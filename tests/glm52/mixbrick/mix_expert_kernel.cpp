/**
 * The shared expert's first half in one MIX kernel instead of two.
 *
 * A layer runs `QuantBatchMatmulV3` for the shared expert's gate and up
 * projections, 11.661 us on AI_CORE, then `DequantSwigluQuant`, 3.140 on the
 * vector core. Two launches, and the INT32 product goes to global memory and
 * comes back. Fusing them keeps the product where the cube left it and pays
 * the cross-core structure once.
 *
 * The cube-to-vector flag direction is the only one the headers settle, so
 * that is the only one used: the cube finishes the whole projection, raises
 * `GATE_READY`, and the vector side dequantises and applies SwiGLU. Both arms
 * run at the same block count so the comparison is of the boundary, not of a
 * parallel decomposition -- the multi-core N split needs the gate and up rows
 * interleaved so a core owns whole pairs, which is a separate change.
 */
// MIX-mode Matmul is driven from the AIV: `Iterate` on the vector side only
// notifies the cube side, which answers when the matrix work is done. That is
// the KFC channel, and it is why a vendor MIX entry carries a
// `__kfc_workspace__`. Calling the API from inside `if ASCEND_IS_AIC` is the
// server half running the client's code with no client, and it deadlocks.
// This macro selects the other route -- Matmul on the cube side only -- and
// `matmul_intf.h` reads it, so it has to precede the include.
#define ASCENDC_CUBE_ONLY

#include "mix_expert_kernel.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

constexpr int64_t HALF = 128;
constexpr int64_t GATE_UP = 2 * HALF;
// CANN's own Matmul-to-vector example uses mode 2 and flag 8. A3 allows
// flag ids 0 through 10 only, and the Matmul library consumes some itself, so
// a hand-written flag beside it is a collision risk -- 8 is what the vendor
// example picks for exactly this handoff. A flag's counter also saturates at
// 15 sets, which a per-layer persistent kernel will have to reckon with.
constexpr uint16_t GATE_READY = 8;

using AQuant = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
using BQuant = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, true>;
using CProduct = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using BiasNone = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

namespace {

/** The cube's half: the whole gate-and-up projection into global memory. */
__aicore__ inline void Project(TPipe& pipe, __gm__ int8_t* x,
                               __gm__ int8_t* gate_up,
                               __gm__ int32_t* partial,
                               __gm__ uint8_t* tiling, int64_t rows)
{
    TCubeTiling cube;
    auto* destination = (uint32_t*)&cube;
    auto* source = (__gm__ uint32_t*)tiling;
    for (uint32_t word = 0; word < sizeof(TCubeTiling) / sizeof(uint32_t);
         ++word) {
        destination[word] = source[word];
    }
    Matmul<AQuant, BQuant, CProduct, BiasNone> mm;
    mm.Init(&cube, &pipe);
    GlobalTensor<int8_t> a;
    GlobalTensor<int8_t> b;
    GlobalTensor<int32_t> c;
    a.SetGlobalBuffer((__gm__ int8_t*)x);
    b.SetGlobalBuffer((__gm__ int8_t*)gate_up);
    c.SetGlobalBuffer((__gm__ int32_t*)partial);
    mm.SetTensorA(a);
    mm.SetTensorB(b, true);
    mm.IterateAll(c);
    mm.End();
}

/** One tile of vector state: a row's product, its two scaled halves. */
struct Epilogue {
    TBuf<TPosition::VECCALC> product_buffer;
    TBuf<TPosition::VECCALC> gate_buffer;
    TBuf<TPosition::VECCALC> up_buffer;
    TBuf<TPosition::VECCALC> work_buffer;
    TBuf<TPosition::VECCALC> scale_buffer;
    TBuf<TPosition::VECCALC> narrow_buffer;
    LocalTensor<int32_t> product;
    LocalTensor<float> gate;
    LocalTensor<float> up;
    LocalTensor<float> work;
    LocalTensor<float> scale;
    LocalTensor<bfloat16_t> narrow;

    __aicore__ inline void Init(TPipe& pipe)
    {
        pipe.InitBuffer(product_buffer, GATE_UP * sizeof(int32_t));
        pipe.InitBuffer(gate_buffer, HALF * sizeof(float));
        pipe.InitBuffer(up_buffer, HALF * sizeof(float));
        pipe.InitBuffer(work_buffer, HALF * sizeof(float));
        pipe.InitBuffer(scale_buffer, GATE_UP * sizeof(float));
        pipe.InitBuffer(narrow_buffer, HALF * sizeof(bfloat16_t));
        product = product_buffer.Get<int32_t>();
        gate = gate_buffer.Get<float>();
        up = up_buffer.Get<float>();
        work = work_buffer.Get<float>();
        scale = scale_buffer.Get<float>();
        narrow = narrow_buffer.Get<bfloat16_t>();
    }
};

/** `silu(gate) * up`, rounded once to BF16 on the way out. */
__aicore__ inline void SwiGlu(Epilogue& held)
{
    Muls(held.work, held.gate, -1.0F, HALF);
    PipeBarrier<PIPE_V>();
    Exp(held.work, held.work, HALF);
    PipeBarrier<PIPE_V>();
    Adds(held.work, held.work, 1.0F, HALF);
    PipeBarrier<PIPE_V>();
    Div(held.gate, held.gate, held.work, HALF);
    PipeBarrier<PIPE_V>();
    Mul(held.gate, held.gate, held.up, HALF);
    PipeBarrier<PIPE_V>();
    Cast(held.narrow, held.gate, RoundMode::CAST_RINT, HALF);
}

/** The vector half: dequantise each row's product and apply SwiGLU. */
__aicore__ inline void Activate(TPipe& pipe, __gm__ int32_t* partial,
                                __gm__ float* gu_scale,
                                __gm__ float* token_scale,
                                __gm__ bfloat16_t* out, int64_t rows)
{
    Epilogue held;
    held.Init(pipe);
    GlobalTensor<int32_t> product_global;
    GlobalTensor<float> scale_global;
    GlobalTensor<float> token_global;
    GlobalTensor<bfloat16_t> out_global;
    product_global.SetGlobalBuffer(partial);
    scale_global.SetGlobalBuffer(gu_scale);
    token_global.SetGlobalBuffer(token_scale);
    out_global.SetGlobalBuffer(out);

    DataCopy(held.scale, scale_global, GATE_UP);
    PipeBarrier<PIPE_ALL>();

    // Two vector subblocks, alternate rows each. At eight rows the work is
    // 1024 elements, so the split is about keeping both subblocks fed rather
    // than about throughput.
    // Two vector subblocks, alternate rows each. The stride is written out
    // rather than taken from `GetTaskRation`, which is not 2 on every path and
    // would make the loop never advance if it returned zero.
    const int64_t subblocks = 2;
    for (int64_t row = GetSubBlockIdx(); row < rows; row += subblocks) {
        DataCopy(held.product, product_global[row * GATE_UP], GATE_UP);
        PipeBarrier<PIPE_ALL>();
        Cast(held.gate, held.product, RoundMode::CAST_NONE, HALF);
        Cast(held.up, held.product[HALF], RoundMode::CAST_NONE, HALF);
        PipeBarrier<PIPE_V>();
        Mul(held.gate, held.gate, held.scale, HALF);
        Mul(held.up, held.up, held.scale[HALF], HALF);
        PipeBarrier<PIPE_V>();
        Muls(held.gate, held.gate, token_global.GetValue(row), HALF);
        Muls(held.up, held.up, token_global.GetValue(row), HALF);
        PipeBarrier<PIPE_V>();
        SwiGlu(held);
        PipeBarrier<PIPE_ALL>();
        DataCopy(out_global[row * HALF], held.narrow, HALF);
    }
}

}  // namespace

// The AIC:AIV ratio is declared once, by KERNEL_TASK_TYPE_DEFAULT in the
// body. `[[bisheng::core_ratio]]` says the same thing again and the sanitizer
// build rejects it outright in this position.
extern "C" __global__ __schedmode__(1) __aicore__ void mix_expert_fused(
    uint64_t sync_base, __gm__ int8_t* x, __gm__ int8_t* gate_up,
    __gm__ int32_t* partial, __gm__ float* gu_scale,
    __gm__ float* token_scale, __gm__ bfloat16_t* out,
    __gm__ uint8_t* tiling, int64_t rows, int64_t stages)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    SetSyncBaseAddr(sync_base);
    TPipe pipe;
    // `stages` localises a hang: bit 0 the cube half, bit 1 the flag, bit 2
    // the vector half.
    if ASCEND_IS_AIC {
        if (stages & 1) {
            Project(pipe, x, gate_up, partial, tiling, rows);
        }
        if (stages & 2) {
            CrossCoreSetFlag<0x0, PIPE_FIX>(GATE_READY);
        }
    }
    if ASCEND_IS_AIV {
        if (stages & 2) {
            CrossCoreWaitFlag(GATE_READY);
        }
        if (stages & 4) {
            Activate(pipe, partial, gu_scale, token_scale, out, rows);
        }
    }
}

// The AIC:AIV ratio is declared once, by KERNEL_TASK_TYPE_DEFAULT in the
// body. `[[bisheng::core_ratio]]` says the same thing again and the sanitizer
// build rejects it outright in this position.
extern "C" __global__ __schedmode__(1) __aicore__ void mix_expert_project(
    uint64_t sync_base, __gm__ int8_t* x, __gm__ int8_t* gate_up,
    __gm__ int32_t* partial, __gm__ uint8_t* tiling, int64_t rows)
{
    // Also a MIX kernel, so that the staged arm's matmul is the same kernel
    // kind as the fused one's and the comparison is of the boundary alone.
    // An ND input makes the Matmul library convert to NZ, which is not purely
    // a cube activity.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    SetSyncBaseAddr(sync_base);
    TPipe pipe;
    if ASCEND_IS_AIC {
        Project(pipe, x, gate_up, partial, tiling, rows);
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void mix_expert_activate(__gm__ int32_t* partial, __gm__ float* gu_scale,
                         __gm__ float* token_scale, __gm__ bfloat16_t* out,
                         int64_t rows)
{
    TPipe pipe;
    Activate(pipe, partial, gu_scale, token_scale, out, rows);
}

void mix_expert_fused_launch(uint32_t blocks, void* stream, uint64_t sync_base,
                             void* x, void* gate_up, void* partial,
                             float* gu_scale, float* token_scale, void* out,
                             void* tiling, int64_t rows, int64_t stages)
{
    mix_expert_fused<<<blocks, nullptr, stream>>>(
        sync_base, (__gm__ int8_t*)x, (__gm__ int8_t*)gate_up,
        (__gm__ int32_t*)partial, gu_scale, token_scale,
        (__gm__ bfloat16_t*)out, (__gm__ uint8_t*)tiling, rows, stages);
}

void mix_expert_staged_launch(uint32_t blocks, void* stream,
                              uint64_t sync_base, void* x, void* gate_up,
                              void* partial, float* gu_scale,
                              float* token_scale, void* out, void* tiling,
                              int64_t rows)
{
    mix_expert_project<<<blocks, nullptr, stream>>>(
        sync_base, (__gm__ int8_t*)x, (__gm__ int8_t*)gate_up,
        (__gm__ int32_t*)partial, (__gm__ uint8_t*)tiling, rows);
    mix_expert_activate<<<blocks, nullptr, stream>>>(
        (__gm__ int32_t*)partial, gu_scale, token_scale,
        (__gm__ bfloat16_t*)out, rows);
}
