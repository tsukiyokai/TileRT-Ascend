/**
 * The routed experts with INT4 weights, which halves the dominant term.
 *
 * The INT8 form of this kernel is weight-streaming bound at 66% of the die's
 * bandwidth, and the model is W4A8, so it carries twice the bytes it needs
 * to. The cube's only INT4 instruction is `mad_s4`, which wants both operands
 * INT4, and the `Matmul` library passes a homogeneous INT4 pair while
 * rejecting a mixed one. So the INT8 activation becomes two INT4 halves --
 * the technique CANN's own `a8w4_msd` stages use -- and the halves stack
 * along M rather than taking a matmul each, so the weight is still read once:
 *
 *     x = 16 * high + low,  C[r] = 16 * C'[r] + C'[ROWS + r]
 *
 * Two signed INT4 reach [-136, 119], so an INT8 of 120 or more has no exact
 * pair. Rather than carry a correction term, the quantisation divides by 119.
 * That is 0.56 dB of range against 4-bit weights, and it makes every stage
 * exact.
 *
 * [int4-weights-20261020](../../../docs/evidence/ascend-decode/int4-weights-20261020/README.md)
 */
#define ASCENDC_CUBE_ONLY

#include "routed_int4_kernel.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

constexpr int64_t HIDDEN = 6144;
constexpr int64_t HALF = 128;
constexpr int64_t GATE_UP = 2 * HALF;
#ifndef ROWS_PER_EXPERT
#define ROWS_PER_EXPERT 2
#endif
constexpr int64_t ROWS = ROWS_PER_EXPERT;
// The two INT4 halves of the activation, stacked.
constexpr int64_t STACKED = 2 * ROWS;
// The cube works in sixteen-row tiles and the `Matmul` library reads and
// writes the whole tile whatever M says, so every buffer along M carries one.
// Without it the sanitizer reported reads of 928 and writes of 992 bytes off
// the ends of `stacked`, `middle`, `partial` and `wide`.
constexpr int64_t M_TILE = 16;
constexpr int64_t SLACK = 64;
constexpr int64_t BLOCK_FLOATS = 8;
constexpr int64_t SCALE_SPAN = ((ROWS + BLOCK_FLOATS - 1) / BLOCK_FLOATS)
                               * BLOCK_FLOATS;
constexpr int64_t TILE = 2048;
constexpr float QUANTUM_SPAN = 127.0F;

/** The address `elements` INT4 values along from `base`.
 *
 * `int4b_t` is a struct, so neither its `sizeof` nor a `GlobalTensor`
 * subscript can be trusted to mean a nibble: a write addressed through
 * either landed nowhere in the allocation, while the same write at offset
 * zero landed every time, and the stale bytes it should have overwritten
 * read identically across completely different input data. So the arithmetic
 * goes through a byte pointer, where the unit is not in doubt.
 */
__aicore__ inline __gm__ int4b_t* Nibbles(__gm__ int4b_t* base,
                                          int64_t elements)
{
    return (__gm__ int4b_t*)((__gm__ int8_t*)base + elements / 2);
}

constexpr uint16_t SPLIT_READY = 4;
constexpr uint16_t GATE_READY = 5;
constexpr uint16_t MIDDLE_READY = 6;
constexpr uint16_t DOWN_READY = 7;

using APacked = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
using BPacked = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, true>;
using CProduct = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using BiasNone = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

namespace {

__aicore__ inline void Publish()
{
    DataSyncBarrier<MemDsbT::DDR>();
}

__aicore__ inline void LoadTiling(TCubeTiling& cube, __gm__ uint8_t* tiling)
{
    auto* destination = (uint32_t*)&cube;
    auto* source = (__gm__ uint32_t*)tiling;
    for (uint32_t word = 0; word < sizeof(TCubeTiling) / sizeof(uint32_t);
         ++word) {
        destination[word] = source[word];
    }
}

/** One INT4 projection at full width, both operands packed. */
__aicore__ inline void Project(TPipe& pipe, __gm__ int4b_t* rows,
                               __gm__ int4b_t* weight,
                               __gm__ int32_t* product, int64_t columns,
                               int64_t inner, __gm__ uint8_t* tiling)
{
    TCubeTiling cube;
    LoadTiling(cube, tiling);
    Matmul<APacked, BPacked, CProduct, BiasNone> mm;
    mm.Init(&cube, &pipe);
    GlobalTensor<int4b_t> a;
    GlobalTensor<int4b_t> b;
    GlobalTensor<int32_t> c;
    a.SetGlobalBuffer(rows, M_TILE * inner);
    b.SetGlobalBuffer(weight, columns * inner);
    c.SetGlobalBuffer(product, M_TILE * columns);
    mm.SetOrgShape(STACKED, columns, inner);
    mm.SetTensorA(a);
    mm.SetTensorB(b, true);
    mm.IterateAll(c);
    mm.End();
}

/** The vector state every stage shares, allocated once a launch.
 *
 * A `TPipe` hands out buffers cumulatively, so calling `InitBuffer` inside a
 * stage that runs once an expert exhausts UB and the buffers start to
 * overlap. That showed as the low half of every split reading zero while the
 * high half read a plain x/16. Allocating once at 6144 wide still overlapped
 * -- ten buffers of a row came to 144 KB -- and the symptom moved rather than
 * went: the high half stayed right while the low half, written one `Emit`
 * later, came back as noise. So every stage works a tile at a time and the
 * footprint stops depending on the hidden size.
 */
struct Workspace {
    TBuf<TPosition::VECCALC> narrow_buffer;
    TBuf<TPosition::VECCALC> value_buffer;
    TBuf<TPosition::VECCALC> other_buffer;
    TBuf<TPosition::VECCALC> work_buffer;
    TBuf<TPosition::VECCALC> half_buffer;
    TBuf<TPosition::VECCALC> code_buffer;
    TBuf<TPosition::VECCALC> spare_half_buffer;
    TBuf<TPosition::VECCALC> spare_code_buffer;
    TBuf<TPosition::VECCALC> staging_buffer;
    TBuf<TPosition::VECCALC> scale_buffer;
    TBuf<TPosition::VECCALC> token_buffer;
    TBuf<TPosition::VECCALC> hold_buffer;
    LocalTensor<int8_t> narrow;
    LocalTensor<float> value;
    LocalTensor<float> other;
    LocalTensor<float> work;
    LocalTensor<half> half_value;
    LocalTensor<int4b_t> code;
    LocalTensor<half> spare_half;
    LocalTensor<int4b_t> spare_code;
    LocalTensor<int32_t> staging;
    LocalTensor<float> scale;
    LocalTensor<float> token;
    LocalTensor<float> activation;

    __aicore__ inline void Init(TPipe& pipe, int64_t span)
    {
        pipe.InitBuffer(narrow_buffer, (span + SLACK) * sizeof(int8_t));
        pipe.InitBuffer(value_buffer, (span + SLACK) * sizeof(float));
        pipe.InitBuffer(other_buffer, (span + SLACK) * sizeof(float));
        pipe.InitBuffer(work_buffer, (span + SLACK) * sizeof(float));
        pipe.InitBuffer(half_buffer, (span + SLACK) * sizeof(half));
        pipe.InitBuffer(code_buffer, (span + SLACK) / 2 + SLACK);
        narrow = narrow_buffer.Get<int8_t>();
        value = value_buffer.Get<float>();
        other = other_buffer.Get<float>();
        work = work_buffer.Get<float>();
        half_value = half_buffer.Get<half>();
        code = code_buffer.Get<int4b_t>();
        pipe.InitBuffer(spare_half_buffer, (span + SLACK) * sizeof(half));
        pipe.InitBuffer(spare_code_buffer, (span + SLACK) / 2 + SLACK);
        spare_half = spare_half_buffer.Get<half>();
        spare_code = spare_code_buffer.Get<int4b_t>();
        pipe.InitBuffer(staging_buffer, (span + SLACK) * sizeof(int32_t));
        pipe.InitBuffer(scale_buffer, (span + SLACK) * sizeof(float));
        pipe.InitBuffer(token_buffer, (SCALE_SPAN + SLACK) * sizeof(float));
        pipe.InitBuffer(hold_buffer, (ROWS * HALF + SLACK) * sizeof(float));
        staging = staging_buffer.Get<int32_t>();
        scale = scale_buffer.Get<float>();
        token = token_buffer.Get<float>();
        activation = hold_buffer.Get<float>();
    }
};

/** Split `value` into its two INT4 halves: high into `value`, low into
 *  `work`.
 *
 * `high` is `floor(x / 16)`, which lands in [-8, 7] for every INT8, and the
 * remainder `x - 16 * high` lands in [0, 15] -- unsigned, which the cube
 * cannot read. Taking eight off it gives an INT4, and the eight comes back as
 * `8 * sum(w)` a column, which depends only on the weight and so is computed
 * once when the weight is loaded. Five operations, no clamp, and the whole
 * INT8 range exact.
 */
__aicore__ inline void SplitHalves(Workspace& held, int64_t span)
{
    Muls(held.other, held.value, 1.0F / 16.0F, span);
    PipeBarrier<PIPE_V>();
    Floor(held.other, held.other, span);
    PipeBarrier<PIPE_V>();
    Muls(held.work, held.other, -16.0F, span);
    PipeBarrier<PIPE_V>();
    Add(held.work, held.work, held.value, span);
    PipeBarrier<PIPE_V>();
    Adds(held.work, held.work, -8.0F, span);
    PipeBarrier<PIPE_V>();
    DataCopy(held.value, held.other, span);
    PipeBarrier<PIPE_ALL>();
}

/** Write `span` INT4 values into `sink` at element `at`.
 *
 * One tensor over the whole destination, subscripted -- not a fresh tensor
 * whose base carries the offset. That shape loses the offset: both halves of
 * a split landed at zero, and which one survived depended on the order the
 * copies happened to complete, so the high half looked correct for as long
 * as it won. An isolated probe writing the same two halves through one
 * tensor and one staging pair lands both, at every offset from 256 to 24576,
 * under `__mix__` and `ASCENDC_CUBE_ONLY` alike.
 */
__aicore__ inline void Emit(LocalTensor<half>& staging,
                            LocalTensor<int4b_t>& packed,
                            LocalTensor<float>& source,
                            GlobalTensor<int4b_t>& sink, int64_t at,
                            int64_t span)
{
    Cast(staging, source, RoundMode::CAST_RINT, span);
    PipeBarrier<PIPE_V>();
    Cast(packed, staging, RoundMode::CAST_RINT, span);
    PipeBarrier<PIPE_ALL>();
    DataCopy(sink[at], packed, span);
    PipeBarrier<PIPE_ALL>();
}

/** Stack one expert's rows as two INT4 halves. */
__aicore__ inline void Split(Workspace& held, __gm__ int8_t* rows,
                             __gm__ int4b_t* stacked, __gm__ int4b_t* witness)
{
    GlobalTensor<int8_t> rows_global;
    GlobalTensor<int4b_t> stacked_global;
    GlobalTensor<int4b_t> witness_global;
    rows_global.SetGlobalBuffer(rows, ROWS * HIDDEN);
    stacked_global.SetGlobalBuffer(stacked, M_TILE * HIDDEN);
    witness_global.SetGlobalBuffer(witness, TILE);

    for (int64_t row = 0; row < ROWS; ++row) {
        for (int64_t done = 0; done < HIDDEN; done += TILE) {
            const int64_t count = (HIDDEN - done < TILE) ? HIDDEN - done
                                                         : TILE;
            DataCopy(held.narrow, rows_global[row * HIDDEN + done], count);
            PipeBarrier<PIPE_ALL>();
            Cast(held.half_value, held.narrow, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Cast(held.value, held.half_value, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            SplitHalves(held, count);
            Emit(held.half_value, held.code, held.value, stacked_global,
                 row * HIDDEN + done, count);
            Emit(held.half_value, held.code, held.work, stacked_global,
                 (ROWS + row) * HIDDEN + done, count);
            if (row == 0 && done == 0) {
                Emit(held.half_value, held.code, held.work, witness_global, 0,
                     count);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
}

/** `silu(gate) * up`, in place in `value`. */
__aicore__ inline void SwiGlu(Workspace& held, int64_t span)
{
    Muls(held.work, held.value, -1.0F, span);
    PipeBarrier<PIPE_V>();
    Exp(held.work, held.work, span);
    PipeBarrier<PIPE_V>();
    Adds(held.work, held.work, 1.0F, span);
    PipeBarrier<PIPE_V>();
    Div(held.value, held.value, held.work, span);
    PipeBarrier<PIPE_V>();
    Mul(held.value, held.value, held.other, span);
}

/** Recombine the two halves of one row of a product into `into`. */
__aicore__ inline void Combine(LocalTensor<float>& into,
                               LocalTensor<float>& spare,
                               GlobalTensor<int32_t>& product,
                               LocalTensor<int32_t>& staging,
                               GlobalTensor<float>& bias, int64_t at,
                               int64_t high, int64_t low, int64_t span)
{
    DataCopy(staging, product[high], span);
    PipeBarrier<PIPE_ALL>();
    Cast(into, staging, RoundMode::CAST_NONE, span);
    PipeBarrier<PIPE_V>();
    Muls(into, into, 16.0F, span);
    PipeBarrier<PIPE_V>();
    DataCopy(staging, product[low], span);
    PipeBarrier<PIPE_ALL>();
    Cast(spare, staging, RoundMode::CAST_NONE, span);
    PipeBarrier<PIPE_V>();
    Add(into, into, spare, span);
    PipeBarrier<PIPE_V>();
    DataCopy(spare, bias[at], span);
    PipeBarrier<PIPE_ALL>();
    Add(into, into, spare, span);
    PipeBarrier<PIPE_V>();
}

/** Recombine, activate, quantise and split one expert's rows.
 *
 * The row maximum is over this expert's own 128 columns and this core owns
 * every one of them, so nothing crosses blocks.
 */
__aicore__ inline void Activate(Workspace& held, __gm__ int32_t* product,
                                __gm__ float* gu_bias, __gm__ float* gu_scale,
                                __gm__ float* token_scale,
                                __gm__ int4b_t* middle,
                                __gm__ float* out_scale)
{
    LocalTensor<int32_t> staging = held.staging;
    LocalTensor<float> scale = held.scale;
    LocalTensor<float> token = held.token;
    LocalTensor<float> activation = held.activation;

    GlobalTensor<int32_t> product_global;
    GlobalTensor<float> scale_global;
    GlobalTensor<float> token_global;
    GlobalTensor<int4b_t> middle_global;
    GlobalTensor<float> out_scale_global;
    GlobalTensor<float> bias_global;
    middle_global.SetGlobalBuffer(middle, M_TILE * HALF);
    bias_global.SetGlobalBuffer(gu_bias, GATE_UP);
    product_global.SetGlobalBuffer(product, M_TILE * GATE_UP);
    scale_global.SetGlobalBuffer(gu_scale, GATE_UP);
    token_global.SetGlobalBuffer(token_scale, SCALE_SPAN);
    out_scale_global.SetGlobalBuffer(out_scale, SCALE_SPAN);

    DataCopy(scale, scale_global, GATE_UP);
    DataCopy(token, token_global, SCALE_SPAN);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::UB>();

    for (int64_t row = 0; row < ROWS; ++row) {
        Combine(held.value, held.work, product_global, staging, bias_global,
                0, row * GATE_UP, (ROWS + row) * GATE_UP, GATE_UP);
        // The gate half stays in `value`; the up half moves to `other`.
        DataCopy(held.other, held.value[HALF], HALF);
        PipeBarrier<PIPE_ALL>();
        Mul(held.value, held.value, scale, HALF);
        Mul(held.other, held.other, scale[HALF], HALF);
        PipeBarrier<PIPE_V>();
        const float of_row = token.GetValue(row);
        Muls(held.value, held.value, of_row, HALF);
        Muls(held.other, held.other, of_row, HALF);
        PipeBarrier<PIPE_V>();
        SwiGlu(held, HALF);
        PipeBarrier<PIPE_V>();
        DataCopy(activation[row * HALF], held.value, HALF);
        Abs(held.work, held.value, HALF);
        PipeBarrier<PIPE_V>();
        ReduceMax(token[row], held.work, held.work, HALF, false);
        PipeBarrier<PIPE_ALL>();
    }
    DataSyncBarrier<MemDsbT::UB>();

    for (int64_t row = 0; row < ROWS; ++row) {
        const float largest = token.GetValue(row);
        const float quantum = largest / QUANTUM_SPAN;
        Muls(held.value, activation[row * HALF], 1.0F / quantum, HALF);
        PipeBarrier<PIPE_V>();
        // Round to an integer before splitting, so both halves are exact.
        Cast(held.half_value, held.value, RoundMode::CAST_RINT, HALF);
        PipeBarrier<PIPE_V>();
        Cast(held.value, held.half_value, RoundMode::CAST_NONE, HALF);
        PipeBarrier<PIPE_V>();
        SplitHalves(held, HALF);
        Emit(held.half_value, held.code, held.value, middle_global,
             row * HALF, HALF);
        Emit(held.half_value, held.code, held.work, middle_global,
             (ROWS + row) * HALF, HALF);
        PipeBarrier<PIPE_ALL>();
        token.SetValue(row, quantum);
    }
    PipeBarrier<PIPE_ALL>();
    // The scale leaves by MTE: a scalar store to global memory from a vector
    // core does not land.
    DataCopy(out_scale_global, token, SCALE_SPAN);
}

/** Recombine and scale one expert's down product into BF16. */
__aicore__ inline void Narrow(Workspace& held, __gm__ int32_t* product,
                              __gm__ float* down_bias,
                              __gm__ float* down_scale,
                              __gm__ float* middle_scale,
                              __gm__ bfloat16_t* out, int64_t from,
                              int64_t upto)
{
    LocalTensor<int32_t> staging = held.staging;
    LocalTensor<float> scale = held.scale;
    LocalTensor<float> row_scale = held.token;
    // BF16 output reuses the half-width buffer the casts already need.
    LocalTensor<bfloat16_t> narrow = held.half_buffer.Get<bfloat16_t>();

    GlobalTensor<int32_t> product_global;
    GlobalTensor<float> scale_global;
    GlobalTensor<float> middle_global;
    GlobalTensor<bfloat16_t> out_global;
    GlobalTensor<float> bias_global;
    bias_global.SetGlobalBuffer(down_bias, HIDDEN);
    product_global.SetGlobalBuffer(product, M_TILE * HIDDEN);
    scale_global.SetGlobalBuffer(down_scale, HIDDEN);
    middle_global.SetGlobalBuffer(middle_scale, SCALE_SPAN);
    out_global.SetGlobalBuffer(out, ROWS * HIDDEN);

    DataCopy(row_scale, middle_global, SCALE_SPAN);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::UB>();

    for (int64_t row = 0; row < ROWS; ++row) {
        const float of_row = row_scale.GetValue(row);
        for (int64_t done = from; done < upto; done += TILE) {
            const int64_t count = (upto - done < TILE) ? upto - done : TILE;
            Combine(held.value, held.work, product_global, staging,
                    bias_global, done, row * HIDDEN + done,
                    (ROWS + row) * HIDDEN + done, count);
            DataCopy(scale, scale_global[done], count);
            PipeBarrier<PIPE_ALL>();
            Mul(held.value, held.value, scale, count);
            PipeBarrier<PIPE_V>();
            Muls(held.value, held.value, of_row, count);
            PipeBarrier<PIPE_V>();
            Cast(narrow, held.value, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_ALL>();
            DataCopy(out_global[row * HIDDEN + done], narrow, count);
        }
    }
}

}  // namespace

extern "C" __global__ __mix__(1, 2) __schedmode__(1)
__aicore__ void routed_int4_fused(
    uint64_t sync_base, __gm__ int8_t* rows, __gm__ int4b_t* stacked,
    __gm__ int4b_t* gate_up, __gm__ float* gu_bias, __gm__ float* gu_scale,
    __gm__ int4b_t* down, __gm__ float* down_bias, __gm__ float* down_scale,
    __gm__ float* token_scale,
    __gm__ int32_t* partial, __gm__ int4b_t* middle,
    __gm__ float* middle_scale, __gm__ int32_t* wide, __gm__ bfloat16_t* out,
    __gm__ uint8_t* gu_tiling, __gm__ uint8_t* down_tiling,
    __gm__ int4b_t* witness, int64_t experts, int64_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);

    const int64_t per_block = (experts + blocks - 1) / blocks;
    TPipe pipe;

    if ASCEND_IS_AIC {
        const int64_t first = GetBlockIdx() * per_block;
        for (int64_t step = 0; step < per_block; ++step) {
            const int64_t expert = first + step;
            if (expert >= experts) {
                break;
            }
            CrossCoreWaitFlag<0x03, PIPE_MTE2>(SPLIT_READY);
            Project(pipe, Nibbles(stacked, expert * M_TILE * HIDDEN),
                    Nibbles(gate_up, expert * GATE_UP * HIDDEN),
                    partial + expert * M_TILE * GATE_UP, GATE_UP, HIDDEN,
                    gu_tiling);
            Publish();
            CrossCoreSetFlag<0x02, PIPE_FIX>(GATE_READY);
            CrossCoreWaitFlag<0x03, PIPE_MTE2>(MIDDLE_READY);
            Project(pipe, Nibbles(middle, expert * M_TILE * HALF),
                    Nibbles(down, expert * HIDDEN * HALF),
                    wide + expert * M_TILE * HIDDEN, HIDDEN, HALF,
                    down_tiling);
            Publish();
            CrossCoreSetFlag<0x02, PIPE_FIX>(DOWN_READY);
        }
    }
    if ASCEND_IS_AIV {
        const int64_t first = (GetBlockIdx() / 2) * per_block;
        const int64_t sub = GetSubBlockIdx();
        const int64_t share = HIDDEN / 2;
        const int64_t from = sub * share;
        Workspace held;
        held.Init(pipe, TILE);
        for (int64_t step = 0; step < per_block; ++step) {
            const int64_t expert = first + step;
            if (expert >= experts) {
                break;
            }
            // Mode 3 answers the cube whether one subblock sets or both, so
            // the narrow work stays on subblock zero and the row maximum
            // stays core-local.
            if (sub == 0) {
                Split(held, rows + expert * ROWS * HIDDEN,
                      Nibbles(stacked, expert * M_TILE * HIDDEN), witness);
                Publish();
                CrossCoreSetFlag<0x03, PIPE_MTE3>(SPLIT_READY);
            }
            CrossCoreWaitFlag<0x02, PIPE_MTE2>(GATE_READY);
            if (sub == 0) {
                Activate(held, partial + expert * M_TILE * GATE_UP,
                         gu_bias + expert * GATE_UP,
                         gu_scale + expert * GATE_UP,
                         token_scale + expert * SCALE_SPAN,
                         Nibbles(middle, expert * M_TILE * HALF),
                         middle_scale + expert * SCALE_SPAN);
                Publish();
                CrossCoreSetFlag<0x03, PIPE_MTE3>(MIDDLE_READY);
            }
            CrossCoreWaitFlag<0x02, PIPE_MTE2>(DOWN_READY);
            Narrow(held, wide + expert * M_TILE * HIDDEN,
                   down_bias + expert * HIDDEN,
                   down_scale + expert * HIDDEN,
                   middle_scale + expert * SCALE_SPAN,
                   out + expert * ROWS * HIDDEN, from, from + share);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

void routed_int4_launch(uint32_t blocks, void* stream, uint64_t sync_base,
                        void* rows, void* stacked, void* gate_up,
                        float* gu_bias, float* gu_scale, void* down,
                        float* down_bias, float* down_scale,
                        float* token_scale, void* partial, void* middle,
                        float* middle_scale, void* wide, void* out,
                        void* gu_tiling, void* down_tiling, void* witness,
                        int64_t experts)
{
    routed_int4_fused<<<blocks, nullptr, stream>>>(
        sync_base, (__gm__ int8_t*)rows, (__gm__ int4b_t*)stacked,
        (__gm__ int4b_t*)gate_up, gu_bias, gu_scale, (__gm__ int4b_t*)down,
        down_bias, down_scale, token_scale, (__gm__ int32_t*)partial,
        (__gm__ int4b_t*)middle, middle_scale, (__gm__ int32_t*)wide,
        (__gm__ bfloat16_t*)out, (__gm__ uint8_t*)gu_tiling,
        (__gm__ uint8_t*)down_tiling, (__gm__ int4b_t*)witness, experts,
        blocks);
}
