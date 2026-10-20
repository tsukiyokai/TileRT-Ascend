/**
 * Does the A3 cube multiply two INT4 operands, driven by hand?
 *
 * The high-level `Matmul` will not pass `int4b_t` in any configuration,
 * because it plans for global memory to hold INT4 while the L1 tile holds
 * INT8 and A3's MTE2 does not widen on the way in. The chip's own path is
 * open: INT4 copies into A1, B1 and A2, and the library's `load_to_l0_utils`
 * already answers `LOAD2DTRANSPOSE` for `int4b_t`. So this drives L1 and L0
 * directly -- `DataCopy`, `LoadData`, `Mmad` -- instead of `IterateAll`.
 *
 * Both operands must be INT4: `Mmad` lists
 * "dst: int32_t, fm: int4b_t, filter: int4b_t" and dispatches to `mad_s4`,
 * and this part has no mixed-width mad. An INT8 activation reaches an INT4
 * weight by splitting into two INT4 halves and recombining 16 × high + low,
 * which is what the vendor's `a8w4_msd` stages do -- so this probe settles
 * the primitive those two mads rest on.
 *
 * The first question is arithmetic, not layout, so both operands are filled
 * with ones and every element of the product should be K whatever order the
 * fractals are in. A second filling, bytes of 0x21, makes the nibbles of a
 * byte differ: the sum over K becomes 96 if both halves are read, 64 or 128
 * if only one is.
 */
#include "int4_mad_kernel.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr int64_t ROWS = 16;
constexpr int64_t INNER = 64;
constexpr int64_t COLUMNS = 16;
// An INT4 fractal covers 64 of K, so one tile is one repeat.
constexpr int64_t INT4_FRACTAL_K = 64;

}  // namespace

extern "C" __global__ __mix__(1, 2) __aicore__ void int4_mad(
    __gm__ int4b_t* left,
    __gm__ int4b_t* right, __gm__ int32_t* out)
{
    // A cube-only task type was rejected at submission -- the queue reported
    // `sqe_type=3(place holder), errType=0x20(sq sw status error)`. The MIX
    // shape this project already runs cube code through is what works.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    if ASCEND_IS_AIV {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::A1> left_stage;
    TBuf<TPosition::B1> right_stage;
    TBuf<TPosition::A2> left_tile;
    TBuf<TPosition::B2> right_tile;
    TBuf<TPosition::CO1> product_tile;
    pipe.InitBuffer(left_stage, ROWS * INNER / 2);
    pipe.InitBuffer(right_stage, COLUMNS * INNER / 2);
    pipe.InitBuffer(left_tile, ROWS * INNER / 2);
    pipe.InitBuffer(right_tile, COLUMNS * INNER / 2);
    pipe.InitBuffer(product_tile, ROWS * COLUMNS * sizeof(int32_t));

    LocalTensor<int4b_t> left_l1 = left_stage.Get<int4b_t>();
    LocalTensor<int4b_t> right_l1 = right_stage.Get<int4b_t>();
    LocalTensor<int4b_t> left_l0 = left_tile.Get<int4b_t>();
    LocalTensor<int4b_t> right_l0 = right_tile.Get<int4b_t>();
    LocalTensor<int32_t> product = product_tile.Get<int32_t>();

    GlobalTensor<int4b_t> left_global;
    GlobalTensor<int4b_t> right_global;
    GlobalTensor<int32_t> out_global;
    left_global.SetGlobalBuffer(left, ROWS * INNER);
    right_global.SetGlobalBuffer(right, COLUMNS * INNER);
    out_global.SetGlobalBuffer(out, ROWS * COLUMNS);

    DataCopy(left_l1, left_global, ROWS * INNER);
    DataCopy(right_l1, right_global, COLUMNS * INNER);
    PipeBarrier<PIPE_ALL>();

    // One fractal a repeat, contiguous, no transposition.
    LoadData(left_l0, left_l1,
             LoadData2DParams(0, (uint8_t)(INNER / INT4_FRACTAL_K), 1, 0, 0,
                              false, 0));
    LoadData(right_l0, right_l1, LoadData2DParams(0, 1, 1, 0, 0, false, 0));
    PipeBarrier<PIPE_ALL>();

    Mmad(product, left_l0, right_l0,
         MmadParams((uint16_t)ROWS, (uint16_t)COLUMNS, (uint16_t)INNER, false,
                    0, false, false, false));
    PipeBarrier<PIPE_ALL>();

    Fixpipe(out_global, product,
            FixpipeParamsV220((uint16_t)COLUMNS, (uint16_t)ROWS,
                              (uint16_t)0, (uint32_t)COLUMNS, false));
}

void int4_mad_launch(void* stream, void* left, void* right, void* out)
{
    int4_mad<<<1, nullptr, stream>>>((__gm__ int4b_t*)left,
                                      (__gm__ int4b_t*)right,
                                      (__gm__ int32_t*)out);
}
