/**
 * Does the A3 cube multiply an INT4 right-hand side, driven by hand?
 *
 * The high-level `Matmul` will not pass `int4b_t` in any configuration,
 * because it plans for global memory to hold INT4 while the L1 tile holds
 * INT8 and A3's MTE2 does not widen on the way in. The chip's own path is
 * open: INT4 copies into A1, B1 and A2, and the library's `load_to_l0_utils`
 * already answers `LOAD2DTRANSPOSE` for `int4b_t`. So this drives L1 and L0
 * directly -- `DataCopy`, `LoadData`, `Mmad` -- instead of `IterateAll`.
 *
 * The first question is not layout but arithmetic, so both operands are
 * filled with ones and every element of the product should be K. That holds
 * whatever order the fractals are in. A second filling, bytes of 0x21, makes
 * the two nibbles of a byte differ: the sum over K becomes 96 rather than 64
 * if both halves are read and 64 or 128 if only one is.
 */
#include "int4_mad_kernel.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr int64_t ROWS = 16;
constexpr int64_t INNER = 64;
constexpr int64_t COLUMNS = 16;
// An INT8 fractal covers 32 of K, an INT4 fractal 64.
constexpr int64_t INT8_FRACTAL_K = 32;

}  // namespace

extern "C" __global__ __aicore__ void int4_mad(__gm__ int8_t* left,
                                               __gm__ int4b_t* right,
                                               __gm__ int32_t* out)
{
    TPipe pipe;
    TBuf<TPosition::A1> left_stage;
    TBuf<TPosition::B1> right_stage;
    TBuf<TPosition::A2> left_tile;
    TBuf<TPosition::B2> right_tile;
    TBuf<TPosition::CO1> product_tile;
    pipe.InitBuffer(left_stage, ROWS * INNER * sizeof(int8_t));
    pipe.InitBuffer(right_stage, COLUMNS * INNER / 2);
    pipe.InitBuffer(left_tile, ROWS * INNER * sizeof(int8_t));
    pipe.InitBuffer(right_tile, COLUMNS * INNER / 2);
    pipe.InitBuffer(product_tile, ROWS * COLUMNS * sizeof(int32_t));

    LocalTensor<int8_t> left_l1 = left_stage.Get<int8_t>();
    LocalTensor<int4b_t> right_l1 = right_stage.Get<int4b_t>();
    LocalTensor<int8_t> left_l0 = left_tile.Get<int8_t>();
    LocalTensor<int4b_t> right_l0 = right_tile.Get<int4b_t>();
    LocalTensor<int32_t> product = product_tile.Get<int32_t>();

    GlobalTensor<int8_t> left_global;
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
             LoadData2DParams(0, (uint8_t)(INNER / INT8_FRACTAL_K), 1, 0, 0,
                              false, 0));
    LoadData(right_l0, right_l1, LoadData2DParams(0, 1, 1, 0, 0, false, 0));
    PipeBarrier<PIPE_ALL>();

    Mmad(product, left_l0, right_l0,
         MmadParams((uint16_t)ROWS, (uint16_t)COLUMNS, (uint16_t)INNER, false,
                    0, false, false, false));
    PipeBarrier<PIPE_ALL>();

    DataCopy(out_global, product, ROWS * COLUMNS);
}

void int4_mad_launch(void* stream, void* left, void* right, void* out)
{
    int4_mad<<<1, nullptr, stream>>>((__gm__ int8_t*)left,
                                      (__gm__ int4b_t*)right,
                                      (__gm__ int32_t*)out);
}
