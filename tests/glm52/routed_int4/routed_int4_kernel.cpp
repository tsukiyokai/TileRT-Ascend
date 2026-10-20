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
 *     x = 16 * high + low,  C[r] = 16 * C'[r] + C'[STRIDE + r]
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
// The cube works in sixteen-row tiles and the `Matmul` library reads and
// writes the whole tile whatever M says, so every buffer along M carries one.
// Without it the sanitizer reported reads of 928 and writes of 992 bytes off
// the ends of `stacked`, `middle`, `partial` and `wide`.
constexpr int64_t M_TILE = 16;
// Where the low INT4 half starts in the M-stacked activation. Fixed at half a
// cube tile, so one tiling serves every row count: the cube reads a whole
// sixteen-row tile whatever M says, and the vector stages read only the rows
// an expert actually has.
constexpr int64_t STRIDE = M_TILE / 2;
// Every routed expert a rank holds; the group list carries one count each.
constexpr int64_t MAX_EXPERTS = 256;
// The two INT4 halves of the activation, stacked.
constexpr int64_t STACKED = 2 * STRIDE;
constexpr int64_t SLACK = 64;
constexpr int64_t BLOCK_FLOATS = 8;
constexpr int64_t SCALE_SPAN = ((STRIDE + BLOCK_FLOATS - 1) / BLOCK_FLOATS)
                               * BLOCK_FLOATS;
#ifndef TILE_SPAN
#define TILE_SPAN 2048
#endif
constexpr int64_t TILE = TILE_SPAN;
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
using BPacked = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
using CProduct = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using BiasNone = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

namespace {

__aicore__ inline void Publish()
{
    DataSyncBarrier<MemDsbT::DDR>();
}

/* `PipeBarrier` is the single-pipe primitive -- it orders instructions of the
 * same pipe that carry a data dependency. Crossing pipes is what
 * `SetFlag`/`WaitFlag` are for, so `PIPE_ALL` is not defined to cover a
 * cross-pipe hazard: it is absent from the headers' own `SUPPORTED_PIPE` and
 * the Ascend checker flags it as `AKA3001`. These three name the hazards this
 * kernel actually has.
 *
 * All three name vector events, so they may only be called from the vector
 * half: on the cube half of a `__mix__` kernel `WaitFlagImpl` guards those
 * cases behind `#ifndef SPLIT_CORE_CUBE` and falls through to `break`, while
 * `SetFlagImpl` has no such guard -- the flag would be raised and never
 * consumed.
 */

/** Order a load into UB against the vector work that reads it. */
__aicore__ inline void AfterLoad()
{
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}

/** Order vector work against the store that carries it out. */
__aicore__ inline void BeforeStore()
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
}

/** Order a load into UB against the scalar read that follows it.
 *
 * A `GetValue` is a scalar-unit read, so the edge is `MTE2_S` and not
 * `MTE2_V` -- the vector edge says nothing about when the scalar unit may
 * look. `Narrow` reads its per-row scale with `GetValue` straight after the
 * copy that brings it in, and subblock one goes there with a cold buffer
 * where subblock zero has just run `Activate` on the same tensors. That is
 * why half the columns were wrong and the other half exact: 24571 of 24576 in
 * columns 3072 and above, 0 of 24576 below.
 */
__aicore__ inline void AfterLoadForScalar()
{
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}

/** Order a load against the vector work it is about to overwrite.
 *
 * The reverse direction, and the one a reused staging buffer needs. `MTE2_V`
 * says the data has arrived; `V_MTE2` says the consumer is finished and the
 * buffer may be refilled. Nothing in this kernel stalls MTE2, so without it
 * the next load runs ahead and clobbers what its consumer has not read --
 * which is how one `staging` tensor served two consecutive loads in
 * `Combine` and produced a row maximum that disagreed with the reference
 * while the product feeding it was exact.
 */
__aicore__ inline void BeforeLoad()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}

/** Order a store against the vector work that overwrites what it read. */
__aicore__ inline void AfterStore()
{
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
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
    b.SetGlobalBuffer(weight, inner * columns);
    c.SetGlobalBuffer(product, M_TILE * columns);
    mm.SetOrgShape(STACKED, columns, inner);
    mm.SetTensorA(a);
    mm.SetTensorB(b, false);
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
    // A reduction's destination is written a whole 32-byte block at a time,
    // so the row maxima need a block each and cannot share the buffer that
    // holds the incoming per-token scales: row zero's reduction would stamp
    // over row one's scale, which is exactly what it did -- row zero's
    // quantum matched the reference and row one's did not.
    TBuf<TPosition::VECCALC> rowmax_buffer;
    TBuf<TPosition::VECCALC> group_buffer;
    TBuf<TPosition::VECCALC> start_buffer;
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
    LocalTensor<float> rowmax;
    LocalTensor<int32_t> group;
    LocalTensor<int32_t> start;

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
        pipe.InitBuffer(hold_buffer,
                        (STRIDE * HALF + SLACK) * sizeof(float));
        staging = staging_buffer.Get<int32_t>();
        scale = scale_buffer.Get<float>();
        token = token_buffer.Get<float>();
        activation = hold_buffer.Get<float>();
        pipe.InitBuffer(rowmax_buffer,
                        STRIDE * BLOCK_FLOATS * sizeof(float));
        rowmax = rowmax_buffer.Get<float>();
        pipe.InitBuffer(group_buffer, (MAX_EXPERTS + SLACK) * sizeof(int32_t));
        group = group_buffer.Get<int32_t>();
        pipe.InitBuffer(start_buffer, (MAX_EXPERTS + SLACK) * sizeof(int32_t));
        start = start_buffer.Get<int32_t>();
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
    BeforeStore();
    // `DataCopyPad`, not `DataCopy`. Every `CastImpl` on this part opens with
    // `if ASCEND_IS_AIV`, so on the cube half the two casts above compile to
    // nothing -- while plain `DataCopy` to global memory carries no
    // `ASCEND_IS_AIC` guard and is issued anyway, with a UB pointer, on a
    // core that has no UB. `DataCopyPad` returns early on the cube half by
    // construction. Two writers to one address, one of them writing whatever
    // that pointer resolves to, is the shape of this kernel's defect: stable
    // nonsense 6144 bytes into `stacked` and the same nonsense again 6144
    // further on.
    DataCopyPad(sink[at], packed,
                DataCopyExtParams{1, (uint32_t)(span / 2), 0, 0, 0});
    // The next call's cast writes this staging while this copy is reading
    // it. `PipeBarrier<PIPE_ALL>` does not order a vector write against an
    // outstanding MTE3 read: with it, two calls in a row through one staging
    // pair gave the right halves on some runs and garbage on others, from the
    // same binary. CANN's own `a8w4_msd_pre` uses these flags for exactly
    // this pair.
    AfterStore();
}

/** Stack one expert's rows as two INT4 halves. */
__aicore__ inline void Split(Workspace& held, __gm__ int8_t* rows,
                             __gm__ int4b_t* stacked,
                             __gm__ int4b_t* stacked_low,
                             __gm__ int4b_t* witness, int64_t rows_held)
{
    GlobalTensor<int8_t> rows_global;
    GlobalTensor<int4b_t> high_global;
    GlobalTensor<int4b_t> low_global;
    GlobalTensor<int4b_t> witness_global;
    rows_global.SetGlobalBuffer(rows, rows_held * HIDDEN);
    high_global.SetGlobalBuffer(stacked, STRIDE * HIDDEN);
    // The low region's base comes in already offset. Deriving it here with a
    // cast produced a tensor whose writes never landed, while the identical
    // write through a tensor built on a plain argument landed every time.
    low_global.SetGlobalBuffer(stacked_low, STRIDE * HIDDEN);
    witness_global.SetGlobalBuffer(witness, TILE);

    for (int64_t row = 0; row < rows_held; ++row) {
        for (int64_t done = 0; done < HIDDEN; done += TILE) {
            const int64_t count = (HIDDEN - done < TILE) ? HIDDEN - done
                                                         : TILE;
            DataCopy(held.narrow, rows_global[row * HIDDEN + done], count);
            AfterLoad();
            Cast(held.half_value, held.narrow, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Cast(held.value, held.half_value, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            SplitHalves(held, count);
            Emit(held.half_value, held.code, held.value, high_global,
                 row * HIDDEN + done, count);
            Emit(held.half_value, held.code, held.work, low_global,
                 row * HIDDEN + done, count);
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
    AfterLoad();
    Cast(into, staging, RoundMode::CAST_NONE, span);
    PipeBarrier<PIPE_V>();
    Muls(into, into, 16.0F, span);
    PipeBarrier<PIPE_V>();
    BeforeLoad();
    DataCopy(staging, product[low], span);
    AfterLoad();
    Cast(spare, staging, RoundMode::CAST_NONE, span);
    PipeBarrier<PIPE_V>();
    Add(into, into, spare, span);
    PipeBarrier<PIPE_V>();
    BeforeLoad();
    DataCopy(spare, bias[at], span);
    AfterLoad();
    Add(into, into, spare, span);
    PipeBarrier<PIPE_V>();
}

/** Recombine, activate, quantise and split one expert's rows.
 *
 * The row maximum is over this expert's own 128 columns and this core owns
 * every one of them, so nothing crosses blocks.
 */
__aicore__ inline void Activate(int64_t rows_held, Workspace& held,
                                __gm__ int32_t* product,
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
    GlobalTensor<int4b_t> middle_high;
    GlobalTensor<int4b_t> middle_low;
    GlobalTensor<float> out_scale_global;
    GlobalTensor<float> bias_global;
    middle_high.SetGlobalBuffer(middle, STRIDE * HALF);
    middle_low.SetGlobalBuffer(Nibbles(middle, STRIDE * HALF),
                               STRIDE * HALF);
    bias_global.SetGlobalBuffer(gu_bias, GATE_UP);
    product_global.SetGlobalBuffer(product, M_TILE * GATE_UP);
    scale_global.SetGlobalBuffer(gu_scale, GATE_UP);
    token_global.SetGlobalBuffer(token_scale, SCALE_SPAN);
    out_scale_global.SetGlobalBuffer(out_scale, SCALE_SPAN);

    DataCopy(scale, scale_global, GATE_UP);
    DataCopy(token, token_global, SCALE_SPAN);
    AfterLoad();
    AfterLoadForScalar();

    for (int64_t row = 0; row < rows_held; ++row) {
        Combine(held.value, held.work, product_global, staging, bias_global,
                0, row * GATE_UP, (STRIDE + row) * GATE_UP, GATE_UP);
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
        ReduceMax(held.rowmax[row * BLOCK_FLOATS], held.work, held.work,
                  HALF, false);
        PipeBarrier<PIPE_ALL>();
    }
    DataSyncBarrier<MemDsbT::UB>();

    for (int64_t row = 0; row < rows_held; ++row) {
        const float largest = held.rowmax.GetValue(row * BLOCK_FLOATS);
        const float quantum = largest / QUANTUM_SPAN;
        Muls(held.value, activation[row * HALF], 1.0F / quantum, HALF);
        PipeBarrier<PIPE_V>();
        // Round to an integer before splitting, so both halves are exact.
        //
        // One cast, both operands float. A `RoundMode` is a tie-break over
        // the DESTINATION type's representable set, not an instruction to
        // take an integer: float to half with CAST_RINT rounds to the nearest
        // half, and only float to float rounds to an integer. The two-step
        // version here computed ToHalf(x), which leaves the low nibble in
        // [-8, 8) as a real -- so whenever x mod 16 reaches 15.5 the stored
        // nibble is +8 and the INT4 cast saturates it to 7, one code unit
        // short. The band that bites is x in [-0.5, 0), and a SwiGLU output
        // measured against its own row maximum crowds exactly there.
        Cast(held.value, held.value, RoundMode::CAST_RINT, HALF);
        PipeBarrier<PIPE_V>();
        SplitHalves(held, HALF);
        Emit(held.half_value, held.code, held.value, middle_high,
             row * HALF, HALF);
        Emit(held.half_value, held.code, held.work, middle_low, row * HALF,
             HALF);
        PipeBarrier<PIPE_ALL>();
        token.SetValue(row, quantum);
    }
    BeforeStore();
    // The scale leaves by MTE: a scalar store to global memory from a vector
    // core does not land.
    DataCopyPad(out_scale_global, token,
                DataCopyExtParams{1, (uint32_t)(SCALE_SPAN * sizeof(float)),
                                  0, 0, 0});
}

/** Recombine and scale one expert's down product into BF16. */
__aicore__ inline void Narrow(int64_t rows_held, Workspace& held,
                              __gm__ int32_t* product,
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
    out_global.SetGlobalBuffer(out, rows_held * HIDDEN);

    DataCopy(row_scale, middle_global, SCALE_SPAN);
    AfterLoadForScalar();

    for (int64_t row = 0; row < rows_held; ++row) {
        const float of_row = row_scale.GetValue(row);
        for (int64_t done = from; done < upto; done += TILE) {
            const int64_t count = (upto - done < TILE) ? upto - done : TILE;
            Combine(held.value, held.work, product_global, staging,
                    bias_global, done, row * HIDDEN + done,
                    (STRIDE + row) * HIDDEN + done, count);
            DataCopy(scale, scale_global[done], count);
            AfterLoad();
            Mul(held.value, held.value, scale, count);
            PipeBarrier<PIPE_V>();
            Muls(held.value, held.value, of_row, count);
            PipeBarrier<PIPE_V>();
            Cast(narrow, held.value, RoundMode::CAST_RINT, count);
            BeforeStore();
            DataCopyPad(out_global[row * HIDDEN + done], narrow,
                        DataCopyExtParams{
                            1, (uint32_t)(count * sizeof(bfloat16_t)), 0, 0,
                            0});
            AfterStore();
        }
    }
}

}  // namespace

extern "C" __global__ __mix__(1, 2) __schedmode__(1)
__aicore__ void routed_int4_fused(
    uint64_t sync_base, __gm__ int8_t* rows, __gm__ int4b_t* stacked,
    __gm__ int4b_t* stacked_low, __gm__ int4b_t* gate_up, __gm__ float* gu_bias, __gm__ float* gu_scale,
    __gm__ int4b_t* down, __gm__ float* down_bias, __gm__ float* down_scale,
    __gm__ float* token_scale,
    __gm__ int32_t* partial, __gm__ int4b_t* middle,
    __gm__ float* middle_scale, __gm__ int32_t* wide, __gm__ bfloat16_t* out,
    __gm__ uint8_t* gu_tiling, __gm__ uint8_t* down_tiling,
    __gm__ int4b_t* witness, __gm__ int32_t* group_count,
    __gm__ int32_t* group_start, GM_ADDR workspace, int64_t experts,
    int64_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    // The `Matmul` library takes scratch from the system workspace, and this
    // kernel never gave it one. Two regions of `stacked` held identical bytes
    // 6144 apart, neither of them a half of the split, which is the shape of
    // a wild write rather than of a lost one.
    SetSysWorkspace(workspace);

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
        GlobalTensor<int32_t> count_global;
        GlobalTensor<int32_t> start_global;
        count_global.SetGlobalBuffer(group_count, MAX_EXPERTS);
        start_global.SetGlobalBuffer(group_start, MAX_EXPERTS);
        DataCopy(held.group, count_global, MAX_EXPERTS);
        DataCopy(held.start, start_global, MAX_EXPERTS);
        AfterLoadForScalar();
        for (int64_t step = 0; step < per_block; ++step) {
            const int64_t expert = first + step;
            if (expert >= experts) {
                break;
            }
            const int64_t held_rows = held.group.GetValue(expert);
            const int64_t held_start = held.start.GetValue(expert);
            if (held_rows == 0) {
                // The handshake is counted, so an empty expert still has to
                // arrive at every flag the cube half sets and waits on.
                if (sub == 0) {
                    CrossCoreSetFlag<0x03, PIPE_MTE3>(SPLIT_READY);
                }
                CrossCoreWaitFlag<0x02, PIPE_MTE2>(GATE_READY);
                if (sub == 0) {
                    CrossCoreSetFlag<0x03, PIPE_MTE3>(MIDDLE_READY);
                }
                CrossCoreWaitFlag<0x02, PIPE_MTE2>(DOWN_READY);
                continue;
            }
            // Mode 3 answers the cube whether one subblock sets or both, so
            // the narrow work stays on subblock zero and the row maximum
            // stays core-local.
            if (sub == 0) {
                Split(held, rows + held_start * HIDDEN,
                      Nibbles(stacked, expert * M_TILE * HIDDEN),
                      Nibbles(stacked_low, expert * M_TILE * HIDDEN),
                      Nibbles(witness, expert * TILE), held_rows);
                Publish();
                CrossCoreSetFlag<0x03, PIPE_MTE3>(SPLIT_READY);
            }
            CrossCoreWaitFlag<0x02, PIPE_MTE2>(GATE_READY);
            if (sub == 0) {
                Activate(held_rows, held,
                         partial + expert * M_TILE * GATE_UP,
                         gu_bias + expert * GATE_UP,
                         gu_scale + expert * GATE_UP,
                         token_scale + expert * SCALE_SPAN,
                         Nibbles(middle, expert * M_TILE * HALF),
                         middle_scale + expert * SCALE_SPAN);
                Publish();
                CrossCoreSetFlag<0x03, PIPE_MTE3>(MIDDLE_READY);
            }
            CrossCoreWaitFlag<0x02, PIPE_MTE2>(DOWN_READY);
            Narrow(held_rows, held, wide + expert * M_TILE * HIDDEN,
                   down_bias + expert * HIDDEN,
                   down_scale + expert * HIDDEN,
                   middle_scale + expert * SCALE_SPAN,
                   out + held_start * HIDDEN, from, from + share);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

void routed_int4_launch(uint32_t blocks, void* stream, uint64_t sync_base,
                        void* rows, void* stacked, void* stacked_low,
                        void* gate_up,
                        float* gu_bias, float* gu_scale, void* down,
                        float* down_bias, float* down_scale,
                        float* token_scale, void* partial, void* middle,
                        float* middle_scale, void* wide, void* out,
                        void* gu_tiling, void* down_tiling, void* witness,
                        void* group_count, void* group_start,
                        void* workspace, int64_t experts)
{
    routed_int4_fused<<<blocks, nullptr, stream>>>(
        sync_base, (__gm__ int8_t*)rows, (__gm__ int4b_t*)stacked,
        (__gm__ int4b_t*)stacked_low, (__gm__ int4b_t*)gate_up, gu_bias,
        gu_scale, (__gm__ int4b_t*)down, down_bias, down_scale, token_scale,
        (__gm__ int32_t*)partial, (__gm__ int4b_t*)middle, middle_scale,
        (__gm__ int32_t*)wide, (__gm__ bfloat16_t*)out,
        (__gm__ uint8_t*)gu_tiling, (__gm__ uint8_t*)down_tiling,
        (__gm__ int4b_t*)witness, (__gm__ int32_t*)group_count,
        (__gm__ int32_t*)group_start, (GM_ADDR)workspace, experts, blocks);
}
