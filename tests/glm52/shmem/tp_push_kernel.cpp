/**
 * One-sided TP all-reduce: push-to-all, one cross-PE sync, ordered local sum.
 *
 * Every rank pushes its whole partial into the slot it owns in every peer's
 * window, so after one synchronization each rank holds all pe_size partials
 * locally. Each core then sums its own stripe over those slots in rank order,
 * which keeps the result bit-reproducible.
 *
 * There are no flags and no polling. A reduction needs no producer/consumer
 * core split: the split exists in examples/allgather because its consumers must
 * pull peers' payloads through a staging window, whereas here the push already
 * delivered every payload before the sync released anyone.
 *
 * Two planes alternate, which bounds how far a rank may run ahead with a single
 * sync per round: reaching round r+2 requires passing the sync of round r+1,
 * which every peer reaches only after finishing its round-r reduction.
 */
#include "tp_push_kernel.h"
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

using namespace AscendC;

constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t PLANES = 2;
// The window head keeps the region the host still reserves for flags; the
// payload planes follow it.
constexpr int64_t SIGNAL_SLOT_INT32 = 16;
constexpr int64_t MAX_SLICES = 32;

// UB comes from a TPipe so the three buffers cannot overlap each other: the
// staging area for the push, and the accumulator and addend for the sum.
constexpr int64_t STAGE_UB_BYTES = 32 * 1024;
constexpr int64_t REDUCE_UB_BYTES = 32 * 1024;

ACLSHMEM_DEVICE int64_t AlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

struct TpPushLayout {
    __gm__ float* data;
    int64_t stride;
    int64_t elements;
};

ACLSHMEM_DEVICE TpPushLayout TpPushPlane(__gm__ uint8_t* window, int rows, int hidden, int64_t pe_size, int64_t plane)
{
    TpPushLayout layout;
    layout.elements = static_cast<int64_t>(rows) * hidden;
    layout.stride = AlignUp(layout.elements * sizeof(float), ALIGN_BYTES) / sizeof(float);
    const int64_t head_int32 = PLANES * MAX_SLICES * SIGNAL_SLOT_INT32;
    layout.data = (__gm__ float*)((__gm__ int32_t*)window + head_int32) + plane * pe_size * layout.stride;
    return layout;
}

/** Core `core` sends the whole partial to one peer, so a fan-out of pe_size costs one put per core. */
ACLSHMEM_DEVICE void TpPushPublish(const TpPushLayout& layout, __gm__ float* partial, int64_t core, int64_t my_pe,
                                   int64_t pe_size, __ubuf__ float* stage)
{
    if (core >= pe_size) {
        return;
    }
    // Staggering the start spreads the first writes of different ranks over distinct links.
    const int64_t peer = (my_pe + core) % pe_size;
    aclshmemx_mte_put_nbi(layout.data + my_pe * layout.stride, partial, stage, STAGE_UB_BYTES, layout.elements, peer,
                          EVENT_ID0);
}

/** Sum every rank's slot in rank order over this core's stripe. */
ACLSHMEM_DEVICE void TpPushReduce(const TpPushLayout& layout, __gm__ float* out, int64_t core, int64_t cores,
                                  int64_t pe_size, const LocalTensor<float>& acc, const LocalTensor<float>& term)
{
    const int64_t stripe = AlignUp((layout.elements + cores - 1) / cores, ALIGN_BYTES / sizeof(float));
    const int64_t begin = core * stripe;
    if (begin >= layout.elements) {
        return;
    }
    const int64_t span = (begin + stripe <= layout.elements) ? stripe : layout.elements - begin;
    const int64_t chunk = REDUCE_UB_BYTES / static_cast<int64_t>(sizeof(float));

    for (int64_t offset = 0; offset < span; offset += chunk) {
        const int64_t count = (span - offset < chunk) ? span - offset : chunk;
        GlobalTensor<float> source;
        source.SetGlobalBuffer(layout.data + begin + offset, count);
        DataCopy(acc, source, count);
        PipeBarrier<PIPE_ALL>();
        for (int64_t pe = 1; pe < pe_size; ++pe) {
            source.SetGlobalBuffer(layout.data + pe * layout.stride + begin + offset, count);
            DataCopy(term, source, count);
            PipeBarrier<PIPE_ALL>();
            Add(acc, acc, term, count);
            PipeBarrier<PIPE_ALL>();
        }
        GlobalTensor<float> sink;
        sink.SetGlobalBuffer(out + begin + offset, count);
        DataCopy(sink, acc, count);
        PipeBarrier<PIPE_ALL>();
    }
}

/**
 * Publish by storing straight into the peer's address.
 *
 * aclshmem_ptr maps our symmetric pointer to the same offset in the peer's
 * window, after which a remote write is an ordinary MTE store. This is how the
 * shmem_mega_moe example communicates, and it skips the library's staging
 * protocol and its trailing quiet.
 */
ACLSHMEM_DEVICE void TpPushPublishDirect(const TpPushLayout& layout, __gm__ float* partial, int64_t core,
                                         int64_t my_pe, int64_t pe_size, const LocalTensor<float>& stage_tensor)
{
    if (core >= pe_size) {
        return;
    }
    const int64_t peer = (my_pe + core) % pe_size;
    __gm__ float* remote = (__gm__ float*)aclshmem_ptr((__gm__ void*)layout.data, peer);
    const int64_t chunk = STAGE_UB_BYTES / static_cast<int64_t>(sizeof(float));

    for (int64_t offset = 0; offset < layout.elements; offset += chunk) {
        const int64_t count = (layout.elements - offset < chunk) ? layout.elements - offset : chunk;
        GlobalTensor<float> source;
        source.SetGlobalBuffer(partial + offset, count);
        DataCopy(stage_tensor, source, count);
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<float> sink;
        sink.SetGlobalBuffer(remote + my_pe * layout.stride + offset, count);
        DataCopy(sink, stage_tensor, count);
        PipeBarrier<PIPE_ALL>();
    }
}

/**
 * Publish with the dependency named rather than a whole-pipeline barrier.
 *
 * Two staging buffers and two events: MTE2 fills one while MTE3 drains the
 * other, which is how QuantizeLocalTokens in examples/shmem_mega_moe keeps the
 * load and store units both busy.
 */
ACLSHMEM_DEVICE void TpPushPublishPipelined(const TpPushLayout& layout, __gm__ float* partial, int64_t core,
                                            int64_t my_pe, int64_t pe_size, const LocalTensor<float>& first,
                                            const LocalTensor<float>& second)
{
    if (core >= pe_size) {
        return;
    }
    const int64_t peer = (my_pe + core) % pe_size;
    __gm__ float* remote = (__gm__ float*)aclshmem_ptr((__gm__ void*)layout.data, peer);
    __gm__ float* slot = remote + my_pe * layout.stride;
    const int64_t chunk = STAGE_UB_BYTES / static_cast<int64_t>(sizeof(float));
    const event_t events[2] = {EVENT_ID0, EVENT_ID1};

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(events[0]);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(events[1]);
    int64_t index = 0;
    for (int64_t offset = 0; offset < layout.elements; offset += chunk) {
        const int64_t count = (layout.elements - offset < chunk) ? layout.elements - offset : chunk;
        const LocalTensor<float>& buffer = (index % 2 == 0) ? first : second;
        const event_t event = events[index % 2];
        ++index;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        GlobalTensor<float> source;
        source.SetGlobalBuffer(partial + offset, count);
        DataCopy(buffer, source, count);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);

        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        GlobalTensor<float> sink;
        sink.SetGlobalBuffer(slot + offset, count);
        DataCopy(sink, buffer, count);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(events[0]);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(events[1]);
}

/**
 * Sum every rank's slot over this core's stripe with the loads pipelined.
 *
 * The addend is double buffered so peer p+1's load is in flight while peer p
 * is being added, and the order of the additions is still ascending rank, so
 * the result stays bit-reproducible.
 */
ACLSHMEM_DEVICE void TpPushReducePipelined(const TpPushLayout& layout, __gm__ float* out, int64_t core, int64_t cores,
                                           int64_t pe_size, const LocalTensor<float>& acc,
                                           const LocalTensor<float>& first, const LocalTensor<float>& second)
{
    const int64_t stripe = AlignUp((layout.elements + cores - 1) / cores, ALIGN_BYTES / sizeof(float));
    const int64_t begin = core * stripe;
    if (begin >= layout.elements) {
        return;
    }
    const int64_t span = (begin + stripe <= layout.elements) ? stripe : layout.elements - begin;
    const int64_t chunk = REDUCE_UB_BYTES / static_cast<int64_t>(sizeof(float));
    const event_t events[2] = {EVENT_ID0, EVENT_ID1};

    for (int64_t offset = 0; offset < span; offset += chunk) {
        const int64_t count = (span - offset < chunk) ? span - offset : chunk;
        GlobalTensor<float> source;
        source.SetGlobalBuffer(layout.data + begin + offset, count);
        DataCopy(acc, source, count);
        SetFlag<AscendC::HardEvent::MTE2_V>(events[0]);
        WaitFlag<AscendC::HardEvent::MTE2_V>(events[0]);

        // Prime the pipeline with the first peer; each iteration then issues
        // the next load before waiting on the current one.
        SetFlag<AscendC::HardEvent::V_MTE2>(events[1]);
        source.SetGlobalBuffer(layout.data + layout.stride + begin + offset, count);
        DataCopy(first, source, count);
        SetFlag<AscendC::HardEvent::MTE2_V>(events[0]);

        for (int64_t pe = 1; pe < pe_size; ++pe) {
            const int64_t current = (pe - 1) % 2;
            const int64_t next = pe % 2;
            const LocalTensor<float>& ready = (current == 0) ? first : second;
            const LocalTensor<float>& incoming = (next == 0) ? first : second;

            if (pe + 1 < pe_size) {
                WaitFlag<AscendC::HardEvent::V_MTE2>(events[next]);
                source.SetGlobalBuffer(layout.data + (pe + 1) * layout.stride + begin + offset, count);
                DataCopy(incoming, source, count);
                SetFlag<AscendC::HardEvent::MTE2_V>(events[next]);
            }
            WaitFlag<AscendC::HardEvent::MTE2_V>(events[current]);
            Add(acc, acc, ready, count);
            SetFlag<AscendC::HardEvent::V_MTE2>(events[current]);
        }
        WaitFlag<AscendC::HardEvent::V_MTE2>(events[0]);
        WaitFlag<AscendC::HardEvent::V_MTE2>(events[1]);

        SetFlag<AscendC::HardEvent::V_MTE3>(events[0]);
        WaitFlag<AscendC::HardEvent::V_MTE3>(events[0]);
        GlobalTensor<float> sink;
        sink.SetGlobalBuffer(out + begin + offset, count);
        DataCopy(sink, acc, count);
        SetFlag<AscendC::HardEvent::MTE3_V>(events[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(events[0]);
    }
}

ACLSHMEM_DEVICE __gm__ int32_t* TpPushFlags(__gm__ uint8_t* window, int64_t plane)
{
    return (__gm__ int32_t*)window + plane * MAX_SLICES * SIGNAL_SLOT_INT32;
}

/**
 * Tell peer `core` that our slot in its window is complete.
 *
 * The flag must not become visible before the data it describes, so the store
 * pipeline is drained first -- a fence on one queue rather than a barrier over
 * every core and rank.
 */
ACLSHMEM_DEVICE void TpPushSignal(__gm__ uint8_t* window, int64_t core, int64_t my_pe, int64_t pe_size,
                                  int64_t plane, int32_t token)
{
    if (core >= pe_size) {
        return;
    }
    const int64_t peer = (my_pe + core) % pe_size;
    aclshmemx_signal_op(TpPushFlags(window, plane) + my_pe * SIGNAL_SLOT_INT32, token, ACLSHMEM_SIGNAL_SET,
                        (int)peer);
}

/**
 * Sum every rank's slot, each add gated only by its own producer's flag.
 *
 * The additions stay in ascending rank order so the result is bit-reproducible,
 * but the wait for peer p happens immediately before peer p's add rather than
 * after every peer has arrived.
 */
ACLSHMEM_DEVICE void TpPushReduceSignalled(const TpPushLayout& layout, __gm__ float* out, __gm__ uint8_t* window,
                                           int64_t core, int64_t cores, int64_t pe_size, int64_t plane, int32_t token,
                                           const LocalTensor<float>& acc, const LocalTensor<float>& term)
{
    const int64_t stripe = AlignUp((layout.elements + cores - 1) / cores, ALIGN_BYTES / sizeof(float));
    const int64_t begin = core * stripe;
    if (begin >= layout.elements) {
        return;
    }
    const int64_t span = (begin + stripe <= layout.elements) ? stripe : layout.elements - begin;
    const int64_t chunk = REDUCE_UB_BYTES / static_cast<int64_t>(sizeof(float));
    __gm__ int32_t* flags = TpPushFlags(window, plane);

    for (int64_t offset = 0; offset < span; offset += chunk) {
        const int64_t count = (span - offset < chunk) ? span - offset : chunk;
        for (int64_t pe = 0; pe < pe_size; ++pe) {
            if (offset == 0) {
                aclshmem_signal_wait_until(flags + pe * SIGNAL_SLOT_INT32, ACLSHMEM_CMP_GE, token);
            }
            GlobalTensor<float> source;
            source.SetGlobalBuffer(layout.data + pe * layout.stride + begin + offset, count);
            if (pe == 0) {
                DataCopy(acc, source, count);
                SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                continue;
            }
            DataCopy(term, source, count);
            SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            Add(acc, acc, term, count);
            SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<float> sink;
        sink.SetGlobalBuffer(out + begin + offset, count);
        DataCopy(sink, acc, count);
        SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
}

/**
 * Check one element of this core's stripe and count a mismatch.
 *
 * The position rotates with the round so 400 rounds cover 400 offsets, which
 * the host's single end-of-run comparison cannot do: with two planes it only
 * ever sees the last round of each.
 */
ACLSHMEM_DEVICE void TpPushProbe(__gm__ float* out, __gm__ int32_t* errors, int64_t core, int64_t cores,
                                 int64_t elements, int64_t round, float expected)
{
    const int64_t stripe = AlignUp((elements + cores - 1) / cores, ALIGN_BYTES / sizeof(float));
    const int64_t begin = core * stripe;
    if (begin >= elements) {
        return;
    }
    const int64_t span = (begin + stripe <= elements) ? stripe : elements - begin;
    GlobalTensor<float> probe;
    probe.SetGlobalBuffer(out + begin + round % span, 1);
    dcci_cacheline((__gm__ uint8_t*)(out + begin + round % span));
    if (probe.GetValue(0) != expected) {
        GlobalTensor<int32_t> counter;
        counter.SetGlobalBuffer(errors + core * SIGNAL_SLOT_INT32, 1);
        counter.SetValue(0, counter.GetValue(0) + 1);
        dcci_cacheline((__gm__ uint8_t*)(errors + core * SIGNAL_SLOT_INT32));
    }
}

/**
 * Publish a bf16 partial, pipelined, straight into the peer's window.
 *
 * Half the bytes of the float32 path for the same values, since the sum is
 * accumulated in float inside the reduction rather than on the wire.
 */
ACLSHMEM_DEVICE void TpPushPublishBf16(const TpPushLayout& layout, __gm__ bfloat16_t* partial, int64_t core,
                                       int64_t my_pe, int64_t pe_size, const LocalTensor<bfloat16_t>& first,
                                       const LocalTensor<bfloat16_t>& second)
{
    if (core >= pe_size) {
        return;
    }
    const int64_t peer = (my_pe + core) % pe_size;
    __gm__ bfloat16_t* remote = (__gm__ bfloat16_t*)aclshmem_ptr((__gm__ void*)layout.data, peer);
    __gm__ bfloat16_t* slot = remote + my_pe * layout.stride * 2;
    const int64_t chunk = STAGE_UB_BYTES / static_cast<int64_t>(sizeof(bfloat16_t));
    const event_t events[2] = {EVENT_ID0, EVENT_ID1};

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(events[0]);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(events[1]);
    int64_t index = 0;
    for (int64_t offset = 0; offset < layout.elements; offset += chunk) {
        const int64_t count = (layout.elements - offset < chunk) ? layout.elements - offset : chunk;
        const LocalTensor<bfloat16_t>& buffer = (index % 2 == 0) ? first : second;
        const event_t event = events[index % 2];
        ++index;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        GlobalTensor<bfloat16_t> source;
        source.SetGlobalBuffer(partial + offset, count);
        DataCopy(buffer, source, count);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);

        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        GlobalTensor<bfloat16_t> sink;
        sink.SetGlobalBuffer(slot + offset, count);
        DataCopy(sink, buffer, count);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(events[0]);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(events[1]);
}

/**
 * Sum every rank's bf16 slot in float, in rank order, and store bf16.
 *
 * The widening and narrowing happen in UB around work the reduction already
 * does, so the host no longer needs a Cast on either side.
 */
ACLSHMEM_DEVICE void TpPushReduceBf16(const TpPushLayout& layout, __gm__ bfloat16_t* out, int64_t core, int64_t cores,
                                      int64_t pe_size, const LocalTensor<float>& acc, const LocalTensor<float>& term,
                                      const LocalTensor<bfloat16_t>& narrow)
{
    const int64_t stripe = AlignUp((layout.elements + cores - 1) / cores, ALIGN_BYTES / sizeof(bfloat16_t));
    const int64_t begin = core * stripe;
    if (begin >= layout.elements) {
        return;
    }
    const int64_t span = (begin + stripe <= layout.elements) ? stripe : layout.elements - begin;
    const int64_t chunk = REDUCE_UB_BYTES / static_cast<int64_t>(sizeof(float));

    for (int64_t offset = 0; offset < span; offset += chunk) {
        const int64_t count = (span - offset < chunk) ? span - offset : chunk;
        for (int64_t pe = 0; pe < pe_size; ++pe) {
            GlobalTensor<bfloat16_t> source;
            source.SetGlobalBuffer((__gm__ bfloat16_t*)layout.data + pe * layout.stride * 2 + begin + offset, count);
            DataCopy(narrow, source, count);
            SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            if (pe == 0) {
                Cast(acc, narrow, AscendC::RoundMode::CAST_NONE, count);
            } else {
                Cast(term, narrow, AscendC::RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, term, count);
            }
            PipeBarrier<PIPE_V>();
        }
        Cast(narrow, acc, AscendC::RoundMode::CAST_RINT, count);
        SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<bfloat16_t> sink;
        sink.SetGlobalBuffer(out + begin + offset, count);
        DataCopy(sink, narrow, count);
        SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
}

ACLSHMEM_DEVICE void tp_push_once(__gm__ float* partial, __gm__ float* out, __gm__ uint8_t* window, int rows,
                                  int hidden, int64_t round, int stage, __ubuf__ float* stage_ub,
                                  __gm__ int32_t* errors, const LocalTensor<float>& stage_tensor,
                                  const LocalTensor<float>& stage_second, const LocalTensor<float>& term_second,
                                  const LocalTensor<float>& acc, const LocalTensor<float>& term)
{
    const int64_t core = GetBlockIdx();
    const int64_t cores = GetBlockNum();
    const int64_t my_pe = aclshmem_my_pe();
    const int64_t pe_size = aclshmem_n_pes();
    const int64_t plane = (stage == 9) ? 0 : round % PLANES;
    const TpPushLayout layout = TpPushPlane(window, rows, hidden, pe_size, plane);

    // stage 1 is the push half alone and stage 2 the sum half alone; neither waits on a peer, so
    // isolating a half can expose a cost but can no longer deadlock.
    // stage 3 moves nothing and only synchronises, which measures the floor the
    // 17.5 us push half is sitting on: it decides whether reduce-scatter plus
    // all-gather, eight times fewer bytes for a second sync, can win.
    if (stage == 3) {
        aclshmem_quiet();
        aclshmemx_sync_vec_all();
        return;
    }
    if (stage == 4) {
        TpPushPublishDirect(layout, partial, core, my_pe, pe_size, stage_tensor);
    } else if (stage == 5 || stage == 6 || stage == 7 || stage == 8 || stage == 9) {
        TpPushPublishPipelined(layout, partial, core, my_pe, pe_size, stage_tensor, stage_second);
    } else if (stage != 2) {
        TpPushPublish(layout, partial, core, my_pe, pe_size, stage_ub);
        // Drain this core's remote stores before anything claims they landed.
        if (stage >= 4) {
            SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        }
        if (stage == 9) {
        TpPushReducePipelined(layout, out, core, cores, pe_size, acc, term, term_second);
        // No peer may begin its next push until every peer has finished
        // reading, which is what makes a single plane safe.
        aclshmemx_sync_vec_all();
    } else if (stage == 10) {
        // Stage 10 orders and synchronises inside its own branch.
    } else if (stage == 8) {
            TpPushSignal(window, core, my_pe, pe_size, round % PLANES, (int32_t)round + 1);
        } else {
            if (stage != 6) {
                aclshmem_quiet();
            }
            aclshmemx_sync_vec_all();
        }
    }
    if (stage == 8) {
        TpPushReduceSignalled(layout, out, window, core, cores, pe_size, round % PLANES, (int32_t)round + 1, acc, term);
        TpPushProbe(out, errors, core, cores, layout.elements, round, (float)(pe_size * (pe_size - 1) / 2));
    } else if (stage == 7) {
        TpPushReducePipelined(layout, out, core, cores, pe_size, acc, term, term_second);
    } else if (stage != 1) {
        TpPushReduce(layout, out, core, cores, pe_size, acc, term);
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void tp_push_kernel(
    uint64_t fftsAddr, __gm__ float* partial, __gm__ float* out, __gm__ uint8_t* window, __gm__ int32_t* errors,
    int rows, int hidden, int rounds, int round_base, int stage)
{
    util_set_ffts_config(fftsAddr);
    // One carve-up of UB for the whole launch; a TPipe inside the round loop would redo it every round.
    TPipe pipe;
    TBuf<TPosition::VECCALC> staging;
    TBuf<TPosition::VECCALC> accumulator;
    TBuf<TPosition::VECCALC> addend;
    pipe.InitBuffer(staging, STAGE_UB_BYTES);
    TBuf<TPosition::VECCALC> staging_second;
    pipe.InitBuffer(staging_second, STAGE_UB_BYTES);
    TBuf<TPosition::VECCALC> addend_second;
    pipe.InitBuffer(addend_second, REDUCE_UB_BYTES);
    pipe.InitBuffer(accumulator, REDUCE_UB_BYTES);
    pipe.InitBuffer(addend, REDUCE_UB_BYTES);
    LocalTensor<float> stage_tensor = staging.Get<float>();
    LocalTensor<float> stage_second = staging_second.Get<float>();
    LocalTensor<float> term_second = addend_second.Get<float>();
    LocalTensor<float> acc = accumulator.Get<float>();
    LocalTensor<float> term = addend.Get<float>();
    __ubuf__ float* stage_ub = (__ubuf__ float*)stage_tensor.GetPhyAddr();

    for (int round = 0; round < rounds; ++round) {
        tp_push_once(partial, out, window, rows, hidden, static_cast<int64_t>(round_base) + round, stage, stage_ub,
                     errors, stage_tensor, stage_second, term_second, acc,
                     term);
    }
}

void tp_push_demo(uint32_t block_dim, void* stream, uint64_t fftsAddr, float* partial, float* out, uint8_t* window,
                  int32_t* errors, int rows, int hidden, int rounds, int round_base, int stage)
{
    tp_push_kernel<<<block_dim, nullptr, stream>>>(fftsAddr, partial, out, window, errors, rows, hidden, rounds,
                                                   round_base, stage);
}
