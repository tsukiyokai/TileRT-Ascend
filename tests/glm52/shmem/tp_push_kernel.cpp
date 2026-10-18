/**
 * One-sided TP reduction, built on the structure the SDK's allgather uses.
 *
 * Each rank writes its own partial into its own symmetric window and raises
 * its own flag; consumers then pull each peer's flag into UB with a
 * non-blocking get and read that peer's data once the flag says it is there.
 * Nothing blocks on a remote condition, which is what an untimed
 * `aclshmem_signal_wait_until` does and what earned aicore 507015 here.
 *
 * The reduction sums the partials in rank order, so the result is
 * bit-reproducible — the trajectory gate needs that, and EP16 failed it
 * precisely by changing a summation's association.
 */
#include "tp_push_kernel.h"
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

using namespace AscendC;

// The window's head is the flag region and the payload follows, as in
// examples/allgather. Slots are spaced 16 int32 apart so no two share a cache
// line; cann/shmem PR 913 widened combine's for the same reason.
constexpr int64_t SIGNAL_SLOT_INT32 = 16;
constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t PLANES = 2;

// A3 has 192 KiB of UB in total, shared by every consumer in the kernel.
constexpr int64_t FLAG_UB_BASE = 32;
constexpr int64_t STAGE_UB_BASE = 1024;
constexpr int64_t STAGE_UB_BYTES = 96 * 1024;
constexpr int64_t REDUCE_UB_BASE = 112 * 1024;
constexpr int64_t REDUCE_UB_SPAN = 32 * 1024;

ACLSHMEM_DEVICE int64_t AlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

struct TpPushLayout {
    __gm__ int32_t* flags;
    __gm__ float* data;
    int64_t stride;
    int64_t elements;
};

ACLSHMEM_DEVICE TpPushLayout TpPushPlane(__gm__ uint8_t* window, int rows, int hidden, int64_t pe_size, int64_t plane)
{
    TpPushLayout layout;
    layout.elements = static_cast<int64_t>(rows) * hidden;
    layout.stride = AlignUp(layout.elements * sizeof(float), ALIGN_BYTES) / sizeof(float);
    const int64_t flag_int32 = PLANES * pe_size * SIGNAL_SLOT_INT32;
    layout.flags = (__gm__ int32_t*)window + plane * pe_size * SIGNAL_SLOT_INT32;
    layout.data = (__gm__ float*)((__gm__ int32_t*)window + flag_int32) + plane * pe_size * layout.stride;
    return layout;
}

/** Write our partial into our own slot, then raise our own flag. */
ACLSHMEM_DEVICE void TpPushPublish(const TpPushLayout& layout, __gm__ float* partial, int64_t my_pe, int32_t ticket)
{
    __ubuf__ float* stage = (__ubuf__ float*)(STAGE_UB_BASE);
    aclshmemx_mte_put_nbi(layout.data + my_pe * layout.stride, partial, stage, STAGE_UB_BYTES, layout.elements, my_pe,
                          EVENT_ID0);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    aclshmemx_signal_op(layout.flags + my_pe * SIGNAL_SLOT_INT32, ticket, ACLSHMEM_SIGNAL_SET, my_pe);
}

/**
 * Poll every peer's flag without blocking on any one of them, then sum this
 * core's stripe in rank order once all of them have been observed.
 */
ACLSHMEM_DEVICE void TpPushConsume(const TpPushLayout& layout, __gm__ float* out, int64_t core, int64_t cores,
                                   int64_t pe_size, int32_t ticket)
{
    __ubuf__ int32_t* seen = (__ubuf__ int32_t*)(FLAG_UB_BASE);
    for (int64_t pe = 0; pe < pe_size; ++pe) {
        seen[pe * 8] = 0;
    }
    int64_t ready = 0;
    while (ready < pe_size) {
        for (int64_t pe = 0; pe < pe_size; ++pe) {
            if (seen[pe * 8] == ticket) {
                continue;
            }
            aclshmem_int32_get_nbi(seen + pe * 8, layout.flags + pe * SIGNAL_SLOT_INT32, 1, pe);
            PipeBarrier<PIPE_ALL>();
            if (seen[pe * 8] == ticket) {
                ++ready;
            }
        }
    }

    const int64_t stripe = (layout.elements + cores - 1) / cores;
    const int64_t begin = core * stripe;
    if (begin >= layout.elements) {
        return;
    }
    const int64_t span = (begin + stripe <= layout.elements) ? stripe : layout.elements - begin;

    LocalTensor<float> acc;
    LocalTensor<float> term;
    acc.address_.logicPos = static_cast<uint8_t>(TPosition::VECOUT);
    acc.address_.bufferAddr = REDUCE_UB_BASE;
    acc.address_.dataLen = REDUCE_UB_SPAN;
    term.address_.logicPos = static_cast<uint8_t>(TPosition::VECOUT);
    term.address_.bufferAddr = REDUCE_UB_BASE + REDUCE_UB_SPAN;
    term.address_.dataLen = REDUCE_UB_SPAN;

    const int64_t chunk = REDUCE_UB_SPAN / static_cast<int64_t>(sizeof(float));
    for (int64_t offset = 0; offset < span; offset += chunk) {
        const int64_t count = (span - offset < chunk) ? span - offset : chunk;
        GlobalTensor<float> source;
        source.SetGlobalBuffer(layout.data + begin + offset, count);
        DataCopy(acc, source, count);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        for (int64_t pe = 1; pe < pe_size; ++pe) {
            source.SetGlobalBuffer(layout.data + pe * layout.stride + begin + offset, count);
            DataCopy(term, source, count);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            Add(acc, acc, term, count);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        GlobalTensor<float> sink;
        sink.SetGlobalBuffer(out + begin + offset, count);
        DataCopy(sink, acc, count);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

ACLSHMEM_DEVICE void tp_push_once(
    __gm__ float* partial, __gm__ float* out, __gm__ uint8_t* window, int rows, int hidden, int64_t round, int stage)
{
    const int64_t core = GetBlockIdx();
    const int64_t cores = GetBlockNum() * GetSubBlockNum();
    const int64_t my_pe = aclshmem_my_pe();
    const int64_t pe_size = aclshmem_n_pes();
    const TpPushLayout layout = TpPushPlane(window, rows, hidden, pe_size, round % PLANES);
    const int32_t ticket = static_cast<int32_t>(round / PLANES) + 1;

    // Producer and consumer cores are disjoint, as in examples/allgather, so
    // no core waits on work it owes itself.
    if (core == 0) {
        if (stage != 2) {
            TpPushPublish(layout, partial, my_pe, ticket);
        }
    } else if (stage != 1) {
        TpPushConsume(layout, out, core - 1, cores - 1, pe_size, ticket);
    }
}

extern "C" __global__ __aicore__ void tp_push_kernel(
    uint64_t fftsAddr, __gm__ float* partial, __gm__ float* out, __gm__ uint8_t* window, int rows, int hidden,
    int rounds, int round_base, int stage)
{
    util_set_ffts_config(fftsAddr);
    // A MIX body runs on both core types and the cube side has no vector
    // units, so its stores would corrupt the output.
    if ASCEND_IS_AIC {
        return;
    }
    for (int round = 0; round < rounds; ++round) {
        tp_push_once(partial, out, window, rows, hidden, static_cast<int64_t>(round_base) + round, stage);
    }
}

void tp_push_demo(uint32_t block_dim, void* stream, uint64_t fftsAddr, float* partial, float* out, uint8_t* window,
                  int rows, int hidden, int rounds, int round_base, int stage)
{
    tp_push_kernel<<<block_dim, nullptr, stream>>>(fftsAddr, partial, out, window, rows, hidden, rounds, round_base,
                                                   stage);
}
