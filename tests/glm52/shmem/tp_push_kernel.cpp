/**
 * One-shot push plus ordered local reduction for the TP16 decode reduction.
 *
 * Each rank publishes its own [rows, hidden] float partial into its slot on
 * every peer, announces it with one signal, then waits on its own slots and
 * sums them in rank order. The only synchronisation is a local spin on local
 * flags: no barrier and no collective. The fixed order makes the result
 * bit-reproducible, which the trajectory gate needs.
 *
 * Two planes alternate so a round's push cannot overwrite data the previous
 * round is still reducing. The plane and the signal value both come from an
 * absolute round index the host supplies, so successive launches continue the
 * sequence instead of restarting it.
 */
#include "tp_push_kernel.h"
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

using namespace AscendC;

constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t SLOT_INT32 = ALIGN_BYTES / static_cast<int64_t>(sizeof(int32_t));
constexpr int64_t PLANES = 2;
// A3 has 192 KiB of UB in total, so the put's staging area and the
// reduction's two buffers have to be carved out of that one budget rather
// than stacked: placing the reduction at a 192 KiB offset put it out of
// bounds entirely and its stores silently did nothing.
constexpr int64_t UB_TOTAL_BYTES = 192 * 1024;
constexpr int64_t REDUCE_UB_SPAN = 32 * 1024;
constexpr int64_t REDUCE_UB_BASE = 1024;
constexpr int64_t UB_STAGE_BASE = REDUCE_UB_BASE + PLANES * REDUCE_UB_SPAN;
constexpr int64_t UB_STAGE_BYTES = UB_TOTAL_BYTES - UB_STAGE_BASE - 1024;

ACLSHMEM_DEVICE int64_t AlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

struct TpPushLayout {
    __gm__ float* data;
    __gm__ int32_t* status;
    int64_t stride;
    int64_t elements;
};

ACLSHMEM_DEVICE TpPushLayout TpPushPlane(__gm__ uint8_t* window, int rows, int hidden, int64_t pe_size, int64_t plane)
{
    TpPushLayout layout;
    layout.elements = static_cast<int64_t>(rows) * hidden;
    layout.stride = AlignUp(layout.elements * sizeof(float), ALIGN_BYTES) / sizeof(float);
    const int64_t data_floats = PLANES * pe_size * layout.stride;
    layout.data = (__gm__ float*)window + plane * pe_size * layout.stride;
    layout.status = (__gm__ int32_t*)((__gm__ float*)window + data_floats) + plane * pe_size * SLOT_INT32;
    return layout;
}

/** Publish our partial to every peer's slot for us, then announce it. */
ACLSHMEM_DEVICE void TpPushPublish(const TpPushLayout& layout, __gm__ float* partial, int64_t core, int64_t my_pe,
                                   int64_t pe_size, int32_t ticket)
{
    if (core >= pe_size) {
        return;
    }
    // Rotating the start spreads the first writes over distinct links.
    const int64_t peer = (my_pe + core) % pe_size;
    __ubuf__ float* ub = (__ubuf__ float*)(UB_STAGE_BASE);
    aclshmemx_mte_put_nbi(layout.data + my_pe * layout.stride, partial, ub, UB_STAGE_BYTES, layout.elements, peer,
                          EVENT_ID0);
    aclshmemx_mte_quiet();
    aclshmemx_signal_op(layout.status + my_pe * SLOT_INT32, ticket, ACLSHMEM_SIGNAL_SET, peer);
}

/** Sum every rank's slot in rank order over this core's stripe. */
ACLSHMEM_DEVICE void TpPushReduce(const TpPushLayout& layout, __gm__ float* out, int64_t core, int64_t cores,
                                  int64_t pe_size)
{
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
    // A MIX kernel body runs on both core types, and the vector side has
    // GetSubBlockNum() blocks per cube block. Using GetBlockNum() alone made
    // cube and vector duplicate stripes 0..3 while vector blocks 4..7 fell off
    // the end, so the cube side -- which has no vector units -- overwrote the
    // output with rubbish. The reduction belongs to the vector cores only.
    const int64_t core = GetBlockIdx();
    const int64_t cores = GetBlockNum() * GetSubBlockNum();
    const int64_t my_pe = aclshmem_my_pe();
    const int64_t pe_size = aclshmem_n_pes();
    const TpPushLayout layout = TpPushPlane(window, rows, hidden, pe_size, round % PLANES);
    // One ticket per plane visit, so a stale signal from two rounds back can
    // never satisfy this round's wait.
    const int32_t ticket = static_cast<int32_t>(round / PLANES) + 1;

    if (stage != 2) {
        TpPushPublish(layout, partial, core, my_pe, pe_size, ticket);
        for (int64_t pe = 0; pe < pe_size; ++pe) {
            aclshmem_signal_wait_until(layout.status + pe * SLOT_INT32, ACLSHMEM_CMP_EQ, ticket);
        }
    }
    if (stage != 1) {
        if ASCEND_IS_AIV {
            TpPushReduce(layout, out, core, cores, pe_size);
        }
    }
}

extern "C" __global__ __aicore__ void tp_push_kernel(
    uint64_t fftsAddr, __gm__ float* partial, __gm__ float* out, __gm__ uint8_t* window, int rows, int hidden,
    int rounds, int round_base, int stage)
{
    util_set_ffts_config(fftsAddr);
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
