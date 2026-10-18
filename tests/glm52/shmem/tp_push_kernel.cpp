/**
 * One-shot push plus ordered local reduction for the TP16 decode reduction.
 *
 * Each rank publishes its own [rows, hidden] float partial into slot my_pe of
 * every peer's symmetric window and announces it with one signal. Each rank
 * then waits on its own sixteen slots and sums them in rank order, so the
 * result is bit-reproducible. The only synchronisation is a local spin on
 * local flags: no barrier, and no collective.
 *
 * Signals are compared against a per-round value rather than cleared, so a
 * replay needs no reset stage.
 */
#include "tp_push_kernel.h"
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

using namespace AscendC;

constexpr int64_t UB_STAGE_BYTES = 190 * 1024;
constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t SLOT_INT32 = ALIGN_BYTES / static_cast<int64_t>(sizeof(int32_t));
// The put overload stages through the first 190 KiB of UB, so the reduction
// takes the next region and keeps its two buffers clear of it.
constexpr int64_t REDUCE_UB_BASE = 192 * 1024;
constexpr int64_t REDUCE_UB_SPAN = 32 * 1024;

ACLSHMEM_DEVICE int64_t AlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

// stage 0 = push and reduce, 1 = push only, 2 = reduce only. Separating them
// is the only way to tell which half owns the measured time.
ACLSHMEM_DEVICE void tp_push_once(
    __gm__ float* partial, __gm__ float* out, __gm__ uint8_t* window, int rows, int hidden, int round, int stage)
{
    const int64_t core = GetBlockIdx();
    const int64_t cores = GetBlockNum();
    const int64_t my_pe = aclshmem_my_pe();
    const int64_t pe_size = aclshmem_n_pes();
    const int64_t elements = static_cast<int64_t>(rows) * hidden;
    const int64_t stride = AlignUp(elements * sizeof(float), ALIGN_BYTES) / sizeof(float);

    __gm__ float* data = (__gm__ float*)window;
    __gm__ int32_t* status = (__gm__ int32_t*)(window + pe_size * stride * sizeof(float));
    __ubuf__ float* ub = (__ubuf__ float*)(64);

    // One core per peer publishes our partial into that peer's slot for us.
    // Rotating the start spreads the first writes over distinct links.
    if (stage != 2 && core < pe_size) {
        const int64_t peer = (my_pe + core) % pe_size;
        aclshmemx_mte_put_nbi(data + my_pe * stride, partial, ub, UB_STAGE_BYTES, elements, peer, EVENT_ID0);
        aclshmemx_mte_quiet();
        aclshmemx_signal_op(status + my_pe * SLOT_INT32, round + 1, ACLSHMEM_SIGNAL_SET, peer);
    }

    if (stage != 2) {
        // Every core waits for every slot, so all cores leave the wait
        // together and none reaches the reduction before the data is visible.
        for (int64_t pe = 0; pe < pe_size; ++pe) {
            aclshmem_signal_wait_until(status + pe * SLOT_INT32, ACLSHMEM_CMP_EQ, round + 1);
        }
    }

    if (stage != 1) {
        // Scalar accumulation costs 557 us here, because each element pays
        // sixteen separate global reads. Stage the stripe through UB and add
        // with vector instructions instead.
        const int64_t stripe = (elements + cores - 1) / cores;
        const int64_t begin = core * stripe;
        const int64_t span = (begin + stripe <= elements) ? stripe : elements - begin;
        if (span > 0) {
            LocalTensor<float> acc;
            LocalTensor<float> term;
            acc.address_.logicPos = static_cast<uint8_t>(TPosition::VECOUT);
            acc.address_.bufferAddr = REDUCE_UB_BASE;
            acc.address_.dataLen = span * sizeof(float);
            term.address_.logicPos = static_cast<uint8_t>(TPosition::VECOUT);
            term.address_.bufferAddr = REDUCE_UB_BASE + REDUCE_UB_SPAN;
            term.address_.dataLen = span * sizeof(float);
            GlobalTensor<float> source;
            GlobalTensor<float> sink;
            source.SetGlobalBuffer(data + begin, span);
            DataCopy(acc, source, span);
            PipeBarrier<PIPE_ALL>();
            for (int64_t pe = 1; pe < pe_size; ++pe) {
                source.SetGlobalBuffer(data + pe * stride + begin, span);
                DataCopy(term, source, span);
                PipeBarrier<PIPE_ALL>();
                Add(acc, acc, term, span);
                PipeBarrier<PIPE_ALL>();
            }
            sink.SetGlobalBuffer(out + begin, span);
            DataCopy(sink, acc, span);
            PipeBarrier<PIPE_ALL>();
        }
    }
    aclshmemi_sync_core_soft();
}

extern "C" __global__ __aicore__ void tp_push_kernel(
    uint64_t fftsAddr, __gm__ float* partial, __gm__ float* out, __gm__ uint8_t* window, int rows, int hidden,
    int rounds, int round_base, int stage)
{
    util_set_ffts_config(fftsAddr);
    for (int round = 0; round < rounds; ++round) {
        tp_push_once(partial, out, window, rows, hidden, round_base + round, stage);
    }
}

void tp_push_demo(uint32_t block_dim, void* stream, uint64_t fftsAddr, float* partial, float* out, uint8_t* window,
                  int rows, int hidden, int rounds, int round_base, int stage)
{
    tp_push_kernel<<<block_dim, nullptr, stream>>>(fftsAddr, partial, out, window, rows, hidden, rounds, round_base,
                                                   stage);
}
