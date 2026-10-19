/**
 * What one cube-to-vector handoff costs on this part, and whether it hides.
 *
 * This generation has no `L0C -> UB` path: the chip difference table gives
 * that row as absent for 1971 while David carries 128 to 256 bits, and
 * `L0C -> L1` exists but `L1 -> UB` does not, so a cube result reaches the
 * vector unit only through global memory. A decode round makes roughly 637
 * such handoffs, and the layer-scale kernel that AMD's design rests on
 * assumes the on-chip path. So the cost of the round trip is the gate on that
 * whole route, and it has never been measured here.
 *
 * Three modes, selected by `mode`, all moving the same bytes:
 *
 *   0  the handoff: AIC writes a tile to GM and raises a flag, AIV waits and
 *      reads it. The serialised cost.
 *   1  no dependency: AIV reads a tile the host filled, while AIC writes
 *      elsewhere. Same traffic, no ordering, so the difference from mode 0 is
 *      what the dependency costs rather than what the bytes cost.
 *   2  the flag alone, no payload. Separates the rendezvous from the transfer.
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t TILE_BYTES = 32 * 1024;
constexpr uint16_t TILE_READY = 6;

extern "C" [[bisheng::core_ratio(1, 2)]] __global__ __aicore__ void handoff_kernel(
    __gm__ float* source, __gm__ float* sink, int mode, int iterations)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> staging;
    pipe.InitBuffer(staging, TILE_BYTES);
    LocalTensor<float> tile = staging.Get<float>();
    const int64_t count = TILE_BYTES / static_cast<int64_t>(sizeof(float));
    const int64_t core = GetBlockIdx();

    GlobalTensor<float> from;
    GlobalTensor<float> to;
    from.SetGlobalBuffer(source + core * count, count);
    to.SetGlobalBuffer(sink + core * count, count);

    for (int step = 0; step < iterations; ++step) {
        if ASCEND_IS_AIC {
            if (mode != 2) {
                DataCopy(to, tile, count);
            }
            CrossCoreSetFlag<0, PIPE_MTE3>(TILE_READY);
        }
        if ASCEND_IS_AIV {
            if (mode == 0 || mode == 2) {
                CrossCoreWaitFlag(TILE_READY);
            }
            if (mode != 2) {
                DataCopy(tile, mode == 0 ? to : from, count);
                SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                Muls(tile, tile, (float)1.0009765625, count);
            }
        }
    }
}

void handoff_demo(uint32_t block_dim, void* stream, float* source, float* sink,
                  int mode, int iterations)
{
    handoff_kernel<<<block_dim, nullptr, stream>>>(source, sink, mode, iterations);
}
