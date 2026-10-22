#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "front_contract.h"

namespace megakernel_front {
using namespace AscendC;

// Prolog's own operands: int8 activations and weights with an int32 product,
// as `indexer_front`'s `Project<int8_t, int32_t>` uses. The bf16 build keeps
// the earlier comparison available.
#ifdef MEGAKERNEL_FRONT_BF16
using Element = bfloat16_t;
using Accumulator = float;
#else
using Element = int8_t;
using Accumulator = int32_t;
#endif

using Activation = MatmulType<TPosition::GM, CubeFormat::ND, Element, false>;
using Weight = MatmulType<TPosition::GM, CubeFormat::ND, Element, false>;
using Product = MatmulType<TPosition::GM, CubeFormat::ND, Accumulator>;
using Projection = Matmul<Activation, Weight, Product, Product>;

__aicore__ inline void LoadTiling(TCubeTiling& cube, __gm__ uint8_t* bytes)
{
    auto* destination = (uint32_t*)&cube;
    auto* source = (__gm__ uint32_t*)bytes;
    for (uint32_t word = 0; word < sizeof(TCubeTiling) / sizeof(uint32_t);
         ++word) {
        destination[word] = source[word];
    }
}

// One invocation over `slice`. In `SetOrgShape` the weight's row stride is `N`
// and the product's is `Kc`, so a core computes a column window of the
// concatenation while both tensors keep the concatenated layout.
__aicore__ inline void Project(Projection& mm, const Config& config,
                               const Addresses& addresses, const Slice& slice,
                               int64_t column)
{
    if (slice.width == 0) {
        return;
    }
    const int64_t stride = config.weight_stride;
    GlobalTensor<Element> a, b;
    GlobalTensor<Accumulator> c;
    a.SetGlobalBuffer((__gm__ Element*)addresses.activation,
                      config.rows * config.reduction);
    b.SetGlobalBuffer((__gm__ Element*)addresses.weight + column,
                      (config.reduction - 1) * stride + slice.width);
    c.SetGlobalBuffer((__gm__ Accumulator*)addresses.product + column,
                      (config.rows - 1) * stride + slice.width);
    mm.SetOrgShape(config.rows, stride, config.reduction, config.reduction,
                   stride);
    mm.SetSingleShape(config.rows, slice.width, config.reduction);
    mm.SetTensorA(a);
    mm.SetTensorB(b, false);
    mm.IterateAll(c);
    mm.End();
}

// Vector work on the AIV half with no barrier against the cube, so its cost
// appears only once the cube stops covering it. Each pass sums a chunk and the
// total is published, which both makes the work observable and keeps the
// compiler from eliding it.
__aicore__ inline void Reduce(TPipe& pipe, const Config& config,
                              const Addresses& addresses)
{
    if (config.vector_passes == 0) {
        return;
    }
    constexpr int64_t kChunk = 1024;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 2 * kChunk * sizeof(float));
    LocalTensor<float> held = buffer.Get<float>();
    LocalTensor<float> work = held[kChunk];
    GlobalTensor<float> witness;
    witness.SetGlobalBuffer((__gm__ float*)addresses.witness, kMaxWitness);

    const int64_t index = GetBlockIdx();
    float total = 0.0f;
    for (int64_t pass = 0; pass < config.vector_passes; ++pass) {
        for (int64_t chunk = 0; chunk < kVectorChunks; ++chunk) {
            Duplicate(held, static_cast<float>(pass + 1), kChunk);
            PipeBarrier<PIPE_V>();
            Abs(held, held, kChunk);
            PipeBarrier<PIPE_V>();
            ReduceSum(work, held, work, kChunk);
            PipeBarrier<PIPE_ALL>();
            total += work.GetValue(0);
        }
    }
    const int64_t slot = index * kWitnessStride;
    witness.SetValue(slot, total);
    DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(witness[slot]);
}

__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void Kernel(uint64_t sync, Config config, Addresses addresses)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync);
    SetSysWorkspace((GM_ADDR)addresses.system);
    TPipe pipe;
    if ASCEND_IS_AIC {
        TCubeTiling cube;
        LoadTiling(cube, (__gm__ uint8_t*)addresses.tiling);
        Projection mm;
        mm.Init(&cube, &pipe);
        const int64_t index = GetBlockIdx();
#ifdef MEGAKERNEL_FRONT_SPLIT
        for (int64_t part = 0; part < config.part_count; ++part) {
            const Slice slice = SliceFor(config.part_columns[part],
                                         config.blocks, index);
            Project(mm, config, addresses, slice,
                    PartOffset(config, part) + slice.offset);
        }
#else
        const Slice slice = SliceFor(TotalColumns(config), config.blocks,
                                     index);
        Project(mm, config, addresses, slice, slice.offset);
#endif
    }
    if ASCEND_IS_AIV {
        Reduce(pipe, config, addresses);
    }
    // A skewed pipeline pays one grid barrier a step, so these price the step
    // against the work they would let overlap.
    for (int64_t step = 0; step < config.barriers; ++step) {
        SyncAll<false>();
    }
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
}

}  // namespace megakernel_front

extern "C" int megakernel_front_launch(void* stream, uint64_t sync,
    const megakernel_front::Config* config,
    const megakernel_front::Addresses* addresses)
{
    if (config == nullptr || addresses == nullptr ||
        !megakernel_front::Valid(*config, *addresses)) {
        return -1;
    }
    megakernel_front::Kernel<<<config->blocks, nullptr, stream>>>(
        sync, *config, *addresses);
    return 0;
}
