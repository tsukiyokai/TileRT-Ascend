#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "front_contract.h"

namespace megakernel_front {
using namespace AscendC;

using Activation = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using Weight = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using Product = MatmulType<TPosition::GM, CubeFormat::ND, float>;
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
    GlobalTensor<bfloat16_t> a, b;
    GlobalTensor<float> c;
    a.SetGlobalBuffer((__gm__ bfloat16_t*)addresses.activation,
                      config.rows * config.reduction);
    b.SetGlobalBuffer((__gm__ bfloat16_t*)addresses.weight + column,
                      (config.reduction - 1) * stride + slice.width);
    c.SetGlobalBuffer((__gm__ float*)addresses.product + column,
                      (config.rows - 1) * stride + slice.width);
    mm.SetOrgShape(config.rows, stride, config.reduction, config.reduction,
                   stride);
    mm.SetSingleShape(config.rows, slice.width, config.reduction);
    mm.SetTensorA(a);
    mm.SetTensorB(b, false);
    mm.IterateAll(c);
    mm.End();
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
