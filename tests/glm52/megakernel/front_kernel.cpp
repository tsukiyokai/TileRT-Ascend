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

// One invocation over `slice`. `SetOrgShape`'s `N` is the weight's row stride
// and `Kc` the product's, so each tensor keeps its own layout: a column window
// of the wide matrix writing the concatenated product, or a core's contiguous
// matrix writing its own product block.
__aicore__ inline void Project(Projection& mm, const Config& config,
                               GM_ADDR hidden, GM_ADDR weight, GM_ADDR product,
                               const Slice& slice,
                               int64_t weight_offset, int64_t weight_stride,
                               int64_t product_offset, int64_t product_stride)
{
    if (slice.width == 0) {
        return;
    }
    GlobalTensor<bfloat16_t> a, b;
    GlobalTensor<float> c;
    a.SetGlobalBuffer((__gm__ bfloat16_t*)hidden, config.rows * kHidden);
    b.SetGlobalBuffer((__gm__ bfloat16_t*)weight + weight_offset,
                      (kHidden - 1) * weight_stride + slice.width);
    c.SetGlobalBuffer((__gm__ float*)product + product_offset,
                      (config.rows - 1) * product_stride + slice.width);
    mm.SetOrgShape(config.rows, weight_stride, kHidden, kHidden, product_stride);
    mm.SetSingleShape(config.rows, slice.width, kHidden);
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
        for (int64_t part = 0; part < kParts; ++part) {
            const Slice slice = SliceFor(kPartColumns[part], config.blocks,
                                         index);
            const int64_t column = PartOffset(part) + slice.offset;
            Project(mm, config, (GM_ADDR)addresses.hidden,
                    (GM_ADDR)addresses.weight, (GM_ADDR)addresses.product,
                    slice, column, kColumns, column, kColumns);
        }
#else
        const Slice slice = SliceFor(kColumns, config.blocks, index);
        Project(mm, config, (GM_ADDR)addresses.hidden,
                (GM_ADDR)addresses.weight, (GM_ADDR)addresses.product,
                slice, slice.offset, kColumns, slice.offset, kColumns);
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
