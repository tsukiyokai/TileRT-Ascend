/** Torch entry for the layer's first dependency level. */
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "front_contract.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {
using namespace megakernel_front;

int64_t SyncBase()
{
    uint64_t address = 0;
    uint32_t length = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&address, &length) == 0 && address != 0,
                "rtGetC2cCtrlAddr failed");
    return static_cast<int64_t>(address);
}

at::Tensor BuildTiling(int64_t rows, int64_t blocks)
{
    const Slice widest = SliceFor(kColumns, blocks, 0);
    auto host = at::empty({static_cast<int64_t>(kTilingStride)},
                          at::dtype(at::kByte));
    const int status = megakernel_front_build_tiling(
        static_cast<int>(rows), static_cast<int>(widest.width),
        static_cast<int>(kColumns), host.data_ptr());
    TORCH_CHECK(status == 0, "megakernel_front_build_tiling failed: ", status);
    return host;
}

void Forward(const at::Tensor& hidden, const at::Tensor& weight,
             const at::Tensor& product, const at::Tensor& tiling,
             const at::Tensor& system, int64_t blocks, int64_t sync_base)
{
    TORCH_CHECK(hidden.dim() == 2 && hidden.size(1) == kHidden,
                "hidden must be [rows, ", kHidden, "]");
    TORCH_CHECK(weight.sizes() == at::IntArrayRef({kHidden, kColumns}),
                "weight must be [", kHidden, ", ", kColumns, "]");
    TORCH_CHECK(product.sizes() == at::IntArrayRef({hidden.size(0), kColumns}),
                "product must be [rows, ", kColumns, "]");
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16
                && weight.scalar_type() == at::kBFloat16
                && product.scalar_type() == at::kFloat,
                "bf16 operands and a float product");
    for (const at::Tensor& owner : {hidden, weight, product, tiling, system}) {
        TORCH_CHECK(owner.is_contiguous() && owner.device().is_privateuseone(),
                    "every owner must be contiguous and on the device");
    }

    const Config config{hidden.size(0), blocks};
    const Addresses addresses{
        reinterpret_cast<uint64_t>(hidden.data_ptr()),
        reinterpret_cast<uint64_t>(weight.data_ptr()),
        reinterpret_cast<uint64_t>(product.data_ptr()),
        reinterpret_cast<uint64_t>(tiling.data_ptr()),
        reinterpret_cast<uint64_t>(system.data_ptr())};
    TORCH_CHECK(Valid(config, addresses), "invalid configuration or addresses");

    const c10::DeviceGuard guard(hidden.device());
    const int status = megakernel_front_launch(
        c10_npu::getCurrentNPUStream().stream(),
        static_cast<uint64_t>(sync_base), &config, &addresses);
    TORCH_CHECK(status == 0, "megakernel_front_launch failed: ", status);
}

TORCH_LIBRARY_FRAGMENT(megakernel_front, m)
{
    m.def("sync_base() -> int");
    m.def("build_tiling(int rows, int blocks) -> Tensor");
    m.def("forward(Tensor hidden, Tensor weight, Tensor product, "
          "Tensor tiling, Tensor system, int blocks, int sync_base) -> ()");
}

TORCH_LIBRARY_IMPL(megakernel_front, CompositeExplicitAutograd, m)
{
    m.impl("sync_base", TORCH_FN(SyncBase));
    m.impl("build_tiling", TORCH_FN(BuildTiling));
}

TORCH_LIBRARY_IMPL(megakernel_front, PrivateUse1, m)
{
    m.impl("forward", TORCH_FN(Forward));
}

}  // namespace
