/**
 * The INT4 routed-expert kernel as a torch operator, so the graph can call it.
 *
 * A vendor-package override cannot express this kernel: it replaces two
 * operators reached through different packages, and swapping an `.o` behind a
 * fixed prototype preserves the operator boundary by construction -- the
 * second grouped matmul would still be issued, and the global-memory round
 * trip between them, which is where the fusion win lives, would survive. So
 * the caller is what changes, and the caller is reachable: the MoE body is
 * already an opaque custom op, `torch.ops.vllm.moe_forward_shared`, with no
 * grouped matmul in the compiled graph at all, which means the quant method
 * is re-entered on every eager invocation and once during capture. Replacing
 * a method there needs no `register_fake` and no graph-break reasoning.
 *
 * The INT4 kernel rather than the INT8 one, for two reasons that point the
 * same way. The checkpoint's weights are packed INT4 with a nonzero per
 * output channel `scale_bias`, and this kernel takes both -- packed codes and
 * the correction -- where the INT8 kernel takes neither, so it would need the
 * weights widened and would still compute the wrong thing. And widening is
 * what made the INT8 path a per-layer experiment at all: 302 MB a layer, 22.65
 * GB over the model, against a 64 GB part already holding 50.
 *
 * [scale-bias-20261020](../../../docs/evidence/ascend-decode/scale-bias-20261020/README.md)
 */
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "routed_int4_kernel.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {

constexpr int64_t HIDDEN = 6144;
constexpr int64_t HALF = 128;
constexpr int64_t GATE_UP = 2 * HALF;
constexpr int64_t M_TILE = 16;
constexpr int64_t BLOCK_FLOATS = 8;
constexpr int64_t TILE_BYTES = 2048 / 2;
constexpr size_t TILING_BYTES = 256;
constexpr size_t MATMUL_WORKSPACE = 16u << 20;

/** The sync base, the tilings and the scratch, built once a shape.
 *
 * `rtGetC2cCtrlAddr` and the tiling upload belong outside capture, and every
 * buffer here is shaped by the layer rather than by the call, so one instance
 * serves every invocation at a given row count.
 */
class Resident {
 public:
    Resident(int64_t experts, int64_t rows)
        : rows_(rows), scale_stride_((rows + BLOCK_FLOATS - 1)
                                     / BLOCK_FLOATS * BLOCK_FLOATS)
    {
        uint32_t length = 0;
        TORCH_CHECK(rtGetC2cCtrlAddr(&sync_base_, &length) == 0,
                    "rtGetC2cCtrlAddr failed");
        const int64_t stacked_rows = 2 * rows;
        gate_up_tiling_ = Upload(stacked_rows, GATE_UP, HIDDEN, GATE_UP);
        down_tiling_ = Upload(stacked_rows, HIDDEN, HALF, HIDDEN);
        stacked_ = Hold(experts * M_TILE * HIDDEN / 2);
        stacked_low_ = static_cast<char*>(stacked_) + rows * HIDDEN / 2;
        partial_ = Hold(experts * M_TILE * GATE_UP * sizeof(int32_t));
        middle_ = Hold(experts * M_TILE * HALF / 2);
        middle_scale_ = static_cast<float*>(
            Hold(experts * scale_stride_ * sizeof(float)));
        wide_ = Hold(experts * M_TILE * HIDDEN * sizeof(int32_t));
        witness_ = Hold(experts * TILE_BYTES);
        workspace_ = Hold(MATMUL_WORKSPACE);
    }

    uint64_t sync_base() const { return sync_base_; }
    void* gate_up_tiling() const { return gate_up_tiling_; }
    void* down_tiling() const { return down_tiling_; }
    void* stacked() const { return stacked_; }
    void* stacked_low() const { return stacked_low_; }
    void* partial() const { return partial_; }
    void* middle() const { return middle_; }
    float* middle_scale() const { return middle_scale_; }
    void* wide() const { return wide_; }
    void* witness() const { return witness_; }
    void* workspace() const { return workspace_; }

 private:
    static void* Hold(size_t bytes)
    {
        void* block = nullptr;
        TORCH_CHECK(aclrtMalloc(&block, bytes, ACL_MEM_MALLOC_HUGE_FIRST)
                    == ACL_SUCCESS, "aclrtMalloc failed for ", bytes);
        return block;
    }

    static void* Upload(int rows, int columns, int inner, int width)
    {
        std::vector<uint8_t> shaped(TILING_BYTES, 0);
        TORCH_CHECK(routed_int4_build_tiling(rows, columns, inner, width,
                                             shaped.data()) == 0,
                    "routed_int4_build_tiling failed");
        void* block = Hold(TILING_BYTES);
        TORCH_CHECK(aclrtMemcpy(block, TILING_BYTES, shaped.data(),
                                TILING_BYTES, ACL_MEMCPY_HOST_TO_DEVICE)
                    == ACL_SUCCESS, "tiling upload failed");
        return block;
    }

    int64_t rows_;
    int64_t scale_stride_;
    uint64_t sync_base_ = 0;
    void* gate_up_tiling_ = nullptr;
    void* down_tiling_ = nullptr;
    void* stacked_ = nullptr;
    void* stacked_low_ = nullptr;
    void* partial_ = nullptr;
    void* middle_ = nullptr;
    float* middle_scale_ = nullptr;
    void* wide_ = nullptr;
    void* witness_ = nullptr;
    void* workspace_ = nullptr;
};

/** One `Resident` a row count, since the engine's draft width is fixed. */
Resident& ResidentFor(int64_t experts, int64_t rows)
{
    static std::vector<std::pair<int64_t, Resident*>> held;
    for (const auto& entry : held) {
        if (entry.first == rows) {
            return *entry.second;
        }
    }
    held.emplace_back(rows, new Resident(experts, rows));
    return *held.back().second;
}

/** The stream this call is on, asked for without draining it.
 *
 * The draining accessor costs about a millisecond a call on a busy worker
 * against two microseconds for this one, and a cached pointer is worse than
 * either: it has been seen to send kernels onto another thread's stream.
 */
void* CurrentStream()
{
    return c10_npu::getCurrentNPUStreamNoWait();
}

/** One layer's routed experts, gate-up through down, in a single launch.
 *
 * `gate_up` and `down` are the checkpoint's packed INT4 codes as they are --
 * `[expert][N][K/2]` and `[expert][K][N/2]`, which is what the kernel reads,
 * so neither a transpose nor a widening is needed. `gu_bias` and `down_bias`
 * are the checkpoint's `scale_bias`, added once after the MSD recombination
 * and before the activation scale, which is the convention the host reference
 * documents.
 */
at::Tensor RoutedExperts(const at::Tensor& rows, const at::Tensor& gate_up,
                         const at::Tensor& gu_bias, const at::Tensor& gu_scale,
                         const at::Tensor& down, const at::Tensor& down_bias,
                         const at::Tensor& down_scale,
                         const at::Tensor& token_scale, int64_t experts,
                         int64_t blocks)
{
    TORCH_CHECK(rows.dtype() == at::kChar, "rows must be INT8");
    TORCH_CHECK(gate_up.dtype() == at::kChar, "gate_up must be packed INT4");
    TORCH_CHECK(down.dtype() == at::kChar, "down must be packed INT4");
    TORCH_CHECK(gu_bias.dtype() == at::kFloat, "gu_bias must be FP32");
    TORCH_CHECK(down_bias.dtype() == at::kFloat, "down_bias must be FP32");

    const int64_t per_expert = rows.size(0) / experts;
    at::Tensor out = at::empty({rows.size(0), HIDDEN},
                               rows.options().dtype(at::kBFloat16));
    Resident& held = ResidentFor(experts, per_expert);
    routed_int4_launch(
        static_cast<uint32_t>(blocks), CurrentStream(), held.sync_base(),
        rows.data_ptr(), held.stacked(), held.stacked_low(),
        gate_up.data_ptr(), gu_bias.data_ptr<float>(),
        gu_scale.data_ptr<float>(), down.data_ptr(),
        down_bias.data_ptr<float>(), down_scale.data_ptr<float>(),
        token_scale.data_ptr<float>(), held.partial(), held.middle(),
        held.middle_scale(), held.wide(), out.data_ptr(),
        held.gate_up_tiling(), held.down_tiling(), held.witness(),
        held.workspace(), experts);
    return out;
}

}  // namespace

TORCH_LIBRARY(tilert, library)
{
    library.def(
        "routed_experts(Tensor rows, Tensor gate_up, Tensor gu_bias, "
        "Tensor gu_scale, Tensor down, Tensor down_bias, Tensor down_scale, "
        "Tensor token_scale, int experts, int blocks) -> Tensor");
}

TORCH_LIBRARY_IMPL(tilert, PrivateUse1, library)
{
    library.impl("routed_experts", &RoutedExperts);
}
