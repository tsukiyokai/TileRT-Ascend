/**
 * The routed-expert kernel as a torch operator, so the graph can call it.
 *
 * A vendor-package override cannot express this kernel: it replaces two
 * operators reached through different packages, and swapping an `.o` behind a
 * fixed prototype preserves the operator boundary by construction -- the
 * second grouped matmul would still be issued and the global-memory round
 * trip between them, which is where the fusion win lives, would survive. So
 * the caller is what changes, and the caller is Python: the MoE body is
 * already an opaque custom op, `torch.ops.vllm.moe_forward_shared`, with no
 * grouped matmul in the compiled graph at all, which means the quant method
 * is re-entered on every eager invocation and once during capture. Replacing
 * a method there needs no `register_fake` and no graph-break reasoning.
 *
 * [integration-route-20261020](../../../docs/evidence/ascend-decode/integration-route-20261020/README.md)
 */
#include <torch/extension.h>
#include <torch/library.h>

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "routed_expert_kernel.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {

constexpr int64_t HIDDEN = 6144;
constexpr int64_t HALF = 128;
constexpr int64_t GATE_UP = 2 * HALF;
constexpr int64_t ROWS = 2;
constexpr int64_t M_TILE = 16;
constexpr int64_t SCALE_SPAN = 8;
constexpr size_t TILING_BYTES = 256;

/** The cross-core sync base, the tilings and the scratch, built once.
 *
 * `rtGetC2cCtrlAddr` and the tiling upload belong outside capture, and the
 * scratch buffers are shaped by the layer rather than by the call, so one
 * instance serves every invocation.
 */
class Resident {
 public:
    static Resident& Instance(int64_t experts)
    {
        static Resident held(experts);
        return held;
    }

    uint64_t sync_base() const { return sync_base_; }
    void* gate_up_tiling() const { return gate_up_tiling_; }
    void* down_tiling() const { return down_tiling_; }
    void* partial() const { return partial_; }
    void* middle() const { return middle_; }
    float* middle_scale() const { return middle_scale_; }
    void* wide() const { return wide_; }

 private:
    explicit Resident(int64_t experts)
    {
        uint32_t length = 0;
        TORCH_CHECK(rtGetC2cCtrlAddr(&sync_base_, &length) == 0,
                    "rtGetC2cCtrlAddr failed");
        gate_up_tiling_ = Upload(ROWS, GATE_UP, HIDDEN, GATE_UP);
        down_tiling_ = Upload(ROWS, HIDDEN, HALF, HIDDEN);
        partial_ = Hold(experts * M_TILE * GATE_UP * sizeof(int32_t));
        middle_ = Hold(experts * M_TILE * HALF);
        middle_scale_ = static_cast<float*>(
            Hold(experts * SCALE_SPAN * sizeof(float)));
        wide_ = Hold(experts * M_TILE * HIDDEN * sizeof(int32_t));
    }

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
        TORCH_CHECK(routed_build_tiling(rows, columns, inner, width,
                                        shaped.data()) == 0,
                    "routed_build_tiling failed");
        void* block = Hold(TILING_BYTES);
        TORCH_CHECK(aclrtMemcpy(block, TILING_BYTES, shaped.data(),
                                TILING_BYTES, ACL_MEMCPY_HOST_TO_DEVICE)
                    == ACL_SUCCESS, "tiling upload failed");
        return block;
    }

    uint64_t sync_base_ = 0;
    void* gate_up_tiling_ = nullptr;
    void* down_tiling_ = nullptr;
    void* partial_ = nullptr;
    void* middle_ = nullptr;
    float* middle_scale_ = nullptr;
    void* wide_ = nullptr;
};

/** The current stream, without draining it.
 *
 * `_npu_getCurrentRawStream` drains, which costs about a millisecond a call
 * on a busy worker against two microseconds for the accessor that does not,
 * and a cached global pointer has been seen to send kernels onto another
 * thread's stream.
 */
void* CurrentStream()
{
    static auto accessor = torch::jit::getAllOperatorsFor(
        c10::Symbol::fromQualString("npu::_npu_getCurrentRawStreamNoWait"));
    aclrtStream stream = nullptr;
    TORCH_CHECK(aclrtGetCurrentStream(&stream) == ACL_SUCCESS,
                "aclrtGetCurrentStream failed");
    return stream;
}

/** One layer's routed experts.
 *
 * `rows` is the permuted INT8 activation, `gate_up` and `down` are the INT8
 * weights unpacked from the checkpoint's nibble pairs -- which already arrive
 * as [expert][N][K] and [expert][K][N], so neither a transpose nor a format
 * cast is needed.
 */
at::Tensor RoutedExperts(const at::Tensor& rows, const at::Tensor& gate_up,
                         const at::Tensor& gu_scale, const at::Tensor& down,
                         const at::Tensor& down_scale,
                         const at::Tensor& token_scale, int64_t experts,
                         int64_t blocks)
{
    TORCH_CHECK(rows.dtype() == at::kChar, "rows must be INT8");
    TORCH_CHECK(gate_up.dtype() == at::kChar, "gate_up must be INT8");
    TORCH_CHECK(down.dtype() == at::kChar, "down must be INT8");
    TORCH_CHECK(gu_scale.dtype() == at::kFloat, "gu_scale must be FP32");
    TORCH_CHECK(down_scale.dtype() == at::kFloat, "down_scale must be FP32");

    at::Tensor out = at::empty({experts * ROWS, HIDDEN},
                               rows.options().dtype(at::kBFloat16));
    Resident& held = Resident::Instance(experts);
    routed_expert_launch(
        static_cast<uint32_t>(blocks), CurrentStream(), held.sync_base(),
        rows.data_ptr(), gate_up.data_ptr(), gu_scale.data_ptr<float>(),
        down.data_ptr(), down_scale.data_ptr<float>(),
        token_scale.data_ptr<float>(), held.partial(), held.middle(),
        held.middle_scale(), held.wide(), out.data_ptr(),
        held.gate_up_tiling(), held.down_tiling(), experts);
    return out;
}

}  // namespace

TORCH_LIBRARY(tilert, library)
{
    library.def(
        "routed_experts(Tensor rows, Tensor gate_up, Tensor gu_scale, "
        "Tensor down, Tensor down_scale, Tensor token_scale, int experts, "
        "int blocks) -> Tensor");
}

TORCH_LIBRARY_IMPL(tilert, PrivateUse1, library)
{
    library.impl("routed_experts", &RoutedExperts);
}
