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
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "routed_int4_kernel.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {

constexpr int64_t HIDDEN = 6144;
constexpr int64_t HALF = 128;
constexpr int64_t GATE_UP = 2 * HALF;
constexpr int64_t M_TILE = 16;
constexpr int64_t ROWS = ROWS_PER_EXPERT;
constexpr int64_t STACKED = 2 * ROWS;
constexpr int64_t BLOCK_FLOATS = 8;
constexpr int64_t SCALE_STRIDE = (ROWS + BLOCK_FLOATS - 1)
                                 / BLOCK_FLOATS * BLOCK_FLOATS;
constexpr int64_t TILE_BYTES = 2048 / 2;
constexpr size_t TILING_BYTES = 256;
constexpr size_t MATMUL_WORKSPACE = 16u << 20;
// What the kernel copies in one go, so what the caller must provide.
constexpr int64_t MAX_EXPERTS = 256;
constexpr int64_t MAX_TAKEN = 32 * 8;
constexpr int64_t SCALE_SPAN = 8;

/** The sync base, the tilings and the scratch, built once.
 *
 * `rtGetC2cCtrlAddr` and the tiling upload belong outside capture, and every
 * buffer here is shaped by the layer rather than by the call, so one instance
 * serves every invocation. Rows an expert is `ROWS_PER_EXPERT`, compiled into
 * the kernel, so there is nothing to infer from the tensor.
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
    void* stacked() const { return stacked_; }
    void* stacked_low() const { return stacked_low_; }
    void* partial() const { return partial_; }
    void* middle() const { return middle_; }
    float* middle_scale() const { return middle_scale_; }
    void* wide() const { return wide_; }
    void* witness() const { return witness_; }
    void* workspace() const { return workspace_; }

 private:
    explicit Resident(int64_t experts)
    {
        // A probe switch: allocate for a fixed count whatever is asked, so
        // that the size and the walked range can be varied independently.
        // Ninety-six experts fault where eighty do not, at every block count,
        // and the only quantity that tracks that is the scratch size.
        if (std::getenv("TILERT_FIXED_SCRATCH") != nullptr) {
            experts = 256;
        }
        uint32_t length = 0;
        TORCH_CHECK(rtGetC2cCtrlAddr(&sync_base_, &length) == 0,
                    "rtGetC2cCtrlAddr failed");
        gate_up_tiling_ = Upload(STACKED, GATE_UP, HIDDEN, GATE_UP);
        down_tiling_ = Upload(STACKED, HIDDEN, HALF, HIDDEN);
        stacked_ = Hold(experts * M_TILE * HIDDEN / 2);
        stacked_low_ = static_cast<char*>(stacked_) + ROWS * HIDDEN / 2;
        partial_ = Hold(experts * M_TILE * GATE_UP * sizeof(int32_t));
        middle_ = Hold(experts * M_TILE * HALF / 2);
        middle_scale_ = static_cast<float*>(
            Hold(experts * SCALE_STRIDE * sizeof(float)));
        wide_ = Hold(experts * M_TILE * HIDDEN * sizeof(int32_t));
        // One tile an expert, and one tile of slack. The kernel steps this
        // by `TILE` INT4 codes an expert, which is `TILE_BYTES`, so the last
        // expert's write ends exactly at `experts * TILE_BYTES` -- flush
        // against the allocation. Eighty experts survived that and
        // ninety-six did not, which is what a write that is only just inside
        // its mapping looks like.
        witness_ = Hold((experts + 1) * TILE_BYTES);
        workspace_ = Hold(MATMUL_WORKSPACE);
    }

    static void* Hold(size_t bytes)
    {
        void* block = nullptr;
        TORCH_CHECK(aclrtMalloc(&block, bytes, ACL_MEM_MALLOC_HUGE_FIRST)
                    == ACL_SUCCESS, "aclrtMalloc failed for ", bytes);
        if (std::getenv("TILERT_SHOW_SCRATCH") != nullptr) {
            std::printf("scratch %12zu bytes at %p\n", bytes, block);
            std::fflush(stdout);
        }
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

/** The stream this call is on, asked for without draining it.
 *
 * `need_empty` false is the whole point: draining costs about a millisecond a
 * call on a busy worker against two microseconds, and a cached pointer is
 * worse than either, having been seen to send kernels onto another thread's
 * stream. The header also declares `getCurrentNPUStreamNoWait`, which
 * `libtorch_npu.so` does not export -- it links and then fails at `dlopen` --
 * so this is the form the op plugin itself uses.
 */
void* CurrentStream()
{
    return c10_npu::getCurrentNPUStream().stream(false);
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
                         const at::Tensor& token_scale,
                         const at::Tensor& group_count,
                         const at::Tensor& group_start,
                         const at::Tensor& source_row,
                         const at::Tensor& row_weight, int64_t experts,
                         int64_t blocks)
{
    TORCH_CHECK(rows.dtype() == at::kChar, "rows must be INT8");
    TORCH_CHECK(gate_up.dtype() == at::kChar, "gate_up must be packed INT4");
    TORCH_CHECK(down.dtype() == at::kChar, "down must be packed INT4");
    TORCH_CHECK(gu_bias.dtype() == at::kFloat, "gu_bias must be FP32");
    TORCH_CHECK(down_bias.dtype() == at::kFloat, "down_bias must be FP32");

    // Every one of these is read through data_ptr(), so a view with its own
    // strides silently reads the wrong memory. A sliced-then-reshaped scale
    // did exactly that: the weights were right, the activations were right,
    // the reference was right, and the kernel's output did not depend on the
    // scale at all -- a boost of a million moved it by three parts in ten
    // million.
    for (const auto& held : {rows, gate_up, gu_bias, gu_scale, down,
                             down_bias, down_scale, token_scale, group_count,
                             group_start, source_row, row_weight}) {
        TORCH_CHECK(held.is_contiguous(),
                    "every input must be contiguous; data_ptr() carries no "
                    "strides");
    }
    TORCH_CHECK(group_count.dtype() == at::kInt, "group_count must be INT32");
    TORCH_CHECK(group_start.dtype() == at::kInt, "group_start must be INT32");
    // The kernel reads the whole list in one copy, so it must be the whole
    // list. Handing it a shorter one reads past the allocation, and the
    // over-read only faults once it crosses a block boundary -- eighty
    // experts passed where ninety-six reported "the GM address accessed by
    // scalar exceeds 48 bits".
    TORCH_CHECK(group_count.numel() == MAX_EXPERTS, "group_count must hold ",
                MAX_EXPERTS, " entries, got ", group_count.numel());
    TORCH_CHECK(group_start.numel() == MAX_EXPERTS, "group_start must hold ",
                MAX_EXPERTS, " entries, got ", group_start.numel());
    TORCH_CHECK(source_row.numel() == MAX_TAKEN, "source_row must hold ",
                MAX_TAKEN, " entries, got ", source_row.numel());
    TORCH_CHECK(row_weight.numel() == MAX_TAKEN, "row_weight must hold ",
                MAX_TAKEN, " entries, got ", row_weight.numel());
    // One row a token, not one a (expert, row) pair: the kernel adds the
    // eight experts a token visits into the same row.
    at::Tensor out = at::zeros({row_weight.size(0) / 8, HIDDEN},
                               rows.options().dtype(at::kBFloat16));
    Resident& held = Resident::Instance(experts);
    routed_int4_launch(
        static_cast<uint32_t>(blocks), CurrentStream(), held.sync_base(),
        rows.data_ptr(), held.stacked(), held.stacked_low(),
        gate_up.data_ptr(), gu_bias.data_ptr<float>(),
        gu_scale.data_ptr<float>(), down.data_ptr(),
        down_bias.data_ptr<float>(), down_scale.data_ptr<float>(),
        token_scale.data_ptr<float>(), held.partial(), held.middle(),
        held.middle_scale(), held.wide(), out.data_ptr(),
        held.gate_up_tiling(), held.down_tiling(), held.witness(),
        group_count.data_ptr(), group_start.data_ptr(),
        source_row.data_ptr(), row_weight.data_ptr<float>(),
        held.workspace(), experts);
    if (std::getenv("TILERT_SHOW_STACKED") != nullptr) {
        // What Split wrote for the cube to read: the activation's two INT4
        // halves, M-stacked. The cube's product is zero, so either this is
        // empty or the matmul does not find it.
        const int64_t bytes = experts * M_TILE * HIDDEN / 2;
        at::Tensor codes = at::empty({bytes}, out.options().dtype(at::kChar));
        TORCH_CHECK(aclrtMemcpy(codes.data_ptr(), bytes, held.stacked(),
                                bytes, ACL_MEMCPY_DEVICE_TO_DEVICE)
                    == ACL_SUCCESS, "stacked readback failed");
        return codes;
    }
    if (std::getenv("TILERT_SHOW_PARTIAL") != nullptr) {
        // The cube's INT32 product, before any dequantisation. A host
        // reference computes this exactly, so comparing it separates a wrong
        // matmul from a wrong recombination.
        const int64_t span = experts * M_TILE * GATE_UP;
        at::Tensor product = at::empty({span},
                                       out.options().dtype(at::kInt));
        TORCH_CHECK(aclrtMemcpy(product.data_ptr(), span * sizeof(int32_t),
                                held.partial(), span * sizeof(int32_t),
                                ACL_MEMCPY_DEVICE_TO_DEVICE) == ACL_SUCCESS,
                    "partial readback failed");
        return product;
    }
    if (std::getenv("TILERT_SHOW_MIDDLE") != nullptr) {
        // The quantum Activate chose, read back so a host reference can
        // compare it directly rather than inferring it from the output.
        at::Tensor quantum = at::empty({experts * SCALE_SPAN},
                                       out.options().dtype(at::kFloat));
        TORCH_CHECK(aclrtMemcpy(quantum.data_ptr(),
                                experts * SCALE_SPAN * sizeof(float),
                                held.middle_scale(),
                                experts * SCALE_SPAN * sizeof(float),
                                ACL_MEMCPY_DEVICE_TO_DEVICE) == ACL_SUCCESS,
                    "middle_scale readback failed");
        return at::stack({out.reshape({-1}).slice(0, 0, quantum.numel()),
                          quantum});
    }
    return out;
}

}  // namespace

TORCH_LIBRARY(tilert, library)
{
    library.def(
        "routed_experts(Tensor rows, Tensor gate_up, Tensor gu_bias, "
        "Tensor gu_scale, Tensor down, Tensor down_bias, Tensor down_scale, "
        "Tensor token_scale, Tensor group_count, Tensor group_start, "
        "Tensor source_row, Tensor row_weight, int experts, int blocks) "
        "-> Tensor");
}

TORCH_LIBRARY_IMPL(tilert, PrivateUse1, library)
{
    library.impl("routed_experts", &RoutedExperts);
}
