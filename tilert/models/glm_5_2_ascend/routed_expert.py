"""Send a layer's routed experts to the hand-written INT4 kernel.

The kernel replaces what `GroupedMatmulSwigluQuantV2` and `GroupedMatmul` do
together, and those are reached through different packages -- the first
through `torch.ops._C_ascend`, the second through `torch_npu` as a CANN
built-in. A vendor-package override swaps an `.o` behind a fixed prototype, so
it preserves the operator boundary and the global-memory round trip between
the two would survive; the fusion is only expressible at the caller. The
caller is reachable because the MoE body is already an opaque custom op, so
the quant method is re-entered on every eager invocation and once during
capture.

Nothing about the weights is rearranged, but when they are taken matters.
The checkpoint stores packed INT4 codes -- `w13_weight (256, 128, 6144)` is
`[expert][N][K/2]` and `w2_weight (256, 3072, 128)` is `[expert][K][N/2]` --
which is what the kernel reads, and `w13_scale_bias` and `w2_scale_bias` are
the per-output-channel FP32 correction the asymmetric quantisation needs,
added once after the MSD recombination and before the activation scale.

`process_weights_after_loading` then casts both weights to fractal NZ, because
the vendor's fused operator dispatches to `aclnnGroupedMatmulSwigluQuantWeightNzV2`
and that variant requires five-dimensional storage. A kernel that reads ND
would read NZ bytes as ND and be silently wrong, so `install` must be called
while the weights are still as loaded, and it holds its own reference to them
-- which costs what the weights cost, once a converted layer.
"""
import ctypes
import os

LIBRARY_DIR = os.environ.get("TILERT_ROUTED_LIB_DIR", "/workspace/src/int4-so")
EXPERTS_PER_RANK = 16
BLOCKS = 16


def _load():
    """Bring in the kernel's library, before torch sees the operator.

    `libascendcl` is a runtime dependency of the kernel object, and loading
    the extension with `RTLD_GLOBAL` first is what lets `load_library`
    resolve it.
    """
    import torch

    ctypes.CDLL(f"{LIBRARY_DIR}/routed_int4_torch.so", mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(f"{LIBRARY_DIR}/routed_int4_torch.so")


def install(layers, experts=EXPERTS_PER_RANK, blocks=BLOCKS):
    """Point `layers`' routed experts at the kernel; return how to undo it.

    The replacement is per object rather than per class, so the layers left
    out keep the vendor path and the comparison is within one run.
    """
    _load()
    import torch

    undo = []
    for layer in layers:
        method = layer.quant_method
        held = (method.w13_weight.data.clone(),
                method.w13_scale_bias.data.squeeze(-1).clone(),
                method.w13_weight_scale.data.squeeze(-1).clone(),
                method.w2_weight.data.clone(),
                method.w2_scale_bias.data.squeeze(-1).clone(),
                method.w2_weight_scale.data.squeeze(-1).clone())
        original_first = method.apply_gmm1_act_quant
        original_second = method.apply_gmm2

        def first(compute, _held=held, _layer=layer, _method=method):
            """Do the whole chain, and leave the second call a hand-back.

            The vendor path splits it across two methods with a
            global-memory round trip between them; this does both and
            remembers the result for the second to return.
            """
            rows, token_scale = _method._quant_hidden_states(
                compute.hidden_states, compute.dynamic_scale)
            _layer._tilert_output = torch.ops.tilert.routed_experts(
                rows, *_held, token_scale, experts, blocks)
            return _layer._tilert_output, token_scale

        def second(compute, hidden_states, act_out_scale, _layer=layer):
            return _layer._tilert_output

        method.apply_gmm1_act_quant = first
        method.apply_gmm2 = second
        undo.append((method, original_first, original_second))

    def restore():
        for method, original_first, original_second in undo:
            method.apply_gmm1_act_quant = original_first
            method.apply_gmm2 = original_second

    return restore
