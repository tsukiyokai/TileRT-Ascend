"""Send a layer's routed experts to the hand-written kernel.

The kernel replaces what `GroupedMatmulSwigluQuantV2` and `GroupedMatmul` do
together, and those are reached through different packages -- the first
through `torch.ops._C_ascend`, the second through `torch_npu` as a CANN
built-in. A vendor-package override swaps an `.o` behind a fixed prototype, so
it preserves the operator boundary and the global-memory round trip between
the two would survive; the fusion is only expressible at the caller. The
caller is reachable because the MoE body is already an opaque custom op, so
the quant method is re-entered on every eager invocation and once during
capture.

The weights need no rearrangement. `w13_weight` arrives as
`(256, 128, 6144)` INT8 with two nibbles a byte, which unpacks to
`(256, 256, 6144)` -- `[expert][N][K]`, which is what the kernel reads -- and
`w2_weight (256, 3072, 128)` unpacks to `(256, 6144, 128)`. The scales are
already FP32 per output channel. Widening INT4 into INT8 containers is
lossless and yields the identical `mad`, so no MSD correction is needed on
this path either.

What it costs is memory: 302 MB a layer of widened weights, so the whole model
would want 22.65 GB that a 64 GB part at utilisation 0.78 does not have. This
is a per-layer experiment by construction, which is why `install` takes the
layers to convert.
"""
import ctypes
import os

LIBRARY_DIR = os.environ.get(
    "TILERT_ROUTED_LIB_DIR", "/workspace/src/routed-so",
)
EXPERTS_PER_RANK = 16
BLOCKS = 16


def _load():
    """Bring in the kernel's library, before torch sees the operator.

    `libascendcl` is a runtime dependency of the kernel object, and loading
    the extension with `RTLD_GLOBAL` first is what lets `load_library` resolve
    it.
    """
    import torch

    ctypes.CDLL(f"{LIBRARY_DIR}/routed_expert_torch.so",
                mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(f"{LIBRARY_DIR}/routed_expert_torch.so")


def _unpack(packed):
    """Two INT4 nibbles a byte to one INT8 an element, low nibble first.

    The checkpoint's packing contract: byte `i` holds element `2i` in its low
    nibble and `2i + 1` in its high one, and a row of an `(M, N)` tensor
    starts at byte `r * N / 2`.
    """
    import torch

    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)
    widened = torch.stack((low, high), dim=-1)
    return widened.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def install(layers, experts=EXPERTS_PER_RANK, blocks=BLOCKS):
    """Point `layers`' routed experts at the kernel; return how to undo it.

    `layers` are the MoE layer objects whose quant method is to be replaced.
    The replacement is per object rather than per class, so the layers left
    out keep the vendor path and the comparison is within one run.
    """
    _load()
    import torch

    undo = []
    for layer in layers:
        method = layer.quant_method
        gate_up = _unpack(method.w13_weight.data).contiguous()
        down = _unpack(method.w2_weight.data).contiguous()
        gu_scale = method.w13_weight_scale.data.squeeze(-1).contiguous()
        down_scale = method.w2_weight_scale.data.squeeze(-1).contiguous()
        original_first = method.apply_gmm1_act_quant
        original_second = method.apply_gmm2

        def fused(rows, token_scale, _gate_up=gate_up, _down=down,
                  _gu_scale=gu_scale, _down_scale=down_scale):
            return torch.ops.tilert.routed_experts(
                rows, _gate_up, _gu_scale, _down, _down_scale, token_scale,
                experts, blocks)

        def first(*arguments, _fused=fused, **named):
            """Do the whole chain here and leave the second call a no-op."""
            rows, token_scale = arguments[0], arguments[1]
            layer._tilert_output = _fused(rows, token_scale)
            return layer._tilert_output

        def second(*arguments, **named):
            return layer._tilert_output

        method.apply_gmm1_act_quant = first
        method.apply_gmm2 = second
        undo.append((method, original_first, original_second))

    def restore():
        for method, original_first, original_second in undo:
            method.apply_gmm1_act_quant = original_first
            method.apply_gmm2 = original_second

    return restore
