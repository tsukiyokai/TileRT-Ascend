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

The weights need repacking, and when they are taken matters. The checkpoint
nibble-packs along the output axis: `w13_weight (256, 128, 6144)` is
`[expert][N/2][K]`, which its scale's 256 entries confirm -- 128 is N halved,
6144 is K -- and `w2_weight (256, 3072, 128)` is the same shape of thing with
N 6144 and K 128. The kernel packs along K instead, two neighbours of a row
to a byte. So `install` unpacks along N and repacks along K, once a layer.

`w13_scale_bias` and `w2_scale_bias` are the per-output-channel FP32
correction the asymmetric quantisation needs, added once after the MSD
recombination and before the activation scale.

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
HIDDEN = 6144
GATE_UP = 256
EXPERTS_PER_RANK = 16
BLOCKS = 16


def _repack(packed, columns):
    """The checkpoint's packing to the kernel's, through the wide form."""
    wide = unpack_codes(packed)
    assert wide.shape[1] == columns, (wide.shape, columns)
    return pack_along_inner(wide)


def unpack_codes(packed):
    """`[expert][N/2][K]` to `[expert][N][K]`, low nibble first.

    The one place that knows which axis the checkpoint pairs along. The shapes
    permit the other reading, so a second implementation that gets it wrong
    runs and agrees with nothing -- which has already happened twice, once in
    this module and once in the reference written to check it.
    """
    import torch

    experts, halved, inner = packed.shape
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)
    return torch.stack((low, high), dim=2).reshape(experts, 2 * halved, inner)


def pack_along_inner(codes):
    """`[expert][N][K]` to `[expert][N][K/2]`, which is what the kernel reads."""
    import torch

    experts, columns, inner = codes.shape
    pairs = codes.reshape(experts, columns, inner // 2, 2)
    return (((pairs[..., 1] & 0x0F) << 4) | (pairs[..., 0] & 0x0F)).to(
        torch.int8).contiguous()


def slice_corrections(gu_scale, gu_bias, down_scale, down_bias,
                      rank=0, world=16):
    """The scales and corrections a rank owns.

    Three different shapes, and only one of them is what it looks like.
    `gate_proj.weight_scale` and `scale_bias` are `[N][1]` over the full
    intermediate, so a rank takes its rows. `down_proj.weight_scale` is
    `[N][1]` over the hidden size, which TP does not divide, so every rank
    takes all of it. But `down_proj.scale_bias` is `[N][world]`: the down
    projection's K axis is the intermediate, so TP splits it and each rank
    computes a partial sum -- adding the whole correction on every rank would
    add it `world` times. Each rank takes its own column, and the sum over
    ranks is the correction.

    This is what `w4a8.py` means by `torch.empty(num_experts, hidden_sizes,
    16 // self.tp_size)`: at TP16 the third dimension is one.
    """
    per_rank = gu_scale.shape[1] // world
    rows = slice(rank * per_rank, (rank + 1) * per_rank)
    column = min(rank, down_bias.shape[-1] - 1)
    return {
        "gu_scale": gu_scale[:, rows].reshape(gu_scale.shape[0], -1),
        "gu_bias": gu_bias[:, rows].reshape(gu_bias.shape[0], -1),
        "down_scale": down_scale.reshape(down_scale.shape[0], -1),
        "down_bias": down_bias[..., column].reshape(down_bias.shape[0], -1),
    }


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
        held = (_repack(method.w13_weight.data, GATE_UP),
                method.w13_scale_bias.data.squeeze(-1).clone(),
                method.w13_weight_scale.data.squeeze(-1).clone(),
                _repack(method.w2_weight.data, HIDDEN),
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
