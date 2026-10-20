# The term both my kernel and my reference leave out

2026-10-20. Read off the checkpoint on the CPU, with every machine's dies
held by another tenant.

## What the checkpoint says

`quant_model_description.json` carries 235,935 entries, of which 117,311 name
a `weight_offset` or a `scale_bias`. Every expert projection is marked
`W4A8_DYNAMIC` and has four tensors, not two:

    model.layers.30.mlp.experts.0.gate_proj.weight         W4A8_DYNAMIC
    model.layers.30.mlp.experts.0.gate_proj.weight_scale   W4A8_DYNAMIC
    model.layers.30.mlp.experts.0.gate_proj.weight_offset  W4A8_DYNAMIC
    model.layers.30.mlp.experts.0.gate_proj.scale_bias     W4A8_DYNAMIC

And the bias is not zero. Reading the shard directly:

    model.layers.3.mlp.experts.0.gate_proj.scale_bias F32 [2048, 1]
      first 16 values, 16 nonzero:
      [-0.001393, 0.076904, 0.000794, 0.017286, 0.001723, 0.025291, ...]

## What that makes of the zero-mismatch results

The INT8 kernel reports 0 of 49152 at 4/4, 8/8, 16/16 and 44/22, and the INT4
kernel 0 of 540672 after its four defects were fixed. Both were checked
against a host reference written from the same reading of the format that the
kernels implement: symmetric dequantisation, `Σ(x·q)·s`. So the agreement is
real and says what it says -- the kernel computes what the reference computes
-- and that is a statement about consistency. It is not a statement about the
format. Both sides omit the offset.

The vendor does not. `_get_mlp_scale_bias` returns `w13_scale_bias` and
`w2_scale_bias`, and they are passed as `weight_assist_matrix` to
`grouped_matmul_swiglu_quant_v2` and as `bias` to `npu_grouped_matmul` --
per output channel in both, added after dequantisation.

## The fix, and what remains unverified

A column add on the accumulator after dequantisation, one FP32 value an output
channel. The vector pipe is at 1.3% utilisation against a scalar pipe at
62.5%, so the arithmetic is free; what it costs is a buffer for the bias and
the loads to fill it.

What is verified here is that the term exists, is nonzero, and is absent from
my kernels. What is not verified is the exact convention -- whether the vendor
adds it to the dequantised product as `npu_grouped_matmul`'s documented `bias`
contract implies, or folds it some other way, and whether `weight_offset`
needs separate handling beyond it. Both questions are answered by one run
against the vendor operator on a single die, which is the next device work
this path needs; guessing the convention would repeat exactly the error this
page records.

## The shape this also settled

`replay_cann_gmm1.py` asserts the frozen trace's `group_counts` is `(256,)`
with every entry 0 or 1, and `dispatch_a8` is `(8, 6144)`. So at M1 a rank
sees one row per active expert and eight active experts -- a single token
routed to eight of the 256, 16 of which live on this rank. The kernel's
measured shapes, 16 experts of 2 rows and up, are heavier than the decode
round they were meant to serve.
