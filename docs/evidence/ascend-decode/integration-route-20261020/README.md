# How a hand-written kernel reaches the graph, and why one route is closed

## The vendor-package override cannot express this kernel

Not on cost -- on structure. This kernel replaces two operators, and they are
reached through different packages: `GroupedMatmulSwigluQuantV2` through
`torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2`, and `GroupedMatmul`
through `torch_npu.npu_grouped_matmul`, a CANN built-in. A vendor-package
override swaps an `.o` behind a fixed prototype, so it preserves the operator
boundary by construction: after the substituted kernel finished, the graph
would still issue the second grouped matmul, and the global-memory round trip
between the two -- which is where the fusion win lives -- would survive
untouched. A per-operator override can only buy what is internal to one
operator, which is what the four landed `patches/cann-gmm-swiglu-v2/` patches
bought at −1.468 ms a round.

Two further walls if that argument were waived. The operator's ABI is not this
kernel's: `weight` is a TensorList of DT_INT4 in a forced 5-D storage shape
`{E, n/64, k/16, 16, 8}`, `weight_scale` is UINT64 carrying float32 bit
patterns, and `weight_assist_matrix` carries the MSD correction, where this
kernel takes one contiguous INT8 ND buffer and f32 scales. And host tiling
stays in `liboptiling.so`, which an `.o` swap does not rebuild -- and `opc`
will not compile a kernel whose TilingData differs, because it `CDLL`s that
library mid-compile to learn `tiling_data_size`.

## The route that works is the caller, and this project has walked it once

Fusing two operators means changing the caller. The MoE body is already an
opaque custom op: the compiled graph contains `torch.ops.vllm.moe_forward_shared`
and no grouped matmul at all, so Python re-enters the quant method on every
eager invocation and once during capture. No `register_fake`, no
`direct_register_custom_op`, no graph-break reasoning -- replace a method on
the quant-method object, which `cann_head.py` already does for another.

And the precedent is this project's own: `torch-allreduce-20261018` shipped a
hand-written AscendC kernel into the captured production graph as
`aclshmem_torch.so` plus `libtp_push_kernel.so`, measured in-graph at 22.3 µs
against HCCL's 63.5, installed through `onesided_reduce.install()` at the hook
that already carries three other overrides. The path was not missing; it was
unrecognised.

## The weight layout already matches

`w13_weight` is `(256, 128, 6144)` INT8 with two nibbles a byte, which unpacks
to `(256, 256, 6144)` -- `[expert][N][K]`, exactly this kernel's `gate_up`.
`w2_weight (256, 3072, 128)` unpacks to `(256, 6144, 128)`, exactly its
`down`. The scales match as f32. So no transpose, no NZ, and no MSD
correction: widening INT4 into INT8 containers is lossless and yields the
identical `mad`. The branch must come before `_process_scale` and before
`maybe_trans_nz`, because NZ is a one-way door.

## The cost is HBM, not packaging

Widening the routed experts to INT8 costs 302 MB a layer resident, so 75
layers is +22.65 GB against a 64 GB part already showing about 50 GB in use at
utilisation 0.78. **A full-model INT8 arm does not fit.** The INT8 kernel is a
per-layer experiment by construction -- eight to sixteen layers is comfortable
-- and production needs the INT4 form, which still deviates in 49105 of 49152
values after its first projection became exact.

## And one of this kernel's own lines is dead

`KERNEL_TASK_TYPE_DEFAULT` expands through `ENABLE_FEATURE_FOR_COMPILE`, which
is empty unless `__CHECK_FEATURE_AT_PRECOMPILE` is defined, and only an
operator-project driver defines it. In a direct `bisheng` build the macro does
nothing: the 1:2 core split comes entirely from `__mix__(1, 2)`. Conversely an
operator project textually includes the source with `#define __global__ inline`
and generates its own entry point, so `__mix__` would land on an inlined
function and `__schedmode__(1)` would have nothing to propagate to -- and this
project has recorded `__schedmode__(1)` as required for cross-core sync. Moving
to an operator project buys less than dispatch, not more.

## What would tell us the plan is wrong

Numerics against the vendor operator rather than against this project's own
host reference. The 0 of 196608 is against a host model of this kernel's own
dequantisation path; the vendor dequantises through the A3 fixpipe's
1+8+10-bit scale with separate FP32 additions. The integer products agree by
construction and the rounding does not, and the size of that divergence is
unmeasured. It matters more here than elsewhere because this engine's TPOT is
acceptance-coupled, so a kernel that moves tokens produces a TPOT that
measures acceptance rather than the kernel.
