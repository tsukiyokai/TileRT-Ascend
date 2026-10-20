# CANN's own operator is the fastest; three custom packages sit on top of it

2026-10-21, a3-84, one die, one process, 96 rows, sixteen experts. The same
`grouped_matmul_swiglu_quant_v2` call, varying only
`ASCEND_CUSTOM_OPP_PATH`.

    CANN only, no custom opp     66.813 us
    owner overlay only           97.362 us   +46%
    vllm-ascend package only    123.985 us   +86%
    both custom packages        104.929 us
    plus the gmm-fastpath overlay (measured separately)  100.604 us

`activate.sh:18` sets the two custom packages for every engine process, and
`launch_arm.sh:34` prepends a third. So no engine measurement has ever run
CANN's own implementation of this operator.

## This is the state difference

The 61.6 us clean against 153.3 us mixed gap was chased through nine
candidates -- loading a kernel library, running that kernel, building a
communication domain, a second resident fractal-NZ bank, the full two-stage
chain, constant versus random codes, graph capture, weight bank size, and the
TP16 gate-up width -- all innocent, because none of them touched the operator
path. A single-die probe does not source `activate.sh`; the engine does.

## What it is worth and how to take it

66.813 against 104.929 is 38 us a layer, about 3.0 ms a round. Unsetting the
variable wholesale is the wrong way to collect it: the custom packages exist
for other operators and other shapes, and one of them is vllm-ascend's own.
The right form is to let this one operator fall back -- an opp path that does
not carry `grouped_matmul_swiglu_quant_v2` while keeping the rest.

## The shape of the mistake

Three layers of custom operator packages, each added to make something
faster, together costing 57 percent on the operator that dominates the
round's largest line item. None of them was ever measured against the CANN
implementation at the decode shape.
