# The kernel, reached from torch, at the decode shape

2026-10-20, a3-84, one die, with every machine's remaining dies held by other
tenants.

    ran: out (16, 6144) torch.bfloat16 finite True nonzero 98304/98304
    wall 37.193 us a call, 16 experts of 1 row

First number the integration path has produced. `torch.ops.tilert.routed_experts`
dispatches on `PrivateUse1` into the hand-written INT4 kernel, which runs both
projections in one launch and returns a correctly shaped BF16 tensor with no
zero and no infinity in it.

## Why this shape

The frozen trace `replay_cann_gmm1.py` replays asserts `group_counts` is
`(256,)` with every entry 0 or 1, and `dispatch_a8` is `(8, 6144)`. So at M1 a
rank sees one row per active expert: a single token routed to eight of 256
experts, sixteen of which live on this rank. Built with `CAPACITY=1`, since
rows an expert is compiled into the kernel.

## What it is comparable to, and what it is not

The vendor path at M1 costs 98.950 + 58.226 = 157.176 µs across its two
operators. Mine costs 37.193 µs in one, wall clock, torch dispatch included.
But the two are handed different work: the vendor is given all 256 experts
with a mostly-zero group list, where this kernel is given the sixteen that
live on the rank. So the ratio is between paths as the engine issues them, not
between identical workloads -- the vendor's 256-expert list is its own
inefficiency and the engine pays it today.

## What it would be worth, if it holds in place

75 layers at 157 µs is 11.8 ms of a 56.9 ms round, about a fifth. Replacing it
at this cost removes roughly 9 ms a round, which at acceptance 5.273 is a TPOT
of about 9.1 ms against the measured 10.93. That is arithmetic on a per-layer
figure, not a measurement: it assumes the vendor cost is on the critical path
once a layer and that the operator slots in without disturbing what surrounds
it. The integration run is what settles it.

## Three defects the attempt surfaced

`getCurrentNPUStreamNoWait` is declared in `NPUStream.h` and not exported by
`libtorch_npu.so`. A shared library links with it undefined and fails at
`dlopen`, so the build passing said nothing. The exported form is
`getCurrentNPUStream().stream(false)`, whose parameter is named `need_empty`
-- the same call the op plugin makes.

Rows an expert is a compile-time define, `ROWS_PER_EXPERT`, and the binding
was inferring it at runtime from the activation and building tiling for twice
that. Two sources for one number, which agree only by luck. The binding now
reads the define, and `build_so.sh` passes `CAPACITY` to both objects.

The per-row-count `Resident` map was generality for a kernel that has one row
count by construction. Removed.
