# Two patches for the W4A8 expert GMM at batch one

Both apply with `patch -p1` at the root of the `vllm-ascend-unified` checkout,
against `csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/`, in the order
listed. They live here because this repository is where the measurements are;
they belong upstream.

Both are gated on `loopCount == 1`, which is every decode call and no prefill
call: `mLimit = (64 MiB / 2) / (k·1 + 2·n·2)` is 4681 for the TP16 `gate_up`
shape, so `Ceil(M, mLimit)` first reaches 2 at M = 4682 while batch one gives
M = 8 and batch four gives M = 32. Confirmed at runtime rather than by
arithmetic alone: a binary with 14 extra `SyncAll<false>` inside that branch
measures +32.490 µs, which would be zero if the branch were not taken. The
gate is what keeps prefill byte-identical, and it is the line between a
batch-one fast path and a rewrite.

## `gmm1-straight-line.patch` — 6.43 µs per call

At one split the three-stage loop runs `preProcess(idx+1)` for a split that
does not exist and `postProcess(idx-1)` for an index of −1. Both return on
their own guards — `pre.h:125-130` on `idx >= loopCount`, `post.h:361-366` on
`idx >= loopCount || idx < 0` — so the two calls and the barrier pair around
them are pure cost. Replacing the loop with pre(0), sync, mid(0), sync,
post(0) measures −6.43 µs per call over three runs (−6.426, −9.171, −4.865;
σ 0.442, 0.194, 0.274; 0 positive of 40 in each; p = 1.8e-12) with codes and
scales bit-identical. Batch four gives −8.729 and −10.35.

The saving exceeds the 3.54 µs the two barriers cost — priced separately and
additively at 2.30 for a mixed `SyncAll<false>` and 1.25 for an AIV-only
`SyncAll<true>` — because two no-op stage calls, a whole-struct assignment, an
extra `pipe->Reset()` and the loop control also disappear.

## `gmm1-chunk-span.patch` — a further 2.1 µs per call

`UpdateWorkSpaceSplitConfig` derives the group-slot span this split covers by
walking the table, and at one split the answer is closed-form: every
cumulative row count is below `mLimit` by construction, so none of the three
breaks can fire, the loop always reaches the table's end, and the tail clamps
it to `groupListLen - 1`. Writing that answer saves `2·groupListLen + 1`
global scalar reads per core: 2.118, 2.414 and 1.611 µs over three runs
(σ 0.227, 0.300, 0.313; 0 positive of 40 each), bit-identical.

Measured on a toggle binary one arm of which executes exactly the submitted
code, so the figure is for the shipped form rather than an approximation.

## What these do not fix

78 layers of `gate_up` at 8.5 µs is 0.66 ms per round against a 6.866 ms
budget. `down` is out of reach: `w4a8.py:621,643` call
`torch_npu.npu_grouped_matmul`, a CANN built-in outside this package.

And the large item is untouched. A call's cost is linear in `group_list`'s
length at 0.1556–0.1825 µs per slot, worth 37–50 µs for a 256-slot table, but
only the 4.157 µs above is attributed to code. `midProcess` genuinely needs
per-slot counts because its weight addressing accumulates `wBaseOffset` rather
than indexing by `groupIdx`, and `postProcess` walks the table twice more.
Batching the table into UB will not help: the 769 reads these patches remove
cost 5.4 ns each, about ten cycles, so they already hit cache, while the
per-slot length term is about 296 cycles — thirty times apart.

## A third patch this repository already carries

`tests/glm52/patches/gmm1-swiglu-barrier.patch` is not an optimisation. The
shipped object — sha256 `2444c7e7…`, byte-identical to a no-patch rebuild —
given bit-identical inputs four times produces calls 0 and 1 agreeing while 2
and 3 each differ, codes by up to 114 and scales by 0.83 and 1.02, with all
five input tensors verified for zero drift around every call. Adding one
`PipeBarrier<PIPE_V>()` at `post.h:207` makes four consecutive calls agree,
giving `2eb81546…`. That is a live epilogue race in the exact shape production
runs, and the trajectory gate compares tokens bit for bit.
