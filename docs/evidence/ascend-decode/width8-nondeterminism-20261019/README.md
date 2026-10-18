# Width 8 is not a valid result yet: the trajectory diverges run to run

Three arms, one tree, one frozen prefix, 1024 output tokens, three warmups and
three measured requests each, `--target-graph --draft-graph
--draft-projection-tp --compile-target --super-kernel --attention-kernel sfa
--fp32-head`:

| arm | median TPOT | range | rounds per request | token hashes |
|---|---|---|---|---|
| width 4 | 14.932 ms | 0.169 | 307, 307, 307 | `df10d06f911c` ×3 |
| width 8 | 13.276 ms | 0.940 | 220, 236, 238 | three distinct |
| width 8 + hoist | 13.041 ms | 4.080 | 287, 246, 212 | three distinct |

**Width 4 is deterministic and width 8 is not.** The same binary, prefix and
flags give three different token streams at width 8, with acceptance ranging
4.655, 4.339, 4.303 and the round count varying by 8%. The hoist arm is worse:
3.568 to 4.830.

So the 4.08 ms spread is not timing noise — it is different trajectories
taking different numbers of rounds. Which means two things.

The width-8 TPOT is confounded: its difference from width 4 mixes a real
speed change with an acceptance change, and the project's own ablation
established that acceptance changes are not speed. And the hoist's −0.235 ms
here is unreadable, sitting far inside its own arm's spread, against the
−0.7967 it measured at width 4 with tight ranges.

More seriously, **width 8 fails the trajectory gate**, which compares tokens
bit for bit.

## The attribution, and it is already patched in this repository

The installed kernel is the original: the source at
`csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h`
has no barrier after the `DataCopy` at line 209, and the vendor binary is
timestamped 16:20, before any of tonight's builds.

That is exactly the epilogue race measured independently: the shipped object,
given bit-identical inputs four times, produces calls 0 and 1 agreeing while 2
and 3 each differ, codes by up to 114 and scales by 0.83 and 1.02, with all
five input tensors verified for zero drift around every call. Adding one
`PipeBarrier<PIPE_V>()` after that copy makes four consecutive calls agree.
[decomposition](../gmm-fastpath-20261019/README.md)

`tests/glm52/patches/gmm1-swiglu-barrier.patch` carries that line and is not
installed.

## The barrier works, and my M-threshold guess was wrong

Installed as a vendor overlay and measured at 64 consecutive calls per cell,
two independent processes per cell, inputs rewritten from a frozen copy before
every call, and **a scratch allocation churned between calls**:

| object | M=8 | M=32 | M=64 |
|---|---|---|---|
| factory `2444c7e7` | 2 results, 1 of 64 wrong, max diff 68 | **4 results, 3–4 of 64, 115** | 2 results, 1 of 64, 114–115 |
| barrier `2eb81546` | 1 result, 0 wrong | 1 result, 0 wrong | 1 result, 0 wrong |

384 calls with the barrier, zero disagreements. The factory object mismatches
10 of 384, and a difference of 114–115 on an int8 range of ±127 is an element
written wrong rather than rounding drift.

The load order was measured rather than inferred, because installing it in the
wrong order fails silently: barrier first gives eight identical results;
pristine first gives two distinct results with call 4 differing by 99. CANN
takes the first matching vendor, so the prepend is

```
ASCEND_CUSTOM_OPP_PATH=<overlay>/vendors/custom_transformer:/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
```

with the second element spelled exactly so that `bootstrap_custom_op_env`
(`utils.py:454-458`) finds it already present and leaves it in second place.

**But my hypothesis that the race needs a large M is refuted.** The factory
object is worst at M=32 — four distinct results, three or four of 64 wrong —
and M=32 is width 4's verify shape, the arm I observed as clean. M=64 is only
one in 64. So width 4's reproducibility is not because its shape fails to
trigger the race, and the attribution behind this withdrawal is not yet
established.

Two candidates remain, and the decisive test is re-running the three arms with
the overlay prepended. Either the trajectory's sensitivity differs — a
corrupted expert row only probabilistically flips an argmax, and at width 4 a
polluted draft the verifier rejects still yields the same accepted token,
while at width 8 the measured quantity *is* the acceptance rate, so one
flipped decision changes rounds, hash and TPOT together — or width 8 has a
second source, with the `_int64_kv_slots` identity caching as the named
candidate.

## A tight loop is not a determinism test

The first attempt at this matrix fired eight identical calls inside one
process and read **all ten cells as deterministic, including the known-bad
factory object**. A tight loop does not change the workspace the operator
receives, so reads of regions it never writes return the same stale bytes
every time and the race is hidden. Adding an allocation churn between calls
exposed it immediately in three shapes.

The churn is not adversarial, it is faithful: a real round allocates and frees
hundreds of tensors between two expert matmuls.

This pairs with the earlier finding that a harness locally defined a constant
the patch had failed to import, turning a crash into a pass. Both are the same
failure: **the test supplied something the thing under test was missing** —
there a name, here a static environment it had no right to depend on.

## With the barrier installed, width 8 still diverges

Three arms re-run with the overlay prepended, same tree and prefix:

| arm | median | range | rounds | distinct token streams |
|---|---|---|---|---|
| width 4 | 14.533 ms | 0.141 | 307, 307, 307 | **1** |
| width 8 | 12.688 ms | 2.267 | 218, 255, 221 | **3** |
| width 8 + hoist | 13.030 ms | 0.610 | 247, 238, 243 | **3** |

**The barrier is not the cause.** Width 4 stays deterministic and width 8
stays at three streams, so this withdrawal's attribution was wrong and the
caution against accepting it was right. There is a second source at width 8.

The barrier should still be installed — 384 calls with zero disagreements
against the factory object's 10 — but it buys correctness on that operator
rather than reproducibility on the round.

## The next candidate is in my own script

`HCCL_DETERMINISTIC` is unset on this machine, and the benchmark only records
it into the report rather than setting it. With HCCL free to vary its
reduction order, a TP all_reduce's float sum differs run to run, which flips a
marginal `argmax` occasionally — and width 8 has more draft positions for one
to flip in, which is exactly the asymmetry observed. That is one variable and
it is being measured now.

If that is not it either, the remaining named candidate is the
`_int64_kv_slots` identity caching: it invalidates by tensor identity only
(`sfa_v1.py:459-474`), so a reused metadata object makes the check always hit
and `cache_index` keeps the first round's int64 copy. Three independent
safeties prevent it today and none of them is an assertion.

## Why both measurements of width 8 can be right

An earlier investigation reported five runs at width 8 emitting bit-identical
tokens; this one reports three streams from three requests. A race is
sensitive to timing, so both can hold — and that is the reason a determinism
check has to run in the same session as the number it licenses, rather than
being inherited.

## What has to happen before width 8 counts

Build and install the barrier-patched kernel, then re-run width 8 and check
that the three measured requests agree bit for bit and on their round count.
Only then is the median a speed claim. Until that is done, **width 8's
13.276 ms is withdrawn as a TPOT result**, and the width-4 figure of 14.932 ms
on this tree stands as the only deterministic point measured here.

Note what this does not touch: the hoist's −0.7967 ms was measured at width 4
with all six arms token-identical over 319 rounds, so it stands on its own
configuration.
