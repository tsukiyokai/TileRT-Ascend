# Attention's cost is 88 percent fixed, not the KV read

One die, twelve rows, four heads, the cache capacity swept:

| capacity | MB | us | GB/s | us a 1024 rows |
|---|---|---|---|---|
| 512 | 0.6 | 171.75 | 3.4 | 335.44 |
| 1024 | 1.2 | 169.78 | 6.9 | 165.80 |
| 2048 | 2.4 | 172.10 | 13.7 | 84.03 |
| 4096 | 4.7 | 182.48 | 25.9 | 44.55 |
| 8192 | 9.4 | 195.65 | 48.2 | 23.88 |
| 16384 | 18.9 | 207.96 | 90.8 | 12.69 |

Thirty-two times the capacity costs 21 percent more time. The fit is about
**170 us of intercept and 2.3 us a thousand cache rows**, so at the engine's
8192 rows the context is 26 us and the other 170 is independent of what is
read.

## Which corrects a claim made one turn earlier

That turn said attention's cost "equals reading 9.4 MB, a physical quantity".
9.4 MB in 196 us is 48 GB/s, about three percent of this die's HBM -- so it was
never the read, and the arithmetic to check it was one division.

## And it explains two earlier results

The gathered path loses because it keeps the same fixed cost while quadrupling
the bytes: selecting 2048 rows per query row moves 28 MB at width 12 against
the full cache's 9.4, and the bytes were never the binding term.

Sixty-four heads cost 1.85 times four heads rather than sixteen times, for the
same reason -- the fixed term dominates and the head count rides on top of it.

## What it opens

A 170 us intercept on an operator called once a layer is 13.3 ms a round, the
largest fixed cost in the model after the collectives. Nothing about it is the
cache, the heads or the rows. Whether it is the operator's tiling, its
workspace, or its own internal launches is the next question, and the profiler's
per-kernel breakdown inside the FIA call is where it would show.

## Inside the call: one kernel, 49.52 us

Profiled on one die, twenty calls at the engine's shape:

    20 kernel records
    aclnnFusedInferAttentionScoreV4_FusedInferAttentionS   20   49.52 us

One device kernel a call. So of the 171 to 196 us measured around the call,
**49.52 us is on the device and about 120 us is host-side** -- the timing used
`perf_counter_ns` around `torch.npu.synchronize()`, which measures the host
round trip.

That is far more than this project's measured 11.386 us of eager dispatch for a
kernel, so the FIA call spends it in its own argument handling: seventeen
parameters including Python lists for `actual_seq_qlen` and `actual_seq_kvlen`,
rebuilt every call.

## Which corrects the 170 us intercept's attribution

The intercept is real -- the capacity sweep stands -- but it is not the
operator's device work. It is the host path into the operator, and in a graph it
does not exist: today's per-item table measured the same call at 0.374 of its
eager cost, 240 us to 81, and 81 includes the replay's own overhead.

So attention's device cost is **49.52 us a layer, 3.9 ms a round**, not 309 us
and 24 ms. The 309 was eager, host-dominated, and the engine replays its layers
from a graph.

## And it reprices the whole compute half

Every per-operator figure this session took with `perf_counter_ns` around a
synchronise carries the same host tax. The ratios survive -- both arms paid it
-- but the absolute attributions do not, which is why the in-graph table was
the one worth keeping and why the TP16 layer's 485 us of compute is an upper
bound on what a captured layer spends.
