# A layer by device time: the grouped matmul is 56 percent, attention 22

One die, the engine's shapes, twelve rows, profiled rather than timed around a
synchronise.

| kernel | a layer | us | share |
|---|---|---|---|
| `aclnnGroupedMatmulV5` | 1 | **127.17** | 56% |
| `aclnnFusedInferAttentionScoreV4` | 1 | 50.28 | 22% |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 4 | 44.63 | 20% |
| `RmsNorm` | 1 | 3.33 | 1% |
| `aclnnDynamicQuantV2` | 1 | 3.27 | 1% |
| **a layer** | | **228.67** | |
| **a round of 78** | | **17.84 ms** | |

## Which agrees with the engine and overturns today's attribution

17.84 ms a round against the census's 16.4 ms of super kernel time -- within 8
percent, from two unrelated measurements. So this is what a captured layer
spends.

And it inverts the compute ordering. The host-timed TP16 probe put attention
and its projections at 308.97 us against the MoE's 198.88, and that was the
host tax: attention's seventeen-parameter call pays about 120 us of argument
handling that a graph removes. On the device the MoE's grouped matmul is 127.17
us and attention is 50.28.

## So the bank narrowing is aimed at the largest item after all

The grouped matmul is 56 percent of a layer's device time. At 7.53 us an expert
over a 55.57 intercept, cutting sixteen experts to the two a round touches takes
127.17 us to about 71 -- **56 us a layer, 4.4 ms a round against a device-side
round of 17.84**, a quarter of all compute.

And attention, which this session spent three experiments trying to attack, is
50.28 us a layer -- 3.9 ms a round, and its two candidate levers were measured
and excluded for good reasons that no longer matter much.

## The lesson the numbers carry

Every absolute figure taken with `perf_counter_ns` around a synchronise was a
host round trip. The ratios survived because both arms paid it; the ordering did
not. The profiler was available all along.
