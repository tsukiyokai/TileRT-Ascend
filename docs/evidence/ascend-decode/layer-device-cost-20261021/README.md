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

## Correction: a layer's MoE is two calls, 258.17 us, not one at 127.17

The layer account used a single grouped matmul at `inner=1536`, which is
neither the checkpoint's width nor the call count. A layer's MoE runs the fused
gate-and-up bank and then the down bank:

    gate_up  (16, 6144, 2048) int8
    down     (16, 2048, 3072) int8
    302.0 MB, 2.0 kernels, 258.17 us, 1170 GB/s
    a round of 78: 20.14 ms

So the layer's device time is 228.67 - 127.17 + 258.17 = **359.67 us**, and a
round is **28.05 ms** rather than 17.84.

## Which makes the round's account consistent

    device compute          28.05 ms
    156 collectives at 190  29.64
    sum                     57.69
    the measured round      50.73

The sum now exceeds the round, which is what overlap looks like -- and overlap
is what this session measured three times, where a serial pair already costs
less than its parts. The earlier account had compute at 17.84 and summed to
47.48 against 50.73, which looked tidy and was short a call.

## And it reprices the narrowing consistently

Sixteen experts cost 258.17 us a layer and two cost about 35, so the narrowing
saves **223 us a layer, 17.4 ms a round** -- 62 percent of the 28.05 ms compute
term it acts on.

The previous figure of 22.2 ms claimed more saving than the compute term
contained, which was the tell: a lever cannot remove more than the item it acts
on. That arithmetic check is cheaper than any measurement and was available the
moment both numbers existed.

## The MoE's plumbing, which the synthetic layer omitted

The capture counts three operators once a layer that the eight-operator layer
did not hold: `GatherV3` at 78 a round, `ForeachAddList` at 75, and a second
`DynamicQuant`. Profiled at the decode shape -- twelve rows, eight picks each,
ninety-six sorted rows:

| operator | us | ms a round |
|---|---|---|
| the routing gather, 96 of 12 rows | 8.45 | 0.66 |
| the expert combine, 96 into 12 | **26.98** | **2.10** |
| a dynamic quant, 12 rows | 3.37 | 0.26 |
| together | 38.80 | 3.03 |

The combine is the largest: a weighted scatter-add of ninety-six rows back into
twelve, which reads 96 x 6144 bf16 and writes 12 x 6144 with a permutation
between. Its 26.98 us against the gather's 8.45 for the same traffic in the
other direction is the scatter's cost.

## The round's account, as far as it goes

    measured device compute, eight operators   28.05 ms
    the MoE's plumbing, three more              3.03
    collectives in a graph at a round's scale   5.14
    sum                                        36.22
    the measured round                         50.73
    unattributed                               14.51   186 us a layer

Down from 17.54 and still the largest unexplained term. What is left can be
bubbles between kernels inside the graph, operators still not enumerated, or
synchronisation points between device and host -- and only the queued
`--round-phases` and `steady` arms separate those.
