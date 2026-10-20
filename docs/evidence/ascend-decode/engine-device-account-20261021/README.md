# What the engine's device spends a round on, deduplicated by name

From `profile-rope-20261018`, rank 1. Rounds from the attention call count,
244 over 78 layers: 3.13. Kernels appearing once across three rounds are the
domain build and are excluded.

    5544 repeated kernels, 57.27 ms of device time a round

    SuperKernel                610.9/round    52.7 us    32.21 ms   56.2%
    hcom_allGather              10.9/round   882.5 us     9.59 ms   16.7%
    hcom_allReduce               9.6/round   831.1 us     7.97 ms   13.9%
    FusedInferAttentionScore    78.0/round    37.5 us     2.92 ms    5.1%
    Cast                       533.5/round     2.6 us     1.40 ms    2.4%
    MatMulV3                     7.7/round    61.2 us     0.47 ms    0.8%
    Muls                        75.1/round     4.5 us     0.34 ms    0.6%
    MatMulV2                    60.4/round     5.4 us     0.33 ms    0.6%

Device busy of 57.27 ms against a measured round near 52.8: the vector and cube
halves run concurrently so the durations add past the wall clock.

## The super kernel does not reduce the kernel count

610.9 SuperKernel launches a round over 78 layers is 7.8 a layer, at 52.7 us
each. A synthetic layer of eight operators measured 214 us in a graph on one
die -- 26.8 us an operator. So the engine runs about the same number of kernels
a layer and each costs twice what the same work costs on a single die at full
width.

That factor of two is the shard: at TP16 each kernel does a sixteenth of the
work and still pays its own fixed cost. It is the same observation the fused
projections rested on -- the router's 3 MB running at 158 GB/s where q_a's
24 MB reaches 1386 -- now visible at the level of the whole round.

## Where the round's time is, in one line each

    32.21 ms   610 kernels a round, each a sixteenth of an operator's work
    17.56 ms   20.5 collectives a round at 858 us
     2.92 ms   attention, outside the super kernel
     1.40 ms   533 casts at 2.6 us

The first two are 87 percent. Fewer, larger kernels is what the five boundary
merges do -- 21.04 ms measured on one die -- and the collectives are what the
one-sided push replaces. Both are built; neither has run end to end.
