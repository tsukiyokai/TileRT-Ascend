# What the engine's device actually spends a round on

From this project's own capture, `profile-rope-20261018`, rank 1. Rounds from
the attention call count, 244 over 78 layers: 3.13 rounds.

    decode kernels            5544
    device sum              179.14 ms
    device busy a round      57.27 ms

    SuperKernel              32.21 ms a round   56%
    hcom_allGather            9.59 ms           17%
    hcom_allReduce            7.97 ms           14%
    FusedInferAttentionScore   2.92 ms            5%
    Cast                      1.40 ms            2%

## Two readings this settles

**The engine is not paying eager cost inside a graph.** Device busy of 57.27 ms
a round meets or exceeds the measured round of about 52.8 ms at that width,
which happens because the vector and cube halves run concurrently and their
durations add. So the device is saturated and the engine already replays at
device cost. The 104 us a layer that a graph would save, read off a synthetic
eager layer this morning, is not available -- that correction stands withdrawn.

**The round is a SuperKernel plus collectives, not a sequence of operators.**
Fifty-six percent is one fused kernel and 31 percent is HCCL. Every operator
measured on one die today -- attention, the grouped matmul, the projections,
the norms -- lives inside that SuperKernel, and their graph-replay sum of
214 us a layer is 16.7 ms a round against the SuperKernel's 32.21. So either
the SuperKernel carries work my synthetic layer omits, or it carries its own
scheduling cost; the capture does not separate those.

## Which relocates the largest lever

Collectives are 17.56 ms a round of device time, where the one-sided reduce was
costed at 6.5 ms from an operator-level comparison -- 32.488 us against HCCL's
201.995 at sixteen ranks. If the 17.56 ms is what that replaces, the lever is
larger than recorded, and it is already built and correct at sixteen ranks.

That makes the queued one-sided arm the most valuable measurement outstanding,
ahead of the routed kernel. The honest projection stays what the end-to-end
arms produce, not what an operator-level sum suggests.
