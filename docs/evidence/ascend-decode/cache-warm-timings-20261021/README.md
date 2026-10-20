# Today's operator timings were taken out of cache

2026-10-21, a3-84, one die. A 6144x6144 bfloat16 matmul, 72 MB of weight,
rotating over a bank of distinct weights between repeats:

    distinct weights   us a matmul   effective GB/s
                   1        31.056             2264
                   2        30.631             2295
                   4        62.635             1123
                   8        62.284             1129
                  16        61.886             1136

2264 GB/s exceeds what HBM gives, so one or two weights are resident on chip.
From four weights the figure settles at 62 us and 1136 GB/s, which is the real
cost of reading a weight that is not resident.

The engine issues 610.9 kernels a round at 52.7 us each. That sits between the
warm 31 and the cold 62, nearer the cold one -- which is what a model reading a
different weight every layer should do.

## What this means for today's measurements

Every single-operator timing today used one weight repeated: the projections at
12 to 19 us, the grouped matmul at 62 to 70, attention at 85, a layer at 214 in
a graph. They were warm.

The ratios survive, because both sides of every comparison were equally warm:
four projections fused against four separate is still 37.238 against 68.724,
the routed experts in one launch still 72.463 against 181.172, the custom
operator path still 104.929 against 66.813. So the 21.04 ms of boundary merges
holds as a proportion of what the round actually spends.

The absolute figures do not. A layer is 52.7 x 7.8 = 411 us in the engine
against the 214 a warm synthetic layer replays in, and the round's 57.27 ms of
device time is the number to reason from -- not the 16.7 ms the warm sum
suggested.

So the projections of 2.98 and 3.73 ms, both built on warm sums, are optimistic
by an unknown factor below two. The end-to-end arms remain the only figures
worth quoting.

## Cold, the fused projections save more, not less

The expectation above was that cold operators cost more, so a boundary is a
smaller share and fusing saves proportionally less. Measured over eight
rotating weight sets, 52 MB each:

    separate   119.119 us   9.29 ms a round
    fused       73.142 us   5.71 ms a round
    saving      45.978 us   3.59 ms a round   (warm said 2.46)

Both sides got dearer -- 119 against 68.724, 73 against 37.238 -- and the
saving grew by half. Fusing reads the same bytes in one contiguous pull instead
of four separate ones, and that is worth more when the bytes actually come from
memory than when they come from cache.

So the direction of the correction is the opposite of what this page first
claimed: the warm figures understate the boundary merges rather than
overstating them. What stands is that warm absolutes cannot be summed into a
round -- the engine's own 57.27 ms of device time is the denominator -- and that
every lever needs its cold figure before the total means anything.
