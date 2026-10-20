# A layer's routed experts in one kernel, and what it costs

`a3-187`, one die, `trt-asc` with CANN 9.2.0. Hand-written AscendC, built by
`bisheng` at `--npu-arch=dav-2201`, checked against a host reference that
computes both INT32 products exactly and rounds where the device rounds.

## The shape, and why

The expert axis is the split. Three measurements forced it:
[the cube is flat in rows](../routed-split-20261020/README.md) -- 10.742 µs at
two against 11.416 at thirty-two -- so a matmul an expert would be 440 µs a
layer against the vendor's 165; what the cube is short of is cores, 76, 120
and 145 GB/s at one, two and four blocks; and a permuted row belongs to one
expert and therefore to one core, which makes the per-token maximum for the
intermediate's quantisation core-local. So this kernel has no grid barrier at
all -- neither of the two `SyncAll<false>()` calls the shared expert's chain
needs, nor its cross-block reduction.

Each cube block walks its own experts serially: gate and up projection at full
width, flag, dequantise and SwiGLU and requantise per token, flag back, down
projection, flag, scale to BF16. The two vector subblocks of a block share the
6144-wide epilogue; subblock zero alone runs the narrow 128-column activation
and answers the cube.

## What it costs

Steady state over fifty launches a round, three rounds, three separate
processes. Correctness is reported for every configuration, sentinel included,
so an unwritten element cannot pass as a right one.

| experts | blocks | wall, µs | against the host reference |
|---|---|---|---|
| 4 | 4 | 47.108 | 0 of 49152 |
| 8 | 4 | 85.090 | 0 of 98304 |
| 16 | 4 | 162.047 | 0 of 196608 |
| 8 | 8 | 47.755 | 0 of 98304 |
| 16 | 8 | 87.104 | 0 of 196608 |
| 24 | 8 | 125.433 | 0 of 294912 |
| **16** | **16** | **49.276, 49.279, 49.725, 49.815, 49.791, 49.666** | **0 of 196608, every run** |
| 44 | 16 | 128.234 | 0 of 540672 |
| 44 | 22 | 91.953 | 0 of 540672 |

The wall tracks experts a block, not experts: four on four blocks and sixteen
on sixteen cost the same, while sixteen on four costs three times as much.
That is the expert-axis split behaving as designed.

## The figure that matters

Sixteen experts on sixteen blocks is the per-rank shape. The profile is rank
zero's, and at TP16 with 256 routed experts a rank holds sixteen; a round at
width 32 puts 32 tokens through top-8, so 256 selections spread over 256
experts and a rank's own sixteen are each hit once or twice.

So 49.3 to 49.8 µs a layer against the vendor's 165.2 -- `GroupedMatmul-
SwigluQuantV2` at 6.793 ms over 75 layers plus `GroupedMatmul` at 5.596 --
which is 3.3 times, and within 12% of the 45 µs byte floor this chain was
given. Over a round: 3.71 ms against 12.39, saving 8.68 ms, which at
acceptance 7 is 1.24 ms of TPOT.

The comparison is conservative in one direction and incomplete in another.
This kernel also absorbs the BF16 epilogue, which the two vendor ops do not
include; it does not do the final unpermute and weighted sum, which is a
separate op in the vendor graph too.

## What is not established

Rows an expert is fixed at two, the measured average of 1.45 rounded up. The
cube is flat in rows, so the cost should hold under a real group list, but
variable rows are unmeasured and the host side of a group list is a separate
change. And none of this is integrated: the graph still runs the vendor's two
grouped matmuls, so no end-to-end TPOT has moved.
