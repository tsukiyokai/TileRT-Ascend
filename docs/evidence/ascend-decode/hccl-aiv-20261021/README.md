# HCCL_OP_EXPANSION_MODE=AIV: sixteen-rank collectives 1.6 to 1.9x faster

2026-10-21, a3-84, sixteen ranks on one node, bf16 over hidden 6144.

A3's collectives expand on the device-side AI CPU by default. The error message
for an invalid value names the legal ones: `Atlas A3: AI_CPU | AIV`. Switching
to the vector cores:

| rows | KB | | all_reduce | reduce_scatter | all_gather |
|---|---|---|---|---|---|
| 12 | 144 | default | 300.56 | 318.09 | 326.74 |
| 12 | 144 | **AIV** | **183.40** | **177.41** | **175.40** |
| 24 | 288 | default | 332.77 | 311.13 | 306.79 |
| 24 | 288 | **AIV** | **172.75** | **167.78** | **186.11** |
| 48 | 576 | default | 303.35 | 287.69 | 438.59 |
| 48 | 576 | **AIV** | **186.12** | **171.34** | **396.32** |

Three repeats of the all_reduce, no overlap between the arms:

    rep1  default 298.08   AIV 163.04   1.83x
    rep2  default 276.65   AIV 171.30   1.62x
    rep3  default 281.33   AIV 178.31   1.58x

The lowest default is 276.65 and the highest AIV is 178.31, so the separation
is clean rather than a mean difference.

## It takes the fixed cost, which is what the decode pays

The one case that barely improves is the 48-row all_gather: 438.59 to 396.32,
1.11x, and it is the only item actually moving a large payload. Everywhere else
the gain is 1.6 to 1.9x. So AIV removes expansion and synchronisation overhead,
not transport -- exactly the 193 us floor this project measured at two ranks,
where moving the same bytes to pinned host memory costs 37 us.

Decode sizes sit entirely in the fixed-cost regime, so the decode takes the
full gain.

## Size

Collectives are 374 us a layer of the engine's round. At 1.6 to 1.9x that is
140 to 175 us a layer, **11 to 13.6 ms a round** -- an order of magnitude above
the bank narrowing's 4.63, and it is one environment variable.

## What is not yet known

Whether the engine's absorbed collectives take the same path. The SuperKernel
wraps most of them -- the standalone count is 0.26 calls a layer -- and whether
a wrapped collective honours `HCCL_OP_EXPANSION_MODE` is unmeasured. The
variable is read at communicator init, so it should apply to every collective
on that communicator, but "should" is what this page does not get to say.

## The mode is per process, which answers the open question

Setting the variable after the communicator exists, and then building a second
communicator over the same ranks:

    started with AIV, flipped to AI_CPU
      before the flip            183.73 us
      after, same communicator   181.88
      a new communicator         181.89

    started unset, flipped to AIV
      before the flip            317.52 us
      after, same communicator   298.58
      a new communicator         319.77

Neither direction moves, and a **freshly built communicator keeps the mode the
process started with**. So the variable is read once when HCCL loads, not at
communicator init and not per call.

Which settles what the previous section could not say. The mode is a property
of the process, so every collective in it takes the same path -- including the
ones the SuperKernel wraps, which is most of them. The 11 to 13.6 ms a round
applies to the whole round rather than to the 0.26 standalone calls a layer.

And it makes the application trivial: set `HCCL_OP_EXPANSION_MODE=AIV` in the
engine's environment. No code.
