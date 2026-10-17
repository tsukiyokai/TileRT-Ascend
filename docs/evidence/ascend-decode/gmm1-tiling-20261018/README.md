# Existing CANN GMM1 tiling screen

A3 shared die 0, one TP16 weight shard, real layer-3 M4 routed inputs:
32 active experts, each with one row. The [private host patch](base-n128.patch)
changes only `baseParams.baseN` and `mmTiling.baseN` from 256 to 128 for
the fixed A8W4 MSD shape K=6144/N=256. The validated SwiGLU device binary
and its barrier remain unchanged; the other SDK tiling fields are preserved.

| Clone + complete GMM1 graph | p50 (µs/call) | p95 |
|---|---|---|
| Existing baseN=256 | 121.525 | 482.393 |
| Candidate baseN=128 | 124.588 | 488.564 |

Each case has ten calls/graph, ten warmup graphs and thirty timing samples.
MSD mutates its input, so each captured call clones the same frozen A8 rows.
Eight eager checks and the last saved graph output match the actual M4
fixture exactly. The [audit](m4-audit.json) independently recomputes timings
and verifies 4,128 output elements per saved graph; other timed outputs
were not archived. A separate debug execution confirms final `baseN=128`
and passes all eight functional replays. SDK logs printed before the
override still show 256; the final operator tiling is 128.

There is no measured gain. Retain the existing tiling and stop this branch.
These shared-device times do not measure model TPOT. Raw fixtures, the
58-check source audit, vendor package, launches and logs remain under
`/home/shanshan/trt-asc/offline/cann-gmm1-basen128-20261018-01/` on a3-gs.
