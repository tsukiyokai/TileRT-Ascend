# Execution boundary experiments

A3 `a3-gs:trt-asc`, GLM-5.2-W4A8C8, TP16, batch one, frozen 8K
prefix. The target remains request TPOT ≤2 ms.

## Small-packet greedy selection

The candidate replaces full-vocabulary gather followed by argmax with the
pinned vLLM `LogitsProcessor.get_top_tokens`. Projection, quantization,
FP32 TP sums, AI_CPU HCCL and the M4 target graph stay unchanged; head
selection remains eager. Four head calls per steady round send 193,600
versus 80 logical bytes per rank. Collective count remains four.

[Precision and timing audit](top1-audit.json): 2,176 same-hidden head
checks and 4,512 token IDs agree, including M1/M2/M4 projections. Sixty-four
serial/speculative/graph/reset snapshots match the previous execution at
the tensor-bit level. Both long runs preserve all sixteen ranks' six
1024-token trajectories, proposals, acceptance and cache states.

| Sequential run | Three measured TPOT values (ms) | Median (ms) |
|---|---|---|
| Small packet | 22.062, 22.117, 22.292 | 22.117 |
| Matched full vocabulary | 20.674, 21.060, 20.229 | 20.674 |

Each run has three warmups, 303 rounds/request and actual acceptance
1023/303. The matched sources differ only in the two head implementations.
Small packets are 6.98% slower in this control and are not the default.
Fewer bytes do not establish lower latency; additional eager device work
has not been separately timed. These three-request measurements exclude
HTTP/HIXL and do not satisfy teacher or formal performance acceptance.

## Symmetric-buffer reduction

[SHMEM audit](shmem-audit.json) binds source, binaries, launch parameters
and logs for SHMEM `24c15d21`. The frozen push implementation passes two
PEs on dies 0/1: 12 KiB BF16 payload, four tiles, fixed-rank FP32
accumulation and one BF16 output rounding. Each rank checks 1,204,224
values through epoch 196. Thirty samples give rank-MAX event p50 27.77 µs
and wall p50 90.32 µs. Inputs rotate independently; this is not the
156-dependent-sum HCCL workload or the TP16 ≤6 µs acceptance.

TP16 driver mapping, ACL mapping and peer-pull variants all terminate
with runtime error 507015 before timing. The push ELF's first fault is a
DCCI after remote MTE3 output; its L2 hint is present. The precise peer
access failure is unresolved. The separate built-in barrier has an
FFTS parameter-offset inconsistency; replacing that barrier enables the
two-PE run but does not fix TP16. Keep working HCCL for decode controls.

## Raw artifacts

Under `/home/shanshan/trt-asc/offline/` on a3-gs:

- `cann-small-top1-20261018-01/`: precision reports, snapshots, frozen
  candidate source, patch and CPU audit.
- `cann-small-top1-long-20261018-01/`: candidate timing and CPU audit.
- `cann-full-vocab-control-20261018-01/`: matched control and CPU audit.
- `shmem-oneshot-20261017-01/`: both source variants, binaries, per-rank
  logs, CSV samples and ELF audit. The failed prototype stays here.
