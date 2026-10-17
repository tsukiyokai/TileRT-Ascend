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

## Observed expert unions

[Route audit](route-audit.json) checks 2,431 calls/rank from one graph
prewarm round, one warmup and one diagnostic 1024-output request. All
sixteen JSONL files are byte-identical; request trajectories and states
match the full-vocabulary control. A captured device copy refreshes every
layer's final expert IDs on replay, after poisoning the buffers with −1.
This instrumentation supplies no performance measurement.

For the diagnostic's 303 M4 rounds, target expert unions across four rows
average 27.121 per layer (median 28, range 14–32); conditioned MTP averages
29.729. Applying these IDs to the fixed 8K logical-read formula gives
6.833–7.481 GB/round for one/four cache passes. At measured A=1023/303 and
die 0 stream bandwidth, the HBM scenarios are 1.753–1.919 ms/token, leaving
0.247–0.081 ms for work that cannot overlap. Actual HBM reads, cache reuse
and model bandwidth remain unmeasured; the trajectory grows toward 9K.

## Communication decision

Continue one bounded HCCL-resource probe, then a producer/consumer fusion
experiment if peer access passes. The [open A3 deterministic executor](https://gitcode.com/cann/hcomm/blob/2b4f2a4ed628cbeb48d76eead015ef220ea5400b/src/legacy/ascend910/algorithm/base/alg_aiv_template/all_reduce/aiv_all_reduce_91093_deter.h)
already performs peer pull and reduction within one AIV kernel. This is
the source model; the installed trace proves AIV execution, without binding
it to that exact algorithm. Full
replication can remove its result-gather stage and GM reduction tree, but
raises per-rank network bytes from `2(P−1)B/P` to `(P−1)B`: eightfold at
TP16 for equal dtype, fourfold for BF16 transport versus current FP32.
At 48 KiB BF16, one-shot sends 720 KiB/rank; an 8 µs window requires at
least 92.16 GB/s of outgoing payload bandwidth before protocol work.

The measured 156-sum FP32 96 KiB chain is 5.044 ms. A hypothetical
156×8 µs scheme would use 1.248 ms, or release 1.124 ms/token of
communication budget at the observed acceptance. This is a budget
scenario, not a predicted TPOT gain. It still consumes 0.370 ms/token,
more than the unoverlapped budget above. Fusion/overlap and reduced reads
are therefore part of this route's acceptance, alongside standalone speed.

The next probe uses the installed `HcomGetAlgExecParam` interface to
establish AIV resources and inspect the opaque context. It needs verified
peer-table ABI before a 32-byte device-read check. Keep initialization,
mapping, ordered reduction and end-to-end latency as separate results.

The [initial resource queries](hccl-query-audit.json) both hit the 60 s
timeout. In the phased second run, all sixteen ranks first pass the
6144-element FP32 control, then enter the native shim without returning.
The individual API wait is not yet localized; no peer read ran.
Source inspection identifies a graph-workspace registration prerequisite.
A variant registers 64 KiB under the same tag and marks individual native
calls; it is CPU-built. Its launch preflight finds another job on all
sixteen dies, so this variant has no device result yet.

## Raw artifacts

Under `/home/shanshan/trt-asc/offline/` on a3-gs:

- `cann-small-top1-20261018-01/`: precision reports, snapshots, frozen
  candidate source, patch and CPU audit.
- `cann-small-top1-long-20261018-01/`: candidate timing and CPU audit.
- `cann-full-vocab-control-20261018-01/`: matched control and CPU audit.
- `cann-route-unions-20261018-02/`: final expert IDs and CPU audit.
- `shmem-oneshot-20261017-01/`: both source variants, binaries, per-rank
  logs, CSV samples and ELF audit. The failed prototype stays here.
- `hccl-peer-probe-20261018-01/`: installed-header query shim, sources,
  bounded runs and host phase markers.
