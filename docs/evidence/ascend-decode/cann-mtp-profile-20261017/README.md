# CANN M4 stage capture

Environment: `a3-gs:trt-asc`, GLM-5.2-W4A8C8, TP16, batch one,
full 8192-row frozen prefix, M4 target ACL graph, eager draft and FP32
TP reduction with `AI_CPU`. The fixture first repeats serial, speculative,
graph and dirty-reset validation on the same target/draft objects. All
sixteen ranks pass before three warmup requests and one captured request.
Each request emits six tokens in three rounds and resets position 8197
to zero.

The [capture audit](profile-audit.json) binds all sixteen validation,
profile and input receipts to the actual source and 1570 raw files
(177,694,819 bytes). The [CPU analysis](summary.json) exports ranks 0/14
and assigns every compute task without ambiguous flows. Graph identity
also appears in the actual exported ModelID49 execution.

## Observed stages

Rank zero, captured rounds two and three. Stage windows run from one
stage's CPU begin to the next stage's begin, including dependent waits.

| Window | Round 2 (ms) | Round 3 (ms) |
|---|---|---|
| Two chained drafts | 16.766 | 16.794 |
| Verify → conditioned draft | 49.020 | 49.348 |
| Conditioned draft → commit | 7.135 | 6.946 |
| Commit → round end | 1.525 | 1.517 |
| Whole round | 74.492 | 74.651 |

Within target verification, compute interval unions are 40.698/40.576 ms;
HCCL unions are 5.313/5.811 ms. The CANN parser labels 4.100/4.598 ms as
Notify_Wait and 1.213/1.213 ms as transit. These intervals can overlap and
are not additive critical-path terms or bare-network measurements.

| Target operator | Calls/round | Duration sum, rounds 2/3 (ms) |
|---|---|---|
| GroupedMatmulSwigluQuantV2 | 75 | 7.807 / 7.727 |
| GroupedMatmul | 75 | 5.538 / 5.526 |
| SparseFlashAttention | 78 | 7.042 / 7.037 |
| QuantBatchMatmulV3 | 255 | 3.830 / 3.809 |
| MlaPrologV3 | 78 | 2.977 / 2.967 |

The 2649 ModelID49 rows include 157 communication envelopes and 157
AICPU rows; they are not 2649 independent compute launches. Triton RoPE
tasks are assigned separately. In one aligned collective, rank 14 enters
0.67451 ms later than rank 0, but their ends differ by only 0.00105 ms.
This proves pairwise arrival waiting; two ranks do not identify the global
last arrival across sixteen.

## Interpretation and reproduction

The [profiler-off sessions](../cann-graph-20261017/README.md#m4-1024-output-diagnostic)
have median TPOT 38.70 ms for 32 outputs and 20.44 ms for 1024 outputs,
using the same source and prefix. Actual tokens/round rise from 1.824 to
3.376. This capture locates the short session's work; the profiled
74.5 ms round does not replace either performance measurement.
GMM is the largest target family, but its presence here does not establish
an additional cost relative to vLLM, which also uses these operators.
The [baseline extraction](baseline-comparison.json) proves actual backbone
compilation and ACL replay; this target captures an eager model. Both
launch configurations specify eager drafting, though the baseline logs
eagle-head compilation. Baseline AIV/native reduction also differs from
the FP32/AI_CPU diagnostic. Compare compiler, reduction and control costs
separately before attributing the remaining gap. Then freeze real M4
expert inputs for one-die CANN tiling comparisons.

Run `run_cann_mtp_rounds.py --target-graph --profile-output DIR` with the
same TP16 checkpoint, frozen-prefix case and sixteen-output validation.
Raw artifacts remain on a3-gs under
`/home/shanshan/trt-asc/offline/cann-mtp-profile-20261017-01/`.
`profile-audit.py` checks bindings; `analysis/export.py`, `analyze.py` and
`summarize.py` regenerate the exported stage/communication analysis on CPU.
The source is `ed3cbc0` plus the three overrides recorded in the audit;
HTTP and HIXL are outside this capture.
