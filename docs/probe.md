# Experiment log

Target: A3, GLM-5.2-W4A8C8, one machine, batch one, 8K input, TPOT ≤2 ms.
Experiments run on `a3-gs:trt-asc`; raw artifacts stay on that server.

| Date | Experiment | Observation | Next decision |
|---|---|---|---|
| 2026-10-17 | [M1 ACL graph](evidence/ascend-decode/cann-graph-20261017/README.md#m1-device-result) | Thirty-two steps; graph/eager and reset tensors have identical bits. Three-request internal median 44.68 ms/token. | Measure the M4 session. |
| 2026-10-17 | [M4 target graph](evidence/ascend-decode/cann-graph-20261017/README.md#m4-target-graph-and-session-timing) | Sixteen-output KV comparisons agree. Thirty measured 32-output requests: TPOT p50/p95 38.70/39.02 ms; round median 69.86 ms; actual acceptance 31/17. Draft stays eager. | Capture verify, draft, acceptance and communication stages. |
| 2026-10-17 | [Live PD HTTP](evidence/ascend-decode/pd-live-20261017/README.md) | Fresh 8192-token prefill; sixteen HIXL WRITEs; seven real M1 steps; HTTP 200 with eight tokens; position 8192→8199, reset complete. | Retain this functional check; measure performance after warmup. |
| 2026-10-17 | [Die 0 HBM reads](evidence/a3-hbm-20261017.json) | Four rotating 513 MiB inputs; 100 warmup/1000 samples; 15 checksum checks pass. Event median 1.155 TB/s, wall median 1.019 TB/s. | Use as a streaming-read budget; measure W4 operator traffic separately. |
| 2026-10-17 | [M4 stage capture](evidence/ascend-decode/cann-mtp-profile-20261017/README.md) | Sixteen ranks revalidate; three captured rounds. Later target windows ≈49 ms; expert GMM sums ≈13.3 ms. These are profiled values. | Compare compiler/reduction settings before relative attribution. |
| 2026-10-17 | [M4 long diagnostic](evidence/ascend-decode/cann-graph-20261017/README.md#m4-1024-output-diagnostic) | Same source/prefix; three warmups and three 1024-output requests. Median TPOT 20.44 ms; 303 rounds, actual acceptance 1023/303. All six requests agree. | Isolate eager-vs-compiled target, FP32/AI_CPU reduction and round-control costs. |
| 2026-10-17 | [AIV control](evidence/ascend-decode/cann-graph-20261017/README.md#ai_cpu-versus-aiv-control) | Only HCCL mode changes; all sixteen ranks' tokens, proposals, acceptance and state match. Median TPOT 21.47 ms versus AI_CPU 20.44 ms; no gain. | Retain AI_CPU and compare compiled target execution next. |

The live check loads D after P exits while keeping device KV alive. Its
117.59 s request includes weight loading and first-use initialization;
it is not a steady-state TPOT measurement. Long teacher trajectories and
the 32-request, 1024-output performance workload remain release checks.
