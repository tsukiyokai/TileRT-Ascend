# Experiment log

Target: A3, GLM-5.2-W4A8C8, one machine, batch one, 8K input, TPOT ≤2 ms.
Experiments run on `a3-gs:trt-asc`; raw artifacts stay on that server.

| Date | Experiment | Observation | Next decision |
|---|---|---|---|
| 2026-10-17 | [M1 ACL graph](evidence/ascend-decode/cann-graph-20261017/README.md#m1-device-result) | Thirty-two steps; graph/eager and reset tensors have identical bits. Three-request internal median 44.68 ms/token. | Measure the M4 session. |
| 2026-10-17 | [M4 target graph](evidence/ascend-decode/cann-graph-20261017/README.md#m4-target-graph-and-session-timing) | Sixteen-output KV comparisons agree. Thirty measured 32-output requests: TPOT p50/p95 38.70/39.02 ms; round median 69.86 ms; actual acceptance 31/17. Draft stays eager. | Capture verify, draft, acceptance and communication stages. |
| 2026-10-17 | [Live PD HTTP](evidence/ascend-decode/pd-live-20261017/README.md) | Fresh 8192-token prefill; sixteen HIXL WRITEs; seven real M1 steps; HTTP 200 with eight tokens; position 8192→8199, reset complete. | Retain this functional check; measure performance after warmup. |

The live check loads D after P exits while keeping device KV alive. Its
117.59 s request includes weight loading and first-use initialization;
it is not a steady-state TPOT measurement. Long teacher trajectories and
the 32-request, 1024-output performance workload remain release checks.
