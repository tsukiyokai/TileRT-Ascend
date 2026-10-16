# GLM-5.2 vLLM P+D baseline

Measured 2026-10-17 on `a3-gs:trt-asc`: one A3 host, 16 dies,
GLM-5.2-W4A8C8, TP16/DP1, batch one. vLLM performs both prefill and decode.
MTP uses three drafts; ACL Graph captures four target verification rows.

| Statistic | Result |
|---|---|
| TPOT p50 / p95 | 16.390 / 18.295 ms |
| TPOT mean / p99 | 16.514 / 18.542 ms |
| Token-weighted decode rate | 60.555 tok/s |
| Measured requests / warmups | 32 / 3 |

Each serial request consumes the same 8192 input IDs and returns exactly
1024 output IDs. TPOT is the interval between the first and last streamed
token arrivals divided by 1023; prefill and final usage/DONE events are
outside that interval. MTP events retain their complete token-ID spans.
The [raw requests](benchmark.json) contain IDs, text, arrival timestamps and
usage. The [result](result.json), [command](command.txt),
[runtime](environment.json), [startup](startup-summary.txt) and
[occupancy snapshots](occupancy.before.txt) identify the measurement.

The runtime is Torch/torch-npu 2.12.0, Transformers 5.14.1 and CANN 9.2.0
master20260917. vLLM is `c4df37df`; vLLM-Ascend is `f558b58c`.
[Build patches](source-patches.json) adjust dependencies and custom-op
compilation. `TORCHINDUCTOR_AUTOGRAD_CACHE=0` preserves npugraph compilation
and ACL Graph replay while avoiding the AOT OutputCode cache assertion.

All first tokens are 6500. With temperature zero and seed zero, the 32
complete output sequences differ; [one fixed trajectory](teacher-32.json)
provides teacher tokens for subsequent decode comparisons. Numerical
acceptance uses layer, logit and cache comparisons separately.

At p50, reaching 2 ms requires an 8.20-fold reduction from this baseline,
which already includes MTP and decode graph replay.
