# Experiment log

A3, GLM-5.2-W4A8C8, batch1,8K; target TPOT2.xms, release budget≤2ms.
Current best measured request remains11.6198ms. Whole-model timing needs an
idle host; CPU replay and bounded single-die correctness continue independently.

| Date | Evidence | Result | Next decision |
|---|---|---|---|
| 10-17 | [Live PD HTTP](evidence/ascend-decode/pd-live-20261017/README.md) | Fresh8K prefill,16 HIXL writes,7 M1 steps, HTTP200/eight tokens; reset completes.117.59s includes model load. | Functional evidence only; keep warm decode TPOT separate. |
| 10-17 | [HBM](evidence/a3-hbm-20261017.json) | Die0, rotating513MiB inputs:1.155TB/s event median,1.019TB/s wall;15 checksum checks. | Price actual bytes and expert unions against this scope. |
| 10-18 | [Teacher/head](evidence/ascend-decode/layer-major-20261018/README.md) |78 layers×32 M1 rows; BF16 head31/32 due to a tie, FP32-output head32/32 at M1/M4. | Eight-prompt and long-session acceptance remain. |
| 10-18 | [Complete push](evidence/ascend-decode/execution-gap-20261018/README.md#complete-one-shot-push) | Changing-input/epoch checks pass. Shared-run protocol p50:35.72µs versus matched HCCL59.18µs. | No decoder gain claimed; pursue producer fusion rather than standalone tuning. |
| 10-19 | [Width 8 phases](evidence/ascend-decode/phase-synced-20261019/README.md) | Old configuration: target39.68ms, draft chain10.27ms, conditioned2.08ms; A≈4.507. | Reprice the actual new configuration; profiler gaps are not removable latency. |
| 10-19 | [Pinned staging](evidence/ascend-decode/metadata-h2d-20261019/README.md) | Two paired differences:−3.384/−3.791ms per round. Six pinned requests11.6198–11.6929ms; nine pageable12.3175–12.6394ms, under shared load. Loaded-source hashes distinguish the arms. | Retain pinned staging; refresh idle end-to-end measurements. |
| 10-19 | [Indexed copying](evidence/ascend-decode/lookup-contract-20261019/README.md) | Server CPU, one 8K+1024 trajectory,30 pairs/width: same proposals and boundaries; total lookup work≈1030ms→20.6–20.9ms including index build. Cache ownership, padding and final bonus corrected. | Device comparison prepared at `/workspace/experiments/context-copy-20261019/go.sh`; not launched. |
| 10-19 | [Unigram fallback](evidence/ascend-decode/lookup-contract-20261019/policy-summary.json) | Shortest suffix 2→1, longest 8: width 8/16/32 rounds253/198/174→228/171/147. Actual indexed implementation reproduces these counts. | Optional switch, default off. Width 32 A=6.959 gives13.92ms/round at2ms/token; measure target cost and expert union. |
| 10-19 | [Adaptive width](evidence/ascend-decode/lookup-contract-20261019/adaptive-summary.json) |175 rounds/2240 rows versus fixed32's147/4704. Seven wide rounds accept no draft despite an 8-token match. | Not enabled; compare measured bucket costs before choosing a policy. |
| 10-19 | [C1 SFA](evidence/ascend-decode/sfa-routing-20261019/README.md) | Full 44-op host, ordinary indices, M8/P8192: original C1, bounded C1 and bounded V1 give identical outputs. Same-ELF AB/BA event medians: V1 90.05/92.41 µs, C1 998.38/1002.53 µs; 120 raw samples,104 MiB timing peak reserved. | Reject dispatch-only C1; retain V1. Shared-device hot-input screening, no decoder TPOT gain. Full-K fixture does not exercise the bounds terminal branch. |
| 10-20 | [MLA mask](evidence/ascend-decode/fia-mask-20261020/README.md) | CPU masked/filtered FP64 max_abs 1.943e-16; eight single-die calls pass. All-False/no-mask FIA are bit-identical; safe-gather＋mask is invariant to future KV changes. | No timing or width-gate change. Check joint-query FIA on one full KV cache; initialize unused dense PV rows before launch. |
| 10-20 | [Shared FIA](evidence/ascend-decode/fia-mask-20261020/README.md#four-layer-sharing-and-integration) | M8/C9344, four-layer Graph: SFA425.780/shared364.470µs; paired ratio0.85445,30/30 faster under shared load. Actual binder A/B/reset:24/24 bit-exact versus rebuilt FIA,28 calls,136MiB peak. | Opt-in M8 integration; M1/MTP stay SFA. Frozen teacher, serial equivalence and whole-request gain pending. |
| 10-20 | [M32 shared FIA](evidence/ascend-decode/fia-mask-20261020/joint/m32-screen/summary.json) | Idle die2, four-layer full path:657.170→416.120µs; paired ratio0.63036,30/30 wins.12 shared/rebuilt and24 eager/Graph state comparisons bit-exact,272MiB peak reserved. | Enable explicit M32 candidate for lookup; measure whole-model teacher/serial, expert union and TPOT before promotion. |
| 10-20 | [Copy cursor](evidence/ascend-decode/lookup-contract-20261019/cursor/result.json) | Server CPU, committed pending-in-context contract: M8 rounds228→228, M16 171→170, M32 147→147. M32 verifies4704 rows in either policy. | Do not add a cursor switch; preserve the simpler lookup policy. |

| 10-20 | [Merged tree](evidence/ascend-decode/merge-tree-20261020/README.md) | Same trajectory, chain lengths dealt in four orders: merged k=31 acceptance 8.317/10.129/11.494/13.286. Break-even over lookup alone is 1.303 from the chain's measured 23.3% round share; deal order spans 1.195–1.909 and the boundary pairing is r=+0.358. | Defer the tree and its mask. Reopen only with a mechanism supplying MTP matches at unvisited positions. |
| 10-20 | [Longest tail](evidence/ascend-decode/merge-tree-20261020/result.json) | First sweep of the maximum suffix, k=31: acceptance 6.912/6.959/6.912/6.912/6.820/6.730/6.600 at longest 4/8/12/16/24/32/64. Longest-first prefers a stale long repeat over a fresher short one. | Keep longest 8; stop sweeping it. One trajectory, direction only. |

[Detailed historical experiments](probe-20261019-2340.md) retain the original
records and subsequent corrections. Raw artifacts remain on their producing
server; current local summaries bind their input and source hashes.
