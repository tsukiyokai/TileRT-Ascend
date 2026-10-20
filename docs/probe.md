# Experiment log

A3, GLM-5.2-W4A8C8, batch1,8K; target TPOT2.xms, release budget≤2ms.
The best historical request timing is11.6198ms; its old configuration diverges
from the frozen teacher at offset4. No new whole-model acceptance is claimed.
CPU replay and bounded single-die checks continue while whole hosts are occupied.

| Date | Evidence | Result | Next decision |
|---|---|---|---|
| 10-20 | [FIA row correction](evidence/ascend-decode/shared-fia-model-20261020/fia-row-correction/README.md) | Die15,8 frozen Q/KV/mask fixtures: inner_precise0→2 leaves524,288 BF16 values bit identical;16 saved controls reproduce.117MB peak allocated. | Close this flag-only route; no full-model rerun or timing claim. |
| 10-20 | [Exact query sharing](evidence/ascend-decode/shared-fia-model-20261020/exact-sharing/README.md) | CPU,21 actual M32 producers:2,016 exact selected-set reconstructions. With4 heads and assumed16×128 tiles, union+mask uses45.60% of full-cache rectangle work; exact cascade49.42%. Union has21 segments, cascade987. | Prefer union for structural investigation; count gather and mask costs. No NPU latency or measured HBM saving. |
| 10-20 | [Old chain first difference](evidence/ascend-decode/shared-fia-model-20261020/old-chain-first-difference/README.md) | First wrong output is round2/query0/position8195: recorded target prediction6500 versus teacher2524. All227 scalar commits and deliveries agree with recorded predictions; reads occurred after conditioned MTP. | Capture target output before/after conditioning. A frozen two-round layer-major replay can first isolate the eager target and rejected-tail overwrite. |
| 10-20 | [Norm/quant fusion](evidence/ascend-decode/shared-fia-model-20261020/norm-quant-fusion/README.md) | Actual ACLGraph/SuperKernel135 checks pass. TP16/78layers: changing only3 dense norm/quant sites changes4/4 labels to3/4, reproducing2524→6500 on all ranks.1,264 model artifacts and48 norm snapshots audited;942MB/die peak. OFF bridge384 field comparisons match. | Paused after completion; no experiment process remains. Next: frozen c29341d full compiled33-output request, then accepted-path TPOT. No new timing. |
| 10-20 | [Frozen two rounds](evidence/ascend-decode/shared-fia-model-20261020/two-round-replay/README.md) | `24049d1`:48 CPU contracts; TP16 all78 layers pass4/4 trusted labels.1,264 artifacts checked. Trusted hidden/residual/new KV and final FP32 logits match accepted M32 exactly; index order differs but sets agree. | First wrong position now2524. Isolate compiled numerical path; Graph/MTP/reset and request TPOT remain. |
| 10-20 | [Full M32 teacher](evidence/ascend-decode/shared-fia-model-20261020/layer-major/m32/full/README.md) | Same-source TP16/78layers: SFA32/32,shared-FIA30/32 on all16 ranks; wrong queries3/21. Independent CPU audit validates2,528 artifacts; final norm cross-arm NRMSE0.15993,899MiB/die. SFA query3 has2524/6500 margin0.002617. | Keep teacher gate. FIA row correction is null on8 frozen layers; next is frozen two-round SFA replay; no request TPOT or serial/Graph acceptance. |
| 10-20 | [Dispersed M32 GMM2](evidence/ascend-decode/gmm2-compact-20261020/guard-release/README.md) | `8e58cff`: retain all-small selection; extend to M32/A≥64/count≤32/hot≤A/8.18 actual-publisher CPU cases and58 Graph/CPU-MSD comparisons pass. Production ABBA saves11.18–22.12µs on three reconstructed teacher distributions; all80 output positions match across four arms. | Keep opt-in. Same-path control deltas reverse sign; no stable overhead estimate. Verify complete teacher and actual lookup round cost; no new TPOT. |
| 10-20 | [Late attention](evidence/ascend-decode/shared-fia-model-20261020/late-attention/README.md) | Single die, real M32 layers66–73:19 new-cache pairs bit-exact with frozen M1; index sets match. Shared/rebuilt FIA attention and pre-O are bit-exact. SFA/FIA pre-O NRMSE0.00224–0.00319;147MiB peak. | Local stages through V-up only; preserve the full teacher gate. |
| 10-20 | [Teacher router prior](evidence/ascend-decode/shared-fia-model-20261020/teacher-routes/README.md) | Current M1 kernels reconstruct75×32 routes from frozen residual. Six norm recovery cases bit-exact; actual key33. M8/16/32 mean union44.35/68.07/99.36. M32 global tile guard rejects35 layers although only49/3198 active experts there exceed16 rows. | Guard release above covers75/75 teacher windows. Capture actual lookup suffixes before assigning a full-round benefit. |
| 10-20 | [Selected MTP head](evidence/ascend-decode/reduced-head-20261020/selected-mtp/README.md) | Source permits conditioned body then selected-row head. Old M8 profile: entire conditioned head task sum median219.986µs across14 rank-round samples; a selected head and later submission remain. | Close implementation work at this priority; no new device experiment or latency saving. Keep focus on target verify. |
| 10-20 | [Trusted teacher copying](evidence/ascend-decode/lookup-contract-20261019/teacher33/README.md) | Exact8K+33 frozen outputs; actual indexed drafter, independent scan and existing replay agree. Width8/16/32 each28rounds for32 new outputs;224/448/896 planned verify rows. | Common30 new outputs: saved MTP4 candidates14rounds/56 target rows versus lookup8 26/208; break-even complete round-cost ratio1.857. Live M4-conditioned candidates and timing remain unmeasured. |
| 10-20 | [Layer-resident M32](evidence/ascend-decode/shared-fia-model-20261020/layer-major/m32/README.md) | Real TP16/SFA,8layers×32rows: all gates pass; hidden/residual/indices bit-exact across ranks;899MiB/die peak. Against old M1 rank0:68 boundary/cache pairs bit-exact; index sets match, order differs at layers1–5. CPU actual-loader ABBA136.475/15.968/16.866/135.243ms for36 tensors; bytes/order/ownership and handle cleanup match. | Shared-FIA8 and78-layer teacher remain. Corrected activity gate refused retry; loader gain is CPU loading only. |
| 10-20 | [M16 shared FIA](evidence/ascend-decode/fia-mask-20261020/joint/m16-screen/README.md) | Single die, four-layer Graph: paired-arm medians373.365→328.620µs, paired delta−44.190µs;29/30 ABBA wins.12 shared/rebuilt and24 Graph/reset checks bit-exact;234MiB peak reserved. Real-config CPU8 checks and mask3 tests pass. | Enable explicit M16 between8/32. Full teacher and whole-request width comparison remain. |
| 10-20 | [Copy-only tree](evidence/ascend-decode/lookup-contract-20261019/copy-tree/README.md) | Server CPU, fixed32 rows: flat/tree2/tree4/tree8 need147/135/132/140 rounds;12 scalar/reverse-scan checks pass. Tree4 A=7.75 permits11.36% more complete round cost; setup separately measured. | Audit indexer ancestor masking before implementation. One old trajectory; no NPU tree or TPOT result. |
| 10-20 | [Reduced greedy head](evidence/ascend-decode/reduced-head-20261020/README.md) | Native vLLM `get_top_tokens`:36 CPU cases and real TP2 eager/Graph/reset match. Graph M1/M8/M32 paired wall regressions14.18/20.74/29.85µs;30 pairs,360MiB/rank. | No production change. TP2 evidence closes direct reuse here; TP16 remains unmeasured. |
| 10-20 | [Copy width saturation](evidence/ascend-decode/lookup-contract-20261019/wide-frontier/README.md) | Actual CPU replay, M32/64/128/256:147/144/144/144 rounds;4704/9216/18432/36864 rows. M64+ permits only2.083% more mean round cost than M32. | Keep device comparison at8/16/32; one old unaccepted trajectory supplies no broad-width gain. |
| 10-20 | [GMM1 K512](evidence/ascend-decode/gmm-compact-20261020/k512/README.md) |24 changing-route Graph cases bit-exact;10 ABBA pairs. M32 random saves4.67µs against N128/K256 but costs0.51µs against original N256/K256; shared8 saves3.76µs. | Close GMM1 tiling branch. No production switch; actual lookup expert unions decide whether a specialized policy is useful. |
| 10-20 | [GMM1 N split](evidence/ascend-decode/gmm-compact-20261020/n-split/README.md) |24 changing-route Graph cases bit-exact;10 ABBA pairs. N128 saves3.22/3.14/4.06µs for M1/M8/shared8 but costs6.49µs for M32 random; N64 loses all pairs. | No production policy. K512 follow-up above retains the original N256 anchor. |
| 10-20 | [Route collection](evidence/ascend-decode/shared-fia-model-20261020/README.md) | Teacher collector/runner34/169 CPU checks pass; actual-session wrapper9 CPU fixtures pass. M24/M32 lookup records candidates, positions and routes; diagnostic headline TPOT is null. | Run frozen-input parity and actual rejected-suffix route capture separately. Whole-model routes remain unmeasured. |
| 10-20 | [GMM1 full-A](evidence/ascend-decode/gmm-compact-20261020/full-a/README.md) |24 three-arm changing-route Graph cases bit-exact;10 ABBA screening pairs. Compact128→full-A64 regresses M1/M8/M32-random/shared8 by paired0.54/4.22/1.35/1.51µs. | Reject; no production switch. Six A transfer batches become one with identical total bytes. |
| 10-20 | [Paired C8 cache](evidence/ascend-decode/index-cache-pair-20261020/README.md) | Reuse CANN Norm scatter with FP16 scale byte views. Actual dynamic-quant→write_cache path:18 full-backing eager/Graph/reset checks pass; M1/M8/M32 25.794→8.586/28.966→10.342/28.460→16.716µs,10/10 pairs each. |13-line isolated vLLM-Ascend patch; whole-model teacher/compiled Graph/TPOT pending. FP8/C4/BF16 unchanged. |
| 10-20 | [Adaptive GMM2 tiles](evidence/ascend-decode/gmm2-compact-20261020/adaptive/README.md) | CPU299 schedules/12 metadata transitions; release46 Graph cases and4 checkpoint calls bit-exact. Full-package ABBA: M32 random143.90–146.98→106.18–107.86µs; M8 saves16.77–20.42µs. M1 costs1.18–1.82µs; shared8 costs5.79–7.01µs. | Opt-in `--adaptive-tiles`; keep disabled until actual lookup expert unions and model teacher checks select it. Fallback tile shape alone does not preserve latency. |
| 10-20 | [Compact GMM2](evidence/ascend-decode/gmm2-compact-20261020/README.md) | CPU102 schedules;24 changing-route Graph cases and4 real-checkpoint calls bit-exact. Shared die15,30 ABBA pairs: M1/M8 62.42→34.86/91.27→72.61µs; M32 random149.68→149.88µs, shared8 64.59→41.40µs. | Opt-in B114 overlay; retain NZ/MSD. No M32 random benefit or whole-model TPOT claim. Measure actual lookup expert unions. |
| 10-20 | [Fixpipe reference](evidence/ascend-decode/gmm2-compact-20261020/fixpipe-official-receipt.json) | Official scale contract clears low13 FP32 bits; correction removes3548/49152 synthetic mismatches, then all24 GMM2 cases match. `2624589` adds a four-output CPU check. | Eight sampled real experts already have zero low13 bits; this does not explain the old teacher divergence. |
| 10-20 | [Compact GMM1](evidence/ascend-decode/gmm-compact-20261020/README.md) | CPU63 invariants;24 changing-group Graph cases bit-exact across production ELFs;4 real-checkpoint calls match independent CPU arithmetic. Shared die15, same-ELF30 ABBA pairs: M1/M8/M32-random90.90→49.36/111.56→83.91/163.28→141.16µs; full256-expert M32 also30/30 wins. | Opt-in `--compact-experts`; default rebuild matches old ELF SHA. Whole-model teacher, TP16 critical path and TPOT pending. |
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
| 10-20 | [Frequency continuation](evidence/ascend-decode/shared-fia-model-20261020/teacher/frequency-summary.json) | Server CPU, same old trajectory: M32 rounds147→146; total lookup CPU22.82→45.70ms/request. | Defer the extra policy; one-round reduction is not a device TPOT result. |
| 10-20 | [Model validation](evidence/ascend-decode/shared-fia-model-20261020/README.md) | a3-187 run interrupted during loading; no model result. Old chain-w8 requests3/4/5 differ from teacher at offset4. a3-194 CPU environment checks pass; all16 dies occupied by Ray at01:58. | `d62e1b2` prepares fixed-input M1/M32 before serial/spec/Graph; CPU169 checks pass. First-difference evidence and teacher gate remain separate from continuation consistency. |
| 10-20 | [Final verify capacity](evidence/ascend-decode/shared-fia-model-20261020/cpu/checks.json) | CPU168 checks pass; width8/32 need room for the full last verification.32 field/range checks cover the expanded poisoned tail. | `55ed217` rejects silent width reduction; current8K/B33/M32/C9344 remains valid. |
| 10-20 | [Static package provenance](evidence/ascend-decode/shared-fia-model-20261020/cache/summary.json) | Saved TP16 M8 repeats: stale package6/6 distinct hidden; fresh patched1/6, its cache1/20. Same cache key held different GMM ELFs.4 real-launcher CPU cases isolate cwd/cache while retaining identical source. | `587cabf` compiles each arm under its own output directory. Bind the actual loaded static ELF in the next model run. |
| 10-20 | [Merged tree](evidence/ascend-decode/merge-tree-20261020/README.md) | Same trajectory, chain lengths dealt in four orders: merged k=31 acceptance 8.317/10.129/11.494/13.286. Break-even over lookup alone is 1.303 from the chain's measured 23.3% round share; deal order spans 1.195–1.909 and the boundary pairing is r=+0.358. | Defer the tree and its mask. Reopen only with a mechanism supplying MTP matches at unvisited positions. |
| 10-20 | [Longest tail](evidence/ascend-decode/merge-tree-20261020/result.json) | First sweep of the maximum suffix, k=31: acceptance 6.912/6.959/6.912/6.912/6.820/6.730/6.600 at longest 4/8/12/16/24/32/64. Longest-first prefers a stale long repeat over a fresher short one. | Keep longest 8; stop sweeping it. One trajectory, direction only. |
| 10-20 | [Critical path](evidence/ascend-decode/critical-path-20261020/README.md) | Two complete rounds of the width-8 capture, by time a stream runs alone: every HCCL collective0.000ms, TP reduction5.35ms, backbone compute35.61ms, draft chain3.54ms, no kernel anywhere11.48ms. The FP32 cast bracket is1.99ms over146 sites,26% of the bracketed region; AI_VECTOR_CORE spends8.71ms moving0.35ms of bytes. Matched inside the capture at `1,6144;1,6144`: plain `Add` 1.820µs against `ForeachAddList` 21.282µs, flat in rows, so all structure. | Reopen route3 on the vector and reducer boundaries, not the publication one. New opt-in `--plain-moe-combine` is bit-exact and worth about1.4ms/round. One-sided headroom is2.28ms/round against the measured37.9µs a site, not the6.47 its 63.5µs baseline implies. Profiled round62.4 against unprofiled53.05: shares only, and the11.48ms idle is mostly that inflation. |
| 10-20 | [Combine floor](evidence/ascend-decode/critical-path-20261020/combine-floor-die1.json) | One shared die under full occupancy, no model, both spellings sampled in every batch: plain add9.176／9.520／9.681µs at1／8／32 rows against the multi-tensor add's38.704／38.490／38.318. Flat in rows on both sides; `torch.equal` holds on the result. A second, busier run on the same die: `(x*s)+y` 27.505µs against one `add(y,x,alpha=s)` 14.453, where a plain add is11.779. | Two steps. Bit-exact first: plain add keeping `Muls`,1.46ms/round from the capture's device-side19.46µs. Then one fused axpy,1.98ms/round total, but `alpha` keeps the product in the accumulator so it needs the frozen teacher. Bench absolutes carry host issue; only pairs inside one run count. Request TPOT still needs a free machine. |

| 10-20 | [Launch floor](evidence/ascend-decode/critical-path-20261020/README.md) | Measured per-call floors times one round's real call counts:3008 calls,11.512 and11.503ms across two rounds,82% of it505 MIX_AIC calls at6.5 a layer. Per family against declared bytes: MlaPrologV3 1.1x, GroupedMatmul 4.6x, GroupedMatmulSwigluQuantV2 3.0x, SFA 5.0x, Cast 30.3x, ForeachAddList 91.1x. | 2ms at A=6.959 allows a13.92ms round, so the floor is83% of the budget before arithmetic: fewer launches, not faster kernels. MlaPrologV3 is the in-capture control for what fusion reaches. Price grouped matmuls on declared bytes scaled by the activated union; `DT_INT4` must be in the byte table. |

| 10-20 | [Fused epilogue](evidence/ascend-decode/epilogue-fused-20261020/README.md) | Hand-written AscendC, one shared die, both arms in one process, bit-exact at8 and32 rows (0 of49152 and0 of196608 differ). Device, 64 passes a launch: fused0.517 against three hand-written kernels'1.222µs at8 rows;0.875 against1.665 at32. Issue-bound5.68 against 16.70. The same arithmetic is36.90µs as three vendor operators. | The hand-writing is worth more than the fusion:2.7ms/round against0.05. `dav-c220` `Muls` rejects BF16, so compute in FP32 and `Cast` the two roundings back. Device figures are L2-resident, not bandwidth. |

| 10-20 | [MIX brick](evidence/ascend-decode/mixbrick-20261020/README.md) | Hand-written MIX kernel: cube matmul, cross-core flag, vector dequant and SwiGLU in one launch. 8 rows at four blocks12.307µs against the two-kernel arm's15.093 and against the vendor pair's14.801(`QuantBatchMatmulV3` 11.661＋`DequantSwigluQuant` 3.140). 32 rows 19.731 against22.420. Exact against the two-kernel arm and within1 ulp of a host reference,24 consecutive clean runs over four block counts. | Root cause of the long race: the two halves number the grid differently and the count does not convert — cube `GetBlockIdx` 0–3 with `GetBlockNum` 4, vector 0–7 with the same 4, so a vector core takes its range from `GetBlockIdx()/2`. Mode 2 lets a cube flag only its own two subblocks, so a core sent elsewhere waits on the wrong producer and no barrier helps. Producer needs `DataSyncBarrier<MemDsbT::DDR>`; `PipeBarrier` orders issue, not retirement; the DCache is irrelevant on this path. |


| 10-20 | [Shared expert in one kernel](evidence/ascend-decode/mixbrick-20261020/README.md) | Cube, vector, cube in one hand-written MIX kernel with `SyncAll<false>()` between the two cube stages: gate and up projection, dequant, SwiGLU, static-scale quantisation, down projection, dequant to BF16. 8 rows at four blocks13.863µs against the vendor's four kernels'33.952(`DynamicQuant` 3.84＋`QuantBatchMatmulV3` 11.661＋`DequantSwigluQuant` 3.140＋`QuantBatchMatmulV3` 15.311). 0 of49152 against a host reference,6/6 clean. | 2.45x as one kernel instead of four,1.51ms/round over75 MoE layers. The grid barrier must be at top level outside both halves' branches. Intermediate uses a static scale; per-token dynamic quant needs the row max over128 columns, which the channel split spreads across blocks, so it needs the two-pass structure the barrier is already there for. |

| 10-20 | [Dynamic quant in the chain](evidence/ascend-decode/mixbrick-20261020/README.md) | Per-token dynamic quantisation inside the one-kernel shared expert, correct at1／2／4 blocks (0 of49152;0 of196608 at32 rows). 8 rows at four blocks25.724µs against the vendor's four kernels'33.952,1.32x. Attributed by stage bits on one binary: projection plus activate 14.568, two `SyncAll<false>()` barriers1.710, reduction plus quantise plus down6.791, epilogue2.632. | The barriers were never the cost. The epilogue was, by shape: three 512-wide passes a row to move0.26µs of bytes, 5.393→2.632 with one wide tile. Batching the quantisation rows was worth only1µs, so the earlier11µs attribution was wrong — never price a stage by differencing two code versions. A `DataCopy` moves whole32-byte blocks, so the INT8 slice caps the split at four blocks; eight fails for that reason. |



| 10-20 | [Routed split](evidence/ascend-decode/routed-split-20261020/README.md) | Cube gate/up only: 10.742／10.709／10.717／10.852／11.416µs at2／4／8／16／32 rows — flat. By blocks at8 rows:20.610／13.120／10.795µs at1／2／4, i.e.76／120／145 GB/s on a fixed 1.57MB of weights. | Per-expert matmuls are closed: ~10µs each × 44 experts = 440µs a layer against the vendor's165. The cube is short of cores, not efficiency — our earlier "31x off the bytes" divided per-core traffic by the die's aggregate rate; four cores at145 GB/s are near what four can have. Split the expert axis instead: a permuted row belongs to one expert and so to one core, so the dynamic-quant row max is core-local and the routed chain needs no grid barrier at all. Byte floor is45µs a layer. |

| 10-20 | [Free machine](evidence/ascend-decode/free-machine-20261020/README.md) | First whole-machine arms since the fleet freed, width32 with lookup, three live requests each: base11.0052／10.9691／10.9937 and10.9848／11.0309／10.9811, combine10.9420／10.9553／10.9285 and11.0452／10.9897／10.9737. | New best request TPOT10.9285ms against the recorded 11.6198, and it belongs to the current tree, not to today's flag. The `--plain-moe-combine` pairing is void: all four arms shared `tree/static_kernel_compile_outputs/static_kernel_cache`, below the tree and outside the per-arm caches, so the combine arms restored the package `base-a` compiled. Redo with that cache cleared or `disable_static_kernel_compile_cache` set. |

| 10-20 | [Combine, re-run](evidence/ascend-decode/free-machine-20261020/README.md) | Same tree and flags, launcher current so each arm keeps its own static-kernel cache. Steady-state means: base-a11.0441／combine-a10.9402, base-b11.1460／combine-b10.8769. | `--plain-moe-combine` is real: −0.104 and −0.269ms, mean −0.19 against the predicted −0.21, and inside each pair every combine request beats every base request, ten of ten. Lowest single request10.8528ms. The 0.102ms between base-a and base-b is the between-arm noise, so quote 0.1–0.3ms. |

| 10-20 | [Handshake modes](evidence/ascend-decode/handshake-20261020/README.md) | Four reverse modes × three set-and-wait patterns × one and four rounds, each variant its own process under a timeout. | Reverse modes0 and1 deadlock in every pattern; mode2 is live only when both subblocks set and the cube waits once; mode3 is live in all three. Markers are unreliable (scalar store from a vector core); liveness is the measurement. |
| 10-20 | [Routed experts, one kernel](evidence/ascend-decode/routed-kernel-20261020/README.md) | Expert-axis split, no grid barrier, hand-written AscendC on one die of a3-187. Sixteen experts on sixteen blocks is the per-rank shape at TP16. | 49.3–49.8µs a layer against the vendor's165.2 =3.3x, within12% of the45µs byte floor; 3.71ms a round against12.39, saving 8.68 ⇒1.24ms TPOT at acceptance7. Correct at every configuration up to44 experts／22 blocks: 0 of540672. Not integrated; rows an expert fixed at2. |

| 10-20 | [Capacity sweep](evidence/ascend-decode/routed-kernel-20261020/README.md) | Rows an expert as a build-time capacity, 2／4／8／16／32 at16 experts on16 blocks. | Padding is not free: 49.289／51.969／55.263／63.308／79.820µs, a fit of 47.25µs+1.02µs a row. The cube is flat in rows but the epilogue is linear in them. The fixed term is weight streaming -- 2.36MB a block,16 at once,799GB/s =69% of the die's peak -- so that is the floor to attack, and capacity must follow the group list rather than the worst case. |

| 10-20 | [INT4 weights](evidence/ascend-decode/int4-weights-20261020/README.md) | Seven `MatmulType` variants with an `int4b_t` operand, then six basic-API copy probes. | The capture shows the vendor reading `DT_INT4` in `FRACTAL_NZ`, so A3 does 4-bit weights; vendor165.2µs on18.9MB is114GB/s (10% of peak) against this kernel's49.3µs on37.7MB =765 (66%). The high-level `Matmul` rejects int4 in every combination because it plans to widen to INT8 during the GM→L1 copy and MTE2 will not; but int4 reaches UB, A1, B1 and A2 unaided, and L1→L0B int4 is in the library already. Hand-rolled L1／L0 would be ~24.8µs a layer. |

| 10-20 | [INT4 mad, by hand](evidence/ascend-decode/int4-weights-20261020/README.md) | `DataCopy`→L1, `LoadData`→L0, `Mmad`, `Fixpipe`→GM at M,K,N =16,64,16 with fillings that make the answer layout-independent. | Works: `0x11` gives64 =K everywhere, `0x21` gives96 =32×1+32×2, so both nibbles of a byte are separate INT4 elements. `KERNEL_TYPE_AIC_ONLY` is rejected at submission (`sqe_type=3`, `errType=0x20`); the `__mix__(1,2)` shape works with the vector half returning. Every link of the INT4 path is now proven. |

| 10-20 | [Attention sparsity](evidence/ascend-decode/attention-sparsity-20261020/README.md) | Read `SparseFlashAttention`'s real arguments out of the capture. | It is handed the whole paged cache (`73,128,1,512` =9344 positions, the `--capacity` default) alongside a `1,1,2048` selection. At85.048µs a call,2048 positions would be 55GB/s and9344 would be253 -- so it reads everything and masks after, and the1.382ms recorded as its byte floor was the floor of reading everything. A read that honours the selection is4.6x less traffic. Settle it by timing against `--capacity`. Also unaccounted: `hcom_allGather_` at63554µs over140 calls, largest single13240. |

| 10-20 | [L2 hint on the weight](evidence/ascend-decode/routed-kernel-20261020/README.md) | `SetL2CacheHint(CACHE_MODE_DISABLE)` on both projections' weights, the vendor's own `numBlocksM == 1` practice, at16 experts／16 blocks ×3 processes. | Nothing: 49.238／49.517／49.611µs against49.276／49.725／49.791 unhinted, correctness unchanged. The line is removed. The kernel is at66% of bandwidth and L2 eviction by its own weight stream does not explain the missing third. |

| 10-20 | [The routed kernel is scalar bound](evidence/ascend-decode/routed-kernel-20261020/README.md) | `msprof --aic-metrics=PipeUtilization`,16 experts／16 blocks. | scalar 27.860µs／0.625, MTE2 32.826／0.733, MAC 4.845／**0.109**, vector 0.622／0.013. The cube spends62.5% of its time in scalar and11% multiplying; the vendor's MTE2-bound cases read0.93–0.99 where this reads0.733. "66% of bandwidth" was the wrong account — it explains the null L2 hint, INT4's21% instead of ~45%, and the flat block-count curve. Next lever: hoist `Matmul` out of the expert loop, or drive L1／L0 by hand. |

| 10-20 | [Fixed cube tiles](evidence/ascend-decode/routed-kernel-20261020/README.md) | `SetFixSplit` swept against the library's own choice, from the scalar diagnosis: fewer, larger tiles mean fewer per-iteration address computations. | 128×512 (which fills L0B exactly) gives41.677µs against49.692 auto,42.095 at128×256, 49.874 at256×128,56.410 at64×512 — all0 of196608. Full table38.761／40.000／41.990／74.555 at4／8／16／44 experts, so **3.93x the vendor** at the per-rank shape (was3.3x) =1.32ms TPOT. Hoisting `Matmul` out of the loop gave nothing, so the cost is inside the iteration. |

[Detailed historical experiments](probe-20261019-2340.md) retain the original
records and subsequent corrections. Raw artifacts remain on their producing
server; current local summaries bind their input and source hashes.

## 2026-10-21 头号未决：每行 0.814 ms 测的是什么

代价模型 `round = 45.88 + 0.814/行` 拟合可靠（四宽度、跨 19.3 ms、残差几百
微秒），而六个候选全排除：

- 图分桶 — 引擎只捕获一个尺寸 `cudagraph_capture_sizes: [graph_width]`
- TP 规约 — 两卡下字节涨 8 倍而斜率为负（111.27→93.24 µs）
- 算子行成本 — 一整层 8→32 token 只多 8.07 µs，即 0.026 ms/行/轮
- 轨迹长度 — 臂按 `--max-tokens 1024` 封顶，各宽度同样的 token 与上下文
- Gloo 命令广播 — 宽度 0 时 305.0 µs、64 时 289.6；且命令只带标量
- MTP 自回归草稿 — `--automaton` 走 `_advance_drafted`，一轮只有一次 verify

**它决定 2.x 是否可达**：若那笔钱是实现代价，固定预算树（64 节点、接受
14.510）给 TPOT 1.88 ms；若是物理代价，最好是自适应树的 3.90。二十二臂里
宽度判别五臂排最前。
[证据](evidence/ascend-decode/row-cost-mismatch-20261021/README.md)

## 2026-10-21 定案：一轮的代价在算子边界，不在算术也不在下发

- 每 kernel 下发地板：eager 11.4 µs、**图回放 1.064 µs**。引擎跑图，所以下发只
  占一层 710 µs 的 4%，剩下是设备时间。
- MoE 矩阵乘与行数几乎无关（16 行 146 µs、256 行 151 µs），所以那 12.3 ms 不是
  算术。
- 真正可省的是**算子边界的 HBM 往返**：vendor 四次下发 181.172 µs 对单 kernel
  72.463 µs，省 **8.48 ms/轮**。
- 跨层融合封顶 23%（嵌套 8 层 6.15x，交换因 scratch 复用而 N²，构造无效）。
- 自定义算子包：CANN 自带 grouped matmul 66.813 µs 最快，owner overlay 97.362、
  vllm-ascend 包 123.985、两者 104.929。但该包在 norm 与 gating 上快 22%，所以
  整体 unset 净 1.4 ms，精确回落需正确的取消注册（删文件无效）。
- 起草侧：自适应提案宽度（上轮接受数的两倍）省 **9.7% TPOT**；树在编辑率 0.02
  下接受 14.510，2.x 条件可达。

## 2026-10-21 三条杠杆复核，两条翻转（口径已被上节修正）

- MoE 单 kernel：cube 半边从 UB 读 group count（该核无 UB 访问权）而跳过全部
  专家，79.206 µs 测的是空转。修好后 365.887 µs 对 vendor 157.176，**慢
  2.33x**，杠杆由 +5.8 翻成 −16.3 ms。
- 路由置换吸收：decode 真实形状（12 token × top-8 = 96 行）下 vendor 这对只要
  53.226 µs，不是记录的 101.751，上限由 7.1 降到 **≤4.15 ms**；最朴素的 torch
  替换慢 2.93x（`index_add_` 一项 131.957 µs）。
- 单边规约：未复核，正确性依据是十六卡精确值比对（判据能失败），仍成立 6.5 ms。

## 2026-10-21 MoE 矩阵乘是固定税

16 行 146.431 µs、256 行 150.565 µs——16 倍工作量多 2.8%。所以那 12.3 ms 不到
3% 是算术，与宿主下发、路由同属每层固定税。更快的矩阵乘只能买 3%，所以手写
kernel 的 2.33x 劣势无关紧要，它的价值只在**一次下发覆盖多层**。两次独立调用是
单次的 1.978 倍，线性，正是融合要收的部分。

## 2026-10-21 vendor 算子的成本依赖进程状态

同一算子同一形状：干净进程 61.617 µs，某种混合态 153.325 µs，两个时钟一致到
1%。加载 kernel 库、跑过 kernel、建过通信域**全部排除**。引擎记的 157.2 µs 属
于慢档，而干净进程能跑 61.6——找出引擎落入慢档的条件约值 7 ms，不需写 kernel。

## 2026-10-21 the A3 inventory, first time checked

Six A3 hosts in `~/.ssh/config`: `a3-gs`, `a3-main`, `a3-3`, `a3-4`, `a3-78`,
`a3-84`. Only two answer, and only one is usable:

    a3-gs     7.150.10.55   connection timed out
    a3-main   7.150.12.90   connection timed out
    a3-3      7.150.13.67   connection timed out
    a3-4      7.150.8.233   connection timed out
    a3-78     7.150.10.78   reachable, no trt-asc container, 58.8/65.5 GB held
    a3-84     7.150.14.84   reachable, trt-asc present, tenant holds 54.6 GB

So the whole session ran on the only usable host, and its being held is a
one-machine constraint rather than an unsearched alternative. Worth checking
first on any day the machine looks busy -- this was the first time this project
looked.

## 2026-10-21 the truncated backbone, reopened for measurement and still blocked

The truncated-backbone route was closed earlier on correctness grounds: sixteen
ranks each randomly initialising disagree about the first token. For measuring
a round's step timings and the rope graph's effect that does not matter -- the
tokens need not be right.

So it was reopened against the tenant-held machine, 6.7 GiB free of 64 a die.
Six attempts: 6 layers, 2 layers, `--gpu-memory-utilization 0.09`, `--capacity
1024`. Every one fails during load, at a different local rank each time -- 2,
then 8 -- which is what memory exhaustion looks like when ranks race.

The reason the layer count does not help: vLLM reserves its KV pool from
`gpu_memory_utilization` times the device, independent of how many layers the
model has. Shrinking the pool to fit 6.7 GiB then leaves too little for the
capacity, and the run fails earlier still -- its log is 374 lines against 677.

So the route is sound in principle and unreachable at 10 percent of a device.
It needs either the machine free or a configuration search that costs five to
seven minutes an iteration, and the four levers waiting on it are each already
validated on one die.

## 2026-10-21 the levers are queued behind a watcher

`/workspace/experiments/levers-20261021/watch_and_run.sh` runs detached in the
container: it polls `precheck.sh` every five minutes for sixteen hours and
starts four arms once a die has more than 48 GiB free. The parse was verified
against the current state -- `free 6.7` reads as 6.7 and compares false -- so a
sixteen-hour wait cannot end in a silent no-op.

    baseline      the current path, with AIV on by default
    rope          --rope-graph
    rope-narrow   --rope-graph --narrow-bank
    aicpu         --ai-cpu-collectives, the negative control for AIV

All four carry `--round-phases`, so the first engine-side account of a round's
steps arrives with them. Each arm skips itself if its `rank-0.json` already
exists, so a rerun resumes rather than repeats.

What the arms settle, in order of size: whether the rope graph's 22.54x on one
die becomes 7.25 ms a round in the engine; whether AIV's 1.5 to 1.85x on
standalone collectives shows up against its own negative control; whether the
bank narrowing's 4.63 ms survives the engine's own routing; and where a round's
50.710 ms actually goes.

## 2026-10-21 two implemented switches were never enabled

The best arm's own record carries `hoist_metadata: false` and
`hoist_forward_context: false`. Both are implemented, both are flags on the
bench, and neither has been used in any arm that produced a headline figure.

What `_metadata` rebuilds every round, measured on one die as the host half
only -- a length, a computed-tokens tensor, a positions range, the slot
mapping, and three H2D copies:

    130.62 us a round, 24.56 ms over a request's 188 rounds

That excludes the vLLM objects the engine also builds every round,
`AscendCommonAttentionMetadata` and `AscendAttentionState`, so it is a floor on
what hoisting saves. Against a 50.710 ms round it is 0.26 percent, which is
small on its own and free to take: the code exists and the flags are wired.

Two arms added to the watcher -- `hoist` alone and `all` with everything -- so
the composition is measured rather than assumed.
