# Lessons

- Captured workspaces outlive a forward. Three MTP graphs exhausted the
  2 GiB probe quota; each bucket independently passes with serial graph
  lifetime. Track live graph pools separately from model-weight memory.
  [MTP forward graphs](evidence/ascend-decode/layer-major-20261018/README.md#complete-mtp-forward-graph)
- A chained graph can consume its own previous output. Stage the live
  recycle parent before replay reuses that output; retain the independent
  seed indices across both chain calls. Real alias calls and dirty rollback
  verify this lifetime contract beyond frozen-input replay.
  [Product chain check](evidence/ascend-decode/layer-major-20261018/mtp-chain-graph-audit.json)
- Removing a host round trip need not shorten the measured chain. Merging
  two correct draft replays into one graph saves a paired median 0.089 ms
  and wins 15/30 under shared load, while capture context roughly doubles.
  Retain the prototype as evidence; pursue the observed device execution
  tails before adding it to the default session.
  [Two-chain timing](evidence/ascend-decode/layer-major-20261018/mtp-two-chain-timing-audit.json)
- Matching raw PMU counters to a task does not establish that they count
  only its execution. A 10.52 µs rotation carries 4.81 ms of converted
  AICORE time, while a 115.34 µs instance carries 4.78 µs. Preserve the
  raw occurrence and counter semantics before assigning execution tails
  to arithmetic or preemption.
  [PMU audit](evidence/ascend-decode/layer-major-20261018/mtp-chain-pmu-summary.json)
- ELF provenance must separate verification reads from runtime loading.
  The complete-MoE driver hashes four control/candidate files before one
  runtime control open; treating all successful opens as selection rejects
  a correct baseline. Preserve the ordered verifier prefix and audit the
  subsequent loader access.
  [Complete-MoE control](evidence/ascend-decode/gmm1-tiling-20261018/moe-baseline-audit.json)
- Static selection must preserve numerical fixes. The MoE private package
  keeps GMM1 out of regeneration, and actual ELF-open traces on every rank
  confirm the installed SwiGLU-barrier implementation is still selected.
  [MoE activation](evidence/ascend-decode/execution-gap-20261018/static-super-moe-audit.json)
- A completed prefix capture proves transport, not decode. A live PD check
  binds one request ID across sixteen WRITE acknowledgements, prefill's
  first token, real decode steps, callback tokens and the HTTP response.
  [Live handoff](evidence/ascend-decode/pd-live-20261017/README.md)
- Budget peak HBM during prefill warmup. Two resident weight sets failed
  despite a 1 GiB P cache. For single-machine validation, keep the HIXL
  receive buffer alive, stop P after transfer, then load D inside the
  pending request. Model loading belongs to TTFT.
- On this HIXL version, use shifted physical D placement, separate internal
  listener ports and non-expandable allocator-block registration.
  [Device evidence](evidence/ascend-decode/pd-hixl-20261017/README.md)
- ACL graph capture does not execute the recorded forward. Replay before
  consuming its output; refresh persistent lengths, RoPE, INT64 slots and
  shared indices. Poison unwritten KV so warmup data cannot mask a missed
  write. [Graph evidence](evidence/ascend-decode/cann-graph-20261017/README.md)
- Keep independent HF diagnostics: shared CANN kernels hide common errors.
  Fix the reference before changing numerical criteria. Layer curves locate
  differences; frozen teacher trajectories define release acceptance.
  [Precision policy](plan.md#4-可独立验收)
- Measure accepted tokens, not verification width. The measured M4 session
  commits 31 tokens in 17 rounds for 32 outputs, but 1023 in 303 rounds
  for 1024 outputs. With unchanged code, median TPOT changes from 38.70
  to 20.44 ms; round means remain near 69 ms. Use the long workload to
  budget the 2 ms target, and retain short captures for diagnosis.
- Bind byte estimates to the executing model. The current CANN router has
  BF16 weights and FP32 output; it does not read FP32 router weights.
  `full_logits=False` still gathers the full vocabulary before discarding
  logits. Expert-union scenarios estimate logical reads, not measured HBM
  traffic. [Representation audit](evidence/ascend-decode/cann-traffic-source-audit-20261017.json)
- Reducing collective bytes can add eager device work. The small-packet
  head matches all checked IDs but measures 22.12 ms TPOT versus the
  immediate full-vocabulary control's 20.67 ms; retain the faster path.
  A mapped SHMEM peer pointer also needs actual device-access validation:
  two PEs pass while TP16 fails before timing.
  [Execution boundaries](evidence/ascend-decode/execution-gap-20261018/README.md)
- MTP acceptance does not divide every byte by the verification width.
  Observed M4 target expert unions average 27.121 versus eight for M1.
  The logical-read saving is about 1.72–1.89×, not the 3.376× acceptance.
  Full-replication reduction trades more traffic for fewer stages; count
  its network work and remaining unoverlapped budget before choosing it.
  [Observed unions and budget](evidence/ascend-decode/execution-gap-20261018/README.md#observed-expert-unions)
- Separate functional setup from latency measurement. The TP16 HCCL
  resource query passes under external load after registering graph scratch
  under the same tag; it needs little device memory. Reserve idle devices
  for model timing. Host context, device peer reads and dependent-chain
  latency are separate gates.
  [Registered query](evidence/ascend-decode/execution-gap-20261018/hccl-query-audit.json)
- Compare matched execution paths before rewriting a shared kernel.
  Graph4 can capture an eager model or a compiled model; short and long
  MTP requests also have different acceptance distributions. A large
  operator cost does not establish its incremental cost versus vLLM.
  AIV lowers the isolated FP32 96 KiB graph chain from 10.93 to 5.04 ms,
  while the decode-session median changes from 20.44 to 21.47 ms.
  Communication-only gains and end-to-end gains need separate evidence;
  resource contention or arrival skew remain hypotheses until localized.
  [Matched modes](evidence/ascend-decode/tp-mode-20261017/README.md)
- Launch blocks do not measure active workers. The current SFA M4 source
  assigns four queries to four AIC workers despite a 24-block launch.
  Splitting heads repeats KV traffic; splitting KV adds partial rounding
  and merging. Compare complete existing-CANN pipelines before rewriting.
  [Attention replay](evidence/ascend-decode/sfa-reuse-20261018/README.md)
- Specialize known cache ownership before replacing an attention kernel.
  Identity pages allow two direct INT32 gathers into existing FIA; real
  M4 hot graphs fall from 89.05 to 69.15 µs, including the gathers. Keep
  cross-kernel numerical differences diagnostic and verify frozen teachers
  before selecting a model default. An optional tensor can also be
  uninitialized: this CANN GatherPA needs explicit zero `seq_offset`.
  [Contiguous-cache evidence](evidence/ascend-decode/sfa-reuse-20261018/README.md#contiguous-cache-and-real-m4)
- Peer access and a constant-input sum do not establish a reusable
  collective. Include payload publication, ready/ACK and overwrite
  protection in the timed dependency chain. Registered executor scratch
  can also be smaller than the caller's allocation; grow the actual
  resource request under a fresh tag before enlarging mailboxes.
  [Peer reduction evidence](evidence/ascend-decode/execution-gap-20261018/peer-reduction-audit.json)
- Frozen teacher inputs permit layer order followed by token order without
  batching tokens together. Keep both BF16 hidden/residual components and
  restore that token's shared top-k before each dependent layer. One
  resident layer then validates actual M1 numerics under a small memory
  quota while other jobs retain the machine.
  [Reorder and device control](evidence/ascend-decode/layer-major-20261018/README.md)
- Reduced-residency probes still need the actual model constructor.
  GLM's constructor registers fused-QKV quantization mappings inherited by
  MTP. An embedding-only stub omits them and fails at weight loading.
  [MTP own-head check](evidence/ascend-decode/layer-major-20261018/README.md#mtp-own-head)
- Convert reference weights to the executing dtype before exact arithmetic.
  This checkpoint stores FP32 LM-head rows, but deployment loads BF16.
  BF16 output also merges distinct winning scores: 16.69348 and 16.75541
  both become 16.75. Diagnose output resolution before adjusting global
  layer-error criteria or introducing token-specific tie rules. Existing
  CANN BF16-input/FP32-output matmul recovers 32/32 labels on the frozen
  full-layer outputs without expanding the weights.
  [Head evidence](evidence/ascend-decode/layer-major-20261018/README.md)
- Compare complete collective contracts. TP16 one-shot push includes
  publication and overwrite protection; its single-kernel p50 is 38.06 µs
  versus 59.18 µs for the consecutive BF16/FP32/HCCL/BF16 graph control.
  Keep shared-device screening, idle calibration and decoder timing
  distinct. Double payload/READY slots remove ACK with an explicit causal
  retirement proof and pass at 35.72 µs. This still exceeds the 8 µs budget;
  stop standalone flag tuning. A producer-fusion comparison must retain
  the installed unfused operator as its baseline.
  [Complete protocol](evidence/ascend-decode/execution-gap-20261018/README.md#complete-one-shot-push)
- More send workers do not establish lower collective latency. Three
  disjoint peer groups preserve the full protocol but win only 14/30
  pairs, with a −0.196 µs median gain. Stop the branch. Bind fusion to the
  executed producer's actual tile: O key24 has one N wave of 24 complete
  `[4,256]` tiles, so additional waves cannot supply overlap here.
  [Producer and paired screen](evidence/ascend-decode/execution-gap-20261018/README.md#actual-o-producer)
- A plausible smaller tile still needs a real-shape comparison. Existing
  GMM1 baseN=128 preserves outputs and actually activates for M4, but its
  shared-die p50 is 124.59 µs versus 121.52 µs for baseN=256. Retain the
  existing tiling; a source-level occupancy argument is not a speedup.
  [GMM1 screen](evidence/ascend-decode/gmm1-tiling-20261018/README.md)
- Preserve pairs when summarizing an AB/BA comparison. Independent
  resident-layer graph p50 values differ by 144.38 µs, but the 30 paired
  differences have median 50.25 µs. The MTP chain graph saves a paired
  median 0.468 ms but wins only 19/30 pairs and worsens p95; that does not
  establish a stable gain. Confirm graph pools, outputs and timing scope.
  [Compiled-layer timing](evidence/ascend-decode/execution-gap-20261018/compiled-resident-timing-audit.json)
  · [MTP chain timing](evidence/ascend-decode/layer-major-20261018/mtp-chain-timing-audit.json)
- Split collective waiting from transport before attributing a graph span.
  One shared-layer sample spends 3.657 ms waiting inside O AllReduce, with
  7.62 µs exported transit; another shifts time into Prolog. Three replays
  were already submitted before the first completed. The 4 ms graph span
  is not evidence of Python submission cost or an intrinsic compute floor.
  [Resident-layer profile](evidence/ascend-decode/execution-gap-20261018/resident-profile-audit.json)
- Measure a changed primitive in its complete consumer chain. MTP projection
  sharding saves a paired 2.389 ms in the isolated M1 path, including gather,
  but only 0.522 ms across two complete draft calls and wins 19/30 pairs.
  Retain actual original modules, include new communication/layout work,
  and trace the exposed dependency path before attributing the difference.
  [Projection and chain measurements](evidence/ascend-decode/layer-major-20261018/README.md#mtp-output-projection-sharding)
- Exact replay and agreement with the previous implementation answer
  different questions. Four one-ULP changes in MTP eh output amplify to
  1.76% recycle NRMSE, yet exact rational dots confirm all four new values
  and reject the corresponding old rounding. Fix the same input/state,
  locate the first change, then evaluate proposal quality and acceptance.
  [Fixed-input diagnosis](evidence/ascend-decode/layer-major-20261018/mtp-projection-fixed-c1-audit.json)
  · [Exact dots](evidence/ascend-decode/layer-major-20261018/mtp-projection-dot-oracle.json)
- Measure proposal quality on a fixed teacher trajectory before inferring
  acceptance from hidden drift. Sharding changes five proposal slots across
  thirty windows, yet every window keeps the same continuous coverage.
  This conditioned screen does not execute M4 verification or measure
  accepted tokens in a session.
  [Teacher screen](evidence/ascend-decode/layer-major-20261018/mtp-teacher-proposal-audit.json)
- Bound clock error before subtracting cross-rank timestamps. Empirical
  causal envelopes preserve millisecond arrival skew, but do not resolve
  microsecond completion spread or always identify the last rank. A long
  AI Core TASK measures elapsed execution, not isolated arithmetic; CSV
  Wait Time is a derived gap between summarized tasks. Nested HCCL child
  waits can cross later operations and must not be added to their parents.
  [Complete-chain trace](evidence/ascend-decode/layer-major-20261018/mtp-chain-profile-summary.json)
- A numerically exact rebuild is not a performance-equivalent baseline.
  O publication fusion wins 9.025 µs against rebuilt O plus push, but its
  paired gain is −0.328 µs against the installed O plus push. The same-run
  rebuild costs 10.609 µs more. Retain installed controls before attributing
  a fusion gain; stop this branch without identifying a speculative cause.
  [Installed control](evidence/ascend-decode/execution-gap-20261018/README.md#o-publication-and-complete-reduction)
- A shared machine is not the explanation until it is measured. The same
  configuration on an idle machine gives 21.210 ms against 20.441 ms on the
  contended one. Contention costs less than the measurement spread here;
  reach for it only after an idle control.
  [Idle control](evidence/ascend-decode/compose-20261018/README.md)
- Separate a kernel's fixed cost from its slope before optimising either.
  SFA fits `29.4 µs + 27.9 ns/row` across sixteen times the selected rows,
  and sixteen times the query heads costs 8%. Head count and traffic are
  therefore not the lever; the intercept and the in-flight load count are.
  [Shape scaling](evidence/ascend-decode/sfa-shape-20261018/README.md)
- A slow gather inside a kernel does not prove the hardware gathers slowly.
  The rows SFA charges 27.9 ns for cost nothing marginal through
  `index_select`, which stays flat to 4096 rows and reaches 196 GB/s
  unsaturated. Time the bare memory operation before writing a new kernel.
  [Gather control](evidence/ascend-decode/sfa-shape-20261018/README.md#top-k-at-heads-4)
- The MSD workspace stages the activation, not the weight. Its preprocess
  reads int8 `x` and writes two int4 halves to GM, so at M≤32 the staged
  bytes are ~196 KiB and the roofline is not missing a dequantised-weight
  round trip. Read the kernel before adding a traffic term.
  [Kernel source](https://github.com/vllm-project/vllm-ascend) `csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pre.h`
- Per-round host coordination is one broadcast, not one per operation.
  `command_serial` 1843 over 1818 rounds rules out Gloo command volume as a
  round cost. Count the control messages before optimising the control path.
  [Round control](evidence/ascend-decode/compose-20261018/idle-default-rank0.json)
- A device tensor the operator reads must live on the device. Leaving
  `actual_seq_lengths_*` on the host faults the sparse kernel with aicore
  exception 507015 rather than a shape error; the existing screens move every
  case entry to the device.
  [Screen convention](../tests/glm52/bench_cann_sparse_attention.py)
- A graph makes the kernel count affordable, so it cannot be blamed for the
  round. Dependent trivial kernels cost 1.079 µs each inside a graph against
  10.09 µs eager, which puts the 561 named calls of one M4 verify at 0.59 ms
  of the 6.752 ms round budget. In-graph SFA is 84 times that floor, so the
  cost is inside the kernels. Measure the in-graph floor before proposing
  fusion to save dispatch.
  [Chain probe](evidence/ascend-decode/graph-chain-20261018/README.md)
- Bytes divided by total time is not a streaming rate when the call is mostly
  fixed cost. The expert GMM looks like a fifth of bandwidth at the model's
  shape, but raising the active groups shows it streams at 1152–1261 GB/s,
  the HBM event median, over a 35.8–39.5 µs floor. Sweep the traffic before
  calling a kernel bandwidth bound, and do not spend effort on bytes that are
  already hidden under a fixed cost.
  [GMM shape scaling](evidence/ascend-decode/gmm-shape-20261018/README.md)
- Optimisations that each look like noise can compose. Five flags that were
  individually inconclusive move TPOT from 21.210 to 19.496 ms, and adding
  the compiled target with SuperKernel reaches 16.640 ms, 21.5% below the
  default, with the round falling 20.2%. SuperKernel's earlier screen could
  only report a task count and explicitly denied a latency gain; in
  combination it is worth 14.6%. Measure the combination before discarding
  its parts.
  [Composed set](evidence/ascend-decode/compose-20261018/README.md)
- A flag can be unreachable for a reason outside the operator. `--compile-target`
  failed because `AOTAutogradCache` expects an `OutputCode` while the npugraph
  FX compiler returns `_CompiledFxGraph`; `TORCHINDUCTOR_AUTOGRAD_CACHE=0`
  bypasses it. The largest single gain measured so far was sitting behind a
  framework cache, not behind a kernel.
  [Cache workaround](evidence/ascend-decode/compose-20261018/README.md#the-compiled-target-needed-a-cache-workaround)
- A vendor operator's shape rules decide whether it is ours to use. MegaMoE
  fuses the whole expert layer and ships inside CANN 9.2.0 ready to import,
  and GLM-5.2 fits every A3 constraint at EP16; but its `intermediate_hidden`
  must be at least 512 and a multiple of 512, and TP16 shards ours to 128.
  Read the constraint table before planning a patch: the EP whitelist is one
  line, the intermediate rule is the real wall.
  [MegaMoE availability](evidence/ascend-decode/mega-moe-20261018/README.md)
- Count the calls from the family table, not from the largest operator types.
  One M4 target verify issues 2,416 calls; the five biggest types are 561 of
  them. At the in-graph floor of 1.079 µs the true count costs 2.61 ms, 38%
  of the round budget, where the five types suggest 0.59 ms. Cast/quant/norm
  alone is 941 calls for 3.032 ms, a third of which is the floor.
  [Family table](evidence/ascend-decode/cann-mtp-profile-20261017/summary.json)
- A resident bank the size of L2 measures L2. The first expert-GMM sweep held
  one 256-group bank, which is exactly 201,326,592 bytes and exactly the die's
  L2, and reported a 64→128 marginal rate of 1261.2 GB/s — above the
  1190.77 GB/s the HBM probe measured on one die. A rate above the measured
  ceiling is the tell. Rotating five banks moved the fixed cost from
  35.8–39.5 µs to 60–63 µs.
  [Re-measurement](evidence/ascend-decode/gmm-l2-20261018/README.md)
- Separate speed from acceptance before crediting a flag. Three of the five
  composed flags make the round shorter when removed and TPOT worse, because
  their whole return is speculative acceptance: `--fp32-head` 1.317 ms,
  `gather_fia` 0.216, `--draft-projection-tp` 0.100. Only `--super-kernel`
  and `--draft-graph` leave A untouched and are genuinely faster. A leave-one-out
  that does not report A cannot tell these apart.
  [Attribution](evidence/ascend-decode/ablate-20261018/README.md)
- A quantity named union may be a sum. In the archived profile the family
  durations add to `compute_union_ms` to the last digit, and the stage
  `compute_union_ms` add to the round's. Either the export sums rather than
  unions, in which case 40.698 ms is not a union, or the tasks never overlap,
  in which case the file's own "duration sums are not additive critical path"
  does not apply to it. Do not build an overlap argument on it until a line
  sweep settles which.
  [Profile](evidence/ascend-decode/cann-mtp-profile-20261017/summary.json)
- Subtracting compute from a span merges three different things. The raw
  per-rank analysis carries `activity_union_ms` next to compute and
  collective, and defines its gap as no traced kernel, HCCL or AICPU
  interval, so verify's genuinely idle device time is 1.014 ms where
  span minus compute gives 7.855 ms. Read the exporter's own fields before
  deriving one.
  [Four-way split](evidence/ascend-decode/draft-collectives-20261018/README.md)
- The same collective costs what the schedule around it allows. A reduction
  inside verify waits 26 µs because 157 of them are pipelined behind 0.26 ms
  of per-layer compute; a reduction inside a single-token draft stage waits
  about 1234 µs because only three of them follow a host broadcast with
  nothing to keep sixteen ranks in step. Forty-six times, for 0.6% transit.
  Optimise the schedule that feeds a collective before the collective.
  [Draft collectives](evidence/ascend-decode/draft-collectives-20261018/README.md)
- Instrumentation that cannot wrap the production path yields no attribution.
  A capture requested with `--draft-graph` records `draft_execution: eager`,
  and adding `--compile-target` fails in `sfa_v1.py:1779` on a width-bound
  output buffer. Every stage-level figure therefore describes a superseded
  configuration, which is why the composed set's 57.12 ms round has no
  decomposition. Check that the profiler records the flags you passed.
  [Capture receipt](evidence/ascend-decode/draft-collectives-20261018/README.md)
- A hardcoded field in a receipt can retire a whole line of inquiry. A
  capture requested with `--draft-graph` reported `draft_execution: eager`
  because that string was a literal in `profile_cann_decode.py`, which led to
  the conclusion that the production path cannot be profiled. The receipt's
  own `draft_graph_state` showed one capture and 28 replays. Check a flag
  against a field that is derived, not written.
  [Five-flag profile](evidence/ascend-decode/five-flag-profile-20261018/README.md)
- A round-time difference between two configurations is not a kernel
  difference when the trajectories differ. Dropping `gather_fia` shortens the
  round by 0.95 ms, which read as the screen failing to transfer; the profile
  shows the attention path is in fact 21.3 µs per layer cheaper, and the two
  cases run 307 and 298 rounds at different acceptance over different expert
  sets. Compare kernels with kernels.
  [Retraction](evidence/ascend-decode/five-flag-profile-20261018/README.md#gather_fia-works-and-an-earlier-objection-to-it-was-wrong)
- A negative result expires when the configuration around it changes. AIV
  reduction measured 5.05% worse than AI_CPU under the default configuration
  and was set aside; under the composed set it is 0.463 ms better, 15.978
  against 16.441 ms, with identical trajectories. The difference is one
  environment variable. Re-test rejected settings after the surrounding
  configuration moves.
  [Expansion mode](evidence/ascend-decode/expansion-20261018/README.md)
- An advanced index on a non-contiguous view materialises the whole view.
  `chunk` left the derived rope caches with a row stride of 128 instead of 64,
  so `_cos_cache[positions]` moved 384 MiB on every metadata build — 7.4 GB
  per round, more than the model's weights — at full streaming bandwidth, so
  it looked like an efficient kernel rather than a mistake. Two
  `.contiguous()` calls took TPOT from 15.978 to 14.540 ms with the trajectory
  unchanged. Check the stride of anything you index by tensor.
  [Rope cache](evidence/ascend-decode/rope-contiguous-20261018/README.md)
- A compiled model is traced at the width of its first forward. The runner's
  serial M1 validation phase ran before the M4 phase, so the compiled target
  was fixed at one row and the M4 phase fed four rows into a one-row buffer.
  That single crash had hidden the compiled configuration from the profiler
  all day, and with it the 9% above. Warm a compiled module at the width you
  intend to use.
  [How it was found](evidence/ascend-decode/rope-contiguous-20261018/README.md#how-it-was-found)
- Fusing the collectives away also hides them. Under AIV expansion with
  SuperKernel, `allreduceAicpuKernel` and `allGather` leave the operator table
  entirely and `communication.json` only keeps 0.03 ms bookkeeping entries, so
  the three AIV groups that carry 37.1% of their type cannot be attributed by
  the usual route. Gaining speed can cost observability; check that the thing
  you optimised is still visible.
  [Round composition](evidence/ascend-decode/profile-rope-20261018/README.md)
- Where a fusion boundary falls decides whether a collective is free. Of 157
  SuperKernel groups that contain an all-reduce, 156 begin with a compute
  kernel and average 36 µs; the one that begins and ends with an all-reduce
  averages 4356 µs, 121 times more, because it has no preceding compute inside
  the group to cover sixteen ranks arriving apart. Group names carry this:
  `start_aiv_all_reduce_float` against `start_static_kernel_AddRmsNormBias`.
  Read the boundary, not just the contents.
  [Grouping boundary](evidence/ascend-decode/profile-rope-20261018/README.md#one-grouping-boundary-costs-436-ms-per-round)
- A PyTorch binding is not a kernel. `torch.ops.npu.npu_mm_all_reduce_base`
  has the widest schema in its family and nothing behind it on A3: the device
  refuses with `SoC version ascend910_93 verification failed. This SoC is not
  configured through the AddConfig API of the OpDef class`. Its neighbour
  `npu_mm_reduce_scatter_base` runs but demands `M % world_size == 0`, which
  batch-one decode cannot supply. Probe the operator on the target SoC at the
  target shape before planning on it.
  [Vendor fusions](evidence/ascend-decode/fused-collective-ops-20261018/README.md)
- Benchmark the layout production uses, or measure an artefact. An int8 matmul
  moved half the bytes of its bfloat16 twin in the same time, which looked like
  a two-fold read amplification worth halving the cube's 22.80 ms. Production
  never sees it: `act_quant_type` defaults to `torch.int8`, so
  `process_weights_after_loading` casts the weight to FRACTAL_NZ. The same
  mistake sat in my own kernel, whose ND tile copy has 256-byte runs and
  reaches 0.18× the vendor regardless of tiling.
  [Streaming rate](evidence/ascend-decode/matmul-streaming-20261018/README.md)
- The bfloat16 matmul is at 93.8% of the die's streaming ceiling at four rows,
  and a chain of thirty-two costs the same per call as one. So there is no
  bandwidth and no launch overhead to win by fusing matmuls together; what a
  fused kernel buys is on-chip activations across the round's ~637 cube-vector
  handoffs and a collective in its epilogue. Aim a megakernel at what
  surrounds the matmul.
- Per-call fixed cost, not bandwidth, owns the batch-one expert GMM: at the
  model's 27-group shape the call is 62–63 µs of prologue around ~18 µs of
  streaming, and rows 1 vs 4 differ by less than the ±4 µs session jitter.
  Measure across processes (median of process medians) — a single session's
  twenty batches hide an additive offset that masquerades as a tiling effect.
  [GMM outside L2](evidence/ascend-decode/gmm-clean-20261018/README.md)
- The failure we probed blind is a documented design boundary one repo over.
  hyper-parallel's bindings name aicore 507015 verbatim as what an untimed
  device-barrier spin does under interconnect contention, and their whole
  protocol — never DCCI a remote address, spin only on local flags, converge
  on the host before first remote visibility — is the complement of our
  failed one-shot push. Search for the error code in shipped source before
  black-box bisection.
  [HyperMPMD](evidence/ascend-decode/hyper-mpmd-20261018/README.md)
