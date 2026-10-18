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
- A synchronisation can be load-bearing for a reason that is not
  synchronisation. `.tolist()` on a draft result was removed to stop a device
  stall, and it silently took a snapshot with it: the graphed draft returns
  reused buffers, so the chain's candidates became aliases that the next
  replay overwrote. Output stayed bit-exact and the round's wall time did not
  move — only acceptance fell, 3.433 to 2.079, and TPOT went from 14.540 to
  24.504. When removing a copy, ask what else the copy was doing, and keep a
  gate on acceptance, not only on tokens.
  [Aliasing](evidence/ascend-decode/devtok-aliasing-20261018/README.md)
- A passing test is not evidence of coverage. All three scalar doubles handed
  back fresh lists of immutable ints, so no host test could have caught the
  aliasing above. Model the awkward part of the real contract — reused storage
  — and verify the new case fails on the unfixed binary before trusting it.
- A flat curve across every tuning knob means you are not measuring the thing
  you are tuning. Twelve TileLang tilings all landed at 190–198 GB/s and I
  concluded the kernel was slow; the flatness was the Cython adapter's ~130 µs
  per-call dispatch floor. In an ACL graph the same kernel runs at 1141–1203
  GB/s, 96–101% of the die's peak. Time a hand-written kernel the way
  production will call it, and report the dispatch floor separately.
  [Stack selection](evidence/ascend-decode/stack-selection-20261018/README.md)
- Count the budget before choosing a mechanism. 157 reductions per round at
  63.5 µs each is 9.97 ms, and a 2 ms/token round may only cost 6.87 ms, so no
  amount of fusing or overlapping reaches the target — the number of
  collectives has to fall. That turns EP (−78 reductions) and MTP replication
  (−9) from optimisations into prerequisites, and it is an argument no profile
  of the current round would have produced.
- A difference measured on a broken build does not survive the fix. Batching
  rounds per broadcast looked worth 0.88 ms/token while acceptance was
  collapsed and the request ran 492 rounds; with proposals correct at 298
  rounds it is neutral at chunk 8 and 14.6% worse per round at chunk 32. The
  per-round Gloo broadcast was never the cost — the `tolist` synchronisations
  were. When a bug and a feature are measured together, re-measure the feature
  alone before keeping its number.
  [Aliasing](evidence/ascend-decode/devtok-aliasing-20261018/README.md)
- Cutting the number of collectives is necessary but not sufficient. EP16
  removes 78 of the round's 157 reductions and still loses to TP16 by 1.09 ms
  per round, with the vendor's fused MegaMoe adding another 1.41 ms. At batch
  one there is no statistical smoothing, so the busiest of sixteen buckets
  holds four to five whole experts behind a per-layer combine barrier; the run
  spread grows 0.252 → 0.521 → 0.756 ms across the three cases, which is that
  tail becoming visible. Trade a fixed cost for a tail only where the tail is
  averaged.
  [EP16](evidence/ascend-decode/ep-megamoe-20261018/README.md)
- A vendor flag can be a no-op by design. `enable_fused_mc2=1` is rolled back
  internally to `dispatch_ffn_combine` and only 2 reaches `MegaMoe`, after
  which it rewrites itself to 1, so the effective config echoes the same value
  either way. Confirm an operator engaged by finding its name in the run, not
  by reading back the switch.
- `python3 -I` is not free. Isolated mode drops the PYTHONPATH entry that puts
  CANN's `site-packages` on the path, so a capability probe reported a vendor
  package missing when the real benchmark, launched without `-I`, could import
  it. The hygiene habit for untrusted downloads manufactured a false negative
  about our own environment.
- My ND-artefact retraction was itself wrong, and the way to tell was to
  measure the layout instead of reasoning about it. FRACTAL_NZ buys 5% on
  o_proj, not the two-fold the amplification story predicted, so the int8
  matmul's deficit is real. Its cause is per-call fixed cost: half the bytes
  take the same time as bfloat16, and a smaller weight (`q_a_proj`, 9.44 MiB)
  costs more time than a larger one (`o_proj`, 12.58 MiB). A retraction needs
  its own evidence, exactly like the claim it replaces.
  [NZ rates](evidence/ascend-decode/quant-nz-20261018/README.md)
- Pick the instrument per subject. A single-operator ACL graph is ~6 us slower
  than eager back-to-back calls, because the replay floor is exposed while
  eager dispatch is already hidden behind the previous kernel. Graphs are the
  right instrument for a round, eager back-to-back for one operator; using
  either everywhere produces a wrong number in one of the two places.
- Measure the same quantity with both instruments before believing either.
  Eagerly a projection looks 95% prologue; in a graph the marginal is 8 us of
  floor plus streaming at 96% of the die's ceiling, so the prologue is 42% and
  the bandwidth is already spent. The fusion saving halves too, from 46.9% to
  nothing for two-into-one, while four-into-one still pays 37%. Both numbers
  were right about their own regime and only one describes production.
  [Per-call cost](evidence/ascend-decode/fixed-cost-20261018/README.md)
- The budget, not the profile, decides whether a programme can reach the
  target. A round delivering 3.433 tokens may cost 6.87 ms at 2 ms/token, and
  reading each rank's weights once costs 5.7-6.0 ms, so 84% is unavoidable and
  clearing every structural tax leaves 1.1 ms. That makes tokens-per-round a
  first-class lever rather than a tuning knob -- and a projection costs the
  same at 1, 4, 16 and 64 rows, so widening MTP is nearly free where it
  matters most.
- Recompute the keystone before building on it. The weight read per rank per
  round was recorded as 6.83 GB and 6.00 ms, making 84% of a 2 ms budget look
  unavoidable; counted from the checkpoint's own headers it is 4.24 GB and
  3.56 ms, 52%. Every strategic claim that leaned on the old number -- that
  fusion could not reach the target, that the collective count had to fall,
  that EP was a prerequisite -- inherited its error.
  [Budget](evidence/ascend-decode/budget-20261018/README.md)
- A measured per-call cost is not a constant of nature. 63.5 us per reduction
  became an axiom in my reasoning, and multiplying it by 157 "proved" the
  count had to fall, which sent EP16 to the front of the queue where it lost.
  48 KiB across sixteen ranks at 63.5 us is 11 GB/s, two orders off any
  bandwidth limit, so the figure describes rounds and rendezvous and falls by
  an order of magnitude under a one-shot push. Before dividing a budget by a
  measured cost, ask what that cost is made of.
- Sweep the size before calling a collective's rate a bandwidth. A TP16
  float32 all_reduce costs 157 us for four bytes and 110 us for six
  mebibytes, so the 0.76 GB/s at our shape is not a transfer rate but a
  message divided by a rendezvous. That single sweep converts "the collective
  is expensive" into "the collective is one synchronisation, and the payload
  rides free", which is what makes a one-shot push attack all of it rather
  than part.
  [Reduction floor](evidence/ascend-decode/reduce-floor-20261018/README.md)
- Locate a cost by what it does not depend on. The TP16 reduction turned out
  independent of message size, of rank count and of peer identity, and two
  ranks exchanging 64 bytes already cost 45.98 us; torch's cross-die copy is
  160 us at every size and does not pipeline. Three vendor paths sharing a
  per-operation floor that depends on nothing names the cost as a scheduled
  synchronisation stage, which is the thing a one-shot push removes -- not the
  rounds of transfer, which were already free.
  [Locating](evidence/ascend-decode/rendezvous-locate-20261018/README.md)
- When an API is missing, check whether it is a separate component before
  concluding the platform lacks it. CANN ships no aclshmem in 9.1.0 or 9.2.0
  here, but the SDK is its own repository at gitcode.com/cann/shmem and builds
  for A3 in ten seconds. A whole strategy was being reasoned around an absence
  that was an installation gap.
  [SHMEM install](evidence/ascend-decode/shmem-install-20261018/README.md)
- The 507015 that killed TP16 symmetric memory was ours, not the chip's.
  Sixteen PEs form a domain on the first attempt through the vendor SDK, so
  the recorded conclusion that this hardware cannot do it at scale is
  withdrawn. A negative result about a platform should name the
  implementation it was observed through, or it will outlive its cause.
- The wire was never the cost. A one-sided put of 64 KiB between dies
  measures 2.7484 us while HCCL's reduction of a comparable message costs
  63.5 us in-graph and does not vary with size at all. Twenty-three to
  forty-six fold, and the cheap number is the one that actually carries the
  data. Every estimate of this gap today was derived; one vendor perftest
  binary settled it in a minute, and it had been sitting in an uninstalled
  SDK.
  [One-sided put](evidence/ascend-decode/shmem-install-20261018/README.md#the-wire-measured-at-last)
- An overhead large enough hides the structure underneath it. HCCL's pairwise
  latencies spanned only 68.90 to 95.93 us and made rank 0's own module-mate
  look like its slowest peer, which read as "topology does not matter". A
  one-sided put shows 0.31 us within a module against 0.86 us across one, a
  real 2.7-fold. The topology was there all along, under 70 us of scheduling.
  [Wire](evidence/ascend-decode/shmem-install-20261018/README.md#sub-microsecond-and-the-topology-appears)
- Read the whole source before building a programme on an excerpt. Working
  from a quoted paragraph about the one-shot push, I inferred the architecture
  from a symbol table and got it backwards: the design is the entire model
  statically unfolded into one persistent kernel with no launch sequence, and
  the 459 exported kernels are building blocks compiled into it. Two of its
  three mechanisms -- the streaming handshake that replaces global barriers,
  and prefetch scheduled across the whole forward pass -- were absent from my
  plan entirely, and both are the reason the third one works.
  [Reread](evidence/tilert-ascend/amd-article-reread-20261018.md)
- Split the kernel before blaming the mechanism. My one-sided reduction
  measured 543 us and the obvious suspect was the push; switching
  communication off entirely left 557 us, so the whole cost was my own
  reduction loop. A stage flag that disables half the kernel is cheaper than
  any amount of reasoning about which half is slow.
  [TP push](evidence/ascend-decode/tp-push-20261018/README.md)
- Copying an example's loop shape imports its assumptions. combine_classic
  accumulates with a scalar for-loop because it reduces only bs tokens; reused
  for a full [4, 6144] sum that is sixteen global reads per element. The
  example was right for its own shape and wrong for mine, and the resemblance
  hid the difference.
- Sweep the repeat count before trusting any per-iteration number. Four
  successive attempts to explain a 543 us reduction -- scalar accumulation,
  barrier placement, core count, a soft sync -- were all explaining one kernel
  launch divided by fifty. At 1 round the figure is 26454 us and at 400 it is
  70.8, a constant product. The real cost is 5.2 us. A repeat sweep is two
  minutes and it would have replaced every one of those guesses.
  [TP push](evidence/ascend-decode/tp-push-20261018/README.md)
- More cores made the reduction strictly slower: 48 cost 3.8 times what 8 did,
  monotonically. At batch one a 96 KiB reduction is bound by inter-core
  coordination rather than arithmetic, so the parallelism that helps a large
  kernel is a tax on a small one. Sweep the core count rather than assuming
  the die's width is the target.
- A sentinel separates "computed wrongly" from "never ran". My reduction
  reported every element wrong with the value 0.0, which reads as a bad sum;
  pre-filling the output with -12345 and counting survivors reported
  `untouched 24576 of 24576` and moved the search from the arithmetic to the
  store in one run. Zero is the worst possible sentinel because it is also a
  plausible answer.
  [TP push](evidence/ascend-decode/tp-push-20261018/README.md)
- A MIX kernel body executes on both core types, and the vector side has
  GetSubBlockNum() blocks per cube block. Using GetBlockNum() as the core
  count had cube and vector writing the same stripes while the surplus vector
  blocks addressed past the end -- and the cube side has no vector units, so
  its stores were rubbish. Gate the vector work with ASCEND_IS_AIV and count
  cores as GetBlockNum() * GetSubBlockNum().
- UB is one 192 KiB budget shared by every consumer in the kernel, not a
  per-consumer allowance. Stacking a 190 KiB staging area and then addressing
  a reduction buffer at a 192 KiB offset put it out of bounds, and the stores
  failed silently rather than faulting.
- Read the vendor's own demo before inventing a protocol. Mine was inverted
  from examples/allgather in three ways at once -- flags at the tail instead
  of the head and indexed per rank instead of per core, the producer pushing
  to all peers instead of to itself, and the consumer blocking in
  signal_wait_until instead of polling peers with int32_get_nbi and moving on.
  The third is what cann/shmem PR 897 names as the source of 507015. Adopting
  the demo's structure removed the deadlock outright.
  [TP push](evidence/ascend-decode/tp-push-20261018/README.md)
- A project's issue tracker is primary evidence. cann/shmem PR 913 and PR 897
  describe, in their own words, the 64-byte signal slot that stops adjacent
  ranks sharing a cache line, the inter-core sync before a receive wait, the
  timeout that an untimed wait needs, and the SDMA precheck whose absence
  prints aclError=507015 -- the exact error we had. Searching the repository
  cost two minutes and replaced an evening of guessing.
- Put every eight-byte kernel parameter first. An eight-byte parameter placed
  after a four-byte one is written by the host stub at a non-eight-aligned
  compact offset and read by the device from past the end of the args buffer,
  so it arrives as something else entirely. CANN's own SHMEM does this in
  aclshmemi_barrier_on_stream_kernel, which is why aclshmem_barrier_all has
  never worked on this machine: it logs 507015 on every call and returns void,
  swallowing the error. Nine more instances sit in the SDK's examples.
  [ABI defect](evidence/ascend-decode/shmem-abi-20261018/README.md)
- A silent no-op in a dependency can masquerade as a defect in your own code.
  I spent an evening isolating stages, rewriting a protocol and reading vendor
  demos while the deadlock was a host barrier failing before my first launch
  ever ran. Worse, every measurement I took through it was of something else:
  my push-versus-reduce attribution came out exactly backwards. When a
  dependency's error path returns void, check its log before trusting any
  measurement that crosses it.
- One-sided symmetric memory survives ACL graph capture and replay on A3, and
  the way to establish that was same-binary polarity rather than inference:
  with capture bypassed the failure was identical but for the wrapper status,
  so capture was not in the causal chain. 400 replays then ran 800 kernels
  containing put_nbi, signal_op and SyncAll bit-correct. This was the largest
  unknown in the whole one-sided programme, since production decode is
  entirely graph-captured.
- `aclshmemx_mte_put_nbi`'s `pe` argument decides whose window the data lands
  in, so passing `my_pe` is a local copy and nothing crosses a rank. My
  one-sided all-reduce ran, passed a deadlock hunt and measured a number while
  computing a sum over fifteen memset zeroes -- the premise that "every rank's
  data is already local after publish" was a misreading of the API. A kernel
  that runs is not a kernel that communicates; the payload check that catches
  this is cheap and I had it, but only applied it to the stage that could not
  fail.
  [One-sided all-reduce](evidence/ascend-decode/onesided-allreduce-20261018/README.md)
- A stage flag that disables one side of a handshake does not isolate it, it
  deadlocks it. Turning the publish off while leaving the consumer's flag poll
  running made consumers wait on a flag nobody would ever raise, and I read
  that as "the reduction stalls". Both measurements taken that way were void.
  When splitting a protocol for attribution, remove the wait with the writer.
- Two allocators cannot share one UB. Hand-picked FLAG_UB_BASE and
  STAGE_UB_BASE alongside a TPipe is two schemes dividing the same memory
  without knowing about each other, and "the TPipe allocates on top" is not
  something a TPipe can do. Take every buffer in a kernel from one allocator.
- Check a replacement collective against the thing it replaces, not only
  against a payload invariant. The one-sided reduction passes its own check
  (every element must equal the rank sum) and differs from HCCL in zero of
  24576 elements; the second check is what licenses swapping it into a path
  whose trajectory gate compares tokens bit for bit.
  [Torch operator](evidence/ascend-decode/torch-allreduce-20261018/README.md)
- A library that initialises fine in a standalone script can fail inside a
  framework worker because the calling thread has no device. `attr_init`
  reads the current ACL device and tried −1 inside a vLLM worker, returning
  ACLSHMEM_INNER_ERROR, while the same call returned 0 in a plain process that
  had set its own device. The Python return code said only "internal error";
  /root/shmem/log named it in one line. Read the dependency's own log before
  reasoning about its return value, and set the device context explicitly when
  bringing a device library up from inside someone else's worker.
  [Torch operator](evidence/ascend-decode/torch-allreduce-20261018/README.md)
- A faster collective is not the same change as a fused one. The one-sided
  reduction is 2.84 times the vendor's and still a standalone kernel, so it
  pays its own launch and overlaps the compute it follows by exactly zero --
  the structure the AMD write-up explicitly rejects. Shrinking an 11.29 ms
  item is bounded by that item; hiding it behind the weight stream, which has
  to happen anyway, is not. Measure what a change removes from the critical
  path, not what it removes from its own cost.
  [Fused projection](evidence/ascend-decode/fused-proj-reduce-20261018/README.md)
- Read the vendor's own fused example before assembling a protocol out of its
  library calls. The SHMEM tree ships a 4896-line megakernel that fuses
  routing, dispatch, two GEMMs, SwiGLU and combine into one device pipeline,
  and its entire use of the one-sided API is three calls: `aclshmem_ptr`,
  `aclshmem_my_pe`, `aclshmem_n_pes`. Symmetric memory is addressable, so a
  remote write is an ordinary `DataCopy` to the pointer `aclshmem_ptr`
  returns. The wrapper I had built the kernel on -- `mte_put_nbi` with its
  staging and its trailing `quiet` -- was costing 8 of 22.3 microseconds for
  nothing.
  [Direct store](evidence/ascend-decode/direct-store-push-20261018/README.md)
- Communication overlaps compute by partitioning cores, not by interleaving on
  one core. The official megakernel gives `vectorSubBlockIndex_ == 1` to
  dispatch and `== 0` to driving the cube, for the kernel's whole life, coupled
  by flags. A standalone collective cannot overlap anything because it has no
  cores left running the work it should hide behind -- which is the mechanism
  behind the measured 0.000000 ms of overlap, not a scheduling accident.
- A ratio whose numerator and denominator come from different captures is not
  a measurement. The 7.4 ms "unexplained" residual divided a profiled
  operator sum by an unprofiled wall clock; with the profiler on, the round is
  133.93 ms, not 49.92. Before subtracting two numbers, check that both were
  taken under the same instrument.
- Time the host spends blocked is not time anyone pays. Every
  `aclrtSynchronizeStream` over a millisecond returns while the device is
  still busy, and the next device task starts within 23 microseconds, because
  the call drains one stream and the model runs on others.
- An operator table can book waiting as compute. One reduction group in the
  round costs 4.32 ms where its 156 identically-named siblings average
  36 microseconds; the difference is rank skew absorbed at the round's first
  collective and attributed to the kernel that happened to be there.
  [Round accounting](evidence/ascend-decode/round-accounting-20261018/README.md)
- A bandwidth ceiling measured one read at a time is a launch latency. The
  1190.77 GB/s figure that bounded this project's whole budget argument came
  from an event pair around each single transfer; batching thirty reads inside
  one pair gives 1527-1564 marginal, and the old method's own best sample
  already exceeded its reported maximum.
- An operator that ignores a cache is not evidence the cache is absent. The
  expert GMM moves under 1% as its footprint sweeps from a tenth of L2 to
  thirty-two times it, while a bf16 matmul in the same harness shows 2.21x --
  because the GMM runs at 18% of the die's read rate and its bytes are already
  free. Check whether the thing you are tuning is bound by what you are
  tuning.
- The 2 ms target is a statement about operator efficiency, not about
  bandwidth or arithmetic. Device work is 37.7 ms per round against a 6.866 ms
  budget, and the weight stream runs at 18% of hardware: 37.7 / 5.6 lands on
  the budget. The slack and the host's 5.861 ms share one cause -- barriers
  that exist because the work is chopped into separately launched pieces.
  [Weight floor](evidence/ascend-decode/weight-floor-recount-20261018/README.md)
- An eight-byte kernel parameter placed after a four-byte one is read from
  past the argument buffer on this device. The SDK commits this in its own
  barrier kernel, so aclshmem_barrier_all has never worked here -- one
  sixteen-rank run logged 964 swallowed 507015 errors -- and every measurement
  taken through that barrier was measuring something else. Eight-byte kernel
  parameters go first.
- "Works standalone, fails inside the worker" is a question about signals
  before it is a question about devices. The bootstrap link classed EINTR as a
  link error rather than retrying it, which a vLLM worker hits constantly and
  a quiet torchrun process almost never does. I changed the device-binding call
  three times against a symptom that said device -1, while the device was
  bound correctly and latched once per process where no later call could
  reach it. [Two defects](../patches/cann-shmem-1.7.0/README.md)
- A fence that costs nothing may be doing nothing. aclshmem_quiet early-returns
  when the transport mask is zero, which it is on single-machine HCCS, and
  sync_vec_all states it does not complete outstanding communication. Measuring
  the fence as free was correct and the inference that ordering was therefore
  cheap was not: a DataCopy to a peer is an ordinary MTE3 store that no SHMEM
  call tracks, and the writer must drain its own queue.
- A check that cannot fail is not a check. Having each rank contribute its own
  index makes every subset sum an integer below 2^24, so float32 never rounds
  and the result is constant over all 16! summation orders -- 200000 random
  orders gave one distinct value. "Agrees with HCCL element for element" was
  the same illusion, since the compared values are exactly representable. Both
  claims cannot even hold at once on an order-sensitive input, because a
  rank-ordered sum and HCCL's rotation are different sums. Design the input so
  that the failure you are excluding would change the answer.
- A constant payload hides time, not just order. Writing the input once and
  reusing it for four hundred rounds makes the check blind to a stale plane, to
  round crosstalk, and to an implementation that communicates in round zero
  and re-reads the same buffer 399 times -- which would also report a lower
  per-round cost. Vary the payload with the round, on a period coprime to any
  buffer parity.
- Evidence that something survives graph capture must exercise the mechanism
  the production path uses. The vendor demo carries its per-round variable
  through device memory; mine was a kernel immediate the host incremented, and
  replay runs no host code, so the parity froze at capture. With an odd number
  of reductions per captured graph and two planes, back-to-back replays put the
  same plane on consecutive rounds with one sync between -- a race the constant
  payload could never show.
  [Audit](evidence/ascend-decode/onesided-audit-20261018/README.md)
- An instruction count is only as good as the data type it assumed. I argued
  the cube was unnecessary at batch one from four Axpy instructions per 128
  weights, having silently assumed fp16 weights -- and fp16 weights cost two
  to four times the bytes that are the entire point. The vector unit has no
  int4 and no int8 multiply, so every quantised weight must be widened and
  sign-restored first: 8.13 element-ops against a 3.69 budget, measured at
  3.90 times slower than the vendor operator.
- The floor for a perfect kernel is not the cost of the present one. At batch
  one the expert GMM spends 3.4 microseconds of a hundred on crossing L2,
  while 40% goes to scanning empty routing slots and 23% to the launch floor.
  Ranking work by the perfect kernel's bottleneck put weight bandwidth first
  when the real item was a dense table nobody needed to read.
  [GMM decomposition](evidence/ascend-decode/gmm-fastpath-20261019/README.md)
- Subtracting one stage's work from a fused kernel measures a lower bound, not
  that stage's cost. The call's duration is the critical path after the vector
  and cube paths join at a barrier, so removing work off the critical path
  changes nothing, while a variable that shortens both paths reports the full
  effect. I published an ablation table as a decomposition and it was not one.
- "Linear in N, therefore removing the N things removes the time" skips a step.
  The expert GMM's cost is certainly linear in the routing table's length, and
  only a tenth of it could be attributed to code anyone can delete; the
  per-slot term is thirty times the cost of the scalar reads that were found,
  so the mechanism is not the reads. Localise before paying for an interface
  change.
  [Correction](evidence/ascend-decode/gmm-fastpath-20261019/README.md)
