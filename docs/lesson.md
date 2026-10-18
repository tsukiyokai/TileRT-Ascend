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
