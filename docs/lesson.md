# The bitter lesson

189 entries below, each with the measurement that produced it. They fall into
seven shapes, and naming the shape is faster than remembering the instance.

**An instrument reading its own overhead.** A bandwidth ceiling measured one
read at a time is a launch latency. An eager event pair can read the host. A
per-call figure can be one launch divided by the repeat count. A tight loop
reads a racy kernel as deterministic because it never changes the workspace.

**A check that cannot fail.** Rank-index contributions make every partial sum
an integer below 2²⁴, so the result is constant across all 16! orderings. A
constant payload is blind to a stale buffer. A harness that defines the
constant the patch forgot to import turns a crash into a pass.

**Attribution by coincidence.** Time containment credits a blocking `tolist`
with every concurrent task. A matching count is not a mechanism. Ablating one
stage of a fused kernel gives a lower bound. A hypothesis that fits a threshold
exactly can still be coincidence — check the differential.

**A fix that was never installed.** An override at the wrong index in
`ASCEND_CUSTOM_OPP_PATH` fails silently. A build script that installs some of
its inputs discards a patch and reports success. `ast.parse` passing excludes
syntax errors and nothing else. Read the position back out of the run.

**Reasoning where a control was available.** Dividing bytes by a peak
bandwidth figure has been wrong by several times every time. An instruction
count is only as good as the data type it assumed. A symptom you caused
yourself is not a code defect, and the fix for it can break what worked.

**Correct-looking output from incorrect computation.** Bit-reproducible tokens
can come from margins absorbing a corrupted operator rather than from right
arithmetic. A release gate that compares tokens cannot see it.

**The build and the environment as hidden variables.** A package cache keyed
on operator parameters and an option string, omitting the vendor path and the
header contents, installs a package another tree built — and only at the
capture size, which reads as a width-dependent race in the kernel. When an
ablation's own switch is in that key, the ablation cannot be run. The
activation script unsets the variables you set before it, and leaves another
checkout of this project ahead of yours on the import path. Read back what the
arm actually loaded, from the arm's own record.

The working rule, in one line: **where a control is available, do not reason.**
[workflow.md](workflow.md) carries that as a table of what each error cost
against what its control would have cost.

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
- Before believing a cost was removed, check the total. Deleting two of four
  consumers of a routing table moved 38 microseconds out of the prologue and
  into the stage that ran next, with the call's total unchanged to within a
  microsecond. Each elimination measured a real few-microsecond saving and
  their sum was nearly nothing, because the cold cost belongs to whoever
  touches the data first. A per-part measurement that does not reconcile with
  the whole is a bill changing hands.
- There is no universal per-call floor to multiply by. Graph dispatch is
  1.2 microseconds, one operator's own prologue is 16.4, another's per-call
  cost is 8.3 -- and multiplying the largest by the round's 2,416 calls gives a
  number larger than the round. The useful quantity is a per-operator
  prologue, which is a finite enumeration rather than a constant.
- A build script that installs some of its inputs manufactures false
  negatives. build_gmm1_barrier.sh copies two headers into the overlay while
  the compiler reads the overlay's installed .cpp, so a patch to that file is
  discarded in silence: the build succeeds, the binary's hash is unchanged, the
  measurement shows nothing, and no error appears anywhere.
  [Floor](evidence/ascend-decode/gmm-fastpath-20261019/README.md)
- A docstring that asserts an invariant is not evidence the invariant holds.
  gather_fia passed no mask at all and justified it in prose -- selection
  already enforces causality -- and that is true to width 6 and false at 8,
  where 941 bit-wise KV checks fail and the output diverges from serial
  decoding at the fourth token. The first round looked clean only because its
  tail had been zeroed.
- Look for the constant that silently bounds a parameter you are sweeping.
  Widening the speculative round lost time until max_num_seqs stopped being
  the literal 4, because the compiler filters its static kernel buckets by
  that limit: the same switches worth 8.9 ms per round at four rows were worth
  nothing at eight. The sweep had been measuring a compiler that had stopped
  applying.
  [Width](evidence/ascend-decode/width-20261019/README.md)
- Check which engine can execute the fix before designing around it. I
  proposed that each of 72 cores bulk-copy a shared table into its own UB,
  reasoning that nothing needed sharing; two of the three variants hang,
  because the 24 cube cores have no UB to copy into -- the platform's own
  cube_vector_combine is split. And warming only the vector side made the call
  slower by the batching machinery's cost, which says both engines were paying
  for the first traversal rather than one of them.
- Before measuring an operator, print its eager host cost beside the event
  pair. npu_grouped_matmul's host enqueue is 28.6 microseconds against 60 on
  device, so an eager event pair reads the device and the earlier L2 results
  stand; the custom fused operator's host floor is 98.97 to 109.90, which is
  why the same method read the host there. One perf_counter decides which.
- Ask whether a span is uncapturable or merely uncaptured. The conditioned
  draft ran entirely eager and the reason was a gate written for width one and
  buffers hard-coded to it, not a data-dependent shape or a host round trip --
  and the same trace already contained a captured collective, which should have
  been the clue. Parameterising the graph was worth 1.1155 milliseconds per
  token.
- An assertion that always fails stops being an assertion. The benchmark
  compared against an operation name the session never emits, so rank 0 raised
  on every run and every summary said fail, including the run whose number the
  project had been quoting for a day. Two investigations tripped over it
  separately before anyone read it.
- Separate a call's intercept from its slope before deciding what to fuse. The
  941 elementwise calls carry 1.5 to 2.9 milliseconds of intercept against 0.4
  to 0.9 of element work, which makes them a better target for a resident
  kernel than the expensive operator everyone was looking at. And the two mixed
  barriers inside that operator are intrinsic rather than overhead, because
  fused work doing the same three stages still has to meet twice.
  [Census](evidence/ascend-decode/prologue-census-20261019/README.md)
- Dividing bytes by a peak bandwidth figure has been wrong by several times
  every time tonight. A gather of eight experts out of 256 moves 12.6 MB and
  predicts 4.1 microseconds at the die's marginal rate; measured it is 15.3,
  running at 54% of peak because strided block copies are not a stream. Price
  a transfer by measuring it, especially when it sits on the critical path with
  nothing to overlap it.
- Ask whether the expensive step can be skipped rather than made cheaper. The
  built-in grouped matmul rejects a short group list, which looked like it
  forced a per-layer copy -- but a narrow view over the resident bank is
  accepted, keeps its NZ format and is bit-identical, so a contiguous active
  window costs nothing at all. One check turned a mandatory 15 microseconds
  into a conditional one.
- `pgrep -f` inside `ssh ... docker exec` matches the diagnostic command's own
  argv, so it reports a process that is only itself. I read "two runs are
  contending" from it and spent a turn killing a job that did not exist, while
  the real state was that nothing had started at all.
- `source /workspace/environment/activate.sh` blocks when stdout is not a tty.
  A background launch under `bash -c` hangs there forever and writes an empty
  log; the same script under `bash -lc` runs. Every working launch in this
  project had used `-lc` and I did not notice it was load-bearing.
- A benchmark that requires a flag will exit in seven seconds, and a detached
  launch makes that look like a long-running job. Read the log's first lines
  before setting a waiter, not after it times out.
- A failure that arrives asynchronously lands outside the try block. A probe
  that tested four tensor expressions on an NZ weight reported all four
  passing, because TransData fails after the call returns and the probe never
  synchronised inside the guard. Three of the four actually fail, which moved
  the fix from the call site to load time. One expression per process, with the
  synchronisation inside the try.
- An operator may rewrite its own input. npu_grouped_matmul overwrites its
  int8 activation in place, with a drift up to 251, which made a bit-exactness
  comparison fail at 1.11 until each arm took its activation from a frozen
  copy. A differential harness has to re-source every mutable input per arm,
  not just per experiment.
- A resident output buffer buys nothing inside a captured graph. Allocation is
  decided at capture and replay never reaches the allocator, so index_select
  with out= measured 16.477 against 16.439 microseconds allocating -- inside
  the noise. The optimisation is real on the eager path and empty in the graph,
  which is where production runs.
- Do not race your own agents for an exclusive machine. Three attempts at a
  sixteen-rank measurement failed on free-memory checks because an agent's job
  started inside the window between my settle check and init_device. The
  machine has no lock; the discipline has to be mine.
- `ast.parse` passing excludes syntax errors and nothing else. A patch
  referenced a name that does not exist anywhere in the target tree, and
  "applies cleanly and parses" read as evidence it would run; the branch that
  would have raised was the guard taken on the production configuration's
  first decode. Resolve every Name against its bindings before handing a patch
  over.
- A test that supplies what the code under test forgot to import is worse than
  no test. The harness locally defined the missing constant, so it validated
  the logic and converted a crash into a pass. When a check runs the code in a
  namespace it will not have in production, it is checking something else.
- Bind the device before anything allocates, not when the model is built. The
  benchmark loaded its inputs before constructing the engine and never called
  set_device, so every rank's first allocation landed on device 0 and rank 0
  then saw sixteen contexts on its own die -- 33.12 of 61.27 GiB free against
  52.08 wanted. A single process sees 59.92 free on the same device, which is
  what proved the device was not the problem. The failure is a race on
  start-up order, which is why it appeared intermittently and why one early
  run succeeded.
- 4256 zombie processes are a real problem and not a memory one. A night of
  killed sixteen-rank runs left them unreaped, and they looked like the
  explanation for held HBM until the obvious was recalled: a zombie has
  already released its memory and holds only an exit status.
- A killed torchrun leaves its rendezvous port bound. Three arms sharing one
  --master-port then die on EADDRINUSE in seven seconds, which reads like a
  code error rather than residue from the run you stopped. Give each arm its
  own port.
- Snapshotting one repository does not isolate a measurement whose model code
  lives in another. Five successive sixteen-rank attempts failed for five
  different reasons and none was a fault in the thing being measured; the last
  was an agent's live edit to vllm-ascend-unified, which my TileRT-Ascend
  snapshot never covered. Isolate by what the run imports, not by what the
  work is nominally about.
- Validate a long measurement's launch path with the shortest possible run
  first. Each of those five attempts cost four to forty minutes and fixed one
  symptom: a shared tree, a missing flag, activate.sh blocking off a tty, a
  device-binding race, a bound rendezvous port. An eight-token smoke would
  have surfaced every one of them in three minutes each, and running the real
  1024-token measurement before the path was known-good is what turned one
  debugging session into five.
- The free-memory failure was residue I had created, and the fix I wrote for
  it broke execution. Three sixteen-rank runs reported 33.12 of 61.27 GiB free
  against 52.08 wanted while a single process saw 59.92 on the same device, so
  I concluded the benchmark was missing an early set_device and added one.
  After that every configuration faulted with an aicore exception and 507015 --
  including one an agent had measured successfully an hour before. Removing
  the line restored execution, and the original symptom was device state left
  by the jobs I had been killing. Diagnosing a transient you caused as a code
  defect, and then shipping the fix, costs twice.
  The discriminating test was to run an agent's own frozen tree with the same
  flags: it completed, which located the fault in my tree rather than in the
  device or the configuration.
- Apportion a performance gap before naming its cause. The vector-only
  kernel's 61.5x shortfall factors as 2.2 for arithmetic intensity, 3.0 for
  core occupancy and 6.2 for per-instruction overhead, so the hardware
  property everyone reached for first -- no int4 vector multiply -- is the
  smallest of the three and fixing it alone buys 2x. The factorisation also
  showed that one of the three is an occupancy artefact: at full occupancy the
  marginal byte cost lands within 6% of the DMA floor and the arithmetic
  nearly disappears.
- "This route is closed" has a margin, and the margin decides a different
  question. Replacing one vendor operator with a vector kernel loses 3.90x;
  replacing the whole family with one resident kernel loses 1.32x on the
  arithmetic, which is ~3.5 ms a round against up to 15.4 ms of per-call
  intercept. The first refutation closed the operator substitution and
  accidentally looked like it closed the megakernel. Ask what the ratio is,
  not only which way it points.
  [Vector-only](evidence/ascend-decode/vector-only-20261019/README.md)
- Check determinism in the session that licenses the number, not once in
  another session. Width 8 was reported bit-identical across five runs by one
  investigation and gave three different token streams across three requests
  in mine, and both are true because the cause is a race. A median TPOT over
  requests whose round counts differ by eight percent is not a speed
  measurement.
- A patch sitting in the repository is not a patch that is installed. The
  vendor kernel running in production was the original object, with the
  epilogue race this project had already found, measured and written a
  one-line fix for -- and that fix is why the width-8 trajectory diverges.
  Before attributing a behavioural difference to your change, check which
  binary is actually loaded.
  [Width 8](evidence/ascend-decode/width8-nondeterminism-20261019/README.md)
- A tight loop is not a determinism test. Eight identical calls in one process
  read a known-racy kernel as deterministic in all ten cells of a matrix,
  because the loop never changes the workspace the operator receives and reads
  of regions it never writes return the same stale bytes every time. Churning
  an allocation between calls exposed it in three shapes at once -- and the
  churn is faithful rather than adversarial, since a real round allocates and
  frees hundreds of tensors between two expert matmuls.
- Two of tonight's bad tests share one shape: the test supplied what the thing
  under test was missing. One locally defined a constant the patch had failed
  to import, turning a NameError into a pass; the other held the environment
  static in a way production never would, turning a race into determinism. Ask
  what your harness is providing that the real caller will not.
- Measure the load order of an override, do not infer it. With the overlay
  first, eight calls agree; with the pristine vendor first, two results differ
  by 99 at the fourth call. An override installed in the wrong order fails in
  silence, which is the standard way a fix gets credited for nothing.
- Installing the fix is also how an attribution gets refuted. The vendor
  epilogue barrier clears its operator's race over 384 calls, and width 8 still
  produced three token streams with it installed, so the cause of that
  divergence was never the barrier. A withdrawal can be right about the
  withdrawal and wrong about the reason, and only the fix distinguishes them.
- Eliminate hypotheses with differentials, and say which ones died. Width 8's
  divergence survived the vendor barrier, survived HCCL determinism, and
  survived process isolation -- three named causes, three experiments, three
  refutations, and the search space is now "what differs across processes at
  one width and not another" rather than a list of suspects. Each refutation
  cost one run; guessing from the code would have cost the same and settled
  nothing.
- Read the operator table for the boring operators. Cast is 554 calls and
  1.54 milliseconds a round, 22% of the budget and four times the next
  elementwise item, and nobody had looked at it in a week of profiling because
  it is not an operator anyone optimises. 314 of those calls exist because
  this project forces an FP32 reduction with a bfloat16 round trip around every
  one of the round's 157 reductions -- a choice made for precision that can be
  kept while folding the conversion into loads the kernel already does.
  [Cast census](evidence/ascend-decode/cast-census-20261019/README.md)
- Check whether the vendor already fuses what you were about to fuse, and then
  check whether its pass actually fires. The norm-quant fusion is enabled by
  default and supported on this part, yet it matches 3.3 times a round against
  16.6 unfused norms -- so the opportunity is in the pass's pattern matching,
  not in a new kernel, and a kernel would have re-implemented work that is
  already running.
- Time containment is not attribution. Matching device tasks to the innermost
  Python frame enclosing them credited 1434 of 1590 casts to three tolist
  calls, because tolist blocks the host for 1429 microseconds and every device
  task from every other stream runs inside that window. It is the same error as
  reading a stream synchronisation's duration as a cost, and I made both in one
  night. Use correlation identifiers, or attribute nothing.
- A matching count is not a mechanism. I wrote that 314 of the round's 554
  casts were the FP32 reduction's bfloat16 round trip because 157 reductions
  times two is 314, and the code search afterwards did not support it. The
  arithmetic coincidence was the whole of the evidence.
- A task with no launching API event is a task inside a captured graph. 1491
  of 1590 in-place casts failed to join on connection_id, which is not missing
  data but the signature of replay: no host code runs, so there is nothing to
  join to. That single fact localised them to the target graph and explained
  why they never appeared in the host launch accounting while costing 1.43 ms
  of device time a round.
- When a trace cannot attribute something because it is inside a graph, turn
  the graph off. The same operations reappear as eager calls carrying their
  stacks, and the attribution becomes exact for the cost of one run -- far
  cheaper than inferring from containment, which had already produced two
  withdrawals.
- An override's position has to be verified in the run's own environment, not
  argued from the string you exported. The bootstrap prepends a vendor path
  derived from `dirname(vllm_ascend.__file__)`, so putting an overlay first on
  PYTHONPATH made it prepend that overlay's own unpatched vendor copy ahead of
  my barrier -- which sat at index 4 and did nothing in every run I published
  from. Spelling the in-tree path second does not help, because the bootstrap
  does not prepend that string. Read the recorded runtime environment back out
  of the run.
- A hypothesis that fits the threshold exactly can still be a coincidence. A
  size test in the sparse attention kernel puts its boundary precisely between
  width 4 and width 6, matching the observed asymmetry, and at width 4 half the
  vector cores early-exit for exactly the predicted reason -- and replacing the
  kernel changes nothing. Fit is not evidence; the differential is.
- "Call it twice and compare" is unusable on a collective. Wrapping the TP sum
  that way reported 314 of 314 results unequal with a difference fifteen times
  the value being measured, because two calls on one symmetric window without a
  barrier let the second overwrite what peers are still reading. The probe
  created the non-determinism it measured.
- A race's ignition point is not a line of code. Three independent bisections
  of the same divergence named three different first-differing operators, while
  agreeing that the dense layers are clean and 73 of 75 MoE layers differ. The
  stable statement is the region and the per-forward re-ignition, and looking
  for the one line would have been looking for something that does not exist.
- Bit-reproducible output does not mean correct computation. In a run that
  demonstrably executed the unpatched expert matmul -- non-deterministic at
  exactly that configuration's verify shape, four distinct results over 64
  calls with int8 codes differing by 115 -- the tokens were identical across
  three requests, because the corruption never flipped an argmax. A gate that
  compares tokens cannot see an error the margins absorb. Scope it carefully
  though: production inherits an environment that already puts a patched
  overlay ahead of the factory object, so this was a property of my own runs,
  and I first wrote it as a live defect in the deployed path.
- Inheriting an environment variable can be safer than setting it. activate.sh
  already ships ASCEND_CUSTOM_OPP_PATH with a patched vendor overlay ahead of
  the in-tree one, and the bootstrap's guard then declines to prepend anything
  because the in-tree path is already present. Exporting the variable from
  scratch drops that overlay and re-arms the prepend, which with a Python
  overlay first on PYTHONPATH hands index 0 to that overlay's own unpatched
  copy. I did exactly that in five runs and read the consequence back out of
  their records afterwards.
- The best evidence for an inference is often a run you already have. "Width 4
  passes its gate on margin rather than correctness" was an argument assembled
  from two separate measurements; reading one run's own vendor resolution
  showed it had executed the unpatched object and still produced a single token
  stream over three requests, which is the claim directly.
- Isolating the compile cache re-exposes a bug a warm cache was hiding.
  `AOTAutogradCache.make_entry` asserts the compiled forward is one of torch's
  own `OutputCode` types while the Ascend FX backend returns a
  `_CompiledFxGraph`, and that assertion runs only on the cache *write* path —
  so a warm `VLLM_CACHE_ROOT` never reaches it. Giving paired arms separate
  cache roots, which is right for avoiding shared state, makes both arms cold
  and kills the first one after it has loaded the model on all sixteen ranks.
  `TORCHINDUCTOR_AUTOGRAD_CACHE=0` in both arms costs compile time, not TPOT,
  and cannot bias the comparison. A good practice can arm a latent failure;
  the precondition belongs in the recipe.
- Verify control flow by extraction and compilation, not by reading
  indentation. A missing brace attached an else-if to a chain nested one level
  in, so five stages of a kernel silently skipped their synchronisation and a
  sixth was unreachable -- and the figure I quoted all night, 11.2
  microseconds, was an unsynchronised floor rather than a correct all-reduce.
  Lifting the forty lines out, replacing the calls with prints and compiling
  them found it in one step.
- An order-sensitive input is not automatically precision-sensitive. My
  cancelling construction put the negation at rank 1, so the accumulator
  returns to zero before the small terms arrive and a bf16 accumulator gives
  the same answer as a float one. The discriminating input puts the negation
  last, forcing the accumulator to carry 4096 through every addition where the
  bf16 step is 32. Design the input against the specific claim, not against
  "being sensitive".
  [Corrected](evidence/ascend-decode/onesided-corrected-20261019/README.md)
- A per-call measurement does not predict a round. Four patches whose per-call
  benches promised three milliseconds a round saved made the round 7.7
  milliseconds slower and added fifteen rounds -- a 21.7% regression against an
  expected gain, with the overlay verified live. Both halves of the prediction
  were wrong, not just its size, so the bench does not reach whatever the round
  does. Every expectation on this project's board that came from a per-call
  delta is suspect until measured inside a round.
- A bundled patch needs a gate on each half. The compact-declaration change
  compressed the declared expert count *and* left the weight in ND, and the
  bit-exactness gate varied only the first with the layout held fixed -- the
  layout change was justified on speed and never on numerical equivalence.
  Different tiling accumulates in a different order, which moves low bits,
  which moves draft acceptance, which moves the round count.
- Round count detects a numerical change that token identity hides. The draft
  head's argmax is the sensitive element and the target's is not, so a
  low-bit change leaves the generated tokens bit-identical while the
  accept/reject path shifts. That is the same asymmetry behind width 8's
  irreproducibility, and it means a token-comparing gate must be paired with a
  round-count check.
- A gate that holds fixed what production varies answers a narrower question
  than the one that matters. Four per-call gates on one patch each held
  something constant -- the layout, the declared slot count, the tie structure,
  the call's neighbours -- and each was correct; the regression they missed was
  found by one end-to-end run in an afternoon against a day of per-call work.
  Ask what your gate is holding still.
- The economical explanation for "correct everywhere I can construct, wrong in
  situ" is a perturbation exposing an existing race, not a new error. A patch
  whose arithmetic is bit-identical at every shape tested destabilised the
  round count in production, and the reading that fits is that it shifted
  timing in a section where a race had already been localised. That turns the
  patch from a failed optimisation into a reproducer for the race, in a
  configuration that is otherwise deterministic -- worth more than the
  milliseconds it was meant to buy.
- An effect smaller than your spread is not a result in either direction.
  Three kernel patches measured +0.224 ms a token against a predicted -0.2,
  with arm ranges of 0.35 and 0.59: unconfounded, harmless, and undemonstrated.
  Say which of the three it is rather than reading the sign.
- A profile that is only written on a zero exit is a profile you will lose
  precisely when the run was interesting. The attribution run completed all
  4579 of its checks and produced no trace, because the harness emits
  --profile-output only on success and 1114 checks had failed on a quantised
  cache differing by one step. Decouple the artefact from the verdict.
- Check whether the mechanism you are proposing already runs. I wrote a
  skeleton proposing that a layer be sliced into engine-homogeneous kernels,
  and the configuration already does exactly that: 55 SuperKernel group shapes
  named start-operator to end-operator, the largest being 471 groups that both
  start and end on the vector engine. What it does not do is keep the
  intermediates on chip, which is a different claim and the one worth making.
- A dispatch-level merge and a kernel-body merge are not the same fusion. The
  SuperKernel rows coexist in the operator table with 1669 standalone Cast
  entries, which is direct evidence those operators are outside the groups; and
  a group spanning two different operator types is a sequence of complete
  kernels in one dispatch. With the in-graph dispatch floor at 1.2 microseconds
  against a per-operator prologue of 8.3 to 16.4, the merge that exists removes
  the small term.
- Sum of per-operator time is not occupancy. The earlier overhead figure came
  from adding up operator durations across concurrent tracks, which double
  counts. A union of busy spans against the profile's own wall clock does not,
  and it says the device is idle 39 percent of steady state -- a different
  quantity from the one I had been reasoning about all day.
- Before calling the timeline's largest item a lever, check whether anything
  overlaps it. EVENT WAIT totals 404 milliseconds, more than every SuperKernel
  group combined, and 99.9 percent of it is covered by work on another track.
  Sixteen of 599 waits are real idle. I was one sentence from announcing a
  nonexistent 404 millisecond win.
- The shape of idle picks the lever. Many sub-10-microsecond gaps would mean
  the host cannot feed the device; a few millisecond-scale gaps mean something
  blocks. Measured, it is the second: 172 gaps of 100 microseconds and up hold
  essentially all the idle, and the gaps under 10 microseconds hold 0.2
  percent. That rules out launch rate at this granularity and points at the
  sampler, which turns out to hold 68 percent of the idle behind four families
  that each pay a host round trip per call.
- A cache key that omits what the compiler actually reads turns every
  experiment downstream of it into a lie. The static-kernel package cache keyed
  on dumped operator parameters, an option string and a determinism flag, and
  left out the vendor path and the header contents -- so a run installed a
  package another tree had built, and only at the capture size. Every arm I ran
  for a day was confounded with "did this arm happen to recompile", and the
  option arms that came out clean came out clean because they changed the key.
  When an ablation's switch participates in a cache key, that ablation cannot
  be run.
- An intermittent fault that tracks one configuration dimension is not
  therefore a race in that dimension's code. This one tracked width because
  static kernels are compiled for the capture size alone, so the captured width
  got a stale package and every other width went through dynamic dispatch to
  the patched object. I spent the day eliminating candidates inside the kernel
  while the carrier was the build system.
- Reproducibility bought by changing the parallel layout is not correctness.
  Expert parallelism gave a stable token stream for weeks and it was a
  different stream from width 4's throughout; the arm that is both stable and
  correct appeared only once the real defect was fixed.
- Three launches died to this machine's environment before one ran, and each
  failure looked like a result. `activate.sh` unsets PYTHONPATH and exports the
  vendor search path whose first entry is the barrier-patched overlay, so
  anything set before it is discarded -- the first attempt therefore ran the
  unpatched in-tree object and read three different token streams from what it
  called its baseline, which would have been a publishable-looking finding.
  `set_env.sh` reads `$ZSH_VERSION` unguarded, so sourcing it under `set -u`
  exits instantly and both arms finish in zero seconds with no record. And
  `activate.sh` leaves another checkout of this project on PYTHONPATH, which
  wins `import tilert` even with the tree under test ahead of it on the path --
  passing a script path from your own tree does not make Python import your own
  tree. Running from inside it does.
- A chain that patches between arms must assert the tree's state before each
  arm, or a rerun measures the patch twice and calls one of them a baseline.
  And it must clear an arm's output directory before any check that can exit
  early, or a stale record from the previous attempt satisfies the "did this
  arm produce anything" test. Both bugs were in the same forty-line script.
- Ranking idle by the operator that ends at each gap's start is adjacency, not
  causation, and I built a 68 percent attribution on it. The one piece a
  controlled experiment could reach -- the metadata copies, 18 of the 98
  milliseconds -- came back negative: removing them made the round's floor
  0.278 milliseconds worse. The operators precede the idle without producing
  it, and the rest of that ranking inherits the same standing.
- A copy_ from a CPU tensor into a device tensor does not synchronise the
  device. I called four families host round trips because they sat next to
  gaps, and then traded two asynchronous DMA copies and a host arange for
  three device kernel launches on a path that runs nine times a round.
- Judge a change on the statistic that is stable, not on the one the goal is
  stated in. The round's tenth percentile holds to 0.6 percent across six
  requests in two independent runs; the TPOT medians of these two arms overlap
  and would have read as noise. Same data, opposite conclusions.
- Stability within a run is not reproducibility between runs, and I spent a
  conclusion on the difference. A round's tenth percentile holds to 0.6 percent
  across six requests inside one run, which is why I trusted a three-against-
  three separation; then the same configuration run later from another
  directory read a floor 0.201 milliseconds higher, against a patch effect of
  0.278. Measured against the comparable run the patch was worth 0.077 --
  nothing. An effect near the between-run floor has to be measured with the
  arms alternated inside one session, not one arm after the other.
- Keep a same-configuration arm in every chain. The only reason this was caught
  within the hour is that the next experiment's control happened to repeat the
  previous experiment's baseline. Without it the wrong conclusion would have
  stood, and it was already written down.
- Python imports the name `sitecustomize` once, so two directories on
  PYTHONPATH each holding one means only the first ever runs. Placing a new
  post-import hook ahead of the existing one would have silently dropped the
  static-kernel cache fix, which changes what the measured binary is -- the
  exact class of error the new hook was built to avoid. Chain the other one by
  path instead of by position.
- An arm's floor is its fastest observation, not its mean. Across 28 arms the
  within-arm spread of a round's p10 is under 0.6 ms in twenty and between 1.9
  and 2.9 in a few, and the slow request's position is not systematic: the
  means by position are 54.766, 54.607, 54.625 and the first request is
  slowest in 15 of 28. So the spread is occasional interference, which a mean
  carries and a minimum discards. On the minimum, two identical
  configurations differ by 0.163 ms, the change I withdrew comes out at 1.04
  times that, and the three kernel patches at 9.3. On the mean the first read
  1.39 and looked real.
- Read the run's own verdict file before diagnosing its exit code. Every arm
  today exited nonzero and I spent a round of investigation concluding the
  cause was "not yet named" -- while summary.json, which this harness has
  always written, carried each rank's validation error in full. One said
  KeyError natural_accepted on a single rank, a field TimedBackend records on
  the advance_mtp branch and not on the advance_mtp_n branch this
  configuration uses, so the pass gate had been failing on a missing field
  rather than on anything about the run. Another said all sixteen ranks
  reported "M4 conditioned graph counts differ", which was the exact defect I
  was about to measure with an A/B whose treatment arm had captures 0 and
  replays 0.
- Fixing a hardcoded width in one place is not fixing it. enable_graph built
  the conditioned graph at a literal 4 and _graph selected it at a literal 4;
  I changed the first, called the result a fix, and the graph was then
  captured at the right width and never replayed. The repair is to make one of
  them ask the other -- the selector now compares against the graph's own
  width, and there is no literal left to disagree with.
- A missing hardware path is not automatically a large cost. I established
  from three sources that this generation has no L0C-to-UB route and built an
  argument on it -- that the layer-scale kernel AMD's design rests on cannot
  work here. Measured, the global-memory round trip it forces costs 1.364
  microseconds a tile and the ordering costs nothing, so 637 of them come to
  0.87 milliseconds against a 52 millisecond round. The absence is real and
  almost free, and the argument was about magnitude without ever measuring one.
- Under core_ratio(1, 2) one AIC block's CrossCoreSetFlag faces two AIV
  blocks' CrossCoreWaitFlag and the kernel deadlocks. The first handoff
  benchmark hung on exactly that, and shmem/fused_proj_reduce_kernel.cpp has
  carried the same pattern unrun since it was written -- it compiles, and
  compiling was all it was ever asked to do.
