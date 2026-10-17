# Lessons

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
