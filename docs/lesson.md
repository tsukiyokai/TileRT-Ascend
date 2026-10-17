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
- Compare matched execution paths before rewriting a shared kernel.
  Graph4 can capture an eager model or a compiled model; short and long
  MTP requests also have different acceptance distributions. A large
  operator cost does not establish its incremental cost versus vLLM.
