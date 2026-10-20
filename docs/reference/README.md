# docs/reference — collected source material

Verbatim source material for the GLM-5.2-W4A8C8 / Ascend A3 ultra-low-latency decode work.
Nothing here is original analysis; every item is quoted as published, under its URL, author
and date, with a one- or two-line `## Relevance` note tying it to our measured numbers. Each
file ends with `## Gaps`.

Our measured baseline, for reading the Relevance notes:
decode round 53.05 ms unprofiled; best TPOT 11.6198 ms; prompt-lookup acceptance ~7.1
tokens/round; ~3008 kernels/round; per-call launch floors AI_VECTOR_CORE 0.70 us,
AI_CORE 2.30 us, MIX_AIV 12.04 us, MIX_AIC 13.58 us (505 MIX_AIC calls/round, 6.5/layer),
giving an 11.5 ms launch floor with 82% on MIX; TP reduce 5.35 ms and 100% serial.
Per family, ms spent vs ms of bytes it must move: SparseFlashAttention 6.973 / 1.382;
GroupedMatmulSwigluQuantV2 6.793 / 2.272; GroupedMatmul 5.596 / 1.212;
QuantBatchMatmulV3 3.810 / 0.780; MlaPrologV3 2.950 / 2.745; Cast 2.300 / 0.076;
ForeachAddList 1.622 / 0.018.

## Files

| File | Covers |
| --- | --- |
| `megakernel-persistent-decode.md` | Hazy Research Llama-1B megakernel (full text), MPK / Mirage Persistent Kernel, FlashMoE/FlashDMoE, **CANN SuperKernel device-side source**, ThunderMLA, Event Tensor / ETC, plus a cross-system comparison table |
| `ascend-fusion-mechanics.md` | AscendC kernel programming, MIX core semantics, cross-core sync APIs, FFTS+, buffer/tiling, SuperKernel and graph-mode launch fusion — public CANN docs and GitCode source |
| `ascend-launch-overhead-and-sync.md` | **Internal Wiki only — do not publish.** Measured MIX head-overhead decomposition, the CrossCoreSetFlag counter limit and MlaPrologV3's real CV pipeline, the 0–7 event-id budget, A3 buffer capacities, `blade/runtime` (an existing Ascend megakernel runtime with `aclrtLaunchMega`), an internal plan for an Ascend MPK, heterogeneous workers, Async-MoE scoreboard sync |
| `fused-moe-expert.md` | gate+up → SwiGLU → quant → down fusion, grouped GEMM at tiny M, W4A8 / antiquant layouts, weight streaming |
| `mla-decode-fusion.md` | MlaPrologV3, SparseFlashAttention / DSA, FusedInferAttentionScore, FlashMLA, weight absorption |
| `low-latency-tp-reduce.md` | one-shot allreduce, symmetric memory, reduction fused into the producing kernel, MC2, LCCL, aclshmem |
| `speculative-acceptance.md` | prompt lookup, SuffixDecoding, EAGLE/Medusa/tree drafts, MTP, acceptance-length measurements |
| `tilelang-ascend.md` | **Deprioritized** (hand-written AscendC was chosen instead). Kept for the AscendC primitives it documents: pipe flag protocol, real L1/UB address maps, Cube↔Vector handoff, and the tile-ai repo inventory |

## The five findings that change the plan

0. **A working, public, hand-written Ascend megakernel already exists, and its grid barrier is
   copy-pasteable.** `huawei-csl/megagdn-pto` fuses seven stages into one `<<< >>>` launch using
   nothing more exotic than `__aicore__` functions calling each other, with
   `SyncAllMegaKernel<false>()` between stages; it is merged into the SGLang NPU ecosystem and
   gives ~15% faster Qwen3.5/3.6 prefill in vllm-ascend. The full A2/A3 and A5 barrier
   implementations are reproduced verbatim in `megakernel-persistent-decode.md` §8.4. Read that
   section first — it is the shortest path from here to compiling code.
1. **Two Ascend megakernel runtimes also exist.** CANN's `SuperKernel`
   (`gitcode.com/cann/graph-autofusion`) is open source, written in AscendC, and its device-side
   interpreter — GM task queues, `TYPE_FUNC`/`TYPE_SYNC`/`TYPE_PRELOAD` dispatch, ICache
   preload, GM notify/wait — is reproduced verbatim in `megakernel-persistent-decode.md` §4,
   with its fusion-boundary enums in §4b. A second, more ambitious one (`blade/runtime`, Host
   API `aclrtLaunchMega`, AICPU scheduler, per-core 128-slot HBM rings, MMIO doorbell) is
   documented internally in `ascend-launch-overhead-and-sync.md` §10 but was not found publicly.
   Deciding whether either is obtainable comes before writing a runtime.
2. **The 13.58 us MIX floor is itemized, and a third of it disappears with static shapes.**
   `ascend-launch-overhead-and-sync.md` §1: ~4 us irreducible hardware, 2.46 us kernel-args
   prefetch, 2.3 us TilingData copy, 1.3 us other — and the TilingData copy falls from 2.845 us
   to 0.038 us when the shape is const. Batch=1 specialization makes every shape const.
3. **The synchronization budget is the binding Ascend constraint.** `CrossCoreSetFlag` flagIds
   are **0-10** on A2/A3 (`ascend-fusion-mechanics.md` §4, §20 — note
   `ascend-launch-overhead-and-sync.md` §4.7 carries a conflicting "0-7" claim that is
   corrected in place; use 0-10). `Matmul` with N objects already consumes `[0, 2N-1]`, plus
   flagId 15 under SuperKernel 1:2, leaving about 2-10 hand-allocatable. Each counter saturates
   at 15 Sets, no two consecutive Sets of the same flagId are allowed, and exceeding it
   deadlocks non-deterministically (a real defect, DTS2025123130366). Three documented ways
   round it: GM counters with `dcci` polling (SuperKernel's `SkMemoryWaitFlag`), a dedicated AIV
   scoreboard tracking only the latest-completing producer, and power-of-two-decreasing expert
   grouping `{8,4,2,1,1}`.
3b. **Four A3-specific hazards to design around from day one.** On 2201 `CrossCoreWaitFlag`
   ignores its `pipe` argument and blocks *all* pipes, so every AIC→AIV handoff is a full-pipe
   barrier on the waiter. Mode 4 (AIC to a single AIV) is 3510-only, and mode 2 is documented
   to leak across MIX blocks at blockDim > 1 and deadlock the producer — with no mode-4 escape
   on A3. `__schedmode__`/batchmode is mandatory, not advisory: a vendor probe measured 3 hangs
   in 49 runs without it and 0 in 240 with, concentrated on small shapes, i.e. our regime. And
   **A3 has no L0C→UB path** (950PR added it), so cube→vector is
   `L0C → Fixpipe → GM → MTE2 → UB`; the measured penalty is small only because the write lands
   in L2 (`aiv_mte2_time` 320.543 → 57.145 us when fused, −82.2%), so the goal is "don't miss
   L2", not "don't touch GM". Conversely `VECOUT → A1 → A2` vector→cube *is* available, and
   `TPosition::CO1` keeps an accumulator in L0C across `Iterate` calls. All in
   `ascend-fusion-mechanics.md`.
4. **Fusion is not unconditionally a win at batch=1, and two sources say so with numbers.**
   MegaMoE's own authors report the fused path is *worse* than Dispatch&Combine at small shape
   because pre-synchronization outweighs the overlap (§13.6), and measure ~50% compute inflation
   from AIC/AIV MTE contention when compute and comm overlap on one core. MonoMoE separately
   measured that fusing the TP all-reduce does not pay because it is terminal.
5. **Acceptance is already at the published ceiling; the round time has to come down.**
   `speculative-acceptance.md`: SuffixDecoding's best reported is 7.8 mean accepted tokens/step
   on SWE-Bench against PLD's 3.2, and our measured 7.1 is at that state of the art. vllm-ascend
   also caps `num_speculative_tokens + 1 <= 16` via `npu_fused_infer_attention_score`.

## Reading order if you are about to write the megakernel

0. `megakernel-persistent-decode.md` §8 — MegaGDN. The minimum viable Ascend megakernel, with
   a working grid barrier for A2/A3 and A5, and the fact that licenses the whole approach:
   one `__aicore__` function may call another, so stages compose without an interpreter.
1. `megakernel-persistent-decode.md` §4 and §4b — CANN SuperKernel. An Ascend megakernel
   interpreter also exists, is open source, and its device-side dispatch loop, task
   descriptors, core sync and GM notify/wait are reproduced there verbatim — along with the
   enums that decide what it will and will not fuse.
2. `ascend-launch-overhead-and-sync.md` §10 — `blade/runtime`. A second, more ambitious Ascend
   megakernel runtime exists internally with an AICPU scheduler and per-core HBM rings. Decide
   whether to obtain it before reimplementing it.
3. `ascend-launch-overhead-and-sync.md` §1 — where the 13.58 us actually goes, and why static
   shapes remove 2.8 us of it for free.
4. `ascend-launch-overhead-and-sync.md` §3, §4 and §12.3 — the synchronization budget (8 event
   ids per pipe pair, 15 outstanding sets per flag) and the two ways round it (GM counters,
   scoreboard AIV).
5. `megakernel-persistent-decode.md` §1 and §2 — the two canonical designs, for the paged
   on-chip allocator and the AOT/JIT task-launch distinction.

## Provenance rule

`ascend-launch-overhead-and-sync.md` is sourced entirely from `wiki.huawei.com` and is marked
internal at the top of the file. Every other file is public material. Keep that boundary when
anything here is copied elsewhere.
