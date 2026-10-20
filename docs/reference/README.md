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
First fused brick measured 2026-10-20: 12.307 us at 8 rows / 4 blocks, against 15.093 for the same work as two
kernels and 14.801 for the vendor pair (`QuantBatchMatmulV3` 11.661 + `DequantSwigluQuant` 3.140) — 17% faster
than the two vendor operators as one kernel, exact against the two-kernel arm, 24 consecutive clean runs.
Per family, ms spent vs ms of bytes it must move: SparseFlashAttention 6.973 / 1.382;
GroupedMatmulSwigluQuantV2 6.793 / 2.272; GroupedMatmul 5.596 / 1.212;
QuantBatchMatmulV3 3.810 / 0.780; MlaPrologV3 2.950 / 2.745; Cast 2.300 / 0.076;
ForeachAddList 1.622 / 0.018.

## Files

| File | Covers |
| --- | --- |
| `megakernel-persistent-decode.md` | Hazy Research Llama-1B megakernel (full text), MPK / Mirage Persistent Kernel, FlashMoE/FlashDMoE, **CANN SuperKernel device-side source**, ThunderMLA, Event Tensor / ETC, plus a cross-system comparison table |
| `ascend-fusion-mechanics.md` | AscendC kernel programming, MIX core semantics, cross-core sync APIs, FFTS+, buffer/tiling, SuperKernel and graph-mode launch fusion — public CANN docs and GitCode source. **§31 is the A3 cube→vector GM visibility rule** (`DataCacheCleanAndInvalid` + `DataSyncBarrier<DDR>` on both sides; `CrossCoreSetFlag` alone is not a fence). **§42-§45 are the `int4b_t` and intra-core-sync block**: §42 sub-byte addressing from the shipped headers (and the hard negative on `SetGlobalBuffer`'s length), §43 the authoritative `PipeBarrier` vs `SetFlag`/`WaitFlag` rule with the per-core-half `HardEvent` split, §44 the two shipped vendor int4 kernels and the 950PR unzip tax, §45 the public found-and-fixed W4A4 erratum |
| `ascend-launch-overhead-and-sync.md` | **Internal Wiki only — do not publish.** Measured MIX head-overhead decomposition, the CrossCoreSetFlag counter limit and MlaPrologV3's real CV pipeline, the 0–7 event-id budget, A3 buffer capacities, `blade/runtime` (an existing Ascend megakernel runtime with `aclrtLaunchMega`), an internal plan for an Ascend MPK, heterogeneous workers, Async-MoE scoreboard sync |
| `fused-moe-expert.md` | gate+up → SwiGLU → quant → down fusion, grouped GEMM at tiny M, W4A8 / antiquant layouts, weight streaming. **Part 8 is grouped matmul over 256 experts inside a hand-written kernel** — `group_list_type`'s three modes, per-expert weight addressing, the 128-group TensorList cap. **§B7 is arXiv 2605.13915 "Multi-Scale Dequant"** — one of four things CANN calls MSD, with the M-stacked single-GEMM form as its equation (16); read §B7.-1 first for the four-family taxonomy and the authoritative A8W4 expansion |
| `w4a8-msd-int4-mmad.md` | The A8W4 MSD decomposition as CANN actually ships it: the four MSD families and which kernels hold each, the documented nibble split and `weightAssistMatrix` / `bias` / `yOffset` correction term, and the `Mmad` both-operands-int4 wall |
| `mla-decode-fusion.md` | MlaPrologV3, SparseFlashAttention / DSA, FusedInferAttentionScore, FlashMLA, weight absorption |
| `low-latency-tp-reduce.md` | one-shot allreduce, symmetric memory, reduction fused into the producing kernel, MC2, LCCL, aclshmem. **Part 7 is the in-kernel collective without a host barrier** — `aclshmem_quiet`'s implementation, epoch/magic tagging, and the A3 AMO limits |
| `speculative-acceptance.md` | prompt lookup, SuffixDecoding, EAGLE/Medusa/tree drafts, MTP, acceptance-length measurements |
| `tilelang-ascend.md` | **Deprioritized** (hand-written AscendC was chosen instead). Kept for the AscendC primitives it documents: pipe flag protocol, real L1/UB address maps, Cube↔Vector handoff, and the tile-ai repo inventory |

## The findings that change the plan

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
3c. **A cross-core flag is a counter, not a fence — this is what breaks `mix_expert_kernel.cpp`.** Nothing in the
   `CrossCoreSetFlag` / `CrossCoreWaitFlag` documentation claims any memory-ordering or cache effect: set increments
   a counter, wait decrements it. On DAV_2201 three independent third-party kernels measured that a GM buffer written
   by one core is *not* reliably visible to another after the flag alone, with the producer's write still in the write
   buffer when the flag arrives — 「flag 已到、写回仍在途」, symptom 「AIV 读到 vTmp 全 0」, and 「加探针即掩盖」, a
   probe masks it. The fix is `DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>` plus
   `DataSyncBarrier<MemDsbT::DDR>()` — producer before the set, consumer after the wait, before the `DataCopy`.
   `PipeBarrier<PIPE_ALL>()` cannot substitute; it orders issue, not retirement. One measured A/B says the producer's
   *barrier* and the consumer's *invalidate* are the load-bearing halves (9/60 → 1/60 failures), and another says
   hardening only the suspected handoff is useless (19/2000 unchanged) while hardening every cross-core buffer removes
   the drift. The ordinary single-operator path gets a whole-DCache flush appended by the Bisheng compiler and another
   from SuperKernel after every sub-kernel; a 78-layer megakernel crosses no operator boundary and must place all of
   them itself. All verbatim in `ascend-fusion-mechanics.md` §31, with the official API pages for both intrinsics.

4. **Fusion is not unconditionally a win at batch=1, and two sources say so with numbers.**
   MegaMoE's own authors report the fused path is *worse* than Dispatch&Combine at small shape
   because pre-synchronization outweighs the overlap (§13.6), and measure ~50% compute inflation
   from AIC/AIV MTE contention when compute and comm overlap on one core. MonoMoE separately
   measured that fusing the TP all-reduce does not pay because it is terminal.
5. **Acceptance is already at the published ceiling; the round time has to come down.**
   `speculative-acceptance.md`: SuffixDecoding's best reported is 7.8 mean accepted tokens/step
   on SWE-Bench against PLD's 3.2, and our measured 7.1 is at that state of the art. vllm-ascend
   also caps `num_speculative_tokens + 1 <= 16` via `npu_fused_infer_attention_score`.

6. **A foreign-core `WaitFlag` is a silent no-op on dav-2201 while its `SetFlag` still fires.**
   `ascend-fusion-mechanics.md` §43.5: `WaitFlagImpl` wraps its `MTE2_V` / `V_MTE2` / `MTE3_V` / `V_MTE3`
   cases in `#ifndef SPLIT_CORE_CUBE`, so on the cube half of a `__mix__` kernel the wait falls through to a
   bare `default: break;` — nothing — while `SetFlagImpl` has no such guard and `SetFlagInternal` screens per
   core only for `__NPU_ARCH__ == 3510`. No `static_assert`, no log, no diagnostic. The operational rule:
   every `SetFlag` / `WaitFlag` / `FetchEventID` naming a vector event must sit inside `if ASCEND_IS_AIV` and
   every cube event inside `if ASCEND_IS_AIC`; an `__aicore__` helper shared by both halves must never name a
   `HardEvent` unguarded. `M_V`, `V_M`, `MTE1_V`, `V_MTE1` must never be named at all on 2201 — their Waits
   are compiled out on *both* halves. The legal set is 13 events on an AIV and 24 on an AIC, not the 35 in
   `HardEvent`. And `PipeBarrier` is the 单流水 primitive, 「阻塞相同流水」 — it is a different instrument from
   `SetFlag`/`WaitFlag`, not a stronger one, so `PIPE_ALL` is not defined to cover a cross-pipe hazard
   (§43.1-§43.2). `PipeBarrier<PIPE_S>()` 「会引发硬件错误」.
7. **`sizeof(int4b_t) == 1` but `SizeOfBits<int4b_t> == 4`, and that gap has already caused one
   found-and-fixed W4A4 defect on this part.** `ascend-fusion-mechanics.md` §42 and §45. Tensor APIs count
   nibbles (`operator[]` does `offset / INT4_TWO`, silently flooring an odd index and *not* adjusting
   `bufferSize_`); raw `__gm__ int4b_t*` arithmetic counts bytes. Within one API family the Level-2
   `DataCopy(gm, ub, count)` counts elements while `Nd2NzParams`'s `dValue`/`srcDValue` count bytes. The GM
   contract is low nibble = even element, row *r* at byte `r*N/2`; K and N must be multiples of 64; and
   `torch.int4` is *unpacked*, one byte per element, so anything sized or indexed from a torch view is 2× out.
   Separately, `SetGlobalBuffer`'s length argument is **inert on NPU** — nothing on the `DataCopy` path reads
   `bufferSize_` — and the Level-0 `DataCopy(GlobalTensor, LocalTensor, DataCopyParams)` **silently emits
   nothing** if the source `TPosition` maps to neither UB nor L1, because that `else` arm is a debug-only check.
8. **The A3 int4 Cube path does not exist on 950PR/950DT, and the sanctioned replacement costs the whole W4
   win.** `ascend-fusion-mechanics.md` §44.3: the vendor's own `matmul_s4` unzips int4x2 to int8 on the vector
   core and **writes the unpacked int8 back to GM** for the Cube to read, so the W4 bandwidth advantage is zero
   at the point of consumption — plus two vector `Cast` passes (int4→half→int8; there is no direct int4↔int8
   intrinsic), 4 bytes of UB per unpacked element held simultaneously, and it runs per kernel call. Any
   layer-fusion design banking on A3 int4 needs its weight path re-derived for 950, not retuned.

## Corrections to earlier findings in this file

Each of these overturns something stated above, with the newer source winning. Read them before acting on
finding 3.

- **flagId on A3 is 0-15, not 0-10, and each AIC and each AIV has its own 16.** Established twice independently
  (`ascend-fusion-mechanics.md` §38.4 from the current `CrossCoreSetFlag` page; §40 from a separate pass). The range
  only *looks* like 0-10 because `SyncAll<false>` reserves 11/12/13 and `SyncAll<true>` reserves 14. Out-of-range
  ids are silently truncated to the low 4 bits, so an id reaching 16 aliases onto 0 — inside `Matmul`'s `[0, 2N-1]`.
  Counter depth is 15 (two of three sources; the API pages say 16), and overflow is a hard error, not a hang.
- **There is no flag-counter reset primitive.** A counter only falls by executing matching Waits (§40). Mode 2
  AIC→AIV requires exactly M Waits per Set with each AIV owning its own counter, and unpaired use is explicitly
  「未定义行为」 (§38.2). The three published disciplines are drain-at-a-barrier (MegaGDN), a 15-deep credit window
  (`catlass` `CrossCoreSetFlagWithReverse`), and flagId rotation (MegaMoE). The unbounded alternative is a GM
  counter, which touches no flagId at all (§40).
- **`PIPE_S` is an unsupported pipe for every sync mode A3 has — and it is the header default** for
  `CrossCoreWaitFlag` (§38.3). Write the pipe out.
- **Mode 2 only reaches a cube's own two subblocks**: AIC[i] can flag AIV[2i] and AIV[2i+1] and nothing else, so the
  cube's output tile must be the exact union of its two AIVs' input tiles (§37, with the shipped two-line tiling
  recipe in §41.3).
- **The two halves of a MIX kernel number the grid differently.** `GetBlockNum()` returns the AIC count on *both*
  halves, while the AIV's `GetBlockIdx()` is a global AIV index in `[0, 2N)`. Shipped code works around it with
  `aivNum = GetBlockNum() * GetSubBlockNum()` and the comment 「doc say aivNum has already mul subNum, by we test
  are not」 (§39.2). `SetDim` counts AIV in `matmul_fused`, AIC in `rope_matrix` — check per API, never assume.
- **The cube→vector path has no cache-coherence problem on A3, only an ordering one.** On 2201 the three caches are
  L2 (shared, no coherence issue), DCache (per-core, Scalar-only) and ICache (read-only); FIXPIPE→GM and MTE2→UB
  both go through L2 (§35.1). What is needed is `DataSyncBarrier<MemDsbT::DDR>()` before raising the flag, not
  `DataCacheCleanAndInvalid`. Finding 3b's "no L0C→UB path" is confirmed, and the L1 escape is closed too:
  `Fixpipe(L0C→UB)`, `DataCopy(L0C→UB)` and `DataCopyL1ToUB` are all 不支持 on A3 (§35.4).
- **`GroupedMatmul` is not in `ops-nn`.** It is in `hicann/ops-transformer` under `gmm/`, with the fused MoE
  megakernel in `mc2/mega_moe` including an A3-specific `mega_moe_kernel_a3.hpp` (`fused-moe-expert.md` Part 8).

## Where to look things up

`asc.gitcode.com/api/` is a complete static mirror of the Ascend C API reference: path-addressable, newer than the
`doc_center` copies, and every page carries an explicit 「产品支持情况」 block naming Atlas A3. Start there.
`hiascend.com/document/detail/...` returns an empty body to a fetcher; `hiascend.com/doc_center/source/...` works
but its id space is undiscoverable. CANN 9.2.0 is not published on the portal at all — 900 is the newest text.
Details and the URL shape in `ascend-fusion-mechanics.md` §35.0.

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
6. `ascend-fusion-mechanics.md` §37-§41 — the MIX brick, start to finish, learned the hard way. §38 is the
   authoritative `CrossCoreSetFlag` semantics (pairing arity, counter residue, the `PIPE_S` trap). §39 is two
   complete vendor kernels: a shipped multi-block `ASCENDC_CUBE_ONLY` + flag handoff (`rope_matrix`) and the KFC
   route where the AIV drives the cube and waits on a `TQue`. §41 is the per-layer kernel: the Matmul object's
   lifetime across stages, `SyncAll<false>()` as a mid-kernel phase boundary with a shipped three-barrier example,
   and the pairing tiling. §35 is the A3 cache model that makes the handoff rule one instruction.

## Provenance rule

`ascend-launch-overhead-and-sync.md` is sourced entirely from `wiki.huawei.com` and is marked
internal at the top of the file. Every other file is public material. Keep that boundary when
anything here is copied elsewhere.
