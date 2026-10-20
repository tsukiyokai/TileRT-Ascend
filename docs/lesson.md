# Lessons

1. Bind an experiment to its actual loaded source, host library, device ELF,
   compiler and input hashes. Read the activated install.info; 84/187 currently
   use toolkit 9.2.0 / V100R001C25B114. A directory label or older inventory is
   insufficient. Rebuild the control with the same compiler as the candidate.
2. Measure the attributed change. Profiled gaps include instrumentation and
   host synchronization; kernel sums can count overlap twice. Pair unprofiled
   A/B requests and report the actual critical-path difference.
3. Small-array hot replay measures a cache-resident regime. HBM calibration
   needs rotating data beyond L2. A peer-read loop omits push/flags and does not
   measure a complete AllReduce.
4. Zero AICore utilization does not make a resident model free. Check HBM,
   processes, container startup and shared-pool ownership. Use CPU first,
   then bounded single-die correctness; reserve whole hosts for dependent work.
5. CANN and the tested path can share an error. Keep independent HF arithmetic
   diagnostics; fix the reference before changing tolerances. Layer metrics
   locate faults, while frozen teacher and serial/speculative output checks
   gate release. Same-configuration graph/reset should reproduce exactly.
   A3 Fixpipe truncates scale's low13 FP32 bits; full-precision scale in an
   oracle is a different computation. GMM2 mutates INT8 input: restore it for
   every A/B call. Do not infer wide-verify gains from M1/M8 active-table gains.
6. Lookup owns target KV only. Its context includes the pending output; MTP KV
   stays at the injected P−1 length. Switching back requires genuine conditioned
   MTP rows, not a logical length update. Clone hidden values before graph reuse.
7. Replay the deployed padding and clip the final bonus. Every policy walks
   its own accepted boundaries. Predictions beyond a rejected draft belong to
   that rejected context; an old MTP trace cannot score arbitrary new branches.
8. A suffix match is not a correct continuation. Adding prompt material can
   displace a better shorter match. Index only occurrences with a known next
   token; otherwise the current tail hides the useful previous occurrence.
9. GLM-5.2's index_topk_freq4 is cross-layer reuse, not four-token pooling.
   Production uses RoPE64 and standard token indices; the KPool path rejects
   that configuration. Checkpoint, instantiated class and loaded source decide
   the contract. C4 synthetic correctness does not establish applicability.
10. Preserve dataflow when removing a boundary. C1 can reuse original token
    indices, but pipeline drain must test bounds before reading another index.
    A fixture's extra backing element does not fix a production out-of-bounds
    access. Registered operator coverage must survive a host-library replacement.
    Correct C1 dispatch is still about 11x slower in the paired M8 screen;
    selecting a Cube template does not remove per-token gather overhead.
11. Compiled overlays can silently keep old entry files. Copy every patched
    installed header/entry, bind the selected variant JSON to the new ELF SHA,
    and check actual tiling keys and mappings in the consuming process. Explicitly
    bind the experiment import root rather than relying on import side effects.
    Static kernel cache keys omit vendor source identity and live below cwd,
    outside VLLM_CACHE_ROOT. Compile each arm in its own fresh output directory.
12. Pinning seq_lens removes a queue-dependent host stall without adding device
    work. Keep measured paired savings separate from the best absolute request.
    More speculative rows can expand the expert union; a 64-expert sample is
    not a worst-case bound for width 32 with top8 routing.
13. A serial/speculative mismatch does not identify future-token leakage.
    Inspect selected IDs and causal thresholds, then perturb only forbidden
    KV slots. An empirical width limit is not an indexer causality boundary.
    A width fixture must update written KV length and right-down causal
    thresholds together; changing query width alone shifts the valid prefix.
14. Inspect the entire dtype chain: uint8 scatter uses AIV, but OR with bool
    promotes back to uint8 and can select AI_CPU. A bool view preserves storage
    and selects LogicalOr; removing one fallback is not a complete-path win.
15. Reuse selection masks by producer generation, not tensor address. Rebuild
    at each producer on every forward; consumers keep their own Q/KV. Full-KV
    dense PV requires finite masked rows because zero times NaN is still NaN.

16. Read the state machine, not the call site. `_advance_drafted` proposes
    before it commits, which reads as a drafter one token behind; `resolve`
    settles it, because `accepted` holds the target's predictions and the next
    pending is the last of them, so the context already ends at the pending.
    A one-line "fix" derived from the call-site ordering was a regression,
    shipped to a gated experiment tree before the state machine was read.
17. When an offline policy can only be priced through an assumption, measure
    the assumption's range before building on it. A merged policy must advance
    along its own boundaries; dealing another policy's recorded match lengths
    independently of position moved the answer across break-even, so the route
    is not established rather than promising. Scoring any policy at a different
    policy's boundaries also double-counts one long repeat per boundary spanned.
18. A driver launched through `docker exec` from an ssh session dies with that
    session, leaving its `sleep` children as PID 1 zombies; a waiter watching
    for records then waits on work that is not running. Launch with
    `docker exec -d` and `setsid`. `pkill -f` inside `docker exec bash -c`
    matches the wrapper's own argv and kills itself first — kill by PID.

19. After the first free-running branch, later predictions no longer test the
    frozen teacher's inputs. Force all32 inputs in both M1 and wide execution;
    keep label acceptance separate from serial/speculative consistency and
    save the first difference before resetting the prefix.

20. Removing a dense expert scan from one consumer can move its cold cost to
    another. Publish one compact table and replace every consumer; retain
    original expert IDs, row prefixes and tile order. Mixed-core barriers
    publish DMA completion but do not replace scalar DCache invalidation.
21. Compare captured outputs only after explicit graph replay and device sync.
    Verify the installed OPC headers and actual ELF, not just the source tree;
    patches rooted at `csrc/` and its parent need different strip depths.

22. A dynamic fallback preserves the chosen arithmetic, not the old critical
    path. Moving a barrier before Matmul initialization can cost time even
    when the selected tile shape is unchanged. Compare complete release ELFs;
    a test switch that skips a metadata read also changes the control's cost.

23. Kernel templates do not override host API dtype checks. For pure copies,
    a dtype view can preserve FP16 bytes while satisfying an INT8-only path;
    a numeric cast cannot. Check the real producer, nonzero storage offsets
    and the entire backing allocation before wiring the replacement.

24. Count transferred bytes separately from transfer batches. With one N
    tile, full-K A residency merges disjoint K transfers without reducing
    their bytes; its earlier loads can also change overlap. Price the whole
    kernel and isolate the tile-size change before attributing a gain.

25. Layer-major replay must carry implicit state as well as tensors. One
    complete teacher chunk preserves producer-mask ownership; multiple
    chunks need separate state. Report partial-layer execution separately
    from final teacher acceptance. Reuse checkpoint shard handles within
    one loading scope, then close them even if the consumer fails.

26. Recovering a fused norm from its saved residual requires the executed
    kernel to normalize that rounded residual. Bind its actual tiling key
    and compare saved outputs before reconstructing router inputs. Teacher
    routes still omit rejected speculative suffixes.

27. A bound on repeated weight reads is not a latency criterion. At the
    same one-eighth hot-expert fraction, wide GMM2 loses with16/32 active
    groups and wins with64. Check concentrated routes and complete release
    binaries before extending an adaptive tile condition.

28. Equal selected sets do not fix floating-point reduction order. Preserve
    ordered IDs when locating a cross-width first difference. A later query's
    changed cache cannot explain an earlier query through causal attention;
    separate those observations before tracing their effects.

Evidence: [GMM2 guard boundary](evidence/ascend-decode/gmm2-compact-20261020/guard-counts/README.md),
[teacher routes](evidence/ascend-decode/shared-fia-model-20261020/teacher-routes/README.md),
[full-A screen](evidence/ascend-decode/gmm-compact-20261020/full-a/README.md),
[paired cache](evidence/ascend-decode/index-cache-pair-20261020/README.md),
[adaptive GMM2](evidence/ascend-decode/gmm2-compact-20261020/adaptive/README.md),
[compact GMM1](evidence/ascend-decode/gmm-compact-20261020/README.md),
[fixed teacher](evidence/ascend-decode/shared-fia-model-20261020/README.md),
[merged tree](evidence/ascend-decode/merge-tree-20261020/README.md),
[lookup](evidence/ascend-decode/lookup-contract-20261019/README.md),
[SFA routing](evidence/ascend-decode/sfa-routing-20261019/README.md),
[pinned staging](evidence/ascend-decode/metadata-h2d-20261019/README.md),
[measurement workflow](workflow.md).
[Detailed historical lessons and their evidence](lesson-20261019-2340.md).

29. Local attention parity does not bound full-model drift. Same-source M32 shared-FIA reaches30/32 teacher labels, with final norm NRMSE0.15993 after78layers; keep it opt-in. At the old chain's first divergent prefix, SFA's2524/6500 margin is0.002617: capture logits and valid KV before attributing a token flip to state corruption. [Evidence](evidence/ascend-decode/shared-fia-model-20261020/layer-major/m32/full/README.md).

30. Removing a memory write must not silently remove a rounding boundary. Dense norm/quant fusion changes451/6144 INT8 codes on frozen inputs. Disabling it preserves eager values in actual ACLGraph/SuperKernel replay. Changing only the three dense fusion sites reproduces the2524→6500 token flip after78 layers on all16 ranks; preserve the BF16 boundary before measuring the corrected full request. Use fresh processes and caches: patterns register globally and the installed compiler hash omits this flag. [Evidence](evidence/ascend-decode/shared-fia-model-20261020/norm-quant-fusion/README.md).

31. A kernel's cost is not the round's cost. Measure the time each stream runs
    alone: every HCCL collective runs0.000ms alone while the TP reduction runs
    5.35ms alone, so one is free and the other is wholly critical path, and
    "no decoder gain claimed" reads as cheap when it only meant unmeasured.
    Normalise by the backbone's78 layers, not by an even division of the
    capture — that read22% low. Forcing FP32 on a reduction costs1.99ms a
    round in cast launches alone; folding the widen and narrow into a
    rank-ordered reducer is the same association, so it is bit-exact.
    [Evidence](evidence/ascend-decode/critical-path-20261020/README.md).

32. A decomposition has a floor before any kernel computes. Multiply the
    measured per-call floors by a round's actual call counts: 3008 calls give
    11.512 ms, and 2 ms a token at A=6.959 allows 13.92 ms. 83% of the budget
    is gone before arithmetic, so the target needs fewer launches, not faster
    kernels; 82% of that floor is 505 MIX_AIC calls, 6.5 a layer. The control
    sits in the same capture: `MlaPrologV3`, one fused operator over several
    projections, runs at 1.1x its declared bytes while every unfused
    neighbour runs at 3x to 91x. Price a grouped matmul against the declared
    weight bytes scaled by the activated union, and put `DT_INT4` in the byte
    table or the weights vanish from the floor entirely.
    [Evidence](evidence/ascend-decode/critical-path-20261020/README.md).

33. On this path a vendor operator costs 30x what its arithmetic needs, and
    the operator count costs less than that excess. The MoE epilogue is
    36.90 us as three vendor operators, 1.222 as three hand-written ones and
    0.517 fused -- so hand-writing is worth 2.7 ms a round and the fusion
    0.05. Keep both instruments: one launch a call measures what the host
    pays, many passes inside a launch measure what an in-graph replay pays,
    and the second is L2-resident so it is never a bandwidth figure. `Muls`
    on `dav-c220` rejects BF16, which is why the vendor's own costs 8.641 us
    to scale 98 KiB; compute in FP32 and `Cast` the roundings back where the
    unfused stages have them, or the result is more accurate and not
    comparable.
    [Evidence](evidence/ascend-decode/epilogue-fused-20261020/README.md).

34. A raw `<<<>>>` MIX kernel needs `__schedmode__`. Without batch
    scheduling the AIC and its two AIVs are not dispatched together, so a
    cross-core wait can be posted with nobody to answer it; a vendor probe
    measured 3 hangs in 49 runs on small shapes, and our first two runs hung.
    For the handoff after `Matmul` use what CANN's own example uses, mode 2
    and flag 8 from `PIPE_FIX`; mode 0 is the all-AIV grid sync and wants
    batchmode just as much. A3 allows flag ids 0 through 10 only, the Matmul
    library consumes some itself, and a flag's counter saturates at 15 sets --
    so a kernel persisting across 78 layers cannot give each layer its own
    flag. A3 also has no L0C to UB path, so cube output reaches the vector
    side through global memory, which is cheap only because it lands in L2.
    `rtGetC2cCtrlAddr` returns the sync base address but only once a device is
    bound. [Reference](reference/ascend-fusion-mechanics.md).

35. Learn the kernel debugger before the kernel needs it. Four blind
    bisections cost more than the whole toolchain survey did, and two of the
    three real defects came out of tools rather than reasoning. The
    instrumented compile rejects an attribute list the plain one accepts,
    which is how `[[bisheng::core_ratio(1, 2)]]` was found to be a second
    declaration of the ratio `KERNEL_TASK_TYPE_DEFAULT` already sets -- it
    hung every run. And `--cce-aicore-arch=dav-c220` drops `__schedmode__`
    with no warning while `--npu-arch=dav-2201` keeps it, which only the
    runtime log's `schemMode` field shows. `synccheck` reports at kernel exit,
    so it is silent on a hang; a stage bitmask on the kernel argument
    localises one in four runs. [debug.md](debug.md).

36. Check the intermediate, not the output. An INT32 product that verifies
    against a host reference every run while the output is wrong on a
    different set of rows each time says the producer is right and the
    consumer reads early -- and it took the product check to see that, because
    a matmul landing in the wrong place and a vector stage reading the wrong
    place are indistinguishable downstream. Two defects hid there: a channel
    range derived from `GetBlockIdx`, which does not mean the same thing on
    the two halves of a MIX kernel, and a scalar read of global memory from a
    vector core, which the mixed-core barrier does not cover. And the test
    that matters for a sync bug is whether removing the sync changes anything:
    the same error distribution with and without the flag proves it was never
    ordering. [Evidence](evidence/ascend-decode/mixbrick-20261020/README.md).

37. In a MIX kernel the two halves number the grid differently and the count
    does not convert between them: at four blocks the cube reports
    `GetBlockNum` 4 with `GetBlockIdx` 0-3, the vector side reports the same 4
    with `GetBlockIdx` 0-7 plus `GetSubBlockIdx`. Subblocks 2i and 2i+1 belong
    to cube i, and mode 2 lets a cube flag only those two -- so a vector core
    whose range came from `GetBlockIdx()` reads a slice another cube wrote and
    waits on the wrong producer, which no barrier can order. Measure the
    numbering before deriving anything from it; three rounds of barrier-adding
    went into a defect a four-line probe settled. The producer does need
    `DataSyncBarrier<MemDsbT::DDR>()` -- `PipeBarrier` orders a pipe's issue,
    not its retirement -- and on 2201 the DCache is irrelevant to a
    cube-to-vector handoff, which runs entirely through the shared L2.
    [Evidence](evidence/ascend-decode/mixbrick-20261020/README.md).

38. A hung single-die experiment blocks the whole machine, and the symptom
    names the die. `EI0015 Communication_Error_Ranktable_Detect ... List of
    unconnected ranks: "[1,]"` meant a kernel of ours had been squatting
    device 1 for seventeen minutes -- a launch past the physical core count,
    which hangs -- so rank 1 could never join HCCL. It cost two whole-machine
    arms on the first free machine of the day. Kill single-die work before
    arming a whole-machine queue, and read an unconnected-rank list as a
    device number rather than a network problem. Two scheduling rules follow:
    `chain.sh` exits on a failing arm, so an unproven arm goes last or it
    takes the queue with it -- a one-sided-reduce arm referencing a module the
    experiment tree predates killed fourteen arms behind it -- and an arm must
    be checked against the tree it will run in, not against the repo.

39. An experiment tree is a snapshot, and a fix in the repository is not a fix
    in the tree. The whole-machine pairing of `--plain-moe-combine` measured
    0.02 ms where the kernel figure predicted 0.21, because all four arms
    replayed one static-kernel package: static kernel packages are cached
    below cwd, the repository's `launch_arm.sh` already `cd`s into the arm's
    own output directory for exactly that reason, and the 10-19 tree's copy
    still `cd`s into the tree. One line apart. Before trusting a paired arm,
    diff the launcher and the engine the tree will actually run against the
    repository, the same way an arm's flags have to be checked against the
    tree rather than the repo.

40. A handshake that was never load-bearing was never tested. The shared
    expert's chain set a cross-core flag from the vector side to the cube, and
    it worked -- but the ordering came from the `SyncAll` that followed, so the
    flag's mode was free to be wrong. The routed experts made that direction
    carry a real dependency and the first launch hung with no output at all.
    A sweep settled it in two minutes:
    [reverse modes 0 and 1 are dead in every pattern, mode 2 admits exactly
    one, mode 3 admits all three](evidence/ascend-decode/handshake-20261020/README.md).
    When a mechanism moves from decoration to load-bearing, test it in
    isolation first -- one process a variant under a timeout, where finishing
    is the measurement.

41. The chip said no and the vendor's own source said how. `Mmad` takes INT4
    only with both operands INT4, A3 has no mixed-width mad, and a vector
    expansion of the weights is orders too slow -- so W4A8 looked shut. The
    answer was on disk: `opp/built-in/.../grouped_matmul_swiglu_quant/` ships
    `a8w4_msd_pre/mid/post`, which split the INT8 activation into two INT4
    halves and recombine 16 × high + low, two `mad_s4` a weight tile. Before
    concluding a quantisation scheme is impossible on a part, read the
    vendor's kernels for the op that already does it.

42. Hand-rolling past a library's refusal is the second-best answer. The
    `Matmul` library rejects INT4 because it plans to widen it to INT8
    entering L1; driving L1 and L0 directly proved the primitive and then
    sank into the micro-synchronisation of a reused L0 slot, where
    `PipeBarrier<PIPE_ALL>` gave right answers at K of 128 and 6144 and wrong
    ones between. The answer was to remove the reason for the refusal
    instead: a homogeneous INT4 pair needs no widening and compiles in every
    combination, and the activation's two INT4 halves stack along M so the
    weight is still read once. Ask what the library objects to before
    replacing the library.

43. A probe whose signal is ambiguous is worth nothing. Searching for where a
    misplaced INT4 half had landed returned a confident offset for several
    rounds -- and it was meaningless, because the test data was an arithmetic
    progression whose low nibbles are a sawtooth of period sixteen, matching
    itself at every aligned phase. A pseudo-random generator turned the same
    search into a −1 that could be acted on. Before trusting a probe, ask what
    else its answer is consistent with.

44. Change one variable. The INT4 write that would not land was confounded
    between its source tensor and its destination offset. Swapping the order
    of the two writes left the failure where it was, and sending the
    known-good source to the suspect offset failed too -- two runs, and the
    offset was named without a guess. The sanitizer did the same for two
    earlier defects, each fix halving the deviating count, which is how a real
    defect behaves.

45. `PipeBarrier` is the single-pipe primitive and `PIPE_ALL` is not a
    stronger version of it. It is absent from the headers' own
    `SUPPORTED_PIPE`, the Ascend checker flags it `AKA3001`, and crossing
    pipes is what `SetFlag`/`WaitFlag` are for. Every kernel in this
    repository leaned on it -- 66 sites across ten -- and the ones that pass
    do so on behaviour nobody specified, which is the class of defect a
    published repro saw vanish on a driver upgrade rather than on a fix. Name
    the hazard: `MTE2_V` after a load, `V_MTE3` before a store, `MTE3_V`
    before overwriting what a store is reading. And on the cube half of a
    `__mix__` kernel a vector-event `WaitFlag` compiles to nothing while its
    `SetFlag` still fires, so a helper that names a `HardEvent` must only be
    called from the half that owns it.

46. A store can be issued on a core that cannot have produced its data. Every
    `CastImpl` opens with `if ASCEND_IS_AIV`, so on the cube half of a
    `__mix__` kernel a cast is silently nothing -- but plain `DataCopy` to
    global memory carries no `ASCEND_IS_AIC` guard, while `DataCopyPad` does.
    The store goes out anyway, with a UB pointer, on a core with no UB, and
    two writers race on one address with nothing reported. Six rounds of
    elimination -- the arithmetic, the write mechanism, the declared length,
    the cross-pipe events, where the base is computed, the alignment -- and
    the cause was a missing guard in the library, not anything in the
    addressing. Use `DataCopyPad` for every UB-to-global store in code both
    halves can reach, and audit the rest for the same shape: it is silent on
    both sides.
