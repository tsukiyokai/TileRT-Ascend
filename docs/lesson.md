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

46. State a count in the unit the call means, for a type whose `sizeof` lies.
    Six rounds of elimination on an INT4 store that landed at offset zero and
    nowhere else -- the arithmetic, the write mechanism, the declared buffer
    length, the cross-pipe events, where the base was computed, the alignment
    -- and what fixed it was `DataCopyPad` with an explicit byte count in
    place of `DataCopy` with an element count. `int4b_t` has a `sizeof` of one
    and four bits of data, so every length in a sub-byte path is a two-times
    trap: a store of 2048 elements writing 2048 bytes overruns the next tile
    and spills 1024 bytes past the region, which is exactly the duplicate that
    appeared 6144 bytes in and again 6144 further on. Give such calls bytes,
    not elements, and read back the raw bytes at both candidate offsets before
    believing any of it -- the readback was the one link this elimination list
    never checked.

47. Never read TPOT without its two factors and the output tokens.
    `sparse_block_size = 16` lowered TPOT from 10.93 to 4.59 ms and was a
    catastrophe: its round went from 56.9 to 116.0 ms while the model
    collapsed into emitting token zero forever, which the prompt-lookup
    drafter guesses perfectly -- 25.6 of 32 drafts accepted against a normal
    5.27. A metric that is a ratio can be improved by destroying its
    denominator's meaning. Report round time, accepted a round, and the first
    tokens of the output, every time. And read acceptance from the engine's
    own round records: the gaps between delivered tokens are one a token
    however many a round accepted, which reads as a flat 1.001 and is a
    misread of the harness rather than a measurement.

48. Check which of your own artifacts already knows the thing before
    announcing that none of them do. The checkpoint is asymmetric W4A8: every
    expert projection carries a `weight_offset` and a nonzero `scale_bias`,
    and `layers.3...gate_proj.scale_bias` is F32 `[2048, 1]` with no zero in
    the first sixteen values. I read that off the checkpoint on the CPU, found
    no bias input in the INT8 kernel, and wrote down that the kernel and my
    host reference shared the omission -- which was wrong twice.
    `reference_weights.py` documents the convention exactly, "checkpoint
    correction added once after 16H+L", and applies it before the activation
    scale; and the INT4 kernel takes `gu_bias` and `down_bias` and loads them.
    Only the INT8 kernel lacks the term. So the real finding is smaller and
    more useful than the one I nearly filed: the integration binding was
    wired to the one kernel of the three that cannot serve this checkpoint,
    and moving it to the INT4 kernel also retires the memory blocker, since
    packed codes need no widening and so no 302 MB a layer.

49. The packing axis is not the packing convention. Both sides pair two INT4
    codes in a byte, low nibble first, and `_pack_to_int32` is only a
    `view(torch.int32)` over that -- which I read as the layouts agreeing and
    wrote down that nothing needed rearranging. They disagree about which axis
    the pair runs along. The checkpoint pairs along the output axis:
    `w13_weight (256, 128, 6144)` is `[expert][N/2][K]`, and its scale's 256
    entries are what settle it -- 128 is N halved and 6144 is K, not the other
    reading. The kernel pairs along K, two neighbours of a row to a byte. Same
    nibble order, same byte, orthogonal axis, and no error anywhere until the
    vendor operator asked for `[E][K][N]` and named the sizes. Two tensors can
    agree on every convention you thought to check and still not be the same
    tensor; when a shape is `(256, 128, 6144)` and the halved axis could be
    either, the scale's length is the witness.

50. Experts can be tensor-sharded rather than distributed. Each rank here
    holds all 256 of them and a sixteenth of each one's intermediate axis --
    the checkpoint's per-expert `gate_proj.weight` is `(1024, 6144)` packed
    along the output axis, so that projection's N is 2048, and the engine's
    `w13_weight (256, 128, 6144)` is two projections at 64 bytes each, 2048/16
    apiece. I had read it as 256 experts over 16 ranks, sixteen to a rank,
    which the arithmetic permits and the data does not. The cost of the wrong
    reading is a factor of two on `down_proj`, whose K is the intermediate
    itself: a rank's share is 2048/16 outputs, not 1024/16 bytes, and nothing
    objects to the narrower slice except the engine's own declared shape.

51. One error code, several causes, and the log was on disk the whole time.
    `aclshmem` returns -4 from `setup_heap` for at least three unrelated
    reasons, and the module's own docstring records one of them -- EINTR
    misclassified in the bootstrap handshake -- which is not the one in front
    of me, since `EINTR` is already in the SDK's reconnect set. I blocked
    signals on the strength of that comment, then bound the thread's device,
    then moved the bind earlier, three fixes in a row that each left the log
    byte-identical. `SHMEM_LOG_LEVEL` turns on the SDK's own trace under
    `/root/shmem/log/`, and it names the failing call, the value it was given
    and the line it came from. A returned code tells you something failed; it
    is not a diagnosis, and treating it as one costs a launch each time.

52. The experiment trees are snapshots, and `bench_cann_mtp.py` moves with
    `cann_model.py`. Shipping the repository's bench into a tree whose model
    file is older fails at `CannModel.__init__() got an unexpected keyword
    argument 'plain_moe_combine'`, and the failure is shaped to mislead: every
    rank raises the same TypeError, but rank 0 writes no record where the
    others do, so it reads as a rank-0 problem -- and rank 0 is exactly the
    one that builds the drafter. The remedy is to find a tree whose
    `cann_model.py` has the same md5 and take its bench, then patch that in
    place. Minimal patches to the tree's own files, never whole-file
    replacement from the repository.

53. The round is 55x its own memory floor, and two thirds of it is neither
    kernel. One decode round moves 1618 MB a die -- 1249 MB of W4 weights at
    TP16 plus a 368 MB unshardable MLA latent at 8K -- which is 1.011 ms at
    1600 GB/s against a measured 55.424 ms. The attention kernel at 89.4 µs a
    layer and the vendor's grouped matmuls at 157.2 µs account for 19.2 ms of
    it across seventy-eight layers; 36.2 ms is something else. So the target
    is not against physics, it is 14.8x the floor, and the levers that matter
    are not the two kernels: replacing the MoE with my fused one takes 19.2 ms
    to 9.9 ms, nine milliseconds off fifty-five. Compute the floor before
    choosing what to optimise.

54. Fit the round against the width before choosing a lever. Four widths at
    one drafter give `round = 45.88 ms + 0.814 ms a row`, residuals a few
    hundred microseconds. The fixed term is seventy-eight layers of issue and
    reduction and does not care how many rows are verified, and it alone
    decides what is reachable: granting perfect acceptance, it floors TPOT at
    4.637 ms at width 12 and 2.248 ms at width 32, so 2.x ms is unreachable
    without moving it however good the drafter becomes. It also explains the
    width optimum -- 12 beats 16 and 32 because a large fixed cost wants a
    small width -- and therefore dates the tree: the tree needs budget, budget
    needs a large width, and a large width needs the fixed cost gone first.

55. Fuse along the dataflow, not along the file. I absorbed the RMSNorm and
    the per-token quantisation into the expert kernel because they sit next to
    it in the source and because the fixed cost wants fewer operators a layer.
    Both are upstream of the router, not of the experts:
    `token_dispatcher.py` takes `dynamic_scale` from
    `token_dispatch_input.routing.pertoken_scale` and `npu_moe_init_routing`
    returns the sorted hidden states already INT8 with their scales permuted
    alongside. So the norm feeds the gate -- the router cannot run before it --
    and quantising inside the expert kernel would either recompute the same
    quantum eight times a token or force the dispatch to carry BF16 at twice
    the bytes. The kernel's original signature was right. What the expert
    kernel can absorb is downstream of the routing: the gather the dispatch
    materialises, and the weighted unpermuted sum the combine performs. Read
    the caller's dataflow before deciding what is adjacent.

56. Debug at the smallest configuration that still reproduces, not at the
    production one. Eleven sixteen-rank arms returned `aclshmem init returned
    -4` and nothing else, across four sessions. A two-rank harness with no
    engine found the cause on its first run -- a `TypeError` in my own device
    binding, printed plainly -- and a sixteen-rank one then came up correctly
    while another tenant held two of the dies, which settled that neither the
    SDK nor the rank count was ever at fault. An arm costs twenty minutes,
    buries tracebacks in `launch.log`, and collapses several causes into one
    returned code; the harness costs a minute and prints them. The rule is not
    "prefer small experiments" but something sharper: a configuration that
    cannot show you the error is not a place to look for it.

57. Pick the ratio that binds. I folded the shared expert in as sixteen
    one-row matmuls over one weight bank and argued it was affordable because
    `aic_mac_ratio` is 0.109 -- the cube is nearly idle. It cost double,
    71.625 us against 37.193, because the kernel is not cube bound: its
    `aic_mte2_ratio` is 0.733, and sixteen matmuls over one bank read that
    bank sixteen times. Both numbers came from the same profile I had already
    written down. Doing it properly needs one M=16 matmul, which needs the
    rows contiguous and the work split along N across blocks, which needs
    grid-wide barriers -- so the measurement settles the megakernel's
    foundation: mixbrick's grid-barrier structure carrying routed_int4's MSD
    path, because per-pair flags do not stretch across an N-split.

58. A microbenchmark with nothing in it measures the runtime's fast path out.
    An empty kernel taking no argument and touching no memory launched in
    64.654 µs; the same kernel given one argument and one store launched in
    3.384. The first number contradicted something I already knew -- a fused
    kernel of mine costs 10.2 µs a call including its launch -- and that
    contradiction is what caught it. It also overturned my own attribution:
    at 3.4 µs a launch, a layer's operators over seventy-eight layers cost 6.6
    ms of host issue, not the 36 ms residual I had assigned to "issue and
    synchronisation". The cost is in the drain, which grows with the launch
    count like a queue rather than sitting still like a fixed cost.

59. On 187 a container is destroyed a few minutes after it starts, and
    `docker events` is the only thing that says so. Three hours of its event
    stream show four `destroy` actions: my `trt-asc` twice and the other
    tenant's `505bvl_omni_c_c0` and `505bvl_omni_d_d0` once each, so the
    reaper is indiscriminate rather than aimed at me. The arm launched at
    22:29:29 was killed and destroyed at 22:33:14 -- three minutes
    forty-five. `docker ps -a` shows nothing afterwards, which is why every
    earlier instance looked like a mystery: a destroyed container leaves no
    exit code to read. a3-84's container runs for days, so the reaper is that
    host's and not the fleet's.
    It reattributes four failures I had explained otherwise -- an NFS prefetch
    storm, host memory exhaustion, and my own `docker rm -f` inside a
    diagnostic (that one was real) -- and it fixes the division of labour:
    whole-machine arms belong on a3-84, and 187 is for single-die probes that
    finish inside a minute. Every single-die measurement today ran there
    without once hitting it.

60. Measure the operator the engine calls, not the one you would write. I
    priced the gating as a torch topk plus a hand-written sigmoid and
    normalisation, 71 µs a layer, and planned to fuse it. The engine does not
    do that: `router_factory.py` selects `npu_moe_gating_top_k` and
    `device_op.py:1228` calls it, one operator for the sigmoid, the group
    selection, the top k, the renormalisation and the scaling factor, at
    20.798 µs. So my routing accounting over-counted by 3.56 ms and the lever
    I was about to build was mostly already there. This is the fourth
    attribution of mine that measurement overturned today, and the four share
    a shape: I measured what I assumed the engine does. Read the call site
    first.

61. Find the outermost opaque boundary before deciding what to fuse. I chose
    the quant method for the integration because it looked like the smallest
    change -- and it sits between the permutation and the combine, so it can
    deliver the grouped matmuls and nothing else. The boundary that holds the
    whole block is one custom op: `ascend_moe_forward_shared_sp` takes the
    unpermuted hidden states and the router logits and returns the shared and
    routed outputs, and `AscendMoERunner` keeps it in `_forward_entry`, an
    instance attribute, so substituting it is one assignment a layer and
    brings the gating, the permutation, both sets of experts, the combine and
    the shared path's reduce-scatter inside. Fusion is worth what the boundary
    encloses, which is this project's whole argument; I nearly violated it in
    my own integration design. This constrains design where lesson 60
    constrains measurement: read the call chain outward before pricing a
    saving, not just before measuring one.

62. Separate the variables before explaining anything. An aicore exception at
    ninety-six experts cost ten hypotheses -- slot indices, an over-read group
    list, scratch addresses, a machine difference, a self-inconsistent probe,
    the indexing path, a flush buffer, the allocation size, the block count,
    event ids -- and every one was disproven by an experiment that took a
    minute. The experiment that settled it changed one variable with the rest
    held: eighty experts at five rounds a block passed and the same eighty at
    six rounds faulted, so the limit is the rounds and not the experts. Nine
    of the ten guesses were arithmetic about how many bytes some buffer was
    short; none of that arithmetic was checkable in a kernel, which is why
    each could be made to sound right. The rule: when a threshold is stable,
    find the one quantity that tracks it by holding the others, and do that
    before the first explanation rather than after the tenth.

63. The comment in the file was about the half I did not apply it to. `TPipe`
    hands out buffers cumulatively, and this kernel already said so -- in a
    comment written after that mistake had cost a debugging session on the
    vector workspace. The cube half constructs a `Matmul` and calls
    `Init(&cube, &pipe)` on the same pipe, twice a round, and the sixth round
    is where ten instantiations become twelve and it runs out: "the GM address
    accessed by scalar exceeds 48 bits", at any expert count, any block count,
    on both machines, and unmoved by replacing the cross-core flags with grid
    barriers. Hoisting the two matmuls to one construction a launch lifts the
    limit and the engine's own 256 experts over sixteen rounds then runs. Ten
    hypotheses preceded it, nine of them arithmetic about buffer sizes. When a
    file carries a hard-won comment about a resource, check every consumer of
    that resource before looking anywhere else.

64. Count the barriers, not the arithmetic. At the engine's 256 experts the
    kernel spent 214.5 µs where the vendor spends 157, and the sparse routing
    was slower than the dense one -- an expert with no rows skipped its
    matmuls and still arrived at all four of its round's barriers, sixty-four
    of them over sixteen rounds. Since the scratch is sized per expert, every
    intermediate can be live at once, so the loops inverted: stage outside,
    expert inside, four barriers a launch at any expert count. 214.5 to
    79.206 µs, and 1.98x the vendor rather than 0.73x. A fused kernel's cost
    is its synchronisation before it is its arithmetic, and the sparse-slower-
    than-dense inversion is the signature.

65. `down_proj.scale_bias` is `[N][world]`, not `[N][1]`. The down
    projection's K axis is the intermediate, which TP divides, so each rank
    computes a partial sum and must add only its own column of the
    correction -- adding the whole thing on every rank adds it sixteen times.
    Its sibling `weight_scale` is `[N][1]` because it indexes the output
    channel, which TP does not divide, and `gate_proj`'s pair are `[N][1]`
    over the full intermediate so a rank takes rows there. Three shapes, one
    of them not what it looks like, and `w4a8.py` says so in passing:
    `torch.empty(num_experts, hidden_sizes, 16 // self.tp_size)`. A single-die
    probe with random weights cannot see this; it needed the real checkpoint
    and the real TP.

66. A criterion that cannot fail for the right reason validates nothing.
    Every kernel check today reported "finite, nonzero, the right rows
    written", and the configuration they all ran in -- random weights with
    random scales -- produced an absmax of 2.4e-17, which satisfies all three
    and is arithmetically dead. Varying the weights and scales separately
    showed only one healthy combination, so the failure is dynamic range: the
    middle quantisation normalises by a per-row max and then three scales
    around 1e-4 multiply, against bfloat16's smallest normal of 1.2e-38. The
    vendor avoids it by dequantising inside L0C through the fixpipe, a
    mechanism this project had filed as an optimisation and which is a
    correctness prerequisite. Timing measurements survive; the correctness
    claims behind them do not, and they are withdrawn.

67. Two criteria disagreeing localises what one criterion never could. The
    kernel matched the vendor under a constant weight to a bfloat16 ULP and
    produces nothing at all under random weights -- 98304 of 98304 elements
    unequal, the vendor at absmax 26.25 and the kernel at zero. A constant
    weight makes every column's product identical, so column offsets, K-axis
    mismatches and index displacements leave it unchanged: that mode cannot
    see structural error, which is the second criterion today that could not
    fail for the reason it was being trusted for. The pair together points at
    misalignment rather than arithmetic. Prefer the vendor to a hand-written
    reference: mine disagreed with the kernel about whether the scale bias
    goes inside or outside the scale, so it could not settle which side was
    wrong, while the vendor needs none of my checkpoint conventions to hold.

68. Bisect before hypothesising, and count the defects before explaining them.
    Two archived kernel builds against one judge: a04e6ce gives 0.47461 where
    the vendor gives 0.45117, ba67cf3 gives 0.00000. So there are two
    independent defects -- a structural one the stage-major restructuring
    introduced, and a five-percent numerical one that predates it -- and every
    one of the ten failed hypotheses had been trying to explain both at once
    with a single mechanism. The commit that produced the session's best
    kernel timing, 79.206 us against the vendor's 157.176, is the one whose
    output is absent; the timing stands and the saving it implied does not.

69. The cube core cannot read unified buffer, and the failure is silent. The
    stage-major kernel staged its group counts in a `TPosition::VECCALC`
    buffer for the cube half to read: on dav-2201 the cube core's local memory
    is L1 and the L0s, every `GetValue` returned zero, every expert was
    skipped and no matmul ran. The vector half reads the same counts from
    unified buffer correctly, so the split stage produced a plausible result
    and only the cube stage did nothing -- an absent product with healthy
    inputs, which is indistinguishable from an arithmetic fault by inspection.
    A scalar read from global memory fixes it and costs 256 loads a launch
    against the matmul each one guards.

70. A broken correctness criterion corrupts the timings too. With the cube
    half skipping every expert, this kernel measured 79.206 us at 256 experts
    and was recorded as 1.98x the vendor. Running its matmuls, it measures
    365.887 us against the vendor's 157.176 -- 2.33x slower, and the 5.8 ms a
    round it promised becomes about 16.3 ms a round worse. The fastest
    possible kernel is one that does nothing, so a speed number taken beside
    an unfalsifiable correctness check is not a measurement of the work. The
    vendor's advantage is structural: fractal-NZ weights dequantised inside
    L0C through the fixpipe, against ND weights dequantised on the vector side
    with a unified-buffer round trip per expert.

71. Price a lever at the shape it will run at. The permutation pair was
    budgeted from 45.772 + 55.979 us and is 22.045 + 31.181 at the decode
    shape -- twelve tokens, top eight, 96 sorted rows -- so the saving ceiling
    falls from 7.1 ms a round to 4.15. A measurement taken at one shape and
    carried into another budget inflates whatever it is subtracted from. The
    plainest replacement is 2.93x the vendor, with `index_add_` alone at
    131.957 us against the vendor's whole unpermute at 31.181, so the ceiling
    is not currently reachable either.

72. The MoE matmul is fixed cost. Sixteen rows cost 146.431 us and 256 cost
    150.565 -- sixteen times the work for 2.8 percent more time -- so under
    three percent of the round's 12.3 ms MoE line item is arithmetic. The
    premise that it is computation, reducible by a faster matmul, is wrong:
    a faster matmul buys three percent of it, which makes the hand-written
    kernel's 2.33x disadvantage beside the point and its value purely as a
    vehicle for covering several layers in one launch. The round model's
    45.88 ms fixed term is the sum of per-layer taxes paid once each because
    each is a separate launch, and that is the only exit the measurements
    leave open.
