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

73. A vendor baseline is a property of the process. The same grouped matmul at
    the same shape costs 61.617 us in a clean process and 153.325 us after a
    communication domain has been built and a custom kernel has run -- 2.5x,
    with two clocks agreeing within one percent. The engine is the mixed case,
    so the round's recorded 157.2 us a layer is right, but any clean-probe
    baseline understates what the engine pays and any mixed-probe baseline
    overstates what a clean kernel would. Compare only within one state, and
    record which state it was.

74. The round's fixed term is a property of the process, not of the operators.
    Measured in one clean process at the decode shape, the MoE matmul, both
    halves of the routing, the norm with its quantisation and a local
    reduction sum to 15.38 ms a round, against about 40 ms for the same items
    in the engine's accounting -- and the MoE matmul alone has been measured at
    61.6 us clean against 153.3 mixed. So the 45.88 ms fixed term that makes
    2.x ms look unreachable is not a floor; it is the cost of whatever state
    the engine is in. A clean-state round would be near 18 ms and a TPOT near
    3.3 at the acceptance already achieved. Six candidate causes are ruled out:
    loading the kernel library, running the kernel, building a communication
    domain, a second resident NZ weight bank, the full two-stage vendor chain,
    and constant versus random codes.

75. Compare in one process, with the arguments already built. The same two
    implementations at the same shape gave four different answers today: 1.98x
    faster (nothing ran), 2.33x slower (a different expert count), 25.7 percent
    faster (clean process against mixed), and 16.6x slower (seven tensors
    allocated inside the timed call). The surviving number is 67.336 us for
    the vendor against 72.933 for the kernel -- 8.3 percent behind, while the
    kernel still gives away five percent of numerical accuracy, fractal-NZ
    weights and the fixpipe. The discipline that caught the last two was
    refusing a cross-process comparison, which is worth more than any of the
    four numbers.

76. Fusing layers means interchanging the loops, not nesting them. Wrapping the
    whole five-stage body in a layer loop made eight layers cost 6.15x one
    launch, because the four grid barriers a layer become 4N. The saving was 23
    percent -- the launch and the entry. Stage-major across layers takes the
    barriers to four for the whole group, which is the only form of cross-layer
    fusion the measurements support.

77. Cross-layer fusion is capped at 23 percent, not an order of magnitude.
    Nesting eight layers in one launch costs 6.15x one launch while multiplying
    the grid barriers by eight: if barriers dominated, the scaling would exceed
    linear, and it is below. So 77 percent of a layer's cost is the per-layer
    work itself, accumulating linearly, and only the launch and the entry can
    be shared. The loop interchange that should have taken the barriers from 4N
    to four measured about N squared instead, because the layers crowded one
    reused scratch -- the second configuration today that could not answer the
    question it was built for. The honest ceiling for one launch over many
    layers is the 23 percent nesting already collects.

78. An optimisation overlay left switched on became the tax it was meant to
    remove. Every engine arm prepends a custom operator overlay to
    `ASCEND_CUSTOM_OPP_PATH`, and that overlay carries its own
    `grouped_matmul_swiglu_quant_v2`. At the decode shape the same call costs
    70.450 us without it and 100.604 with -- 43 percent, about 2.3 ms a round,
    collectable by not setting one variable. It was built to make that operator
    faster and has been active in every engine measurement since, including the
    ones that set the round's budget.

79. CANN's own operator was the fastest all along. The same grouped matmul at
    the decode shape costs 66.813 us with no custom operator path, 97.362 with
    the owner overlay, 123.985 with vllm-ascend's own package and 104.929 with
    both -- and every engine process sets both while every arm prepends a
    third. That is the whole of the 61.6 against 153.3 us state gap, chased
    through nine innocent candidates because none of them touched the operator
    path: a single-die probe does not source activate.sh and the engine does.
    About 3.0 ms a round, to be collected by letting this one operator fall
    back rather than by unsetting the variable, since the packages exist for
    other operators: measured since, the package is 23 percent faster on the
    norm with its quantisation and 22 on the gating, so a wholesale unset nets
    1.4 ms rather than 3.0. There is no systematic conclusion here, only a
    per-operator fact -- and trimming the grouped matmul out of a copy of the
    package does not deregister it, leaving 116.376 us and dragging the other
    two operators between states.

80. [Corrected by 82.] The round is the kernel count times 28.8 us. A single elementwise multiply
    over twelve elements costs 28.824 us on one die at this batch -- 8.5x the
    3.384 us an empty kernel suggested -- and a layer issues about 25 kernels.
    25 x 28.8 is 720 us against the round's measured 710 us a layer, so the
    round is launches and the arithmetic is noise. That is why every operator
    today measured 30 to 70 us whatever its shape, why sixteen times the rows
    moved the grouped matmul 2.8 percent, and what the 45.88 ms fixed term is.
    TPOT = kernels a layer x 28.8 us x 78 / accepted, which gives 10.3 ms at 25
    kernels and 5.44 accepted against the measured 9.38. Reaching 2.x ms needs
    about five kernels a layer. The megakernel is therefore necessary, for the
    third reason tried and the first that measures: not to beat the vendor's
    arithmetic, not to fuse across layers (capped at 23 percent), but to
    collapse 25 launches within a layer into a handful.

81. The unit of comparison is the launch. This kernel was measured three ways
    and withdrawn twice: 1.98x faster (nothing ran), 2.33x slower against one
    vendor operator at 256 experts, 8.3 percent behind at the decode shape.
    Priced against what it actually replaces -- four vendor launches against
    one -- it saves 8.48 ms a round, 181.172 us against 72.463. All the
    measurements were right and only one answers the question, and which one
    that is became clear only after the per-kernel floor was measured at
    28.824 us for a twelve-element multiply.

82. The per-kernel floor is eager dispatch, and the engine runs a graph. One
    twelve-element multiply costs 11.386 us a call eager and 1.064 us under
    captured replay, eleven times less. So lesson 80's model -- the round as
    launch count times a 28.8 us floor -- is wrong, and its agreement with the
    round's 710 us a layer was a coincidence of two wrong numbers. In a graph,
    25 kernels a layer cost 27 us, under four percent. The 8.48 ms the MoE
    kernel saves is real but its mechanism is the three HBM round trips the
    vendor chain makes for its intermediates, not the three launches: those are
    worth 0.25 ms. The case for one kernel over a chain is residency, which is
    the classical argument, reached after measuring the wrong quantity twice.

83. Engine-side optimisation reaches about 7 ms, and the last factor of three
    is drafting. Issue under graph replay is 1.064 us a kernel, so launches are
    four percent of a layer; the decode attention is already three boundaries;
    the routed experts in one kernel save 8.48 ms and the one-sided reduce 6.5,
    taking the round to about 40 ms. TPOT is then 40 over acceptance, so 2.9 ms
    needs 13.8 tokens accepted a round against today's 5.441. Corrected since:
    that is conditional on the edit rate, not impossible. At width 32 and
    budget 64 the tree accepts 14.542 at an edit rate of 0.02, 6.851 at 0.05
    and 3.890 at 0.10, and the measured 5.441 places the workload near 0.05.
    An agentic coding model's output is bimodal -- long near-verbatim copies
    punctuated by short rewrites -- so a single width tuned for the mean is
    wrong at both ends, and the drafter should widen inside a copy stretch and
    narrow inside a rewrite.

84. A proposal ceiling is not a quota. Sizing the automaton's proposal at twice
    the last round's acceptance keeps 83 percent of the acceptance for 39
    percent of the mean width, which is TPOT 6.75 against 7.18 at an edit rate
    of 0.02 and 12.02 against 13.24 at 0.05 -- because a round costs 0.814 ms a
    proposed row and rows that were never going to hold are paid for anyway.
    The first signal tried, the matched suffix length, measured worse at every
    rate: it is typically under nine on this corpus, so a threshold at eight
    never fired, and the automaton's advantage comes from the most frequent
    continuation of a short match rather than from a long match. Scoring any of
    this inside a loop whose columns share one advance locks the feedback at the
    narrowest width; an adaptive column needs its own trajectory.

85. A change that alters shared code rewrites its own baseline. The adaptive
    ceiling went straight into `propose`, so the fixed-width column it was
    being compared against fell from 5.393 to 2.896 and the comparison was
    against a contaminated control -- the third time today a measurement was
    taken against something the change had already moved. Behaviour that
    changes a default belongs behind a constructor argument, with the control
    pinned explicitly. The factor sweep that found this also settled it: two is
    best, 12.01 against 12.62, 12.54 and 12.97 at 1.5, 3 and 4.

86. Read the call site, not the operator's parameter list. Tree drafting was
    recorded as blocked on the attention kernel because `atten_mask` is absent
    from the sparse operator. But the engine calls it with
    `layout_query="TND"`, which has no batch axis, `sparse_indices` as a
    tensor indexed by token, and `sparse_block_size=1` -- so each query row
    already selects its own visible set, which is what a tree mask is. The
    blocker is host-side index construction, not the kernel. A probe written
    against `torch_npu.npu_sparse_flash_attention` also failed on a required
    `query_rope` that the `_C_ascend` operator the engine calls does not need:
    two implementations behind one name, and the call site is the
    specification.

87. Measure a tensor-parallel operator at its shard. Seven per-layer
    projections measured 364 us a layer unsharded and 109.447 at the TP16
    shard the engine runs -- o_proj alone fell from 178 us to 13.176, since its
    200 MB weight is split sixteen ways. At the shard each costs 12 to 19 us
    whatever its size, the router's 3 MB running at 158 GB/s where q_a's 24 MB
    reaches 1386, so the small ones are boundary-bound. Concatenating the four
    that read the hidden states along their output axis makes them one matmul:
    68.724 us to 37.238, 2.46 ms a round, relative difference 4.058e-04 on
    magnitudes near 308 where bfloat16's ULP is 1 to 2 -- reblocking, not
    approximation. And an absolute difference means nothing without the
    magnitude beside it: 0.125 was both the measured difference and what an
    error of one part in a hundred would have looked like.

88. A fusion pass that rarely matches may be protecting you. The cast census
    filed `norm_quant_fusion_pass` firing on 3.3 of 156 possible sites a round
    as a defect to investigate. Measured at the decode shape,
    `npu_add_rms_norm_quant` is 210.996 us against 43.532 for the two separate
    operators -- making the pass match more often would cost 13 ms a round. Its
    sibling `npu_add_rms_norm` is the opposite, 19.189 against 37.913, worth
    1.46 ms. Fusion is not a direction, it is a per-operator measurement.

89. The boundary-merging mechanism is exhausted at 20.3 ms a round. Five merges
    taken -- routed experts 8.48 ms, one-sided reduce 6.50, four input
    projections 2.46, residual-with-norm 1.46, CANN's own grouped matmul 1.40 --
    and four closed by measurement: the norm with its quantisation at 4.8x
    slower, o_proj with its residual add where addmm is slower than two calls,
    the attention chain where the vendor already fused everything collectable,
    and cross-layer nesting at 6.15x for eight layers. That leaves a 35.1 ms
    round of device time in tuned operators that cost 12 to 19 us each
    regardless of size, so the last factor of two must come from doing less a
    round rather than from merging more: fewer layers per token, or more tokens
    accepted per evaluation.

90. The checkpoint shipped a trained draft head and nobody loaded it.
    `num_hidden_layers: 78` with `num_nextn_predict_layers: 1`, and the weight
    index carries layers 0 to 78 -- 79 of them. Layer 78 is the MTP head. The
    engine passes `speculative_config=None` and refuses a drafter at
    `cann_model.py:263`, so every acceptance figure measured here comes from a
    host-side suffix automaton that can only repeat what it has seen, which is
    why acceptance falls from 10.020 to 1.824 as the edit rate goes from 0.02
    to 0.20. A trained head predicts instead of recalling. One extra layer a
    round is 1.3 percent, and acceptance of 15 against today's 9.288 is the
    difference between 4.9 ms and 2.4.

91. Read the whole chain of branches, not the one that mentions your name.
    Searching `hf_config_override` for `glm_moe_dsa` found it only in a boolean
    and a `glm5_next` branch that does not apply, which read as "no MTP
    architecture for this model". Two `if`s earlier the same function maps
    `glm_moe_dsa` to `deepseek_mtp` and then to `DeepseekV32MTPModel`. The
    checkpoint's draft head is fully supported upstream; what was missing was
    only the engine passing a speculative config.

92. Modelling a mechanism you have not read is a way to generate numbers that
    look like evidence. Prefixing the automaton's proposal with one
    probabilistically-correct token measured 1.597 to 1.948 accepted against
    the automaton's 4.902, which reads as "the pairing fails" -- but prefixing
    shifts the whole proposal by a position, so a wrong head token voids
    everything behind it. MTP drafts autoregressively against the target's own
    context and has no second drafter to invalidate. The probe measured a
    splice of my own invention.

93. Published measurements beat a guess, and they withdrew the day's largest
    lever. The checkpoint's MTP head was projected to lift acceptance to 15 and
    TPOT to 2.4 ms. Deployment figures and vLLM's own spec-decode counters put
    it at 78 to 85 percent acceptance and a mean accepted length near 1.85 at
    k=1, and 2.6 to 3.9 at k=3 to 5, with vLLM-Ascend documenting that
    DeepSeek's single MTP layer does not guarantee accuracy above k=1. The
    adaptive tree over the suffix automaton already measures 9.288. MTP is an
    order of magnitude behind it, because it predicts in general where the
    automaton recalls on a distribution that largely copies its input. The 15
    was mine and nothing published approaches it.

94. Wasted bandwidth can be the cheaper option. This project's own evidence
    showed the sparse attention operator reading the whole 10.3 MB cache and
    applying its 2048-position selection as a mask afterwards, and read that as
    a missed gather. Measured: the contiguous read runs at 380 GB/s and the
    scattered selection at 115, so a fifth of the bytes costs three quarters of
    the time, and adding the gather's own 16.332 us makes pre-gathering 35
    percent slower. The operator is right and the lever is closed.

95. The grouped matmul does not read by its group list. Eight busy experts cost
    112.850 us and 256 cost 111.524, a 32-fold range in bytes for a 33 percent
    spread with no ordering, so a narrower draft saves nothing in the MoE --
    removing a saving the adaptive width was assumed to carry. Compacting the
    70 touched experts into a contiguous bank makes the operator 32 us faster
    and costs 3688 us to build, 115 times the saving. With attention's cache
    this makes two operators today whose apparent waste is the cheaper choice,
    both of which this project had recorded as opportunities.

96. The round model's row cost is 31 times the operators'. A whole layer goes
    from 135.190 us at 8 tokens to 143.263 at 32 -- 0.026 ms a row over 78
    layers -- against the model's fitted 0.814 ms, and the fit is sound with
    residuals of a few hundred microseconds across a 19.3 ms span. The drafter
    is excluded at 0.1 to 0.3 us a round on the host. So the money is in the
    sixteen-rank reduction or the graph's static token buckets, and if it is the
    buckets then the constraint that shaped every drafting decision today is an
    artefact: a fixed-budget tree at 64 nodes and 14.510 accepted becomes a
    TPOT of 1.88 ms instead of being priced out. One number, worth more than
    every lever measured today together.

97. A cost model fitted across configurations can measure the trajectory
    instead of the configuration. `round = 45.88 + 0.814 a row` fits four
    widths with residuals of a few hundred microseconds, and all three
    mechanisms that could produce the row term are excluded: the engine
    captures one graph size so there are no buckets, a two-rank all-reduce is
    flat to negative in the row count, and a whole layer costs 0.336 us a row
    against the model's 10.4. What is left is that each width ran its own
    trajectory -- wider drafts accept more, advance further, grow the KV faster
    and pay more for attention later -- so the fitted slope is mostly context
    length. Compare at constant trajectory length, not constant round count.

98. Five mechanisms excluded and the slope still stands. `round = 45.88 + 0.814
    a row` survives: graph buckets (one captured size), the TP reduction (flat
    to negative over an eightfold byte range), per-row operator cost (0.336 us
    a row a layer against 10.4 claimed), trajectory length (the arms cap tokens
    not rounds), and the Gloo command broadcast (305 us at width 0, 289.6 at
    64). A fit with residuals of a few hundred microseconds across a 19.3 ms
    span is not noise, so something real is being measured and it has not been
    named. Record that as the state rather than picking the least-disproven
    story.

99. The round model's slope is the width's fixed cost, not the row's marginal
    cost, and the records already said so. Grouping each arm's own rounds by
    matched drafts: within auto12, zero matches cost 50.49 ms and six cost
    50.65; within auto16, 53.30 against 53.90. That is 0.03 to 0.10 ms a row,
    agreeing with the 0.026 measured on one die. Across arms the medians are
    48.2, 50.5 and 53.5 ms at widths 8, 12 and 16 -- 0.66 ms per unit of width,
    paid every round whatever the draft achieves, because the engine captures
    one graph at `graph_width` and a wider graph costs more to replay even with
    empty rows. Nine mechanisms were hypothesised and excluded before the
    grouping that settles it, and the data for that grouping was in the records
    the whole time.

100. [Narrowed below.] The width cost is in graph replay, not in any operator.
     Attention is flat in both picks and width -- 109.057 us at width 8 with 2048 picks against
     103.898 with 512, and 107.370 at width 16 -- so the per-row scoring
     hypothesis fails and `--index-topk` stays closed. Every operator measured
     today is flat in the rows, yet the engine pays 0.66 ms per unit of width
     every round, and a bare operator call does not include the graph. Also:
     dividing one record's `replays` by another's `rounds` gave exactly 6.0 and
     looked like six replays a round; it was three warmup plus three measured
     requests. Two scopes divided by each other, for the third time today.

101. A substitute operator's flatness is not the operator's. The attention
     measured flat in picks and width was
     `npu_fused_infer_attention_score_v2` on a gathered layout; the engine
     calls `npu_sparse_flash_attention` on a paged cache with a block table,
     whose tiling function rejected four attempts at an argument set. So
     attention is not excluded as the carrier of the 0.66 ms a width unit -- it
     is the only per-query-row operator in a layer and the only one left
     unmeasured, which makes it the leading candidate rather than a cleared
     one.

102. Find the dispatch layer before trying to remove something from it. Three
     attempts to deregister one operator from a custom CANN package failed in
     three different ways -- removing its TBE kernel directory, removing its
     `binary_info_config.json` entry, and localising its symbols with
     `objcopy`, which edits the static symbol table and not `.dynsym`. The
     operator is resolved as a dynamic symbol from
     `op_api/lib/libcust_opapi.so`, and withholding that library drops the
     grouped matmul from 104.9 to 70.147 us while the norm and gating keep
     their TBE kernels. Net 2.14 ms a round against 1.4 for unsetting the path
     wholesale. The third failure is what located the layer; the first two were
     guesses at it.

103. The width cost is capture, not the operators, and a synthetic graph shows
     it without the machine. Eight layers of projections and norms: eager goes
     568.7 to 554.4 us across a fourfold row range -- flat and slightly
     decreasing -- while replay goes 527.0 to 582.9, monotonically up, 2.3 us a
     row. Scaled to 78 layers that is 1.77 ms a row a round against the round
     model's 0.814, high by about twice in the direction an eight-layer
     extrapolation should be. So capture fixes shapes into the task list and a
     list built for more rows costs more to replay even where the operators are
     flat. Ten mechanisms were hypothesised for this before a graph was
     captured at two widths, which was always the direct test.

104. [Half of this is re-corrected by 122.] Varying two parameters at once
     produced both of today's drafting wins, and both withdraw. The tree's 14.510 accepted came from budget 64 with a width
     ceiling of 32; at equal budget the tree matches the automaton to within
     half a percent -- 9.178 against 9.221 at 16 -- because a suffix match is
     usually right and branching buys nothing on this corpus. The adaptive
     budget's win came from a mean width of 12.6 against a fixed 32; at equal
     budget it loses acceptance for nodes it cannot spend, 7.830 against 9.221,
     and with the width term now known to be per unit of *captured* width,
     which the configuration fixes, asking for fewer nodes saves nothing. The
     configuration to run is the plain automaton at a captured width near 24,
     TPOT 3.73 ms with the merges.

105. A bandwidth floor is not a target unless the operator only moves bytes.
     Sparse attention runs at 121 GB/s where a contiguous read of its cache
     reaches 449, which read as six milliseconds a round of waste -- exactly
     what separated the best projection from 2.x. But the operator also scores,
     softmaxes and weighs, and doing that densely in torch costs 95.271 us
     against the operator's 85.048: the vendor is 11 percent faster than a
     plain implementation of the same work. The 22.316 us read is the floor of
     reading, not of attending.

106. Check an item-by-item budget against the whole before trusting it. A
     complete layer runs in 304.965 us against 317.476 for the sum of its
     parts -- 3.9 percent of cross-operator overlap, not a factor -- and
     317.476 over 78 layers is 24.76 ms against the 24.84 the fixed term
     becomes after the merges, agreeing to a third of a percent. So the round
     has no unattributed remainder, and the statement that 2.x ms is outside
     the identified mechanisms stops being an inference and becomes an
     accounted fact: attention is 30.6 percent of a layer and near its floor,
     the grouped matmul 19.5 and already the fastest of four implementations,
     and the remaining six items are operator fixed cost on twelve rows, which
     is what the boundary merges already collected.

107. The item-by-item layer account was eager and the engine runs a graph.
     Summing a layer's parts gave 317.476 us and matched the round model's
     post-merge fixed term over 78 layers to a third of a percent, which read
     as "no unattributed remainder" and put 2.x outside reach. But the same
     layer replays in 213.913 us -- 5.6 percent above its profiled device-busy
     202.514 -- so capture removes the 32 percent eager spends on issue. At
     214 us a layer the post-merge round is 16.7 ms and TPOT at width 24 is
     2.98. An agreement between two numbers is not a closed account when both
     were measured in the wrong execution mode.

108. The engine's round is a SuperKernel plus collectives, and its device is
     saturated. From the project's own capture: 57.27 ms of device-busy a round
     against a measured round near 52.8 -- the halves run concurrently so
     durations add past the wall clock -- of which 56 percent is one
     SuperKernel and 31 percent is HCCL. So the 104 us a layer that graph
     capture saves on a synthetic eager layer is not available in the engine,
     and that correction withdraws. It also relocates the largest lever:
     collectives are 17.56 ms of device time where the one-sided reduce was
     costed at 6.5 from an operator comparison, so the measurement outstanding
     that matters most is the one already built.

109. Count the calls before pricing a replacement for them. The one-sided
     reduce was costed at 6.5 ms a round by multiplying an operator-level
     saving, 201.995 us against 32.488, by the 156 calls a TP16 decode was
     assumed to make. The capture says 41 calls a round at 831 us each -- a
     tenth the count at four times the cost -- so the arithmetic was wrong
     twice and landed in the right order of magnitude by cancellation. Forty-one
     collectives at 830 us looked like 35.1 ms of device time -- wrong, because
     grouping by the type column averaged the one-off domain build into the
     per-round figure. Of 128 collective records 64 are a repeated AivKernel at
     858.4 us and the rest appear once each at 2126 to 9343 us. The decode
     figure is 20.4 calls a round at 858 us, 17.5 ms, which is what an earlier
     by-type sum had already given. Deduplicate by name before averaging by
     type.

110. The super kernel does not reduce the kernel count. Deduplicated by name,
     the engine issues 610.9 SuperKernel launches a round over 78 layers --
     7.8 a layer at 52.7 us each -- where a synthetic eight-operator layer
     replays in 214 us on one die, 26.8 us an operator. Same count, twice the
     cost each, and the factor of two is the shard: at TP16 a kernel does a
     sixteenth of the work and still pays its fixed cost. 32.21 ms of kernels
     plus 17.56 of collectives is 87 percent of the round, which is exactly
     what the boundary merges and the one-sided push address.

111. Repeating one operator measures it out of cache. A 72 MB weight matmul
     costs 31.056 us when one weight is repeated -- 2264 GB/s, above what HBM
     gives -- and 62 us from four distinct weights onward, 1136 GB/s. Every
     single-operator timing today repeated one weight: the projections at 12 to
     19 us, the grouped matmul at 62 to 70, attention at 85, a layer at 214 in a
     graph. The engine's 52.7 us a kernel sits nearer the cold figure, as a
     model reading a different weight every layer should. The comparisons
     survive because both sides were equally warm. And the first guess at the
     direction was wrong: measured cold over eight rotating weight sets, the
     fused projections save 3.59 ms a round against the warm 2.46, because
     fusing turns four separate weight pulls into one contiguous pull and that
     is worth more from memory than from cache. So warm figures understate the
     merges. What stands is that warm absolutes cannot be summed into a round --
     the engine's 57.27 ms of device time is the denominator -- and every lever
     needs its cold figure.

112. A profiler's wait column is queueing, not a share of the kernel. Cast
     shows 21.47 ms of wait against 1.40 ms of duration -- 1532 percent -- so
     the column is time spent queued before the kernel starts and cannot split
     a collective into transfer and synchronisation. It still answers a
     question: the collectives wait 13 and 18 percent of their own length
     against Cast's 1532, so their 17.56 ms a round is largely execution, and
     the one-sided push is worth 8.8 to 14.7 ms depending on whether its 6.2x
     advantage survives the engine's message size -- a range only sixteen ranks
     can narrow.

113. The one-sided push is a latency optimisation, so batching defeats it. Two
     ranks, float32: it beats HCCL 6.90x at 24 KB, 9.91x at the decode shape's
     288 KB, 2.17x at 1.5 MB and loses at 6 MB, because HCCL spends 189 to 258
     us on rendezvous regardless of size and the push skips it. The engine
     makes 20.5 collective calls a round at 858 us where a layer-wise reduction
     would make 156, so it already batches -- into the regime where the push
     loses. The lever is therefore not "replace HCCL" but "unbatch and push",
     156 pushes at 26 us being 4.1 ms a round against 17.56, and the two are
     alternatives rather than complements.

114. An absorbed reduction costs 30 us where a standalone one costs 858. Six
     super kernels a round declare an all-reduce in their name and cost about
     30 us each; the twenty standalone collectives cost 858. So the lever on
     the round's 17.56 ms of collectives is absorption, not a faster primitive:
     the two-rank scaling shows the one-sided push is a latency optimisation
     that loses past a megabyte, and 858 us a call is well past that. Absorbing
     the remaining twenty the way six already are is worth about 17 ms a round
     and the mechanism is already in the compiler -- so the question is what
     condition decides which form it emits, which is answerable by reading.

115. Check the configuration before quoting numbers from two experiments. The
     absorption lever took its call count from a capture whose round is 52.8 ms
     and met its refutation in an ablation whose round is 16.6 -- the same day
     on the same machine, three times apart -- and a 16.6 ms round cannot hold
     the 17.56 ms of collectives the lever is built on. The within-capture
     ratio survives: absorbed 30 us against standalone 858, measured side by
     side. The 17 ms does not, because it multiplies a per-call difference from
     one configuration by a count from another.

116. Read the source run's configuration before deriving anything from its
     capture. `profile-rope-20261018` records `max_tokens: 6`, `mtp_width: 4`
     and no `--super-kernel`: a six-token profiling run at width four, not
     decode at steady state. Today's whole device-time account came from it --
     SuperKernel 32.21 ms a round, collectives 17.56, the 20.4 call count, the
     30 us against 858 absorbed comparison -- and all of it is void, because
     the round count was obtained by dividing 244 attention calls by 78 layers
     on the assumption that every call was decode, where six output tokens at
     an acceptance near three give about two rounds and the rest is warmup and
     prefill. This was the first thing to check and it was checked last.

117. This project has never profiled steady-state decode. All three device
     captures -- profile-rope, profile-compiled, profile-stacks, all
     20261018 -- record `max_tokens: 6` and `mtp_width: 4`, while every
     decision is taken against a width-12 round of 55.424 ms. So the round's
     item-by-item table rests on single-die operator measurements, most of them
     warm, and nothing device-side has ever been measured at the configuration
     that matters. The first thing to do with a free machine is that capture,
     before any lever is re-measured.

118. A truncated backbone is four of our own literals away from running. Both
     reachable machines leave 6.8 and 3.8 GiB against the model's 16.7, so an
     eight-layer arm was tried: the gate passed and it started, then hit the
     weight loader walking every tensor the index lists, our cache bind
     expecting 78 main and 21 indexer layers, our MTP guard requiring
     num_hidden_layers == 78, and vLLM's MTP completeness check demanding
     weights for a layer the checkpoint has at 78 and the truncation puts at 8.
     Five are cleared without touching the untruncated path -- the fifth was
     the waiver, which had to go around the draft model's own load rather than
     the target's, since a ContextVar does not reach the other object. The
     sixth is not a literal: our own check that the official loader loaded the
     MTP head cannot pass under truncation, because the head is the
     checkpoint's layer 78 and a shortened model has nowhere to put it.
     Truncation and MTP are incompatible by construction, so a truncated arm
     needs a session without the draft engine -- which `--automaton` would
     permit, since it only calls `target.verify`. Hard-coded depths are what
     make a model un-shrinkable, and shrinking it is what lets a shared machine
     be used at all.

119. The truncated-backbone route ends on two constraints that oppose each
     other. Nine in all, seven cleared: the tree not reaching the container,
     the loader walking every tensor the index lists, our cache bind expecting
     78 main and 21 indexer layers, our MTP guard requiring depth 78, vLLM's
     MTP completeness check, our check that the official loader loaded the MTP
     head, the cache's eight uses of the draft engine, and the bench record's
     four snapshots. The two left: truncation needs dummy weights because the
     loader walks the full index, and dummy weights cannot satisfy the injected
     prefix because its first token was computed with the real ones. Forcing
     either would weaken a real guarantee for the sake of a measurement. Five
     of the seven were our own hard-coded depths, which is what makes a model
     impossible to shrink for any purpose at all.

120. The project's best result belongs to a different drafter than the plan
     credits. `lever84-20261020/auto12` records `drafter: "lookup"`, and the
     bench writes that field as `"automaton" if args.automaton else "lookup"`,
     so the 9.3794 ms TPOT is `LookupDrafter`'s and the suffix automaton has
     never run in the engine. The directory names -- auto8, auto12, auto16 --
     were read as the drafter rather than as the sweep. The same reading found
     that `--automaton` did not imply the host-drafted path, so all
     thirty-five arms queued today would have measured the MTP engine under
     names that say drafter. Read the record's own fields, not the directory's
     name.

121. Dummy weights and tensor parallelism are incompatible, which closes the
     truncated-backbone triangle. Under `load_format="dummy"` each of sixteen
     ranks initialises its own random weights, so they disagree about the first
     token and `start_mtp`'s check fires -- correctly, since catching that is
     what it is for. A shard is not a copy, so seeding identically would not
     help. The route needed dummy weights because the loader walks every tensor
     the checkpoint index lists; it needed real weights because TP16 needs one
     model. Running this engine in a fraction of its memory needs a loader that
     skips absent layers, which is inside vLLM. Fifteen constraints were cleared
     on the way, eleven of them our own, and the attempt found the two
     corrections that mattered more than the route.

122. The corpus was the proxy, not the finding. Drafters compared on this
     repository's source files put the tree within half a percent of the
     automaton at equal budget, so the tree's branching was recorded as worth
     nothing here. On the engine's own stream -- the capture case's 8192-token
     prompt and the 1024 tokens a real run produced -- the tree leads the
     automaton by 9 percent at width 12 and 20 at width 32, and the automaton
     leads lookup by 13 to 21. Source files repeat themselves differently from
     a model rewriting its input. The real stream was available all along, in
     the records of the run that set the project's best result.

123. The tree's optimum is a sixteen-node tree with four branches, and
     branching barely matters. On the engine's own stream, acceptance climbs
     monotonically to 8.678 at budget 64 while TPOT worsens past 16, because
     0.66 ms a verified row outruns what the extra acceptance buys; two, four
     and eight branches differ by one to two percent throughout. Depth beats
     breadth here, which is why a single-path automaton comes within 10 percent
     of a tree. With the fixed term the measured merges leave, 30.15 ms, the
     floor is 6.12 ms -- the target bounded out of the identified mechanisms a
     second time, now with measured data on both sides.

124. The workload's repetition, not the drafter, is the binding limit. For each
     position in the engine's own output, the longest run that appeared
     verbatim before: 83.8 percent of positions have four tokens, 57.9 have
     sixteen, 37.2 have thirty-two. An oracle copy drafter -- always finding
     the longest prior run -- accepts 9.660 at a sixteen-row budget, which is
     TPOT 4.21 ms against the perfect predictor's 2.39. So no copy-based
     drafter reaches 2.9 ms on this workload however well implemented, and the
     tree at 6.649 is already 69 percent of the oracle. Reaching the target
     needs prediction on the 42 percent of positions where nothing long enough
     was seen -- and the published MTP figures, 1.85 to 3.9 accepted, are below
     both. Corrected by the oracle's budget curve: acceptance saturates at
     13.474 by budget 64, which is what 2.9 ms needs, but 64 rows cost 42.2 ms
     at 0.66 a row and put TPOT at 5.37. So the binding constraint is the row,
     not the workload -- at 0.1 ms a row the same recall gives 2.72 ms. And the
     row cost is the graph's: replay rises with the captured width where the
     operators are flat eagerly, so a graph charges for rows that carry
     nothing.

125. [Withdrawn by 128.] The row cost is entirely the graph's, and that makes
     2.x an engineering question. A layer's operators are flat in the row count cold -- 126.969 us
     at 8 rows against 121.955 at 32, negative at the margin -- as they were
     warm, so the operator-side row cost over 78 layers is between 0 and -0.02
     ms against the round model's fitted 0.66. The whole of it is capture: the
     same sequence replays 527.0 to 582.9 us across a fourfold row range while
     eagerly it goes 568.7 to 554.4. And the copy oracle saturates at 13.474
     accepted by budget 64, exactly what 2.9 ms needs, so the only thing
     between recall and the target is 64 rows at 0.66 ms each. At 0.10 it is
     2.72 ms; at zero, 2.24.

126. Several captured graph sizes cost about 195 MiB each at 78 layers, and the
     increment does not depend on the size. Five sizes is about 1 GiB against
     the 48 GiB a rank holds, so capturing a list where the engine captured one
     cannot fail for memory -- which was the risk that would have killed the
     arm in the queue without answering anything. That the increment is flat in
     the size is itself evidence: the cost is the task list, not the data,
     which is the same thing the row cost turned out to be.

127. Holding several captured graphs does not slow the small one. Size 4
     replays at 516.989 us alone and 516.696 with four larger graphs resident,
     and size 32 is 12.9 percent dearer than size 4 -- so replaying the
     smallest graph that covers a draft is free of interference, which was the
     lever's last unverified premise. Every link of the chain from the round
     model to a one-line configuration change is now measured on one die or in
     the arms' own records.

128. The capture spread is a ninth of the fitted row cost, so the chain was not
     closed. Seven operators including the grouped matmul, captured at four
     widths: 9.52 ms a round at four rows against 10.70 at twenty-four, a
     spread of 1.17 ms where the round model charges 13.2 over the same twenty
     rows. So the operators are flat in rows and the captured width costs 0.05
     ms a row rather than 0.66, and five sixths of the slope is still
     unattributed. Multi-size capture is worth about 1.2 ms a round -- real and
     small -- and the arithmetic that reached 2.24 ms by removing the whole row
     cost is withdrawn. A chain is only as closed as its weakest measured link,
     and this one was an extrapolation from five operators to the engine's
     layer.

129. Attention is the only candidate whose magnitude fits the row cost, and it
     is two points from two captures. The engine's sparse attention rejected
     six argument sets on one die, so its scaling comes from 85.048 us a call
     at width 8 and 37.500 at mtp_width 4 -- 11.9 us a row, 0.93 ms a row over
     78 layers against the fitted 0.66. Right order, wrong provenance: the
     points are from different captures, which is the error of mixing a 52.8 ms
     round with a 16.6 ms one, made earlier today. If it holds the row cost is
     attention's own work rather than overhead, the copy oracle's 4.18 ms is
     the floor, and the queued cap12 and cap24 arms settle it as a side effect.

130. An evidence page that does not name its run leaves a number that cannot be
     used. `attention-sparsity-20261020` reports 85.048 us a call "from this
     project's own capture, width 8" without naming it; the three captures on
     the machine are all `mtp_width: 4` and its allGather count of 140 matches
     none of them. So the figure came from a fourth capture of unknown
     configuration and cannot be compared against anything -- which turned the
     attention row-scaling candidate from weakly measured into unmeasured. Name
     the run.

131. Attention is flat in the query rows, which exhausts the single-die
     candidates for the row cost. Through this project's own `shared_kv_fia` --
     which runs where six attempts at the engine's sparse operator were
     rejected -- 123.229 us at one query row against 123.203 at twenty-four,
     marginal between -0.39 and +0.19 us a row, about ±0.015 ms a row over 78
     layers against the fitted 0.66. With the projections flat warm and cold,
     the grouped matmul flat in rows, and capture worth 0.05, five sixths of
     the slope is unattributed and nothing left to try on one die. It is
     sixteen-rank behaviour or nothing.

132. The per-row attention mask was already in use. `shared_kv_fia` passes
     `atten_mask` of shape (1, 1, width, capacity) and is known to run, which
     is exactly what tree verification needs -- while today's `tree_mask` and
     `tree_attention` were built to feed `sparse_indices` to an operator that
     rejected every probe. The shorter path was in the codebase.

133. The shorter tree path is gated at width six. `install_gather_fia` raises
     above `GATHER_FIA_MAX_WIDTH` because "widths above six failed
     serial-equivalence checks" -- the failure `fia-mask-20261020` recorded as
     unisolated -- and a tree wants 16 to 24. So the per-row mask already in
     the codebase is not a shortcut: using it means isolating a correctness
     failure first. It also explains why the engine runs SFA rather than
     gather_fia, which is not an oversight but an unvalidated width. That gate
     now bounds two things: the tree, and gather_fia's own 21.3 us a layer.

134. The sentinel index cannot explain the width-six gate. `selection_mask`
     scatters zeros at `topk_indices` without filtering the `-1` the engine
     uses for unused slots, where `sfa_v1.py` filters it explicitly -- and a
     `-1` scatters to column `capacity - 1`. But the causal term masks that
     column for every query whose slot is below it, and at an 8K context in a
     9344 cache none reaches it. Excluded on paper, which is where a suspect
     with an arithmetic answer should be settled.
