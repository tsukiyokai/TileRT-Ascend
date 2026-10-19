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
12. Pinning seq_lens removes a queue-dependent host stall without adding device
    work. Keep measured paired savings separate from the best absolute request.
    More speculative rows can expand the expert union; a 64-expert sample is
    not a worst-case bound for width 32 with top8 routing.
13. A serial/speculative mismatch does not identify future-token leakage.
    Inspect selected IDs and causal thresholds, then perturb only forbidden
    KV slots. An empirical width limit is not an indexer causality boundary.
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

Evidence: [merged tree](evidence/ascend-decode/merge-tree-20261020/README.md),
[lookup](evidence/ascend-decode/lookup-contract-20261019/README.md),
[SFA routing](evidence/ascend-decode/sfa-routing-20261019/README.md),
[pinned staging](evidence/ascend-decode/metadata-h2d-20261019/README.md),
[measurement workflow](workflow.md).
[Detailed historical lessons and their evidence](lesson-20261019-2340.md).
