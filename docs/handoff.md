# Handoff, 2026-10-21

## Where the target stands

Measured: TPOT 9.3794 ms, round 50.73 ms at p50 over 188 rounds with a spread
of 1.06x, 5.441 tokens accepted a round. The target needs a round of 15.78 ms
at that acceptance.

## The round's account, 88 percent attributed, one provenance throughout

    a layer, 21 kernels, one graph replay, harness floor removed   497.03 us
    78 layers                                                       38.77 ms
    two reductions a layer, measured in a chain on sixteen ranks      6.10
    sum                                                              44.87
    the measured round                                               50.73
    unattributed                                                      5.86

Every term is one replay of a whole unit against an engine that replays whole
units. Sums of separately-timed operators are not comparable to it and were
withdrawn.

## The levers

| lever | size | state |
|---|---|---|
| narrow the expert bank to the experts a round touches | 17.41 ms a round | implemented, `--narrow-bank`, bit-exact on five gap patterns, verified through the installer at real shapes, never run in the engine |
| hoist the metadata, with the rope refresh captured | 7.28 ms, floor 0.13 | implemented, `--hoist-metadata --rope-graph`; the rope half only exists when hoisting is on |
| AIV collective expansion | about zero | default on; the host path it improves is amortised away inside a graph |

The bank is the only one whose size is both large and fully evidenced: the
grouped matmul is bandwidth-bound at 1170 to 1255 GB/s, a rank's sixteen
experts are 302 MB and cost 258.17 us a layer across two calls, the two a round
touches cost about 35, and the checkpoint's router bias spans 0.02 to 0.03
across 256 experts so routing is not clustered.

## What is queued

`/workspace/experiments/levers-20261021/watch_and_run.sh` runs detached in the
container, polls every five minutes for sixteen hours, and starts seven arms
once a die has 48 GiB free: baseline, hoist-rope, narrow, hoist, all, aicpu
(AIV's negative control), steady (`--profile-last`). All carry
`--round-phases`. Predictions are in
`docs/evidence/ascend-decode/predictions-20261021/`.

`steady` matters most: the project has never captured a steady-state decode
round, and every figure read from the six-token `profile-rope-20261018` had to
be withdrawn or re-derived.

## What is closed, by measurement rather than argument

Drafting: acceptance saturates at 4.289 offline and the engine's own 5.441 is
exact, and four independent routes to the copy oracle's 13.474 all fail -- its
advantage is knowing the answer. Within-layer overlap: the attention side is a
strict chain and the MoE's fork is cube against cube, where concurrency costs
18.49 us. Fewer ranks: a rank owns 256/ranks experts so narrowing does not make
TP8 affordable. The gather_fia gate: raising it is 19 percent slower at width
12 and its failure is one bf16 ULP. Cross-layer fusion: 6.15x.

## The method lessons that cost the most

`docs/lesson.md` holds 214 entries. The ones that changed the most work:

- Figures from different scopes can sit side by side and cannot overturn each
  other. The collective term read 29.64, 10.36, 5.11, 1.41 and 6.10 ms across
  the day, each measured, and only the last was measured in a chain.
- Before attributing a residual, check the numerator belongs to the
  denominator. 374 us a layer was derived by dividing a round-wide fit by 78.
- A lever cannot remove more than the item it acts on. That arithmetic caught a
  probe using the wrong inner width.
- Measure the instrument. The replay harness charges 26.58 us before the graph
  does anything, and it was counted as a finding twice.
- Read the class the engine instantiates, not the base class whose name
  matches. The narrowing wrapped `get_mlp_weights` where the W4A8 method has
  `_get_mlp_weights` with four return values, and it installed silently.
