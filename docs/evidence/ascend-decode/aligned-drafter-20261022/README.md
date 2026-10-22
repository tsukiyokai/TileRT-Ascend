# A cursor into the prompt is worth 1.78x the automaton, and widening pays again

2026-10-22, CPU, no device. The question was why the project's drafters propose
so little a round -- 3.400 accepted at a budget of 256, against a copy oracle's
13.474 -- when `LookupDrafter.propose` already returns `context[end:end+count]`
and so is not capped by its match length at all.

## The defect is the anchor, not the proposal length

`_latest` maps a tail to its most recent occurrence over prompt **plus every
committed token**, so as generation grows the drafter's own output becomes the
most recent occurrence of everything. Counting where each round's anchor lands,
on a perturbed source file at width 32:

| edit rate | rounds | anchor in source | in output | no anchor |
|---|---|---|---|---|
| 0.00 | 352 | 350 | 0 | 2 |
| 0.02 | 1188 | 551 | 255 | 382 |
| 0.05 | 2311 | 753 | 591 | 967 |

A third of rounds at a 2 percent edit rate, and two fifths at 5 percent, have
**no anchor at all** and so accept exactly one token: the tail they would
search for is what the edit just rewrote. Another fifth to a quarter anchor in
the drafter's own output rather than in the source it is reproducing.

## The cursor has neither failure

`AlignedDrafter` keeps a place in the source instead of searching for one. The
target's rejection is itself the edit's position, so the next round proposes
from the source token after it, and the index it re-anchors against covers the
source only. Accepted tokens a round, same stream, same acceptance rule:

| edit rate | width | lookup | aligned | gain |
|---|---|---|---|---|
| 0.00 | 12 | 12.924 | 12.998 | 1.01x |
| 0.00 | 32 | 32.494 | 32.963 | 1.01x |
| 0.00 | 64 | 63.544 | 64.989 | 1.02x |
| 0.02 | 12 | 6.932 | 10.446 | 1.51x |
| 0.02 | 32 | 10.086 | 18.968 | 1.88x |
| 0.02 | 64 | 11.181 | 24.704 | 2.21x |
| 0.05 | 32 | 4.920 | 11.683 | 2.37x |
| 0.10 | 32 | 2.860 | 6.340 | 2.22x |

Blind rounds go to zero, because the cursor always has something to propose.

In the repository's own comparator at width 32 the ordering holds against every
drafter it carries: at a 2 percent edit rate automaton 12.431 and tree 12.292
against aligned **22.097**, which is TPOT 5.79 against **3.26** on the round
model's measured terms; at 5 percent, 6.055 against 14.144.

## It overturns this morning's conclusion that widening is spent

With tail matching, acceptance grew sublinearly while the row cost grew
linearly, so the optimum width sat at or below 32. With the cursor, acceptance
goes 10.446, 18.968, 24.704 across widths 12, 32 and 64 -- near enough linear
that widening pays again. That curve was the drafter's saturation, not physics.

## What it does to the target

Carrying the measured ratio to the engine's own stream, where the automaton
takes 6.872 at width 32, the cursor projects about 12.2. Then

    fixed 45.88, row 0.814   (45.88 + 26.05) / 12.2 = 5.90 ms
    fixed 45.88, row 0.458   (45.88 + 14.66) / 12.2 = 4.96   shared read
    fixed 15.97, row 0.458   (15.97 + 14.66) / 12.2 = 2.51   and eleven passes

So all three legs are now measured rather than assumed, and the drafter is the
only one of them that costs nothing on the device.

## Scope

A constructed stream: a repository source file, and that file again with a
share of its positions replaced. It overstates acceptance against the engine's
own stream by about 1.8x, which is why the projection above carries the ratio
rather than the level. The cursor also assumes the output reproduces the source
roughly in order; its re-anchor handles jumps, but a reordering workload is
untested. `EfficientEdit` (arXiv 2506.02780) is where the mechanism comes from:
"there is no need to determine an optimal draft length for reuse, as a single
forward pass can verify a long sequence of tokens", and rejection "implicitly
identifies the potential starting position".
