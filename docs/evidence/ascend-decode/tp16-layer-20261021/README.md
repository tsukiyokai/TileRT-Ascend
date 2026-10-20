# A complete layer on sixteen ranks, at a decode round's width

Every per-layer figure this session used came from one die with no collective in
it; every collective figure came from a standalone benchmark or a
prefill-scoped capture. This is the number both halves lacked. One layer's
weights a rank -- 12 MB of attention shards and a 151 MB expert bank -- fit what
the tenant leaves.

Sixteen ranks, twelve rows, AIV with determinism:

    compute only, no collective     493.22 us
    with two all_reduce calls       870.63 us
    the collectives cost            377.42 us

    a round of 78 layers             67.91 ms
      of which compute               38.47 ms
      of which collectives           29.44 ms

## Which vindicates a figure this session withdrew

The 374 us a layer of collective cost was derived as 588 minus 214 -- the
engine's fitted term over 78 against a synthetic layer in a graph -- and then
withdrawn as an artefact of dividing a round-wide fit by the layer count.

Measured directly, a layer's two reductions cost **377.42 us**. The derivation
was right and the retraction was wrong: the quantity exists and the arithmetic
found it.

## What the total does and does not say

67.91 ms a round against a measured 50.710 means this synthetic layer is an
upper bound, not a prediction. It runs eight separate operators where the engine
fuses them into super kernels and overlaps its streams, which is exactly the 637
super kernel calls a round the census counted.

So the useful readings are the parts: collectives are 29.44 ms a round **after
AIV is already enabled**, which is 43 percent of this bound, and the naive
unfused compute is 38.47.

## And it sets the ceiling on the day's levers

The rope graph, the bank narrowing and AIV act on compute and dispatch. The
29.44 ms of collective time remains, and the five exits measured today --
smaller messages, explicit overlap, fewer calls, fewer ranks, a hand-rolled
primitive -- are all closed. At 5.441 accepted, 29.44 ms of collectives alone is
5.41 ms of TPOT, above the 2.9 target before a single layer computes anything.

That is the session's hardest finding and it is now measured rather than
inferred: the target is unreachable while every layer reduces twice.
