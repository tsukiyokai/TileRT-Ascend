# The grouped matmul's cost follows the bank, not the rows

2026-10-21, a3-84, one die, 96 sorted rows held fixed, INT8 into INT32 in a
graph.

| experts | bank MB | us | GB/s |
|---|---|---|---|
| 1 | 9.4 | 55.57 | 170 |
| 2 | 18.9 | 54.24 | 348 |
| 4 | 37.7 | 73.85 | 511 |
| 8 | 75.5 | 105.51 | 716 |
| 16 | 151.0 | 168.58 | 896 |

Linear in the bank: 7.53 us an expert over an intercept of 55.57, which
predicts 168.6 at sixteen against 168.58 measured. Bandwidth climbs to 896 GB/s
at the full bank, so the operator is reading the whole thing at close to HBM
speed whatever it was asked to compute. That is why it is flat in rows: 2.8
percent from 16 rows to 256.

## What a batch-one round actually touches

Each row picks eight of two hundred and fifty-six experts, and a rank owns
sixteen. Under uniform routing the expected number of a rank's own experts
touched, and what the bank would cost if only those were handed over:

| rows | picks | touched | us | saved | ms a round |
|---|---|---|---|---|---|
| 1 | 8 | 0.49 | 55.6 | 113.0 | 8.81 |
| 5 | 40 | 2.32 | 65.5 | 103.1 | 8.04 |
| 12 | 96 | 5.01 | 85.8 | 82.8 | 6.46 |
| 24 | 192 | 8.45 | 111.7 | 56.9 | 4.43 |

At the acceptance ceiling of 4.289 the engine proposes a handful of rows, so a
rank sees two to five of its sixteen experts. Narrowing the bank to those is
worth **82.8 to 103.1 us a layer, 6.5 to 8.0 ms a round** -- against the 4.25
ms the target needs.

## First lever this session whose size exceeds the gap

And the mechanism is calibrated rather than assumed: the cost is linear in the
bank with a measured slope, so the saving follows directly from how many
experts a round touches.

## Three premises to check before believing it

Routing may not be uniform. A real stream's topk distribution is on this
machine and can be counted rather than modelled -- and a copy-heavy workload
repeating its own tokens may well route them to the same experts, which would
*help*, since fewer distinct experts means a narrower bank.

The narrowing must not cost a copy. `npu_grouped_matmul` takes a list of
weights, so handing it a subset should be a matter of which tensors are in the
list rather than moving any bytes -- but the vendor's path builds that list
once at load, and whether a per-round list costs anything is unmeasured.

And this is distinct from a lever already closed. "Narrowing the draft to touch
fewer experts" changed the drafter to route to fewer; this changes nothing about
routing and only declines to read experts no row selected.

## The saving depends entirely on routing concentration, which is unmeasured

| rows | routing | touched | us | saved | ms a round |
|---|---|---|---|---|---|
| 5 | uniform over 256 | 2.43 | 66.4 | 102.2 | 7.97 |
| 5 | favouring 64 | 7.87 | 107.3 | 61.2 | 4.77 |
| 5 | favouring 32 | 12.14 | 139.5 | 29.1 | 2.27 |
| 5 | favouring 16 | 15.52 | 164.9 | 3.6 | 0.28 |
| 12 | uniform over 256 | 5.02 | 85.9 | 82.7 | 6.45 |
| 12 | favouring 32 | 15.48 | 164.6 | 3.9 | 0.30 |
| 12 | favouring 16 | 16.00 | 168.5 | 0.0 | 0.00 |

So the lever is worth between 0 and 8 ms a round depending on a quantity this
project has never measured. At twelve rows, routing concentrated onto 32 of the
256 experts already touches a rank's full sixteen and the saving is nil.

## And the argument offered for it was wrong

This page said a copy-heavy workload repeating its tokens would route them to
the same experts, "which would help, since fewer distinct experts means a
narrower bank". That holds globally and not per rank. Concentration means the
touched experts cluster, and if the cluster falls inside a rank's own sixteen
that rank touches all of them. Fewer distinct experts globally does not imply
fewer on any given rank.

So the premise cannot be argued either way. It has to be counted: the topk
indices of a real decode round, per rank, which no saved artifact on this
machine holds -- `moe_capture.py` exists to record exactly that and has not been
run against a decode.

## Where that leaves the lever

Mechanism calibrated: cost is linear in the bank at 7.53 us an expert over a
55.57 intercept, 896 GB/s at the full bank, flat in rows. That part is
measured and stands.

Size unknown: 0 to 8 ms a round, set by the routing concentration. One capture
of a decode round's topk indices settles it, and that capture is the same one
the queued arms need the machine for.

## The router's own bias says routing is not clustered

`mlp.gate.e_score_correction_bias` is 256 trained values added to the gate's
scores before the top eight are taken -- the model's own load-balancing term,
readable from the checkpoint with no machine at all.

| layer | min | max | std | top8 - median |
|---|---|---|---|---|
| 10 | 6.8174 | 7.0632 | 0.0403 | 0.0460 |
| 11 | 6.8044 | 7.0193 | 0.0343 | 0.0460 |
| 13 | 6.0188 | 6.2417 | 0.0243 | 0.0200 |
| 14 | 5.9079 | 6.1098 | 0.0271 | 0.0210 |
| 17 | 7.5610 | 7.7459 | 0.0264 | 0.0200 |

The range is 0.02 to 0.03 across 256 experts, so the bias barely distinguishes
them: it is a near-constant term, which is what a well-trained balancing
correction should be. Counting the experts it leaves within one standard
deviation of the eighth-best gives 89 to 162 of 256 by layer.

Priced over that contention set, five rows of eight picks each:

| contending | touched | us | saved | ms a round |
|---|---|---|---|---|
| 89 | 2.02 | 63.3 | 105.2 | 8.21 |
| 130 | 2.16 | 64.3 | 104.2 | 8.13 |
| 162 | 2.22 | 64.8 | 103.8 | 8.09 |

Insensitive: 8.09 to 8.21 ms across the whole range. Forty picks spread over at
least 89 experts touch about two of a rank's sixteen however the 89 are chosen.

## So the lever's size narrows from 0-to-8 to about 8

To zero it, routing would have to concentrate below 32 experts, and the
narrowest contention set the bias permits is 89.

What this does not cover: the bias is one of two terms. The score itself is
`hidden @ gate.weight`, and a token's scores across layers may correlate in a
way the bias cannot show. So concentration is excluded on the bias side and
unmeasured on the score side -- which needs hidden states, and the one saved
tensor on this machine holds a single row.

## A probe bug worth recording

The first pricing printed 8.04 ms for every layer because its touched-expert
expression contained `contending / GLOBAL * (1 / contending)`, which cancels
the quantity being varied. A constant column across a swept parameter is the
signal, and it was visible before the numbers were believed.
