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
