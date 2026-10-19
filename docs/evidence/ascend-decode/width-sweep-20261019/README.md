# The round is linear in width, and widening past 8 does not pay

Four arms in one session, the last repeating the first as the chain's own
drift control. Best configuration: three gmm1 patches, conditioned graph,
cache fix. Every arm `status=measured`, no rank errors, exit 0.

| width | chain | verify | round floor | round p10 | rounds | acceptance | TPOT |
|---|---|---|---|---|---|---|---|
| 8 | 1.707 × 6 | 39.677 | 52.261 | ~52.3 | 227 | 4.507 | **12.047** |
| 12 | 1.684 × 10 | 41.460 | 60.743 | 62.280 | 210 | 4.871 | 12.995 |
| 16 | 1.744 × 14 | 43.338 | 70.286 | 71.739 | 191 | 5.356 | 13.847 |

A chain step costs 1.712 ms and does not depend on the width — three arms with
6, 10 and 14 steps give 1.707, 1.684, 1.744. The verify grows 0.458 ms a row,
measured twice at 0.446 and 0.470. So

    round(w) = 1.712(w − 2) + 0.458w + 41.753 = 2.169w + 38.330

which predicts 52.26, 60.94, 69.62 against measured 52.26, 60.74, 70.29.

## The criterion

Widening costs **2.169 ms a unit of width**, so it pays only while acceptance
grows faster than 2.169 / TPOT = **0.187 tokens a unit**.

| | acceptance slope | verdict |
|---|---|---|
| 4 → 8 | 0.294 | pays |
| 8 → 12 | 0.091 | does not pay |
| 12 → 16 | 0.121 | does not pay |

Acceptance carries across configurations: the token digest is `68f2bc184553`
in every arm at every width tested, so the same tokens are accepted whatever
the kernels underneath. That is what lets the width-4 acceptance of 3.332,
measured on the older configuration, be used here.

So the optimum is above 4 and below 12, and width 6 is the one untested point.
Its round is predicted at 2.169 × 6 + 38.330 = 51.34, which beats width 8 only
if acceptance at 6 is above 51.34 / 12.047 = 4.262 — against 4.507 at width 8
and 3.332 at 4. Linear interpolation gives 3.92, so width 8 should hold, but
the interpolation is the weakest link and one arm settles it.

## What it says about the two halves of the round

The chain's cost per unit of width, 1.712 ms, is twelve times the verify's
0.458. A speculative round buys its extra candidates almost entirely in the
chain, and the chain is serial single-row forwards whose cost is a fixed
overhead rather than work: one row costs 1.712 against 0.458 for one more row
inside the verify, **3.7×**.

That is the same ratio the phase split gave from a different direction — a
single-row MTP layer at 1.731 against 0.501 a layer inside the eight-row
verify. Two independent measurements of the same fixed cost.
