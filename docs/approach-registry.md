# Approach registry

Grouped by the idea each route uses, not by wording, so that convergence is
visible. A route is blocked when it stalls at something as hard as the goal
itself; it reopens only on a materially new mechanism.

| family | idea | state | holder |
|---|---|---|---|
| A — collective elimination | the collective's cost is rendezvous, not transfer; remove the rendezvous and put the push inside compute | **active**, measured | root, plus the in-worker init blocker |
| B — tokens per round | raise the denominator instead of lowering the numerator: a round that delivers more tokens may cost proportionally more | **active**, unmeasured | agent |
| C — weight bytes and residency | 52% of the budget is one pass over the weights; cut the bytes or keep a layer's active experts on chip | **active**, unmeasured | agent |
| D — round structure and host tax | ~7.4 ms of the round is in no operator, and 16 ranks arrive 4.4 ms apart | **active**, partly measured | agent |
| E — attention block | 78 o_proj reductions, ~637 cube/vector handoffs, 556 `Cast` calls per round | **dormant** — measured but nobody holds it | — |
| F — adversarial audit | every claim above is a candidate for refutation | **active** | agent |
| G — parallel layout | EP16 and the vendor's fused MoE | **blocked** — measured slower than TP16 per round (+1.09 ms, and +1.41 more with the fusion), spread growing 0.252 → 0.756 ms, the batch-one tail signature. Reopens only on EP4×TP4 or hot-expert replication, which change the tail's shape rather than retrying it | — |
| H — call fusion for its own sake | fuse matmuls to amortise per-call cost | **narrowed** — bf16 matmul is already at 93.8% of the die's ceiling and a chain of 32 costs the same per call as one, so there is no bandwidth and no launch to win; what remains is the 8 µs in-graph operator floor, worth 37% at 4→1 | folded into A |

## Why the portfolio is shaped this way

A, B and C attack different terms of the same inequality. A round must fit
6.87 ms at acceptance 3.433; A shrinks what the round spends, B enlarges what
it may spend, C lowers the floor underneath both. They can all be right at
once and none subsumes another, so none should be dropped for the elegance of
another.

The agents on B, C and D were not told which family the root favours. Early
independence matters more than early agreement: the collective route gives
the most attractive reduction right now, and an agent told so would very
likely find a way to agree with it.

F exists because seven of today's eleven retractions were the same instrument
error — reasoning from whichever figure arrived first, when the same quantity
differs by one to three orders of magnitude depending on how it is measured.
That error is invisible from inside the route that commits it.
