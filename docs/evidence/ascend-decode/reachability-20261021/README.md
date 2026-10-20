# What 2.x ms requires, after a day of remeasurement

The round is 55.424 ms and TPOT is round over tokens accepted. Today's
measurements fix both halves of what is left.

## The engine side is nearly exhausted

Per-kernel issue under graph replay is 1.064 us, so 25 kernels a layer cost 27
us of the layer's 710. The rest is device time in operators the vendor has
already fused:

- The decode attention is three boundaries, not many:
  `mla_preprocess_only_decode` covers the q and kv projections, both norms,
  the rope and the cache write; then sparse attention; then `o_proj`. Little
  to collapse.
- The routed experts are four boundaries and one kernel replaces them,
  measured at 72.463 us against 181.172 -- 8.48 ms a round, from three HBM
  round trips removed rather than three launches.
- The TP reduction is 6.5 ms by the one-sided push, already built and correct
  at sixteen ranks.

Those two together take the round to about 40 ms. At today's acceptance of
5.441 that is a TPOT near 7.4 ms.

## Which puts the target on the drafter

    TPOT = 40 ms / tokens accepted

2.9 ms needs 13.8 accepted a round. Width twelve accepts 5.441 and width
thirty-two accepts 6.053, so widening does not reach it -- the acceptance
curve is flat where the cost curve is not. 13.8 from a width of 32 means an
acceptance rate of 43 percent against today's 19.

So the remaining factor is not engine work. It is either a drafter that is
right far more often, or a round that verifies far more tokens for the same
cost. For an agentic coding model that copies its input, the first is the one
with headroom: the automaton drafter already lifted acceptance from 5.273 to
6.053 at width 32 by matching the longest suffix with any continuation, and it
still proposes a single linear path.

## The honest position

Engine-side optimisation reaches about 7 ms. The last factor of three is a
drafting problem, and tree verification -- which needs the attention kernel to
take a per-row mask it does not currently take -- is the only mechanism
identified that changes the acceptance curve's shape rather than its position.
