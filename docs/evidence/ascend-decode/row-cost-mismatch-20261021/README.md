# The round model's 0.814 ms a row is 31 times what the operators cost

2026-10-21, a3-84, one die. A layer's operators -- norm, the fused input
projection, q_b, o_proj, the shared down projection, the dynamic quant and the
grouped matmul -- at four token counts:

     tokens   layer us   marginal us a token
          8    135.190
         12    132.616               -0.644
         16    136.508                0.165
         32    143.263                0.336

Eight microseconds between 8 tokens and 32. Over 78 layers that is
**0.026 ms a row**, against the round model's fitted **0.814**.

The fit is sound -- four widths, residuals of a few hundred microseconds on
values of 52 to 72 ms, and 19.3 ms separating width 8 from width 32 -- so the
cost is real and it is not in the operators. It is in something a single die
cannot reach:

- the sixteen-rank TP reduction, whose bytes do scale with rows (147 KB at
  width 12 against 393 KB at 32), though small-message HCCL is latency-bound
  and should not produce 19 ms; or
- the graph's static token buckets, where a wider width lands in a larger
  bucket and pays for it.

The drafter is excluded: the automaton and the tree each cost 0.1 to 0.3 us a
round on the host, measured on CPU over an 8192-token context.

## Why this decides the target

If the row cost is an implementation artefact rather than physics, the
constraint that has shaped every drafting decision today disappears. A tree's
nodes stop being expensive, and the fixed-budget tree -- 14.510 accepted a
round at an edit rate of 0.02, which the adaptive budget gave up as too costly
-- becomes the right configuration.

    fixed term after the boundary merges   25.58 ms
    64 nodes at 0.026 ms a row              1.66 ms
    round                                  27.25 ms
    accepted                               14.510
    TPOT                                    1.88 ms

Against 3.90 ms for the adaptive tree at the fitted 0.814. So this one number
is worth more than every lever measured today put together, and it is a
question about the engine rather than about an operator.

## The discriminator

Width 12 against width 13. If the cost is the graph's buckets, two widths in
one bucket differ by the operator figure -- about 0.03 ms -- while two widths
across a boundary differ by a bucket's worth. If it is the TP reduction, the
difference is smooth in the width. One pair of arms settles it, and they need
the whole machine.

## Three candidates excluded, and what is left

- **Graph buckets.** The engine sets `cudagraph_capture_sizes: [graph_width]`,
  one size, so every width captures its own graph and there is no step to pay
  at a boundary. Excluded by reading the config.
- **The TP reduction.** Two ranks, hidden 6144, float32: 111.27 us at 8 rows
  and 93.24 at 64, an eightfold range in bytes with a negative slope
  throughout. Not row-sensitive. Sixteen ranks may differ but the direction
  does not support it.
- **Per-row operator cost.** The page this corrects attributed 10.4 us a layer
  a row to scalar and issue pressure. A whole layer measures 0.336 us a row.
  Excluded, 31 times over.

What remains is trajectory coupling. The four widths each ran their own
trajectory: a wider draft accepts more, so each round advances further, the KV
grows faster, and later rounds pay more for attention. `round(width)` then
mixes the width's own cost with the context length its acceptance produces,
and the 0.814 ms a row is mostly the second.

If that is right, widening does not cost what the model says, a tree's nodes
are not expensive, and the fixed-budget tree at 14.510 accepted becomes
reachable -- the 1.88 ms line above.

## The discriminator, corrected

Not adjacent widths -- buckets are excluded, so that tests nothing. Fix the
number of tokens generated rather than the number of rounds, and compare
widths over the same trajectory length. If the round cost is flat in the width
at constant trajectory, the row term is an artefact of the measurement.

## Five candidates excluded; the slope is sound and what it measures is not identified

Added to the three above:

- **Trajectory length.** The arms cap at `--max-tokens 1024`, so every width
  generates the same token count over the same growing context. Width changes
  the round count, not the trajectory.
- **The Gloo command broadcast.** A round's scope is "Gloo, draft chain,
  verify, MTP recompute, commit", and the Gloo part is one
  `broadcast_object_list` a round. Measured over two ranks it is flat in the
  draft size: 305.0 us at width 0, 289.6 at 64.

So the fitted 0.814 ms a row is real -- four widths, 19.3 ms across the range,
residuals of a few hundred microseconds -- and none of the five mechanisms that
could produce it survives measurement. The honest position is that the slope is
sound and what it measures has not been identified, which is why the queued
width arms lead the queue: this one number decides whether a tree's budget is
affordable, and with it whether 2.x ms is reachable.

## A smaller finding on the way past

The Gloo command broadcast costs 305 us an object against 106 us for an
equivalent integer tensor -- threefold on two ranks, and object broadcast
pickles, sends a length, then sends bytes, so sixteen ranks will be worse.
One a round at 55 ms is half a percent, but the fix is to broadcast a
fixed-size tensor instead of a Python object, and the drafter is deterministic
so each rank could recompute the draft from the same state at 0.1 us rather
than receive it at all.
