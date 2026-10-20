# Computing every head on every rank beats sharding plus a reduce

The collective floor is 120 us with the better expansion mode and resists every
setting, so the only way past is fewer calls. One call a layer is the reduce
after attention's output projection, and it exists because the heads are
sharded.

At batch one the alternative is redundancy: every rank computes the whole
attention over a replicated KV cache, and the reduce disappears.

2026-10-21, a3-84, one die, 12 rows, 8192 cache rows, 9.4 MB of cache.

| heads | us | per head |
|---|---|---|
| 4 | 87.17 | 21.79 |
| 8 | 91.53 | 11.44 |
| 16 | 115.53 | 7.22 |
| 32 | 135.94 | 4.25 |
| 64 | 161.03 | 2.52 |

Sixteen times the heads costs 1.85 times as much. At twelve rows the attention
is dominated by reading the 9.4 MB cache, and that read does not scale with the
head count.

## The trade

    sharded     4 heads   87.17 us + one reduce at 120 =  207 us a layer
    redundant  64 heads  161.03 us + no reduce          =  161 us a layer

**46 us a layer, 3.6 ms a round**, against the better expansion mode. Against
the default mode's 200 us floor it is 287 against 161 -- 126 us a layer, 9.8 ms
a round.

The cost is memory: the KV cache replicated on every rank is 9.4 MB a layer,
734 MB for 78 layers, which is 1.5 percent of the 47 GiB a rank already holds.

## What this does not yet establish

That the engine's attention can be asked to compute every head. Its MLA path
shards the heads as part of the weight layout, so q_b and o_proj are sharded
with them -- undoing that means each rank holding the full q_b and o_proj
weights, which is sixteen times the attention weights rather than the cache.
Whether that fits, and whether the output projection's own reduce disappears
with it, is the next question.

And the 120 us saved assumes one collective a layer attributable to attention.
The round's measured collective cost is 374 us a layer, which is about two
calls, so the other one belongs to the MoE and is untouched by this.
