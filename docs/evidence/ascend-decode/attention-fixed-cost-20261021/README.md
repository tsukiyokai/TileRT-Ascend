# Attention's cost is 88 percent fixed, not the KV read

One die, twelve rows, four heads, the cache capacity swept:

| capacity | MB | us | GB/s | us a 1024 rows |
|---|---|---|---|---|
| 512 | 0.6 | 171.75 | 3.4 | 335.44 |
| 1024 | 1.2 | 169.78 | 6.9 | 165.80 |
| 2048 | 2.4 | 172.10 | 13.7 | 84.03 |
| 4096 | 4.7 | 182.48 | 25.9 | 44.55 |
| 8192 | 9.4 | 195.65 | 48.2 | 23.88 |
| 16384 | 18.9 | 207.96 | 90.8 | 12.69 |

Thirty-two times the capacity costs 21 percent more time. The fit is about
**170 us of intercept and 2.3 us a thousand cache rows**, so at the engine's
8192 rows the context is 26 us and the other 170 is independent of what is
read.

## Which corrects a claim made one turn earlier

That turn said attention's cost "equals reading 9.4 MB, a physical quantity".
9.4 MB in 196 us is 48 GB/s, about three percent of this die's HBM -- so it was
never the read, and the arithmetic to check it was one division.

## And it explains two earlier results

The gathered path loses because it keeps the same fixed cost while quadrupling
the bytes: selecting 2048 rows per query row moves 28 MB at width 12 against
the full cache's 9.4, and the bytes were never the binding term.

Sixty-four heads cost 1.85 times four heads rather than sixteen times, for the
same reason -- the fixed term dominates and the head count rides on top of it.

## What it opens

A 170 us intercept on an operator called once a layer is 13.3 ms a round, the
largest fixed cost in the model after the collectives. Nothing about it is the
cache, the heads or the rows. Whether it is the operator's tiling, its
workspace, or its own internal launches is the next question, and the profiler's
per-kernel breakdown inside the FIA call is where it would show.
