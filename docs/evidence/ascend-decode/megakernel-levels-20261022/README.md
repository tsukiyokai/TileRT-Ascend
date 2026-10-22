# The concatenation wins at one dependency level and loses at the next

The kernel is now parameterised by the level rather than specialised to the
first, so adding a level costs a dictionary entry. The second level says the
mechanism is not universal.

| level | reduction | columns | rows | fused us | split us | saving us |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| hidden | 6144 | 3040 | 1 | 40.26, 40.01 | 82.54, 83.08 | 42.82 |
| hidden | 6144 | 3040 | 12 | 39.56, 39.92 | 84.90, 84.79 | **44.98** |
| hidden | 6144 | 3040 | 32 | 40.65, 40.64 | 86.26, 86.45 | 45.80 |
| query_norm | 2048 | 5120 | 1 | 29.71, 30.22 | 28.65, 28.60 | −1.57 |
| query_norm | 2048 | 5120 | 12 | 31.87, 31.42 | 29.04, 29.75 | **−2.12** |
| query_norm | 2048 | 5120 | 32 | 32.31, 32.70 | 30.78, 30.80 | −1.90 |

Two alternating passes a row count, eight changing inputs each, one die. The
arms' product digests are identical in every case; the levels sit within
6.1e-4 and 2.1e-4 of a float reference against magnitudes of 376 and 230.

## What separates them

The first level concatenates four projections of the normalized hidden -- query
latent 2048, kv latent and rope 576, indexer key 160, router gate 256 -- and so
removes three invocations over a reduction of 6144. The second concatenates two
projections of the query norm -- query up 1024 and indexer query 4096 -- and so
removes one over a reduction of 2048.

One invocation removed at a short depth does not pay for the wider per-core
slice it forces. The second level's 5120 columns are already 512-byte aligned at
10240 bytes a row, so alignment is not the difference. The fused arm is selected
for the first level and the split arm for the second.

## The limit on reading these numbers

The operands are bf16 and the product float, which isolates the invocation count
from the quantization contract but makes the first level read 37.36 MB where the
deployed W4A8 projections read roughly a quarter of that. Both arms shrink with
the bytes, and the saving need not shrink in proportion -- it is a count of
invocations, not of bytes -- but nothing here establishes the figure for the
engine's weights. That measurement is the next step, not a restatement of this
one.

[`run.sh`](run.sh) is the alternating runner; the JSON files are its output, one
a level an arm a row count a pass.
