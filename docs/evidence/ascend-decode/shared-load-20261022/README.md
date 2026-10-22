# An operation costs 2.35 microseconds inside a load and 24.3 with its own

The pass budget says a layer must go from about 48 passes over its data to about
eleven, and that removing a pass means two operations sharing one load. This
measures what that is worth. It is worth a factor of ten.

## The measurement

The front stage's vector half, twelve rows, two passes over 72 chunks of 1024
elements, one die, five repeats. Both arms do the same arithmetic; only the
loads differ. `one load` keeps the chunk in unified buffer across every
operation; `n loads` gives each operation its own load and store.

| operations | one load us | n loads us | saving |
| ---: | ---: | ---: | ---: |
| 1 | 59.37 | 59.74 | 0.37 |
| 2 | 60.82 | 83.40 | 22.58 |
| 4 | 61.10 | 133.84 | 72.74 |
| 8 | 75.84 | 229.91 | 154.07 |

The shared arm is nearly flat -- 59.37 to 61.10 from one operation to four --
while the separate arm grows linearly. The marginals:

    an extra operation sharing the load      2.35 us
    an extra operation bringing its own     24.30 us

A factor of 10.3. The cube's product is exact at every point, maximum
difference zero against an integer reference.

## What it settles

Every negative result today pointed here. Fewer kernel launches is 1.064
microseconds each in a graph. Fewer operator calls is nothing by itself, since
an in-kernel pass costs what a call costs. Rescheduling passes to overlap the
machine's halves is bounded by the vector work there is to hide, 23.40
microseconds in the FFN body. Dividing the expert loop more finely to balance it
buys another reduction walk. And cutting the cube into pipeline steps buys a
walk a step, which is why the front stage lost 70 microseconds that way.

This is the one that pays, and it is the same rule the cube obeys: the front
stage's 26.87 microseconds came from walking the reduction once for four
concatenated projections rather than four times. Load once, compute several
things, store once -- on either half of the machine.

## Scope

One die, synthetic vector work, the front stage's own data size. The 24.30
microseconds an extra load is that size's figure and does not transfer directly
to a layer's other passes, whose data differs. What transfers is the ratio and
the rule.

[`run.sh`](run.sh) is the sweep; the JSON files are its output.
