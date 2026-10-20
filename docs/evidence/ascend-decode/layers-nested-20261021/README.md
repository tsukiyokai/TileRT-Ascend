# Nesting layers inside one launch pays the tax N times

2026-10-21, a3-84, one die, one process, decode shape.

    vendor                            69.388 us
    mine, 1 layer in one launch       71.839 us
    mine, 2 layers in one launch     124.402 us   1.73x
    mine, 4 layers in one launch     230.181 us   3.20x
    mine, 8 layers in one launch     442.088 us   6.15x

Eight layers cost 6.15 times one, not once. So one launch covering several
layers does not amortise the per-layer cost; it saves about 23 percent, which
is the launch and the entry, and nothing more.

## What this corrects

Yesterday's reading of the same tax -- that it is a per-execution constant and
that one execution over several layers would pay it once -- is wrong. The
constant is independent of rows and of weight size but it accumulates with
layers, so it lives inside each layer's work rather than at the operator's
entry.

The mechanism with that shape is the grid barrier. The stage-major kernel takes
four `SyncAll` a layer, and nesting layers multiplies them by the layer count:
4N barriers for N layers, which is exactly the observed scaling.

## What it points at instead

Stage-major across layers, not layer-major across stages. Do every layer's
split, then every layer's gate projection, then every layer's activation: the
barrier count falls from 4N to 4 for the whole group. That is a loop
interchange in the kernel body, and it is the version the measurement
actually supports.
