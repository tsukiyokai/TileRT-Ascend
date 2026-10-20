# The frontier: both halves in one table

`TPOT = 78 layers x per-layer / accepted`, rows excluded because the fit's
0.66 ms a row is unattributed. Every figure measured today.

| per layer (us) | lookup@12 5.441 | tree@16 5.394 | automaton@32 6.872 | oracle@16 9.660 | oracle@24 11.011 |
|---|---|---|---|---|---|
| **588** engine now, fitted term / 78 | 8.43 | 8.51 | 6.68 | 4.75 | 4.17 |
| **386** after the measured merges | 5.54 | 5.59 | 4.39 | 3.12 | 2.74 |
| **317** the layer's parts, warm | 4.55 | 4.59 | 3.60 | 2.56 | 2.25 |
| **214** the same layer in a graph | 3.07 | 3.09 | **2.43** | 1.73 | 1.52 |
| **203** its profiled device-busy | 2.90 | 2.93 | 2.30 | 1.64 | 1.43 |

The engine's own cell is 8.43 against 9.3794 measured; the difference is the
row term the fit charges and nothing explains.

## Nine cells reach 2.9 ms, and one of them is made of two measurements

The honest reading is the cell at 214 us a layer and 6.872 accepted: **2.43
ms**. Both numbers are measured -- 214 is the synthetic layer replayed in a
graph on one die, 6.872 is the automaton at width 32 on a real stream -- and
neither needs the copy oracle, which is a bound rather than a drafter.

So 2.x is inside two things already measured, and what stands between is a
single quantity: the engine pays 588 us a layer where the same operators
replay in 214.

    588 - 214 = 374 us a layer = 29.2 ms a round

That is the same unattributed four fifths of the row cost, seen per layer
instead of per row, and it is now the only thing between the measurements and
the target.

## What it is not

Not the operators: every one is flat in rows, warm and cold, and their warm sum
is 317 against the engine's 588. Not capture width: 0.05 ms a row. Not the TP
reduction at two ranks, the Gloo command, the trajectory, the drafter, or graph
buckets -- all excluded. What remains unexamined is sixteen-rank behaviour,
which needs the machine the arms are queued on.

## Why this table is worth more than its cells

Every earlier attempt priced one half and assumed the other. The merges were
costed against a fixed acceptance; the drafters against a fixed round. Neither
reaches 2.9 alone and the product of their measured bests does, which is why
the single unattributed quantity -- 374 us a layer -- is the whole project
rather than one lever among several.
