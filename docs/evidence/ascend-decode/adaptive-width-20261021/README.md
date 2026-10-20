# Sizing the proposal by the last acceptance buys 6 to 9 percent of TPOT

2026-10-21, CPU, 34 source files from this repository, width ceiling 32.

| edit rate | automaton @32 | adaptive | mean width | TPOT @32 | TPOT adaptive |
|---|---|---|---|---|---|
| 0.02 | 10.020 | 8.318 | 12.6 | 7.18 | **6.75** |
| 0.05 | 5.434 | 4.362 | 8.1 | 13.24 | **12.02** |

The adaptive column accepts less and costs less. A round is 45.88 ms plus
0.814 a proposed row, so thirty-two rows to have eight accepted pays for
twenty-four that were never going to hold; twice the last round's acceptance
keeps 83 percent of the acceptance for 39 percent of the width.

## Two signals tried, one wrong

The first attempt sized the proposal by `matched_length` -- how long a suffix
the automaton found a continuation for -- on the reasoning that a long match
means a copy stretch. It measured worse at every edit rate, and the mean width
column said why: the matched length is typically under nine on this corpus, so
a threshold at eight almost never fired. The automaton's advantage does not
come from long matches; it comes from the most frequent continuation of a
short one. That was a wrong account of its own mechanism.

## And one measurement construction that had to be fixed

Scoring the adaptive column inside the existing loop made it worse still,
because every column there shares one advance -- `step` is the best column's
acceptance -- so a column that proposed four rows was advanced ten and its
feedback loop locked at the narrowest width. It needs its own trajectory,
which doubles the evaluator's cost and is why the sweep now runs two edit
rates rather than four.

## Where it sits

Orthogonal to the tree. The tree raises acceptance (14.510 at 0.02) and needs
a per-row attention mask the kernel does not take; this lowers the width's cost
and needs nothing. Both are needed: engine work reaches about 40 ms a round,
and 2.9 ms at 40 ms needs 13.8 accepted, which the tree reaches at a low edit
rate while this makes each round cheaper.
