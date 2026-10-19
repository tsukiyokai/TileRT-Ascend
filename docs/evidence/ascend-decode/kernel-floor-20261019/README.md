# A replayed kernel costs 19.30 µs, so the fixed cost is the kernel count

`tests/glm52/handoff/floor_kernel.cpp`, one die, vector only. Four bodies of
the same loop, measured both as a long loop divided by its iterations and as a
single launch.

| mode | an iteration | one launch | body |
|---|---|---|---|
| 0 | 0.014 µs | 26.700 µs | empty |
| 1 | 0.233 | 20.320 | one 32 KiB read |
| 2 | 0.274 | 31.061 | read + a vector pass |
| 3 | 0.699 | 17.450 | read + pass + write back |

The body's real work goes from 0.014 to 0.699 µs an iteration, so the spread
between those four launches — 17.45 to 31.06 — is noise. **A launch costs
20 to 30 µs and does not depend on what the kernel does.**

## The verify's fixed cost is that, times its kernels

The target graph launches 7467 kernels across the capture, which over about
four verifies is **1867 a verify**. The verify's fixed cost — the part that
does not move with rows, measured at 36.02 ms — divided by that count is
**19.30 µs a kernel**.

| | a kernel | a verify |
|---|---|---|
| a bare launch on an idle die | 24.00 µs | 44.80 ms |
| the in-graph dispatch floor | 1.20 | 2.24 |
| **what the fixed cost implies** | **19.30** | **36.02** |

19.30 sits between the two, which is where a replayed kernel should sit. So
the fixed cost is not a property of any operator; it is the number of them.

## What 2 ms requires, as an integer

| the verify must cost | kernels | from 1867 |
|---|---|---|
| 9.013 ms (a whole round's budget) | 467 | 4.0× fewer |
| 4 ms | 207 | 9.0× fewer |

## Every fusion on the board, re-priced

| kernels removed | a verify saves | share | the fusion |
|---|---|---|---|
| 1787 | 34.49 ms | 86.9% | a layer in one kernel, 1867 → 80 |
| 483 | 9.32 | 23.5% | every `Cast` folded into its producer |
| 305 | 5.89 | 14.8% | MegaMoE: the MoE block's 6 kernels into 1, 61 layers |
| 192 | 3.71 | 9.3% | attention's four operators into one, 64 layers |
| 61 | 1.18 | 3.0% | the two grouped matmuls into one |

A verify of one kernel a layer would be 1.54 ms, inside the 9.013 a whole
round may cost.

## Two of my own conclusions this overturns

**MegaMoE's fusion is worth 5.89 ms, not 0.78.** I priced it at a 1.2 µs
dispatch floor plus a 1.364 µs GM round trip, which is the cost of the
boundary and not the cost of the kernel. 7.5× too low.

**`Cast` is the second largest prize, not negligible.** This morning I took it
off the board because the device spends 2.86 µs a span on it. That measured
its work. As 1935 separate kernels in the capture, about 483 a verify, it
carries 9.32 ms of launches to do 1.4 ms of work — and it is the easiest of
the five to remove, since a type conversion belongs in its neighbour's
epilogue rather than in a kernel of its own.
