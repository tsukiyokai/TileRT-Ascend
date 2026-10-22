# The FFN's vector half is idle 82 percent of its window

The routed body's own stage trace already holds the decomposition, and it says
the two halves of the machine take turns idling. At twelve rows, 24 cube cores
and 48 vector cores, medians over the per-core records:

| half | work us | barrier us | idle share |
| --- | ---: | ---: | ---: |
| cube | 71.25 | 67.52 | 49% |
| vector | 23.40 | 111.74 | 82% |

The window's median is 136.40 microseconds, and each half's work plus barrier
accounts for it: 138.77 for the cube and 135.14 for the vector. Total work is
94.65 microseconds in a 136.40 window.

## Where each half waits

| stage | cube work | cube wait | vector work | vector wait |
| --- | ---: | ---: | ---: | ---: |
| restore_dispatch | 0.17 | 6.90 | 1.07 | 2.28 |
| metadata_and_split | 0.26 | 15.30 | 6.26 | 9.29 |
| gate_up_matmul | 40.35 | 10.94 | 0.02 | 51.31 |
| activate | 0.28 | 11.54 | 4.04 | 7.79 |
| down_matmul | 29.65 | 6.52 | 0.02 | 36.15 |
| store_routes | 0.28 | 12.93 | 9.87 | 3.33 |
| combine | 0.27 | 3.38 | 2.12 | 1.60 |

The vector cores lose 87.46 microseconds waiting out the two matmuls, and the
cube loses 39.77 waiting out the vector stages. The remaining 27.75 of cube wait
is cube waiting on cube -- the expert loop's imbalance.

## What closes it

Within one expert the chain is strict: metadata feeds gate/up, which feeds
activate, which feeds down, which feeds the store and the combine. So the
vector work cannot hide under the cube work it produces. But a round touches
about five experts a rank, and expert `e`'s activate is independent of expert
`e+1`'s gate/up. Pipelining the loop puts each half's work under the other's.

The cube's 71.25 microseconds is the floor. Perfect pipelining would leave
roughly that plus one expert's fill, call it 75, against today's 136.40 -- about
61 microseconds a layer, or 4.76 milliseconds a round. It is the largest single
mechanism still on the table, and the shared expert already demonstrates it:
the `ROUTED_SHARED_PIPELINE` variant places shared cube projections beside
routed vector stages and measures 187.17 against 219.24 microseconds at twelve
rows, a 32 microsecond saving.

This also sizes the capacity measured separately. A cube stage hides about 10.5
microseconds of vector work for nothing, and the FFN's whole vector half is
23.40 -- so two cube stages' worth of capacity covers all of it.

[`decompose.py`](decompose.py) reads the trace; [`width12.txt`](width12.txt) and
[`width32.txt`](width32.txt) are its output. The trace is
`megakernel-device-20261021/a3-84/routed-stage-profile.json`, taken with the
instrumented library, so the absolute window carries its instrumentation cost;
the split between work and waiting is what this page uses.
