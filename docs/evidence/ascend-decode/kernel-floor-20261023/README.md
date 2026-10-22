# The per-kernel floor, rechecked on today's stack: fusion saves 45 percent

2026-10-23, a3-180, one die, no model, `bench_kernel_floor.py`, with a tenant
holding 40781 MB a die -- which is why this was the measurement available.

The whole route to 2.x rests on one claim: a round's cost is its pass count,
not any pass's inner loop, so fusion is the only lever. The numbers behind it
were measured on another host and another CANN. Rechecked here:

| operation | median | min | p90 |
|---|---|---|---|
| add_32_rows | 11.836 us | 10.751 | 12.401 |
| foreach_add_32_rows | 51.752 | 50.626 | 53.381 |
| cast_round_trip_8_rows | 26.965 | 25.818 | 29.135 |
| scale_then_add_8_rows | 27.635 | 26.238 | 29.788 |
| fused_axpy_8_rows | 15.083 | 13.656 | 15.595 |

A kernel that moves almost nothing costs **11.836 us**, which is the 12 us a
pass the megakernel ledger prices with. Two steps as two kernels cost 27.635;
the same arithmetic as one kernel costs 15.083 -- **45 percent less, and the
difference is 12.55 us, one kernel floor**. So the claim holds on this stack
and with this CANN, and it is the saving per fused pass rather than an
extrapolation.

`foreach_add_32_rows` is 4.4 times `add_32_rows`: the batched API pays more
floor than it saves at this size, so it is not a fusion route.
