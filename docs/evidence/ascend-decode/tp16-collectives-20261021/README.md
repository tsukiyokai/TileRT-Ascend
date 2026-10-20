# Sixteen-rank collectives are pure latency, and they are the missing 374 us

2026-10-21, a3-84, sixteen ranks on one node, bf16 over hidden 6144. Needs
dies but almost no memory, which is what a tenant holding 55 GB of each leaves.

| rows | KB | all_reduce | reduce_scatter | all_gather |
|---|---|---|---|---|
| 1 | 12.0 | 289.62 | 292.96 | 332.53 |
| 4 | 48.0 | 292.16 | 314.89 | 309.25 |
| 12 | 144.0 | 304.12 | 277.27 | 282.01 |
| 24 | 288.0 | 290.53 | 266.67 | 286.32 |
| 48 | 576.0 | 306.08 | 268.72 | 423.56 |

Flat. Twelve kilobytes and 576 cost the same to five percent, so at decode
sizes a sixteen-rank collective is **latency, not bandwidth**: 270 to 330 us a
call whatever it carries.

## Which closes the 2.7x

The engine fits 588 us a layer; the same operators replay in 214 on one die
with no collective in them. The difference is 374 us a layer, and one
all_reduce plus one all_gather at these sizes is 575 us. The engine's
SuperKernel absorbs most of its collectives -- the standalone count is 0.26
calls a layer -- so a layer carries a fraction of that pair, and 374 sits
inside it.

So the unattributed four fifths of the round has a name and a measurement, and
it is the one thing the synthetic layer omitted all along.

## What it rules out, and what it leaves

Ruled out: narrowing the draft, reducing the width, shrinking any message. The
cost does not follow the bytes -- 48x the payload costs 6 percent more.

Left: fewer calls, or overlap. And the overlap measurement from earlier today
says vector work hides under cube work but cube does not hide under cube; a
collective is neither, so whether it overlaps compute is a separate question
this does not answer.

## A port collision worth recording

The run failed three times with `EI0020`: the tenant's job holds the NPU
socket port 16666 on every die. `HCCL_IF_BASE_PORT` does not move it --
`HCCL_NPU_SOCKET_PORT_RANGE=27000-27200` does. Sixteen-rank collective work is
therefore possible on a machine whose dies are memory-held, which is worth
knowing for every future day the machine is busy.

## Explicit overlap with compute is slower

Sixteen ranks, twelve rows, an all_reduce against a matmul:

    compute alone               89.14 us
    all_reduce alone           300.46
    sum                        389.61
    larger of the two          300.46
    serial, one stream         343.01
    concurrent, two streams    376.70
    concurrency gains          -33.69 us

Serial on one stream is already 46 us below the sum, so the hardware overlaps
part of it unasked. Putting the collective on a side stream behind an event
costs 33.69 us more than that -- the third time today the same shape appears:
making a dependency explicit serialises what was already overlapping and adds
the event's cost. The other two were vector-under-cube and the MoE fork.

## So the collective's three exits are all measured

    smaller messages   flat to five percent over a 48x payload range -- no
    explicit overlap   33.69 us worse than leaving it alone
    fewer calls        the only one left

And fewer calls is already mostly spent: the SuperKernel absorbs them, leaving
0.26 standalone calls a layer. What a round pays is largely inside that kernel,
where the count is not a host-side choice.

## Which is the honest shape of the remaining gap

The round is compute plus collectives, both now measured: 214 us a layer in a
graph and 374 us a layer of collective latency. The bank narrowing takes 59.4
from the first. Nothing measured today takes anything from the second, and its
three mechanisms are each closed by a measurement rather than by argument.

## The fourth exit: 193 us of it is independent of the rank count

One all_reduce of 144 KB, one world size a run:

| ranks | us |
|---|---|
| 2 | 193.60 |
| 4 | 193.11 |
| 8 | 238.27 |
| 16 | 296.21 |

Two ranks already cost 193 us. So the rank count explains 103 of the 296, and
**193 us is a fixed per-call cost** -- not bandwidth, not topology, not the
number of participants.

## Which explains why capture cannot touch it

A kernel's issue under graph replay is 1.064 us, measured earlier in this
project. A collective's fixed cost is 193, a hundred and eighty times that. So
the collective's overhead does not sit on the path a graph captures, and the
0.877 capture ratio of the operators around it is the same story: what capture
removes is host dispatch, and this is not host dispatch.

## And it prices the one structural alternative

Dropping to eight ranks would save 58 us a call and to four 103 us, at the cost
of doubling or quadrupling the per-rank compute. At batch one that trade is
less lopsided than it looks, because every operator is flat in rows and the
grouped matmul's cost follows its bank -- a rank holding twice the experts pays
7.53 us each for twice as many. Sixteen experts at 7.53 is 120 us a layer; at
TP8 a rank would hold thirty-two.

So the trade is 58 us of collective against roughly 120 us of bank, and it
loses. TP16 is the right rank count for this workload, which is now measured
rather than assumed -- and the 193 us floor is what no rank count avoids.

## The 193 us is synchronisation, not transport

144 KB out of die 0, one process, two devices, no HCCL:

| destination | us | GB/s |
|---|---|---|
| die 1 | 219.64 | 0.66 |
| die 2 | 217.98 | 0.66 |
| die 4 | 244.03 | 0.59 |
| die 8 | 218.50 | 0.66 |
| die 15 | 218.51 | 0.66 |
| pinned host memory | **37.09** | |

Two readings, both useful. A die-to-die copy through torch is 218 us -- worse
than HCCL's 193 us floor, and flat across every destination, so building a
reduction from peer copies is not the way out. And the same payload reaches
pinned host memory in 37.09 us.

So moving 144 KB off a die costs 37 us. HCCL's 193 us per call is therefore
**synchronisation and setup, not transport**: five times the cost of the data
movement it performs, and independent of the rank count, the payload and the
destination.

## Which is the honest end of this thread

Four exits closed by measurement -- smaller messages, explicit overlap, fewer
calls, fewer ranks -- and now the fifth, a hand-rolled primitive, closed too:
the peer copy is more expensive than the collective it would replace.

What remains is the gap between 37 us of transport and 193 us of call: a cost
that is neither bandwidth nor participants nor host dispatch. Whatever it is
lives inside HCCL, and 78 layers of it is the largest single item in this
project's round.
