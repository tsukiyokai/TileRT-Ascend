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
