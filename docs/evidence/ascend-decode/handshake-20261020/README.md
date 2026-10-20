# Which cross-core handshake survives being load-bearing

`a3-187`, one die. Each variant runs as its own process under a twelve-second
timeout: finishing is live, the timeout is a deadlock. The handshake runs
once or four times a launch, because a surplus set passes once and then
races.

The cube-to-vector direction was already settled -- mode 2, and one set
raises the counter of both of a cube's vector subblocks. The other direction
had never carried a dependency: the shared expert's chain set a flag that
way, but the ordering it relied on came from the `SyncAll` after it, so a
wrong mode there could not show. Making it load-bearing hung the routed
experts' first launch with no output at all.

| reverse mode | both subblocks set, cube waits once | subblock zero sets, cube waits once | both set, cube waits twice |
|---|---|---|---|
| 0 | deadlock | deadlock | deadlock |
| 1 | deadlock | deadlock | deadlock |
| 2 | **live** | deadlock | deadlock |
| 3 | **live** | **live** | **live** |

Identical at one round and at four.

So mode 2 in the reverse direction admits exactly one shape, the obvious
reading of this project's own rule -- split the work, never the handshake --
and it is not the shape the routed experts want, because duplicating the
narrow activation on both subblocks would duplicate the write of the
intermediate. Mode 3 is insensitive to how many set and how many wait, which
lets subblock zero answer alone.

## Liveness is the signal; the markers are not

Each core stamps its progress into its own 32-byte block. Those values
disagree between sweeps -- mode 2's live variant reported the cube at 402 in
one sweep and at -1 in the next, and mode 3's reported a subblock one round
behind -- because the stamp is a scalar store to global memory followed by a
cache flush, and this project already knows that path does not reliably land
from a vector core. Liveness is what the timeout measures, and that is what
the table reports.

## Mode 3 orders, and the proof is elsewhere

Liveness alone cannot tell a working handshake from one that never blocks, and
a mode that never blocks would be live in every pattern. The routed experts'
correctness check settles it: with mode 3 and subblock zero answering alone,
the chain is right at 0 of 540672 values. A handshake that did not order would
have the down projection read an intermediate still being written.
