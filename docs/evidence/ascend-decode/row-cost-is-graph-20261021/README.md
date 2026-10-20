# The row cost is entirely the graph's, cold as well as warm

A layer's operators over six rotating 77 MB weight sets, so nothing is
resident:

    rows   us a layer   marginal us a row
       8      126.969
      12      116.493          -2.619
      16      116.427          -1.318
      24      117.941          -0.564
      32      121.955          -0.209

Flat, and negative at the margin. Warm it was flat too -- 8.07 us across the
same range. So the operators do not care how many rows they process, whether or
not their weights are resident, and over 78 layers the row cost on the operator
side is between 0 and -0.02 ms.

The engine's round model fits 0.66 ms a row.

## What this establishes

The entire row cost is graph machinery. Capture fixes the shapes into the task
list, and a list built for more rows costs more to replay even though every
operator in it is flat in rows -- measured directly: replay rises 527.0 to
582.9 us across a fourfold row range while the same sequence eagerly goes 568.7
to 554.4.

## And what it does to the target

The copy oracle saturates at 13.474 accepted by budget 64, which is what 2.9 ms
needs. The only thing stopping it is 64 rows at 0.66 ms:

    row cost 0.66 ms   TPOT 5.37 ms
    row cost 0.20          3.19
    row cost 0.10          2.72
    row cost 0.00          2.24

So 2.x is reachable by recall alone, and the single question between here and
there is why graph replay charges for rows whose operators do not. That is an
engineering question about capture, answerable on one die, and it is now the
most valuable open item in this project -- worth more than every operator lever
measured today put together.

## Where to look

Capture bakes a tiling choice per operator for the captured shape. An operator
flat in rows eagerly chooses its tiling from the actual shape each call; in a
graph it is fixed at the captured width. So a graph at width 64 may be running
every operator with a tiling meant for 64 rows while only a few carry data --
and the cost would then be the tiling's, not the rows'. The test is to capture
at one width and compare against capturing at another with the same data, which
the queued width arms do end to end and a synthetic graph can do on one die.

## The multi-size capture's memory cost

Eight layers a graph, five sizes captured and held together on one die:

    size   reserved MiB   delta MiB
       2          116.0         22.0
       4          136.0         20.0
       8          156.0         20.0
      16          176.0         20.0
      24          196.0         20.0

About 20 MiB a size for eight layers, so roughly 195 MiB a size at 78 -- and
the increment does not depend on the size, which says the cost is the task list
rather than any data, consistent with the row cost being capture's.

Five sizes at the engine's depth is about 1 GiB against the 48 GiB a rank holds
at 0.78 utilisation with the model taking 17. So the lever cannot fail for
memory, and the configuration caps the list at the requested width -- {2, 4, 8,
16, 24} at width 24.

## And holding several graphs does not slow the small one

    size 4, captured alone            516.989 us
    size 4, four larger held too      516.696 us   ratio 0.999
    size 32, held alongside           583.245 us   12.9 percent dearer

So the lever's premise holds: a small graph replays at its own cost whether or
not larger ones are resident, and the small one is 12.9 percent cheaper than
the wide one -- 0.65 ms a round at this synthetic layer's five operators.

The engine's layer has about 7.8 kernels including the attention and the MoE,
so the spread there should be wider than 12.9 percent; the round model's 0.66 ms
a row across a 24-row width is 15.8 ms, which is 29 percent of the round. What
the arms measure is where in that range the real engine sits.

## The chain, with every link measured

    round = 45.88 ms + 0.66 a row            arms' own records, within-arm
    operators flat in rows, cold and warm    126.969 -> 121.955 us a layer
    so the row cost is capture's             replay 527 -> 583, eager 569 -> 554
    copy oracle saturates at 13.474          the engine's own token stream
    2.9 ms needs 13 at the merged fixed term 30.15 ms
    removing the row cost reaches 2.24       arithmetic on the above
    the removal is a list, not a rewrite     cudagraph_capture_sizes
    several sizes cost 195 MiB each          synthetic capture
    and do not slow each other               0.999
