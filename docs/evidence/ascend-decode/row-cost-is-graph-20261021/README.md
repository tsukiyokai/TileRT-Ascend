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
