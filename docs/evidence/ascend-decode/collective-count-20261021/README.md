# Forty-one collectives a round, at 830 microseconds each

From `profile-rope-20261018`, rank 1, 3.13 rounds by attention call count.

    hcom_allGather   68 calls   21.7 a round   882.5 us each   19.18 ms a round
    hcom_allReduce   60 calls   19.2 a round   831.1 us each   15.94 ms a round

Forty-one calls a round, 35.1 ms of device time, against a measured round near
52.8 ms. The halves overlap so not all of it is on the critical path, but it is
the largest single item in the capture by a wide margin.

## What this overturns

The one-sided reduce was costed at 6.5 ms a round from an operator comparison:
32.488 us for the push against HCCL's 201.995 at sixteen ranks, times the 156
calls a TP16 decode was assumed to make -- two a layer over 78 layers.

Both numbers were wrong. The engine makes 41 calls a round, not 156, and each
costs 831 us, not 202. So the comparison was between the wrong message size and
the wrong call count, and it happened to land in the right order of magnitude
for the wrong reasons.

## The question this raises

Twenty-two allGathers a round does not correspond to any obvious layer
structure: 78 layers divided by 21.7 is 3.6. And an 830 us collective at TP16
is large -- the one-sided probe moved its payload in 32 us. So either these
carry far more data than a hidden-state reduction, or they are waiting rather
than transferring.

Which makes the next question concrete and machine-free: find the call sites.
If any of the 41 are redundant, that is worth more than every operator lever
measured today put together.
