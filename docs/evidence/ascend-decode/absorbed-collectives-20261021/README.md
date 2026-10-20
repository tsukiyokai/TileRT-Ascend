# A reduction absorbed into a super kernel costs 30 microseconds, not 858

From `profile-rope-20261018`, rank 1, 3.13 rounds. Super kernels whose name
declares a collective:

    1.0/round   3282.3 us   3.15 ms   sk_1277_full_model_start_aiv_all_reduce_float_
    1.0/round     31.1 us   0.03 ms   sk_1413_...
    1.0/round     31.0 us   0.03 ms   sk_1501_...
    1.0/round     29.6 us   0.03 ms   sk_1305_...
    1.0/round     29.5 us   0.03 ms   sk_1529_...
    1.0/round     29.4 us   0.03 ms   sk_1345_...
    absorbed total 3.29 ms a round

Against the standalone collectives: 20.4 calls a round at 858 us, 17.56 ms.

So the engine already absorbs some reductions into super kernels, and an
absorbed one costs about 30 us where a standalone one costs 858 -- a factor of
28. `sk_1277` at 3282 us is the exception and is a single call a round, so it
carries something else.

## Which redirects the largest lever

The one-sided push was the candidate for the 17.56 ms. Two-rank scaling says it
is a latency optimisation: 9.91x at the decode shape's 288 KB, 2.17x at 1.5 MB,
and a loss at 6 MB. At 858 us a call the standalone collectives are well past
that crossover, so the push probably does not help them.

Absorption does, and the engine demonstrates it: 30 us against 858. If the 20.4
standalone calls were absorbed the way six already are, 17.56 ms becomes about
0.6 -- **a saving near 17 ms a round**, larger than every lever measured today
together, and the mechanism is already in the codebase rather than needing a
new primitive.

## What to find out next

Why those six are absorbed and the twenty are not. The names are
`sk_<id>_full_model_start_aiv_all_reduce_float_end_...`, so the compiler emits
an absorbed form when some condition holds; the standalone ones are where it
does not. That condition is the lever, and it is a question about the
compilation pass rather than about an operator -- which makes it answerable by
reading, without the machine.
