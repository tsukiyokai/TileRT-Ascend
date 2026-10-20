# The residual 5 percent: two mechanisms excluded, cause open

2026-10-21. Constant weights, sixteen experts, one row each.

    vendor   absmax 0.45117
    kernel   absmax 0.47461    ratio 1.0520, max difference 0.126953

A constant weight makes the block analytic -- every code one, every bias zero,
so a row's gate-up product is the row's own sum. Computed in float64 on the
host:

    analytic                           0.343068
    recombination coefficient 15.0     0.321931
    recombination coefficient 16.0     0.343068
    recombination coefficient 17.0     0.362906
    without the middle quantisation    0.343163

## What this excludes

The MSD recombination coefficient: reaching 0.45 from this model needs a
coefficient near 21, which is not a candidate value.

The intermediate quantisation: removing it entirely moves the result by three
parts in ten thousand, 0.343068 to 0.343163. It cannot produce five percent.

## What it says about the model

Both measured sides exceed the analytic value by 30 to 38 percent, so the
analytic model is the one that is wrong and cannot adjudicate between them.
The likeliest error in it is the per-token quantum: it takes each expert's
first slot where each row carries its own.

## Why this is now parked

The kernel's value is established at 8.48 ms a round and this error is near
the quantisation noise of a W4A8C8 checkpoint, so it does not block the
engine-side measurement that matters next. It is recorded as an open gap
rather than chased further -- the cost of one more hypothesis here exceeds
what it buys.
