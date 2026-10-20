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

## The SwiGLU's form is excluded, and absmax cannot see it anyway

The vendor takes `swiglu_limit` and the engine passes 0.0 while the kernel has
no limit, which made the activation's form a candidate. Computed on the host
over the constant-weight case's own distribution, against the measured ratio of
1.0520:

    no limit                   absmax ratio 1.0000
    gate clamped at 0 above    absmax ratio 0.0002
    gate clamped at 0 below    absmax ratio 1.0000
    product clamped at 0       absmax ratio 1.0000
    silu with beta 1.702       absmax ratio 1.0000

None produces 1.0520. And the four that give exactly 1.0000 show why this
comparison was weak to begin with: under a constant weight the absmax is set by
a single row, and every variant agrees on that row's sign, so absmax has almost
no power to distinguish activation forms. That is lesson 66 again -- a
criterion has to be able to fail for the reason you care about.

Six candidates are now excluded: the MSD recombination coefficient, the
intermediate quantisation, a non-contiguous scale tensor, the per-token scale
slots, matmul reuse across calls, and the activation's form.

The gap stays parked. The kernel's value is the launch merge, measured as a
ratio with both sides under identical conditions -- 8.54 ms a round cold -- and
five percent sits inside a W4A8C8 checkpoint's quantisation noise. What blocks
progress is the absent steady-state capture and the absent end-to-end run, not
this.
