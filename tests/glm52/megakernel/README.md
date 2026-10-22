# The megakernel's front stage

A decode matmul's cost follows its reduction length, paid once an invocation,
while its output axis is spread across cores for free. See
[the measurements](../../../docs/evidence/ascend-decode/megakernel-design-20261022/README.md):
at K=16384 and twelve rows, widening the output axis 24-fold costs 1.5 times the
wall clock, and splitting one invocation into four at fixed bytes costs 4.4
times as much.

So the layer's first dependency level belongs in one invocation. Four
projections read the normalized hidden at K=6144 -- the query latent at 2048
columns, the key-value latent with its rope at 576, the indexer's key at 160 and
the router's gate at 256 -- and this entry computes all 3040 columns in a single
mixed AIC/AIV launch, giving each core one contiguous slice aligned to 32.

`MEGAKERNEL_FRONT_SPLIT=1` builds the control arm. It is the same kernel, the
same tiling and the same concatenated output; only the invocation count differs,
because each core issues one call a projection instead of one over its slice.
`orgKc` holds the product's row stride at the full width whatever a core
computes, so both arms write the same layout.

## Measured

One die, four widths, two alternating passes a width, eight changing inputs:
the fused arm costs 38.43 microseconds at twelve rows against the split arm's
82.62, a saving of 44.36 a layer or 3.46 milliseconds a round. Both arms produce
identical product digests at every input and every width. The fused arm is flat in rows -- 38.18 at one and
38.27 at thirty-two -- while the split arm rises 4.4, because it pays the
reduction depth four times. Its weight row stride must be 512-byte aligned,
which the contract enforces: 3040 columns cost 48.12 microseconds and the padded
3072 cost 38.43. See
[the evidence](../../../docs/evidence/ascend-decode/megakernel-front-20261022/README.md).

## Scope

bf16 operands and a float product, which isolates the invocation count from the
quantization contract. The deployed projections are W4A8 with per-channel
scales; that and the load-time weight concatenation are separate steps. Rows 1
to 32, blocks up to 24, one die.

## Build and check

    bash tests/glm52/megakernel/build.sh                      # fused
    MEGAKERNEL_FRONT_SPLIT=1 bash tests/glm52/megakernel/build.sh
    python3 -I tests/glm52/megakernel/check.py \
        tests/glm52/megakernel/build-fused/megakernel_front.so --rows 12

`check.py` runs eight changing inputs through the arm, compares each product
against a float reference, and reports a digest of the product so the two arms
can be compared exactly. They must agree digest for digest: the concatenation
changes which invocation computes a column, not the reduction that produces it.
