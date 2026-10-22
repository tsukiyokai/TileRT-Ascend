# The front stage built in Ascend C: 33.88 microseconds a layer

The layer's first dependency level now runs as one matmul invocation instead of
four. The kernel is [`tests/glm52/megakernel`](../../../../tests/glm52/megakernel/README.md);
`MEGAKERNEL_FRONT_SPLIT=1` builds the control from the same source, the same
tiling and the same concatenated output, differing only in how many invocations
each core issues.

## The comparison

One idle die of a3-194, CANN 9.2, four widths, two alternating passes a width,
eight changing inputs and five timing repeats a pass.

| rows | fused us | split us | saving us | digests equal |
| ---: | --- | --- | ---: | --- |
| 1 | 48.00, 47.97 | 81.30, 81.21 | 33.30 | 8 of 8 |
| 2 | 48.39, 48.07 | 81.14, 81.26 | 32.87 | 8 of 8 |
| 12 | 48.37, 48.52 | 82.40, 82.34 | 33.88 | 8 of 8 |
| 32 | 49.24, 49.59 | 85.58, 85.03 | 35.99 | 8 of 8 |

The two passes of a width agree to within 0.4 microseconds and the arms' sample
ranges do not overlap. At the engine's twelve rows the saving is 33.88
microseconds a layer, or 2.64 milliseconds a round.

## It computes the same thing

Every width runs eight changing inputs through both arms, and the product's
digest is identical between them at every input. That is the check that matters:
concatenating the output axis changes which invocation computes a column, not
the reduction that produces it, so anything other than equality would mean the
slicing is wrong. Both arms also match a float reference to 5.5e-4 against
magnitudes near 320, which is bf16 accumulation.

## It is the reduction depth, not the rows

The fused arm costs 48.00 microseconds at one row and 49.59 at thirty-two, a
rise of 1.6 over a 32-fold change in rows. The split arm rises 4.3 over the same
range, because it pays the depth four times. This is the same shape the
shape probes found and the reason the concatenation works.

## Scope

bf16 operands and a float product, which isolates the invocation count from the
quantization contract. The deployed projections are W4A8 with per-channel
scales, and the concatenated weight layout is a load-time step; neither is built
here. One die, no tensor-parallel reduction, no model.

[`run.sh`](run.sh) is the alternating runner and the JSON files are its output,
one a width a pass an arm.
