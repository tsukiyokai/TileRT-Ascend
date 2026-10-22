# Skewing pays only where the cube is already divided into invocations

The upper bound on skewed co-scheduling assumed the cube's work can be cut into
pipeline steps. It cannot be cut for free: each step is another matmul
invocation and each invocation walks the reduction again.

## The measurement

The front stage at `prolog`, twelve rows, one die. Zero steps runs the halves
one after the other with a barrier between, which is what a phase boundary
gives; more steps interleave the cube's step `s` with the vector's step `s-1`
inside one barrier region a step.

| steps | us at 2 vector passes | against serial | us at 4 passes | against serial |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 56.18 | — | 79.49 | — |
| 1 | 53.79 | −2.39 | 79.77 | +0.28 |
| 2 | 62.63 | +6.45 | 80.23 | +0.74 |
| 4 | 99.68 | +43.50 | 100.40 | +20.91 |
| 8 | 126.37 | +70.19 | 136.38 | +56.89 |

Monotonically worse with depth. At eight steps the extra 70 microseconds is
seven more walks of a 6144 reduction, which the increment fit prices at 4.41
microseconds per thousand of K. The vector work available to hide is 25
microseconds at two passes. The trade is 70 against 25.

## The criterion it gives

Skewing is worth it exactly when the cube work is **already** divided into
independent invocations, so interleaving adds barriers and no walks.

That is the difference between this result and the one that works. The front
stage is one invocation by design -- that is what earns its 26.87 microseconds --
so cutting it for a pipeline gives back more than the pipeline saves. The FFN's
expert loop is already one invocation an expert, three or four to a core, so
interleaving expert `e`'s activate with expert `e+1`'s gate/up costs only the
barriers. `ROUTED_SHARED_PIPELINE` demonstrates the same thing from the other
side, placing the shared expert's cube work beside the routed vector stages for
32 microseconds.

So the skew belongs in the expert loop and in the MLA tail's head loop, not in
the projections. And the earlier upper bound of 1.4 to 2.0 milliseconds, which
assumed the whole layer could be skewed, holds only for the part of the layer
whose cube work is already divided.

## Scope

The cube's product is exact at every depth, maximum difference zero against an
integer reference. The vector half's witness does not verify on the split arms:
it accumulates atomically across steps and no longer matches the closed-form
total, so the vector side has timing but not a correctness gate here. The
witness verified on the unsplit arms, where the vector work is one range.

[`run.sh`](run.sh) is the sweep; the JSON files are its output.
