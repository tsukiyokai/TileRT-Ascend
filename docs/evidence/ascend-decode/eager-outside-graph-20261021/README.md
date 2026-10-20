# A round makes 1108 eager calls outside its graph, worth 11.4 ms of dispatch

The census splits the capture's kernels by name family. `sk_*` is the captured
super kernel; anything beginning `aclnn` is an eager aclnn call, which pays host
dispatch -- 11.386 us measured in this project against 1.064 us under graph
replay.

Per round, from `profile-rope-20261018` rank 1 over its three rounds:

| family | calls a round | device ms a round |
|---|---|---|
| collective | 43 | 36.62 |
| super kernel (captured) | 637 | 33.58 |
| aclnn (eager) | **1108** | 7.40 |
| other | 81 | 0.42 |

    1108 eager calls at 11.386 us dispatch = 12.62 ms
    the same under replay at 1.064 us      =  1.18 ms
    difference                               11.44 ms a round

Against the round's unaccounted 12 ms. The match is close enough to name the
residual, and the remedy is the one the engine already applies to the rest of
the model: capture.

## It also corrects the layer figure

The super kernel is not 78 calls of 210 us. It is **637 calls a round totalling
33.58 ms** -- so the layer's own device time is twice what the 210 us reading
suggested, and the earlier 16.4 ms was the cost of 78 of them rather than all
637.

Device time now sums above the measured round (36.62 + 33.58 + 7.40 = 77.6 ms
against 50.710), which means the streams overlap. That is consistent with the
earlier finding that a single stream already recovers part of a serial sum, and
it means device totals cannot be added into a round budget without knowing the
overlap.

## What is actionable

The eager calls are the residual and they are host dispatch, not device work --
7.40 ms of device time against 12.62 of dispatch. Which of the 1108 are
capturable is the next question, and the names say where to look: the census's
largest eager families were `aclnnInplaceCopy_CastAiCore_Cast` at 530 a round
and `aclnnMuls` at 75.

## Where the 530 casts a round come from

`refresh_hoisted_rope` -> `refresh_rope`, in `cann_model.py`:

    torch.index_select(rope._cos_cache, 0, positions, out=rope._cos_mla[:width])
    torch.index_select(rope._sin_cache, 0, positions, out=rope._sin_mla[:width])
    for cos, sin in destinations:
        cos.copy_(rope._cos_mla[:width])
        sin.copy_(rope._sin_mla[:width])

`aclnnInplaceCopy_CastAiCore_Cast` is `copy_`'s kernel, and 530 a round over
two copies an entry puts about 265 entries in `destinations`.

They are not redundant. `rope_destinations` already deduplicates by
`data_ptr()`, and its comment names the reason: a builder configured for
speculative decoding keeps independent RoPE storage and copies the shared
gather into it, so every distinct buffer has to be written every round.

## But the addresses are fixed, so the copies are capturable

The destinations come from hoisted metadata -- the addresses a replay captured
-- so they do not change between rounds. `index_select`'s output and every
`copy_`'s target are the same pointers every time, which is exactly the
condition a graph needs.

So the 11.44 ms is not work to remove but dispatch to capture: 1108 eager calls
at 11.386 us against 1.064 under replay. The engine already captures the model
forward; this is the per-round metadata refresh that sits outside it.

Two ways in, both untried: capture the refresh in its own small graph replayed
before the model's, or make the speculative builders share the MLA buffers so
`destinations` is empty and the loop disappears. The first is ours to do; the
second is a vLLM-Ascend builder configuration.

## The pattern is capturable, 22.5x, and bit-exact

One die, the refresh's shape replicated with 265 destinations -- two
`index_select` calls writing into fixed outputs, then 530 `copy_` calls:

    eager          7588.89 us
    graph replay     336.63 us   22.54x
    saved            7252.26 us a round
    bit-exact after a replay at new positions: True

The correctness check is the one that matters: `positions` is moved to a
different range, the graph is replayed, the result is kept, and then the eager
path is run at those same positions. They agree exactly -- so the capture holds
the addresses and reads whatever the buffer contains at replay, which is the
premise `RopeGraph` rests on.

And the eager figure is higher than the estimate: 7.59 ms measured against 6.06
ms from 532 calls at 11.386 us. So the dispatch estimate was conservative.

## Which makes this the session's largest validated lever

7.25 ms a round on a 50.710 ms round, mechanism measured on a die and
bit-exact, with the engine-side implementation committed behind `--rope-graph`.
What remains is running it in the engine, which needs the sixteen ranks the
tenant holds.

## The full list, and attention is on it

| kernel | a round | dispatch ms |
|---|---|---|
| `aclnnInplaceCopy_CastAiCore_Cast` | 530 | 6.03 |
| `aclnnFusedInferAttentionScoreV4` | **81** | 0.93 |
| `aclnnAbs_AbsAiCore_Abs` | 78 | 0.89 |
| `aclnnMuls_Muls_Muls` | 75 | 0.85 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 63 | 0.72 |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 38 | 0.43 |
| `aclnnLayerNormWithImplMode` | 24 | 0.28 |
| `aclnnIndex_IndexCheck_IndexCheck` | 19 | 0.22 |
| total | 1108 | 12.62 |

    the rope graph captures 580 of them, 6.61 ms
    the remainder is 528 calls, 6.01 ms

## Attention runs eager, once a layer

Eighty-one FusedInferAttentionScore calls a round is 78 layers plus three, so
**every layer's attention is an eager call** rather than part of a super kernel.
Each pays 11.386 us of dispatch and, from today's measurement, about 120 us of
the call's own argument handling -- seventeen parameters including Python lists
rebuilt every time.

`Abs` at 78 and `Muls` at 75 are also once a layer: the dynamic quant's absmax
and scale. `Matmul` at 63 is most of the projections.

So of the 528 calls the rope graph leaves, 297 are the once-a-layer items --
attention, the quant's two parts, and the projections -- and all of them are
outside the captured graph.

## Which is a larger lever than the rope refresh

The rope graph removes 6.61 ms of dispatch. The once-a-layer items are 3.39 ms
of dispatch by the same count, but attention alone carries about 120 us a call
of argument handling that the dispatch figure does not include: 78 calls times
120 us is **9.4 ms a round**, which no lever has touched and which capture
removes by construction.

That is the largest single item left in the host account, and it follows from
the same mechanism already implemented for the rope refresh.
