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
