# The traffic floor, and the target is inside it once the bank is narrowed

At batch one every operator in a layer is bandwidth-bound -- the grouped matmul
measured 1170 to 1255 GB/s, the projections 11 us each for about 12 MB, the
attention 48 GB/s at 196 us of which 170 was host. So a round's cost has a
physical floor: the bytes it must read, at the rate the die reads them.

A layer a rank, from the checkpoint's own shapes, at the 1200 GB/s the grouped
matmul achieves:

| item | MB | us |
|---|---|---|
| attention weights | 190.2 | 158.5 |
| KV cache, 8K context | 9.4 | 7.8 |
| expert bank, all sixteen | 302.0 | 251.7 |
| expert bank, the two touched | 37.8 | 31.5 |

    full bank    501.6 MB a layer   39.12 GB a round   32.60 ms   TPOT 5.99
    narrowed     237.3 MB a layer   18.51 GB a round   15.43 ms   TPOT 2.84

## Which is the first statement that the target is reachable

**2.84 ms.** Not a projection from a lever's ratio but the time to move the
bytes, at a rate this session measured on this die. The condition is the bank
narrowed to what a round touches and everything else running at bandwidth.

The measured round is 50.73 ms against a narrowed floor of 15.43, so the engine
pays 3.3 times its traffic. That ratio is what the levers and the queued arms
are about.

## And it names the next target

With the bank narrowed, the attention weights are 190.2 MB of the 237.3 a layer
-- 80 percent. Twelve times the KV cache and five times the narrowed bank.

Within them, `kv_b_proj` is `[28672, 512]` in **BF16**, which is 29.4 MB where
every other attention weight is int8. Carrying it at int8 would take the layer
from 237.3 to 222.6 MB, a floor of 14.47 ms and a TPOT of 2.66 -- and whether
that is numerically acceptable is a question for the checkpoint's own
quantisation, not for this page.

The larger question the floor poses: 190 MB of attention weights a layer a rank
at TP16 means 3.0 GB a layer across the node, which is the model's own size
divided by nothing. Reading all of it once a round is what a dense forward
costs, and no amount of speculation changes it -- only accepting more tokens a
round does, which is why 5.441 accepted is the other half of every figure here.

## Correction: the attention weights are a rank's share, 11.9 MB not 190.2

190.2 MB a layer is the whole layer across sixteen ranks -- this session had
already divided it once, recording 0.93 GB a rank over 78 layers. A rank's share
is 11.9 MB a layer.

| item | MB a rank | us |
|---|---|---|
| attention weights | 11.9 | 9.9 |
| KV cache, 8K, replicated | 9.4 | 7.8 |
| expert bank, all sixteen | 302.0 | 251.7 |
| expert bank, the two touched | 37.8 | 31.5 |

    full bank    323.3 MB a layer   21.01 ms a round   TPOT 3.86
    narrowed      59.0 MB a layer    3.84 ms a round   TPOT 0.71

## Which makes the bank the whole story

It is 93 percent of a layer's traffic at full and 64 percent narrowed. The
attention weights are 4 percent and the KV cache 3 percent, so the previous
section's claim that attention dominates and `kv_b_proj` is the next target is
withdrawn entirely -- it rested on the undivided figure.

## And it moves the floor far below the target

A narrowed round's traffic is **3.84 ms**, a TPOT of 0.71. The engine's 50.73 ms
is thirteen times that. So the target is not near the floor, it is well above
it, and what stands between is implementation rather than physics: per-call
costs, collectives, and whatever the arms attribute.

That is a better position than the previous section's 2.84 ms, which made the
target look like a near-miss against physics. It is not. The bytes are cheap;
the round is expensive for other reasons.
