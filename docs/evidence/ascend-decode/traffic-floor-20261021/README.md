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
