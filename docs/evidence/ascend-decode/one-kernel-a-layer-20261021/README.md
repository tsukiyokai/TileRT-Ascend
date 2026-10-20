# The target, restated: the six small operators at zero

The layer post-merge on one die, and an exact coincidence.

    the layer                       317.475 us
      attention                      96.959
      grouped matmul                 61.910
      the two together              158.869
      the six small ones            158.606   half the layer

    2.9 ms at the measured acceptance ceiling of 4.289 needs
      per layer                     159.463 us
      the two large ones alone are  158.869   a difference of 0.6

So the target is not an unspecified factor of two. It is the six small
operators costing nothing: 158.6 us a layer, 12.37 ms a round, for operators
whose own arithmetic at five rows is negligible. Twenty to forty-five
microseconds for twelve rows of a projection or a norm is the boundary, not the
work.

## As launches

    now        8 operators x 78 layers   624 a round
    folded     2 operators x 78 layers   156 a round
               468 boundaries removed at 26.4 us each = 12.37 ms

## Why this is a different claim from the one already refuted

Cross-layer fusion was measured and closed: nesting eight layers in one launch
costs 6.15 times one, because the layers crowd one reused scratch. That is the
**across** direction. This is the **within** direction, which is where the
15.73 ms of merges already came from -- four routed launches into one, four
projections off the hidden states into one. The mechanism is proven at that
scale; the question is how much further it goes inside a layer.

## What stands between

Six operators folded into two means the norm, the dynamic quant, the input
projection, q_b, o and the shared down projection all execute inside the
attention kernel or the grouped matmul kernel, sharing their tiles and never
returning to global memory. Three facts bear on whether that is reachable:

- The grouped matmul is flat in rows from 16 to 256 (+2.8%), so it has idle
  vector capacity at five rows -- the small operators are vector work.
- A 72 MB bf16 matmul runs at 2264 GB/s with one weight repeated against 1136
  with four distinct, so the fold's value depends on the access pattern it
  creates, not only on the launches it removes.
- The custom routed kernel already does the whole expert chain in one launch
  and measured 8.54 ms saved against four vendor launches -- the same mechanism
  at the largest scale available.

That is where the remaining factor of two lives, and it is one structural
change rather than a search.
