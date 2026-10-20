# The attention operator reads a cache it was told to skip

From this project's own capture, `kernel_details.csv` on rank zero, width 8.
What `SparseFlashAttention` is handed:

| argument | shape | dtype |
|---|---|---|
| query | `1,4,512` | BF16 |
| key | `73,128,1,512` | BF16 |
| value | `73,128,1,512` | BF16 |
| selected positions | `1,1,2048` | INT32 |
| lengths | `1,73` | INT32 |
| query rope | `1,4,64` | BF16 |
| key rope | `73,128,1,64` | BF16 |

`73 × 128 = 9344`, which is the `--capacity` default exactly, so the key and
value arguments are the whole paged cache. The `1,1,2048` argument is the
`index_topk`: of those 9344 positions only 2048 are attended.

## Which of the two it is, the numbers decide

256 calls at 85.048 µs, which over 78 layers is 6.63 ms a round and matches
the 6.973 recorded for the family.

Reading only the selected 2048 positions is `2048 × 512 × 2` bytes for each of
key and value plus `2048 × 64 × 2` for each rope part: 4.7 MB a layer, so
318 µs a round at the die's 1.155 TB/s. Against that, 85 µs a call is 55 GB/s
-- 4.8% of peak, which no kernel doing a contiguous gather would hit.

Reading the whole 9344 is 21.5 MB a layer, so 1.45 ms a round -- which is the
1.382 ms this project recorded as the family's byte floor. Against that, 85 µs
is 253 GB/s, 22% of peak, which is an ordinary figure for a gather.

So the recorded floor was the floor of reading everything, and the sparsity the
operator is named for buys nothing in bandwidth: the 2048 positions look to be
applied as a mask after the read rather than as a gather before it.

That makes attention worth more than it looked. A read that honours the
selection is 4.6 times less traffic; at the same 22% efficiency it would be
1.44 ms a round instead of 6.63, and at the efficiency the routed kernel
reaches on this part -- 66% of peak -- it would be 0.48 ms.

## The experiment that settles it

Time against `--capacity`, holding everything else. If the operator reads the
whole cache its cost is proportional to capacity; if it honours the selection
its cost is flat once capacity exceeds 2048. One sweep of a flag that already
exists, and nothing to build either way.

## And a term that was not in the budget at all

`hcom_allGather_` is the largest single consumer in the capture: 140 calls,
63554 µs, the largest single call 13240 µs. A 13 ms all-gather inside a decode
step is not moving data, it is waiting for the slowest rank. The project's
round budget has carried `hcom_allReduce_` at 5.35 ms a round with no overlap
and has never accounted for this one.
