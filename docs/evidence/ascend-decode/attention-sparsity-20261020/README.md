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


## Corrected: the denominator is wrong in kind, not only in magnitude

Three things overturn the reading above.

The operator is already at or below the public state of the art. A reference
AscendC `npu_sparse_flash_attention` at batch 1, 128 query heads, topk 2048
and head dim 512 + 64 costs 98 to 100 µs a call on A3; this one costs 89.4 at
four query heads. And the published times are flat across KV lengths from
2560 to 48000, so the cost tracks the sparse window rather than the sequence
-- which is the behaviour of an operator that already honours its selection.

The binding limit is not bandwidth. A vector core has sixteen outstanding
MTE2 requests, and a scattered gather on this part necessarily becomes one
descriptor per block because there is no gather DMA: `Gather` takes `__ubuf__`
sources on the vector pipe, and the multi-dimensional `DataCopy` is
unsupported on A3. So the family is latency bound in the number of
descriptors, and a bytes roofline is the wrong denominator in kind. The
vendor's own controlled A/B on this part separates the terms: holding bytes
identical and raising `blockCount` from 1 to 64 takes MTE2 from 548.161 to
220.772 µs, 2.48 times, while raising `blockLen` from 128 B to 2048 B buys
only 1.09.

And the byte estimate itself was too small. `KV_N` supports only 1, so the
MQA latent is not tensor-parallel shardable and all sixteen ranks read the
same bytes. At `index_topk = 2048`, a 656-byte C8 row and 78 layers that is
104.8 MB a decode step, which against 1.382 ms implies 75.8 GB/s rather than
the comfortable figure the page assumed.

What survives is the lever, and it is cheaper than a kernel.
`sparse_block_size` is an existing attribute of the operator this engine
already calls, hardcoded to 1 at five sites in
`vllm_ascend/device/device_op.py`. Above 4 it selects the C template, which
drops the `kvMergeGm_` staging round trip entirely -- `MergeKv` has exactly
one call site and it is guarded by `if constexpr (TEMPLATE_MODE ==
V_TEMPLATE)` -- and cuts the descriptor count by the block factor, which is
the term that actually binds. The cost is accuracy, since selection becomes
block-pooled, and GLM-5.3-Flash's own `index_kpool: 4` shows the model family
treats that as acceptable. One constant, one arm, and the answer is a
measurement rather than an argument.
