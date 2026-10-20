# The width-six gather_fia gate: four places read, cause not found

`cann_attention.GATHER_FIA_MAX_WIDTH = 6` with a comment -- "widths above six
failed serial-equivalence checks" -- and no evidence page behind it. Read today
with no machine:

## What `gathered_fia` does with the width

    width = ql_nope.shape[0]
    indices = topk_indices.reshape(-1)
    latent = kv_cache[0].view(-1, 512).index_select(0, indices)
    key = latent.view(width, 1, 2048, 512)
    ... actual_seq_qlen=[1] * width, actual_seq_kvlen=[2048] * width

Fully parameterised. The width comes from the input's own shape and every
reshape and sequence list is built from it. No constant, no assumption, nothing
that changes at seven.

## What the topk buffer does

`sfa_v1._update_indexcache_topk_indices` writes into
`self.topk_indices_buffer[:num_tokens, :topk_tokens]`, so the buffer is
two-dimensional and sliced by the round's own token count. It imposes no width
limit, and it is the write path rather than the read path `gathered_fia` uses --
which receives `topk_indices` as an argument.

So the non-contiguity hypothesis fails: the indices `gathered_fia` reshapes are
the caller's tensor, not a slice of that buffer.

## Excluded, and what is left

Excluded today: a hardcoded width in `gathered_fia`, a width limit in the topk
buffer's shape, non-contiguity from buffer slicing, and the sentinel-index
suspect closed earlier in this project.

What is left is what the comment describes and nothing records: a
serial-equivalence check that failed at width 8 or above, whose output nobody
kept. The gate is therefore a recorded observation with no located cause, and
the cheapest way forward is to re-run the comparison and read the failure rather
than to keep reading the code -- four readings have now found nothing.
