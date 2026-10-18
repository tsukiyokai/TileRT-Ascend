"""CANN gather/FIA for batch one with a contiguous BF16 MLA cache.

The caller supplies 2048 valid causal indices per query. Logical token
positions equal physical cache rows; selection already enforces causality --
but only up to width 6. At width 8 and beyond an early row sees rows written
later in the same verify, because the call passes sparse_mode 0 with
actual_seq_kvlen uniformly 2048 and so applies no mask at all. Measured: the
output diverges from serial decoding at token 4 with 941 bit-wise KV checks
failing, while the first round is clean because its tail was zeroed.
"""

from types import MethodType


def gathered_fia(
    impl, ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
    actual_seq_lengths_query, actual_seq_lengths_key, block_table=None,
):
    """Gather selected latent/RoPE rows, then reuse the installed CANN FIA."""
    import torch_npu

    width = ql_nope.shape[0]
    indices = topk_indices.reshape(-1)
    latent = kv_cache[0].view(-1, 512).index_select(0, indices)
    rope = kv_cache[1].view(-1, 64).index_select(0, indices)
    key = latent.view(width, 1, 2048, 512)
    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        ql_nope.view(width, 4, 1, 512), key, key,
        query_rope=q_pe.view(width, 4, 1, 64),
        key_rope=rope.view(width, 1, 2048, 64), num_query_heads=4,
        num_key_value_heads=1, softmax_scale=impl.scale, input_layout="BNSD",
        sparse_mode=0, block_table=None, block_size=0,
        actual_seq_qlen=[1] * width, actual_seq_kvlen=[2048] * width,
    )
    return output.view(width, 4, 512)


GATHER_FIA_MAX_WIDTH = 6


def install_gather_fia(impl, width=4):
    """Bind one model instance without changing the installed vLLM classes."""
    if width > GATHER_FIA_MAX_WIDTH:
        raise ValueError(
            f"gather_fia drops causality past width {GATHER_FIA_MAX_WIDTH}; "
            "use the sfa kernel")
    impl._execute_sparse_flash_attention_process = MethodType(
        gathered_fia, impl,
    )
