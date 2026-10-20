"""Feed the draft tree's per-row index set to the attention that already takes one.

The sparse attention the engine calls selects, for each query row, the key
positions that row may see: `layout_query="TND"` has no batch axis, so a token
is a row, `sparse_indices` is indexed by that row and `sparse_block_size=1`
makes a block one token. The engine fills that selection from an indexer; with
`skip_topk` it takes the selection from `topk_indices_buffer` instead.

A tree mask is the same selection with a different rule -- the committed prefix
plus a node's own ancestors -- so the wiring is to own that buffer and write
the tree into it each round. `tree_mask.ancestor_rows` builds the rows and
`-1` is the operator's own sentinel for an unused slot.
"""
from .tree_mask import ancestor_rows


def install(layers, picks):
    """Own each layer's index buffer; return how to fill it and how to undo.

    `layers` are the attention objects. `picks` is the operator's per-row
    selection width, which the checkpoint's `index_topk` sets.
    """
    import torch

    held = []
    for layer in layers:
        held.append((layer, layer._skip_topk, layer.topk_indices_buffer))

    buffer = None

    def write(parents, prefix_length, device):
        """Put this round's tree in the buffer every layer reads."""
        nonlocal buffer
        rows = ancestor_rows(parents, prefix_length, picks)
        wanted = torch.tensor(rows, dtype=torch.int32, device=device)
        if buffer is None or buffer.shape != wanted.shape:
            buffer = wanted
            for layer, _, _ in held:
                layer._skip_topk = True
                layer.topk_indices_buffer = buffer
        else:
            buffer.copy_(wanted)
        return buffer

    def restore():
        for layer, skip, original in held:
            layer._skip_topk = skip
            layer.topk_indices_buffer = original

    return write, restore
