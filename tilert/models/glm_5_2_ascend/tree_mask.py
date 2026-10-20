"""Turn a draft tree into the per-row index set sparse attention already takes.

The attention operator the engine calls selects, for each query row, the key
positions that row may see -- `layout_query="TND"` has no batch axis, so a
token is a row, `sparse_indices` is indexed by that row, and
`sparse_block_size=1` makes a block one token. The engine uses this to give
each token its own top-k blocks. A tree mask is the same mechanism with a
different index set: a draft token sees the committed prefix and its own
ancestors, and nothing else in the tree.

`-1` is the operator's own sentinel for an unused slot, which `sfa_v1.py:175`
relies on when it sorts the selections.
"""
EMPTY = -1


def ancestor_rows(parents, prefix_length, picks, prefix_picks=None):
    """One row of key positions a draft token may attend to.

    `parents[i]` is the index within the tree of node `i`'s parent, or -1 for
    a root. The tree's nodes occupy key positions `prefix_length + i`, so a
    node's visible set is the prefix plus the positions of its ancestors plus
    its own.

    `prefix_picks` caps how much of the committed prefix each row carries,
    which is what makes the row fit `picks`: the prefix is shared by every row
    and the engine already selects it sparsely, so the caller passes the
    selection it would have made anyway.
    """
    if prefix_picks is None:
        prefix_picks = max(0, picks - len(parents))
    shared = list(range(max(0, prefix_length - prefix_picks), prefix_length))
    rows = []
    for node in range(len(parents)):
        chain = []
        walk = node
        while walk >= 0:
            chain.append(prefix_length + walk)
            walk = parents[walk]
        chain.reverse()
        row = shared + chain
        if len(row) > picks:
            # The ancestors are what correctness needs; the prefix is already
            # a selection, so it is the prefix that gives way.
            row = row[len(row) - picks:]
        rows.append(row + [EMPTY] * (picks - len(row)))
    return rows


def visible(rows, node):
    """The positions a row actually selects, for checking a construction."""
    return [position for position in rows[node] if position != EMPTY]

def ancestor_bitmap(parents, prefix_length, capacity, device=None):
    """The tree's visibility as the mask `shared_kv_fia` already takes.

    `cann_attention.selection_mask` states the convention -- True excludes a
    cache row -- and builds its mask by scattering zeros at the selected
    positions. This does the same with the tree's rule: a node sees the
    committed prefix and its own ancestors, so those positions are zero and
    everything else is one.

    This is the shorter of the two paths into the engine. The other feeds
    `sparse_indices` to the operator the engine calls, which rejected seven
    argument sets from a probe; `shared_kv_fia` passes a mask of shape
    (1, 1, width, capacity) and is known to run.
    """
    import torch

    width = len(parents)
    bitmap = torch.ones((width, capacity), dtype=torch.uint8, device=device)
    bitmap[:, :prefix_length] = 0
    for node in range(width):
        walk = node
        while walk >= 0:
            bitmap[node, prefix_length + walk] = 0
            walk = parents[walk]
    return bitmap.view(torch.bool)
