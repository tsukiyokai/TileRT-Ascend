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
