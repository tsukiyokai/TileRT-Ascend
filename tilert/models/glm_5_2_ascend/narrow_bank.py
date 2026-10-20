"""Hand the grouped matmul only the experts a round's rows actually touched.

`npu_grouped_matmul`'s cost follows the bank it is given, not the rows: on one
die with 96 rows held fixed, 1, 2, 4, 8 and 16 experts of 6144x1536 INT8 cost
55.57, 54.24, 73.85, 105.51 and 168.58 us -- linear at 7.53 us an expert, with
bandwidth reaching 896 GB/s at the full bank. It reads the whole bank at HBM
speed whatever it was asked to compute, which is why it is flat in rows from 16
to 256.

At batch one a round proposes a handful of rows, each picking eight of 256
experts, so a rank of sixteen sees two to five of its own touched. The
checkpoint's `e_score_correction_bias` spans 0.02 to 0.03 across those 256, so
routing is not clustered and that count holds.

Narrowing moves no bytes. `bank[touched]` costs 703 us on one die, three times
the saving, but the operator takes a list of one tensor per expert and the
engine's own call already passes `w1 if isinstance(w1, list) else [w1]`.
Splitting into sixteen views is free -- 194.71 us against the fused tensor's
195.65 in a graph -- and passing six of them is 134.85. The saving is the
subset, not the split.

Both the weights and the group list have to agree on which slots exist, so the
filter belongs where a caller sees both at once.

The group list arrives as cumulative counts (type 0) or counts (type 1); every
quantisation method here calls `cumsum_group_list` with those two, and the
sparse type 2 branch cannot run at all since no caller passes the `expert_num`
it needs. So an empty slot is a zero or a repeat of its neighbour, and never a
negative step.
"""
import torch


def expert_counts(group_list, group_list_type):
    """Rows an expert, from either group-list convention.

    Type 0 is cumulative, type 1 is counts already. A slot is touched when its
    count is positive, and a negative count means the cumulative list is not
    monotonic -- which no path here produces, so it is an assertion rather
    than a case to handle.
    """
    counts = (torch.diff(group_list, prepend=group_list.new_zeros(1))
              if group_list_type == 0 else group_list)
    assert (counts >= 0).all(), counts
    return counts


def narrow(weights, group_list, group_list_type):
    """`(weights, group_list)` holding only the slots with rows.

    Returns the inputs unchanged when every slot is touched, so the caller
    pays nothing for a round that needs the whole bank.
    """
    counts = expert_counts(group_list, group_list_type)
    keep = (counts > 0).nonzero(as_tuple=True)[0]
    if keep.numel() == counts.numel():
        return weights, group_list
    picked = [weights[index] for index in keep.tolist()]
    kept = counts[keep]
    return picked, (kept.cumsum(0) if group_list_type == 0 else kept)


def install(quant_method):
    """Narrow both grouped matmuls to the touched experts; return the undo.

    The engine's W4A8 method reads its weights through `_get_mlp_weights`,
    which returns four values -- `(w1, w1_scale, w2, w2_scale)` -- and builds
    the group list from the round's `MoEMlpComputeInput`. The two have to agree
    on which slots exist, so the filter wraps that reader and the two methods
    that call it.

    An earlier version wrapped `get_mlp_weights` with two return values, a name
    and an arity the W4A8 class does not have. It installed without error and
    narrowed nothing.
    """
    original = quant_method._get_mlp_weights
    split = {}
    keep = [None]

    def splitting(layer):
        """The bank as one tensor an expert, cached a layer, filtered a round.

        Splitting is free -- 194.71 us against the fused tensor's 195.65 in a
        graph -- and the views are built once rather than a round.
        """
        if id(layer) not in split:
            held = original(layer)
            split[id(layer)] = tuple(
                [piece[index] for index in range(piece.shape[0])]
                if hasattr(piece, "shape") and piece.dim() >= 1 else piece
                for piece in held)
        pieces = split[id(layer)]
        if keep[0] is None:
            return tuple(_rejoin(piece) for piece in pieces)
        return tuple(_subset(piece, keep[0]) for piece in pieces)

    quant_method._get_mlp_weights = splitting
    undo = [("_get_mlp_weights", original)]

    for name in ("apply_gmm1_act_quant", "apply_gmm2"):
        method = getattr(quant_method, name, None)
        if method is None:
            continue

        def filtering(mlp_compute_input, *rest, _method=method):
            """Shrink the round's group list, then let the method run."""
            group_list = mlp_compute_input.group_list
            counts = expert_counts(group_list,
                                   mlp_compute_input.group_list_type)
            chosen = (counts > 0).nonzero(as_tuple=True)[0]
            if chosen.numel() == counts.numel():
                return _method(mlp_compute_input, *rest)
            kept = counts[chosen]
            mlp_compute_input.group_list = (
                kept.cumsum(0) if mlp_compute_input.group_list_type == 0
                else kept)
            keep[0] = chosen.tolist()
            try:
                return _method(mlp_compute_input, *rest)
            finally:
                keep[0] = None
                mlp_compute_input.group_list = group_list

        setattr(quant_method, name, filtering)
        undo.append((name, method))

    def restore():
        for name, method in undo:
            setattr(quant_method, name, method)

    return restore


def _subset(piece, chosen):
    """The chosen experts of a split bank, or the value if it was not split."""
    if isinstance(piece, (list, tuple)):
        return [piece[index] for index in chosen]
    return piece


def _rejoin(piece):
    """A split bank as the list the operator accepts, or the value itself."""
    if isinstance(piece, (list, tuple)):
        return list(piece)
    return piece
