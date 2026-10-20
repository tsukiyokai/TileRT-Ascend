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
"""
import torch


def expert_counts(group_list, group_list_type):
    """Rows an expert, from either group-list convention.

    Type 0 is cumulative, type 1 is counts already. A slot is touched when its
    count is positive.
    """
    if group_list_type == 0:
        return torch.diff(group_list, prepend=group_list.new_zeros(1))
    return group_list


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

    The engine reads its weights through `get_mlp_weights` and its group list
    from the round's `MoEMlpComputeInput`, and the two have to agree, so the
    filter wraps the methods that hold both: `apply_gmm1_act_quant` and
    `apply_gmm2`. Each reads `mlp_compute_input.group_list`, which is the
    round's own, so the subset is recomputed every round at no cost beyond the
    nonzero scan.
    """
    original_weights = quant_method.get_mlp_weights
    split = {}

    def splitting(layer):
        """The bank as one tensor an expert, cached a layer.

        Splitting is free -- 194.71 us against the fused tensor's 195.65 in a
        graph -- and the views have to be built once rather than a round.
        """
        if id(layer) not in split:
            first, second = original_weights(layer)
            split[id(layer)] = (
                [first[index] for index in range(first.shape[0])],
                [second[index] for index in range(second.shape[0])],
            )
        return split[id(layer)]

    quant_method.get_mlp_weights = splitting
    undo = [("get_mlp_weights", original_weights)]

    for name in ("apply_gmm1_act_quant", "apply_gmm2"):
        original = getattr(quant_method, name)

        def filtering(mlp_compute_input, *rest, _original=original):
            """Shrink the round's group list, then let the method run.

            Both methods read `mlp_compute_input.group_list` and pass it
            straight to the operator, and both reach their weights through
            `get_mlp_weights`, which now returns a list. So narrowing is one
            substitution on the input: the slots with no rows leave the group
            list, and `splitting`'s list is indexed to match.
            """
            group_list = mlp_compute_input.group_list
            counts = expert_counts(group_list,
                                   mlp_compute_input.group_list_type)
            keep = (counts > 0).nonzero(as_tuple=True)[0]
            if keep.numel() == counts.numel():
                return _original(mlp_compute_input, *rest)
            kept = counts[keep]
            mlp_compute_input.group_list = (
                kept.cumsum(0) if mlp_compute_input.group_list_type == 0
                else kept)
            held = quant_method.get_mlp_weights
            quant_method.get_mlp_weights = (
                lambda layer, _keep=keep.tolist(), _held=held: tuple(
                    [side[index] for index in _keep] for side in _held(layer)))
            try:
                return _original(mlp_compute_input, *rest)
            finally:
                quant_method.get_mlp_weights = held
                mlp_compute_input.group_list = group_list

        setattr(quant_method, name, filtering)
        undo.append((name, original))

    def restore():
        for name, method in undo:
            setattr(quant_method, name, method)

    return restore
