"""Checks for narrow_bank, on the host: no NPU, no model."""
import sys

import torch

sys.path.insert(0, "/home/shanshan/repo/tile-ai/TileRT-Ascend")
from tilert.models.glm_5_2_ascend.narrow_bank import expert_counts, narrow


def check_counts_from_cumulative():
    group_list = torch.tensor([3, 3, 7, 7, 10])
    counts = expert_counts(group_list, 0)
    assert counts.tolist() == [3, 0, 4, 0, 3], counts.tolist()


def check_counts_passed_through():
    group_list = torch.tensor([3, 0, 4, 0, 3])
    assert expert_counts(group_list, 1).tolist() == group_list.tolist()


def check_narrow_drops_empty_slots():
    weights = [f"w{index}" for index in range(5)]
    group_list = torch.tensor([3, 3, 7, 7, 10])
    picked, kept = narrow(weights, group_list, 0)
    assert picked == ["w0", "w2", "w4"], picked
    assert kept.tolist() == [3, 7, 10], kept.tolist()


def check_full_bank_is_returned_unchanged():
    weights = [f"w{index}" for index in range(3)]
    group_list = torch.tensor([2, 4, 6])
    picked, kept = narrow(weights, group_list, 0)
    assert picked is weights
    assert kept is group_list


def check_counts_convention_is_preserved():
    weights = [f"w{index}" for index in range(4)]
    counts = torch.tensor([2, 0, 5, 1])
    picked, kept = narrow(weights, counts, 1)
    assert picked == ["w0", "w2", "w3"], picked
    assert kept.tolist() == [2, 5, 1], kept.tolist()




def check_install_narrows_a_fake_method():
    """The installer's substitution, against a stand-in for the engine."""
    class Layer:
        pass

    class Method:
        def __init__(self):
            self.layer = Layer()
            self.seen = None

        def get_mlp_weights(self, layer):
            bank = torch.arange(5 * 2 * 2).reshape(5, 2, 2)
            return bank, bank

        def apply_gmm1_act_quant(self, compute):
            first, _ = self.get_mlp_weights(compute.layer)
            self.seen = (len(first), compute.group_list.tolist())
            return "out"

        def apply_gmm2(self, compute, hidden, scale):
            return "out2"

    class Compute:
        def __init__(self, layer):
            self.layer = layer
            self.group_list = torch.tensor([3, 3, 7, 7, 10])
            self.group_list_type = 0

    from tilert.models.glm_5_2_ascend.narrow_bank import install

    method = Method()
    compute = Compute(method.layer)
    restore = install(method)
    assert method.apply_gmm1_act_quant(compute) == "out"
    assert method.seen == (3, [3, 7, 10]), method.seen
    # The round's own group list is put back, so a later reader sees it whole.
    assert compute.group_list.tolist() == [3, 3, 7, 7, 10]
    restore()
    method.apply_gmm1_act_quant(compute)
    assert method.seen == (5, [3, 3, 7, 7, 10]), method.seen


def check_a_non_monotonic_list_is_rejected():
    """A cumulative list that steps backwards is a bug, not an empty slot."""
    try:
        expert_counts(torch.tensor([3, 2, 7]), 0)
    except AssertionError:
        return
    raise AssertionError("a backwards step should not pass")

def main():
    checks = [value for name, value in sorted(globals().items())
              if name.startswith("check_")]
    for check in checks:
        check()
        print(f"  ok  {check.__name__}")
    print(f"{len(checks)} checks passed")


if __name__ == "__main__":
    raise SystemExit(main())
