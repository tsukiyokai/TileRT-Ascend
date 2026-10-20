"""The tree mask's construction, checked against the tree's own structure."""
import sys

sys.path.insert(0, "/home/shanshan/repo/tile-ai/TileRT-Ascend")
from tilert.models.glm_5_2_ascend.tree_mask import ancestor_rows, visible


def ancestors(parents, node):
    chain = []
    walk = node
    while walk >= 0:
        chain.append(walk)
        walk = parents[walk]
    return set(chain)


def main():
    # A tree where node 0 is the root, 1 and 2 branch from it, 3 from 1.
    parents = [-1, 0, 0, 1]
    prefix = 100
    picks = 8
    rows = ancestor_rows(parents, prefix, picks)
    failures = []
    for node in range(len(parents)):
        seen = visible(rows, node)
        want = {prefix + index for index in ancestors(parents, node)}
        tree_part = {position for position in seen if position >= prefix}
        if tree_part != want:
            failures.append(f"node {node}: sees {sorted(tree_part)}, "
                            f"wants {sorted(want)}")
        if len(rows[node]) != picks:
            failures.append(f"node {node}: row is {len(rows[node])} long")
        # A sibling must not be visible: that is the whole point of the mask.
        for other in range(len(parents)):
            if other not in ancestors(parents, node) and \
                    prefix + other in tree_part:
                failures.append(f"node {node} sees non-ancestor {other}")
    # And a budget too small for the chain must still keep the ancestors.
    tight = ancestor_rows(parents, prefix, 3)
    deepest = visible(tight, 3)
    if not {prefix, prefix + 1, prefix + 3}.issubset(set(deepest)):
        failures.append(f"tight budget dropped an ancestor: {deepest}")
    print("\n".join(failures) if failures
          else f"all rows correct at picks {picks} and at a tight budget")


if __name__ == "__main__":
    sys.exit(main())
