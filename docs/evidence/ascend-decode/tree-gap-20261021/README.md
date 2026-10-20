# Where the tree loses to the copy oracle

Ninety-one rounds over 400 tokens of a real stream, budget 16, against an
oracle that always finds the longest prior run.

    tree takes    3.41 a round
    oracle has    5.48
    the tree falls short in 46 of 91 rounds
    where the oracle found what the tree missed: prompt 35, output 11
    shortfall: 1 token in 17 rounds, 2 in 8, 3 in 7, 4 or more in 8
    rounds where the tree got nothing and the oracle had a run: 29

## The shape of the loss

Two thirds of the shortfall is rounds where the tree proposes nothing that
holds -- 29 of 46 -- rather than rounds where it goes not quite far enough. And
three quarters of what it misses, 35 of 46, the oracle found in the **prompt**
rather than in the output generated so far.

## Which names the defect

`TreeDrafter` builds a suffix automaton over the prompt followed by the
generated tokens and matches from the current context's suffix, descending
suffix links when a long tail has no continuation. After an edit the suffix is
broken, so the descent finds a short tail or none -- while the same material
still sits in the prompt at a position the broken suffix no longer points to.

The oracle does not match by suffix; it matches by looking for the current
token everywhere and extending. That is the difference, and it is 29 rounds in
91.

## The change it implies

On a round where the suffix descent yields nothing, restart the match from the
other occurrences of the current token in the prompt rather than giving up.
The automaton already holds those positions -- that is what its transitions
are -- so this is a search-order change rather than a new index.

Measurable on the twelve streams, machine-free, and worth up to the 31 percent
the tree is behind the oracle: 6.649 accepted toward 9.660, which at budget 16
and the merged fixed term is TPOT 6.12 ms toward 4.21.
