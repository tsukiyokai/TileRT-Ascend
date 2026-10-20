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

## The root fallback was the wrong fix, and it says what the right one is

Implemented and A/B tested on twelve streams at budget 16:

    suffix only              mean 5.578   min 2.169   max 6.649
    with the root fallback   mean 5.578   min 2.169   max 6.649

Identical, so the fallback never fired: `_match_state` does not return None on
these streams. The suffix descent always finds a class with continuations.

So the 29 rounds where the tree got nothing were not rounds where it found
nothing to propose -- they were rounds where it proposed and every branch was
wrong. The diagnosis above is corrected: the gap is in **which source the match
picks**, not in whether a match exists.

And that names the real difference. The tree ranks continuations by how often
they follow the matched suffix; the oracle picks the occurrence whose
continuation extends furthest. Those differ exactly where a frequent short
continuation sits beside a rare long one -- which is what an edit creates: the
pre-edit text is frequent, the post-edit continuation is not.

The change to test is therefore ranking by extendable length rather than by
frequency, over the same occurrence set the automaton already holds. The code
is reverted meanwhile, since an unfired fallback is dead code.
