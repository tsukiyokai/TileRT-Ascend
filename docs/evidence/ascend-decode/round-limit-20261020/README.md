# The routed kernel cannot walk more than five experts a block

2026-10-20, a3-84, one die, with another tenant on the machine.

Sixteen experts, varying only how many a block walks:

    blocks 16 (rounds 1): wall  47.783 us
    blocks  8 (rounds 2): wall  66.070 us
    blocks  4 (rounds 4): wall 114.925 us
    blocks  3 (rounds 6): aicore exception
    blocks  2 (rounds 8): aicore exception

And eighty experts, which isolates the rounds from the count:

    experts 80 blocks 16 (rounds 5): wall 147.514 us
    experts 80 blocks 14 (rounds 6): aicore exception
    experts 80 blocks 13 (rounds 7): aicore exception
    experts 81 blocks 17 (rounds 5): wall 149.426 us

The limit is five rounds a block. It does not depend on the expert count, the
block count by itself, the indexing, the scratch size, or the machine.

## Not introduced by the absorption

Every earlier measurement of this kernel -- the numerics against the vendor,
the real checkpoint weights, the variable group list, the absorbed gather and
scatter -- ran at one expert a block. The limit was always there and nothing
had exercised it. The real shape is what exposed it: a rank holds 256 experts,
and sixteen blocks walking them is sixteen rounds.

## Why raising the block count does not rescue it

Five rounds over 256 experts needs 52 blocks. A3 has 24 AI cores a die, and
`numBlocks` above the physical core count makes the framework insert the wrong
synchronisation -- a trap recorded earlier in this project. So the reachable
ceiling is 24 blocks, eleven rounds, still past the limit.

## What the mechanism is not

The device reports "the GM address accessed by scalar exceeds 48 bits". Ten
explanations were tried and each was disproven by experiment: a slot index
past the valid assignments, an over-read of the group list, an abnormal
scratch address, a machine difference, a self-inconsistent probe table, the
indexing path, the witness buffer's flush allocation, the scratch size, the
block count, and the manually allocatable event ids -- the flag constants are
fixed at 4, 5, 6 and 7 and do not grow with rounds.

What survives is the shape of the fact: the cross-core flags are counted, each
round sets two on each half, and nothing drains them between rounds. A counter
that saturates after five rounds fits every observation, and nothing else
tried does.

## What it settles

The paired `CrossCoreSetFlag`/`WaitFlag` structure cannot serve 256 experts,
so it has to be replaced by the grid barrier mixbrick uses -- `SyncAll<false>()`
on flagIds 11, 12 and 13, outside the manually allocated range. That was
already the conclusion from the shared-expert fold, which needed a grid
barrier for the N-split. The two lines meet here, and this time the reason is
a measured hard limit rather than a design preference.
