# Two thirds of this trajectory lives past the chain's ceiling

A width-8 round offers seven draft slots, so the MTP chain cannot accept more
than seven tokens however accurate it is. This asks how much of the output is
out of its reach, and answers it from a finished run with no device work.

`tests/glm52/ngram_acceptance.py` replays the record's own token stream as a
context-lookup drafter: take the last n tokens, find the most recent earlier
occurrence, propose what followed it. The accepted prefix is the longest
common prefix of that proposal and what the run actually emitted, so a drafter
that never ran has an exactly computable acceptance.

## Measured, `conditioned-20261019/on-1`, 1024 tokens

| drafter | width | rounds | acceptance | hit | max |
|---|---|---|---|---|---|
| MTP chain, measured | 8 | 227 | 4.511 | 91.2% | 7 |
| context lookup k=7 | 8 | 280 | 3.657 | 46.4% | 7 |
| context lookup k=15 | 16 | 226 | 4.531 | 37.2% | 15 |
| context lookup k=23 | 24 | 214 | 4.785 | 33.2% | 23 |
| context lookup k=31 | 32 | 202 | 5.069 | 30.2% | 31 |

Lookup lengths 2 to 8, longest first. The three measured requests are
bit-identical, so one of them is the trajectory.

At the same width the chain wins — 4.511 against 3.657 — because it hits on
91.2% of rounds and lookup on 46.4%. The two are not competing at the same
width, though. They have different shapes:

**At k=23, 17.8% of rounds match more than seven drafts, and those rounds
carry 68.7% of every token accepted.** At k=31 it is 17.8% of rounds and
71.9% of tokens. The chain cannot reach any of it, at any accuracy, because
width 8 has seven slots.

This agrees with what the chain's own histogram says. 18.94% of its rounds
match all seven — the predictable stretches — and lookup finds runs of 20 to
31 in the same places. Two independent measurements, one structure: this
trajectory is long predictable runs separated by hard tokens.

Scope: a record carries the generated tokens and not the 8192-token prompt, so
this searches the generation only. A deployed drafter also searches the
prompt, which on an agentic trajectory is where the copyable material is.
Every number above is a lower bound.

## Why the width is nearly free

`kernel_mla.h:536-549` sets `usedCoreNum` to the query count and returns from
cores 8 to 23, so a width-8 verify runs eight core groups with a one-row
critical path and **two thirds of the cube idle**
([sfa-explained](../sfa-explained-20261019/README.md)). The measured row cost
agrees: +0.045 ms a row from width 6 to 8, +0.112 from 8 to 12, +0.117 from 12
to 16 ([width sweep](../width-sweep-20261019/README.md)).

So `splitKV` being dead code and the die being two-thirds idle is not a
separate problem to solve — a wider verify fills those cores with rows, and
the rows are what buy acceptance.

## The arithmetic, projected

Extrapolating the measured row cost, a width-32 verify is about
39.68 + 0.112 × 24 = 42.4 ms. A lookup drafter needs no draft forward at all,
so the chain's 10.27 ms and the conditioned pass's 2.08 both go:

| configuration | round | acceptance | TPOT |
|---|---|---|---|
| chain, width 8 (today) | 53.05 | 4.507 | 11.77 |
| lookup only, width 32 | 43.4 | 5.069 | **8.56** |
| chain + lookup, width 32 | 55.8 | ≥ 5.07 | ≤ 11.0 |

Lookup only reaches the 9 ms milestone with no kernel work, no training and
no device-side drafter. But it is the wrong configuration to ship: on a
non-repetitive stretch it accepts one token a round, which at a 43.4 ms round
is 43 ms a token. The chain's 91.2% hit rate is what protects those stretches,
so the two belong in one tree — and the merged acceptance is the number that
decides the width, which is not measured yet.

## What gates it

The merged acceptance needs the chain's proposals at each round, and the
record kept only the accepted prefix. `mtp_session.py` now records
`candidates` and `predictions` — both already produced by the round, so this
costs nothing — and with them any tree topology's acceptance is computable
offline from one run, along with the per-position acceptance rates that decide
where a tree should branch.

## Branches are worth nothing; depth is everything

A tail that repeats several times offers several continuations, so the obvious
move is to verify them all. Measured, with each branch costing k more rows:

| branches | width | acceptance | hit |
|---|---|---|---|
| 1 | 8 | 3.657 | 46.4% |
| 2 | 15 | 3.864 | 46.8% |
| 4 | 29 | 3.894 | 47.1% |
| 8 | 57 | 3.908 | 47.3% |

Eight branches at eight times the width buy 0.9 points of hit rate. When the
longest tail match exists, the most recent occurrence's continuation is
almost always the right one; when no match exists, no number of branches
creates one. **Lookup's misses are structural, not a ranking problem**, so its
complement has to be a model and not more of itself.

At k=15 the same holds: 4.531 / 4.741 / 4.785 / 4.830 for 1 / 2 / 4 / 8.

## A merge estimate that does not hold

Evaluating lookup at the chain's own 227 round boundaries gives acceptance
10.489 and a merged 11.405, which would be 4.89 ms a token. It is wrong:
10.489 × 227 is 2381 tokens out of a 1023-token stream. Sampling at another
drafter's boundaries counts the predictable runs once per boundary that lands
in them, and a drafter accepting ten a round would have reached the end in 93
rounds rather than 227.

Only a walk where the drafter drives its own boundaries is consistent, which
is what the table at the top of this page is. The merge cannot be simulated
that way without the chain's proposals at arbitrary positions — the reason
`mtp_session.py` now records `candidates` and `predictions`.
