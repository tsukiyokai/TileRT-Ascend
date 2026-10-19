# Four patches that passed per-call bit-exactness cost 3.15 ms a token

Paired, one session, one tree, one frozen prefix, width 4, the composed flags,
each arm its own cache root, each figure read from that arm's own record.

| | baseline | four kernel patches + compact declaration |
|---|---|---|
| TPOT | 14.9121 / 14.4877 / 14.3187 | 17.6350 / 17.8901 / 17.5922 |
| median | 14.4877 ms | **17.6350 ms** |
| range | 0.5934 | 0.2979 |
| rounds | 311, 311, 311 | **326, 335, 326** |
| token stream | one, `df10d06f911c` | one, **`df10d06f911c`** |
| winning vendor index | 3, barrier only | 1, the four-patch object |

**+3.15 ms a token, a 21.7% regression, against an expected −0.9.** The
overlay resolved to the patched object, so this is a measurement of the
patches and not of nothing.

## The two facts worth more than the number

**The token stream is bit-identical to baseline** — the same digest across all
three requests in both arms. A 1024-token generation agreeing token for token
is a stronger correctness statement than any of the four per-call gates, and
it covers the kernel patches and the compact declaration together. Whatever is
wrong, it is not correctness.

**The round count moved and stopped being constant within the arm**, 311 to
326/335/326, while the tokens did not. Decomposed: per-round time goes
14.4877 × 1023 / 311 = 47.6 ms to 17.6350 × 1023 / 326 = 55.3 ms, so the round
is 7.7 ms slower *and* there are fifteen more of them. The per-call bench
predicted 3 ms a round saved. **Both halves are wrong, which means the
per-call bench does not reach whatever this is.**

## The gate's hole, found by its own author

The compact-declaration patch bundles two changes: compressing the declared
expert count, and leaving `w2` in ND instead of converting it to FRACTAL_NZ.
The bit-exactness gate compared ND-declaring-256 against ND-declaring-8 — the
declaration varied, the layout held fixed. **It never compared NZ against ND.**
Leaving `w2` in ND was justified on speed alone, 5.603 µs a call, and never on
numerical equivalence.

If the two layouts tile the reduction differently they accumulate in a
different order, so a bf16 down-projection output can differ in its low bits.
That changes no value materially but it moves draft acceptance, and acceptance
sets the round count. **Identical final tokens with a different accept/reject
path is exactly that signature.**

## What this does to the rest of the board

Every expectation on this project's list derived from a per-call measurement is
now suspect, because here a per-call prediction of −3 ms a round produced
+7.7. That includes the kernel three's 0.66 ms, the compact declaration's 2.39,
and the one-sided reduction's 2.48 — the last of which at least was measured
as a whole protocol rather than as a per-call delta, but still outside a round.

Nothing from this set should land on today's evidence.

## A unifying observation

The draft head's argmax is the sensitive element and the target's is not. Width
8 diverges run to run because it has more draft positions for a marginal
decision to flip in; this regression moves acceptance because the draft sees
different low bits from a different reduction order. In both cases the final
tokens are stable while the accept/reject path is not — which is why round
count, not token identity, is the quantity that detects a numerical change
here.
[width 8](../width8-nondeterminism-20261019/README.md)

## In flight

A third arm with the four kernel patches and `_COMPACT_EXPERT_DECLARATION`
false, which also restores `w2` to NZ, separating the kernel three from the
declaration-and-layout pair. At 311 rounds and a median near 14.49 the kernel
three are clean and the declaration is the problem; if it also regresses, all
four need re-examining inside a round rather than per call.
