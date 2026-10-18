# Removing a synchronisation removed a snapshot, and only acceptance noticed

`a3-84` idle, TP16, the composed flag set with `HCCL_OP_EXPANSION_MODE=AIV`,
`--max-tokens 1024 --warmup 3 --requests 3`, profiler off. The change under
test: proposal tokens travel as device scalars so a round issues every kernel
before the host reads anything
([commit](../../../../tilert/models/glm_5_2_ascend/mtp_round.py)).

| case | TPOT median (ms) | rounds | A | round (ms) | tokens vs reference |
|---|---|---|---|---|---|
| reference (host ints) | 14.540 | 298 | 3.433 | 49.92 | — |
| device tokens, chunk 1 | 24.504 | 492 | 2.079 | 50.74 | identical |
| device tokens, chunk 8 | 23.972 | — | — | — | identical |
| device tokens, chunk 32 | 23.622 | — | — | — | identical |

Three facts together name the defect. The output tokens are bit-identical to
the reference, so correctness held. The round's wall time is unchanged at
50.74 ms against 49.92, so nothing got slower. But acceptance fell from 3.433
to 2.079 and the request needed 492 rounds instead of 298. The regression is
entirely in proposal quality.

## The cause

`--draft-graph` means `CannGraph.forward` returns the graph's own persistent
output buffers — its docstring says so: "Returned hidden storage is reused on
replay." The previous `_seed` read `token_ids` through `.tolist()`, which
incidentally took a host snapshot. Replacing that with a device view
(`result.token_ids[row]`) left the chain holding aliases:

1. replay one writes the token buffer, `_seed` keeps a view of it
2. the view is appended to `candidates`
3. replay two **overwrites the same buffer**, so the candidate already
   appended now reads the newer token

By verification time the earlier candidates hold later proposals, so the
drafts mismatch and acceptance collapses. The target still verifies from
`candidates[0]`, the pending token, which is a host integer — which is
exactly why the delivered tokens stayed correct and the gate passed.

The fix is one device clone per proposal, `result.token_ids[row].clone()`:
a one-element copy, no host transfer, snapshot restored.

## The host suite could not see it, and now can

Every scalar double returned fresh Python lists of immutable ints, so they
were accidentally immune to the contract the real engines have. The round
suite gained `Cell`, a mutable token slot whose only detach is `clone()`, and
`RecycledDraft`, which refills the same cells on every forward. The new case
runs a plain and a recycled draft to completion and requires equal output
**and equal round counts**.

Polarity checked both ways on the same binary: with the clone removed the
case fails 2 subtests; with it restored, 10/10 pass. A test that only passes
is not evidence it would have caught anything.

## A measurement that survives the bug

All three cases carry the same acceptance regression, so the differences
between them isolate one variable — the per-round command broadcast. Batching
32 rounds per command is 0.88 ms/token faster than one round per command
(23.622 against 24.504). At 1023 tokens over 492 rounds that is about
1.83 ms per round removed by dropping 31 of every 32 broadcasts, so **the
per-round host command costs roughly 1.9 ms of the round**. The gap measured
*between* commands is only 0.017 ms median, which is consistent: the
broadcast is inside the operation's wall, not between operations.
