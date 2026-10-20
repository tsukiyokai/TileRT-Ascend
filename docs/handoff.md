# Where this stands, 2026-10-21

TPOT best measured end to end: **9.3794 ms** (automaton drafter at width 12,
a3-84). Target 2.x. Everything below is single-die or CPU scope unless marked.

## The round, as now understood

    round(width) = 45.88 ms + 0.66 ms per unit of width

The second term is the **width's** fixed cost, not the row's marginal cost.
Settled from the arms' own records: grouping each arm's rounds by matched
drafts, six more matches cost 0.16 to 0.60 ms within an arm, while the medians
across arms are 48.2, 50.5 and 53.5 ms at widths 8, 12 and 16. One graph is
captured at `graph_width` and replayed once a round, so a wider graph costs
more to replay even when its rows carry nothing.
[Evidence](evidence/ascend-decode/row-cost-mismatch-20261021/README.md)

Consequence: a tree's budget is expensive because of the captured width, not
because nodes cost anything to verify. Match the captured width to the tree's
typical node count -- the adaptive tree averages 13.1 nodes, so width 16 --
rather than to its ceiling.

## Levers measured, awaiting the engine

| lever | a round | how |
|---|---|---|
| routed experts, four launches into one | 8.48 ms | `--routed-kernel` |
| TP reduction by one-sided push | 6.50 ms | `--onesided-reduce` |
| four projections off the hidden states | 2.46 ms | unwired |
| residual add with the norm | 1.46 ms | unwired |
| grouped matmul to CANN's own | 1.40 ms | `STOCK_OPP=1`, net 1.4 |
| adaptive draft width | 9.7 percent | on by default with `--automaton` |

20.30 ms of merges takes the round to about 33 ms at width 16; with the
adaptive tree's 9.288 accepted that is a TPOT near 3.57 ms.

## The queue

Twenty-seven arms on a3-84, 200 attempts at 300 s each, about 16 hours:
`index-topk` (five), routed kernel (four), operator dispatch (four), width
discriminator (five), MTP (five), super kernel (four). The machine has been
tenant-occupied since 00:17 with 54.5 GiB held on the busiest die against a
47 GiB gate.

## Open, in order of what they decide

1. **What in graph replay scales with width.** Attention is the leading
   candidate and the only layer operator not measurable from one die --
   `aclnnSparseFlashAttention` rejects five argument sets with a tiling
   failure. The width arms answer it.
2. **The 20.30 ms end to end.** None of it has run in the engine.
3. ~~The tree mask from the decode loop.~~ Done: `--tree` selects the
   drafter, `MtpRoundExecutor` dispatches on `propose_tree`, the mask writer
   is injected by the attention installer, and the accepted path commits
   through the existing `resolve` -- the path is itself a chain, so the linear
   match, the EOS check, the output budget and the cache commit all apply.
   Four checks on CPU.
4. **Width 8 and above disagree between serial and speculative decoding.**
   Recorded in `fia-mask-20261020`, never isolated, and every acceptance
   figure here is from width 12. Now has a discriminator: a chain draft's
   tree mask is exactly the causal prefix, so if the cause is future leakage
   the tree path agrees at width 16 where the chain path does not, and the
   gate can go. Queued as `causal-20261021`.
5. The routed kernel's residual five percent against the vendor, parked.
6. Deregistering one operator from the custom package; deleting its files
   leaves it at 116.376 us instead of CANN's 66.813.

## Closed by measurement today

Cross-layer fusion (23 percent ceiling), `npu_add_rms_norm_quant` (4.8x
slower), `o_proj` with its residual add, the attention chain, pre-gathering the
KV selection (35 percent slower), `--index-topk` (flat in picks), MTP as a
drafter (published acceptance 1.85 to 3.9 against the automaton's 9.288),
narrowing the draft to touch fewer experts (cost is flat in the expert count),
and compacting the expert bank (115x the saving).
