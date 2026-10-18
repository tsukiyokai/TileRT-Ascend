# In-graph cost of one kernel

A3 `a3-84` die 0, shared with another tenant's idle `vllm serve`, one rank,
no model. [The chain probe](../../../../tests/glm52/bench_graph_chain.py)
captures N dependent one-row BF16 adds in a single NPU graph and replays it,
so host dispatch is already removed and the slope over N is the device-side
cost of a kernel that moves 12 KiB. `out=` keeps the chain serial. Fifty
replays per batch after fifty warmup replays, twenty batches, one chain
length per process.

| chain length | replay median (µs) | marginal (µs/kernel) |
|---|---|---|
| 1 | 8.546 | |
| 2 | 8.469 | −0.076 |
| 4 | 13.388 | 2.459 |
| 8 | 20.631 | 1.811 |
| 16 | 29.000 | 1.046 |
| 32 | 45.191 | 1.012 |
| 64 | 78.995 | 1.056 |
| 128 | 147.187 | 1.066 |
| 256 | 288.325 | 1.103 |

The marginal cost settles at 1.01–1.10 µs from sixteen kernels upward; a
least-squares line over 8–256 gives 1.079 µs per kernel with a 12 µs
intercept, and the first two lengths sit on the replay overhead alone.
The eager back-to-back floor for the same operation is 10.09 µs
([floor probe](../sfa-shape-20261018/README.md#per-kernel-floor)), so a graph
removes about nine microseconds per kernel, a factor of 9.5. That is the
mechanism behind the measured M1 graph result of 256.84→44.68 ms.

This prices the kernel count rather than condemning it. One M4 target
verify issues 2,416 calls, not the 561 the five largest operator types
suggest: the profile's family table sums 150 grouped GMM, 234 MLA prolog
and attention, 255 quantized projections, 941 cast/quant/norm, 225 MoE
routing and combine, 548 other traced compute, 21 index selections and 42
RoPE calls to exactly the 40.698 ms compute union
([family table](../cann-mtp-profile-20261017/summary.json)). At 1.079 µs
that floor is 2.61 ms, which is 38% of the 6.866 ms round budget at the
observed acceptance.

So launch volume does not explain today's round of 57.12 ms — in-graph SFA
costs 90.3 µs per call, 84 times this floor, and the round cost sits inside
the kernels. But the call count is a first-order claim on the 2 ms budget
and has to fall as well. The clearest case is cast/quant/norm: 941 calls for
3.032 ms, 3.2 µs each, of which 1.02 ms is the floor alone. Together with
the 548 other traced compute calls that is 1,489 calls and 9.32 ms at 6.3 µs
each, almost all of it small vector work whose cost the floor dominates.
