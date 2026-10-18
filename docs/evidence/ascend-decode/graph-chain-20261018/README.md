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

This bounds the kernel count rather than condemning it. The named operator
families in one M4 target verify are 561 calls (75 GMM-SwiGLU, 75 GMM,
78 SFA, 255 QuantBatchMatmul, 78 MlaProlog), which is 0.59 ms at this rate;
even a thousand kernels per round costs 1.08 ms against the 6.752 ms round
budget. Launch volume is therefore not what keeps the round at 71.6 ms.
In-graph SFA costs 90.3 µs per call, 84 times this floor, so the round cost
sits inside the kernels. Fusion has to be justified by locality and traffic,
not by dispatch.
