# CANN decode execution

Environment: `a3-gs:trt-asc`, GLM-5.2-W4A8C8, TP16, batch one,
8192 captured prefix rows, capacity 9344, memory utilization 0.85, eager
CANN kernels and FP32 TP sums. Requests use the production rank command
loop; prefix readiness, reset, HTTP and HIXL are outside internal TPOT.

## CPU thread affinity

[Audited results](affinity.json) record two sequential runs from the same
source snapshot. Each uses one warmup and three 32-step requests.

| Request TPOT (ms) | Original affinity | Bound threads |
|---|---|---|
| 1 | 222.156 | 227.715 |
| 2 | 256.843 | 246.645 |
| 3 | 301.381 | 286.644 |
| Median | 256.843 | 246.645 |

The medians differ by 3.97%; the ranges overlap. These short sequential
runs do not establish a causal speedup. Thread affinity does not explain
the roughly 250 ms eager cost.

The benchmark's `--cpu-affinity` uses the installed
`CpuAlloc.build_cpu_pools()`, `allocate()` and `bind_threads()` after model
loading. Actual per-thread masks match the plan on all sixteen ranks.
It does not call `run_all()`, change IRQ masks or migrate NUMA pages.
All eight 33-token outputs are identical. Rank report hashes, checkpoint,
source, runtime settings and clocks are checked in the receipt.

Raw runs remain on a3-gs under
`/home/shanshan/trt-asc/offline/cann-affinity-20261017/{bound,original}/`.
The source is `12ae8e2` plus the hashed benchmark override in the receipt.

## ACL Graph input updates

The optional `CannModel.enable_graph()` captures only the M1 model
forward with the installed `ACLGraphWrapper`; head and argmax remain
outside. Eager warmup precedes capture. Before each replay, the original
metadata builders update token, position, sequence length, slot mapping
and the persistent RoPE buffers. A separate persistent INT64 slot tensor
is also refreshed: the SFA warmup cache otherwise retains an old cast.
Full indexer layers generate and copy top-k indices within the graph;
shared layers read that graph-updated buffer.

In the installed rope64/BF16 path, SFA and indexer pass device sequence
length tensors to the kernels. Python `max_seq_len` is not an argument of
this branch. The pinned source paths and hashes are retained by each run.

The first wrapper call records a graph and returns uncomputed output;
actual replay is the subsequent branch of `acl_graph.py:294-302`.
The initial probe returned zero hidden/logits and token 0 instead of 382
at position 8192. The adapter now replays before exposing the first output.
The checker also prefills unwritten KV rows with NaN or an INT8 sentinel,
so retained eager rows cannot conceal missing graph writes. Numerical
diagnostics, reset, prefix retention and actual ACL graph entries are
checked before any profiler-off timing request.

## M1 device result

[Independent CPU audit](graph-audit.json) rehashes all 1536 snapshots
(725,712,384 bytes). Graph/eager, reset/eager and reset/graph each compare
100,352 tensors across sixteen ranks; every tensor has identical bits.
These tensors contain final logits, final normalized hidden, token IDs
and all layers' new KV rows. They do not contain every intermediate hidden.
Each rank also executes 579 full-prefix checks across the three phases.

The actual wrapper holds one `UpdatableGraph` with the M1 descriptor and
the recorded token/position addresses. Numerical diagnosis records one
capture and 64 replays; the four subsequent requests add 128 forwards.
Every rank completes command serial 137 and finishes reset at position 0.

| Profiler-off request | Internal TPOT (ms) |
|---|---|
| Warmup | 44.8040 |
| Measured 1 | 44.7035 |
| Measured 2 | 44.6755 |
| Measured 3 | 44.6391 |

Median is 44.6755 ms. Each output equals the 33-token eager M1 trajectory;
the [rank-zero receipt](graph-rank-0.json) retains callback clocks. This
compares with 256.8426 ms in the preceding unbound eager short run, about
5.75 times lower. Both are internal 32-step results, without HTTP/HIXL.
They are not the MTP3/graph4 vLLM baseline or 1024-token acceptance.

Run [check_cann_graph.py](../../../../tests/glm52/check_cann_graph.py)
with sixteen ranks, the checkpoint and capture-case paths, `--steps 32`,
`--requests 3 --warmup 1`, and a fresh `--output`. The exact launch and
source snapshot remain on a3-gs at
`/home/shanshan/trt-asc/offline/cann-graph-20261017-02/`.
Its `audit.py` regenerates `graph-audit.json` on CPU. The prior failed
capture and zero-output snapshots remain in sibling `cann-graph-20261017-01`.

The adapter currently captures M1. M4 target verification and the MTP
draft layer still need their own graph execution and timing. HF remains
the independent numerical diagnostic; the eight-input frozen teacher and
1024-token release checks are separate.

## M4 serial and speculative execution

The [MTP audit](mtp-audit.json) covers the same full-P 8K case, real target
and MTP weights, width four and sixteen output tokens. All sixteen ranks
produce the same tokens in serial, speculative and dirty-reset runs.
The audit rehashes 48 snapshots (70,818,384 bytes). Serial/speculative and
speculative/reset each compare 3136 tensors containing 47,088 new cache
rows; all bits agree. Full-prefix retention is checked on the device;
the CPU snapshots contain only new rows.

The eight rounds have natural acceptance counts `[3,1,1,3,1,2,2,3]`, mean
2.0. The output budget trims the final count from three to two, so the
actual fifteen committed rows average 1.875 per round. Including prefill's
first token, sixteen tokens are emitted; main and MTP lengths finish at
8207, with the last token pending. This short trace does not establish
the acceptance distribution for 1024 outputs.

This run uses eager M4, without timing. At a sustained acceptance of two,
the 2 ms/token goal permits only 4 ms per round. M4 graph execution and
actual round timing must precede further performance attribution.

Run [run_cann_mtp_rounds.py](../../../../tests/glm52/run_cann_mtp_rounds.py)
with sixteen ranks, the checkpoint and capture-case paths, `--width 4`,
`--max-tokens 16 --capacity 9344 --ignore-eos`, memory utilization 0.85
and a fresh output directory. Launch, frozen source, rank reports,
snapshots and `audit.py` remain on a3-gs under
`/home/shanshan/trt-asc/offline/cann-mtp-rounds-20261017-01/`.
