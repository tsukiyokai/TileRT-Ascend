# Existing CANN sparse-attention alternatives

A3 `a3-gs:trt-asc`, CANN 9.2.0, Torch/torch-npu 2.12.0, BF16
latent512 + RoPE64, four heads/rank, 2048 selected keys. The
[audit](audit.json) binds source and raw outputs. Runs use shared die 0.

## Numerical replay

The real input is layer 0/rank 0 at position 8192, M=1. An explicitly
synthetic M=4 case changes queries, KV rows and each query's causal indices.
All three pipelines use the same values and reversed physical-page table.
Each pipeline's two executions have identical output bits.

| Pipeline versus original SFA | Real M1 NRMSE | Synthetic M4 NRMSE |
|---|---|---|
| Device paged gather → existing nonpaged FIA | 0.000521 | 0.000876 |
| Four 512-slot SFA partitions → FP32 AttentionUpdate | 0.002370 | 0.002153 |

Maximum absolute BF16 difference is 0.0078125 in all four comparisons.
Peak Torch reserved memory is 96 MiB for M1 and 182 MiB for M4. These
cross-kernel diagnostics do not release model precision or establish speed.
The [single replay entry](../../../../tests/glm52/compare_cann_sparse_attention.py)
performs eight explicit attention/merge calls, saves six raw outputs and
checks repeatability. It has no performance loop or model loading.

The device-gather/FIA prototype also passes changing-input ACL replay for
M1 and synthetic M4: one capture, replay A/B/reset A. All six persistent
inputs change for B; context, gathered latent/rope and physical indices
match eager bits. Reset A overwrites poisoned output/intermediate buffers.
The independent raw-output check covers 17,756,160 replay elements.

```bash
python tests/glm52/compare_cann_sparse_attention.py \
  --input /workspace/offline/cann-mla8k-20261017/early-bind/result.pt \
  --output /workspace/offline/attention-replay --device 0
```

## Execution choices

The pinned arch22 SFA assigns M4 to four active AIC workers, each iterating
over four 512-slot KV segments. This is derived from source; the profile's
24 launch blocks are not an active-core count. Its 78 calls sum to about
7 ms in the existing profiled target round, versus the 6.752 ms total
round budget at observed acceptance 1023/303.

Head-as-batch passes real M1 bit comparison but repeats selected KV
traffic fourfold; both one and four heads still pad to sixteen matrix
rows. The two alternatives above expose KV parallelism instead. FIA uses
one virtual batch per query, preserving each query's selected-key order.
The split variant retains SFA gather and merges four partials with
`L_i=max_i+log(sum_i)`, `O=Σ O_i exp(L_i−logsumexp(L))`. Partials are
BF16; conversion to FP32 for merging cannot undo that extra rounding.

Existing SFA already stages selected KV through GM. External gather is
not automatically an additional whole-KV round trip: both payload
traversal models total 9,175,040 bytes/query before padding and other
workspace. Its added launch boundary and actual traffic need measurement.
The source's incomplete `FLASH_DECODE` branch is not enabled.

## Complete graph screening

The [benchmark](../../../../tests/glm52/bench_cann_sparse_attention.py)
captures ten complete calls per graph, warms ten graphs and rotates
pipeline order over thirty samples. Same-stream event time is divided by
ten; mapping, gather, input transforms and merging are included.

| Pipeline | Real M1 p50 (µs) | Synthetic M4 p50 (µs) |
|---|---|---|
| Original SFA | 86.67 | 88.44 |
| Paged gather → FIA | 91.11 | 136.26 |
| Split SFA → AttentionUpdate | 59.83 | 95.59 |

All 120 saved graph outputs match their eager outputs bit for bit before
and after timing. Sixteen external jobs share the host; p95 spans
437–506 µs. These hot-input measurements screen candidates, not idle-device
latency or decode TPOT. Retain original SFA for M4.

A short eager trace contains all six pipeline labels and 44 kernels;
all six contexts match the nonprofiled replay. In its second iteration,
FIA takes 36.24 µs, while its mapping and gather kernels take 96.46 µs.
The full pipeline takes 132.70 µs versus SFA's 88.38 µs. Split SFA takes
53.28 µs plus 48.58 µs of transforms and merging. The first original-SFA
call is a 2734.84 µs outlier. This shared-device Level0 trace supplies no
active-core or HBM counters.

TileRT's current cache is contiguous and its block table is identity.
Next compare direct token-index gathers with existing dual-cache
`GatherPaKvCache`, then feed FIA. Capture real M4 inputs and retain the
frozen-teacher end-to-end gate before changing the model default.

## Raw artifacts

Under `/home/shanshan/trt-asc/offline/` on a3-gs:

- `cann-attention-reuse-20261018-01/`: repository runner, real M1 and
  synthetic M4 inputs, graph samples, short trace, raw outputs and
  independent CPU audits.
- `cann-sfa-head-batch-20261018/`: pinned SFA source/profile contract and
  head-as-batch numerical diagnostic.
- `cann-fia-gather-20261018-01/`: FIA schema/source contract, independently
  checked CPU/device gathers and page-permutation cases.
- `cann-sfa-splitkv-20261018-01/`: partials/max/sum and FP64 merge
  reference; device-only arithmetic matches the earlier CPU bridge.
