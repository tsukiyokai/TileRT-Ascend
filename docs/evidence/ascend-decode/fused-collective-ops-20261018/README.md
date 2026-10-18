# Both vendor fusions of matmul and collective are closed to us

The ROCm library this project ships for AMD has no standalone reduction. Six
of its twenty-nine kernels carry the collective as an epilogue —
`fused_moe_allreduce_v4`, `fused_moe_allreduce_w8a8_v4`,
`pure_mla_allreduce`, `unprojo_allreduce`, `eh_proj_allreduce`,
`padded_allreduce_add`
([kernel inventory](../../tilert-ascend/libtilert_glm52_rocm-elf.txt), and
`kernels.tsv` in the extraction). Our round spends 9.97 ms on 157 separate
reductions, so the same fusion is the obvious move. Both vendor routes to it
are blocked, measured on `a3-84`, two ranks on dies 2 and 3, CANN 9.2.0.

## The all-reduce fusion is not built for this SoC

`torch.ops.npu.npu_mm_all_reduce_base` exists and its schema is the widest of
the family: `dequant_scale` with `pertoken_scale` for W8A8, `antiquant_scale`
for weight-only, `x3` for a post-reduction residual, `group_sizes` for the
grouped case, `comm_quant_scale_1/2` for a quantised wire, and `comm_turn`.
One operator covers four of the six ROCm families. Every variant is refused
([probe](probe_mm_ar_variants.py)):

```
bf16 × bf16                     MatmulAllReduceBaseKernelNpuOpApi.cpp:276
bf16 × bf16 + x3                ...:274
bf16 × int8 antiquant           ...:312
int8 × int8 pertoken dequant    ...:297
bf16 × bf16 comm_turn=2         ...:276
```

The device message behind the first probe names the cause:

```
Failed to execute operator MatmulAllReduce_12. Reason: 1.SoC version
ascend910_93 verification failed. This SoC is not configured through the
AddConfig API of the OpDef class.
```

So the kernel is absent from the A3 operator package rather than wrong for
our shapes. The PyTorch binding is there; nothing stands behind it.

## The reduce-scatter fusion exists but not at our shape

Two neighbours in the same family do run on A3
([inventory](probe_fused_comm_inventory.py)):

| operator | verdict |
|---|---|
| `npu_mm_reduce_scatter_base` | runs; `(8, 2048) × (2048, 6144)` at world 2 gives `(4, 6144)` |
| `npu_all_gather_base_mm` | runs; gives `[(8, 6144), (8, 2048)]` |
| `npu_quant_mm_reduce_scatter` | not established — the probe's own arguments were wrong |
| `npu_all_gather_quant_mm` | not established — same |

That reduce-scatter would serve as the fusion if an all-gather followed it,
and vLLM-Ascend's `KimiOProjMMReduceScatterOp` at
`vllm_ascend/ops/linear_op.py:156` is exactly that construction, which is
evidence the shape is the usual one on this hardware. It is not ours
([row constraint](probe_rs_rows.py)):

| rows | world | result |
|---|---|---|
| 2 | 2 | `(1, 6144)` |
| 4 | 2 | `(2, 6144)` |
| 5 | 2 | `The M-axis in input of Matmul should be be divisible by world_size.` |
| 7 | 2 | same refusal |
| 16 | 2 | `(8, 6144)` |

The operator splits the token axis across ranks. Decode at batch one presents
four rows to sixteen ranks, and four is not divisible by sixteen, so this
route is closed by arithmetic. Padding to sixteen rows would quadruple the
attention and projection work to save a collective worth 36 µs.

## What this settles

Fusing the collective into its producer is right — the ROCm library proves the
design — but no vendor operator will do it for us at batch one on A3. It has
to be hand-written, which puts it inside the same kernel work as the
cube-vector fusion rather than ahead of it as a cheap first step.

The costly reduction is also not the one to start with. Of the 157 per round,
156 cost about 36 µs each, which at 48 KiB across sixteen ranks is close to
rendezvous latency and leaves little for fusion to overlap. The first costs
4.36 ms because it absorbs the ranks' entry skew
([round composition](../profile-rope-20261018/README.md)), and that skew comes
from the host driving each round, which no operator fusion addresses.
