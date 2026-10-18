# 554 Cast calls a round, and 314 of them are ours

Read from `profile-rope-20261018`'s `op_statistic.csv` on rank 0, which covers
3.01 rounds (`FusedInferAttentionScore` 244 calls against 81 per round), so
per-round figures divide by three.

| operator | calls (export) | total µs | per round | µs per round |
|---|---|---|---|---|
| **`Cast`** | 1669 | 4628.479 | **554** | **1537** |
| `Muls` | 235 | 1110.214 | 78 | 368 |
| `Abs` | 235 | 207.158 | 78 | 68.8 |
| `Slice` | 132 | 176.2 | 44 | 58.5 |
| `RmsNorm` | 50 | 170.098 | 16.6 | 56.5 |
| `DynamicQuant` | 30 | 73.85 | 10 | 24.5 |
| `AddRmsNormBias` | 10 | 33.48 | 3.3 | 11.1 |
| `DequantSwigluQuant` | 10 | 31.863 | 3.3 | 10.6 |

`Cast` alone is **1.54 ms a round, 22% of the 6.866 ms a round may cost** at
acceptance 3.433 — the largest elementwise item by a factor of four and one
nobody had looked at.

## The vendor's norm-quant fusion barely fires

`vllm_ascend/compilation/passes/norm_quant_fusion_pass.py` fuses add, RMS norm
and quantise into `npu_add_rms_norm_quant`; it is gated on
`ascend_compilation_config.fuse_norm_quant`, which defaults to `True`
(`ascend_config.py:124`), and on `GRAPH_NORM_QUANT_FUSION`, which is in
`_STANDARD_CAPABILITIES` (`hardware_profile.py:270-271`), so it is enabled
here.

But the profile shows `AddRmsNormBias` at only 3.3 calls a round while
`RmsNorm` alone runs 16.6 times and `DynamicQuant` 10. So the pass covers a
small minority of the pairs it exists for, and the elementwise intercept the
census priced is what remains **after** the vendor fusion. Writing our own
norm-quant kernel would be redoing a pass that already runs; the open question
is why it matches so rarely, which is a pattern-matching question in the pass
rather than a kernel one.

## 314 of the casts are this project's own choice

`cann_model.py:254-259` overrides the TP sum to force FP32:

```python
def _tp_reduce(self, value):
    if value.dtype == torch.bfloat16:
        return self._tp_sum_original(value.float()).bfloat16()
    return self._tp_sum_original(value)
```

Two casts per reduction, 157 reductions a round, **314 Cast calls — 57% of the
554**. The one-sided replacement inherits exactly the same round trip
(`onesided_reduce.py`'s `install`: `reduce.sum(value.float()).bfloat16()`).

And it is avoidable rather than intrinsic. The one-sided kernel already
accumulates in FP32 internally — `LocalTensor<float> acc` in `TpPushReduce` —
so it can take bf16 in, accumulate in FP32 across the sixteen slots, and emit
bf16, with the casts folded into the loads and stores it already performs.
That removes 314 dispatches and their intercepts without changing the
arithmetic the reduction performs. The FP32 reduction was adopted because
bf16 accumulation across sixteen ranks loses precision the trajectory gate
notices; accumulating in FP32 inside the kernel keeps that property.

The remaining 240 casts are unattributed. `--fp32-head` is the obvious
candidate for some of them, and it is worth 1.317 ms of acceptance by the
project's own ablation, so it is not free to drop.

## What this changes

The census put the elementwise family's intercept at 1.5–2.9 ms a round and
called it the fattest target for a resident kernel. That is right, but the
first move is not a kernel: it is to stop generating 314 of the calls, which
needs an interface change in the one-sided reduction and no new arithmetic.
[census](../prologue-census-20261019/README.md)
