# Layer-major frozen-teacher replay

The [runner](../../../../tests/glm52/run_cann_layer_major.py) keeps one
decoder layer resident and executes its teacher tokens sequentially at
M=1/DecodeOnly. A3 `a3-gs:trt-asc`, TP16, W4A8C8, frozen 8192-token
prefix; sixteen external jobs remain present. The Torch quota is 2 GiB/die.

For fixed teacher inputs, node `(layer, token)` depends on that token's
preceding-layer hidden/residual, earlier KV rows in the same layer and the
same token's latest full-index selection. Layer order followed by token
order satisfies all dependencies. Carry hidden and residual separately;
save each token's top-2048 and restore it at every shared-index layer.
This preserves M1 shapes and numerical configuration during the reorder.

| Device control | Raw tensor comparisons | Elements compared |
|---|---|---|
| [2 layers × 2 tokens](dense-audit.json), sixteen ranks | 416 | 1,355,776 |
| [7 layers × 2 tokens](mixed-index-audit.json), sixteen ranks | 1472 | 5,716,000 |

All comparisons are bit exact against frozen token-major CANN tensors.
They include merged layer boundaries, new KV and selected indices; the
separate hidden/residual components also match their following-layer
inputs. Layers 3–5 restore shared selections per token. Layer 6 covers
INT8 index keys and FP16 scales. Peak allocated memory is 940,005,376
bytes/die, with layer ownership released between iterations.

The indexed loader removes the C8 quant class's unused `q_rot/k_rot`
parameters before checking weight coverage. The installed class creates
empty placeholders and its postprocess is a no-op; actual C8 execution
uses the indexer's `q_hadamard/k_hadamard` buffers. The layer-6 bit control
includes the resulting keys, scales, indices and hidden boundaries.

Teacher inputs and successor labels come directly from
[teacher-32.json](../vllm-baseline-20261017/teacher-32.json). Its prompt hash
identifies the original JSON file; the captured prefix hash identifies
compact token IDs. The runner checks both representations against the
same original prompt. Final norm and LM head execute once per M1 row.
Complete 78-layer/32-step agreement remains a separate end-to-end check;
the controls above establish the reorder, not full teacher acceptance.
There is no latency or TPOT measurement in this execution order.

```bash
torchrun --standalone --nproc-per-node=16 \
  tests/glm52/run_cann_layer_major.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --capture-case /workspace/offline/prefix8k-allocation-20261017/capture/case-00 \
  --teacher-json docs/evidence/ascend-decode/vllm-baseline-20261017/teacher-32.json \
  --prompt-json /workspace/experiments/glm52/prompt-8192-code-review.json \
  --layers 78 --steps 32 --attention-kernel sfa --output /workspace/offline/teacher
```

Use the validated SwiGLU overlay, deterministic AI_CPU HCCL and 200 MiB
HCCL buffers, as in the frozen controls. Server artifacts reside under
`/home/shanshan/trt-asc/offline/cann-layer-major-20261018-{01,02}/`:
frozen source, launch/environment records, raw per-layer tensors and audits.
