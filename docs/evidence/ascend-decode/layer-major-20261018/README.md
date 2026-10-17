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

## Complete 78-layer teacher execution

The [full execution audit](full-teacher-audit.json) checks 1,264 artifacts,
2,464 exact hidden/residual handoffs covering 484,442,112 elements, and
81,788,928 unique causal index entries. All sixteen final logit tensors
are bit identical. Peak allocation is 940,374,016 bytes/die. Execution
takes 1,171.65 seconds in this residency order; it supplies no decode timing.

Every rank matches 31/32 frozen successors. The only mismatch is zero-based
step 19, position 8211, input 323: prediction 279 versus label 64019.
Both BF16 logits equal 16.75, with zero top-1 margin.

The [exact production-weight diagnostic](head-production-ties-exact-audit.json)
first converts checkpoint FP32 head rows to the deployed BF16 dtype, then
forms exact rational dot products with the frozen BF16 normalized hidden.
Scores for 279 and 64019 are 16.6934759437 and 16.7554145380: both correctly
round to 16.75. All three tied-maximum rows, seven candidates in total,
select the frozen labels at this higher output resolution. This is a
CPU diagnostic. Earlier raw-FP32-weight dot products in the server artifacts
are superseded by this dtype-correct calculation.

The [CANN FP32-head audit](head-fp32-audit.json) then executes the existing
`aten.mm.dtype(BF16, BF16, FP32)` on those frozen normalized outputs.
All sixteen actual `[9680,6144]` vocabulary shards run sequentially on one
die; weights remain BF16. M1 and M4 each match 32/32 successor labels,
with 4,956,160 FP32 logits bit identical between query widths. All winners
are unique; the smallest margin is 0.00261688. Peak allocation is
142,149,120 bytes. Seven exact production-weight dot products differ by
at most 1.1684e-5, six FP32 ULP; this is not a correctly-rounded FP32 claim.
The production constructor uses default global padding 64 and needs no
extra rows at vocabulary 154880. Earlier manually padded 9728-row probes
are superseded for product-shape comparisons.

The target and independent MTP head expose `--fp32-head` through the
precision/timing runners, retaining BF16 weights and the original gather
and vocabulary crop. The default remains BF16. The [actual TP16 M1/M4
processor, gather and crop](head-tp-audit.json) match the same-shape frozen
FP32 reference on all ranks, including all 32 teacher successors. Each width
checks 79,298,560 logits against the reference and between ranks.

The [local head Graph audit](head-graph-audit.json) replays all sixteen
shards sequentially on one die at M1/M4. Thirty-two graphs consume the
32 changing normalized hidden rows through 640 replays. All 9,912,320
FP32 logits match their eager artifacts bit for bit; both widths preserve
32/32 successors. Peak allocation is 141,264,896 bytes.

The [TP16 head Graph audit](head-tp-graph-audit.json) executes the official
processor, HCCL gather and vocabulary crop with one actual weight shard per
die. M1/M4 eager, changing-input Graph and recaptured reverse-order Graph
produce 96 saved matrices: all 475,791,360 FP32 values match the canonical
concatenated reference bit for bit. Within-rank, cross-rank and M1/M4
comparisons also agree; each variant retains 32/32 successors. Sixty-four
captures supply 1,280 replays, with output poisoned to NaN before each
replay and every graph reset before shutdown. Peak allocation is
143,805,952 bytes/die. The independent MTP head and complete new-head
trajectory remain separate checks. These head-only runs supply no TPOT.

The process exits normally and postflight shows sixteen healthy dies with
only the existing external jobs. One CPU `resource_tracker` shared-memory
warning remains in the log; explicit process-group destruction was not
observed. Raw head-Graph artifacts and audit stay in
`/home/shanshan/trt-asc/offline/cann-fp32-head-tp-graph-20261018-01/`.

The [reference configuration audit](reference-config-audit.json) binds
the baseline's original vendor, MTP3/graph4/async settings, and the
separate MTP1/eager prefill request that supplied this KV. Prompt IDs and
first token agree; the baseline request's KV was not saved. The prior
same-configuration native/TileRT/reset paths also choose 279 at position
8211. These observations distinguish the wrapper comparison, configuration
changes and prefix provenance; they do not identify a specific cause for
the original service's nondeterminism. HF remains the independent
diagnostic. Release acceptance remains eight prompts with 32/32 successors
each, plus the long trajectory check.

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
The command above retains the BF16 head and its 31/32 result. Add
`--fp32-head` to exercise the new output dtype through the complete runner;
the head-only evidence does not substitute for that execution.
