# CANN model execution for M1 decode

[CannModel](../../../tilert/models/glm_5_2_ascend/cann_model.py) reuses the
pinned vLLM-Ascend model, weight processing and CANN operators. TileRT
owns prefix injection and the decode loop. `NPUWorker` initializes the
device and TP groups and loads the model; the adapter calls the model
directly, with no vLLM scheduler, serving process or speculative model.

The handoff state for serial decode is
`(main_KV[0:P], position=P, pending=prefill_first_token)`.
The first call consumes that pending token and produces its successor.
All P cache rows remain intact. Replaying the last prompt token would
require the prefill and decode arithmetic to agree at that position;
[the inspected norm and fused prolog paths differ](prefill-norm-prolog-20261017.json).

The adapter binds the existing paged latent/RoPE and INT8/BF16 index
layouts to 78 main attention layers and 21 indexers. It uses the existing
SFA and indexer metadata builders for one decode row. Logical TP rank
selects the weight shard; physical device selection remains independent.
Reset invalidates cache length. The next request replaces its valid rows.

The [device entry](../../../tests/glm52/run_cann_model.py) saves logits,
normalized hidden state, new cache rows and source hashes. Acceptance is
complete-prefix identity, one-row position advance, exact reset replay,
then next-token and numerical comparison against the supplied reference.
CPU API checks establish the pinned interfaces; full model execution and
PD serving integration require device acceptance. MTP follows M1 numerical
acceptance and retains its separate target-hidden and draft-cache state.

## Host verification

The [installed API probe](cann-adapter-cpu-api-20261017.json) constructs the
real engine configuration, cache bindings, SFA/indexer metadata and forward
context on CPU at positions 0, 1, 8192 and 8193. Torch-NPU remains
uninitialized. The [source check](cann-model-static-20261017.json) records
21 passing syntax and pinned-interface checks. The request, TP, MTP and
connector [regressions](cann-model-host-checks-20261017/check_ascend_pd.json)
pass 120 cases across five reports in that directory. These checks cover
the host interfaces; they do not execute weights or NPU kernels. The
[PD entry check](cann-model-host-checks-20261017/check_cann_pd_entry.json)
also verifies explicit Torch-NPU registration before NPU device parsing.

The [current CPU HF reference](hf-cold-current-20261017.json) completes
teacher tokens `[154820, 198]` and exact reset replay. Its raw traces remain
at `/workspace/experiments/glm52/hf-cold-current-20261017-attempt-01`.
The [comparison check](cann-layer-comparison-cpu-20261017.json) accepts the
identical trace and rejects changed or missing intermediate hidden/logits.
For the cold device test, use `--tokens 154820 198 --steps 2 --layer-trace`
and `--reference` with that trace directory.

## Device acceptance

After a complete HIXL prefix capture, stop the producer and receiver and
run the continuation alone on an idle A3. The capture manifest must include
all 16 rank receipts and the original 8192 prompt token IDs.

```bash
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
ASCEND_CUSTOM_OPP_PATH=/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer \
  HCCL_BUFFSIZE=200 HCCL_OP_EXPANSION_MODE=AIV \
  HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_HOST_SOCKET_PORT_RANGE=auto \
  OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1 \
PYTHONPATH="$PWD:${PYTHONPATH:-}" TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -m torch.distributed.run --standalone --nproc-per-node=16 \
  tests/glm52/run_cann_model.py \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --capture-case /workspace/experiments/glm52/prefix-8k-20261017-attempt-05/case-00 \
  --steps 2 --capacity 9344 --layer-trace \
  --output /workspace/experiments/glm52/cann-m1-8k-20261017-attempt-01
```

Every rank saves both forwards, verifies all injected rows remain exact,
and repeats from reset. `--reference` adds a matching independent HF trace
with the same prefix hashes and teacher tokens. Final logits, normalized
hidden and all 78 new KV rows are covered. `--layer-trace` adds the
79 hidden states and 78 layer logits used by the HF trace; routing and
index choices remain outside this adapter's coverage. The report keeps
`precision_verified=false` until the coverage in [the plan](../../plan.md)
is complete.

[CannTrace](../../../tilert/models/glm_5_2_ascend/cann_trace.py) captures
each preceding layer's merged residual from the next input norm. A decoder
layer's first output alone is an FFN increment, not its completed hidden.
The last hidden and logits come from the original forward. The other 77
states receive one batched final norm/head projection on every TP rank;
their BF16 tensors are compared against the independent HF trace. The
[CPU hook checks](cann-trace-cpu-20261017.json) pass 27 cases with no NPU
initialization. Hooks and auxiliary projections run only in this diagnostic
mode and are excluded from TPOT measurement.

## Saved-trace comparison

Generate the HF oracle with the captured `consumed_tokens` and the same
main-prefix files. Then run [the offline comparator](../../../tests/glm52/compare_saved_cann.py)
on CPU; it reads all 16 rank reports and tensor files without loading
model weights. It verifies file hashes, checkpoint/prompt/teacher identity,
exact reset coverage, all 79 hidden states, 78 logits and new KV rows.
Results are written separately from both inputs. The
[host validator checks](check-compare-saved-cann-20261017.json) pass 24
cases, including a changed intermediate hidden on rank 15.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tests/glm52/compare_saved_cann.py \
  --native /workspace/experiments/glm52/cann-m1-8k-20261017-attempt-01 \
  --reference /workspace/experiments/glm52/hf-cann-m1-8k-20261017 \
  --output /workspace/experiments/glm52/compare-cann-m1-8k-20261017.json
```

## PD serving

The [M1 PD entry](../../../tests/glm52/serve_cann_pd.py) uses the same
adapter with the existing HTTP request lifecycle and HIXL receiver.
Logical rank r uses physical die `(r+1)%16` to avoid HIXL self connections.
Its default memory guard is 0.4 for two resident models; actual combined
HBM use must be checked when starting P+D. The standalone continuation
must pass before this serving experiment and TPOT measurement. Finish the
request and confirm `/decode_status` is idle before stopping the service.
