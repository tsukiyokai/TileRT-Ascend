# Ascend prefill/decode handoff

This path connects a TP16 vLLM-Ascend producer to rank-local CANN decode
engines through HIXL. It serves one fresh, unchunked greedy request at a time.
Serving supports serial M=1 and MTP verification widths 2/4 through the same
TP16 request loop. Serial decode remains the default precision baseline.

## Ownership and completion

- Each decode rank owns one registered receive allocation described by
  [layout v2](../../../tilert/pd_vllm/ascend_layout.py): 196 regions,
  including 79 MLA latent/RoPE pairs, 22 index keys and 16 FP16 scales.
- A [central receiver](../../../tilert/pd_vllm/ascend_receiver.py) publishes
  sixteen rank-specific HIXL endpoints. Source rank r writes destination r;
  equal virtual addresses in different processes are separate allocations.
- The [V1 connector](../../../tilert/pd_vllm/ascend_connector.py) binds the
  actual cache views and scheduler block tables. Its `wait_for_save` returns
  after producer synchronization, synchronous WRITE and the matching
  request/rank/generation acknowledgement. Scheduler blocks remain live
  throughout the read. Metadata hashes identify the checkpoint recipe and
  tensor-to-file manifest on both sides.
- Main caches transfer P rows; MTP transfers P-1 target-conditioned parent
  rows. The receiver publishes ready only after all sixteen ranks finish
  the same generation. The slot remains occupied throughout decode.
- [Injection](../../../tilert/pd_vllm/ascend_cache.py) consumes P-1 rows in
  each cache, then replays the final prompt token through the main model.
  This restores main length P, normalized target hidden[P-1], and its greedy
  first token. Serving compares that token with the prefill result before
  emitting it. No hidden-state tensor crosses the wire.
- Cancellation retains admitted writers until their WRITE completions.
  A disconnect without completion quarantines the slot. Stop both sides
  before replacing registered allocations or restarting an endpoint.
  A cancelled prefill also retires its slot when no HTTP decode was started;
  a later HTTP announcement cannot revive that generation.

The pinned runner calls connector finalization after MTP proposal:
vLLM-Ascend `f558b58c`, `model_runner_v1.py:2981–2999`; the vLLM MTP first
pass shifts input IDs but retains parent positions (`c4df37df`,
`llm_base_proposer.py:1803–1836`). The connector requires all 79 MLA save
callbacks. This excludes a skipped MTP pass, prefix hits and partial prompts.

## MTP delivery and cache retention

`--mtp-width 2` or `4` selects
[AscendMtpPDEngine](../../../tilert/models/glm_5_2_ascend/mtp_session.py).
The decode side emits the prefill token once; it counts toward `max_tokens`.
Only requests needing a successor start MTP. Every rank completes a whole
propose/verify/recompute/commit round before rank zero delivers its accepted
tokens. Cancellation is checked between deliveries and before the next round.

For prompt length P and N delivered tokens, normal completion and
cancellation retain main length `P + max(0,N-1)`. Discovering an EOS retains
`P+N`; that EOS is not delivered. Cancellation during an accepted span
discards computed but undelivered rows on every rank. Draft length is capped
by main length; a request cancelled before the first round still has P-1
draft rows. A callback that raises has not completed delivery of that token.

`last_stats` separates completed rounds, verified positions, logical accepted
tokens and delivered `output_tokens`. Logical acceptance includes EOS and
tails withheld after cancellation; it is not response usage. Each span also
records its delivered count. The final emitted token stays pending outside
main KV unless it was consumed to discover EOS.

The TP commands are inject, start, advance, finish and reset, ordered by
serial, request ID and main position. A command failure after broadcast can
leave ranks at different serials. The backend then rejects further commands
locally; HTTP cleanup retains the slot and lock. Restart all sixteen ranks
instead of broadcasting a recovery reset.

## PD configuration

Use the existing `/opt/trt-asc/venv` and CANN 9.2.0 in `trt-asc`. Start the
decode service with sixteen ranks after device precision passes and the
host is free:

```bash
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
PYTHONPATH="$PWD:${PYTHONPATH:-}" TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  ACL_OP_INIT_MODE=1 OMP_NUM_THREADS=1 HCCL_NPU_SOCKET_PORT_RANGE=auto \
  /opt/trt-asc/venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=16 --module tilert.pd_vllm.ascend_server \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --hixl-host "${TRT_HIXL_IP:?set the business IP}" \
  --hixl-library /workspace/experiments/hixl-build/build-sync-contract/libtilert_hixl.so \
  --capacity 9344 --mtp-width 1 --ctrl-port 5556 --http-port 18080 \
  --output /workspace/experiments/glm52/pd-serving
```

The [launcher](../../../tilert/pd_vllm/ascend_server.py) registers one HIXL
endpoint per rank at ports 26100–26115 and gathers their descriptors before
starting HTTP. It preloads main/MTP experts. `--expert-bank` selects the
separately validated CANN GMM candidate; the default keeps the per-expert
baseline. No device timing is inferred from startup or CPU checks.
After serial and MTP numerical acceptance, use `--mtp-width 4` for the
speculative path in the traffic budget. The output metadata records the
selected width and implementation hashes.

Set the vLLM producer to TP16, PP/DP/PCP/DCP=1, expert parallel off,
`max_num_seqs=1`, chunked prefill and prefix caching off, block size 128,
and a fixed positive MTP draft count with `parallel_drafting=false`.
Prompt length plus that count must fit
the draft model length; the scheduler token budget must fit the whole
prompt. Use the same local ModelSlim checkpoint and mixed C8 index recipe.
The [prefix capture receiver](measurement.md#capture-original-vllm-prefixes)
uses the same producer configuration for frozen numerical inputs, before
the decode service is started.
The V1 connector configuration is:

```json
{
  "kv_connector": "AscendTileRTConnector",
  "kv_connector_module_path": "tilert.pd_vllm.ascend_connector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "tilert_host": "DECODE_BUSINESS_IP",
    "tilert_ctrl_port": 5556,
    "tilert_max_seq_len": 9344,
    "hixl_host": "PREFILL_BUSINESS_IP",
    "hixl_port_base": 26000,
    "hixl_library": "/workspace/experiments/hixl-build/build-sync-contract/libtilert_hixl.so"
  }
}
```

The existing router can use the decode HTTP endpoint; requests explicitly
use `temperature=0`. Prefill produces the first token. Decode first replays
the prompt row during injection, then emits that token and consumes it only
when a successor is needed. The final emitted token stays outside main KV.
During measured decode, the prefill service must be quiescent.

## CPU evidence and remaining device work

[Connector checks](check-ascend-connector.json) exercise actual local TCP
admission/completion on sixteen ranks with synthetic source addresses and
fake HIXL WRITE. They cover P=1/128/129/8192, mixed C8 scales, distinct
cache-group block tables, page strides and producer block lifetime.
[Control checks](ascend-pd-control.json) cover request ownership, cancellation,
generation checks, prompt replay and individual token wakeups.
[MTP TP checks](mtp-serving/check-mtp-tp.json) and
[MTP session checks](mtp-serving/mtp-session.json)
exercise the real command executors, rank engines and round controller with
independent scalar caches. They compare accepted/rejected spans, EOS,
capacity tails, callback cancellation and repeated requests with serial AR.
[Runner metadata checks](mtp-serving/native-runner.json) reject missing
layers/fields and corrupted prompt/new rows, and verify 322 prefix/new
comparison scopes, including INT8 keys and FP16 scales. Tensor arithmetic is
replaced in this metadata check.

Streaming uses [TokenQueue](../../../tilert/pd_vllm/token_queue.py), waking
the response on every token instead of the former 200-ms poll. A decoder
thread still alive after a cancellation join timeout retains its request
slot and HTTP lock until it terminates; a failed reset retains ownership.

These checks execute on the a3-gs host CPU without Torch. Actual HIXL
Initialize/registration/DMA, NPU prefix injection/replay, V1 callbacks,
HTTP delivery timing and full numerical parity remain device acceptance
work. Follow the precision→profiling→calibration order in
[the development record](offline-development.md).

After main and MTP HF parity, run
[run_native_mtp_session.py](../../../tests/glm52/run_native_mtp_session.py)
on sixteen ranks. It uses the production serving factory, HCCL command
channel and rank engine; compares normal/cancelled/repeated requests with
serial CANN AR plus all valid target-conditioned MTP cache rows; and checks
every rank before reset. Frozen prefix rows compare exactly; replay/new rows
compare separately to preserve their numerical error scale. Its CLI matches
`run_native_mtp_rounds.py`. A cold prompt is
given with `--tokens`; an 8K case uses paired `--prefix`/`--mtp-prefix` and
the last prompt token. This runner checks device execution consistency;
HIXL and HTTP delivery timing are separate acceptance measurements.
