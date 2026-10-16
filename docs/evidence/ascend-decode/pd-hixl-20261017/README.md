# A3 HIXL handoff checks

Environment: `a3-gs:trt-asc`, CANN 9.2.0, Torch/Torch-NPU 2.12.0. The
transport bridge SHA256 is
`cee498b49da610c77a50d00de200b7bc4ddf1c1304bf6a3dea19cb6f870864f5`.

| Check | Observation |
|---|---|
| Independent processes, die 0 → die 0 | [Connect status 103900](same-die.json); receive allocation remains unchanged. |
| Independent processes, die 0 → die 1 | [1 KiB WRITE passes](cross-die.json); all receiver bytes equal the source pattern. |
| Production host command channel | [16 real Gloo processes](gloo-idle.json) receive all four commands in order after 141 seconds of idle time; none imports Torch-NPU. |
| TP16 metadata and admission | [30 PD checks](check-shifted-pd.json), [29 connector checks](check-shifted-connector.json), and [20 session checks](check-session.json) pass on CPU. |

Run [check_transport.py](../../../../tests/hixl/check_transport.py) with
`--sender-device 0 --receiver-device 0` or `1`; each JSON records the full
command, source hash, library hash and receiver result. The installed HIXL
`comm` engine rejects self-device connections before `TransferSync`:
[original errors](hixl-self-device-20261017/pid-30305-connect-error.log),
[engine selection](hixl-self-device-20261017/pid-30305-engine-selection.log),
and [versioned source audit](../hixl-self-device-20261017.json).

The receiver explicitly selects physical die `(LOCAL_RANK+1)%16`; its
logical rank and weight shard remain unchanged. Source rank r writes
destination rank r. Visible device order stays unchanged.

The request command channel uses a separate CPU Gloo group.
[check_control_channel.py](../../../../tests/glm52/check_control_channel.py)
executes its actual object broadcasts with CPU Gloo as the default group;
the next device run also checks coexistence with the HCCL computation group.
The earlier
[8K attempt](prefix-8k-20261017-attempt-02/inputs.json) waited
[141.05 seconds](prefix-8k-20261017-attempt-02/control-idle-check.json), then
failed at the HIXL self-device connection. Its
[receiver log](prefix-8k-20261017-attempt-02/receiver.log) and
[client error](prefix-8k-20261017-attempt-02/client.log) preserve that run.

The next 8K attempt used the shifted device mapping. An external job then
occupied all sixteen dies before prefill initialization completed:
[occupancy snapshot](prefix-8k-20261017/occupancy-conflict.json).
Only this experiment's passive receiver was stopped;
[the external processes remain](prefix-8k-20261017/occupancy-after-stop.json).
Full 8K KV transfer and decode continuation await an idle A3.
