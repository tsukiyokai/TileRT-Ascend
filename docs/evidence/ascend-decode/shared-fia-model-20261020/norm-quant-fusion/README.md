# Dense post-attention norm/quant fusion

The historical chain-w8 compiled artifact, SHA451d95d43dcc1a18a3ddcc5c9463c23f829eda08860391b75b50c9090f097ebe,
contains exactly three AddRMSNorm→DynamicQuant substitutions, at dense
layers0/1/2. [Exact calls and consumers](compiled-contract.json) bind beta,
epsilon1e-5, mask[true,false] and no smooth scale. Eager execution produces
BF16 normalized values before quantization.

A3-194/trt-asc, die9, CANN9.2/B114: the [single-die contrast](result.json)
uses frozen HF layer0 post-attention operands and current checkpoint gamma.
M8 repeats that M1 pair; it is not the historical M8 candidate input.

| Width | Different INT8 values | Maximum code difference | Scale difference | Residual |
|---|---:|---:|---:|---|
|1|451/6144|1|2.56449e-5|bit identical|
|8|3608/49152|1|2.56449e-5 on all8 rows|bit identical|

Exit0; peak Torch allocation17,617,920 bytes. [Prepared operands](prepared.json),
[API registration](cpu-api.json), [probe](probe.py) and [launch](run.sh)
retain the contract. Raw outputs SHA2295bcbf57630c520f9563854a9baddca46fbc6b507d83d2c9aaceb8eab7e6d8
remain at `a3-194:trt-asc:/workspace/experiments/m32-full-teacher-20261020/norm-fusion/device-01/`.

The actual config [passes10 CPU checks](config-check.json): fusion is off,
Graph/static/SuperKernel stay on, and no NPU initializes. Fresh-process
vLLM/NGE passes retain separate norm/quant only in the OFF arm.

The [compiled device contrast](compiled/raw-audit.json) passes on die9:
one outer Graph capture per arm, original→changed→restored inputs in fixed
buffers,18 raw q/scale/residual comparisons. OFF matches unfused execution
bit for bit; ON reproduces3608 changed INT8 codes on the repeated M8 input.
Actual compiled IR retains static/SuperKernel settings. Peak allocation
is17.5MB; no timing. [Probe](compiled/check.py), [launcher](compiled/launch.py),
[raw auditor](compiled/audit.py), [OFF IR](compiled/device-off/compiled-ir.py).
Each arm uses a fresh process and cache: global pattern registration and
the compiler hash, which omits the fusion flag, otherwise confound the switch.

The production ACLGraphWrapper/UpdatableGraph path also passes on A3-84
with SuperKernel enabled: [135 independent CPU checks](acl-graph/acl-01-independent-cpu-audit.json)
verify the saved tensors, input changes, restoration and actual compiled IR.
Each arm captures once and replays three times; peak allocation17,400,832B.
The ON/OFF output bytes match the raw-Graph contrast. This covers the norm
subgraph; full-model compilation and request delivery remain separate.

Next propagate only these three substitutions through the frozen target
rounds. Historical compiled device ELF identity remains unknown; the local
operator contrast alone does not attribute the old token flip.
