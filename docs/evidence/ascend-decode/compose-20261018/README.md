# The composed optimisation set on an idle machine

`a3-84`, whole machine idle, TP16, source `04a6628`, the same frozen 8K
prefix and checkpoint for every case. `--max-tokens 1024 --capacity 9344
--warmup 3 --requests 3`, profiler off. Cases run back to back after a wait
for the driver to reclaim HBM.

| case | flags beyond `--target-graph` | TPOT (ms) | tok/s | rounds | A | round (ms) |
|---|---|---|---|---|---|---|
| default | — | 21.210 | 47.1 | 303 | 3.376 | 71.61 |
| no-compile | `--draft-graph --draft-projection-tp --attention-kernel gather_fia --fp32-head` | 19.496 | 51.3 | 319 | 3.207 | 62.52 |
| composed | the above plus `--compile-target --super-kernel` | 16.640 | 60.1 | 298 | 3.433 | 57.12 |

Medians of the three measured requests; the spread within each case is small
(composed: 16.637, 16.640, 16.704). The three requests of a case agree on
round count, so each trajectory is deterministic.
[default](idle-default-rank0.json), [no-compile](idle-no-compile-rank0.json),
[composed](idle-composed-all-rank0.json); token dumps and per-round records
were dropped from the archived reports.

The composed set is 21.5% below the default and the round falls 20.2%, from
71.61 to 57.12 ms. The compiled target with SuperKernel accounts for
19.496 → 16.640, that is 2.856 ms or 14.6%, and this is the first measured
latency gain for SuperKernel: the earlier screen could only report a task
count falling from 44 to 34 and stated that it did not establish a latency
gain. Against the vLLM-Ascend baseline p50 of 16.390 ms this is a ratio of
1.015, where `--target-graph` alone was 1.247.

Distance to the target is now 8.32×, from 10.61× at the default.

## The compiled target needed a cache workaround

`--compile-target` failed on this stack before this run: `AOTAutogradCache`
asserts its entry is an `OutputCode`, while the npugraph FX compiler returns
`npugraph_ex.npu_fx_compiler._CompiledFxGraph`, so `aot_stage2_inference`
raised `AssertionError: expected OutputCode, got <class
'...._CompiledFxGraph'>` during model construction. Setting
`TORCHINDUCTOR_AUTOGRAD_CACHE=0` bypasses `_cache_inference_info` and the
compiled path runs. Every compiled measurement needs that variable; without
it `--compile-target` and therefore `--super-kernel` are unreachable.

## What this does not establish

`precision_verified` is false in all three cases, and acceptance moves with
the configuration (3.376, 3.207, 3.433), so the numerics changed: `--fp32-head`,
`--draft-projection-tp` and `gather_fia` each alter rounding, the draft
proposals differ, and the delivered trajectories are not the same tokens.
TPOT remains comparable because it is time per delivered token over a fixed
1024 outputs, but the composed configuration still owes the frozen-teacher
and eight-input acceptance of the plan's section 4.
