# Idle-machine TPOT for the current default

`a3-84`, whole machine idle at launch, TP16, source `04a6628`, the same
frozen 8K prefix and checkpoint as the `a3-gs` runs. `--target-graph` only,
`--max-tokens 1024 --capacity 9344 --warmup 3 --requests 3`, profiler off.
[Distilled rank-0 report](idle-default-rank0.json); token dumps and per-round
records were dropped.

| request | TPOT (ms) |
|---|---|
| warmup | 20.543 |
| warmup | 20.789 |
| warmup | 20.619 |
| measured | 20.491 |
| measured | 21.210 |
| measured | 22.461 |

Median of the measured requests is 21.210 ms, mean 21.387. The `a3-gs`
median for the same configuration is 20.441 ms, so an idle machine is 3.8%
slower and contention does not explain the 20 ms. Each request runs 303
rounds for 1024 outputs, so A = 3.376 as before; the trajectory is
unchanged. One graph capture and 1818 replays confirm exactly one target
replay per round. `model_tp_reduction` is `float32`, so the FP32 transport
override is active, and `command_serial` of 1843 over 1818 rounds puts Gloo
at about one broadcast per round rather than one per operation.

The composed set `--draft-graph --draft-projection-tp --compile-target
--super-kernel --attention-kernel gather_fia --fp32-head` has still not been
measured end to end: the first attempt failed in `_init_device` because the
driver had not reclaimed HBM from the previous case, and by the retry another
tenant held 19.8 GiB per die, which would have made the comparison
meaningless. Leave a reclaim wait between cases and require an idle machine.
