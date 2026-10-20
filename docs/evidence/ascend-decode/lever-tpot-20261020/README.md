# Each lever, measured end to end

One arm, one variable, read the record. Host 7.150.10.194, the same warm
directory for every arm so the 245 static kernels are compiled once, and the
same flags otherwise -- `--target-graph --compile-target --super-kernel
--attention-kernel sfa --fp32-head --lookup`, which is what `launch_arm.sh`
supplies. Six requests an arm; the first is a warm-up and the mean is over the
other five.

| arm | one variable | requests, ms | mean |
|---|---|---|---|
| tail8 | the drafter's tail at 8, as shipped | 10.9518 10.9262 10.9466 10.9217 10.9224 | **10.9337** |
| tail32 | the drafter's tail at 32 | 11.1580 11.1157 11.0746 11.2049 11.1642 | 11.1435 |

## The drafter's tail is worth −1.9%, not nothing

Lengthening it costs 0.21 ms, and the CPU harness could not have seen why.
That harness counts accepted tokens; it does not count what the drafter spends
to propose them. `LookupDrafter.commit` builds one tuple key a suffix length
for every committed token, so a tail of 32 does twenty-four hashes of up to
thirty-two elements where a tail of 8 does six of up to eight -- four times
the host work, per token, against a round that commits about seven. The
acceptance the longer tail buys at this workload's copy rate is small, by that
same harness's own curve, and the host cost is not.

So the first of the four levers to get an end-to-end number gets a negative
one, and the useful part is the flaw it exposes: an acceptance harness that
ignores the proposer's cost can only ever report an upper bound.

## And the tiling lever does not cross the operand width

`SetFixSplit(16, 128, 512)` is worth 16% on the INT8 routed kernel and costs
the INT4 one a factor of two. Swept at sixteen experts on sixteen blocks:

| baseN | baseK | INT4 wall, µs |
|---|---|---|
| the library's own choice | | 39.480 |
| 128 | 512 | 80.243 |
| 128 | 1024 | 79.756 |
| 256 | 512 | 39.276 |

The reasoning that produced the INT8 corner -- narrow N, deep K, fill L0B,
fewest K iterations -- was derived from a profile of the INT8 kernel and does
not transfer. For INT4 the library already picks near the corner and the fixed
split is removed rather than retuned.
