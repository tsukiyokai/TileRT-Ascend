# Width 8 is not a valid result yet: the trajectory diverges run to run

Three arms, one tree, one frozen prefix, 1024 output tokens, three warmups and
three measured requests each, `--target-graph --draft-graph
--draft-projection-tp --compile-target --super-kernel --attention-kernel sfa
--fp32-head`:

| arm | median TPOT | range | rounds per request | token hashes |
|---|---|---|---|---|
| width 4 | 14.932 ms | 0.169 | 307, 307, 307 | `df10d06f911c` ×3 |
| width 8 | 13.276 ms | 0.940 | 220, 236, 238 | three distinct |
| width 8 + hoist | 13.041 ms | 4.080 | 287, 246, 212 | three distinct |

**Width 4 is deterministic and width 8 is not.** The same binary, prefix and
flags give three different token streams at width 8, with acceptance ranging
4.655, 4.339, 4.303 and the round count varying by 8%. The hoist arm is worse:
3.568 to 4.830.

So the 4.08 ms spread is not timing noise — it is different trajectories
taking different numbers of rounds. Which means two things.

The width-8 TPOT is confounded: its difference from width 4 mixes a real
speed change with an acceptance change, and the project's own ablation
established that acceptance changes are not speed. And the hoist's −0.235 ms
here is unreadable, sitting far inside its own arm's spread, against the
−0.7967 it measured at width 4 with tight ranges.

More seriously, **width 8 fails the trajectory gate**, which compares tokens
bit for bit.

## The attribution, and it is already patched in this repository

The installed kernel is the original: the source at
`csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h`
has no barrier after the `DataCopy` at line 209, and the vendor binary is
timestamped 16:20, before any of tonight's builds.

That is exactly the epilogue race measured independently: the shipped object,
given bit-identical inputs four times, produces calls 0 and 1 agreeing while 2
and 3 each differ, codes by up to 114 and scales by 0.83 and 1.02, with all
five input tensors verified for zero drift around every call. Adding one
`PipeBarrier<PIPE_V>()` after that copy makes four consecutive calls agree.
[decomposition](../gmm-fastpath-20261019/README.md)

`tests/glm52/patches/gmm1-swiglu-barrier.patch` carries that line and is not
installed.

## Why both measurements of width 8 can be right

An earlier investigation reported five runs at width 8 emitting bit-identical
tokens; this one reports three streams from three requests. A race is
sensitive to timing, so both can hold — and that is the reason a determinism
check has to run in the same session as the number it licenses, rather than
being inherited.

## What has to happen before width 8 counts

Build and install the barrier-patched kernel, then re-run width 8 and check
that the three measured requests agree bit for bit and on their round count.
Only then is the median a speed claim. Until that is done, **width 8's
13.276 ms is withdrawn as a TPOT result**, and the width-4 figure of 14.932 ms
on this tree stands as the only deterministic point measured here.

Note what this does not touch: the hoist's −0.7967 ms was measured at width 4
with all six arms token-identical over 319 rounds, so it stands on its own
configuration.
