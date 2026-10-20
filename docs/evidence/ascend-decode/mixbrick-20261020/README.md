# A MIX kernel that works: cube, cross-core flag, vector, in one launch

The shared expert's gate-and-up projection, its dequantisation and its SwiGLU
in one hand-written AscendC kernel against the same work as two. This is the
pattern a per-layer fused kernel needs, so the point of the brick is that the
pattern runs at all; the absolute cost is not competitive yet and is said so
below.

`tests/glm52/mixbrick/`, one shared die, no model, both arms in one process at
the same block count. Raw output in [die1.txt](die1.txt).

| | 8 rows | 32 rows |
|---|---|---|
| fused, µs | 31.414–31.873 | 36.952–37.015 |
| cube kernel plus vector kernel, µs | 35.761–36.176 | 45.459–45.548 |
| saved | 4.2–4.4 | 8.4–8.6 |
| ratio | 1.13–1.14 | 1.23 |
| against the host reference, 1 ulp | 0 of 1024 | 0 of 4096 |
| against the two-kernel arm, exact | 0 of 1024 | 0 of 4096 |

The INT32 product is exact and the host reference reproduces it; SwiGLU is a
transcendental, so one unit in the last place of BF16 is agreement. The two
arms agree bit for bit.

## Five defects, in the order they had to be removed

Each one hung every launch, and four blind bisections found none of them.

**`[[bisheng::core_ratio(1, 2)]]` is not `__mix__(1, 2)`.** The macro has two
definitions, one per compilation pass — `__attribute__((aic))` for the cube
half, `core_ratio` for the mix — while the C++11 attribute applies
unconditionally to both. The sanitizer build rejects the attribute outright in
that position, which is how it surfaced.

**`--cce-aicore-arch=dav-c220` drops `__schedmode__` silently.** No
`-Wignored-attributes`, no error, and the runtime reports `schemMode=0`.
`--npu-arch=dav-2201` on the same source reports `schemMode=1`. Batch
scheduling is what makes a cross-core wait answerable at all.

**MIX `Matmul` is driven from the vector side.** `Iterate` on the AIV notifies
the AIC, which answers when the matrix work is done — the KFC channel, and the
reason a vendor MIX entry carries a `__kfc_workspace__`. Calling the API
inside `if ASCEND_IS_AIC` is the server half running the client's code with no
client. `#define ASCENDC_CUBE_ONLY` before `lib/matmul_intf.h` selects the
cube-side-only route instead.

**Mode 0 does not cross from cube to vector.** Mode 0 synchronises cores of
the same kind, so `CrossCoreSetFlag<0x0, PIPE_FIX>` from the cube raises a
flag only another cube would consume and the waiting vector core never sees
it. Mode 2 is the crossing one. This was the last and longest-lived of the
five.

**`GetSubBlockIdx()` inside the shared vector routine.** A MIX kernel's two
vector subblocks take alternate rows; a vector-only kernel is one core and
must take them all. Reading the subblock index inside the routine wrote every
other row in the two-kernel arm — 512 of 1024, exactly half, and the fused arm
was right the whole time.

## How each one was found

A stage bitmask on the kernel argument, one bit per stage, with the cube's set
and the vector's wait on separate bits. Four runs attribute a hang to one side
of one handshake:

| stages | what runs | result |
|---|---|---|
| 2 | the cube's set, nobody waiting | returns — the set is fine |
| 8 | the vector's wait, nobody setting | hangs — the vector half is alive |
| 16 | the vector sets, the cube waits | returns — vector to cube works |
| 10 | the cube sets, the vector waits | hung — so only this direction was wrong |

That instrument is cheaper than every tool in [debug.md](../../debug.md) and
should be reached for first. `synccheck` reports at kernel exit and is silent
on a hang; `msdebug` segfaults on `breakpoint set --name` followed by `run`.

## The channel split, and a race it exposed

Splitting the 256-wide axis across blocks takes 8 rows from 28.182 µs at one
block to 14.175 at four and 14.140 at eight — 2x, and at parity with the
vendor's own two operators for the same work (`QuantBatchMatmulV3` 11.661 plus
`DequantSwigluQuant` 3.140 = 14.80 µs) while being one kernel instead of two.
Each block projects its own run of channels twice, once inside each half, so
SwiGLU's gate `i` and up `i` stay on one core and no weight permutation is
needed. [raw](split.txt)

Two defects came out of it, both found by checking the INT32 product in
`partial` directly against a host reference rather than only the output: a
matmul that lands in the wrong place and a vector stage that reads the wrong
place are indistinguishable downstream.

`GetBlockIdx` and `GetBlockNum` do not agree across the halves — the cube sees
`blocks` of them, the vector side two per cube block. A channel range derived
from either directly gives the two halves different answers, and the columns
one half wrote the other never read; they read back as zeros in exactly the
slices the odd blocks owned. `GetBlockIdx() * blocks / GetBlockNum()` maps
both onto the cube's numbering.

And a scalar read of global memory from a vector core is not covered by the
mixed-core barrier. `token_scale` read as zero for every row but the first,
which showed as a zeroed output while the product verified. Staging it into
UB first fixes the common case.

## The flag orders nothing, and that is where this stops

The brick is not yet reliable. Across identical runs the error count is 0,
256, 288, 448, 448, 896 of 1024, and the INT32 product in `partial` verifies
every single time — so the cube writes the right values and the vector half
reads some of them before they arrive.

One experiment settles what it is not. Running with the cross-core flag
disabled gives 448, 448, 0; running with it gives 288, 448, 896. **The same
distribution either way: the flag is establishing no order at all.** It does
block — a wait with no set hangs — so it is a semaphore that releases without
publishing.

Already tried and insufficient, in this order: `PipeBarrier<PIPE_ALL>()` after
the projections and before the set; the same after the wait;
`DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE,
DcciDst::CACHELINE_OUT>` on both sides, which is the spelling
`distribute_barrier.h` uses and whose implementation is gated on
`__NPU_ARCH__ == 2201`; and `SetFlag`/`WaitFlag<HardEvent::MTE2_S>` before the
scalar read.

So the handshake needs to be something other than a bare
`CrossCoreSetFlag<0x02, PIPE_FIX>` plus `CrossCoreWaitFlag`, and the next
change is to find out what the working hand-written kernels actually use
rather than to try a fifth barrier.

## What it is not yet

`BLOCK_DIM` is 1, so one cube core pulls the whole 1.57 MB of gate-and-up
weights. The vendor's `QuantBatchMatmulV3` does the same shape in 11.661 µs
across four blocks; this kernel takes about 31. The fusion saves 4.2 µs a
layer at that block count, 0.32 ms a round over 75 layers, but the absolute is
three times the vendor's and the next change is the multi-core N split.

That split needs the gate and up rows interleaved in the weight so a core owns
whole pairs — SwiGLU multiplies gate `i` by up `i`, and an N split on the
natural layout puts them on different cores. The permutation is free at load
time. After it, the dynamic quantisation's row maximum becomes a cross-block
reduction, which is the next sync question.
