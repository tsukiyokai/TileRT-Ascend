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

## Six defects found, and the handshake still is not right

Each of these was real and each is fixed. None of them was the whole thing.

A `GlobalTensor` needs its extent. `SetGlobalBuffer(ptr)` without a length
leaves it unset, and then only the offset-zero row reads back correctly: row 0
right, rows 1 through 7 wrong, every run, from a `partial` that verified
against the host every run.

An output buffer needs a sentinel, not a zero. Four runs reading 896 wrong and
four reading 0 looked like a race until the output was pre-filled with 0x7777
and the count held at 896 every time: the allocator was handing back the
previous process's buffer, and an untouched output and a zeroed one are
indistinguishable. The sentinel also proved every element *is* written —
`untouched 0 of 1024` — which ruled out coverage and left arithmetic.

`PipeBarrier` is an issue-order barrier, 「阻塞相同流水」. The one that waits
for memory accesses to retire is `DataSyncBarrier<MemDsbT::DDR>()`, and that
is why three `PipeBarrier<PIPE_ALL>()` changed nothing. `MemDsbT::UB` after
the staging copy took the four-block arm from 2 of 6 clean to 5 of 6.

One set, not one per waiter. Under mode 2 a single `CrossCoreSetFlag` wakes
both of a cube's vector subblocks; consecutive sets of one flagId are not
allowed, and a surplus set leaves a counter that survives the process.

Flag 8 is `AIV_INTER_BLOCK_BARRIER` in `catlass` and the vendor advises
against sharing an id with the Matmul high-level API, which this kernel also
uses. Moved to 4.

The cube's C stays at `TPosition::GM`. The vendor's cube-only sample declares
it `VECIN` on the kernel side; copying that made the four-block arm worse,
448 448 448 224 896 0 against 0 0 448 0 0 0.

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
`CrossCoreSetFlag<0x02, PIPE_FIX>` plus `CrossCoreWaitFlag`.

The documentation does not settle it. CANN's own 220x architecture page says
the flag is what 「确保数据从L0C成功搬运到GM后，再从GM搬运到UB」 and mentions
no cache maintenance; the vendor's `matmul_fused_manual` sample carries no
DCCI, no `DataSyncBarrier` and no `PipeBarrier` at all, and passes. Against
that, three third-party kernels on the same 2201 measure that the flag alone
does not publish, one with this exact symptom — the consumer reads all zeros,
and 「加探针即掩盖」, adding a probe hides it. The sample's evidence is one
verification run at `numBlocks = 1` on a single tile, which cannot tell
correct from one-in-three.

Our own measurements are evidence on the second side: at four blocks the arm
is clean in 3 of 6 runs with every documented barrier in place, and at eight
blocks never. At one block it alternates, which the sentinel showed is partly
the allocator rather than the kernel. Treat the cache operations as a
workaround with a measured effect, not as the rule, and A/B them once the
handshake is correct — `ENTIRE_DATA_CACHE` is 「耗时较大」 and a per-layer
kernel would pay it per stage.
[reference](../../reference/ascend-fusion-mechanics.md)

The 2201 cache model closes the coherence question and does not fix this. L2
is shared across cores, the DCache is per-core but used only by the Scalar
unit, and the ICache is read-only; a cube-to-vector handoff runs L0C, Fixpipe,
GM, MTE2, UB — entirely through the shared L2. So there was never a coherence
problem on this path and the DCache flushes were aimed at the wrong cache.
The documented need is completion ordering, one
`DataSyncBarrier<MemDsbT::DDR>()` on the producer, and with exactly that
protocol the arm is still clean in only 3 of 6 runs at four blocks.

Two hypotheses are now closed by measurement. Two `Matmul` objects sharing one
`TPipe` is not the cause: one projection over the whole 256-wide axis
alternates 896, 0 identically to two projections over halves. And the
per-row scalar scale is not the cause: making every row's scale identical
leaves the failure unchanged.

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
