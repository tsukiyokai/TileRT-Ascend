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

## The channel split, and the eighth defect

| 8 rows, µs | 1 block | 2 | 4 | 8 |
|---|---|---|---|---|
| fused | 22.737 | 14.914 | **12.307** | 12.936 |
| cube kernel plus vector kernel | 25.626 | 17.732 | 15.093 | 15.621 |
| ratio | 1.13 | 1.19 | 1.23 | 1.21 |

At 32 rows, four blocks: 19.731 against 22.420. Every configuration is exact
against the two-kernel arm — 0 of 1024 and 0 of 4096 — and within one unit in
the last place of a host reference that computes the INT32 product exactly.
Twenty-four consecutive clean runs across four block counts. [raw](split.txt)

The vendor does these two stages as `QuantBatchMatmulV3` 11.661 µs plus
`DequantSwigluQuant` 3.140, which is 14.801. This kernel does both in 12.307,
one launch instead of two — 17% faster than the pair, and the pattern a
per-layer kernel is made of.

Each block projects its own run of channels twice, once inside the gate half
and once inside the up half, so SwiGLU's gate `i` and up `i` stay on one core
and no weight permutation is needed.

## The defect, and the seven that hid it

**The two halves of a MIX kernel number the grid differently, and the count
does not convert between them.** Measured, not assumed: at four blocks the
cube reports `GetBlockNum` 4 with `GetBlockIdx` 0 to 3, while the vector side
reports `GetBlockNum` 4 with `GetBlockIdx` 0 to 7 and `GetSubBlockIdx` 0 or 1.
Vector subblocks 2i and 2i+1 belong to cube i, so a vector core's range comes
from `GetBlockIdx() / 2`.

That is also why no barrier could fix it. Under mode 2 a cube raises a flag
only for its own two vector subblocks; a vector core sent to a slice another
cube wrote is waiting on the wrong producer, and ordering a write that was
never the one it reads cannot help. The symptom was exactly this shape: the
slices a core's own cube wrote came back right, the rest held whatever had
landed, the count moved with scheduling, and `partial` downloaded after the
kernel verified every time because every cube does finish — just not always
before some vector core has read.

Seven other defects were real and fixed on the way, none of them sufficient.
A `GlobalTensor` needs its extent, or only the offset-zero row reads back.
An output buffer needs a sentinel rather than a zero, or an untouched buffer
and a zeroed one are indistinguishable and the allocator's residue looks like
a race. `[[bisheng::core_ratio(1, 2)]]` is not `__mix__(1, 2)`: the macro has
one definition per compilation pass, the attribute applies to both, and the
mismatch hangs every launch. `--cce-aicore-arch=dav-c220` drops `__schedmode__`
silently where `--npu-arch=dav-2201` keeps it, visible only as `schemMode` in
the runtime log. MIX `Matmul` is driven from the vector side through the KFC
channel unless `ASCENDC_CUBE_ONLY` is defined before the include. Mode 0 does
not cross from cube to vector and mode 2 does. And flag 8 is
`AIV_INTER_BLOCK_BARRIER` in `catlass`, which the Matmul API also uses.

## What the cache model says, and what it does not

On NPU architecture 2201 the L2 cache is shared across cores, the DCache is
per-core but used only by the Scalar unit, and the ICache is read-only. A
cube-to-vector handoff runs L0C, Fixpipe, GM, MTE2, UB — entirely through the
shared L2 — so there is no coherence problem on it and every DCache flush
tried here was aimed at the wrong cache. The producer needs completion
ordering, one `DataSyncBarrier<MemDsbT::DDR>()`, and `PipeBarrier` is not that:
it orders a pipe's issue, not its retirement.

That much is now settled and the kernel carries exactly it. What the cache
model could not do was find the defect, and three rounds of barrier-adding
were spent before the grid numbering was measured. The instrument that
mattered was a stage bitmask with the cube's set and the vector's wait on
separate bits, a sentinel in the output, and a probe writing
`GetBlockIdx`/`GetBlockNum`/`GetSubBlockIdx` from both halves into a buffer.
[reference](../../reference/ascend-fusion-mechanics.md)

## Hardened against two documented rules it was relying on by luck

`PIPE_S` is listed as unsupported for every cross-core mode this part has, and
it is the header's default — so a bare `CrossCoreWaitFlag` passes an
unsupported value. The consumer is now `CrossCoreWaitFlag<0x02, PIPE_MTE2>`,
which is what it actually blocks on, and the result stays 24 of 24 clean.

One set raises the counter of *both* of a cube's vector subblocks, and each
must wait exactly once or the leftover count survives into the next launch —
which is a second mechanism that produces exactly the period-two alternation
seen here. The handshake therefore stays outside any branch the two subblocks
could take differently: split the work, never the handshake.

Two corrections to earlier notes in this file and in `lesson.md`: flagIds on
A3 run 0 to 15, not 0 to 10, each AIC and each AIV has its own sixteen, and an
out-of-range id is silently truncated to its low four bits — so an id of 16
aliases onto 0. And consecutive `CrossCoreSetFlag` calls from one core are
unordered, which a kernel raising a stage flag per layer has to design around.

## The whole shared expert, cube to vector to cube, in one kernel

The second projection, 128 to 6144, makes the kernel cube, vector, cube. Its K
axis is all 128 intermediate columns, which the first stage spread across
blocks, so what stands between the two cube stages is not a per-pair flag but
`SyncAll<false>()` — the three-way MIX barrier, cube waits for all vector, all
cube sync, all vector wait for cube. It sits at top level, outside both
halves' branches, because every core must arrive even where its own stage work
is skipped.

| 8 rows, µs | 1 block | 2 | 4 | 8 |
|---|---|---|---|---|
| whole chain, one kernel | 23.802 | 16.465 | **13.863** | 14.468 |

Correct at every block count, 0 of 49152 against a host reference that
computes both INT32 products exactly, and 6 of 6 clean runs at four blocks.
[raw](chain.txt)

The vendor spends four kernels on this: `DynamicQuant` 3.84 µs,
`QuantBatchMatmulV3` 11.661 for gate and up, `DequantSwigluQuant` 3.140, and
`QuantBatchMatmulV3` 15.311 for the down projection — 33.952 µs a layer. One
kernel does it in 13.863, **2.45x**, and over 75 MoE layers that is 1.51 ms a
round.

## Per-token dynamic quantisation, correct and expensive

The static scale is gone. Each block reduces its own channel slice to a
per-row maximum, a barrier makes every block's partial readable, each block
reduces across blocks and quantises its own slice, a second barrier makes the
whole intermediate readable, and then the down projection runs. Two grid
barriers, because the down projection reads every slice and the row maximum
spans every slice. [raw](dynamic.txt)

| 8 rows, µs | 1 block | 2 | 4 |
|---|---|---|---|
| static scale | 23.802 | 16.465 | 13.863 |
| per-token dynamic | 61.678 | 36.948 | **29.343** |

Correct: 0 of 49152 at one, two and four blocks, 0 of 196608 at 32 rows, and
6 of 6 clean runs. Eight blocks fails entirely and is not yet understood.

So the chain is still ahead of the vendor's four kernels — 29.180 against
33.952, 1.16x — but the 2.45x the static scale showed was bought by skipping
the reduction.

The 15 µs is now attributed, and it is not where it looked. A stage bit that
runs the two barriers with nothing between or after them prices them apart
from the work:

| at four blocks, 8 rows | µs |
|---|---|
| gate and up projection, dequantisation, SwiGLU, partial maxima | 14.125 |
| plus the two `SyncAll<false>()` barriers, nothing else | 16.494 |
| plus the reduction, quantisation, down projection and its dequantisation | 29.180 |

**The two grid barriers cost 2.37 µs, 1.18 each.** The down projection and its
epilogue cost 1.556 in the static-scale measurement. So the quantisation pass
alone is about 11 µs, and the reason is its shape, not its arithmetic: per
block it walks eight rows, and each row does a 32-element global copy, a
scalar multiply, two casts and a 32-element copy back. Forty operations on
32 elements each, where one vector repeat is 64 floats — all of it fixed
overhead.

The fix is to batch the rows: a strided `DataCopy` brings the whole slice in
at once, the casts run over the whole buffer, and only the per-row scalar
multiply stays per row. Thirteen operations instead of forty. Two earlier
guesses were wrong and are recorded as such — the per-row scalar read of the
reduction's result, and a 32 KiB copy of the whole maxima workspace, each
changed the total by under 0.3 µs.

Two findings from building it. A reduction's destination is written a whole
32-byte block at a time, so one float a row stamps over the next seven rows'
slots — give each row its own aligned block. And a scalar store to global
memory from a vector core does not land: the per-row scales read back as zeros
while the partial maxima that produced them read back exactly right. Gather on
chip and copy out once, by MTE. That is the same rule as LCCL's, which moves
its flags by MTE and never by scalar store, and it follows from the DCache
being the one incoherent cache on this part.

The reference had to be corrected too, not the tolerance. The kernel quantises
float to INT8 through half, which is the Ascend C idiom, and a host reference
rounding once instead of twice disagreed on 4587 of 49152 outputs — 9.3%,
identical at every block count, which is what told us it was arithmetic rather
than a race.

## Next

Attribute the 15.5 µs, then the same chain for the routed experts, which is
this shape with a group dimension.
