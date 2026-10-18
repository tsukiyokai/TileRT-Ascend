# The layer megakernel: skeleton before arithmetic

Written 2026-10-19, after the night's measurements closed every incremental
route. This fixes the structure; the arithmetic goes in afterwards, because
the structure is what cannot be changed later.

## Why there is no other route

A round may cost 6.866 ms at acceptance 3.433. The three honest terms are
3.886–5.873 ms of weight stream, 3.70 ms of one-sided reductions, and 5.861 ms
of host launch path. Any two exceed the budget. All three sum to 13.4 ms at
the optimistic end.
[arithmetic](evidence/ascend-decode/onesided-audit-20261018/README.md)

They are not three problems. The host path exists because the round is 2,416
separately launched pieces; the collective cannot overlap because a standalone
kernel leaves no cores running the work it should hide behind, measured at
0.000000 ms of overlap; and the weight stream runs at 18% of what the die
delivers because the vendor's pipeline synchronises between its stages even
when there is one split. One cause, three symptoms.
[round accounting](evidence/ascend-decode/round-accounting-20261018/README.md),
[weight floor](evidence/ascend-decode/weight-floor-recount-20261018/README.md)

## What the vendor's own megakernel does

`examples/shmem_mega_moe` in the CANN SHMEM tree is 4,896 lines and fuses
routing, cross-rank dispatch, two expert GEMMs, SwiGLU and the combine into
one device pipeline. Three things in it are structural rather than incidental.

Communication and compute overlap by **partitioning cores**, not by
interleaving on one core. In `Process()`, `vectorSubBlockIndex_ == 1` runs
`BuildRoutingMetadataAndDispatch` while `== 0` runs `RunFirstProjection`, for
the kernel's whole life, coupled by flags. Half the vector cores are a
communication engine that never exits.

Its entire use of the one-sided API is three calls — `aclshmem_ptr`,
`aclshmem_my_pe`, `aclshmem_n_pes`. Symmetric memory is addressable, so a
remote write is an ordinary `DataCopy` to the pointer `aclshmem_ptr` returns.
It never calls `put_nbi`, `quiet`, or `barrier_all`. Replacing the wrapper
with a direct store measured 22.4 → 14.4 µs here.
[direct store](evidence/ascend-decode/direct-store-push-20261018/README.md)

It pipelines with paired `SetFlag`/`WaitFlag` on named hardware events —
`MTE3_V`, `MTE3_MTE2`, `MTE2_V`, `V_MTE3` — across a `PIPELINE_BUFFER_COUNT`
double buffer with explicit prologue and steady phases. Not one
`PipeBarrier<PIPE_ALL>` in a hot path. Naming the dependency instead of
draining the pipeline took 14.4 → 12.5 µs.

## The skeleton, in the order the constraints impose

### 1. Core partition, decided once

A3 gives 24 cube and 48 vector cores per die. The partition is static for the
kernel's life, because a flag protocol between roles that change is a protocol
nobody can reason about.

**The cube is mandatory, and an earlier version of this document said
otherwise on a bad count.** `dav-c220-vec` has no int4 and no int8 multiply —
`Intrinsic_vmul` takes float16, float32, int32 and int16 — so every quantised
weight must be widened and sign-restored before it can be multiplied, which
costs 8.13 vector element-ops per W4 weight against a budget of 3.69 at
1540 GB/s. I had counted four instructions per 128 weights by silently
assuming fp16 weights, and fp16 weights cost two to four times the bytes that
are the whole point.

This is settled by measurement rather than estimate. The repository's TileLang
W4A8 GEMV, now verified bit-exact against a CPU reference on device over 18
cases, runs `(256, 6144)` at 31.398 µs — 25.05 GB/s, 1.63% of what the die
delivers. One layer's batch-one expert work is 453.05 µs against the vendor
operator's 116.2, **3.90× slower**, and every identified headroom together
(waking all 48 cores rather than 16, a wider `block_k`, fewer output pairs)
lands at parity, not a win. Per core the kernel reaches 16.3% of nominal
throughput because each vector instruction takes about 23 cycles where its ALU
work is 4.3 ns, and the weight load is capped by DMA granularity — 256-byte
segments at stride 6144 — at 71 GB/s, which is 4.6% of the die and not a
bandwidth limit at all.

The premise behind the bad count was also wrong in a way worth keeping:
routed decode has a reuse factor of exactly 1 at batch one, since top-8 over
256 experts gives each active expert a single row. There is no MAC
amplification to exploit and no reuse to cache.

So: the cube does the products, the vector cores do normalisation,
quantisation, the SwiGLU epilogue and the ordered reduction, and a reserved
subset of them owns remote writes. The MIX structure stands.
[decomposition](evidence/ascend-decode/gmm-fastpath-20261019/README.md)

### 2. On-chip budget, decided once

UB is 196,608 bytes per core and that is the whole budget — a measurement that
has already cost one silent corruption, when a reduction buffer at a 192 KiB
offset wrote nowhere and reported success. Every buffer comes from one
`TPipe`.

Reserved regions must be respected if any SHMEM call is used: the SDK places
SDMA at 191 KiB, RDMA at 190, UDMA at 189, with
`ACLSHMEM_INTERNAL_UB_BUF_START_ADDR` at 188 KiB. Using `aclshmem_ptr` plus
`DataCopy` avoids the staging buffer `put_nbi` requires, which is a strong
reason to prefer it in an epilogue where UB is already tight.

### 3. Flag protocol, decided before any arithmetic

Three rules the night produced.

Eight-byte kernel parameters go first. An eight-byte argument after a
four-byte one is read from past the args buffer on this device; the SDK
commits this in its own barrier kernel, so `aclshmem_barrier_all()` has never
worked here and swallowed 964 instances of 507015 in one run.
[two defects](../patches/cann-shmem-1.7.0/README.md)

A remote write is ordered only by the writer draining its own store queue.
`aclshmem_quiet` early-returns when the transport mask is zero, which it is on
single-machine HCCS, and `sync_vec_all` states it does not complete
outstanding communication. An `MTE3_S` event pair is what orders a
`DataCopy` to a peer.

Correctness must not depend on a parity. Two planes with alternating parity
needs a round counter, and a graph replay runs no host code, so a
host-incremented counter freezes at capture; with an odd number of reductions
per captured graph, adjacent rounds then share a plane. Synchronising after
the reduction removes the hazard instead of scheduling around it, and one
plane suffices. Cost is one rendezvous, 5.016 µs at sixteen ranks, which grows
as the logarithm of rank count.

### 4. Weight schedule, part of the skeleton

A round reads 5.985 GB per rank with `--draft-projection-tp`. Only the verify
forward walks 78 layers; the three drafts walk layer 78 alone, so the stack is
traversed once per round and **there is no second read for a cache to serve**.
Layer-major reordering wins 1.555× on a synthetic stream and loses 7% on the
real GMM, and even a working reorder is worse than the single pass already is.

L2 residency is real at 2.07–2.21× with total footprint below 0.5×L2 as the
criterion, but no residency API exists on A3: `NORMAL`, `PERSISTENT` and
`LAST` compile to byte-identical objects because the implementation is gated
on architecture values that exclude 2201, and the vendor's own tree never uses
either hint. What exists is prefetch, and a prefetched line is an ordinary
first victim.

So the schedule is a stream, not a cache plan: weights enter once, in layer
order, and every consumer of a tile must be ready when it arrives. That is
what makes the fusion boundary a layer rather than an operator — a tile read
once must feed the projection, its push, and its reduction before it is
evicted.

One caution against over-reading this. **At batch one the expert GMM is not
weight-bound**: crossing L2 costs it 3.36 µs of a 98 µs call, while 40% goes
to scanning 248 empty `group_list` slots and 23% to the kernel launch floor.
The weight stream is the floor for a *perfect* kernel, not the cost of the
present one, and the two must not be confused when ranking work.

### 5. Then the arithmetic

Normalisation, quantisation, projection, push, ordered reduction, residual,
next normalisation. Each is small and each is replaceable without touching
anything above.

## The first measurable brick

`tests/glm52/shmem/fused_proj_reduce_kernel.cpp` compiles with a NZ-weight
`Matmul<>` and a per-tile push in one `KERNEL_TYPE_MIX_AIC_1_2` body, which
established that the two can share a kernel. After the correction above it is
the right shape: the cube does the product because nothing else can.
`tests/glm52/shmem/vector_proj_kernel.cpp` is kept as the record of the
refuted alternative.

It has no number yet. The gate is ≥1.3× against the operators it replaces, at
the real shapes, with bank rotation to defeat L2, and with a payload that
varies per round so the check can see a stale buffer.

And one discipline the GMM work made non-negotiable: **measure inside a
captured graph, never eager**. That operator is host-bound in eager mode and
an event pair around it reads the host — 109.90 µs of enqueue against a
113.04 µs reading — which silently inverted one conclusion already.

## What this document must not become

A plan whose parts are each justified by the whole. Every item above cites a
measurement, and the three that do not — the core partition's instruction
count, the vector path's feasibility, and the claim that a layer is the right
fusion boundary — are the next three things to measure, not to assume.
