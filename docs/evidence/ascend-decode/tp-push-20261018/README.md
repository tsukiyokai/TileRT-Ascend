# A hand-written one-sided TP reduction: mechanism works, my kernel does not

The gate was a sixteen-rank one-shot push plus ordered local reduction,
measured against HCCL's 63.5 µs. The mechanism is confirmed available and the
numerics are correct; the kernel I wrote is two orders too slow and the fault
is squarely in its reduction stage.
[kernel](../../../../tests/glm52/shmem/tp_push_kernel.cpp),
[harness](../../../../tests/glm52/shmem/main.cpp).

## What works

Built as an example inside the installed SHMEM SDK tree. Each rank publishes
its `[4, 6144]` float partial into slot `my_pe` of every peer's symmetric
window with `aclshmemx_mte_put_nbi`, drains with `aclshmemx_mte_quiet`,
announces with `aclshmemx_signal_op(..., SET, peer)`; every core then spins
`aclshmem_signal_wait_until` on all sixteen local slots and sums in rank
order. No barrier and no collective.

Each rank contributes its own index as the payload, so the sum must be
`pe_size*(pe_size-1)/2` in every element and a wrong order cannot pass. At two
PEs it reports `wrong 0 of 24576, expected 1.0 got 1.0`. **The one-sided
reduction is numerically correct.**

## What does not, and the measurement that found it

| configuration | per reduction |
|---|---|
| push and reduce | 543 µs |
| **reduce only, no communication at all** | **557 µs** |
| reduce only, vectorised through UB | 358 µs |

Splitting the kernel into push-only and reduce-only stages was the whole
diagnosis: with communication switched off entirely the cost is unchanged. So
none of the 543 µs is the push — **it is all my reduction loop**.

The first version accumulated with a scalar `for` over elements, so each of
24576 elements paid sixteen separate global reads. Staging the stripe through
UB and using `Add` brought it to 358 µs, which is still absurd for 96 KiB —
roughly 1 µs of traffic at the measured rate. The remaining fault is visible
in the code: it re-copies all sixteen partials into UB every round and places
`PipeBarrier<PIPE_ALL>` inside the loop, draining the whole pipeline on each
of the sixteen adds. The vectorised variant also hangs in stage 0, so its
flag protocol and its UB region are interacting.

Two incidental notes. `LocalTensor` is constructed by assigning
`address_.logicPos`, `address_.bufferAddr` and `address_.dataLen` directly,
as `examples/notifywait/main.cpp:120` does; `SetAddrWithOffset` does not have
a matching overload here. And the seven-argument
`aclshmemx_mte_put_nbi(dst, src, buf, ub_size, elem_size, pe, sync_id)` is the
GM-to-GM form that stages through a UB scratch, which is the one to use when
the source is in global memory — the five-argument form requires the source
already in UB.

## Where this leaves the argument

Nothing here contradicts the case for the one-sided route, and one thing
sharpens it: with the push removed the time did not move, which is the same
result as the size sweep and the wire measurement arriving from a third
direction. The push is not the cost. The remaining work is a competent local
reduction, which is ordinary vector-kernel engineering rather than a question
about the platform.

Established today and unaffected: the wire is 0.31 µs within a module and
0.84 µs across one
([wire](../shmem-install-20261018/README.md#sub-microsecond-and-the-topology-appears));
sixteen PEs form a symmetric domain; and the SDK's own `combine_classic`
performs push plus ordered local reduce at our shape, `-pes 16 -bs 4 -h 6144`,
with `[Combine] check passed`. That example reduces only `bs` tokens with a
scalar loop, which is why its loop shape was a poor model for a full
`[4, 6144]` sum and why copying it cost this detour.
