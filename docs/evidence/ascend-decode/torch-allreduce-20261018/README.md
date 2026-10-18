# The one-sided all-reduce, callable from torch

The reduction measured 22.3 µs as a bare kernel
([one-sided all-reduce](../onesided-allreduce-20261018/README.md)), but the
decode path reduces torch tensors. It is now a torch operator, and it agrees
with HCCL element for element.

```
world 16  shape (4, 6144) float32  96 KiB
one-sided   28.181 us   wrong 0 of 24576  expected 120.0  got 120.0
HCCL       142.850 us   differs from one-sided in 0 elements
speedup    5.07x
```

Two checks, not one. The payload check — each rank contributes its own index,
so every element must be `pe_size*(pe_size-1)/2 = 120` and a wrong summation
order cannot pass — and agreement with the vendor collective on the same
inputs, which is zero differing elements out of 24576.

The 28.181 µs is about 6 µs above the bare kernel's 22.3, which is torch's
dispatch; the production path runs inside an ACL graph, where that floor
collapses to roughly 1 µs per kernel
([graph chain](../graph-chain-20261018/README.md)). HCCL's 142.850 µs here is
its eager figure, consistent with the 126 µs measured directly; in-graph it
is 63.5 µs, so the honest in-graph comparison is 63.5 against something near
22.3.

## How it attaches to the decode path

`AllReduce` follows the SDK's own `AllGather` class in
`examples/torch_binding/src/torch_bindings.cpp`: it allocates a symmetric
window lazily on first use, sized for two planes of `pe_size` slots, and
`compute(out, in)` issues the kernel on the current NPU stream with a round
counter so the planes alternate. `Manager` from the same binding does the
bring-up — `attr_init` with a TCP config store, and `finalize`.

The attachment point in this project needs no new plumbing.
`cann_model.py:183` already overrides `_tp_sum_group._all_reduce_out_place`
with `_tp_reduce` to force FP32 reductions, and that override is the single
funnel through which all 157 reductions per round pass. Replacing its body
with `reducer.compute` puts the one-sided path in production.

## Three build facts worth keeping

`torch_binding` is gated behind `PYEXPAND_EXAMPLE`, not `USE_EXAMPLES`, so it
needs `cmake -DUSE_EXAMPLES=ON -DPYEXPAND_EXAMPLE=ON`. Configuring in the
build directory avoids `scripts/build.sh`, which resets the cache and silently
drops every example target while leaving the previous `build/bin/`
executables in place.

The binding links `${ACLSHMEM_ALL_KERNEL_TARGETS}`, which
`aclshmem_add_collective_example` populates automatically, but the kernel
library is not recorded as a runtime dependency — so `aclshmem_torch.so`
fails to load with `undefined symbol: _Z12tp_push_demo...` unless
`libtp_push_kernel.so` is `dlopen`ed with `RTLD_GLOBAL` first.

And the registered class namespace is `ShmemOps`, from
`torch::jit::class_<ShmemOps::CLASS_NAME>("ShmemOps", #CLASS_NAME)` — not the
`shmem_ops` the directory name suggests.
