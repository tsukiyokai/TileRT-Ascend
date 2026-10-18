"""Replace the TP sum with a one-sided push and an ordered local reduction.

The vendor collective costs 63.5 microseconds in-graph for this round's 96 KiB
message and none of that is transfer: a four-byte reduction costs as much as a
six-megabyte one, so the figure is rendezvous. Writing straight into every
peer's symmetric window and summing locally in rank order measures 22.3
microseconds as a kernel and agrees with HCCL element for element. Over the
round's 157 reductions that is 9.97 milliseconds against 3.50.

The sum is taken in rank order, so the result is bit-reproducible — the
trajectory gate compares tokens bit for bit, and EP16 failed it precisely by
changing a summation's association.
"""

import ctypes
import os

LIBRARY_DIR = os.environ.get(
    "TILERT_SHMEM_LIB_DIR", "/workspace/src/shmem-build/shmem/build/lib",
)
HEAP_BYTES = 1024 * 1024 * 1024


def _load():
    """Resolve the kernel globally before the binding that references it.

    The binding links the kernel target but does not record the library as a
    runtime dependency, so loading it first is what makes `tp_push_demo`
    resolvable.
    """
    import torch

    ctypes.CDLL(f"{LIBRARY_DIR}/libtp_push_kernel.so", mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(f"{LIBRARY_DIR}/aclshmem_torch.so")


class OneSidedReduce:
    """Own the symmetric domain for one rank and reduce FP32 tensors in it."""

    def __init__(self, rank, device, world_size, endpoint):
        import torch

        _load()
        # attr_init reads the calling thread's ACL device, and inside a vLLM
        # worker that thread may have none: it then tries device -1 and
        # setup_heap fails with ACLSHMEM_INNER_ERROR.
        with torch.npu.device(device):
            self.manager = torch.classes.ShmemOps.Manager()
            status = self.manager.attr_init(rank, world_size, HEAP_BYTES,
                                            endpoint)
            if status != 0:
                raise RuntimeError(f"aclshmem init returned {status}")
            self.reducer = torch.classes.ShmemOps.AllReduce()

    def sum(self, value):
        """Return the TP sum of `value`, which the caller keeps in FP32."""
        import torch

        out = torch.empty_like(value)
        self.reducer.compute(out, value)
        return out

    def close(self):
        self.manager.finalize()
        self.manager = None
        self.reducer = None


def install(engine, endpoint):
    """Point the engine's TP sum at a one-sided reduction.

    The engine already funnels every TP reduction through one override to
    force FP32, so this replaces that override's body and inherits the
    bfloat16 round trip.
    """
    import torch

    reduce = OneSidedReduce(engine.rank, engine.device, 16, endpoint)

    def one_sided(value):
        if value.dtype == torch.bfloat16:
            return reduce.sum(value.float()).bfloat16()
        return reduce.sum(value)

    engine._tp_sum_group._all_reduce_out_place = one_sided
    engine.tp_reduction = "one_sided_float32"
    return reduce
