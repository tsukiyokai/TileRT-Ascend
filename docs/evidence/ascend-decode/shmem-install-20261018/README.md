# Symmetric memory installed and brought up on all sixteen dies

The one-sided push needed an API that this machine did not have. CANN's
official SHMEM SDK supplies it, and it now builds, installs and initialises
here.

## What was missing and where it came from

No `aclshmem` SDK ships in CANN 9.1.0 or 9.2.0 on `a3-84` — only
`pypto/experimental/shmem.py`, a PTO wrapper with no runtime behind it. The
SDK is a separate CANN component at `gitcode.com/cann/shmem`, v1.7.0, tagged
2026-10-16.

The container has no outbound network, so the clone ran on the workstation
and the tree was shipped in as a tarball. Build and install took under a
minute in total:

```
cd shmem && CMAKE_BUILD_PARALLEL_LEVEL=32 bash scripts/build.sh
  -> install/aarch64/SHMEM_1.7.0_linux-aarch64.run      (exit 0, 10 s)
./SHMEM_1.7.0_linux-aarch64.run --quiet \
  --install-path=/workspace/opt/shmem --install
  -> /workspace/opt/shmem/shmem/1.7.0                   (exit 0)
```

`scripts/build.sh` with no arguments targets A2/A3, and the quickstart lists
800I/800T A2/A3 explicitly. Installed into the workspace, so the system CANN
is untouched; `set_env.sh` only prepends to `LD_LIBRARY_PATH` and `PATH`.
Delivered: `libshmem.so`, `libshmem_utils.so`, and two bootstrap plugins,
`aclshmem_bootstrap_uid.so` and `aclshmem_bootstrap_config_store.so`.

## Bring-up, and a retraction

The bundled `examples/init` runs in a `default` mode that needs no MPI — it
bootstraps through a TCP config store rather than broadcasting a unique id,
so each rank is a plain process with no launcher. Two PEs first:

```
pe 0: shmem init SUCCESS
pe 1: shmem init SUCCESS
[SUCCESS] pe 0: demo run success!   [SUCCESS] pe 1: demo run success!
```

Then all sixteen, which is the scale that defeated this project before:

```
bash run.sh -mode default -pesize 16 -gnpus 16
  init SUCCESS x 16
  === Execution succeeded (mode: default, pesize: 16)! ===
```

A first attempt reported eight failures because `GNPU_NUM` defaults to 8
while `-pesize` was 16, so only half the ranks started and the domain could
not form — the flag is `-gnpus`, not inferred.

**This retracts the recorded conclusion that TP16 symmetric memory dies with
507015 on this hardware.** That failure belonged to a hand-rolled mapping
path, not to the scale or the chip: sixteen PEs form a domain on the first
try through the vendor SDK. The HyperMPMD dissection predicted exactly this,
since its own bindings name 507015 as what an untimed device-barrier spin
does rather than as a property of sixteen ranks
([dissection](../hyper-mpmd-20261018/README.md)).

## What this unblocks

The gate the whole one-sided argument waited on
([locating](../rendezvous-locate-20261018/README.md)) is open. Three vendor
paths were measured carrying a per-operation floor that depends on nothing
about the data — HCCL reduction 63.5 µs in-graph, HCCL at two ranks and
64 bytes 45.98 µs, `tensor.copy_()` across dies 160 µs without pipelining —
and none of them can measure the wire. A kernel writing into a mapped peer
address can, and the SDK now provides the mapping.

## The wire, measured at last

`examples/shmem_perftest` builds `ascendc_perftest`, which issues a one-sided
put between two named dies:

```
./build/bin/ascendc_perftest -t put -d float   --block-list 4,1024,98304,1572864 --device1 0 --device2 1 --loop-count 100

block_size 98304, datasize 65536:  time 2.7484 us
[Verification] SUCCESS: All cores' first values transferred correctly!
```

**A one-sided put of 64 KiB from die 0 to die 1 costs 2.7484 µs**, verified
correct. Against HCCL's 63.5 µs in-graph and 126 µs eager for a 96 KiB
reduction, that is 23 to 46 times cheaper — and the 2.75 µs already contains
the transfer, which HCCL's figure did not depend on at all.

So the derived "single-digit microseconds" is now a measurement. Everything
the earlier localisation attributed to a scheduled synchronisation stage is
confirmed as removable: the wire was never the cost.

The run dies afterwards on `std::vector larger than max_size()` inside the
example's own sizing code at `datasize=131072`, which is a harness bug
unrelated to the transport.

## Graph capture is a real risk, and it is the vendor's

`aclgraph_demo` fails on this machine: `sync stream failed` with outputs
stuck at zero across every size, `retCode=0x91` in the plog. That is the
vendor's own demonstration of symmetric memory inside an ACL graph, not our
code, so capture compatibility is now a named open risk rather than a
hypothesis — and the production decode path is entirely graph-captured.

`notifywait` remains to be read: it is the local flag spin that replaces a
barrier.
