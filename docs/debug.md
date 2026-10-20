# Debugging a hand-written kernel on A3

The CUDA and ROCm equivalents all exist here, and they are all installed in
`trt-asc`. The table is the map; the sections below are what each one
actually did on a real hang, including where two of them stopped being useful.

| CUDA | ROCm | Ascend | on PATH |
|---|---|---|---|
| `cuda-gdb` | `rocgdb` | `msdebug` (LLDB 19, reports 26.2.0) | yes |
| `compute-sanitizer` | device ASAN | `mssanitizer` | yes |
| `cuobjdump` | `llvm-objdump` | `msobjdump` | yes |
| Nsight | `rocprof` | `msprof` | yes |
| device `printf` | device `printf` | `AscendC::printf`, `DumpTensor`, read back with `show_kernel_debug_data` | tool present |
| CPU emulation | — | `tikicpulib` with `ASCENDC_CPU_DEBUG` | library present |
| `CUDA_LAUNCH_BLOCKING` | `AMD_SERIALIZE_KERNEL` | `ASCEND_LAUNCH_BLOCKING` | env |
| `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` | — | exception dump via `NPU_COLLECT_PATH` | env |
| — | — | `simulator`, `operator_cmp` | tools present |

Run everything under `source /opt/ascend/cann-9.2.0/set_env.sh`, or
`libruntime_common` fails to resolve `ErrorManager::ATCReportErrMessage`.

## The runtime log is the first stop, not the last

```sh
export ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1
./program > /tmp/run.log 2>&1
```

Two lines carry most of what a MIX launch gets wrong:

```
StreamLaunchKernelV1: kernel_name=mix_expert_fused, arg_size=72, taskRation=2,
  funcType=4, coreDim=1, mixType=3
FillFftsAicAivCtxForDavinciTask: set schemMode=0, taskType=0
```

`taskRation`, `coreDim` and `mixType` say how the kernel was actually
dispatched, and `schemMode` says whether batch scheduling is on. Reading that
`schemMode=0` is what turned a week-shaped problem into a one-flag problem.

## The architecture flag decides whether `__schedmode__` exists

`__schedmode__(N)` expands to `__attribute__((cce_schedule_mode(N)))` and is
the only batch-scheduling lever a raw `<<<>>>` launch has. Under
`--cce-aicore-arch=dav-c220` the attribute compiles without complaint, emits
no `-Wignored-attributes` warning, and the runtime still reports
`schemMode=0`. Under `--npu-arch=dav-2201` the same source reports
`schemMode=1`.

So `ARCH=--npu-arch=dav-2201` is the default in
`tests/glm52/mixbrick/build.sh`. A kernel that needs cross-core
synchronisation and is built with the other flag has no batch scheduling and
no diagnostic saying so.

## mssanitizer

Four tools, the same four as `compute-sanitizer`: `memcheck` (default),
`racecheck`, `initcheck`, `synccheck`. It refuses a kernel built without
instrumentation —

```
[mssanitizer] WARN: synccheck is not supported on the kernel that without
`--cce-enable-sanitizer' compiler option, thus it's temporarily ignored.
```

— so `SANITIZE=1` adds that flag. Never time an instrumented build.

It earned its place before it reported anything. The instrumented compile
rejects an attribute list the plain compile accepts, which is how
`[[bisheng::core_ratio(1, 2)]]` was found to be a second, conflicting
declaration of the AIC:AIV ratio already given by
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` in the body. Removing it
made a kernel that had hung every run return.

Its limit: `synccheck` reports at kernel exit, so on a kernel that never exits
it prints `Start synccheck sanitizer on kernel ...` and nothing else. A hang
is not its case.

## msdebug

LLDB with the familiar interface — `--batch`, `--one-line`, `--attach-pid`,
`--wait-for`. `DEBUG=1` adds `-g` to the device compile so there is something
to break on.

```sh
msdebug -b -o "run 8 1 1 1 2 1" -o "quit" -- ./mixbrick
```

runs and stops the process. But `-o "breakpoint set --name <kernel>"` followed
by `run` segfaults `msdebug.bin` inside
`SBDebugger::RunCommandInterpreter`, so breaking on a device kernel by name
does not work this way yet. Unfinished: whether a source-location breakpoint,
an attach to a running process, or `--wait-for` gets inside the kernel.

## Bisect the kernel from the host, with a stage mask

A hang tells you nothing unless you can run one side of it. Give the kernel a
bitmask argument, one bit per stage, and sweep it:

```
stages=1  cube half only              returns
stages=5  cube and vector halves      returns
stages=2  the cross-core flag alone   hangs
stages=3  cube and flag               hangs
```

Four runs, and the cross-core flag is the whole problem — not the Matmul, not
the vector code, not their interaction. That is a cheaper instrument than any
of the tools above and it should be reached for first.

## Two traps in the build itself

Do not pipe the compiler through `head`: it dies of SIGPIPE and the script
reports a build failure with no errors anywhere in sight. Capture to a log
instead.

Do not grep that log for `error` alone. Warnings carry the answers —
`-Wignored-attributes` is how you learn an attribute was dropped.

## What the MIX route needs, beyond the above

`ASCENDC_CUBE_ONLY`, defined before `lib/matmul_intf.h`. MIX-mode `Matmul` is
driven from the *vector* side: `Iterate` on the AIV notifies the AIC, which
answers when the matrix work is done. That is the KFC channel, and it is why a
vendor MIX entry carries a `__kfc_workspace__` parameter. Calling the API from
inside `if ASCEND_IS_AIC` without this macro is the server half running the
client's code with no client.
[reference](reference/ascend-launch-overhead-and-sync.md)
