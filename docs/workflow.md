# Workflow, revised from one night's failures

Written 2026-10-19 after a night in which six measurement-instrument errors
were caught, five of them mine, and in which five consecutive attempts at one
16-rank measurement failed for five different reasons without once being a
fault in the thing measured.

## The one rule the night produced

**Where a control is available, do not reason.** Every error below had an
experiment that would have settled it in one run, and in each case I argued
instead.

| what I did | what the control would have cost |
|---|---|
| divided bytes by peak bandwidth to price a transfer, three times, wrong by 3.7× each | one measurement |
| inferred "the build can be frozen" from address stability, twice, both silently wrong | one polarity run |
| attributed 314 casts to a round trip because 157 × 2 = 314 | one grep, which refuted it |
| attributed 1434 casts to the Python frame spanning them | a `connection_id` join |
| concluded the cube was unnecessary from an instruction count that assumed fp16 weights | reading my own kernel |
| "fixed" a free-memory symptom I had caused, and the fix broke execution | running an agent's working tree |
| ranked device idle by what precedes each gap, then built a 68% attribution on it | one controlled removal, which the design could not resolve |
| trusted a round's p10 holding to 0.6% inside a run as reproducibility between runs | a control arm repeating the previous chain's configuration |
| expected per-call measurements to predict rounds, twice, wrong in both directions | measuring in a round, which is the same run |
| eliminated kernel candidates for a day while the carrier was a cache key | reading back which binary the arm loaded, from the arm's own record |

## 2026-10-19 additions

Four habits the day argues for, each with the failure that bought it.

**Measure the quantity you intend to change, in the configuration you intend
to ship.** A profiler adds host work to a device the host feeds, so every
figure it produced had to be hedged and one was then refuted. The round's own
record already carried per-round elapsed time; reading that settled an A/B
whose TPOT medians overlapped.

**Keep one arm in every chain that repeats the previous chain's
configuration.** It is the only reason a wrong conclusion was caught within
the hour, and it costs one arm. Two identical configurations differed by 0.201
ms in a round's p10, which is the size of the effects being looked for.

**Alternate the arms when the effect is near that floor.** A B A B A B inside
one session, paired pass by pass. One arm after the other cannot separate the
patch from the drift, and the published conclusion that it could had to be
withdrawn.

**Check that the parts sum to the whole.** The phase hook's first report had
the phases totalling 124.505 ms against a 57 ms round, which is how it
announced that it was averaging warmup rounds together with measured ones. A
decomposition that does not add up is reporting its own defect.


## Measurement

Smoke the launch path with the shortest possible run before paying for the
real one. Five failures at 4–40 minutes each — a shared tree carrying a live
edit, a missing flag, `activate.sh` blocking off a tty, a device-binding race,
a bound rendezvous port — would each have surfaced in an 8-token run in three
minutes.

Measure inside a captured graph unless you have checked the host cost. An
eager event pair can read the host: 109.90 µs of enqueue behind a 113.04 µs
reading on one operator, while another's host cost is 28.6 against 60 on
device and eager is safe. One `perf_counter` beside the event pair decides
which case you are in.

Churn an allocation between calls in any determinism check. Eight identical
calls in a tight loop read a known-racy kernel as deterministic in all ten
cells of a matrix, because the loop never changes the workspace the operator
receives.

Vary the payload with the round, on a period coprime to any buffer parity. A
constant input is blind to a stale plane, to round crosstalk, and to an
implementation that communicates once and re-reads — which would also report a
*lower* per-round cost.

Design the check so the failure you are excluding changes the answer. Having
each rank contribute its own index made every partial sum an integer below
2²⁴, so the check was constant across all 16! orderings: 200000 random orders
gave one distinct value.

Check determinism in the session that licenses the number. Width 8 was
bit-identical across five runs in one investigation and gave three streams
across three requests in another; both are true because the cause is a race.

## Attribution

Join on correlation identifiers, never on time containment. A blocking
`tolist` spans 1429 µs and swallows every concurrent task on every other
stream. A task that joins to no API event is not missing data — it is a task
inside a captured graph.

A matching count is not a mechanism. Check the total when you believe a cost
was removed: deleting two of four consumers of a routing table moved 38 µs
from the prologue into the next stage with the call's total unchanged.

Ablation of one stage in a fused kernel gives a lower bound, not that stage's
cost, because the call is a critical path after the vector and cube paths meet
at a barrier.

Apportion a gap before naming its cause. A 61.5× shortfall factored as
2.2 × 3.0 × 6.2, and the hardware property everyone reached for first was the
smallest of the three.

## Delegation

Brief with what you have ruled out, in the same form you would want it. Three
agents each refuted a proposal of mine with a named mechanism and a file:line,
and twice what they refuted would have computed silently wrong results.

Isolate by what the run imports, not by what the work is about. Snapshotting
one repository does not isolate a measurement whose model code lives in
another.

Do not race your own agents for an exclusive machine. The machine has no lock,
so the discipline has to be mine; when it is held, the productive work is
everything that needs no device — reading a profile, pricing an item, writing
and compiling a kernel.

Ask for the narrowest established bracket rather than a conclusion when an
agent has been running long. "The first round where two runs diverge, and what
was identical up to it" is actionable; a list of suspects is not.

## What a report must contain

Every number its measurement conditions. Every claim about code a file:line.
Every causal claim a differential rather than a correlation. A retraction when
the evidence does not establish what was claimed — eleven on the first day,
six instrument errors on the second, and the ones caught by an agent's control
rather than mine are the reason to keep the adversarial pass running.

## 2026-10-19 evening: the launch recipe belongs in code

Ten launches failed in one evening, none of them a fault in the thing
measured, and three of the five causes were already in this repo's own list of
environment facts "each of which cost a launch". They were kept as prose. The
evening's three drivers — 38, 42 and 44 lines — each rebuilt the same recipe
and each dropped a different piece: the autograd cache, the gate, the port
range. Details in [lesson.md](lesson.md); what matters here is where they now
live.

**`launch_arm.sh`** is one arm: the guarded source, the overlay prepend
`activate.sh` undoes, a cache root and a port range per arm rather than per
pass, `TORCHINDUCTOR_AUTOGRAD_CACHE=0`, the gate, and the md5 of the file
under test. Exit 2 means the gate refused, so a caller waits instead of
consuming the arm.

**`chain.sh`** is a list of them: `chain.sh <out> <tree> <overlay>
tag:width:flags...`, one port range per arm, `round_clock.py` after each. It
replaces `chain_alternating.sh`, which hard-coded the directory it looked for
records in — so every experiment but one checked the wrong place.

So an experiment is now a list of arms and a patched file, and the recipe is
not part of it. Four habits the evening also argues for:

- Run the tool that exists. `paired.py` has done arm floors and paired deltas
  since this morning; I did them by hand three times.
- A container can take a SIGKILL from outside mid-chain. A waiter that only
  greps a log waits hours on a dead one, so check `docker inspect` too.
- Read load against the core count. 39 looked like contention; the box has 320
  cores, and a3-84 was the only host at 4% HBM while the rest were at 89–94%.
- Stage the standby before it is needed. a3-187 carries the same tree, code
  and gate, verified by its own round checks, so a switch costs one launch.

Launch the driver detached, or the gate waits for nothing:

```sh
docker exec -d trt-asc setsid bash -c "exec bash /workspace/experiments/go.sh > /workspace/experiments/go.log 2>&1"
```

A driver started from an interactive `docker exec` dies with the ssh session
that opened it. Two did; their `sleep 180` children reparented to PID 1 as
zombies, and a waiter watching for records watched for two hours while nothing
ran. The symptom is a log whose last gate line is hours old with no driver in
`ps`. To stop one, read its PID and kill that: `pkill -f` inside
`docker exec bash -c` matches the wrapper's own argv and kills itself first,
leaving the driver running.

Ship a file with `scp` then `docker cp <file>`, and compare md5 on both sides.
`docker cp /dev/stdin` once truncated a driver to 15 bytes, and `bash -n`
passed on the fragment.

## 2026-10-20: do not squat on the gate

Two drivers retried the occupancy gate 150 and 13 times. Between 22:04 and
09:36 a co-tenant held 60.2 GiB a die against the 47 the gate wants, and all
three hosts read 59–62 of 64 GiB a die. Nothing ran for eleven and a half
hours. Re-arming for 400 attempts was the same mistake with a bigger number:
a retry loop is not progress, it is a bet that someone else will finish, and
it reports nothing while it waits.

Sort an experiment by what it actually needs, and run the cheapest tier that
can still decide something:

- One shared die, a few hundred MiB, no model. Paired microbenchmarks of a
  single operator. These run under full occupancy, in seconds, and they
  confirmed the MoE combine's 4x and its flatness in rows on the same day the
  whole fleet was unavailable.
- Sixteen dies, no model. Collective and protocol screens, minutes.
- Sixteen dies with the model resident. Acceptance, expert union, request
  TPOT. Only this tier needs the gate.

The last tier's queue must be short and ordered by what it decides, not by
what is convenient to append. Fourteen arms is an hour or two of exclusive
machine; a co-tenant's window will not be that long. Carry the arms that
settle one question each and drop the rest.

When the gate refuses, bound the wait to something a real window would
satisfy — tens of minutes, not a working day — and spend the refusal on the
first tier instead.

## 2026-10-20: the recipe a hand-written MIX kernel needs

Four things that are not discoverable from the headers, each of which cost a
build cycle.

A standalone AscendC program builds with `bisheng -xcce
--cce-aicore-arch=dav-c220`, include roots from the *source* tree under
`$CANN/aarch64-linux/asc` and `tikcpp`, linking `-lascendcl -lruntime`. Add
`-lstdc++` the moment the host side uses `std::vector`, or the link fails on
`operator new` and `__gxx_personality_v0`. And do not pipe the compiler
through `head`: it dies of SIGPIPE and the script reports a build failure with
no errors in sight.

A cube tiling comes from `matmul_tiling::MatmulApiTiling` on the host. Take
the `GetTiling(AscendC::tiling::TCubeTiling&)` overload -- the `optiling`
one drags in the operator registry's `TilingDef`. The link line needs
`-ltiling_api -lplatform -lunified_dlog -lc_sec -lregister`, and the program
must run under `set_env.sh` or `libruntime_common` fails to resolve
`ErrorManager::ATCReportErrMessage`. `tests/glm52/mixbrick/` does this and
prints the tiling; 8 rows gives baseM 16, 32 rows baseM 32, both with
baseK 128 over the 6144 axis.

Cross-core synchronisation in a raw-launched MIX kernel needs the sync base
address, which the host gets from `rtGetC2cCtrlAddr` -- declare it
`extern "C"`, it is not in the public headers -- and the kernel installs with
`AscendC::SetSyncBaseAddr`. That is all `util_set_ffts_config` does in the
SHMEM tree. Then `CrossCoreSetFlag<0, PIPE_FIX>` from the cube side and
`CrossCoreWaitFlag` from the vector side. The reverse direction is not settled
by the headers and a flag under `core_ratio(1, 2)` faces two waits per set, so
design the first bricks to need the cube-to-vector direction only.

`dav-c220` vector instructions are typed more narrowly than the model is:
`Muls` takes half, float, int16 or int32 and rejects BF16. Compute in FP32 and
`Cast` the roundings back where the unfused stages have them, or the result is
more accurate and no longer comparable.

Kernel debugging has its own page now: [debug.md](debug.md) maps the CUDA and
ROCm tools onto `msdebug`, `mssanitizer`, `msprof`, `msobjdump` and the CPU
simulation library, and records what each one did on a real hang. Two results
from it belong in any build script: `--npu-arch=dav-2201` rather than
`--cce-aicore-arch=dav-c220`, because the latter silently drops
`__schedmode__`; and never grep a compiler log for `error` alone, because the
warnings carry the answers.

## Pre-flight for a whole-machine queue

Three arms died today on things a single command would have caught, so this is
a checklist, not advice. Run it against the tree the arms will execute in, not
against the repository.

```sh
B=$TREE/tests/glm52/bench_cann_mtp.py
grep -oE '\-\-[a-z-]+' $B | sort -u          # every flag the tree accepts
md5sum $TREE/tests/glm52/launch_arm.sh       # against the repository's
ls $TREE/tilert/models/glm_5_2_ascend/       # every module an arm imports
```

What each one caught. `--lookup-unigram` is absent from the 10-19 tree and so
is the `lookup_min_ngram` plumbing behind it, so that arm failed in seven
seconds and took the queue with it. `onesided_reduce` was absent for the same
reason and killed fourteen arms behind it. And the tree's `launch_arm.sh` was
one line behind the repository's — `cd "$tree"` against `cd "$out"` — which
put the static-kernel cache in the tree instead of each arm's own directory,
so every arm replayed the first one's compiled package and a paired
measurement read zero.

Then two ordering rules. `chain.sh` exits on a failing arm, so anything
unproven goes last. And kill single-die work first: a kernel of ours squatting
device 1 made rank 1 unable to join HCCL, which reports as
`EI0015 ... unconnected ranks: "[1,]"` and looks like a network fault.


## One arm, one variable, and a warm directory

The queueing apparatus is gone. `chain.sh`, the per-sweep drivers, the settle
loops and the waiters each cost more than they returned: every failure today
was in the machinery rather than in an experiment -- a lost env var, a gate
read once before launching, a settle reading a field that does not exist in
the container, one failing arm taking fourteen with it, a `pkill -f` matching
its own shell. What survives is `launch_arm.sh`, which encodes the real traps,
and one rule: run one arm, change one thing, read the record.

The cold-cache cost is the reason the rule needs a second half. A first arm on
a fresh machine compiles 245 static kernels at 8 to 12 seconds each -- the
progress bar's own estimate is 32 to 46 minutes -- where the same arm on a
machine with a warm cache takes five and a half. The 101 MB of output lands in
the arm's own directory, because `launch_arm.sh` does `cd "$out"` for exactly
the reason recorded above: a shared static-kernel cache voided a paired
measurement, since its key covers the dumped operator parameters and the
super-kernel option string and not the vendor source.

So isolate by what the change touches, not by habit. An arm that changes only
host-side Python -- a drafter module, a flag the graph does not see -- can run
from an already-warm directory with `--output` pointing elsewhere, and the
compiled kernels are reused: `grep -c "static kernel compile start"` returns
zero, which is the check. An arm that changes an operator or monkeypatches a
vendor call needs its own directory, or it will replay a package compiled
before the change.

And read what the launcher already passes before claiming a flag is off.
`launch_arm.sh` supplies `--target-graph --compile-target --super-kernel
--attention-kernel sfa --fp32-head` itself, so the seven `store_true` flags
being off in the bench's own defaults does not mean they are off in a run.
Two of the largest were already on in every number this project has recorded.
