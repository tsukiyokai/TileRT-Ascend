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
