# The bitter lesson

The entries below are a day-by-day record. What they keep teaching, in the
order a mistake usually happens:

## Before measuring

- Measure the instrument first. The replay harness charges 26.58 us before the
  graph does anything, and it was counted as a finding twice.
- Read the class the engine instantiates, not the base class whose name
  matches. A wrapper on `get_mlp_weights` installed silently where the W4A8
  method has `_get_mlp_weights` with four return values.
- Check a knob is not already in the baseline. `HCCL_OP_EXPANSION_MODE=AIV` was
  exported by two scripts in this repository before it was "found".
- Check a flag is registered before `parse_args`. Four were not, and argparse
  said only "unrecognized arguments" at run time.

## While measuring

- Measure a cost where it sits, not where it is convenient. A reduction is
  32.77 us alone in a graph and 44.62 in a chain, because a chain lets arrivals
  drift.
- Use the checkpoint's own shapes. A probe at `inner=1536` priced the expert
  bank at half its real width and counted one call where a layer makes two.
- Vary one thing. Two drafting "wins" came from arms that moved the budget and
  the width together.
- A pass criterion that cannot fail is not a criterion. Random weights with
  random scales produce absmax 2e-17, which "finite and nonzero" accepts.
- Name the capture a figure came from. Three readings had to be withdrawn
  because their capture was `max_tokens: 6`.

## When attributing

- Figures from different scopes sit side by side and cannot overturn each
  other. The collective term read 29.64, 10.36, 5.11, 1.41 and 6.10 ms, each
  measured, at five different scopes.
- Check the numerator belongs to the denominator. 374 us a layer came from
  dividing a round-wide fit by 78.
- A lever cannot remove more than the item it acts on. That arithmetic caught
  the wrong-width probe without any measurement.
- A suspect cannot supply more than its own total. The round model's row term
  is 9.77 ms at width 12 and the whole collective chain is 5.125, so blaming
  the collective was arithmetically impossible before the sweep that then
  measured its share at 2.3 percent.
- A residual names only what the probe covered. The 0.026 ms a row that looked
  31 times too small excluded SFA, the indexer and routing -- 24.6 ms of the
  round, and the only width-dependent stages left once the collective fell.
- Check that the discriminator is runnable in the code before planning a run
  for it. "Width 12 against 13" was the headline next step for a day, and 13 is
  not in `cann_graph.py`'s capture whitelist.
- A per-row figure without its width is not comparable. The gather's marginal
  row costs 13.70 us at width 12 and 2.53 at 32, so one linear term smooths a
  curve with regimes and any single slope inherits the span it was fitted on.
- Ask what grows with the parameter before blaming a stage for it. Routing is
  12.1 ms a round and looked like the biggest row-cost suspect; its buffers are
  fixed at `MAX_ROUTES = 32 x 8` and its tensors are tiny, so it is launch
  count, not rows.
- Compare like with like. A sum of separately-launched kernels is not what an
  engine that fuses them spends.
- A broken index reads exactly like a failed mechanism. Twice in one day.
- Vary one thing. The bank probe moved the weight tensor's size and the active
  group count together, read the whole effect as the tensor's, and carried the
  project's largest lever for a day. The control it never ran -- a full tensor
  with most groups empty -- took one measurement and killed it: 87.23 us against
  87.10 for a tensor of only the touched experts.
- Before pricing a lever, check the vendor has not already taken it. Five CANN
  grouped-matmul kernels skip an empty group with one `continue`, and this
  repository already anchored that line in a test nobody read.
- Tensor parallelism shards each expert; it does not deal experts out. A rank
  carries all 256 and a round touches 81, which the trace's own routing array
  says plainly. Carrying an expert-parallel intuition across moved the layer's
  traffic floor from 81 microseconds to 41 and the headroom from 2.9x to 5.7x.
- Derive a count that a distribution fixes rather than assuming it. "A round
  touches two of a rank's sixteen experts" was an assumption; eight distinct
  picks of 256 over twelve rows gives 5.07, and the routed fixture's own 81
  distinct of 256 confirms it.
- Two independent paths agreeing is evidence only if they are independent. The
  native layer's 574.15 us and the vendor layer's 575.22 looked like a
  cross-validation until both turned out to skip the empty expert the same way.
- A figure measured at one row does not price the engine's twelve. The KV read's
  294 GB/s was a one-row latency bound; at twelve rows the gather does 1094 and
  the layout step it justified does not exist.
- A regression someone else measured is a finding, not noise. The packed-KV
  candidate lost 240 of 240 paired comparisons, which said there was nothing to
  win before any mechanism was proposed for why.

## When implementing
- Four launch-path defects on a fresh host, each wearing another's clothes.
  The engine stack is not portable by copying its trees; what it imports has to
  be checked one import at a time.
  - `vllm_ascend.egg-info` under `vendor-metadata` is the only declaration of
    `ascend = vllm_ascend:register`, so leaving that directory off `PYTHONPATH`
    makes `VLLM_PLUGINS=ascend` select nothing and the engine die on an empty
    device string inside `DeviceConfig` -- a device message with no device in it.
  - No pinned vllm lists `indexer_kv_dtype="int8"`, which `cann_model.py:131`
    hardcodes for the W4A8C8 checkpoint, so it faults in pydantic before any
    device work. The literal is a validation list; `patches/vllm/` carries it.
  - The image's `triton` is two wheels on top of each other, 3.5.0's files over
    triton_ascend 3.2.2, and vllm imports `triton.experimental.gluon`
    unconditionally when `find_spec("triton")` succeeds. There is no env
    override for `HAS_TRITON`, so removing the package is the switch.
  - `fla_npu` can be neither kept nor dropped: `vllm_ascend/ops/gdn.py` imports
    it for Qwen3.5, and its `libcust_opapi.so` wants GLIBCXX_3.4.29 while the
    image and the host both ship libstdc++ 6.0.28. The same image's
    gcc-toolset-12 has 6.0.30.
- Append to `PYTHONPATH`, never assign it. `set_env.sh` puts CANN's `tbe` there
  and GE imports it; overwriting the variable turns that into
  `OpCompileProcessor init failed`, which this project had recorded as an
  environment fault in its own right. Read the Python traceback, not the Ascend
  log's tail: the real line was `No module named 'tbe'`.
- A fusion that changes accumulation order changes the answer. The FFN's
  clearest shared-load instance -- StoreRoutes scattering 1.18 MB that
  CombineTokens reads back and sums -- is blocked because summing eight bf16
  values in a different order is visible, and the stage's gate is bitwise
  against the stock chain. The hardware allows it; the contract does not.
- The cost is a pass over the data, in a kernel or out of one. A Torch sum over
  98304 floats takes 11.82 microseconds, a topk 18.59, and an in-kernel AscendC
  pass over 73728 takes 12.46 -- and the increment survives graph capture, so it
  is device time. Fusing operator calls buys nothing by itself; fusing *passes*
  is the whole lever, which is why one reduction walk instead of four paid 26.87
  and rescheduling the same passes paid 2.94.
- Price an operation's floor before replacing it. Selecting 2048 of 8192 over
  twelve rows costs 17.93 microseconds, a full sort of the same data 17.86, and
  a threshold count 21.09. The algorithm does not enter: 393 KB in 18
  microseconds is 22 GB/s, so the cost is the operator call. The selector's
  54.84 is about three calls, not an expensive algorithm.
- Separate waiting on the other half from waiting on your own. The FFN's cube
  waits 67.52 microseconds, of which 39.77 is for the vector stages -- which
  skewing can take, up to the 23.40 of vector work that exists -- and 27.75 is
  for other cube cores. The second is 81 experts not dividing by 24 cores, and
  levelling it means splitting an expert across cores, which buys another
  reduction walk. Not all idle time is recoverable.
- A pipeline step on the cube costs a reduction walk. Cutting the front stage
  into eight steps to interleave the vector half added 70 microseconds of walks
  to hide 25 of vector work. Skewing pays only where the cube is already divided
  into independent invocations -- the expert loop, the head loop -- not where one
  invocation was the point.
- Fit the increment, not the total. Fitting invocation cost against K gave
  `11.6 + 0.000985 K` and the 11.6 was read as a per-invocation fixed cost; it
  is the kernel's, paid once. Fitting the *extra* invocation gave an intercept
  of -1.58 and settled it. The wrong reading had already produced a conclusion
  that 2 ms needs model-structural change.
- A mechanism that wins once is not a rule, and the variable may not be the one
  you named. Concatenating projections that share an activation saves 31 
  microseconds at a reduction of 6144 and loses 2 at 2048, both removing exactly
  one invocation -- so depth decides, not the count.
- Check the grouping exists before pricing it. The first reading of this stage
  put four projections on one activation and measured 44.98 microseconds. The
  engine cannot use it: the indexer's key reads the hidden as bf16 where Prolog
  reads it as int8, and the router's gate is read after attention. The adoptable
  grouping is two projections and 31.06.
- Align a weight's row stride to 512 bytes. The concatenated front stage used
  3040 bf16 columns, 6080 bytes a row, and took 48.12 microseconds; padding to
  3072 columns and 6144 bytes took 38.43, a fifth off for 32 columns of zeros.
  Every stride from 3072 to 8192 costs the same, so it is the alignment and not
  the width. The unaligned kernel also looked 20 percent worse than the vendor's
  matmul on the same shape, which is what sent the search to tiling and core
  counts first.
- A define that selects types must reach every translation unit. The dtype flag
  went only to the kernel compile, so a bf16 device half was linked against an
  int8 host half; the launch faulted on the core with EE9999 and the only visible
  clue was `element_bytes` reporting 1 from a library named bf16.
- A cross-half barrier deadlocks against a spin-wait. `SyncAll` counts the cube
  and vector halves together, so a producer that barriers while the consumer
  spins on its flag hangs the core for ten seconds and reports `aicore timeout`.
- Decide a trade by its ratio before building the protocol for it. Row-axis
  pipelining buys the producer's duration and costs one matmul invocation a
  group; at a reduction of 6144 that is 25 microseconds against a producer of
  under one, so no handoff protocol could have made it pay.
- Work with no observable result is not work. A vector stage that computed in
  unified buffer and published nothing reported under one microsecond at every
  load up to 64 passes; publishing its total and checking the arithmetic turned
  that into 12.46 microseconds a pass. The compiler had removed the loop.
- One float a core shares a cache line with seven others, and a flush writes the
  whole line. The witness failed intermittently until each core got its own line.
- A broken kernel keeps the clock. The packed weight variant produced NaN and
  errors of 464 against magnitudes of 331, and timed within noise of the correct
  arm at every width. Only the digest comparison caught it.
- Read the API's meaning from a working call in the same repository, not from the
  parameter's name. In `SetOrgShape(M, N, Ka, Kb, Kc)` the weight's row stride is
  `N` and `Kb` is the reduction length; putting the stride in `Kb` compiles, runs,
  and is silently wrong when the layout actually differs.
- A Python error inside a graph capture is reported as a capture failure. Read
  the traceback before suspecting the hardware.
- Code appended after the point that consumes it never runs. Three times in one
  day: a check below `main`, a check below `__main__`, flags below
  `parse_args`.
- `pkill -f` matches the remote shell's own argv. Write the script to a file.
- A chained `cat >> ... && git commit` commits when the first half failed.
- A table whose header is one column short of its print is read one place left,
  and every value still looks plausible. `tree_draft_eval.py` labelled thirteen
  columns over twelve values for a day; the automaton's acceptance was being
  read as a long chain's.
- Do not overwrite a driver script while it runs. Bash reads a script
  incrementally, so the old driver carried on with a half-replaced view and a
  second driver started beside it; two sixteen-rank arms then contended for the
  same dies and the same HCCL port range. Copy to a fresh path and launch that.
- A killed rank that is reading the checkpoint sits in `D` state until its NFS
  read returns, so `pgrep` still counts it after SIGKILL. Drain to zero before
  relaunching, or the next arm collides with a corpse that still holds a device
  context -- the HBM figure is already back to idle and says nothing.
- `npu_sparse_flash_attention` lives in the compiled `vllm_ascend_C`. No tree
  on a fresh host carries it and the only built copy nearby is
  cpython-312/torch-2.10 against this stack's 3.11/2.12, so `--attention-kernel
  sfa` cannot run and the recorded 9.3794 ms cannot be reproduced there.
  `vllm-ascend-8237/csrc` on a3-194 is what a build would start from.
- Read the drafter before blaming its budget. Four separate figures pointed at
  "the proposal is too short" -- 3.400 accepted at budget 256, 128 to 256
  buying 1.3 percent, a 13.474 oracle, an external probe reaching 9.65 at
  budget 20 -- and `propose` was already returning the whole remaining span.
  The defect was the anchor: an index over prompt plus output makes the
  drafter's own output the most recent occurrence of everything.
- A saturating curve is a property of the instrument until you have changed the
  instrument. Acceptance looked sublinear in width, so the width optimum looked
  pinned at 32; with a cursor drafter it is near linear and widening pays.
- A truncated model is not a slower model, it is a broken one. `--layers 10`
  emitted token 1170 and then 255 zeros -- two distinct tokens in 256 -- so a
  drafter that indexes its own output accepted the entire width every round:
  the automaton read 23.273 at width 32 and TPOT read 1.71 ms. Both are the
  degenerate-repetition artefact, not a measurement. "The arm produced
  non-warmup requests" is not enough; the output has to be more than one token
  wide, which `distinct_output_tokens` now records and gates.
- Two of three drafters agreeing is not a check when they share the defect. On
  that degenerate stream lookup gave 1.969, the automaton 23.273 and the cursor
  1.000 -- and the cursor was the only one that was right, because 255 zeros
  occur nowhere in the prompt and there is nothing to copy.
- More readers on a slow shared mount is less throughput. Sixteen ranks each
  read 3 MB/s and block in `__lock_page_killable` on each other's pages, for
  about 48 MB/s against a single stream's 79. Two arms died of this before the
  worker's `wchan` was read; the progress log said "10% (1/6)" for forty
  minutes and said nothing about why.
- A constructed stream can be the mechanism under test wearing a disguise. The
  perturbed-file corpus is one monotone copy of the prompt, which is precisely
  what a cursor drafter tracks and what a tail search must re-find every round,
  so it paid the cursor 1.88x. On the engine's own streams the cursor loses
  2.6x, because real output repeats itself rather than reproducing the prompt
  in order -- and indexing prompt plus output, which looked like the defect, is
  the whole advantage there. Validate a drafter on the target's own output
  before building on the number, and keep both distributions when they
  disagree: the disagreement is the finding.
- Check that a feature's second half is wired before measuring its first. The
  tree drafter proposes and `accepted_path` resolves, but
  `tree_attention.install` has no caller, so `mtp_round.tree_mask` stays None
  and off-path nodes attend to the indexer's selection instead of their own
  ancestors. A ladder of seven arms was about to measure that.
- Assign the property, not the field behind it. `sfa_v1`'s `skip_topk` setter
  also recomputes `skip_indexer_pre_process`, which `runtime_has_indexer`
  reads, so `tree_attention.install` writing `_skip_topk` left the indexer
  running under the mask that was supposed to replace it. The vendor's own
  `set_skip_topk` assigns the property; the copy in this repository did not.
- Locate a patch by the traceback's frames, not by the file whose name fits.
  The strict loader that rejected `rot.weight` is vllm's `deepseek_v2.py`,
  which is what GLM-5.2's target class actually is; the first patch went into
  `vllm_ascend/models/glm5next/model.py` and changed nothing.
- The pinned pair had never run end to end -- `plan.md` verified it as far as
  `--help` -- so its offsets against `cann_mtp.py` come out one per launch:
  `rot.weight` has no owner, `has_own_lm_head` is gone, `lm_head` is now
  `head` and deferred. Each fix moved the failure one layer deeper, which is
  how you know the direction is right; none of them is a measurement.
- A container that vanishes from `docker ps -a` was removed, not reaped. A
  `sleep infinity` probe on the same host lived through five samples over two
  minutes, so the name was the problem: `trt-asc` collides with someone
  else's cleanup, and `shanshan-trt` does not.
- Invariance is cheaper to test than speed. The question that invalidated every
  TPOT reading overnight -- does the verification width change what is decoded
  -- has a host-side answer that costs no die and 0.2 seconds, and it had never
  been asked above width 4. A control that cannot fail is not evidence: the two
  deliberately broken engines in the same file are what make the passing check
  worth citing.
- A drafter must not raise the round's budget. `propose_tree` floored it at
  two, which is invisible at every width an arm has run and fatal at width 2 --
  the width a reference arm would use.
- Check whether the box was taken before suspecting your own residue. Three
  consecutive launch failures read as leaked device memory were three different
  tenants arriving; `npu-smi info -t proc-mem` names the holder in one call,
  and `VLLMWorker_TP` under someone else's container is not your rank.
- An inventory's verdict is not an inventory. `kbenv` ranks a host busy on a
  process name before it consults HBM, and counts a die npu-smi did not report
  as taken, so the one free A3 in the fleet read 占用 with 0 of 16 dies free
  while every card sat at 4 percent. Three hosts were abandoned on that
  reading. `scripts/free_dies.sh` reads the cards.
- The A3 devserver pool rejects key authentication, and ssh then reports a
  correct password as "Permission denied". Force it:
  `-o PreferredAuthentications=password -o PubkeyAuthentication=no`.
- A parameter the operator accepts is not a path the kernel supports. Passing
  the group list as a running sum (`groupListType == 0`) instead of counts
  passes every host-side check and then faults the AI core with 507015 at both
  projections and both expert counts. The engine passes counts because that is
  the validated path, not because nobody looked.
- Check the vendor's own parameter rule before designing around it.
  `aclnnGroupedMatmulV5` requires `size(groupList) == weight.shape[0]` under
  groupType 0, so the lever "hand the kernel a shorter group list" is not
  expressible at all -- it refuses 81 against a 256-shard weight. Four sweep
  points were spent discovering by experiment what the error message says in
  one line. Shortening the walk needs a kernel that enumerates, or a weight
  gather that moves 63.7 MB a call to save nine milliseconds a round.
- A probe that varies one number may be holding two. The expert-GMM sweep
  always handed over a 256-long list and varied only how many entries were
  non-zero, so its `59.37 us + 0.709/expert` folded the walk over all 256
  configured groups into the fixed term. "Narrowing the expert bank is worth
  nothing" was therefore only ever evidence that the bill is not in the bytes.
- Do not assemble a flag combination no build script ships. `ROUTED_ROW_BALANCED=1
  ROUTED_BALANCED_NAMESPACE=0` put the row-balanced schedule under the plain
  `tilert` name that `--routed-kernel` looks up, and the arm died on an AI core
  out-of-bounds scalar access reported asynchronously by the next kernel. The
  tested balanced build registers `tilert_balanced`; the plain name belongs to
  the unbalanced kernel. A build variant is a tested pair, not two free knobs.
- A rank wedged by an AI core fault does not answer SIGTERM. The faulted arm's
  sixteen ranks still held 87 percent of five cards fifty minutes later, and
  `pkill -f` without `-9` had already run against them. The free-memory error
  that followed reads exactly like another tenant arriving; `npu-smi -t
  proc-mem` named them as mine. Reap with `-9`, and check the holder before
  believing either story.
- An AI core fault surfaces in whoever launches next. "The address for the
  scalar to access the internal buffer of AICore is out of bounds" arrived
  through `aclnnSparseFlashAttention` and `aclnnInplaceCopy`, neither of which
  was at fault. Read the error string, not the frame.
- Run the kernel's own single-die check before a sixteen-rank arm.
  `routed_int4/check.py` compares against the vendor chain and times both, so a
  fault costs one die and four minutes instead of an engine start. Two arms
  were spent before this was remembered.
- Stage the checkpoint in tmpfs before the first arm, not after the third. One
  stream off the NAS reads 8.4 MB/s while the fleet runs its own benchmarks,
  and sixteen ranks each reading every shard put one arm's weight load at four
  and a half hours. Sixteen parallel streams reach 256 MB/s, so one staged copy
  into `/dev/shm` costs twenty minutes and every arm after it reads at memory
  speed. `--ipc host` already shares the host's tmpfs into the container, so it
  needs no extra mount. The tenants on this fleet serve their own weights from
  `/dev/shm` for exactly this reason -- read their command lines.
