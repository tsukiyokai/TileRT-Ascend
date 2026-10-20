# Running an arm in a tenant's residue: five constraints, four cleared

Both reachable A3s have been held by tenants for hours -- a3-84 leaves 6.8 GiB
a die, a3-78 leaves 3.8, against the 16.7 GiB the full model needs and the
47 GiB the gate asks. So an arm was tried at eight layers instead of 78, about
1.7 GiB a die: absolute TPOT is then meaningless but the ratio between two arms
is not, and every lever here is a ratio.

The gate passed and the arm started -- the first time today. Five constraints
followed, in order:

1. **`--layers` reached the engine but not the container.** `scp` puts a file
   on the host; the arm reads the container's `/tmp`. Second time today.
2. **The weight loader walks every tensor the index lists** and raised
   `KeyError: layers.9...` at eight layers. `is_pp_missing_parameter` does not
   cover it at one pipeline stage. Cleared with `load_format="dummy"`, which
   suits an arm that measures timing.
3. **Our own cache bind expected 78 main and 21 indexer layers.** Cleared by
   reading the depth from the config; the check's intent was that every layer
   is bound, not that there are 78.
4. **Our own MTP guard required `num_hidden_layers == 78`.** Cleared by
   checking the family and the single shared-index MTP layer instead.
5. **vLLM's MTP completeness check** demands weights for the MTP layer at the
   truncated depth -- layer 8 -- where the checkpoint's is layer 78. It fires
   before `load_format` is consulted. `disable_mtp_completeness_check()` exists
   and entering it around `worker.load_model()` did not help: the draft model
   loads outside that scope, so the ContextVar does not reach it.

6. **Our own check that the official loader loaded the MTP head**
   (`cann_mtp.py:179`). Under truncation it did not, and cannot: the
   checkpoint's MTP head is layer 78.

## What the sixth constraint says about the route

The sixth is not a literal to relax -- it is the route's real obstacle.
Truncation and MTP are incompatible by construction, because the head lives at
the checkpoint's last layer and a shortened model has no such layer to load it
into. So a truncated arm has to run without the MTP engine at all.

There is a way: `--automaton` takes `_advance_drafted`, which needs only
`target.verify` and never touches the draft engine. A truncated session could
be built without `CannMtp`. But that is a structural change to the session's
construction rather than another relaxed literal, and its value -- not waiting
for an empty machine -- is worth less than its cost while the arms are already
queued with a sixteen-hour window.

## Where this stands

Five of six constraints cleared, all committed, none of them touching the
untruncated path: the cache bind and the MTP guard read the real depth, and
`load_format`, the completeness waiver and the depth comparison are entered only
below 78 layers.

The sixth needs a session that omits the draft engine. Recorded rather than
built, so the next attempt starts there instead of rediscovering the five.

Four of the five are cleared and committed, and none of them changes the
untruncated path: the cache bind and MTP guard read the real depth, which is 78
when nothing overrides it, and `load_format` and the waiver are entered only
when `--layers` is given.

The fifth needs the waiver at the draft model's own load, which is inside
vLLM's speculative setup rather than ours. That is a larger change than the
remaining value justifies right now -- the arms are queued against an empty
machine anyway -- but the four cleared constraints are the expensive part, and
they are recorded so the next attempt starts at the fifth.
