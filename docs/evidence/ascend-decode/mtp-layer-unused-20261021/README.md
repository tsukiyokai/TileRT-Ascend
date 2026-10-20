# The checkpoint carries a trained draft head that has never been loaded

`config.json` says `num_hidden_layers: 78` and
`num_nextn_predict_layers: 1`, with `index_share_for_mtp_iteration: True`.
The weight index carries layer indices **0 to 78 inclusive, 79 of them**. So
layer 78 is the multi-token-prediction head, and it is in the checkpoint.

The engine does not load it. `cann_model.py:152` passes
`speculative_config=None`, and line 263 refuses a drafter outright:

    if self.runner.drafter is not None:
        raise ValueError("Target must not load a draft model")

Every acceptance figure this project has measured therefore comes from a
host-side suffix automaton over the token stream -- a drafter that can only
repeat what it has already seen.

## Why this is the largest lever on the board

The automaton's ceiling is structural: it proposes what followed a matching
suffix, so it is excellent inside a copied stretch and blind inside a rewrite.
That is exactly the shape of the acceptance curve measured -- 10.020 a round
at an edit rate of 0.02 falling to 1.824 at 0.20. A trained MTP head predicts
rather than recalls, so it does not have that blindness.

The cost is one extra layer a round, 1.3 percent of 78.

    round 35.1 ms after the boundary merges
    acceptance 9.288 with the adaptive tree    TPOT 4.9 ms
    acceptance 15 if MTP delivers what such heads usually do   TPOT ~2.4 ms

Which is the target, from a component already paid for and shipped in the
weights.

## What has to be checked

That the refusal at line 263 is a guard against loading a *separate* draft
model rather than against the checkpoint's own head -- the message says "draft
model", and vLLM's MTP path is a different mechanism from an external drafter.
And that the engine's speculative config can address layer 78 without
disturbing the 0 to 77 graph capture the round depends on.
