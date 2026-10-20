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

## Upstream支持已确认，架构映射在列

`SpeculativeConfig.hf_config_override`:

    if hf_config.model_type in ("deepseek_v3", "deepseek_v32", "glm_moe_dsa"):
        hf_config.model_type = "deepseek_mtp"
    if hf_config.model_type == "deepseek_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": [
            "DeepseekV32MTPModel" if use_v32_mtp else "DeepSeekMTPModel"]})

with `use_v32_mtp = hf_config.model_type in ("deepseek_v32", "glm_moe_dsa")`.

So this checkpoint's `glm_moe_dsa` becomes `deepseek_mtp` and loads
`DeepseekV32MTPModel`, with `n_predict` taken from the config's
`num_nextn_predict_layers` of 1. The path is complete upstream.

A first reading of the same function found a `glm5_next` branch and concluded
`glm_moe_dsa` had no architecture mapping. That was reading one branch of a
chain of them -- the mapping is two `if`s earlier.

## The head's structure, and what a draft token actually costs

Layer 78 holds 3113 tensors:

    eh_proj.weight            the embedding-and-hidden projection
    enorm.weight hnorm.weight a norm for each of the two inputs
    embed_tokens.weight       its own embedding
    input_layernorm.weight
    mlp.experts.0..255        a complete 256-expert MoE, four tensors each
    mlp.gate.e_score_correction_bias
    (plus the attention tensors)

That is a full transformer layer with an `eh_proj` in front -- exactly what
`DeepseekV32MTPModel` expects, so loading should succeed without adaptation.

It also fixes the cost. The head is not a lightweight projection: running it
costs what a layer costs, about 710 us today or roughly 450 after the boundary
merges. Drafting N tokens means running it N times autoregressively, so

    --mtp-tokens=1   about 1.3 percent of the round
    --mtp-tokens=3   about 4 percent

Against a 35.1 ms round, three draft tokens need acceptance to rise by more
than 6 percent to pay for themselves. A trained head against a suffix
automaton that is blind inside rewrites should clear that by a wide margin, but
the comparison in the queue measures it rather than assuming it.

## A model of the pairing, and why it does not answer the question

Prefixing the automaton's proposal with one token from a source right with
probability p, at an edit rate of 0.05:

    automaton alone          accepted 4.902  rows 6.5  TPOT  8.24
    p = 0.60                 accepted 1.597  rows 4.1  TPOT 24.33
    p = 0.80                 accepted 1.800  rows 4.4  TPOT 21.76
    p = 0.95                 accepted 1.948  rows 4.7  TPOT 20.23

Much worse, and the construction is why: prefixing shifts the automaton's whole
proposal by one position, so a correct head token gains one accepted token
while a wrong one voids the entire proposal behind it. That is not how MTP
drafts. The head shares the target's context and drafts autoregressively --
predict one token, extend the context with it, predict the next -- so there is
no second drafter whose output it can invalidate.

The probe therefore measures a wrong splice, not the pairing, and its numbers
are not evidence about MTP. Recorded so they are not mistaken for it later.

What does stand: with `n_predict 1` the head produces one token a pass, so
`--mtp-tokens=N` costs N full layers where the automaton produces 32 tokens for
nothing. The two have different economics per drafted token, and only a
measured acceptance settles which wins. That is what the queued arms do.
