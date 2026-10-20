# Raising speculative acceptance beyond prompt-lookup — verbatim source material

Collected 2026-10-20 for the GLM-5.2-W4A8C8 / Ascend A3 / TP16 / batch=1 ultra-low-latency effort.
Context this material is judged against: prompt-lookup drafting saturates acceptance near **7.1 tokens/round**; TPOT = round / ~7, so a 2 ms TPOT needs a ~14 ms round; a decode round is 53.05 ms unprofiled and current best TPOT is 11.6198 ms. GLM-5.2 is an agentic coding model that copies its input heavily.

Everything below the heading is quoted verbatim from the named source. Chinese text is left in Chinese.

---

### SuffixDecoding — 7.8 mean accepted tokens/step on SWE-Bench against PLD's 3.2 (the single most relevant measurement)

https://arxiv.org/abs/2411.04975 · https://arxiv.org/html/2411.04975v3
Gabriele Oliaro, Zhihao Jia, Daniel Campos, Aurick Qiao (Snowflake AI Research; Carnegie Mellon University). Code: https://github.com/snowflakedb/ArcticInference

Abstract, verbatim:

> Speculative decoding is widely adopted to reduce latency in large language model (LLM) inference by leveraging smaller draft models capable of handling diverse user tasks. However, emerging AI applications, such as LLM-based agents, present unique workload characteristics: instead of diverse independent requests, agentic frameworks typically submit repetitive inference requests, such as multi-agent pipelines performing similar subtasks or self-refinement loops iteratively enhancing outputs. These workloads result in long and highly predictable sequences, which current speculative decoding methods do not effectively exploit. To address this gap, we introduce SuffixDecoding, a novel method that utilizes efficient suffix trees to cache long token sequences from prompts and previous outputs. By adaptively speculating more tokens when acceptance likelihood is high and fewer when it is low, SuffixDecoding effectively exploits opportunities for longer speculations while conserving computation when those opportunities are limited. Evaluations on agentic benchmarks, including SWE-Bench and Text-to-SQL, demonstrate that SuffixDecoding achieves speedups of up to 5.3×, outperforming state-of-the-art methods—2.8× faster than model-based approaches like EAGLE-2/3 and 1.9× faster than model-free approaches such as Token Recycling. SuffixDecoding is open-sourced at https://github.com/snowflakedb/ArcticInference.

The explicit indictment of prompt-lookup:

> However, existing speculative decoding approaches fall short in meeting these dual requirements. Model-based methods can use significant GPU time when speculating long sequences, and can incur memory contention and kernel-level transitions (Chen et al., 2024b; Li et al., 2024) that must be managed carefully. Conversely, existing model-free approaches, such as prompt-lookup decoding (PLD) (Saxena, 2023), achieve low overhead and rapid token generation, but typically lack adaptivity. These methods speculate a fixed number of tokens irrespective of acceptance likelihood, leading to wasted computational resources on verifying long and improbable draft sequences.

The headline numbers, verbatim:

> We compare SuffixDecoding with EAGLE-2, EAGLE-3, PLD, and Token Recycling on SWE-Bench and AgenticSQL. We also run the Spec-Bench standard dataset, which is a more traditional non-agentic workload. Fig. 4 shows the results. First, on the agentic workloads, SuffixDecoding outperforms all baselines. In AgenticSQL, SuffixDecoding obtains a mean speedup of 5.3× over vanilla decoding, a 2.8× improvement over EAGLE-2/3, and 1.9× higher than Token Recycling. In SWE-Bench, EAGLE-2/3 fail due to their maximum sequence length limitations. SuffixDecoding obtains a mean speedup of 2.5× over vanilla decoding, a 1.7× improvement over PLD, the next best baseline. SuffixDecoding's superior performance in agentic workloads can be attributed to its consistently higher mean accepted tokens per decoding step. In AgenticSQL, SuffixDecoding reaches 6.3 mean accepted tokens per step—substantially higher than EAGLE-3 (3.6 tokens) and Token Recycling (3.2 tokens). In SWE-Bench, SuffixDecoding achieves 7.8 mean accepted tokens per step, while PLD only accepts 3.2 tokens per step on average.

> The hybrid approach also performs well in AgenticSQL, achieving a 4.1× speedup in the tree variant, significantly better than the 1.9× speedup from standalone EAGLE-2/3 and the 2.7× speedup from Token Recycling. These speedups are achieved thanks to the hybrid approach's impressive 7.5 mean accepted tokens per step, more than 2× higher than EAGLE-2/3 and Token Recycling. SuffixDecoding has a slightly lower mean acceptance length of 6.3, but its much lower speculation cost and higher acceptance rate make it the winning solution in agentic tasks (5.3× average speedup compared to the 4.1× speedup of the hybrid approach).

Measurement setup (batch size 1 — the same regime as ours):

> Figure 4: Speculative speedups (top) and mean accepted tokens per step (bottom) compared to vanilla decoding for SuffixDecoding and baseline methods on three benchmarks: Spec-Bench, AgenticSQL, and SWE-Bench. Experiments use Llama-3.1-8B-Instruct on a single H100 GPU with batch size 1. Speedup is measured as the ratio of wall-clock time-per-output-token relative to vanilla decoding. Suffix (tree) and Hybrid (tree) use SuffixDecoding's tree speculation algorithm, which constructs a speculation tree from the suffix tree for parallel verification. Suffix (linear) and Hybrid (linear) use a simpler linear speculation approach that only allows sequential token chains. The hybrid variants combine SuffixDecoding with EAGLE-3, dynamically selecting between suffix-based and model-based speculation based on pattern match confidence. Note that EAGLE-2/3 and Token Recycling failed to run on several SWE-Bench tasks due to long context lengths (>8192 tokens), indicated by missing bars. Spec-Bench represents a non-agentic workload and is included for comparison.

The algorithm — two suffix trees, greedy tree expansion, the scoring function:

> For each request, we consider speculating token sequences from (1) the prompt and output of that request, and (2) the outputs of prior requests. Doing so captures the source of output token repetition from many agentic algorithms, including self-consistency, self-refinement, and multi-agent pipelines.

> SuffixDecoding leverages suffix trees to perform fast pattern matching and find possible continuations of token sequences. Suppose the prompt and output tokens of an ongoing inference is x_{1:t}. Consider a suffix x_{t-p+1:t} of length p, which we will refer to as the pattern sequence. We walk the suffix tree starting from the root node, and at each step taking the child that corresponds to token x_{t-p+i}. If no such child exists, then the pattern is not found and SuffixDecoding reverts to standard non-speculative decoding. Otherwise, after p steps, we arrive at a node whose descending paths are the possible continuations of the pattern sequence.

> In reality, we found it convenient to maintain two different suffix trees: a global tree for the previously generated outputs, and a separate per-request tree for the current ongoing inference request. This circumvents the complexities and overheads due to synchronizing the suffix tree updates from multiple concurrent requests. The global tree can be constructed offline in O(n) time, while the per-request tree can be efficiently constructed and updated online (Ukkonen, 1995).

> Although suffix trees are memory-efficient at O(n) space, the global tree can still become large when there are many previous outputs. However, they only require CPU memory, which is typically plentiful and under-utilized in LLM serving scenarios.

The scoring math (this is the expected-accepted-length calculation):

```text
C(N) = COUNT(N) / sum_{M in CHILDREN(PARENT(N))} COUNT(M)

D(N) = D(PARENT(N)) × C(N),  if N != N_p
     = 1,                    otherwise
```

> where COUNT(N) is the number of occurrences of node N in the reference corpus, which can be computed when constructing the suffix tree. Starting with the single node N_p in our speculation sub-tree, we consider all children of all of its leaf nodes, and add the node N with the highest D(N). This process is repeated until the sub-tree reaches a predetermined size limit, MAX_SPEC.

> Intuitively, C(N) estimates the probability that TOKEN(N) would be the next observed token in a subsequence TOKEN(N_p),…,TOKEN(PARENT(N)), and D(N) estimates the probability that TOKEN(N) would be ultimately accepted by the speculative tree verification, assuming the output tokens follow historical patterns. Thus, SuffixDecoding builds the speculation tree by greedily adding leaf nodes that it believes to be the most likely to be accepted during verification.

```text
SCORE(T_spec) = sum_{N in T_spec} D(N).
```

> Intuitively, if D(N) estimates the probability that node N in a speculation tree T_spec would be accepted, then SCORE(T_spec) estimates the expected number of accepted tokens. SuffixDecoding then selects the T_spec with the highest SCORE as the final speculation tree to be verified.

Adaptive speculation length — the mechanism that breaks the PLD plateau:

> Figure 2: (a) the mean number of accepted tokens increases with the length of the pattern match, which motivates MAX_SPEC = αp. (b) shows that this choice achieves a better trade-off between acceptance rate and speculative speedup.

> To guide how to adaptively set MAX_SPEC, we observed that the number of accepted tokens in practice typically increases with longer pattern sequence lengths p (Fig. 2(a)). Thus, we define MAX_SPEC adaptively as
> MAX_SPEC(p) = αp,
> where α is a user-defined max speculation factor. Fig. 2(b) shows that setting MAX_SPEC adaptively according to the pattern length results in a better trade-off between acceptance rate and speculative speedup. In practice, we found that α ∈ [1,4] works well for agentic applications.

Hybrid fallback and the threshold rule:

> Lastly, we find that SCORE(T_spec) can be used to dynamically decide between using SuffixDecoding or falling back to a model-based speculation method, which is useful for practical scenarios when the workload can be mixed between agentic and more diverse applications. Specifically, for each decoding iteration, we always speculate using SuffixDecoding first. If SCORE(T_spec) > τ, where τ is a configurable threshold, then SuffixDecoding's draft tokens are used. Otherwise, we use a fall-back speculation method, such as EAGLE-3 (Li et al., 2025a).

> As a practical guideline, we find that setting τ close to the mean accepted tokens of the fallback speculator works well for mixed workloads, while using SuffixDecoding alone (τ=0) is optimal for highly repetitive agentic tasks.

> Table 3 shows the wall-clock speedup results on Spec-Bench for various threshold values. For open-ended generation tasks, we find that setting τ close to or slightly exceeding the mean accepted tokens (MAT) of the model-based speculator yields the best performance. For instance, EAGLE-3 achieves a MAT of approximately 4.65 tokens/step on Spec-Bench. The results show that τ ∈ [5,7] produces the highest overall speedups (2.5×), with performance being relatively stable across this range.

> In summary, practitioners should set τ ≈ MAT_fallback for mixed or open-ended workloads, where MAT_fallback is the mean accepted tokens of the fallback model-based speculator. For highly repetitive agentic workloads, using SuffixDecoding alone (τ=0 or very low values) yields the best performance.

Observed tree shapes on structured output:

> Fig. 5 shows an example of a speculation tree constructed by SuffixDecoding. We observed many instances of large speculation trees that branch at each boolean true/false value of several consecutive features. These speculation tree always contains a branch with high acceptance, advancing output generation by dozens of tokens or more in one step.

Corpus-size sensitivity — acceptance rate is nearly flat, speedup is not:

> In Fig. 8, we show the speedup and acceptance rate of SuffixDecoding on WildChat and Magicoder across a range of suffix tree sizes between 256 and 10,000 output examples. First, we note a promising pattern: the speedup consistently improves as the size of the suffix tree grows. This indicates that SuffixDecoding can learn useful patterns even in workloads with lower token repetition, and may be a substitute for model-based methods when a draft model is not available.
>
> Second, perhaps surprisingly, the acceptance rate does not change much even when the suffix tree size varies across almost two orders of magnitude. We believe this is primarily due to the effect of the adaptive speculation length MAX_SPEC = αp. Although less data may mean less certainty in the speculated tokens, the pattern matches are also shorter, which results in fewer speculated tokens.

How to predict in advance whether suffix drafting will pay off:

> SuffixDecoding tends to perform better on more structured tasks compared to very open-ended ones (e.g., AgenticSQL vs WildChat). We can measure this "structuredness" using empirical entropy. The steps are as follows: (1) create a suffix tree from example model outputs (100 examples is typically enough), (2) calculate the entropy of each node's output distribution by determining how often each child node is accessed, and (3) compute a weighted average of this entropy across all nodes. A low average entropy indicates that output tokens are more predictable based on their prefixes, which generally suggests that Suffix Decoding will perform better.

The real agentic-coding deployment setup:

> We conducted the end-to-end SWE-Bench experiment on a 8xH100 80GB GPU cluster, with 1TB RAM. We served the all-hands/openhands-lm-32b-v0.1-ep3 model locally using vLLM, with a tensor parallelism degree of 4 and with prefix caching enabled. We used the flashinfer kernels for sampling. We made some minor modifications to vLLM to record the per-request and per-step statistics of interest (time-per-token latency, throughput, acceptance length, acceptance rate). We used the same settings for all baselines. We ran the OpenHands daemon on the same machine, and used the OpenAI API to interact with the vLLM server. We ran OpenHands with the CodeActAgent (Wang et al., 2024b) with ITERATIVE_EVAL_MODE=true, and a maximum of 100 iterations, as recommended by the OpenHands authors. We used a maximum of 16 concurrent workers to run the SWE-Bench tasks.

Cross-domain transfer without adaptation:

> Fig. 9 shows the results. SuffixDecoding starts from having 4,000 output examples from WildChat, and begins to receive SpiderSQL inference requests. Without any adaptation, SuffixDecoding still achieves 1.5× speedup and 8% acceptance rate, but is far from the 2.6× speedup and 20% acceptance rate it would achieve if it were trained on 500 examples from SpiderSQL instead.

## Relevance
This is the one published measurement directly comparable to our 7.1 tokens/round: **7.8 mean accepted tokens/step on SWE-Bench** for suffix-tree drafting against **3.2 for PLD**, at batch size 1 on an agentic coding workload. Two mechanisms explain the 2.4x gap, and both are available to us: matching against a *global* tree of previous outputs rather than only the current prompt, and `MAX_SPEC = αp` — speculating proportionally to the observed match length instead of a fixed K. Our 7.1 already exceeds their 7.8-benchmark baseline's PLD figure by a wide margin, which suggests our prompt-copy rate is unusually high; the suffix-tree gain over us may therefore be smaller than 2.4x, but the adaptive-length mechanism is orthogonal to the match source and should still apply.

---

### vllm-ascend speculative decoding — the Ascend hard ceiling of 16 tokens per decode round, and the `suffix` method already wired up

https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/speculative_decoding.md
(vLLM Ascend plugin, `main` as of 2026-10-20)

The supported-method table, verbatim:

```text
| Method | Description |
| ------ | ----------- |
| `ngram` | Match n-grams from the prompt |
| `suffix` | Suffix-based pattern matching (requires Arctic Inference) |
| `medusa` | Medusa heads embedded in the target model |
| `eagle` | EAGLE-based draft model |
| `eagle3` | EAGLE-3 based draft model |
| `mtp` | Multi-Token Prediction with shared embedding head |
| `dflash` | Block diffusion-based parallel draft model |
| `dspark` | Semi-autoregressive block drafting with a sequential Markov logit-bias head |
| `draft_model` | Generic external draft LLM |
| `extract_hidden_states` | Extract hidden states for EAGLE training |
```

**The hard ceiling**, verbatim:

```text
!!! note

    On Ascend NPUs, the `npu_fused_infer_attention_score` operator supports a maximum of 16 tokens per decode round. Therefore, `(num_speculative_tokens + 1)` must be ≤ 16.
```

The architecture:

```text
vLLM Ascend implements speculative decoding through a **proposer-verifier** architecture:

1. **Proposer** (`vllm_ascend/spec_decode/`): Generates draft (speculative) tokens using various methods — from simple n-gram matching to neural-network-based draft models.
2. **Rejection Sampler** (`vllm_ascend/sample/`): Verifies draft tokens against the target model's output, accepting matches and rejecting mismatches, with optional optimizations including Block Verify and Entropy Verify.
```

```text
- **`num_speculative_tokens`** (int, required): Number of speculative tokens to generate per forward pass. Auto-filled from the draft model's `n_predict` config (e.g., MTP) or `suffix_decoding_max_tree_depth` (suffix method) when available.
- **`draft_tensor_parallel_size`** (int, optional): Tensor parallelism size for the draft model. Can only be `1` or the same as the target model's tensor parallel size.
- **`disable_padded_drafter_batch`** (bool, default: `False`): Disable input padding for speculative decoding. If set to `True`, speculative input batches can contain sequences of different lengths, which may only be supported by certain attention backends. **Note:** Only effective with `eagle`, `eagle3`, `mtp`, `dflash`, `draft_model`, and `extract_hidden_states` methods.
```

n-gram / prompt-lookup configuration:

```python
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 4,
        },
    )
```

Suffix Decoding section, verbatim:

```text
## Speculating using Suffix Decoding

The following code configures vLLM to use speculative decoding where proposals are generated using Suffix Decoding [(SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications)](https://arxiv.org/abs/2411.04975).

Like n-gram, Suffix Decoding can generate draft tokens by pattern-matching using the last `n` generated tokens. Unlike n-gram, Suffix Decoding (1) can pattern-match against both the prompt and previous generations, (2) uses frequency counts to propose the most likely continuations, and (3) speculates an adaptive number of tokens for each request at each iteration to get better acceptance rates.

Suffix Decoding can achieve better performance for tasks with high repetition, such as code-editing, agentic loops (e.g. self-reflection, self-consistency), and RL rollouts.

!!! note

    Suffix Decoding requires Arctic Inference. You can install it with `pip install arctic-inference`.
```

```python
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        enforce_eager=True,
        speculative_config={
            "method": "suffix",
            "num_speculative_tokens": 15,
        },
    )
```

```shell
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
      --tensor-parallel-size 1 \
      --enforce-eager \
      --speculative-config '{"method": "suffix", "num_speculative_tokens": 15}'
```

MTP, including the TP16 reference configuration:

```text
## Speculating using MTP

MTP (Multi-Token Prediction) boosts inference performance by parallelizing the prediction of multiple tokens, shifting from single-token to multi-token generation. This approach significantly increases generation throughput and achieves multiplicative acceleration in inference speed — all without compromising output quality.
```

```python
    llm = LLM(
        model="deepseek-ai/DeepSeek-V3.2-Exp-W8A8",
        tensor_parallel_size=16,
        enable_expert_parallel=True,
        max_model_len=36768,
        max_num_seqs=10,
        quantization="ascend",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 2,
            "disable_padded_drafter_batch": False,
        },
    )
```

Dynamic speculation — adapting K at runtime:

```text
## Dynamic Speculative Decoding

Dynamic Speculative Decoding adapts the number of draft tokens (K) at runtime, instead of always using a fixed `num_speculative_tokens`. This helps keep speculative decoding beneficial as concurrency and draft confidence change. vLLM Ascend currently provides two approaches:

- **Confidence-based verify length (DSpark / DFlash)**: adjust the per-request verify length from draft confidence. DSpark uses a dedicated confidence head; DFlash is **head-free** and uses `max(softmax(logits))` of the drafted token as a confidence proxy.
- **Batch-size based (Autoregressive)**: select a shared K from concurrency ranges via `num_speculative_tokens_per_batch_size`.
```

```text
When `dynamic_spec_config.method` is set to `"dspark"` or `"dflash"`, the corresponding proposer:

1. Runs the draft model as usual to produce up to `num_speculative_tokens` draft tokens.
2. Estimates per-position acceptance likelihood for each request:
   - **DSpark**: sigmoid output of the DSpark confidence head.
   - **DFlash (head-free)**: `max(softmax(logits))` of the argmax draft token (no extra neural head).
3. Periodically recomputes a shared per-request verify budget from those confidence scores.
4. Allocates that budget across requests (each request keeps at least `min_verify_tokens`), so each request verifies only a prefix of draft tokens.
```

```text
!!! note

    This is an **exploratory** feature and currently targets **model runner v1** only. Supported dynamic methods are `"dspark"` (confidence head) and `"dflash"` (head-free). Among DSpark draft models, only the **Qwen** series is supported today; support for models such as GLM and DeepSeek is being added incrementally.
```

Acceptance-criterion relaxations — Block Verify and Entropy Verify:

```text
vLLM Ascend provides two optional optimizations for the rejection sampler in speculative decoding: **Block Verify** and **Entropy Verify**. These features trade a small amount of output precision for improved inference throughput.

!!! warning

    Both Block Verify and Entropy Verify modify the token acceptance criteria and may cause minor precision degradation (e.g., slightly different output tokens compared to the standard rejection sampler). Evaluate the quality impact on your specific workload before enabling them in production.

### Block Verify

Block Verify evaluates all draft tokens as a block using cumulative probability products, rather than checking each token independently. This can improve the acceptance rate and reduce the overhead of rejection sampling, especially when `num_speculative_tokens >= 3`.

### Entropy Verify

Entropy Verify adjusts the acceptance threshold based on the entropy of the target distribution:

- **High entropy** (uncertain distribution) → lower effective threshold → more tokens accepted
- **Low entropy** (confident distribution) → higher effective threshold → stricter rejection

This entropy-aware threshold is controlled by two parameters:

- **`posterior_threshold`** (default: `0.95`, range: `(0, 1]`): The upper bound of the modified threshold. Even when entropy is very low, the effective threshold will not exceed this value.
- **`posterior_alpha`** (default: `0.4`, range: `>= 0`): Controls how strongly entropy influences the threshold. A higher alpha makes the threshold more sensitive to entropy changes, resulting in a higher acceptance rate for speculative tokens but also greater precision loss. You need to tune this value based on your specific model and dataset. When alpha is `0`, entropy has no effect and the threshold equals `posterior_threshold`.
```

```shell
    vllm serve <model> --additional-config \
        '{"rejection_sampler_config": {"enable_block_verify": true, \
        "enable_entropy_verify": true, "posterior_threshold": 0.95, \
        "posterior_alpha": 0.4}}'
```

Synthetic rejection sampling:

```text
In addition to the default `standard` rejection sampling (which accepts a draft token when `target_prob / draft_prob >= uniform`), vLLM Ascend supports **synthetic** rejection sampling. In this mode each draft token at position `i` is accepted with a configured probability `rates[i]`, compared against a uniform random draw — **independent of the target/draft probability match**. The first rejected position stops the request and emits a recovered token; if every draft token is accepted, the bonus token is appended.
```

## Relevance
The hard constraint that governs everything: **`num_speculative_tokens + 1 ≤ 16`** because `npu_fused_infer_attention_score` takes at most 16 tokens per decode round. Our measured 7.1 accepted tokens/round sits below but not far below that ceiling; pushing acceptance past ~15/round is not possible with this attention operator at all, so the achievable TPOT floor from speculation alone is round/15. `method: "suffix"` with `num_speculative_tokens: 15` is already a supported configuration on Ascend — the maximum the operator allows — and `Block Verify` plus `Entropy Verify` are two documented ways to raise the accepted count at a stated precision cost. Note `synthetic` rejection sampling exists purely as an acceptance-rate instrument, useful for isolating how much of a round-time win comes from acceptance versus from the kernel.

---

### SAM-Decoding — suffix automaton instead of suffix tree, O(1) per step

https://arxiv.org/abs/2411.10666 · https://arxiv.org/html/2411.10666v3 · code: https://github.com/hyx1999/SAM-Decoding
Hu et al.

As published:

> The method is retrieval-based and uses a suffix automaton to find the longest suffix match, rather than fixed n-gram matching. It achieves an average time complexity of O(1) per generation step.
> It builds a static automaton over a text corpus and a dynamic one over the current prompt and generated text. The suffix automaton matches the suffixes of the generating text and retrieves the draft from the text corpus and the generating text respectively according to the matching position.
> Assuming the matching length of the longest suffix implying the quality of retrieval draft tokens, the method can be integrated with other speculative decoding methods, deciding whether to adopt auxiliary decoding techniques.
> Drafts are then verified by the LLM, and accepted tokens are used to extend the automata.
> With Token Recycling, it reports a speedup of 2.27× over autoregressive decoding on Spec-Bench. With EAGLE-2, it reaches a speedup of 2.49×, surpassing all current approaches.
> It is 18%+ faster than other retrieval-based SD methods.

## Relevance
Same two-tree (static corpus + dynamic current-text) structure as SuffixDecoding, but with an O(1)-amortised suffix automaton rather than a suffix tree, and the *same* use of match length as a confidence signal — "assuming the matching length of the longest suffix implying the quality of retrieval draft tokens". That confirms match length as the right adaptive-K signal independently of SuffixDecoding. No mean-accepted-length figures are published, only speedups on Spec-Bench, which is a non-agentic workload — so it does not give a number comparable to our 7.1.

---

### FLUX / decode-phase speculation interaction — the serial cost of draft + verify

https://arxiv.org/abs/2411.04975 (SuffixDecoding, § motivation)

> To efficiently handle the long repetitions common in agent-driven applications, speculative decoding methods must satisfy two critical requirements. First, they need to generate draft tokens rapidly and with minimal overhead, enabling maximal exploitation of long speculation lengths. Second, they must do so adaptively—only generating more draft tokens when acceptance likelihood is high and fewer tokens when acceptance likelihood is low, to prevent verification from becoming a bottleneck.

> Model-based methods can use significant GPU time when speculating long sequences, and can incur memory contention and kernel-level transitions (Chen et al., 2024b; Li et al., 2024) that must be managed carefully.

From the Arctic Inference documentation (https://arcticinference.readthedocs.io/en/stable/suffix-decoding.html):

> a specialized, model-free speculative decoding method... enabling rapid draft token predictions using CPU resources without GPU overhead

> There is a global suffix tree holding outputs from completed requests and a per-request tree built from the current prompt and generated tokens.

## Relevance
The decisive structural property for us: suffix/n-gram drafting is **model-free and runs on CPU**, so the draft stage adds no device work and no extra kernel launches to the round. Every model-based drafter (EAGLE/EAGLE-3/MTP/Medusa) adds a device-side forward per round, which at our ~3008 kernels and 0.70–13.58 us launch floors is a direct addition to round time. That is the quantitative reason to raise acceptance by improving the match source and the adaptive length rather than by adding a neural drafter — and it is why vllm-ascend's own GLM-5.2 reference config sets `"enforce_eager": true` on the MTP drafter (see FILE 1) rather than capturing it in the graph.

---

### Snowflake's production write-up — SuffixDecoding in Arctic Inference and vLLM

https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/
(Snowflake engineering blog)

> SuffixDecoding at Production Scale with Arctic Inference and vLLM

As described in the paper that this blog accompanies, and as surfaced in the Arctic Inference docs:

> Suffix Decoding can achieve better performance for tasks with high repetition, such as code-editing, agentic loops (e.g. self-reflection, self-consistency), and RL rollouts.

## Relevance
Code-editing is named as a first-class target. Our workload — an agentic coding model that copies its input heavily — is exactly the case the method was designed for, which is also why our prompt-lookup acceptance is already as high as 7.1.

---

## Gaps

Looked for and could not find:

- **Any published acceptance-length number above ~8 tokens/round.** The maximum in the literature reviewed is SuffixDecoding's 7.8 mean accepted tokens/step on SWE-Bench (batch=1, Llama-3.1-8B, H100), with a hybrid Suffix+EAGLE-3 variant at 7.5 on AgenticSQL. Nothing reports sustained acceptance in the 10–15 range, and on Ascend `num_speculative_tokens + 1 ≤ 16` caps it regardless. So our measured 7.1 is already at the published state of the art for this workload class, and the remaining TPOT must come from round time, not from acceptance.
- **GLM-5.2's own MTP head documentation.** vllm-ascend's reference GLM-5.2 config uses `"method": "deepseek_mtp"` with `num_speculative_tokens: 3` (see FILE 1), i.e. it borrows the DeepSeek MTP path rather than a GLM-specific head; no GLM-5.2 MTP architecture document or per-head acceptance measurement was found. The `dspark` dynamic method's own note says GLM support "is being added incrementally".
- **Per-benchmark acceptance tables from SuffixDecoding's appendix.** The appendix section headings exist in the HTML ("SWE-Bench: Mean accepted tokens (tokens/step)", "SWE-Bench: Mean Acceptance Rate", "AgenticSQL: …", "Spec-Bench: …") but the figure data itself is in plotted images, not text, so the per-sub-task breakdowns could not be quoted.
- **Measured acceptance for `dflash` / `dspark` / Block Verify / Entropy Verify.** vllm-ascend documents the mechanisms and the parameters but publishes no acceptance-length or throughput deltas for any of them.
- **Tree-attention mask construction details for SpecInfer / Sequoia / HASS, and dynamic-draft-tree expected-accepted-length derivations.** Searched; the retrievable material was abstract-level speedup claims rather than the mask construction or the E[accepted] algebra. SuffixDecoding's `SCORE(T_spec) = Σ D(N)` is the one expected-accepted-length formula captured verbatim here.
- **REST, Token Recycling and Lookahead-decoding primary sources.** Token Recycling appears only as a baseline inside SuffixDecoding's numbers (3.2 mean accepted tokens/step on AgenticSQL); no primary measurement was retrieved for REST or Lookahead decoding.
- **Cursor / Continue "fast apply" speculative-edit acceptance numbers.** No published per-round acceptance figures were found for commercial fast-apply implementations.
