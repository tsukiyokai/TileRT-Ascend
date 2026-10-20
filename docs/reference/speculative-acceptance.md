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

### Single-chain n-gram / prompt-lookup match policies — the exact code and defaults in Saxena's original, HF transformers, vLLM, TRT-LLM and llama.cpp

Sources, all retrieved 2026-10-20 and pinned (a `git clone --filter=blob:none` of each repo; line numbers below refer to that commit):
- https://github.com/apoorvumang/prompt-lookup-decoding @ cada4fe6ec7368d41c386ec3fb94fa11268971eb (Apoorv Saxena, "Prompt Lookup Decoding", November 2023; last commit 2024-08-23)
- https://github.com/huggingface/transformers @ 9f5750c67ad4c2e077dd3883d20503f3d77b2fe5 (2026-10-20)
- https://github.com/vllm-project/vllm @ fce64bc5c2c939a4521bfb9b462c8c74c1c13893 (2026-10-20)
- https://github.com/NVIDIA/TensorRT-LLM @ 083447cd22b228a36cf30d21385c624be905cddd (2026-10-20)
- https://github.com/ggml-org/llama.cpp @ 84663953535d4323e92b4bc90971aa022c305d06 (2026-10-19)

#### 1. Saxena's original (the algorithm every other implementation descends from)

> **TLDR**: We modify speculative decoding where we replace the draft model with simple string matching in the prompt to generate candidate token sequences. This results in significant speedups (2x-4x) in input-grounded tasks, with no effect on output quality. This method can be used with any decoder model without model changes or external datastore, and with both greedy and sampling techniques.

> **Intuition**: In several LLM use cases where you're doing _input grounded generation_ (summarization, document QA, multi-turn chat, code editing), there is high n-gram overlap between LLM input (prompt) and LLM output. This could be entity names, phrases, or code chunks that the LLM directly copies from the input while generating the output. Prompt lookup exploits this pattern to speed up autoregressive decoding in LLMs.

`README.md` lines 28-56:

````python
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
    input_length = input_ids.size(1)

    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = input_ids[0, -ngram_size:].tolist()

        # Create sliding windows of size ngram_size
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

        # Convert ngram to a tensor for comparison
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=2)

        # Get the indices of matches
        match_indices = matches.nonzero(as_tuple=True)[1]

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # Ensure we don't go beyond the length of input_ids and avoid self-match
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]

    # If no match is found, return an empty tensor
    return torch.tensor([], dtype=torch.long, device=input_ids.device)
````

> Input to this function is the same as to the draft model - all the tokens till the current generation step (`input_ids`). It then tries to match last few tokens to somewhere earlier in the prompt. If found, it returns the next-k token continuation as `candidate_input_ids` or candidate sequence. The 2 parameters are `max_ngram_size`, which is the maximum ngram to use when looking for matches in the prompt. `num_pred_tokens` is the candidate sequence length to return after match is found.

> - **GPU**: Single A100 40GB
> - **Model**: Mistral-7B-Instruct-v0.1
> - **Decoding Type**: Greedy decoding
> - **Hyperparams**: Matching max n-gram size = 3, Continuation length = 10

The only coding datum in the README, and the author's own list of open policy questions:

> - roleplay has the worst gain. This is probably because there isn't many ngrams to copy, since each generation is sort of unique.
> - coding has very high gain in 2nd turn, because there is lots of code copying
> - in first turn, extraction has highest gain. This agrees with our hypothesis - in extraction there is definitely n-gram copying, and PLD should help

> - There's probably better ways to do string matching than the current one, and there are several obvious things to improve eg. what to do when there are multiple matches? Whats the ideal length of continuation?
> - We haven't yet tried sampling, although there's no reason it shouldn't work.
>     - Here, one additional thing to test would be whether prompt lookup while sampling can affect hallucination rates, since this artifically increases probability of sampling exact sequences from input (this was suggest by my colleague Shwetha S)
> - Testing actual FLOPs impact and tradeoffs is needed
> - Also need to figure out best hyperparams - 3 and 10 were chosen on very little testing

#### 2. HF transformers `PromptLookupCandidateGenerator` (the reference many engines copy)

`src/transformers/generation/candidate_generator.py` lines 1017-1060:

````python
class PromptLookupCandidateGenerator(CandidateGenerator):
    """
    `CandidateGenerator` class to be used for prompt lookup generation. This class generates candidates by looking up
    likely continuations in the provided prompt (input_ids) itself.
    Read the following blog post for more information: https://github.com/apoorvumang/prompt-lookup-decoding

    Args:
        eos_token_id (`torch.Tensor`, *optional*):
            The token id of the end of sequence token.
        num_output_tokens (`int`, *optional*, defaults to 10):
            The number of tokens to be output as candidate tokens.
        max_matching_ngram_size (`int`, *optional*, defaults to 2):
            The maximum ngram size to be considered for matching in the prompt
        max_length (`int`, *optional*, defaults to 20):
            The number of total maximum tokens that can be generated. For decoder-only models that includes the
            prompt length. Defaults to 20, which is the max length used as default in generation config.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step. In
            prompt lookup assisted generation, they are not used to manipulate probabilities, but rather to find
            forbidden tokens (p = -inf) and block them from being valid candidates.
        vocab_size (`int`, *optional*):
            The size of the vocabulary. Required if `logits_processor` is provided.
    """

    def __init__(
        self,
        eos_token_id: torch.Tensor | None = None,
        num_output_tokens: int = 10,
        max_matching_ngram_size: int = 2,
        max_length: int = 20,
        logits_processor: Optional["LogitsProcessorList"] = None,
        vocab_size: int | None = None,
    ):
        self.num_output_tokens = num_output_tokens
        self.max_matching_ngram_size = max_matching_ngram_size
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.logits_processor = logits_processor
        self.vocab_size = vocab_size

        if self.max_matching_ngram_size <= 0 or self.num_output_tokens <= 0:
            raise ValueError("Invalid max_matching_ngram_size or num_output_tokens")

````

`src/transformers/generation/candidate_generator.py` lines 1076-1112:

````python
            return input_ids, None

        chosen_ids = None
        match_found = False
        for ngram_size in range(min(self.max_matching_ngram_size, input_length - 1), 0, -1):
            # Create sliding windows of size ngram_size
            windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

            # Convert ngram to a tensor for comparison
            ngram_tensor = input_ids[0, -ngram_size:]

            # Find where the windows match the ngram
            matches = (windows == ngram_tensor).all(dim=2)

            # Get the indices of matches
            match_indices = matches.nonzero(as_tuple=True)[1]

            # Iterate through match indices to find a valid continuation
            # TODO (joao): this finds the first valid candidates (left to right), but perhaps we should find the
            # longest valid candidates?
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + self.num_output_tokens
                end_idx = min(end_idx, input_length, self.max_length)

                if start_idx < end_idx:
                    chosen_ids = input_ids[0, start_idx:end_idx]

                    # Check if the each new candidate token is forbidden according to the logits processor. If all
                    # tokens are allowed, we keep `chosen_ids` as is.
                    # 1. create random logits.
                    # 2. apply the logits processor to get output logits for the next token, using the arbitrary
                    #    logits as input.
                    # 3. compare the output logits with the next candidate token. If they are -inf, then the next
                    #    candidate token is forbidden and we don't want to generate it.
                    if self.logits_processor is not None:
                        sequence_with_candidate = input_ids
````

`src/transformers/generation/candidate_generator.py` lines 1130-1154:

````python

                    match_found = True

                    # remove remaining candidate ids if an "eos" token is found, otherwise the target model may
                    # accept eos and the rest as valid, thus not stopping generation after "eos"
                    # NOTE: below code is written based on the fact that assisted decoding supports only bs=1
                    if self.eos_token_id is not None:
                        mask = torch.isin(chosen_ids, self.eos_token_id)
                        match_indices_eos = torch.nonzero(mask)
                        if match_indices_eos.numel() > 0:
                            first_eos_index = match_indices_eos[0].item()
                            chosen_ids = chosen_ids[:first_eos_index]
                    break
            if match_found:
                break

        # In case we didn't find a match return the input sequence unchanged, reverts back to autoregressive decoding
        if not match_found or chosen_ids is None or len(chosen_ids) == 0:
            return input_ids, None

        # Now need extend input_ids with chosen_ids
        chosen_ids = chosen_ids.unsqueeze(0)
        candidate_input_ids = torch.cat((input_ids, chosen_ids), dim=1)
        # assisted_generation expects logits as well, but we don't have those here, so returning None
        return candidate_input_ids, None
````

Config-level documentation of the same two knobs:

`src/transformers/generation/configuration_utils.py` lines 339-342:

````text
        prompt_lookup_num_tokens (`int`, *optional*):
            The number of tokens to be output as candidate tokens.
        max_matching_ngram_size (`int`, *optional*):
            The maximum ngram size to be considered for matching in the prompt. Default to 2 if not provided.
````

#### 3. vLLM `ngram` — config defaults and the match routine

`vllm/config/speculative.py` lines 476-481:

````python
    prompt_lookup_max: int | None = Field(default=None, ge=1)
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: int | None = Field(default=None, ge=1)
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""
````

`vllm/config/speculative.py` lines 1235-1263:

````python
        if self.method in ("ngram", "[ngram]"):
            self.method = "ngram"

        if self.method in ("ngram", "ngram_gpu"):
            # Set default values if not provided
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                if self.prompt_lookup_max is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                if self.prompt_lookup_min is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}"
````

`vllm/v1/spec_decode/ngram_proposer.py` lines 12-36:

````python
class NgramProposer:
    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # Minimum length of the n-gram to match.
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        # Maximum length of the n-gram to match.
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # Number of tokens follow the match. If there are less than k
        # tokens follow the match, we will return the maximum amount of
        # tokens until the end.
        self.k = vllm_config.speculative_config.num_speculative_tokens
        # Maximum length of the model.
        self.max_model_len = vllm_config.model_config.max_model_len

        # Pre-allocate buffers for numba batch propose.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros((max_num_seqs), dtype=np.int32)

        # Threshold of total number of tokens in the batch to enable
        # multi-threading in numba batch propose.
        self.num_tokens_threshold = 8192
````

`vllm/v1/spec_decode/ngram_proposer.py` lines 209-295:

````python
@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """Find the longest n-gram which matches the suffix of the given tokens
    whose length is within [min_ngram, max_ngram] (inclusive).

    If found, we will extract k right after the matched ngram.
    """
    # Do not generate draft tokens is context is shorter than minimum n-gram
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Do not generate draft tokens beyond the max model length.
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip tokens, and the goal become to find longest ngram
    # on the rightmost position which matches the prefix with
    # length [min_n, max_n] (inclusive).
    tokens = origin_tokens[::-1]

    # Longest prefix (not including itself) which is a suffix of
    # the current position.
    #   lps[i] = max{v, where tokens[0:v] == tokens[i+1-v:i+1]}
    #
    # As ngram is capped by max_ngram to save memory, we only need to
    # store lps for the first max_ngram prefix.
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    # lps[0] always equal to 0, we start with index 1
    prev_lps = 0
    i = 1
    while i < total_token:
        # tokens[:prev_lps] is the longest prefix as a suffix of tokens[:i]
        if tokens[prev_lps] == tokens[i]:
            # Token match: tokens[:prev_lps+1] is the longest prefix as
            # a suffix of tokens[:i+1]
            prev_lps += 1
            # Check if we found a longer valid ngram.
            #
            # Update position when longest_ngram matched prev_lps,
            # as we want to get the target n-gram of the earliest position
            # in the original tokens (i.e.
            # latest position in the reversed tokens)
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                # Store LPS for the first max_ngram prefix
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                # When prev_lps reached max_ngram, update prev_lps
                # to lps[max_ngram-1] to avoid matching ngram
                # longer than max_ngram
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            # Token mismatch: try the second-longest prefix
            # among all suffix of tokens[:i],
            # which is the longest prefix of tokens[:prev_lps]
            prev_lps = lps[prev_lps - 1]
        else:
            # Token mismatch, and no more prefix (except empty string)
            # as a suffix of tokens[:i]
            i += 1

    if longest_ngram < min_ngram:
        # No valid ngram is found
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip the position back, so in origin_tokens,
    # origin_tokens[total_token-1-position:total_token-1-position+longest_ngram]
    # is the matched ngram, so we should start drafting tokens from
    # total_token-1-position+longest_ngram
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]
````

vLLM's own documentation example (`docs/features/speculative_decoding/n_gram.md`):

`docs/features/speculative_decoding/n_gram.md` lines 7-21:

````python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)
outputs = llm.generate(prompts, sampling_params)
````

#### 4. TensorRT-LLM `NGram` — the pool keeps several matches per pattern, then emits exactly one

`tensorrt_llm/_torch/speculative/ngram.py` lines 14-49:

````python
class NGramPoolManager(BaseResourceManager):
    """
    Drafter for NGram. This class maintains the pattern-matches pairs for NGram drafter.

    For example, one of the existed pairs could be: ["I","love"] -> [["apple", "because", "it", "is"], ["banana", "and"]].

    Here we call ["I","love"] as `pattern`, and [["apple", "because", "it", "is"], ["banana", "and"]] as `matches`.

    `pattern` is a list of token ids. The pool emits corresponding draft tokens from the matches if the pattern appears at the tail of the generated sentence.

    `matches` is a list of candidate draft token ids attaching to a pattern.

    Arguments:
        max_total_draft_tokens: int
            The length maximum of draft tokens (can be understood as length maximum of output draft tokens).

        max_matching_ngram_size: int
            The length maximum of searching tokens (can be understood as length maximum of input tokens to search).

        is_keep_all: bool = True
            Whether to keep all candidate pattern-matches pairs, only one match is kept for each pattern if False.

        is_use_oldest: bool = True
            Whether to provide the oldest match when pattern is hit, the newest one is provided if False.

        is_public_pool: bool = True
            Whether to use a common pool for all requests, or the pool is private for each request if False.

    Members:
        pool: dict[tuple[int], OrderedSet[int]] or dict[int, dict[tuple[int], OrderedSet[int]]]
            If is_public_pool == True, it maps from patterns to matches
            If is_public_pool == False, it maps from request ID to the request-specific pool

        start_index: dict[int, int]
            It maps from request ID to the index of the prompt to update the pool in the next step.
    """
````

`tensorrt_llm/_torch/speculative/ngram.py` lines 84-141:

````python
    def get_draft_tokens(
        self,
        prefix: list[int],
        request_id: int,
        max_sequence_length: int,
    ):
        prefix_len = len(prefix)
        max_draft_token_length_this_step = max_sequence_length - 1 - prefix_len
        if max_draft_token_length_this_step <= 0:  # No draft token is need if the prefix is long enough
            return []

        if request_id not in self.start_index:  # Extend start_index and pool for a new request
            self.start_index[request_id] = 0
            if not self.is_public_pool:
                self.pool[request_id] = {}

        pool = (self.pool if self.is_public_pool else self.pool[request_id])

        # Update pool
        sequence = prefix[self.start_index[request_id]:]
        for size in range(min(self.max_matching_ngram_size, prefix_len - 1), 0,
                          -1):
            # Find each possible pattern-match combination, and use tuple for hash
            for l in range(len(sequence) - size):
                r = min(l + size + self.max_total_draft_tokens, len(sequence))
                pattern = tuple(sequence[l:l + size])
                new_match = tuple(sequence[l + size:r])
                if pattern not in pool or \
                    (not self.is_keep_all and len(new_match) > len(pool[pattern][0])):
                    # Replace the match if
                    # 1. the pattern does not exist in the pool
                    # 2. only one match is kept, and the new match is longer (MRU)
                    pool[pattern] = OrderedSet((new_match, ))
                elif new_match not in pool[pattern]:
                    # Update the matches if the pattern is already existed:
                    # TODO: need a strategy to maintain the short candidates, now we just remove them
                    # Drop all existed matches with small length
                    for match in pool[pattern]:
                        if len(match) < len(new_match):
                            pool[pattern].remove(match)
                    pool[pattern].add(new_match)

        draft_tokens = []
        for size in range(min(self.max_matching_ngram_size, prefix_len - 1), 0,
                          -1):
            pattern = tuple(prefix[-size:])
            if pattern not in pool:
                continue
            draft_tokens = pool[pattern][0 if self.is_use_oldest else -1]
            draft_tokens = list(draft_tokens)[:max_draft_token_length_this_step]
            break

        # Update start_index
        self.start_index[request_id] = max(
            0, prefix_len -
            (self.max_total_draft_tokens + self.max_matching_ngram_size - 1))

        return draft_tokens
````

`tensorrt_llm/_torch/speculative/ngram.py` lines 161-181:

````python
class NGramDrafter(Drafter):

    _needs_padding_kv_extension = True

    def __init__(
        self,
        spec_config: NGramDecodingConfig,
        ngram_pool_manager: NGramPoolManager = None,
    ):
        super().__init__(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            max_concurrency=spec_config.max_concurrency,
            draft_len_schedule=spec_config.draft_len_schedule)
        assert ngram_pool_manager is not None, "NGram needs a resource manager to maintain the pool."
        self.spec_resource_manager = ngram_pool_manager
        self.spec_config = spec_config
        self.max_draft_len = spec_config.max_draft_len
        self.max_total_draft_tokens = spec_config.tokens_per_gen_step - 1
        assert self.max_draft_len == self.max_total_draft_tokens, "NGram only supports linear tree."

````

`tensorrt_llm/llmapi/llm_args.py` lines 3021-3048:

````python
class NGramDecodingConfig(DecodingBaseConfig):
    """Configuration for NGram drafter speculative decoding."""
    decoding_type: Literal["NGram"] = Field(default="NGram")
    max_matching_ngram_size: PositiveInt = Field(
        default=2,
        description=
        "The length maximum of searching tokens (can be understood as length maximum of input tokens "
        "to search).")
    is_keep_all: bool = Field(
        default=True,
        description=
        "Whether to keep all candidate pattern-matches pairs, only one "
        "match is kept for each pattern if False.")
    is_use_oldest: bool = Field(
        default=True,
        description="Whether to provide the oldest match when pattern is hit, "
        "the newest one is provided if False.")
    is_public_pool: bool = Field(
        default=True,
        description="Whether to use a common pool for all requests, or the pool "
        "is private for each request if False.")

    @model_validator(mode="after")
    def validate_ngram_config(self):
        if self.max_draft_len is None or self.max_draft_len <= 0:
            raise ValueError("max_draft_len must be > 0 for NGram")
        self.max_total_draft_tokens = self.max_draft_len  # Current NGram only supports linear tree
        return self
````

`docs/source/features/speculative-decoding.md` lines 83-100:

````markdown
### NGram

The NGram method is an implementation of [this Prompt Lookup Decoding algorithm](https://github.com/apoorvumang/prompt-lookup-decoding).

When the NGram algorithm is used, TRT-LLM will maintain a map from token prefixes to candidate draft sequences. For example, the 3-gram ["The ", " future ", " is"] could map to the draft sequence [" bright", " because"]. The prefixes are token sequences that are extracted from the prompt and the tokens generated by the target model. The NGram pool and matching procedure can be tuned with the following options:

* `max_draft_len`: Maximum draft candidate length.
* `max_matching_ngram_size`: Maximum prompt suffix length to match with keys in the pool.
* `is_public_pool`: If true, a single ngram pool is shared for all requests. Otherwise, each request has its own ngram pool.
* `is_keep_all`: If true, draft candidates will be retained in the pool forever. Otherwise, only the largest draft candidate is retained.
* `is_use_oldest`: If true, the oldest draft candidate is always proposed for a given match. Otherwise, the newest draft candidate is used. Only applicable if `is_keep_all == True` because `is_keep_all == False` means we'll only ever have a single value for each key.

```python
from tensorrt_llm.llmapi import NGramDecodingConfig

speculative_config = NGramDecodingConfig(
    max_draft_len=3, max_matching_ngram_size=4, is_public_pool=True)

````

#### 5. llama.cpp `ngram-simple` / `ngram-map-k` / `ngram-map-k4v` / `ngram-mod`

`docs/speculative.md` lines 126-203:

````markdown
### n-gram Map (`ngram-simple`, `ngram-map-*`)

These implementations search the token history for patterns and use matching sequences as draft candidates.
They require no additional model but rely on patterns that have already appeared in the generated text.
An example to use this approach can be the rewriting of source code by a LLM.

#### n-gram Map (`ngram-simple`)

This implementation looks for the last n-gram in history that matches the current n-gram and creates a draft using the m tokens following the matched n-gram. It is the simplest self-speculative approach with minimal overhead.

```
llama-server [...] --spec-type ngram-simple --spec-draft-n-max 64
```

#### n-gram Map Key (`ngram-map-k`)

This implementation looks for the current n-gram of size n (called the _key_) in the token history. If the key n-gram is followed by the same m tokens (called the _mgram_) multiple times, it creates a draft using these m tokens. This approach requires a minimum number of occurrences (argument `--spec-ngram-map-k-min-hits`, default is 1) before generating drafts.

The number of accepted tokens is stored for each used n-gram.

**Example:**
```
llama-server [...] --spec-type ngram-map-k --spec-draft-n-max 64
```

#### n-gram Map Key-4-Values (`ngram-map-k4v`)

This experimental implementation looks for the current n-gram of size n (called the _key_) in the token history. For each key, up to four _values_ (n-grams of size m, called _mgrams_) are tracked. An internal statistic counts the occurrences of each mgram after the key n-gram. If one mgram is significantly more frequent than the others, it is used as the draft.

The number of accepted tokens is stored for each used n-gram.

**Example:** Server options to be used if there are a lot of longer repetitions.
```
llama-server [...] --spec-type ngram-map-k4v --spec-ngram-map-k4v-size-n 8 --spec-ngram-map-k4v-size-m 8 --spec-ngram-map-k4v-min-hits 2 --spec-draft-n-max 64
```

### n-gram Mod (`ngram-mod`)

Add basic ngram hasher for speculative decoding:

- For each ngram, compute a hash using LCG
- For each computed hash, store the next token
- During speculation, iteratively compute the rolling hash of the last n tokens and pick the next token from the storage

Some characteristics:

- Lightweight (~16 MB)
- Constant memory and complexity
- Can generate variable draft lengths (i.e. m is not fixed)

Currently, a single hash pool is shared across all server slots, so different requests can benefit from each other.

**Sample usage:**

```
# notes:
# - small `n` are not recommended
# - MoEs require long drafts
# - dense models: can reduce `--spec-ngram-mod-n-min` and `--spec-ngram-mod-n-max`

llama-server ... --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
```

Applications:

- Iterating over a block of text/code (e.g. in llama.vim)
- Reasoning models (when they have to repeat their thinking in the final answer)
- Summarization

Example Video:

- See #19164

### Differences between ngram-simple, ngram-map and ngram-mod

- ngram-simple looks for a previous matching n-gram and inserts the following m-gram.
- ngram-map-k looks for a previous matching n-gram and inserts the following m-gram but uses an internal hash-map of n-grams in the current context window.
- ngram-mod uses a hash pool which is shared across all server slots. The hash pool is a map from n-gram hash to the next token (not the next m-gram as in ngram-map).
````

`docs/speculative.md` lines 317-358:

````text
### n-gram Mod Parameters

```
--spec-ngram-mod-n-match                N
                                        ngram-mod lookup length (default: 24)
--spec-ngram-mod-n-min                  N
                                        minimum number of ngram tokens to use for ngram-based speculative decoding (default: 48)
--spec-ngram-mod-n-max                  N
                                        maximum number of ngram tokens to use for ngram-based speculative decoding (default: 64)
```

### n-gram Simple Parameters

```
--spec-ngram-simple-size-n              N
                                        ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-simple-size-m              N
                                        ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-simple-min-hits            N
                                        minimum hits for ngram-simple speculative decoding (default: 1)
```

### n-gram Map Key Parameters

```
--spec-ngram-map-k-size-n               N
                                        ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-map-k-size-m               N
                                        ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-map-k-min-hits             N
                                        minimum hits for ngram-map-k speculative decoding (default: 1)
```

### n-gram Map Key-4-Values Parameters

```
--spec-ngram-map-k4v-size-n             N
                                        ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-map-k4v-size-m             N
                                        ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-map-k4v-min-hits           N
                                        minimum hits for ngram-map-k4v speculative decoding (default: 1)
````

`docs/speculative.md` lines 424-445:

````text
Each speculative decoding implementation prints statistics.

```
draft acceptance rate = 0.57576 (  171 accepted /   297 generated)
statistics ngram_simple: #calls = 15, #gen drafts = 5, #acc drafts = 5, #gen tokens = 187, #acc tokens = 73
statistics draft: #calls = 10, #gen drafts = 10, #acc drafts = 10, #gen tokens = 110, #acc tokens = 98
```

```
draft acceptance rate = 0.70312 (   90 accepted /   128 generated)
statistics ngram_mod: #calls = 810, #gen drafts = 15, #acc drafts = 15, #gen tokens = 960, #acc tokens = 730, dur(b,g,a) = 0.149, 0.347, 0.005 ms
```

```
statistics ngram_map_k: #calls(b,g,a) = 6 1690 26, #gen drafts = 26, #acc drafts = 26, #gen tokens = 1248, #acc tokens = 968, dur(b,g,a) = 2.234, 1.427, 0.016 ms
```


- `#calls(b,g,a)`: number of calls of begin (new prompt), generation and accumulation of this implementations
- `#gen drafts`: number of drafts generated by this implementation
- `#acc drafts`: number of drafts accepted (partially) by the main model
- `#gen tokens`: number of tokens generated by this implementation (including rejected tokens)
````

## Relevance
Four independent codebases agree on the single-chain policy: try the longest suffix first (Saxena `max_ngram_size=3`, HF `max_matching_ngram_size=2`, vLLM `prompt_lookup_min/max` which both default to 5 when neither is given although the field docstring still says "Defaults to 1", TRT-LLM `max_matching_ngram_size=2`), and then take exactly ONE occurrence. Which occurrence differs: HF takes the first hit scanning left to right (its own TODO asks "perhaps we should find the longest valid candidates?"), vLLM takes the longest matched n-gram and, among equal lengths, "the earliest position in the original tokens", TRT-LLM stores every distinct continuation for a pattern in an `OrderedSet` and then emits `pool[pattern][0 if self.is_use_oldest else -1]` (`is_use_oldest=True` by default), llama.cpp `ngram-simple` uses "the last n-gram in history that matches". Nobody in this list verifies two occurrences at once: TRT-LLM states it outright in code (`assert self.max_draft_len == self.max_total_draft_tokens, "NGram only supports linear tree."`) and in `NGramDecodingConfig` ("Current NGram only supports linear tree"). That is the unexploited headroom for our shape: when the same suffix occurs k times earlier in the context with different continuations, a linear drafter pays the whole-round loss every time its single guess is the wrong occurrence, while the other k-1 continuations were already sitting in the pool. The draft-length defaults span a wide range (Saxena 10, HF 10, vLLM docs 5, TRT-LLM docs 3, llama.cpp `ngram-simple` m=48 and `ngram-mod` n_max=64) and llama.cpp's own sample comment says "MoEs require long drafts", which is consistent with our MoE target needing wide drafts to amortise an expensive verify.

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
