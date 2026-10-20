# Megakernel / persistent-kernel LLM decode — source material

Verbatim excerpts. Each item carries URL, author, date. Nothing here is paraphrased except
the short `## Relevance` lines, which tie the item to our own measured numbers
(3008 kernels/round, MIX_AIC launch floor 13.58 us, MIX_AIV 12.04 us, AI_CORE 2.30 us,
AI_VECTOR_CORE 0.70 us, 11.5 ms launch floor of which 82% on MIX, current TPOT 11.6198 ms).

---

## 1. Hazy Research — "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B"

URL: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
Authors: Benjamin Spector\*, Jordan Juravsky\*, Stuart Sul, Owen Dugan, Dylan Lim, Dan Fu, Simran Arora, Chris Ré (Stanford Hazy Research)
Date: May 27, 2025 · 13 min read

This is the canonical megakernel post. Reproduced essentially in full because every paragraph
is load-bearing for our design.

### 1.1 The problem statement and the regime

> There are some applications that benefit from running LLMs really, really fast. This low-latency regime encompasses applications like chatbots and human-in-the-loop workflows, where users care a lot about seeing responses come back immediately.
>
> Given the importance of these low-latency workloads, we wanted to explore just how fast we can run open-source models on modern GPUs. To really stress-test existing systems, we consider an aggressive low-latency scenario where we generate a single sequence with Llama-3.2-1B. This workload is strongly memory bound – our performance is dominated by how fast we can load model weights from GPU global memory.
>
> It turns out that popular LLM inference engines – vLLM and SGLang – are only able to use at most 50% of available GPU bandwidth when running this workload on an H100. The root of the problem, which we'll describe more below, is that existing systems break down a model forward pass into around a hundred separate kernels that each implement a few operations (e.g. RMS norm, attention, an MLP layer + activation, rotary). Each kernel comes with a setup and teardown period and during this time no useful work gets done – for instance, the all-important task of loading model weights is stalled.

> Figure 1: Speed! Results generated with a 32-token prompt and 128 generated tokens, with no speculation

> In this post, we show how we can bypass this problem by merging the entire Llama-1B forward pass into a single "megakernel" that eliminates kernel boundaries altogether. Doing this achieves brr – on an H100, we use 78% of memory bandwidth and outperform existing systems by over 1.5x. (To our knowledge, this is the lowest-latency forward pass for Llama-1B in bfloat16!) In the rest of this post, we'll walk through how and why one would do this. Specifically:
>
> First, we'll talk about how small kernels lead to AI systems that underutilize the GPU's full bandwidth.
>
> Second, we'll describe three important points about how we built our megakernel: how we fused lots of kernels together, how we share hardware resources across them to minimize overhead, and how we synchronize them efficiently.
>
> If you're interested in learning more of the details or using these ideas yourself, we're open-sourcing all of our code here.

### 1.2 "Separate Kernels Kill the Vibe" — the three stall sources

> In general, the way one runs code on a GPU is by launching a "kernel" – a small program that does a well-defined operation (e.g. RMS norm, MLP). Today, all AI workloads run as long sequences of relatively small kernels. To get an initial sense, let's look at the operations in the Llama-1B transformer block, and some example kernel boundaries of how they might be divided up (Figure 2).
>
> Figure 2: An example set of kernel boundaries for the Llama-1B transformer block. Red boxes delineate the work done by individual kernels.
>
> As we described earlier, decoding a single sequence with Llama-1B is a purely memory-bound workload: our performance depends on being able to always be loading weights from GPU global memory. So, why are existing approaches so far from using the full bandwidth of the GPU?
>
> When we dug into it, we noticed a key problem was that the current kernel-based approach to running models introduces stalls that prevent us from constantly loading memory:
>
> First: GPU kernels are launched with a strict ordering, so that a thread block in one kernel can't start until all thread blocks in previous kernels have completely finished. Consequently, every time we start a kernel, we have to wait for all the straggler thread blocks from the prior one to finish. For example, if a kernel runs 512 thread blocks (like our Llama-1B down projection), but we only have 148 streaming multiprocessors (like on a B200), we end up with 80 empty SM's at the end.
>
> Second, as we've previously highlighted, each kernel launch and teardown incurs costs. In principle, NVIDIA's CUDA graphs can help hide costs, but by our measurements they still leave a lot on the table. For a simple dummy kernel (which dumps a start time, sleeps, and dumps an end time) on an H100, we find that running on a CUDA stream incurs a launch cost of about 2.1 microseconds, and with CUDA graphs the launch cost only decreases to around 1.3 microseconds – time spent with the GPU doing no useful work! We'd like to have the GPU spend all of its time doing useful work.
>
> Finally, even after we start the next kernel, we still have to wait to load weights and activations before any compute can start. These latencies leave the GPU sitting idle for thousands of cycles! Ideally, we'd start loading the next weights while the previous computations and stores are happening. NVIDIA has also built a mechanism for this called Programmatic Dependent Launch (PDL), which allows the next kernel to start preparing while the previous kernel is running, but we found it still introduces unnecessary stalls because the PDL synchronization mechanism (cudaGridDependencySynchronize) is very coarse. For example, it means we have to wait for all queries, keys, and values to complete in order to start attention, as opposed to starting heads as soon as they are ready. We'll later show another specific case of where this is useful in Llama-1B.
>
> Taken together, these form the "memory pipeline bubbles" our title references – and they represent a key reason that we're not always loading from memory.  For short operations, these pauses add up, wasting a huge chunk of potential bandwidth. In part, this is because Llama-1B (actually 1.24B parameters) in batch size 1 is just so... small: if each operation is really fast, then the time spent in-between them really starts to matter.

### 1.3 The arithmetic that motivates the whole thing

> To illustrate the magnitude of the problem: for single-sequence generation in 16-bit precision on a single H100, the memory limit is 3.35TB/s / 2.48GB = ~1350 forward passes per second. But with 7 kernel launches per layer, and 16 layers, even with an optimistic 5 us of stalling per kernel (counting stragglers, kernel launch, and memory latencies), generation would run at just ~770 forward passes per second. In practice, it's often worse. On low-latency workloads, GPUs spend only a fraction of their time actually doing any useful work!
>
> So while CUDA does provide some existing features (e.g. graphs, streams, PDL) to partially solve these problems, we wanted to see if a different approach could solve all of these problems, where we just fuse the entire model forward pass into a single kernel.

### 1.4 Issue 1/3 — the on-GPU interpreter and the instruction set

> Next, we'll show you how we fused a whole Llama forward pass into a single kernel, and our methods for resolving three key problems:
>
> Fusing dozens of operations is hard to do from scratch. We need a mechanism for executing these operations within the megakernel.
>
> In order to overlap multiple operations on the same hardware, we need to prevent contention over limited resources, such as shared memory.
>
> The GPU synchronizes after each kernel in the traditional kernel model. Without kernels, we have to synchronize the GPU all by ourselves!

> Traditional kernel fusion generally merges just two or three operations together. In contrast, we need to fuse about a hundred. Consequently, we need to have a sensible abstraction for how we can actually program a megakernel.
>
> Our approach is built on an on-GPU interpreter – essentially a more sophisticated version of our infrastructure underlying ThunderMLA. Our interpreter is designed such that each streaming multiprocessor (SM) within the GPU receives a sequence of instructions (each implemented using the same CUDA template) and executes them. We schedule each SM's instruction sequence ahead of time on the Python side, and notably we can reuse each schedule for hundreds of forward passes!
>
> For our end-to-end Llama forwards pass megakernel, we define the following set of instructions:
>
> A fused RMS norm & QKV & RoPE instruction.
>
> An attention computation instruction.
>
> An attention reduction instruction (for ThunderGQA on long sequences).
>
> An O-projection + residual instruction.
>
> A fused RMS norm & up-gate & SiLU instruction.
>
> A down-projection + residual instruction.
>
> An RMS norm & language modeling head instruction, for computing the final token logits.
>
> We implement each of these instructions using a common CUDA template (with load, store, compute boilerplate functions), facilitating interoperability within our interpreter framework.

### 1.5 Issue 2/3 — paged shared memory (the exact page geometry)

> The instruction-and-interpreter structure lets us cleanly organize our megakernel. However, we haven't yet addressed the key issue: making sure that model weights are always being loaded in order to maximize memory bandwidth utilization.
>
> The reason why a megakernel lets us solve this problem is that we can pipeline memory loads across instructions: our interpreter will start loading the model weights for an instruction as soon as it can, even if a previous instruction is still finishing up (e.g. storing out its results to global memory). It's this tight transitioning between instructions that minimizes the memory bubbles that would otherwise appear if we launched multiple kernels.
>
> However, there's a catch: loading the weights from global memory for the next instruction doesn't do you much good if you have no place to put the data you loaded! More precisely, all of our weight matrices are loaded from GPU global memory into our SM's "shared memory" – NVIDIA's term for the fast memory on each SM. Shared memory is a scarce resource on each SM, and we can't start a load for a new instruction if a previous instruction is using all of it. This necessitates a way to keep track of which instruction is using which piece of shared memory and quickly transition shared memory to the next instruction when the current instruction is done with it.
>
> We accomplish this by paging shared memory. We first divide the first 213kB of shared memory on an H100 into 13 16KiB pages, and use remaining shared memory for special purposes, like storing instruction parameters. To use one of these pages, instructions have to explicitly request and release them from the interpreter. The interpreter automatically passes released pages to the next instruction, allowing them to start issuing memory loads as early as shared memory becomes available.

### 1.6 Issue 3/3 — the counter-based synchronization system

> While megakernels let us minimize pipeline bubbles, they also introduce a new problem: synchronization. The performance limitation with the normal many-kernel execution model is that no thread blocks in a kernel can start until all thread blocks in previous kernels are finished. However, it's precisely this property that makes it easy to manage data dependencies. When a kernel launches, CUDA guarantees that all of the kernel's input tensors have already been produced and are safe to read from immediately.
>
> With megakernels, we have no such guarantees: when an SM starts to execute a new instruction, its inputs might not be ready! To address this, we explicitly synchronize the instructions inside of our megakernel. We accomplish this with a simple counter system. Before the megakernel launches, we initialize an array of counters (i.e. integers) in GPU global memory with a starting value of zero. Whenever an instruction completes, it increments one of these counters. Similarly, whenever a new instruction starts, it must wait for some of these counters to reach a target value, indicating that all of its dependencies have finished.
>
> One optimization this enables is in the big multi-layer perceptrons (MLPs) in Llama-1B.
>
> In a naive implementation using PDL, one must await completing the whole hidden state before beginning the down projection matrix multiply.
>
> We instead produce and consume the intermediate state in four chunks, each with their own counter. This way, an instruction for the down projection only needs to wait for its input chunk to finish.

### 1.7 Results and the full runtime breakdown

> To our knowledge, our H100 megakernel represents the first time anyone has run the forward pass for a 16-bit 1B+ parameter language model in under one millisecond on a GPU. Our B200 implementation pushes this even further to under 680 microseconds per forward pass!
>
> As shown in Figure 1, our megakernel outperforms vLLM and SGLang baselines (which use CUDA graphs and torch compilation):
>
> On an H100, our megakernel runs almost 2.5x faster than vLLM and over 1.5x faster than SGLang.
>
> On a B200, the gap with vLLM rises to over 3.5x, and we remain more than 1.5x faster than SGLang, too.
>
> We're still actually quite a ways off from the theoretical limit on a B200, which is around ~3,000 forward passes per second. Part of this gap is because this theoretical limit is based purely on memory bandwidth – but we still have to wait to load activations. And although these activations are small (and don't cost a lot of bandwidth), there are still latencies in loading them that we can't hide. A breakdown of the runtime of our current B200 forward pass (total runtime 600 microseconds):
>
> 250 microseconds are spent storing activations, awaiting consistency, and loading them. This is about 20% higher than a simple model would suggest: since each instruction has a dependence on the last one, we need to pay two load latencies (check ready, and then load activations) and two store latencies (store activations, then mark ready) per instruction. Using ~500 nanoseconds latency per load / store, this would impose about 200 microseconds of overhead. (We suspect some of the remaining 50 microseconds comes from time spent processing atomics in global memory.)
>
> 200 microseconds are spent actually running RMS norm and matrix-vector computations. 95% of this portion is devoted to matrix-vector. On Blackwell, we find that using the tensor cores is marginally helpful for this; on Hopper, we find it better to simply run on the CUDA cores. This difference comes from the fact that both GPUs have relatively similar CUDA core performance, but Blackwell tensor cores are much faster.
>
> 30 microseconds are spent awaiting weights from global memory (pipelining works!) Of these, 40% are spent in the LM head, which is the best-pipelined part of the whole megakernel due to its homogeneity and huge size.
>
> 40 microseconds are spent on low-level synchronization overhead across warps. A key issue here is that CUDA's asynchronous barriers are relatively slow, even when they're already in the "pass" state, requiring about 60 nanoseconds each time.
>
> 80 microseconds are on setup and various other overheads (e.g. passing instruction barriers, marking pages as complete, etc.)
>
> We think there's probably more to do on each of these, but that'll have to wait for a future update!

### 1.8 Repo build/run commands

URL: https://raw.githubusercontent.com/HazyResearch/Megakernels/main/README.md
Author: HazyResearch
Date: fetched 2026-10-20

````markdown
# Megakernels!

## Installation

Clone this repo and run:

```bash
git submodule update --init --recursive
pip install uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
```

## Low-Latency Llama Demo

First, to compile the megakernel, run:

```bash

# from the repo root
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.12 # adjust if yours is different
export GPU=H100 # options are {H100, B200}, else defaults to B200
cd demos/low-latency-llama
make

```

To start an interactive chat session with the model, run:

```bash

# from the repo root
python megakernels/scripts/llama_repl.py

```

To benchmark the megakernel, run:

```bash

# from the repo root
python megakernels/scripts/generate.py mode=mk prompt="tell me a funny joke about cookies" ntok=100

```
````

## Relevance

The structural argument transfers directly: their 7 launches/layer x 16 layers x 5 us optimistic
stall = ~770 fwd/s against a 1350 fwd/s bandwidth limit is exactly the shape of our 3008
kernels/round x MIX floors = 11.5 ms launch floor against a much smaller byte-movement time
(e.g. GroupedMatmulSwigluQuantV2 6.793 ms spent vs 2.272 ms of bytes). Their fix list maps 1:1 onto
what we must build: an ahead-of-time-scheduled instruction sequence per core, an explicit paged
on-chip buffer allocator (our UB/L1 instead of their 213 kB shared memory in 13 x 16 KiB pages),
and global-memory counters replacing the implicit kernel barrier. Their 250 us of 600 us spent on
"store activations, await consistency, load them" at ~500 ns per load/store is the warning that a
megakernel does not make inter-stage activation handoff free — it converts launch cost into
sync-latency cost, and we must budget for that on Ascend too.

---

## 2. Mirage Persistent Kernel (MPK) — "A Compiler and Runtime for Mega-Kernelizing Tensor Programs"

URL: https://arxiv.org/abs/2512.22219 · HTML: https://arxiv.org/html/2512.22219v1
Authors: Xinhao Cheng, Zhihao Zhang, Yu Zhou, Jianan Ji, Jinchen Jiang, Zepeng Zhao, Ziruo Xiao, Zihao Ye, Yingyi Huang, Ruihang Lai, Hongyi Jin, Bohan Hou, Mengdi Wu, Yixin Dong, Anthony Yip, Songting Wang, Wenqin Yang, Xupeng Miao, Tianqi Chen, Zhihao Jia
Date: arXiv:2512.22219v1 [cs.DC] 22 Dec 2025 (v2 revised 10 Jun 2026)
License: CC BY 4.0
Code: https://github.com/mirage-project/mirage

This is the most complete published description of a megakernel *runtime* (work queues,
schedulers, event graph, paged shared memory, hybrid launch). It is the blueprint closest to
what we need to write.

### 2.1 Abstract

> We introduce Mirage Persistent Kernel (MPK), the first compiler and runtime system that automatically transforms multi-GPU model inference into a single high-performance mega-kernel. MPK introduces an SM-level graph representation that captures data dependencies at the granularity of individual streaming multiprocessors (SMs), enabling cross-operator software pipelining, fine-grained kernel overlap, and other previously infeasible GPU optimizations. The MPK compiler lowers tensor programs into highly optimized SM-level task graphs and generates optimized CUDA implementations for all tasks, while the MPK in-kernel parallel runtime executes these tasks within a single mega-kernel using decentralized scheduling across SMs. Together, these components provide end-to-end kernel fusion with minimal developer effort, while preserving the flexibility of existing programming models.
>
> Our evaluation shows that MPK significantly outperforms existing kernel-per-operator LLM serving systems by reducing end-to-end inference latency by up to 1.7×, pushing LLM inference performance close to hardware limits. MPK is publicly available at https://github.com/mirage-project/mirage.

### 2.2 Why kernel-per-operator loses (three limitations)

> First, modern GPUs implicitly insert a kernel barrier between consecutive launches on the same stream to ensure that all threads from the previous kernel complete before any thread from the next kernel begins. While this mechanism enforces data dependencies, it prevents cross-operator software pipelining, forcing dependent operators to execute strictly sequentially. NVIDIA recently introduced programmatic dependent launch (PDL) [11], which allows partial overlap between kernels on the same stream. However, adopting PDL requires significant engineering effort, as it fundamentally alters kernel structure and control flow.
>
> Second, the kernel-per-operator approach prevents fine-grained compute-communication overlap. Because dependencies are captured only at the coarse operator granularity, the runtime must enforce full-operator completion before launching dependent communication or computation. For example, when a matrix multiplication is followed by an all-reduce in separate kernels, the all-reduce must wait for the entire multiplication to complete, even though each piece of all-reduce depends only on a subset of the multiplication. Exploiting such opportunities requires representing and enforcing data dependencies at a granularity finer than individual kernels.
>
> Finally, the kernel-per-operator approach requires launching hundreds to thousands of kernels for each inference iteration. To mitigate launch overhead, current systems rely heavily on CUDA Graphs, which capture a sequence of GPU operations and launch them with minimal overhead. However, CUDA Graphs are largely static: any changes to control flow, tensor shapes, or data dependencies require re-instantiation or modification of the captured graph, limiting flexibility for dynamic workloads commonly seen in model inference.

> A promising approach to overcoming these limitations is to fuse all computation and communication into a single mega-kernel (also known as a persistent kernel). In this design, the system launches one GPU kernel to execute the entire model, from layer computations to inter-GPU communication, without interruption.
>
> Mega-kernels address the limitations of the kernel-per-operator approach in several ways. First, they eliminate kernel launch overhead by avoiding repeated invocations. Second, by fusing all operator computations into a single kernel, they enable cross-operator software pipelining, allowing data for the next operator to be prefetched while computations for the current operator are still in progress. Third, they support fine-grained overlap of computation and inter-GPU communication, enabling simultaneous execution that more effectively hides communication latency.

### 2.3 ttGraph — the SM-level task/event graph

> This section introduces ttGraph, a representation that expresses the computation of a tensor program at the granularity of individual streaming multiprocessors (SMs). This fine-grained representation exposes additional parallelism and enables optimizations such as cross-operator software pipelining and fine-grained kernel overlap, both of which cannot be supported in existing kernel-per-operator execution model.
>
> Figure 4 illustrates an example ttGraph, where each node represents either a task or an event. Each task—shown as a blue (or orange) rectangle—denotes a unit of computation or communication executed on a single SM. Each event—shown as a green circle—represents synchronization across tasks. Tasks and events alternate in the graph: every task only has outgoing edges to triggering events and incoming edges from dependent events. A task is ready for execution when its dependent events are all activated and notifies its triggering event upon completion. An event is activated once it has received notifications from all tasks associated with it.
>
> This structure captures dependencies at a much finer granularity than traditional computation graphs. For example, multi-GPU LLM serving often involves a MatMul operator followed by an AllReduce operator (Figure 4a). Existing systems generally execute these operators sequentially due to coarse-grained kernel barriers that synchronize entire kernels. In contrast, SM-level task graphs can represent precise task-level dependencies: since AllReduce performs element-wise communication, each of its tasks depends only on one corresponding MatMul task. By inserting fine-grained events between dependent task pairs, MPK can overlap compute-intensive MatMul tasks with communication-intensive AllReduce tasks, maximizing overall GPU utilization.

> **Comparison with CUDA Graphs.**
>
> ttGraphs can be viewed as a lower-level extension of CUDA Graphs, sharing several structural similarities. Like CUDA Graphs, ttGraphs are statically instantiated and encode explicit dependencies among operations. However, while CUDA Graphs capture dependencies only at the kernel level, ttGraphs operate at the granularity of individual SM tasks and sub-kernel events. CUDA Graphs primarily describe kernel launch order and rely on stream semantics for synchronization, which confines overlap and fusion to kernel boundaries. In contrast, ttGraphs explicitly model both intra- and cross-operator dependencies, enabling fine-grained synchronization across SMs and overlapping of computation and communication within a single kernel. This design allows MPK to exploit parallelism that is inaccessible to CUDA Graphs or kernel-level execution models.

### 2.4 Compiler: operator decomposition, dependency analysis, event fusion

> **Operator decomposition.**
>
> The MPK compiler decomposes each operator of the input computation graph into a set of tasks by partitioning the operator's output tensors such that all tasks compute disjoint subsets of the output and can therefore execute in parallel across SMs. Most tensor algebra operators can be partitioned across multiple output dimensions; for example, the output tensor of a matrix multiplication can be tiled along both the row and column dimensions to expose parallelization opportunities.
>
> The performance of a partitioning strategy depends on both the problem shape and the target GPU architecture. To discover an effective strategy, MPK selects a partitioning strategy that minimizes data loading from device memory to shared memory, since accessing device memory is significantly more expensive than accessing shared memory or performing computation on CUDA cores or tensor cores. By default, MPK generates a number of tasks proportional to the number of SMs to promote load balance across SMs during execution. MPK also provides an interface for users to easily specify custom partitioning strategy by setting the desired parallelization degree along each output dimension.

> **Dependency analysis.**
>
> MPK uses events to capture dependencies between tasks. For any two operators that share a tensor, MPK enumerates all pairs of tasks from the two operators and introduces an event e for a task pair (t1,t2) if and only if the output region produced by task t1 overlaps with the input region consumed by task t2. The event serves as a synchronization point indicating that t2 cannot begin execution until t1 has completed producing the required data. Accordingly, MPK inserts two edges (t1,e) and (e,t2) into the resulting ttGraph. This fine-grained dependency analysis preserves all producer-consumer dependencies while exposing maximal parallelism across independent tasks.

> **Event fusion.**
>
> MPK applies two complementary forms of event fusion—successor-set and predecessor-set fusion—to eliminate redundant synchronization points and simplify the constructed ttGraph. For an event e, we define two functions: InTasks(e), the set of tasks that trigger e, and OutTasks(e), the set of tasks that depend on e. These functions allow us to characterize when multiple events exhibit identical dependency structure and can therefore be fused.
>
> First, successor-set fusion merges events that serve as prerequisites for the same set of consumer tasks. Because these tasks cannot begin execution until all such events are activated, representing them separately provides no additional scheduling flexibility.
>
> Definition 4.1 (Successor-set fusion).
> For any two events e1 and e2 of a ttGraph, successor-set fusion applies if and only if OutTasks(e1)=OutTasks(e2). MPK removes events e1 and e2 from T and introduces a fused event e′ with InTasks(e′)=InTasks(e1) ∪ InTasks(e2) and OutTasks(e′)=OutTasks(e1).
>
> As an example, successor-set event fusion fuses events e10 and e14 in Figure 5(b) into a new event (i.e., e4 in Figure 5(c)) since they are both prerequisites for task O1.
>
> Second, predecessor-set fusion merges events that depend on an identical set of producer tasks. Because such events are triggered simultaneously, maintaining them as separate synchronization nodes introduces unnecessary graph complexity.

Figure 5 caption, verbatim:

> Figure 5: The MPK compiler workflow. In (b), Q, K, V, A, O, and R denote the set of tasks produced by decomposing the query projection, key projection, value projection, attention, output projection, and RMSNorm operators, respectively. D1 and D2 in (e) are dummy tasks inserted during ttGraph normalization to guarantee that each task has exactly one triggering event. Finally, (f) shows how MPK linearizes the ttGraph and stores the resulting structure in GPU memory, where both tasks and events follow a uniform, canonical representation.

### 2.5 In-kernel parallel runtime — workers, schedulers, queues

> MPK employs an in-kernel parallel runtime that executes the ttGraph across all SMs within a single mega-kernel. This design eliminates kernel-launch overheads and enables fine-grained control over scheduling, synchronization, and execution order. Once launched, the mega-kernel continuously manages both computation and communication until the inference workload completes.
>
> To support this execution model, MPK partitions a GPU's SMs into workers and schedulers. Each worker runs on one physical SM and maintains an independent task queue. Workers execute a lightweight loop that repeatedly dequeues tasks, performs the associated computation or communication, and signals task completion by notifying the task's triggering event. This design ensures that workers are fully utilized while enabling asynchronous execution across operators.
>
> Schedulers are organized at warp granularity, with each SM hosting four scheduler warps. Each scheduler maintains an event queue and repeatedly polls for newly activated events, dispatching the corresponding tasks to workers. The allocation of workers and schedulers is fixed at kernel-launch time and matches the GPU's physical SM count, avoiding any dynamic role-switching overhead inside the kernel.

> **5.1 Event-Driven Execution**
>
> MPK executes a ttGraph using an event-driven model. Each ttGraph begins with a designated start event (e.g., e0 in Figure 7) that has no prerequisites. This event is initially enqueued into a scheduler's event queue. Upon dequeuing the event, the scheduler (e.g., s1) launches all tasks that depend on it (e.g., AT1,…,AT4). Each launched task is dispatched to a worker, which executes the task and, upon completion, notifies the triggering event associated with that task.
>
> An event becomes activated once all of its prerequisites have completed and thus have collectively triggered the event the required number of times. When an event is activated, it is enqueued into a scheduler's event queue, allowing the runtime to continue propagating execution through the ttGraph. In this way, events serve as the mechanism for driving task execution, enabling fine-grained, asynchronous execution.

### 2.6 Hybrid task launch — JIT vs AOT, and the exact sync-count argument

> A task can be enqueued into a worker's task queue either just-in-time (JIT) or ahead-of-time (AOT). In the JIT mode, a scheduler assigns a task to a worker only after its dependent event has been fully activated; the task can therefore begin execution immediately upon assignment. In the AOT mode, the runtime pre-enqueues the task to a worker before its predecessor event is activated. The worker cannot execute the task until the dependent event is fully activated, and thus waits locally for the dependency to be satisfied.
>
> JIT and AOT approaches involve complementary advantages. On the one hand, JIT task launch enables MPK to adapt to workload imbalance. For example, attention operators in modern LLMs have highly variable execution times due to data-dependent sequence lengths—requests with long sequence lengths take substantially longer to finish than those with short ones. This variance makes static assignment ineffective. Under JIT, MPK launches downstream tasks (e.g., MatMul or AllReduce) only after the attention tasks that trigger them have completed. Workers that finish their attention tasks earlier can execute more downstream tasks, improving end-to-end latency and balancing load across SMs, as illustrated in Figure 7.
>
> On the other hand, JIT task launch involves higher latency due to additional worker–scheduler communication. Figure 8 illustrates the difference for launching task T2 after event e triggered by task T1. Under JIT (Figure 8a), the worker executing T1 must notify a scheduler, which then dequeues event e, launches T2, and enqueues it to a worker's task queue. The receiving worker must then dequeue T2 before execution. This chain requires two synchronizations (worker→scheduler and scheduler→worker). By contrast, under AOT (Figure 8b), T2 is already enqueued to a pre-assigned worker, which only needs to wait for the event activation. Thus, AOT requires only a single synchronization (worker→worker via the event trigger), reducing per-task launch latency.
>
> MPK uses hybrid task launch to combine the advantages of both approaches. During ttGraph construction, the compiler classifies each operator as JIT or AOT based on whether its execution time is data-dependent and likely to induce runtime imbalance. Operators with data-dependent durations (e.g., attention) are marked as JIT, and all downstream tasks remain JIT until encountering a global barrier (i.e., an event that must be triggered by all upstream tasks). Barriers eliminate imbalance, making subsequent operators suitable for AOT. All remaining operators are labeled AOT to minimize dispatch overhead. Labels apply at operator granularity: all tasks generated by the same operator share the same mode.
>
> Workers maintain two queues—one for JIT and one for AOT tasks. Workers always prioritize JIT tasks, as they are ready to execute immediately. When a worker exhausts its JIT queue, it checks whether the first AOT task's dependent event has been fully activated; if so, the worker executes that AOT task. This design ensures that workers block only when no ready work exists.
>
> Schedulers handle only JIT-related events. All AOT tasks are pre-enqueued before execution begins. Because operators typically produce a number of tasks proportional to the number of workers (Section 4), MPK distributes AOT tasks in a round-robin fashion across workers to maintain balanced load. Pre-launching AOT tasks reduces scheduler load and amortizes dispatch overhead, while JIT task launch dynamically maintains workload balance across SMs.

### 2.7 Runtime optimizations — paged shared memory, cross-task pipelining, descriptor prefetch

> **Paged shared-memory abstraction.**
>
> In conventional GPU programming models such as CUDA and Triton, shared memory is a fast on-chip memory that is private to each thread block. Shared memory exists only for the lifetime of a kernel—once the kernel finishes, its shared memory is automatically released. In the current kernel-per-operator approach, each kernel assumes exclusive access to the entire shared memory to comply with this programming model. However, this design prevents cross-task software pipelining (Section 2), which overlaps data loading for a subsequent task with the computation of the current task, since both tasks need to access shared memory.
>
> To enable such pipelining, MPK introduces a paged shared-memory abstraction. Shared memory is partitioned into multiple fixed-size pages, and task implementations are modified to operate on these pages instead of assuming monolithic allocation. A task may acquire one or multiple pages based on its shared-memory footprint and must release the pages when they are no longer needed. Once a task releases any of its pages, it is no longer permitted to acquire more shared memory, ensuring monotonic usage patterns that simplify scheduling. When the current task signals its release, MPK can immediately preallocate available pages for the next task and begin data prefetching. This design enables fine-grained, on-demand allocation of shared memory resources within the mega-kernel execution model.

> **Cross-task software pipelining.**
>
> To enable software pipelining across tasks executed on the same worker (Section 2.1), MPK decomposes each task into a pre-loading phase and a compute phase. The pre-loading phase issues data transfer instructions to fetch a chunk of the required tensor from device memory into shared memory, initializing the intra-task software pipeline but performing no computation.
>
> MPK opportunistically overlaps the compute phase of the current task T1 with the pre-loading phase of the subsequent task T2 when two conditions hold: (1) T1 has already issued all of its own data-transfer instructions, and (2) sufficient shared-memory pages are available for T2's pre-loading phase. Note that such a pipeline does not interfere with T1's execution, as MPK inserts appropriate intra-SM synchronization barriers to ensure that T2's memory transfers do not conflict with ongoing data transfers for T1.

> **Pre-fetching task descriptions.**
>
> Each worker maintains both JIT and AOT task queues in device memory. Every task is associated with a task description that encodes its input tensors, output tensors, and configuration parameters; in our current implementation, each description occupies 352 bytes (Section 6.1). To reduce enqueue/dequeue latency and hide device-memory access costs, MPK employs a lightweight prefetching mechanism that retrieves upcoming task descriptions into shared memory before they are needed.

### 2.8 Implementation details and task-launch overhead techniques

> We implement MPK as a kernel backend for PyTorch. A PyTorch program can be compiled into a MPK mega-kernel via PyTorch's compilation interface by specifying MPK as the backend, i.e., torch.compile(backend=MPK). This call invokes the MPK compiler, which generates a mega-kernel and returns it as a callable PyTorch function. Executing this function issues a single launch of the generated mega-kernel. The current MPK implementation consists of approximately 40K lines of C++, 84K lines of CUDA, and 10K lines of Python. The in-kernel parallel runtime is written in CUDA and uses semaphores in device memory to coordinate workers and schedulers. The MPK compiler, implemented in C++ and Python, automatically transforms an input tensor program into an optimized ttGraph tailored to specific GPU types. For compute tasks, the compiler integrates the Mirage superoptimizer [33] to automatically generate optimized CUDA implementations and uses NVSHMEM [10] to support in-kernel inter-GPU communication.

> **Task-launch overhead.**
>
> Because MPK decomposes computation into tasks that are substantially finer-grained than traditional GPU kernels, minimizing per-task overhead is essential for performance. MPK employs several techniques to keep task-launch costs low. First, the runtime uses extremely lightweight workers and schedulers: event and task queues are implemented as circular buffers in GPU device memory, and enqueue/dequeue operations rely solely on low-cost atomicAdd instructions. Second, MPK adopts a decentralized scheduling strategy in which each scheduler assigns tasks using only local state. This design avoids global coordination and eliminates the communication and synchronization overheads inherent to globally coordinated scheduling.
>
> While MPK currently uses decentralized scheduling, the runtime is designed to support alternative strategies, including globally coordinated scheduling, with only minor code modifications. Exploring these designs and their performance trade-offs is an interesting direction for future work.

> **Supporting runtime dynamism.**
>
> To demonstrate MPK's ability to support highly dynamic workloads, we extend the system with mechanisms required for LLM serving, including continuous batching [36] and paged attention [23]. When processing the start event of a task graph, the scheduler prepares a new decoding iteration by (1) removing completed requests from the previous iteration, (2) admitting newly arrived requests, and (3) updating per-request KV-cache metadata. All of this logic executes within a single task, and the KV-cache metadata is stored in device memory for direct use by paged-attention tasks.
>
> To handle the dynamic batch sizes intrinsic to LLM serving, MPK generates multiple ttGraphs specialized for representative batch sizes (powers of two up to the maximum batch). At runtime, the scheduler selects the appropriate graph based on the current batch size. This approach allows the compiler to produce ttGraphs optimized for specific batch sizes while still preserving flexibility for highly dynamic workloads.

### 2.9 Worker/scheduler split and page geometry per GPU (Table 1 + setup)

> Table 1: MPK configuration in our evaluation.
>
> | GPU | # SMs | # Workers | # Schedulers |
> | --- | --- | --- | --- |
> | A100 | 108 | 104 | 16 |
> | H100 | 132 | 128 | 16 |
> | B200 | 148 | 144 | 16 |

> We evaluate MPK across five widely deployed LLMs that span a range of parameter scales and architectural families, as well as three generations of NVIDIA GPUs: A100, H100, and B200. Table 1 summarizes the MPK configuration for each GPU type. For all experiments, we reserve four SMs for schedulers, allocating a total of 16 scheduler warps (each SM accommodates up to four concurrently active warps). The remaining SMs are allocated to workers. We fix the shared-memory page size to 32KB on all GPUs. This configuration yields 5, 7, and 7 shared-memory pages per SM on A100, H100, and B200, respectively.

> To control for variability in request-arrival patterns, all experiments are conducted under an offline, batched-inference setup, with varying maximum batch sizes. All requests use a fixed prompt length of 64 and decode 1024 tokens.

### 2.10 End-to-end numbers

> Figure 9 shows the end-to-end throughput results. For single-batch inference, MPK improves serving performance by 1.0–1.7× across models and hardware. The improvements are most significant for smaller models and newer GPU generations. This trend arises from three factors:
> (1) kernel-per-operator approaches involve higher kernel-launch overheads, even when using CUDA Graphs;
> (2) these systems suffer from pipeline bubbles because the kernel abstraction prevents cross-task pipelining (Figure 2); and
> (3) SGLang and vLLM perform page allocation and request scheduling on the CPU, adding CPU–GPU synchronization delays.
> These overheads become increasingly significant relative to computation as model sizes shrink and hardware improves.

> The results show that MPK is well-suited for latency-optimized scenarios such as single-batch, low-latency model serving, and can drive LLM inference latency closer to hardware limits. For example, on Qwen3-8B running on an A100 GPU, MPK reduces per-token decoding latency from 14.5 ms—already achieved by highly optimized systems such as vLLM and SGLang—to 12.5 ms, approaching the theoretical lower bound of roughly 10 ms (estimated from loading 16 GB of model parameters over a 1.6 TB/s memory bandwidth).

### 2.11 MoE case study — hybrid balancer and fused gather-GEMM

> Figure 10: Comparing MPK with existing systems for Qwen3-30B-A3B on B200. Each value represents the actual MoE runtime in microseconds of each approach (lower is better), and the numbers above the bars indicate the speedup achieved by MPK-Hybrid-MoE over SGLang-MoE.

> To efficiently serve dynamic workloads such as mixture-of-experts (MoE) models, MPK includes two MoE-specific optimizations: (1) a hybrid workload balancer and (2) a fused gather–GEMM implementation.
>
> **Hybrid workload balancer.**
>
> Because the number of tokens routed to each expert is known only at runtime, predetermining an effective workload partition is challenging. A naive static approach assigns a fixed group of SMs to a pre-designated subset of experts. However, when token routing is highly skewed, this strategy leads to severe load imbalance: some SM groups become oversubscribed while others remain underutilized. At the opposite extreme, a fully dynamic approach based on persistent Grouped-GEMM [8] can balance work across SMs, but introduces significant fine-grained synchronization overheads.
>
> MPK introduces a hybrid strategy that combines the strengths of both approaches. At compile time, the compiler statically partitions the work into expert-specific tasks. At runtime, each task receives a meta-tensor produced by the topk–softmax kernel containing global MoE information: specifically, the number of activated experts and the number of tokens assigned to each expert. Using this global metadata, tasks dynamically refine their workload allocation, splitting work uniformly while avoiding the synchronization overheads associated with fully dynamic scheduling. As shown in Figure 10, the hybrid approach consistently outperforms purely static partitioning across all batch sizes.
>
> **Fused gather-GEMM.**
>
> To leverage tensor memory accelerators (TMAs) on Hopper and Blackwell GPUs, current MoE implementations perform a gather step to pack tokens routed to the same expert into a contiguous memory layout. In the case of Qwen3-30B-A3B with batch size 1 on SGLang, this preprocessing step can account for up to 11% of total MoE execution time. Moreover, in MPK, introducing additional preprocessing tasks would further increase scheduling overhead.
>
> To address these issues, MPK replaces the TMA-based gather with an asynchronous token-level copy integrated directly into the data-loading phase of the GEMM tasks. This fusion eliminates the standalone gather kernel and avoids additional scheduling points while still enabling efficient memory movement. As a result, MPK with fused gather–GEMM achieves consistent speedups over SGLang's implementation.

### 2.12 Multi-GPU: collectives compiled into in-kernel tasks

> We further evaluate MPK's scalability across multiple GPUs on an NVIDIA H100 DGX instance. As with competing baselines, we adopt tensor model parallelism as introduced in Megatron-LM [27]. Users can easily specify tensor-parallel execution by inserting AllReduce layers after attention and gated MLP blocks; MPK then automatically compiles these collective operators into (1) inter-GPU data-transfer tasks (implemented using NVSHMEM's nvshmem_signal_wait_until) and (2) local reduction tasks. This decomposition transforms synchronous collective communication into fully asynchronous tasks, allowing it to integrate with MPK's task-based, event-driven runtime.
>
> Figure 11 shows the performance results. Compared to PyTorch, which uses a combination of hand-optimized kernels, CUDA Graphs, and torch.compile, MPK's mega-kernel execution improves throughput by up to 10×. Compared to highly optimized serving systems such as SGLang and vLLM, MPK achieves 1.1–1.4× speedups when scaling to 8 H100 GPUs. These gains arise from three key optimizations missing in kernel-per-operator baselines: (1) MPK integrates page allocation and request scheduling directly inside the mega-kernel, eliminating CPU-side dispatch overheads; (2) MPK's asynchronous execution model overlaps compute tasks with collective communication; and (3) MPK eliminates kernel barriers and enables cross-task software pipelining.

### 2.13 Ablations

> Figure 12: Ablation study on cross-task pipelining. We measure the performance of the final linear layer in Qwen3-8B on an NVIDIA B200 GPU and report the actual runtime in microseconds (lower is better), and the number shows the relative speed-up of MPK-Pipe over MPK-No-Pipe.

> **Cross-task pipelining.**
>
> As described in Section 5.3, MPK enables cross-task pipelining by preloading chunks of input tensors for the next task in parallel with the computation of the current task. Figure 12 shows an ablation study to evaluate the impact of cross-task pipelining on the final linear layer in Qwen3-8B. Cross-task pipelining reduces the task runtime by 1.2-1.3× and even outperforms cuBLAS compiled kernels.

> **Compute-communication overlap.**
>
> MPK captures fine-grained dependency between tasks (see Figure 5), allowing the MPK runtime to opportunistically overlap compute and communication tasks. Figure 13 shows an ablation study to evaluate the impact of such overlap. Specifically, we can disable such overlap by only capturing coarse-grained dependency between collective operators (e.g., AllReduce) and previous tasks using a single event (as shown in Figure 5c). Figure 13 shows the results. Enabling compute–communication overlap reduces per-iteration latency by 1.1×.

### 2.14 Related-work note on prior megakernels

> **Mega-kernels.**
>
> Prior efforts on mega-kernels largely rely on manual design. For example, FlashDMoE fuses mixture-of-experts computation with inter-GPU communication into a single handcrafted mega-kernel [13]. As another example, Spector et al. manually developed a low-latency mega-kernel for LLaMA-1B [30, 9]. These approaches require extensive engineering effort and deep GPU expertise, and they do not generalize across models or GPUs.

## Relevance

MPK gives us the runtime architecture we would otherwise have to invent: worker cores with
per-core task queues, a small number of scheduler cores polling an event queue, events as
counters in device memory activated by N notifications, AOT pre-enqueue for statically balanced
work (one sync) versus JIT dispatch for data-dependent work (two syncs), a paged on-chip buffer
allocator with the "once you release you may not acquire" monotonicity rule, and 352-byte task
descriptors prefetched into on-chip memory. The AOT/JIT distinction is directly actionable for us:
our MoE routed-expert work at batch=1 is data-dependent (top-8 of 256) and is the JIT case, while
the dense/attention/projection chain is AOT. Section 6.4's statement that SGLang's gather step is
up to 11% of MoE time at batch=1, and MPK's fix of folding the gather into the GEMM's data-loading
phase, bears on our GroupedMatmul 5.596 ms vs 1.212 ms gap. Section 6.5's compilation of AllReduce
into in-kernel transfer tasks + local reduce tasks is the template for removing our 5.35 ms
fully-serial TP reduction.

---

## 3. FlashMoE / FlashDMoE — "Fast Distributed MoE in a Single Kernel"

URL: https://arxiv.org/abs/2506.04667 · HTML v3: https://arxiv.org/html/2506.04667v3
Authors: Osayamen Jonathan Aimuyo et al. (Cornell) — code https://github.com/osayamenja/FlashMoE
Date: arXiv June 2025; v3 renamed FlashMoE; NeurIPS 2025 poster https://neurips.cc/virtual/2025/poster/119124

The only published megakernel that fuses a *distributed MoE layer* — compute + inter-GPU
communication — into one persistent kernel. Actor model with warp specialization.

### 3.1 Abstract

> The computational sparsity of Mixture-of-Experts (MoE) models enables sub-linear growth in compute cost as model size increases, thus offering a scalable path to training massive neural networks. However, existing implementations suffer from low GPU utilization, significant latency overhead, and a fundamental inability to leverage task locality, primarily due to CPU-managed scheduling, host-initiated communication, and frequent kernel launches. To overcome these limitations, we develop FlashMoE, a fully GPU-resident MoE operator that fuses expert computation and inter-GPU communication into a single persistent GPU kernel. FlashMoE enables fine-grained pipelining of dispatch, compute, and combine phases, eliminating launch overheads and reducing idle gaps. Unlike existing work, FlashMoE obviates bulk-synchronous collectives for one-sided, device-initiated, inter-GPU (R)DMA transfers, thus unlocking payload efficiency, where we eliminate bloated or redundant network payloads in sparsely activated layers. When evaluated on an 8-H100 GPU node with MoE models having up to 128 experts and 16K token sequences, FlashMoE achieves up to 9× higher GPU utilization, 6× lower latency, 5.7× higher throughput, and 4× better overlap efficiency compared to state-of-the-art baselines—despite using FP32 while baselines use FP16. FlashMoE shows that principled GPU kernel-hardware co-design is key to unlocking the performance ceiling of large-scale distributed ML. We provide code at https://github.com/osayamenja/FlashMoE.

### 3.2 Kernel-count table — the single most quotable number

> | MoE Implementation | GPU Ops |
> | --- | --- |
> | FlashMoE (this work) | 1 |
> | COMET [12] | 33 |
> | Megatron-LM CUTLASS [13, 16] | 85 |
> | Megatron-LM TE [13, 16] | 261 |
> | Megatron-LM + DeepEP [1] | 432 |
> | DeepSpeedMoE [11] | 550 |
>
> Table 1: We report number of GPU operations launched by MoE implementations by profiling with Nsight Systems [17]. We count operations in a single MoE layer (Gate → Dispatch → Expert → Combine) on 2 A100 GPUs, where each GPU has 32 experts. FlashMoE is the first to fully fuse the distributed MoE layer into a single GPU kernel.

> **Kernel Launch Overhead.**  We compare the kernel launch overheads between FlashMoE and existing baselines. Table 1 shows the number of kernel launches during a single forward pass: FlashMoE launches exactly one persistent kernel, while the baselines launch up to 550 short-lived kernels to perform the same computation. Figure 4 provides a visual comparison using CUDA API traces captured by NSight Systems, illustrating the difference between FlashMoE and DeepEP. DeepEP exhibits many small CUDA API calls, with frequent stalls between individual operators, leading to increased GPU idle time (Figure 4(a)). In contrast, FlashMoE maintains high GPU utilization by avoiding launch overhead and synchronization gaps—achieving 93.17% GPU utilization compared to 14% for DeepEP.

> Figure 4: 4(a) shows GPU utilization averaged across 100 MoE forward passes on 2 A100s with 300 GB/s unidirectional bandwidth, where we observe up to 90% idle time, due to kernel launch gaps and non-overlapping communication.

### 3.3 Actor model — Processor / Scheduler / Subscriber, and the block/warp split

> **Actor-based model.**
>
> The design of FlashMoE is based on the actor model of concurrent computation [26, 27, 28]. We implement this model by specializing GPU thread blocks and warps into three distinct actor roles: (1) Processor (§F.1), (2) Subscriber (§F.3), and (3) Scheduler (§F.2). The Processor performs compute (GEMMs and element-wise operations) and tile communication. We use CUTLASS [19] as the underlying infrastructure for high-performance BLAS routines and NVSHMEM for kernel-initiated communication [18]. The Subscriber and Scheduler perform administrative functions. Specifically, the Scheduler assigns computational tasks to available thread blocks. Our key innovation is making the Scheduler both multithreaded, enabling high scheduling throughput, and work-conserving, ensuring consistently high GPU SM utilization. On the other hand, the Subscriber decodes tile packets from peer GPUs to task descriptors (§3.1). Of the N thread blocks on a GPU, we specialize N−1 to adopt the Processor role. We specialize the last block as the Operating System (OS). Within this block, we specialize three warps for the Subscriber role and one warp for the Scheduler role. This split of thread blocks across actors is intentional: our goal is to use few resources for administrative tasks while reserving bulk of the resources for performing MoE computation tasks.

> **In-kernel Block scheduling and Tile parallelism.**
>
> FlashMoE implements tile-level parallelism, meaning it partitions input token matrices into smaller, independent units called tiles, which are processed by blocks but managed (scheduled and constructed) by warps. We specialize every thread block, except one, as processors to perform compute. In addition, we designate a dedicated Operating System (OS) block (4 warps) to perform administrative tasks of (1) scheduling computational work to processors (scheduler), and (2) decoding computational tasks from messages received from other GPUs (subscriber). This design allows FlashMoE to dynamically assign tasks to GPU blocks based on readiness, ensuring that no GPU SM remains idle throughout the lifetime of the MoE operator.

### 3.4 Algorithm 1 — the whole kernel, verbatim structure

> Algorithm 1  FlashMoE Distributed MoE Fused Kernel
>
> Input: A, O ∈ R^{S×H}, X ∈ R^{E×H×D}, N
>
> ```
>  1  begin
>  2    T_phi, G_phi <- FusedGate(A)
>  3    if blockId+1 < N then
>  4      Dispatch(T_phi, A)
>  5      processor::start()
>  6    else
>  7      if warpID == 0 then
>  8        scheduler::start()
>  9      else
> 10        subscriber::start(T_phi, G_phi, O, X)
> 11      end if
> 12    end if
> 13  end
> ```

> Note that A ∈ R^{S×H} is the input token matrix; O ∈ R^{S×H} the output matrix; and X ∈ R^{E×H×D} is a 3-D tensor of expert weights, where E denotes the number of local experts for the executing GPU, H is the embedding dimension, D is the FFN intermediate dimension and S is the sequence length. T_phi ∈ (N×R)^{E×C} is a routing table data structure, where T_phi(e,c)=(i,w) indicates that token i at slot c dispatches to expert e. w is the combine weight (Equation 2) and C is expert capacity. ... G_phi ∈ R^{S×E} captures the affinity scores produced by the gate (Equation 3).

### 3.5 Processor loop (Algorithm 2 body, as rendered)

```
 2    signal <- 0
 3    // shared memory variables
 4    task <- {}
 5    interrupt <- False
 6    complete <- False
 7    while interrupt == False do
 8      if warpId == 0 then
 9        if threadId == 0 then
10          awaitTaskFromScheduler(interrupt, signal)
11          FencedNotifyRQ(ready)
12        end if
13        syncwarp()
14        warpReadTQ(tQ, signal, task)
15      end if
16      syncthreads()
17      if interrupt == False then
18        switch task.Type do
19          case GEMM_0 do
20            // fused GEMM, epilogue and async tile staging
21            fGET(GEMM_0, task)
22            if threadId == 0 then
23              complete <- NotifyTileCompletion()
24            end if
25            syncthreads()
26            if complete == True then
27              NotifySchedulerNextGEMM(tQ)
28            end if
29          end case
30          case GEMM_1 do
31            // fused GEMM, epilogue and async tile transfer
32            fGET(GEMM_1, task)
33          end case
34          case Combine do
35            combine(task)
36          end case
37        end switch
38      end if
39    end while
```

### 3.6 Tile dimension selection — concrete numbers and the reasoning

> **Determining tile dimensions in FlashMoE.**
>
> Selecting appropriate tile dimensions in FlashMoE is crucial to ensure efficient GPU utilization. An undersized tile underutilizes the GPU, while excessively large tiles create register pressure, causing performance-degrading register spills to local memory. After careful parameter sweeps, we choose tile dimensions of (128, 64). Our key insights are: increasing tile width significantly raises the register usage per thread, potentially triggering costly spills; increasing tile height without adjusting thread count increases workload per thread, harming performance. Raising the thread count per block beyond our fixed value of 128 threads reduces the number of concurrent blocks, negatively affecting SM occupancy. Larger thread-block sizes also increase overhead from intra-block synchronization (__syncthreads() barriers), further degrading performance. Thus, our chosen tile dimensions balance register usage, shared-memory constraints, and GPU occupancy to deliver optimal performance.

### 3.7 Unified task descriptor abstraction

> **Unified task abstraction.**
>
> We unify the FFN and combine operations under a common abstraction called a task. Tasks provide a uniform interface for communicating tile-level work among Subscribers, Schedulers, and Processors. Formally, a task descriptor t ∈ T is defined as a tuple:
>
> t = (M, ⋆, φ)
>
> where M is a set of metadata (e.g., device ID, tile index), ⋆ is a binary tensor operation (specifically, matrix multiplication · or Hadamard product ⊙), and φ is an element-wise activation function (e.g., ReLU or identity).
>
> We define a task t operating on input tensors A, B, D, producing output tensor C, as follows:
>
> F_t(A,B,C,D) := C <- φ(A ⋆_t B + D)      (4)
>
> The operator ⋆_t (instantiated from ⋆) may behave differently depending on the task metadata M, and the result of A ⋆_t B is accumulated into D.
>
> In practice, we implement each task defined by Equation 4 as a single fused `__device__` decorated function which the Processor (Algorithm 2) invokes at runtime. Fusion for t entails applying φ and the succeeding addition operation to registers storing the results of the binary operator ⋆_t.

The FFN and combine instantiations, verbatim:

> t1 = (M, ·, φ1),   t2 = (M, ·, φ2),
> F_t1(A,B1,C1,D1) := C1 <- φ1(A B1 + D1),
> F_t2(C1,B2,C2,D2) := C2 <- φ2(C1 B2 + D2).
>
> Whereas, the expert-combine operation is formalized as:
>
> t3 = (M, ⊙, φ2),
> F_t3(A,S,C,C) := C <- φ2(A ⊙ S + C).

### 3.8 Symmetric tensor layout — the non-blocking one-sided communication trick

> Within a single GPU device, the actors in FlashMoE communicate through the GPU's memory subsystem. Specifically, the Scheduler and Subscriber actors exchange data via fast shared memory, while other actor pairs communicate through global memory. For communication across multiple devices, FlashMoE uses device-initiated communication, leveraging the one-sided PGAS (Partitioned Global Address Space) programming model [30]. However, achieving scalable and correct one-sided memory accesses in PGAS without costly synchronization is a known challenge [1, 31]. We address this challenge with a provably correct and scalable solution: a symmetric tensor layout L, supporting fully non-blocking memory accesses. We define L as:
>
> L ∈ R^{P×R×B×E×C×H}
>
> where: P is the expert parallel world size, R identifies communication rounds (i.e., two rounds, one for token dispatch and one for combine), B is number of staging buffers, E is the number of local experts, C is the upscaled expert capacity (§3.2.1) and H is the token embedding dimension. Our core insight to enable non-blocking communication is temporal buffering. Specifically, we overprovision memory for the underlying token matrix by at least 2·r times, where r is the number of communication rounds in the dependency graph, and the factor of 2 accounts for separate buffers for incoming and outgoing data within each communication round. For MoE models, we have 2·r = 4. This modest increase in memory usage eliminates the need for synchronization during one-sided data transfers.
>
> Theorem 3.1. The symmetric tensor layout L is write-write conflict-free.

> Empirically, we find Size(L) ≈ 4·Size(T), contributing memory overhead ≤2% of memory capacity for inference of popular models.

> **3.2.1 In-place Padding for Payload Efficiency**
>
> Due to the dynamic and uneven distribution of tokens in MoE dispatch [32], GPUs commonly receive fewer tokens than their predefined expert capacity. Current MoE frameworks [11] typically pad these buffers with null tokens before computation, unnecessarily increasing communication payloads and degrading performance. In contrast, we propose in-place padding, performing padding directly within the local symmetric tensor buffers and thus eliminating excess network communication.
>
> As we show in Figure 7(a) as a reference, each cell Ei is sized according to the expert capacity C. We further align this capacity to ensure divisibility by the tile block size bM=128, guaranteeing safe and aligned memory reads by Processor threads consuming remote tokens.

### 3.9 Technical challenges list (the engineering checklist)

> **Technical challenges.**
>
> Realizing the single-kernel design of FlashMoE required solving several technical challenges to achieve high performance: (1) lightweight computational dependency management; (2) navigating optimal SM occupancy configurations; (3) implementing in-device BLAS operations; (4) minimizing inter- and intra-device synchronization overheads; (5) implementing transfer-awareness by leveraging DMA over Unified Virtual Addressing (UVA) when available. In addressing these challenges, FlashMoE's design presents a radical departure from traditional synchronous AlltoAll collectives, where GPUs exhibit significant idle time during layer execution. For device-initiated communication, FlashMoE uses NVSHMEM [18] to establish a global address space across all GPUs for Direct Memory Access (DMA) communication. For in-device BLAS, FlashMoE develops custom high-performance GEMM operations via CUTLASS [19].

### 3.10 Limitations, verbatim

> **Engineering complexity.**  Fully fused, persistent kernels demand deep GPU + distributed-systems expertise; future work may investigate compiler/DSL abstractions to lower this barrier.
>
> **FP16 inefficiency.**  Our FP16 path is suboptimal (§H) due to insufficient tuning. We anticipate addressing this gap with autotuned GEMM operators like cuBLASDx [35] or CUTLASS builders.
>
> **Training support.**  This work targets inference; enabling training will require fusing backward computation and gradient communication with new bookkeeping and task descriptors.

## Relevance

This is the closest published analogue to the MoE part of our problem: it is the existence proof
that a routed-MoE layer plus its collectives can be one kernel, and it gives the actor split we
could copy onto Ascend (N−1 compute blocks + one "OS" block of 4 warps = 1 scheduler warp +
3 subscriber warps). Table 1 (1 vs 33/85/261/432/550 GPU ops for the same MoE layer) is the direct
precedent for collapsing our 3008 kernels/round. The "complete <- NotifyTileCompletion(); if
complete then NotifySchedulerNextGEMM" pattern in Algorithm 2 is exactly the GEMM0→GEMM1 chaining
we need for gate/up → SwiGLU → down inside one kernel (our GroupedMatmulSwigluQuantV2 6.793 ms vs
2.272 ms of bytes). The symmetric-tensor 4x overprovisioning theorem is the mechanism that makes
one-sided device-initiated transfer safe without a barrier, which is what our 5.35 ms serial TP
reduce needs.

---

## 4. CANN `SuperKernel` (graph-autofusion) — an Ascend-native megakernel interpreter, open source

This is the single most actionable find for us: CANN ships an open-source, device-side
task-queue interpreter that runs many sub-kernels inside one `__global__` entry on Ascend.
It is structurally the same design as the Hazy Research interpreter and MPK's in-kernel
runtime, already expressed in AscendC, already handling MIX core modes.

Repo: https://gitcode.com/cann/graph-autofusion (CANN org on GitCode/AtomGit)
Raw access pattern that works: `https://raw.gitcode.com/cann/graph-autofusion/raw/master/<path>`
Fetched 2026-10-20.

> Read together with `ascend-fusion-mechanics.md` §6 and §7, which cover the *host* side of the
> same system: the public C API (`aclskOptimize`, `aclskScopeBegin`/`aclskScopeEnd`,
> `aclskScopeVerify`, the full option enum) and how vllm-ascend wires it up via
> `enable_super_kernel`, including a real GLM-5.2-W8A8C8 A3 config. This section covers the
> device side — the interpreter, its task descriptors, its synchronization, and (in §4b) the
> rules that decide what gets fused at all.

### 4.1 Principle — README verbatim (Chinese, as published)

URL: https://gitcode.com/cann/graph-autofusion/blob/master/super_kernel/README.md
Raw: https://raw.gitcode.com/cann/graph-autofusion/raw/master/super_kernel/README.md

> # SuperKernel
>
> ## 原理介绍
>
>    SuperKernel 是一种面向网络图模型的调度优化技术。其核心思想是：基于网络图模型中算子的先验信息（如算子类型、前后序依赖关系等），结合即时编译（JIT）能力，将整个网络模型重新编译为单一算子，从而显著降低算子调度开销，并借助 ICache 预取、Early-Start、同步优化、子 Kernel 拆分等优化手段进一步提升性能。
>
> ***
>    SuperKernel 的设计初衷是将多个子算子融合成一个 SuperKernel，以节省 N-1 次算子调度开销。然而，为维持子算子间的执行顺序，通常需插入全核同步操作，这在一定程度上削弱了调度优化的收益。得益于编译阶段即可获取全部子算子的先验信息，SuperKernel 可在此基础上实施更多深层优化。
>
> 1. **ICache Preload 优化**
>
>    SuperKernel 融合全部算子后，其二进制体积较大。系统在加载算子时通常仅预取入口处指令，导致 SuperKernel 内部大量指令未被预加载至指令缓存（ICache），从而引发较高的 ICache Miss。为此，我们引入 ICache Preload 机制：在当前子算子开始执行前，预加载其后续子算子的代码段，从而有效减少后续算子执行时的 ICache Miss。
>
> 2. **Early-Start 优化**
>
>    在常规调度中，必须等待前序算子全部执行完成后，才能启动后续算子。然而，多数前序算子的末尾指令为 MTE 数据搬运指令，而后续算子的起始指令通常为与输入数据无关的初始化标量指令。由于这两类指令分属不同计算单元，具备并发执行的条件。Early-Start 技术在前序算子的搬运指令前插入 Set 同步点，在后续算子的初始化指令后插入 Wait 同步点，从而实现两个子算子的部分指令并发执行，提升整体执行效率。
>
> 3. **同步优化**
>
>    为保障执行顺序正确，SuperKernel 在各子算子调度之间会插入全核同步操作。对于 Kernel Type 为 Mix 1:2 的混合类型算子，完整的全核同步需等待所有 Aicore 的 Vector 核与 Cube 核均完成同步。SuperKernel 在编译时能够识别每个子算子的类型，因此可针对前后子算子的 Kernel Type 定制同步范围。例如，对于连续的 Vector 算子，仅需执行全 Vector 核同步即可。通过细粒度控制同步范围，可有效降低子算子间的同步开销。
>
> 4. **子 Kernel 拆分**
>
>    在多核系统中，当多个计算核心执行同一段代码时，会并发访问内存中的同一指令地址。这种对同一地址的并发访问会在共享的 L2 Cache 层面形成串行化访问队列，引发资源争用，削弱多核并行带来的性能增益。为解决该问题，SuperKernel 将子 Kernel 代码复制为多份副本，使不同核心能根据核 ID 映射到不同的物理地址执行。这一方法有效缓解了多核对同一指令地址的争用，显著提升算子执行效率。
>
>    此外，SuperKernel 还支持基于内存语义的 Notify 与 Wait 事件，以适配 Tiling 下沉与 Weight 预取等场景。Tiling 下沉算子指的是 Tiling 计算依赖前序算子的输出结果，为避免主机与设备间的频繁交互，将 Tiling 计算部署于 AICpu 执行的算子。若 SuperKernel 融合了该 Tiling 算子的前序算子，则需在前序算子执行完成后通过 Notify 事件通知 AICpu 启动 Tiling 计算；若融合了 Tiling 下沉算子本身，则需通过 Wait 事件等待 AICpu 完成 Tiling 计算后再执行 Device 侧计算。Weight 预取则借助 CMO 任务调用专用硬件单元 SDMA，将数据提前加载至 L2 Cache，以提升计算效率。SDMA 与 Aicore 之间的协作正是通过内存语义的 Notify/Wait 事件实现的。

Directory structure, verbatim:

```text
super_kernel/
├── docs                   # 文档介绍
├── examples               # 示例脚本或 Notebook，演示典型用法
├── include                # SuperKernel 对外头文件
├── kernel                 # 设备侧 Kernel 实现
├── scripts                # 脚本路径
├── src                    # SuperKernel 业务代码
│   ├── aot                # AOT 编译与优化实现
│   └── jit                # JIT 编译实现
├── tests                  # 测试工程目录
│   ├── aot                # AOT 测试
│   ├── fixtures           # 共享测试数据与夹具
│   ├── st                 # System Test
│   ├── ut                 # Unit Test
│   ├── utils              # 通用校验与工具函数
│   └── conftest.py        # pytest 公共配置
├── CMakeLists.txt         # CMake 配置文件
├── pyproject.toml         # 项目元信息与打包配置
├── README.md              # 中文说明文档
├── README_en.md           # 英文说明文档
└── requirements-dev.txt   # python 依赖配置文件
```

Developer-guide note fixing the target SoC, verbatim
(https://raw.gitcode.com/cann/graph-autofusion/raw/master/docs/zh/super_kernel/developer_guide.md):

> - `soc_version`（`fixtures/config.py`）：指定默认 SoC 信息，当前默认值为 `Ascend910_9391`；如需使用其他版本，可在场景内覆盖或增加新的 fixture。

### 4.2 The device-side entry points — one `__global__` per MIX ratio

Source: `super_kernel/kernel/sk_entry.asc`
Raw: https://raw.gitcode.com/cann/graph-autofusion/raw/master/super_kernel/kernel/sk_entry.asc

```cpp
#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define SK_ENTRY_ENABLE_SIMT 1
#include "simt_api/asc_simt.h"
#endif
#include "sk_common.h"
#include "sk_flag_diag.h"
#include <cstdio>

typedef void (*sk_sub_func)(const __gm__ void *param, const sk::SkSystemArgs *sysArgs);
```

```cpp
extern "C" __global__ __attribute__((aligned(512))) __mix__(0, 1) void sk_entry_aiv(GM_ADDR skDevArgs) {
  spk_entry_impl<0, 1, false, false>(skDevArgs);
}

extern "C" __global__ __attribute__((aligned(512))) __mix__(1, 0) void sk_entry_aic(GM_ADDR skDevArgs) {
  spk_entry_impl<1, 0, false, false>(skDevArgs);
}

extern "C" __global__ __attribute__((aligned(512))) __mix__(1, 1) void sk_entry_mix11(GM_ADDR skDevArgs) {
  spk_entry_impl<1, 1, false, false>(skDevArgs);
}

extern "C" __global__ __attribute__((aligned(512))) __mix__(1, 2) void sk_entry_mix12(GM_ADDR skDevArgs) {
  spk_entry_impl<1, 2, false, false>(skDevArgs);
}
```

Template signature of the interpreter, verbatim:

```cpp
template <uint8_t aic, uint8_t aiv, bool enable_debug = false, bool enable_profiling = false,
          bool enable_op_trace = false, bool enable_early_start = false>
__aicore__ inline void spk_entry_impl(GM_ADDR skDevArgs) {
  const __gm__ SkHeaderInfo *sk_header = (const __gm__ SkHeaderInfo *)(get_para_base());
```

### 4.3 The dispatch loop — queue selection, then switch on task type

Note `que_offset` picks a *different copy* of the queue per core
(`(((uint8_t)get_coreid()) % 4) * aicQueSize`) — the sub-kernel-split / L2-ICache-contention
mitigation, expressed in the queue addressing.

```cpp
  uint32_t que_offset = 0;
  if ASCEND_IS_AIC {
    que_offset = sk_header->aicQueOffset + (((uint8_t)get_coreid()) % 4) * sk_header->aicQueSize;
    ...
  }
  if ASCEND_IS_AIV {
    que_offset = sk_header->aivQueOffset + (((uint8_t)get_coreid()) % 4) * sk_header->aivQueSize;
    ...
  }
  const __gm__ TaskQue *taskQue = (const __gm__ TaskQue *)((const __gm__ uint8_t *)sk_header + que_offset);
  for (auto i = 0; i < taskQue->taskCnt; i++) {
    const __gm__ TaskInfo *task = taskQue->taskInfos + i;
    if (task->type == SkTaskType::TYPE_PRELOAD) {
      auto blockId = AscendC::GetBlockIdx();
      if (blockId < task->numBlocks) {
        preload((const void *)(task->entry[((uint8_t)get_coreid()) % task->entryCnt]), task->args);
        dc_preload((__gm__ uint64_t *)(task->extraInfo), 0);
        dc_preload((__gm__ uint64_t *)(task->extraInfo) + 8, 0);
        dc_preload((__gm__ uint64_t *)(task + 1), 0);
      }
    } else if (task->type == SkTaskType::TYPE_FUNC) {
      auto blockId = AscendC::GetBlockIdx();
      if (blockId < task->numBlocks) {
        ...
        sk::SkSystemArgs sysArgs = {
            static_cast<uint16_t>(blockId), static_cast<uint16_t>(task->numBlocks),
            static_cast<uint16_t>(task->extraInfo),  // extraInfo field used for early start sync config
        };

        if ((task->debugOptions & 0x4) != 0) {
          dcci(nullptr, 1, 2);
        }

        ((sk_sub_func)(task->entry[((uint8_t)get_coreid()) % task->entryCnt]))((const __gm__ void *)task->args,
                                                                               &sysArgs);
        ...
        if ((task->debugOptions & 0x20) != 0) {
          dcci(nullptr, 1, 2);
        }
        ...
      }
      ...
    } else if (task->type == SkTaskType::TYPE_SYNC) {
      ...
        if constexpr (enable_early_start) {
          sk::AutoCoreSyncImpl<aic, aiv>((SkCoreSyncType)task->args, task->numBlocks, task->extraInfo);
        } else {
          sk::AutoCoreSyncImpl<aic, aiv>((SkCoreSyncType)task->args);
        }
      ...
    } else if (task->type == SkTaskType::TYPE_EVENT_NOTIFY) {
      GM_ADDR eventAddr = reinterpret_cast<GM_ADDR>(task->args);
      const uint64_t eventValue = task->entry[0];
      if ASCEND_IS_AIC {
        sk::NotifyFunc<true>(eventAddr, eventValue);
      }
      if ASCEND_IS_AIV {
        sk::NotifyFunc<false>(eventAddr, eventValue);
      }
    } else if (task->type == SkTaskType::TYPE_EVENT_WAIT) {
      GM_ADDR eventAddr = reinterpret_cast<GM_ADDR>(task->args);
      const uint64_t eventValue = task->entry[0];
      const uint32_t waitFlag = static_cast<uint32_t>(task->extraInfo);
      if ASCEND_IS_AIC {
        sk::WaitFunc<true>(eventAddr, eventValue, waitFlag);
      }
      if ASCEND_IS_AIV {
        sk::WaitFunc<false>(eventAddr, eventValue, waitFlag);
      }
    } else if (task->type == SkTaskType::TYPE_EVENT_RESET) {
      GM_ADDR eventAddr = reinterpret_cast<GM_ADDR>(task->args);
      const uint64_t eventValue = task->entry[0];
      if ASCEND_IS_AIC {
        sk::ResetFunc<true>(eventAddr, eventValue);
      }
      if ASCEND_IS_AIV {
        sk::ResetFunc<false>(eventAddr, eventValue);
      }
    }
  }
  ...
  pipe_barrier(PIPE_ALL);
```

### 4.4 Full-core barrier and the per-direction sync primitives

```cpp
template <uint8_t aic, uint8_t aiv>
__aicore__ inline void FullCoreSyncImpl() {
  if constexpr (aic == 1 && aiv == 0) {
    AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(AscendC::SYNC_AIC_FLAG);
    AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIC_FLAG);
  } else if constexpr (aic == 0 && aiv == 1) {
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(AscendC::SYNC_AIV_ONLY_ALL);
    AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIV_ONLY_ALL);
  } else {
    if ASCEND_IS_AIC {
      AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(AscendC::SYNC_AIC_FLAG);
      AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIC_FLAG);
      AscendC::CrossCoreSetFlag<0x02, PIPE_FIX>(AscendC::SYNC_AIC_AIV_FLAG);
      AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIV_FLAG);
    }
    if ASCEND_IS_AIV {
      AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(AscendC::SYNC_AIV_ONLY_ALL);
      AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIV_ONLY_ALL);
      AscendC::CrossCoreSetFlag<0x02, PIPE_MTE3>(AscendC::SYNC_AIV_FLAG);
      AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIC_AIV_FLAG);
    }
  }
}
```

```cpp
template <uint8_t aic, uint8_t aiv>
__aicore__ inline void AutoCoreSyncImpl(SkCoreSyncType syncType) {
  switch (syncType) {
    case SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC:
      if ASCEND_IS_AIC {
        AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(AscendC::SYNC_AIC_FLAG);
        AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIC_FLAG);
      }
      return;
    case SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV:
      if ASCEND_IS_AIV {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(AscendC::SYNC_AIV_ONLY_ALL);
        AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIV_ONLY_ALL);
      }
      return;
    case SkCoreSyncType::INTER_SYNC_SET_AIC_TO_AIV:
      if ASCEND_IS_AIC {
        AscendC::CrossCoreSetFlag<0x02, PIPE_FIX>(AscendC::SYNC_AIC_AIV_FLAG);
      }
      return;
    case SkCoreSyncType::INTER_SYNC_SET_AIV_TO_AIC:
      if ASCEND_IS_AIV {
        AscendC::CrossCoreSetFlag<0x02, PIPE_MTE3>(AscendC::SYNC_AIV_FLAG);
      }
      return;
    case SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV:
      if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIC_AIV_FLAG);
      }
      return;
    case SkCoreSyncType::INTER_SYNC_WAIT_AIV_TO_AIC:
      if ASCEND_IS_AIC {
        AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIV_FLAG);
      }
      return;
    case SkCoreSyncType::ALL_SYNC:
      FullCoreSyncImpl<aic, aiv>();
      return;
    default:
      AscendC::Trap();
      return;
  }
}
```

### 4.5 Early-Start — the bitmask form of a split set/wait

```cpp
template <uint8_t aic, uint8_t aiv>
__aicore__ inline void AutoCoreSyncImpl(SkCoreSyncType syncType, uint8_t numBlocks, uint64_t syncConfig) {
  if (syncConfig == 0) {
    AutoCoreSyncImpl<aic, aiv>(syncType);
    return;
  }

  if (AscendC::GetBlockIdx() < numBlocks) {
    return;
  }

  if ASCEND_IS_AIC {
    if ((syncConfig & static_cast<uint64_t>(SkEarlyStartMask::AIC_TO_AIC_SET)) != 0) {
      AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(AscendC::SYNC_AIC_FLAG);
    }
    if ((syncConfig & static_cast<uint64_t>(SkEarlyStartMask::AIC_TO_AIC_WAIT)) != 0) {
      AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIC_FLAG);
    }
    if ((syncConfig & static_cast<uint64_t>(SkEarlyStartMask::AIC_TO_AIV_SET)) != 0) {
      AscendC::CrossCoreSetFlag<0x02, PIPE_FIX>(AscendC::SYNC_AIC_AIV_FLAG);
    }
    if ((syncConfig & static_cast<uint64_t>(SkEarlyStartMask::AIV_TO_AIC_WAIT)) != 0) {
      AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIV_FLAG);
    }
  }

  if ASCEND_IS_AIV {
    if ((syncConfig & static_cast<uint64_t>(SkEarlyStartMask::AIV_TO_AIV_SET)) != 0) {
      AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(AscendC::SYNC_AIV_ONLY_ALL);
    }
    if ((syncConfig & static_cast<uint64_t>(SkEarlyStartMask::AIV_TO_AIV_WAIT)) != 0) {
      AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIV_ONLY_ALL);
    }
    if ((syncConfig & static_cast<uint64_t>(SkEarlyStartMask::AIV_TO_AIC_SET)) != 0) {
      AscendC::CrossCoreSetFlag<0x02, PIPE_MTE3>(AscendC::SYNC_AIV_FLAG);
    }
    if ((syncConfig & static_cast<uint64_t>(SkEarlyStartMask::AIC_TO_AIV_WAIT)) != 0) {
      AscendC::CrossCoreWaitFlag(AscendC::SYNC_AIC_AIV_FLAG);
    }
  }
}
```

### 4.6 Memory-semantic Notify / Wait — the GM counter mechanism, spelled out

This is the Ascend equivalent of Hazy Research's "array of counters in GPU global memory",
and of MPK's "semaphores in device memory". Note the `dcci` cache-invalidate inside the spin
loop — that is what makes the poll see remote writes.

```cpp
      } else if (flag == static_cast<uint32_t>(SkMemoryWaitFlag::EQ)) {
        dcci(waitLock, 0, 2);
        while (*waitLock != value) {
          dcci(waitLock, 0, 2);
        }
      } else if (flag == static_cast<uint32_t>(SkMemoryWaitFlag::AND)) {
        dcci(waitLock, 0, 2);
        while ((*waitLock & value) == 0) {
          dcci(waitLock, 0, 2);
        }
      } else {
        dcci(waitLock, 0, 2);
        while ((~(*waitLock | value)) == 0) {
          dcci(waitLock, 0, 2);
        }
      }
```

```cpp
template <bool aic_flag>
__aicore__ inline void ResetFunc(GM_ADDR param, uint64_t value) {
  if constexpr (aic_flag) {
    if (get_block_idx() == 0) {
      __gm__ uint64_t *resetLock = reinterpret_cast<__gm__ uint64_t *>(param);
      *resetLock = value;
      dcci(resetLock, 0, 2);
    }
  } else {
    if (AscendC::GetBlockIdx() == 0) {
      __gm__ uint64_t *resetLock = reinterpret_cast<__gm__ uint64_t *>(param);
      *resetLock = value;
      dcci(resetLock, 0, 2);
    }
  }
}
```

### 4.7 All data structures and enums, verbatim

Source: `super_kernel/src/aot/sk_common.h`
Raw: https://raw.gitcode.com/cann/graph-autofusion/raw/master/super_kernel/src/aot/sk_common.h

```cpp
enum class SkKernelArch : uint32_t {
  UNKNOWN = 0,
  DAV_2201,
  DAV_3510,
};

constexpr std::array<SkKernelArch, 2> SK_SUPPORTED_KERNEL_ARCHS = {
    SkKernelArch::DAV_2201,
    SkKernelArch::DAV_3510,
};

constexpr uint32_t SK_EVENT_DAV_2201_CORE_NUM = 75;
constexpr uint32_t SK_EVENT_DAV_3510_CORE_NUM = 108;
constexpr uint32_t SK_DAV_2201_TICK_US_MULTIPLIER = 50;
constexpr uint32_t SK_DAV_3510_TICK_US_MULTIPLIER = 1000;

constexpr uint32_t MAX_SCOPE_NAME_LEN = 256;
constexpr char DEFAULT_SK_SCOPE_NAME[] = "default_sk_scope_name";
```

```cpp
enum class SkKernelType : uint8_t {
  AIC_ONLY = 1,
  AIV_ONLY = 2,
  MIX_AIV_1_0 = 3,
  MIX_AIC_1_0 = 4,
  MIX_AIC_1_1 = 5,
  MIX_AIC_1_2 = 6,
  DEFAULT = 0xFF,
};

constexpr size_t SK_KERNEL_TYPE_COUNT = 6;
```

```cpp
enum class SkTaskType : uint8_t {
  TYPE_FUNC,
  TYPE_SYNC,
  TYPE_PRELOAD,
  TYPE_EVENT_NOTIFY,
  TYPE_EVENT_WAIT,
  TYPE_EVENT_RESET,
  TYPE_MAX,
};

enum class SkCoreSyncType : uint8_t {
  ALL_SYNC = 0,
  CROSS_SYNC_AIC_TO_AIC,
  CROSS_SYNC_AIV_TO_AIV,
  INTER_SYNC_SET_AIC_TO_AIV,
  INTER_SYNC_SET_AIV_TO_AIC,
  INTER_SYNC_WAIT_AIC_TO_AIV,
  INTER_SYNC_WAIT_AIV_TO_AIC,
  SYNC_NONE,
};
```

```cpp
// Memory wait rule for VALUE_WAIT tasks.
// The numeric values must stay aligned with runtime's aclrtValueWait flag definition.
enum class SkMemoryWaitFlag : uint32_t {
  GEQ = 0x0,  // Wait until *addr >= value
  EQ = 0x1,   // Wait until *addr == value
  AND = 0x2,  // Wait until (*addr & value) != 0
  NOR = 0x3,  // Wait until ~(*addr | value) != 0
};

constexpr uint64_t SK_DEFAULT_NOTIFY_VALUE = 1;
constexpr uint64_t SK_DEFAULT_WAIT_VALUE = 1;
constexpr uint64_t SK_DEFAULT_RESET_VALUE = 0;
constexpr uint32_t SK_DEFAULT_WRITE_FLAG = 0;
```

```cpp
struct TaskInfo {
  uint32_t index;
  SkTaskType type;
  SkKernelType relatedType = SkKernelType::DEFAULT;
  uint8_t numBlocks;
  uint8_t entryCnt;
  uint64_t entry[4];
  // 根据bit位确定option选项
  // 1： dcci_disable_on_kernel
  // 2： debug_sync_all
  // 4： dcci_before_kernel_start
  // 8： dcci_after_kernel_end
  // 16： debug_cross_core_sync_check
  // 32： enable_dcci_after_func - 直接指示kernel是否需要在func执行后插入dcci
  //      该bit由host端根据disableDcci和afterKernelEnd综合计算得出
  //      kernel侧只需检查此bit即可，无需组合判断
  uint64_t debugOptions;
  uint64_t extraInfo;
  uint64_t args;
  uint32_t argsSize;
  uint8_t isSimtKernel;
  uint8_t reservedList[3];
};
```

```cpp
enum class SkEarlyStartMask : uint32_t {
  NONE = 0U,
  AIC_TO_AIC_SET = 1U << 0,
  AIC_TO_AIC_WAIT = 1U << 1,
  AIC_TO_AIV_SET = 1U << 2,
  AIV_TO_AIC_WAIT = 1U << 3,
  AIV_TO_AIV_SET = 1U << 4,
  AIV_TO_AIV_WAIT = 1U << 5,
  AIV_TO_AIC_SET = 1U << 6,
  AIC_TO_AIV_WAIT = 1U << 7,
  SPLIT_CORE_CTRL = 1U << 15,
};
```

```cpp
struct TaskQue {
  uint32_t taskCnt;
  uint32_t cap;
  TaskInfo taskInfos[0];
};

struct SkHeaderInfo {
  uint32_t aicQueSize;
  uint32_t aivQueSize;
  uint32_t aicQueOffset;
  uint32_t aivQueOffset;
  uint32_t counterOffset;
  uint32_t dfxOffset;
  uint32_t eventConfigOffset;  // 算子打印事件配置偏移量
  uint32_t nodeCnt;
  uint64_t modelIdIndexAndSkScopeId;
  uint64_t totalSize;
};
```

### 4.8 Entry-point variants and the AOT API

The 16 declared entry points are the cross product of {aiv, aic, mix11, mix12} x
{plain, _debug, _dump_profiling, _debug_dump_profiling}:

```cpp
void sk_entry_aiv / sk_entry_aic / sk_entry_mix11 / sk_entry_mix12
void sk_entry_aiv_debug / sk_entry_aic_debug / sk_entry_mix11_debug / sk_entry_mix12_debug
void sk_entry_aiv_dump_profiling / sk_entry_aic_dump_profiling
void sk_entry_mix11_dump_profiling / sk_entry_mix12_dump_profiling
void sk_entry_aiv_debug_dump_profiling / sk_entry_aic_debug_dump_profiling
void sk_entry_mix11_debug_dump_profiling / ...
```

The AOT host API, from the internal design document (see
`ascend-launch-overhead-and-sync.md` for the Wiki URL):

```cpp
// src/aot/super_kernel.cpp
aclError aclskOptimize(aclmdlRI model, aclskOptions *options) {
    SuperKernelGraph graph(model);
    graph.InitSKGraph();

    SuperKernelOptionsManager opts;
    opts.ParseOptions(options);

    SuperKernelOptimizer optimizer(opts);
    optimizer.Process(graph);

    return ACL_SUCCESS;
}
```

The runtime API added for prefetch sizing:

```cpp
aclError aclrtGetFunctionSize(aclrtFuncHandle funcHandle, size_t *aicSize, size_t *aivSize);

/**
 * @ingroup rts_kernel
 * @brief get kernel pc start address in device, if kernel is mix, the output param aicAddr and aivAddr are both valid,
 * otherwise, there will be only one valid address.
 * @param funcHandle the kernel
 * @param aicAddr output param: kernel pc start address of aicore
 * @param aivAddr output param: kernel pc start address of aivector
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtsFuncGetAddr(const rtFuncHandle funcHandle, void **aicAddr, void **aivAddr);
```

### 4.9 Known limits of SuperKernel as shipped

From the internal 串讲 notes (Wiki `WIKI2026022410189615`, 学习笔记):

> 目前 superkernel 现在有诸多限制：
>
> * 未支持 aclgraph，据说有支持计划，后续可以关注下
> * 存量的 TBE 算子无法融入 superkernel，目前有补丁办法可以解决，但是未合入正式的主线方案，据说是 2026/01 在 CANN 包中体现
> * torch 原生的算子无法融入 superkernel <-- 其实没有原生算子这么一说，torch.xxx 这种算子在昇腾上就是用 TBE 实现的~
> * ATB 加速算子不支持融入 superkernel
>
> 换句话说，​**superkernel 目前只支持融入使用 Ascend C 写的算子**​~

> ## 八、重要技术决策
>
> | 决策 | 说明 |
> |------|------|
> | **串行调度** | 第一版采用纯串行调度，暂不支持 Cube/Vector 并行 |
> | **Mix1:2 强制提升** | 混合算子检测 Mix1:2 后强制提升模式 |
> | **待确定功能** | DFX 调试能力、双流并行调度、ACLGraph 正式接口集成 |

> **Mix1:1 → Mix1:2 自适应**：
>
> - 检测序列中是否包含 Mix1:2 算子
> - 若存在，强制整个 SuperKernel 按 Mix1:2 模式启动
> - 防止因 Vector 核数量不足导致的同步错误

And the reported benefit, verbatim from `WIKI202601229899315` (runtime 组):

> 相对于单算子下发，SuperKernel技术可以减少任务调度等待时间和调度开销，同时利用Task间隙资源进一步优化算子头开销。
>
> superKernel的参数中融合多个子kernel，在superkernel内部直接调用算子，仅需一次stars调度后便可执行多个kernel。由于kernel内的调用路径比stars更短，性能更好，尤其是大量连续的aicore算子时，一个superkernel可以融合上千个子算子，调用收益可达15%以上，为了获得该收益ac IGraph需要支持superkernel。

> * 两个Task之间的调度开销降低
> * 每个Task结束后会执行Cache Flush操作，将所有修改的Cache内容刷新出去，其中包含一些不必要的栈空间
> * 算子调度后的核启动时间，包括同地址访问的排队延迟、Scalar执行初始化等环节
>   superkernel从这三个方面去减少

## Relevance

This is the Ascend megakernel substrate we were about to write from scratch, and it already
exists, is open source, is written in AscendC, and compiles for `Ascend910_9391`. Direct
mappings to what we need: `TaskQue`/`TaskInfo` in GM is the ahead-of-time instruction
sequence; `TYPE_FUNC` with `entry[((uint8_t)get_coreid()) % task->entryCnt]` is the
instruction dispatch plus the L2-ICache-contention fix (up to 4 code copies); `TYPE_PRELOAD`
with `preload()` + `dc_preload()` is the weight/code prefetch that keeps MTE2 busy across
stage boundaries; `SkMemoryWaitFlag::{GEQ,EQ,AND,NOR}` spinning on a `__gm__ uint64_t` with
`dcci` in the loop is precisely the global-memory-counter dependency mechanism from Hazy
Research §1.6 and MPK §5.1, and it is *not* subject to the 8-event-id / 15-count limits of
`CrossCoreSetFlag`; `SkEarlyStartMask` is the mechanism that overlaps the tail MTE of one
stage with the scalar init of the next. Two caveats to design around. First, as shipped the
scheduler is 串行 (serial) — it inserts a barrier between sub-kernels and explicitly does not
yet run Cube and Vector stages concurrently, so we cannot get the MPK-style JIT work-stealing
for free. Second, mixing any MIX_AIC_1_2 sub-kernel forces the whole SuperKernel to 1:2 mode,
which matters for us because our 505 MIX_AIC calls/round would set the mode for everything
else. The claimed benefit (15%+, "一个superkernel可以融合上千个子算子") is for launch
overhead only, consistent with our 11.5 ms launch floor being the thing to attack but *not*
enough on its own — 15% of 11.6198 ms does not reach 2.x ms, so we need the fusion plus the
Early-Start/preload overlap plus the elimination of the 5.35 ms serial TP reduce.

---

## 5. ThunderMLA — "FlashMLA, Faster and Fused-er!" (the interpreter template, and MLA decode as a megakernel)

URL: https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla
Authors: Benjamin Spector, Aaryan Singhal, Dan Fu, Chris Ré (Stanford Hazy Research)
Date: Mar 4, 2025 · 10 min read

The precursor to the Llama-1B megakernel, and the one that is *specifically about MLA decode*.
Same attention mechanism family as our MLA + SparseFlashAttention.

### 5.1 The claim and the workload

> We've been playing with some new schedulers to deal with variable length sequences. (This is a common case in LLM inference, when you're serving requests from many different users.) Like everyone else, we were really excited about DeepSeek MLA so we decided to look into it!
>
> We're excited to introduce ThunderMLA, our response to the performance challenges of large language model inference. ThunderMLA is a completely fused "megakernel" for decode that's 20-35% faster than DeepSeek's FlashMLA on diverse workloads. It turns out some simple scheduling tricks can get you pretty far! And although this release is focused on attention decoding, we think these techniques apply much more broadly.
>
> Figure 0: A caricature of the main message of this post. Fused, scheduled megakernels (B) can be much more performant than separate kernels (A)!
>
> Please be warned: this release is more art than product, although to our surprise, people are already using it in production!

### 5.2 Why decode attention stalls — and the exact baseline numbers

> Attention decode kernels can struggle on variable-prompt workloads. Whereas prefill usually allows for a single kernel to fill the GPU, parallelizing across batch and sequence, when decoding small batches and just a few queries at a time, one needs to use other tricks like FlashInfer to keep the GPU full, but incurs other costs in doing so. The core issue lies in the substantial overhead created by:
>
> Running two separate kernel launches
>
> Tail effects between kernel executions
>
> Limited batch sizes when running large reasoning models (which appear, at the present, to be the future of AI)
>
> These constraints can severely limit performance. For intuition, consider the following decode scenario with imbalanced inputs:
>
> A batch of 4 prompts, of lengths [4641, 45118, 1730, 1696]
>
> Generating 4 new tokens (e.g., for a speculator)
>
> 8-way tensor parallel -- that is, 16 heads per GPU for DeepSeek R1
>
> With 4 new tokens, FlashMLA runs in 52 us on an SXM H100, achieving just 144 TFLOPS and 1199 GB/s (out of the 939 TFLOPS / 3300 GB/s advertised by NVIDIA).
>
> As a partial solution, we're introducing ThunderMLA, a completely fused mega-kernel that represents the first step in our broader effort to rethink CUDA programming; on the same workload, ThunderMLA runs in just 41 us, achieving 183 TFLOPs and 1520 GB/s.
>
> ThunderMLA applies classical computing ideas (like job scheduling) to modern AI workloads through:
>
> A novel interpreter template for building megakernels
>
> The actual ThunderMLA megakernel
>
> Two simple schedulers
>
> Advanced timing infrastructure

### 5.3 The three costs of the many-kernel model, and the instruction-tensor fix

> Traditionally, GPU workloads are written as sequences of kernels that exchange data through global memory (L2 cache or HBM). While this approach allows kernels to be reused, it introduces significant overhead, illustrated in Figure 1:
>
> Kernels must be set up and torn down; each of these can consume several microseconds
>
> Single-wave tail effects can often be substantial in these small-batch, fast-decoding scenarios
>
> Using multiple kernels prevents data reuse, thereby increasing memory bandwidth required by the GPU
>
> Figure 1: a stylized Gantt chart of various kernel overheads. Sometimes, these costs are negligible, but often times they're not!
>
> Our solution eliminates these kernel launches by creating a simple virtual instruction set on the GPU. Instead of launching separate kernels, we fuse them into a single "megakernel," and then pass this megakernel its instructions at runtime through an instruction tensor, which in turn can dramatically modify its behavior. Within the megakernel, GPU SMs read from this instruction tensor, decide what work to perform next, and then execute it. Although we're initially focused on these megakernels for attention decoding scenarios, we think they're an important idea in general -- stay tuned!
>
> We've implemented this through an interpreter template within ThunderKittens, allowing arbitrary simple kernels to be plugged together into a megakernel, each with its own opcode and instructions. Key benefits include:
>
> Vastly simplified pipelining within instructions, across loop boundaries
>
> Asynchronous fetching of the future instructions, minimizing instruction overhead
>
> Deep pipelines that hide data fetch latencies within and across instructions

### 5.4 Size of the implementation, and the template instantiation

> Our implementation adapts a slot attention kernel and token reduction kernel into this interpreter template with just a few hundred lines of device code, and uses a global tensor as a semaphore to synchronize dependencies across instructions. Once both the two constituent kernels ("partial" and "reduction") of the megakernel for FlashMLA are implemented, the megakernel can be instantiated as interpreter::kernel<config, partial_template, reduction_template>. Altogether, this fused megakernel is just 250 lines of device code in ThunderKittens.

### 5.5 The full results table, verbatim

> | Kernel | B 1, Seq 64k, Q 1 | B 64, Seq 256-1024 (random), Q 4 | B 64, Seq 512 Q 2 | B 132, Seq 4k, Q 4 |
> | --- | --- | --- | --- | --- |
> | FlashMLA (μs, TFLOPs, GB/s) | 55.0, 42, 1378 | 47.0, 124, 1212 | 39.0, 59, 1092 | 226.0, 333, 2839 |
> | ThunderMLA (μs, TFLOPs, GB/s) | 44.5, 52, 1700 | 39.5, 147, 2022 | 28.6, 80, 1489 | 210.0, 358, 3055 |
> | Speedup (%) | 23.6% | 19.0% | 36.3% | 7.6% |

### 5.6 The two schedulers — static heap, and makespan-backwards

> Static Scheduler
> Our first approach uses a simple static scheduler that:
>
> Divides jobs into small pieces
>
> Synthesizes a k-way reduction tree to merge operations
>
> Uses a heap-based priority queue to allocate work to available SMs
>
> This yields good performance across various settings, though it struggles to optimally overlap reduction and compute because it doesn't consider task timing when creating instructions. An example of a proposed schedule from the static scheduler can be seen in figure 2A.
>
> Makespan Backwards Scheduler
> Our second, more advanced scheduler improves performance by optimizing the makespan more aggressively:
>
> We know the very last event of the kernel is a single reduction task for each token, so we can work backwards from it
>
> We then run heuristic rollouts to determine optimal execution path
>
> Assigns processor IDs to each task
>
> Taken together, this can shave an additional 10% off execution time, as visualized in figure 2B.
>
> Figure 2: Two simulated Gantt charts from our schedulers. (A) our simple static scheduler. Although this approach doesn't overlap very well, it nonetheless reduces both tail effects and kernel launch times. (B) Our makespan-focused backwards scheduler. Of particular note is that the schedule is determined relative to the end of the kernel, rather than its beginning.

### 5.7 Why a slow (1–2 ms) scheduler is still acceptable — the amortization argument

> Note that while these schedulers are currently slower than the kernel itself (currently requiring 1-2ms), this approach remains viable because:
>
> Schedules can be reused across dozens of layers on large models
>
> Generation can happen asynchronously on the CPU while previous batches are running
>
> Schedules between similar batch sizes change infrequently (e.g., a batch size of 4 only needs schedule regeneration at most every 8 forward passes on average)
>
> Even when schedules change, they usually change in simple enough ways that a simple update is possible

### 5.8 Tooling note — standard profilers do not work on megakernels

> During development, we realized that NVIDIA's tools are increasingly inadequate for understanding the performance of heterogeneous and dataflow-driven kernels. NCU, while an excellent tool we love very much, isn't as well-suited for analyzing these kernels where hardware resources are more constraining than thread execution.
>
> To address this, we've developed custom timing infrastructure that produces detailed Gantt charts of kernel execution, tracking events such as:
>
> Consumer setup phases

## Relevance

The 250-lines-of-device-code figure and the `interpreter::kernel<config, partial_template,
reduction_template>` instantiation are the scope estimate for a staged megakernel if the
interpreter substrate already exists — which on Ascend it does (CANN SuperKernel, §4). Their
B=1, Seq 64k, Q=1 column (44.5 us, 1700 GB/s) is the closest published point to our batch=1
regime, and the fact that even a fully fused MLA decode reaches only ~52% of H100 bandwidth at
B=1 is the warning that fusing attention alone will not close our SparseFlashAttention gap
(6.973 ms spent vs 1.382 ms of bytes = 5.0x). §5.7's amortization argument applies to us with
*more* force: at batch=1 with a fixed 8K sequence, one schedule serves all 78 layers and every
round, so we can afford an arbitrarily expensive offline scheduler. §5.8 is a practical
warning: msprof/op_summary will stop being informative once the kernel count collapses, and we
should plan on in-kernel `get_sys_cnt()` timestamping (which CANN SuperKernel already
implements — see its `SkKernelEventRecord` / `WriteEventRecord` path in §4).

---

## 6. Event Tensor / ETC — "A Unified Abstraction for Compiling Dynamic Megakernel"

URL: https://arxiv.org/pdf/2604.13327 · https://www.alphaxiv.org/abs/2604.13327
Venue: MLSys 2026 — https://proceedings.mlsys.org/paper_files/paper/2026/hash/53d3f45797970d323bd8a0d379c525aa-Abstract-Conference.html
Authors: Hongyi Jin, Bohan Hou, Songting Wang, Ruihang Lai, et al. (CMU et al.)
Date: arXiv April 2026

The newest of the three megakernel compilers, and the one that addresses the thing our MoE
decode actually needs: *data-dependent* dynamism (top-8 of 256 experts, known only at runtime).

### 6.1 Abstract, verbatim

> Modern GPU workloads, especially large language model (LLM) inference, suffer from kernel launch overheads and coarse synchronization that limit inter-kernel parallelism. Recent megakernel techniques fuse multiple operators into a single persistent kernel to eliminate launch gaps and expose inter-kernel parallelism, but struggle to handle dynamic shapes and data-dependent computation in real workloads. We present Event Tensor, a unified compiler abstraction for dynamic megakernels. Event Tensor encodes dependencies between tiled tasks, and enables first-class support for both shape and data-dependent dynamism. Built atop this abstraction, our Event Tensor Compiler (ETC) applies static and dynamic scheduling transformations to generate high-performance persistent kernels. Evaluations show that ETC achieves state-of-the-art LLM serving latency while significantly reducing system warmup overhead.

### 6.2 The launch-overhead arithmetic, verbatim — note the 5–10 us vs 2 us ratio

> The first source of overhead arises from kernel launches. Current systems such as PyTorch (Paszke et al., 2019) launch GPU kernels sequentially from the host CPU (Figure 1, upper left). During LLM inference, each auto-regressive decoding step may involve hundreds or even thousands of fine-grained operations, where the launch overhead cannot be effectively amortized. Each kernel launch typically incurs 5–10 μs of latency, while the fastest kernels may complete in 2 μs, making the launch overhead dominant.
>
> The second source of overhead stems from kernel boundaries, which enforce implicit synchronization between consecutive kernels. In many cases, later kernels depend only ...

### 6.3 Language constructs — Device Function, Event Tensor, Graph Function

> **2.1 Language Constructs**
>
> We first introduce the main language constructs in Event Tensor–based programs.
>
> Device Function. A device function defines a grid of tasks launched in parallel on the GPU. Each launch is parameterized by a multidimensional coordinate, where each coordinate identifies a task tile executed on a streaming multiprocessor (SM). Each task can include specialized logic, such as warp specialization or tensor core calls.
>
> Event Tensor. An Event Tensor is a multi-dimensional structure whose elements represent events—the completion of task sets at the SM level—following established practices in parallel programming systems (Blumofe et al., 1995; Treichler et al., 2014; Bauer et al., 2012; Dagum & Menon, 1998). Each element has an initial wait count recording the number of tasks it depends on and supports several operations: E[i].notify() signals task completion, E[i].wait() blocks until the event is triggered, and in dynamic scheduling, events can also trigger dependent tasks. Compared with approaches that manage standalone events individually, Event Tensors greatly reduce task-graph management overhead when scaling to millions of fine-grained events in real LLM inference workloads.
>
> Graph function. A graph function represents a computational graph consisting of call device calls that explicitly launch device functions with specified task shapes. Unlike traditional computational graphs, it includes both data tensors and Event Tensors. Each device function launch can annotate explicit input/output dependencies and coordinate mappings that track fine-grained task relationships through Event Tensors.
>
> Figure 3 shows an example Event Tensor–based program. The general program can be viewed as a compact representation of task graphs in the form of “producer task → event → consumer task”. We use generic lambda functions to represent event task relations. These dependency annotations implicitly map to event notifications at the end of each producer task and waiting at the beginning of each consumer task.

### 6.4 Figure 3 — the minimal program, verbatim

```python
class IRModule:
    @device_func
    def partial_sum(i: int, j: int, A: Tensor, B: Tensor):
        # i, j are task coordinates
        # The following tasks can be implemented at CTA/cluster level
        B[i*32 : i*32+32, j] = sum(A[i*32 : i*32+32, j*32 : j*32+32])

    @device_func
    def final_sum(i: int, B: Tensor, C: Tensor):
        # i is the task coordinate
        # The following tasks can be implemented at CTA/cluster level
        C[i*32 : i*32+32] = sum(B[i*32 : i*32+32, :])

    @graph_func
    def main_graph(A: Tensor(("n*32", 128))) -> Tensor(("n*32",)):
        n = sym_var()  # an integer symbolic variable
        E = ETensor((n,), wait_count=4)
        B: Tensor((n * 32, 4)) = call_device(
            IRModule.partial_sum, tile_num=(n, 4),
            args=[A], in_edges={}, out_edges={E: "ij->i"},
        )
        C: Tensor((n * 32,)) = call_device(
            IRModule.final_sum, tile_num=(n,),
            args=[B], in_edges={E: "i->i"}, out_edges={},
        )
        return C
```

> Figure 3. Example Event Tensor–based program.

Dependency edges are given in einsum notation (`"ij->i"`, `"i->i"`, `"bh->bh"`, `"h->bh"`).

### 6.5 Shape dynamism — symbolic shapes, no recapture

> Figure 4. Event Tensor handles shape dynamism with symbolic-shape tensors that define a template for dependency graphs. At runtime, the template is instantiated with concrete shape values (e.g., producing a 1 × 2 graph for batch size 1 or a 2 × 2 graph for batch size 2) without recompilation or repeated graph capture.

### 6.6 Data-dependent dynamism — the MoE mechanism, verbatim

This is the section to copy for our 256-expert / top-8 routing.

> **2.4 Supporting Data-Dependent Dynamism**
>
> A critical challenge in modern workloads is handling irregular, data-dependent task graphs. A representative example is the Mixture-of-Experts (MoE) layer (Shazeer et al., 2017), where input tokens are dynamically routed to different expert sub-networks based on routing decisions computed at runtime. An efficient MoE implementation typically first groups tokens according to their assigned experts and then uses GroupGEMM operators to compute the results for all tokens within each group.
>
> Dynamic routing introduces fine-grained, data-dependent task dependencies that are unknown at compile time (specifically, which experts GroupGEMM tile processes which tokens). This dynamism poses a significant challenge for traditional compilers and schedulers, which assume a static task graph with fixed dependencies. To efficiently represent such dynamic workloads, we need an abstraction that can (1) determine at runtime which tasks each consumer task depends on, and (2) trigger a variable number of consumer tasks based on runtime data. The Event Tensor abstraction is designed precisely for this purpose. It manages data-dependent task graphs through two core mechanisms:
>
> Data-Dependent Event Update. Unlike conventional task graphs that support only static dependencies (Figure 5a), the Event Tensor abstraction allows dynamic event dependencies (Figure 5b). In the MoE example, runtime routing decisions stored in the topk tensor determine which grouping tiles (one per token) update which events (one per expert). Each expert's event counter is initialized to the number of tokens routed to it, and this initialization occurs dynamically at runtime, together with the computation of topk.
>
> Data-Dependent Task Triggering. Similarly, an event can trigger a runtime-dependent number of tasks. Based on routing decisions in topk, we can compute how many tokens each expert must process and, therefore, how many GroupGEMM tiles each expert requires. As illustrated in Figure 5b, this information is encoded in the tensor exp indptr, which stores the prefix sum of GroupGEMM tiles to be triggered per expert. Leveraging exp indptr, we enable data-dependent triggering, where expert i activates tiles in the range (exp indptr[i], exp indptr[i+1]).
>
> Together, these two mechanisms allow the compiler to generate megakernels that efficiently adapt to highly dynamic workloads—cases that static task graphs handle poorly. Combined with the symbolic-shape support described above, all compilation in ETC occurs offline; at inference time, the compiled binary handles both shape and data-dependent dynamism with zero compilation overhead.
>
> Note that the overall dependency chain remains strictly feed-forward (Figure 2, right): Attention Output → Token Routing (TopK) → Token Grouping → Token Computation (GroupGEMM). Computing TopK depends only on the preceding Attention output—a standard static dependency. The data-dependent Event Tensor mechanisms described above govern only the later stages, where routing results dynamically determine which grouping tasks notify which expert events and how many GroupGEMM tiles each expert triggers.

(`indptr` footnote, verbatim: "indptr is a term commonly used in sparse matrix representations
such as the compressed sparse row (CSR) format.")

### 6.7 Fused GEMM + ReduceScatter — before/after static scheduling, verbatim

This is the exact shape of our problem: one persistent device function containing both the
matmul tiles and the reduce-scatter tiles, coordinated by an Event Tensor, with the collective
expressed as `multimem_ld_reduce_cta`. Contrast with our 5.35 ms fully serial TP reduce.

Before transformation — two device functions plus a distributed Event Tensor:

```python
# Constants: M, N, K (matrix dimensions),
# BLK_M, BLK_N (block sizes), WORLD_SIZE (number of devices)
LOCAL_M = M // WORLD_SIZE

@device_func
def matmul(i: int, j: int, A: Tensor, B: Tensor, C: Tensor):
    # i is the task coordinate
    with cta():
        matmul_cta(C[i*BLK_M : (i+1)*BLK_M, j*BLK_N : (j+1)*BLK_N],
                   A[i*BLK_M : (i+1)*BLK_M, :],
                   B[:, j*BLK_N : (j+1)*BLK_N])

@device_func
def reduce_scatter(i: int, j: int, C: Tensor, D: Tensor):
    # i, j are task coordinates
    with cta():
        offset = get_rank() * LOCAL_M
        multimem_ld_reduce_cta(
            D[i*BLK_M : (i+1)*BLK_M, j*BLK_N : (j+1)*BLK_N],
            C[offset + i*BLK_M : offset + (i+1)*BLK_M,
              j*BLK_N: (j+1)*BLK_N]
        )

@graph_func
def main_graph(A: Tensor((M, K)), B: Tensor((K, N))) -> Tensor((LOCAL_M, N)):
    # E is a distributed Event Tensor, sharded on dimension 0.
    # Each device holds a shard of shape (LOCAL_M // BLK_M, N // BLK_N)
    E = ETensor((M // BLK_M, N // BLK_N), wait_count=WORLD_SIZE, shard="S[0]")
    C: Tensor((M, N)) = call_device(
        matmul, tile_num=(M // BLK_M, N // BLK_N),
        args=[A, B], in_edges={}, out_edges={E: "ij->ij"},
    )
    E_local = E.local_view()
    D: Tensor((LOCAL_M, N)) = call_device(
        reduce_scatter, tile_num=(LOCAL_M // BLK_M, N // BLK_N),
        args=[C], in_edges={E_local: "ij->ij"}, out_edges={},
    )
    return D
```

After static scheduling transformation — one persistent device function with `SM_COUNT` tiles,
an in-kernel tile scheduler, and explicit notify/wait:

```python
@device_func
def fused_matmul_rs(
    sm_id: int, A: Tensor, B: Tensor, C: Tensor, E: ETensor, D: Tensor
):
    # sm_id is the task coordinate
    with cta():
        tile_scheduler = init_tile_scheduler(sm_id)
        while tile_scheduler.valid():
            task_idx, task_type = tile_scheduler.get_task()
            if task_type == 0:
                i, j = task_idx
                matmul_cta(C[...], A[...], B[...])
                E[i, j].notify()
            else:
                i, j = task_idx
                offset = get_rank() * LOCAL_M
                E[i + offset // BLK_M, j].wait()
                multimem_ld_reduce_cta(D[...], C[...])
            tile_scheduler.next_tile()

@device_func
def main_graph(
    A: Tensor((M, K)), B: Tensor((K, N))
) -> Tensor((LOCAL_M, N)):
    E = ETensor(
        (M // BLK_M, N // BLK_N), wait_count=WORLD_SIZE, shard="S[0]"
    )
    C: Tensor((M, N)) = empty((M, N))  # Allocate intermediate tensor C
    D: Tensor((LOCAL_M, N)) = call_device(
        fused_matmul_rs, tile_num=(SM_COUNT,), args=[A, B, C, E],
        in_edges={}, out_edges={},
    )
    return D
```

> Figure 6. GEMM + Reduce-Scatter before and after static scheduling transformation. Two separate device functions are fused into a single persistent function, with explicit notify and wait calls on the Event Tensor to coordinate dependencies.

### 6.8 Reported results, verbatim

> We evaluate ETC on a diverse set of LLM serving workloads, comparing against highly competitive, industry-level baselines (e.g., vLLM and SGLang) that already employ aggressive optimizations such as CUDA Graphs, Programmatic Dependent Launch (PDL), and torch.compile. Our results show that ETC achieves substantial speedups over these systems while efficiently supporting both shape- and data-dependent dynamism, without requiring runtime graph recapture or recompilation. For tensor-parallel workloads, our compiler-driven overlap of computation and communication achieves up to 1.40x speedup on fused GEMM and Reduce-Scatter kernels. For data-dependent workloads such as MoE, our megakernels outperform specialized libraries by up to 1.23x. In dynamic-shape, low-batch inference scenarios, ETC matches or exceeds the performance of these highly optimized inference systems, while reducing engine warm-up overheads by up to 3.5x. Achieving even moderate speedups over such strong baselines translates to substantial economic value at datacenter scale. Beyond raw speed, ETC achieves true ahead-of-time (AOT) compilation for dynamic workloads, completely eliminating runtime compilation ...

> By unifying previously manual fusion and hand-crafted scheduling techniques (Spector et al., 2025; Cheng et al., 2025) into compiler transformations, ETC significantly reduces engineering effort to construct megakernels, while improving their runtime performance.

## Relevance

§6.6 is the design we should copy for MoE routing in a persistent kernel, because it is the
only published mechanism that lets a *statically scheduled* megakernel handle a runtime
top-k: initialize each expert's event counter to the token count routed to it (computed with
topk), and use an `exp_indptr` prefix-sum tensor so expert `i` triggers tiles in
`[exp_indptr[i], exp_indptr[i+1])`. At batch=1 with top-8 of 256, only 8 experts are live and
every count is 1, so this collapses to something very cheap — but it still requires that the
dependency structure be expressible without recompiling, which the CANN SuperKernel task queue
(a flat `TaskInfo` array built on the host) currently is not. §6.7's `fused_matmul_rs` is the
template for turning our 5.35 ms serial TP reduce into tiles interleaved with the producing
matmul's tiles in one persistent kernel; `multimem_ld_reduce_cta` is the NVLS primitive, and
the Ascend question is what plays that role (see `low-latency-tp-reduce.md`). §6.2's "each
kernel launch typically incurs 5–10 μs of latency, while the fastest kernels may complete in
2 μs" is the same ratio we measured with far worse constants: our MIX_AIC floor is 13.58 us.

---

## 7. Comparison table — what each system does, for picking our design

Assembled from the quotes above. Every cell is sourced from the corresponding section.

| | Hazy Megakernel (§1) | ThunderMLA (§5) | MPK (§2) | FlashMoE (§3) | ETC / Event Tensor (§6) | CANN SuperKernel (§4) |
| --- | --- | --- | --- | --- | --- | --- |
| Hardware | H100 / B200 | H100 | A100/H100/B200 | 8x H100 | NVIDIA | Ascend (Ascend910_9391; DAV_2201, DAV_3510) |
| Scope | whole Llama-1B fwd | MLA decode only | whole model, multi-GPU | one distributed MoE layer | whole model, multi-GPU | arbitrary AscendC subgraph |
| Work unit | instruction (7 types) | opcode + instruction tensor | task + event (ttGraph) | tile + task descriptor | task + Event Tensor element | `TaskInfo` in a `TaskQue` |
| Scheduling | AOT in Python, reused | AOT, static heap or makespan-backwards | hybrid JIT (scheduler warps) / AOT (round-robin) | fully dynamic, 1 scheduler warp | static + dynamic transforms, symbolic shapes | AOT on host, serial |
| Dependency mechanism | GM counters, incremented on completion | global tensor as semaphore | device-memory semaphores, atomicAdd queues | shared mem (sched↔sub), GM (sched↔proc), NVSHMEM signal | Event Tensor, `notify()`/`wait()`, einsum edges | `CrossCoreSetFlag/WaitFlag` + GM `SkMemoryWaitFlag` spin w/ `dcci` |
| On-chip buffer mgmt | 13 x 16 KiB pages of 213 kB | — | pages, 32 KB each (5/7/7 per SM) | tiles (128, 64), 128 thr/block | — | not managed; sub-kernels own UB/L1 |
| Cross-stage prefetch | interpreter hands freed pages to next instruction | async instruction fetch, deep pipelines | pre-loading phase + compute phase | — | — | `TYPE_PRELOAD` + `preload()`/`dc_preload()` |
| Collectives in-kernel | no | no | yes, `nvshmem_signal_wait_until` | yes, one-sided NVSHMEM (R)DMA | yes, `multimem_ld_reduce_cta` | `TYPE_EVENT_NOTIFY/WAIT` (AICPU/SDMA, not inter-device) |
| Data-dependent routing | no | variable seq len via scheduler | hybrid MoE balancer w/ meta-tensor | expert capacity C, in-place padding | `topk` event init + `exp_indptr` triggering | no |
| Headline number | 78% of H100 BW; <1 ms H100, <680 us B200 | 20–35% over FlashMLA; 41 us vs 52 us | up to 1.7x; 14.5 → 12.5 ms/token Qwen3-8B A100 | 1 vs 550 GPU ops; 93.17% vs 14% util | 1.40x GEMM+RS; 1.23x MoE; 3.5x less warmup | "调用收益可达15%以上" |

---

## 4b. CANN SuperKernel — what it will and will not fuse (host-side source)

Source files, all under `https://raw.gitcode.com/cann/graph-autofusion/raw/master/super_kernel/src/aot/`
Fetched 2026-10-20. This closes the "what gets fused" question left open in §4.

### 4b.1 `FusionFailReason` — the complete list of reasons a node cannot join a SuperKernel

`sk_node.h`:

```cpp
// Unfused reason
enum class FusionFailReason {
  CAN_FUSE,                // 0: Can fuse (default)
  OP_UNSUPPORT,            // 1: Failed to resolve SuperKernel bind map for the operator
  DYNAMIC_TASK_UNSUPPORT,  // 2: Operator dynamically refreshes task info at runtime, SK does not support fusing
                           // dynamically changing tasks
  NOT_IN_SCOPE,            // 3: Operator is not within user-marked fusion range
  IN_UNFUSIBLE_SCOPE,      // 4: User actively marked this operator as unfusible
  EXCEED_CORE_MAX,         // 5: Operator requires more cores than device maximum
  RESET_TYPE_NODE,         // 6: reset type node placed at end
  ISOLATED_EVENT,          // 7: Isolated event exists
  EXIST_DEADLOCK,          // 8: Deadlock exists
  SCOPE_FUSE_PART,         // 9: Scope processing skipped part of the original scope
  EXTERNAL_DEPEND,         // 10: Event has external dependency
  UNSUPPORT_EVENT_TYPE,    // 11: Unsupported event type
  MEMORY_WAIT_NODE_ONLY,   // 12: No memory write exists, meaning the memory write is outside modelRI,
  MEMORY_WRITE_NODE_ONLY,  // 13: only exists memory write nodes, mask it as unfusible
  DEFAULT_NODE,            // 14: default node uses aicpu resources, mask it as unfusible
  KERNEL_ATTR_GET_FAILED,  // 15: Failed to get kernel attribute for SuperKernel fusion
  EXCEED_SCOPE_MAX,        // 16: Exceeded maximum scope number limit for SuperKernel fusion
};

// Bindmap related fail reason detail
enum class BindmapFailReason : uint8_t {
  NONE,                      // 0: No error
  BINDMAP_INIT_EMPTY,        // 1: Bindmap empty after initialization
  BINHDL_NULL,               // 2: binHdl is null
  FUNCHDL_NULL,              // 3: funcHdl is null
  FUNC_NOT_FOUND,            // 4: Original function is not found in bind map
  BIN_DEV_ADDR_GET_FAILED,   // 5: Failed to get binary device address
  FUNC_ADDR_GET_FAILED,      // 6: Failed to get function address
  BINDMAP_ENTRY_CONFLICT,    // 7: Duplicated bind map entry has inconsistent value
```

### 4b.2 `ScopeBreakReason` — why a fused scope is cut

`sk_scope_info.h`:

```cpp
/*!
 * \brief Reasons why a scope boundary was created (scope was split)
 */
enum class ScopeBreakReason : uint8_t {
  NONE,
  UNFUSIBLE_NODE,
  DEADLOCK_DETECTED,
  SYNCALL_OP_DROP,
  DEBUG_PER_OP_MAX_CORE,
};
```

### 4b.3 The stream walk that terminates a scope

`sk_scope_split.cpp`:

```cpp
  // Check scopeBitFlags match
  if (node->GetScopeBitFlags() != currentScopeBitFlags_) {
    SK_LOGI("Stream %u: node %s scopeBitFlags mismatch, terminating stream", streamIdx, node->Format().c_str());
    state.isTerminated = true;
    return;
  }

  // Check fusibility
  if (!node->IsFusible()) {
    SK_LOGI("Stream %u: node %s is not fusible, terminating stream", streamIdx, node->Format().c_str());
    state.isTerminated = true;

    ScopeBreakInfo breakInfo;
    breakInfo.SetReason(ScopeBreakReason::UNFUSIBLE_NODE)
        .SetTriggerNode(node->GetNodeId(), streamIdx)
        .SetDetail("unfused node causes scope break");
    scopeSplitResultReporter_->ReportNewBreak(*currentScope_, std::move(breakInfo));
```

### 4b.4 How the scope's kernel type and block count are chosen

`sk_scope_split.cpp`, `ScopeCoreInfoCalculator`. Note the `(minVectorNum + 1) >> 1` — in 1:2
mode the block count is the number of *AIC*, so the AIV requirement is halved and rounded up.
Note also that a single `MIX_AIC_1_2` node forces `requireMix12` for the whole scope, which is
the mechanism behind the "Mix1:1 → Mix1:2 自适应" note in §4.9.

```cpp
bool ScopeCoreInfoCalculator::UpdateScopeCoreConstraints(const SuperKernelBaseNode &node,
                                                         ScopeCoreConstraints &constraints) {
  constraints.minCubeNum = std::max(constraints.minCubeNum, node.GetCubeNum());
  constraints.minVectorNum = std::max(constraints.minVectorNum, node.GetVecNum());
  constraints.requireMix12 = constraints.requireMix12 || node.GetKernelType() == SkKernelType::MIX_AIC_1_2;

  if (!node.RequiresExactCoreMatch()) {
    return true;
  }

  const bool vectorOnly = node.GetCubeNum() == 0 && node.GetVecNum() != 0;
  auto &exactCoreNum = vectorOnly ? constraints.exactVectorNum : constraints.exactCubeNum;
  const uint32_t taskCoreNum = vectorOnly ? node.GetVecNum() : node.GetCubeNum();
  if (exactCoreNum.has_value() && exactCoreNum.value() != taskCoreNum) {
    return false;
  }
  exactCoreNum = taskCoreNum;
  return true;
}
```

```cpp
  const auto getCandidates = [](uint32_t minCubeNum, uint32_t minVectorNum, bool requireMix12) {
    if (minCubeNum == 0 && minVectorNum == 0) {
      return std::vector<ScopeCoreInfo>{{}};
    } else if (requireMix12) {
      const uint32_t vectorBlocks = static_cast<uint32_t>((static_cast<uint64_t>(minVectorNum) + 1U) >> 1U);
      return std::vector<ScopeCoreInfo>{{SkKernelType::MIX_AIC_1_2, std::max(minCubeNum, vectorBlocks)}};
    } else if (minCubeNum == 0 && minVectorNum != 0) {
      return std::vector<ScopeCoreInfo>{{SkKernelType::AIV_ONLY, minVectorNum}};
    } else if (minCubeNum != 0 && minVectorNum == 0) {
      return std::vector<ScopeCoreInfo>{{SkKernelType::AIC_ONLY, minCubeNum}};
    } else {
      const uint32_t vectorBlocks = static_cast<uint32_t>((static_cast<uint64_t>(minVectorNum) + 1U) >> 1U);
      const ScopeCoreInfo mix12{SkKernelType::MIX_AIC_1_2, std::max(minCubeNum, vectorBlocks)};
      const ScopeCoreInfo mix11{SkKernelType::MIX_AIC_1_1, std::max(minCubeNum, minVectorNum)};
      return minVectorNum <= minCubeNum ? std::vector<ScopeCoreInfo>{mix11, mix12}
                                        : std::vector<ScopeCoreInfo>{mix12, mix11};
    }
  };
```

```cpp
ScopeCoreInfo BuildDebugPerOpScopeCoreInfo(const SuperKernelBaseNode &node, int64_t maxDeviceCubeNum,
                                           int64_t maxDeviceVectorNum) {
  if (node.RequiresExactCoreMatch()) {
    if (node.GetCubeNum() == 0 && node.GetVecNum() != 0 && (node.GetVecNum() & 1U) != 0U) {
      return {SkKernelType::AIV_ONLY, node.GetVecNum()};
    } else {
      const uint32_t numBlocks = node.GetCubeNum() == 0 ? node.GetVecNum() / 2U : node.GetCubeNum();
      return {SkKernelType::MIX_AIC_1_2, numBlocks};
    }
  } else {
    const int64_t numBlocks = std::min(maxDeviceCubeNum, maxDeviceVectorNum / 2);
    return {SkKernelType::MIX_AIC_1_2, static_cast<uint32_t>(numBlocks)};
  }
}
```

Other scope-splitting passes present in the same file, by the break reason they set:
`DEADLOCK_DETECTED` (a `[DeadlockRefine]` pass that splits a scope into `scopeBefore` /
`scopeAfter` around the deadlocking node and calls `SetFusionFailReason(FusionFailReason::EXIST_DEADLOCK)`),
and `SYNCALL_OP_DROP` (a `[ScheModeSplit]` pass). `ISOLATED_EVENT` is set directly during the
initial walk.

## Relevance

This is the acceptance test for adopting SuperKernel as our substrate. The two entries that
decide it for us are `DYNAMIC_TASK_UNSUPPORT` ("Operator dynamically refreshes task info at
runtime, SK does not support fusing dynamically changing tasks") and `DEFAULT_NODE` ("default
node uses aicpu resources"). Our MoE routing produces exactly a runtime-refreshed task set
(top-8 of 256 chosen per token), so a routed-expert GroupedMatmul is a candidate for
`DYNAMIC_TASK_UNSUPPORT` and would break the scope there — which is precisely the gap Event
Tensor's `exp_indptr` triggering (§6.6) was designed to fill and SuperKernel has no answer
for. `EXCEED_CORE_MAX` and the `requireMix12` propagation also matter: one MIX_AIC_1_2 node
anywhere in the scope forces the whole SuperKernel to 1:2, and with 505 MIX_AIC calls a round
that will happen. The practical next step is to run `aclskOptimize` over one decode round and
read the `PrintScopeBreakReasonReport()` output — it will tell us exactly how many scopes our
3008 kernels collapse into and why each boundary exists, which is a measurement we can get
before writing any kernel code.


---

## 8. MegaGDN — a real, working, public Ascend megakernel (and its grid barrier)

Repo: https://github.com/huawei-csl/megagdn-pto (fork: https://github.com/vloncar/megagdn-pto)
Report: `blog/mega_gdn_en.md` — "MegaGDN: One fused kernel cuts Qwen3.5/3.6 inference TTFT by 15%
(accelerate vLLM-Ascend by PTO-ISA)"
Dates: repo created 2026-06-22, last push 2026-10-13; presented on the CANN official account
2026-05-19; merged into the SGLang NPU ecosystem 2026-05-20
(https://github.com/sgl-project/sgl-kernel-npu/pull/462)
Built on PTO-ISA: https://gitcode.com/cann/pto-isa

This is the artifact that matters most in this file: a short, readable, **production** Ascend
megakernel that fuses seven stages into one `<<< >>>` launch, with a working grid-wide barrier
for both A2/A3 and A5. It is not a design document.

### 8.1 What it claims

> **TL;DR** This repo provides **1.5~3x faster**<sup>[1]</sup> NPU kernels for chunk GDN<sup>[2]</sup>, with integration to [vllm-ascend](https://github.com/vllm-project/vllm-ascend) to enable **~15% faster prefill** for Qwen3.5/3.6 models<sup>[3]</sup>.
>
> - [1] Compared to the triton kernels used [in vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/v0.19.1rc1/vllm_ascend/ops/triton/fla)/[sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu/tree/2026.05.01/python/sgl_kernel_npu/sgl_kernel_npu/fla). See [Performance highlights](#performance-highlights).
> - [2] All 6 stages: `chunk_cumsum`, `scaled_dot_kkt`, `solve_tril`, `wy_fast`, `chunk_h`, `chunk_o`. Plus a merged metakernel to save kernel launch overhead.
> - [3] Tested model sizes: `0.8B`, `2B`, `4B`, `9B`, `27B`, `35B-A3B`. (`122B-A10B`, `397B-A17B` are also compatible in principle, yet to test)

> CANN 8.5.0~9.0.0 are verified on A2/A3; the experimental Ascend A5 support requires CANN >= 9.1.0.

### 8.2 Why they wrote a megakernel instead of using ACL Graph — verbatim

> ## Host overhead in eager PyTorch
>
> Kernel-only optimizations alone are not enough to speed up end-to-end inference on Atlas A2/A3 systems -- those machines are notoriously easy to become host-bound when running PyTorch eager mode. Profiling vLLM prefill (original Triton backend) shows large bubbles on the device side, due to host-side Python overhead and kernel launch overhead.
>
> The standard mitigation is [ACL Graph](https://docs.vllm.ai/projects/ascend/en/v0.18.0/developer_guide/Design_Documents/ACL_Graph.html), the NPU analogue of CUDA Graph. vLLM-Ascend's graph path applies to **`decode_only`**; prefill still runs in eager Torch. See [ACL graph limitations](https://docs.vllm.ai/projects/ascend/en/v0.18.0/developer_guide/Design_Documents/ACL_Graph.html#limitations). Prefill tensor shapes change often, but capturing and replaying graphs prefer fixed static shapes. Supporting dynamic-shape graphs means padding, bucketing, and other workarounds.

### 8.3 The fusion method — "simpler but effective"

> ## Simpler way for megakernel fusion
>
> To not struggle with graph mode workarounds, we take a simpler but effective approach: we have the source code for all 6 computation stages, each only a few hundred lines of C++. We just merged them into one NPU kernel function, launched once with the familiar `<<< >>>` syntax.
>
> Between the stages, we must fence across all cores for memory ordering. We use `TSyncAll` that is [just added to PTO-ISA](https://gitcode.com/cann/pto-isa/pull/878).
>
> The fused source code is at path `megagdn-pto/kernels/pto/mega_kernel.cpp`. The megakernel source stays short and readable. The `launch_mega_kernel` kernel entry is a few hundred lines of C++. Stages reuse the single-kernel sources without copy-paste duplication. **This is legal because one AICORE function can call another AICORE function, similar to CUDA `__forceinline__ __device__`.**

That last sentence is the licence for the whole approach on Ascend.

### 8.4 THE GRID BARRIER — `SyncAllMegaKernel`, working code for A2/A3 and A5

Source: `kernels/pto/include/kernel_utils.h` (533-line `mega_kernel.cpp` uses it between every
stage). This is the single most reusable piece of code found in this entire research pass.

```cpp
AICORE inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId) {
  constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
  constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;
  return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) +
          ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

template <bool isAIVOnly = true>
AICORE inline void SyncAllA2A3() {
// These constants are defined on A5
#if __CCE_AICORE__ == 220
  constexpr uint16_t SYNC_AIV_FLAG = 12;
  constexpr uint16_t SYNC_AIC_FLAG = 11;
  constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
  constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
  pipe_barrier(PIPE_ALL);
  if constexpr (isAIVOnly) {
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
    wait_flag_dev(SYNC_AIV_ONLY_ALL);
    return;
  }
#if defined(__DAV_CUBE__)
  wait_flag_dev(SYNC_AIV_FLAG);
  ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
  wait_flag_dev(SYNC_AIC_FLAG);
  ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_VEC__)
  ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
  wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
#endif
}

// ── A5 ── mirrors include/pto/npu/a5/SyncAll.hpp:101-111
// Requires: getFFTSMsg from <pto/npu/a5/SyncAll.hpp> (A5 defines its own)
AICORE inline void SyncAllMixA5() {
#if __CCE_AICORE__ != 220
  pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
  // 1. Wait for both paired AIVs: one flag ID per subblock (base, base +
  // SYNC_FLAG_ID_MAX).
  wait_intra_block(PIPE_S, SYNC_AIV_FLAG);
  wait_intra_block(PIPE_S, SYNC_AIV_FLAG + SYNC_FLAG_ID_MAX);
  // 2. Global all-AIC barrier.
  ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));
  wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
  // 3. Release both paired AIVs.
  set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
  set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG + SYNC_FLAG_ID_MAX);
#elif defined(__DAV_VEC__)
  // Set on MTE3 so prior stores drain; wait on PIPE_S.
  set_intra_block(PIPE_MTE3, SYNC_AIV_FLAG);
  wait_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
#endif
#endif
}

template <bool isAIVOnly = true>
AICORE inline void SyncAllMegaKernel() {
#if __CCE_AICORE__ == 220
  SyncAllA2A3<isAIVOnly>();
#else
  static_assert(isAIVOnly == false, "No support for AIV only global sync");
  SyncAllMixA5();
#endif
}
```

```cpp
/**
 * @brief Pipe in-core barrier for vector core that is a no-op for A5.
 *
 */
AICORE inline void PipeBarrierVec() {
#if __CCE_AICORE__ == 220
  pipe_barrier(PIPE_V);
#endif
}
```

### 8.5 The megakernel body — stage list, barriers, early-exit instrumentation

`kernels/pto/mega_kernel.cpp` header comment, verbatim:

```cpp
// mega_kernel.cpp — GDN Mega-Kernel (group-value / GQA): all PTO stages in one
// launch
//
// Same pipeline as pto_mega_kernel, but scaled_dot_kkt / wy_fast / chunk_h /
// chunk_o use runtime H/Hg dispatch from dynamic_bsnd_groupvalue; cumsum still
// uses H (value heads) like dynamic_bsnd.
//
// Stages:
//   1. cumsum      (Vec)
//   2. transpose   (Vec)
//   3. kkt         (Cube+Vec)  — K has Hg heads; β,g,A use H value heads
//   4. solve_tril  (Cube)
//   5. wy_fast     (Vec+Cube)
//   6. chunk_h     (Cube+Vec)
//   7. chunk_o     (Cube+Vec)
```

Compile-time shape parameters, with the explicit "one .so serves every head count" note — the
exact static/dynamic tradeoff we face:

```cpp
#ifndef GDN_D
#define GDN_D 128
#endif
#ifndef GDN_C
#define GDN_C 128
#endif
// GDN_MAX_HEADS: compile-time ceiling on the value-head count. num_heads is a
// RUNTIME argument (one .so serves every head count), so the transpose/cumsum
// UB tiles are sized for this worst case; any num_heads <= GDN_MAX_HEADS works
// with the unused columns zero-padded. Must match the host-side _MAX_HEADS
// guard and the GDN_MAX_HEADS in chunk_cumsum.cpp.
#ifndef GDN_MAX_HEADS
#define GDN_MAX_HEADS 64
#endif

#include <runtime/rt_ffts.h>

#include <pto/pto-inst.hpp>
#include <type_traits>

#include "acl/acl.h"
#include "kernel_utils.h"

using namespace pto;
using namespace kernel_utils;
```

The body: note `set_ffts_base_addr(ffts_addr)` first, then stage calls separated by
`SyncAllMegaKernel<false>()`, with `MEGA_STOP_AFTER_*` macros for per-stage bisecting.

```cpp
AICORE inline void mega_kernel_impl(
    __gm__ uint8_t* q_ptr, __gm__ uint8_t* k_ptr, __gm__ uint8_t* v_ptr,
    ...
    int64_t total_tokens, uint32_t num_matrices, uint64_t ffts_addr) {
  set_ffts_base_addr(ffts_addr);

  constexpr int32_t D = GDN_D;
  constexpr int32_t C = GDN_C;

  if (num_key_heads == 0 || (static_cast<uint32_t>(H) % num_key_heads) != 0) {
    return;
  }

  mk_cumsum::cumsum_kernel<C>(reinterpret_cast<__gm__ float*>(g_in_ptr),
                              reinterpret_cast<__gm__ float*>(g_sum_ptr),
                              reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr),
                              batch_size, seq_len, H, ffts_addr);

#ifdef MEGA_STOP_AFTER_CUMSUM
  pipe_barrier(PIPE_ALL);
  return;
#endif

  SyncAllMegaKernel<false>();

#ifdef MEGA_STOP_AFTER_SYNC1
  return;
#endif

  mega_transpose_TH_to_HT<float>(...);
  mega_transpose_TH_to_HT<half>(...);

#ifdef MEGA_STOP_AFTER_TRANSPOSE
  pipe_barrier(PIPE_ALL);
  return;
#endif

  SyncAllMegaKernel<false>();

  mk_kkt::kkt_kernel<D, C>(..., ffts_addr);
```

The AIC/AIV handshake drain between stages — note the explicit A2-vs-A5 divergence comment:

```cpp
// Drain the kkt handshake: Vec released both workspace slots (flags 2/3) one
// last time after Cube's final iteration, so consume them before the next
// stage. A2: Vec is a separate core → FFTS cross-core flag. A5: both Vec
// sub-blocks share Cube's core → each signals its own intra-block
//     flag (base, base + 16).
#if defined(__DAV_CUBE__)
  pipe_barrier(PIPE_ALL);
#if __CCE_AICORE__ == 220
  wait_flag_dev(2);
  wait_flag_dev(3);
#else
  WaitBothVecOnA5<PIPE_MTE2>(2);
  WaitBothVecOnA5<PIPE_MTE2>(3);
  pipe_barrier(PIPE_ALL);
#endif
#endif
```

### 8.6 Entry point and host launch

```cpp
extern "C" __global__ AICORE void launch_mega_kernel(
    __gm__ uint8_t* q_ptr, __gm__ uint8_t* k_ptr, __gm__ uint8_t* v_ptr,
    ...
    __gm__ uint8_t* kkt_ws_ptr,
    __gm__ uint8_t* wy_ws_a1_ptr, __gm__ uint8_t* wy_ws_a2_ptr,
    __gm__ uint8_t* h_ws_ptr, __gm__ uint8_t* o_ws_qk_ptr,
    __gm__ uint8_t* o_ws_qs_ptr, __gm__ uint8_t* o_ws_gated_ptr,
    uint32_t num_heads, uint32_t num_key_heads, int64_t batch_size,
    int64_t seq_len, int64_t total_tokens, uint32_t num_matrices,
    uint64_t ffts_addr) {
  // num_heads is a runtime kernel argument (one .so serves every head count).
  // Guard the compile-time UB ceiling; the host validates before launch too.
  if (num_heads == 0 || num_heads > GDN_MAX_HEADS) {
    return;
  }
  mega_kernel_impl(...);
}

extern "C" void call_kernel(
    uint32_t block_dim, void* stream, uint8_t* q, uint8_t* k, uint8_t* v,
    ...
    int64_t total_tokens, uint32_t num_matrices) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_mega_kernel<<<block_dim, nullptr, stream>>>(
      q, k, v, ..., num_matrices, fftsAddr);
}
```

Note `rtGetC2cCtrlAddr(&fftsAddr, &fftsLen)` on the host and `set_ffts_base_addr(ffts_addr)`
on the device — the FFTS control address must be fetched and passed in; it is the prerequisite
for `ffts_cross_core_sync` working at all.

### 8.7 Other files in the same kernel directory

```
chunk_cumsum.cpp      24829
chunk_h.cpp           40744
chunk_h_kda.cpp       35107
chunk_o.cpp           66473
chunk_o_kda.cpp       37959
gate_cumsum_kda.cpp   15396
include/              (kernel_utils.h, 5189 bytes)
kkt_kda.cpp           21886
mega_kernel.cpp       19694   (533 lines)
mega_kernel_kda.cpp   14776
scaled_dot_kkt.cpp    37959
tri_inverse.cpp       43005
wy_fast.cpp           44739
wy_kda.cpp            46981
```

## Relevance

This refutes the assumption that an Ascend megakernel has to be built from scratch, and it
does so with the smallest possible design: no interpreter, no task queue, no scheduler — just
`__aicore__` functions called in sequence from one `__global__` entry, with
`SyncAllMegaKernel<false>()` between stages. §8.3's "one AICORE function can call another
AICORE function" is the enabling fact, and §8.4 is the barrier we would otherwise have had to
derive. Against our 3008 kernels/round, this pattern collapses a chain of stages at the cost
of one full-core barrier per stage boundary — which is exactly the structure CANN SuperKernel
imposes (§4.9, 串行调度) but written by hand and therefore tunable: we can drop barriers where
a finer flag suffices, which SuperKernel cannot.

Two caveats before copying. First, MegaGDN targets **prefill**, where stages are large and one
barrier per boundary is cheap; at batch=1 decode our stages are tiny and the barrier cost is
proportionally much larger, so a naive stage-by-stage port reproduces the serialization that
Hazy Research §1.6 and MPK §5.1 specifically avoid with per-dependency counters. Second, the
`GDN_MAX_HEADS` pattern — compile-time ceiling, runtime count, zero-padded columns, "one .so
serves every head count" — is the opposite of our situation: we are allowed to specialize for
batch=1 and should burn the shapes in rather than pad.

## 9. `aclshmem` / CANN SHMEM — the official in-kernel cross-rank communication library (`gitcode.com/cann/shmem`)

Repo: https://gitcode.com/cann/shmem (GitCode, org `cann`)
Docs site: https://shmem-doc.pages.dev/
SIG: https://gitcode.com/cann/community/tree/master/CANN/sigs/shmem
Mailing list: https://mailweb.cann.osinfra.cn/mailman3/lists/shmem.cann.osinfra.cn/
Releases: v1.7.0 (2026/10), v1.6.0 (2026/07), v1.3.0 (2026/04), first online 2025/12
License: CANN Open Software License Agreement Version 2.0
Contributor credit: 华南理工大学 陆璐教授团队 (SCUT, Prof. Lu Lu's group)
Raw file base used below: `https://raw.gitcode.com/cann/shmem/raw/master/<path>`
Retrieved 2026-10-20.

This is the answer to "is there a one-sided all-reduce issuable from device code on Ascend".
There is, it is first-party, it is open source under the `cann` org, and it is an OpenSHMEM
dialect. Name: SHMEM / `aclshmem`. The README calls it 「基于对称内存的昇腾分布式内存通信加速库」.

### 9.1 README — what it is, and the AICore-direct-drive claim

`README.md`, verbatim:

> ## 最新动态
>
> 🚀 [2026/10] [SHMEM v1.7.0](https://gitcode.com/cann/shmem/releases/v1.7.0) 正式发布。
>
>    - Ascend 950PR&950DT系列产品 RDMA/UDMA 新增 QP-specific put/get 接口，扩展聚合提交能力，支持 UDMA 多链路并发。
>    - A2/A3 SDMA 新增 QP-specific put/get 接口，支持多 QP 通信。
>    - 完善安装与运行时诊断，新增 MegaMoE、UDMA AllReduce 示例及多 QP 性能测试。
>
> 📌 [2026/07] [SHMEM v1.6.0](https://gitcode.com/cann/shmem/releases/v1.6.0) 发布：完善 Ascend 950PR&950DT系列产品 RDMA、UDMA/MTE 与 SIMT 能力，改进 Python wheel 安装与 `shmem-config` 诊断。
>
> 📌 [2026/04] [SHMEM v1.3.0](https://gitcode.com/cann/shmem/releases/v1.3.0) 发布：新增 AICore 直驱能力和 40+ 通信接口，并完善 DFX 工具链。
>
> 🔥 [2025/12] SHMEM 项目首次上线。

> ## 一、项目简介
>
> SHMEM 是面向昇腾平台的多机多卡内存通信库，通过封装 Host 侧与 Device 侧接口，实现跨设备的高效内存访问与数据同步。其核心价值在于：
>
> - 支持 AICore 直驱 MTE、xDMA 使能 D2D/D2H/H2D/D2rH/rH2D 通信
> - 简化分布式场景下的卡间通信逻辑，降低算子开发门槛
> - 与 CANN 生态深度适配，支持通算融合类算子的快速部署

> ## 二、核心功能
>
> **1. 双侧接口体系**
>
> - Host 侧：负责初始化、内存堆管理、通信域（Team）创建及全局同步
> - Device 侧：提供远程内存访问（RMA）、设备级同步及通信域操作
>
> 接口设计贴合昇腾算子开发范式，支持 Host 与 Device 协同工作流。
>
> **2. 高性能通信优化**
>
> - 内置 MTE、xDMA 引擎支持，实现远程内存直接读写，减少数据传输延迟
> - 兼容 MPI 通信框架，支持集合通信（Allgather/Allreduce 等）场景
> - 针对昇腾硬件特性优化数据传输路径，提升多卡协同效率

> **4. 通信通路覆盖**
>
> 下图展示了 SHMEM 支持的全链路通信通路（以910A3为例），覆盖 Host 侧与 Device 侧的不同传输引擎：
>
> 如图所示，SHMEM 支持多种通信引擎和通路：
>
> - **MTE Engine**：芯片级内存传输引擎，支持 D2D、D2H、H2D、D2rH、rH2D 多种通路
> - **xDMA Engine**：高速直接内存访问引擎，支撑主机内/主机间高效数据传输

Note the figure caption: 「以910A3为例」 — the full-link communication-path diagram is drawn
*for 910 A3*, which is our part.

> ## 四、典型使用场景
>
> **1. 通算融合类算子开发**：基于 Device 侧内存直接访问接口，开发融合「计算+通信」的自定义算子（如 matmul+allreduce），减少卡间数据拷贝，提升算子执行效率。

The include tree (verbatim from the README's 代码结构 block):

```text
├── include/                            # 对外头文件
│   ├── shmem.h                         # SHMEM 所有对外 API 汇总
│   ├── device/                         # Device 侧头文件
│   │   ├── gm2gm/                      # AICore 驱动 gm2gm 数据面接口
│   │   │   └── engine/                 # AICore 直驱 gm2gm 低阶接口
│   │   ├── team/                       # Device 侧通信域管理
│   │   └── ub2gm/                      # AICore 驱动 ub2gm 数据面接口
│   │       └── engine/                 # AICore 直驱 ub2gm 低阶接口
```

The actual header list (walked via the GitCode contents API,
`https://api.gitcode.com/api/v5/repos/cann/shmem/contents/include/...`):

```
include/shmem.h
include/device/shmem_def.h
include/device/gm2gm/shmem_device_rma.h          45447 B
include/device/gm2gm/shmem_device_amo.h          42372 B
include/device/gm2gm/shmem_device_p2p_sync.h     35212 B
include/device/gm2gm/shmem_device_so.h           26487 B
include/device/gm2gm/shmem_device_cc.h            6013 B
include/device/gm2gm/shmem_device_mo.h            2476 B
include/device/gm2gm/engine/shmem_device_mte.h
include/device/gm2gm/engine/shmem_device_sdma.h
include/device/gm2gm/engine/shmem_device_rdma.h
include/device/gm2gm/engine/shmem_device_udma.h
include/device/ub2gm/shmem_device_rma.h          12047 B
include/device/ub2gm/engine/shmem_device_mte.h
include/device/team/shmem_device_team.h
include/device_simt/gm2gm/shmem_device_simt_rma.h
include/device_simt/gm2gm/shmem_device_simt_amo.h
include/device_simt/gm2gm/engine/shmem_device_simt_mte.h
include/device_simt/ub2gm/shmem_device_simt_rma.h
include/device_simt/ub2gm/engine/shmem_device_simt_mte.h
include/device_simt/team/shmem_device_simt_team.h
include/device_simt/shmem_simt_common_types.h
include/host/...
include/host_device/shmem_common_macros.h
include/host_device/shmem_common_types.h         17971 B
```

### 9.2 `include/device/gm2gm/shmem_device_cc.h` — THE DEVICE-SIDE COLLECTIVE CONTRACT

This file is short and is the single most important text in this section. Reproduced in full
minus the licence header. Note the first block: the all-core barrier **requires a MIX kernel**.

```cpp
// clang-format off
/*
 * Collective programming contract:
 * - The all-core barrier and sync APIs require a MIX kernel; all participating local AIC and AIV
 *   cores must enter. Vector-only sync APIs involve only AIV cores. The _all forms select
 *   ACLSHMEM_TEAM_WORLD without changing the local core participation scope.
 * - Every PE in a team must call its collectives in the same order. Do not overlap collective
 *   calls on the same team. ACLSHMEM_TEAM_INVALID (-1) is a no-op for team-taking APIs;
 *   other invalid handles are not covered by this rule.
 * - Ordinary symmetric-memory stores from participating cores are synchronized on supported
 *   HCCS/UB paths. The Cube scalar unit is outside this visibility guarantee.
 * - Device collectives cover Device-issued work; Host-issued work needs the corresponding
 *   Host-side completion and ordering.
 */
// clang-format on

/*!
 * \file shmem_device_cc.h
 * \brief shmem device Collective Communication APIs
 */
#ifndef _DEVICE_GM2GM_ACLSHMEM_DEVICE_CC_H_
#define _DEVICE_GM2GM_ACLSHMEM_DEVICE_CC_H_

#include "host_device/shmem_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config)
 * @brief Set runtime ffts address. Call this at MIX Kernel entry point (if the kernel contains barrier calls).
 *
 * @param config              [config] ffts config, acquired by util_get_ffts_config()
 */
ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config);
#define shmemx_set_ffts_config util_set_ffts_config

/**
 * @brief Complete prior communication and synchronize the PEs in a team.
 * @note All local cores in the declared participation scope must enter, and all team PEs
 *       must call collectives in the same order. On return, prior communication covered
 *       by this Device API is complete and its updates are visible to the destination PE.
 *       The source buffer of a completed transfer may then be reused. QP-specific transfers
 *       require their matching QP-specific quiet.
 *
 * @param[in] team Team to synchronize.
 * @return 0 after the collective (also for ACLSHMEM_TEAM_INVALID); Device execution errors
 *         are not reported through this return value.
 */
ACLSHMEM_DEVICE int aclshmem_barrier(aclshmem_team_t team);
#define shmem_barrier aclshmem_barrier

/**
 * @brief Perform aclshmem_barrier over ACLSHMEM_TEAM_WORLD with the same local core scope.
 */
ACLSHMEM_DEVICE void aclshmem_barrier_all(void);
#define shmem_barrier_all aclshmem_barrier_all

/**
 * @brief Collective synchronization over a team in which only vector cores participate. This routine does not
 * ensure
 *        completion of remote memory updates issued through ACLSHMEM communication APIs. Cube cores may call
 * the API
 *        but take no effect.
 *
 * @param team              [in] team to synchronize
 */
ACLSHMEM_DEVICE void aclshmemx_sync_vec(aclshmem_team_t team);

/**
 * @brief aclshmemx_sync_vec over ACLSHMEM_TEAM_WORLD.
 */
ACLSHMEM_DEVICE void aclshmemx_sync_vec_all(void);

/**
 * @brief Compatibility wrapper for aclshmemx_sync_vec.
 *
 * @param team              [in] team to synchronize
 */
[[deprecated("aclshmemx_barrier_vec is deprecated, please use aclshmemx_sync_vec instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_vec(aclshmem_team_t team);
#define shmemx_barrier_vec aclshmemx_barrier_vec

/**
 * @brief Compatibility wrapper for aclshmemx_sync_vec_all.
 */
[[deprecated("aclshmemx_barrier_all_vec is deprecated, please use aclshmemx_sync_vec_all instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_all_vec(void);
#define shmemx_barrier_all_vec aclshmemx_barrier_all_vec

/**
 * @brief Wait for all team PEs to reach this synchronization point.
 * @note On supported HCCS/UB paths, ordinary symmetric-memory stores from participating
 *       cores are synchronized within the applicable visibility scope. This call does not
 *       complete outstanding NBI communication. Call the matching quiet before sync when
 *       communication completion or source-buffer reuse is required, or use barrier instead.
 *       All cores in the declared participation scope must enter in the same order.
 *
 * @param[in] team Team to synchronize.
 */
ACLSHMEM_DEVICE void aclshmem_sync(aclshmem_team_t team);

ACLSHMEM_DEVICE int aclshmem_team_sync(aclshmem_team_t team);

/**
 * @brief Perform aclshmem_sync over ACLSHMEM_TEAM_WORLD with the same local core scope.
 */
ACLSHMEM_DEVICE void aclshmem_sync_all(void);

#ifdef __cplusplus
}
#endif

#include "gm2gm/shmemi_device_cc.h"

#endif // _DEVICE_GM2GM_ACLSHMEM_DEVICE_CC_H_
```

Four things in that file matter to us directly:

1. `util_set_ffts_config(uint64_t config)` — aliased `shmemx_set_ffts_config` — **"Call this at
   MIX Kernel entry point (if the kernel contains barrier calls)"**, and `util_get_ffts_config()`
   is the host-side getter. Same mechanism as MegaGDN's `rtGetC2cCtrlAddr` +
   `set_ffts_base_addr` (§8.6): the FFTS control address must be plumbed in as a kernel argument.
2. **"The all-core barrier and sync APIs require a MIX kernel; all participating local AIC and
   AIV cores must enter."** First-party confirmation that the device-side cross-rank barrier is
   not available to a non-MIX kernel.
3. `aclshmemx_sync_vec(team)` / `aclshmemx_sync_vec_all()` — **a vector-only cross-rank sync**.
   "Cube cores may call the API but take no effect." But: "This routine does not ensure
   completion of remote memory updates issued through ACLSHMEM communication APIs" — so it is
   a rendezvous, not a completion. You must `aclshmem_quiet()` first.
4. **"The Cube scalar unit is outside this visibility guarantee."** Ordinary symmetric-memory
   stores are synchronized on HCCS/UB paths, *except* those issued from the Cube scalar unit.

### 9.3 `include/device/gm2gm/shmem_device_mo.h` — memory ordering; `fence == quiet` on Ascend

Full body, verbatim:

```cpp
/**
 * @brief The aclshmem_quiet routine ensures completion of all operations on symmetric data objects issued by the
 *        calling PE.
 *        On systems with only scale-up network (HCCS), updates are globally visible, whereas on systems with
 *        both scale-up network HCCS and scale-out network (RDMA), ACLSHMEM only guarantees that updates to the
 *        memory of a given PE are visible to that PE.
 *        Quiet operations issued on the CPU and the NPU only complete communication operations that were
 *        issued from the CPU and the NPU, respectively. To ensure completion of NPU-side operations from
 *        the CPU, using aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 */
ACLSHMEM_DEVICE void aclshmem_quiet(void);
#define shmem_quiet aclshmem_quiet

/**
 * @brief In OpenACLSHMEM specification, aclshmem_fence assures ordering of delivery of Put, AMOs, and memory store routines
 *        to symmetric data objects, but does not guarantee the completion of these operations.
 *        However, due to hardware capabilities, we implemented aclshmem_fence same as aclshmem_quiet, ensuring both ordering
 *        and completion.
 *        Fence operations issued on the CPU and the NPU only order communication operations that were issued from the
 *        CPU and the NPU, respectively. To ensure completion of NPU-side operations from the CPU,
 *        using aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 */
ACLSHMEM_DEVICE void aclshmem_fence(void);
#define shmem_fence aclshmem_fence
```

Single-machine TP16 is HCCS-only, so the strong clause applies: **"On systems with only scale-up
network (HCCS), updates are globally visible."** There is no weaker visibility caveat for us.

### 9.4 `include/device/shmem_def.h` — the UB budget the library takes, and batched submit

```cpp
#include "kernel_operator.h"
#include "host_device/shmem_common_types.h"

constexpr uint64_t ACLSHMEM_INTERNAL_UB_BUF_START_ADDR = 188 * 1024;
constexpr uint32_t UB_ALIGN_SIZE = 32;
constexpr uint32_t UB_ALIGN_SIZE_64 = 64;
constexpr uint32_t ACLSHMEM_NUM_CQE_PER_POLL_CQ = 100;

/**
 * @struct non_contiguous_copy_param
 * @brief Non-Contiguous Datacopy Param.
 *
 * - uint32_t repeat: Data move times
 * - uint32_t length: Data move unit length
 * - uint32_t src_ld: Src data leading dimension. Interval between the head of the repeat and the
 *   head of the following repeat.
 * - uint32_t dst_ld: Dst data leading dimension.
 */
struct non_contiguous_copy_param {
    uint32_t repeat;
    uint32_t length;
    uint32_t src_ld;
    uint32_t dst_ld;
};

/**
 * @brief State shared by aclshmemx_defer_t and aclshmemx_submit_t actions in one active batch.
 *
 * Value-initialize the object as aclshmemx_submit_state_t state{} before the first operation.
 * The library maintains pending_count and resets it to zero after a successful submit; callers must not modify the
 * field directly. Use one state object for only one active batch, and keep it alive until every action that references
 * it is no longer used.
 *
 * @note If submit validation or completion polling fails, the calling device kernel is aborted
 *       and pending_count is not reset.
 * @note reserved is retained for ABI-compatible extension and must remain zero.
 */
struct aclshmemx_submit_state_t {
    uint32_t pending_count = 0; ///< Number of operations currently added to the batch.
    uint32_t reserved = 0;      ///< Reserved for future extension; callers must keep it zero.
};

/**
 * @brief Action that adds the current operation to a batch without submitting the batch.
 */
struct aclshmemx_defer_t {
    aclshmemx_submit_state_t& state; ///< Submit state for the active batch.
    ACLSHMEM_DEVICE explicit aclshmemx_defer_t(aclshmemx_submit_state_t& st) : state(st) {}
};

/**
 * @brief Action that adds the current operation and submits all operations in the batch.
 */
struct aclshmemx_submit_t {
    aclshmemx_submit_state_t& state; ///< Submit state for the active batch.
    ACLSHMEM_DEVICE explicit aclshmemx_submit_t(aclshmemx_submit_state_t& st) : state(st) {}
};
```

`ACLSHMEM_INTERNAL_UB_BUF_START_ADDR = 188 * 1024` is a hard budget line: on a 192 KB UB the
library reserves the top 4 KB for itself. Any megakernel that links `aclshmem` loses that.

### 9.5 `include/device/gm2gm/shmem_device_p2p_sync.h` — the one-sided signal/wait pair

```cpp
/**
 * @brief The aclshmemx_signal_op operation updates sig_addr with signal using operation sig_op on the specified PE.
 *        This operation can be used together with aclshmem_signal_wait_until for efficient
 *        point-to-point synchronization.
 *        On Ascend950, RDMA is used as the communication engine, supporting signal_set with guaranteed atomicity.
 *
 * @param sig_addr              [in] Symmetric address of the signal word to be updated.
 * @param signal                [in] The value used to update sig_addr.
 * @param sig_op                [in] Operation used to update sig_addr with signal. Supported operations:
 *                                   ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                    [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmemx_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);
#define shmemx_signal_op aclshmemx_signal_op

/**
 * @brief This routine can be used to implement point-to-point synchronization between PEs or between threads within
 *        the same PE. A call to this routine blocks until the value of sig_addr at the calling PE satisfies the wait
 *        condition specified by the comparison operator, cmp, and comparison value, cmp_val.
 *
 * @param sig_addr              [in] Local address of the source signal variable.
 * @param cmp                   [in] The comparison operator that compares sig_addr with cmp_val. Supported operators:
 *                                   ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
 *                                   ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * @param cmp_val               [in] The value against which the object pointed to by sig_addr will be compared.
 * @return Return the contents of the signal data object, sig_addr, at the calling PE that satisfies the wait condition.
 */
ACLSHMEM_DEVICE int32_t aclshmem_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val);
#define shmem_signal_wait_until aclshmem_signal_wait_until
```

The same header also machine-generates, for every type in `ACLSHMEM_TYPE_FUNC`
(`half float double int8 int16 int32 int64 uint8 uint16 uint32 uint64 char bfloat16`):

```cpp
ACLSHMEM_DEVICE void   aclshmem_##NAME##_wait_until(__gm__ TYPE *ivar, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE void   aclshmem_##NAME##_wait(__gm__ TYPE *ivar, TYPE cmp_value)
ACLSHMEM_DEVICE void   aclshmem_##NAME##_wait_until_all(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_any(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_some(__gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE void   aclshmem_##NAME##_wait_until_all_vector(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_any_vector(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_wait_until_some_vector(__gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values)
ACLSHMEM_DEVICE int    aclshmem_##NAME##_test(__gm__ TYPE *ivars, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_any(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_some(__gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp, TYPE cmp_value)
ACLSHMEM_DEVICE int    aclshmem_##NAME##_test_all_vector(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_any_vector(__gm__ TYPE *ivars, size_t nelems, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values)
ACLSHMEM_DEVICE size_t aclshmem_##NAME##_test_some_vector(__gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values)
```

`_test*` are the non-blocking polls — the primitive you want in a persistent kernel so a core
can do other work instead of spinning.

### 9.6 `include/device/gm2gm/shmem_device_so.h` — put-with-signal (data-then-flag as one op)

```cpp
/**
 * @brief Synchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE
 *       then update sig_addr
 *
 * @note This interface supports single-core invocation only; multiple cores or writers are unsupported.
 * @note For UDMA, transfers over 256 MB are split internally before updating sig_addr.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param elem_size         [in] Number of bytes copied from src to dst.
 * @param sig_addr          [in] Symmetric address of the signal word to be updated.
 * @param signal            [in] The value used to update sig_addr.
 * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:
 *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_putmem_signal(
    __gm__ void* dst, __gm__ void* src, size_t elem_size, __gm__ int32_t* sig_addr, int32_t signal, int sig_op, int pe);
#define shmem_putmem_signal aclshmem_putmem_signal

template <typename T>
ACLSHMEM_DEVICE void aclshmem_putmem_signal(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, size_t elem_size, __gm__ int32_t* sig_addr,
    int32_t signal, int sig_op, int pe);
```

and the per-type expansion macro:

```cpp
#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL(NAME, TYPE)                                                                \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_signal(                                                              \
        __gm__ TYPE* dst, __gm__ TYPE* src, size_t elem_size, __gm__ int32_t* sig_addr, int32_t signal, int sig_op, \
        int pe)
ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL);
```

```cpp
/**
 * @brief Standard RMA Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |half       | half      |
 * |float      | float     |
 * |double     | double    |
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 * |int64      | int64     |
 * |uint8      | uint8     |
 * |uint16     | uint16    |
 * |uint32     | uint32    |
 * |uint64     | uint64    |
 * |char       | char      |
 * |bfloat16   | bfloat16  |
 */
#define ACLSHMEM_TYPE_FUNC(FUNC) \
    FUNC(half, half);            \
    FUNC(float, float);          \
    FUNC(double, double);        \
    FUNC(int8, int8_t);          \
    FUNC(int16, int16_t);        \
    FUNC(int32, int32_t);        \
    FUNC(int64, int64_t);        \
    FUNC(uint8, uint8_t);        \
    FUNC(uint16, uint16_t);      \
    FUNC(uint32, uint32_t);      \
    FUNC(uint64, uint64_t);      \
    FUNC(char, char);            \
    FUNC(bfloat16, bfloat16_t)
```

There is also a non-contiguous (strided) put-with-signal taking `non_contiguous_copy_param`:

```cpp
ACLSHMEM_DEVICE void aclshmem_NAME_put_signal(__gm__ TYPE *dst, __gm__ TYPE *src,
    const non_contiguous_copy_param &copy_params, __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
```

The critical constraint, repeated on every signalling variant:
**"This interface supports single-core invocation only; multiple cores or writers are
unsupported."**

### 9.7 `include/host_device/shmem_common_types.h` — enums, limits, and the sync-pool arithmetic

```cpp
enum aclshmem_data_op_type_t {       // (values quoted as found)
    ACLSHMEM_DATA_OP_SDMA = 0x02,
    ACLSHMEM_DATA_OP_ROCE = 0x04,
    ACLSHMEM_DATA_OP_UDMA = 0x08,
    ACLSHMEM_DATA_OP_MAX  = 0x08,
};

/**
 * @brief Memory type of NPU or host
 */
enum aclshmem_mem_type_t {
    HOST_SIDE = 0, // The memory address allocated by shmem is on the host side
    DEVICE_SIDE    // The memory address allocated by shmem is on the device side
};

/**
 * @brief Signal operations, used by the signaler in peer-to-peer synchronization
 */
enum aclshmem_signal_op_type_t { ACLSHMEM_SIGNAL_SET, ACLSHMEM_SIGNAL_ADD };

/**
 * @brief Signal comparison operations, used by the signalee in peer-to-peer synchronization
 */
enum aclshmem_cmp_op_type_t {
    ACLSHMEM_CMP_EQ = 0,
    ACLSHMEM_CMP_NE,
    ACLSHMEM_CMP_GT,
    ACLSHMEM_CMP_GE,
    ACLSHMEM_CMP_LT,
    ACLSHMEM_CMP_LE
};

typedef int aclshmem_team_t;
typedef uint64_t aclshmemx_team_uniqueid_t;

/// \brief Maximum number of Processing Elements (PE) supported by ACLSHMEM (16384)
#define ACLSHMEM_MAX_PES 16384
/// \brief Maximum number of teams supported by ACLSHMEM (2048)
#define ACLSHMEM_MAX_TEAMS 2048
/// \brief Size of the signal structure (8 bytes)
#define ACLSHMEM_SIGNAL_SIZE 8
/// \brief Maximum capacity of ACLSHMEM local memory (40GB)
#define ACLSHMEM_MAX_LOCAL_SIZE (40UL * 1024 * 1024 * 1024)
/// \brief Padding bytes for the team configuration structure, used for memory alignment (48 bytes)
#define TEAM_CONFIG_PADDING 48

/* arch related */
/// \brief Size of the scalar data cache line (64 bytes), used for cache alignment optimization
#define SCALAR_DATA_CACHELINE_SIZE 64
/// \brief Size of the L2 cache line (512 bytes), used for advanced cache alignment optimization
#define L2_CACHELINE_SIZE 512
/// \brief ACLSHMEM memory page size (2MB), complying with system memory paging specifications
#define ACLSHMEM_PAGE_SIZE (1024UL * 1024UL * 2)
/// \brief notify_addr offset (14KB)
#define ACLSHMEM_STARS_NOTIFY_ADDR_OFFSET (14 * 1024)
/// \brief SDMA flag data length
#define ACLSHMEM_SDMA_FLAG_LENGTH 64
/// \brief SDMA shared workspace capacity, covering channel metadata, notify IDs, and three flag regions for up to 72
/// AIVs
#define ACLSHMEM_SDMA_WORKSPACE_SIZE (28U * 1024U)
/// \brief Minimum UB scratch size in bytes for UDMA PIPE_MTE3 WQE staging
#define ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE 128U
/// \brief Minimum UB scratch size in bytes for RDMA MTE3 WQE staging
#define ACLSHMEM_RDMA_MTE_STAGING_UB_SIZE 128U

#define ALIGN_TO(size, page) (((size) + (page) - 1) / (page) * (page))

/* synchronization related */
/// \brief Memory size of the ACLSHMEM synchronization bit, consistent with the scalar data cache line size
#define ACLSHMEM_SYNCBIT_SIZE SCALAR_DATA_CACHELINE_SIZE

// npu level sync
/// \brief Number of log bits required for NPU-level synchronization (5 bits), calculated as: ceil(log₈(16384)) = 5
#define SYNC_LOG_MAX_PES 5
/// \brief Task group dissemination coefficient (8) for ACLSHMEM barrier synchronization, used for synchronization
/// algorithm optimization
#define ACLSHMEM_BARRIER_TG_DISSEM_KVAL 8
/// \brief Total size of the NPU-level synchronization array, calculated from sync bit size, log bits and dissemination
/// coefficient
#define SYNC_ARRAY_SIZE (ACLSHMEM_SYNCBIT_SIZE * SYNC_LOG_MAX_PES * ACLSHMEM_BARRIER_TG_DISSEM_KVAL)
#define SYNC_COUNTER_SIZE ACLSHMEM_SYNCBIT_SIZE
#define SYNC_POOL_SIZE (SYNC_ARRAY_SIZE * ACLSHMEM_MAX_TEAMS)
#define SYNC_COUNTERS_SIZE (SYNC_COUNTER_SIZE * ACLSHMEM_MAX_TEAMS)

// core level sync
/// \brief Maximum number of AIV (AI Core) supported per NPU (72)
#define ACLSHMEM_MAX_AIV_PER_NPU 72
/// \brief Number of log bits required for core-level synchronization (7 bits), calculated as: ceil(log₂(72)) = 7
#define ACLSHMEM_LOG_MAX_AIV_PER_NPU 7
/// \brief Total size of the core-level synchronization pool, calculated from AIV count, log bits and sync bit size
#define ACLSHMEM_CORE_SYNC_POOL_SIZE (ACLSHMEM_MAX_AIV_PER_NPU * ACLSHMEM_LOG_MAX_AIV_PER_NPU * ACLSHMEM_SYNCBIT_SIZE)
#define ACLSHMEM_CORE_SYNC_COUNTER_SIZE ACLSHMEM_SYNCBIT_SIZE

#define ACLSHMEM_EXTRA_SIZE_UNALIGNED SYNC_POOL_SIZE
#define ACLSHMEM_EXTRA_SIZE ALIGN_TO(ACLSHMEM_EXTRA_SIZE_UNALIGNED, ACLSHMEM_PAGE_SIZE)
```

Read the barrier arithmetic: the all-PE barrier is a **dissemination barrier with radix
K = 8** (`ACLSHMEM_BARRIER_TG_DISSEM_KVAL 8`) over `ceil(log_8(16384)) = 5` rounds, each round
having its own 64-byte sync bit, times 2048 teams. For TP16 the radix-8 dissemination needs
`ceil(log_8 16) = 2` rounds. The *core-level* barrier is radix-2 over
`ceil(log_2 72) = 7` rounds, 64 B per round per AIV: `72 * 7 * 64 = 32256 B` of GM.

So the device barrier is a **software, GM-resident, log-depth tree over 64-byte cache lines** —
not an FFTS flag. That is why it costs what it costs, and why `aclshmemx_sync_vec` exists.

## Relevance

This is the sanctioned answer to the question this file was opened to answer, and it is better
than expected: a one-sided `put`, a one-sided `signal_op`, a `put`+`signal` fused into one
ordered operation, a non-blocking `test`, and a blocking `signal_wait_until` — all callable
from `__aicore__` device code, all first-party, all open source. §9.5+§9.6 are literally the
epoch-tagged-flag protocol we were going to hand-roll, with `ACLSHMEM_SIGNAL_ADD` giving the
monotone counter that removes the stale-flag race without zeroing buffers.

Three facts land directly on our measurements. (a) `shmem_device_cc.h` states the all-core
barrier **requires a MIX kernel** and that all local AIC *and* AIV cores must enter — the same
conclusion our `KERNEL_TYPE_AIC_ONLY` rejection forces on us, arrived at from the library side.
(b) `aclshmemx_sync_vec` is an AIV-only cross-rank sync, i.e. there is a sanctioned path that
does *not* drag Cube into the rendezvous — relevant because our 49.3 us MoE chain is MIX and we
do not want to pay a full-core barrier to publish an all-reduce. (c) **"The Cube scalar unit is
outside this visibility guarantee"** — if the Cube scalar unit writes the activation we are
about to all-reduce, the collective does not cover it; the write must come from AIV or from an
explicit `put`.
Two hard costs to budget: `ACLSHMEM_INTERNAL_UB_BUF_START_ADDR = 188 * 1024` takes the top 4 KB
of UB, and `aclshmem_putmem_signal` is **single-core only** ("multiple cores or writers are
unsupported"), so the publish step serialises onto one AIV unless we shard by destination PE.

## 10. `examples/tp_allreduce_udma` — a complete device-side TP all-reduce in AscendC, first-party

URL: https://gitcode.com/cann/shmem/tree/master/examples/tp_allreduce_udma
Raw: https://raw.gitcode.com/cann/shmem/raw/master/examples/tp_allreduce_udma/tp_allreduce_udma_kernel.cpp
Author: Huawei Technologies Co., Ltd. (copyright header reads `Copyright (c) 2026`)
Date: shipped with SHMEM v1.7.0, 2026/10 ("新增 MegaMoE、UDMA AllReduce 示例")
Wiki writeup referenced by the README: https://gitcode.com/cann/shmem/wiki/operator-practice-tp-allreduce-udma
Retrieved 2026-10-20.

This is a TP=2 all-reduce written as reduce-scatter + local reduce + all-gather, with the
communication issued from `__aicore__` code by AIV cores driving UDMA. Reproduced nearly in
full; the licence header and the profiling wrappers are the only things cut.

### 10.1 The constants and the kernel entry — note `[[bisheng::core_ratio(0, 1)]]`

```cpp
#include "tp_allreduce_udma_kernel.h"

#include "kernel_operator.h"
#include "shmem.h"
#include "utils/prof/shmemi_prof.h"

namespace {

constexpr uint32_t TP_SIZE = 2;
#if defined(ACLSHMEM_RELAY_SUPPORT)
constexpr uint32_t BOX_SIZE = 4;
constexpr uint32_t TAILCUT_PATH_COUNT = 3;
#endif
constexpr uint32_t REDUCE_TILE_BYTES = 16 * 1024;
constexpr uint32_t REDUCE_QUEUE_DEPTH = 1;
constexpr uint32_t UDMA_WQE_SCRATCH_BYTES = 256;
constexpr int32_t PROF_FRAME_TOTAL = 0;
constexpr int32_t PROF_FRAME_REDUCE_SCATTER = 1;
constexpr int32_t PROF_FRAME_LOCAL_REDUCE = 2;
constexpr int32_t PROF_FRAME_ALLGATHER = 3;
```

```cpp
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void tp_allreduce_udma_int32_t_kernel(
    uint64_t fftsConfig, GM_ADDR workspace, uint32_t elements, int teamId, int32_t perfMode, int32_t mode,
    int32_t stage, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight)
{
    TpAllReduceUdmaImpl<int32_t>(
        fftsConfig, workspace, elements, teamId, perfMode, mode, stage, directWeight, relay0Weight, relay1Weight);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void tp_allreduce_udma_float16_t_kernel(...)

void launch_tp_allreduce_udma_int32_t(
    uint32_t blockDim, void* stream, uint64_t fftsConfig, uint8_t* workspace, uint32_t elements, aclshmem_team_t tpTeam,
    int32_t perfMode, int32_t mode, int32_t stage, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight)
{
    tp_allreduce_udma_int32_t_kernel<<<blockDim, nullptr, stream>>>(
        fftsConfig, workspace, elements, static_cast<int>(tpTeam), perfMode, mode, stage, directWeight, relay0Weight,
        relay1Weight);
}
```

`[[bisheng::core_ratio(0, 1)]]` — **0 cube : 1 vector**, i.e. an AIV-only kernel declared through
a Bisheng C++ attribute rather than through a `KERNEL_TYPE_*` macro. The companion
`examples/barrier_all/barrier_all_kernel.cpp` (§10.4) uses `[[bisheng::core_ratio(1, 2)]]`, the
1:2 MIX ratio. This is the first-party spelling of the kernel-type declaration in current CANN
examples.

### 10.2 The communication primitives actually called

```cpp
template <typename T>
ACLSHMEM_DEVICE void SendShardDirect(
    __gm__ T* remoteDst, __gm__ T* localSrc, __ubuf__ T* udmaBuffer, uint32_t elementCount, int32_t targetPe)
{
    aclshmemx_udma_put_nbi(remoteDst, localSrc, udmaBuffer, elementCount, targetPe);
    aclshmemx_udma_quiet(targetPe);
}
```

and the multi-path ("tailcut") variant, which is the interesting one: three cores each submit a
different *path* to the same destination PE, two of them bouncing through a relay PE, then
**one** `AscendC::SyncAll<true>()` and **one** core calls quiet:

```cpp
template <typename T>
ACLSHMEM_DEVICE void SendShardTailcut(
    __gm__ T* remoteDst, __gm__ T* localSrc, __ubuf__ T* udmaBuffer, uint32_t elementCount, int32_t targetPe,
    int32_t myPe, uint32_t coreIdx, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight)
{
    int32_t relay0 = -1;
    int32_t relay1 = -1;
    bool enableTailcut = GetRelayPeers(myPe, targetPe, relay0, relay1);

    if (!enableTailcut) {
        if (coreIdx == 0) {
            SendShardDirect(remoteDst, localSrc, udmaBuffer, elementCount, targetPe);
        }
        return;
    }

    if (coreIdx < TAILCUT_PATH_COUNT) {
        uint32_t offset = 0;
        uint32_t currentElements = 0;
        GetTailcutSegment(coreIdx, elementCount, directWeight, relay0Weight, relay1Weight, offset, currentElements);
        if (currentElements > 0) {
            if (coreIdx == 0) {
                aclshmemx_udma_put_nbi(remoteDst + offset, localSrc + offset, udmaBuffer, currentElements, targetPe);
            } else {
                int32_t relayPe = coreIdx == 1 ? relay0 : relay1;
                aclshmemx_udma_relay_put_nbi(
                    remoteDst + offset, localSrc + offset, udmaBuffer, currentElements, targetPe, relayPe);
            }
        }
    }

    // Every active-path NBI call must return before core0 starts scanning the completion slots.
    AscendC::SyncAll<true>();
    if (coreIdx == 0) {
        aclshmemx_udma_quiet(targetPe);
    }
}
```

`GetRelayPeers` / `GetTailcutSegment` — the relay topology and the weighted split:

```cpp
ACLSHMEM_DEVICE bool GetRelayPeers(int32_t myPe, int32_t targetPe, int32_t& relay0, int32_t& relay1)
{
    int32_t boxBase = myPe / static_cast<int32_t>(BOX_SIZE) * static_cast<int32_t>(BOX_SIZE);
    uint32_t relayCount = 0;
    for (uint32_t i = 0; i < BOX_SIZE; ++i) {
        int32_t candidate = boxBase + static_cast<int32_t>(i);
        if (candidate == myPe || candidate == targetPe) {
            continue;
        }
        if (relayCount == 0) {
            relay0 = candidate;
        } else if (relayCount == 1) {
            relay1 = candidate;
        }
        ++relayCount;
    }
    return relayCount == 2;
}

ACLSHMEM_DEVICE void GetTailcutSegment(
    uint32_t pathIdx, uint32_t totalElements, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight,
    uint32_t& offset, uint32_t& elementCount)
{
    uint64_t totalWeight =
        static_cast<uint64_t>(directWeight) + static_cast<uint64_t>(relay0Weight) + static_cast<uint64_t>(relay1Weight);
    if (totalWeight == 0) {
        offset = 0;
        elementCount = 0;
        return;
    }
    // Cumulative floor boundaries keep zero-weight paths empty while covering the complete range.
    uint32_t directEnd = static_cast<uint32_t>(static_cast<uint64_t>(totalElements) * directWeight / totalWeight);
    uint32_t relay0End = static_cast<uint32_t>(
        static_cast<uint64_t>(totalElements) * (static_cast<uint64_t>(directWeight) + relay0Weight) / totalWeight);
    if (pathIdx == 0) {
        offset = 0;
        elementCount = directEnd;
    } else if (pathIdx == 1) {
        offset = directEnd;
        elementCount = relay0End - directEnd;
    } else {
        offset = relay0End;
        elementCount = totalElements - offset;
    }
}
```

### 10.3 The body — symmetric-memory layout, UB WQE scratch, and the STAGE split

```cpp
template <typename T>
ACLSHMEM_DEVICE void TpAllReduceUdmaImpl(
    uint64_t fftsConfig, GM_ADDR workspace, uint32_t elements, int teamId, int32_t perfMode, int32_t mode,
    int32_t stage, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight)
{
    util_set_ffts_config(fftsConfig);

    aclshmem_team_t tpTeam = static_cast<aclshmem_team_t>(teamId);
    int32_t rankInTp = aclshmem_team_my_pe(tpTeam);
    int32_t tpSize = aclshmem_team_n_pes(tpTeam);
    if (tpSize != TP_SIZE || elements % TP_SIZE != 0) {
        return;
    }

    int32_t peerInTp = rankInTp ^ 1;
    int32_t myGlobalRank = -1;
#if defined(ACLSHMEM_RELAY_SUPPORT)
    if (mode == static_cast<int32_t>(TpAllReduceUdmaMode::TAILCUT)) {
        myGlobalRank = aclshmem_team_translate_pe(tpTeam, rankInTp, ACLSHMEM_TEAM_WORLD);
    }
#endif
    int32_t peerGlobalRank = aclshmem_team_translate_pe(tpTeam, peerInTp, ACLSHMEM_TEAM_WORLD);
    uint32_t shardElements = elements / TP_SIZE;
    uint32_t coreIdx = AscendC::GetBlockIdx();
    uint32_t coreCount = AscendC::GetBlockNum();

    __gm__ T* input = reinterpret_cast<__gm__ T*>(workspace);
    __gm__ T* reduceScatterRecv = input + elements;
    __gm__ T* output = reduceScatterRecv + elements;

    __gm__ T* shardForPeer = input + peerInTp * shardElements;
    __gm__ T* peerRecvSlot = reduceScatterRecv + rankInTp * shardElements;
    __gm__ T* localShard = input + rankInTp * shardElements;
    __gm__ T* receivedPeerShard = reduceScatterRecv + peerInTp * shardElements;
    __gm__ T* reducedLocalShard = output + rankInTp * shardElements;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> udmaBuffer;
    pipe.InitBuffer(udmaBuffer, UDMA_WQE_SCRATCH_BYTES);
    AscendC::LocalTensor<uint8_t> udmaUb = udmaBuffer.GetWithOffset<uint8_t>(UDMA_WQE_SCRATCH_BYTES, 0);
    __ubuf__ T* udmaUbAddr = reinterpret_cast<__ubuf__ T*>(udmaUb.GetPhyAddr());

    TpAllReduceUdmaStage currentStage = static_cast<TpAllReduceUdmaStage>(stage);
    if (currentStage == TpAllReduceUdmaStage::REDUCE_SCATTER) {
        ProfStart(perfMode, PROF_FRAME_TOTAL);
        ProfStart(perfMode, PROF_FRAME_REDUCE_SCATTER);
        SendShard(
            peerRecvSlot, shardForPeer, udmaUbAddr, shardElements, peerGlobalRank, myGlobalRank, coreIdx, mode,
            directWeight, relay0Weight, relay1Weight);
        ProfEnd(perfMode, PROF_FRAME_REDUCE_SCATTER);
        return;
    }
    if (currentStage != TpAllReduceUdmaStage::LOCAL_REDUCE_ALLGATHER) {
        return;
    }
    ...
    if (reduceElements > 0) {
        ReduceTwoShards(
            localShard + reduceOffset, receivedPeerShard + reduceOffset, reducedLocalShard + reduceOffset,
            reduceElements, pipe);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll<true>();

    ProfStart(perfMode, PROF_FRAME_ALLGATHER);
    SendShard(
        reducedLocalShard, reducedLocalShard, udmaUbAddr, shardElements, peerGlobalRank, myGlobalRank, coreIdx, mode,
        directWeight, relay0Weight, relay1Weight);
    ProfEnd(perfMode, PROF_FRAME_ALLGATHER);
    ProfEnd(perfMode, PROF_FRAME_TOTAL);
}
```

and the header that declares the stage enum, i.e. the admission that the example does **two
launches**:

```cpp
enum class TpAllReduceUdmaMode : int32_t {
    BASELINE = 0,
    TAILCUT = 1,
};

enum class TpAllReduceUdmaStage : int32_t {
    REDUCE_SCATTER = 0,
    LOCAL_REDUCE_ALLGATHER = 1,
};
```

The local reduce, for completeness — a plain double-buffer-free UB tile loop, `AscendC::Add`:

```cpp
template <typename T>
ACLSHMEM_DEVICE void ReduceTwoShards(
    __gm__ T* localShard, __gm__ T* peerShard, __gm__ T* outputShard, uint32_t shardElements, AscendC::TPipe& pipe)
{
    AscendC::GlobalTensor<T> localGm; ... localGm.SetGlobalBuffer(localShard); ...
    AscendC::TQue<AscendC::TPosition::VECIN, REDUCE_QUEUE_DEPTH> localQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, REDUCE_QUEUE_DEPTH> peerQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, REDUCE_QUEUE_DEPTH> outputQueue;
    pipe.InitBuffer(localQueue, REDUCE_QUEUE_DEPTH, REDUCE_TILE_BYTES);
    pipe.InitBuffer(peerQueue, REDUCE_QUEUE_DEPTH, REDUCE_TILE_BYTES);
    pipe.InitBuffer(outputQueue, REDUCE_QUEUE_DEPTH, REDUCE_TILE_BYTES);

    constexpr uint32_t tileElements = REDUCE_TILE_BYTES / sizeof(T);
    for (uint32_t offset = 0; offset < shardElements; offset += tileElements) {
        ...
        AscendC::DataCopyPad(localUb, localGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(peerUb, peerGm[offset], copyParams, padParams);
        ...
        AscendC::Add(outputUb, localUb, peerUb, currentElements);
        ...
        AscendC::DataCopyPad(outputGm[offset], outputUb, copyParams);
    }
}
```

### 10.4 `examples/barrier_all/barrier_all_kernel.cpp` — the MIX device-side cross-rank barrier, in full

Raw: https://raw.gitcode.com/cann/shmem/raw/master/examples/barrier_all/barrier_all_kernel.cpp

```cpp
#include "kernel_operator.h"
#include "shmem.h"
#include "barrier_all_kernel.h"

namespace barrier_all_demo {

extern "C" [[bisheng::core_ratio(1, 2)]] __global__ __aicore__ void device_barrier_all_demo(
    uint64_t fftsAddr, GM_ADDR buffer)
{
    util_set_ffts_config(fftsAddr);
    int pe = aclshmem_my_pe();
    int n_pes = aclshmem_n_pes();
    AscendC::PipeBarrier<PIPE_ALL>();

    // Only AIV sub-block 0 submits the puts; other AIV sub-blocks and AIC
    // still participate in the collective barrier below.
    if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() == 0) {
            auto data = reinterpret_cast<__gm__ uint8_t*>(buffer);
            // The high-level API selects MTE for same-machine PEs and RDMA for cross-machine PEs.
            for (int peer = 0; peer < n_pes; ++peer) {
                if (peer == pe) {
                    continue;
                }
                aclshmem_putmem_nbi(data + pe * kSlotBytes, data + pe * kSlotBytes, kSlotBytes, peer);
            }
        }
    }

    aclshmem_barrier_all();
}

void launch_barrier_all(void* stream, uint64_t fftsAddr, uint32_t* buffer)
{
    device_barrier_all_demo<<<1, nullptr, stream>>>(fftsAddr, reinterpret_cast<uint8_t*>(buffer));
}
} // namespace barrier_all_demo
```

Note: **"The high-level API selects MTE for same-machine PEs and RDMA for cross-machine PEs."**
Single machine TP16 ⇒ `aclshmem_putmem_nbi` goes over MTE, not RDMA/UDMA. And note blockDim is
literally `1` in the launch — one AIC + its two AIVs.

### 10.5 The README's constraints — THE PLATFORM LIMIT

`examples/tp_allreduce_udma/README.md`, 约束 section, verbatim:

> ## 约束
>
> - `baseline` 与 `tailcut` 算子依赖 UB-Mesh 互联，以及 AIV Core 直驱 UDMA 的能力，**当前仅支持 Ascend 950PR&950DT系列产品**。
> - `tailcut` 需借助其他 NPU 中转部分通信数据，**当前 relay 功能仅支持 Atlas 950 SuperPoD**；非该形态部署下，无法保证绕路转发可用，请勿使用 `tailcut` 模式。
> - TP 大小固定为 2。
> - `elements` 必须为正数且能被 2 整除。
> - Tailcut 模式要求 PE 数量能被 4 整除，且 AIV Core 数量至少为 3。
> - `--ratio` 仅用于 Tailcut 模式，三个权重必须为非负整数且总和大于 0。
> - 每阶段的流量边界按 `floor(shard_elements * 累计权重 / 总权重)` 计算，权重为 0 的路径不发送数据。

> ## 示例概述
>
> 本示例在单机启动多个 PE，并将相邻的两个 PE 划分为一个 TP=2 通信组。每个 TP 组独立执行
> `reduce-scatter + local reduce + all-gather`，不同 TP 组之间不传输 AllReduce 数据。
>
> 示例通过 `--mode` 提供两种通信模式，并复用相同的 Host 初始化、Local Reduce、性能统计和
> golden 数据校验逻辑：
>
> - `baseline`：Core 0 调用 `aclshmemx_udma_put_nbi` 直发数据，并调用 `aclshmemx_udma_quiet` 等待完成。
> - `tailcut`：将通信分片按比例切分到一条直发路径和两条绕路路径，Core 0、1、2 分别提交 direct、relay0、relay1 路径，随后由 Core 0 调用一次 `aclshmemx_udma_quiet`。
>
> Local Reduce 按 tile 切分给所有启动的 AIV Core 计算。Tailcut 模式将每四个连续 PE 视为一个
> box，当前 TP 对之外的两个 PE 作为 relay PE，默认流量权重为 `2:1:1`。

Parameter defaults, for the shape of the measurement they do:

> | `-pes` | PE 数量；baseline 支持 2、4、8、16，tailcut 支持 4、8、16 | 未指定时 baseline 为 2、tailcut 为 4 |
> | `-elements` | 每个 PE 输入 Tensor 的元素数量 | 1024 |
> | `-type` | 数据类型，支持 `int32_t`、`float16_t` | `float16_t` |
> | `-aiv` | Kernel 使用的 AIV Core 数量；tailcut 至少为 3，且不能超过设备的 Vector Core 数量 | 未指定时 baseline 为 1、tailcut 为 3 |

### 10.6 `docs/api/sync_barrier_api_usage.md` — the sanctioned in-kernel ordering recipes

> ## 三、Device kernel 中的使用顺序
>
> 以下两种顺序均要求 team 内所有 PE 和声明参与范围内的核按相同顺序调用集体接口：
>
>     // 方式一：NBI -> barrier
>     aclshmem_putmem_nbi(dst, src, bytes, peer);
>     aclshmem_barrier_all();
>     // barrier 返回后，先前通信完成；可读取对应目标数据、复用源缓冲区。
>
>     // 方式二：NBI -> quiet -> sync
>     aclshmem_putmem_nbi(dst, src, bytes, peer);
>     aclshmem_quiet();     // 本 PE 的先前通信完成，可复用本地源缓冲区
>     aclshmem_sync_all();  // 等待所有 PE 到达；之后再读取其他 PE 的写入结果
>
> 普通对称内存 store 的同步/可见性与通信 API 的完成范围不同：sync 同步参与核在受支持的 HCCS/UB 内存域中的普通存储，但不能代替 quiet 完成 NBI；barrier 同时提供通信完成与 PE 到达同步。**对混合核，Cube 标量单元不在这些 API 的同步保证范围内**；参与核范围见 Device public header。Host 发出的通信应使用 Host 对应接口完成，Device 发出的通信应在 Device 执行范围内完成。

> **屏障、同步与通信完成：**
>
> - sync 表示指定 team 的 PE 到达同一同步点；对参与核此前的普通对称内存存储提供适用范围内的同步与可见性，但不保证此前发出的非阻塞通信已完成。Host 的 aclshmem_team_sync 与流式 sync 均应按这一最低保证使用；不要以 sync 代替通信完成操作。
> - quiet 完成调用 PE 在对应 Host 或 Device 执行范围内先前发出的通信，但不等待其他 PE 到达同步点。
> - barrier 先完成相应执行范围内此前发出的通信，再等待指定 team 的所有 PE 到达同步点。所有集体调用必须由 team 内 PE 按相同顺序参与。
> - 在 Device kernel 中，NBI 后如需读取远端结果或复用源缓冲区，使用 barrier；或者先 quiet 完成本 PE 通信，再 sync 等待其他 PE。fence 仅用于建立顺序，不作为通用的通信完成保证。

## Relevance

Two things to take and one to reject.

Take: `[[bisheng::core_ratio(aic, aiv)]]` is the current first-party way to declare the core mix
of a kernel — `(0, 1)` for AIV-only, `(1, 2)` for the A2/A3 MIX ratio. That is a second,
attribute-based spelling alongside the `KERNEL_TYPE_*` macros, and it is what CANN's own 2026
examples use. If `KERNEL_TYPE_AIC_ONLY` is rejected at submission with
`sqe_type=3, errType=0x20`, `[[bisheng::core_ratio(1, 0)]]` is the obvious next thing to try —
the notation can at least *express* cube-only, and it is the notation the toolchain itself ships.

Take: the ordering recipe `NBI -> quiet -> sync` with the explicit note that `sync` alone does
**not** complete an NBI. Our one-sided all-reduce therefore needs, per layer, exactly: put_nbi
(or putmem_signal) → quiet → a cross-rank rendezvous. `aclshmemx_sync_vec_all()` (§9.2) is the
AIV-only rendezvous, so the whole sequence can stay off Cube.

Reject: the structure of this example is *not* what we want. `TpAllReduceUdmaStage` splits the
all-reduce into **two kernel launches** with a host barrier between reduce-scatter and
local-reduce — exactly the host round trip we are trying to delete. Nothing in the example says
this is required; the stage split appears to exist so the benchmark can time the halves
separately (the three `PROF_FRAME_*` IDs). But it does mean the first-party example does not
demonstrate a single-launch all-reduce; §11 does.

Hard limit to note: the UDMA path, i.e. "AIV Core 直驱 UDMA", is 「当前仅支持 Ascend 950PR&950DT
系列产品」 — A5 only, NOT A3. On A3 the engine is MTE/SDMA (§10.4: "The high-level API selects MTE
for same-machine PEs"), and §9.1's communication-path figure is drawn 「以910A3为例」, so MTE
one-sided put is the A3 path. The UB cost is small: `UDMA_WQE_SCRATCH_BYTES = 256`, and the
header constants say `ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE 128U` /
`ACLSHMEM_RDMA_MTE_STAGING_UB_SIZE 128U` minimum.

## 11. `examples/shmem_mega_moe` — a MoE forward pass as ONE device pipeline, with a hand-rolled epoch-tagged cross-rank barrier

URL: https://gitcode.com/cann/shmem/tree/master/examples/shmem_mega_moe
Raw base: https://raw.gitcode.com/cann/shmem/raw/master/examples/shmem_mega_moe/
Author: Huawei Technologies Co., Ltd. (`Copyright (c) 2026`)
Date: shipped with SHMEM v1.7.0, 2026/10 ("新增 MegaMoE、UDMA AllReduce 示例")
Retrieved 2026-10-20.

This is the closest public artifact to what we are building: token routing, cross-rank dispatch,
two expert GEMMs, SwiGLU, and cross-rank combine, all inside one device-side pipeline. And it
does **not** use `aclshmem_barrier_all()` in the hot path — it hand-rolls an epoch-tagged
cross-rank barrier over symmetric memory, and it writes remote results with a **plain
`DataCopyPad` into a `GlobalTensor` based at `aclshmem_ptr(localBase, peerRank)`**.

### 11.1 README — the goal statement and the division of labour

`README.md`, verbatim:

> # SHMEM MegaMoE 示例（Ascend950）
>
> ## 样例介绍
>
> 本样例展示了一个面向 Ascend950 的 SHMEM 原生 MegaMoE 前向流水线。算子将
> Token 路由、跨 Rank 数据派发、两次专家矩阵乘、SwiGLU 激活和多专家结果合并放在同一个
> Device 侧流水线中，避免在各阶段之间回到 Host 或重复组织通信。
>
> 实现中各组件的职责如下：
>
> - ACLSHMEM 提供多 Rank 对称内存、远端地址访问和全局同步能力；
> - AscendC AIV 核完成路由掩码生成、Token 量化、SwiGLU 后处理和最终结果合并；
> - Catlass AIC 核完成专家网络的两次 MXFP8 GEMM；
> - AIC/AIV 之间通过标志位和双缓冲协作，使矩阵计算、后处理与通信可以流水执行。

> ## 整体数据流
>
> 算子的主流程位于 `kernel/arch35/shmem_mega_moe_kernel.h` 的 `Process()` 中，数据路径可概括为：
>
> ```text
> 本 Rank BF16 Token
>         |
>         v
> 生成目标 Rank 路由掩码 -----> 远端 Rank 的 SHMEM 对称内存
>         |
>         v
> BF16 -> FP8(E4M3/E5M2) + E8M0 Scale
>         |
>         v
> 目标 Rank 统计并接收 Token，生成 routing metadata
>         |
>         v
> Catlass GEMM 1: model_dim -> ffn_dim
>         |
>         v
> AIV SwiGLU + 再量化: ffn_dim -> ffn_dim / 2
>         |
>         v
> Catlass GEMM 2: ffn_dim / 2 -> model_dim
>         |
>         v
> 根据 routing metadata 写回源 Rank 的 combinedTokens
>         |
>         v
> 源 Rank 按 routing_weights 累加 Top-K 结果 -> BF16 输出
> ```

> ### 1. 生成跨 Rank 路由掩码
>
> `BuildSendMasks` 遍历本 Rank 的 `expert_ids`，将每个 Token 的每条路由转换为目标 Rank 和
> 目标局部专家，并生成按目标 Rank 划分的 Token 位图。位图和待发送 Token 数直接写入目标
> Rank 的 `inboundMask` 对称内存，使接收方无需由 Host 参与即可知道需要拉取哪些 Token。

> ### 4. 第一次专家投影与 SwiGLU
>
> `RunFirstProjection` 使用 Catlass 执行第一次 MXFP8 GEMM。AIC 核将结果分块写入 UB，AIV 核
> 消费这些分块，完成以下操作：
>
> 1. 将 `ffn_dim` 宽度的输出切分为 Gate 和 Up；
> 2. 计算 `SiLU(Gate) * Up`；
> 3. 将 BF16 激活重新量化为 FP8，并生成第二次 GEMM 所需的 E8M0 Scale；
> 4. 通过双缓冲标志通知 AIC 对应分块已经就绪。
>
> 因此，第一次 GEMM、SwiGLU 和再量化无需把完整中间张量落回全局内存后再启动新的算子。

> ### 6. 远端写回专家结果
>
> 第二次投影结束后，AIV 根据路由元数据把每一行专家结果映射回源 Rank、源 Token 和 Top-K
> 槽位，并写入源 Rank 的 `combinedTokens` 对称内存。写回过程使用乒乓标志同步数据生产和消费，
> 防止不同轮次复用缓冲区时覆盖尚未被读取的数据。

> ## AIC/AIV 分工与同步
>
> | 处理单元 | 主要工作 |
> | --- | --- |
> | AIC | 两次 Catlass MXFP8 GEMM、矩阵分块调度 |
> | AIV | 路由掩码生成、Token 量化、路由数据搬运、SwiGLU、再量化和最终 Combine |
>
> 每个局部专家对应一组计算资源。运行时会查询设备上的 AIC/AIV 数量，并将结果写入 Tiling；
> 也可以通过环境变量覆盖核数。AIC 和 AIV 使用以下机制协调：
>
> - 第一次投影采用双缓冲，AIC 生产 GEMM 分块，AIV 消费分块并执行 SwiGLU；
> - 输入就绪、激活就绪和 Combine 就绪状态通过独立标志区传递；
> - 路由派发按固定行数分波次处理，避免整批 Token 串行等待；
> - 多 Rank 阶段边界使用 ACLSHMEM 同步，确保远端写入对接收方可见。
>
> 运行前会检查 `local_expert_count` 不大于可用 AIC 数量。如果资源不足，程序会直接报错，
> 避免多个局部专家错误地竞争同一组 AIC 流水资源。

And the MIX-ratio constraint, from the 参数说明 section:

> 设置 `AIC_NUM` 或 `AIV_NUM` 时，覆盖值不能超过运行时检测到的硬件核数，并且必须保持
> `AIC_NUM:AIV_NUM=1:2` 的 MIX Kernel 比例。建议同时设置两个变量，例如
> `AIC_NUM=4 AIV_NUM=8`，避免只覆盖一个变量导致比例校验失败。

Symmetric-memory regions:

> | 区域 | 用途 |
> | --- | --- |
> | 控制与同步区 | 保存跨 Rank 屏障、标志位等控制信息 |
> | `inboundMask` | 接收其他 Rank 写入的 Token 路由位图和计数 |
> | `quantizedTokens` | 保存本 Rank 已量化的 Token，供目标 Rank 远端读取 |
> | `combinedTokens` | 接收远端专家返回的、尚未按权重累加的 Token 结果 |
>
> 主要缓冲区按 512 字节对齐，以满足搬运和矩阵计算要求。样例默认申请 1 GiB 对称内存。

Workspace flag regions:

> | `inputReadyFlags` | 路由输入分波次就绪标志 |
> | `activationReadyFlags` | SwiGLU 激活分块就绪标志 |

### 11.2 THE EPOCH-TAGGED CROSS-RANK BARRIER — `SynchronizeRanks()`, verbatim

File: `kernel/arch35/shmem_mega_moe_kernel.h`, lines 1035–1057.

```cpp
// SynchronizeRanks uses the first 48 KiB of symmetric memory for cross-rank synchronization.
template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::SynchronizeRanks()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.symmetricMemory.localBase);
    __gm__ int32_t* syncCount =
        (__gm__ int32_t*)(params_.symmetricMemory.localBase + 48 * 1024 + vectorCoreIndex_ * 64);
    int count = ShmemMegaMoeReadGmBypassDCache(syncCount) + 1;
    for (int i = vectorCoreIndex_; i < rankCount_; i += vectorCoreCount_) {
        __gm__ int32_t* remoteSynchronizationFlag =
            reinterpret_cast<__gm__ int32_t*>(GetRemoteAddress(params_.symmetricMemory.localBase, i, 0)) + rankId_ * 16;
        ShmemMegaMoeWriteGmBypassDCache(remoteSynchronizationFlag, count);
        auto syncCheck = syncRank + i * 16;
        WaitForRankGeneration(syncCheck, count);
    }
    ShmemMegaMoeWriteGmBypassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}
```

and the three helpers it rests on, from `kernel/arch35/shmem_mega_moe_types.h` (lines 42–64,
210–225) — **the exact code of how a receiver knows a remote write landed without a barrier**:

```cpp
namespace AscendC {
template <typename T>
__aicore__ inline T ShmemMegaMoeReadGmBypassDCache(__gm__ T* addr)
{
#if !defined(__NPU_ARCH__)
    return *addr;
#elif defined(SHMEM_MEGA_MOE_USE_NEW_BYPASS_DCACHE)
    return ReadGmBypassDCache(addr);
#else
    return ReadGmByPassDCache(addr);
#endif
}

template <typename T>
__aicore__ inline void ShmemMegaMoeWriteGmBypassDCache(__gm__ T* addr, T value)
{
#if !defined(__NPU_ARCH__)
    *addr = value;
#elif defined(SHMEM_MEGA_MOE_USE_NEW_BYPASS_DCACHE)
    WriteGmBypassDCache(addr, value);
#else
    WriteGmByPassDCache(addr, value);
#endif
}
} // namespace AscendC
```

```cpp
__aicore__ inline void WaitForRankGeneration(__gm__ int32_t* signal, int32_t generation)
{
    do {
        // The same slot is reused by successive barrier generations. A faster rank may publish
        // the next generation before a slower rank samples the current one, so equality can wait
        // forever after the value has advanced past generation.
        if (ShmemMegaMoeReadGmBypassDCache(signal) >= generation) {
            return;
        }
    } while (true);
}

__aicore__ inline GM_ADDR GetRemoteAddress(GM_ADDR localBase, uint32_t rankId, uint64_t offset)
{
    return reinterpret_cast<GM_ADDR>(aclshmem_ptr(localBase, rankId)) + offset;
}
```

Read the protocol off that code:

- The epoch ("generation") is a **monotone per-core counter**, read from the core's own private
  slot at `localBase + 48 KiB + coreIdx * 64`, incremented by 1 per barrier. It is never reset.
- Rank `r` announces arrival by **writing `count` into rank `i`'s slot at
  `aclshmem_ptr(localBase, i) + r * 16` int32s** = `r * 64` bytes = one 64-byte cache line per
  rank, inside the first 48 KiB of symmetric memory.
- The receiver spins on its *own* `syncRank + i * 16` with **`>= generation`, not `==`**, and the
  comment states exactly why: the slot is reused across generations and a fast rank can race
  ahead, so equality deadlocks. This is the whole trick, in one `if`.
- Read and write both **bypass the data cache** (`ReadGmBypassDCache` / `WriteGmBypassDCache`,
  spelled `ReadGmByPassDCache` on older CANN — the example CMake probes which spelling the
  installed headers use; see `SHMEM_MEGA_MOE_BYPASS_DCACHE_MODE=OLD|NEW` in the README). No
  `dcci` in the loop — the bypass variant replaces it.
- **`if constexpr (g_coreType == AIC) { return; }`** — the AIC cores do not participate at all.
  The barrier is AIV-only, then a local `SyncAll<true>()`.
- Work is sharded across AIVs by `for (int i = vectorCoreIndex_; i < rankCount_; i += vectorCoreCount_)`,
  so with 16 ranks and ≥16 AIVs each AIV handles one peer.

### 11.3 The other spin-wait — a per-tile completion counter with a backoff

`kernel/arch35/shmem_mega_moe_kernel.h`, lines 872–876:

```cpp
    __gm__ int32_t* completionCounter =
        (__gm__ int32_t*)rowGroupCounters.GetPhyAddr() + expertIndex * expertCounterStride_ +
        assignedRowGroup * vectorCoreCount_ * INT_CACHELINE + vectorCoreIndex_ * INT_CACHELINE;
    while (AscendC::ShmemMegaMoeReadGmBypassDCache(completionCounter) != requiredTileCount) {
        AscendC::Nop<200>();
    }
```

`AscendC::Nop<200>()` is the backoff in the spin. Note that here the comparison *is* `!=` —
because this counter is reset between rounds (`ResetRowGroupFlags()`), unlike the never-reset
generation counter in §11.2.

### 11.4 The remote write is a PLAIN `DataCopyPad` — no library call in the hot path

`kernel/arch35/shmem_mega_moe_combine_pipeline.h`, `CombineTokens`, verbatim:

```cpp
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(
    uint32_t rowOffset, uint32_t columnOffset, uint32_t n, LocalTensor<int32_t>& routingMetadata,
    LocalTensor<ElementMMadOut2>& secondProjectionOutput, BlockShape& actualBlockShape,
    const ShmemMegaMoeKernelParams& params)
{
    int32_t tileRowCount = Get<SHAPE_ROW_INDEX>(actualBlockShape);
    AscendC::GlobalTensor<ElementMMadOut2> remoteOutput;
    uint64_t combinedTokenOffset = params.symmetricMemory.combinedTokens - params.symmetricMemory.localBase;
    AscendC::DataCopyExtParams outputCopyParams{1, 0, 0, 0, 0};
    outputCopyParams.blockCount = 1;
    outputCopyParams.blockLen = Get<SHAPE_COLUMN_INDEX>(actualBlockShape) * sizeof(ElementMMadOut2);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t tileRowIndex = 0; tileRowIndex < tileRowCount; ++tileRowIndex) {
        uint32_t routingRecordOffset = tileRowIndex * ROUTING_RECORD_WIDTH;
        uint32_t destinationRank = routingMetadata.GetValue(routingRecordOffset + RANK_ID);
        uint32_t tokenIndex = routingMetadata.GetValue(routingRecordOffset + TOKEN_ID);
        uint32_t routeSlotIndex = routingMetadata.GetValue(routingRecordOffset + ROUTE_SLOT_INDEX);
        remoteOutput.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(
            GetRemoteAddress(params.symmetricMemory.localBase, destinationRank, combinedTokenOffset)));
        uint64_t destinationOffset =
            (static_cast<uint64_t>(tokenIndex) * params.config->expertsPerToken + routeSlotIndex) * n + columnOffset;
        AscendC::DataCopyPad(
            remoteOutput[destinationOffset],
            secondProjectionOutput[tileRowIndex * Get<SHAPE_COLUMN_INDEX>(actualBlockShape)], outputCopyParams);
    }
}
```

`remoteOutput` is an ordinary `AscendC::GlobalTensor` whose base address is
`aclshmem_ptr(localBase, destinationRank)`. The write is an ordinary `AscendC::DataCopyPad` on
MTE3. **There is no `aclshmem_put` in the data path at all.** The library is used once, to
translate the symmetric base address; after that a peer's HBM is just a `__gm__` pointer.

The dispatch side reads a peer's memory the same way (`kernel/arch35/shmem_mega_moe_kernel.h`:581):

```cpp
    remoteQuantizedRows.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(
        GetRemoteAddress(params_.symmetricMemory.localBase, sourceRank, quantizedTokenOffset_)));
```

and the mask publish (line 448):

```cpp
        outputGlobal.SetGlobalBuffer(
            (__gm__ uint8_t*)GetRemoteAddress(params_.symmetricMemory.localBase, dstRank, dstOffset));
        DataCopyPad(outputGlobal, maskBuffer, maskCopyConfig);
```

### 11.5 The AIC↔AIV flag discipline inside the same megakernel

`kernel/arch35/shmem_mega_moe_kernel.h` — the exact `CrossCore*Flag` calls, with flag IDs:

```cpp
    // AIV side, end of CountReceivedTokens():
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(2);
```

```cpp
    if constexpr (Mode == ShmemMegaMoeMatmulStage::kFirstProjection) {
        if constexpr (g_coreType == AscendC::AIC) {
            CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(2 + 16);
            uint64_t expertCountOffset =
                static_cast<uint64_t>(expertIndex) * 8U * cubeCoreCount_ + 8U * cubeBlockIndex_;
            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                receivedTokenCountsGlobal_[expertCountOffset]);
            Get<SHAPE_ROW_INDEX>(state.problemShape) = receivedTokenCountsGlobal_.GetValue(expertCountOffset);
            CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_S>(3);
        }
        if constexpr (g_coreType == AscendC::AIV) {
            if (vectorSubBlockIndex_ == 0) {
                CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(3);
                uint64_t expertCountOffset =
                    static_cast<uint64_t>(expertIndex) * 8U * cubeCoreCount_ + 8U * cubeBlockIndex_;
                DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                    receivedTokenCountsGlobal_[expertCountOffset]);
                Get<SHAPE_ROW_INDEX>(state.problemShape) = receivedTokenCountsGlobal_.GetValue(expertCountOffset);
            } else {
                Get<SHAPE_ROW_INDEX>(state.problemShape) = receivedTokenCount_;
            }
        }
    }
```

Three observations. (a) The mode is a single named constant `SYNC_AIC_AIV_MODE` used for **both**
directions — AIV→AIC (set on `PIPE_MTE3`, flag 2) and AIC→AIV (set on `PIPE_S`, flag 3). (b) The
AIC waits on flag `2 + 16`, i.e. the **AIV sub-block offset is 16**: sub-block 0 writes flag `n`,
sub-block 1 writes flag `n + 16`. On arch35 the paired AIVs share the Cube core's flag space and
are distinguished by that `+16`. (c) A `DataCacheCleanAndInvalid<..., CacheLine::ENTIRE_DATA_CACHE,
DcciDst::CACHELINE_OUT>` is needed on **both** sides around the flag before the payload is read —
the flag orders, the dcci makes the payload visible.

Core indices are derived as (lines 117–118):

```cpp
    uint32_t cubeBlockIndex_ = GetBlockIdx() / GetTaskRation();
    uint32_t vectorCoreIndex_ = GetBlockIdx();
```

and the atomic-counter wave flag (line 701):

```cpp
            __gm__ int32_t* waveReadyFlags = matmulBuffers.waveReadyFlags;
            for (int32_t w = firstWaveIndex; w <= finalWaveIndex; ++w) {
                ...
                AtomicAdd(waveReadyFlags + w, dispatchedRows);
            }
```

### 11.6 `Process()` — the whole forward pass, one function

```cpp
template <SHMEM_MEGA_MOE_TEMPLATE_DECL>
__aicore__ inline void ShmemMegaMoePipeline<SHMEM_MEGA_MOE_TEMPLATE_ARGS>::Process()
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    InitSendQuantBuffers();
    BuildSendMasks();
    ResetPipelineFlags();
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        ResetRowGroupFlags();
    }
    QuantizeLocalTokens();
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }
    SynchronizeRanks();

    InitDispatchBuffers();
    ...
    for (int localExpertId = 0; localExpertId < localExpertCount_; localExpertId++) {
        if constexpr (g_coreType == AIV) {
            if (vectorSubBlockIndex_ == 1) {
                CountReceivedTokens(localExpertId);
            }
        }
        if (!PrepareExpertProblem<ShmemMegaMoeMatmulStage::kFirstProjection>(firstProjectionState, localExpertId)) {
            continue;
        }
        UpdateMatmulBuffers<ShmemMegaMoeMatmulStage::kFirstProjection>(matmulBuffers, firstProjectionState);
        if constexpr (g_coreType == AIV) {
            if (vectorSubBlockIndex_ == 1) {
                BuildRoutingMetadataAndDispatch(matmulBuffers, localExpertId);
            }
        }
        if (vectorSubBlockIndex_ == 0) {
            ShmemMegaMoeKernel::RunFirstProjection<
                QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                epilogueOp_, firstProjectionState.problemShape, matmulBuffers, schedulerStartBlock_, vectorSyncState_);
        }
    }
    FinishFirstProjectionSync(vectorSyncState_);
    if constexpr (g_coreType == AIV) {
        if (vectorSubBlockIndex_ == 1) {
            CopyExpertTokenCounts();
        }
    }

    vectorSyncState_ = 0;
    processedTokenCount_ = 0;
    ExpertPipelineState secondProjectionState{initShape, initOffset};
    InitCombineBuffers();

    for (uint32_t expertIndex = 0; expertIndex < localExpertCount_; expertIndex++) {
        if (!PrepareExpertProblem<ShmemMegaMoeMatmulStage::kSecondProjection>(secondProjectionState, expertIndex)) {
            continue;
        }
        UpdateMatmulBuffers<ShmemMegaMoeMatmulStage::kSecondProjection>(matmulBuffers, secondProjectionState);

        ShmemMegaMoeKernel::RunSecondProjection<
            CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
            params_, secondProjectionState.problemShape, matmulBuffers, schedulerStartBlock_, vectorSyncState_,
            processedTokenCount_, secondProjectionBufferIndex_, expertIndex, expertCounterStride_);

        if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
            RunCombine(matmulBuffers, secondProjectionState, expertIndex);
        }
    }
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        FinishSecondProjectionSync(vectorSyncState_, secondProjectionBufferIndex_);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    if constexpr (g_coreType == AIV) {
        InitOutputBuffers();
        SynchronizeRanks();
        CombineRoutedTokens();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
```

Count the global barriers in one MoE layer: `SyncAll<true>()` ×3 plus `SynchronizeRanks()` ×2
(each of which itself ends in a `SyncAll<true>()`). So five full-core barriers and two cross-rank
rendezvous for a whole MoE layer including dispatch and combine.

Shapes used in the README's run script: `TOKEN_COUNT=256 MODEL_DIM=4096 FFN_DIM=1024
EXPERTS_PER_TOKEN=6 LOCAL_EXPERT_COUNT=4`, `RANK_SIZE=2`, 1 GiB symmetric memory,
`MAX_RECEIVED_TOKENS` auto = `token_count * rank_size * min(experts_per_token, local_expert_count)`.
Numerics tolerance: E4M3 5% relative, E5M2 15% relative, absolute 1.

## Relevance

This is the design we want, written by the vendor, with the exact code for the part we were going
to invent. Four transfers:

1. **The epoch-tagged protocol is `>=` on a never-reset monotone generation counter, read and
   written with `ReadGmBypassDCache` / `WriteGmBypassDCache`, one 64-byte cache line per rank, in
   the first 48 KiB of symmetric memory.** The in-source comment names the exact failure mode of
   the naive version ("equality can wait forever after the value has advanced past generation").
   That is §2 of our brief, answered from first-party source.
2. **A peer's symmetric memory is just a `__gm__` pointer.** `GetRemoteAddress()` =
   `aclshmem_ptr(localBase, rank) + offset`, then ordinary `DataCopyPad` on MTE3. Our 49.3 us MoE
   chain can write a peer's HBM with the MTE3 instructions it already issues; no library call
   enters the hot loop, so no UB-budget or single-core-writer penalty from
   `aclshmem_putmem_signal` (§9.6).
3. **The cross-rank barrier is AIV-only by construction** — `if constexpr (g_coreType == AIC)
   { return; }`. This is a *deliberate departure* from `shmem_device_cc.h`'s own contract
   (§9.2: the all-core barrier "require[s] a MIX kernel; all participating local AIC and AIV cores
   must enter"). The vendor's own megakernel does not use `aclshmem_barrier_all()` in the hot
   path; it hand-rolls precisely so Cube never enters the rendezvous. Given our measurement that
   `SyncAll` already eats 11–13 of 16 flags, that is the right trade: the cross-rank part costs
   zero FFTS flags, because it is GM polling, not flags.
4. **The AIV sub-block flag offset is 16** (`CrossCoreWaitFlag<...>(2 + 16)`), matching MegaGDN's
   A5 `SYNC_FLAG_ID_MAX` pattern in §8.4. On dav-2201 we measured 16 flags per core; on arch35
   the same base-plus-16 scheme is used to address the two sub-blocks, which is consistent with
   16 being the per-sub-block flag count rather than the per-core total.

The caveat: this example is `Ascend950` only (`shmem_mega_moe 仅支持 Ascend950`) and uses Catlass
arch35 GEMMs and MXFP8. But §11.2–§11.5 are platform-independent AscendC: `aclshmem_ptr`,
`DataCopyPad`, `ReadGmBypassDCache`, `CrossCoreSetFlag`, `AtomicAdd`, `Nop<200>`. Nothing in the
barrier protocol needs A5.

## 12. The MTE engine layer — `aclshmemx_mte_*`: remote put/get/atomic-add issued from AscendC, and the ONE atomic that works on A2/A3

Header: https://raw.gitcode.com/cann/shmem/raw/master/include/device/gm2gm/engine/shmem_device_mte.h
Doc: https://raw.gitcode.com/cann/shmem/raw/master/docs/api/atomic_api_sync_async_comparison.md
Author: Huawei Technologies Co., Ltd. (`Copyright (c) 2025`)
Date: current `master` as of 2026-10-20 (SHMEM v1.7.0)

This is the lowest layer, the one that matters for A3. The high-level `aclshmem_*` RMA calls
dispatch to this on a single machine ("The high-level API selects MTE for same-machine PEs and
RDMA for cross-machine PEs" — §10.4).

### 12.1 `aclshmem_ptr` — the whole symmetric-memory abstraction, in one function

```cpp
/**
 * @brief Translate a local symmetric address to the corresponding symmetric address on the specified PE.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] Target PE number.
 * @return The corresponding symmetric address on the specified PE.
 * @note The supported access method for the returned address depends on the transport and topology.
 */
ACLSHMEM_DEVICE __gm__ void* aclshmem_ptr(__gm__ void* ptr, int pe);
#define shmem_ptr aclshmem_ptr
```

That is it. It lives in the *MTE engine* header, is callable from device code, and returns a
`__gm__ void*` you can hand to `GlobalTensor::SetGlobalBuffer` (§11.4).

### 12.2 `aclshmemx_mte_put_nbi` / `get_nbi` — four overloads each, with the `sync_id` argument

```cpp
/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local
 * device.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, uint32_t elem_size, int pe, uint32_t sync_id);

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size,
    const non_contiguous_copy_param& copy_params, int pe, uint32_t sync_id);
#define shmem_mte_get_mem_nbi aclshmemx_mte_get_nbi

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, uint32_t elem_size, int pe, uint32_t sync_id);

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, const non_contiguous_copy_param& copy_params,
    int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync pipeline.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id);

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf,
    const non_contiguous_copy_param& copy_params, int pe, uint32_t sync_id);
#define shmem_mte_put_mem_nbi aclshmemx_mte_put_nbi

/**
 * @brief Asynchronous interface. Clear instruction pipes and flush data cache to GM.
 */
ACLSHMEM_DEVICE void aclshmemx_mte_quiet();
```

Note what `aclshmemx_mte_quiet()` actually is: **"Clear instruction pipes and flush data cache to
GM."** That is a local pipe barrier plus a cache flush — not a network fence. Cheap.

### 12.3 `aclshmemx_mte_atomic_add` — THE A2/A3 PRIMITIVE, verbatim with its platform table

```cpp
/**
 * @brief Atomic add operation. Adds the value to the destination address.
 *        WARNING: Use sync_id in device_state.mte_config for pipeline synchronization.
 *        NOTE: This is an asynchronous interface. The final operation differs by platform and type:
 *              - On Ascend910B/Ascend910C: The atomic add is performed by MTE2 unit.
 *              - On Ascend950:
 *                - uint32_t/int64_t/uint64_t: scalar AtomicAdd (Scalar unit).
 *                - Other types: UB + MTE3 (MTE3 atomic add).
 *              If there are data dependencies, developers need to insert synchronization according
 *              to actual situations. Scalar write to UB before MTE3 read requires S_MTE3 event
 *              synchronization; MTE3 write to GM before subsequent reads requires MTE3_MTE2
 *              event synchronization.
 *
 * | Data Type                      | Ascend910B/Ascend910C | Ascend950 |
 * |--------------------------------|-----------------------|-----------|
 * | int8_t/int16_t/half/bfloat16_t | ✓                     | ✓         |
 * | int32_t/float                  | ✓                     | ✓         |
 * | uint32_t/int64_t/uint64_t      | ✗                     | ✓         |
 *
 * @note The MTE transport operates over the in-die interconnect and does not support
 *       cross-PCIe (inter-node) communication.
 *
 * @param dst               [in] Symmetric address of the destination data.
 * @param value             [in] Value to be added.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_atomic_add(__gm__ T* dst, T value, int32_t pe);
```

The other MTE atomics all carry `Supported hardware platform: Ascend950.` in their doc comment —
`aclshmemx_mte_atomic_fetch`, `_set`, `_swap`, `_compare_swap`, `_inc`, `_fetch_inc`,
`_fetch_add`. Each also repeats:

> @note The MTE transport operates over the in-die interconnect and does not support
>       cross-PCIe (inter-node) communication.

### 12.4 `docs/api/atomic_api_sync_async_comparison.md` — the authoritative per-platform table

> # ACLSHMEM Atomic 原子操作接口用法语义总结
>
> > 本文档基于实际代码实现编写，覆盖各层接口：标准接口 `aclshmem_<type>_atomic_<op>`、MTE 后端 `aclshmemx_mte_atomic_<op>`、RDMA 后端 `aclshmemx_roce_atomic_<op>`。
>
> ## 一、接口层次关系
>
> ```text
> aclshmem_<type>_atomic_<op>()          ← 标准接口
>     │
>     ├── topo == MTE  → aclshmemx_mte_atomic_<op>()   ← MTE 后端
>     └── topo == ROCE → aclshmemx_roce_atomic_<op>()  ← RDMA 后端
> ```
>
> 标准接口根据 `device_state->topo_list[pe]` 自动分派到 MTE 或 RDMA 后端，并在后端调用之上额外添加同步保证。

> ## 二、MTE 后端接口：`aclshmemx_mte_atomic_<op>`
>
> > 头文件：`include/device/gm2gm/engine/shmem_device_mte.h`
> > 实现文件：`src/device/gm2gm/engine/shmem_device_mte.hpp`
>
> **核心特征：所有 MTE 后端接口均为异步**，不论是否带返回值。接口仅发起硬件原子操作即返回，无 quiet/fence。
>
> 同步原语：`aclshmemx_mte_quiet()` 或 Kernel Runtime 级 `SetFlag/WaitFlag`。
>
> ### 2.1 接口清单
>
> | 接口完整名 | 返回值 | 支持类型 | 硬件平台 |
> |---|---|---|---|
> | `aclshmemx_mte_atomic_fetch` | T | int32, uint32, float, int64, uint64 | 950 |
> | `aclshmemx_mte_atomic_set` | void | uint32, uint64 | 950 |
> | `aclshmemx_mte_atomic_swap` | T | uint32, uint64 | 950 |
> | `aclshmemx_mte_atomic_compare_swap` | T | uint32, uint64 | 950 |
> | `aclshmemx_mte_atomic_inc` | void | int32, uint32, float, int64, uint64 | 950 |
> | `aclshmemx_mte_atomic_add` | void | int8, int16, bf16, half, int32, float (A2/A3/950) + uint32, uint64, int64 (950) | A2/A3/950 |
> | `aclshmemx_mte_atomic_fetch_inc` | T | int32, uint32, float, int64, uint64 | 950 |
> | `aclshmemx_mte_atomic_fetch_add` | T | int32, uint32, float, int64, uint64 | 950 |
>
> ### 2.2 同步要求
>
> 所有 MTE 后端接口调用后，如果存在**标量计算单元（Scalar）与搬运单元（MTE2/MTE3）之间的数据依赖**，开发者需根据实际数据流插入同步：
>
> | 时机 | 场景 | 同步方式 |
> |---|---|---|
> | 调用前 | 其他单元（如 MTE2）正在写同一 UB 区域 | `SetFlag/WaitFlag` on `MTE2_S` |
> | 调用后 | 后续操作依赖原子操作在 GM 中的结果 | `SetFlag/WaitFlag` on `MTE3_MTE2` |
> | 调用后 | 需要复用同一 UB 区域写入新数据 | `SetFlag/WaitFlag` on `MTE3_S` |
>
> ### 2.3 `aclshmemx_mte_atomic_add` 的类型差异（重要）
>
> `aclshmemx_mte_atomic_add` 在 Ascend950 上按类型有两种实现路径：
>
> | 类型组 | 支持类型 | 实现方式 | 同步要点 |
> |---|---|---|---|
> | UB + MTE3 路径 | int8, int16, bf16, half, int32, float | Scalar 写操作数到 UB → MTE3 读 UB → MTE3 atomic add 写 GM | 需关注 UB 相关 fence（MTE2_S、MTE3_S）和 GM fence（MTE3_MTE2） |
> | Scalar AtomicAdd 路径 | uint32, uint64, int64 | Scalar 单元直接执行标量 AtomicAdd 写 GM，不经过 UB 和 MTE3 | 不需要 UB/MTE3 相关 fence；仅在 Scalar↔MTE2/MTE3 有 GM 数据依赖时按通用规则同步 |

And on the standard-interface layer, the note that symmetric memory is an *interface* convention,
not an engine restriction:

> 标准 AMO 接口会将用户传入的 `dst`、`dest` 或 `source` 交给 `aclshmem_ptr(..., pe)` 换算为指定 PE 上的
> 目标地址，因此这些参数必须指向对称内存。该要求来自标准接口的地址换算方式，并不表示 MTE 引擎本身
> 只能访问对称内存；传输层自动分派不会改变这一接口语义。

> ```text
> aclshmem_<type>_atomic_<op>(dst, val, pe)
>     │
>     ├── topo_list[pe] & MTE
>     │   ├── fetch 变体:    调用 aclshmemx_mte_atomic_fetch_<op>() + PipeBarrier<PIPE_ALL>
>     │   ├── swap/cas:      调用 aclshmemx_mte_atomic_swap/compare_swap() + PipeBarrier<PIPE_ALL>
>     │   ├── add/inc:       透传 aclshmemx_mte_atomic_add/inc()，未加 PipeBarrier
>     │   └── set/and/or/xor: 透传底层异步接口，未加 PipeBarrier
>     │
>     └── topo_list[pe] & ROCE
>         ├── fetch 变体:    透传 aclshmemx_roce_atomic_fetch_<op>()（后端已 quiet）
>         ├── swap/cas:      透传 aclshmemx_roce_atomic_swap/compare_swap()（后端已 quiet）
>         ├── add/inc:       透传 aclshmemx_roce_atomic_add/inc()（后端异步，未加 quiet）
>         └── set/and/or/xor: 透传底层异步接口（未加 quiet）
> ```

And the RDMA concurrency warning, for completeness (does not apply to single-machine TP16):

> ### 3.3 并发限制
>
> > **警告**：使用 RDMA 作为底层传输时，不支持对同一 PE 的并发 RMA/AMO 操作。请使用 `device_state.rdma_config` 中的 `sync_id` 进行流水线同步。

## Relevance

This is the single most actionable finding in this whole pass. On Ascend910B/Ascend910C — our
part — `aclshmemx_mte_atomic_add` is **supported**, works on **bf16, half, int8, int16, int32 and
float**, and **"the atomic add is performed by MTE2 unit"**. Every other MTE atomic is
Ascend950-only.

That means the one-sided all-reduce does not need a reduce-scatter/all-gather structure, a
staging buffer, a flag protocol, or a receiver-side add at all. The minimal TP16 all-reduce on A3
is: each rank atomically adds its own partial-sum tiles into *the same symmetric output buffer on
each of the 16 peers*, i.e. `aclshmemx_mte_atomic_add(dst_sym, value, pe)` for `pe = 0..15`, then
one epoch-tagged rendezvous (§11.2) to know all 16 contributions have landed. The reduce happens
in the interconnect/MTE2 path, not in anybody's vector unit. For a 4096-wide bf16 activation at
batch=1 the payload per peer is 8 KB.

Three facts to carry into the design:
- The atomic add is **asynchronous and has no implicit fence** ("所有 MTE 后端接口均为异步 … 接口仅
  发起硬件原子操作即返回，无 quiet/fence"). Ordering is ours: `MTE3_MTE2` event before any read of
  the result, `MTE2_S` before, `MTE3_S` to reuse the UB.
- `aclshmemx_mte_quiet()` is only "clear instruction pipes and flush data cache to GM" — a local
  operation. It does not wait on peers. The peer-arrival wait must be the §11.2 generation
  counter.
- "The MTE transport operates over the in-die interconnect and does not support cross-PCIe
  (inter-node) communication" — fine for one machine, and a hard stop for anything beyond it.

Unverified on our side: whether a *remote* `atomic_add` across HCCS on 910C has the same cost as
the familiar local `AtomicAdd`, and whether 16 ranks all atomically adding into the same cache
line serialise. The 64-byte-per-rank layout in §11.2 exists precisely to avoid that for flags;
the same padding discipline presumably applies to the payload if it is small.

## 13. `AscendC::Hccl` — the in-kernel HCCL client, official API reference (asc.gitcode.com, built 2026/10/20)

URL root: https://asc.gitcode.com/api/SIMD-API/adv_api/HCCL_communication/
Pages quoted: `HCCL_Kernel/HCCL_usage.html`, `HCCL_Kernel/HCCL_template_params.html`,
`HCCL_Kernel/AllReduce.html`, `HCCL_Kernel/Commit.html`, `HCCL_Kernel/Wait.html`,
`HCCL_Kernel/InterHcclGroupSync.html`, `HCCL_Kernel/BatchWrite.html`,
`HCCL_Kernel/GetWindowsInAddr.html`
Source repo: https://gitcode.com/cann/asc-devkit/ (branch `master`)
Page footer, verbatim: 「版本 master 提交 5f50fc4c6ada构建于 2026/10/20 11:47:29包含未提交修改」
Disclaimer footer, verbatim: 「免责声明：本站内容由 asc-devkit 仓 master 分支自动编译生成，属于持续开发版本，可能存在缺陷，仅供预览与参考。如需稳定及商用资料，请查阅官方 昇腾社区。」
Retrieved 2026-10-20. This is the newest copy of the Ascend C API reference in existence —
built the same day.

Full page list under `HCCL_communication/`, for anyone needing a page not quoted here:

```
HCCL-Context/{GetHcclContext, SetHcclContext, HCCL_Context_intro}
HCCL_Kernel/{HCCL_usage, HCCL_template_params, InitV2, Init_deprecated,
             SetCcTilingV2, SetCcTiling_deprecated,
             AllReduce, AllGather, ReduceScatter, AlltoAll, AlltoAllV, AlltoAllvWrite, BatchWrite,
             Commit, Wait, Iterate, Query, InterHcclGroupSync, Finalize,
             GetWindowsInAddr, GetWindowsOutAddr, GetRankId, GetRankDim, GetQueueNum, QueueBarrier}
HCCL_MC2/{HCCL_MC2_usage, Mc2AcquireCcResCtx, Mc2CcKernelLaunch, Mc2GetCcArgs, Mc2FreeCcArgs,
          Mc2SetCcAlgConfig, Mc2SetCcCommEngine, Mc2SetCcSrcDataType, Mc2SetCcDstDataType,
          Mc2SetCcReduceType, CheckOpResSufficient}
HCCL_Tiling/{HCCL_Tiling_usage, HCCL_Tiling_constructor, GetTiling, TilingData_struct,
             SetAlgConfig, SetCommBlockNum, SetCommEngine, SetDebugMode, SetGroupName, SetOpType,
             SetQueueNum, SetReduceType, SetSkipBufferWindowCopy, SetSkipLocalRankCopy,
             SetStepSize, CheckOpResSufficient}
```

### 13.1 `HCCL使用说明` — the mechanism, verbatim; and the key disappointment

> HCCL使用说明
>
> Ascend C提供一组HCCL通信类高阶API，方便算子核函数（Kernel）开发用户在AI Core侧灵活管理通算融合算子中计算与通信任务的执行顺序。
>
> HCCL为集合通信任务客户端，主要对外提供了集合通信原语接口（以下统称为Prepare接口），对标集合通信C++接口，当前支持AllReduce、AllGather、ReduceScatter、AlltoAll、AlltoAllV、AlltoAllvWrite、BatchWrite接口。**本章的所有接口运行在AI Core上，且不执行通信任务，而是由用户调用Prepare接口将对应类型的通信任务信息发送给AI CPU或CCU服务端，并在合适的时机通过Commit接口通知AI CPU或CCU上的服务端执行对应的通信任务。** 注意，当前Ascend 950PR&950DT系列产品上仅支持CCU服务端。
>
> 所谓合适的时机，取决于用户编排的是先通信后计算的任务，还是先计算后通信的任务。对于这两种场景，简述如下：
>
> 先通信后计算的任务：典型的如AllGather通信+Matmul计算任务编排。此场景下，用户在调用AllGather接口下发通信任务之后，通过AllGather接口返回的该通信任务标识handleId，可立即调用Commit接口通知服务端执行该handleId对应的任务，同时用户调用Wait阻塞接口等待服务端通知handleId对应的通信任务执行结束，待该通信任务结束后，再执行计算任务。
>
> 先计算后通信的任务：典型的如Matmul计算+AllReduce通信任务编排。此场景下，用户可以先调用AllReduce接口通知服务端先下发通信任务，再调用Matmul计算接口进行计算，这样AllReduce任务的组装和任务下发及执行过程可以被Matmul的计算流水所掩盖，待计算任务完成后，调用Commit接口通知服务端执行AllReduce任务，无须调用Wait接口等待通信任务执行结束。
>
> 当后续无通信任务时，调用Finalize接口，通知服务端后续无通信任务，执行结束后退出，客户端检测并等待最后一个通信任务执行结束。

> 注意
>
> 对于Atlas A3系列产品，在AI CPU作为服务端的场景中，HCCL通信API的功能依赖开放AI CPU用户态下发调度任务，存在一定的安全风险，用户需要自行确保AI Core自定义算子的安全可靠，防止恶意攻击行为。

That first bolded sentence is the whole architecture: **the AI Core does not perform the
collective.** It builds a task descriptor and rings a doorbell; an **AI CPU** (on A3) or a
**CCU** (on A5) server executes it. This is a device-side *submission*, not a device-side
collective.

The five steps, verbatim with the code:

> 实现AI Core下发一个通信任务的具体步骤如下：
>
> 创建HCCL对象，并调用初始化接口InitV2。
>
> ```
> // 传initTiling地址的调用方式，推荐使用该方式
> GET_TILING_DATA_WITH_STRUCT(
>     AllGatherCustomTilingData, tilingData, tilingGM); // AllGatherCustomTilingData为对应算子头文件定义的结构体
>
> Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl; // 通过模板入参的方式选择硬件类型
> GM_ADDR contextGM = GetHcclContext<0>(); // AscendC自定义算子核函数（Kernel）中，通过此方式获取HCCL context
>
> hccl.InitV2(contextGM, &tilingData);
> ```
>
> 当调用InitV2接口时，必须使用标准C++语法定义TilingData结构体的开发方式。
>
> 设置对应通信算法的Tiling地址。通过SetCcTilingV2接口设置对应通信算法的Tiling地址，调用Commit接口后，该地址被发送到服务端由服务端解析。SetCcTilingV2接口必须与InitV2接口配合使用。
>
> ```
> if (SetCcTilingV2(offsetof(AllGatherCustomTilingData, mc2CcTiling)) != HCCL_SUCCESS) {
>     return;
> }
> ```
>
> 用户通过对应的Prepare接口异步下发对应类型的通信任务，并获取到该任务的标识handleId，服务端接收到后开始通信任务的展开和下发。
>
> ```
> auto handleId =
>     hccl.ReduceScatter<false>(aGM, cGM, recvCount, AscendC::HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, strideCount, 1);
> ```
>
> 用户调用Commit接口通知服务端执行handleId对应的通信任务。
>
> ```
> // 等待通信任务执行时机成熟，调用Commit接口通知服务端执行
> hccl.Commit(handleId);
> ```
>
> 用户调用Wait阻塞接口，等待服务端执行完对应的通信任务。
>
> ```
> auto ret = hccl.Wait(handleId);
>
> // 调用核间同步接口，防止部分核执行较快退出，触发Hccl析构，影响执行较慢的核
> // 开发者可根据实际的业务场景，选择调用SyncAll、CrossCoreSetFlag、CrossCoreWaitFlag接口，保证全部核的任务完成后再退出执行
> ```
>
> 用户调用Finalize接口，通知服务端后续无通信任务，执行结束后退出；客户端检测并等待最后一个通信任务执行结束。
>
> ```
> hccl.Finalize();
> ```
>
> 注意：若HCCL对象的模板参数未指定下发通信任务的核，则Prepare接口仅能在AIC或AIV之一上运行，调用步骤2到步骤5的接口前，必须指定接口代码运行在AIC或AIV核上，实现时如下代码所示。
>
> ```
> // 通过内置常量g_coreType来判断AIC核或者AIV核
> if (g_coreType == AIV) {
> // if (g_coreType == AIC) {
>         调用HCCL接口
> }
> ```

The two enums, verbatim:

```cpp
enum HcclDataType {
    HCCL_DATA_TYPE_INT8 = 0,     /* int8 */
    HCCL_DATA_TYPE_INT16 = 1,    /* int16 */
    HCCL_DATA_TYPE_INT32 = 2,    /* int32 */
    HCCL_DATA_TYPE_FP16 = 3,     /* half或float16 */
    HCCL_DATA_TYPE_FP32 = 4,     /* float */
    HCCL_DATA_TYPE_INT64 = 5,    /* int64 */
    HCCL_DATA_TYPE_UINT64 = 6,   /* uint64 */
    HCCL_DATA_TYPE_UINT8 = 7,    /* uint8 */
    HCCL_DATA_TYPE_UINT16 = 8,   /* uint16 */
    HCCL_DATA_TYPE_UINT32 = 9,   /* uint32 */
    HCCL_DATA_TYPE_FP64 = 10,    /* float64 */
    HCCL_DATA_TYPE_BFP16 = 11,   /* bfloat16 */
    HCCL_DATA_TYPE_INT128 = 12,  /* int128预留类型，暂不支持 */
    HCCL_DATA_TYPE_HIF8 = 14,    /* hif8 */
    HCCL_DATA_TYPE_FP8E4M3 = 15, /* fp8e4m3 */
    HCCL_DATA_TYPE_FP8E5M2 = 16, /* fp8e5m2 */
    HCCL_DATA_TYPE_FP8E8M0 = 17, /* fp8e8m0 */
    HCCL_DATA_TYPE_RESERVED      /* reserved */
};

enum HcclReduceOp {
    HCCL_REDUCE_SUM = 0,  /* sum */
    HCCL_REDUCE_PROD = 1, /* prod */
    HCCL_REDUCE_MAX = 2,  /* max */
    HCCL_REDUCE_MIN = 3,  /* min */
    HCCL_REDUCE_RESERVED  /* reserved */
}
```

### 13.2 `HCCL模板参数` — pinning the comm client to ONE named core

产品支持情况, verbatim:

> Ascend 950PR&950DT系列产品：支持
> Atlas A3系列产品：支持
> Atlas A2系列产品：支持
> Atlas 200I/500 A2推理产品：不支持
> Atlas推理系列产品AI Core：不支持
> Atlas推理系列产品Vector Core：不支持
> Atlas训练系列产品：不支持

```cpp
template <HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU, const auto& config = DEFAULT_CFG>
class Hccl;
```

> | 参数名称 | 描述
> | serverType | 支持的服务端类型。HcclServerType枚举类型，取值如下。
> HCCL_SERVER_TYPE_AICPU：AI CPU服务端。
> HCCL_SERVER_TYPE_CCU：CCU服务端。
> HCCL_SERVER_TYPE_END：预留参数，不支持使用。
>
> 对于Ascend 950PR&950DT系列产品，当前仅支持HCCL_SERVER_TYPE_CCU。
> **对于Atlas A3系列产品，当前仅支持HCCL_SERVER_TYPE_AICPU。**
> 对于Atlas A2系列产品，当前仅支持HCCL_SERVER_TYPE_AICPU。
>
> | config | 用于指定向服务端下发任务的核。HcclServerConfig类型，定义如下代码所示，含义如下。
>
> type：向服务端下发任务的核的类型。CoreType类型，可取值为：
> DEFAULT：表示不指定AIC核或者AIV核。
> ON_AIV：表示指定为AIV核。
> ON_AIC：表示指定为AIC核。
>
> blockId：向服务端下发任务的核的ID。
>
> 默认值DEFAULT_CFG = {CoreType::DEFAULT, 0}。

```cpp
enum HcclServerType {
    HCCL_SERVER_TYPE_AICPU = 0,
    HCCL_SERVER_TYPE_CCU,
    HCCL_SERVER_TYPE_END // 预留参数，不支持使用
};

struct HcclServerConfig {
    CoreType type;
    int64_t blockId;
};

enum class CoreType : uint8_t { DEFAULT, ON_AIV, ON_AIC };
```

> 调用示例
>
> 通过如下传入模板参数config的方式创建Hccl类对象，指定HCCL客户端仅在AIV的10号核上发送通信消息给服务端，替代通过调用GetBlockIdx接口的方式指定运行的核。
>
> ```cpp
> static constexpr HcclServerConfig HCCL_CFG = {CoreType::ON_AIV, 10};
> // 选择AICPU作为服务端
> Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU, HCCL_CFG> hccl;
>
> // 选择CCU作为服务端
> Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU, HCCL_CFG> hccl;
> ```

### 13.3 `AllReduce` — signature, A3 support, limits

> 产品支持情况
>
> Ascend 950PR&950DT系列产品：支持
> Atlas A3系列产品：支持
> Atlas A2系列产品：支持
> Atlas 200I/500 A2推理产品：不支持
> Atlas推理系列产品AI Core：不支持
> Atlas推理系列产品Vector Core：不支持
> Atlas训练系列产品：不支持

```cpp
template <bool commit = false>
__aicore__ inline HcclHandle AllReduce(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t count,
                                       HcclDataType dataType, HcclReduceOp op, uint8_t repeat = 1)
```

> | commit | 输入 | bool类型。参数取值如下：
> true：在调用Prepare接口时，Commit同步通知服务端可以执行该通信任务。
> false：在调用Prepare接口时，不通知服务端执行该通信任务。
>
> | dataType | 输入 | AllReduce操作的数据类型，目前支持float、half（即float16）、int8_t、int16_t、int32_t、bfloat16_t数据类型，即支持取值为HCCL_DATA_TYPE_FP32、HCCL_DATA_TYPE_FP16、HCCL_DATA_TYPE_INT8、HCCL_DATA_TYPE_INT16、HCCL_DATA_TYPE_INT32、HCCL_DATA_TYPE_BFP16。
> | op | 输入 | Reduce的操作类型，目前支持sum、max、min操作类型，即支持取值为HCCL_REDUCE_SUM、HCCL_REDUCE_MAX、HCCL_REDUCE_MIN。
> | repeat | 输入 | 一次下发的AllReduce通信任务个数。repeat取值≥1，默认值为1。当repeat>1时，每个AllReduce任务的sendBuf和recvBuf地址由服务端自动算出，计算公式如下：
>
> sendBuf[i] = sendBuf + count* sizeof(datatype) * i, i∈[0, repeat)
> recvBuf[i] = recvBuf + count* sizeof(datatype) * i, i∈[0, repeat)
>
> 返回值说明
> 返回该任务的标识handleId，handleId大于等于0。调用失败时，返回 -1。
>
> 约束说明
> 调用本接口前确保已调用过InitV2和SetCcTilingV2接口。
> 若HCCL对象的config模板参数未指定下发通信任务的核，该接口只能在AIC核或者AIV核两者之一上调用。**若HCCL对象的config模板参数中指定了下发通信任务的核，则该接口可以在AIC核和AIV核上同时调用，接口内部会根据指定的核的类型，只在AIC核、AIV核二者之一下发该通信任务。**
>
> 对于Atlas A2系列产品，一个通信域内，所有Prepare接口的总调用次数不能超过63。
> **对于Atlas A3系列产品，一个通信域内，所有Prepare接口和InterHcclGroupSync接口的总调用次数不能超过63。**
> 对于Ascend 950PR&950DT系列产品，一个通信域内，所有Prepare接口的总调用次数不能超过63。
> 对于Ascend 950PR&950DT系列产品，通信服务端为CCU时，最大支持8卡的全链接，默认仅支持FullMesh算法，并且单次最大通信数据量不能超过256M。

The worked example, verbatim:

```cpp
extern "C" __global__ __aicore__ void all_reduce_custom(GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    auto sendBuf = xGM;       // xGM为AllReduce的输入GM地址
    auto recvBuf = yGM;       // yGM为AllReduce的输出GM地址
    uint64_t sendCount = 300; // 每张卡上均有300个float16的数据
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    REGISTER_TILING_DEFAULT(AllReduceCustomTilingData); // AllReduceCustomTilingData为对应算子头文件定义的结构体
    GET_TILING_DATA_WITH_STRUCT(AllReduceCustomTilingData, tilingData, tilingGM);

    Hccl hccl;
    GM_ADDR contextGM = AscendC::GetHcclContext<0>(); // AscendC自定义算子核函数（Kernel）中，通过此方式获取HCCL context

    if (AscendC::g_coreType == AIV) { // 指定AIV核通信
        hccl.InitV2(contextGM, &tilingData);
        auto ret = hccl.SetCcTilingV2(offsetof(AllReduceCustomTilingData, mc2CcTiling));
        if (ret) {
            return;
        }
        HcclHandle handleId1 =
            hccl.AllReduce<true>(sendBuf, recvBuf, sendCount, HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp);
        hccl.Wait(handleId1);
        AscendC::SyncAll<true>(); // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死
        hccl.Finalize();
    }
}
```

### 13.4 `Commit` / `Wait` — exact signatures and the counting rule

```cpp
using HcclHandle = int8_t;

__aicore__ inline void    Commit(HcclHandle handleId)
__aicore__ inline int32_t Wait(HcclHandle handleId)
```

Both: 产品支持情况 — Ascend 950PR&950DT：支持 / Atlas A3：支持 / Atlas A2：支持.

> Commit — 功能说明
> 非细粒度通信时，每调用一次本接口，则通知服务端可以执行handleId对应的任务一次。细粒度通信时，每调用一次本接口，则通知服务端可以执行handleId对应的步长长度的子通信任务。
>
> 约束说明
> 非细粒度通信时，本接口的调用次数应该与Prepare的repeat次数一致。细粒度通信时，本接口的调用次数应该与通信任务的总步骤数/步长*Prepare的repeat次数一致。
> 本接口在AIC核或者AIV核上调用必须与对应的Prepare接口的调用核保持一致。

> Wait — 功能说明
> 阻塞AI Core，非细粒度通信时，等待handleId对应的通信任务完成，细粒度通信时，等待handleId对应的步长长度的子通信任务完成。handleId调用Wait接口的顺序，须和Prepare接口一致。**该接口默认在所有核上工作，用户也可以在调用前通过GetBlockIdx指定其在某一个核上运行。**
>
> 返回值说明
> 0，表示成功
> -1，表示失败

Note the asymmetry: `Commit` must be on the same core as the Prepare; `Wait` "defaults to
working on all cores" and can be narrowed with `GetBlockIdx`.

### 13.5 `InterHcclGroupSync` — cross-communication-domain dependency, A3 ONLY

> 产品支持情况
>
> Ascend 950PR&950DT系列产品：**不支持**
> Atlas A3系列产品：**支持**
> Atlas A2系列产品：**不支持**
> Atlas 200I/500 A2推理产品：不支持
> Atlas推理系列产品AI Core：不支持
> Atlas推理系列产品Vector Core：不支持
> Atlas训练系列产品：不支持
>
> 功能说明
> 用于等待一个跨通信域的通信任务完成。调用该接口后，本通信域后续下发的通信任务，均等待指定的srcGroupID通信域中的srcHandleID通信任务执行完成，才开始执行。

```cpp
__aicore__ inline void InterHcclGroupSync(int8_t srcGroupID, HcclHandle srcHandleID)
```

> 约束说明
> 调用本接口前确保已调用过InitV2和SetCcTilingV2接口。
> 本接口在AIC核或者AIV核上调用必须与对应的Prepare接口的调用核保持一致。
> 一个通信域内，所有Prepare接口和InterHcclGroupSync接口的总调用次数不能超过63。

The example, verbatim — two communication domains in one kernel, chained device-side:

```cpp
extern "C" __global__ __aicore__ void alltoall_allgather_custom(GM_ADDR xGM, GM_ADDR alltoallGM, GM_ADDR allgatherGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(
        AlltoAllAllGatherCustomTilingData); // AlltoAllAllGatherCustomTilingData为对应算子头文件定义的结构体
    GET_TILING_DATA_WITH_STRUCT(AlltoAllAllGatherCustomTilingData, tilingData, tilingGM);
    GM_ADDR contextGM0 = AscendC::GetHcclContext<0>();
    GM_ADDR contextGM1 = AscendC::GetHcclContext<1>();

    Hccl hccl0;
    Hccl hccl1;
    HcclDataType dtype = HcclDataType::HCCL_DATA_TYPE_FP16;
    const uint64_t dataCount = 10U;
    const uint64_t strideCount = 0U;
    if (AscendC::g_coreType == AIV) { // 仅使用AIV核进行通信
        hccl0.InitV2(contextGM0, &tilingData);
        hccl1.InitV2(contextGM1, &tilingData);
        hccl0.SetCcTilingV2(offsetof(AlltoAllAllGatherCustomTilingData, alltoallTiling));
        hccl1.SetCcTilingV2(offsetof(AlltoAllAllGatherCustomTilingData, allgatherTiling));

        // 通信域0下发1个AlltoAll任务
        auto group0_handle = hccl0.AlltoAll(xGM, alltoallGM, dataCount, dtype, strideCount);

        // 通信域1下发跨域依赖任务，保证通信域1后续的AllGather任务在通信域0的AlltoAll执行结束后，才开始执行
        hccl1.InterHcclGroupSync(0, group0_handle);
        // 通信域1下发1个AllGather任务
        auto group1_handle = hccl1.AllGather(alltoallGM, allgatherGM, dataCount, dtype, strideCount);

        hccl0.Commit(group0_handle);
        hccl1.Commit(group1_handle);
        hccl0.Wait(group0_handle);
        hccl1.Wait(group1_handle);

        AscendC::SyncAll<true>(); // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死
        hccl0.Finalize();
        hccl1.Finalize();
    }
}
```

### 13.6 `GetWindowsInAddr` / `GetWindowsOutAddr` — a peer rank's HCCL window as a plain `GM_ADDR`, on A3

> 产品支持情况
>
> Ascend 950PR&950DT系列产品：**不支持**
> Atlas A3系列产品：**支持**
> Atlas A2系列产品：**支持**
>
> 功能说明
> **获取卡间通信数据WindowsIn起始地址，可用来直接作为计算的输入输出地址，减少拷贝。** 该接口默认在所有核上工作，用户也可以在调用前通过GetBlockIdx指定其在某一个核上运行。

```cpp
__aicore__ inline GM_ADDR GetWindowsInAddr(uint32_t rankId)
```

> | rankId | 输入 | 待查询的卡的Id。
>
> 返回值说明
> 返回对应卡的卡间通信数据WindowsIn起始地址。当rankId非法时，返回nullptr。
>
> 约束说明
> 无

```cpp
auto winInAddr = hccl.GetWindowsInAddr(0);
auto winOutAddr = hccl.GetWindowsOutAddr(0);
auto rankId = hccl.GetRankId();
auto rankDim = hccl.GetRankDim(); // 4张卡
```

Note 「约束说明 无」 — *no constraints*. And the pair
`SetSkipBufferWindowCopy` / `SetSkipLocalRankCopy` exist on the Tiling side precisely so you can
tell HCCL not to copy into/out of the window when you have already written it yourself.

### 13.7 `BatchWrite` — the one-sided batched remote write, A3 supported / A5 NOT

> 产品支持情况
>
> Ascend 950PR&950DT系列产品：**不支持**
> Atlas A3系列产品：**支持**
> Atlas A2系列产品：**支持**
>
> 功能说明
> 集合通信BatchWrite的任务下发接口，返回该任务的标识handleId给用户。**BatchWrite实现了一种点对点通信，这是一种直接传输数据的通信模式，能够同时将多份数据发送到不同的Global Memory地址上。**
>
> 对于Atlas A3系列产品，BatchWrite通信支持在相同或不同的昇腾AI Server之间进行。
>
> 对于Atlas A2系列产品，BatchWrite通信必须在不同昇腾AI Server（通常是8卡或16卡的昇腾NPU设备组成的服务器形态的统称）之间进行。

```cpp
template <bool commit = false>
__aicore__ inline HcclHandle BatchWrite(GM_ADDR batchWriteInfo, uint32_t itemNum, uint16_t queueID = 0U)
```

The descriptor layouts — **they differ between A2 and A3**:

```cpp
// Atlas A3系列产品
struct BatchWriteItem {
    uint64_t type;
    uint32_t res1[5];
    uint32_t length;
    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;
    uint32_t dstAddrLow;
    uint32_t dstAddrHigh;
    uint32_t res2[4];
};
```

```cpp
// Atlas A2系列产品
struct BatchWriteItem {
    uint64_t localBuf;     // 本端发送数据的window地址
    uint64_t remoteBuf;    // 对端接收数据的window地址
    uint64_t count;        // 发送的数据个数
    uint32_t dataType;     // 发送的数据类型
    uint32_t remoteRankId; // 发送数据的目的卡号
};
```

> 对于Atlas A3系列产品：
> type：预留参数，取值为0。
> res1[5]：预留参数，无需填写该值。
> length：待拷贝数据的长度。
> srcAddrLow：待拷贝数据的源地址低32位。
> srcAddrHigh：待拷贝数据的源地址高32位。
> dstAddrLow：待拷贝数据的目的地址低32位。
> dstAddrHigh：待拷贝数据的目的地址高32位。
> res2[4]：预留参数，无需填写该值。
>
> | itemNum | 输入 | 批量任务的个数。该参数取值必须与batchWriteInfo中通信任务信息的组数一致。
> 对于Atlas A3系列产品，该参数取值不能大于等于2048。
> | queueID | 输入 | 指定当前通信所在的队列ID，默认值为0。Atlas A2系列产品，该参数仅支持取值为0。
>
> 约束说明
> 一个通信域内，所有Prepare接口和InterHcclGroupSync接口的总调用次数不能超过63。
> 对于Atlas A2系列产品，当前接口仅支持不同AI Server间的通信，同时通信任务信息中指定的目的卡号不能是本卡号。
> **通信任务信息写入batchWriteInfo前，必须通过调用DataCacheCleanAndInvalid接口，保证预期的数据成功刷新到Global Memory上。**

On A3 the descriptor is a bare `(length, srcAddr64, dstAddr64)` triple — **no rank field at
all**, because the destination address is already a globally resolvable address. Up to 2047
items per call.

## Relevance

`AscendC::Hccl` is the *sanctioned* in-kernel collective, it is A3-supported, and it is not what
we want — but it tells us exactly why, and it hands us two pieces we do want.

Why it is not it: 「本章的所有接口运行在AI Core上，且不执行通信任务，而是由用户调用Prepare接口将
对应类型的通信任务信息发送给AI CPU或CCU服务端」. On A3 the only server type is
`HCCL_SERVER_TYPE_AICPU`. So every layer's all-reduce is a round trip to an **AI CPU** — off the
AI Core but still off-chip-of-the-core, with the AI CPU doing task expansion and submission. That
is cheaper than a host round trip but it is not one-sided, and it caps us at **63 Prepare calls
per communication domain** — with 1 all-reduce per layer and GLM-5.2's layer count, a single
persistent kernel covering the whole forward pass would blow that 63 limit outright. That is a
hard structural argument for the §11/§12 route over this one.

What we do want from it:

1. **`static constexpr HcclServerConfig HCCL_CFG = {CoreType::ON_AIV, 10};
   Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU, HCCL_CFG> hccl;`** — pin the comm client to one
   named AIV. With that, `AllReduce`/`Commit` can be called from code compiled for all cores and
   only core 10 issues. This is exactly the "one core is the communication core" structure a
   megakernel wants, and it removes the `if (g_coreType == AIV)` / `GetBlockIdx()` guard clutter.
2. **`GetWindowsInAddr(rankId)` / `GetWindowsOutAddr(rankId)` return a peer rank's HCCL window as
   a plain `GM_ADDR`, are A3-supported, and carry 「约束说明 无」.** Combined with §11.4 (a remote
   `GM_ADDR` is writable by ordinary `DataCopyPad`) and §12.3 (`aclshmemx_mte_atomic_add` works on
   910B/910C), this means we can run the one-sided all-reduce **inside the HCCL window buffers
   that HCCL has already registered**, without `aclshmem` and without a separate symmetric heap.
   `SetSkipBufferWindowCopy` / `SetSkipLocalRankCopy` on the Tiling side exist to let us own that
   copy.
3. **`BatchWrite` on A3 takes a bare `(length, srcAddr64, dstAddr64)` descriptor with no rank
   field, up to 2047 items.** That is the shape of a scatter-to-all-peers in one submission.

One CONTRADICTION-adjacent datapoint for our flag budget: the HCCL usage page's own advice at the
end of `Wait` is 「开发者可根据实际的业务场景，选择调用SyncAll、CrossCoreSetFlag、CrossCoreWaitFlag
接口，保证全部核的任务完成后再退出执行」 — i.e. the vendor expects `SyncAll` *and* raw
`CrossCoreSetFlag`/`WaitFlag` to coexist with an HCCL object in the same kernel. If `SyncAll`
really consumes 11–13 of 16 flags as we measured, then HCCL's own internal flag usage plus a
`SyncAll` plus our own handshakes cannot all fit; either HCCL uses none, or one of the three must
go. Worth checking what `Hccl::Wait` uses internally before assuming it is free.

## 14. `HCCL MC2` host-side interfaces — how the device-side comm server is actually stood up

URL: https://asc.gitcode.com/api/SIMD-API/adv_api/HCCL_communication/HCCL_MC2/HCCL_MC2_usage.html
Source repo: https://gitcode.com/cann/asc-devkit/ branch `master`, commit `5f50fc4c6ada`
Built 2026/10/20 11:47:29. Retrieved 2026-10-20.

This page answers "what is the `Mc2*` server/client split" concretely, and it carries one
syntactic surprise worth more than the rest of the page: a kernel declared
`__global__ __vector__`.

### 14.1 The flow, verbatim

> 说明
> 本章节描述MC2内置通信算子的Host侧接口。用户通过这些接口配置通信任务、申请通信资源上下文并启动通信服务端。接口同时支持AI CPU和CCU两种通信服务端，二者共用同一组参数配置和资源上下文接口。
>
> HCCL MC2接口由Host侧参数配置、资源上下文申请和通信服务端启动三部分组成。用户在Host侧配置通信任务参数后，接口内部会完成算法选择、通信资源申请以及设备侧资源上下文的写入。
>
> **通信引擎用于选择通信服务端的执行方式：AICPU_TS表示AI CPU通信服务端，CCU_MS和CCU_SCHED表示CCU通信服务端。其中，CCU_SCHED表示CCU调度模式，建议CCU场景使用该模式。**
>
> 正式申请资源前，通信引擎设置为CCU_SCHED时，可以调用CheckOpResSufficient预检查通信资源是否充足。
>
> 典型流程如下：
>
> 调用Mc2GetCcArgs创建通信参数对象。
>
> 调用配置接口填充通信参数。
> Mc2SetCcCommEngine：设置通信引擎。
> Mc2SetCcSrcDataType：设置输入数据类型。
> Mc2SetCcDstDataType：设置输出数据类型。
> Mc2SetCcReduceType：设置Reduce操作类型，仅AllReduce、ReduceScatter等归约类任务需要配置。
> Mc2SetCcAlgConfig：设置通信算法。
>
> 调用Mc2AcquireCcResCtx申请通信资源上下文，获得设备侧资源上下文地址和大小。
>
> 调用Mc2CcKernelLaunch启动通信服务端。AI CPU路径传入独立于AI Core任务的AI CPU任务流；CCU路径可传入nullptr。
>
> 在AI Core核函数中，将步骤3返回的资源上下文传入InitV2，再调用AllGather、AllReduce、AlltoAll等HCCL高阶API下发客户端通信任务。
>
> 参数对象不再使用后，调用Mc2FreeCcArgs释放。
>
> 上述流程中的第3步与第5步必须使用同一个资源上下文。ccResCtx是Mc2AcquireCcResCtx返回的设备侧通信资源上下文地址，包含通信算法和通信资源相关信息；该资源由通信域生命周期管理，用户不需要调用单独的接口释放。
>
> Mc2GetCcArgs创建的参数对象默认通信引擎为AICPU_TS，即AI CPU通信服务端。若使用CCU路径，必须调用Mc2SetCcCommEngine显式设置通信引擎，推荐设置为CCU_SCHED。

### 14.2 The AI CPU vs CCU table — with the engine NUMBERS

> AI CPU与CCU路径的主要差异如下：
>
> | 项目 | AI CPU路径 | CCU路径 |
> |---|---|---|
> | commEngine取值 | **2，AICPU_TS**，表示AI CPU通信服务端 | **5，CCU_MS；或6，CCU_SCHED，推荐使用6** |
> | 服务端启动stream | 必须传入独立的aclrtStream | 传入nullptr，launch stream由资源上下文内部携带 |
> | AI Core侧初始化 | hccl.InitV2(contextGM) | hccl.InitV2(contextGM) |
> | HCCL模板参数 | HCCL_SERVER_TYPE_AICPU | HCCL_SERVER_TYPE_CCU |
> | 退出机制 | AI Core侧调用Finalize后服务端退出 | AI Core侧调用Finalize后服务端退出 |
> | rank数 | 必须大于1 | 必须大于1 |

### 14.3 The CCU (A5) host sequence, verbatim

```cpp
void* ccArgs = nullptr;
void* ccResCtx = nullptr;
uint32_t ccResCtxSize = 0U;
constexpr uint8_t kCcuSchedEngine = 6U; // CCU_SCHED
const uint8_t ccType = static_cast<uint8_t>(AscendC::HcclCMDType::HCCL_CMD_ALLTOALL);
char algConfig[] = "sole[mesh]";

Mc2Result ret = Mc2GetCcArgs(&ccArgs);
ret = Mc2SetCcCommEngine(ccArgs, kCcuSchedEngine);
ret = Mc2SetCcSrcDataType(ccArgs, HCCL_DATA_TYPE_FP32);
ret = Mc2SetCcDstDataType(ccArgs, HCCL_DATA_TYPE_FP32);
ret = Mc2SetCcAlgConfig(ccArgs, algConfig);

ret = Mc2AcquireCcResCtx(comm, ccType, ccArgs, &ccResCtx, &ccResCtxSize);
(void)Mc2FreeCcArgs(ccArgs);

ret = Mc2CcKernelLaunch(nullptr, ccResCtx, ccResCtxSize);

// 将ccResCtx传入AI Core核函数，并在核函数内调用InitV2和AlltoAll接口。
all_to_all_kernel<<<1, nullptr, streamAiv>>>(sendBuf, recvBuf, ccResCtx, dataCount);
```

Note `char algConfig[] = "sole[mesh]";` — the algorithm is selected by a *string*.

### 14.4 The AI CPU (A3) host sequence and the kernel-side client — note `__global__ __vector__`

> 以下示例展示AllGather任务使用AI CPU通信路径时的调用顺序。**AI CPU服务端会阻塞等待AI Core客户端消息，因此aicpuStream与streamAiv必须为两条不同的任务流，且建议先同步AI Core任务流、再同步AI CPU任务流。**

```cpp
void* ccArgs = nullptr;
void* ccResCtx = nullptr;
uint32_t ccResCtxSize = 0U;
constexpr uint8_t kAicpuTsEngine = 2U; // AICPU_TS
const uint8_t ccType = static_cast<uint8_t>(AscendC::HcclCMDType::HCCL_CMD_ALLGATHER);

Mc2Result ret = Mc2GetCcArgs(&ccArgs);
ret = Mc2SetCcCommEngine(ccArgs, kAicpuTsEngine);
ret = Mc2SetCcSrcDataType(ccArgs, HCCL_DATA_TYPE_FP32);
ret = Mc2SetCcDstDataType(ccArgs, HCCL_DATA_TYPE_FP32);

ret = Mc2AcquireCcResCtx(comm, ccType, ccArgs, &ccResCtx, &ccResCtxSize);
(void)Mc2FreeCcArgs(ccArgs);

ret = Mc2CcKernelLaunch(aicpuStream, ccResCtx, ccResCtxSize);

all_gather_kernel<<<1, nullptr, streamAiv>>>(sendBuf, recvBuf, ccResCtx, sendCount);
```

> AI Core核函数中的AI CPU客户端示例片段如下：

```cpp
__global__ __vector__ void all_gather_kernel(
    __gm__ void* sendBuf, __gm__ void* recvBuf, __gm__ void* contextGM, uint64_t sendCount)
{
    if (g_coreType != AscendC::AIV) {
        return;
    }
    AscendC::InitSocState();
    AscendC::Hccl<AscendC::HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl;
    hccl.InitV2((GM_ADDR)contextGM, nullptr);
    AscendC::HcclHandle handleId = hccl.AllGather<true>(
        (GM_ADDR)sendBuf, (GM_ADDR)recvBuf, sendCount,
        AscendC::HcclDataType::HCCL_DATA_TYPE_FP32, 0);
    hccl.Wait(handleId);
    AscendC::SyncAll<true>();
    hccl.Finalize();
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

Three things in nine lines: (a) the kernel is declared **`__global__ __vector__`**, not
`__global__ __aicore__` — a third spelling of the core-type declaration, alongside the
`KERNEL_TYPE_*` macros and `[[bisheng::core_ratio(a, b)]]` from §10.1; (b)
`AscendC::InitSocState()` is called before touching HCCL; (c) `hccl.InitV2(contextGM, nullptr)` —
the tiling pointer can be `nullptr` when the MC2 resource context already carries the algorithm.

## Relevance

The engine numbers are the useful residue: `commEngine` is `2 = AICPU_TS`, `5 = CCU_MS`,
`6 = CCU_SCHED`. On A3 only `2` exists, which closes the question of whether we can get a CCU-like
in-fabric scheduler on our part: we cannot.

The structural cost is now explicit: on A3, `AscendC::Hccl` requires **a second stream carrying a
blocking AI CPU server kernel** (`Mc2CcKernelLaunch(aicpuStream, ...)`, and 「AI CPU服务端会阻塞等待
AI Core客户端消息」). For a TPOT 2.x ms target at batch=1 that is a persistent AI CPU thread
spinning for the whole decode, plus a cross-processor handshake per all-reduce, plus the 63-Prepare
cap from §13.3. Against §12's `aclshmemx_mte_atomic_add` — which is an MTE2 instruction and needs
no server at all — the MC2 route is strictly heavier for our case. Its advantage is that it
handles algorithm selection and multi-node; we need neither.

`__global__ __vector__` is the actionable syntactic item: it is a per-kernel core-type qualifier
used in current first-party docs, and the obvious cube-side counterpart to test against our
`KERNEL_TYPE_AIC_ONLY` rejection.

## 15. `ascend_mega_kernel` — a third party building exactly our thing, with real-silicon probe results on the flag primitives

Repo: https://gitcode.com/liruixin_dvc/ascend_mega_kernel (branch `main`)
Files quoted: `README.md`, `docs/05-megakernel-design.md` (101 KB, 422 lines, "Mega Kernel 设计方案 v1.2"),
`docs/23-reference-repos.md`, `aiv_sync_probe_pack/README.md`
Author: `liruixin_dvc` (GitCode). Project start date stated in README: 探索日期 2026-09-26.
Dated entries quoted below run 2026-09-26 → 2026-10-18.
Target: Ascend950PR (dav-3510), CANN 9.1.0, single card, Qwen3.8-Flash-Next-MXFP4.
Retrieved 2026-10-20 via `https://api.gitcode.com/api/v5/repos/liruixin_dvc/ascend_mega_kernel/contents/<path>?ref=main`
(`raw.gitcode.com` returns 「暂不支持预览」 for files this size; the contents API returns base64).

Different chip from ours (3510, not 2201) and a different model, but this is the only public
project that has actually *measured* the cross-core flag primitives at scale while building a
per-layer megakernel, and several of its findings bear directly on ours. Read the arch caveat on
every number: **these are dav-3510 results.**

### 15.1 README — the module inventory, i.e. what a per-layer Ascend megakernel decomposes into

> # ascend_mega_kernel
>
> 面向昇腾 950（Ascend950PR）的 mega kernel 开发：在单卡 950PR + CANN 9.1.0 上探索大算子融合与持久化 kernel 技术，目标是对 Qwen3.8-Flash-Next-MXFP4 做高性能推理。

> | `m0/` | M0 bring-up：mix(1,2) 全核启动 + BufferID/CrossCore 最小验证。核间屏障为**只有 AIV 的 mode 0** `CrossCoreSetFlag<0,PIPE_MTE3>`/`CrossCoreWaitFlag<0>`（非 `SyncAll`；AIC 不参与）；56 AIV 真机 13/13 `ALL PASS` | ✅ |
> | `m1_mxfp4_gemm/` | 单核 MXFP4 GEMM（basic API 全链路，10 用例逐位 PASS） | ✅ |
> | `m2_mxfp4_quant/` | MXFP4 激活量化 AIV kernel（对 quant_ref.py 三层链逐字节一致） | ✅ |
> | `m3_grouped_gemm/` | 多核 grouped MXFP4 GEMM（28 AIC 专家槽位×N 切分，**mode2+mode0+mode2**） | ✅ |
> | `m4_gdn_recurrent/` | GDN decode 递推核（48 head，寄存器 VF，state 预取流水） | ✅ |
> | `m5_swiglu_quant/` | SwiGLU + MXFP4 量化融合 epilogue | ✅ |
> | `m7_router_topk/` | Router softmax+top-10 核（Sort32+2路MrgSort，**m=1 实测 0.066ms**） | ✅ |
> | `m8_permute/` | MoE permute/unpermute 核（专家槽位重排，golden 逐位一致） | ✅ |
> | `m9_gdn_prolog/` | GDN conv1d+l2norm+gating 融合预处理核 | ✅ |
> | `m11_bf16_gemm/` | bf16 GEMM（GDN in/out_proj 用，非 MX 路径） | ✅ |
> | `m12_rmsnorm_gated/` | RMSNormGated（GDN 输出门控+归一化，per-head 128 维） | ✅ |
> | `m13_moe_layer/` | **MoE block 整层 kernel**（mix(1,2) 单次启动跑全链 S1-S10，682 判据 + 118 项 numpy 独立校验全 PASS） | ✅（层界 I/O 待改） |
> | `m14_gdn_layer/` | **GDN 整层 kernel**（S1-S7 单次 mix 启动，114 kernel 判据 + 103 numpy 校验 PASS） | ✅（层界 I/O 待改） |
> | `m15_layer_loop/` | **48 层循环骨架**（host 循环，每层一次 `__mix__(1,2)` 启动）：36 层 GDN 直接复用 m14 整层 kernel，12 层 full attention 暂为**占位直通**；层间只经 GM bf16 残差流交接，conv/ssm state 常驻 GM | ✅ |
> | `probe_sync_quirks/` | 同步/sort/下标类问题常备复现器（MrgSort 26 变体 + VF idx 矩阵 + 证据归档） | ✅ |
> | `m0_syncall_probe/` | **最小探针：只测全核同步**（两个独立可执行文件，无环境变量）：`m0sk_mix` = AIC+AIV 双边 `SyncAll<false>`（官方"融合算子"形态）；`m0sk_aiv` = 仅 AIV `SyncAll<true>`（官方"纯 Vector"形态，**负控**）。判据 = 56 个 AIV 的"屏障前/后"标记 + kernel 能返回 | **已关闭（2026-10-05）** |

The probe/evidence directories, which are the unusual part of this repo:

```
aiv_sync_probe_pack/  m0_syncall_probe/  probe_aic_gm_dma/  probe_aiv_sync/
probe_crosscore_tail/ probe_cube_fp32/   probe_host_dma/    probe_mask_lanes/
probe_sync_quirks/    probe_ub2l1/       probe_ub_bufid_tail/ probe_v_align/
probe_vf_ldst/        probe_vf_loop/
```

### 15.2 `docs/05` §2 — the chosen granularity, and the mode-selection rule

> ## 2. 总体形态（已确认）
>
> - **粒度**：不是整个网络一个 kernel，而是**每个 transformer layer 一个 kernel**（48 层 → 48 次启动/decode step）。
> - **核配置**：整个 persistent kernel 以 **mix mode（AIC:AIV = 1:2）** 启动，用满所有核。
>   - 本机 28 AIC + 56 AIV（PG 降频 binning 版，满配 die 为 32 AIC）——**核数必须 API 获取，禁止硬编码**：`PlatformAscendC::GetCoreNumAic()` / `GetCoreNumAiv()`，blockDim 用 `CalcTschNumBlocks(aicCoreNum, aivCoreNum)` 计算；mix 启动 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`。
>   - **kernel 内部的分工是自由的**：同一 kernel 内可以划分 AIC-only 区域、AIV-only 区域、mix 区域。

THE MODE RULE, verbatim:

> - 核间同步用 **CrossCore 系列**，模式选用规则（用户明确）：
>   - **mode 2**：单个 AIC ↔ 它所属 AICore 内的 2 个 AIV（及反向）——适用于 **AICore 内 CV 直连的数据交接**，典型如 attention 的 **FIXP L0C→UB**、cube 输入的 **UB→L1** 这类点对点（核组内）同步；**1 set(AIC) 配 2 wait(AIV)，或 2 set(AIV) 配 1 wait(AIC) 算配对**；
>   - **mode 0**：**GM 介导的 op 间交接必须用 mode 0**——前一个任务输出到 GM、后一个任务从 GM 读时，不是特定一个核对一个核（前后 op 的 tiling 不一定一一对应），本质是 all-to-all。**mode 0 仅同类型**（"全部 AIC 之间"或"全部 AIV 之间"二选一，不支持 AIC 组 set、AIV 组 wait 的跨类型用法）；跨类型 all-to-all 的标准组合（SyncAll mix 序列）：AIVs→AIC（mode 2）→ 全体 AIC barrier（mode 0）→ AIC→AIVs（mode 2）→（如需）全体 AIV barrier（mode 0）。

THE HARD CONSTRAINTS, verbatim:

> - **CrossCore 硬约束（M2 调研实证）**：
>   - **Set 侧 pipe：模式 0/1/2 禁止 PIPE_S/PIPE_ALL**，只能 PIPE_V/M/MTE1/MTE2/MTE3/FIX（模式 4 可加 PIPE_S）——"同步挂在计算 pipe 上"是强制的；
>   - **Wait 侧 pipe 在 950 上生效**：只阻塞指定 pipe 的后续指令下发（已下发照常执行）——无依赖的 MTE2 预取可从同步旁流过，但必须用窄 pipe（PIPE_M/PIPE_FIX 等），不用默认的 PIPE_S；
>   - 同一核连续多个 CrossCoreSetFlag **硬件不保证执行顺序** → 相邻不同同步点用不同 flagId；
>   - 同 flagId 跨模式复用（同核）须前一模式全部 set/wait drain 完；numBlocks ≤ 物理核数。
>   - **batch mode（硬性要求，用户 2026-10-10）**：后续**所有** mix kernel（`__global__ __mix__(1,2)` 形态）**一律**加 `__schedmode__(1)`。跨核同步走 FFTS 报文，屏障成立要求整个 task 的所有核一次性原子下发；`__schedmode__(0)` 下 FFTS 允许分批 / 交叠下发，"全体到齐"的计数永远凑不齐 ⇒ 死等（launch 不返回、空转）。

And the design's own in-kernel-sync policy, which is worth reading as an argument:

> - 核内 pipeline 同步**只用 BufferID**（`get_buf`/`rls_buf`），**禁用 set_flag / wait_flag 系列**表达跨 pipe 数据可见性（理由：set/wait flag 是单向同步；BufferID 是 mutex 语义、双向同步）。
> - **PIPE_S 同步限制（硬性要求）**：除**有值依赖**的场景外……**禁止把同步挂在 PIPE_S 上**。理由：PIPE_S 同步会阻塞 main scalar 继续发射异步指令，摧毁"scalar 跑在指令前面"的跨 op 预取重叠机制。……**【2026-10-05 人类裁决订正：该例外作废】** 跨核链式 `wait→set` **不需要**把 wait 挂 PIPE_S（裁决原话「**所有说需要 PIPE_S 的都是扯淡**」）。
> - **资源全静态自管理（硬性要求）**：尽量不使用 AscendC 资源管理函数（TPipe / TBuf / TQue / TBufPool / AllocTensor 等）——所有 **BufferID、CrossCore flagId、buffer 地址全部由我们自己管理，编译期静态分配**（一份静态资源表：BufferID 枚举、flagId 枚举、UB/L1 地址偏移常量）。

### 15.3 `docs/05` §4.1 — THE RESOURCE TABLE, including the flagId budget

> | CrossCore flagId | **每核 16 个（0-15），模式 4 下 AIC 侧 0-31**；"每 id 计数 15 次" = 每（核,flagId) 一个 **4bit 计数器**（0-15），未配平累计超 15 报错。**SyncAll 占 11-14(+28/29)、Matmul 高阶 API 占 0-7**；**本项目不用 SyncAll/高阶 API → 0-15 全可用**（早期资料"0-10"是保守窗口，已修正） | 层内多处同步需规划复用；相邻同步点必须用不同 flagId |
> | BufferID | **每核用户可用 0-27**（MAX_TBUFID=31，28-31 保留给框架）；token 初始全部空闲；release 的 mode：`false` = CANN `ASC_LOCK_BLOCK`（默认，「阻塞」）、`true` = `ASC_LOCK_NON_BLOCK`；**两种模式都等本 pipe 已发射指令落地**（不是「立即生效」），`true` 额外等此前同 id 的释放 ⇒ `true` 更保守；**全项目 release 一律 `false`**；不可重入 | AIC 与 AIV 的 token 空间独立，pipe 不命名空间——不同 pipe 并发用同一 id 会互踩 |
> | SSBuf | 3KB，AIC↔AIV 核间通信 | 小控制信息传递 |
> | UB / AIV | **256 KB 物理 / 248 KB 总量 / 216 KB 可用**（`aclrtGetDeviceInfo(0, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE)` = **221,184 B**，2026-10-17 本机实测）。差额 32 KB 是编译期保留区（`__ASC_USE_RESERVED_UBUF__` / `--cce-disable-asc-reserved-ubuf` 相关）⇒ **UB 预算按 216 KB 算**，按 248 KB 排布会静默越界（表现为跨窗互相覆盖、数值"整数倍"错且不可复现） |
> | L1 Buffer / AIC | 512KB（1 AIC + 2 AIV 共享） |
> | L0A/L0B / AIC | 64KB + 4KB MX |
> | L0C / AIC | 256KB（256×256 tile） |
> | Scalar I-cache | AIC 32KB / AIV 16KB |

and the hardware basis line:

> 硬件依据：950 = 6 pipeline（CUBE/VEC/MTE1/MTE2/MTE3/FIXP）+ 独立 Scalar 单元（乱序、每周期最多发射 5 条指令）；BufferID 见白皮书 §4.1.6（get_buf=acquire、rel_buf=release）；**CrossCore 同步 2.0 由 STARS2.0 调度器支撑（flagId API 窗口 0–10、每 id 计数 ≤15，硬件 flag 池远大于此窗口）**。
>
> **推论**：GM 边界的 mode 0 同步应插在计算 pipe 上，这样无依赖的 MTE2 权重预取指令可以从同步旁边流过去（同步只阻塞依赖方，不阻塞搬运）。

### 15.4 `docs/05` §6 — THE KERNEL-TYPE CONSTRAINT, one line, our §4 answered

> | **核函数类型约束**：AIC_ONLY/AIV_ONLY **不开启调度模块、CrossCore 不可用**——layer kernel 必须 MIX 类型（本设计满足） |

### 15.5 `docs/05` §6.1 — the pipe/core-type matching rule that makes a wrong flag a SILENT no-op

> | **CrossCore `SetFlag/WaitFlag<mode, pipe>` 必须 pipe 类与核型匹配**（M24 实测；M17 挂死根因） | 实现只在 pipe 类与核型匹配时**发射指令**：`IsSplitVectorPipe={S,V,MTE2,MTE3}` 仅 AIV 发射、`IsSplitCubePipe={S,MTE1,MTE2,FIX,M}` 仅 AIC 发射 ⇒ **不匹配时是静默空操作（不报错），对侧 wait 永久挂死**。**用户澄清**：这里的原始错误是用法本身错——**AIC 侧写 GM 用的是 FIXP，不可能用 MTE3**（**根因：AIC 不能「直接」访问 UB ⇒ UB→GM 对 AIC 不可路由**）；pipe 类匹配规则只是让这类误用表现为静默挂死而不是编译报错 | **AIC 侧 set/wait 只能 `PIPE_S/MTE1/MTE2/FIX/M`；AIV 侧只能 `PIPE_S/V/MTE2/MTE3`**。写同步前先问"这个核型的这条 pipe 到底存不存在"；**flagId 8-11 无保留冲突，pipe 类才是雷** |

> | **硬规则：AIC 写 GM 必须走 `Fixpipe`，不能走 MTE3 DMA；配套：AIC 上的同步事件不得挂 `PIPE_MTE3`**（M24 + M35 + 用户裁定，2026-09-26） | **三次独立撞到同一根**：① **同步类**（M24 实测；M17 全 kernel 挂死根因）：`CrossCoreSetFlag/WaitFlag<…, PIPE_MTE3>` 在 AIC 上是**静默空操作**——`IsSplitCubePipe={S,MTE1,MTE2,FIX,M}` 不含 MTE3 ⇒ 指令根本不发射，对侧 wait 永久挂死；② **数据类**（M35 `probe_aic_blockidx.asc` 实测）：AIC 用「标量写 UB → MTE3 DMA 写 GM」**全部读回哨兵**；③ **用户裁定原话**："**AIC 写 GM 用的是 FIXP，而不是 MTE3**，所以你这个同步本来就是错的" | **后果＝静默失效**：不报错、不挂死（同步类表现为对侧永久等待，数据类表现为 GM 里仍是旧值/哨兵）——属**最难查的一类错**。 |

Two more rows worth having:

> | `GetBlockNum()` 在 mix kernel 内的语义（M22 实测） | **`__mix__(1,2)` 下 AIV 视角 `GetBlockNum()` 返回 AIC 数（28），不是 AIV 数**——AIV 侧要用核数需 `×2`（或在 host 用 `GetCoreNumAiv()` 传入） |
> | **同 pipe MTE2 背靠背复用同一 buffer** | 完成序无保证；**`get_buf/rls_buf` 对同 pipe 无效**（mode 0 与 drain 均实测不保序，**60/60 错 vs PipeBarrier 0/60**）。`PipeBarrier<PIPE_MTE2>` 是同 pipe 内部同步的正确且唯一有效指令……**官方口径同**：`asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/Lock.md:93` 逐字「连续调用的、具有相同id与pipe的两对Lock与Unlock**不能实现单流水（参数pipe指定）内不同指令之间的同步，单流水内多个指令之间的同步请使用PipeBarrier接口**」；旗标侧 `.../SetFlag_WaitFlag_ISASI.md:99` 逐字「**相同流水、相同eventID下，连续使用SetFlag会引发未定义行为，此时再执行PipeBarrier<PIPE_ALL>会出现卡死现象**」 |

> | **BufferID 只覆盖核内**（用户澄清，2026-09-26） | **BufferID（get/rls）是核内 token，管不了跨核可见性**。凡"AIV 产出 → AIC 消费"的交接……**必须用 CrossCore 同步**（AIV 侧在拷贝完成后 set、AIC 侧 wait），漏了就会"AIC 读太早、数据不在"（表现为下游恒 0，且难以从数值上定位） |

And the one honest unknown about graph capture, which is also ours:

> - **aclgraph 对"kernel 内同步"的交互要求**：具体到本项目的 **CrossCore flagId**、**BufferID（`Mutex`）**、`SyncAll`，**没有找到**任何出处说明图捕获/重放对它们有额外要求。已查渠道：CANN 9.1.0 本机头（`acl_rt.h` / `runtime/stream.h`）、asc-devkit 官方 example `04_aclgraph`、……——**均只讲 Stream 上任务的捕获/重放，没有讲与核内同步的交互**。官方 example 只演示了「捕获一个内部用 `PipeBarrier` 的核函数」这一个**正例**，**不能**据此推断 CrossCore / `Mutex`(BufferID) 也一定无约束。⇒ **实施 aclgraph 前必须在设备上就这几个同步点单独验证**。

### 15.6 `aiv_sync_probe_pack/README.md` — THE FFTS MESSAGE ENCODING AND THE RESERVED FLAG IDS

The two probes, verbatim:

```cpp
__schedmode__(1) __global__ __mix__(1, 2) void aiv_mode0_sync_kernel()
{
    if ASCEND_IS_AIV {
        AscendC::CrossCoreSetFlag<0 /*CC_MODE0*/, PIPE_MTE3>(5);   // 用户号 5（避开 CANN 保留号 11..14）
        AscendC::CrossCoreWaitFlag<0 /*CC_MODE0*/, PIPE_S>(5);
    }
    // AIC 不参与
}
```

```cpp
__schedmode__(1) __global__ __mix__(1, 2) void aiv_syncall_kernel()
{
    if ASCEND_IS_AIV {
        AscendC::SyncAll</*isAIVOnly=*/true>();   // 就一次，不循环
    }
    // AIC 不参与（SyncAll<true> 在 AIC 上本来就是空操作）
}
```

> `SyncAll<true>()` 展开即 `ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL))`
> + `wait_flag_dev(PIPE_S, SYNC_AIV_ONLY_ALL)`（`asc/impl/basic_api/dav_3510/kernel_operator_sync_impl.h:157-176`），
> 用的是**保留号 14 这个"一次 set 打给全体 AIV"的广播号**，默认 config `{PIPE_ALL, PIPE_ALL}`。

> ## 已经掌握的事实（都是设备实测或 CANN 源码里读出来的）
>
> 1. ~~**mode0 的生产者只有 AIC**：AIV 发 mode0 set 必挂~~ —— **2026-10-17 本机实测已推翻**：
>    56 个 AIV 各自 `CrossCoreSetFlag<0>(5)` + `CrossCoreWaitFlag<0>(5)` 能正常返回（`rc=0`）。
>    负控（只让 AIV block 0 set、其余只 wait）才会挂死（内部 30 s 超时 + 清理阶段继续卡住，`rc=124`）
>    ⇒ **mode0 的 wait 要全体 AIV 到齐，set 本身在 AIV 上是可用的**。
>    （早前"逐 id 16/16 全挂"的记录未能复现，怀疑当时二进制里夹了别的主动异常/未开 batch mode。）
> 2. **16 个 flag id 行为完全一致**，没有任何 id 残留计数 —— "某个 id 坏了"这个假设已被证伪。
>    逐 id 的 wait 探测里，**每次的基线 launch（不 set 不 wait）都能返回**，说明一次挂死不会污染后续 launch。
> 3. **跨核同步必须配 batch mode**（`__schedmode__(1)`）：`__schedmode__(0)` 下 FFTS 允许分批下发 /
>    算子交叠，"整个 task 到齐"的计数永远凑不齐 ⇒ 死等（表现为 launch 不返回、空转 250~360 W、util 0%）。
> 4. FFTS 报文编码（`kernel_operator_sync_impl.h:131-134`）：
>    `GetffstMsg(mode, flagId) = 0x1 + ((mode & 0x3) << 4) + ((flagId & 0xf) << 8)`
>    —— **mode 只占 2 bit、flagId 只占 4 bit（0..15）；CANN 保留 11=AIC / 12=AIV / 13=AIC+AIV / 14=AIV_ONLY_ALL。**

Measured table:

> | 程序 | 全核 28 AIC / 56 AIV | 半机 16 AIC / 32 AIV |
> |---|---|---|
> | `aiv_syncall` | **PASS**（`rc=0`） | **PASS**（`rc=0`） |
> | `aiv_mode0_sync` | **PASS**（`rc=0`） | **PASS**（`rc=0`） |
>
> - 负控（只让 AIV block 0 set）确认会挂死：说明 PASS 不是"自己 set 满足自己 wait"。

And the build-flow gotcha, which matters if you want `__schedmode__`:

> 走的是**自建 ASC 流**（`find_package(ASC)` + `LANGUAGES ASC`，直接编 `.asc`），
> **不是** CANN 官方的 `ascendc_kernel_cmake` 流。两者差别（实测）：
> **官方流会把核函数再包含一遍并改名成 inline、并把 `__schedmode__` / `__mix__` 属性判为非法**
> （`error: ... needs to appear with kernel func.`）；官方流里 task type / MIX 比例由
> `KERNEL_TASK_TYPE_DEFAULT(...)` 声明，batch mode 是 host 侧 `SetScheduleMode(1)`。

Timeout discipline worth copying:

> **程序内部另有 30 秒的 API 级超时**：两个探针都用
> `aclrtSynchronizeStreamWithTimeout(stream, 30000)`（单位**毫秒**，`acl/include/acl/acl_rt.h:2763-2769`）替代
> `aclrtSynchronizeStream`，卡在屏障里 30 秒就会返回非 0 值。
> **判据都是"这次 launch 能不能返回"**：`rc=0` = 同步成立、kernel 返回；`rc=124` = 卡在屏障里（超时）。

### 15.7 `docs/23` §4–§6 — the mode-0 barrier SCALE CLIFF, and its retraction

> ⇒ 本机结论：**52 ✓ / 54 ✗**（2× 现场复现）。M0 / m24 / `m0_syncall_probe`（都是 56 AIV）在本机挂死的原因就是这个；**代码正确性不能据此判负**。

> **⚠ 跨机器校正（人类 2026-10-05 报告）**：**另一台机器上 28 blocks（56 AIV）是 OK 的** ⇒ 这个悬崖**不是** 950PR 的普适硬件规则，而是**本机特有的**（本机 = 异常的那台）。

> ## 5. 新卡复现：52/54 悬崖**不复现**（2026-10-05）
>
> | blocks | AIV 数 | 结果 | 输出自证 |
> |---|---|---|---|
> | 16 | 32 | **通过** rc=0（~7–8 s） | `output[0]`=496=Σ(0..31) |
> | 26 | 52 | **通过** rc=0（~7–8 s） | `output[0]`=1326=Σ(0..51) |
> | 27 | 54 | **通过** rc=0（~7–8 s，3/3 复现） | `output[0]`=1431=Σ(0..53) |
> | 28 | 56（满核） | **通过** rc=0（~7–8 s） | `output[0]`=1540=Σ(0..55) |
>
> - 四发各编出**不同 md5** 的 demo；`output[0]` 恰等于 `NB*(2*NB-1)`（全体 AIV 的 `blockIdx` 之和）
>   ⇒ 54/56 个 AIV **全部**到达 mode-0 屏障并返回。
>
> **结论**：§4 的"52 ✓ / 54 ✗"是本项目**旧卡**现象，在新卡上**不复现**（54、56 AIV 均通过）；旧卡上的
> `m0` / `m24` / `m0_syncall_probe` 挂死应标注为**卡相关**，不能在新卡上默认成立。

The reproduction recipe for the official sample, which is the cheapest possible test of the
primitive on any machine:

```bash
# 1) 编官方样例（需给本地 CANN 9.1.0 补一个符号：它只有 AscendC::Std::ceil_division，
#    官方新样例用 AscendC::Std::ceil_div —— 在 /tmp 的拷贝里加个 4 行 shim 即可）
cp -r <asc-devkit>/examples/01_simd_cpp_api/03_basic_api/05_sync_control/cross_core_set_wait_flag /tmp/asc_ex_src
#    改 cross_core_set_wait_flag.h:31（本 checkout）的 NUM_BLOCKS，加 shim，然后：
cmake -B /tmp/asc_ex/s0_NB -S /tmp/asc_ex_src -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=0 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/asc_ex/s0_NB -j8
# 2) 生成输入 + 跑 + 验
cd /tmp/asc_ex/s0_NB && /usr/local/python3.12.13/bin/python3 /tmp/asc_ex_src/scripts/gen_data.py -scenarioNum 0
timeout 60 ./demo ; echo rc=$?
# 3) 判据：rc=0 ⇒ 通；rc=124 ⇒ 挂。数据自证：output[0] 应等于 Σ(0..2*NUM_BLOCKS-1)
```

> **待做（未跑）**：官方 `SCENARIO_NUM=2`（融合场景，8 blocks）是否通过；`__mix__(1,1)` + 28 blocks（28 AIC+28 AIV=56 核）用来判别"限制的是总核数还是 AIV/block 数"；以及有没有发射配置能让 28 blocks 共驻（`__schedmode__(1)` / task type / `SetScheduleMode`）。

### 15.8 A performance number, for calibration

`docs/25-cube-qsa-perf-notes.md`, summarized in the README index:

> **CUBE 版 QSA 性能现状与待优化点（2026-10-11）**：`caus_prefill4097` 实测 **31.75 ms / MFU 1.29%**（延迟受限）、当前 tiling（Mmad `M16`、softmax `8 头×64 token`）、四处待办（M 改 token tile、`SpcSoftmaxVf` 重写、逐 chunk `PIPE_ALL`、N 乒乓深度）

and `docs/19`:

> GDN prefill chunk 扫描选型：把 m18 的 **15.51 ms/层** 拆开、与 **1.37 ms/层** 对齐复算

## Relevance

Two of our four measured claims are contradicted or re-explained here, and the kernel-type
question is answered in one line.

**CONTRADICTION 1 — "mode 3 is the only reverse (vector→cube) mode that works in all set/wait
patterns" is probably an artifact of a pipe/core-type mismatch, not a property of the mode.**
§15.2's mode rule says **mode 2 is explicitly bidirectional**: 「1 set(AIC) 配 2 wait(AIV)，或
2 set(AIV) 配 1 wait(AIC) 算配对」. §15.5 then gives the mechanism that would make a reverse
mode-2 appear broken: `IsSplitCubePipe = {S, MTE1, MTE2, FIX, M}` — **MTE3 is not in it** — so
`CrossCoreWaitFlag<2, PIPE_MTE3>` on the AIC side 「是静默空操作（不报错），对侧 wait 永久挂死」.
Any reverse-direction experiment that put `PIPE_MTE3` on the cube side would fail silently and
look like "mode 2 reverse doesn't work". The prescribed pipe sets are
**AIC: `PIPE_S/MTE1/MTE2/FIX/M`; AIV: `PIPE_S/V/MTE2/MTE3`**. Re-run our reverse-mode-2 test with
`PIPE_FIX` or `PIPE_M` on the AIC wait before concluding mode 3 is needed. Note also §15.6 fact 4:
`GetffstMsg` masks `mode & 0x3`, so mode 3 *encodes* but is outside the documented set (0/1/2 on
A3) — if mode 3 "works", it is working undocumented.

**CONTRADICTION 2 — SyncAll's flag occupancy is 11-14, not 11-13, and 28/29 on top under mode 4.**
§15.3: 「SyncAll 占 11-14(+28/29)、Matmul 高阶 API 占 0-7」. §15.6 independently shows why 14 is in
the list: `SyncAll<true>` uses flag **14** = `SYNC_AIV_ONLY_ALL`, the broadcast ID, and nothing
else. So the AIV-only barrier costs exactly one flag, and our "11-13" is the `isAIVOnly=false`
(mix) case. Four reserved IDs: **11=AIC, 12=AIV, 13=AIC+AIV, 14=AIV_ONLY_ALL**. If we drop
`SyncAll<false>` for `SyncAll<true>` plus a hand-rolled AIC barrier, we get 11/12/13 back.

**Our §4 answered in one line** (§15.4): 「AIC_ONLY/AIV_ONLY **不开启调度模块、CrossCore 不可用**
——layer kernel 必须 MIX 类型」. Not "AIC-only is unsupported" — *AIC_ONLY/AIV_ONLY do not turn on
the scheduling module at all*, so FFTS cross-core sync has no hardware to run on. That is a
coherent cause for a submission-time rejection with `sqe_type=3, errType=0x20`: the SQE declares a
task type whose scheduler the dispatcher cannot wire up.

**Three things to adopt immediately.** (a) `__schedmode__(1)` on every mix kernel — §15.6 fact 3
says cross-core sync *requires* batch mode because FFTS otherwise dispatches the task's cores in
batches and the all-arrived count never completes; the failure mode is a launch that never
returns at 250–360 W and 0% util. We have not mentioned this anywhere in this file and it would
present as a mysterious hang. (b) `aclrtSynchronizeStreamWithTimeout(stream, 30000)` instead of
`aclrtSynchronizeStream`, so a barrier bug is a return code instead of a wedged card. (c) The
official sample `examples/01_simd_cpp_api/03_basic_api/05_sync_control/cross_core_set_wait_flag`
with `NUM_BLOCKS` bisected is the cheapest possible scale test of the primitive, with a
self-checking output (`output[0] == NB*(2*NB-1)`).

**One number to steal for our own flag accounting**: the counter is a **4-bit per (core, flagId)
counter**, 「未配平累计超 15 报错」. A persistent whole-forward-pass kernel that sets a flag once
per layer without a matching wait overflows after 15 layers. That is a structural argument for
per-layer launches or for a strict set/wait balance per layer.

Caveat: dav-3510, not dav-2201. The flagId count (16), the 4-bit counter, the reserved IDs
(11-14), the `GetffstMsg` encoding, the `IsSplit*Pipe` sets and the AIC_ONLY/AIV_ONLY constraint
are all read out of CANN headers rather than measured, so they are likely shared; the scale-cliff
and the 216 KB UB budget are machine- and chip-specific and should not be carried over.

## 16. `AscendC::CrossCoreSetFlag` — official Ascend C API reference (asc.gitcode.com, built from `cann/asc-devkit` master)

URL: https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.html
Raw markdown source: https://raw.gitcode.com/cann/asc-devkit/raw/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md
GitCode page link printed on the doc page: https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md
Author: Huawei Technologies Co., Ltd. — Ascend C SIG (`cann/asc-devkit` repo, auto-built docs site). Page last-commit shown on page: `5f50fc4c6ada`
Date: fetched 2026-10-20; site disclaimer: "本站内容由 asc-devkit 仓 master 分支自动编译生成，属于持续开发版本"

### 16.1 「产品支持情况」 block — VERBATIM (with the site's own `npu=` build annotations)

```markdown
## 产品支持情况<a id="section1550532418810"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- Atlas 200I/500 A2推理产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas推理系列产品AI Core：不支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- Atlas推理系列产品Vector Core：不支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- Atlas训练系列产品：不支持
<!-- end id7 -->
```

### 16.2 Header path and 功能说明 — VERBATIM

> 头文件路径为：`"basic_api/kernel_operator_block_sync_intf.h"`。
>
> 在核间同步场景中，CrossCoreSetFlag接口和[CrossCoreWaitFlag](CrossCoreWaitFlag_ISASI.md)接口配对工作，使用时需传入核间同步的标记ID（flagId），每个ID对应一个用于控制同步的计数器。
>
> - CrossCoreSetFlag：AI Core中的AIC/AIV在完成本核前置位的pipe流水任务后，调用该接口，向调度模块上报本核任务已完成。调度模块收集所有参与核的上报信息，作为全局完成标志。
> - CrossCoreWaitFlag：用于阻塞本核后续指令的下发（已下发指令可正常执行），直至调度模块确认所有应调用CrossCoreSetFlag的核均已上报完成，随即解除阻塞，允许该核继续下发指令。

### 16.3 THE AUTHORITATIVE MODE LIST — VERBATIM

> 如[图1](#fig37581010773)所示（图中block代表AIC，sub_block代表AIV），同步控制分为以下几种模式：
>
> - 模式0：AI Core核间的同步控制。对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行；对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行。
> - 模式1：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了CrossCoreSetFlag，CrossCoreWaitFlag后续的指令才会执行。
> - 模式2：AI Core内部，AIC与所有AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后，两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。
> - 模式4：AI Core内部，AIC与单个AIV之间的同步控制。当在AIC核执行CrossCoreSetFlag之后，AIV0或者AIV1上CrossCoreWaitFlag后续的指令才会继续执行；AIV0或者AIV1执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。
>
> > [!NOTE]说明
> > - 不同型号支持的模式不同，具体支持情况请参考[modeId支持的取值说明](#modeId支持的取值说明)。
> > - 每个模式的具体执行逻辑与细节可以参考[关键特性说明](key_features.md)。

NOTE: the enumerated set is `{0, 1, 2, 4}`. **There is no 模式3 / mode 3 anywhere on this page** (nor on the WaitFlag page, nor in `key_features.md`). Mode 3 is a hole in the documented mode space.

Figure referenced: `../../../../figures/block_subblock_relationship_3510.png` ("图1 同步控制模式示意图").

### 16.4 EXACT C++ SIGNATURE — VERBATIM

```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
```

> CrossCoreSetFlag的模板参数`modeId`和`pipe`均**没有默认值**，其取值用于配置硬件指令，调用接口时必须**显式传入**。与本接口不同，CrossCoreWaitFlag的两个模板参数均具有默认值，且在不同产品和同步模式下的生效情况不同，因此**两个接口对模板参数的传入要求并不相同**。

表1 模板参数说明 — VERBATIM:

| 参数名 | 描述 |
| --- | --- |
| modeId | 核间同步的模式，不同产品对同步模式的支持情况请参见[modeId支持的取值说明](#modeId支持的取值说明)。<br>各个模式支持的对应核函数（Kernel）类型请参照[表3](#table3)。 |
| pipe | 标识在哪条流水的前序指令完成后，CrossCoreSetFlag才向调度模块发送通知。支持的流水类型为PIPE_V、PIPE_M、PIPE_MTE1、PIPE_MTE2、PIPE_MTE3、PIPE_FIX，不支持PIPE_S和PIPE_ALL。不同产品对流水类型的支持情况请参见[pipe支持的流水类型说明](#pipe支持的流水类型说明)。 |

表2 参数说明 — VERBATIM:

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| flagId | 输入 | 核间同步的标记。不同产品对flagId的取值范围说明请参见[flagId取值范围说明](#flagId取值范围说明)。 |

### 16.5 WHICH ARCH SUPPORTS WHICH MODE — VERBATIM

```markdown
### modeId支持的取值说明<a id="modeId支持的取值说明"></a>

不同产品对同步模式的支持情况如下：
<!-- npu="950" id8 -->
- 针对Ascend 950PR&950DT系列产品，支持的同步模式为：模式0、模式1、模式2、模式4。
<!-- end id8 -->
<!-- npu="A3,910b" id9 -->
- 针对如下产品，支持的同步模式为：模式0、模式1、模式2。
    <!-- npu="A3" id10 -->
    - Atlas A3系列产品
    <!-- end id10 -->
    <!-- npu="910b" id11 -->
    - Atlas A2系列产品
    <!-- end id11 -->
<!-- end id9 -->
```

So for Atlas A3 (dav-2201 / 910_9392): modes 0, 1, 2 ONLY. Mode 4 is `npu="950"`-gated (Ascend 950PR & 950DT, i.e. arch 3510).

### 16.6 PIPE RULES FOR THE SET PIPE — VERBATIM

```markdown
### pipe支持的流水类型说明<a id="pipe支持的流水类型说明"></a>

- 核间同步的模式为模式0、1、2时，
    - 支持的流水类型为PIPE_V、PIPE_M、PIPE_MTE1、PIPE_MTE2、PIPE_MTE3、PIPE_FIX。
    - 不支持的流水类型为PIPE_ALL、PIPE_S。
<!-- npu="950" id12 -->
- 针对Ascend 950PR&950DT系列产品，核间同步的模式为模式4时，
    - 支持的流水类型为PIPE_V、PIPE_M、PIPE_MTE1、PIPE_MTE2、PIPE_MTE3、PIPE_FIX、PIPE_S。
    - 不支持的流水类型为PIPE_ALL。
<!-- end id12 -->
```

Reading for A3: the SET pipe may be any of `PIPE_V | PIPE_M | PIPE_MTE1 | PIPE_MTE2 | PIPE_MTE3 | PIPE_FIX`. `PIPE_FIX` is legal as a set pipe on A3 (this is what you use after a Cube/FixPipe write, the canonical `CrossCoreSetFlag<0x2, PIPE_FIX>` pattern). `PIPE_S` is FORBIDDEN as a set pipe on A3 — PIPE_S only becomes legal for the set side on 950 mode 4.

### 16.7 flagId RANGE — VERBATIM

```markdown
### flagId取值范围说明<a id="flagId取值范围说明"></a>

- 核间同步的模式为模式0、1、2时，支持的取值范围为0-15。
<!-- npu="950" id13 -->
- 针对Ascend 950PR&950DT系列产品，核间同步的模式为模式4时，支持的取值范围情况如下：
    - AIV0发起的flagId 0-15的CrossCoreSetFlag操作对应AIC CrossCoreWaitFlag中flagId 0-15的操作。
    - AIV1发起的flagId 0-15的CrossCoreSetFlag操作对应AIC CrossCoreWaitFlag中flagId 16-31的操作。
    - AIC发起的flagId 0-15的CrossCoreSetFlag操作对应AIV0 CrossCoreWaitFlag中flagId 0-15的操作。
    - AIC发起的flagId 16-31的CrossCoreSetFlag操作对应AIV1 CrossCoreWaitFlag中flagId 0-15的操作。
<!-- end id13 -->

此外，当本接口与Matmul高阶API或SyncAll接口同时使用时，开发者自行使用的flagId还需注意避免与这些接口内部已占用的flagId冲突，详见[核间同步flagId占用说明](key_features.md#inter_core_sync_flagid_usage)。
```

### 16.8 FULL 「约束说明」 — VERBATIM, COMPLETE

```markdown
## 约束说明<a id="section633mcpsimp"></a>

- 由于当核函数（Kernel）类型为KERNEL_TYPE_AIC_ONLY或KERNEL_TYPE_AIV_ONLY时，硬件不会开启调度模块，也就无法正常进行核间同步，因此不同的同步模式配置核函数（Kernel）类型或函数执行空间限定符的情况如下：
    - 在纯Vector/Cube场景下（模式0或模式1），建议设置核函数（Kernel）类型为KERNEL_TYPE_MIX_AIV_1_0或KERNEL_TYPE_MIX_AIC_1_0，其它支持的核函数（Kernel）类型请参考表3。
    - 对于Vector和Cube混合场景（模式2和模式4），需根据AI Core中AIC和AIV的比例灵活配置核函数（Kernel）类型，不同模式支持的函数执行空间限定符和核函数（Kernel）类型请参照表3。

        **表3**  模式与支持的核函数（Kernel）类型配置<a id="table3"></a>

        | 模式 | 支持的函数执行空间限定符 | 支持的核函数（Kernel）类型配置 |
        | --- | --- | --- |
        | 0 | __mix__(0, 1)、__mix__(1, 0)、__mix__(1, 1)、__mix__(1, 2) | KERNEL_TYPE_MIX_AIV_1_0、KERNEL_TYPE_MIX_AIC_1_0、KERNEL_TYPE_MIX_AIC_1_1、KERNEL_TYPE_MIX_AIC_1_2 |
        | 1 | __mix__(1, 1)、__mix__(1, 2) | KERNEL_TYPE_MIX_AIC_1_1、KERNEL_TYPE_MIX_AIC_1_2 |
        | 2 | __mix__(1, 1)、__mix__(1, 2) | KERNEL_TYPE_MIX_AIC_1_1、KERNEL_TYPE_MIX_AIC_1_2 |
        | 4 | __mix__(1, 2) | KERNEL_TYPE_MIX_AIC_1_2 |

- 以下接口内部实现中使用了CrossCoreSetFlag和CrossCoreWaitFlag会占用一部分flagId，开发者同时使用CrossCoreSetFlag/CrossCoreWaitFlag与这些接口时，需注意避免flagId使用冲突，各接口flagId占用的完整说明参见核间同步flagId占用说明。

- 用户需要确保配套使用CrossCoreSetFlag和CrossCoreWaitFlag，否则会出现未定义行为。
- 核间同步的模式为模式0、1、2时，模板参数pipe不支持设置为PIPE_ALL，PIPE_S。
<!-- npu="950" id14 -->
- 针对Ascend 950PR&950DT系列产品，核间同步的模式为模式4时，模板参数pipe不支持配置为PIPE_ALL。
<!-- end id14 -->
- 一个核连续发出的CrossCoreSetFlag，硬件不保证执行顺序。例如AIV依次发射CrossCoreSetFlag<0, PIPE_V>(0)，CrossCoreSetFlag<0, PIPE_V>(1)，实际的执行顺序可能是flagId=1的先执行。
- flagId相关的约束：
    - 对于模式0、1、2，每个AIC和每个AIV都各自有16个flagId，支持的取值范围为0-15。如果flagId的值超出该范围，则截取低4bit（例如，flagId=17时，截取后为1）。
    <!-- npu="950" id15 -->
    - 针对Ascend 950PR&950DT系列产品，核间同步的模式为模式4时，AIC有32个flagId，支持的取值范围为0-31，如果flagId的值超出该范围，则截取低5bit（例如，flagId=35时，截取后为3）；AIV有16个flagId，支持的取值范围为0-15，如果flagId的值超出该范围，则截取低4bit（例如，flagId=17时，截取后为1）。
    <!-- end id15 -->
    - 每个flagId都对应一个计数器，当调用CrossCoreWaitFlag时，若计数器值为0则会阻塞后续指令下发，已下发指令可正常执行；当调度模块感知到所有参与同步的核（具体包含哪些核与设置的核间同步模式有关）完成（调用了CrossCoreSetFlag）同步后，会将与设置的flagId对应的计数器的值增加1。此时，计数器值为非0，阻塞解除，并且将对应计数器的值减去1进行还原。具体执行逻辑与细节可以参考关键特性说明。每一个计数器计数范围为0-15。如果调用CrossCoreSetFlag的次数超过15次，计数器的值超出该范围，则会异常报错，中断流程。
- 核间同步的模式为模式0、1、2时，同一个flagId用于不同核间同步模式的约束：
    - 同一核上，若同一个flagId需用于不同核间同步模式，须在模式切换前完成前一个模式的所有同步操作——即确保该flagId关联的所有CrossCoreSetFlag与配套CrossCoreWaitFlag调用均已执行完毕。
    - 对于不同的核，可以直接将同一flagId用于不同的核间同步模式，具体包括以下2种场景：
        - 多个AI Core之间，使用flagId=0同步所有的AIC（模式0）；单个AI Core内，使用flagId=0同步所有AIV（模式1）。
        - 单个AI Core内，使用flagId=0同步所有AIV（模式1）；另一个AI Core内，使用flagId=0同步AIC与所有AIV（模式2）。
- 使用该接口模式0时，建议开启batchmode模式，使算子独占全部所需核资源，否则可能因满足以下条件导致死锁：
    - 多流并发场景（≥2条执行流）。
    - ≥2个算子并发执行。
    - 所有并发算子的核数总和超过物理核数。
    - ≥2个并发算子使用了核间同步功能。

    具体而言，在多流场景下，某条流的核间同步算子虽分配到n个物理核，但可能仅有n-m个核先被调度执行，而其余m个核因被其他流的核间同步算子抢占而尚未启动。先启动的n-m个核执行到核间同步时等待剩余m核完成，而剩余m核因被其他流的核间同步算子占用而无法释放，形成死锁。
    核函数（Kernel）直调场景下通过__schedmode__(mode)限定符来设置batchmode模式；工程化算子开发场景下，通过TilingContext的SetScheduleMode接口来设置batchmode模式，具体请参考《基础数据结构和接口》。
```

### 16.9 调用示例 — VERBATIM

表4 样例描述:

| SCENARIO_NUM取值 | 业务场景 | 使用的同步模式 |
| --- | --- | --- |
| 0 | 纯Vector计算场景（16个AIV） | mode0（AIV全核同步） |
| 1 | 纯Vector计算场景（2个AIV） | mode1 |
| 2 | Cube与Vector融合计算场景 | mode2（AIC等AIV）、mode2（AIV等AIC）、mode0（AIC全核同步） |

```cpp
    AscendC::Muls(xLocal, xLocal, float(AscendC::GetBlockIdx()), this->blockLength);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

    // Unified Buffer（UB）到GM搬运启用原子累加。
    AscendC::SetAtomicAdd<float>();
    // DataCopy属于PIPE_MTE3流水操作。
    AscendC::DataCopy(atomicResultGm, xLocal, this->blockLength);
    // 当本AIV完成前置PIPE_MTE3(DataCopy)流水操作后，通知其他AIV核，本AIV已经完成。
    AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(0);
    // 阻塞本AIV继续往下执行指令，直到其他AIV全部都完成PIPE_MTE3流水操作，才解除阻塞往下执行。
    AscendC::CrossCoreWaitFlag<0>(0);
    // 关闭原子累加。
    AscendC::DisableDmaAtomic();

    if (AscendC::GetBlockIdx() == 0) {
        AscendC::DataCopy(yLocal, atomicResultGm, this->blockLength);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopy(atomicResultGm, yLocal, this->blockLength);
        return;
    }
```

Full sample: https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/03_basic_api/05_sync_control/cross_core_set_wait_flag

## Relevance

This is THE authoritative spec and it both confirms and contradicts our measurements. Confirms: 16 flagIds per AIC and per AIV on A3 (range 0-15, overflow truncates to low 4 bits), counter depth 0-15 with hard error past 15 sets. Contradicts: the documented mode set on A3 is exactly {0,1,2} — mode 3 is NOT a documented mode at all, and mode 2 is specified as FULLY BIDIRECTIONAL (AIC->both AIV, and both AIV->AIC), so there is no documented need for a "reverse mode". Also hard constraint: `PIPE_S` is illegal as the SET pipe on A3; set-order between consecutive CrossCoreSetFlag on one core is NOT guaranteed by hardware.

## 17. `AscendC::CrossCoreWaitFlag` — official Ascend C API reference (the mode/pipe template args are INERT on A3)

URL: https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreWaitFlag_ISASI.html
Raw markdown source: https://raw.gitcode.com/cann/asc-devkit/raw/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreWaitFlag_ISASI.md
Author: Huawei Technologies Co., Ltd. — Ascend C SIG (`cann/asc-devkit`). Doc topic id `ZH-CN_TOPIC_0000001787470374`
Date: fetched 2026-10-20 (master branch, continuously built)

### 17.1 产品支持情况 — identical gating to SetFlag, VERBATIM

```markdown
<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- Atlas 200I/500 A2推理产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas推理系列产品AI Core：不支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- Atlas推理系列产品Vector Core：不支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- Atlas训练系列产品：不支持
<!-- end id7 -->
```

Header path, VERBATIM: 头文件路径为：`"basic_api/kernel_operator_block_sync_intf.h"`。

### 17.2 EXACT C++ SIGNATURE — VERBATIM (note the defaults)

```cpp
template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
```

表1 模板参数说明 — VERBATIM:

| 参数名 | 描述 |
| --- | --- |
| modeId | 核间同步的模式，默认值为0。不同产品对该参数的生效情况及支持的流水类型请参见[模板参数默认值及生效情况](#template-parameter-defaults)。<br>不同产品对同步模式的支持情况请参见[modeId支持的取值说明](#modeId支持的取值说明)。<br>各个模式支持的对应核函数（Kernel）类型请参照[表3](#table3)。 |
| pipe | 设置该接口阻塞的流水类型，默认值为PIPE_S。不同产品对该参数的生效情况及支持的流水类型请参见[模板参数默认值及生效情况](#template-parameter-defaults)。 |

### 17.3 ★「模板参数默认值及生效情况」 — THE MOST IMPORTANT BLOCK — VERBATIM

```markdown
### 模板参数默认值及生效情况<a id="template-parameter-defaults"></a>

<!-- npu="950" id8 -->
- 针对Ascend 950PR&950DT系列产品，两个模板参数的使用描述如下：

    - modeId
        - `modeId`不直接作为硬件指令的输入，CrossCoreWaitFlag会根据该参数选择对应的硬件指令。
        - 模式0、模式1和模式2使用相同的硬件指令。省略`modeId`时使用默认值`0`，不影响接口功能；但**建议显式传入与CrossCoreSetFlag相同的`modeId`**，便于确认两个接口的同步模式匹配。
        - 模式4使用的硬件指令与模式0、模式1和模式2不同，**必须显式传入`modeId=4`**，否则接口会按照默认的模式0执行，导致同步行为不符合预期。
    - pipe
        - `pipe`作为硬件指令的输入，参数取值**生效**，用于指定CrossCoreWaitFlag阻塞的流水。
        - 省略`pipe`时使用默认值`PIPE_S`。
        - 模式0、模式1、模式2和模式4支持`PIPE_S`、`PIPE_V`、`PIPE_M`、`PIPE_MTE1`、`PIPE_MTE2`、`PIPE_MTE3`、`PIPE_FIX`，不支持`PIPE_ALL`。
<!-- end id8 -->
<!-- npu="A3,910b" id9 -->
- 针对如下产品，**参数取值不影响硬件指令功能**，两个模板参数的使用描述如下：

    <!-- npu="A3" id10 -->
    - Atlas A3系列产品
    <!-- end id10 -->
    <!-- npu="910b" id11 -->
    - Atlas A2系列产品
    <!-- end id11 -->

    `modeId`和`pipe`均不作为硬件指令的输入，建议按如下方式配置：

    - modeId：省略该参数不影响接口功能；但**建议显式传入与CrossCoreSetFlag相同的`modeId`**，便于确认两个接口的同步模式匹配。
    - pipe：**建议省略该参数并使用默认值`PIPE_S`**。无论`pipe`取值为何，CrossCoreWaitFlag都会阻塞**全部流水**的后续指令。
<!-- end id9 -->
```

Decoded for A3 / dav-2201: on the WAIT side BOTH template parameters are purely decorative. `modeId` does not pick a different hardware instruction and `pipe` does not narrow the block — the wait always blocks ALL pipes ("无论pipe取值为何，CrossCoreWaitFlag都会阻塞全部流水的后续指令"). The mode therefore lives ENTIRELY in the SET instruction on 2201. Only on 950/3510 does the wait side differentiate (mode 4 selects a different hardware instruction, and `pipe` becomes a real narrowing input).

### 17.4 modeId / flagId ranges — VERBATIM

```markdown
### modeId支持的取值说明<a id="modeId支持的取值说明"></a>

不同产品对同步模式的支持情况如下：
<!-- npu="950" id12 -->
- 针对Ascend 950PR&950DT系列产品，支持的同步模式为：模式0、模式1、模式2、模式4。
<!-- end id12 -->
<!-- npu="A3,910b" id13 -->
- 针对如下产品，支持的同步模式为：模式0、模式1、模式2。
    <!-- npu="A3" id14 -->
    - Atlas A3系列产品
    <!-- end id14 -->
    <!-- npu="910b" id15 -->
    - Atlas A2系列产品
    <!-- end id15 -->
<!-- end id13 -->

### flagId取值范围说明<a id="flagId取值范围说明"></a>

- 核间同步的模式为模式0、1、2时，支持的取值范围为0-15。
<!-- npu="950" id16 -->
- 针对Ascend 950PR&950DT系列产品，核间同步的模式为模式4时，支持的取值范围情况如下：
    - AIV0发起的flagId 0-15的CrossCoreSetFlag操作对应AIC CrossCoreWaitFlag中flagId 0-15的操作。
    - AIV1发起的flagId 0-15的CrossCoreSetFlag操作对应AIC CrossCoreWaitFlag中flagId 16-31的操作。
    - AIC发起的flagId 0-15的CrossCoreSetFlag操作对应AIV0 CrossCoreWaitFlag中flagId 0-15的操作。
    - AIC发起的flagId 16-31的CrossCoreSetFlag操作对应AIV1 CrossCoreWaitFlag中flagId 0-15的操作。
<!-- end id16 -->
```

### 17.5 约束说明 — the parts that differ from the SetFlag page — VERBATIM

> - CrossCoreWaitFlag必须与[CrossCoreSetFlag](CrossCoreSetFlag_ISASI.md)接口配合使用，避免计算核一直处于阻塞阶段。建议显式配置与配对CrossCoreSetFlag相同的modeId，以便确认Set与Wait使用相同的同步模式。
>
> - 以下接口内部实现中使用了CrossCoreSetFlag和CrossCoreWaitFlag会占用一部分flagId，开发者同时使用CrossCoreSetFlag/CrossCoreWaitFlag与这些接口时，需注意避免flagId使用冲突，各接口flagId占用的完整说明参见[核间同步flagId占用说明](key_features.md#inter_core_sync_flagid_usage)。
>
> - flagId相关的约束：
>     - 对于模式0、1、2，每个AIC和每个AIV都各自有16个flagId，支持的取值范围为0-15。如果flagId的值超出该范围，则截取低4bit（例如，flagId=17时，截取后为1）。
>     - 针对Ascend 950PR&950DT系列产品，核间同步的模式为模式4时，AIC有32个flagId，支持的取值范围为0-31，如果flagId的值超出该范围，则截取低5bit（例如，flagId=35时，截取后为3）；AIV有16个flagId，支持的取值范围为0-15，如果flagId的值超出该范围，则截取低4bit（例如，flagId=17时，截取后为1）。
>     - 每个flagId都对应一个计数器，当调用[CrossCoreWaitFlag](CrossCoreWaitFlag_ISASI.md)时，若计数器值为0则会阻塞后续指令下发，已下发指令可正常执行；当调度模块感知到核间同步（CrossCoreSetFlag）全部完成后，会将对应CrossCoreWaitFlag的计数器的值增加1。此时，计数器值为非0，阻塞解除，并且将对应计数器的值减去1进行还原。具体执行逻辑与细节可以参考[关键特性说明](key_features.md)。flagId对应的计数器计数范围为0-15。如果计数器的值超出该范围，则会异常报错，中断流程。
> - 模式0、1、2下，同一个flagId用于不同核间同步模式的约束：
>     - 同一核上，若同一个flagId需用于不同核间同步模式，须在模式切换前完成前一个模式的所有同步操作——即确保该flagId关联的所有CrossCoreSetFlag与配套CrossCoreWaitFlag调用均已执行完毕。
>     - 对于不同的核，可以直接将同一flagId用于不同的核间同步模式，具体包括以下2种场景：
>         - 多个AI Core之间，使用flagId=0同步所有的AIC（模式0）；单个AI Core内，使用flagId=0同步所有AIV（模式1）。
>         - 单个AI Core内，使用flagId=0同步所有AIV（模式1）；另一个AI Core内，使用flagId=0同步AIC与所有AIV（模式2）。

表3 is byte-identical to the SetFlag page's 表3 (mode -> `__mix__` qualifier -> KERNEL_TYPE table); see section 9.8.

## Relevance

The decisive fact for a hand-written A3 engine: `CrossCoreWaitFlag<mode, pipe>` on dav-2201 compiles to a mode-agnostic, pipe-agnostic, all-pipe barrier on the local flagId counter. Any "mode N works / doesn't work" observation on A3 is therefore a statement about the SET instruction's mode field and about which core's counter it bumps, never about the wait. Explains why an undocumented set-mode can appear to "work in all set/wait patterns": the wait side cannot distinguish it.

## 18. 「关键特性说明」 — the official flagId OCCUPATION table (arch 2201 vs 3510) and the per-mode mechanism

URL: https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.html
Anchor for the occupation table: https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.html#inter_core_sync_flagid_usage
Raw markdown: https://raw.gitcode.com/cann/asc-devkit/raw/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.md
Author: Huawei Technologies Co., Ltd. — Ascend C SIG (`cann/asc-devkit`)
Date: fetched 2026-10-20

### 18.1 概述 — VERBATIM

> 开发者使用CrossCoreSetFlag和CrossCoreWaitFlag接口进行核间同步时，需要传入核间同步标记flagId，每个flagId对应一个用于控制同步的计数器。
>
> 部分接口的内部实现也会占用一部分flagId。当开发者同时使用这些接口与CrossCoreSetFlag/CrossCoreWaitFlag时，若flagId发生冲突（即相互独立的同步操作复用了同一个flagId），其对应的计数器会被错误地共享和修改，导致核间同步行为异常：可能出现阻塞无法解除而使程序卡死，或同步提前完成、阻塞提前解除而导致计算结果错误。
>
> 因此，当开发者同时调用CrossCoreSetFlag/CrossCoreWaitFlag与这些占用flagId的接口时，应避开各接口内部已占用的flagId，选用未被占用的flagId。
>
> 核间同步分为模式0、1、2与模式4，两者的底层硬件指令不同，flagId相互独立，因此各接口flagId的占用情况按所属模式和NPU架构版本分别参见模式0、1、2的flagId占用和模式4的flagId占用。

### 18.2 ★ 表1 模式0、1、2 flagId占用情况 — VERBATIM (NPU架构版本 column names 2201 explicitly)

| 接口 | 限定条件 | 占用flagId | 各计算核占用情况 | NPU架构版本 |
| --- | --- | --- | --- | --- |
| SyncAll | isAIVOnly=true | 14 | AIV：14 | 2201、3510 |
| SyncAll | isAIVOnly=false | 11、12、13 | AIC：11、12、13；AIV：12、13 | 2201、3510 |
| SetNextTaskStart | SuperKernel子核函数（Kernel）中调用 | 11、14 | AIC：11；AIV：14 | 2201、3510 |
| WaitPreTaskEnd | SuperKernel子核函数（Kernel）中调用 | 11、12、13、14 | AIC：11、12、13；AIV：12、13、14 | 2201、3510 |
| Matmul | 定义N个Matmul对象 | [0, 2N-1] | AIV/AIC按对象与subBlock分配 | 2201 |
| Matmul | SuperKernel融合场景（GetTaskRatio返回2，即1:2配比） | 15 | AIV：15 | 2201 |

### 18.3 表2 模式4 flagId占用情况 — VERBATIM (all rows are 3510-only; NONE apply to 2201/A3)

| 接口 | 限定条件 | 占用flagId | 各计算核占用情况 | NPU架构版本 |
| --- | --- | --- | --- | --- |
| SyncAll | isAIVOnly=false（核函数（Kernel）使用__mix__(1, 2)修饰时） | 12、13、28、29 | AIC：12、13、28、29；AIV：12、13 | 3510 |
| WaitPreTaskEnd | SuperKernel子核函数（Kernel）中调用 | 12、13、28、29 | AIC：12、13、28、29；AIV：12、13 | 3510 |
| Matmul | 定义N个Matmul对象 | [0, N+17] | AIC/AIV按对象、eventType、subBlock分配 | 3510 |
| Matmul | A矩阵和B矩阵同时开启IBSHARE的场景 | 15 | AIV：15 | 3510 |
| TSCM | 从UB发起的L1 Buffer（TSCM）队列 | 0~11（从11向下逐个分配） | AIV | 3510 |
| DataCopy（GM→L1） | 无 | 1、17 | AIC：1、17；AIV：1 | 3510 |

### 18.4 各接口限定条件说明 — VERBATIM

> **SyncAll**
>
> - isAIVOnly=true：纯Vector算子的全核同步，占用模式0、1、2的flagId为14（AIV）。
> - isAIVOnly=false：Mix（包含Cube和Vector计算）算子的全核同步，占用情况随NPU架构与核函数（Kernel）类型不同：
>     - 2201架构：占用模式0、1、2的flagId为11、12、13。
>     - 3510架构：
>         - 核函数（Kernel）使用__mix__(1, 1)修饰时，占用模式0、1、2的flagId为11、12、13。
>         - 核函数（Kernel）使用__mix__(1, 2)修饰时，占用模式0、1、2的flagId为11，以及模式4的flagId为12、13、28、29。
>
> **SetNextTaskStart / WaitPreTaskEnd**
>
> - 二者为SuperKernel场景专用接口，仅在算子以SuperKernel方式融合、且子核函数（Kernel）中调用时才生效。
> - SetNextTaskStart：占用模式0、1、2的flagId为11（AIC）、14（AIV）。
> - WaitPreTaskEnd：
>     - 2201架构：占用模式0、1、2的flagId为11、12、13、14。
>     - 3510架构：占用模式0、1、2的flagId为11、12、13、14，以及模式4的flagId为12、13、28、29。
>
> **Matmul**
>
> 占用flagId的范围与定义的Matmul对象数目N相关（N最大为4），并且随NPU架构不同而不同，参见REGIST_MATMUL_OBJ。
>
> - 2201架构：占用模式0、1、2的flagId为[0, 2N-1]（N=4时占用[0,7]）；SuperKernel融合（1:2配比，即GetTaskRatio返回2）场景下还占用flagId为15。
> - 3510架构：占用模式4的flagId为[0, N+17]（N=4时占用[0,21]）；A矩阵和B矩阵同时开启IBSHARE的场景下还占用flagId为15。
>
> **TSCM**
>
> - 仅当使用从UB发起的L1 Buffer（TSCM）队列时才触发，参见TSCM。
> - 占用模式4的flagId，范围为0~11：第1个TSCM队列占用11，第2个占用10，依次向下递减分配，最多可分配到0。
>
> **DataCopy（GM→L1）**
>
> - 在3510架构下占用模式4的flagId为1、17（AIC：1、17；AIV：1）。参见DataCopy（GMToL1连续数据搬运）。

### 18.5 各个核间同步控制实现原理 — VERBATIM (note the explicit "模式4...仅在NPU架构版本3510上支持")

> 如图1所示，本章节将配合时序图介绍CrossCoreSetFlag和CrossCoreWaitFlag配合使用时支持的四种同步控制模式各自实现的原理。
>
> - 模式0：AI Core核间的同步控制。对于AIC全核场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的全部流水或者由模板参数pipe指定的流水（与NPU架构有关）中的指令才会执行；对于AIV全核场景，同步所有的AIV核，直到所有的AIV核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的全部流水或者由模板参数pipe指定的流水（与NPU架构有关）中的指令才会执行。
> - 模式1：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了CrossCoreSetFlag，CrossCoreWaitFlag后续的全部流水或者由模板参数pipe指定的流水（与NPU架构有关）中的指令才会执行。
> - 模式2：AI Core内部，AIC与AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后，两个AIV上CrossCoreWaitFlag后续的全部流水或者由模板参数pipe指定的流水（与NPU架构有关）中的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的全部流水或者由模板参数pipe指定的流水（与NPU架构有关）中的指令才能执行。
> - 模式4：AI Core内部，AIC与单个AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后，单个AIV上CrossCoreWaitFlag后续的由模板参数pipe指定的流水中的指令才会继续执行；单个AIV执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的由模板参数pipe指定的流水中的指令才能执行。**该模式仅在NPU架构版本3510上支持。**
>
> 说明 模式0、1、2下CrossCoreWaitFlag阻塞的流水类型因NPU架构而异：
>
> - 在NPU架构版本2201上，CrossCoreWaitFlag传入的pipe参数无效，无论取值为何都默认阻塞全部流水的后续指令，建议省略pipe参数并使用默认值。
> - 在NPU架构版本3510上，CrossCoreWaitFlag传入的pipe参数生效，阻塞由模板参数pipe指定的流水的后续指令。

### 18.6 The scenario the timing diagrams assume — VERBATIM

> 下述同步特性均以如下场景配置为例：核函数（Kernel）使用__mix__(1, 2)修饰，即每个AI Core包含1个AIC和2个AIV，并设置逻辑核数numBlocks=2，即共启动2个AI Core，因此存在2个AIC和4个AIV。为便于描述，将2个AIC分别编号为AIC0、AIC1；AI Core0中的2个AIV分别编号为AIV0-0、AIV0-1，AI Core1中的2个AIV分别编号为AIV1-0、AIV1-1。各核中与flagId对应的计数器初始值均为0。
>
> 说明
>
> 以下针对模式0、1、2的描述，仅针对NPU架构版本2201有效，此时CrossCoreWaitFlag传入的pipe参数无效，因此采用默认值，默认阻塞全部流水的后续指令。

> **多AI Core中AIC或者AIV全核同步（模式0）**
>
> 多AI Core中AIC或者AIV全核同步需要配套使用模式0的CrossCoreSetFlag和CrossCoreWaitFlag接口。该场景可细分为两类：
>
> 2个AI Core中的2个AIC进行全核同步。
>
> 当2个AIC都执行完CrossCoreSetFlag时，每个AIC对应flagId的计数器增加1。若AIC该flagId的计数器为非0，被CrossCoreWaitFlag所阻塞的所有流水中的指令才会执行下去，该flagId的计数器减去1。

## Relevance

Confirms exactly which flagIds the library steals on 2201 and in which configuration: Mix SyncAll takes {11,12,13}, AIV-only SyncAll takes {14} alone, and Matmul takes [0, 2N-1]. For a hand-written engine that also instantiates Matmul objects, the only flagIds guaranteed free on 2201 with 4 Matmul objects + Mix SyncAll are {8, 9, 10, 15} — and 15 is also taken under SuperKernel 1:2 fusion.

## 19. ★★★ THE SMOKING GUN — the actual dav_c220 (`__NPU_ARCH__ == 2201`) lowering: `ffts_cross_core_sync` / `wait_flag_dev` / `set_ffts_base_addr` with the EXACT message bit layout

URL: https://raw.gitcode.com/cann/asc-devkit/raw/master/impl/basic_api/dav_c220/kernel_operator_sync_impl.h
Browsable: https://gitcode.com/cann/asc-devkit/blob/master/impl/basic_api/dav_c220/kernel_operator_sync_impl.h
Also: https://raw.gitcode.com/cann/asc-devkit/raw/master/impl/basic_api/kernel_operator_block_sync_intf_impl.h (arch dispatch)
Also: https://raw.gitcode.com/cann/asc-devkit/raw/master/include/basic_api/kernel_operator_block_sync_intf.h (declarations)
Author: Huawei Technologies Co., Ltd. (`Copyright (c) 2025 Huawei Technologies Co., Ltd.` / impl dispatcher is `(c) 2026`)
Date: fetched 2026-10-20 from `cann/asc-devkit` master

### 19.1 Arch dispatch — which impl file your A3 build actually uses — VERBATIM

From `impl/basic_api/kernel_operator_block_sync_intf_impl.h`:

```cpp
#if __NPU_ARCH__ == 1001
#include "dav_c100/kernel_operator_sync_impl.h"
#elif __NPU_ARCH__ == 2002
#include "dav_m200/kernel_operator_sync_impl.h"
#elif __NPU_ARCH__ == 2201
#include "dav_c220/kernel_operator_sync_impl.h"
#elif __NPU_ARCH__ == 3002
#include "dav_m300/kernel_operator_common_impl.h"
#elif __NPU_ARCH__ == 3003
#include "dav_l300/kernel_operator_sync_impl.h"
#elif __NPU_ARCH__ == 3102
#include "dav_m310/kernel_operator_common_impl.h"
#elif __NPU_ARCH__ == 3510
#include "dav_3510/kernel_operator_sync_impl.h"
#elif __NPU_ARCH__ == 9201
#include "dav_9201/kernel_operator_sync_impl.h"
#elif __NPU_ARCH__ == 9202
#include "dav_9202/kernel_operator_sync_impl.h"
#elif (__NPU_ARCH__ == 5102)
#include "dav_m510/kernel_operator_sync_impl.h"
#elif (__NPU_ARCH__ == 5162)
#include "dav_5162/kernel_operator_sync_impl.h"
```

### 19.2 Public declarations — VERBATIM from `include/basic_api/kernel_operator_block_sync_intf.h`

```cpp
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 5162) || \
                              (__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
template <bool isAIVOnly = true, const SyncAllConfig& config = DEFAULT_SYNC_ALL_CONFIG>
__aicore__ inline void SyncAll();
#else
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll();
#endif

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId);

template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);
```

Also declared in the same header (same namespace `AscendC`), for completeness of the inter-core-sync family:

```cpp
template <bool isAIVOnly = true>
__aicore__ inline void IBSet(
    const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, int32_t blockIdx,
    int32_t eventID);

template <bool isAIVOnly = true>
__aicore__ inline void IBWait(
    const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, int32_t blockIdx,
    int32_t eventID);

template <bool isAIVOnly = true>
__aicore__ inline void SyncAll(
    const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, const int32_t usedCores = 0);

namespace BlockSync {
enum SyncScope {
    InterBlock,
};

template <SyncScope Scope = SyncScope::InterBlock>
__aicore__ inline void MakeBarrier(uint8_t groupID, uint16_t arriveNum);
} // namespace BlockSync
```

Note `kernel_operator_block_sync_intf.h` begins with:

```cpp
#if defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
#error "kernel_operator_block_sync_intf.h cannot be used with compile flag --enable-simt enabled."
#endif
```

### 19.3 Thin wrappers — VERBATIM from `impl/basic_api/kernel_operator_block_sync_intf_impl.h`

```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
{
    NotifyEventImpl<modeId, pipe>(flagId);
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
{
    WaitEventImpl<modeId, pipe>(flagId);
}
```

And the legacy `NotifyEvent`/`WaitEvent` aliases — note that on everything that is NOT 3510/5102/5162/9201/9202 (i.e. on 2201) `NotifyEvent<pipe>` HARD-CODES `subBlockSyncMode = 0x02`:

```cpp
#if (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 5162) || (__NPU_ARCH__ == 9201) || \
    (__NPU_ARCH__ == 9202)
template <pipe_t pipe, uint8_t subBlockSyncMode = 2>
__aicore__ inline void NotifyEvent(uint16_t flagId)
{
    NotifyEventImpl<subBlockSyncMode, pipe>(flagId);
}

template <pipe_t pipe = PIPE_S, uint8_t mode = 0>
__aicore__ inline void WaitEvent(uint16_t flagId)
{
    WaitEventImpl<mode, pipe>(flagId);
}
#else
template <pipe_t pipe>
__aicore__ inline void NotifyEvent(uint16_t flagId)
{
    constexpr uint8_t subBlockSyncMode = 0x02;
    NotifyEventImpl<subBlockSyncMode, pipe>(flagId);
}

template <pipe_t pipe = PIPE_S>
__aicore__ inline void WaitEvent(uint16_t flagId)
```

### 19.4 ★★★ THE FFTS MESSAGE ENCODING AND THE RESERVED SyncAll FLAG NAMES — VERBATIM from `impl/basic_api/dav_c220/kernel_operator_sync_impl.h`

```cpp
constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

__aicore__ inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}
```

Decoded FFTS sync-message word layout on dav-2201:

```
bit  0      : 0x1          constant (valid / opcode bit)
bits [3:1]  : 0            unused
bits [5:4]  : mode & 0x3   <-- MODE FIELD IS ONLY 2 BITS WIDE
bits [7:6]  : 0            unused
bits [11:8] : flagId & 0xf <-- FLAG FIELD IS 4 BITS => exactly 16 flagIds
```

Two hard consequences for arch 2201 / Atlas A3:
1. `flagId & 0xf` — the 4-bit field is the mechanical reason the doc says "每个AIC和每个AIV都各自有16个flagId" and "如果flagId的值超出该范围，则截取低4bit". 16 flags per core, confirmed at the encoding level.
2. `mode & 0x3` — the mode field holds only values 0,1,2,3. `CrossCoreSetFlag<4, ...>` on 2201 therefore SILENTLY DEGRADES TO MODE 0 (`4 & 0x3 == 0`), with no compile error and no runtime trap. And mode 3 IS a legal, distinct, encodable hardware mode on 2201 that the public documentation never names.

### 19.5 ★ `NotifyEventImpl` / `WaitEventImpl` / `set_ffts_base_addr` for 2201 — VERBATIM

```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void NotifyEventImpl(uint16_t flagId)
{
    ffts_cross_core_sync(pipe, GetffstMsg(modeId, flagId));
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void WaitEventImpl(uint16_t flagId)
{
    (void)modeId;
    wait_flag_dev(flagId);
}

__aicore__ inline void SetSyncBaseAddrImpl(uint64_t config) { set_ffts_base_addr(config); }

__aicore__ inline void SetSyncBaseAddr(uint64_t config) { SetSyncBaseAddrImpl(config); }
```

Intrinsic signatures on 2201, as actually called:
- `ffts_cross_core_sync(pipe_t pipe, uint16_t msg)` — 2 args: pipe, and the packed `GetffstMsg(mode, flagId)` word.
- `wait_flag_dev(uint16_t flagId)` — ONE arg only. `modeId` is explicitly discarded by `(void)modeId;`; `pipe` is not even referenced. This is the source-level proof of the doc statement "modeId和pipe均不作为硬件指令的输入".
- `set_ffts_base_addr(uint64_t config)` — exposed publicly as `AscendC::SetSyncBaseAddr(uint64_t config)`.

### 19.6 ★ `SyncAllImpl` for 2201 — exactly which flags, which pipes, which modes — VERBATIM

```cpp
template <bool isAIVOnly = true>
__aicore__ inline void SyncAllImpl()
{
    ... (SuperKernel auto-sync-all block elided; see 12.8)

    PipeBarrier<PIPE_ALL>();
    if constexpr (isAIVOnly) {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
        return;
    }
    // trans software sync to hardware sync
    if ASCEND_IS_AIC {
        wait_flag_dev(SYNC_AIV_FLAG);
        ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
        wait_flag_dev(SYNC_AIC_FLAG);
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
        return;
    }
    if ASCEND_IS_AIV {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
        wait_flag_dev(SYNC_AIC_AIV_FLAG);
        return;
    }
}
```

Reading off the exact budget on 2201:
- `SyncAll<true>` (AIV-only): consumes exactly ONE flagId, `14`, with mode 0 set on `PIPE_MTE3`.
- `SyncAll<false>` (Mix): consumes exactly THREE flagIds, `11`, `12`, `13`. The AIC leg is `wait(12) -> set<mode0,PIPE_FIX>(11) -> wait(11) -> set<mode2,PIPE_MTE3>(13)`; the AIV leg is `set<mode2,PIPE_MTE3>(12) -> wait(13)`.
- It is prefixed by an unconditional `PipeBarrier<PIPE_ALL>()`.
- Note the asymmetry worth copying: the AIC->AIC mode-0 set uses `PIPE_FIX`, while every mode-2 cross-subblock set uses `PIPE_MTE3`.

### 19.7 ★★ MODE 2 IS USED IN BOTH DIRECTIONS BY HUAWEI'S OWN LIBRARY — VERBATIM, WITH HUAWEI'S OWN DIRECTION COMMENTS

From `SetNextTaskStartV3Impl` / `WaitPreTaskEndV3Impl` in the same 2201 file — these comments are Huawei's, not mine:

```cpp
__aicore__ inline void SetNextTaskStartV3Impl(uint32_t earlyStartConfig)
{
    if ASCEND_IS_AIC {
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIC_TO_AIC_SET) {
            ffts_cross_core_sync(AIC_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG)); // SET:C--->C
        }
        ...
            ffts_cross_core_sync(AIV_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL)); // SET: V--->V
```

```cpp
__aicore__ inline void WaitPreTaskEndV3Impl(uint32_t earlyStartConfig)
{
    if ASCEND_IS_AIC {
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIC_TO_AIC_WAIT) {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG); // WAIT: C--->C
        }
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIC_TO_AIV_SET) {
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG)); // SET: C--->V
        }
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIV_TO_AIC_WAIT) {
            wait_flag_dev(AscendC::SYNC_AIV_FLAG); // WAIT: V--->C
        }
    }
    if ASCEND_IS_AIV {
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIV_TO_AIV_WAIT) {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL); // WAIT: V--->V
        }
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIV_TO_AIC_SET) {
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG)); // SET: V--->C
        }
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIC_TO_AIV_WAIT) {
            wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG); // WAIT: C--->V
        }
    }
}
```

The two lines to stare at:

```cpp
ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG)); // SET: C--->V
ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));     // SET: V--->C
```

Same mode `0x02`, same pipe `PIPE_MTE3`, opposite directions. The DIRECTION is not carried by the mode at all — it is carried by WHICH CORE ISSUES the `ffts_cross_core_sync` (`ASCEND_IS_AIC` vs `ASCEND_IS_AIV`), and the two directions are kept apart purely by using DIFFERENT flagIds (13 for C->V, 12 for V->C).

### 19.8 `SoftSyncAllImpl` (the GM/UB software fallback used by `SyncAll(gmWorkspace, ubWorkspace, usedCores)`) — VERBATIM opening

```cpp
template <bool isAIVOnly = true>
__aicore__ inline void SoftSyncAllImpl(
    __gm__ int32_t* gmWorkspaceAddr, __ubuf__ int32_t* ubWorkspaceAddr, const int usedCores)
{
    if ASCEND_IS_AIC {
        return;
    }
    __sync_all_stub(usedCores, isAIVOnly);
    PipeBarrier<PIPE_ALL>();

    int32_t totalBlocks = isAIVOnly ? GetBlockNum() : (GetTaskRationImpl() * GetBlockNum());
    totalBlocks = usedCores != 0 ? usedCores : totalBlocks;
```

Note: the workspace form of `SyncAll` is an AIV-only spin on GM; `if ASCEND_IS_AIC { return; }` means the AIC leg is a no-op. It consumes NO flagIds at all.

### 19.9 Cross-check against dav_3510 — proves `mode & 0x3` is NOT an oversight

From `impl/basic_api/dav_3510/kernel_operator_sync_impl.h`, same constants and the SAME 2-bit mask:

```cpp
constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;
constexpr uint16_t SYNC_FLAG_ID_MAX = 16;

__aicore__ inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}
```

and mode 4 is implemented by an ENTIRELY DIFFERENT PAIR OF INTRINSICS, never by widening the mode field:

```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void NotifyEventImpl(uint16_t flagId)
{
    if constexpr (modeId == 4) {
        if constexpr (IsSplitVectorPipe<pipe>()) {
            if ASCEND_IS_AIV {
                set_intra_block(pipe, flagId);
            }
        }
        if constexpr (IsSplitCubePipe<pipe>()) {
            if ASCEND_IS_AIC {
                set_intra_block(pipe, flagId);
            }
        }
    } else {
        if constexpr (IsSplitVectorPipe<pipe>()) {
            if ASCEND_IS_AIV {
                ffts_cross_core_sync(pipe, GetffstMsg(modeId, flagId));
            }
        }
        if constexpr (IsSplitCubePipe<pipe>()) {
            if ASCEND_IS_AIC {
                ffts_cross_core_sync(pipe, GetffstMsg(modeId, flagId));
            }
        }
    }
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void WaitEventImpl(uint16_t flagId)
{
    if constexpr (modeId == 4) {
        if constexpr (IsSplitVectorPipe<pipe>()) {
            if ASCEND_IS_AIV {
                wait_intra_block(pipe, flagId);
            }
        }
        if constexpr (IsSplitCubePipe<pipe>()) {
            if ASCEND_IS_AIC {
                wait_intra_block(pipe, flagId);
            }
        }
    } else {
        if constexpr (IsSplitVectorPipe<pipe>()) {
            if ASCEND_IS_AIV {
                wait_flag_dev(pipe, flagId);
            }
        }
        if constexpr (IsSplitCubePipe<pipe>()) {
            if ASCEND_IS_AIC {
                wait_flag_dev(pipe, flagId);
            }
        }
    }
}

__aicore__ inline void SetSyncBaseAddrImpl(uint64_t config) { return; }
```

Three arch-difference facts fall out:
- `wait_flag_dev` has arity 1 on 2201 (`wait_flag_dev(flagId)`) and arity 2 on 3510 (`wait_flag_dev(pipe, flagId)`). Code written against 3510 does not port to 2201 by a recompile.
- mode 4 on 3510 = `set_intra_block` / `wait_intra_block`, a different instruction pair entirely. There is no mode-4 bit pattern to reach on 2201.
- `set_ffts_base_addr` exists ONLY on 2201; on 3510 `SetSyncBaseAddrImpl` is `{ return; }`, a deliberate no-op.

### 19.10 Also on 2201: `SYNC_FLAG_ID_MAX` is absent

The constant `constexpr uint16_t SYNC_FLAG_ID_MAX = 16;` appears in the 3510 file but NOT in the dav_c220 file. 2201 has no compile-time or runtime flagId bound check in this layer — the only bound is the silent `& 0xf` truncation.

## Relevance

This is the single most load-bearing section. It fixes the flag budget mechanically (4-bit field = 16 flags, matching our measurement), fixes SyncAll's exact footprint ({11,12,13} Mix / {14} AIV-only), and — critically — shows the mode field is only 2 bits on 2201, so mode 3 is real-but-undocumented while mode 4 silently aliases to mode 0. It also proves mode 2 is bidirectional in Huawei's own shipped code, with direction set by the issuing core and disambiguated by flagId, not by mode.

## 20. `AscendC::SyncAll` — official API reference (three prototypes; the `config` prototype is NOT supported on A3)

URL: https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.html
Raw markdown: https://raw.gitcode.com/cann/asc-devkit/raw/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md
Author: Huawei Technologies Co., Ltd. — Ascend C SIG. Doc topic id `ZH-CN_TOPIC_0000001538296817`
Date: fetched 2026-10-20

### 20.1 Three separate 产品支持情况 blocks — one per prototype — VERBATIM

```markdown
### 软同步原型
- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持
- Atlas 200I/500 A2推理产品：不支持
- Atlas推理系列产品AI Core：支持
- Atlas推理系列产品Vector Core：不支持
- Atlas训练系列产品：支持
- Kirin X90处理器系列产品：支持
- Kirin 9030处理器系列产品：支持

### 硬同步（不支持config指定流水）原型
- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持
- Atlas 200I/500 A2推理产品：不支持
- Atlas推理系列产品AI Core：不支持
- Atlas推理系列产品Vector Core：不支持
- Atlas训练系列产品：不支持
- Kirin X90处理器系列产品：支持
- Kirin 9030处理器系列产品：支持

### 硬同步（支持config指定流水）原型
- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持        <-- A3 CANNOT use the SyncAllConfig form
- Atlas A2系列产品：不支持
- Atlas 200I/500 A2推理产品：不支持
- Atlas推理系列产品AI Core：不支持
- Atlas推理系列产品Vector Core：不支持
- Atlas训练系列产品：不支持
- Kirin X90处理器系列产品：不支持
- Kirin 9030处理器系列产品：不支持
```

### 20.2 EXACT SIGNATURES — VERBATIM

```cpp
// 软同步
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, const int32_t usedCores = 0)

// 硬同步 - 不支持config指定流水  (THIS is the A3 form)
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll()

// 硬同步 - 支持config指定流水  (3510/950 ONLY)
template <bool isAIVOnly = true, const SyncAllConfig& config = DEFAULT_SYNC_ALL_CONFIG>
__aicore__ inline void SyncAll()
```

`SyncAllConfig`, VERBATIM from the `config` row of 表1:

```cpp
struct SyncAllConfig {
    pipe_t triggerPipe;
    pipe_t waitPipe;
};

// 默认使用全部流水来进行触发和等待行为。
constexpr SyncAllConfig DEFAULT_SYNC_ALL_CONFIG = {PIPE_ALL, PIPE_ALL};
```

> **该配置仅当isAIVOnly=True时有效。**
> 控制SyncAll函数的行为，在多个AI Core之间进行流水同步时，指定哪些流水（pipe）用于触发和等待。
> - **triggerPipe**：指定哪个流水用于"发送触发信号"。仅支持MTE2、MTE3、PIPE_ALL。
> - **waitPipe**：指定哪个流水用于"接收等待信号"。仅支持MTE2、MTE3、PIPE_ALL。

### 20.3 功能说明 — what the Mix hard-sync actually does (THREE syncs in one call) — VERBATIM

> 头文件路径为：`"basic_api/kernel_operator_block_sync_intf.h"`。
>
> SyncAll是核间同步控制接口，根据不同的函数原型介绍其功能：
>
> - 硬件同步接口：
>     用于多个核之间的同步控制，支持以下同步场景：
>     - 纯Vector算子的全核同步，可参考[CrossCoreSetFlag的模式0](key_features.md)。
>     - 融合算子的全核同步，一次性实现三个核间同步能力（参考[CrossCoreSetFlag的关键特性说明](key_features.md)）：首先完成单个AI Core内，AIC与所有AIV同步（AIC等AIV）；接着完成多个AI Core间，所有AIC的同步；最后完成单个AI Core内，所有AIV与AIC同步（AIV等AIC）。
>
> - 软件同步接口：
>     仅支持纯Vector算子的多核（参与同步的核数可通过入参指定）同步。每一个AIV都向全局内存对应位置写入信号值1，接着每一个AIV都持续轮询全局内存中所有AIV对应的信号值，直到所有AIV核对应的信号值都变为1。
>
> 在纯Vector算子场景中，若所有AIV核默认参与同步，推荐采用性能更优的硬件同步接口；若需指定部分AIV核参与同步，则应使用软件同步接口，并通过入参usedCores完成配置。

表1 isAIVOnly — VERBATIM:

> | isAIVOnly | 控制SyncAll作用于纯Vector算子或Mix（包含Cube和Vector计算）算子。可选值：<br>• **true**（默认值）：纯Vector算子的全核同步，仅执行Vector核的全核同步。<br>• **false**：Mix（包含Cube和Vector计算）算子的全核同步，先分别完成Vector核和Cube核的全核同步，再执行两者之间的同步（软同步接口不支持此功能）。 |

### 20.4 硬件同步接口约束 — VERBATIM

> - 硬件同步接口约束：
>     - 使用硬同步接口时，需根据场景使用合适的核函数（Kernel）修饰符：
>         - 在纯Vector算子，需使用\_\_mix\_\_\(0, 1\)修饰核函数（Kernel），而不是使用\_\_vector\_\_，具体原因请参考[CrossCoreSetFlag约束说明](CrossCoreSetFlag_ISASI.md#约束说明)中的核函数（Kernel）类型配置说明。
>         - 对于Mix（包含Cube和Vector计算）算子，需使用\_\_mix\_\_\(1, 1\)或\_\_mix\_\_\(1, 2\)修饰核函数（Kernel）。
>     - 对于纯Vector算子，模板参数isAIVOnly必须设置为true，否则会导致程序卡死。
>     - 对于Mix（包含Cube和Vector计算）算子，模板参数isAIVOnly设置为true，会导致接口实际只能完成Vector全核同步与预期同步行为不符。
>     - 使用该接口时，建议开启batchmode模式，使算子独占全部所需核资源，否则可能因满足以下条件导致死锁：
>         - 多流并发场景（≥2条执行流）。
>         - ≥2个算子并发执行。
>         - 所有并发算子的核数总和超过物理核数。
>         - ≥2个并发算子使用了核间同步功能。
>     - SyncAll硬件同步接口内部实现中使用了[CrossCoreSetFlag](CrossCoreSetFlag_ISASI.md)和[CrossCoreWaitFlag](CrossCoreWaitFlag_ISASI.md)进行核间同步控制，因此开发者同时使用CrossCoreSetFlag（或CrossCoreWaitFlag）和SyncAll硬件同步接口时，需注意避免flagId冲突，各接口flagId占用的完整说明参见[核间同步flagId占用说明](key_features.md#inter_core_sync_flagid_usage)。
>
> - 硬件同步接口和软件同步接口公共约束：使用该接口进行多核控制时，算子调用时指定的逻辑AI Core核数numBlocks必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致核函数（Kernel）"卡死"现象。

软件同步接口约束 — VERBATIM:

> - 软件同步接口约束：
>     - gmWorkspace缓存申请的空间大小要求大于等于GetBlockNum()*32Bytes，并且缓存的值需要初始化为0。
>     - ubWorkspace申请的空间大小要求大于等于GetBlockNum()*32Bytes。
>     - usedCores传入数值不能超过算子调用时指定的逻辑AI Core中AIV的数量：GetBlockNum()*GetTaskRatio()，不传此参数表示全核软同步。

## Relevance

The A3-relevant SyncAll is `template <bool isAIVOnly> void SyncAll()` with no config. The `SyncAllConfig{triggerPipe, waitPipe}` prototype is explicitly 不支持 on A3 — do not plan around pipe-scoped SyncAll on 2201. Hard constraint for a hand-written engine: a pure-Vector kernel must be `__mix__(0, 1)`, NOT `__vector__`, or the scheduler block is never enabled and the sync hangs.

## 21. ★★ THE C-API SIBLING FAMILY — each mode gets a NAME, and the per-core-type pipe tables are SHARPER than the C++ page

URL (overview, the mode->name table): https://asc.gitcode.com/api/SIMD-API/c_api/sync/inter_core_sync/inter_core_sync_overview.html
URL (mode 0): https://asc.gitcode.com/api/SIMD-API/c_api/sync/inter_core_sync/asc_sync_inter_arrive.html , .../asc_sync_inter_wait.html
URL (mode 1): https://asc.gitcode.com/api/SIMD-API/c_api/sync/inter_core_sync/asc_sync_subblock_arrive.html , .../asc_sync_subblock_wait.html
URL (mode 2): https://asc.gitcode.com/api/SIMD-API/c_api/sync/inter_core_sync/asc_sync_block_arrive.html , .../asc_sync_block_wait.html
URL (mode 4): https://asc.gitcode.com/api/SIMD-API/c_api/sync/inter_core_sync/asc_sync_intra_arrive.html , .../asc_sync_intra_wait.html
Raw markdown base: https://raw.gitcode.com/cann/asc-devkit/raw/master/docs/zh/api/SIMD-API/c_api/sync/inter_core_sync/
Author: Huawei Technologies Co., Ltd. — Ascend C SIG
Date: fetched 2026-10-20

### 21.1 ★ 表2 支持的核间同步场景 — THE MODE->NAME->SCOPE TABLE — VERBATIM

| 接口名称 | 同步范围 | 对应模式 | 功能简述 |
| --- | --- | --- | --- |
| [asc_sync_inter_arrive](asc_sync_inter_arrive.md)/[asc_sync_inter_wait](asc_sync_inter_wait.md) | 组间同步，即不同group之间所有block（所有AIC或所有AIV）之间的同步。 | 模式0（同步等待全部AIC或全部AIV执行结束） | `asc_sync_inter_arrive`为生产者，负责发送通知；`asc_sync_inter_wait`为消费者，负责阻塞等待通知到达。 |
| [asc_sync_subblock_arrive](asc_sync_subblock_arrive.md)/[asc_sync_subblock_wait](asc_sync_subblock_wait.md) | 组内subblock间的同步，即同一group内不同subblock（AIV）之间的同步。 | 模式1（单个AI Core内，全部AIV之间的同步） | `asc_sync_subblock_arrive`为生产者，负责发送通知；`asc_sync_subblock_wait`为消费者，负责阻塞等待通知到达。 |
| [asc_sync_block_arrive](asc_sync_block_arrive.md)/[asc_sync_block_wait](asc_sync_block_wait.md) | 同一group内block（AIC）与所有subblock（AIV）之间的同步。 | 模式2（单个AI Core内，AIC与全部AIV之间的同步） | `asc_sync_block_arrive`为生产者，负责发送通知；`asc_sync_block_wait`为消费者，负责阻塞等待通知到达。 |
| [asc_sync_intra_arrive](asc_sync_intra_arrive.md)/[asc_sync_intra_wait](asc_sync_intra_wait.md) | 同一group内block（AIC）与单个subblock（AIV）之间的同步。 | 模式4（单个AI Core内，AIC核与单个AIV之间同步） | `asc_sync_intra_arrive`为生产者，负责发送通知；`asc_sync_intra_wait`为消费者，负责阻塞等待通知到达。 |

Every one of these pages says 「[四种核间同步模式](../system_sync_overview.md#fig_sync_control_mode)」 — FOUR modes, named 0/1/2/4. The word 「四种」 is Huawei's; it is a closed enumeration and mode 3 is not in it.

表1 group配置 — VERBATIM (the block/subblock mapping that decides which mode is even legal):

| 算子类型 | 函数执行空间限定符 | 核函数（Kernel）类型 | block/subblock | 1：N |
| --- | --- | --- | --- | --- |
| Cube算子 | `__cube__` | `KERNEL_TYPE_AIC_ONLY` | AIC为block，无subblock | 不涉及 |
| Vector算子 | `__vector__` | `KERNEL_TYPE_AIV_ONLY` | AIV为block，无subblock | 不涉及 |
| Mix算子 | `__mix__(1, 0)` | `KERNEL_TYPE_MIX_AIC_1_0` | AIC为block，无subblock | 不涉及 |
| Mix算子 | `__mix__(0, 1)` | `KERNEL_TYPE_MIX_AIV_1_0` | AIV为block，无subblock | 不涉及 |
| Mix算子 | `__mix__(1, 1)` | `KERNEL_TYPE_MIX_AIC_1_1` | AIC为block，AIV为subblock | 1：1 |
| Mix算子 | `__mix__(1, 2)` | `KERNEL_TYPE_MIX_AIC_1_2` | AIC为block，AIV为subblock | 1：2 |

> AIC/AIV多核结构如下图2所示，AIC/AIV按group划分。一个group内包含1个block和N个subblock（N≥1），其中block表示"主核"，每个subblock表示一个"从核"。
> ... AIV为block且AIC为subblock的配置（即Vector核为主、Cube核为辅的模式）当前不支持。

### 21.2 EXACT C SIGNATURES — VERBATIM

```c
__aicore__ inline void asc_sync_inter_arrive(pipe_t pipe, int64_t flag_id)      // mode 0 producer
__aicore__ inline void asc_sync_subblock_arrive(pipe_t pipe, int64_t flag_id)   // mode 1 producer
__aicore__ inline void asc_sync_block_arrive(pipe_t pipe, int64_t flag_id)      // mode 2 producer
__aicore__ inline void asc_sync_block_wait(pipe_t pipe, int64_t flag_id)        // mode 2 consumer
__aicore__ inline void asc_sync_intra_arrive(pipe_t pipe, uint64_t sync_id)     // mode 4 producer (NOTE: uint64_t sync_id, not flag_id)
```

Header for all of them: `"c_api/sync/sync.h"`. 流水类型 for all of them: `PIPE_S`.

### 21.3 ★★★ THE flagId COUNTER WIDTH, STATED EXPLICITLY — VERBATIM

> | flag_id | 输入 | 核间同步的标记，用于标识同一组同步信号。取值范围为[0, 15]，每个`flag_id`各自拥有独立的**4位计数器**。 |

And:

> - 每个计数器最多连续累加15次（此时计数器的值为15），必须保证计数器的值不超过15，否则触发异常。
> - 本接口不阻塞`pipe`流水中的后续指令。
> - `flag_id`取值范围为[0, 15]，超出范围值会被按位宽截断处理为低4位（例如，flagId=16时，截取后为0；flagId=17时，截取后为1）。

So: 16 flagIds, each with its own independent 4-BIT counter, max 15 pending sets, 16th set traps. The truncation example here (`flagId=16` -> `0`) is more informative than the C++ page's (`flagId=17` -> `1`).

### 21.4 ★★★ THE REAL PIPE RULES FOR 2201, PER CORE TYPE — VERBATIM (this is SHARPER than the C++ CrossCoreSetFlag page)

For mode 0 (`asc_sync_inter_arrive`) AND mode 2 (`asc_sync_block_arrive`) the tables are IDENTICAL:

```markdown
- 针对[NPU架构版本3510]，AIC和AIV支持的`pipe`取值如[表2]所示。

    **表2** NPU架构3510中AIC和AIV支持的`pipe`取值

    | 核类型 | 支持的pipe取值 |
    | :--- | :--- |
    | AIC | `PIPE_M`、`PIPE_MTE1`、`PIPE_MTE2`、`PIPE_FIX` |
    | AIV | `PIPE_MTE2`、`PIPE_MTE3`、`PIPE_V` |

- 针对[NPU架构版本2201]，AIC和AIV支持的`pipe`取值如[表3]所示。

    **表3** NPU架构2201中AIC和AIV支持的`pipe`取值

    | 核类型 | 支持的pipe取值 |
    | :--- | :--- |
    | AIC | `PIPE_M`、`PIPE_MTE1`、`PIPE_MTE2`、`PIPE_MTE3`、`PIPE_FIX` |
    | AIV | `PIPE_MTE2`、`PIPE_MTE3`、`PIPE_V` |
```

For mode 1 (`asc_sync_subblock_arrive`), VERBATIM:

> - 本接口仅支持AIV调用，AIV支持的`pipe`取值如下：`PIPE_MTE2`、`PIPE_MTE3`、`PIPE_V`。

★ THE PIPE_FIX / PIPE_MTE3 / PIPE_S RULE FOR dav-2201, decoded:
- `PIPE_FIX` as a SET pipe: AIC ONLY. It is in the AIC row and absent from the AIV row. A vector core cannot set on PIPE_FIX.
- `PIPE_V` as a SET pipe: AIV ONLY. Absent from the AIC row.
- `PIPE_MTE3` as a SET pipe: LEGAL ON BOTH on 2201 — and note the 3510 AIC row DROPS `PIPE_MTE3`. So `CrossCoreSetFlag<2, PIPE_MTE3>` from the AIC is a 2201-only idiom; it will not survive a port to 950.
- `PIPE_M` / `PIPE_MTE1`: AIC only.
- `PIPE_S` on the SET/arrive side: absent from BOTH 2201 rows for modes 0/1/2. Only mode 4 (3510-only) admits `PIPE_S` on arrive.
- The intersection legal on both core types on 2201 is exactly `{PIPE_MTE2, PIPE_MTE3}`.

### 21.5 ★ THE WAIT SIDE ON 2201 — "硬件不支持配置核间同步模式和流水类型" — VERBATIM

From `asc_sync_block_wait` 约束说明:

```markdown
- 针对`asc_sync_block_arrive`接口，传入的`pipe`参数在不同NPU架构中**均生效**；针对`asc_sync_block_wait`接口，传入的`pipe`参数**是否生效与NPU架构有关**，具体情况如下：
    - 针对[NPU架构版本3510]，硬件支持配置核间同步模式和流水类型，输入参数`pipe`**生效**，此时`asc_sync_block_wait`会阻塞由**`pipe`指定的流水**的后续指令。AIC和AIV支持的`pipe`取值如[表2]所示。

        **表2** NPU架构3510中AIC和AIV支持的`pipe`取值

        | 核类型 | 支持的pipe取值 |
        | :--- | :--- |
        | AIC | `PIPE_S`、`PIPE_M`、`PIPE_MTE1`、`PIPE_MTE2`、`PIPE_FIX` |
        | AIV | `PIPE_S`、`PIPE_MTE2`、`PIPE_MTE3`、`PIPE_V` |
    - 针对[NPU架构版本2201]，硬件不支持配置核间同步模式和流水类型，输入参数`pipe`**不生效**，此时`asc_sync_block_wait`会阻塞**全部流水**的后续指令。
```

The clause 「硬件不支持配置核间同步模式和流水类型」 is the explicit authoritative statement that on dav-2201 the WAIT instruction has NO mode field and NO pipe field. It matches `(void)modeId; wait_flag_dev(flagId);` in section 12.5 exactly.

### 21.6 ★ MODE 2 IS BIDIRECTIONAL — the official two-scenario description — VERBATIM

From both `asc_sync_block_arrive` and `asc_sync_block_wait` 功能说明:

> 本接口与[asc_sync_block_wait](asc_sync_block_wait.md)配对使用，实现单AI Core内AIC与全部AIV之间的同步（[四种核间同步模式](../system_sync_overview.md#fig_sync_control_mode)中的模式2），核间同步实现的原理如下：
>
> - 全部AIV等待单个AIC的场景：
>   - 单AI Core内AIC执行`asc_sync_block_arrive`后向调度模块发送通知，接着调度模块将该AI Core内M个AIV各自对应`flag_id`的计数器增加1。
>   - 各AIV上配对的`asc_sync_block_wait`检测到对应`flag_id`的计数器非0后解除阻塞并将计数器减1。
> - 单个AIC等待全部AIV的场景：
>   - 单AI Core内M个AIV都执行`asc_sync_block_arrive`后向调度模块发送通知，接着调度模块将AIC对应`flag_id`的计数器增加1。
>   - AIC上配对的`asc_sync_block_wait`检测到对应`flag_id`的计数器非0后解除阻塞并将计数器减1。

Both directions, ONE mode (mode 2), with the SAME function. There is no separate "reverse" mode in the official model on 2201. Note the fan-out/fan-in asymmetry: AIC->AIV is a BROADCAST (one arrive bumps M AIV counters); AIV->AIC is a GATHER (all M AIV must arrive before the single AIC counter is bumped once).

### 21.7 Mode 4 (3510-only) `sync_id` semantics and the cross-core ID MAPPING — VERBATIM, for contrast

> | sync_id | 输入 | 核间同步的标记，用于标识同一组同步信号。每个`sync_id`各自拥有独立的4位计数器。一个AI Core由1个AIC与2个AIV构成，AIC侧拥有32个`sync_id`（0~31），每个AIV侧各拥有16个`sync_id`（0~15）。 |

> - 与其它核间同步模式不同，本接口与`asc_sync_intra_wait`配对使用时，**不要求**`arrive`与`wait`传入的`sync_id`相同，而是要求二者符合如下跨核ID映射关系，否则会出现未定义行为。
>     - AIV0中调用的`asc_sync_intra_arrive`接口的`sync_id`取值为0~15，分别与AIC中调用的`asc_sync_intra_wait`接口的`sync_id`取值0~15对应。
>     - AIV1中调用的`asc_sync_intra_arrive`接口的`sync_id`取值为0~15，分别与AIC中调用的`asc_sync_intra_wait`接口的`sync_id`取值16~31对应。
>     - AIC中调用的`asc_sync_intra_arrive`接口的`sync_id`取值为0~15，分别与AIV0中调用的`asc_sync_intra_wait`接口的`sync_id`取值0~15对应。
>     - AIC中调用的`asc_sync_intra_arrive`接口的`sync_id`取值为16~31，分别与AIV1中调用的`asc_sync_intra_wait`接口的`sync_id`取值0~15对应。
> - 每个计数器最多连续累加15次（此时计数器的值为15），必须保证计数器的值不超过15，否则触发异常。
> - 本接口不阻塞`pipe`流水中的后续指令。
> - AIC的`sync_id`取值范围为[0, 31]，超出范围值会被按位宽截断处理为低5位（例如，syncId=32时，截取后为0；syncId=33时，截取后为1）。
> - AIV的`sync_id`取值范围为[0, 15]，超出范围值会被按位宽截断处理为低4位（例如，syncId=16时，截取后为0；syncId=17时，截取后为1）。

表2 AIC和AIV支持的`pipe`取值 (mode 4, 3510 only) — VERBATIM:

| 核类型 | 支持的pipe取值 |
| :--- | :--- |
| AIC | `PIPE_S`、`PIPE_M`、`PIPE_MTE1`、`PIPE_MTE2`、`PIPE_FIX` |
| AIV | `PIPE_S`、`PIPE_MTE2`、`PIPE_MTE3`、`PIPE_V` |

And the mode-4 worked timing narrative, VERBATIM:

> 以图1为例，演示1个AI Core中AIV0与AIC进行同步（AIV0-1发起asc_sync_intra_arrive）。
>
> AIC 0中在执行asc_sync_intra_wait后，此时AIC 0 sync_id=0的计数器为0，后续所有指令全部被阻塞，需要等到1个AIV执行完asc_sync_intra_arrive。
>
> - AIV 0-0不需要执行asc_sync_intra_arrive。
> - AIV 0-1的PIPE_MTE3指令全部执行完毕后，asc_sync_intra_arrive生效。此时调度模块感知1个AIV已执行完asc_sync_intra_arrive，因此将AIC 0 sync_id=0的计数器值增加为1。AIC 0检测到对应的sync_id=0的计数器变为1，则AIC 0核解除阻塞，继续执行后续PIPE_FIX的指令，并且将计数器值减去1。

### 21.8 Kernel-qualifier legality, per mode — VERBATIM from each page's first 约束 bullet

> - mode 0 (`asc_sync_inter_*`): 支持的函数执行空间限定符为`__mix__(0, 1)`、`__mix__(1, 0)`、`__mix__(1, 1)`、`__mix__(1, 2)`。
> - mode 1 (`asc_sync_subblock_*`): 支持的函数执行空间限定符为`__mix__(1, 1)`、`__mix__(1, 2)`。
> - mode 2 (`asc_sync_block_*`): 支持的函数执行空间限定符为`__mix__(1, 1)`、`__mix__(1, 2)`。
> - mode 4 (`asc_sync_intra_*`): 支持的函数执行空间限定符为`__mix__(1, 2)`。
>
> (all four pages share:) 调用本接口的核函数（Kernel）不能使用`__cube__`或`__vector__`函数执行空间限定符。使用这两种函数执行空间限定符时，硬件不会开启调度模块，无法正常进行核间同步。

### 21.9 A COMPLETE, COMPILABLE MODE-2 EXAMPLE (AIV->AIC gather) — VERBATIM

From `asc_sync_block_arrive` 调用示例. Note `constexpr int64_t FLAG_ID = 8;` — Huawei picks 8, i.e. above the Matmul `[0,2N-1]` band and below the SyncAll `11..14` band.

```cpp
/*
 * 本示例说明：
 * - 采用__mix__(1, 2)标识核函数（Kernel），同一AI Core内包含1个AIC和2个AIV。
 * - AIV0和AIV1分别将矩阵从GM搬入UB，通过asc_get_sub_block_id区分不同的AIV。
 * - 两个AIV将矩阵从UB搬运至L1 Buffer（PIPE_MTE3）操作执行完后，才能执行asc_sync_block_arrive向调度模块发送通知。
 * - AIC中从L1 Buffer搬运数据到L0A Buffer、L0B Buffer（PIPE_MTE1）的操作被asc_sync_block_wait阻塞，直到两个AIV均执行了asc_sync_block_arrive后才能执行。
 */
__global__ __mix__(1, 2) void asc_sync_block_kernel(__gm__ half* input, __gm__ float* output)
{
    asc_init();
    __cbuf__ half matrix_a_l1[MATRIX_ELEMENTS], matrix_b_l1[MATRIX_ELEMENTS];
    __ubuf__ half input_ub[MATRIX_ELEMENTS];
    __ca__ half matrix_a_l0[MATRIX_ELEMENTS];
    __cb__ half matrix_b_l0[MATRIX_ELEMENTS];
    __cc__ float output_l0[MATRIX_ELEMENTS];

    if ASC_IS_AIV {
        asc_copy_gm2ub_align(input_ub, input, MATRIX_BYTES);
        asc_sync_notify(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        if (asc_get_sub_block_id() == 0) {
            asc_copy_ub2l1(matrix_a_l1, input_ub, MATRIX_BYTES);
        } else {
            asc_copy_ub2l1(matrix_b_l1, input_ub, MATRIX_BYTES);
        }
        asc_sync_block_arrive(PIPE_MTE3, FLAG_ID);
    }

    if ASC_IS_AIC {
        asc_sync_block_wait(PIPE_MTE1, FLAG_ID);
        asc_copy_l12l0a(matrix_a_l0, matrix_a_l1, 0, 0, 1, 1, 1, 1);
        asc_copy_l12l0b_transpose(matrix_b_l0, matrix_b_l1, 0, 0, 1, 1, 1, 1);
        asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID0);
        asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID0);
        asc_mmad(output_l0, matrix_a_l0, matrix_b_l0, MATRIX_DIM, MATRIX_DIM, MATRIX_DIM, asc_unit_flag_mode::DISABLE, false, false, true);
        ...
    }
    asc_sync_pipe(PIPE_ALL);
}
```

Note: `numBlocks = 1` in this example's host code, i.e. mode 2 is demonstrated intra-AI-Core only.

## Relevance

The C-API family is the authoritative answer to the "PIPE_FIX vs PIPE_MTE3 vs PIPE_S" question that the C++ page blurs: on 2201 the set pipe must be in the ISSUING CORE's row — AIC {M, MTE1, MTE2, MTE3, FIX}, AIV {MTE2, MTE3, V} — and PIPE_S is in NEITHER. It also states the counter is a 4-BIT counter with a 15-set ceiling that TRAPS, and that mode 2 is officially bidirectional with broadcast-out / gather-in asymmetry.

## 22. `set_ffts_base_addr` / `asc_set_ffts_base_addr` / `asc_get_ffts_base_addr` — the FFTS sync register base address (A3/A2 ONLY, not on 950)

URL: https://asc.gitcode.com/api/SIMD-API/c_api/utils/sys_var/asc_set_ffts_base_addr.html
URL: https://asc.gitcode.com/api/SIMD-API/c_api/utils/sys_var/asc_get_ffts_base_addr.html
Raw markdown: https://raw.gitcode.com/cann/asc-devkit/raw/master/docs/zh/api/SIMD-API/c_api/utils/sys_var/asc_set_ffts_base_addr.md (and `asc_get_ffts_base_addr.md`)
Author: Huawei Technologies Co., Ltd. — Ascend C SIG
Date: fetched 2026-10-20

### 22.1 产品支持情况 — INVERTED relative to CrossCoreSetFlag — VERBATIM

```markdown
<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- Atlas 200I/500 A2推理产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas推理系列产品AI Core：不支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- Atlas推理系列产品Vector Core：不支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- Atlas训练系列产品：不支持
<!-- end id7 -->
```

This matches the source exactly: dav_c220's `SetSyncBaseAddrImpl` calls `set_ffts_base_addr(config)`, while dav_3510's is `{ return; }` (section 12.9).

### 22.2 功能说明 + EXACT SIGNATURES — VERBATIM

> 头文件路径为：`"c_api/utils/sys_var.h"`。
>
> 在[asc_sync_block_wait](../../sync/inter_core_sync/asc_sync_block_wait.md)和[asc_sync_block_arrive](../../sync/inter_core_sync/asc_sync_block_arrive.md)之前使用，设置核间同步寄存器的基地址。需要在Host侧调用接口aclrtGetHardwareSyncAddr获取，并作为参数传入。
>
> ```cpp
> aclError aclrtGetHardwareSyncAddr(void **addr);
> ```

```cpp
__aicore__ inline void asc_set_ffts_base_addr(uint64_t config)
__aicore__ inline int64_t asc_get_ffts_base_addr()
```

> | config | 输入 | 核间同步的基地址。取值范围[0, 2^48-1]。 |

流水类型 for both: `PIPE_S`.

### 22.3 约束说明 — VERBATIM

> (asc_set_ffts_base_addr)
> - 在使用[asc_sync_block_wait]和[asc_sync_block_arrive]之前**必须**使用本接口设置基地址。
>
> (asc_get_ffts_base_addr)
> 本接口为只读查询接口，读取的是核间同步寄存器的基地址，调用方不可直接设置或修改返回的基地址值；基地址需通过[asc_set_ffts_base_addr](asc_set_ffts_base_addr.md)预先配置后方可用于核间同步。

### 22.4 Worked example, with the exact bisheng command line for dav-2201 — VERBATIM

> 以[NPU架构版本2201]为例，编译运行命令如下：
>
> ```bash
> bisheng example.asc -o main --npu-arch=dav-2201 && ./main
> ```

```cpp
/*
 * 本示例说明：
 * - ffts_addr由Host侧调用aclrtGetHardwareSyncAddr获得。
 * - 核函数（Kernel）将ffts_addr转换为uint64_t类型并设置核间同步寄存器基地址。
 * - 核函数（Kernel）回读基地址，Host侧检查设置值与回读值是否一致。
 */
__global__ __vector__ void asc_get_ffts_base_addr_kernel(
    __gm__ uint64_t* output, __gm__ uint64_t* ffts_addr)
{
    asc_init();
    const uint64_t config = reinterpret_cast<uint64_t>(ffts_addr);
    asc_set_ffts_base_addr(config);
    output[0] = config;
    output[1] = static_cast<uint64_t>(asc_get_ffts_base_addr());
    asc_dcci_single(output);
}
```

Host side, VERBATIM:

```cpp
    aclrtGetHardwareSyncAddr(&ffts_addr);
    asc_get_ffts_base_addr_kernel<<<1, 0>>>(
        output_device, reinterpret_cast<uint64_t*>(ffts_addr));
```

And the simpler snippet from the set page:

```cpp
// Host侧调用接口aclrtGetHardwareSyncAddr获取核间同步基地址ffts_addr
uint64_t config = *(__gm__ uint64_t*)ffts_addr;
asc_set_ffts_base_addr(config);
```

Note the two pages disagree on how to derive `config`: the SET page does `*(__gm__ uint64_t*)ffts_addr` (dereference), the GET page does `reinterpret_cast<uint64_t>(ffts_addr)` (the pointer value itself). The C++ wrapper `AscendC::SetSyncBaseAddr(uint64_t config)` passes `config` straight through.

## Relevance

On A3 the FFTS base address MUST be programmed before any cross-core sync when you drive the C-API path directly (「必须」), via host `aclrtGetHardwareSyncAddr`. This is a hard prerequisite a hand-written engine bypassing the AscendC C++ layer can silently miss. It is also A3/A2-only — the knob does not exist on 950.

## 23. ★★ HUAWEI'S OWN REFERENCE SAMPLE for all four modes, CANN >= 9.2.0, with the A3 vs 950 arch split spelled out

URL: https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/03_basic_api/05_sync_control/cross_core_set_wait_flag
Raw kernel: https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/03_basic_api/05_sync_control/cross_core_set_wait_flag/cross_core_set_wait_flag.h
Raw README: https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/03_basic_api/05_sync_control/cross_core_set_wait_flag/README.md
Author: Huawei Technologies Co., Ltd. (`cann/asc-devkit` examples)
Date: fetched 2026-10-20

### 23.1 Supported product / CANN version matrix — VERBATIM (matches our exact target)

> ## 本样例支持的产品及CANN软件版本
>
> | 产品 | CANN软件版本 |
> |------|-------------|
> | Ascend 950PR&950DT系列产品 | >= CANN 9.2.0 |
> | Atlas A3系列产品 | >= CANN 9.2.0 |
> | Atlas A2系列产品 | >= CANN 9.2.0 |

> 本样例介绍核间同步接口CrossCoreSetFlag和CrossCoreWaitFlag的模式0、模式1、模式2和模式4，并通过纯Vector计算以及Cube与Vector融合计算场景说明各模式的使用方法。**其中模式4仅支持Ascend 950PR/Ascend 950DT。**

README mode table — VERBATIM (HTML table, transcribed):

| 同步控制模式 | 说明 |
| --- | --- |
| mode 0 | 对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到`CrossCoreSetFlag`时，`CrossCoreWaitFlag`后续的指令才会执行。 |
| mode 0 | 对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到`CrossCoreSetFlag`时，`CrossCoreWaitFlag`后续的指令才会执行。 |
| mode 1 | 单个AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了`CrossCoreSetFlag`，`CrossCoreWaitFlag`后续的指令才会执行。 |
| mode 2 | 在AIC核执行`CrossCoreSetFlag`之后，两个AIV上`CrossCoreWaitFlag`后续的指令才会继续执行。 |
| mode 2 | 两个AIV都执行`CrossCoreSetFlag`后，AIC上`CrossCoreWaitFlag`后续的指令才能执行。 |
| mode 4 | 单个AI Core内部，AIC与指定的单个AIV之间进行同步控制。AIV0和AIV1通过不同的`flagId`映射分别与AIC同步。 |

Scenario map — VERBATIM:

> | `SCENARIO_NUM` | `0`（默认）、`1`、`2`、`3` | 场景编号：0（纯Vector模式0）、1（纯Vector模式1）、2（Cube+Vector模式0/2）、3（Cube+Vector模式4，仅NPU架构`dav-3510`） |

### 23.2 ★ The sample's flagId constant block — VERBATIM (note: it DELIBERATELY collides with SyncAll's 11/12/13/14)

```cpp
// 模式0的flagId
constexpr uint16_t SYNC_AIC_FLAG = 11;
// 模式2的flagId,AIC等AIV
constexpr uint16_t SYNC_AIV_AIC_FLAG = 12;
// 模式2的flagId,AIV等AIC
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
// 模式4中，AIV1的flagId映射到AIC时需要加16。
constexpr uint16_t SYNC_MODE4_INPUT_FLAG = 14;
constexpr uint16_t SYNC_MODE4_OUTPUT_FLAG = 15;
constexpr uint16_t SYNC_MODE4_AIV1_OFFSET = 16;
```

(This sample does not call `SyncAll`, so the reuse of 11/12/13 is safe there — but it is the exact band `SyncAll<false>` claims per section 11.2/12.6. Do not copy these constants into a kernel that also calls `SyncAll`.)

### 23.3 ★★ THE ARCH-COMPAT WRAPPER AND ITS COMMENT — VERBATIM

```cpp
// dav-3510将CrossCoreWaitFlag的pipe模板参数作为硬件指令输入，模式0/1/2不支持默认的PIPE_S；
// dav-2201上该参数不影响硬件指令，保持省略参数可兼容两类架构。
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreWaitFlagForArch(uint16_t flagId)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    AscendC::CrossCoreWaitFlag<modeId, pipe>(flagId);
#else
    AscendC::CrossCoreWaitFlag<modeId>(flagId);
#endif
}
```

README restatement — VERBATIM:

> (3) `CrossCoreWaitFlag`的`pipe`模板参数在NPU架构`dav-2201`上不影响硬件指令，但在NPU架构`dav-3510`上会生效。由于NPU架构`dav-3510`的模式0、模式1和模式2**不支持默认的`PIPE_S`**，样例代码通过架构条件分别处理：NPU架构`dav-3510`显式传入`PIPE_MTE2`或`PIPE_FIX`，NPU架构`dav-2201`保留默认参数行为。

NOTE A DOC-VS-SAMPLE CONFLICT (3510 only, not A3): the `CrossCoreWaitFlag` reference page claims for 950 that 「模式0、模式1、模式2和模式4支持`PIPE_S`、`PIPE_V`、...」, but this sample and its README both say 「NPU架构`dav-3510`的模式0、模式1和模式2不支持默认的`PIPE_S`」. The sample is the more conservative source. Irrelevant for A3, where the parameter is inert either way.

### 23.4 ★★★ MODE 2 USED IN BOTH DIRECTIONS, SAME modeId, DIFFERENT flagId AND DIFFERENT SET PIPE — VERBATIM

AIC leg (`ProcessAIC`), 2201 path:

```cpp
        } else {
            // 模式2：AIC等待本AI Core内2个AIV完成精度转换。
            CrossCoreWaitFlagForArch<2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG);
        }

        CopyIn(a1Local, b1Local);
        SplitA(a1Local, a2Local);
        SplitBTranspose(b1Local, b2Local);
        Compute(a2Local, b2Local, c1Local);
            CopyOut(c1Local);

            // 模式0：8个AIC全核同步，确保原子累加结果正确。
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(SYNC_AIC_FLAG);
            CrossCoreWaitFlagForArch<0, PIPE_FIX>(SYNC_AIC_FLAG);

            // 模式2：AIC通知本AI Core内2个AIV可以执行LeakyRelu。
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
```

AIV leg (`ProcessAIV`), 2201 path:

```cpp
        } else {
            // 模式2：AIV通知本AI Core内AIC精度转换完成。
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);

            // 模式2：AIV等待本AI Core内AIC完成矩阵乘。
            CrossCoreWaitFlagForArch<2, PIPE_MTE2>(SYNC_AIC_AIV_FLAG);
            AscendC::DataCopy(cLocal, CVectorGM, C_AIV_BLOCKS_LENGTH);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
```

The canonical A3 cube<->vector handshake, extracted:

```
AIC:  CrossCoreWaitFlag<2>(12)                  // gather: wait both AIV
      ... matmul ...
      CrossCoreSetFlag<0, PIPE_FIX>(11)         // mode 0: all-AIC barrier
      CrossCoreWaitFlag<0>(11)
      CrossCoreSetFlag<2, PIPE_FIX>(13)         // broadcast: release both AIV

AIV:  ... cast ...
      CrossCoreSetFlag<2, PIPE_MTE3>(12)        // gather contribution
      CrossCoreWaitFlag<2>(13)                  // wait AIC
```

Mode 2 is BOTH directions. The set pipe differs because of the per-core-type pipe tables in section 14.4: AIC sets on `PIPE_FIX` (its last cube/fixpipe stage), AIV sets on `PIPE_MTE3` (its last store stage). `PIPE_FIX` is not available to the AIV at all.

### 23.5 Mode 0 and mode 1 in a pure-Vector kernel — VERBATIM

```cpp
    __aicore__ inline void ProcessMode0()
    {
        ...
        // UB 到 GM 搬运启用原子累加：搬运至 atomicResult 的数据与原值累加后覆盖原值
        AscendC::SetAtomicAdd<float>();
        // DataCopy属于PIPE_MTE3流水操作
        AscendC::DataCopy(atomicResultGm, xLocal, this->blockLength);
        // 当本AIV完成前置PIPE_MTE3(DataCopy)流水操作后，通知其他AIV核，本AIV已经完成
        AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(0);
        // 阻塞本AIV继续往下执行指令，直到其他AIV全部都完成PIPE_MTE3流水操作，才解除阻塞往下执行。
        CrossCoreWaitFlagForArch<0, PIPE_MTE2>(0);
        // 关闭原子累加
        AscendC::DisableDmaAtomic();
        ...
    }
    __aicore__ inline void ProcessMode1()
    {
        // 16个aiv，GetBlockIdx取值为0-15。
        if ((AscendC::GetBlockIdx() == 2) || (AscendC::GetBlockIdx() == 3)) {
            ...
            AscendC::SetAtomicAdd<float>();
            AscendC::DataCopy(atomicResultGm, xLocal, this->blockLength);
            AscendC::CrossCoreSetFlag<1, PIPE_MTE3>(0);
            CrossCoreWaitFlagForArch<1, PIPE_MTE2>(0);
            AscendC::DisableDmaAtomic();
            ...
```

Note mode 1 is demonstrated on `GetBlockIdx() == 2 || == 3`, i.e. the two AIVs of AI Core 1 — mode 1 is strictly the two subblocks of ONE AI Core.

### 23.6 ★ 注意事项 — the pure-Vector trap and the kernel qualifier — VERBATIM

> #### 3.3纯Vector计算场景
> (1)在使用`CrossCoreSetFlag`与`CrossCoreWaitFlag`核间同步接口时，**即使是纯Vector计算场景，核函数也不能使用`__vector__`修饰符。**
> 本样例核函数采用`__mix__(1, 2)`修饰，但在纯Vector场景下，由于仅执行矢量计算，必须通过`ASCEND_IS_AIV`宏确保程序仅在AIV核上运行，否则会导致程序卡死。
>
> ```cpp
> AscendC::InitSocState();
> KernelCrossCoreSetFlag op;
> if ASCEND_IS_AIV {
>     op.Init(x, z, dataLength);
>     op.Process();
> }
> AscendC::PipeBarrier<PIPE_ALL>();
> ```

> #### 3.1 Cube与Vector融合计算场景
> (1) Cube与Vector融合计算场景中，融合算子（配置`__mix__(1, 2)`）中需要通过`ASCEND_IS_AIV`/`ASCEND_IS_AIC`进行AIV和AIC核代码的隔离。
>
> ```cpp
> AscendC::InitSocState();
> KernelMmad op;
> if ASCEND_IS_AIC {
>     op.InitAIC(A, B, c);
>     op.ProcessAIC();
> }
> if ASCEND_IS_AIV {
>     op.InitAIV(a, b, A, B, c);
>     op.ProcessAIV();
> }
> AscendC::PipeBarrier<PIPE_ALL>();
> ```

> (2) `GetBlockIdx`的返回值（获取当前核的index）在AIC和AIV的取值范围不同，其取值与算子设置的逻辑核数和一个AI Core中的AIC与AIV的比例有关。本样例中设置`NUM_BLOCKS=8`、AIC与AIV的比例为1:2，因此`GetBlockIdx`的返回值在AIC和AIV的取值范围分别为0-7和0-15。
>
> (3)样例采用静态Tensor编程范式需要手动插入核内同步。另外，静态Tensor编程方式中需要开发者手动调用`InitSocState()`接口初始化全局状态寄存器。
>
> (2)模式4仅支持Ascend 950PR/Ascend 950DT（对应NPU架构`dav-3510`），且Kernel必须配置为`__mix__(1, 2)`。

> 本样例针对融合算子（配置`__mix__(1, 2)`）场景展开，每个AI Core内部有1个AIC和2个AIV。配置逻辑核数`numBlocks = 8`，对应8个AIC与16个AIV。

## Relevance

Huawei's own CANN-9.2.0 sample for A3 does exactly the thing our measurement called impossible: it uses mode 2 for BOTH cube->vector and vector->cube, on the same kernel, in the same iteration, with mode 2 on both sides. The only differences between the two directions are the flagId (13 vs 12) and the set pipe (PIPE_FIX for AIC, PIPE_MTE3 for AIV). This is the pattern to copy.

## 24. Mode 2 + `PIPE_FIX` in a shipped CANN operator (`cann/ops-transformer`) — the real-world cube->vector idiom

URL: https://gitcode.com/cann/ops-transformer/blob/master/experimental/posembedding/rope_matrix/README.md
Raw: https://raw.gitcode.com/cann/ops-transformer/raw/master/experimental/posembedding/rope_matrix/README.md
Author: Huawei / CANN `ops-transformer` contributors (operator library built on CANN)
Date: fetched 2026-10-20

### 24.1 VERBATIM

> By profiling, we find memory bound for both C and V on 910B. Thus, no need CV pipeline, we just simplify run V after C.
> Using ```AscendC::CrossCoreSetFlag``` and ```AscendC::CrossCoreWaitFlag``` to control pipeline.

> Then use ```ASCEND_IS_AIC, ASCEND_IS_AIV``` to isolation cube/vector process and ```CrossCoreSetFlag, CrossCoreWaitFlag``` to control communication between cube and vector.
>
> ```bash
>     KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // dry kernel type: KERNEL_TYPE_MIX_xxx
>     TPipe tpipe;
>     TCubeTiling tilingLocal;
>     RopeMatrix::CopyTiling(&tilingLocal, tiling);
>     if ASCEND_IS_AIC {
>         ...
>         AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8); // set flag after matmul
>     }
>     if ASCEND_IS_AIV {
>         AscendC::CrossCoreWaitFlag(0x8); // wait matmul outputs
>         ...
>     }
> ```

Three things to note in this shipped operator:
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` — the `__mix__(1,2)` equivalent required by mode 2 per 表3 (section 9.8).
- `CrossCoreSetFlag<0x2, PIPE_FIX>(0x8)` — mode 2, set on `PIPE_FIX` from the AIC immediately after the matmul/fixpipe. flagId `0x8` = 8, chosen above the Matmul `[0, 2N-1]` band.
- `CrossCoreWaitFlag(0x8)` on the AIV with BOTH template args omitted — legal because on 2201 they are inert (section 10.3). This is the idiomatic A3 form.

## Relevance

Independent confirmation from a shipped operator (not a doc, not a sample) that `<2, PIPE_FIX>` set on AIC + bare `CrossCoreWaitFlag(flagId)` on AIV is the production cube->vector pattern on A2/A3, and that omitting the wait-side template args is intentional, not sloppy.

## 25. ★★★ VERSION CONTRADICTION: hiascend.com official CANN 9.0.0 doc says flagId range on A3 is 0-10, NOT 0-15

URL: https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0273.html
(the `/document/detail/...` equivalent that is WAF-blocked to fetchers: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0273.html)
Companion WaitFlag page: https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0274.html
Author: Huawei Technologies Co., Ltd. — official 昇腾社区 CANN API reference, 社区版 9.0.0
Date: fetched 2026-10-20. NOTE: this is an OLDER doc generation than asc.gitcode.com (which tracks asc-devkit master / CANN 9.2.0).

### 25.1 产品支持情况 table — VERBATIM (this generation uses the "Atlas 350 加速卡" name, not "Ascend 950PR&950DT")

| 产品 | 是否支持 |
| --- | --- |
| Atlas 350 加速卡 | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品 AI Core | x |
| Atlas 推理系列产品 Vector Core | x |
| Atlas 训练系列产品 | x |

### 25.2 功能说明 — note "面向分离模式的核间同步控制接口" — VERBATIM

> 面向分离模式的核间同步控制接口。
>
> 该接口和CrossCoreWaitFlag接口配合使用。使用时需传入核间同步的标记ID(flagId)，每个ID对应一个用于控制同步的计数器。
>
> 同步控制分为以下几种模式，如图1所示：
>
> 模式0：AI Core核间的同步控制。对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行；对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行。
>
> 模式1：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了CrossCoreSetFlag，CrossCoreWaitFlag后续的指令才会执行。
>
> 模式2：AI Core内部，AIC与AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后， 两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。
>
> 模式4：AI Core内部，AIC与AIV之间的同步控制。AIV0与AIV1可单独触发AIC等待。比如，在AIC核执行CrossCoreSetFlag之后， AIV0上CrossCoreWaitFlag后续的指令才会继续执行；AIV0执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。
>
> 其中，模式4仅支持Atlas 350 加速卡。

Same closed set {0, 1, 2, 4}. Again NO mode 3, in a doc generation two releases older.

### 25.3 表1 模板参数说明 — VERBATIM (note the PIPE_S statement)

> | modeId | 核间同步的模式，取值如下：<br>模式0：AI Core核间的同步控制。<br>模式1：AI Core内部，Vector核（AIV）之间的同步控制。<br>模式2：AI Core内部，Cube核（AIC）与Vector核（AIV）之间的同步控制。<br>模式4：AI Core内部，AIC与AIV之间的同步控制。AIV0与AIV1可单独触发AIC等待。 |
> | pipe | 设置这条指令所在的流水类型，流水类型可参考硬件流水类型。<br>**特别地，PIPE_S流水类型仅Atlas 350 加速卡支持。** |

### 25.4 ★★★ 表2 参数说明 — THE flagId RANGE DISAGREEMENT — VERBATIM

> | flagId | 输入 | 核间同步的标记。
> **Atlas A2 训练系列产品 / Atlas A2 推理系列产品，取值范围是0-10。**
> **Atlas A3 训练系列产品 / Atlas A3 推理系列产品，取值范围是0-10。**
> Atlas 350 加速卡，取值范围如下：
> - AIV0发起的flagId 0-10的CrossCoreSetFlag操作对应AIC CrossCoreWaitFlag中flagId 0-10的操作。
> - AIV1发起的flagId 0-10的CrossCoreSetFlag操作对应AIC CrossCoreWaitFlag中flagId 16-26的操作。
> - AIC发起的flagId 0-10的CrossCoreSetFlag操作对应AIV0 CrossCoreWaitFlag中flagId 0-10的操作。
> - AIC发起的flagId 16-26的CrossCoreSetFlag操作对应AIV1 CrossCoreWaitFlag中flagId 0-10的操作。
> |

★ THE CONTRADICTION, STATED PLAINLY:
- CANN 9.0.0 (hiascend, this page): A3 flagId range is **0-10**. 350/950 ranges are **0-10** and **16-26**.
- CANN 9.2.0 (asc-devkit master, section 9.7): A3 flagId range is **0-15**, each core has **16 flagIds**, out-of-range truncates to low 4 bits. 950 mode-4 ranges are **0-15** and **16-31**, AIC has **32 flagIds**, truncates to low 5 bits.
- The SOURCE CODE (section 12.4): `(flagId & 0xf)` — a 4-bit field, i.e. 0-15 physically, with 0-15 encodable and 16+ wrapping.

Resolution: the hardware range is 0-15 (the `& 0xf` mask is dispositive). `0-10` in the 9.0.0 doc is a CONSERVATIVE / RESERVED-AWARE bound, not the hardware bound — it carves out 11-15 for the library's internal users (SyncAll 11/12/13/14, SuperKernel-fused Matmul 15, SetNextTaskStart/WaitPreTaskEnd 11/14 — see section 11.2). The 9.2.0 docs replaced the opaque `0-10` with the honest `0-15` PLUS an explicit occupation table. Every third-party blog that says "flagId range 0-10" is reciting the 9.0.0 doc.

### 25.5 约束说明 — VERBATIM (shorter than 9.2.0; note the Matmul advice is stronger)

> 使用该同步接口时，需要按照如下规则设置Kernel类型：
> - 在纯Vector/Cube场景下，需设置Kernel类型为KERNEL_TYPE_MIX_AIV_1_0或KERNEL_TYPE_MIX_AIC_1_0。
> - 对于Vector和Cube混合场景，需根据实际情况灵活配置Kernel类型。
>
> 因为Matmul高阶API内部实现中使用了本接口进行核间同步控制，所以**不建议开发者同时使用该接口和Matmul高阶API**，否则会有flagID冲突的风险。
>
> **同一flagId的计数器最多设置15次。**
>
> 使用该接口模式0时，建议开启batchmode模式，使算子独占全部所需核资源，否则可能因满足以下条件导致死锁：
> - 多流并发场景（≥2条执行流）。
> - ≥2个算子并发执行。
> - 所有并发算子的核数总和超过物理核数。
> - ≥2个并发算子使用了核间同步功能。
>
> Kernel直调场景下通过__schedmode__(mode)限定符来设置batchmode模式；工程化算子开发场景下，通过TilingContext的SetScheduleMode接口来设置batchmode模式。

### 25.6 调用示例 — VERBATIM (the older `g_coreType` idiom, and mode 0/1/2 side by side)

```cpp
// 使用模式0的方式同步所有的AIV核
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag(0x8);
}

// 使用模式1的方式同步当前AICore内的所有AIV子核
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag(0x8);
}

// 注意：如果调用高阶API,无需开发者处理AIC和AIV的同步
// 以Matmul为例：AIC侧做完Matmul计算后通知AIV进行后处理
if (g_coreType == AscendC::AIC) {
    // Matmul处理
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
}

// AIV侧等待AIC Set消息, 进行Vector后处理
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreWaitFlag(0x8);
    // Vector后处理
}
```

父主题：核间同步

## Relevance

A hard, citable version contradiction. If a design doc says "flagId 0-10 on A3" it is quoting CANN 9.0.0; the CANN 9.2.0 doc and the `& 0xf` mask both say 0-15. For a hand-written engine on CANN 9.2.0 the usable range is 0-15 minus whatever the library claims; the 9.0.0 "0-10" is best read as the safe subset. Also: `PIPE_S` as a set pipe is 350/950-only in this generation too — never legal on A3.

## 26. Independent third-party spec of the mode semantics and the pipe-selection rule (PTOAS issue #985)

URL: https://github.com/hw-native-sys/PTOAS/issues/985
Title: `[Feature] DSL facade 跨核同步接口能力不完整`
Author: hw-native-sys/PTOAS contributors (a compiler/DSL project whose AscendC codegen passes these parameters straight through)
Date: fetched 2026-10-20. The issue itself cites `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0273.html`

### 26.1 The parameter spec — VERBATIM

> 当前客户项目 向用户暴露了 XXX_cross_core_set_flag(mode_id, pipe, flag_id) 和 XXX_cross_core_wait_flag(mode_id, pipe, flag_id) 两个跨核同步原语，三个参数的含义如下：
>
> | 参数 | 类型 | 说明 |
> | --- | --- | --- |
> | mode_id | int | 同步模式，取值 0/1/2/4，控制信号传播范围（跨 block 全核同步 / AIV 间同步 / CV 广播汇聚 / CV 逐子核 1:1） |
> | pipe | str | 流水线类型，如 PIPE_FIX / PIPE_MTE3 / PIPE_V / PIPE_M 等，指定信号发射或等待的流水线 |
> | flag_id | int 或 PrimExpr | 信号量编号，A5 平台范围 0-31（0-15 对应 AIV0，16-31 对应 AIV1） |
>
> 在 AscendC codegen 中，这三个参数直接透传为 CrossCoreSetFlag<mode_id, pipe>(flag_id) / CrossCoreWaitFlag<mode_id, pipe>(flag_id)，功能完整。

### 26.2 ★ The four modes with their typical scenarios — VERBATIM

> mode_id 的四种模式
>
> | mode_id | 含义 | 典型场景 |
> | --- | --- | --- |
> | 0 | 跨 block 同类型核同步（所有 AIC↔AIC 或所有 AIV↔AIV） | 多个 block 分担计算，需要所有同类型核到达屏障 |
> | 1 | block 内 AIV 子核间同步（AIV0 ↔ AIV1） | 两个 AIV 子核协作处理数据，需要互相通知 |
> | 2 | block 内 AIC↔AIV 同步，广播+汇聚（一次 set 到达两个 AIV，一次 wait 等到两个 AIV 都完成） | AIC 算完 GEMM 一次性通知两个 AIV 来搬数据 |
> | 4 | block 内 AIC↔AIV 同步，1:1 逐子核（AIV0 和 AIV1 可单独触发） | AIC 分别通知 AIV0 和 AIV1，用不同 flag_id 区分目标子核 |

Note the mode-2 gloss: 「广播+汇聚（一次 set 到达两个 AIV，一次 wait 等到两个 AIV 都完成）」 — broadcast AND gather, one mode, both directions. Independent of Huawei's own wording in sections 9.3/14.6, same conclusion.

### 26.3 ★ The pipe-selection rule, stated as an engineering rule — VERBATIM

> pipe 指定信号在哪条流水线上发射（set 端）或阻塞（wait 端）。pipe 的选择必须与实际数据流经过的流水线匹配，**选错会导致数据竞争。**
>
> pipe 的合法取值取决于具体场景，不应被固定为某个值。不同 mode_id 和不同操作侧（Cube/Vector）需要的 pipe 不同。例如：
>
> - GEMM 场景中，Cube 侧 L0C→UB 搬运走 PIPE_FIX，所以 Cube 侧 set/wait 用 PIPE_FIX
> - GEMM 场景中，Vector 侧 UB→GM 搬运走 PIPE_MTE3，所以 Vector 侧 set/wait 用 PIPE_MTE3
> - AscendC 官方文档中，mode 0 的 set 示例使用 PIPE_MTE3（不是 PIPE_FIX）
> - 其他场景中，可能还需要 PIPE_V、PIPE_M、PIPE_MTE2 等
>
> 如果 wait 端 stall 了错误的流水线，需要同步的流水线不会被阻塞，导致读到未就绪数据。

> flag_id 参数
>
> A5 平台上 flag_id 地址空间为 0-31：0-15 对应 AIV0，16-31 对应 AIV1（+16 偏移）。mode 4 场景中，AIC 需要用不同的 flag_id 分别通知两个 AIV 子核。

### 26.4 The facade gap (confirms mode 2 / mode 4 are the two DISTINCT hardware instruction families) — VERBATIM

> 问题 1：DSL facade 不支持 mode 0 和 mode 1
>
> 当前 PTO DSL facade 只提供了两组跨核同步函数：
>
> - set_cross_flag / wait_cross_flag（对应 mode 2 语义）
> - set_intra_flag / wait_intra_flag（对应 mode 4 语义）

The DSL names `cross` vs `intra` line up exactly with the two hardware instruction families found in section 12.9: `ffts_cross_core_sync`/`wait_flag_dev` (modes 0/1/2) vs `set_intra_block`/`wait_intra_block` (mode 4). Independent structural confirmation.

## Relevance

Third-party, non-Huawei confirmation that the mode space is {0,1,2,4} and that mode 2 is a broadcast+gather bidirectional mode. The quoted pipe rule — "pipe 的选择必须与实际数据流经过的流水线匹配，选错会导致数据竞争" — is the operational principle behind the AIC=PIPE_FIX / AIV=PIPE_MTE3 asymmetry; the set pipe must be the pipe that actually carried the data you are publishing.

## 27. Hardware pipe-type semantics (what PIPE_FIX / PIPE_MTE3 / PIPE_S actually gate) — official CANN 8.0.RC2.2 「同步控制简介」

URL: https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC22/apiref/opdevgapi/atlasascendc_api_07_0262.html
Author: Huawei Technologies Co., Ltd. — official 昇腾社区 CANN API reference, 商用版 8.0.RC2.2
Date: fetched 2026-10-20

### 27.1 表1 指令流水类型和相关说明 — VERBATIM

| 流水类型 | 含义 |
| --- | --- |
| PIPE_S | 标量流水线，使用Tensor GetValue函数时为此流水 |
| PIPE_V | 矢量计算流水及L0C->UB数据搬运流水 |
| PIPE_M | 矩阵计算流水 |
| PIPE_MTE1 | L1->L0A、L1->L0B数据搬运流水 |
| PIPE_MTE2 | GM->L1、GM->L0A、GM->L0B、GM->UB数据搬运流水 |
| PIPE_MTE3 | UB->GM、UB->L1数据搬运流水 |
| PIPE_FIX | L0C->GM、L0C->L1数据搬运流水，当前版本暂不支持该流水类型。 |

(The trailing 「当前版本暂不支持该流水类型」 on PIPE_FIX is a CANN-8.0.RC2.2-era caveat about PIPE_FIX in the *intra-core* SetFlag/WaitFlag table; PIPE_FIX IS supported as a CrossCoreSetFlag set pipe on 2201 per sections 9.6 / 14.4 / 16.4 / 17.1.)

### 27.2 Why the set pipe matters — VERBATIM

> 同步控制分类
>
> 对上述并行流水的同步控制分为两种：
>
> - 多流水同步：通过SetFlag/WaitFlag进行不同流水线间的同步控制。
>     - SetFlag：当前序指令的所有读写操作都完成之后，当前指令开始执行，并将硬件中的对应标志位设置为1。
>     - WaitFlag：当执行到该指令时，如果发现对应标志位为0，该队列的后续指令将一直被阻塞；如果发现对应标志位为1，则将对应标志位设置为0，同时后续指令开始执行。
> - 单流水同步：通过PipeBarrier完成同一流水线内的同步控制，用于在同一流水线内部约束执行顺序。其作用是，保证前序指令中所有数据的读写工作全部完成，后序指令才能执行。

> Cube计算单元
>
> Cube侧所有流水同步都由Ascend C框架完成，不需要算子开发者插入同步。

### 27.3 Pipe -> core-type mapping, derived

Cross-referencing 20.1 against the per-core-type pipe tables of section 14.4 for arch 2201:

| pipe | 数据通路 | legal as SET pipe on 2201 AIC | legal as SET pipe on 2201 AIV |
| --- | --- | --- | --- |
| PIPE_S | 标量流水线 | NO | NO |
| PIPE_V | 矢量计算流水及 L0C->UB 搬运 | NO | YES |
| PIPE_M | 矩阵计算流水 | YES | NO |
| PIPE_MTE1 | L1->L0A、L1->L0B | YES | NO |
| PIPE_MTE2 | GM->L1、GM->L0A/L0B、GM->UB | YES | YES |
| PIPE_MTE3 | UB->GM、UB->L1 | YES (2201 only; dropped on 3510) | YES |
| PIPE_FIX | L0C->GM、L0C->L1 | YES | NO |

So the AIC's natural "I have published my matmul result" set pipe is `PIPE_FIX` (its L0C->GM/L1 drain), and the AIV's natural "I have published my vector result" set pipe is `PIPE_MTE3` (its UB->GM/L1 drain). `PIPE_S` is never a legal CrossCoreSetFlag pipe on 2201 — which is exactly why `CrossCoreWaitFlag`'s default of `PIPE_S` is harmless (it is inert on 2201, section 10.3) but `CrossCoreSetFlag` has NO default and must be given an explicit, core-appropriate pipe.

## Relevance

Settles "PIPE_FIX vs PIPE_MTE3 vs PIPE_S" causally rather than by table lookup: the set pipe must name the pipe that physically moved the data being published. AIC publishes out of L0C -> PIPE_FIX. AIV publishes out of UB -> PIPE_MTE3. PIPE_S moves nothing and is rejected on the set side on 2201 while being the inert default on the wait side.

## 28. ★★ INDEPENDENT CORROBORATION of the 2201 lowering and the mode-2 CREDIT semantics (compiler-builtin level)

URL: https://github.com/hw-native-sys/pto-isa/issues/308 — `[Bug][CPU_SIM] Missing FFTS mode-2 set/wait implementation breaks A2/A3 simulation`
URL: https://github.com/hw-native-sys/PTOAS/issues/1513 — `[Bug] Explicit sync.set/sync.wait emit hardware-only intrinsics and fail A2/A3 CPU simulator compilation`
URL: https://github.com/hw-native-sys/simpler/issues/2188 — `[Bug] Explicit FFTS synchronization fails to compile on a2a3sim`
Author: hw-native-sys PTO / PTOAS / simpler contributors (compiler + simulator teams, not Huawei docs)
Date: fetched 2026-10-20. Stated environment: `CANN Version 9.0.0`, `Host Platform Linux (aarch64)`, `Ascend A2/A3 (--pto-arch=a3, a2a3sim)`

### 28.1 ★ THE COMPILER BUILTIN NAMES AND THE EXACT GENERATED A2/A3 SEQUENCE — VERBATIM

From PTOAS #1513, "The relevant generated C++ is":

```cpp
set_ffts_base_addr(v2);
// Cube section:
const int64_t v3 = 0;
uint16_t v4 = getFFTSMsg(FFTS_MODE_VAL, v3);
__builtin_cce_ffts_cross_core_sync(PIPE_FIX, v4);
// Vector section:
__builtin_cce_wait_flag_dev(0);
```

From pto-isa #308, the minimal compile probe — VERBATIM:

```cpp
#include <cstdint>
#include <pto/pto-inst.hpp>
using namespace pto;

void compile_probe() {
    uint16_t message = getFFTSMsg(FFTS_MODE_VAL, 0);
    __builtin_cce_ffts_cross_core_sync(PIPE_FIX, message);
    __builtin_cce_wait_flag_dev(0);
}
```

compiled with — VERBATIM:

```bash
g++-15 -std=c++23 -D__CPU_SIM -D__DAV_CUBE__ \
-I /path/to/pto-isa/include -fsyntax-only repro_ffts_cpu.cpp
```

Four symbols, VERBATIM:

> This fails with missing definitions for:
> - `FFTS_MODE_VAL`
> - `getFFTSMsg`
> - `__builtin_cce_ffts_cross_core_sync`
> - `__builtin_cce_wait_flag_dev`

This is the independent confirmation that on A2/A3:
- the SET lowers to `__builtin_cce_ffts_cross_core_sync(pipe, uint16_t message)` — pipe plus a single packed 16-bit message word, exactly as `GetffstMsg` builds it (section 12.4).
- the WAIT lowers to `__builtin_cce_wait_flag_dev(flagId)` — ONE argument, no pipe, no mode. (`PIPE_MTE2` was requested on the wait side by the DSL and is simply dropped: `pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, ...)` becomes `__builtin_cce_wait_flag_dev(0)`.)
- `set_ffts_base_addr` is part of the emitted prologue.

> Source evidence
>
> The NPU A2/A3 **TSync.hpp** defines `FFTS_MODE_VAL` and `getFFTSMsg`; hardware compilation supports the corresponding synchronization intrinsics.

> PTOAS v0.60, **PTOToEmitC.cpp:1618-1706**: `createFFTSMsg` emits `getFFTSMsg` / `FFTS_MODE_VAL`; the set/wait helpers directly choose `__builtin_cce_ffts_cross_core_sync` / `__builtin_cce_wait_flag_dev`.
> PTOAS v0.60, `PTOSyncSetToEmitC` / `PTOSyncWaitToEmitC`: both static and dynamic event paths use those helpers.
> PTO ISA's CI-pinned A2/A3 TSync.hpp defines the FFTS helper, while the corresponding CPU TSync.hpp does not.
> PTOAS v0.60: `3bc3d673564051d1f655e7224dbeb5d6177de4f7`.

### 28.2 ★★★ THE MODE-2 SEMANTICS AS A CREDIT PROTOCOL, with the failure mode spelled out — VERBATIM

From pto-isa #308, "Requested behavior":

> The immediate downstream requirement is **A2/A3 FFTS mode 2, base count 1, event IDs 0–15**, with actual CPU synchronization semantics:
>
> - An AIC signal makes one notification available to **each** of its two paired AIV lanes.
> - An AIC wait completes only after consuming one notification **from each** AIV lane. **Two signals from one lane must not substitute for the other lane.**
> - Notifications **accumulate** and are consumed by waits; repeated notifications **must not collapse into a boolean**.
> - Signaling/waiting preserves the ordering and visibility of producer data before consumer reads.
> - State works across independently loaded AIC/AIV shared libraries and follows the runtime's device/group and task/run lifetime boundaries, including reset.

From simpler #2188, "Expected Behavior":

> The A2/A3 simulator should compile and execute explicit FFTS mode-2 events with device-compatible synchronization semantics:
>
> - An AIC notification is visible to **both** AIV lanes.
> - An AIC wait completes only after **both** AIV lanes notify it.
> - **Notifications are consuming credits**, so queued and repeated events do not collapse into one boolean flag.
> - Event state is isolated by simulated device and cluster and reset between runs.

「event IDs 0–15」 — a third independent statement of 16 flagIds on A2/A3, from the compiler/simulator side rather than the docs.

「Notifications are consuming credits」 + 「repeated notifications must not collapse into a boolean」 — this is the counter (not a flag bit) semantics, matching the 「每个flagId都对应一个计数器 ... 每一个计数器计数范围为0-15」 of section 9.8 and the 「4位计数器」 of section 14.3.

「Two signals from one lane must not substitute for the other lane」 — the mode-2 gather is per-lane, not a count-to-2. This is the sharpest correctness statement found anywhere about mode 2, and it is NOT in the official docs.

### 28.3 The DSL-level call that produced that A3 sequence — VERBATIM (PyPTO reproducer)

```python
        pl.system.set_ffts(workspace)
        a = pl.load(x, [block * 32, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
        acc = pl.tile.matmul(a, a)
        pl.store(acc, [0, 0], scratch)
        pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
        for sub in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            pl.system.set_ffts(workspace)
            pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
            chunk = pl.load(scratch, [sub * 16, 0], [16, 32], target_memory=pl.MemorySpace.Vec)
```

Note: `ffts_mode=2`, `pipe=FIX` on the AIC set; `pipe=MTE2` on the AIV wait (dropped on A3); `flag_id = 0`. And from simpler #2188: 「Calls `pl.system.set_ffts(workspace)` on AIC **and both AIV lanes**」 — the FFTS base address must be programmed on EVERY participating core, not just one.

> Native A2/A3 execution passes, confirming that this is specific to the simulator compatibility path.

### 28.4 Message-word arithmetic cross-check

A literal message value `0x121` is reported in this ecosystem for `ffts_cross_core_sync(PIPE_FIX, 0x121)`. Decoding with the layout of section 12.4:

```
0x121 = 0b0001_0010_0001
  bit 0      = 1          -> the constant 0x1
  bits [5:4] = 0b10 = 2   -> mode 2
  bits [11:8]= 0b0001 = 1 -> flagId 1

GetffstMsg(2, 1) = 0x1 + ((2 & 0x3) << 4) + ((1 & 0xf) << 8)
                 = 1 + 32 + 256 = 289 = 0x121   ✓
```

The decode is exact. This independently validates the bit layout derived from `GetffstMsg` in section 12.4 against a hand-written literal from a different codebase.

## Relevance

Confirms from the compiler/simulator side, with no reference to Huawei prose, that on A2/A3 the wait builtin takes only a flagId (no mode, no pipe), that event IDs run 0-15, and that the flag is a consuming CREDIT COUNTER not a boolean. Its unique contribution: the mode-2 gather is PER-LANE — "two signals from one lane must not substitute for the other lane" — a correctness constraint absent from every official page.

## 29. Mode-value survey across EVERY arch implementation in asc-devkit master: only 0x0 and 0x02 ever appear; mode 3 appears nowhere

URL: https://raw.gitcode.com/cann/asc-devkit/raw/master/impl/basic_api/<arch>/kernel_operator_sync_impl.h
for `<arch>` in `dav_c100`, `dav_m200`, `dav_c220`, `dav_l300`, `dav_3510`, `dav_9201`, `dav_m510`, `dav_5162` (`dav_9202` returns 404 on master; `dav_m300`/`dav_m310` route to `kernel_operator_common_impl.h` instead)
Author: Huawei Technologies Co., Ltd. (`cann/asc-devkit`)
Date: fetched 2026-10-20

### 29.1 Method and result

Every `GetffstMsg(mode, flagId)` call site across all fetched arch headers was enumerated:

```
$ grep -ohE 'GetffstMsg\(0x[0-9a-fA-F]+' arch_*.h | sort | uniq -c | sort -rn
     68 GetffstMsg(0x02
     28 GetffstMsg(0x0
```

Exactly two mode values appear anywhere in Huawei's own cross-core sync implementations, across every supported NPU architecture: `0x0` (mode 0, same-core-type barrier) and `0x02` (mode 2, AIC<->all-AIV). Mode 1 is never used internally (it is a user-facing mode only). **Mode 3 does not appear in a single call site, a single constant, a single `if constexpr`, or a single comment.**

### 29.2 The 2-bit mask is uniform across all five archs that pack the message — VERBATIM

```
impl/basic_api/dav_c220/kernel_operator_sync_impl.h:126:    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
impl/basic_api/dav_3510/kernel_operator_sync_impl.h:109:    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
impl/basic_api/dav_9201/kernel_operator_sync_impl.h:114:    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
impl/basic_api/dav_m510/kernel_operator_sync_impl.h:102:    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
impl/basic_api/dav_5162/kernel_operator_sync_impl.h:102:    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
```

Identical expression, identical masks, five different architectures. `mode & 0x3` is a deliberate, stable, architecture-wide 2-bit field — not a typo and not a 2201 quirk.

### 29.3 What this means for "mode 3" and "mode 4" on dav-2201

- `CrossCoreSetFlag<3, pipe>(flagId)` on 2201 emits a message with bits[5:4] = `0b11`. The encoding is reachable, distinct from modes 0/1/2, and completely unspecified by Huawei: no doc names it, no library call site uses it, no `if constexpr` branches on it.
- `CrossCoreSetFlag<4, pipe>(flagId)` on 2201 emits bits[5:4] = `4 & 0x3 = 0b00` — i.e. it SILENTLY BECOMES MODE 0. There is no `#error`, no `static_assert`, no trap. A kernel ported from 950 to A3 with `<4, ...>` left in will compile, run, and barrier the wrong set of cores. (On 3510 mode 4 is caught by `if constexpr (modeId == 4)` and routed to `set_intra_block`/`wait_intra_block`; on 2201 that branch does not exist — see section 12.9.)
- Any value `modeId >= 5` likewise aliases: `5->1`, `6->2`, `7->3`, `8->0`, ...

## Relevance

Settles the mode-space question exhaustively rather than by absence-of-evidence: mode 3 is an encodable bit pattern that Huawei's documentation never defines and Huawei's code never emits, on any arch. The actionable trap is the other direction — `<4>` on A3 silently degrades to `<0>` through `mode & 0x3`, which is a real portability footgun with no diagnostic.

## 30. ★★★ THE ARCH-2201-SPECIFIC PAGE: 「NPU架构版本2201」 hardware spec — contains a 「避免连续设置」 constraint found nowhere else

URL: https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html
Raw markdown: https://raw.gitcode.com/cann/asc-devkit/raw/master/docs/zh/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.md
Companion 3510 page: https://raw.gitcode.com/cann/asc-devkit/raw/master/docs/zh/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_3510.md
Author: Huawei Technologies Co., Ltd. — Ascend C 编程指南 / 硬件实现 / 体系结构规格, arch 2201
Date: fetched 2026-10-20

### 30.1 核间同步 section opening and mode list — VERBATIM (A3's own arch page; 模式0/1/2 only, no mode 4)

```markdown
-   核间同步

    当不同核之间操作同一块全局内存时，可能存在读后写、写后读以及写后写等数据依赖问题，需要进行核间同步控制。

    核间同步控制分为以下几种模式，如下图所示：

    -   模式0：AI Core核间的同步控制。对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行；对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行。
    -   模式1：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了CrossCoreSetFlag，CrossCoreWaitFlag后续的指令才会执行。
    -   模式2：AI Core内部，AIC与AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后，两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。
```

(Figure: `../../../../figures/aicore.png`. The 3510 page lists the same three plus 「模式4：AI Core内部，AIC与AIV之间的同步控制（1:1）。AIV0与AIV1可单独触发AIC等待。」 and annotates mode 2 as 「（1:2）」.)

### 30.2 The canonical worked explanation of the A3 cube->vector handshake — VERBATIM

> 下面以模式2为例说明AIC与同一AI Core内AIV之间的同步过程：AIC将L0C的计算结果搬运到GM后，调用`CrossCoreSetFlag<0x2, PIPE_FIX>(ID1)`；两个AIV通过`CrossCoreWaitFlag<0x2>(ID1)`确保数据从L0C成功搬运到GM后，再将GM的数据搬运到UB。其中，`0x2`表示模式2，`PIPE_FIX`表示流水类型，`ID1`为同步标记ID。**CrossCoreWaitFlag只显式传入modeId，未传入pipe模板参数。** CrossCoreSetFlag之前PIPE_FIX流水的指令完成后，CrossCoreWaitFlag后续**全部流水**的指令才能执行，具体原因请参考"默认值"。

> [CrossCoreSetFlag]和[CrossCoreWaitFlag]接口配合使用。使用时需传入核间同步的标记ID(flagId)，即上图中的ID1，每个ID对应一个**初始值为0**的计数器。执行CrossCoreSetFlag后ID对应的计数器**增加1**；执行CrossCoreWaitFlag时如果对应的计数器数值为0则**阻塞不执行**；如果对应的计数器**大于0，则计数器减一**，同时后续指令开始执行。

### 30.3 ★★★ THE FOUR NUMBERED CONSTRAINTS, ARCH 2201 — VERBATIM

```markdown
    需要注意以下几点：

    -   **成对使用与一致性要求**

        CrossCoreSetFlag和CrossCoreWaitFlag必须成对使用，否则可能导致算子超时问题。两个接口的参数一致性要求如下：

        - modeId：本架构版本的硬件指令不使用CrossCoreWaitFlag的modeId，但建议显式配置与CrossCoreSetFlag相同的modeId，以便确认两个接口的同步模式匹配。
        - pipe：CrossCoreWaitFlag的pipe不生效，无需与CrossCoreSetFlag的pipe保持一致。
        - flagId：CrossCoreSetFlag与CrossCoreWaitFlag必须使用相同的flagId。

    -   **避免连续设置**

        不允许连续设置同一个flagId，以防止计数器状态混乱。

    -   **与其它接口的flagId冲突**

        部分接口内部实现中使用了CrossCoreSetFlag和CrossCoreWaitFlag，会占用一部分flagId，开发者同时使用这些接口与CrossCoreSetFlag/CrossCoreWaitFlag时，需注意避免flagId使用冲突，各接口flagId占用的完整说明（含各NPU架构版本的差异）参考核间同步flagId占用说明。

    -   **计数器限制**

        同一flagId的计数器最多可以设置15次。

    -   **默认值**<a id="cross_core_wait_flag_default_values"></a>

        - modeId：默认值为0。建议显式配置与CrossCoreSetFlag相同的modeId。
        - pipe：默认值为PIPE_S。CrossCoreWaitFlag接口提供pipe模板参数，但本架构版本的硬件指令不区分等待的流水，因此pipe取值不影响实际执行效果，CrossCoreWaitFlag会阻塞全部流水的后续指令。
```

★ 「**不允许连续设置同一个flagId，以防止计数器状态混乱。**」 — "It is NOT PERMITTED to set the same flagId consecutively, to prevent counter-state corruption." This appears on BOTH the 2201 and 3510 arch pages (3510 line 326-328, identical wording) but on NEITHER API reference page.

★ THIS IS IN TENSION WITH THE DEPTH-15 STORY. The API reference (section 9.8) says 「每一个计数器计数范围为0-15。如果调用CrossCoreSetFlag的次数超过15次，计数器的值超出该范围，则会异常报错，中断流程。」 — implying up to 15 outstanding sets are fine. The arch page says consecutive sets on the same flagId are flatly disallowed. Practical reading: the 15-deep counter exists and overflow traps, but Huawei does not support using it as a credit pipeline on a single flagId — one set must be matched by its wait before the next set on that flagId. If you want N-deep pipelining, use N DISTINCT flagIds.

★ Note also the 3510 page's differing default-value text, VERBATIM:
> - pipe：默认值为PIPE_S，此时阻塞PIPE_S流水的后续指令。需要阻塞其它流水时，必须显式配置对应的pipe。
and its flagId consistency rule:
> - flagId：模式0、模式1和模式2下，CrossCoreSetFlag与CrossCoreWaitFlag必须使用相同的flagId；模式4下，两个接口的flagId必须符合对应关系。
and its pipe rule:
> - pipe：CrossCoreSetFlag的pipe表示调用该接口前需要完成的流水，CrossCoreWaitFlag的pipe表示需要阻塞的流水，无需保持一致。

### 30.4 The analogous INTRA-core SetFlag/WaitFlag constraints on 2201, for contrast — VERBATIM

> -   建议通过AllocEventID或者FetchEventID接口获取EventID，以确保其合法性和有效性。
> -   EventID的数量有限，使用后应立即调用ReleaseEventID释放资源，避免EventID耗尽，影响系统正常运行。
> -   SetFlag和WaitFlag必须成对使用，且SetFlag和WaitFlag的参数必须完全一致（包括模板参数和事件ID）。如果不匹配，可能导致当前核的计算异常，或影响下一个核的算子执行，引发timeout问题。
>
>     例如，`SetFlag<HardEvent::S_MTE3>(1)`和`SetFlag<HardEvent::MTE3_MTE1>(1)`设置的不是同一个EventID，因为其模板参数不同。只有当模板参数和事件ID完全一致时，才表示同一个EventID。
>
> -   不允许连续设置同一个EventID，因为这可能导致事件状态混乱或未被正确处理。
> -   **不建议手动插入TEventID，不能手动插入6和7的TEventID，因为它们可能被系统预留或用于特殊用途。**

(That last bullet is about INTRA-core `TEventID`, a different and smaller ID space than the cross-core `flagId` — do not conflate them. TEventID 6 and 7 are reserved; flagId 11-15 are the cross-core reserved band.)

## Relevance

This is the single most arch-targeted authoritative page for dav-2201 and it carries a constraint absent from every API page: 「不允许连续设置同一个flagId」. Read together with the depth-15 counter, the engine design rule is: one outstanding set per flagId at a time; pipeline depth must come from distinct flagIds, not from the counter. It also restates, in arch-spec voice, that the 2201 wait instruction uses neither modeId nor pipe.

## 31. Per-mode counter mechanics and the EXACT pairing arithmetic (key_features mode 0 / 1 / 2 sections, arch 2201)

URL: https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.html
Raw markdown: https://raw.gitcode.com/cann/asc-devkit/raw/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.md
Author: Huawei Technologies Co., Ltd. — Ascend C SIG
Date: fetched 2026-10-20
Scope note from the page itself: 「以下针对模式0、1、2的描述，仅针对NPU架构版本2201有效」

### 31.1 模式0 — mechanism + reference code — VERBATIM

> ## 多AI Core中AIC或者AIV全核同步（模式0）
>
> 多AI Core中AIC或者AIV全核同步需要配套使用模式0的CrossCoreSetFlag和CrossCoreWaitFlag接口。该场景可细分为两类：
>
> - 2个AI Core中的2个AIC进行全核同步。
>
>     当2个AIC都执行完CrossCoreSetFlag时，每个AIC对应flagId的计数器增加1。若AIC该flagId的计数器为非0，被CrossCoreWaitFlag所阻塞的所有流水中的指令才会执行下去，该flagId的计数器减去1。
>
>     对于每个AIC，CrossCoreSetFlag和CrossCoreWaitFlag都必须配对使用。
>
> - 2个AI Core中的4个AIV进行全核同步。
>
>     当4个AIV都执行完CrossCoreSetFlag时，每个AIV核对应flagId的计数器增加1。若AIV该flagId的计数器为非0，被CrossCoreWaitFlag所阻塞的所有流水中的指令才会执行下去，该flagId的计数器减去1。
>
>     对于每个AIV，CrossCoreSetFlag和CrossCoreWaitFlag都必须配对使用。

```cpp
// 进行核间同步时，即使只有AIC参与同步也不能用__cube__修饰核函数（Kernel），具体原因请参考CrossCoreSetFlag的约束说明。
if (ASCEND_IS_AIC) {
    // 每个核都应该有类似如下的成对调用CrossCoreSetFlag和CrossCoreWaitFlag。
    // modeId必须配置为0，flagId要一致；CrossCoreWaitFlag的pipe使用默认值。
    AscendC::CrossCoreSetFlag<0, PIPE_M>(0);       // 待前置PIPE_M流水任务完成后，通知调度模块。
    AscendC::CrossCoreWaitFlag<0>(0);              // 直到所有AIC都完成前置PIPE_M流水，调度模块更新flagId=0的计数器后，解除所有流水后续指令的阻塞。
}
```

> AIC0中在执行CrossCoreWaitFlag后，此时AIC0 flagId=0的计数器为0，后续所有流水中的指令全部被阻塞，需要等到2个AIC核均执行完CrossCoreSetFlag。
>
> AIC1的CrossCoreSetFlag执行完后，此时调度模块感知到2个AIC均已执行完CrossCoreSetFlag，因此所有AIC各自的flagId=0的计数器值增加为1。AIC0和AIC1检测到各自对应的flagId=0的计数器变为1，都解除所有流水后续指令的阻塞，继续执行后续指令，并且将计数器值减去1。

Figure: `../../../../figures/inter_core_aic_all_sync.png`. Note the set pipe here is `PIPE_M` — a third legal AIC set pipe alongside PIPE_FIX.

### 31.2 ★ 模式1 — mechanism, reference code, and the CAUTION — VERBATIM

> ## 单个AI Core中AIV全核同步（模式1）
>
> 该场景需要配套使用模式1的CrossCoreSetFlag和CrossCoreWaitFlag接口。
>
> 1个AI Core中的2个AIV进行全核同步。当2个AIV全部都执行完CrossCoreSetFlag时，每个AIV对应flagId的计数器增加1。若AIV该flagId的计数器为非0，被CrossCoreWaitFlag所阻塞的所有流水中的指令才会执行下去，该flagId的计数器减去1。对于每个AIV，CrossCoreSetFlag和CrossCoreWaitFlag都必须配对使用。
>
> > [!CAUTION]注意
> > **该模式中，不同AI Core的AIV核之间不会互相影响同步。**

```cpp
// 进行核间同步时，即使只有AIV参与同步也不能用__vector__修饰核函数（Kernel），具体原因请参考CrossCoreSetFlag的约束说明。
if (ASCEND_IS_AIV) {
    if (AscendC:: GetBlockIdx() <= 1) {
        // 参与同步的2个AIV属于第0个AI Core。
        // 每个AIV都应该有类似如下的成对调用CrossCoreSetFlag和CrossCoreWaitFlag。
        // modeId必须配置为1，flagId要一致；CrossCoreWaitFlag的pipe使用默认值。
        AscendC::CrossCoreSetFlag<1, PIPE_MTE3>(0);       // 待前置PIPE_MTE3流水任务完成后，通知调度模块。
        AscendC::CrossCoreWaitFlag<1>(0);                  // 直到该AI Core中的所有AIV都完成前置PIPE_MTE3流水，调度模块更新flagId=0的计数器后，解除所有流水后续指令的阻塞。
    }
}
```

> AIV0-0中在执行CrossCoreWaitFlag后，此时AIV0-0 flagId=0的计数器为0，后续所有流水中的指令全部被阻塞，需要等到2个AIV均执行完CrossCoreSetFlag。
>
> AIV0-1的CrossCoreSetFlag执行完后，此时调度模块感知到2个AIV均已执行完CrossCoreSetFlag，因此将AIV0-0和AIV0-1各自的flagId=0的计数器值增加为1。AIV0-0和AIV0-1检测到各自对应的flagId=0的计数器变为1，都解除所有流水后续指令的阻塞，继续执行后续指令，并且将计数器值减去1。

Figure: `../../../../figures/single_core_aiv_all_sync.png`. Mode 1 is strictly per-AI-Core — the same flagId used in mode 1 on two different AI Cores does not interfere (this is the mechanism behind the 「对于不同的核，可以直接将同一flagId用于不同的核间同步模式」 allowance in section 9.8).

### 31.3 ★★ 模式2 — THE EXACT PAIRING ARITHMETIC (1:2 and 2:1) — VERBATIM

> ## 单个AI Core中AIC与AIV全核同步（模式2）
>
> 单个AI Core中AIC与AIV全核同步需要配套使用模式2的CrossCoreSetFlag和CrossCoreWaitFlag接口。该场景可细分为两类：
>
> - 1个AI Core中的AIC执行CrossCoreSetFlag，对应的2个AIV都执行CrossCoreWaitFlag。
>
>     当AIC执行完CrossCoreSetFlag时，2个AIV对应flagId的计数器增加1。若AIV该flagId的计数器为非0，则解除AIV所有流水后续指令的阻塞，CrossCoreWaitFlag后的指令继续发射，该flagId的计数器减去1。
>
>     **总计AIC 1次调用CrossCoreSetFlag，对应2个AIV各1次CrossCoreWaitFlag才算配对使用。**
>
> - 1个AI Core中的2个AIV都执行CrossCoreSetFlag，对应的AIC执行CrossCoreWaitFlag。
>
>     当2个AIV全都执行完CrossCoreSetFlag时，AIC对应flagId的计数器增加1。若AIC该flagId的计数器为非0，则解除AIC所有流水后续指令的阻塞，CrossCoreWaitFlag后的指令继续发射，该flagId的计数器减去1。
>
>     **总计2个AIV各1次CrossCoreSetFlag，对应AIC 1次调用CrossCoreWaitFlag才算配对使用。**

```cpp
// AIC和AIV均参与核间同步，需分别使用对应宏包裹代码段，具体原因请参考CrossCoreSetFlag的约束说明。
if (ASCEND_IS_AIV) {
    if (AscendC:: GetBlockIdx() <= 1) {
        // 参与同步的AIV属于第0个AI Core。
        // 每个AIV都应该有类似如下的CrossCoreSetFlag调用。
        // modeId必须配置为2，flagId要一致。
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(0);       // 待前置PIPE_MTE3流水任务完成后，通知调度模块。必须由所有AIV执行。
    }
}
if (ASCEND_IS_AIC) {
    if (AscendC:: GetBlockIdx() == 0) {
        // 参与同步的AIC属于第0个AI Core。
        // AIC需要有类似如下的CrossCoreWaitFlag调用。
        // CrossCoreWaitFlag的pipe使用默认值。
        AscendC::CrossCoreWaitFlag<2>(0);                 // 直到该AI Core中的所有AIV都完成前置PIPE_MTE3流水，调度模块更新flagId=0的计数器后，AIC解除所有流水后续指令的阻塞。
    }
}
```

> AIC0中在执行CrossCoreWaitFlag后，此时AIC0 flagId=0的计数器为0，后续所有流水中的指令全部被阻塞，需要等到2个AIV均执行完CrossCoreSetFlag。
>
> 当AIV0-1的CrossCoreSetFlag在Vector指令1执行完后，前置PIPE_V的指令全部完成，CrossCoreSetFlag执行完成，但是此时AIV0-0未执行CrossCoreSetFlag。因此AIC0 flagId=0的计数器还是为0，AIC的所有流水中的指令依然被阻塞。
>
> 当AIV0-0的Vector指令1、2全部执行完毕后，前置PIPE_V的指令全部完成，CrossCoreSetFlag生效。此时调度模块感知到2个AIV均已执行完CrossCoreSetFlag，因此将AIC0 flagId=0的计数器值增加为1。AIC0检测到对应的flagId=0的计数器变为1，则AIC0核解除所有流水后续指令的阻塞，继续执行后续指令Matrix指令2，并且将计数器值减去1。

Figure: `../../../../figures/single_core_aic_aiv_sync_aiv_setflag.png`.

★ Note the ASYMMETRIC CARDINALITY, which is the trap for anyone treating mode 2 as a plain semaphore:
- AIC->AIV (broadcast): 1 set -> 2 counter increments (one per AIV). Pair = 1 set : 2 waits.
- AIV->AIC (gather): 2 sets -> 1 counter increment (on the AIC). Pair = 2 sets : 1 wait.

So on arch 2201, the mode-2 AIV->AIC direction REQUIRES BOTH AIVs TO SET. A single AIV setting does not advance the AIC's counter at all — 「但是此时AIV0-0未执行CrossCoreSetFlag。因此AIC0 flagId=0的计数器还是为0」. This is exactly the per-lane gather the compiler team independently specified in section 21.2 (「Two signals from one lane must not substitute for the other lane」). If you need to release the AIC on one AIV alone, mode 2 CANNOT do it on 2201 — that is precisely what mode 4 is for, and mode 4 does not exist on 2201.

## Relevance

The mode-2 gather on 2201 is all-AIV-or-nothing: both subblocks of the AI Core must set before the AIC's counter moves. For a hand-written engine with `__mix__(1,2)` where only one AIV lane has work, mode 2 will deadlock the AIC unless the idle lane also issues its set. That is the most likely real cause of a "reverse mode 2 doesn't work" observation.

## 32. ★★★ CAN A FLAG CROSS AN AIC->AIV BOUNDARY AT blockDim > 1? NO — and here is the field report that hit it

### 32.1 The authoritative statements (official, already quoted above, collected here for this specific question)

URL: https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.html
URL: https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.html
URL: https://asc.gitcode.com/api/SIMD-API/c_api/sync/inter_core_sync/inter_core_sync_overview.html
Author: Huawei Technologies Co., Ltd. — Ascend C SIG
Date: fetched 2026-10-20

Every official description of modes 1, 2 and 4 is scoped to a SINGLE AI Core / a SINGLE group. VERBATIM, the scoping words:

> 模式1：**AI Core内部**，AIV核之间的同步控制。
> 模式2：**AI Core内部**，AIC与所有AIV之间的同步控制。
> 模式4：**AI Core内部**，AIC与单个AIV之间的同步控制。

Only mode 0 crosses AI Cores, and it crosses only between cores of the SAME TYPE. VERBATIM:

> 模式0：**AI Core核间**的同步控制。对于AIC场景，同步所有的AIC核 ...；对于AIV场景，同步所有的AIV核 ...

The C-API overview puts it in group terms, VERBATIM (表2, section 14.1):

> | asc_sync_inter_arrive/asc_sync_inter_wait | **组间同步**，即不同group之间所有block（所有AIC或所有AIV）之间的同步。 | 模式0 |
> | asc_sync_subblock_arrive/asc_sync_subblock_wait | **组内subblock间的同步**，即同一group内不同subblock（AIV）之间的同步。 | 模式1 |
> | asc_sync_block_arrive/asc_sync_block_wait | **同一group内**block（AIC）与所有subblock（AIV）之间的同步。 | 模式2 |
> | asc_sync_intra_arrive/asc_sync_intra_wait | **同一group内**block（AIC）与单个subblock（AIV）之间的同步。 | 模式4 |

And the mode-1 CAUTION, VERBATIM (section 24.2):

> > [!CAUTION]注意
> > 该模式中，**不同AI Core的AIV核之间不会互相影响同步。**

ANSWER: NO. At `blockDim > 1` there is no mode that lets AIC #i signal AIV #j of a DIFFERENT AI Core. The cross-AI-Core reach of the FFTS flag mechanism is: mode 0 only, same core type only. A cube->vector dependency that crosses AI Cores must be composed — e.g. mode 2 (AIC -> its own AIVs) plus mode 0 (all-AIV barrier), which is exactly what `SyncAll<false>` does in three stages (section 13.3: 「首先完成单个AI Core内，AIC与所有AIV同步（AIC等AIV）；接着完成多个AI Core间，所有AIC的同步；最后完成单个AI Core内，所有AIV与AIC同步（AIV等AIC）」).

### 32.2 The field report — a team that hit this and had to re-tile

URL: https://zhuanlan.zhihu.com/p/2000250579956540357
Title: 端到端打通transformer仓experimental路径首个开源MIX算子
Author: Zhihu 专栏 article by the CANN `ops-transformer` experimental-path contributors (the same work whose README is section 17)
Date: fetched via search index 2026-10-20. NOTE: zhuanlan.zhihu.com returns HTTP 403 to both `curl` and WebFetch, so this sentence is reproduced as returned by the search index; the companion gitcode README (section 17) is the fetchable primary artifact of the same work.

VERBATIM (Chinese preserved):

> 由于 CrossCoreSetFlag 仅支持核内同步（比如说20个Cube和40个Vector，Cube0仅能和Vector0和Vector1同步），所以需要将 matmul 的 tiling 修改为切 S 轴的方式

and

> 使用 ASCEND_IS_AIC / ASCEND_IS_AIV 区分 Cube 和 Vector 部分，再通过 CrossCoreSetFlag、CrossCoreWaitFlag 控制 C 和 V 之间的通信关系

The concrete consequence they report: with 20 AIC and 40 AIV, `Cube0` can only sync with `Vector0` and `Vector1`. Because the cube->vector flag cannot cross the AI Core boundary, the matmul TILING had to be changed to split the S axis so that each AIC's output is consumed only by its own two AIVs. This is a tiling constraint imposed by the sync topology, not a performance tweak.

## Relevance

Answers the blockDim>1 question definitively: a mode-2 flag is confined to one AI Core's {1 AIC, N AIV} group, so a hand-written engine must TILE so that each AIC's output is consumed by its own paired AIVs — or pay for a composed mode2+mode0+mode2 barrier. A real team changed their matmul tiling to the S axis for exactly this reason.

## 33. CONSOLIDATED LEDGER: where the sources agree, where they contradict each other, and where they contradict the four measured claims

This section is DERIVED (my cross-reference of the quoted material above), not a quotation. Every row points at the section holding the verbatim evidence.

### 33.1 Against the four measured claims

| Measured claim | Verdict | Evidence |
| --- | --- | --- |
| "16 flags per core on dav-2201" | **CONFIRMED, three independent ways** | Doc: 「对于模式0、1、2，每个AIC和每个AIV都各自有16个flagId，支持的取值范围为0-15」 (§16.8). Source: `(flagId & 0xf)` — a 4-bit field (§19.4). Compiler team: 「event IDs 0–15」 (§28.2). C-API: 「取值范围为[0, 15]，每个`flag_id`各自拥有独立的4位计数器」 (§21.3). |
| "`AscendC::SyncAll` consumes 11-13 of them" | **CONFIRMED for the Mix form, INCOMPLETE overall** | `SyncAll<false>` consumes exactly {11, 12, 13} — official table (§18.2) and source (§19.6). BUT `SyncAll<true>` (AIV-only, the DEFAULT template argument) consumes {14} and NOT 11-13. And `SetNextTaskStart`/`WaitPreTaskEnd` claim {11,14} / {11,12,13,14}, and SuperKernel-fused Matmul claims {15}. So the full reserved band is 11-15, not 11-13. (§18.2, §18.4, §19.6) |
| "sync mode 2 is cube->vector" | **CONTRADICTED — mode 2 is BIDIRECTIONAL** | Doc: 「在AIC核执行CrossCoreSetFlag之后，两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；**两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行**」 (§16.3). Library source, Huawei's own comments: `GetffstMsg(0x02, SYNC_AIC_AIV_FLAG) // SET: C--->V` and `GetffstMsg(0x02, SYNC_AIV_FLAG) // SET: V--->C` (§19.7). Huawei's CANN 9.2.0 A3 sample uses mode 2 in both directions in one kernel (§23.4). C-API spells out both scenarios (§21.6). Direction comes from WHICH CORE ISSUES the set, kept apart by flagId — not from the mode. |
| "mode 3 is the only reverse (vector->cube) mode that works in all set/wait patterns" | **CONTRADICTED on both halves** | (a) Mode 3 is NOT a documented mode. The mode space is the closed set {0,1,2,4} in CANN 9.2.0 (§16.3), CANN 9.0.0 (§25.2), the arch-2201 spec page (§30.1), the C-API 「四种核间同步模式」 (§21.1), and a third-party DSL spec (§26.2). Mode 3 appears in ZERO call sites, constants, branches or comments across ALL arch impls (§29.1). (b) No "reverse mode" is needed: mode 2 already carries AIV->AIC (rows above). (c) Why mode 3 can LOOK like it works: `mode & 0x3` makes 3 a reachable distinct encoding (§19.4), AND on 2201 `WaitEventImpl` does `(void)modeId; wait_flag_dev(flagId);` so the WAIT CANNOT DISTINGUISH ANY MODE (§19.5, §17.3, §30.3). Any set-mode that bumps the right counter will appear to "work in all set/wait patterns" — including, trivially, mode 3 if its hardware semantics happen to overlap mode 2's. The observation is real; the inference that mode 3 is the only working reverse mode is not supported. |

### 33.2 Contradictions BETWEEN sources (all verbatim-sourced above)

| Topic | Source A | Source B | Resolution |
| --- | --- | --- | --- |
| flagId range on A3 | CANN 9.0.0 hiascend: 「Atlas A3 ... 取值范围是0-10」 (§25.4) | CANN 9.2.0 asc-devkit: 「支持的取值范围为0-15」 + 「各自有16个flagId」 (§16.7, §16.8) | Hardware range is 0-15 (`& 0xf` is dispositive, §19.4). `0-10` is the old doc's conservative bound carving out the library-reserved 11-15. Every blog repeating "0-10" is reciting CANN 9.0.0. |
| Counter depth usability | API ref: 「每一个计数器计数范围为0-15。如果调用CrossCoreSetFlag的次数超过15次 ... 异常报错」 (§16.8) | arch-2201 spec: 「**不允许连续设置同一个flagId**，以防止计数器状态混乱」 (§30.3) | Both hold. The counter physically holds 0-15 and traps on overflow, but Huawei does not sanction consecutive sets on one flagId. Pipeline depth must come from DISTINCT flagIds. The arch pages carry this rule; the API pages do not. |
| `PIPE_S` as SET pipe | C++ page: modes 0/1/2 「不支持的流水类型为PIPE_ALL、PIPE_S」 (§16.6) | C-API 2201 tables omit PIPE_S from BOTH the AIC and AIV rows (§21.4); CANN 9.0.0: 「PIPE_S流水类型仅Atlas 350 加速卡支持」 (§25.3) | Agreement: PIPE_S is never a legal set pipe on 2201. |
| Set pipe legality, per core type | C++ page lists one flat set {PIPE_V, PIPE_M, PIPE_MTE1, PIPE_MTE2, PIPE_MTE3, PIPE_FIX} for modes 0/1/2 (§16.6) | C-API splits it: 2201 AIC {M, MTE1, MTE2, MTE3, FIX}, AIV {MTE2, MTE3, V} (§21.4) | The C-API is SHARPER and correct: the flat C++ list is the UNION. PIPE_FIX is AIC-only, PIPE_V is AIV-only. Use the per-core-type table. |
| `PIPE_S` on WAIT side, 950/3510 | CrossCoreWaitFlag page: for 950 「模式0、模式1、模式2和模式4支持`PIPE_S`、...」 (§17.3) | asc-devkit sample + its README: 「NPU架构`dav-3510`的模式0、模式1和模式2**不支持默认的`PIPE_S`**」 (§23.3) | Unresolved doc-vs-sample conflict; 3510-only, irrelevant on A3 where the parameter is inert. Trust the sample. |
| `PIPE_FIX` support | CANN 8.0.RC2.2 intra-core pipe table: 「PIPE_FIX | L0C->GM、L0C->L1数据搬运流水，**当前版本暂不支持该流水类型**」 (§27.1) | PIPE_FIX is the canonical AIC CrossCoreSetFlag pipe in CANN 9.x docs, samples and shipped operators (§16.6, §21.4, §23.4, §24.1, §30.2) | The 8.0.RC2.2 caveat is about intra-core `SetFlag`/`WaitFlag` HardEvents, not about CrossCoreSetFlag. PIPE_FIX IS a valid cross-core set pipe on 2201. |

### 33.3 Facts no single source states but that follow from the above

- `CrossCoreSetFlag<4, pipe>(flagId)` compiled for `__NPU_ARCH__ == 2201` SILENTLY BECOMES MODE 0, because `4 & 0x3 == 0` and the 2201 impl has no `if constexpr (modeId == 4)` branch. No diagnostic. (§19.4 + §19.9 + §29.3)
- `modeId >= 5` likewise aliases mod 4: `5->1`, `6->2`, `7->3`, `8->0`. (§19.4)
- On 2201 the mode lives ENTIRELY in the set instruction; the wait is a mode-agnostic, pipe-agnostic, all-pipe block on the local flagId counter. (§19.5 + §17.3 + §21.5 + §30.3)
- With 4 Matmul objects + `SyncAll<false>` on 2201, the library claims flagIds [0,7] ∪ {11,12,13}; the free set is {8, 9, 10, 14, 15}, shrinking to {8, 9, 10} if SuperKernel fusion (15) or SuperKernel task sync (14) is also in play. Huawei's own C-API sample picks `FLAG_ID = 8` and the shipped `rope_matrix` operator picks `0x8`. (§18.2 + §21.9 + §24.1)
- The mode-2 AIV->AIC gather on 2201 requires BOTH AIVs of the AI Core to set. One AIV setting alone leaves the AIC counter at 0 and the AIC blocked forever. (§31.3 + §28.2). This is the most likely real cause of an observed "mode 2 reverse doesn't work" in a kernel where only one AIV lane has work.

## Relevance

The two highest-value items: (1) mode 2 IS the bidirectional cube<->vector mode on A3 and Huawei's own CANN 9.2.0 sample uses it both ways in one kernel, so no undocumented mode is needed; (2) the reason an undocumented mode can appear to work is that the 2201 WAIT instruction has no mode field at all — `(void)modeId; wait_flag_dev(flagId);` — so the wait cannot distinguish modes and "works in all set/wait patterns" is not evidence about the mode.

# Topic 2 — LCCL / Lcal device-side one-sided all-reduce and the epoch-tagged (MAGIC) flag protocol

## 34. LCCL `sync_collectives.h` — the complete epoch-tagged flag protocol (THE core file)

URL: https://github.com/egretwAlker/cpl-ascend-transformer-boost/blob/master/src/kernels/lcal/src/ascendc_kernels/sync_collectives.h
(upstream: `ascend-transformer-boost` / ATB, `src/kernels/lcal/src/ascendc_kernels/sync_collectives.h`; same file is vendored verbatim into `vllm-project/vllm-ascend` at `csrc/utils/inc/kernel/sync_collectives.h` and `sgl-project/sgl-kernel-npu` at `csrc/deepep/ops/op_kernel/sync_collectives.h`)
Author: Huawei Technologies Co., Ltd. (CANN Open Software License Agreement Version 1.0)
Date: Copyright 2024; file present in ATB releases through 2025

This is the whole file verbatim. It is the answer to "how does a receiving rank know a remote write
has landed without a host barrier".

```cpp
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_SYNC_H
#define LCCL_SYNC_H

#include "comm_args.h"

using namespace AscendC;
using namespace Lcal;

constexpr int64_t FLAG_UNIT_INT_NUM = 4;
constexpr int64_t SYNC_UNIT_SIZE = FLAG_UNIT_INT_NUM * sizeof(int64_t);
constexpr int64_t MAGIC_OFFSET = 32;
constexpr int64_t MAGIC_MASK = ~((1LL << MAGIC_OFFSET) - 1);
#ifdef ENABLE_LCCL_MIX
constexpr int32_t LCAL_BLOCK_NUM_MULTI = 2;
#else
constexpr int32_t LCAL_BLOCK_NUM_MULTI = 1;
#endif

class SyncCollectives {
public:
    __aicore__ inline SyncCollectives() {}

    __aicore__ inline void Init(int rank, int rankSize, GM_ADDR *shareAddrs)
    {
        this->rank = rank;
        this->rankSize = rankSize;
        this->shareAddrs = shareAddrs;
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum() * LCAL_BLOCK_NUM_MULTI;
        segmentCount = GetBlockNum() * LCAL_BLOCK_NUM_MULTI * FLAG_UNIT_INT_NUM;
        localSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]);
        basicSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        blockOuterSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + segmentCount + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        TPipe pipe;
        pipe.InitBuffer(tBuf, GetBlockNum() * SYNC_UNIT_SIZE);
    }

    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag(localSyncAddr + eventID * FLAG_UNIT_INT_NUM, v);
    }

    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, v);
    }

    __aicore__ inline int32_t CalEventIdByMulBlockNum(int32_t blockMultiplier, int32_t targetCoreId)
    {
        return (blockMultiplier * blockNum) + targetCoreId;
    }

    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank,
        int32_t breakCycle = 0)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v, breakCycle);
    }

    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(basicSyncAddr, value);
    }
    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag((__gm__ int64_t*)(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM, value);
    }

    __aicore__ inline void WaitInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM, 1, value);
    }

    __aicore__ inline void WaitRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }

    __aicore__ inline bool CheckRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        return CheckOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }

    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(blockOuterSyncAddr, value);
    }

    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        __gm__ int64_t *flagAddr = GetOuterFlagAddr(setRank, setBlock);
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(flagAddr, value);
    }

    __aicore__ inline void WaitOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t *flagAddr = GetOuterFlagAddr(waitRank, waitBlock);
        WaitOneRankPartFlag(flagAddr, 1, value);
    }

    __aicore__ inline void WaitOneRankOuterFlag(int32_t magic, int32_t eventID, int64_t rank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t *flagAddr;
        flagAddr = GetOuterFlagAddr(rank, 0);
        WaitOneRankPartFlag(flagAddr, blockNum, value);
    }
    __aicore__ inline void WaitAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t *flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            WaitOneRankPartFlag(flagAddr, flagNum, value);
        }
    }

    __aicore__ inline bool CheckAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock,
        int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t *flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            if (!CheckOneRankPartFlag(flagAddr, flagNum, value)) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void WaitAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        WaitAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    __aicore__ inline bool CheckAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        return CheckAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    __aicore__ inline void SetFlag(__gm__ int64_t* setAddr, int64_t setValue)
    {
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        GlobalTensor<int64_t> globalSet;
        globalSet.SetGlobalBuffer(setAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localSet = tBuf.GetWithOffset<int64_t>(1, 0);
        localSet.SetValue(0, setValue);
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(globalSet, localSet, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);

        tBuf.FreeTensor(localSet);
    }

    __aicore__ inline void WaitFlag(__gm__ int64_t* waitAddr, int64_t waitValue)
    {
        WaitOneRankPartFlag(waitAddr, 1, waitValue);
    }

    __aicore__ inline int64_t GetFlag(__gm__ int64_t* waitAddr)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(1, 0);
        DataCopy(localWait, globalWait, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        int64_t res = localWait.GetValue(0);
        tBuf.FreeTensor(localWait);
        return res;
    }

    __aicore__ inline void WaitOneRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank,
                                                    int64_t startBlock, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t *flagAddr;
        flagAddr = GetOuterFlagAddr(waitRank, startBlock);
        WaitOneRankPartFlag(flagAddr, flagNum, value);
    }

    __aicore__ inline int64_t GetInnerFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM);
    }

    __aicore__ inline int64_t GetOuterFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + segmentCount + waitBlock * FLAG_UNIT_INT_NUM);
    }

private:
    __aicore__ inline int64_t MergeMagicWithValue(int32_t magic, int32_t value)
    {
        return (static_cast<int64_t>(magic) << MAGIC_OFFSET) | static_cast<int64_t>(value);
    }

    __aicore__ inline __gm__ int64_t* GetInnerFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + flagBlock * FLAG_UNIT_INT_NUM;
    }

    __aicore__ inline __gm__ int64_t* GetOuterFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + segmentCount + flagBlock * FLAG_UNIT_INT_NUM;
    }

    __aicore__ inline void WaitOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue,
        int32_t breakCycle = 0)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;
        do {
            if (breakCycle > 0) {
                int64_t systemCycleBefore = AscendC::GetSystemCycle();
                int64_t systemCycleAfter = AscendC::GetSystemCycle();
                while (systemCycleAfter - systemCycleBefore < breakCycle) {
                    systemCycleAfter = AscendC::GetSystemCycle();
                };
            }
            DataCopy(localWait, globalWait, flagNum * FLAG_UNIT_INT_NUM);
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            isSync = true;
            for (auto i = 0; i < flagNum; ++i) {
                int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
                if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                    isSync = false;
                    break;
                }
            }
        } while (!isSync);
        tBuf.FreeTensor(localWait);
    }

    __aicore__ inline void WaitOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        WaitOneRankPartFlag(waitAddr, blockNum, checkValue);
    }

    __aicore__ inline bool CheckOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        DataCopy(localWait, globalWait, flagNum * FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        bool isSync = true;
        for (auto i = 0; i < flagNum; ++i) {
            int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
            if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                isSync = false;
                break;
            }
        }
        tBuf.FreeTensor(localWait);
        return isSync;
    }

    __aicore__ inline bool CheckOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        return CheckOneRankPartFlag(waitAddr, blockNum, checkValue);
    }

    int rank;
    int rankSize;
    int blockIdx;
    int blockNum;
    GM_ADDR *shareAddrs;
    int64_t segmentCount;
    __gm__ int64_t* localSyncAddr;
    __gm__ int64_t* basicSyncAddr;
    __gm__ int64_t* blockOuterSyncAddr;
    TBuf<QuePosition::VECCALC> tBuf;
};

#endif // LCCL_SYNC _H
```

Exact arithmetic to extract, stated plainly:

```
FLAG_UNIT_INT_NUM = 4                      // each flag occupies 4 x int64 = 32 B (one UB block); only [0] carries the value
SYNC_UNIT_SIZE    = 4 * 8 = 32 B
MAGIC_OFFSET      = 32
MAGIC_MASK        = ~((1LL << 32) - 1)     // = 0xFFFFFFFF00000000

written value     = (int64_t(magic) << 32) | int64_t(eventID)     // MergeMagicWithValue

inner flag addr   = (int64_t*)shareAddrs[rank] + blockIdx * 4
outer flag addr   = (int64_t*)shareAddrs[rank] + segmentCount + blockIdx * 4
segmentCount      = GetBlockNum() * LCAL_BLOCK_NUM_MULTI * 4      // in int64 units

spin predicate (the ONLY place staleness is decided):
    if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) { not ready }
    // i.e. epoch must match EXACTLY, and within the epoch the eventID must have advanced >= expected
```

## Relevance
This is the exact protocol we need: a 64-bit flag where the high 32 bits are the epoch (magic) and the
low 32 bits are a monotone per-epoch event/stage id, with a spin that demands exact epoch equality plus
`>=` on the stage. Note `ENABLE_LCCL_MIX` doubling `LCAL_BLOCK_NUM_MULTI` to 2 — LCCL already has a MIX
(AIC+AIV) build in which the flag array is sized for twice the block count, which is precisely our
megakernel's shape.

## 35. `op_def.h` — `SET_MAGIC`: the epoch counter computed ON DEVICE in MIX mode (no host round trip)

URL: https://github.com/egretwAlker/cpl-ascend-transformer-boost/blob/master/src/kernels/lcal/src/ascendc_kernels/op_def.h
Author: Huawei Technologies Co., Ltd. (CANN Open Software License Agreement Version 1.0)
Date: Copyright 2024

This is the single most important file for a megakernel: under `ENABLE_LCCL_MIX`, LCCL does **not** take
the epoch from the host. `blockIdx == 0` performs an **atomic add of 1** into `commArgs->magics[rankSize-1]`
in device global memory, then every block reads it back, and that read value becomes `magic`. Full file:

```cpp
#ifndef LCCL_OP_DEF_H
#define LCCL_OP_DEF_H
#define GET_COMM_ARGS \
    GlobalTensor<int> commArgsGm; \
    commArgsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(commArgs), 5); \
    int rank = commArgsGm.GetValue(0); \
    int localRank = commArgsGm.GetValue(1); \
    int rankSize = commArgsGm.GetValue(2); \
    int localRankSize = commArgsGm.GetValue(3); \
    uint32_t extraFlag = commArgsGm.GetValue(4); \
    GM_ADDR dumpAddr = (reinterpret_cast<__gm__ CommArgs *>(commArgs))->dumpAddr; \
    int32_t lcalBlockNum = GetBlockNum()

#ifdef ENABLE_LCCL_MIX
#define SET_MAGIC \
do { \
    __gm__ CommArgs * commArgsTmp = reinterpret_cast<__gm__ CommArgs *>(commArgs); \
    PipeBarrier<PIPE_ALL>(); \
    SetAtomicNone(); \
    SetMaskNormImpl(); \
    SetSyncBaseAddr(commArgsTmp->fftsVal); \
    SetVectorMask<int32_t>((uint64_t)-1, (uint64_t)-1); \
    PipeBarrier<PIPE_ALL>(); \
    LocalTensor<int32_t> localSet; \
    localSet.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN); \
    localSet.address_.bufferAddr = reinterpret_cast<uint64_t>((__ubuf__ int32_t *)96); \
    GlobalTensor<int32_t> magicGt; \
    magicGt.SetGlobalBuffer((__gm__ int32_t *)commArgsTmp->magics); \
    if (GetBlockIdx() == 0) { \
        SetAtomicOpType<int32_t>(Op::ADD); \
        localSet.SetValue(0, 1); \
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0); \
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0); \
        DataCopyExtParams dataCopyParams(1, sizeof(int32_t), 0, 0, 0); \
        DataCopyPad(magicGt[rankSize - 1], localSet, dataCopyParams); \
        AscendC::SetAtomicNone(); \
        PipeBarrier<PIPE_ALL>(); \
    } \
    SyncAll<true>(); \
    DataCopyExtParams dataCopyParams(1, sizeof(int32_t), 0, 0, 0); \
    DataCopyPadExtParams<int32_t> padParams; \
    DataCopyPad(localSet, magicGt[rankSize - 1], dataCopyParams, padParams); \
    AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0); \
    AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0); \
    magic = static_cast<int64_t>(localSet.GetValue(0)); \
    PipeBarrier<PIPE_ALL>(); \
    constexpr int32_t aivNumPerAic = 2; \
    lcalBlockNum = GetBlockNum() * aivNumPerAic; \
} while (0)
#else
#define SET_MAGIC \
do {} while (0)
#endif

#define GET_IPC_MEM_ARGS(type) \
do { \
    SET_MAGIC; \
    GlobalTensor<GM_ADDR> peerMemsAddrGm; \
    peerMemsAddrGm.SetGlobalBuffer(&(reinterpret_cast<__gm__ CommArgs *>(commArgs))->peerMems[0], \
                       LCAL_MAX_RANK_SIZE); \
    for (int i = 0; i < rankSize; ++i) { \
        shareAddrs[i] = (__gm__ type *) (peerMemsAddrGm.GetValue(i) + \
                        (magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET)); \
    } \
    AscendC::PipeBarrier<PIPE_ALL>(); \
} while (0) \

#define CLASS_OP_LAUNCH(name, type) \
do { \
    name<type> opKernel(rank, rankSize, extraFlag); \
    opKernel.Init(KERNELS_ARGS_CALL()); \
    opKernel.Process();                 \
} while (0)

#define CLASS_OP_QUANT_910A5_LAUNCH(name, outputType, addType, inputType) \
do { \
    name<outputType, addType, inputType> opKernel(rank, rankSize, extraFlag); \
    opKernel.Init(KERNELS_ARGS_CALL()); \
    opKernel.Process();                 \
} while (0)

#define LCCL_TYPE_FUNC(fun) \
    fun(int,);fun(int8_t,);fun(int16_t,);fun(int64_t,); \
    fun(float,);fun(float16_t,);fun(bfloat16_t,)

#ifdef ENABLE_LCCL_MIX
#define LCCL_TYPE_AIC_FUNC(fun) \
    fun(int, _mix_aic); fun(int8_t, _mix_aic); fun(int16_t, _mix_aic); fun(int64_t, _mix_aic); \
    fun(float, _mix_aic); fun(float16_t, _mix_aic); fun(bfloat16_t, _mix_aic)

#define LCCL_TYPE_AIV_FUNC(fun) \
    fun(int, _mix_aiv); fun(int8_t, _mix_aiv); fun(int16_t, _mix_aiv); fun(int64_t, _mix_aiv); \
    fun(float, _mix_aiv); fun(float16_t, _mix_aiv); fun(bfloat16_t, _mix_aiv)
#else
#define LCCL_TYPE_AIC_FUNC(fun) \
    (void)0

#define LCCL_TYPE_AIV_FUNC(fun) \
    fun(int,); fun(int8_t,); fun(int16_t,); fun(int64_t,); \
    fun(float,); fun(float16_t,); fun(bfloat16_t,)
#endif

#define LCCL_VADD_910B_TYPE_FUNC(fun) \
    fun(int);fun(int16_t); \
    fun(float);fun(float16_t)

#define LCCL_QUANT_LOW_TYPE_FUNC(fun) \
    fun(int8_t)

#endif
```

The host-side half that makes `SyncAll<true>()` legal inside that MIX kernel — `fftsVal` comes from
`rtGetC2cCtrlAddr`, is shipped in `CommArgs`, and is re-installed on device with `SetSyncBaseAddr()`:

```cpp
// src/kernels/lcal/src/lcal_comm.cpp  (LcalComm::SyncCommArgs)
    if (isEnableMix_) {
        uint64_t fftsVal = 0;
        uint32_t fftsLen = 0;
        int error = rtGetC2cCtrlAddr(&fftsVal, &fftsLen);
        if (error != RT_ERROR_NONE) {
            MKI_LOG(ERROR) << "rtGetC2cCtrlAddr err:" << error;
            return LCAL_ERROR_MKIRT;
        }
        commArgs_.fftsVal = fftsVal;
    }
```

And the MIX launch geometry — blockDim is halved and the resource limit queried is the **CUBE** core
count, not the vector core count:

```cpp
// src/kernels/lcal/src/lccl.cpp  Lccl::GetBlockNum
uint32_t Lccl::GetBlockNum(LcalType cclType, uint32_t rankSize, int64_t dataSize,
                           int localRankSize, uint32_t extraFlag) const
{
    if (comm_ == nullptr) {
        MKI_LOG(ERROR) << "comm is nullptr " << __LINE__;
        return 0;
    }
    uint32_t limitVal = 0;
    aclrtDevResLimitType limitType = aclrtDevResLimitType::ACL_RT_DEV_RES_VECTOR_CORE;
    uint32_t blockNum = GetKernelBlockNum(cclType, rankSize, dataSize, localRankSize, extraFlag);
    if (comm_->isEnableMix_) {
        constexpr uint32_t aivNumPerAic = 2;
        if (blockNum % aivNumPerAic == 1) {
            MKI_LOG(ERROR) << "Lccl not support odd block number at msprof op enabled!";
            return 0;
        }
        blockNum = blockNum / aivNumPerAic;
        limitType = aclrtDevResLimitType::ACL_RT_DEV_RES_CUBE_CORE;
    }

    int res = GetAclResInCurThread(static_cast<int>(limitType), limitVal);
    if (res == LCAL_SUCCESS) {
        MKI_LOG(DEBUG) << "Required blockNum(" << blockNum <<
            ") limit:(limitVal=" << limitVal << ", limitType=" << static_cast<int>(limitType) << ")";
        if (blockNum > limitVal) {
            MKI_LOG(ERROR) << "Insufficient blockDim: Required blockNum(" << blockNum <<
                ") exceeds limit (limitVal=" << limitVal << ", limitType=" << static_cast<int>(limitType) << ")";
            return 0;
        }
    }
    return blockNum;
}
```

MIX is also force-enabled whenever op-level profiling is on, and selects a different registered kernel
group (`opGroup`):

```cpp
// src/kernels/lcal/src/lcal_comm.cpp  LcalComm::InitCommon
    const char *lcclDeterministic = Mki::GetEnv("LCCL_DETERMINISTIC");
    if (lcclDeterministic && (string(lcclDeterministic) == "1" || string(lcclDeterministic) == "true")) {
        deterministic_ = true;
        commArgs_.extraFlag |= ExtraFlag::DETERMINISTIC;
    }
    if (GetChipName() == ChipName::CHIP_910B2C) {
        commArgs_.extraFlag |= ExtraFlag::TOPO_910B2C;
    }
    if (GetChipName() >= ChipName::CHIP_910_9391) {
        commArgs_.extraFlag |= ExtraFlag::TOPO_910_93;
    }
    if (GetChipName() > ChipName::CHIP_910_9362) {
        commArgs_.extraFlag |= ExtraFlag::TOPO_910A5;
    }
    if (GetCoreNum(GetChipName()) > AI_CORE_NUM_20) {
        commArgs_.extraFlag |= ExtraFlag::IS_GREATER_THAN_40_AIV;
    }

    ReportTiming report("LcclReporting", rank_, false, nullptr, nullptr);
    MKI_LOG(INFO) << "LcalComm::InitCommon ReportTiming " << std::hex << ReportTiming::ProfilingStatus() << std::dec;
    if (ReportTiming::ProfilingStatus() == ReportTiming::PROF_TASK_TIME_DUMP) {
        isEnableMsprofOp_ = true;
        isEnableMix_ = true;
    }

    int32_t opGroup = 0;
    if (isEnableMsprofOp_) {
        opGroup = 0;
    } else if (isEnableMix_) {
        opGroup = 1;
    } else {
        constexpr int32_t normalOpGroup = 2;
        opGroup = normalOpGroup;
    }
    MKI_LOG(INFO) << "LcalComm::InitCommon RegistKernel opGroup " << opGroup;
    RegistKernel(opGroup);
```

Note the constructor signature: the fifth parameter is spelled `isEnableMagic` but is stored into
`isEnableMix_` — magic-on-device and MIX are the same switch:

```cpp
// src/kernels/lcal/src/lcal_comm.cpp:736-737
LcalComm::LcalComm(int rank, int rankSize, int commDomain, int bufferSize, int isEnableMagic)
    : rank_(rank), rankSize_(rankSize), commDomain_(commDomain), bufferSize_(bufferSize), isEnableMix_(isEnableMagic)
```

```cpp
// src/kernels/lcal/include/lcal_comm.h
    LcalComm(int rank, int rankSize, int commDomain, int bufferSize, int isEnableMagic);
    ...
    int64_t magic_ = 1;          // host-side epoch counter, starts at 1
    bool isEnableMix_ = false;
```

## Relevance
This is the mechanism for an epoch counter with **no host involvement**: one `DataCopyPad` with
`SetAtomicOpType<int32_t>(Op::ADD)` into a 4-byte device slot, a grid sync, then every block reads the
same epoch. Combined with `magic % PING_PONG_SIZE` selecting one of two IPC buffers, successive
all-reduces never alias, so flags are never zeroed. Also directly contradicts "no grid barrier needed":
LCCL's MIX path does call `SyncAll<true>()`, and makes it work by installing `rtGetC2cCtrlAddr`'s value
via `SetSyncBaseAddr()` — see section 48 for why that matters to us.

## 36. `lccl_op.h` — exact one-shot / two-shot / big-data thresholds for AllReduce (device-side dispatch)

URL: https://github.com/egretwAlker/cpl-ascend-transformer-boost/blob/master/src/kernels/lcal/src/ascendc_kernels/lccl_op.h
Author: Huawei Technologies Co., Ltd. (CANN Open Software License Agreement Version 1.0)
Date: Copyright 2024

The algorithm choice happens **inside the kernel**, on device, from `len * sizeof(type)`. Verbatim:

```cpp
#define LCCL_ALL_REDUCE_FUNC_AUTO_DEF(type, suffix) \
extern "C" __global__ __aicore__ void LcalAllReduce_##type##suffix(KERNELS_ARGS_FUN()) { \
    if ASCEND_IS_AIV { \
    GET_COMM_ARGS; \
    constexpr int32_t quickOneshotRankSize = 2; \
    constexpr int32_t threeStepNum = 3; \
    constexpr int32_t smallRankSize = 8; \
    constexpr int32_t oneshotDataSize = 16 * 1024; \
    constexpr int64_t quantSmallDataSize = 512 * 1024; \
    constexpr int32_t cceSmallDataSize = 2 * 1024 * 1024; \
    constexpr int32_t smallDataSize910a3 = 32 * 1024 * 1024; \
    constexpr int32_t rankSize910a3 = 16; \
    constexpr int32_t scaleCountMax = 12 * 1024 * 1024; \
    __gm__ type * shareAddrs[LCAL_MAX_RANK_SIZE]; \
    GET_IPC_MEM_ARGS(type); \
    if ((extraFlag & ExtraFlag::TOPO_PCIE) != 0) { \
        if (len * sizeof(type) < SIZE_OF_8M) {      \
            LcalAllReduce2npuWrite<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else { \
            LcalAllReduce2npuBigDataWrite<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } \
    } else if ((extraFlag & ExtraFlag::QUANT_FP16) != 0 && std::is_same_v<type, int8_t>) { \
        if (len * sizeof(type) <= oneshotDataSize) { \
            CLASS_OP_QUANT_LAUNCH(AllReduceOneShot, half, int8_t); \
        } else if (len * sizeof(type) <= quantSmallDataSize) { \
            CLASS_OP_QUANT_LAUNCH(AllReduceTwoShot, half, int8_t); \
        } else if (scaleCount * rankSize <= scaleCountMax) { \
            CLASS_OP_QUANT_LAUNCH(AllReduceBigData, half, int8_t); \
        } else { \
            AllReduceBigData<half, int8_t> opTmp(localRank, localRankSize, extraFlag); \
            opTmp.Init(KERNELS_ARGS_CALL()); \
            opTmp.SupportBigScale(); \
            input = output; \
            CLASS_OP_LAUNCH(AllReduceBigData, half); \
        } \
    } else if ((extraFlag & ExtraFlag::TOPO_910B2C) != 0 && rankSize > smallRankSize) { \
        if (len * sizeof(type) < cceSmallDataSize) {      \
            LcalAllReduceTwoShot910B2C<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else { \
            LcalAllReduceBigData910B2C<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } \
    } else if ((extraFlag & ExtraFlag::DETERMINISTIC) != 0) { \
        constexpr uint32_t maxAivNum = 40; \
        const bool isAivNumSupport = ((extraFlag & ExtraFlag::IS_GREATER_THAN_40_AIV) != 0 ||  \
            rankSize * threeStepNum <= maxAivNum); \
        if ((extraFlag & ExtraFlag::TOPO_910_93) != 0) { \
            if (rankSize % quickOneshotRankSize == 1 || rankSize == quickOneshotRankSize || \
                (rankSize <= rankSize910a3 && len * sizeof(type) <= smallDataSize910a3 && isAivNumSupport)) { \
                LcalAllReduceDeterministicBigData<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
            } else { \
                CLASS_OP_LAUNCH(AllReduceHierarchyDoubleRing, type); \
            } \
        } else if (len * sizeof(type) < SMALL_DATA_SIZE) { \
            LcalAllReduceDeterministic<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else { \
            LcalAllReduceDeterministicBigData<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } \
    } else if ((extraFlag & ExtraFlag::TOPO_910_93) != 0 && lcalBlockNum != rankSize && \
        (rankSize == quickOneshotRankSize || len * sizeof(type) > smallDataSize910a3)) { \
        if (rankSize == quickOneshotRankSize) { \
            LcalAllReduce2npuBigDataWrite<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else if (rankSize % quickOneshotRankSize == 0) { \
            CLASS_OP_LAUNCH(AllReduceHierarchyDoubleRing, type); \
        } else { \
            CLASS_OP_LAUNCH(AllReduceBigDataSio, type); \
        } \
    } else { \
        if (len * sizeof(type) < cceSmallDataSize or lcalBlockNum == rankSize) {      \
            if (rankSize == quickOneshotRankSize && lcalBlockNum != rankSize) { \
                LcalAllReduce2npuRead<type>(ALLREDUCE_ARGS_CALL(type)); \
            } else { \
            LcalAllReduceTwoShot<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
            } \
        } else { \
            LcalAllReduceBigData<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } \
    } \
    } \
}
```

Thresholds, extracted (our case: TP16 → `rankSize == 16`, `TOPO_910_93` set on 910_9392, bf16):

```
oneshotDataSize      = 16 * 1024          // 16 KB   — ONLY used on the int8->fp16 quant path
quantSmallDataSize   = 512 * 1024         // 512 KB  — quant two-shot ceiling
cceSmallDataSize     = 2 * 1024 * 1024    // 2 MB    — the general two-shot / big-data split
smallDataSize910a3   = 32 * 1024 * 1024   // 32 MB   — A3 (910_93) hierarchy/double-ring entry
SIZE_OF_8M           = 8 * 1024 * 1024               // PCIe 2-rank split
SMALL_DATA_SIZE      = 1 * 1024 * 1024    // 1 MB    // from comm_args.h; deterministic non-A3 split
rankSize910a3        = 16
maxAivNum            = 40
scaleCountMax        = 12 * 1024 * 1024

=> TP16 bf16 non-deterministic on A3, payload <= 32 MB:
   falls through the TOPO_910_93 branch (len*sizeof <= 32MB), lands in the final else:
     len*sizeof(type) <  2 MB   -> LcalAllReduceTwoShot<type>
     len*sizeof(type) >= 2 MB   -> LcalAllReduceBigData<type>
   batch=1 TP16 hidden-size payloads are a few KB  -> ALWAYS LcalAllReduceTwoShot.
   (The templated AllReduceOneShot class is reachable only via the quant path, <=16 KB.)
```

Host-side block-dim selection for the same decision (`src/kernels/lcal/src/lccl.cpp`):

```cpp
uint32_t GetAllReduceBlockNum(uint32_t rankSize, int64_t dataSize, uint32_t extraFlag)
{
    constexpr uint32_t twoBlockNum = 2;
    constexpr uint32_t threeStepNum = 3;
    constexpr uint32_t dbRingBlockNum = 34;
    constexpr int64_t smallDataSize = 1 * 1024 * 1024;
    constexpr uint32_t smallRankSize = 8;
    constexpr uint32_t cceSmallDataSize = 2 * 1024 * 1024;
    constexpr uint32_t quickOneshotRankSize = 2;
    const int64_t quantSmallDataSize = ((extraFlag & ExtraFlag::QUANT_FP16) != 0) ? (smallDataSize / 2) : smallDataSize;
    constexpr int32_t smallDataSize910a3 = 32 * 1024 * 1024;

    if ((extraFlag & ExtraFlag::TOPO_PCIE) != 0) {
        return rankSize * twoBlockNum;
    } else if ((extraFlag & ExtraFlag::QUANT_FP16) != 0) {
        return dataSize <= quantSmallDataSize ? rankSize : rankSize * twoBlockNum;
    } else if ((extraFlag & ExtraFlag::TOPO_910B2C) != 0 && rankSize > smallRankSize) {
        return dataSize < cceSmallDataSize ? rankSize : (rankSize / twoBlockNum * threeStepNum + twoBlockNum);
    } else if ((extraFlag & ExtraFlag::DETERMINISTIC) != 0) {
        return GetAllReduceDetermBlockNum(rankSize, dataSize, extraFlag);
    }

    if (GetParallel()) {
        return rankSize;
    }

    if ((extraFlag & ExtraFlag::TOPO_910_93) != 0 && dataSize > smallDataSize910a3 &&
        (rankSize != quickOneshotRankSize)) {
        return rankSize % quickOneshotRankSize == 0 ? dbRingBlockNum : rankSize * threeStepNum;
    }
    return (rankSize == quickOneshotRankSize || dataSize >= cceSmallDataSize) ? rankSize * twoBlockNum : rankSize;
}

uint32_t GetAllReduceDetermBlockNum(uint32_t rankSize, int64_t dataSize, uint32_t extraFlag)
{
    constexpr uint32_t quickOneshotRankSize = 2;
    constexpr uint32_t twoBlockNum = 2;
    constexpr uint32_t threeStepNum = 3;
    constexpr uint32_t rankSize910a3 = 16;
    constexpr uint32_t dbRingBlockNum = 34;
    constexpr int64_t smallDataSize = 1 * 1024 * 1024;
    constexpr int32_t smallDataSize910a3 = 32 * 1024 * 1024;
    if ((extraFlag & ExtraFlag::TOPO_910_93) != 0) {
        constexpr uint32_t maxAivNum = 40;
        const bool isAivNumSupport = ((extraFlag & ExtraFlag::IS_GREATER_THAN_40_AIV) != 0 ||
            rankSize * threeStepNum <= maxAivNum);
        if (rankSize % quickOneshotRankSize == 1 || rankSize == quickOneshotRankSize ||
            (rankSize <= rankSize910a3 && dataSize <= smallDataSize910a3 && isAivNumSupport)) {
            return rankSize * threeStepNum;
        } else {
            return dbRingBlockNum;
        }
    }
    if (dataSize < smallDataSize) {
        return rankSize * twoBlockNum;
    }
    return rankSize * threeStepNum;
}
```

Note the env switch `LCCL_PARALLEL`, which collapses block count to `rankSize` and therefore, via
`lcalBlockNum == rankSize` in the kernel, forces the two-shot path:

```cpp
bool GetParallel()
{
    static int parallel = -1;
    if (parallel == -1) {
        static const char *ENV = Mki::GetEnv("LCCL_PARALLEL");
        parallel = (ENV && (string(ENV) == "1" || string(ENV) == "true")) ? 1 : 0;
        MKI_LOG(INFO) << "LCCL_PARALLEL is " << parallel;
    }
    return static_cast<bool>(parallel);
}
```

For TP16 non-deterministic bf16 with a small payload, `GetAllReduceBlockNum` returns `rankSize` = **16
blocks** (one block per rank), which in-kernel means `lcalBlockNum == rankSize` → `LcalAllReduceTwoShot`.

## Relevance
For batch=1 TP16 the payload is a few KB, far below every threshold, so the only algorithm that matters
to us is `LcalAllReduceTwoShot` at `blockDim == rankSize == 16` — 16 AIV blocks, one per peer. That is
the shape to copy, and it is small enough to fuse: 16 blocks, one flag set and one flag wait per block.

## 37. `allreduce_two_shot.h` — the actual two-shot choreography and its flag sequence

URL: https://github.com/egretwAlker/cpl-ascend-transformer-boost/blob/master/src/kernels/lcal/src/ascendc_kernels/allreduce_two_shot.h
Author: Huawei Technologies Co., Ltd. (CANN Open Software License Agreement Version 1.0)
Date: Copyright 2024
The whole flag choreography of `Process()` is **two flag writes and three flag waits per block** — no
`SyncAll`, no hardware event IDs beyond `EVENT_ID0`:

```cpp
    FORCE_INLINE_AICORE void Process()
    {
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
        if (blockIdx >= rankSize) {
            DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
            return;
        }
        if constexpr (std::is_same_v<T, U>) {
            Collectives::CpGM2GM(inputIpcGt, inputGt, dataNumPreBlock, COPYONLY);
        } else {
            if (peerRank == rank) {
                if (!isEnableScale) {
                    Collectives::CpGM2GM(inputIpcGt, inputGt, dataNumPreBlock, COPYONLY);
                } else if (!isVectorScale) {
                    CpGM2GM(inputIpcGt, inputGt, dataNumPreBlock, COPYONLY, firstScale, offset);
                } else {
                    CpGM2GM(inputIpcGt, inputGt, dataNumPreBlock, COPYONLY, scaleGt, scaleNum, offset);
                }
            } else {
                GlobalTensor<U> inputIpcGtTmp;
                inputIpcGtTmp.SetGlobalBuffer((__gm__ U*)inputIpcGt.GetPhyAddr());
                Collectives::CpGM2GM(inputIpcGtTmp, inputGt, dataNumPreBlock, COPYONLY);
            }
        }
        sync.SetInnerFlag(magic, 1);

        sync.WaitInnerFlag(magic, 1, rank, coreSegmentedIdx + rank * corePerRank);
        sync.WaitInnerFlag(magic, 1, peerRank, coreSegmentedIdx + rank * corePerRank);
        if (peerRank != rank) {
            if constexpr (std::is_same_v<T, U>) {
                Collectives::CpGM2GM(processIpcGt, srcIpcGt, ipcDataNumPreBlock, atomOp);
            } else {
                GlobalTensor<U> srcIpcGtTmp;
                srcIpcGtTmp.SetGlobalBuffer((__gm__ U*)srcIpcGt.GetPhyAddr());
                if (!isEnableScale) {
                    Collectives::CpGM2GM(processIpcGt, srcIpcGtTmp, ipcDataNumPreBlock, atomOp);
                } else if (!isVectorScale) {
                    CpGM2GM(processIpcGt, srcIpcGtTmp, ipcDataNumPreBlock, atomOp, firstScale, offset);
                } else {
                    CpGM2GM(processIpcGt, srcIpcGtTmp, ipcDataNumPreBlock, atomOp, scaleGt, scaleNum, offset);
                }
            }
        }

        if (!(extraFlag & ExtraFlag::RDMA)) {
            sync.SetOuterFlag(magic, 1);
            sync.WaitOneRankOuterFlag(magic, 1, peerRank);
            Collectives::CpGM2GM(outputGt, processedIpcGt, dataNumPreBlock, COPYONLY);
        }
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
    }
```

`Init()` — note `blockNum = rankSize;` is forced, so `corePerRank = 1`, `coreSegmentedIdx = 0`,
`peerRank = blockIdx`. Block *i* is the agent for peer *i*:

```cpp
    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
        if constexpr(!std::is_same_v<T, U>) {
            BuildScaleOffset(scale, scaleCount, offset);
        }

        if (blockIdx >= rankSize) {
            DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
            return;
        }

        blockNum = rankSize;

        __gm__ CommArgs *localArgs = reinterpret_cast<__gm__ CommArgs *>(commArgs);

        int localRankSize = localArgs->localRankSize <= 0 ? rankSize : localArgs->localRankSize;
        int globalRankSize = localArgs->rankSize <= 0 ? rankSize : localArgs->rankSize;
        int serverNum = globalRankSize / localRankSize;
        int64_t ipcBuffMaxSizeAligned = IPC_BUFF_MAX_SIZE / (globalRankSize + serverNum - 1) /
            QUEUE_DEPTH / sizeof(T) /scaleNum * scaleNum * QUEUE_DEPTH * sizeof(T) * globalRankSize;
        ipcDataPerParagraphSize = ipcBuffMaxSizeAligned / localRankSize;
        int64_t ipcDataPerParagraphNum = ipcDataPerParagraphSize / sizeof(T);
        atomOp = op;
        corePerRank = blockNum / rankSize;
        coreSegmentedIdx = blockIdx % corePerRank;
        peerRank = blockIdx / corePerRank;
        perRankDataNum = GetDataCount(len, rankSize) / scaleNum * scaleNum;
        curRankDataNum = (rank == rankSize - 1) ? (len - rank * perRankDataNum) : perRankDataNum;
        pullRankDataNum = perRankDataNum;
        if (peerRank == rankSize - 1) {
            pullRankDataNum = len - peerRank * perRankDataNum;
        }
        pullBlockDataNum = GetDataCount(pullRankDataNum, corePerRank);
        dataNumPreBlock = pullBlockDataNum;
        if (coreSegmentedIdx == corePerRank - 1) {
            dataNumPreBlock = pullRankDataNum - coreSegmentedIdx * pullBlockDataNum;
        }
        buffOffsetNum = peerRank * perRankDataNum + coreSegmentedIdx * pullBlockDataNum +
                        ipcDataPerParagraphNum * peerRank;

        curBlockDataNum = GetDataCount(curRankDataNum, corePerRank);
        ipcDataNumPreBlock = curBlockDataNum;
        ipcbuffOffsetNum = rank * perRankDataNum + coreSegmentedIdx * curBlockDataNum + ipcDataPerParagraphNum * rank;

        inputGt.SetGlobalBuffer((__gm__ U*)input + buffOffsetNum - ipcDataPerParagraphNum * peerRank, dataNumPreBlock);
        inputIpcGt.SetGlobalBuffer((__gm__ T*)(shareAddrs[rank] + IPC_DATA_OFFSET) + buffOffsetNum, dataNumPreBlock);
        srcIpcGt.SetGlobalBuffer((__gm__ T*)(shareAddrs[peerRank] + IPC_DATA_OFFSET) + ipcbuffOffsetNum,
                                            ipcDataNumPreBlock);
        processIpcGt.SetGlobalBuffer((__gm__ T*)(shareAddrs[rank] + IPC_DATA_OFFSET) + ipcbuffOffsetNum,
                                    ipcDataNumPreBlock);

        processedIpcGt.SetGlobalBuffer((__gm__ T*)(shareAddrs[peerRank] + IPC_DATA_OFFSET) + buffOffsetNum,
                dataNumPreBlock);
        outputGt.SetGlobalBuffer((__gm__ T*)output + buffOffsetNum - ipcDataPerParagraphNum * peerRank,
                                 dataNumPreBlock);
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
    }

    constexpr static int QUEUE_DEPTH = 4;
```

Protocol summarised exactly:

```
block i (i = peerRank), one block per rank, rankSize blocks total:
  1. CpGM2GM  own input shard -> OWN ipc buffer at offset (i*perRankDataNum + ipcDataPerParagraphNum*i)
  2. SetInnerFlag(magic, 1)                                       // writes to OWN flag slot [blockIdx]
  3. WaitInnerFlag(magic, 1, rank,     coreSegmentedIdx + rank*corePerRank)   // own slot for MY shard
     WaitInnerFlag(magic, 1, peerRank, coreSegmentedIdx + rank*corePerRank)   // PEER's slot for MY shard
  4. CpGM2GM with atomOp:  PEER ipc[my shard] -> OWN ipc[my shard]  (atomic add; the reduce)
  5. SetOuterFlag(magic, 1)                                       // OUTER region, OWN slot [blockIdx]
  6. WaitOneRankOuterFlag(magic, 1, peerRank)                     // ALL blockNum outer slots of peer
  7. CpGM2GM  PEER ipc[peer shard, now fully reduced] -> own output     (the broadcast half)
```

## Relevance
This is the complete recipe for a device-side one-sided all-reduce at TP16: 16 blocks, each the agent
for one peer, with exactly 2 flag stores and 3 flag spins, and the "reduce" being nothing but a
`CpGM2GM` with the hardware atomic-add enabled on a peer's IPC window. All synchronisation lives in
global memory; it consumes **none** of our 16 hardware flag IDs — `SetFlag`/`WaitFlag` inside
`SyncCollectives` only ever uses `EVENT_ID0`.

## 38. `sync_collectives.h` as vendored into vllm-ascend and sgl-kernel-npu — the documented variant, and the "resume where you left off" spin

URL: https://github.com/vllm-project/vllm-ascend/blob/main/csrc/utils/inc/kernel/sync_collectives.h
URL: https://github.com/sgl-project/sgl-kernel-npu/blob/main/csrc/deepep/ops/op_kernel/sync_collectives.h
Author: vLLM Ascend / SGLang contributors, re-licensing Huawei's LCCL file (namespace changed `Lcal` -> `Moe`)
Date: present in vllm-ascend main and sgl-kernel-npu main as of 2025-2026

Same protocol, but with the semantics written out in the header comments — this is the authoritative
statement of what the two 32-bit halves mean:

```cpp
    /**
     * @brief Set the flag for the specified eventID of the designated card, with the value being a combination of magic
     * and value.
     * @param magic The operator batch, which will be combined into the high 32 bits of the flag value to be set.
     * @param value The specific value to be set, which will be the low 32 bits of the flag value to be set.
     * @param eventID Physically, it is an offset from the shared memory base address (requires scaling, not an absolute
     * value).
     * @param rank This rank is the rankId corresponding to the peerMems array in the CommArgs structure, not a global
     * or local id. (Local is not applicable in the 91093 scenario, and global is not applicable in the 910B
     * multi-machine scenario.)
     */
```

Three differences from the ATB original that matter:

1. `Init` takes the `TBuf` from the caller instead of constructing its own `TPipe` — necessary when the
   sync object lives inside a larger fused kernel that already owns the pipe:

```cpp
    __aicore__ inline void Init(int rank, int rankSize, GM_ADDR *shareAddrs, TBuf<QuePosition::VECCALC> &tBuf)
    {
        this->rank = rank;
        this->rankSize = rankSize;
        this->shareAddrs = shareAddrs;
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum();
        // Length of a single indicator segment
        segmentCount = GetBlockNum() * FLAG_UNIT_INT_NUM;
        // Initialize the intra-card/inter-card synchronization address corresponding to the current core.
        localSyncAddr = (__gm__ int64_t *)(shareAddrs[rank]);
        basicSyncAddr = (__gm__ int64_t *)(shareAddrs[rank]) + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        blockOuterSyncAddr = (__gm__ int64_t *)(shareAddrs[rank]) + segmentCount + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        this->tBuf = tBuf;
    }
```

2. The spin loop remembers how far it already validated, so a 16-peer wait does not re-read peers that
   already arrived (`checkedFlagNum`):

```cpp
    __aicore__ inline void WaitOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;
        int64_t checkedFlagNum = 0;
        do {
            // Copy global synchronization flags to local
            DataCopy(localWait, globalWait[checkedFlagNum * FLAG_UNIT_INT_NUM],
                     (flagNum - checkedFlagNum) * FLAG_UNIT_INT_NUM);
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB

            // Check if the synchronization flags are equal to checkValue
            isSync = true;
            int64_t remainToCheck = flagNum - checkedFlagNum;
            for (auto i = 0; i < remainToCheck; ++i) {
                // Continue waiting if any core has not reached the checkValue phase
                int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
                if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                    isSync = false;
                    checkedFlagNum += i;
                    break;
                }
            }
        } while (!isSync);
    }
```

3. `SetFlag` is annotated — note that **no `dcci` appears anywhere in this file**. The entire visibility
   argument rests on `DataCopy` UB->GM through MTE3 and `DataCopy` GM->UB through MTE2, bracketed by
   `HardEvent::MTE3_S` / `HardEvent::MTE2_S` / `HardEvent::S_MTE3` waits, all on `EVENT_ID0`:

```cpp
    __aicore__ inline void SetFlag(__gm__ int64_t* setAddr, int64_t setValue)
    {
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        GlobalTensor<int64_t> globalSet;
        globalSet.SetGlobalBuffer(setAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localSet = tBuf.GetWithOffset<int64_t>(1, 0);
        localSet.SetValue(0, setValue);

        // Copy global synchronization flag to local
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);  // Wait for SetValue to complete
        DataCopy(globalSet, localSet, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);  // Wait for UB->GM to complete
    }
```

sgl-kernel-npu additionally exposes a range wait over `[eventID, eventID + flagNum)`:

```cpp
    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t *)(shareAddrs[this->rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v);
    }

    /**
     * @brief Wait for the flags starting from the specified eventID on the specified card to become
     *        a value composed of the combination of magic and value.<br>
     *        Note: [eventID, eventID + flagNum)
     */
    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank, int64_t flagNum)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t *)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, flagNum, v);
    }
```

## Relevance
Confirms the LCCL protocol is the de-facto standard device-side sync on Ascend — identical bit layout
reproduced in two independent inference stacks. The vllm/sgl `Init(..., TBuf&)` signature is the one to
copy for a megakernel, and the `checkedFlagNum` optimisation matters at TP16 where a single wait scans
16 peer slots.

## 39. `hccl_shmem.hpp` (sgl-kernel-npu) — raw `__gm__` load/store flags WITH explicit `dcci`, plus HCCL symmetric-memory window access

URL: https://github.com/sgl-project/sgl-kernel-npu/blob/main/csrc/deepep/ops/op_kernel/dispatch_ffn_combine_kernel/utils/hccl_shmem.hpp
Author: SGLang / sgl-kernel-npu contributors
Date: present in sgl-kernel-npu main (2025-2026)

A completely different flag family from LCCL's: plain pointer dereference plus an explicit cache-line
invalidate. This is the `dcci` idiom, with the compiler-barrier trick required to stop it being hoisted:

```cpp
#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
constexpr int32_t MAX_RANK_SIZE = 32;
constexpr int32_t SHMEM_MEM = 700 * MB_SIZE;

constexpr uint16_t SEND_SYNC_EVENT_ID = 9;
constexpr uint16_t RECV_SYNC_EVENT_ID = 10;

constexpr uint32_t SELF_STATE_OFFSET = 256 * 1024;
constexpr uint32_t STATE_OFFSET = 512;

FORCE_INLINE_AICORE void AicSyncAll()
{
    AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(8);
    AscendC::CrossCoreWaitFlag<0x0>(8);
}

template <typename T>
FORCE_INLINE_AICORE void gm_store(__gm__ T *addr, T val)
{
    *((__gm__ T *)addr) = val;
}

template <typename T>
FORCE_INLINE_AICORE T gm_load(__gm__ T *cache)
{
    return *((__gm__ T *)cache);
}

template <typename T>
FORCE_INLINE_AICORE void gm_dcci(__gm__ T *addr)
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(addr));

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

FORCE_INLINE_AICORE int32_t gm_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    do {
        gm_dcci((__gm__ uint8_t *)sig_addr);
        if (*sig_addr == cmp_val) {
            return *sig_addr;
        }
        if (*sig_addr == cmp_val + 1) {
            return *sig_addr;
        }
    } while (true);
    return -1;
}

FORCE_INLINE_AICORE void gm_signal_wait_until_ne(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    do {
        AscendC::LocalTensor<int32_t> ub;
        ub.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
        ub.address_.bufferAddr = 0;
        AscendC::GlobalTensor<int32_t> sig;
        sig.SetGlobalBuffer(sig_addr);
        AscendC::DataCopy(ub, sig, 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        if (ub(0) != cmp_val) {
            return;
        }
    } while (true);
    return;
}
```

The symmetric-memory addressing — HCCL "window" buffers, obtained from the device-side HCCL context,
with the peer's window reached through `remoteRes[index].nextDevicePtr`:

```cpp
class HcclShmem
{
public:
#ifdef HCCL_COMM  // HCCL needs to initialize the HCCL context
    __gm__ HcclOpResParamCustom *WinContext_{nullptr};
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    AscendC::LocalTensor<int32_t> ub;
    FORCE_INLINE_AICORE
    HcclShmem()
    {
        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        WinContext_ = (__gm__ HcclOpResParamCustom *)contextGM0;

        m_rank = WinContext_->localUsrRankId;
        m_rankSize = WinContext_->rankSize;
        m_segmentSize = WinContext_->winSize;
    }
#else
    FORCE_INLINE_AICORE
    HcclShmem()
    {
        m_segmentSize = SHMEM_MEM;
    }
    FORCE_INLINE_AICORE
    void initShmem(GM_ADDR symmetricPtr_, size_t rank, size_t rankSize)
    {
        symmetricPtr = symmetricPtr_;
        m_rank = rank;
        m_rankSize = rankSize;
    }
#endif

    FORCE_INLINE_AICORE
    GM_ADDR operator()() const
    {  // No parameters: return pointer to local peermem
#ifdef HCCL_COMM
        return (GM_ADDR)(WinContext_->localWindowsIn);
#else
        return reinterpret_cast<GM_ADDR>(shmem_ptr(symmetricPtr, m_rank));
#endif
    }

    FORCE_INLINE_AICORE
    GM_ADDR operator()(int32_t index) const
    {  // With index parameter: return pointer to the base address of remote peermem
#ifdef HCCL_COMM
        return (
            GM_ADDR)((index == m_rank)
                         ? WinContext_->localWindowsIn
                         : ((HcclRankRelationResV2Custom *)(WinContext_->remoteRes[index].nextDevicePtr))->windowsIn);
#else
        return reinterpret_cast<GM_ADDR>(shmem_ptr(symmetricPtr, index));
#endif
    }

    FORCE_INLINE_AICORE
    GM_ADDR operator()(int64_t offset, int32_t rankId) const
    {
#ifdef HCCL_COMM
        if (offset < 0 || offset >= m_segmentSize) {
            return nullptr;
        }
        if (rankId < 0 || rankId >= m_rankSize) {
            return nullptr;
        }
        return (GM_ADDR)((rankId == m_rank)
                             ? WinContext_->localWindowsIn
                             : ((HcclRankRelationResV2Custom *)(WinContext_->remoteRes[rankId].nextDevicePtr))
                                   ->windowsIn) +
               offset;
#else
        return reinterpret_cast<GM_ADDR>(shmem_ptr((symmetricPtr + offset), rankId));
#endif
    }
```

A full cross-rank barrier built on the above — note the flag region is carved out of the **last 1 MB of
the HCCL window**, each peer gets a 16-int32 stride (64 B), and `SyncAll<true>()` is used for the
intra-device half:

```cpp
    FORCE_INLINE_AICORE
    void CrossRankSync()
    {
        uint64_t flag_offset = (m_segmentSize - MB_SIZE) / sizeof(int32_t);
        __gm__ int32_t *sync_counter = (__gm__ int32_t *)(*this)() + flag_offset;
        __gm__ int32_t *sync_base = (__gm__ int32_t *)(*this)() + flag_offset + 2048;
        int count = gm_load(sync_base) + 1;
        int vec_id = AscendC::GetBlockIdx();
        int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        for (int i = vec_id; i < m_rankSize; i += vec_size) {
            __gm__ int32_t *sync_remote = (__gm__ int32_t *)((*this)(i)) + flag_offset + m_rank * 16;
            gm_store(sync_remote, count);
            gm_dcci((__gm__ uint8_t *)sync_remote);
            auto sync_check = sync_counter + i * 16;
            gm_signal_wait_until_eq_for_barrier(sync_check, count);
        }

        AscendC::SyncAll<true>();
        gm_store(sync_base, count);
    }

    FORCE_INLINE_AICORE
    __gm__ int32_t *SyncBaseAddr()
    {
        uint64_t flag_offset = (m_segmentSize - MB_SIZE) / sizeof(int32_t);
        return (__gm__ int32_t *)(*this)() + flag_offset + 2048;
    }
```

## Relevance
Two things we need. First, the `gm_dcci` idiom with `__asm__ __volatile__("")` on both sides — the
compiler-barrier is mandatory or the invalidate gets hoisted out of the spin. Second, `GetTaskRation()`
in `vec_size = GetBlockNum() * GetTaskRation()`: that is the official way to get the AIV-per-AIC factor
inside a MIX kernel (what LCCL hardcodes as `LCAL_BLOCK_NUM_MULTI`/`aivNumPerAic`). Note `AicSyncAll()`
uses `CrossCoreSetFlag<0x0, PIPE_FIX>(8)` / `CrossCoreWaitFlag<0x0>(8)` — mode `0x0`, flag id 8, from
the cube side; see section 46 for the mode semantics, which bears on our "mode 2 / mode 3" measurement.

## 40. `InitStatusTargetSum` + `CrossRankSyncV2Set/Wait` — the FLIP-FLOP sentinel (no epoch counter) and a VECTORISED 16-peer flag check

URL: https://github.com/sgl-project/sgl-kernel-npu/blob/main/csrc/deepep/ops/op_kernel/dispatch_ffn_combine_kernel/utils/hccl_shmem.hpp
Author: SGLang / sgl-kernel-npu contributors
Date: present in sgl-kernel-npu main (2025-2026)

This is the second way Ascend code avoids the stale-flag race without zeroing: instead of a monotone
epoch, the expected sentinel **alternates between `1.0f` (`0x3F800000`) and `0.0f`** on each call. The
state is read from the window itself, so it survives across kernel launches with no host involvement:

```cpp
    FORCE_INLINE_AICORE
    void InitStatusTargetSum()
    {
        using namespace AscendC;
        uint64_t flag_offset = (m_segmentSize - MB_SIZE) + SELF_STATE_OFFSET;
        // uint64_t self_state_offset = (m_segmentSize - 2 * MB_SIZE);
        //  ep state
        // uint32_t coreIdx = get_block_idx();;
        uint32_t coreIdx = GetBlockIdx();
        GlobalTensor<int32_t> selfStatusTensor;
        selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)((*this)() + flag_offset));
        __asm__ __volatile__("");
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            selfStatusTensor[coreIdx * UB_ALIGN]);
        __asm__ __volatile__("");
        int32_t state = selfStatusTensor(coreIdx * UB_ALIGN);
        if (state == 0) {
            sumTarget_ = static_cast<float>(1.0);
            selfStatusTensor(coreIdx * UB_ALIGN) = 0x3F800000;  // 1.0f
            epStateValue_ = 0x3F800000;                         // 1.0f
        } else {
            sumTarget_ = static_cast<float>(0.0);
            selfStatusTensor(coreIdx * UB_ALIGN) = 0;
            epStateValue_ = 0;
        }
        __asm__ __volatile__("");
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            selfStatusTensor[coreIdx * UB_ALIGN]);
        __asm__ __volatile__("");
    }
```

The set half — one `DataCopy` per destination rank, writing the 4-byte sentinel into the peer's window
at `m_rank * STATE_OFFSET` (512 B per rank):

```cpp
    FORCE_INLINE_AICORE
    void CrossRankSyncV2Set(AscendC::LocalTensor<int32_t> ctrBuffer)
    {
        // subblockid = 0
        uint32_t stateOffset_ = STATE_OFFSET;
        // uint32_t epStateOffsetOnWin_ = m_rank * stateOffset_;

        uint64_t flag_offset = (m_segmentSize - MB_SIZE) + m_rank * stateOffset_;
        // uint64_t flag_offset = (m_segmentSize - MB_SIZE);
        int vec_size = get_block_num();
        int vec_id = get_block_idx();

        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(RECV_SYNC_EVENT_ID);
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SEND_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(SEND_SYNC_EVENT_ID);
        pipe_barrier(PIPE_ALL);

        ctrBuffer.SetValue(0, epStateValue_);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        for (uint32_t dstEpIdx = vec_id; dstEpIdx < m_rankSize; dstEpIdx += vec_size) {
            AscendC::GlobalTensor<int32_t> gmDstStates;
            gmDstStates.SetGlobalBuffer((__gm__ int32_t *)((*this)(flag_offset, dstEpIdx)));
            DataCopy(gmDstStates, ctrBuffer, 8);
        }
        AscendC::CrossCoreWaitFlag(RECV_SYNC_EVENT_ID);
    }
```

The wait half — and this is the trick worth stealing: **all peers' flags are checked with one strided
`DataCopy` + `GatherMask` + `Sum`**, and readiness is `sumOfFlag` landing inside
`[sumTarget_*n - 0.5, sumTarget_*n + 0.5]`. No per-peer scalar loop, no per-peer branch:

```cpp
    FORCE_INLINE_AICORE
    void CrossRankSyncV2Wait(AscendC::LocalTensor<float> statusTensor, AscendC::LocalTensor<float> gatherMaskOutTensor,
                             AscendC::LocalTensor<uint32_t> gatherTmpTensor,
                             AscendC::LocalTensor<float> statusSumOutTensor)
    {
        uint64_t flag_offset = (m_segmentSize - MB_SIZE);
        int vec_size = get_block_num();
        int vec_id = get_block_idx();
        uint32_t stateOffset_ = STATE_OFFSET;

        uint32_t sendRankNum_ = m_rankSize / vec_size;
        uint32_t remainderRankNum = m_rankSize % vec_size;
        uint32_t startRankId_ = sendRankNum_ * vec_id;
        if (vec_id < remainderRankNum) {
            sendRankNum_++;
            startRankId_ += vec_id;
        } else {
            startRankId_ += remainderRankNum;
        }
        uint32_t endRankId_ = startRankId_ + sendRankNum_;
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SEND_SYNC_EVENT_ID);

        AscendC::GlobalTensor<float> epStatusSpaceGlobalTensor_;
        epStatusSpaceGlobalTensor_.SetGlobalBuffer((__gm__ float *)((*this)() + flag_offset));

        if (startRankId_ < m_rankSize) {
            AscendC::PipeBarrier<PIPE_ALL>();
            gatherTmpTensor.SetValue(0, 1);
            uint32_t mask = 1;  // gatherMask + sum
            uint64_t rsvdCnt = 0;
            // DataCopyParams intriParams{static_cast<uint16_t>(sendRankNum_), 1,
            //                         static_cast<uint16_t>((moeSendNum_ > 512) ? 7 : 15), 0};
            AscendC::DataCopyParams intriParams{static_cast<uint16_t>(sendRankNum_), 1, static_cast<uint16_t>(15), 0};

            float sumOfFlag = static_cast<float>(-1.0);
            float minTarget = (sumTarget_ * sendRankNum_) - (float)0.5;
            float maxTarget = (sumTarget_ * sendRankNum_) + (float)0.5;
            AscendC::SumParams sumParams{1, sendRankNum_, sendRankNum_};

            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

            while ((sumOfFlag < minTarget) || (sumOfFlag > maxTarget)) {
                AscendC::DataCopy<float>(
                    statusTensor, epStatusSpaceGlobalTensor_[startRankId_ * stateOffset_ / sizeof(float)], intriParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                GatherMask(gatherMaskOutTensor, statusTensor, gatherTmpTensor, true, mask,
                           {1, (uint16_t)sendRankNum_, 1, 0}, rsvdCnt);

                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                sumOfFlag = statusSumOutTensor.GetValue(0);
            }
        }

        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(RECV_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(RECV_SYNC_EVENT_ID);

        // unpermute
        AscendC::CrossCoreWaitFlag(SEND_SYNC_EVENT_ID);
    }
```

Layout constants, exactly:

```
STATE_OFFSET      = 512            // bytes of window per peer for its status word
SELF_STATE_OFFSET = 256 * 1024     // where a rank keeps its own flip-flop state
flag region       = last MB_SIZE of the HCCL window: offset (winSize - 1 MB)
peer k status at  : window_base(me) + (winSize - 1MB) + k * 512
DataCopyParams    : {blockCount = sendRankNum_, blockLen = 1 (32 B unit), srcStride = 15, dstStride = 0}
                    // srcStride 15 * 32 B = 480 B, + 32 B block = 512 B = STATE_OFFSET. exact.
sentinel          : 0x3F800000 (1.0f) alternating with 0x00000000
readiness test    : sumTarget_*n - 0.5 <= Sum(flags) <= sumTarget_*n + 0.5
```

## Relevance
Two independent, directly usable mechanisms. The flip-flop sentinel removes the epoch counter entirely
(no atomic, no host, no `SyncAll` needed to distribute the epoch) at the cost of requiring every call to
touch every flag. The vectorised check turns a 16-peer spin into one `DataCopy` with `srcStride = 15` +
`GatherMask` + `Sum` — the arithmetic `srcStride*32 + 32 == STATE_OFFSET` is the whole design.

## 41. OFFICIAL vendor doc, arch 2201 — `CrossCoreSetFlag` mode semantics and the EXACT flagId occupancy table. CONTRADICTS our mode-3 and SyncAll-cost measurements

URL: https://github.com/hicann/asc-devkit/blob/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md
URL: https://github.com/hicann/asc-devkit/blob/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.md
Author: Huawei / CANN `asc-devkit` (AscendC devkit, official API reference)
Date: current `master` of hicann/asc-devkit (2026)

### 47a. The modes, verbatim. A3 has modes 0, 1, 2. There is NO mode 3.

```
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持
- Atlas 200I/500 A2推理产品：不支持
- Atlas推理系列产品AI Core：不支持
- Atlas推理系列产品Vector Core：不支持
- Atlas训练系列产品：不支持
```

```
如图1所示（图中block代表AIC，sub_block代表AIV），同步控制分为以下几种模式：

- 模式0：AI Core核间的同步控制。对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行；对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行。
- 模式1：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了CrossCoreSetFlag，CrossCoreWaitFlag后续的指令才会执行。
- 模式2：AI Core内部，AIC与所有AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后，两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。
- 模式4：AI Core内部，AIC与单个AIV之间的同步控制。当在AIC核执行CrossCoreSetFlag之后，AIV0或者AIV1上CrossCoreWaitFlag后续的指令才会继续执行；AIV0或者AIV1执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。
```

```
### modeId支持的取值说明

不同产品对同步模式的支持情况如下：
- 针对Ascend 950PR&950DT系列产品，支持的同步模式为：模式0、模式1、模式2、模式4。
- 针对如下产品，支持的同步模式为：模式0、模式1、模式2。
    - Atlas A3系列产品
    - Atlas A2系列产品
```

Signature and the fact that neither template parameter has a default on `SetFlag`:

```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
```

```
CrossCoreSetFlag的模板参数`modeId`和`pipe`均**没有默认值**，其取值用于配置硬件指令，调用接口时必须**显式传入**。与本接口不同，CrossCoreWaitFlag的两个模板参数均具有默认值，且在不同产品和同步模式下的生效情况不同，因此**两个接口对模板参数的传入要求并不相同**。
```

### 47b. flagId range and counter saturation — exact

```
### flagId取值范围说明

- 核间同步的模式为模式0、1、2时，支持的取值范围为0-15。
```

```
- flagId相关的约束：
    - 对于模式0、1、2，每个AIC和每个AIV都各自有16个flagId，支持的取值范围为0-15。如果flagId的值超出该范围，则截取低4bit（例如，flagId=17时，截取后为1）。
    - 每个flagId都对应一个计数器，当调用CrossCoreWaitFlag时，若计数器值为0则会阻塞后续指令下发，已下发指令可正常执行；当调度模块感知到所有参与同步的核（具体包含哪些核与设置的核间同步模式有关）完成（调用了CrossCoreSetFlag）同步后，会将与设置的flagId对应的计数器的值增加1。此时，计数器值为非0，阻塞解除，并且将对应计数器的值减去1进行还原。具体执行逻辑与细节可以参考关键特性说明。每一个计数器计数范围为0-15。如果调用CrossCoreSetFlag的次数超过15次，计数器的值超出该范围，则会异常报错，中断流程。
```

```
- 一个核连续发出的CrossCoreSetFlag，硬件不保证执行顺序。例如AIV依次发射CrossCoreSetFlag<0, PIPE_V>(0)，CrossCoreSetFlag<0, PIPE_V>(1)，实际的执行顺序可能是flagId=1的先执行。
```

### 47c. Why `KERNEL_TYPE_AIC_ONLY` is rejected — and the documented replacement

```
- 由于当核函数（Kernel）类型为KERNEL_TYPE_AIC_ONLY或KERNEL_TYPE_AIV_ONLY时，硬件不会开启调度模块，也就无法正常进行核间同步，因此不同的同步模式配置核函数（Kernel）类型或函数执行空间限定符的情况如下：
    - 在纯Vector/Cube场景下（模式0或模式1），建议设置核函数（Kernel）类型为KERNEL_TYPE_MIX_AIV_1_0或KERNEL_TYPE_MIX_AIC_1_0，其它支持的核函数（Kernel）类型请参考表3。
    - 对于Vector和Cube混合场景（模式2和模式4），需根据AI Core中AIC和AIV的比例灵活配置核函数（Kernel）类型，不同模式支持的函数执行空间限定符和核函数（Kernel）类型请参照表3。
```

```
        **表3**  模式与支持的核函数（Kernel）类型配置

        | 模式 | 支持的函数执行空间限定符 | 支持的核函数（Kernel）类型配置 |
        | --- | --- | --- |
        | 0 | __mix__(0, 1)、__mix__(1, 0)、__mix__(1, 1)、__mix__(1, 2) | KERNEL_TYPE_MIX_AIV_1_0、KERNEL_TYPE_MIX_AIC_1_0、KERNEL_TYPE_MIX_AIC_1_1、KERNEL_TYPE_MIX_AIC_1_2 |
        | 1 | __mix__(1, 1)、__mix__(1, 2) | KERNEL_TYPE_MIX_AIC_1_1、KERNEL_TYPE_MIX_AIC_1_2 |
        | 2 | __mix__(1, 1)、__mix__(1, 2) | KERNEL_TYPE_MIX_AIC_1_1、KERNEL_TYPE_MIX_AIC_1_2 |
        | 4 | __mix__(1, 2) | KERNEL_TYPE_MIX_AIC_1_2 |
```

### 47d. Mode 0 multi-stream deadlock and the `__schedmode__` batchmode requirement

```
- 使用该接口模式0时，建议开启batchmode模式，使算子独占全部所需核资源，否则可能因满足以下条件导致死锁：
    - 多流并发场景（≥2条执行流）。
    - ≥2个算子并发执行。
    - 所有并发算子的核数总和超过物理核数。
    - ≥2个并发算子使用了核间同步功能。

    具体而言，在多流场景下，某条流的核间同步算子虽分配到n个物理核，但可能仅有n-m个核先被调度执行，而其余m个核因被其他流的核间同步算子抢占而尚未启动。先启动的n-m个核执行到核间同步时等待剩余m核完成，而剩余m核因被其他流的核间同步算子占用而无法释放，形成死锁。
    核函数（Kernel）直调场景下通过__schedmode__(mode)限定符来设置batchmode模式；工程化算子开发场景下，通过TilingContext的SetScheduleMode接口来设置batchmode模式
```

### 47e. THE flagId OCCUPANCY TABLE for arch 2201 (our chip), verbatim from `key_features.md`

```
### 模式0、1、2的flagId占用

**表1**  模式0、1、2flagId占用情况

| 接口 | 限定条件 | 占用flagId | 各计算核占用情况 | NPU架构版本 |
| :--- | :--- | :--- | :--- | :--- |
| SyncAll | `isAIVOnly=true` | 14 | AIV：14 | 2201、3510 |
| SyncAll | `isAIVOnly=false` | 11、12、13 | AIC：11、12、13；AIV：12、13 | 2201、3510 |
| SetNextTaskStart | SuperKernel子核函数（Kernel）中调用 | 11、14 | AIC：11；AIV：14 | 2201、3510 |
| WaitPreTaskEnd | SuperKernel子核函数（Kernel）中调用 | 11、12、13、14 | AIC：11、12、13；AIV：12、13、14 | 2201、3510 |
| Matmul | 定义N个Matmul对象 | `[0, 2N-1]` | AIV/AIC按对象与subBlock分配 | 2201 |
| Matmul | SuperKernel融合场景（GetTaskRatio返回2，即1:2配比） | 15 | AIV：15 | 2201 |
```

```
#### SyncAll

- **isAIVOnly=true**：纯Vector算子的全核同步，占用模式0、1、2的flagId为14（AIV）。
- **isAIVOnly=false**：Mix（包含Cube和Vector计算）算子的全核同步，占用情况随NPU架构与核函数（Kernel）类型不同：
    - **2201架构**：占用模式0、1、2的flagId为11、12、13。
    - **3510架构**：
        - 核函数（Kernel）使用`__mix__(1, 1)`修饰时，占用模式0、1、2的flagId为11、12、13。
        - 核函数（Kernel）使用`__mix__(1, 2)`修饰时，占用模式0、1、2的flagId为11，以及模式4的flagId为12、13、28、29。
```

```
#### Matmul

- 占用flagId的范围与定义的**Matmul对象数目N**相关（N最大为4），并且随NPU架构不同而不同，参见REGIST_MATMUL_OBJ。
- **2201架构**：占用模式0、1、2的flagId为[0, 2N-1]（N=4时占用`[0,7]`）；SuperKernel融合（1:2配比，即GetTaskRatio返回2）场景下还占用flagId为15。
```

And the consequence of a collision, stated by the vendor:

```
部分接口的内部实现也会占用一部分flagId。当开发者同时使用这些接口与CrossCoreSetFlag/CrossCoreWaitFlag时，若flagId发生冲突（即相互独立的同步操作复用了同一个flagId），其对应的计数器会被错误地共享和修改，导致核间同步行为异常：可能出现阻塞无法解除而使程序卡死，或同步提前完成、阻塞提前解除而导致计算结果错误。
```

Plus the rule for reusing one flagId across modes:

```
- 核间同步的模式为模式0、1、2时，同一个flagId用于不同核间同步模式的约束：
    - 同一核上，若同一个flagId需用于不同核间同步模式，须在模式切换前完成前一个模式的所有同步操作——即确保该flagId关联的所有CrossCoreSetFlag与配套CrossCoreWaitFlag调用均已执行完毕。
    - 对于不同的核，可以直接将同一flagId用于不同的核间同步模式，具体包括以下2种场景：
        - 多个AI Core之间，使用flagId=0同步所有的AIC（模式0）；单个AI Core内，使用flagId=0同步所有AIV（模式1）。
        - 单个AI Core内，使用flagId=0同步所有AIV（模式1）；另一个AI Core内，使用flagId=0同步AIC与所有AIV（模式2）。
```

### 47f. The matching implementation, from `impl/basic_api/dav_c220/kernel_operator_sync_impl.h`

```cpp
constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

__aicore__ inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}
```

```cpp
template <bool isAIVOnly = true>
__aicore__ inline void SyncAllImpl()
{
    ...
    PipeBarrier<PIPE_ALL>();
    if constexpr (isAIVOnly) {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
        return;
    }
    // trans software sync to hardware sync
    if ASCEND_IS_AIC {
        wait_flag_dev(SYNC_AIV_FLAG);
        ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
        wait_flag_dev(SYNC_AIC_FLAG);
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
        return;
    }
    if ASCEND_IS_AIV {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
        wait_flag_dev(SYNC_AIC_AIV_FLAG);
        return;
    }
}
```

The super-kernel task-handoff routine, showing that BOTH directions use mode `0x02` and the direction
is carried by the flagId, with the vendor's own `// SET: C--->V` / `// SET: V--->C` comments:

```cpp
__aicore__ inline void WaitPreTaskEndV3Impl(uint32_t earlyStartConfig)
{
    if ASCEND_IS_AIC {
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIC_TO_AIC_WAIT) {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG); // WAIT: C--->C
        }
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIC_TO_AIV_SET) {
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG)); // SET: C--->V
        }
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIV_TO_AIC_WAIT) {
            wait_flag_dev(AscendC::SYNC_AIV_FLAG); // WAIT: V--->C
        }
    }
    if ASCEND_IS_AIV {
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIV_TO_AIV_WAIT) {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL); // WAIT: V--->V
        }
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIV_TO_AIC_SET) {
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG)); // SET: V--->C
        }
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIC_TO_AIV_WAIT) {
            wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG); // WAIT: C--->V
        }
    }
}

template <pipe_t AIV_PIPE, pipe_t AIC_PIPE>
__aicore__ inline void SetNextTaskStartV3Impl(uint32_t earlyStartConfig)
{
    if ASCEND_IS_AIC {
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIC_TO_AIC_SET) {
            ffts_cross_core_sync(AIC_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG)); // SET:C--->C
        }
    }
    if ASCEND_IS_AIV {
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIV_TO_AIV_SET) {
            ffts_cross_core_sync(AIV_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL)); // SET: V--->V
        }
    }
}
```

## Relevance — FOUR CONTRADICTIONS with our measurements

1. "`sync mode 2 is cube->vector; mode 3 is the only reverse (vector->cube) mode that works`" is wrong
   on both halves. Mode 2 is **bidirectional** by the vendor's definition and by `SyncAllImpl`'s own
   use: AIV does `GetffstMsg(0x02, SYNC_AIV_FLAG)` for V->C and AIC does `GetffstMsg(0x02,
   SYNC_AIC_AIV_FLAG)` for C->V — same mode, different flagId. **Mode 3 is not a documented mode on
   A3 at all** (A3 supports 0, 1, 2; mode 4 is 950/3510 only). `GetffstMsg` masks `mode & 0x3`, so
   passing 3 produces a bit pattern with no specified behaviour. Direction is a property of the
   *flagId's* hardwired routing, not of the mode.

2. "`SyncAll consumes 11-13`" is only true for `SyncAll<false>`. **`SyncAll<true>` (isAIVOnly, the
   template default) occupies flagId 14 and nothing else.** If our barrier is vector-only we get
   11, 12, 13 back — three flags recovered for free.

3. Mode 2 is **intra-AI-Core only** ("AI Core内部") — it cannot be a grid barrier. Only mode 0 crosses
   AI Cores. A MIX grid barrier therefore necessarily costs three flags (mode-0 AIC leg, plus two
   mode-2 bridge legs), which is exactly why `SyncAll<false>` is 11/12/13 and not one flag.

4. "`KERNEL_TYPE_AIC_ONLY is rejected at submission with sqe_type=3, errType=0x20`" now has a cause
   and a documented fix: AIC_ONLY/AIV_ONLY do not start the scheduler module, so cross-core sync is
   impossible; the vendor's prescription for a pure-cube kernel that still needs mode-0 sync is
   **`KERNEL_TYPE_MIX_AIC_1_0`** / `__mix__(1, 0)`.

Two more constraints we must design around: each flagId counter saturates at 15 un-consumed sets and
then **raises a hardware error and aborts** (a layer-looping megakernel must pair every set with a
wait within 15 iterations), and consecutive `CrossCoreSetFlag` calls from one core have **no ordering
guarantee**. Also: Matmul high-level API burns `[0, 2N-1]` for N Matmul objects on 2201, and flagId 15
when `GetTaskRatio()` returns 2 — so with 4 Matmul objects, flags 0-7 and 15 are gone, leaving only
8, 9, 10 plus whatever `SyncAll` does not take.

## 42. Symmetric / IPC memory on Ascend — exact API signatures and LCCL's actual bring-up path

URL: https://github.com/leideng/CANN-8.1.RC1/blob/master/Ascend/ascend-toolkit/8.1.RC1/aarch64-linux/include/acl/acl_rt.h
URL: https://github.com/egretwAlker/cpl-ascend-transformer-boost/blob/master/src/kernels/lcal/src/lcal_comm.cpp
Author: Huawei Technologies Co., Ltd. (CANN 8.1.RC1 headers; ATB lcal)
Date: CANN 8.1.RC1 release headers; ATB Copyright 2024

### 48a. The ACL shareable-handle family, verbatim with doc comments

```c
/**
 * @ingroup AscendCL
 * @brief share the handle that created by the process itself to other process
 * @param [in] handle   mem handle created by aclrtMallocPhysical
 * @param [in] handleType  reserved param, must be MEM_HANDLE_TYPE_NONE
 * @param [in] flags  reserved param, must be 0
 * @param [out] shareableHandle  shareable Handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemExportToShareableHandle(aclrtDrvMemHandle handle,
                                                             aclrtMemHandleType handleType, uint64_t flags,
                                                             uint64_t *shareableHandle);

/**
 * @ingroup AscendCL
 * @brief import a mem allocation from a shareable Handle
 * @param [in] shareableHandle  shareable Handle
 * @param [in] deviceId  used to generate the handle in the specified Device Id
 * @param [out] handle handle in the process
 */
ACL_FUNC_VISIBILITY aclError aclrtMemImportFromShareableHandle(uint64_t shareableHandle,
                                                               int32_t deviceId, aclrtDrvMemHandle *handle);

/**
 * @ingroup AscendCL
 * @brief set the process whitelist, only the process configured in the whitelist can use this shareableHandle
 */
ACL_FUNC_VISIBILITY aclError aclrtMemSetPidToShareableHandle(uint64_t shareableHandle,
                                                             int32_t *pid, size_t pidNum);

/**
 * @ingroup AscendCL
 * @brief get the mem allocation granularity by the option
 */
ACL_FUNC_VISIBILITY aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp *prop,
                                                              aclrtMemGranularityOptions option,
                                                              size_t *granularity);

/**
 * @ingroup AscendCL
 * @brief Get the pid for the current process on the physical device
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceGetBareTgid(int32_t *pid);
```

Virtual-memory primitives these pair with:

```c
ACL_FUNC_VISIBILITY aclError aclrtReserveMemAddress(void **virPtr, ...);
ACL_FUNC_VISIBILITY aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle, ...);
ACL_FUNC_VISIBILITY aclError aclrtMapMem(void *virPtr, ...);
```

Peer access:

```c
/**
 * @brief checking whether current device and peer device support the p2p feature
 *
 * @param canAccessPeer [OUT]   pointer to save the checking result
 * @param deviceId [IN]         current device id
 * @param peerDeviceId [IN]     peer device id
 *
 * @see aclrtDeviceEnablePeerAccess | aclrtDeviceDisablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId);

/**
 * @brief enable the peer device to support the p2p feature
 *
 * @param peerDeviceId [IN]   the peer device id
 * @param flags [IN]   reserved field, now it must be zero
 *
 * @see aclrtDeviceCanAccessPeer | aclrtDeviceDisablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags);

ACL_FUNC_VISIBILITY aclError aclrtDeviceDisablePeerAccess(int32_t peerDeviceId);
```

### 48b. What LCCL actually does — `rtIpc*`, not the ACL shareable handle

LCCL does not use `aclrtMemExportToShareableHandle`. It allocates one contiguous device buffer per
rank, names it with `rtIpcSetMemoryName`, exchanges the 128-byte name over a TCP socket, whitelists
peer PIDs (and on A3, peer SDIDs — the superpod path), and opens each peer's buffer with
`rtIpcOpenMemory`. Verbatim:

```cpp
int LcalComm::InitMem()
{
    constexpr int32_t bufferSizeUint = 1024 * 1024;
    int lcalBuffSize = bufferSize_ * bufferSizeUint + LCAL_FLAG_BUFF_BYTES;

    MKI_LOG(DEBUG) << "lcal buffer size " << lcalBuffSize;
    aclError ret = aclrtMalloc(
        reinterpret_cast<void **>(&peerMem_[rank_]), lcalBuffSize,
        (GetChipName() == ChipName::CHIP_310P3) ? ACL_MEM_MALLOC_HUGE_FIRST_P2P : ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "allocate device mem error " << __FILE__ << ":" << __LINE__ << " " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    MKI_LOG(DEBUG) << "peerMem[rank" << rank_ << "], allocate finished.";
    aclrtMemset(peerMem_[rank_], lcalBuffSize, 0, lcalBuffSize);
    return LCAL_SUCCESS;
}

int LcalComm::SetMemoryName(string &name)
{
    char nameModified[IPC_NAME_SIZE] = {};
    int memRank = rank_;
    constexpr int32_t bufferSizeUint = 1024 * 1024;
    int lcalBuffSize = bufferSize_ * bufferSizeUint + LCAL_FLAG_BUFF_BYTES;
    if (rtIpcSetMemoryName(peerMem_[memRank], lcalBuffSize, nameModified, IPC_NAME_SIZE) != RT_ERROR_NONE) {
        return LCAL_ERROR_INTERNAL;
    }
    name = nameModified;
    return LCAL_SUCCESS;
}

int LcalComm::SetIpcPidSdid(string &name, const uint32_t *pids, const int64_t *sdids) const
{
    for (int i = 0; i < rankSize_; ++i) {
        if (i == rank_) {
            continue;
        }

        if (physicalInfo_.chipName < ChipName::CHIP_910_9391) {
            int32_t pidInt32 = pids[i];
            int rtRet = rtSetIpcMemPid(name.c_str(), &pidInt32, HCCL_IPC_PID_ARRAY_SIZE);
            if (rtRet != RT_ERROR_NONE) {
                MKI_LOG(ERROR) << "err " << rtRet;
                return LCAL_ERROR_INTERNAL;
            }
        } else {
            int32_t pidInt32 = pids[i];
            int rtRet = rtSetIpcMemorySuperPodPid(name.c_str(), sdids[i], &pidInt32, HCCL_IPC_PID_ARRAY_SIZE);
            if (rtRet != RT_ERROR_NONE) {
                MKI_LOG(ERROR) << "err " << rtRet;
                return LCAL_ERROR_INTERNAL;
            }
        }
    }
    return LCAL_SUCCESS;
}

int LcalComm::OpenIpcMem(const char names[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE])
{
    static mutex mut;
    lock_guard<mutex> lock(mut);
    for (int i = 0; i < rankSize_; ++i) {
        if (i == rank_) {
            continue;
        }
        if (SkipUnusedChannel910B2C(rank_, i, GetChipName())) {
            continue;
        }
        int ret = rtIpcOpenMemory(reinterpret_cast<void **>(&peerMem_[i]), names[i]);
        if (ret != RT_ERROR_NONE) {
            CloseIpcMem();
            MKI_LOG(ERROR) << "rank : " << rank_ << " localRank : " << localRank_ << " peerMem: " << i <<
                " IpcOpenMemory err " << ret;
            return LCAL_ERROR_INTERNAL;
        }
    }
    ipcMemInited_ = true;
    return LCAL_SUCCESS;
}
```

Peer access is enabled per peer device, gated on the topology value from `rtGetPairDevicesInfo`:

```cpp
int LcalComm::EnablePeerAccess()
{
    physicalInfo_.chipName = GetChipName();
    for (auto &dev : devList_) {
        if (devId_ == dev) {
            continue;
        }
        if (SkipUnusedChannel910B2C(dev, devId_, GetChipName())) {
            continue;
        }

        int64_t value = 0;
        if (rtGetPairDevicesInfo(devId_, dev, 0, &value) != RT_ERROR_NONE) {
            MKI_LOG(WARN) << devId_ << " & " << dev << " pair devices info failed to get";
        } else {
            MKI_LOG(DEBUG) << devId_ << " <-----> " << dev << ", halGetPairDevicesInfo: *value = " << value;
        }

        if (value == TOPOLOGY_HCCS || value == TOPOLOGY_SIO || value == TOPOLOGY_HCCS_SW ||
            GetChipName() == ChipName::CHIP_910B2C) {
            physicalInfo_.physicalLink = PhysicalLink::HCCS;
            commArgs_.extraFlag &= ~(ExtraFlag::TOPO_PCIE);
        } else if (physicalInfo_.physicalLink == PhysicalLink::RESERVED) {
            physicalInfo_.physicalLink = PhysicalLink::PCIE;
            commArgs_.extraFlag |= ExtraFlag::TOPO_PCIE;
            if (rankSize_ > PING_PONG_SIZE) {
                MKI_LOG(ERROR) << "do not support pcie > 2 rank! rankSize_ = " << rankSize_;
                return LCAL_ERROR_INTERNAL;
            }
        }

        physicalInfo_.coreNum = GetCoreNum(physicalInfo_.chipName);

        if (physicalInfo_.chipName == ChipName::CHIP_310P3 && value == 0) {
            MKI_LOG(WARN) << "warn aclrtDeviceEnablePeerAccess is skipped! peerDeviceId = " << dev;
            continue;
        }

        aclError ret = aclrtDeviceEnablePeerAccess(dev, 0);
        if (ret != ACL_SUCCESS) {
            MKI_LOG(ERROR) << "err aclrtDeviceEnablePeerAccess failed peerDeviceId = " << dev << " ,rank = " << rank_
                           << ", value = " << value << ", flags = " << 0 << "," << __LINE__ << ": " << ret;
            return LCAL_ERROR_INTERNAL;
        }
    }
    MKI_LOG(DEBUG) << "EnablePeerAccess succeed" << rank_;
    return LCAL_SUCCESS;
}
```

### 48c. The exact buffer layout

```cpp
// src/kernels/lcal/include/lcal_types.h
constexpr int LCAL_FLAG_BUFF_BYTES = 4 * 1024 * 1024;     // 4 MB, = 2 ping-pong halves x 2 MB of flags

// src/kernels/lcal/include/comm_args.h
constexpr int LCAL_MAX_RANK_SIZE = 128; // lcal通信库最大支持的npu卡数
constexpr int RANK_SIZE_TWO = 2;  // 可用SIO的规模，以及是否需要跨卡搬运数据核的分界规模
constexpr int64_t IPC_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr int64_t IPC_DATA_OFFSET = 2 * 1024 * 1024; // 前2MB作为flag标志位，之后100MB作为数据存储
constexpr int64_t SYNC_FLAG_BIT_NUM = 10;  // cce 算子在用
constexpr int64_t MEM_DMA_UNIT_INT_NUM = 4;
constexpr int64_t EVENT_ID_MASK = 0xFFFFFFFF;
constexpr int64_t PING_PONG_SIZE = 2;
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;
constexpr int64_t SMALL_DATA_SIZE = 1 * 1024 * 1024;
constexpr int64_t UB_SINGLE_PING_PONG_ADD_SIZE_MAX = UB_SINGLE_DMA_SIZE_MAX / 2;
constexpr int UB_ALIGN_SIZE = 32;

// 2step算法中，2个aiv真正用作数据预处理
constexpr int64_t PRE_CORE_REAL_NUM = 2;
constexpr int64_t AIV_PER_AICORE = 2;
constexpr int DFX_COUNT = 50;
constexpr int64_t VADD_MAX_REPEAT = 255;
constexpr int64_t VADD_UNIT_BYTE = 256;
// vadd单位粒度是256B，vadd最大repeat次数为255，两个相乘的结果
constexpr int64_t MAX_VADD_SIZE = VADD_MAX_REPEAT * VADD_UNIT_BYTE;
constexpr int64_t BLOCK_UNIT_BYTE = 32;
constexpr int64_t VADD_UNIT_TO_BLOCK_UNIT_RATIO = VADD_UNIT_BYTE / BLOCK_UNIT_BYTE;    // 8
constexpr bool ATOMIC_ENABLE = false;
```

And `CommArgs`, the single struct `aclrtMemcpy`'d to device once at init — this is the whole
device-visible comm state, including the per-rank magic array:

```cpp
struct ExtraFlag {
    static constexpr uint32_t RDMA = 1;
    static constexpr uint32_t TOPO_910B2C = 1 << 1;
    static constexpr uint32_t TOPO_910_93 = 1 << 2;
    static constexpr uint32_t DETERMINISTIC = 1 << 3;
    static constexpr uint32_t QUANT_FP16 = 1 << 4;
    static constexpr uint32_t QUANT_FP32 = 1 << 5;
    static constexpr uint32_t TOPO_910A5 = 1 << 6;
    static constexpr uint32_t QUANT_DELAY = 1 << 7;
    static constexpr uint32_t QUANT_CURRENT = 1 << 8;
    static constexpr uint32_t TOPO_PCIE = 1 << 9;
    static constexpr uint32_t IS_GREATER_THAN_40_AIV = 1 << 16;
};

struct CommArgs {
    int rank = 0;           // attr rank_id, global rank
    int localRank = -1;
    int rankSize = 0; // global rank size
    int localRankSize = -1;  // 此参数是指fullmesh互联的卡数
    uint32_t extraFlag = 0; // 32 bit map，具体每一位的含义就在此文件正上方
    GM_ADDR peerMems[LCAL_MAX_RANK_SIZE] = {}; // 传入初始化获得的buff，所有allreduce都是同一个参数
    /**
     * @param sendCountMatrix 大小是rankSize*rankSize的一维数组
     * eg: sendCountMatrix[1] 的数值，对应二维数组的[0][1]，表示 卡0 要给 卡1 发送的数据个数
     */
    int64_t sendCountMatrix[LCAL_MAX_RANK_SIZE * LCAL_MAX_RANK_SIZE] = {}; // for all2allv
    int64_t dfx[DFX_COUNT] = {};
    GM_ADDR dumpAddr = nullptr;
    int32_t magics[LCAL_MAX_RANK_SIZE] = {0};
    uint64_t fftsVal = 0;
};
```

Address derivation on device (from `Collectives::Init`), the only place ping-pong is applied:

```cpp
        GlobalTensor<GM_ADDR> peerMemsAddrGm;
        peerMemsAddrGm.SetGlobalBuffer(&(reinterpret_cast<__gm__ CommArgs *>(commArgs))->peerMems[0],
                                        LCAL_MAX_RANK_SIZE);
        for (int i = 0; i < rankSize; ++i) {
            shareAddrs[i] = peerMemsAddrGm.GetValue(i) +
                            (magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET);
        }
```

```
layout of one rank's peerMem (one aclrtMalloc, zeroed once at init with aclrtMemset):

  ping-pong half h = magic % 2, base = peerMem + h * (100 MB + 2 MB)
    base + 0                             .. + segmentCount*8   : INNER flags, 32 B per block
    base + segmentCount*8                .. + 2*segmentCount*8  : OUTER flags, 32 B per block
    base + IPC_DATA_OFFSET (2 MB)        .. + 2 MB + 100 MB     : data
  total allocation = bufferSize_ MB + LCAL_FLAG_BUFF_BYTES (4 MB)
```

## Relevance
`aclrtMemset` once at init is the ONLY zeroing of the flag region in the entire library — after that
the magic epoch alone keeps old values from being mistaken for new ones. Also note LCCL's production
path is `rtIpcSetMemoryName`/`rtIpcOpenMemory` with `rtSetIpcMemorySuperPodPid` for A3 superpods, not
the ACL shareable-handle API; for TP16 inside one A3 superpod that is the call sequence to replicate.

## 43. `aclnnInplaceAllReduce` / ATB `LinearParallel` backend selection and the LCCL env vars

URL: https://github.com/egretwAlker/cpl-ascend-transformer-boost/blob/master/src/atb/runner/lcal_runner.cpp
URL: https://github.com/egretwAlker/cpl-ascend-transformer-boost/blob/master/src/kernels/lcal/include/lcal_api.h
Author: Huawei Technologies Co., Ltd. (CANN Open Software License Agreement Version 1.0)
Date: Copyright 2024

### 49a. THE KEY FINDING: device-side magic is turned on exactly when the launch mode is GRAPH_LAUNCH_MODE

ATB decides the flag at comm construction. `magicNumberDisabled_` is a misnomer: it is passed straight
into the `isEnableAutoMagicNum` parameter.

```cpp
LcalRunner::LcalRunner(const std::string &name, RunnerType runnerType, int32_t rank, int32_t rankSize,
                       const infer::CommMode commMode, const std::string &commDomain, Context &context)
    : Runner(name), runnerType_(runnerType), rank_(rank), rankSize_(rankSize), commMode_(commMode),
      commDomain_(commDomain)
{
    magicNumberDisabled_ = context.GetLaunchMode() == GRAPH_LAUNCH_MODE;
    ATB_LOG(INFO) << GetLogPrefix() << "LcalRunner::LcalRunner " << runnerType_ << " called, rank : "
                  << rank << "/" << rankSize << " commMode: " << commMode_
                  << " commDomain: " << commDomain_ << " magicNumberDisabled: " << magicNumberDisabled_;
    InitLcalComm();
}
```

```cpp
std::shared_ptr<Lcal::LcalComm> LcalRunner::CreateLcalComm()
{
    LcalCommPtr comm = nullptr;
    if (commMode_ == infer::CommMode::COMM_MULTI_PROCESS) {
        auto commonId = ParseCommDomain(commDomain_);
        ATB_LOG(INFO) << GetLogPrefix() << "Lccl COMM_MULTI_PROCESS commDomain is : "
            << commonId.first << ":" << commonId.second;
        if (commonId.first == -1) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Invalid commDomain: " << commonId.first;
            return std::shared_ptr<Lcal::LcalComm>();
        }
        lcalErrorCode_ = LcalCommInitRankWithCustDomainSize(commonId.first, commonId.second, rankSize_,
                                                            rank_, &comm, magicNumberDisabled_);
    } else if (commMode_ == infer::CommMode::COMM_MULTI_THREAD) {
        lcalErrorCode_ = LcalCommInitThread(rank_, rankSize_, commDomain_.c_str(), &comm);
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "Invalid commMode: " << commMode_;
        return std::shared_ptr<Lcal::LcalComm>();
    }
    ...
}
```

The chain, end to end:

```
context.GetLaunchMode() == GRAPH_LAUNCH_MODE
  -> LcalRunner::magicNumberDisabled_ = true
  -> LcalCommInitRankWithCustDomainSize(..., isEnableAutoMagicNum = true)
  -> new LcalComm(rank, rankSize, commDomain, bufferSize, isEnableAutoMagicNum)
  -> LcalComm::isEnableMix_ = true                     (lcal_comm.cpp:737)
  -> commArgs_.fftsVal = rtGetC2cCtrlAddr(...)         (lcal_comm.cpp:203-212)
  -> RegistKernel(opGroup = 1)                          -> the ENABLE_LCCL_MIX kernel binaries
  -> device-side SET_MAGIC: atomic-add on magics[rankSize-1], SyncAll<true>, read back
```

### 49b. Public C API — the full `lcal_api.h`

```cpp
#ifndef LCAL_API_H
#define LCAL_API_H

#include <string>
#include <hccl.h>
#include <comm_args.h>
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef void *LcalCommPtr;
#define LCAL_UNIQUE_ID_BYTES 128
typedef struct { char internal[LCAL_UNIQUE_ID_BYTES]; } LcalUniqueId;

int LcalGetUniqueId(LcalUniqueId *uniqueId, int commDomain);

int LcalCommInitRankLocal(int rankSize, int rank, LcalCommPtr *comm);

int LcalCommInitRank(LcalUniqueId commId, int rankSize, int rank, LcalCommPtr *comm);

int LcalCommInitRankWithCustDomainSize(int commDomain, int bufferSize, int rankSize, int rank, LcalCommPtr *comm,
                                       const bool isEnableAutoMagicNum = false);

int LcalCommInitRankWithDomain(int commDomain, int rankSize, int rank, LcalCommPtr *comm);

int LcalGetCommArgsDev(LcalCommPtr comm, GM_ADDR &commArgsPtr);

int LcalGetCommArgsHost(LcalCommPtr comm, Lcal::CommArgs *&commArgsPtr);

void LcalPrintDFX2Log(LcalCommPtr comm);

int LcalCommInit(int rank, int rankSize, LcalCommPtr *comms);

int LcalCommInitAll(uint32_t ndev, int32_t* devices, LcalCommPtr *comms);

int LcalCommInitThread(int rank, int rankSize, const char *uid, LcalCommPtr *comms);

int LcclAllReduce(void *sendBuf, void *recvBuf, int64_t count, HcclDataType dataType, HcclReduceOp op,
                  LcalCommPtr comm, aclrtStream stream);

int LcclAllGather(void *sendBuf, void *recvBuf, int64_t sendCount, HcclDataType dataType, LcalCommPtr comm,
                  aclrtStream stream);

int LcclReduceScatter(void *sendBuf, void *recvBuf, int64_t recvCount, HcclDataType dataType, HcclReduceOp op,
                      LcalCommPtr comm, aclrtStream stream);

int LcclBroadcast(void *buf, int64_t count, HcclDataType dataType, int root, LcalCommPtr comm,
                  aclrtStream stream);

int LcclCommDestroy(LcalCommPtr comm);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // LCAL_API_H
```

Note `LcalGetCommArgsDev(comm, commArgsPtr)` — this is the official way to obtain the device pointer to
`CommArgs` (containing `peerMems[]`, `magics[]`, `fftsVal`) and hand it to **your own** kernel.

### 49c. Buffer sizing and commDomain `id:size` syntax

```cpp
// src/kernels/lcal/include/lcal_types.h
constexpr int LCAL_COMM_BUFFER_SIZE = 200; // 单位MB
constexpr int LCAL_FLAG_BUFF_BYTES = 4 * 1024 * 1024;
```

```cpp
// src/kernels/lcal/src/lcal_wrap.cpp
int LcalCommInitRankWithCustDomainSize(int commDomain, int bufferSize, int rankSize, int rank, LcalCommPtr *comm,
    const bool isEnableAutoMagicNum)
{
    MKI_LOG(INFO) << "using lcal c++ api! rank : " << rank << ", rankSize : " << rankSize << ", commDomain:" <<
        commDomain << ", bufferSize:" << bufferSize << ", isEnableAutoMagicNum:" << isEnableAutoMagicNum;
    if (comm == nullptr) {
        MKI_LOG(ERROR) << "lcal comm ptr is nullptr!";
        return LCAL_ERROR_INTERNAL;
    }

    constexpr int minBufferSize = LCAL_COMM_BUFFER_SIZE;
    if (bufferSize < minBufferSize) {
        MKI_LOG(ERROR) << "lcal comm buffer size " << bufferSize << " MBytes should not be less than " <<
            minBufferSize << " MBytes!";
        return LCAL_ERROR_INTERNAL;
    }
    ...
    auto *c = new (std::nothrow) LcalComm(rank, rankSize, commDomain, bufferSize, isEnableAutoMagicNum);
```

```
200 MB minimum + 4 MB flags = 204 MB = 2 * (IPC_BUFF_MAX_SIZE 100 MB + IPC_DATA_OFFSET 2 MB)
=> the 200 MB minimum exists precisely so that the magic%2 ping-pong has two full 102 MB halves.
```

```cpp
std::pair<int32_t, int32_t> LcalRunner::ParseCommDomain(const std::string &commDomain) const
{
    constexpr int32_t defaultDomainSize = 200;
    constexpr int32_t maxDomainId = 65535;
    if (commDomain.empty()) {
        return {0, defaultDomainSize};
    }

    size_t colonPos = commDomain.find(':');
    if (colonPos != std::string::npos) {
        std::string idStr = commDomain.substr(0, colonPos);
        std::string sizeStr = commDomain.substr(colonPos + 1);
        ...
        return {id, size};
    }

    // 兼容旧格式：只传编号
    ...
}
```

### 49d. ATB backend selection: `backend == "lccl"` vs `"hccl"` is a param, not an env var

```cpp
// src/ops/ops_infer/all_reduce/all_reduce_operation.cpp
std::shared_ptr<Runner> AllReduceOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.backend == "hccl") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<AllReduceHcclRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<AllReduceHcclRunner>(param_, param_.hcclComm);
        }
    } else if (param_.backend == "lccl") {
        return std::make_shared<AllReduceLcclRunner>(param_, context);
    }
    return std::shared_ptr<Runner>();
}
```

Constraints the vendor enforces for the lccl backend:

```cpp
    if (opParam.backend != "hccl" && opParam.backend != "lccl") {
        ATB_LOG(ERROR) << "backend is " << opParam.backend << "backend must either be hccl or lccl";
    ...
    if (opParam.backend == "lccl" && GetSingleton<Config>().Is310P()) {
        ATB_LOG(ERROR) << "AllReduce lccl is not support in Atlas inference products";
    ...
    if (opParam.backend == "lccl" && opParam.allReduceType == "prod") {
        ATB_LOG(ERROR) << "allReduceType is " << opParam.allReduceType << "lccl not support prod ";
    ...
    if (param_.backend == "lccl" && inTensorDesc.dtype == ACL_INT64) {
        ATB_LOG(ERROR) << "lccl allreduce is not support int64";
```

And the `LCCL_PARALLEL` error surface:

```cpp
    if (ret == Lcal::LCAL_ERROR_PARA_CHECK_FAIL) {
        ATB_LOG(ERROR) << "ret: " << ret << " LCCL_PARALLEL should be 0 or fasle";
        return ERROR_INVALID_SINGLE_OPERATION_PARAM;
    }
```

### 49e. Env vars found in the LCCL source, exhaustively

```
LCCL_DETERMINISTIC   -> "1" or "true"  -> deterministic_ = true; extraFlag |= ExtraFlag::DETERMINISTIC
                        (lcal_comm.cpp LcalComm::InitCommon)
LCCL_PARALLEL        -> "1" or "true"  -> GetParallel() == true -> blockNum collapses to rankSize,
                        which in-kernel forces lcalBlockNum == rankSize -> the TwoShot path
                        (lccl.cpp GetParallel)
```

There is no `ATB_USE_LCCL` in this tree; backend choice is the `backend` string in
`atb::infer::AllReduceParam` / `LinearParallelParam`. MIX / device-magic is not env-selectable either —
it is `context.GetLaunchMode() == GRAPH_LAUNCH_MODE`, or `ReportTiming::ProfilingStatus() ==
PROF_TASK_TIME_DUMP`.

## Relevance
This is the most actionable thing in the whole document. The vendor's own reason for computing the
epoch on device is that the host cannot bump it when the launch is a captured graph — the kernel args
are frozen. A megakernel is the same situation taken further. So the supported, shipping recipe for
"no return to host between layers" already exists: build with `ENABLE_LCCL_MIX`, construct the comm
with `isEnableAutoMagicNum = true`, pass `fftsVal` from `rtGetC2cCtrlAddr`, and get the epoch from a
device atomic add.

## 44. The OFFICIAL in-kernel collective API (`Hccl` kernel class) — and why on A3 it is NOT one-sided

URL: https://github.com/hicann/asc-devkit/tree/master/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel
Author: Huawei / CANN `asc-devkit`
Date: current `master` of hicann/asc-devkit (2026)

### 50a. `Hccl` class template — on A3 the server is AICPU, full stop

```
template <HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU, const auto& config = DEFAULT_CFG>
class Hccl;
```

```
enum HcclServerType {
    HCCL_SERVER_TYPE_AICPU = 0,
    HCCL_SERVER_TYPE_CCU,
    HCCL_SERVER_TYPE_END // 预留参数，不支持使用
};

struct HcclServerConfig {
    CoreType type;
    int64_t blockId;
};

enum class CoreType : uint8_t { DEFAULT, ON_AIV, ON_AIC };
```

```
| serverType | 支持的服务端类型。HcclServerType枚举类型，取值如下。
HCCL_SERVER_TYPE_AICPU：AI CPU服务端。
HCCL_SERVER_TYPE_CCU：CCU服务端。
HCCL_SERVER_TYPE_END：预留参数，不支持使用。
对于Ascend 950PR&950DT系列产品，当前仅支持HCCL_SERVER_TYPE_CCU。
对于Atlas A3系列产品，当前仅支持HCCL_SERVER_TYPE_AICPU。
对于Atlas A2系列产品，当前仅支持HCCL_SERVER_TYPE_AICPU。 |
| config | 用于指定向服务端下发任务的核。HcclServerConfig类型...
type：向服务端下发任务的核的类型。CoreType类型，可取值为：
DEFAULT：表示不指定AIC核或者AIV核。
ON_AIV：表示指定为AIV核。
ON_AIC：表示指定为AIC核。
blockId：向服务端下发任务的核的ID。
默认值DEFAULT_CFG = {CoreType::DEFAULT, 0}。 |
```

```
static constexpr HcclServerConfig HCCL_CFG = {CoreType::ON_AIV, 10};
// 选择AICPU作为服务端
Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU, HCCL_CFG> hccl;

// 选择CCU作为服务端
Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU, HCCL_CFG> hccl;
```

### 50b. `AllReduce` signature and the 63-task ceiling

```
template <bool commit = false>
__aicore__ inline HcclHandle AllReduce(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint8_t repeat = 1)
```

```
| commit | 输入 | bool类型。参数取值如下：
true：在调用Prepare接口时，Commit同步通知服务端可以执行该通信任务。
false：在调用Prepare接口时，不通知服务端执行该通信任务。 |
```

```
| repeat | 输入 | 一次下发的AllReduce通信任务个数。repeat取值≥1，默认值为1。当repeat>1时，每个AllReduce任务的sendBuf和recvBuf地址由服务端自动算出，计算公式如下：

sendBuf[i] = sendBuf + count* sizeof(datatype) * i, i∈[0, repeat)

recvBuf[i] = recvBuf + count* sizeof(datatype) * i, i∈[0, repeat)

注意：当设置repeat>1时，须与count参数配合使用，规划通信数据地址。 |
```

```
## 约束说明

-   调用本接口前确保已调用过InitV2和SetCcTilingV2接口。
-   若HCCL对象的config模板参数未指定下发通信任务的核，该接口只能在AIC核或者AIV核两者之一上调用。若HCCL对象的config模板参数中指定了下发通信任务的核，则该接口可以在AIC核和AIV核上同时调用，接口内部会根据指定的核的类型，只在AIC核、AIV核二者之一下发该通信任务。
-   对于Atlas A2系列产品，一个通信域内，所有Prepare接口的总调用次数不能超过63。
-   对于Atlas A3系列产品，一个通信域内，所有Prepare接口和InterHcclGroupSync接口的总调用次数不能超过63。
-   对于Ascend 950PR&950DT系列产品，一个通信域内，所有Prepare接口的总调用次数不能超过63。
-   对于Ascend 950PR&950DT系列产品，通信服务端为CCU时，最大支持8卡的全链接，默认仅支持FullMesh算法，并且单次最大通信数据量不能超过256M。
```

### 50c. The canonical in-kernel AllReduce, verbatim from the doc

```cpp
extern "C" __global__ __aicore__ void all_reduce_custom(GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    auto sendBuf = xGM;       // xGM为AllReduce的输入GM地址
    auto recvBuf = yGM;       // yGM为AllReduce的输出GM地址
    uint64_t sendCount = 300; // 每张卡上均有300个float16的数据
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    REGISTER_TILING_DEFAULT(AllReduceCustomTilingData); // AllReduceCustomTilingData为对应算子头文件定义的结构体
    GET_TILING_DATA_WITH_STRUCT(AllReduceCustomTilingData, tilingData, tilingGM);

    Hccl hccl;
    GM_ADDR contextGM = AscendC::GetHcclContext<0>(); // AscendC自定义算子核函数（Kernel）中，通过此方式获取HCCL context

    if (AscendC::g_coreType == AIV) { // 指定AIV核通信
        hccl.InitV2(contextGM, &tilingData);
        auto ret = hccl.SetCcTilingV2(offsetof(AllReduceCustomTilingData, mc2CcTiling));
        if (ret) {
            return;
        }
        HcclHandle handleId1 =
            hccl.AllReduce<true>(sendBuf, recvBuf, sendCount, HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp);
        hccl.Wait(handleId1);
        AscendC::SyncAll<true>(); // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死
        hccl.Finalize();
    }
}
```

The multi-round split variant, showing `repeat` and the matching `Wait` count:

```cpp
extern "C" __global__ __aicore__ void all_reduce_custom(GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    constexpr uint32_t tileNum = 2U;   // 首块数量
    constexpr uint64_t tileLen = 128U; // 首块数据个数
    constexpr uint32_t tailNum = 1U;   // 尾块数量
    constexpr uint64_t tailLen = 44U;  // 尾块数据个数
    auto sendBuf = xGM;                // xGM为AllReduce的输入GM地址
    auto recvBuf = yGM;                // yGM为AllReduce的输出GM地址
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    REGISTER_TILING_DEFAULT(AllReduceCustomTilingData);
    GET_TILING_DATA_WITH_STRUCT(AllReduceCustomTilingData, tilingData, tilingGM);

    Hccl hccl;
    GM_ADDR contextGM = AscendC::GetHcclContext<0>();
    if (AscendC::g_coreType == AIV) {                 // 指定AIV核通信
        hccl.InitV2(contextGM, &tilingData);
        auto ret = hccl.SetCcTilingV2(offsetof(AllReduceCustomTilingData, mc2CcTiling));
        if (ret != HCCL_SUCCESS) {
            return;
        }
        // 2个首块处理
        constexpr uint32_t tileRepeat = tileNum;
        // 除了sendBuf和recvBuf入参不同，对2个首块处理的其余参数相同。故使用repeat=2，第2个首块AllReduce任务的sendBuf、recvBuf将由API内部自行更新
        HcclHandle handleId1 =
            hccl.AllReduce<true>(sendBuf, recvBuf, tileLen, HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp, tileRepeat);
        // 1个尾块处理
        constexpr uint32_t kSizeOfFloat16 = 2U;
        sendBuf += tileLen * tileNum * kSizeOfFloat16;
        recvBuf += tileLen * tileNum * kSizeOfFloat16;
        constexpr uint32_t tailRepeat = tailNum;
        HcclHandle handleId2 =
            hccl.AllReduce<true>(sendBuf, recvBuf, tailLen, HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp, tailRepeat);

        for (uint8_t i = 0; i < tileRepeat; i++) {
            hccl.Wait(handleId1);
        }
        hccl.Wait(handleId2);
        AscendC::SyncAll<true>(); // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死
        hccl.Finalize();
    }
}
```

### 50d. `Commit` / `Wait` — exact signatures and pairing rules

```
__aicore__ inline void Commit(HcclHandle handleId)
```
```
__aicore__ inline int32_t Wait(HcclHandle handleId)
```
```
| handleId | 输入 | 对应通信任务的标识ID，只能使用Prepare原语接口的返回值。
using HcclHandle = int8_t; |
```

```
## 约束说明   (Commit)
-   调用本接口前确保已调用过InitV2和SetCcTilingV2接口。
-   入参handleId只能使用Prepare原语对应接口的返回值。
-   非细粒度通信时，本接口的调用次数应该与Prepare的repeat次数一致。细粒度通信时，本接口的调用次数应该与通信任务的总步骤数/步长*Prepare的repeat次数一致。
-   本接口在AIC核或者AIV核上调用必须与对应的Prepare接口的调用核保持一致。
```

```
## 功能说明   (Wait)
阻塞AI Core，非细粒度通信时，等待handleId对应的通信任务完成，细粒度通信时，等待handleId对应的步长长度的子通信任务完成。handleId调用Wait接口的顺序，须和Prepare接口一致。该接口默认在所有核上工作，用户也可以在调用前通过GetBlockIdx接口指定其在某一个核上运行。
```

### 50e. `GetWindowsInAddr` — the supported way to get a peer's symmetric window from inside a kernel

```
__aicore__ inline GM_ADDR GetWindowsInAddr(uint32_t rankId)
```

```
## 功能说明

获取卡间通信数据WindowsIn起始地址，可用来直接作为计算的输入输出地址，减少拷贝。该接口默认在所有核上工作，用户也可以在调用前通过GetBlockIdx接口指定其在某一个核上运行。
```

```
| rankId | 输入 | 待查询的卡的Id。 |
```
```
返回对应卡的卡间通信数据WindowsIn起始地址。当rankId非法时，返回nullptr。
```

```cpp
REGISTER_TILING_DEFAULT(ReduceScatterCustomTilingData);
GET_TILING_DATA_WITH_STRUCT(ReduceScatterCustomTilingData, tilingData, tilingGM);
Hccl hccl;
GM_ADDR contextGM = AscendC::GetHcclContext<0>();
hccl.InitV2(contextGM, &tilingData);

auto winInAddr = hccl.GetWindowsInAddr(0);
auto winOutAddr = hccl.GetWindowsOutAddr(0);
auto rankId = hccl.GetRankId();
auto rankDim = hccl.GetRankDim(); // 4张卡
```

## Relevance
This is the official in-kernel collective and it is NOT what we want on A3: `serverType` is forced to
`HCCL_SERVER_TYPE_AICPU` there (CCU is 950-only), so every `AllReduce` is a message to an AICPU server
plus a `Wait` spin — a host-free but not core-free round trip, with a hard ceiling of 63 Prepare calls
per comm domain. The genuinely useful parts for us are `GetWindowsInAddr(rankId)` /
`GetWindowsOutAddr(rankId)`, which hand us the peer symmetric-memory base inside the kernel with no
IPC bring-up of our own, and `GetHcclContext<0>()`. Also note every official example ends with
`AscendC::SyncAll<true>()` before `Finalize()` — the vendor's own stated reason is that core 0 racing
ahead to `Finalize()` makes the other cores' `Wait` hang.

## 45. `ipc_queue.h` + `allreduce_big_data.h` — the low 32 bits as a MONOTONE STAGE COUNTER within one epoch (how to do many stages with ONE flag slot)

URL: https://github.com/egretwAlker/cpl-ascend-transformer-boost/blob/master/src/kernels/lcal/src/ascendc_kernels/ipc_queue.h
URL: https://github.com/egretwAlker/cpl-ascend-transformer-boost/blob/master/src/kernels/lcal/src/ascendc_kernels/allreduce_big_data.h
Author: Huawei Technologies Co., Ltd. (CANN Open Software License Agreement Version 1.0)
Date: Copyright 2024

This is how LCCL pipelines without allocating a flag per stage: the high 32 bits stay at `magic`, and
the low 32 bits hold a loop counter that only increases. `WaitOneRankPartFlag`'s `v < checkValue`
clause makes `WaitInnerFlag(magic, count, ...)` mean "peer's counter has reached at least `count`".
`EVENT_ID_MASK = 0xFFFFFFFF` is how the counter is read back out. Full `ipc_queue.h`:

```cpp
#ifndef LCCL_IPC_QUEUE_H
#define LCCL_IPC_QUEUE_H
#include "sync_collectives.h"
using namespace AscendC;

template <typename T>
class IpcQueue {
public:
    FORCE_INLINE_AICORE IpcQueue() {}

    FORCE_INLINE_AICORE void Init(SyncCollectives *sync, int64_t magic, GM_ADDR workSpace, uint64_t bufferNum,
                                   uint64_t blockNum)
    {
        this->sync = sync;
        this->magic = magic;
        depth = bufferNum / blockNum;
        front = 0;
        rear = 0;
        count = 0;
        this->blockNum = blockNum;
        buff.SetGlobalBuffer((__gm__ T*)workSpace, bufferNum);
        blockIdx = GetBlockIdx();
    }

    FORCE_INLINE_AICORE bool Full()
    {
        if ((rear + 1) % depth == front) {
            return true;
        }
        return false;
    }

    FORCE_INLINE_AICORE GlobalTensor<T> EnQue()
    {
        uint64_t rearOld = rear;
        rear = (rear + 1) % depth;
        return buff[rearOld * blockNum];
    }

    FORCE_INLINE_AICORE void DeQue(int checkRank, int checkBlock = -1)
    {
        if (!Full()) {
            return;
        }
        if (checkBlock == -1) {
            checkBlock = blockIdx;
        }
        sync->WaitInnerFlag(magic, count, checkRank, checkBlock);
        PipeBarrier<PIPE_ALL>();
        int64_t val = sync->GetInnerFlag(checkRank, checkBlock) & EVENT_ID_MASK;
        count = val + 1;
        front = (val + 1) % depth;
    }

    FORCE_INLINE_AICORE void DeQue(int *rankList, int checkCount, int checkBlock = -1)
    {
        if (!Full()) {
            return;
        }
        if (checkBlock == -1) {
            checkBlock = blockIdx;
        }
        int64_t minIndex = LLONG_MAX;
        for (int i = 0; i < checkCount; i++) {
            sync->WaitInnerFlag(magic, count, rankList[i], checkBlock);
            PipeBarrier<PIPE_ALL>();

            int64_t val = sync->GetInnerFlag(rankList[i], checkBlock) & EVENT_ID_MASK;
            if (minIndex > val) {
                minIndex = val;
            }
        }
        count = minIndex + 1;
        front = (minIndex + 1) % depth;
    }
    FORCE_INLINE_AICORE void DeQue(int *rankList, int *blockIdxList, int checkCount)
    {
        if (!Full()) {
            return;
        }

        int64_t minIndex = LLONG_MAX;
        for (int i = 0; i < checkCount; i++) {
            sync->WaitInnerFlag(magic, count, rankList[i], blockIdxList[i]);
            PipeBarrier<PIPE_ALL>();

            int64_t val = sync->GetInnerFlag(rankList[i], blockIdxList[i]) & EVENT_ID_MASK;
            if (minIndex > val) {
                minIndex = val;
            }
        }
        count = minIndex + 1;
        front = (minIndex + 1) % depth;
    }

    FORCE_INLINE_AICORE GlobalTensor<T> ReadFront()
    {
        uint64_t frontOld = front;
        front = (front + 1) % depth;
        return buff[frontOld * blockNum];
    }

private:
    int64_t magic;
    uint64_t depth;
    uint64_t front;
    uint64_t rear;
    uint64_t count;
    uint64_t blockNum;
    GlobalTensor<T> buff;
    SyncCollectives *sync;
    int blockIdx;
};
#endif // LCCL_IPC_QUEUE_H
```

The producer / consumer split: blocks `[0, rankSize)` produce, blocks `[rankSize, 2*rankSize)` consume.
Note the flag value is `count`, the loop index — one flag slot, unbounded stages:

```cpp
    FORCE_INLINE_AICORE void Producer()
    {
        int64_t loopCount = CeilDiv(curRankDataNum, curBlockNum);
        int64_t remain = curRankDataNum;
        int count = 0;
        while (count < loopCount) {
            inputQue.DeQue(rankList, coreIdxList, rankSize);
            GlobalTensor<T> outputGm = inputQue.EnQue();
            int64_t copyNum = (remain < curBlockNum) ? remain : curBlockNum;
            if constexpr (std::is_same_v<T, U>) {
                Collectives::CpGM2GMPingPong(copyNum * sizeof(T), inputGt[count * curBlockNum], outputGm, COPYONLY);
            } else {
                ...
            }
            sync.SetInnerFlag(magic, count);

            remain = remain - curBlockNum;
            count = count + 1;
        }
    }

    FORCE_INLINE_AICORE void Consumer()
    {
        int64_t atomLoopCount = CeilDiv(pullRankDataNum, curBlockNum);
        int64_t atomRemain = pullRankDataNum;
        int64_t loopCount = CeilDiv(curRankDataNum, curBlockNum);
        int64_t remain = curRankDataNum;
        int count = 0;
        while (count < loopCount || count < atomLoopCount) {
            if (peerRank != rank && count != atomLoopCount) {
                sync.WaitInnerFlag(magic, count, rank, rank);
                sync.WaitInnerFlag(magic, count, peerRank, rank);

                GlobalTensor<T> inputGm = srcQue.ReadFront();
                GlobalTensor<T> outputGm = dstQue.EnQue();

                int64_t atomCopyNum = (atomRemain < curBlockNum) ? atomRemain : curBlockNum;
                if constexpr (std::is_same_v<T, U>) {
                    Collectives::CpGM2GMPingPong(atomCopyNum * sizeof(T), inputGm, outputGm, atomOp);
                } else {
                    GlobalTensor<U> inputGmTmp;
                    inputGmTmp.SetGlobalBuffer((__gm__ U*)inputGm.GetPhyAddr());
                    CpGM2GMWithScale(atomCopyNum, inputGmTmp, outputGm, atomOp);
                    }
                    atomRemain = atomRemain - curBlockNum;
                }
                sync.SetOuterFlag(magic, count);
                if (count == loopCount) {
                    break;
                }
                sync.WaitOneRankPartOuterFlag(magic, count, peerRank, rankSize, rankSize);
                if (!(extraFlag & ExtraFlag::RDMA)) {
                    GlobalTensor<T> pullGm = pullQue.ReadFront();
                    int64_t copyNum = (remain < curBlockNum) ? remain : curBlockNum;
                    Collectives::CpGM2GMPingPong(copyNum * sizeof(T), pullGm, outputGt[count * curBlockNum], COPYONLY);
                }

                sync.SetInnerFlag(magic, count);
                remain = remain - curBlockNum;
                count = count + 1;
        }
    }
```

Queue sizing arithmetic from `AllReduceBigData::Init`, exact:

```cpp
        constexpr static int QUEUE_DEPTH = 4;
        ...
        if (blockIdx >= PING_PONG_SIZE * rankSize) { return; }       // 2 * rankSize blocks used
        perStepBlockNum = rankSize;
        int globalRankSize = localArgs->rankSize <= 0 ? rankSize : localArgs->rankSize;
        int localRankSize = localArgs->localRankSize <= 0 ? rankSize : localArgs->localRankSize;
        int serverNum = globalRankSize / localRankSize;
        int64_t ipcBuffMaxSizeAligned = IPC_BUFF_MAX_SIZE / (globalRankSize + serverNum - 1) /
            QUEUE_DEPTH / sizeof(T) /scaleNum * scaleNum * QUEUE_DEPTH * sizeof(T) * globalRankSize;
        curBlockSize = ipcBuffMaxSizeAligned / localRankSize / QUEUE_DEPTH;
        curBlockNum = curBlockSize / sizeof(T);
        int64_t perQueSize = ipcBuffMaxSizeAligned / localRankSize;
        int64_t perQueNum = perQueSize / sizeof(T);

        for (int i = 0; i < rankSize; ++i) {
            rankList[i] = i;
            coreIdxList[i] = rankSize + blockIdx % perStepBlockNum;
        }
        peerRank = blockIdx % perStepBlockNum;

        if (blockIdx / perStepBlockNum == 0) {
            inputQue.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + inputIpcGtOffsetNum,
                perQueNum, curBlockNum);
        } else {
            srcQue.Init(&sync, magic, shareAddrs[peerRank] + IPC_DATA_OFFSET + rank * perQueSize,
                perQueNum, curBlockNum);
            dstQue.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + rank * perQueSize,
                perQueNum, curBlockNum);
            pullQue.Init(&sync, magic, shareAddrs[peerRank] + IPC_DATA_OFFSET + peerRank * perQueSize,
                perQueNum, curBlockNum);
        }
```

## Relevance
This is the answer to "how do I sync N layers inside one megakernel with 16 flag slots": you don't
need N flags, and you don't need N epochs. Keep `magic` fixed and use the low 32 bits as the layer
index — `SetInnerFlag(magic, layer)` then `WaitInnerFlag(magic, layer, peer, block)`. The `v >=
checkValue` comparison makes a peer that has run ahead still satisfy an earlier wait, which is exactly
the property a pipelined layer loop needs. `QUEUE_DEPTH = 4` is the vendor's chosen in-flight depth.

## 46. `moe_distribute_base.h` / `notify_dispatch_a2.h` (sgl-kernel-npu, the Ascend DeepEP analogue) — raw-pointer flag spins and the `HcclOpResParam` window structs

URL: https://github.com/sgl-project/sgl-kernel-npu/blob/main/csrc/deepep/ops2/op_kernel/notify_dispatch_a2.h
URL: https://github.com/sgl-project/sgl-kernel-npu/blob/main/csrc/deepep/ops2/op_kernel/moe_distribute_base.h
Author: SGLang / sgl-kernel-npu contributors (vendoring Huawei HCCL device structs)
Date: present in sgl-kernel-npu main (2025-2026)

### 52a. A PER-BLOCK epoch counter kept in the HCCL window — no grid barrier needed to distribute it

LCCL's MIX `SET_MAGIC` needs `SyncAll<true>()` because one block does the atomic and all blocks read
it. `notify_dispatch_a2` avoids that entirely: **each block owns its own magic slot**, read-modify-
writes it, and uses it. Since block *i* on rank A only ever synchronises with block *i* on rank B,
per-block epochs stay consistent without any cross-core barrier. Verbatim:

```cpp
        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();

        hccl_.InitV2(contextGM0, &tilingData);
        hccl_.SetCcTilingV2(offsetof(NotifyDispatchA2TilingData, mc2CcTiling1));
        this->winContext_[COMM_EP_IDX] = (__gm__ HcclOpResParam *)contextGM0;
        notifyMemoryOffset = winContext_[COMM_EP_IDX]->winSize - IPC_BUFF_MAX_SIZE * 2;
        // 设置并自增magic
        magicTensor_.SetGlobalBuffer((__gm__ uint64_t *)(hccl_.GetWindowsInAddr(rank) + IPC_DATA_OFFSET -
                                                         blockNum * sizeof(uint64_t) * EXP_TOKEN_COUNT_FLAG_CNT +
                                                         notifyMemoryOffset));

        pipe.InitBuffer(this->tBuf, TEMP_BUF_LEN);
        LocalTensor<uint64_t> tempLocal = tBuf.Get<uint64_t>();
        PipeBarrier<PIPE_ALL>();
        tempLocal(0) = magicTensor_.GetValue(blockIdx * EXP_TOKEN_COUNT_FLAG_CNT) + 1;
        // 使用atomic方式实现+1
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);  // 等待SetValue完成
        DataCopy(magicTensor_[blockIdx * EXP_TOKEN_COUNT_FLAG_CNT], tempLocal, EXP_TOKEN_COUNT_FLAG_CNT);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);  // 等待DataCopy完成
        magic = magicTensor_.GetValue(blockIdx * EXP_TOKEN_COUNT_FLAG_CNT);
        PipeBarrier<PIPE_ALL>();
        // 初始化目标rank的shareAddrs
        for (int i = 0; i < rankSize; i++) {
            this->shareAddrs[i] =
                hccl_.GetWindowsInAddr(i) + notifyMemoryOffset + (magic % PING_PONG_SIZE) * IPC_BUFF_MAX_SIZE;
        }
```

```cpp
        windowInGM_ = this->shareAddrs[rank];
        windowOutGM_ =
            hccl_.GetWindowsOutAddr(rank) + (magic % PING_PONG_SIZE) * IPC_BUFF_MAX_SIZE + notifyMemoryOffset;
```

```
magic region is carved immediately BELOW the data region of the window:
  magicTensor_ base = GetWindowsInAddr(rank) + notifyMemoryOffset + IPC_DATA_OFFSET
                      - blockNum * sizeof(uint64_t) * EXP_TOKEN_COUNT_FLAG_CNT
  slot for this block = magicTensor_[blockIdx * EXP_TOKEN_COUNT_FLAG_CNT]
  notifyMemoryOffset  = winContext_->winSize - IPC_BUFF_MAX_SIZE * 2   // last 2 IPC buffers of the window
  ping-pong stride    = IPC_BUFF_MAX_SIZE, selected by magic % PING_PONG_SIZE
```

### 52b. A DIFFERENT magic merge — `magic * 2 + value`, not a 32-bit shift

```cpp
    __aicore__ inline uint64_t MergeMagicWithValue(uint64_t magic, uint64_t value)
    {
        // magic as the high part, eventID as the low part, combined into a value for comparison
        return (magic * 2ULL + value);
    }
```

The spin predicate is textually identical to LCCL's, so with this merge the `MAGIC_MASK` equality
degenerates and the `v < checkValue` monotonicity does the real work:

```cpp
    // Wait for a part of synchronization flags within a rank
    __aicore__ inline void WaitOneRankPartFlag(__gm__ uint64_t *waitAddr, int64_t flagNum, uint64_t checkValue)
    {
        GlobalTensor<uint64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<uint64_t> localWait = tBuf.GetWithOffset<uint64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;
        uint64_t checkedFlagNum = 0;
        do {
            // Copy global synchronization flags to local
            DataCopy(localWait, globalWait[checkedFlagNum * FLAG_UNIT_INT_NUM],
                     (flagNum - checkedFlagNum) * FLAG_UNIT_INT_NUM);
            SetWaitEvent<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB

            // Check if the synchronization flags are equal to checkValue
            isSync = true;
            uint64_t remainToCheck = flagNum - checkedFlagNum;
            for (auto i = 0; i < remainToCheck; ++i) {
                // Continue waiting if any core has not reached the checkValue phase
                uint64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
                if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                    isSync = false;
                    checkedFlagNum += i;
                    break;
                }
            }
        } while (!isSync);
    }

    __aicore__ inline void SetInnerFlag(uint64_t magic, uint64_t eventID, int64_t setRank, int64_t setBlock)
    {
        uint64_t value = MergeMagicWithValue(magic, eventID);
        // SetFlag((__gm__ uint64_t *)(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM, value);
        __gm__ uint64_t *setAddr = (__gm__ uint64_t *)(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM;

        SetWaitEvent<HardEvent::MTE3_S>(EVENT_ID0);
        SetWaitEvent<HardEvent::MTE2_S>(EVENT_ID0);
        GlobalTensor<uint64_t> globalSet;
        globalSet.SetGlobalBuffer(setAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<uint64_t> localSet = tBuf.GetWithOffset<uint64_t>(1, 0);
        localSet.SetValue(0, value);

        // Copy global synchronization flag to local
        SetWaitEvent<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(globalSet, localSet, FLAG_UNIT_INT_NUM);
        SetWaitEvent<HardEvent::MTE3_S>(EVENT_ID0);
    }

    // Wait for a single inner-card synchronization flag
    __aicore__ inline void WaitInnerFlag(uint64_t magic, uint64_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        uint64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankPartFlag((__gm__ uint64_t *)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM, 1, value);
    }
```

Remote flag write to every peer in the server, with the exact offset arithmetic (and the vendor's own
worked example in the comments):

```cpp
        for (int i = 0; i < localRankSize; ++i) {
            int32_t curServerRankId = serverId * localRankSize + i;
            for (int j = 0; j < serverNum; ++j) {
                // 给当前server每卡写入serverNum个标记，位置为 rank + j * localRankSize
                int32_t offset = rank + j * rankSize;  // rank0: 0,16 / rank8: 8,24
                // rank0,server0: 0-7,16-23  rank8,server1: 8-15,24-31
                SetInnerFlag(magic, 1, curServerRankId, offset);
            }
        }
```

```cpp
            // server0: 0-7,16-23   server1: 8-15,24-31
            int32_t offset = (i / serverNum + serverId * localRankSize) + (i % serverNum) * rankSize;
            WaitInnerFlag(magic, 1, rank, offset);

            remoteGt.SetGlobalBuffer((__gm__ T *)(shareAddrs[targetRankId] + IPC_DATA_OFFSET +
                                                  serverTarRankId * queSize +
                                                  STATUS_ENTRY_SIZE));  // 该rank上的第server块
            CpGM2GMPingPong<T>(recvCount * sizeof(T), remoteGt, recvDataOutputGt[serverTarRankId * this->len],
                               COPYONLY);
```

### 52c. `HcclOpResParam` — the exact device-visible HCCL context layout (the symmetric-memory descriptor)

```cpp
struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HcclOpResParam {
    // local resource
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId;  // usrrankid
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    // aicore detect remote window
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
    uint32_t rWinOffset;
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;

    // config parameters
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM];

    // communicate retry
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;
    uint64_t tinyMem;  // for all2all
    uint64_t tinyMemSize;
    // zero-copy
    uint64_t zeroCopyHeadPtr;
    uint64_t zeroCopyTailPtr;
    uint64_t zeroCopyRingBuffer;
    uint64_t zeroCopyIpcPtrs[16];
    uint32_t zeroCopyDevicePhyId[16];

    bool utraceStatusFlag;
};

constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19;
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2;
constexpr uint32_t AICPU_MAX_RANK_NUM = 128 * 1024;
```

A range `dcci` helper (cache-line-granular write-through over a byte range), and the dataplane-mode
probe which shows an **AIV dataplane** exists alongside HOST and AICPU:

```cpp
__aicore__ inline void cacheWriteThrough(__gm__ uint8_t *sourceAddr, uint64_t length)
{
    __gm__ uint8_t *start =
        (__gm__ uint8_t *)((uint64_t)sourceAddr / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    __gm__ uint8_t *end =
        (__gm__ uint8_t *)(((uint64_t)sourceAddr + length) / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint32_t i = 0; i <= end - start; i += AscendC::CACHE_LINE_SIZE) {
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(global[i]);
    }
}

__aicore__ inline DataplaneMode GetDataplaneMode(GM_ADDR contextGM0)
{
    __gm__ HcclA2CombineOpParam *winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM0;
    CombinedCapability *capability = winContext_->capability;
    uint64_t capabilitySize = winContext_->capabilitySize;
    DataplaneMode dataplaneMode = DataplaneMode::HOST;
    if (capability == 0) {
        dataplaneMode = DataplaneMode::AICPU;
        return dataplaneMode;
    }
    uint64_t dataplaneModeBitmap = capability->dataplaneModeBitmap;
    if ((dataplaneModeBitmap & 0x02) == 0x02) {
        dataplaneMode = DataplaneMode::AICPU;
    }
    if ((dataplaneModeBitmap & 0x04) == 0x04) {
        dataplaneMode = DataplaneMode::AIV;
    }
    return dataplaneMode;
}
```

Also note the top-level `Process()` is AIV-gated and uses `SyncAll<true>()` (the one-flag, flagId-14
variant) between phases, not `SyncAll<false>`:

```cpp
    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            // 第一阶段，处理server间通信
            if (serverNum > 1) {
                ProcessBetweenServer();
            }

            // 第二阶段，处理server内通信
            ProcessWithinServer();
            SyncAll<true>();

            // 交换后的数据拆分和计算输出
            SplitAndCalcData();
            SyncAll<true>();

            hccl_.Finalize();
        }
    }
```

## Relevance
The per-block magic slot is the single cleanest answer to our constraint of no grid barrier: give each
block its own epoch counter in the shared window, read-modify-write it at entry, and the epochs of
block *i* across all ranks advance in lockstep because block *i* only talks to block *i*. No atomic,
no `SyncAll`, no host. `HcclOpResParam.remoteRes[i].nextDevicePtr -> HcclRankRelationResV2.windowsIn`
is the exact pointer chase for a peer's symmetric window, and `winSize - 2*IPC_BUFF_MAX_SIZE` is where
these kernels put their flag/ping-pong region inside it.

## 47. `catlass/arch/cross_core_sync.hpp` — the canonical fix for the 15-set counter saturation, and Catlass's own flagId allocation

URL: https://github.com/hicann/catlass/blob/master/include/catlass/arch/cross_core_sync.hpp
Author: Huawei Technologies Co., Ltd. (CANN Open Software License Agreement Version 2.0)
Date: Copyright 2025

Whole file verbatim. The leading comment is the authoritative statement of the saturation hazard and
the reverse-sync remedy:

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_ARCH_CROSS_CORE_SYNC_HPP
#define CATLASS_ARCH_CROSS_CORE_SYNC_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Arch {

// A certain cross core flag can be continuously set up to 15 times without waiting. In scenarios where there is only
// one-way synchronization between cores and the synchronization operation is executed multiple times, if the core
// performing the set operation runs faster, it may result in the flag being set more than 15 times consecutively,
// leading to a system freeze. To prevent such a freeze, we need to wait for the waiting cores to perform a reverse
// synchronization operation after executing the set operation MAX_REVERSE_DEPTH times.
constexpr uint32_t MAX_REVERSE_DEPTH = 15;

using FlagID = uint16_t;
constexpr FlagID AIV_INTER_BLOCK_BARRIER = 8;
constexpr FlagID AIC_INTER_BLOCK_BARRIER = 9;
constexpr FlagID AIV_INTER_SUBBLOCK_BARRIER = 10;
constexpr FlagID FFTS_MAX_FLAG = 7;

struct CrossCoreFlag {
    CATLASS_DEVICE
    CrossCoreFlag() : id(0)
    {}

    CATLASS_DEVICE
    CrossCoreFlag(FlagID id) : id(id)
    {}

    FlagID id;
};

template <uint32_t REVERSE_DEPTH_ = MAX_REVERSE_DEPTH>
struct CrossCoreFlagWithReverse {
    CATLASS_DEVICE
    CrossCoreFlagWithReverse() : id(0), reverseId(0)
    {}

    CATLASS_DEVICE
    CrossCoreFlagWithReverse(FlagID id, FlagID reverseId) : id(id), reverseId(reverseId)
    {}

    FlagID id;
    FlagID reverseId;
    uint32_t count{0};
};

template <uint8_t MODE, int32_t CORE_TYPE>
struct BarrierFlag {
    static_assert(MODE != 255 && MODE == 255, "Unsupported cross core barrier flag, can not find the specialization.");
};

template <>
struct BarrierFlag<0x0, AscendC::AIV> {
    static constexpr FlagID ID = AIV_INTER_BLOCK_BARRIER;
};

template <>
struct BarrierFlag<0x0, AscendC::AIC> {
    static constexpr FlagID ID = AIC_INTER_BLOCK_BARRIER;
};

template <>
struct BarrierFlag<0x1, AscendC::AIV> {
    static constexpr FlagID ID = AIV_INTER_SUBBLOCK_BARRIER;
};

template <uint8_t MODE, pipe_t PIPE>
CATLASS_DEVICE void CrossCoreBarrier()
{
    constexpr FlagID flagId = BarrierFlag<MODE, g_coreType>::ID;
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flagId);
    AscendC::CrossCoreWaitFlag(flagId);
}

template <uint8_t MODE, pipe_t PIPE>
CATLASS_DEVICE void CrossCoreSetFlag(CrossCoreFlag& flag)
{
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.id);
}

CATLASS_DEVICE
void CrossCoreWaitFlag(CrossCoreFlag& flag)
{
    AscendC::CrossCoreWaitFlag(flag.id);
}

template <uint8_t MODE, pipe_t PIPE>
CATLASS_DEVICE void CrossCoreWaitFlag(CrossCoreFlag& flag)
{
    AscendC::CrossCoreWaitFlag<MODE, PIPE>(flag.id);
}

template <uint8_t MODE, pipe_t PIPE, uint32_t REVERSE_DEPTH>
CATLASS_DEVICE void CrossCoreSetFlagWithReverse(CrossCoreFlagWithReverse<REVERSE_DEPTH>& flag)
{
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.id);
    if (++flag.count >= REVERSE_DEPTH) {
        AscendC::CrossCoreWaitFlag(flag.reverseId);
        flag.count = 0;
    }
}

template <uint8_t MODE, pipe_t PIPE, uint32_t REVERSE_DEPTH>
CATLASS_DEVICE void CrossCoreWaitFlagWithReverse(CrossCoreFlagWithReverse<REVERSE_DEPTH>& flag)
{
    AscendC::CrossCoreWaitFlag(flag.id);
    if (++flag.count >= REVERSE_DEPTH) {
        AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.reverseId);
        flag.count = 0;
    }
}

} // namespace Catlass::Arch

#endif // CATLASS_ARCH_CROSS_CORE_SYNC_HPP
```

Note the mode mapping asserted by the `BarrierFlag` specialisations — it matches the official doc and
contains no mode 3:

```
BarrierFlag<0x0, AscendC::AIV> -> flagId 8    // mode 0 = inter-BLOCK (across AI Cores), vector side
BarrierFlag<0x0, AscendC::AIC> -> flagId 9    // mode 0 = inter-BLOCK (across AI Cores), cube side
BarrierFlag<0x1, AscendC::AIV> -> flagId 10   // mode 1 = inter-SUBBLOCK (the 2 AIVs of one AIC)
FFTS_MAX_FLAG = 7                              // Catlass treats 0..7 as reserved/available-to-matmul
```

Consolidated flagId budget on arch 2201 from sections 47 and 53 — all 16 are accounted for:

```
  0 .. 7   Matmul high-level API, [0, 2N-1] for N objects (N<=4);  Catlass FFTS_MAX_FLAG boundary
  8        Catlass AIV_INTER_BLOCK_BARRIER        (mode 0, AIV)
  9        Catlass AIC_INTER_BLOCK_BARRIER        (mode 0, AIC)
 10        Catlass AIV_INTER_SUBBLOCK_BARRIER     (mode 1, AIV)
 11,12,13  AscendC SyncAll<false>  (AIC: 11,12,13; AIV: 12,13)  + SetNextTaskStart/WaitPreTaskEnd
 14        AscendC SyncAll<true>   (AIV only)                    + SetNextTaskStart (AIV)
 15        Matmul in SuperKernel 1:2 ratio (GetTaskRatio() == 2), AIV
```

## Relevance
`CrossCoreSetFlagWithReverse` is the pattern we must adopt if the megakernel does one-way core-to-core
signalling in a layer loop: after 15 un-waited sets the hardware counter overflows and, per the vendor
doc, "会异常报错，中断流程". Catlass's remedy is a paired reverse flag drained every 15 sets. Also,
Catlass independently confirms modes {0, 1} for same-type/subblock and never uses 3 — so a megakernel
that leans on a mode-3 behaviour is leaning on something neither the doc nor any vendor library uses.

## 48. Provenance note — the official upstream of every LCCL file quoted above

URL: https://github.com/hicann/ascend-transformer-boost/tree/master/src/kernels/lcal/src/ascendc_kernels
URL: https://gitcode.com/cann/ascend-transformer-boost
Author: Huawei Technologies Co., Ltd.
Date: official mirror live since 2025-09

The `egretwAlker/cpl-ascend-transformer-boost` tree quoted throughout is byte-identical to the official
`hicann/ascend-transformer-boost` mirror except for the licence header, which upstream has advanced
from v1.0 to v2.0. Diff of `sync_collectives.h`, official vs. the copy quoted above — the ONLY
difference in the whole file:

```diff
3,4c3,4
<  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
<  * CANN Open Software License Agreement Version 2.0 (the "License").
---
>  * This file is a part of the CANN Open Software.
>  * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
```

Same for `op_def.h`. Directory listing of the official upstream `ascendc_kernels` directory:

```
91093
CMakeLists.txt
allgather.h
allreduce_big_data.h
allreduce_one_shot.h
allreduce_quant.h
allreduce_two_shot.h
collectives.h
datacopy_gm2gm.h
datacopy_gm2gm_delay.h
ipc_queue.h
lccl_op.h
lccl_op1.cpp
lccl_op2.cpp
op_def.h
reduce_scatter.h
sync_collectives.h
```

## Relevance
Every constant, flag name and threshold quoted in sections 40-43 and 48-51 is current upstream vendor
code, not a stale fork. Raw fetch pattern that works:
`https://raw.githubusercontent.com/hicann/ascend-transformer-boost/master/src/kernels/lcal/src/ascendc_kernels/sync_collectives.h`

## 49. The SOFT `SyncAll` — a vendor grid barrier that consumes ZERO flagIds, supports a SUBSET of cores, and is itself epoch-tagged with no zeroing

URL: https://github.com/hicann/asc-devkit/blob/master/impl/basic_api/dav_c220/kernel_operator_sync_impl.h
URL: https://github.com/hicann/asc-devkit/blob/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md
Author: Huawei / CANN `asc-devkit`
Date: current `master` of hicann/asc-devkit (2026)

### 55a. The three prototypes, and which exist on A3

```cpp
- 软同步

    template <bool isAIVOnly = true>
    __aicore__ inline void SyncAll(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, const int32_t usedCores = 0)

- 硬同步
    - 不支持config指定流水

        template <bool isAIVOnly = true>
        __aicore__ inline void SyncAll()

    - 支持config指定流水

        template <bool isAIVOnly = true, const SyncAllConfig& config = DEFAULT_SYNC_ALL_CONFIG>
        __aicore__ inline void SyncAll()
```

```
### 硬同步（支持config指定流水）原型
- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：不支持
- Atlas A2系列产品：不支持
```

So on A3 only the soft form and the plain hard form exist; the `SyncAllConfig{triggerPipe, waitPipe}`
variant is 950-only.

```
| config | **该配置仅当isAIVOnly=True时有效。**
struct SyncAllConfig {
    pipe_t triggerPipe;
    pipe_t waitPipe;
};

// 默认使用全部流水来进行触发和等待行为。
constexpr SyncAllConfig DEFAULT_SYNC_ALL_CONFIG = {PIPE_ALL, PIPE_ALL};
控制SyncAll函数的行为，在多个AI Core之间进行流水同步时，指定哪些流水（pipe）用于触发和等待。
triggerPipe：指定哪个流水用于"发送触发信号"。仅支持MTE2、MTE3、PIPE_ALL。
waitPipe：指定哪个流水用于"接收等待信号"。仅支持MTE2、MTE3、PIPE_ALL。 |
```

### 55b. What the two forms actually do, verbatim

```
SyncAll是核间同步控制接口，根据不同的函数原型介绍其功能：

- 硬件同步接口：
    用于多个核之间的同步控制，支持以下同步场景：
    - 纯Vector算子的全核同步，可参考CrossCoreSetFlag的模式0。
    - 融合算子的全核同步，一次性实现三个核间同步能力：首先完成单个AI Core内，AIC与所有AIV同步（AIC等AIV）；接着完成多个AI Core间，所有AIC的同步；最后完成单个AI Core内，所有AIV与AIC同步（AIV等AIC）。

- 软件同步接口：
    仅支持纯Vector算子的多核（参与同步的核数可通过入参指定）同步。每一个AIV都向全局内存对应位置写入信号值1，接着每一个AIV都持续轮询全局内存中所有AIV对应的信号值，直到所有AIV核对应的信号值都变为1。

在纯Vector算子场景中，若所有AIV核默认参与同步，推荐采用性能更优的硬件同步接口；若需指定部分AIV核参与同步，则应使用软件同步接口，并通过入参usedCores完成配置。
```

```
| isAIVOnly | 控制SyncAll作用于纯Vector算子或Mix（包含Cube和Vector计算）算子。可选值：
• **true**（默认值）：纯Vector算子的全核同步，仅执行Vector核的全核同步。
• **false**：Mix（包含Cube和Vector计算）算子的全核同步，先分别完成Vector核和Cube核的全核同步，再执行两者之间的同步（软同步接口不支持此功能）。 |
```

### 55c. Constraints — including two ways to hang

```
- 硬件同步接口约束：
    - 使用硬同步接口时，需根据场景使用合适的核函数（Kernel）修饰符：
        - 在纯Vector算子，需使用__mix__(0, 1)修饰核函数（Kernel），而不是使用__vector__，具体原因请参考CrossCoreSetFlag约束说明中的核函数（Kernel）类型配置说明。
        - 对于Mix（包含Cube和Vector计算）算子，需使用__mix__(1, 1)或__mix__(1, 2)修饰核函数（Kernel）。
    - 对于纯Vector算子，模板参数isAIVOnly必须设置为true，否则会导致程序卡死。
    - 对于Mix（包含Cube和Vector计算）算子，模板参数isAIVOnly设置为true，会导致接口实际只能完成Vector全核同步与预期同步行为不符。
    - 使用该接口时，建议开启batchmode模式，使算子独占全部所需核资源，否则可能因满足以下条件导致死锁：
        - 多流并发场景（≥2条执行流）。
        - ≥2个算子并发执行。
        - 所有并发算子的核数总和超过物理核数。
        - ≥2个并发算子使用了核间同步功能。
    - SyncAll硬件同步接口内部实现中使用了CrossCoreSetFlag和CrossCoreWaitFlag进行核间同步控制，因此开发者同时使用CrossCoreSetFlag（或CrossCoreWaitFlag）和SyncAll硬件同步接口时，需注意避免flagId冲突
```

```
- 软件同步接口约束：
    - gmWorkspace缓存申请的空间大小要求大于等于GetBlockNum()*32Bytes，并且缓存的值需要初始化为0。目前常见的有两种初始化方式：
        - 通过在host侧进行初始化操作，确保传入该接口时，gmWorkspace缓存已经初始化为0；
        - 在核函数（Kernel）侧初始化的时候对gmWorkspace缓存初始化，需要注意的是，每个核上都需要初始化全部的gmWorkspace缓存空间。
    - ubWorkspace申请的空间大小要求大于等于GetBlockNum()*32Bytes。
    - usedCores传入数值不能超过算子调用时指定的逻辑AI Core中AIV的数量：GetBlockNum()*GetTaskRatio()，不传此参数表示全核软同步。
```

```
- 硬件同步接口和软件同步接口公共约束：使用该接口进行多核控制时，算子调用时指定的逻辑AI Core核数numBlocks必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致核函数（Kernel）"卡死"现象。
```

### 55d. `SoftSyncAllImpl` verbatim — a self-incrementing epoch with no zeroing, zero flagIds

```cpp
__aicore__ inline void ClcSyncCount(
    __gm__ int32_t* localSyncGM, __ubuf__ int32_t* ubWorkspaceAddr, const int32_t blockIdx, const int32_t totalBlocks,
    bool isFirst, int32_t& count)
{
    if (isFirst) {
        __ubuf__ int32_t* localUbAddr = ubWorkspaceAddr + (blockIdx * DEFAULT_BLK_NUM);
        *(reinterpret_cast<__ubuf__ int32_t*>(localUbAddr)) = 1;
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        copy_ubuf_to_gm(
            static_cast<__gm__ void*>(localSyncGM), static_cast<__ubuf__ void*>(localUbAddr), 0, 1, 1, 0, 0);
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        count = 1;
        for (int32_t i = 0; i < totalBlocks; i++) {
            if (i != blockIdx) {
                count += *(reinterpret_cast<__ubuf__ int32_t*>(ubWorkspaceAddr) + i * ONE_BLK_FLOAT_NUM);
            }
        }
    } else {
        for (int32_t i = 0; i < totalBlocks; i++) {
            count += *(reinterpret_cast<__ubuf__ int32_t*>(ubWorkspaceAddr) + i * ONE_BLK_FLOAT_NUM);
        }
    }
}

__aicore__ inline int64_t GetBlockNum();
template <bool isAIVOnly = true>
__aicore__ inline void SoftSyncAllImpl(
    __gm__ int32_t* gmWorkspaceAddr, __ubuf__ int32_t* ubWorkspaceAddr, const int usedCores)
{
    if ASCEND_IS_AIC {
        return;
    }
    __sync_all_stub(usedCores, isAIVOnly);
    PipeBarrier<PIPE_ALL>();

    int32_t totalBlocks = isAIVOnly ? GetBlockNum() : (GetTaskRationImpl() * GetBlockNum());
    totalBlocks = usedCores != 0 ? usedCores : totalBlocks;
#ifdef __ASCENDC_ENABLE_SUPER_KERNEL__
    int32_t blockIdx = GetBlockIdxImpl();
#else
    int32_t blockIdx = isAIVOnly ? get_block_idx() : GetBlockIdxImpl();
#endif

    __gm__ int32_t* localSyncGM = gmWorkspaceAddr + (blockIdx * DEFAULT_BLK_NUM);
    __ubuf__ int32_t* localUbAddr = ubWorkspaceAddr + (blockIdx * DEFAULT_BLK_NUM);

    copy_gm_to_ubuf(static_cast<__ubuf__ void*>(localUbAddr), static_cast<__gm__ void*>(localSyncGM), 0, 1, 1, 0, 0);
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    int32_t curValue = *(reinterpret_cast<__ubuf__ int32_t*>(localUbAddr)) + 1;
    bool isFirst = curValue == 1 ? true : false;
    *(reinterpret_cast<__ubuf__ int32_t*>(localUbAddr)) = curValue;
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    copy_ubuf_to_gm(static_cast<__gm__ void*>(localSyncGM), static_cast<__ubuf__ void*>(localUbAddr), 0, 1, 1, 0, 0);
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    int32_t totalBlockCount = ONE_BLK_FLOAT_NUM * totalBlocks;
    uint16_t blockLen = totalBlockCount / AscendCUtils::GetC0Count(sizeof(int32_t));
    while (true) {
        copy_gm_to_ubuf(
            static_cast<__ubuf__ void*>(ubWorkspaceAddr), static_cast<__gm__ void*>(gmWorkspaceAddr), 0, 1, blockLen, 0,
            0);
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        int32_t count = 0;
        ClcSyncCount(localSyncGM, ubWorkspaceAddr, blockIdx, totalBlocks, isFirst, count);
        event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        if (count >= (totalBlocks * curValue)) {
            break;
        }
    }
    __sync_all_stub(usedCores, isAIVOnly);
}
```

The protocol, stated exactly:

```
slot for core i      : gmWorkspaceAddr + i * DEFAULT_BLK_NUM        (8 x int32 = 32 B per core)
on entry             : curValue = *own_slot + 1 ;  *own_slot = curValue   (NEVER reset to 0)
readiness            : sum_{i<totalBlocks} slot[i] >= totalBlocks * curValue
totalBlocks          : isAIVOnly ? GetBlockNum() : GetTaskRationImpl() * GetBlockNum()
                       overridden by usedCores when usedCores != 0
hardware flagIds     : NONE. only intra-core HardEvent ids, dynamically taken via
                       GetTPipePtr()->FetchEventID(HardEvent::...)
AIC                  : returns immediately ("if ASCEND_IS_AIC { return; }")
```

## Relevance
This is the barrier to use inside a megakernel. It costs **zero** of the 16 `CrossCoreSetFlag` flagIds,
cannot saturate a 15-deep counter, and its `usedCores` parameter lets us barrier only the 16 comm
blocks instead of the whole grid. It is also itself a worked example of exactly the epoch discipline
we are after: the workspace is zeroed once (host side or first kernel), never again, and the barrier
generation number *is* the stored value, with readiness decided by `sum >= n * generation`. The one
caveat the vendor flags: on a pure-vector kernel use `__mix__(0, 1)`, not `__vector__`, or the hard
form hangs; and in a MIX kernel `isAIVOnly=true` silently syncs only the vector cores.

## 50. `datacopy_gm2gm.h` + `collectives.h` — the one-sided reduce itself: `SetAtomicAdd` on a PEER's IPC window, and the UB staging constants

URL: https://github.com/hicann/ascend-transformer-boost/blob/master/src/kernels/lcal/src/ascendc_kernels/datacopy_gm2gm.h
URL: https://github.com/hicann/ascend-transformer-boost/blob/master/src/kernels/lcal/src/ascendc_kernels/collectives.h
Author: Huawei Technologies Co., Ltd. (CANN Open Software License Agreement Version 2.0 upstream)
Date: Copyright 2024

There is no special "remote reduce" primitive. LCCL's entire reduction is an ordinary GM->UB->GM copy
with the hardware atomic enabled, where the destination GM pointer happens to be inside another rank's
IPC window. The op code maps onto `SetAtomicAdd` / `SetAtomicMax` / `SetAtomicMin`:

```cpp
    template <typename T>
    FORCE_INLINE_AICORE void SetAscendCAtomic(int op)
    {
        SetAtomicType<T>();
        switch (op) {
            case ADD:
                SetAtomicAdd<T>();
                return;
            case MUL:
                return;
            case MAX:
                SetAtomicMax<T>();
                return;
            case MIN:
                SetAtomicMin<T>();
                return;
            default:
                ;
        }
    }
```

```cpp
enum Op : int {
    COPYONLY = -1,
    ADD = 0,
    MUL = 1,
    MAX = 2,
    MIN = 3
};
```

Note `MUL` deliberately falls through to no atomic — this is why ATB rejects `allReduceType == "prod"`
on the lccl backend.

The copy engine, with the UB staging geometry and the exact event-id discipline:

```cpp
class DataCopyGM2GM {
    constexpr static int32_t UB_HEAD_OFFSET = 64;
    constexpr static int32_t BLOCK_SIZE_PIECE = BLOCK_SIZE / (sizeof(T) + sizeof(U)) / ALIGN_SIZE * ALIGN_SIZE;
    constexpr static int32_t INPUT_BLOCK_SIZE = std::is_same_v<T, U> ? BLOCK_SIZE : BLOCK_SIZE_PIECE * sizeof(U);
    constexpr static int32_t OUTPUT_BLOCK_SIZE = std::is_same_v<T, U> ? BLOCK_SIZE : BLOCK_SIZE_PIECE * sizeof(T);
public:
    FORCE_INLINE_AICORE DataCopyGM2GM() {}
    FORCE_INLINE_AICORE void Init(const GlobalTensor<T>& outputGt, const GlobalTensor<U>& inputGt,
        const uint32_t calCount, int op)
    {
        inputGm = inputGt.GetPhyAddr();
        outputGm = outputGt.GetPhyAddr();
        inputUB = (__ubuf__ U*)(UB_HEAD_OFFSET);
        if constexpr (std::is_same_v<T, U>) {
            outputUB = (__ubuf__ T*)inputUB;
        } else {
            outputUB = (__ubuf__ T*)(UB_HEAD_OFFSET + INPUT_BLOCK_SIZE);
        }
        this->op = op;
        dataSizeRemain = calCount * sizeof(T);
    }

    FORCE_INLINE_AICORE void Process()
    {
        SetAtomic(op);
        int64_t i = 0;
        while (dataSizeRemain >= OUTPUT_BLOCK_SIZE) {
            CpGM2UB(inputUB, (__gm__ U*)inputGm + i * INPUT_BLOCK_SIZE / sizeof(U), INPUT_BLOCK_SIZE);
            AscendC::SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            if constexpr (!std::is_same_v<T, U>) {
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                CastImpl(outputUB, inputUB, RoundMode::CAST_NONE, INPUT_BLOCK_SIZE / sizeof(U));
                AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            }
            CpUB2GM((__gm__ T*)outputGm + i * OUTPUT_BLOCK_SIZE / sizeof(T), (__ubuf__ T*)outputUB,
                OUTPUT_BLOCK_SIZE);
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            i += 1;
            dataSizeRemain -= OUTPUT_BLOCK_SIZE;
        }
        if (dataSizeRemain > 0) {
            CpGM2UB(inputUB, (__gm__ U*)inputGm + i * INPUT_BLOCK_SIZE / sizeof(U),
                dataSizeRemain / sizeof(T) * sizeof(U));
            AscendC::SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            ...
            CpUB2GM((__gm__ T*)outputGm + i * OUTPUT_BLOCK_SIZE / sizeof(T), (__ubuf__ T*)outputUB,
                dataSizeRemain);
            PipeBarrier<PIPE_ALL>();
        }
        UnsetAtomic(op);
    }
```

The double-buffered variant used by `CpGM2GMPingPong`, with the UB split constants:

```cpp
class Collectives {
    constexpr static int32_t UB_HEAD_OFFSET = 96;
    constexpr static int32_t UB_MID_OFFSET = UB_HEAD_OFFSET + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + ALIGN_SIZE;
    ...
    template <typename T>
    FORCE_INLINE_AICORE void DataCopyWrapPingPong(const GlobalTensor<T>& inputGT, const GlobalTensor<T>& outputGT,
        int64_t dataSizeRemain, int op, TBuf<QuePosition::VECCALC> tbuf)
    {
        if (dataSizeRemain <= 0) {
            return;
        }
        LocalTensor<T> localUB[2];
        localUB[0] = tbuf.GetWithOffset<T>(UB_SINGLE_PING_PONG_ADD_SIZE_MAX, 0);
        localUB[1] = tbuf.GetWithOffset<T>(UB_SINGLE_PING_PONG_ADD_SIZE_MAX, UB_SINGLE_PING_PONG_ADD_SIZE_MAX);

        int inputOffsetNum = 0;
        int outputOffsetNum = 0;

        PipeBarrier<PIPE_ALL>();
        if (op != COPYONLY) {
            SetAscendCAtomic<T>(op);
        }
        PipeBarrier<PIPE_ALL>();

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int64_t i = 0; dataSizeRemain > 0; i++) {
            uint32_t size = dataSizeRemain > UB_SINGLE_PING_PONG_ADD_SIZE_MAX ?
                UB_SINGLE_PING_PONG_ADD_SIZE_MAX : dataSizeRemain;
            TEventID eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            DataCopyWrap(localUB[(i & 1) ? 0 : 1], inputGT[inputOffsetNum], size);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            DataCopyWrap(outputGT[outputOffsetNum], localUB[(i & 1) ? 0 : 1], size);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            dataSizeRemain -= size;
            inputOffsetNum += (size / sizeof(T));
            outputOffsetNum += (size / sizeof(T));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);

        if (op != COPYONLY) {
            SetAtomicNone();
        }
        PipeBarrier<PIPE_ALL>();
    }
```

Relevant constants again, for sizing:

```
UB_SINGLE_DMA_SIZE_MAX            = 190 * 1024          // 190 KB
UB_SINGLE_PING_PONG_ADD_SIZE_MAX  = 190 * 1024 / 2      // 95 KB per ping-pong half
UB_ALIGN_SIZE                     = 32
Collectives::UB_HEAD_OFFSET       = 96
Collectives::UB_MID_OFFSET        = 96 + 95 KB + ALIGN_SIZE
DataCopyGM2GM::UB_HEAD_OFFSET     = 64
```

## Relevance
The reduce is free: `SetAtomicAdd<bf16>()` plus a `CpUB2GM` whose destination is `shareAddrs[peer] +
IPC_DATA_OFFSET + ...`. No separate reduce pass, no scratch. Our megakernel already does GM<->UB
staging, so the whole all-reduce is "turn on the atomic, point the store at a peer's window, set a
flag". Note the hardware HardEvent ids used in LCCL are only EVENT_ID0/1/3 — these are intra-core
pipe events, entirely disjoint from the 16 `CrossCoreSetFlag` flagIds discussed in sections 47 and 53.

## 51. Mirror access recipe — verified working endpoints for all of the above

URL: https://gitcode.com/cann/ascend-transformer-boost
URL: https://gitee.com/ascend/cann-hccl
URL: https://github.com/hicann/ascend-transformer-boost
Author: n/a (access notes)
Date: verified 2026-10-20

Verified HTTP results from this machine (behind the corporate proxy already in the environment):

```
https://gitcode.com/cann/ascend-transformer-boost            -> 200   318960 bytes
https://gitcode.com/cann/hccl                                -> 200   232901 bytes
https://gitcode.com/cann/catlass                             -> 200   284889 bytes
https://gitcode.com/cann/lccl                                -> 404           (DOES NOT EXIST;
                                                                 lcal/lccl lives INSIDE ascend-transformer-boost)
https://gitee.com/ascend/cann-hccl                           -> 200    14560 bytes
```

Raw single-file fetch that works on GitCode (returns the v2.0-licence upstream copy):

```
curl -s -L --max-time 60 \
  'https://raw.gitcode.com/cann/ascend-transformer-boost/raw/master/src/kernels/lcal/src/ascendc_kernels/sync_collectives.h' \
  -o out.h -w '%{http_code} %{size_download}\n'
# -> 200 12376

curl -s -L --max-time 60 \
  'https://raw.gitcode.com/cann/ascend-transformer-boost/raw/master/src/kernels/lcal/src/ascendc_kernels/op_def.h' \
  -o out.h -w '%{http_code} %{size_download}\n'
# -> 200 4793   (contains SET_MAGIC, 3 occurrences)
```

GitHub mirrors, fetched via authenticated `gh` (fastest; returns base64 in `.content`):

```
gh api repos/hicann/ascend-transformer-boost/contents/src/kernels/lcal/src/ascendc_kernels/sync_collectives.h --jq '.content' | base64 -d
gh api repos/hicann/asc-devkit/contents/impl/basic_api/dav_c220/kernel_operator_sync_impl.h --jq '.content' | base64 -d
gh api repos/hicann/catlass/contents/include/catlass/arch/cross_core_sync.hpp --jq '.content' | base64 -d
gh api repos/vllm-project/vllm-ascend/contents/csrc/utils/inc/kernel/sync_collectives.h --jq '.content' | base64 -d
gh api repos/sgl-project/sgl-kernel-npu/contents/csrc/deepep/ops2/op_kernel/notify_dispatch_a2.h --jq '.content' | base64 -d
```

Repos that turned out NOT to carry anything on this topic, checked and negative:

```
huawei-csl/megagdn-pto      -> tree contains only .github/skills + pre-commit config; no kernels
Ascend/pytorch              -> no npu_symmetric_memory / SymmetricMemory hits
<any repo>                  -> no HcclGetCommWindow / HcclBufferGet symbol anywhere in GitHub code search
<any repo>                  -> no aclnnInplaceAllReduce in GitHub code search
```

Chinese-language community search was run with the terms requested (`LCCL 源码解析`, `lccl allreduce
magic flag 昇腾 核内通信`, `昇腾 LCCL 低时延通信库 one-shot allreduce 原理 2MB flag 标志位 IPC
buffer 解析`, `ascend-transformer-boost lccl 源码 sync_collectives magic 标志位`) across zhihu /
blog.csdn.net / bbs.huaweicloud.com / gitcode. No blog or forum post covers LCCL's magic-flag
internals; the only material is the vendor source and the vendor's own Chinese API reference, both of
which are quoted directly in the sections above. The one external idea worth recording, from a
Triton-on-symmetric-heap writeup of one-shot all-reduce, is the alternative to an epoch: a per-tile
counter incremented on every peer and compared against `N-1`, then reset to 0 — i.e. a resettable
counter rather than a monotone epoch, with `cdiv(numel, BLOCK_SIZE)` flags on the symmetric heap.

## Relevance
Confirms there is no separate `cann/lccl` repository to chase — LCCL is a subdirectory of
ascend-transformer-boost, mirrored identically on GitCode, Gitee's cann-hccl sibling, and
github.com/hicann. All quoted material in sections 40-56 is reachable from at least two of these.

## 52. FIRST-HAND VERIFICATION of the dav_c220 (arch 2201) sync lowering — fetched and read directly, 2026-10-20

URL: https://raw.gitcode.com/cann/asc-devkit/raw/master/impl/basic_api/dav_c220/kernel_operator_sync_impl.h
Browsable: https://gitcode.com/cann/asc-devkit/blob/master/impl/basic_api/dav_c220/kernel_operator_sync_impl.h
File size as fetched: 22,381 bytes. HTTP 200.
Author: Huawei Technologies Co., Ltd. Branch `master`, fetched 2026-10-20.

§19 and §41 above arrive at the same conclusions through intermediated routes. Because those
conclusions contradict two of our own measurements, the file was fetched and read directly. It
does. Here is the verbatim text, with the fetched line numbers.

### 52.1 The flag constants — lines 117–127, verbatim

```cpp
constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

__aicore__ inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}
```

`mode & 0x3` → 2 bits, so modes 0/1/2/3 all *encode*. `flagId & 0xf` → 4 bits, so 16 flag IDs,
0–15. Reserved: **11 = AIC, 12 = AIV, 13 = AIC+AIV, 14 = AIV_ONLY_ALL.**

### 52.2 `SyncAllImpl` — lines 363–403, the complete body, verbatim

```cpp
template <bool isAIVOnly = true>
__aicore__ inline void SyncAllImpl()
{
#if defined(__ASCENDC_SUPERKERNEL_AUTO_SYNC_ALL__)
    if (g_superKernelAutoSyncAllEnable) {
        if (GetBlockIdxImpl() == 0) {
            g_superKernelAutoSyncAllSyncIdx += 1;
            // |    32bits  |   16bits  |   8bits   |   8bits   |
            // |  sync idx  | sync type |   is end  | is valid  |
            *reinterpret_cast<__gm__ uint64_t*>(g_superKernelAutoSyncAllConfigGmAddr) =
                (static_cast<uint64_t>(g_superKernelAutoSyncAllSyncIdx)
                 << SK_AUTO_SYNC_ALL_SYNC_IDX_CONFIG_BIT_OFFSET) |
                (static_cast<uint64_t>(isAIVOnly) << SK_AUTO_SYNC_ALL_SYNC_TYPE_CONFIG_BIT_OFFSET) |
                static_cast<uint64_t>(SK_AUTO_SYNC_ALL_VALID_MAGIC_NUM);
            dcci(
                reinterpret_cast<__gm__ uint64_t*>(g_superKernelAutoSyncAllConfigGmAddr),
                cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        }
    }
#endif

    PipeBarrier<PIPE_ALL>();
    if constexpr (isAIVOnly) {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
        return;
    }
    // trans software sync to hardware sync
    if ASCEND_IS_AIC {
        wait_flag_dev(SYNC_AIV_FLAG);
        ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
        wait_flag_dev(SYNC_AIC_FLAG);
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
        return;
    }
    if ASCEND_IS_AIV {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
        wait_flag_dev(SYNC_AIC_AIV_FLAG);
        return;
    }
}
```

Read it line by line:

| what | mode | flagId | set pipe | issuing core |
|---|---|---|---|---|
| `SyncAll<true>` whole barrier | 0 | **14** | `PIPE_MTE3` | AIV |
| `SyncAll<false>` step 1: V→C gather | **2** | **12** | `PIPE_MTE3` | **AIV** |
| `SyncAll<false>` step 2: all-AIC barrier | 0 | **11** | `PIPE_FIX` | AIC |
| `SyncAll<false>` step 3: C→V release | **2** | **13** | **`PIPE_MTE3`** | **AIC** |

### 52.3 `SoftSyncAllImpl` — lines 63–114, the ZERO-flagId barrier, verbatim

```cpp
__aicore__ inline int64_t GetBlockNum();
template <bool isAIVOnly = true>
__aicore__ inline void SoftSyncAllImpl(
    __gm__ int32_t* gmWorkspaceAddr, __ubuf__ int32_t* ubWorkspaceAddr, const int usedCores)
{
    if ASCEND_IS_AIC {
        return;
    }
    __sync_all_stub(usedCores, isAIVOnly);
    PipeBarrier<PIPE_ALL>();

    int32_t totalBlocks = isAIVOnly ? GetBlockNum() : (GetTaskRationImpl() * GetBlockNum());
    totalBlocks = usedCores != 0 ? usedCores : totalBlocks;
#ifdef __ASCENDC_ENABLE_SUPER_KERNEL__
    int32_t blockIdx = GetBlockIdxImpl();
#else
    int32_t blockIdx = isAIVOnly ? get_block_idx() : GetBlockIdxImpl();
#endif

    __gm__ int32_t* localSyncGM = gmWorkspaceAddr + (blockIdx * DEFAULT_BLK_NUM);
    __ubuf__ int32_t* localUbAddr = ubWorkspaceAddr + (blockIdx * DEFAULT_BLK_NUM);

    copy_gm_to_ubuf(static_cast<__ubuf__ void*>(localUbAddr), static_cast<__gm__ void*>(localSyncGM), 0, 1, 1, 0, 0);
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    int32_t curValue = *(reinterpret_cast<__ubuf__ int32_t*>(localUbAddr)) + 1;
    bool isFirst = curValue == 1 ? true : false;
    *(reinterpret_cast<__ubuf__ int32_t*>(localUbAddr)) = curValue;
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    copy_ubuf_to_gm(static_cast<__gm__ void*>(localSyncGM), static_cast<__ubuf__ void*>(localUbAddr), 0, 1, 1, 0, 0);
    ...
        if (count >= (totalBlocks * curValue)) {
            break;
        }
    }
    __sync_all_stub(usedCores, isAIVOnly);
}
```

Same `curValue = *own_slot + 1`, spin-until-`sum >= totalBlocks * curValue` epoch discipline as
MegaMoE's `SynchronizeRanks` (§11.2) and LCCL's `SET_MAGIC` (§35) — the vendor uses one idea in
three places. Note `if ASCEND_IS_AIC { return; }`: the soft barrier is AIV-only by construction,
and `usedCores` lets it barrier a **subset**.

### 52.4 `SetNextTaskStart` / `WaitPreTaskEnd` — the SuperKernel early-start flags, and AIC on `PIPE_MTE3` again

Lines 129–191, verbatim (abridged where repetitive):

```cpp
template <pipe_t AIV_PIPE, pipe_t AIC_PIPE>
__aicore__ inline void SetNextTaskStartV3Impl(uint32_t earlyStartConfig)
{
    if ASCEND_IS_AIC {
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIC_TO_AIC_SET) {
            ffts_cross_core_sync(AIC_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG)); // SET:C--->C
        }
    }
    if ASCEND_IS_AIV {
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIV_TO_AIV_SET) {
            ffts_cross_core_sync(AIV_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL)); // SET: V--->V
        }
    }
}

template <pipe_t AIV_PIPE = PIPE_MTE3, pipe_t AIC_PIPE = PIPE_FIX, bool FORCE = false>
__aicore__ inline void SetNextTaskStartImpl()
{
    ...
        ffts_cross_core_sync(AIC_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG));
    ...
        ffts_cross_core_sync(AIV_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
```

```cpp
    } else if (g_super_kernel_early_start_config == ASCENDC_SUPER_KERNEL_EARLY_START_AIC_TO_AIV) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
        }
    } else if (g_super_kernel_early_start_config == ASCENDC_SUPER_KERNEL_EARLY_START_AIV_TO_AIC) {
        if ASCEND_IS_AIC {
            ...
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));
        }
```

Note the default template arguments of `SetNextTaskStartImpl`:
**`AIV_PIPE = PIPE_MTE3, AIC_PIPE = PIPE_FIX`** — the vendor's own default pairing for mode 0.

## Relevance

This is our own reading of the shipped arch-2201 source, so it settles the three disputed items.

**Our "sync mode 2 is cube→vector" is incomplete, and "mode 3 is the only reverse (vector→cube)
mode that works in all set/wait patterns" is wrong as stated.** `SyncAllImpl<false>` uses
**mode 2 in both directions in the same function**: the AIV branch issues
`GetffstMsg(0x02, SYNC_AIV_FLAG)` — that *is* a vector→cube mode-2 set — and the AIC branch
issues `GetffstMsg(0x02, SYNC_AIC_AIV_FLAG)` for cube→vector. Direction is carried by **which
core issues the set** and disambiguated by **which flagId** (12 for V→C, 13 for C→V), not by the
mode. `wait_flag_dev(flagId)` on 2201 takes one argument and no mode at all, so the wait side
cannot distinguish modes even in principle. Mode 3 encodes (`mode & 0x3`) but appears in no call
site in this file. Whatever we observed as "mode 3 works in all patterns" is consistent with mode
3 simply bumping some counter that the mode-blind wait then consumes — it is not evidence that
mode 2 reverse is broken. The thing to re-test is the **pairing cardinality**: mode 2 gathers
*both* AIVs of an AI Core into one AIC credit, so if only one of the two sub-blocks issues the
set, the AIC waits forever.

**Our "SyncAll consumes 11-13" is exactly right — for `SyncAll<false>`.** `SyncAll<true>`, which
is the **default template argument**, costs **flag 14 and nothing else**. So the AIV-only grid
barrier is one flag, not eleven-to-thirteen, and switching from `SyncAll<false>` to
`SyncAll<true>` plus a hand-rolled AIC barrier frees 11/12/13 — three of our sixteen. Better
still, `SoftSyncAllImpl` (§52.3) costs **zero flagIds** (GM + UB workspace only) and takes a
`usedCores` argument, so it can barrier a subset — at the price of a GM round trip per barrier,
which at our 49.3 us/layer may well be the wrong trade, but it is a real third option.

**§15.5's claim that AIC cannot set on `PIPE_MTE3` does not hold on arch 2201.** That claim is
read out of dav-3510 (`IsSplitCubePipe = {S, MTE1, MTE2, FIX, M}`). On dav_c220, Huawei's own
`SyncAllImpl` has the AIC branch issue `ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02,
SYNC_AIC_AIV_FLAG))`, and `WaitPreTaskEnd` does the same twice more. MegaGDN's A2/A3 barrier
(§8.4) independently matches: cube uses `PIPE_FIX` for the mode-0 AIC↔AIC leg and `PIPE_MTE3` for
the mode-2 AIC→AIV leg. So the pipe/core-type restriction is **arch-specific**: on 2201 the
cube-side mode-2 set is on `PIPE_MTE3`, and `PIPE_FIX` is for the mode-0 cube barrier. Do not
port the 3510 rule.

One more thing to take: `SoftSyncAllImpl` and `SynchronizeRanks` (§11.2) and `SET_MAGIC` (§35) are
the same protocol at three scales — per-core, per-rank, per-collective — all with a never-reset
monotone counter and a `>=` (or `>= N*k`) comparison. If we build one epoch primitive and use it
for both the grid barrier and the cross-rank rendezvous, we are following the vendor's own
practice rather than inventing.

## 53. `SetNextTaskStart` / `WaitPreTaskEnd` and SuperKernel `early_start` — A3 + CANN 9.2.0 MEASURED decode numbers, from a public issue

URL: https://github.com/hw-native-sys/pypto/issues/2948
Title: `[Performance] 跨 incore task 的核内 Scalar/Vector-Cube 并发：对标 SuperKernel early_start`
Author: `nalinaly`
Opened: 2026-09-30T03:31:32Z. Status: open, no comments.
Referenced vendor doc: https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md
Fetched verbatim via `gh api repos/hw-native-sys/pypto/issues/2948`, 2026-10-20.

This is the only public source found with **Atlas A3 + CANN 9.2.0-beta.2 measured decode numbers**
for the official kernel-fusion alternative to a hand-written megakernel. It is also the issue that
names the two AscendC APIs that §52.4 showed burning flagIds 11 and 14.

### 53.1 The ask, and the measured baseline — verbatim

> ## 诉求
>
> 请 PyPTO 在**跨 incore task 的边界上**引入核内 Scalar 与 Vector/Cube 的并发，
> 对标 CANN SuperKernel 的 `early_start`（`AscendC::SetNextTaskStart()` /
> `AscendC::WaitPreTaskEnd()`）所达到的「算子头部 Scalar 指令提前执行」效果。
>
> 理由不是理论推演：在同一批 decode 形状上实测，**Native 光靠打开 SuperKernel
> 就快 6.0%~15.2%**，而这项加速 PTO 结构上拿不到——它是 aclgraph 捕获图之上的
> 二进制融合，PTO 算子不走那条路（PTO 侧静态 Kernel 装包数恒为 0）。

> ## 一、实测依据：SuperKernel 单独值 6.0%~15.2%
>
> 一个 decode 注意力半层算子，单层、单卡回放，Atlas A3 + CANN 9.2.0-beta.2。
> 两侧唯一差别是 `super_kernel_optimize`，其余
> （`static_kernel_compile: True`、`force_eager: False`、`inplace_pass: False`）
> 完全一致：
>
> ```python
> torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=False,
>               options={"force_eager": False, "inplace_pass": False,
>                        "static_kernel_compile": True,
>                        "super_kernel_optimize": <False|True>})
> ```
>
> 生效判据取 `compiler.static_super_flags`（`[false]` / `[true]`），**不是装包数**
> ——后者只说明 static kernel 装了包，与 SuperKernel 无关，我们一开始就是在这里
> 读错的。每档在**同一张卡**上按 `sk0_a → sk1_a → sk1_b → sk0_b` 的 ABBA 顺序跑，
> 每侧 20 个正式设备事件，两次取平均，超过 1.25×p50 的样本剔除：
>
> 纯 device 耗时，单位 μs：
>
> | history / batch | sk=0 min | sk=0 mean | sk=0 max | sk=1 min | sk=1 mean | sk=1 max | 收益(mean) |
> | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
> | 128K / B4 | 927.22 | 1002.68 | 1134.20 | 795.89 | 850.18 | 914.52 | **15.21%** |
> | 128K / B8 | 923.95 | 930.21 | 971.02 | 856.61 | 860.10 | 864.71 | **7.54%** |
> | 128K / B16 | 1217.71 | 1223.75 | 1230.43 | 1144.26 | 1150.18 | 1157.05 | **6.01%** |
> | 128K / B24 | 1400.50 | 1408.32 | 1419.78 | 1276.96 | 1289.71 | 1296.77 | **8.42%** |
> | 8K / B16 | 1031.78 | 1070.23 | 1167.88 | 914.63 | 953.20 | 1005.21 | **10.93%** |
> | 8K / B24 | 1179.19 | 1198.65 | 1253.34 | 1094.69 | 1119.64 | 1177.53 | **6.59%** |
> | 8K / B32 | 1302.34 | 1323.32 | 1365.85 | 1178.36 | 1208.52 | 1362.42 | **8.67%** |
>
> 128K 平均 9.29%、8K 平均 8.73%，按 128K:8K = 8:2 加权 **9.18%**。
>
> 判噪声的依据（同侧两槽的 mean 之差 vs 该档收益的绝对值，μs）：
>
> | history / batch | sk=0 两槽漂移 | sk=1 两槽漂移 | 收益绝对值 |
> | --- | ---: | ---: | ---: |
> | 128K / B4 | 45.2 | 75.3 | **152.5** |
> | 128K / B8 | 19.1 | 3.9 | **70.1** |
> | 128K / B16 | 15.0 | 20.4 | **73.6** |
> | 128K / B24 | 12.6 | 3.1 | **118.6** |
> | 8K / B16 | 40.7 | 50.7 | **117.0** |
> | 8K / B24 | 7.7 | 75.8 | **79.0** |
> | 8K / B32 | 1.7 | 13.7 | **114.8** |
>
> 每一档的收益都大于该档自身的漂移，最紧的一档（8K/B24）也有 79.0 vs 75.8。
>
> 数据卫生上踩过一个坑，一并记在这里：8K/B32 第一轮的 `sk0_b` 整槽被外部负载污染
> （p50 1651.85，同档另一槽 1313.17），这是**整槽偏移**，样本级的 1.25×p50 过滤
> 挡不住，表里是重跑一整轮 ABBA 后的值。筛查办法是除样本级过滤外还要看同侧两槽的
> 均值差，超过该档自身极差就重跑那一档。

### 53.2 What `early_start` actually is — the vendor's wording, quoted in the issue

> [SuperKernel 融合优化功能](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md)
> 把收益来源写得很明确：
>
> > 与单算子下发相比，SuperKernel技术能够优化任务调度的等待时间和调度开销，
> > **并可利用 Task 间隙资源进一步优化算子头开销**。

> **1）`early_start`（核心诉求）**
>
> > 开启后，SuperKernel 会识别并生效算子内适配的 `AscendC::SetNextTaskStart()` /
> > `AscendC::WaitPreTaskEnd()` 任务间同步接口，达到**算子头部 Scalar 指令提前执行**
> > 的优化效果。
>
> 这对接口在 CANN 9.2 的 AscendC 头文件里是实打实存在的
> （`asc/impl/basic_api/kernel_operator_common_intf_impl.h`，注释写着
> "In SuperKernel fusion mode, set wait flag between two operators"；
> 实现在各 `dav_*/kernel_operator_sync_impl.h` 的 `SetNextTaskStartImpl`，
> 通过 `ffts_cross_core_sync` 发跨核 flag）。语义是：
> **前一个 task 在它的计算 pipe 走到某一点时放行，后一个 task 的头部 Scalar 段
> （地址运算、tiling 参数、偏移推导）先跑起来，只有真正要读前者结果时才
> `WaitPreTaskEnd()` 阻塞。** 于是后继 task 的 Scalar 头被藏进前一个 task 的
> Vector/Cube 执行窗口里。
>
> **2）`auto_op_parallel`（次要）**
>
> > 在确保算子间依赖关系不变的前提下，可以将多条流上的算子按照 Cube/Vector 类型
> > 并行排布，获得相对较优的算子执行流水。

### 53.3 The three-way correspondence table, and the CUDA PDL design rules worth copying

> ### 3.3 三家的对应关系
>
> | 概念 | CUDA PDL | CANN SuperKernel `early_start` | PyPTO 现状 |
> | --- | --- | --- | --- |
> | 前驱放行后继 | `cudaTriggerProgrammaticLaunchCompletion()` | `AscendC::SetNextTaskStart()` | 无 |
> | 后继真正取数据时才阻塞 | `cudaGridDependencySynchronize()` | `AscendC::WaitPreTaskEnd()` | 无（整个 task 串行） |
> | 开关 | `cudaLaunchAttributeProgrammaticStreamSerialization` | `early_start: 1` | 无 |
> | 被重叠掉的东西 | kernel preamble + launch latency | 算子头部 Scalar 指令 | task 头部 Scalar + AICPU 派发 |
> | 硬件门槛 | sm_90+ | **A2/A3** | —— |
>
> **两家独立做到了同一件事**，这说明它不是某个后端的实现细节，而是这类
> "多任务顺序下发到同一批核"的架构上的通用优化点。

> 三条设计取舍值得 PyPTO 直接照搬：
>
> - **触发点可以不写，语义安全退化。** "If the primary kernel doesn't execute the
>   trigger, it implicitly occurs after all thread blocks in the primary kernel exit."
>   ——不标注就是原来的串行语义，**向后兼容、可逐个任务渐进采用**，不需要一次性
>   改完所有算子。
> - **同步点必须写，正确性责任在消费侧。** "the secondary thread blocks might launch
>   before data written by the primary kernel is visible. As such, … it **must always**
>   use `cudaGridDependencySynchronize` or other means to verify that the result data
>   from the primary is available."
> - **重叠是机会性的，不能当正确性前提。** "this behavior is opportunistic and not
>   guaranteed to lead to concurrent kernel execution. Reliance on concurrent execution
>   in this manner is unsafe and can lead to deadlock."

> **依赖是对数据的，不是对整块执行体的。**
> 凡是不依赖前驱结果的部分（preamble / Scalar 头 / 下一片的搬运），
> 都应该允许与前驱重叠。

### 53.4 The gap budget on the non-SuperKernel path — 44 tasks, 346 gaps, 241.8 core·μs

> ## 五、PTO 侧的量级
>
> 同一个算子的 DFX 泳道（只取 pid 4 Worker View，8K/B16）：一次 decode 调用是
> **44 个不同的 incore task、44 次派发、1009 个块事件**，铺在 72 条 AICore
> 核道上（**AIC 24 + AIV 48**）。核道上相邻两个 task 之间 **≤5 μs 的碎隙有 346 处、
> 合计 241.8 核·μs**——这部分是纯交接开销，不含依赖等待（大空隙另有成因，
> 是真实的数据流阻塞）。该档 **AIC 总空闲占 span 的 25%**。
>
> 本项目此前实测的图重放开销是**每任务约 2.3 μs、每块约 0.27 μs**，
> 按这个量上面这一次调用对应约 **374 μs** 的 AICPU 侧派发工作。
>
> 碎隙只是**看得见**的那一半；`early_start` 真正要藏的是每个 task `dur`
> **内部**那段 Scalar 头，泳道里它算在任务执行时长中，从外面看不出来。
> 所以实际可回收的量应当大于 241.8 核·μs。

### 53.5 What PyPTO already has, and the acceptance protocol

> PyPTO **已经有**同核硬件单元阶梯的概念，只是作用域限定在单个 pipeline loop 体内。
> `CanonicalizeIOOrder`（`docs/en/dev/passes/32-canonicalize_io_order.md`）：
>
> > reorders statements along a **same-core hardware-unit stage ladder**
> > (scalar → load → compute → store) … Lifting scalar compute is what unlocks
> > the load cluster.
>
> 当前形态下，同一个核上相邻两个 incore task 完全串行：前一个 task 的 Vector/Cube
> 流水排干、kernel 退出、AICPU 派发下一个、下一个 task 的 Scalar 头才开始跑。
> 在 pypto 与 simpler 代码库里检索确认目前**没有任何** `SetNextTaskStart` /
> `WaitPreTaskEnd` / early-start 相关设施。

> ## 七、建议的验收口径
>
> 1. 在同一批 decode 形状上，以同卡 ABBA、每侧 20 事件测 PTO 算子开关前后的纯
>    device 耗时，报 min/mean/max；判据是每档 Δ 大于该档自身同侧两槽的漂移。
> 2. 逐 bit 精度不变——该机制只改调度，不改任何运算次序。
> 3. 泳道上可核对：相邻 task 在同一核道上出现时间重叠，且 ≤5 μs 碎隙数量下降。

### 53.6 Related leads not yet read

- `sk_raw_project` — https://gitcode.com/sjtulxh/sk_raw_project — described as
  「基于 CANN 与 PyTorch NPU 生态的手写 SuperKernel 融合验证项目」: a hand-written
  SuperKernel fusion validation, pybind extension called from Python, a C++ task queue
  launching the SK entry, fusing two add operators with a correctness check. The only public
  example of hand-driving the SuperKernel entry found.
- hiascend forum thread `0297204111998064411`,
  https://www.hiascend.com/dev/forum/thread-0297204111998064411-1-1.html (2026-01),
  「昇腾平台大模型算子高性能优化实践：从绑核到动态融合」. Quoted secondhand in
  `ascend_mega_kernel`'s `docs/03-ascend950-megakernel.md` §3.2 as:
  > **"MegaKernel 通过整合多算子逻辑为单一 Kernel，结合绑核策略让任务常驻物理核，可大幅提升硬件资源利用率"**
  — on 910B (16 physical cores) + MindSpore 2.3 + DLCompiler/DLBlas, the case being
  **Qwen3-Next AttentionProlog (Matmul+RmsNorm+RoPE) fused into a MegaKernel, achieving
  MLA (Cube) and VPU (Vector) parallelism**. That page is behind a cookie wall; the
  `ascend_mega_kernel` doc flags it as taken from search snippets, 「细节未全文核验」.
- `ascend_mega_kernel`'s `docs/03` §3.3 verdict on whether AscendC can do a CUDA-style
  persistent kernel, verbatim:
  > | 任务常驻物理核、绑核 | **社区实践存在**（绑核 + 核内循环），无独立公开 API 名称 | hiascend 论坛 MegaKernel 帖 |
  > | CUDA 式 persistent kernel（kernel 常驻 + device 侧全局 barrier + host 仅投喂工作） | **无公开的一等公民 API/官方范式**；最接近的是 SuperKernel + SyncAll 全核同步 + UB Memory/URMA 的 device 侧 load/store/atomic（950 提供硬件原语） | 综合白皮书 §4.1.6/§4.6 与 CANN 文档，未发现官方 "persistent kernel" 条目 |
  >
  > 即：**"一次启动、核内长驻循环、tile/层间无重启"在单算子粒度上是 AscendC 原生能力；"整网一个 megakernel"在昇腾上的官方路径是 SuperKernel（图级、二进制融合），而非手写常驻 kernel**。

## Relevance

This is the number our 49.3 us vs 165.2 us has to be measured against, because it is the free
alternative. On **our exact platform** — Atlas A3, CANN 9.2.0-beta.2, decode, one attention
half-layer, single card — flipping one `torch.compile` option buys **6.01%–15.21%, weighted
9.18%**, with ABBA methodology and a per-shape noise floor check. It needs no hand-written
kernel. Any argument for a hand-written megakernel has to clear that bar and then some, and the
honest framing of our 3.3x is that it is against the *eager* vendor path, not against the
SuperKernel-fused one.

Two things to adopt directly:

1. **`AscendC::SetNextTaskStart()` / `AscendC::WaitPreTaskEnd()` are real AscendC APIs in CANN
   9.2, present on A3, and §52.4 shows their dav_c220 implementation**: `SetNextTaskStartImpl`
   defaults to `AIV_PIPE = PIPE_MTE3, AIC_PIPE = PIPE_FIX` and sets mode 0 on flag **11**
   (`SYNC_AIC_FLAG`, cube) or flag **14** (`SYNC_AIV_ONLY_ALL`, vector); `WaitPreTaskEndImpl`
   waits on 11/12/13/14 depending on `g_super_kernel_early_start_config`. So these are not
   compiler magic — they are two flagIds and a documented pipe pair, callable by hand. **Inside
   our megakernel the same mechanism is available without SuperKernel at all**: issue the
   next stage's address/tiling scalar work before waiting on the previous stage's flag. That is
   the Hazy Research "start loading the next weights while the previous computations are
   happening" idea (§1.2) with a first-party Ascend primitive behind it.
2. **The measurement protocol.** Same card, ABBA order `a0 → b0 → b1 → a1`, 20 device events per
   side, discard samples above 1.25×p50, and then the decisive check: the per-shape Δ must exceed
   **that shape's own two-slot drift**. The issue documents a slot-level contamination
   (p50 1651.85 vs 1313.17 on the same shape) that sample-level filtering could not catch. Our
   49.3 vs 165.2 should be re-reported this way before it is quoted again.

The structural argument for our route survives, and it is sharpened: SuperKernel is 「aclgraph 捕获
图之上的二进制融合」 and `early_start` only hides the *scalar head* of each task. It leaves the 44
dispatches, the 346 inter-task gaps and the 25% AIC idle in place. A single kernel removes all of
those. But the 9.18% is the part of that already harvested for free, so our marginal claim is
against ~1.09x, not 1.0x.

## 54. `torch_npu` symmetric memory, HyperParallel `symm_mem`, and the "Mega Kernel" layer — including `task type 105`

Sources, all retrieved 2026-10-20:
- https://github.com/pytorch/pytorch/issues/190628 — `[Distributed][SymmetricMemory] Allow allocators to customize tensor wrapping for externally allocated memory`, author `MrMC-Sqz`, opened 2026-07-21T03:37:22Z
- https://github.com/mindspore-ai/hyper-parallel/issues/658 — `【RFC】HyperParallel Symmetric Memory — 子 Group、完整集合通信、反向传播及 Hccl 互操作`, author `rainyhorse`, opened 2026-09-15T09:31:10Z (mirror of https://gitcode.com/mindspore/hyper-parallel/issues/658; the prior RFC is #59)
- https://gitcode.com/pengqihw/DocsAndTests (branch `main`) — `HP 对称内存能力接入 torch_npu`
- https://docs.pytorch.org/docs/main/symmetric_memory.html — upstream API reference

### 54.1 The torch_npu allocation chain — symmetric memory on NPU IS `aclshmem`

From pytorch#190628, verbatim:

> ### Observed failure
>
> This was observed with an out-of-tree PrivateUse1 NPU backend using an ACLSHMEM-based symmetric-memory allocator.
>
> The allocation path is:
>
> ```text
> torch.distributed._symmetric_memory.empty(device="npu")
>   -> _SymmetricMemory.empty_strided_p2p()
>   -> c10d::symmetric_memory::empty_strided_p2p()
>   -> NPUSHMEMSymmetricMemoryAllocator::alloc()
>   -> Aclshmem_malloc()
>   -> generic at::from_blob()
> ```

> `torch.distributed._symmetric_memory.empty()` allows a device backend to customize symmetric-memory allocation through `SymmetricMemoryAllocator::alloc()`.
>
> However, after the device pointer is allocated, `empty_strided_p2p()` always wraps it using the generic `at::from_blob()` implementation.
>
> An out-of-tree backend can therefore customize:
>
> - allocation
> - deallocation
> - rendezvous
> - multicast capability
>
> but cannot customize how the allocated device pointer is converted into a tensor.

The current upstream code, quoted in the issue:

```cpp
auto allocator = get_allocator(device.type());
void* dev_ptr =
    allocator->alloc(alloc_size, device.index(), group_name);

auto options = at::TensorOptions()
                   .dtype(dtype)
                   .device(device);

return at::from_blob(
    dev_ptr,
    size,
    stride,
    [allocator = std::move(allocator)](void* ptr) {
      allocator->free(ptr);
    },
    options);
```

> Source:
> https://github.com/pytorch/pytorch/blob/b76127025ce802917f4a29e46e4cbddea4626c81/torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp#L263-L292

The concrete NPU failure, verbatim — worth recording because it is the kind of error that costs a
day:

> The NPU backend needs that final Tensor metadata to initialize its storage descriptor. A normal NPU tensor factory performs this initialization, but the generic symmetric-memory wrapping path does not.
>
> A later quantized matrix multiplication operator reads the incomplete storage format metadata and fails with:
>
> ```text
> RuntimeError: unknown format type: 1869833586
> ```

and the proposed fix:

```cpp
class SymmetricMemoryAllocator
    : public c10::intrusive_ptr_target {
 public:
  virtual at::Tensor make_tensor(
      void* ptr,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides,
      c10::ScalarType dtype,
      c10::Device device,
      std::function<void(void*)> deleter = {});

  // Existing allocator APIs...
};
```

### 54.2 HyperParallel `symm_mem` — Push-mode one-sided communication, the primitive set, and the node constraint

From mindspore-ai/hyper-parallel#658, verbatim:

> ### 1.1 背景
>
> 对称内存（Symmetric Memory）基于昇腾芯片内 Device 间共享内存，以 **Push 模式**实现单边通信——发起方可直接通过 Python API 读写远端对称内存，**无需目标 rank 显式参与，消除了传统集合通信的同步屏障和中间数据拷贝**。基础原语（put/get/signal/wait/put_with_signal）及 `shmem_allgather` 等接口设计详见 [#59](https://gitcode.com/mindspore/hyper-parallel/issues/59)。
>
> 当前待补齐的能力：
>
> | 能力 | PyTorch | MindSpore |
> |------|---------|-----------|
> | 对称内存基础原语（put/get/signal/wait/put_with_signal） | ✅ | ✅ |
> | `shmem_allgather` | ✅ | ✅ |
> | `shmem_alltoall` | ✅ | ❌ |
> | MC2 融合算子（AG+MatMul / MatMul+RS） | ✅ | ❌ |
> | 内存分配（empty / Manager.malloc / MemPool） | ✅ | ✅ |

> ### 1.2 现状与问题
>
> 当前实现存在三大能力缺口：
>
> 1. **不支持子 Group**——所有对称内存操作基于全局 `world_size`，无法用于 FSDP/HSDP 中不同 DP group、TP group 等场景。典型需求：DP group 内用对称内存做 AllReduce，TP group 内用 Hccl 做 AllGather，两组互不干扰。
>
> 2. **缺少高级集合通信封装**——用户需自行基于 put/get/signal 原语组装 AllReduce / ReduceScatter / Send / Recv，开发门槛高、容易出错。
>
> 3. **无自动微分支持**——`shmem_allgather` / `shmem_alltoall` 无法在训练中直接使用，因为缺少反向传播实现。

The API surface being added, verbatim:

```python
# 新增 group 参数（所有接口统一）
symm.shmem_put(target, target_offset, src, src_offset, size, target_rank, group=None)
symm.shmem_get(target, target_offset, src, src_offset, size, target_rank, group=None)
symm.shmem_signal_op(signal, signal_offset, signal_value, signal_op, target_rank, group=None)
symm.shmem_put_with_signal(target, target_offset, src, src_offset,
                            size, signal, signal_offset, signal_value, signal_op, target_rank,
                            group=None)
symm.shmem_allgather(output_tensor, input_tensor, group=None)
symm.shmem_alltoall(send_tensor_list, receive_tensor, receive_list, group=None)
symm.shmem_allreduce(tensor, op='sum', group=None)
symm.shmem_reduce_scatter(output_tensor, input_tensor, op='sum', group=None)
symm.shmem_send(tensor, dst_rank, group=None)
symm.shmem_recv(tensor, src_rank, group=None)
```

> **`target_rank` 语义变更**：当指定 `group` 时，`target_rank` 变为 **group 内局部 rank**（0 ~ group.size-1），框架内部负责映射到全局 rank。
>
> **约束**：
> - **`group` 内所有 rank 必须处于同一物理节点（对称内存仅在节点内 Device 间共享）。**
> - 对已创建的 group 缓存其 rank 映射关系，避免重复初始化开销。

The target scenarios, verbatim — note the first one:

> | 场景 | 说明 |
> |------|------|
> | **MoE 通算掩盖** | TP/EP group 内对称内存替代 Hccl，减少同步开销，与多核并行协同 |
> | **FSDP/HSDP 梯度同步** | DP group 内用 `shmem_reduce_scatter` / `shmem_allreduce`，降低延迟 |
> | **序列并行 / Context Parallel** | CP group 内用对称内存通信，与计算流水线重叠 |
> | **张量并行 MC2** | 子 Group MC2，通算掩盖效率进一步提升 |
> | **参数 Broadcast** | `shmem_send` / `shmem_recv` 点对点通信，无 barrier |

Design goals including the precision one:

> | **低精度通信高精累加** | AllReduce / ReduceScatter 支持低精度数据传输 + 高精度本地累加 |
> | **Hccl 混用互操作** | 同一训练任务中 Hccl 集合通信与对称内存通信可安全混用 |
> | **DFX 可观测性** | 内存用量监控、日志分级、**signal 超时检测**、性能 Profiling 集成 |

### 54.3 `DocsAndTests` — the three-layer bridge, and ★ `task type 105`

`README.md` of https://gitcode.com/pengqihw/DocsAndTests, verbatim:

> # HP 对称内存能力接入 torch_npu
>
> 将 Hyper-parallel (HP) 的对称内存三层通信能力接入 torch_npu，用户通过 `torch.ops.symm_mem.*` 统一入口使用全部能力。
>
> ## 快速理解
>
> HP 的三层能力通过**三种不同的桥接方式**接入，**均不产生编译依赖**：
>
> | 层 | 能力 | 桥接方式 | 对 HP 编译依赖 |
> |---|------|---------|:---:|
> | 第 1 层 | 通信原语（put_mem_signal 等 5 个） | `SymmMemBridge` 双向注入（函数指针）；推荐由 HP 独立 bridge .so 暴露 C ABI | 零 |
> | 第 2 层 | 融合算子（fused_all_gather_matmul 等） | 运行时 `from hyper_parallel import` | 零 |
> | 第 3 层 | **Mega Kernel（moe_ffn_fwd/bwd）** | `torch::ops::hyper_parallel::*` dispatch | 零 |
>
> 核心设计是一个独立接口合约 `symm_mem_ops.h`，定义 `SymmMemBridge` 结构体——torch_npu 填入 `memory_ops`（分配/释放），HP 填入 `comm_ops`（通信原语函数指针），双方通过单一 `extern "C"` 入口 `hpshmem_register_ops(&bridge)` 完成握手。

> ## 关键设计决策
>
> - **接口合约**：`SymmMemBridge` 双向注入——torch_npu 给 HP 内存管理能力，HP 给 torch_npu 通信原语能力。双方零编译依赖。
> - **SHMEM 初始化**：保留 HP standalone 路径；启用 torch_npu-managed 模式后，由 torch_npu 负责唯一初始化，HP 跳过自己的 `attr_init`。
> - **.so 运行时选择**：根据配置加载 `libshmem.so`（官方）或 `libhpshmem.so`（HP patch 版本），确保进程只加载一份。
> - **三层独立**：三层各自走不同的桥接路径，第 1 层通过 `bridge.comm_ops`，第 2 层通过 Python import，**第 3 层（Mega Kernel）不经过 bridge——直接通过 CANN 算子内部 task type 105 调用 aclshmem**。

> - **主要风险**：Phase 4 Mega Kernel 的 **CANN vendor 库预加载时序**（问题 Q3）

## Relevance

Three takeaways.

**The NPU symmetric-memory stack is `aclshmem` all the way down.** `torch.distributed._symmetric_memory.empty(device="npu")` → `NPUSHMEMSymmetricMemoryAllocator::alloc()` → `Aclshmem_malloc()`. So §9–§12 is not a parallel universe: it *is* the substrate under torch_npu's symm_mem, under HyperParallel's `symm.shmem_*`, and under whatever MoE Mega Kernel ships on top. If we build on `aclshmem_ptr` + MTE put/atomic-add we are on the same road the vendor's own PyTorch integration is on, which matters for coexistence with whatever the framework allocates.

**★ `task type 105`.** The DocsAndTests design states that the Mega Kernel layer reaches `aclshmem` 「直接通过 CANN 算子内部 **task type 105** 调用」, bypassing the bridge entirely. That is a concrete numeric task type in the launch path, and it is the first public number of its kind we have found. Against our `KERNEL_TYPE_AIC_ONLY` rejection with `sqe_type=3, errType=0x20` — which is a *task-type/SQE* level rejection — a sanctioned task type 105 for "CANN operator that calls aclshmem internally" is the obvious thing to look up next in the runtime's task-type table. It also hints that a kernel doing in-kernel cross-rank communication may need a *different declared task type* than a plain compute kernel, which would be a second reason beyond §15.4's scheduling-module argument for why a naive type declaration gets refused.

**The one-sided semantics are stated plainly and they are what we want**: 「以 **Push 模式**实现单边通信——发起方可直接通过 Python API 读写远端对称内存，**无需目标 rank 显式参与，消除了传统集合通信的同步屏障和中间数据拷贝**」, with the primitive set `put / get / signal / wait / put_with_signal` — exactly §9.5–§9.6's device-side set, surfaced to Python. And the hard constraint is the one we can live with: 「`group` 内所有 rank 必须处于同一物理节点（**对称内存仅在节点内 Device 间共享**）」 — single machine, TP16, which is our target exactly.

Two smaller items worth carrying. 「低精度通信高精累加」 — low-precision transfer with
high-precision local accumulation — is the right shape for a bf16 TP16 all-reduce and argues
against a naive bf16 `atomic_add` chain across 16 ranks on numerical grounds (16 sequential bf16
accumulations into one cell). And 「signal 超时检测」 is listed as a DFX requirement, i.e. the
vendor also expects device-side signal waits to hang and plans for a timeout — cf. §15.6's
`aclrtSynchronizeStreamWithTimeout`.


---

## Gaps

Looked for and did not find:

- **The Hazy Research megakernel's actual device code.** The blog (§1) is reproduced in full
  and the repo's build/run commands are in §1.8, but I did not read
  `demos/low-latency-llama/*` — the instruction templates, the page request/release API, or
  the counter wait/increment primitives. That is the highest-value remaining read for anyone
  copying the design: https://github.com/HazyResearch/Megakernels
- **MPK's in-kernel runtime source.** The paper (§2) describes workers, schedulers, circular
  task queues with `atomicAdd`, and 352-byte task descriptors, but I quoted only the paper.
  The CUDA runtime is at https://github.com/mirage-project/mirage and would give the exact
  queue, semaphore and page-allocator code.
- **Event Tensor / ETC code.** No repository was found; the MLSys 2026 page and arXiv PDF
  were the only sources. The data-dependent `exp_indptr` triggering mechanism (§6.6) is
  described but not shown in code.
- **ThunderMLA's interpreter template.** §5.4 names
  `interpreter::kernel<config, partial_template, reduction_template>` and says the whole thing
  is 250 lines, but the ThunderKittens interpreter header itself was not retrieved.
- **X/Twitter threads.** Searched for technical replies to the megakernel announcements
  (Spector, Juravsky, HazyResearch, mirage-project). The search returned only secondary blog
  coverage and unrelated Spectre results; no thread with technical replies was retrievable.
  If those threads matter, they need a direct handle lookup rather than keyword search.
- **FlashInfer persistent decode.** Searched specifically for a persistent / CUDA-graph-free
  decode path in FlashInfer. The retrievable material describes the opposite design (plan +
  CUDA graph capture, `use_cuda_graph=False` by default on
  `BatchDecodeWithPagedKVCacheWrapper`). Either the persistent path does not exist or it is
  not documented; checking the repo for `persistent` in the decode sources would settle it.
- **CANN SuperKernel's host-side optimizer.** §4 covers `kernel/sk_entry.asc` and
  `src/aot/sk_common.h`; §4b covers the fusion-boundary logic from `sk_node.h`,
  `sk_scope_info.h` and the scope-splitting walk plus core-selection in `sk_scope_split.cpp`
  (78 KB, read selectively). Still not read: `sk_optimizer.cpp` (the top-level `Process()` /
  `Schedule()` flow), `sk_task_builder.cpp` (how the dual queues and the up-to-4 entry copies
  are actually built, including where `entryCnt` is decided), `sk_lock_detector.cpp` and
  `sk_verify_scope_split.cpp` (the deadlock analysis that produces `EXIST_DEADLOCK`), and the
  whole `src/jit/superkernel/*.py` tree. All are at
  `https://raw.gitcode.com/cann/graph-autofusion/raw/master/super_kernel/src/aot/<file>`.
- **Any megakernel number for a MoE model at batch=1 with top-k routing on Ascend.** MPK §6.4
  gives Qwen3-30B-A3B on B200 and FlashMoE gives 8xH100, both NVIDIA. MegaGDN (§8) is a real
  Ascend megakernel with a real number (~15% faster prefill, 1.5–3x kernel) but it is a linear-
  attention prefill workload, not MoE decode. Nothing covers MoE + top-k + batch=1 on Ascend;
  the internal Ascend material is in `ascend-launch-overhead-and-sync.md` and is design
  documentation, not measurement.
- **MegaGDN's remaining sources.** §8 covers `mega_kernel.cpp` and `include/kernel_utils.h`.
  Not read: the per-stage kernels (`chunk_o.cpp` is 66 KB, `wy_kda.cpp` 47 KB,
  `tri_inverse.cpp` 43 KB) which would show how a PTO-ISA stage is written to be *callable from
  inside another kernel*; the Chinese report `blog/mega_gdn_zh.md`; the PTO-ISA `TSyncAll` PR
  (https://gitcode.com/cann/pto-isa/pull/878); and the vllm-ascend / sgl-kernel-npu integration
  PRs (sgl-project/sgl-kernel-npu#462, #517, #533).
- **PTO-ISA itself.** `gitcode.com/cann/pto-isa` is the virtual tile ISA MegaGDN is written
  against, and `CANN/pypto` / `CANN/pto-isa` appear in the GitCode CANN org listing. Whether
  hand-written AscendC or PTO-ISA is the better substrate for our kernel was not evaluated.
