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
