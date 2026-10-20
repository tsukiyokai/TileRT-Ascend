# Low-latency TP reduction — verbatim source material

Collected 2026-10-20 for the GLM-5.2-W4A8C8 / Ascend A3 / TP16 / batch=1 ultra-low-latency effort.
Context this material is judged against: a decode round is 53.05 ms unprofiled; our one-shot push "SuperKernel" TP reduction costs 5.35 ms/round and is 100% serial with compute, while every HCCL collective fully overlaps and is free; per-call launch floors AI_VECTOR_CORE 0.70 us, AI_CORE 2.30 us, MIX_AIV 12.04 us, MIX_AIC 13.58 us; ~3008 kernels/round.

Everything below the heading is quoted verbatim from the named source. Chinese text is left in Chinese.

---

## Part 1 — One-shot / push-based allreduce for tiny messages

### vLLM `custom_all_reduce.cuh` — one-shot (`cross_device_reduce_1stage`) and two-shot kernels

https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh
(vLLM project, `main` as of 2026-10-20; fetched from `raw.githubusercontent.com/vllm-project/vllm/main/csrc/custom_all_reduce.cuh`)

```cuda
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from
  // all ranks.

  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}
```

The strategy selection in the same file (`CustomAllreduce::allreduce`):

```cuda
    const char* env_algo = std::getenv("VLLM_CUSTOM_ALLREDUCE_ALGO");
    bool force_1stage = false;
    bool force_2stage = false;
    if (env_algo != nullptr) {
      if (std::strcmp(env_algo, "1stage") == 0 ||
          std::strcmp(env_algo, "oneshot") == 0) {
        force_1stage = true;
      } else if (std::strcmp(env_algo, "2stage") == 0 ||
                 std::strcmp(env_algo, "twoshot") == 0) {
        force_2stage = true;
      } else {
        throw std::runtime_error(
            "Invalid VLLM_CUSTOM_ALLREDUCE_ALGO: " + std::string(env_algo) +
            ". Valid values: 1stage, oneshot, 2stage, twoshot");
      }
    }
...
    if (force_1stage) {                                 \
      KL(ngpus, cross_device_reduce_1stage);            \
    } else if (force_2stage) {                          \
      KL(ngpus, cross_device_reduce_2stage);            \
    } else {                                            \
      if (world_size_ == 2) {                           \
        KL(ngpus, cross_device_reduce_1stage);          \
      } else if (fully_connected_) {                    \
        if ((world_size_ <= 4 && bytes < 512 * 1024) || \
            (world_size_ <= 8 && bytes < 256 * 1024)) { \
          KL(ngpus, cross_device_reduce_1stage);        \
        } else {                                        \
          KL(ngpus, cross_device_reduce_2stage);        \
        }                                               \
      }                                                 \
    }
```

CUDA-graph note from the same class (why peer pointers cannot be kernel args at capture time):

```cuda
  // Stores rank data from all ranks. This is mainly for cuda graph purposes.
  // For cuda graph to work, all kernel arguments must be fixed during graph
  // capture time. However, the peer pointers are not known during graph
  // capture time. Therefore, during capture, we increment the rank data
  // pointer and use that as the argument to the kernel. The kernel arguments
  // are stored in graph_unreg_buffers_. The actual peer pointers will be
  // filled in at the memory pointed to by the pointers in
  // graph_unreg_buffers_ when the IPC handles are exchanged between ranks.
  //
  // The overall process looks like this:
  // 1. Graph capture.
  // 2. Each rank obtains the IPC handles for each addresses used during cuda
  // graph capture using get_graph_buffer_ipc_meta.
  // 3. (In Python) all gather the IPC handles.
  // 4. Obtain the peer pointers by opening the IPC handles, and store them in
  // the rank data array at corresponding positions.
```

## Relevance
This is the exact structure of our SuperKernel push reduce: barrier, read all 16 peer buffers, sum, barrier. Note the crossover thresholds: at world_size <= 8 vLLM leaves one-shot at **256 KB**, and at 16 ranks (our TP16) the `fully_connected_` branch has no clause at all — upstream vLLM does not use a custom one-shot beyond 8 ranks. Our 16-die one-shot is outside the regime anyone upstream tuned.

---

### vLLM `custom_collective_common.cuh` — the signal/flag exchange and barrier cost

https://github.com/vllm-project/vllm/blob/main/csrc/custom_collective_common.cuh
(vLLM project, `main` as of 2026-10-20)

```cuda
namespace vllm {
constexpr int kMaxCustomCollectiveRanks = 16;

// Maximal number of blocks in allreduce kernel.
constexpr int kMaxBlocks = 36;

// Default number of blocks in allreduce kernel.
#ifndef USE_ROCM
inline constexpr int defaultBlockLimit = 36;
inline CUpointer_attribute rangeStartAddrAttr =
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR;
#else
inline constexpr int defaultBlockLimit = 16;
inline hipPointer_attribute rangeStartAddrAttr =
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR;
#endif

// Counter may overflow, but it's fine since unsigned int overflow is
// well-defined behavior.
using FlagType = uint32_t;

// Two sets of peer counters are needed for two syncs: starting and ending an
// operation. The reason is that it's possible for peer GPU block to arrive at
// the second sync point while the current GPU block haven't passed the first
// sync point. Thus, peer GPU may write counter+1 while current GPU is busy
// waiting for counter. We use alternating counter array to avoid this
// possibility.
struct Signal {
  alignas(128) FlagType start[kMaxBlocks][kMaxCustomCollectiveRanks];
  alignas(128) FlagType end[kMaxBlocks][kMaxCustomCollectiveRanks];
  alignas(128) FlagType _flag[kMaxBlocks];  // incremental flags for each rank
};

struct __align__(16) RankData {
  const void* ptrs[kMaxCustomCollectiveRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxCustomCollectiveRanks];
};
```

```cuda
static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
  #else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
  #endif
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  #else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;"
               : "=r"(flag)
               : "l"(flag_addr));
  #endif
  return flag;
}

// This function is meant to be used as the first synchronization in the all
// reduce kernel. Thus, it doesn't need to make any visibility guarantees for
// prior memory accesses. Note: volatile writes will not be reordered against
// other volatile writes.
template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg,
                              int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x];
    // Write the expected counter value to peer and wait for correct value
    // from peer.
    st_flag_volatile(peer_counter_ptr, flag);
    while (ld_flag_volatile(self_counter_ptr) != flag);
  }
  __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// This function is meant to be used as the second or the final
// synchronization barrier in the all reduce kernel. If it's the final
// synchronization barrier, we don't need to make any visibility guarantees
// for prior memory accesses.
template <int ngpus, bool final_sync = false>
DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->end[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->end[blockIdx.x][threadIdx.x];
    // Write the expected counter value to peer and wait for correct value from
    // peer.
    if constexpr (!final_sync) {
      st_flag_release(peer_counter_ptr, flag);
      while (ld_flag_acquire(self_counter_ptr) != flag);
    } else {
      st_flag_volatile(peer_counter_ptr, flag);
      while (ld_flag_volatile(self_counter_ptr) != flag);
    }
  }
  if constexpr (!final_sync) __syncthreads();

  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}
```

The ROCm branch states the cost model explicitly:

```cuda
template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg,
                              int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    // simultaneously write to the corresponding flag of all ranks.
    // Latency = 1 p2p write
    __scoped_atomic_store_n(&sg.signals[threadIdx.x]->start[blockIdx.x][rank],
                            flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    // wait until we got true from all ranks
    while (__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x],
                                  __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_DEVICE) < flag);
  }
  __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}
```

And the reduce itself:

```cuda
template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}
```

## Relevance
`Signal` is sized `[36][16]` per phase — the flag matrix is per-block × per-rank, i.e. a 16-rank one-shot pays 16 p2p flag writes per block per barrier, and vLLM's own comment prices a barrier at "Latency = 1 p2p write" only because all 16 writes are issued simultaneously by 16 distinct threads. If our 5.35 ms SuperKernel is barrier-bound rather than bandwidth-bound, this is the structure to compare against: two barriers per reduce, one p2p write-round-trip each.

---

### TensorRT-LLM `customAllReduceKernels.h` — strategy enum, Lamport thresholds, rank limits

https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/customAllReduceKernels.h
(NVIDIA, Copyright 2022-2024, `main` as of 2026-10-20)

```cpp
constexpr size_t WARP_SIZE = 32;
constexpr size_t MAX_ALL_REDUCE_BLOCKS = 24;
// Use max modules to avoid overflow and ABA problem when the block num changes for barrier_flag
// Not a perfect solution, but it has large chance that it is correct
constexpr size_t MAX_ALL_REDUCE_MODULES = std::numeric_limits<uint32_t>::max() / 6 * 6;
constexpr size_t MAX_RANKS_PER_NODE = 16;
constexpr size_t DEFAULT_BLOCK_SIZE = 512;

namespace reduce_fusion::details
{
static constexpr int kBytesPerAccess = 16;
static constexpr int kWarpSize = 32;
static constexpr int kMaxCtaSize = 1024;
static constexpr int kClusterMaxSize = 8;
static constexpr int kLamportTokenNumThreshold = 16;
static constexpr int kLamportHiddenSizeThreshold = 256;
}; // namespace reduce_fusion::details

// Warning: python definition is in tensorrt_llm/functional.py
// they must be kept in sync
enum class AllReduceStrategyType : int8_t
{
    NCCL = 0,
    MIN_LATENCY = 1,
    UB = 2,
    AUTO = 3,
    ONESHOT = 4,
    TWOSHOT = 5,
    LOWPRECISION = 6,
    MNNVL = 7,
    NCCL_SYMMETRIC = 8,
};

enum class AllReduceStrategyConfig : int8_t
{
    USE_MEMCPY = 1 << 0,
    PUSH_MODE = 1 << 1,
};

enum class AllReduceFusionOp : int8_t
{
    NONE = 0,
    RESIDUAL_RMS_NORM = 1,
    LAST_PROCESS_FOR_UB = 2,
    RESIDUAL_RMS_PREPOST_NORM = 3,
    RESIDUAL_RMS_NORM_QUANT_FP8 = 4,
    RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5,
    RESIDUAL_RMS_NORM_OUT_QUANT_FP8 = 6,
    RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 = 7,
    MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM = 8,
    RMS_NORM = 9,
};
```

```cpp
struct AllReduceFusionParams
{
    ...
    // gemm bias
    void const* bias_buffer;
    // residuial add
    void const* residual_buffer;
    // rms norm
    int hidden_size;                             // equal to normalized_shape
    void const* weight_buffer;                   // norm elem-wise affine gamma
    void const* weight_buffer_pre_residual_norm; // for gemma norm before residual
    float eps;
    // new residual
    void* intermediate_buffer;
    void* lamport_peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE * 3];
};

struct AllReduceParams
{
    size_t elts_total;
    size_t elts_per_rank;
    size_t elts_per_block;
    size_t rank_offset;
    size_t ranks_per_node;
    size_t local_rank;
    uint32_t* barrier_flag_ptr;
    uint32_t* barrier_flag_counter_ptr;
    uint32_t* peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
    uint32_t* peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
    void* peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE * 2];
    void* local_output_buffer_ptr;
    void const* local_input_buffer_ptr;

    AllReduceFusionParams fusion_params;

    static AllReduceParams deserialize(int64_t* buffer, size_t tpSize, size_t tpRank, tensorrt_llm::DataType dataType,
        int token_num, int hidden_size, AllReduceFusionOp op);
};

bool configurationSupported(AllReduceStrategyType algo, size_t msg_size, size_t n_ranks, tensorrt_llm::DataType type);

void customAllReduce(kernels::AllReduceParams& params, tensorrt_llm::DataType dataType, AllReduceStrategyType strat,
    AllReduceStrategyConfig config, AllReduceFusionOp fusionOp, cudaStream_t stream);

void residualRmsNorm(
    kernels::AllReduceParams& params, tensorrt_llm::DataType dataType, cudaStream_t stream, AllReduceFusionOp fusionOp);

void lamportInitialize(void* buffer, size_t size, tensorrt_llm::DataType dataType, cudaStream_t stream);

namespace reduce_fusion
{
bool is_lamport_supported(tensorrt_llm::DataType dataType, int token_num, int hidden_size);
}
```

## Relevance
`PUSH_MODE` is a first-class config flag alongside `USE_MEMCPY`, and `MIN_LATENCY` is a *separate* strategy from `ONESHOT` — the Lamport (flag-free, value-as-signal) path gated at `kLamportTokenNumThreshold = 16` tokens and `kLamportHiddenSizeThreshold = 256` hidden. Our batch=1 decode is exactly 1 token, deep inside the Lamport-eligible region. Also `MAX_RANKS_PER_NODE = 16` matches our 16 dies, and the reduce has 10 fusion ops available including `RESIDUAL_RMS_NORM` — i.e. NVIDIA fuses the post-reduce epilogue into the collective rather than launching it separately.

---

### TensorRT-LLM perf cookbook — measured Lamport one-shot vs NCCL-symmetric per-call latency

https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-regression-cookbook/references/communication/allreduce-host-overhead-small-model-tp.md
(NVIDIA TensorRT-LLM in-repo regression cookbook; case `case-allreduce-host-overhead-small-model-tp`, nvbug 6262973, commit `c0e6d795c8d4`, PR #16902 merged 2026-08-11)

```text
- **Failed attempts:** PR **#15157** (bot-filed by `tensorrt-cicd`,
  2026-06-09) — still **OPEN** but **superseded by the merged #16902**
  (untouched since 2026-06-22, no human review at all, only a CodeRabbit
  comment; the code it wanted to revert has since been restructured into
  C++). It attacked the same host overhead by *reverting* rather than
  relocating: (1) flip `_NCCL_SYMMETRIC_ZERO_COPY` back from `"1"` to `"0"`
  in `tensorrt_llm/_torch/distributed/ops.py`; (2) drop
  `inputs_pre_hook_register_nccl_symmetric_memory_window` from
  `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` so the AutoTuner
  benchmarks each tactic on the real production input instead of a
  pre-copied window buffer; (3) change the cache-miss fallback from
  `NCCL_SYMMETRIC.value` to `NCCL.value`. Its own diagnosis is worth
  keeping — the post-fix profile still showed "322 ncclSymk calls @ 473us
  = 152ms" against "Lamport one-shot (4347 calls @ 7.3us = 31.6ms)", i.e.
  the `inputs_pre_hook` "hides NCCL_SYMMETRIC's in-kernel
  cudaMemcpyAsync cost" so the tuner picks it for shapes where
  ONESHOT/TWOSHOT would win — and it reported `total_token_throughput`
  `1.785e+04` bad → `1.863e+04` after fix against a `1.926e+04` good, i.e.
  it recovered only part of the gap. Two files (11/−25 and 6/−3 lines) are
  the whole diff. Lesson: reverting
  the zero-copy default was tried and not taken — the accepted fix moved
  the tactic dispatch to C++ and left the default alone.
```

```text
- **Root cause:** host-side, not kernel-side. Two contributions, both
  public: (1) the zero-copy default makes the GEMM output window-resident
  for NCCL_SYMMETRIC (PR #15157), so registered-NCCL-buffer bookkeeping
  crosses the C++/Python boundary per collective, and the AutoTuner's
  `inputs_pre_hook` pre-copies the input into the NCCL window before each
  tactic benchmark, "hiding NCCL_SYMMETRIC's in-kernel cudaMemcpyAsync
  cost" so the tuner picks it where ONESHOT/TWOSHOT would win (PR #15157);
  and (2) a pre-existing tax — under `AllReduceStrategy.AUTO` the per-call
  bucket lookup and tactic dispatch ran in Python
  (`torch.ops.trtllm.tunable_allreduce`) on every AllReduce, which PR
  #16902 moves to C++ "avoiding Python work during inference". Both only
  matter where the model is small enough that the host is the critical
  path; PR #16902 states it "benefits small, CPU-overhead-sensitive models".
```

```text
- **Fix mechanism:** PR #16902 moves the hot-path AllReduce autotuner bucket
  lookup and tactic dispatch out of Python into C++. It adds
  `autotuned_allreduce` plus a thread-safe native tactic cache in
  `cpp/tensorrt_llm/thop/allreduceOp.cpp` (14 fixed token-bucket cutoffs;
  key = tensor group, fusion-op id, NCCL-window vs non-window mode, input
  scalar dtype, static input shape excluding `size(0)`), with
  `register_allreduce_tactic`, `validate_allreduce_tuning_buckets` and
  `clear_allreduce_tactic_cache` as native control ops.
```

## Relevance
The one hard per-call number in the TensorRT-LLM tree: **Lamport one-shot allreduce = 7.3 us/call** (4347 calls, 31.6 ms total) versus **NCCL symmetric = 473 us/call**. 7.3 us is the order of magnitude a tuned one-shot allreduce should cost. Our 5.35 ms per round, spread over the ~60 TP reductions a GLM-5.2 decode round needs, is ~90 us per reduce — more than 12x the Lamport figure. That gap is the budget.

---

### TensorRT-LLM perf cookbook — AUTO keyed on token count alone picked one-shot where NCCL won

https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-regression-cookbook/references/communication/allreduce-strategy-lut.md
(NVIDIA TensorRT-LLM in-repo regression cookbook; case `case-allreduce-strategy-lut`, nvbugs 5540494 / 5531839 / 5517023, commit `fd4311e6a396`, PR #7870)

```text
# AllReduce AUTO keyed only on token count, so it picked a one-shot kernel where NCCL was faster

- **Symptom:** two directions, both in the PR. Regression: "perf regression due to
  using a one-shot kernel instead of NCCL on A100/H100" — AUTO chose a custom
  kernel that lost to plain NCCL on those SMs. Upside on the fixed path: "3-4 %
  perf gain in concurrencies of 256 and 512" on Deepseek-R1.
- **Root cause:** AUTO decided from **token count alone** — a single threshold
  (default 128, overridable) plus a `world_size <= 2` special case — and below the
  threshold always returned `MIN_LATENCY`, i.e. the one-shot custom kernel. The
  inputs it ignored are exactly the ones that determine the crossover:
  `hidden_size` (message bytes per token), the **fusion op** attached to the
  allreduce (which changes how much work the custom kernel is amortising), and the
  **SM architecture** (the custom kernels' advantage over NCCL is
  generation-dependent, and on A100/H100 it can be negative). A one-dimensional
  heuristic over a five-dimensional crossover surface is right in the region it was
  tuned on and wrong everywhere else — and because it is *deterministic*, it is
  wrong reproducibly, which reads as a product regression rather than a tuning gap.
- **Fix mechanism:** a generated lookup table,
  `AllReduceBestStrategyTable[sm][tp][fusion_op][hidden_size][num_tokens]`,
  populated from measurement for SM90 and SM100, with **out-of-range ⇒ NCCL** as
  the safe default. Explicit user requests for `ONESHOT` / `TWOSHOT` are still
  honoured (AUTO is the only path that changes). The kernels themselves also move
  from stronger orderings to `relaxed` stores plus a single
  `fence.release.sys`. What the PR **removes** matters as much: a `TORCH_CHECK`,
  three `TLLM_LOG_WARNING` fallback paths, and the env overrides
  `OVERRIDE_HEURISTIC_ALLREDUCE_STRATEGY` and
  `ALLREDUCE_AUTO_HEURISTIC_MIN_LATENCY_THRESHOLD_TOKEN_NUM` — so a runbook or
  script that sets either variable is silently a no-op after this commit.
- **Detection signal:** the log line was renamed, which dates any log precisely:
  `grep -n "AllReduceOp runtime strategy for rank" <log>` is post-fix,
  `grep -n "AllReducePlugin strategy for rank" <log>` is pre-fix.
```

## Relevance
Two transferable facts. First, "the kernels themselves also move from stronger orderings to `relaxed` stores plus a single `fence.release.sys`" — the memory-ordering strength of the barrier is itself a measurable lever on a tiny-message one-shot. Second, NVIDIA's own conclusion is that a custom one-shot is *not* unconditionally better than the vendor collective; the decision surface is `[sm][tp][fusion_op][hidden_size][num_tokens]`. Our situation — one collective that is 100% serial while HCCL is free — means the comparison on A3 is not even about kernel time but about overlap, which this table does not model.

---

### TensorRT-LLM perf casebook — MNNVL two-shot, Lamport flag sync, multicast loads

https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization-casebook/references/communication/mnnvl-twoshot-allreduce.md
(NVIDIA TensorRT-LLM in-repo optimization casebook; commit `6e1aee6fd68a`, PR #5934)

```text
# Optimize the collective kernel itself (two-shot / Lamport-sync MNNVL AllReduce)

- **Commits:** `6e1aee6fd68a` [fix] Performance Optimization for MNNVL TwoShot Kernel (#5934).
- **Applies when:** communication-bound + signals: TP AllReduce on a multi-node NVLink (MNNVL) domain; medium/large messages where two-shot (reduce-scatter + allgather) beats one-shot; `allreduce_strategy=MNNVL` or `TWOSHOT` already selected and the AR kernel (not the network) is the bottleneck.
- **Mechanism:** tunes the MNNVL two-shot AllReduce CUDA kernel rather than swapping algorithms. Lamport-style flag sync with triple-buffered flags (`buffer_flags % 3`), grid reworked so only a subset of CTAs do the lamport sync; cross-CTA completion via `red.async.release.global.gpu.add.u32` atomics + spin on `offset_access_ptr < gridDim.x*gridDim.y`; data via vectorized `ld.volatile.global.v4.f32`/`v2.f32` over NVLink-multicast buffers (`mcastDeviceMemory`), assuming 8B-atomic writes and `num_token*token_dim` divisible by ELTS_PER_LOAD (4 BF16 / 2 FP32).
- **Generalizes to:** "make the collective kernel faster instead of changing the algorithm" — buffer-flag/Lamport sync, grid-rightsizing, vectorized multicast loads carry to one-shot AllReduce, allgather/reduce-scatter, custom NVLS/multimem collectives; adapt by matching alignment/divisibility assumptions and the flag lifecycle.
- **Apply via:** `allreduce_strategy=MNNVL` (or `TWOSHOT`) in `_torch/model_config.py` (`AllReduceStrategyType`: NCCL=0, MIN_LATENCY=1, UB=2, AUTO=3, ONESHOT=4, TWOSHOT=5, LOWPRECISION=6, MNNVL=7, NCCL_SYMMETRIC=8).
- **Eligibility:** "num_token*token_dim divisible by ELTS_PER_LOAD (4 for BF16, 2 for FP32), as of #5934"
```

And the sibling autotune case:

https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-optimization-casebook/references/communication/shape-aware-allreduce-autotune.md
(commits `22257457824a` PR #7870, `d272f1a9bcbc` PR #8531)

```text
- **Mechanism:** make the algorithm a function of shape. #7870 adds two `AUTO` heuristics in `customAllReduceUtils.h`: an LP threshold (`SelectStrategyLP`: per-`(SM_major, TP_size)` `(NCCL_num_token_threshold, TWO_SHOT_numel_threshold)` → TWOSHOT if `message_size ≥ two_shot_numel_threshold`, else ONESHOT, else NCCL) and a benchmark-derived LUT (`selectStrategyLookUpTable`, indexed `[sm][tp][fusionOp][log2(hidden)-7][log2(tokens)]`) generated offline by `allreduce_heuristic_code_gen.py`. Also fixes that ONESHOT/TWOSHOT couldn't be overridden, fixes a TWOSHOT perf bug, cleans dispatch. #8531 wires it into the runtime AutoTuner via `AllReduceRunner(TunableRunner)` + `trtllm::tunable_allreduce`, with `AutoTuner.choose_one` picking per-profile.
- **Expected effect:** recovers per-shape best AllReduce time, removes the one-shot-vs-NCCL regression. **PR #7870 reports ~3–4% end-to-end gain for DeepSeek-R1 at concurrency 256 and 512** (fixes a known one-shot regression). Re-measure on your shapes.
- **Knobs:** `allreduce_strategy`, `TLLM_DISABLE_ALLREDUCE_AUTOTUNE`, `OVERRIDE_HEURISTIC_ALLREDUCE_STRATEGY`
```

## Relevance
Triple-buffered Lamport flags (`buffer_flags % 3`) and "grid reworked so only a subset of CTAs do the lamport sync" are two concrete, portable reductions in barrier cost. The second is directly applicable: if only some AI cores participate in the cross-die sync, the barrier's fan-in shrinks from (cores × 16) to (sync-cores × 16).

---

## Part 2 — Ascend: symmetric memory, device-side put, compute-comm fusion

### ACLSHMEM / `cann/shmem` — Ascend symmetric-memory communication library (the Ascend answer to NVSHMEM)

https://gitcode.com/cann/shmem
(Huawei / CANN SIG `shmem`; Copyright (c) 2025 Huawei Technologies Co., Ltd.; v1.7.0 released October 2026; docs at https://shmem-doc.pages.dev/)

```text
# SHMEM
Symmetric Memory-based Ascend Distributed Memory Communication Acceleration Library

## What's New

🚀 [October 2026] SHMEM v1.7.0 is now available.

   - Added QP-specific put/get APIs for Ascend 950PR&950DT products RDMA/UDMA, expanded aggregate submission support, and enabled concurrent UDMA communication across multiple links.
   - Added QP-specific put/get APIs for A2/A3 SDMA to support multi-QP communication.
   - Improved installation and runtime diagnostics, with new MegaMoE and UDMA AllReduce examples and multi-QP performance tests.

📌 [July 2026] SHMEM v1.6.0: expanded Ascend 950PR&950DT products RDMA, UDMA/MTE, and SIMT capabilities, and improved Python wheel installation and `shmem-config` diagnostics.

📌 [April 2026] SHMEM v1.3.0: added AI Core direct-driven capabilities and 40+ communication APIs, with expanded DFX tooling.

🔥 [December 2025] Initial launch of the SHMEM project.
```

```text
## 1. Project Introduction
SHMEM is a multi-server, multi-device memory communication library designed for the Ascend platform. By abstracting host-side and device-side APIs, SHMEM enables fast cross-device memory access and data synchronization. Its core benefits include:
- Support for AI Core direct-driven MTE and xDMA, enabling D2D, D2H, H2D, D2rH, and rH2D communication paths
- Simplified inter-device communication logic in distributed workloads, lowering operator development complexity
- Deep integration with the CANN ecosystem, accelerating deployment of computing-communication fusion operators

## 2. Core Functions

**1. Dual-side API system**

- Host side: initialization, memory heap management, communicator (team) creation, and global synchronization
- Device side: remote memory access (RMA), device-level synchronization, and team operations

The API design follows the Ascend operator development paradigm and supports host-device collaboration.

**2. High-performance communication optimization**

- Built-in MTE and xDMA engines for direct remote memory read/write, minimizing latency
- MPI interoperability, supporting collective communication primitives such as AllGather and AllReduce
- Optimized data transfer paths for Ascend hardware features to improve multi-device collaboration efficiency

**4. Communication path coverage**

The following figure shows the full-link communication paths supported by SHMEM (using Ascend 910A3 as an example), covering different transmission engines on the host and device sides:

- **MTE Engine**: chip-level memory transfer engine, supporting D2D, D2H, H2D, D2rH, and rH2D paths
- **xDMA Engine**: high-speed direct memory access engine, supporting efficient intra-host and inter-host data transfer
```

Code structure (verbatim), showing the device-side data planes:

```text
shmem/                                  # Project root directory
├── include/                            # External header files
│   ├── shmem.h                         # All SHMEM external APIs
│   ├── device/                         # Device-side header files
│   │   ├── gm2gm/                      # Data plane APIs gm2gm driven by AI Core
│   │   │   └── engine/                 # Low-level APIs gm2gm directly driven by AI Core
│   │   ├── team/                       # Device-side team management
│   │   └── ub2gm/                      # Data plane APIs ub2gm driven by AI Core
│   │       └── engine/                 # Low-level APIs ub2gm directly driven by AI Core
│   ├── host/                           # Host-side header files
│   │   ├── data_plane/                 # Host-side data plane APIs
│   │   ├── init/                       # Host-side initialization APIs
│   │   ├── mem/                        # Host-side memory management APIs
│   │   ├── team/                       # Host-side team management APIs
│   │   └── utils/                      # Tools and general auxiliary code
│   └── host_device/                    # Shared directory
├── src/                                # Source code implementation
│   ├── device/                         # Device-side implementation
│   │   ├── gm2gm/                      # Data plane APIs gm2gm directly driven by AI Core
│   │   │   └── engine/                 # Low-level APIs gm2gm directly driven by AI Core
│   │   ├── team/                       # Device-side team management
│   │   └── ub2gm/                      # Data plane APIs ub2gm driven by AI Core
│   │       └── mte/                    # Low-level APIs ub2gm directly driven by AI Core
│   ├── host/                           # Host-side implementation
│   │   ├── sync/                       # Synchronization primitives (barrier/p2p/order)
│   │   ├── transport/                  # Transport layer implementation (such as RDMA, SDMA, and UDMA)
```

```text
## 4. Typical Use Cases

**1. Computing-communication fusion operator development**: Develop custom operators that merge compute and communication (such as Matmul + AllReduce) by leveraging device-side direct memory access APIs. This reduces inter-device data copies and improves operator execution efficiency.

**3. Low-latency inter-device communication**: Employ RDMA-optimized device-side APIs to transfer data among devices in milliseconds, meeting the real-time requirements of latency-sensitive AI inference scenarios.
```

TLS is on by default and must be disabled *before* init:

```c
int32_t ret = aclshmemx_set_conf_store_tls(false, NULL, 0);
```

```text
**Q4: Why is an encryption failure still reported after TLS is disabled?**

A: Call `aclshmemx_set_conf_store_tls` before `aclshmemx_init_attr`. The TLS configuration cannot be modified after initialization.
```

Install and build:

```bash
python3 -m pip install cann-shmem \
  --index-url https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/
shmem-config --version   # Query the installed version
shmem-config --diagnose  # Check native library loading and package integrity
```

```bash
git clone https://gitcode.com/cann/shmem.git
cd shmem
bash scripts/build.sh             # A2/A3 platforms
# bash scripts/build.sh -soc_type Ascend950  # Ascend 950PR&950DT products platform
source install/set_env.sh
```

The umbrella header `include/shmem.h`, which enumerates the whole device-side surface:

```c
#ifndef SHMEM_API_H
#define SHMEM_API_H

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#include "device/gm2gm/shmem_device_amo.h"
#include "device/gm2gm/shmem_device_cc.h"
#include "device/gm2gm/shmem_device_mo.h"
#include "device/gm2gm/shmem_device_p2p_sync.h"
#include "device/gm2gm/shmem_device_rma.h"
#include "device/gm2gm/shmem_device_so.h"
#include "device/gm2gm/engine/shmem_device_mte.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "device/gm2gm/engine/shmem_device_sdma.h"
#include "device/gm2gm/engine/shmem_device_udma.h"
#include "device/ub2gm/shmem_device_rma.h"
#include "device/ub2gm/engine/shmem_device_mte.h"
#include "device/shmem_def.h"
#include "device/team/shmem_device_team.h"

// simt
#if defined(USE_SIMT)
#include "device_simt/gm2gm/shmem_device_simt_rma.h"
#include "device_simt/gm2gm/shmem_device_simt_amo.h"
#include "device_simt/gm2gm/engine/shmem_device_simt_mte.h"
#include "device_simt/ub2gm/shmem_device_simt_rma.h"
#include "device_simt/ub2gm/engine/shmem_device_simt_mte.h"
#include "device_simt/team/shmem_device_simt_team.h"
#endif
#endif

#include "host/shmem_host_def.h"
#include "host/mem/shmem_host_heap.h"
#include "host/init/shmem_host_init.h"
#include "host/utils/shmem_host_exception.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/data_plane/shmem_host_so.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "host/data_plane/shmem_host_cc.h"
#include "host/team/shmem_host_team.h"
#include "host/utils/shmem_log.h"

#endif // SHMEM_API_H
```

## Relevance
This is the Ascend equivalent of `torch.distributed._symmetric_memory` + NVSHMEM device-side put, and it exists in the open on CANN. `device/gm2gm/engine/shmem_device_mte.h` and `shmem_device_sdma.h` are AI-Core-driven direct remote writes — i.e. exactly the push primitive our SuperKernel needs, with a maintained `p2p_sync` and device-side `team` barrier next to it. `ub2gm/` matters even more for batch=1: it writes from Unified Buffer straight to a peer's global memory, skipping the local GM round trip.

---

### ACLSHMEM device-side RMA — `aclshmem_*_put_nbi` / `get_nbi` exact signatures (UB ⇄ peer GM)

https://gitcode.com/cann/shmem/blob/master/include/device/gm2gm/shmem_device_rma.h
(Huawei / CANN `cann/shmem`, master as of 2026-10-20; fetched from `raw.gitcode.com/cann/shmem/raw/master/include/device/gm2gm/shmem_device_rma.h`)

Supported element types, as the generator macro:

```c
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

The push primitive — UB straight to a peer PE's symmetric memory, non-blocking:

```c
/**
 * @brief  Automatically generates aclshmem put nbi functions for different data types (e.g., float, int8_t).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src, uint32_t elem_size, int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on Symmetric memory of the destination data.
 * - **src**         - [in] Pointer on local UB of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_PUT_TYPENAME_MEM_UB_NBI(NAME, TYPE)                                                                   \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src, uint32_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_UB_NBI);
```

AscendC tensor overload, and the non-contiguous (strided) forms:

```c
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src, uint32_t elem_size, int32_t pe)
#define ACLSHMEM_PUT_TYPENAME_MEM_UB_TENSOR_NBI(NAME, TYPE)                                                             \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src,     \
                                                 uint32_t elem_size, int32_t pe)

 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_put_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src, const non_contiguous_copy_param &copy_params, int32_t pe)
 * @par Function Description
 * Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE.
```

The pull primitive (peer GM → local UB), for a read-based one-shot:

```c
/**
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_get_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int pe)
 *
 * @par Function Description
 * Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local UB.
 *
 * @par Parameters
 * - **dst**         - [in] Pointer on local UB of the destination data.
 * - **src**         - [in] Pointer on Symmetric memory of the source data.
 * - **elem_size**   - [in] Number of elements in the destination and source arrays.
 * - **pe**          - [in] PE number of the remote PE.
 */
#define ACLSHMEM_GET_TYPENAME_MEM_UB_NBI(NAME, TYPE)                                                               \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int pe)
```

## Relevance
`aclshmem_bfloat16_put_nbi(__gm__ bfloat16_t *dst, __ubuf__ bfloat16_t *src, uint32_t elem_size, int32_t pe)` is the exact one-sided push our SuperKernel needs, callable from inside an AscendC kernel, with the peer selected by `pe` at runtime (no host round trip, no graph-capture pointer problem). The `__ubuf__` source means the partial sum never has to land in local HBM before being pushed.

---

### ACLSHMEM `shmem_device_mte.h` — `aclshmem_ptr`, MTE put/get with explicit UB staging, device-side atomics

https://gitcode.com/cann/shmem/blob/master/include/device/gm2gm/engine/shmem_device_mte.h
(Huawei / CANN `cann/shmem`, master as of 2026-10-20)

```c
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

```c
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

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local PE to symmetric address on the specified PE.
 * ...
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, const non_contiguous_copy_param& copy_params,
    int pe, uint32_t sync_id);

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(
    AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
    uint32_t sync_id);

#define shmem_mte_put_mem_nbi aclshmemx_mte_put_nbi

/**
 * @brief Asynchronous interface. Clear instruction pipes and flush data cache to GM.
 */
ACLSHMEM_DEVICE void aclshmemx_mte_quiet();
```

Device-side remote atomics (the natural reduce-into-peer primitive):

```c
/**
 * @brief Atomic fetch operation. Returns the value at the source address on the specified PE.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_fetch(__gm__ T* src, int32_t pe);

/**
 * @brief Atomic set operation. Sets the value at the destination address on the specified PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_atomic_set(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Atomic compare and swap operation. Conditionally updates the value at the destination address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_compare_swap(__gm__ T* dst, T cond, T value, int32_t pe);

/**
 * @brief Atomic swap operation. Swaps the value at the destination address.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_swap(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Atomic increment operation. Increments the value at the destination address by 1.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_atomic_inc(__gm__ T* dst, int32_t pe);

/**
 * @brief Atomic add operation. Adds the value to the destination address.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_atomic_add(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Atomic fetch increment operation. Increments the value at the destination address by 1 and returns the old
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_fetch_inc(__gm__ T* dst, int32_t pe);

/**
 * @brief Atomic fetch add operation. Adds the value to the destination address and returns the old value.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_mte_atomic_fetch_add(__gm__ T* dst, T value, int32_t pe);
```

## Relevance
`aclshmem_ptr(ptr, pe)` is the Ascend equivalent of the IPC peer-pointer table vLLM has to patch after CUDA-graph capture — it resolves on device, so a captured ACLGraph does not need the peer addresses baked into kernel args. `aclshmemx_mte_atomic_fetch_inc` is enough to build a counter barrier with one remote atomic instead of a 16-wide flag matrix. `aclshmemx_mte_quiet()` is the only completion fence for NBI puts, so it is the thing to count in a latency budget.

---

### ACLSHMEM `shmem_device_cc.h` — grid-wide device-side barrier without returning to host

https://gitcode.com/cann/shmem/blob/master/include/device/gm2gm/shmem_device_cc.h
(Huawei / CANN `cann/shmem`, master as of 2026-10-20)

The programming contract, verbatim, is the single most important paragraph for a persistent-kernel design:

```c
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
```

```c
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

ACLSHMEM_DEVICE void aclshmem_sync_all(void);
```

Deprecations worth noting so the wrong name is not used:

```c
[[deprecated("aclshmemx_barrier_vec is deprecated, please use aclshmemx_sync_vec instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_vec(aclshmem_team_t team);

[[deprecated("aclshmemx_barrier_all_vec is deprecated, please use aclshmemx_sync_vec_all instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_all_vec(void);
```

## Relevance
This is the direct answer to "how can a persistent kernel do a cross-device reduce without returning to host". Three facts bear on our launch-floor numbers: (1) `aclshmem_barrier` / `aclshmem_sync` require a **MIX** kernel with every AIC and AIV core entering — and MIX_AIV/MIX_AIC are our two most expensive launch classes at 12.04/13.58 us, so an all-core barrier forces the expensive launch class; (2) `aclshmemx_sync_vec` is vector-only and explicitly does *not* complete remote updates, which is the cheap barrier if the data movement is already ordered by something else — AI_VECTOR_CORE launches at 0.70 us; (3) "The Cube scalar unit is outside this visibility guarantee" is a correctness trap for fusing a reduce into a cube-heavy GEMM epilogue.

---

### ACLSHMEM `shmem_device_p2p_sync.h` — signal/wait, the Lamport-style primitive on Ascend

https://gitcode.com/cann/shmem/blob/master/include/device/gm2gm/shmem_device_p2p_sync.h
(Huawei / CANN `cann/shmem`, master as of 2026-10-20)

```c
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

The generated per-type wait/test family (all `ACLSHMEM_DEVICE`, over the 13 RMA types):

```c
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

## Relevance
`aclshmem_*_wait_until_all(ivars, nelems=16, ..., ACLSHMEM_CMP_EQ, flag)` is a single-call 16-way arrival wait, and `aclshmemx_signal_op(..., ACLSHMEM_SIGNAL_ADD, pe)` is a remote add. Together they replace vLLM's `[36][16]` flag matrix with one counter word per rank — the cheapest barrier shape available, and the `_test*` variants let the kernel poll without blocking so the barrier can be overlapped with the tail of the local GEMM instead of being serial.

---

### CATCCOS — CANN Templates for Compute-Communication Overlap Subroutines

https://gitcode.com/cann/catccos
(Huawei / CANN; README fetched 2026-10-20)

````text
# CATCCOS

## 📌 简介

CATCCOS(**CA**NN **T**emplates for **C**ompute-**C**ommunication **O**verlap **S**ubroutines)，中文名为昇腾计算-通信融合算子模板库，是一个聚焦于提供高性能计算通信融合类算子基础模板的代码库。

通过抽象分层的方式将计算-通信算子代码模板化。简化通算融合算子开发，解决易用性问题。内存语义实现计算通信细粒度并行，最大化掩盖度；根据计算通信特征，结合硬件架构深度优化，提供极致性能。

本代码仓为CATCCOS代码仓。结合昇腾生态力量，共同设计研发算子模板，并提供典型通算融合算子的高性能实现代码样例。

## 📁 目录结构说明

```bash
catccos
├── 3rdparty    # 依赖的catlass工程文件
├── docs        # 文档（索引见 docs/README.md，算子见 docs/operators.md）
├── examples    # kernel使用样例
├── include     # 模板头文件
├── pre-commit  # CI 代码检查与自动格式化的配置文件
├── tests       # 单元测试与动态 tiling 测试
├── tools       # AscendTimer 打点计时工具
└── utils       # 算子注册、日志、共享内存等通用辅助代码
```

## 💻 软硬件配套说明

- 硬件平台:
  - **CPU**: `aarch64`/`x86_64`
  - **NPU**: `Atlas A2 训练系列产品`/`Atlas 800I A2 推理产品`/`A200I A2 Box 异构组件`/`Atlas 350 加速卡`

- CANN版本:

| CANN包类别 | 版本要求                    |
| ---------- | --------------------------- |
| 社区版     | 8.5.0.alpha002 及之后版本 |
| 商用版     | 8.5.0及之后版本           |

## 🚀 快速上手

以`matmul_allreduce`算子样例为例，快速上手CATCCOS算子开发：

```bash
  # 用于统一配置 CANN、SHMEM、CATLASS 相关环境变量（A2 默认）
  source ./examples/utils/setup.sh

  # Ascend950 算子需指定 soc 类型（首次编译 SHMEM 时生效）
  source ./examples/utils/setup.sh -soc_type Ascend950
```

```bash
cd examples/matmul_allreduce
bash scripts/build.sh
```

```bash
# bash scripts/run.sh <device_list>
# device_list为选择的设备编号，以启动两卡执行任务为例
bash scripts/run.sh 0,1
```
````

## Relevance
CATCCOS is the Ascend analogue of Flux / TileLink: a template library whose stated goal is "内存语义实现计算通信细粒度并行，最大化掩盖度" — memory-semantics fine-grained compute/comm parallelism to maximise overlap — and it ships a `matmul_allreduce` example built on top of SHMEM. This is the supported route to fusing the reduction into the producing kernel instead of launching it serially, which is precisely the 5.35 ms we need to hide. Note it depends on `catlass`, not on HCCL.

---

### vllm-ascend Super Kernel — operator-binary fusion at ACLGraph capture

https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/graph_mode.md
(vLLM Ascend plugin, `main` as of 2026-10-20)

````text
### Super Kernel optimization

[Super Kernel](https://www.hiascend.com/document/detail/en/Pytorch/latest/devguide/TorchAir/docs/en/npugraph_ex/advanced/superkernel.md) is an **optional** operator-binary fusion optimization. Unlike source-level fusion, it works on compiled kernel binaries: eligible subgraphs are identified, their child kernels are combined into a larger kernel, and synchronization is inserted according to graph dependencies. Compared with launching operators individually, this can reduce task scheduling waits, launch overhead, and operator-head overhead.

In vLLM Ascend, Super Kernel optimization is applied during ACLGraph capture. It depends on both static kernel and Npugraph_ex:

```text
Super Kernel -> static kernel -> Npugraph_ex
```

!!! note

    Static kernel is disabled by default, so Super Kernel is also disabled under the default configuration. When `enable_static_kernel` is explicitly set to `true` and `enable_super_kernel` is omitted, Super Kernel follows static kernel and is enabled. Set `enable_super_kernel` explicitly to override this inherited behavior. Super Kernel cannot be enabled when static kernel is disabled.

    Not every operator is eligible for Super Kernel fusion. An unsupported operator may split the fusion range, so the actual coverage and performance benefit depend on the model and operator sequence. Changing the Super Kernel setting also changes the static-kernel compilation configuration and may cause the static-kernel cache to be rebuilt.
````

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true, "enable_static_kernel":true}}'
```

````text
#### Verifying Super Kernel is active

When Super Kernel optimization is applied during ACLGraph capture, vLLM Ascend emits:

```text
Super kernel optimization is enabled for ACL graph capture.
```

This message confirms that the optimization step was invoked. To verify the resulting fusion coverage and performance benefit, collect an **Ascend Profiling** trace and inspect `kernel_details.csv`. Compare the kernel/task entries and latency with Super Kernel disabled; the profiler also reports the launch core count for generated Super Kernels in the `Block Dim` field.
````

https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/configuration/additional_config.md

```text
| `enable_npugraph_ex`               | bool | `True` | Whether to enable npugraph_ex backend.                                                 |
| `enable_static_kernel` | bool | `False` | Whether to enable static kernel. Suitable for scenarios where shape changes are minimal and some time is available for static kernel compilation. |
| `enable_super_kernel` | bool | Inherits `enable_static_kernel` | Whether to enable Super Kernel optimization. When omitted, it follows `enable_static_kernel`; set it explicitly to override that behavior. Super Kernel requires static kernel. |
| `fuse_norm_quant`  | bool | `True` | Whether to enable fuse_norm_quant pass. |
| `fuse_qknorm_rope` | bool | `True` | Whether to enable fuse_qknorm_rope pass. If Triton is not in the environment, set it to False. |
| `fuse_muls_add` | bool | `True` | Whether to enable fuse_muls_add pass.|
```

## Relevance
"reduce task scheduling waits, launch overhead, and operator-head overhead" is the official framing of what Super Kernel buys — and at ~3008 kernels a round with a 0.70–13.58 us per-call launch floor, launch overhead alone is in the tens of milliseconds. Note the hard dependency chain `Super Kernel -> static kernel -> Npugraph_ex`, and the warning that an unsupported operator *splits the fusion range* — one ineligible op in the attention-to-MLP stretch costs the whole fusion.

---

### vllm-ascend's own GLM-5.2 A3 16-die super-kernel configuration (closest public analogue to our target)

https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/cases/models/configs/GLM/GLM-5.2-W8A8C8-A3_128k-static-super-kernels.yaml
(vLLM Ascend plugin; header says "Optional configuration based on PR #17871; intentionally not registered in weekly_config.yaml")

```yaml
# Optional configuration based on PR #17871; intentionally not registered in weekly_config.yaml.
test_name: "multi-node-GLM-5.2-W8A8C8-MTP-A3_128k-static-super-kernels"
model: "Eco-Tech/GLM-5.2-w8a8c8"
num_nodes: 2
npu_per_node: 16
dp_load_balancing: internal
env_common: &env_common
  LOCAL_WORLD_SIZE: "16"
  HCCL_OP_EXPANSION_MODE: "AIV"
  OMP_PROC_BIND: "false"
  VLLM_USE_MODELSCOPE: "true"
  OMP_NUM_THREADS: "1"
  HCCL_BUFFSIZE: "400"
  PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
  VLLM_ASCEND_ENABLE_FLASHCOMM1: "1"
  VLLM_ENGINE_READY_TIMEOUT_S: "3000"
  SERVER_PORT: 8077

# TODO: need to identify why TP and mtp+1 divisibility rules break on dual-node case

deployment:
  - envs:
      <<: *env_common
    server_cmd: >
      vllm serve Eco-Tech/GLM-5.2-w8a8c8
      --host 0.0.0.0
      --port $SERVER_PORT
      --data-parallel-size 8
      --data-parallel-size-local 4
      --data-parallel-address $LOCAL_IP
      --enable-expert-parallel
      --data-parallel-rpc-port 12981
      --tensor-parallel-size 4
      --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'
      --seed 1024
      --tool-call-parser glm47
      --reasoning-parser glm45
      --enable-auto-tool-choice
      --max-num-seqs 6
      --max-model-len 133120
      --max-num-batched-tokens 4096
      --trust-remote-code
      --gpu-memory-utilization 0.92
      --quantization ascend
      --enable-chunked-prefill
      --enable-prefix-caching
      --async-scheduling
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
      --kv-cache-dtype int8
      --attention_config.indexer_kv_dtype int8
      --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "ascend_compilation_config": {"fuse_muls_add": true, "enable_npugraph_ex": true, "enable_static_kernel": true, "enable_super_kernel": true}, "multistream_overlap_shared_expert": true,"enable_fused_mc2":1,"enable_mlapo":true}'
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'
```

## Relevance
Huawei's own reference GLM-5.2 configuration on A3 uses `HCCL_OP_EXPANSION_MODE: "AIV"` (collective expansion executed on vector cores, not AICPU), `HCCL_BUFFSIZE: "400"`, `enable_fused_mc2: 1`, `multistream_overlap_shared_expert: true`, and `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`. It also pairs super kernels with MTP speculative decoding at `num_speculative_tokens: 3` and `enforce_eager: true` for the drafter. Note it chooses `--tensor-parallel-size 4` with DP8/EP rather than TP16 — the public reference does not run TP16 for GLM-5.2.

---

### vllm-ascend comm-fusion and finegrained-TP knobs

https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/configuration/additional_config.md

```text
| `mc2_comm_alg`                      | str  | `""`    | set dispatch/combine op's `comm_alg` param, only supports `""/"fullmesh"/"hierarchy"/"fullmesh_v2"`. `"hierarchy"` is only supported by A2/A3, and `"fullmesh_v2"` is only supported by A3 now. |
| `enable_mc2_hierarchy_comm`         | bool | `False` | Enable dispatch/combine op inter-node communication by ROCE. This param will be deprecated and be replaced by mc2_comm_alg = "hierarchy" |
| `enable_prefill_mc2`                | bool | `False` | Whether to reserve mc2_token_capacity for prefill batches. When enabled, `max_num_batched_tokens` is used to calculate the mc2_token_capacity instead of the decode-only capacity. In this scenario, the recommended maximum value of `max_num_batched_tokens` is `tp_size * 512`. This is a temporary switch; once MC2 operators are complete for all scenarios, this switch will be removed and MC2 will be enabled by default. |
| `enable_fused_mc2`                  | int  | `0`     | Fused MC2 configuration. `0` disables the fused path, `1` selects dispatch-FFN-combine, and `2` selects CANN MegaMoe when the model and parallel configuration support it. On A5, SiTU models require a CANN MegaMoe wrapper exposing `activation` and `activation_params`; both SiTU parameters are bound from the MoE configuration during initialization. The legacy `VLLM_ASCEND_ENABLE_FUSED_MC2` environment variable is no longer supported. |
| `enable_flashcomm1`                 | bool | `False` | Whether to enable SP MoE. See [Sequence Parallelism](../feature_guide/sequence_parallelism.md). |
| `mega_moe_max_tokens`               | int  | `65536` | Reference per-rank token capacity after dispatch in the fused MC2/MegaMoe path. It is passed as `dispatch_ffn_combine`'s `max_output_size` and CANN MegaMoe buffer's `max_recv_token_num`. |
| `combine_quant_mode`                | int  | `0`     | Fused MC2 configuration. This configuration will be passed as the `comm_quant_mode` argument for the `torch_npu.npu_moe_distribute_combine_v2` operator. |
```

**finegrained_tp_config**

```text
| `oproj_tensor_parallel_size`     | int  | `0` | The custom tensor parallel size of `o_proj`.     |
| `mlp_tensor_parallel_size`       | int  | `0` | The custom tensor parallel size of `mlp`.       |
| `lmhead_tensor_parallel_size`    | int  | `0` | The custom tensor parallel size of `lm_head`.    |
| `embedding_tensor_parallel_size` | int  | `0` | The custom tensor parallel size of `embedding`. |
```

```python
{
    "finegrained_tp_config": {
        "oproj_tensor_parallel_size": 8,
        "mlp_tensor_parallel_size": 8,
        "lmhead_tensor_parallel_size": 8,
        "embedding_tensor_parallel_size": 8,
    },
    "enable_kv_nz": False,
    "multistream_overlap_shared_expert": True,
```

## Relevance
`finegrained_tp_config` is the structural lever that reduces the *number* of 16-way reductions: shrinking `oproj_tensor_parallel_size` / `mlp_tensor_parallel_size` below the global TP turns a 16-way allreduce into an 8-way one plus an allgather, and at batch=1 the reduce is latency- not bandwidth-bound, so halving the fan-in halves the barrier fan-in too.

---

### `cann/asc-comm` — `AscendC::Hcomm` and `AscendC::Ain`: AIV-direct-driven kernel-side one-sided comm (the Ascend NVSHMEM-device-API equivalent)

https://gitcode.com/cann/asc-comm
(Huawei / CANN, "asc-comm项目首次上线" 2026/07; Copyright (c) 2026 Huawei Technologies Co., Ltd.; README fetched 2026-10-20)

```text
# asc-comm

面向昇腾AI处理器通信场景，提供Hcomm与Ain通信API、AIV直驱实现、样例和验证用例

### 🚀 当前能力

- 提供AICore侧Hcomm点对点通信接口，覆盖普通`Init`、`ReadNbi`、`WriteNbi`、`WriteWithNotifyNbi`、`AtomicFAA`、`AtomicCAS`、`Commit`、`Drain`，以及Ascend 950 UB_CTP路径的`MakeBatchHandle`、批量读写、`BatchCommit`和批量`Drain`。
- 提供AIV直驱Hcomm RoCE和UB_CTP/URMA相关实现，主实现位于`src/aicore/hcomm/`。
- 提供AICore侧Ain单边通信接口，覆盖`Put`、`PutValue`、`Get`、`Signal`、`ReadSignal`、`WaitSignal`、`Flush`、`FlushAsync`、`Wait`，以及`AinBarrierSession`集合通信同步原语，主实现位于`src/aicore/ain/`。
- 提供Hcomm UT工程，覆盖`ascend950pr_9599_AIV`的RoCE/URMA普通接口和UB_CTP批量接口，以及`ascend910B1_AIC`基础接口用例。
- 提供SIMT URMA `WriteNbi`、`WriteValueNbi`、`ReadNbi`、`WriteWithNotifyNbi`、`AtomicFAA`、`AtomicCAS`和`Drain`，以及`simt_urma`功能样例和`simt_urma_perftest`性能样例。

## 🚀概述

asc-comm是面向昇腾AI处理器通信场景的开源仓，当前用于承载AICore侧公开API、AIV直驱设备侧实现、API文档、样例和验证能力。

当前公开能力包括`AscendC::Hcomm`点对点通信和`AscendC::Ain`单边通信，面向算子Kernel侧通信数据路径。Hcomm侧使用方通过`AscendC::Hcomm`模板选择通信协议：普通接口通过`ChannelHandle`逐条提交通信任务；Ascend 950 UB_CTP路径还可以通过BatchHandle添加多个通信任务，并通过`BatchCommit`统一提交。两种流程分别通过对应的`Drain`重载管理完成等待。Ain侧使用方通过`AscendC::Ain`模板基于对称窗口（Symmetric Window）发起`Put`/`Get`/`Signal`等单边操作，通过`Flush`或`FlushAsync`+`Wait`管理完成等待。
```

Protocol capability table (verbatim) — this is the A3-vs-950 dividing line:

```text
| 协议 | 能力说明 |
| --- | --- |
| `COMM_PROTOCOL_ROCE` | RoCE点对点通信路径，支持普通`ReadNbi`、`WriteNbi`、`Commit`、`Drain`，不支持`WriteWithNotifyNbi`和BatchHandle接口。 |
| `COMM_PROTOCOL_UB_CTP` | UBC CTP/URMA点对点通信路径，支持普通`ReadNbi`、`WriteNbi`、`WriteWithNotifyNbi`、`AtomicFAA`、`AtomicCAS`、`Commit`、`Drain`；Ascend 950还支持BatchHandle批量接口。 |
```

````text
### 如何使用Hcomm接口

Hcomm Kernel侧使用时包含如下头文件：

```cpp
#include "hcomm/hcomm.h"
```

普通接口调用流程如下：

1. 创建`AscendC::Hcomm`对象，并选择通信协议。
2. 调用`Init`初始化临时工作区。
3. 通过`ReadNbi`、`WriteNbi`、`WriteWithNotifyNbi`、`AtomicFAA`或`AtomicCAS`提交通信任务。
4. 如果提交任务时设置`commit = false`，调用`Commit`显式提交通信任务。
5. 调用`Drain`等待通道上的通信任务完成。
````

````text
### 如何使用Ain接口

Ain Kernel侧使用时包含如下头文件：

```cpp
#include "ain/ain.h"
```

基本调用流程如下：

1. 创建`AscendC::Ain`对象，绑定通信上下文索引。
2. 通过`Put`/`PutValue`/`Get`发起单边读写，或通过`Signal`发起远端原子信号操作。
3. 如果提交任务时设置`AIN_COMMIT_DELAYED`，提交将被延迟，直到后续`AIN_COMMIT_IMMED`任务触发敲门铃；否则立即提交。
4. 完成等待可通过`Flush`或`FlushAsync`+`Wait`管理。
5. 通过`ReadSignal`/`WaitSignal`读取或等待本地信号。
6. 如需集合同步，通过`AinBarrierSession`的`Sync`完成同步。

提交模式说明：

| 模式 | 行为说明 |
| --- | --- |
| `AIN_COMMIT_IMMED` | 组装通信任务后立即敲响门铃，提交给底层引擎执行。 |
| `AIN_COMMIT_DELAYED` | 仅组装通信任务，不敲门铃；延迟到后续`AIN_COMMIT_IMMED`任务触发提交。 |
````

## Relevance
`AIN_COMMIT_DELAYED` is the doorbell-batching lever: 16 peer puts assembled with no doorbell, then one `AIN_COMMIT_IMMED` rings once. On the UB_CTP path `BatchCommit` does the same at the Hcomm layer. For a 16-die one-shot push this collapses 16 submission events into one — but note the table says BatchHandle is **Ascend 950 only** and `WriteWithNotifyNbi` is **not supported on `COMM_PROTOCOL_ROCE`**, which is the A3-relevant protocol. On A3 the available push is plain `WriteNbi` + `Commit` + `Drain`.

---

### `cann/asc-comm` `include/aicore/hcomm/hcomm.h` — exact kernel-side primitive signatures

https://gitcode.com/cann/asc-comm/blob/master/include/aicore/hcomm/hcomm.h
(Copyright (c) 2026 Huawei Technologies Co., Ltd.)

```cpp
namespace AscendC {

/*!
 * @class Hcomm
 * @brief This class mainly provides a series of point-to-point communication primitive interfaces,
 *        benchmarking against Huawei's point-to-point communication C++ interface,
 *        including WriteNbi、ReadNbi and so on.
 *        The typical usage of this class is as follows:
 *          1) Create Hcomm object.
 *          2) Initialize communication channel.
 *          3) Launch comm tasks asynchronously through the corresponding interface,
 *             and the server starts assembling and launching comm tasks as soon as it listens.
 *          4) If commit is false when launching task, call the Commit interface to notify the execution of the
 *             corresponding comm task.
 *          5) Call the Drain interface (blocking) to wait for the server to complete the corresponding comm task.
 * @tparam commProtocol: The communication protocol to use. COMM_PROTOCOL_UB_CTP (implemented through URMA) is used
 *                       by default.
 */
template <CommProtocol commProtocol = COMM_PROTOCOL_UB_CTP>
class Hcomm {
public:
    /*!
     * @brief Initialize Hcomm workspace.
     * @param [in] buff: The UB buffer provided by caller.
     * @param [in] len: The buffer length in bytes.
     * @return 0 indicates success and -1 indicates failure.
     * @note URMA uses buff as its temporary workspace after 32-byte alignment.
     */
    __aicore__ inline int32_t Init(__ubuf__ uint8_t* buff, uint32_t len);
```

```cpp
    /*!
     * @brief The task launching interface of the Write point-to-point communication operator.
     *        (task content: Write data of length len from src to dst through the specified channel.)
     * @tparam commit: true/false true: commit the task immediately; false: do not commit immediately.
     * @tparam commitPipe: The pipe type to use for commit, PIPE_S supported as default.
     * @tparam reqPipe: The pipe type to use for req, PIPE_MTE3 supported as default.
     * @tparam config: URMA WQE control config, only used by URMA. Default: strongly ordered + fence + CQE enabled.
     * @param [in] channel: The handle of the communication channel.
     * @param [out] dst: The destination address of the data.
     * @param [in] src: The source address of the data.
     * @param [in] len: The length of the data to write, using byte as the basic unit. Must be less than 256 MB
     *                  (256 * 1024 * 1024 bytes).
     * @return 0 indicates success and -1 indicates failure.
     * @note Must be called after channel initialization.
     */
    template <
        bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
        auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len);
```

The remote-reduce push — note the platform restriction in the last `@note`:

```cpp
    /*!
     * @brief The task launching interface of the write-with-reduce point-to-point communication operator.
     *        The source data is reduced into the destination data on the remote side.
     * @tparam T: The element data type. int8_t, int16_t, int32_t, uint32_t, half, float and bfloat16_t are supported.
     * @tparam reduceOp: The reduction operation.
     * @tparam commit: true/false true: commit the task immediately; false: do not commit immediately.
     * @tparam commitPipe: The pipe type to use for commit, PIPE_S supported as default.
     * @tparam reqPipe: The pipe type to use for req, PIPE_MTE supported as default.
     * @tparam config: URMA WQE control config, only used by URMA. Default: strongly ordered + fence + CQE enabled.
     * @param [in] channel: The handle of the communication channel.
     * @param [out] dst: The remote destination address of the reduction.
     * @param [in] src: The local source address of the reduction.
     * @param [in] count: The number of elements to reduce.
     * @return 0 indicates success and -1 indicates failure.
     * @note Only the UB_CTP/URMA path supports this interface.
     */
    template <
        typename T, HcommUrmaReduceOp reduceOp, bool commit = true, pipe_t commitPipe = PIPE_S,
        pipe_t reqPipe = PIPE_MTE3, auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteReduceNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t count);
```

```cpp
    /*!
     * @brief The task launching interface of the Write-with-notify point-to-point communication operator.
     * ...
     * @param [in] notifyAddr: The remote notify address.
     * @param [in] notifyVal: The remote notify value.
     */
    template <
        bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
        auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteWithNotifyNbi(
        ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len, GM_ADDR notifyAddr, uint64_t notifyVal);

    /*!
     * @brief The task launching interface of the Fetch-and-add point-to-point communication operator.
     * @tparam T: The data type of the atomic operation. Only int32_t, uint32_t, int64_t, uint64_t is supported.
     * ...
     * @param [out] fetchAddr: The address to store the old value before atomic add.
     * @param [in] addVal: The remote add value.
     */
    template <
        typename T, bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
        auto const& config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t AtomicFAA(ChannelHandle channel, GM_ADDR dst, GM_ADDR fetchAddr, T addVal);

    /*!
     * @brief Informed that tasks submitted on channel can be executed.
     * @tparam pipe: The pipe type to use for commit, PIPE_S supported as default.
     */
    template <pipe_t pipe = PIPE_S>
    __aicore__ inline int32_t Commit(ChannelHandle channel);

    /*!
     * @brief Submit all tasks in the current batch of a batch handle.
     */
    template <typename T, typename HandleTraits<T>::ChannelType* = nullptr>
    __aicore__ inline int32_t BatchCommit(T& batchHandle);

    /*!
     * @brief Block Aicore and drain comm tasks submitted on channel until finish processing.
     * @tparam pipe: The pipe type to use for drain, PIPE_MTE3 supported as default.
     * @param [in] channel: The handle of the communication channel.
     * @return 0 indicates success. A non-zero value indicates failure. For COMM_PROTOCOL_UB_CTP, the underlying
     *         CQ polling error code is returned directly.
     */
    template <pipe_t pipe = PIPE_MTE3>
    __aicore__ inline int32_t Drain(ChannelHandle channel);
```

Cross-AI-Core channel locking (Ascend 950 only):

```cpp
    /*!
     * @brief Acquire the cross-AI-Core lock of a channel.
     * @note This interface is supported only by COMM_PROTOCOL_UB_CTP on Ascend 950. It blocks until the lock is
     * ...
     * @brief Flush the channel state and release its cross-AI-Core lock.
     * @note This interface is supported only by COMM_PROTOCOL_UB_CTP on Ascend 950. It must be called by the AI Core
```

## Relevance
`WriteNbi` has an explicit `commit` template parameter, and the commit/req pipes are template-selectable (`PIPE_S` / `PIPE_MTE3`). That makes the submission cost of a 16-way push statically controllable from inside the kernel: issue 16 `WriteNbi<false>`, then one `Commit`, then one `Drain`. `WriteReduceNbi` would do the whole allreduce in one primitive (remote reduce-into-destination) — but it is UB_CTP/URMA only, i.e. **not available on our A3**; on A3 the reduce has to stay local after a plain push.

---

### `cann/asc-comm` `include/aicore/ain/ain.h` — `Put` / `PutValue` / `Signal` / `WaitSignal` / `AinBarrierSession`

https://gitcode.com/cann/asc-comm/blob/master/include/aicore/ain/ain.h
(Copyright (c) 2026 Huawei Technologies Co., Ltd.)

```cpp
namespace AscendC {
/*!
 * @brief Get a pointer to a remote peer's memory location within a symmetric window.
 * @param [in] peer: Peer rank id within the team.
 * @param [in] window: Symmetric window handle.
 * @param [in] offset: Byte offset into the window.
 * @return An opaque memory handle pointing to the remote peer's location.
 */
AIN_DEVICE HcommMemHandle GetPeerPointer(uint32_t peer, HcclCommSymWindow window, size_t offset);

/*!
 * @class Ain
 * @brief This class provides device-side one-sided communication primitives (put/get) layered on top
 *        of the Hcomm point-to-point engine. It resolves the per-peer communication channel from the
 *        device communication table, translates symmetric-window handles into remote/local addresses,
 *        and submits tasks immediately or defers submission until a subsequent immediate-commit task.
 *        Pending tasks can be awaited via Flush() or FlushAsync()/Wait().
 */
template <unsigned CommEngineMask = AIN_MASK_DEFAULT>
class Ain {
public:
    AIN_DEVICE Ain(uint32_t contextIndex = 0);
    AIN_DEVICE ~Ain();

    /*!
     * @brief Drain every peer channel in the team, blocking until all pending tasks complete.
     */
    AIN_DEVICE void Flush(HcommTeamHandle team);

    /*!
     * @brief Get the channel handle of a given peer without blocking, for deferred wait.
     * @note Each rank currently has only one channel; channel index 0 is used.
     */
    AIN_DEVICE void FlushAsync(HcommTeamHandle team, uint32_t peer, ChannelHandle* outChannelHandle);

    /*!
     * @brief Block until all tasks submitted on the given channel are complete.
     */
    AIN_DEVICE void Wait(ChannelHandle& channelHandle);

    /*!
     * @brief Issue a one-sided write (put) from a local window to a remote peer's window.
     * @tparam RemoteAction: Optional remote-side action tag, AinRemoteNone by default.
     * @tparam DescriptorUbuf: UB workspace descriptor type, AinDescriptorUbuf by default.
     * @tparam CommitFlags: AIN_COMMIT_IMMED rings the doorbell immediately;
     *                      AIN_COMMIT_DELAYED defers submission until a subsequent AIN_COMMIT_IMMED task.
     * @tparam Config: URMA WQE control config, only used by URMA. Default: strongly ordered + fence + CQE enabled.
     * @param [in] remoteAction: Remote signal action fired after write; AinRemoteNone by default.
     */
    template <
        typename RemoteAction = AinRemoteNone, typename DescriptorUbuf = AinDescriptorUbuf,
        AinCommitFlags CommitFlags = AIN_COMMIT_IMMED, auto const& Config = URMA_DEFAULT_CFG>
    AIN_DEVICE void Put(
        HcommTeamHandle team, uint32_t peer, HcclCommSymWindow dstWin, uint64_t dstOffset, HcclCommSymWindow srcWin,
        uint64_t srcOffset, uint64_t bytes, RemoteAction remoteAction = RemoteAction{},
        const DescriptorUbuf& ubuf = DescriptorUbuf{});

    /*!
     * @brief Issue an inline one-sided write (put) of a typed value to a remote peer's window.
     * @tparam RemoteAction: Signal action type, for example AinSignalInc, AinSignalAdd.
     * @tparam Config: URMA WQE control config, only used by URMA.
     *         Default: strongly ordered + fence + CQE + inline enabled.
     */
    template <
        typename T, typename RemoteAction = AinRemoteNone, typename DescriptorUbuf = AinDescriptorUbuf,
        AinCommitFlags CommitFlags = AIN_COMMIT_IMMED, auto const& Config = URMA_INLINE_CFG>
    AIN_DEVICE void PutValue(
        HcommTeamHandle team, uint32_t peer, HcclCommSymWindow dstWin, uint64_t dstOffset, T value,
        RemoteAction remoteAction = RemoteAction{}, const DescriptorUbuf& ubuf = DescriptorUbuf{});

    /*!
     * @brief Issue a one-sided read (get) from a remote peer's window into a local window.
     */
    template <
        typename DescriptorUbuf = AinDescriptorUbuf, AinCommitFlags CommitFlags = AIN_COMMIT_IMMED,
        auto const& Config = URMA_DEFAULT_CFG>
    AIN_DEVICE void Get(
        HcommTeamHandle team, uint32_t peer, HcclCommSymWindow dstWin, uint64_t dstOffset, HcclCommSymWindow srcWin,
        uint64_t srcOffset, uint64_t bytes, const DescriptorUbuf& ubuf = DescriptorUbuf{});

    /*!
     * @brief Atomically update a remote peer's signal via Fetch-and-Add.
     * @param [in] action: Signal action; AinSignalInc adds 1, AinSignalAdd adds a custom value.
     */
    template <
        typename RemoteAction, typename DescriptorUbuf = AinDescriptorUbuf,
        AinCommitFlags CommitFlags = AIN_COMMIT_IMMED, auto const& Config = URMA_DEFAULT_CFG>
    AIN_DEVICE void Signal(
        HcommTeamHandle team, uint32_t peer, RemoteAction action, const DescriptorUbuf& ubuf = DescriptorUbuf{});

    /*!
     * @brief Read the local signal value, masked to the lower `bits` bits.
     * @param [in] bits: Number of low-order bits to read (1-64).
     * @param [in] order: Memory order, defaults to AIN_MEMORY_ORDER_RELAX.
     */
    AIN_DEVICE uint64_t ReadSignal(
        HcommTeamHandle team, HcclCommSymWindow signalWindow, size_t signalOffset, uint32_t bits = 64,
        AinMemoryOrder order = AIN_MEMORY_ORDER_RELAX) const;

    /*!
     * @brief Block until the local signal reaches at least `least` (masked to `bits` bits).
     */
    AIN_DEVICE void WaitSignal(
        HcommTeamHandle team, HcclCommSymWindow signalWindow, size_t signalOffset, uint64_t least, uint32_t bits = 64,
        AinMemoryOrder order = AIN_MEMORY_ORDER_RELAX) const;

private:
    static constexpr CommProtocol commProtocol = MaskToCommProtocol(CommEngineMask);
    uint32_t contextIndex_ = 0;
    Hcomm<commProtocol> hcomm_;
};
```

```cpp
template <unsigned CommEngineMask = AIN_MASK_DEFAULT>
class AinBarrierSession {
public:
    /*!
     * @brief Construct a barrier session over the given team at the given barrier resource index.
     * @param [in] index: Barrier resource index; each index isolates an independent barrier.
     */
    AIN_DEVICE AinBarrierSession(Ain<CommEngineMask>* ain, HcommTeamHandle team, uint32_t index);

    /*!
     * @brief Synchronize with all peers in the team, blocking until completion.
     * @param [in] order: Memory order, defaults to AIN_MEMORY_ORDER_RELAX.
     * @note Implements a ring barrier: Signal every other rank, then WaitSignal from each.
     */
    template <typename DescriptorUbuf = AinDescriptorUbuf>
    AIN_DEVICE void Sync(AinMemoryOrder order, const DescriptorUbuf& ubuf = DescriptorUbuf{});

    /*!
     * @brief Synchronize with all peers in the team with a timeout.
     * @return 0 on success, -1 on timeout.
     */
    template <typename DescriptorUbuf = AinDescriptorUbuf>
    AIN_DEVICE int32_t
    Sync(AinMemoryOrder order, uint64_t timeoutCycles, const DescriptorUbuf& ubuf = DescriptorUbuf{});

private:
    AinBarrierSessionImpl<CommEngineMask> barrierImpl_;
};
```

## Relevance
`AinBarrierSession::Sync` is documented as "Implements a ring barrier: Signal every other rank, then WaitSignal from each" — i.e. exactly vLLM's `barrier_at_start` shape, but as a supported CANN device API with a timeout variant and `AinMemoryOrder` selectable down to `AIN_MEMORY_ORDER_RELAX`. `Put` with `RemoteAction = AinSignalInc` fuses the data push and the arrival signal into one submitted task, which removes one of the two barrier round trips from a one-shot. `GetPeerPointer` plus `index`-isolated barrier resources means several independent reductions can be in flight with separate barriers — relevant when a round needs ~60 reductions and we want them pipelined rather than serialized.

---

### `torch_npu.npu_mm_all_reduce_base` (MC2 MatmulAllReduce) — signature, constraints, and the explicit "do not use in decode" rule

https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_mm_all_reduce_base.md
(Huawei Ascend op-plugin, tag 7.3.0)

```text
# torch_npu.npu_mm_all_reduce_base

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    | √  |

## 功能说明

- API功能：TP切分场景下，实现mm和all_reduce的融合，融合算子内部实现计算和通信流水并行。

- 计算公式：
    $$
    output = allreduce\left(x1 \mathbin{@} \left((x2 + antiquantOffset) * antiquantScale\right) + bias + x3\right)
    $$

> [!NOTE]
> 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 函数原型

```
torch_npu.npu_mm_all_reduce_base(x1, x2, hcom, *, reduce_op='sum', bias=None, antiquant_scale=None, antiquant_offset=None, x3=None, dequant_scale=None, pertoken_scale=None, comm_quant_scale_1=None, comm_quant_scale_2=None, comm_turn=0, antiquant_group_size=0) -> Tensor
```
```

```text
- **hcom** (`str`)：必选参数。通信域handle名，通过`get_hccl_comm_name`接口获取。
- **reduce_op** (`str`)：可选参数。reduce操作类型，当前版本仅支持`sum`，默认值：`sum`。
- **comm_quant_scale_1** (`Tensor`)：可选参数。alltoall通信前后的量化、去量化系数。支持`float16`、`bfloat16`，支持$ND$格式。`x2`为$[k, n]$时shape为$[1, n]$或$[n]$，用户需保证每张卡上数据保持一致且正确。
- **comm_quant_scale_2** (`Tensor`)：可选参数。allgather通信前后的量化、去量化系数。支持`float16`、`bfloat16`，支持$ND$格式。`x2`为$[k, n]$时shape为$[1, n]$或$[n]$，用户需保证每张卡上数据保持一致且正确。
- **comm_turn** (`int`)：可选参数。表示rank间通信切分粒度，默认值：`0`，表示默认的切分方式。当前版本仅支持输入`0`。
```

```text
## 约束说明

- 该接口支持推理场景下使用。
- 增量场景不使能该融合算子，全量场景使能该融合算子。
- 该接口支持图模式。
- 输入`x1`可为2维或者3维、`x2`必须是2维，分别为$(b, s, k)/(m, k)$, $(k, n)$，$k$轴满足mm算子入参要求，$k$轴相等。`bias`当前仅支持一维，且维度大小与`output`的最后一维大小相同。`x3`的shape与`output`的shape相同。
- `x1`不支持输入转置后的tensor，`x2`转置后输入，需要满足shape的第一维大小与`x1`的最后一维相同，满足matmul的计算条件。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 数据类型支持`bfloat16`。
    - `x1`、`x2`不支持为空tensor。
    - 支持1、2、4、8卡，并且仅支持hccs链路all mesh组网。
    - 非量化场景下，$m、k、n$的取值范围均为`[1, 2147483647]`。
    - `comm_quant_scale_1`，`comm_quant_scale_2`的shape应保持一致，dtype与输出的dtype保持一致，且只在全量化场景支持。
- Atlas A2 训练系列产品：一个模型中的通算融合算子（AllGatherMatmul、MatmulReduceScatter、MatmulAllReduce），仅支持相同通信域。
- 在长序列场景，随着$b/s$或者$m$的增大，可能出现内存不足或者计算超时。
```

## Relevance
Three hard blockers for our case, stated by Huawei: (1) "增量场景不使能该融合算子，全量场景使能该融合算子" — MC2 MatmulAllReduce is explicitly *not* to be enabled in the incremental/decode scenario, which is exactly batch=1 TPOT; (2) "支持1、2、4、8卡" — 1/2/4/8 dies only, so TP16 is out of spec; (3) the three MC2 fusion ops in one model must share one communication domain. So the supported MC2 path does not reach our configuration and a custom kernel on top of `asc-comm`/`shmem` is the remaining route.

---

### HCCL / vllm-ascend environment variables that move the collective off the AI CPU

https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/optimization_and_tuning.html
(vLLM Ascend plugin documentation, "Optimization and Tuning"; also mirrored in zh-cn)

Quoted from the page's variable list (the values below are the page's own recommendations):

```text
HCCL_OP_EXPANSION_MODE  — Sets communication scheduling to AI vector cores over RoCE instead of AI CPU.  Recommended: "AIV"
HCCL_INTRA_ROCE_ENABLE  — Uses RDMA instead of SDMA links between two 8P nodes for mesh interconnect.
HCCL_RDMA_TC            — Sets the traffic class of the RDMA NIC.
HCCL_RDMA_SL            — Sets the service level of the RDMA NIC.
HCCL_BUFFSIZE           — Sets the cache size used to share data between two NPUs.
TASK_QUEUE_ENABLE       — Optimizes the operator dispatch queue. It can raise peak memory use and may hurt performance when memory is tight.  Recommended: 2
CPU_AFFINITY_CONF       — Improves performance for CPU-bottlenecked models and keeps NPU-bottlenecked models at similar performance.  Recommended: 1
PYTORCH_NPU_ALLOC_CONF  — expandable_segments:True releases communication-stream memory early so the compute stream can reuse it.
```

And the Chinese edition of the same page, on `HCCL_OP_EXPANSION_MODE`:

> 在该模式下，通信改由 AI 向量核心经 RoCE 发起调度，不再由 AI CPU 负责下发。

Huawei's own HCCL_BUFFSIZE guidance, quoted from the CANN environment-variable reference:

> 对于集合通信算子，当数据量超过 HCCL_BUFFSIZE 的取值时，性能可能会下降。建议将 HCCL_BUFFSIZE 的值设置为大于数据量。

## Relevance
`HCCL_OP_EXPANSION_MODE=AIV` moves collective *scheduling* from the AI CPU to vector cores — which is exactly why HCCL collectives on A3 can overlap with compute for free in our measurement while our own kernel does not. Our 5.35 ms SuperKernel is serial because it occupies the same cores as compute; HCCL's AIV expansion runs the collective's control flow on cores the compute kernel is not using. Any replacement must preserve that property, not just lower the kernel time.

---

## Part 3 — Barrier-free one-shot: the strongest measured results in the literature

### "Every Microsecond Matters: Achieving Near Speed-of-Light Latency in GPU Collectives"

https://arxiv.org/abs/2607.16100 · https://arxiv.org/html/2607.16100v1
Siyuan Shen, Anton Korzh, John Bachan, Tiancheng Chen, Arnav Goel, Ludwig Schneider, Pouya Kousha, Zhenhao He, Sylvain Jeaugey, Kamil Iskra, Nishank Chandawala, Jeff R. Hammond, Torsten Hoefler. Submitted 17 Jul 2026.

Abstract, verbatim:

> GPU collective communication is typically optimized for bandwidth, yet many emerging workloads are increasingly limited by latency. Long-context decode-heavy large language model (LLM) inference is a prime example, where serving large models requires multiple GPUs, and many small collectives lie directly on the critical path of token generation. Therefore, even μs of overhead can impact performance and cost. In this work, we study how to approach the hardware Speed-of-Light (SoL) lower bound for GPU collectives within a scale-up network. We identify key principles for near-optimal designs, including barrier-free synchronization and efficient use of symmetric memory and multicast. Building on NCCL's device-side API, we develop low-latency interfaces for constructing custom collective kernels and use them to implement new symmetric collectives in NCCL. Microbenchmarks show substantial latency reductions for small and medium messages, reducing overhead to within 7% of the absolute SoL lower bound. When integrated into real applications, these kernels improve inter-token latency and throughput in LLM inference and accelerate cuSOLVERMp, demonstrating benefits for both AI inference and traditional HPC workloads.

The headline measurement:

> Figure 1 illustrates this effect for the long-context, small-batch decode setting that we target. On 4 GB200 GPUs, the NCCL low-latency kernel we introduce in this work reduces average latency for small messages from 11.0 μs for NCCL ring to 2.37 μs, yielding an 8.7% ITL reduction for the inference workload of Llama-3.1-70B. Using CoreWeave's on-demand price of $42/hour for 4 GB200 [8] and converting output throughput into cost per 1M output tokens, the measured data implies that each μs removed from AllReduce latency reduces cost by about 0.9%.

Push vs pull, stated as a rule:

> In a one-shot AllReduce, the entire reduction is completed in a single communication phase, where each GPU fetches data from all peers, performs the reduction locally, and writes the result to the output. In pull mode, GPUs read remote data via loads, while in push mode they write data to remote buffers before reducing locally. Push is generally faster, requiring only half a GPU-to-GPU RTT versus a full RTT for remote loads, but needs additional buffering for incoming data. Consequently, push-based one-shot designs are often preferred for latency-sensitive collectives.

> For N ranks reducing M bytes of data, one-shot incurs O(N·M) total communication volume, whereas two-shot reduces this to O(M).

The barrier-cost measurement — the key number for our 5.35 ms:

> After examining several one-shot and two-shot AllReduce implementations in state-of-the-art frameworks and libraries [6, 12, 3, 30], we observe that, regardless of push or pull communication, these designs typically rely on explicit memory barriers to synchronize peers and signal data readiness at the thread-block level. Using NCCL's ncclLsaBarrierSession, which implements a memory barrier for load-store accessible (LSA) devices, we measure the overhead of such synchronization under relaxed memory ordering and report the results in Fig. 3. Although NCCL's implementation may not be fully optimized, alternative designs follow the same fundamental pattern: a flag is propagated to all peers, and each GPU waits until it observes the corresponding signals from every other participant. Therefore, it is representative of the inherent cost in such approaches.

> The measured barrier latency shows that each barrier will incur more than 1 μs of overhead. In many AllReduce kernels, two such barriers are required. As illustrated in Fig. 1, when a small-message AllReduce completes in roughly 5 μs on four GPUs, two barrier calls alone will account for about 40% of the total latency. As emphasized earlier, when targeting near SoL performance, every μs matters. Thus, eliminating these barriers entirely can yield substantial performance gains.

The three barrier-elimination mechanisms, verbatim:

> IV-1 LL
>
> The first technique we introduce is LL, short for low latency, which originates from the LL protocol in NCCL [11] and is also used in libraries such as NVSHMEM [12] and MSCCL++ [30]. Conventional synchronization signals data arrival using explicit flags and enforced ordering between data and signals. LL removes the signaling step by packing the 8-byte flag with the 8-byte data and transmitting them atomically with 16-byte atomic stores, allowing the receiver to determine data readiness by checking the flag directly. This design halves the effective payload bandwidth and doubles scratch buffer usage, making LL mostly suitable for very small messages.
>
> IV-2 Sentinel
>
> Instead of embedding a signal in the transmitted data, the receiving scratch buffer can be initialized with a sentinel value that is unlikely to appear in valid computations, such as the floating-point value -NaN. Data is then written directly to the buffer, and the receiver polls until the value changes from the sentinel, indicating that valid data has arrived. Compared to LL, this approach preserves full effective bandwidth and uses less scratch space, making it more efficient for moderately larger messages. However, it has a few drawbacks. First, unlike LL, which updates its flag each iteration, the sentinel method requires the buffer to be reset before reuse, complicating buffer management. Second, transmitted values must never match the sentinel because a matching value would prevent the receiver from detecting data arrival. Users must therefore exclude such values to ensure correctness.
>
> IV-3 Bidirectional Communication & Double Buffering
>
> Although LL and sentinel synchronization eliminate explicit memory barriers for single exchanges, they are insufficient when messages require multiple iterations due to limited buffer space. In such cases, inputs are partitioned and processed in chunks. Conventional approaches insert barriers between iterations to prevent buffer overwrites. We can eliminate them by using bidirectional communication and double buffering.
>
> Fig. 4 demonstrates the mechanism. Consider two ranks whose scratch space cannot hold all input data simultaneously. In step ①, both ranks begin with the scratch buffer set to Buffer 0. In step ②, they exchange their first chunks C_r^0. Because Rank 0 processes data faster, it completes the reduction R and writes the result to its output while Rank 1 is still processing. In step ③, Rank 0 advances to the next iteration and switches to Buffer 1 to broadcast chunk C_0^1. It then waits for C_1^1 before proceeding, ensuring that data in Buffer 0 is not overwritten before Rank 1 finishes reading it.
>
> This bidirectional exchange, combined with double buffering, ensures that a rank cannot overwrite a peer's buffer for the next iteration until it has received data from that peer in the current one. This assumes that each rank communicates with a given peer at most once per iteration, avoiding multiple stores to the same remote address. In effect, each receive from a peer serves as an implicit permission for the next send, analogous to credit-based flow control. This mechanism allows multiple reduction iterations without costly global memory barriers.

Table I (algorithm trade-offs), reconstructed exactly as published — columns are AllReduce Algorithm / Comm. Volume per GPU / Latency, # Synchronizations / Scratch Space per Iter / Deterministic:

```text
One-shot (LL)             2(N-1)M            1    2ND
One-shot (Sentinel)       (N-1)M             1    ND
Two-shot (LL)             4(N-1)M/N          2    2D
Two-shot (Sentinel)       2(N-1)M/N          2    D
Two-shot (LL128 Atomic)   ≈2(N-1)M/N         2    ≈D/N
```

The SoL derivation and the measured constants:

> In this section, we describe how the SoL lower bound can be estimated. We define the SoL as the minimal data movement required to complete the AllReduce while ignoring all other overheads, such as instruction scheduling and computation. Since small-message latency is determined by the transfer of the smallest unit handled by the memory system, we focus on the movement of a single 128-byte cache line.

> Combining the components above, the SoL latency of an AllReduce can be expressed as
> L_SoL = 2 L_L2_RTT + L_remote_store.
> Under the SoL assumption, sending data to N peers incurs the same latency as sending to a single peer. Therefore, L_SoL represents an absolute lower bound that is independent of the number of ranks involved.

> On two GB200, we measured the L_L2_RTT and L_remote_store as 0.306 µs and 0.792 µs, respectively. Therefore, the SoL latency of an AllReduce is computed to be 1.404 µs.

Scratch-buffer sizing (measured):

> When the buffer is small (< 8 MiB), two-shot kernels are slower than one-shot kernels. Each iteration of the two-shot design requires two synchronization steps, which dominate latency. Although fewer iterations are needed overall, the additional synchronization cost outweighs this benefit. As the buffer size increases, the performance of two-shot kernels improves rapidly and then plateaus once the entire message fits within a single iteration. This indicates that, ideally, achieving the best performance for two-shot kernels requires a scratch buffer large enough to process the data in one iteration.
>
> In one-shot kernels, every rank writes to all peers simultaneously, which means that increasing the buffer size increases the amount of data sent per iteration, which raises instantaneous bandwidth pressure on the NVLink fabric. Hence, beyond a certain point, increasing the scratch size provides little benefit and can even slightly degrade performance.
>
> Based on these results, we select 4 MiB of scratch buffer for one-shot kernels, as they show limited benefit from larger buffers. For two-shot and LL128 atomic kernels, we choose 64 MiB as the default.

Results versus rank count — the part that matters for TP16:

> Our LLBuffer-based one-shot AllReduce achieves the lowest latency for small messages across all GPU counts. As shown in Fig. 11, our kernels incur only about 7% overhead over the SoL bound at 2 GPUs, while competing implementations remain noticeably farther away. Even at 64 GPUs, multicast one-shot variants stay within roughly 70% overhead of the SoL bound. While NCCL AGxLL and MSCCL++ also use LL-style synchronization, our implementation benefits from targeted optimizations including aggressive compile-time unrolling and parallel polling and reduction across ranks, which reduce detection and accumulation latency. We also observe a crossover between LL and sentinel synchronization. LL performs slightly better at very small sizes, whereas sentinel becomes preferable as message size and rank count increase, avoiding the flag overhead of LL.

> After extensively benchmarking the kernels, we modify NCCL to select the best-performing low-latency algorithm based on the collected empirical results. For example, at 4 ranks, messages below 1 MiB use one-shot kernels, while those between 1 and 2 MiB use two-shot kernels.

> Across all models, low-latency kernels consistently improve performance. The best configuration reduces ITL by 7–13% on 4 GPUs and 9–11% on 8 GPUs, with similar gains in throughput. Results hold across diverse model architectures, including dense (Llama), mixture-of-experts (DeepSeek), and hybrid attention (Qwen3-Next). NCCL LL alone performs well because decode steps involve small AllReduce operations, where one-shot LLBuffer kernels approach the SoL bound.

The API design note (why thread-level, and when barriers are unavoidable):

> The low latency APIs presented here are intentionally fine grained. Unlike NVSHMEM, which provides APIs at the thread, warp, and block levels, our design exposes only thread level primitives. This choice maximizes user control and allows implementations to achieve the lowest possible latency. Moreover, the APIs favor bidirectional communication patterns to achieve safety and optimal performance. As an example, algorithms with independent per-iteration communication, such as one-shot and two-shot AllReduce, can operate without barriers because each step uses disjoint buffers and each rank both sends and receives within the same iteration. In contrast, algorithms with inter-step dependencies and without bidirectional communication, such as ring AllReduce, still require explicit memory barriers. In a ring schedule, each rank must receive a chunk before reducing and forwarding it, as advancing early risks overwriting unconsumed data.

> Using the proposed APIs, we implement one-shot, two-shot, and two-shot LL128 atomic AllReduce, along with other collectives. Taking AllReduce as an example, algorithms can be selected via NCCL_SYM_KERNEL: AllReduce_LLBuffer for one-shot, AllReduce_LLBuffer_Twoshot for two-shot, and AllReduce_LL128_Atomic for the LL128 atomic variant. The synchronization mode is controlled by NCCL_SYM_LLBUFFER_SYNC, allowing users to switch between LL and sentinel mechanisms for the same algorithm. Additionally, two-shot AllReduce requires symmetric (LSA) output buffers, while one-shot only relies on symmetric scratch buffers initialized by NCCL and has no such requirements.

Limitations of the LL128-atomic variant:

> The algorithm has several limitations. It requires NVLink to guarantee cache-line–level atomic addition and supports only single- and half-precision types due to the availability of vectorized atomics in CUDA [40, 41]. It is also limited to addition, as the embedded flag relies on commutativity, and CUDA does not provide vectorized atomic multiplication. In practice, this is not too restrictive since addition dominates most target workloads, such as LLM inference. Finally, the algorithm is non-deterministic because floating-point atomic ordering is not guaranteed. A one-shot variant is possible, but different ranks may observe different results, which violates AllReduce semantics [16].

## Relevance
This paper is the closest published analogue of our problem: latency-bound small allreduce on the decode critical path, scale-up domain, batch=1-ish. Three things transfer directly. (1) A barrier costs >1 us and two barriers are ~40% of a 5 us allreduce — so our SuperKernel's cost is almost certainly synchronization, not data. (2) The fix is barrier *elimination*, not barrier optimization: LL (flag packed into the 16-byte atomic store), sentinel (-NaN pre-fill, poll for change), and bidirectional double buffering as credit-based flow control. The ACLSHMEM/`asc-comm` primitives above support all three — `PutValue` with inline config is the LL analogue, `WaitSignal` with `AIN_MEMORY_ORDER_RELAX` is the poll. (3) The scaling evidence is cautionary for TP16: 7% over SoL at 2 GPUs becomes ~70% over SoL at 64 GPUs even with multicast, so at 16 dies the achievable floor is several times the 2-rank floor, and a hierarchical split (see `finegrained_tp_config`) deserves measurement.

---

### PyTorch `torch.distributed._symmetric_memory` — operator maturity table, `one_shot_all_reduce`, and the Triton one-shot kernel

https://docs.pytorch.org/docs/main/distributed/symmetric_memory.html
(PyTorch documentation; "Created On: Oct 24, 2025 | Last Updated On: Oct 15, 2026")

```text
Note

torch.distributed._symmetric_memory is currently in alpha state and under
development. Its Python APIs and every operator in torch.ops.symm_mem may
change without notice. A dispatcher-visible operator is not necessarily a
user-facing API. The maturity table below identifies which operators are
intended for direct use and which are implementation details.
```

The maturity table, verbatim (Operations / Maturity / Scope or requirements / When to use them):

```text
empty, rendezvous, get, and get_mem_pool
  Alpha | General | Allocate, establish, access, and reuse symmetric memory.

fused_all_gather_matmul and fused_matmul_reduce_scatter
  Alpha | General | Overlap an unscaled matrix multiplication with a tensor-parallel collective. Ordinary accelerator tensors are accepted; the operators manage a symmetric workspace. These are also targets of the compiler's tensor-parallel fusion pass.

fused_all_gather_scaled_matmul and fused_scaled_matmul_reduce_scatter
  Experimental | Hardware-specific | FP8 or other scaled matrix multiplication when the accelerator supports aten._scaled_mm. Their interfaces closely follow compiler lowering requirements.

one_shot_all_reduce, one_shot_all_reduce_out, and two_shot_all_reduce_
  Alpha | Hardware-specific | Low-latency, direct-access all-reduce on a symmetric tensor.

multimem_all_reduce_ and multimem_all_gather_out
  Alpha | Hardware-specific | NVIDIA systems with multicast and multimem support, such as NVLink SHARP.

multimem_one_shot_all_reduce, multimem_one_shot_all_reduce_out, and multimem_one_shot_reduce_out
  Experimental | Hardware-specific | Compiler-oriented multimem variants. Their reduction order is not fixed, so all ranks are not guaranteed to receive bitwise-identical results.

get_remote_tensors
  Alpha | Hardware-specific | Create local tensor views of peer allocations when every peer is directly addressable. The views require explicit cross-rank synchronization.

nvshmem_*, all_to_all_vdev*, tile_reduce, and multi_root_tile_reduce
  Alpha | Backend-specific | NVSHMEM or rocSHMEM transport and device-driven collectives. Availability differs by operation and SHMEM implementation.

put_signal and wait_signal
  Experimental | Backend-specific | One-sided signaling; currently implemented only for the NCCL backend.

_low_contention_*, _async_input_mm, _rendezvous, and _barrier
  Internal | Compiler-facing | Compiler and dispatcher implementation details.

stream_write_value32_, memset32_, memcpy_to_multicast_, and raw nccl_* operators
  Internal | Backend implementation | Kernel composition or backend dispatch. Use a documented Python wrapper instead.
```

The motivation section, verbatim — it is the design rationale for exactly what we are building:

> With rapidly evolving parallelization techniques, existing frameworks and libraries often struggle to keep up, and developers increasingly rely on custom implementations directly scheduling communications and computations. In recent years we've witnessed a shift from primarily relying on one-dimensional data-parallelism techniques to multi-dimensional parallelism ones. The latter have different latency requirements for different types of communications and thus require fine-grained overlapping of compute and communications.
>
> To minimize compute interference, they also require the use of copy engines and network interface cards (NICs) to drive communication. [...] This increase in variety indicates the need for finer-grained communication primitives than are offered today by high-level collective APIs, ones that would enable developers to implement specific algorithms tailored for their use cases, such as low-latency collectives, fine-grained compute-communications overlap, or custom fusions.

> PyTorch Symmetric Memory unlocks three new capabilities:
>
> Customized communication patterns: Increased flexibility in kernel writing allows developers to write custom kernels that implement their custom computations and communications, directly tailored to the need of the application. [...]
>
> In-kernel compute-comm fusion: Device-initiated communication capability allows developers to write kernels with both computation and communication instructions, allowing for the fusion of computation and data movement in the smallest possible granularity.
>
> Low-latency remote access: Network transport protocols like RDMA enhance the performance of symmetric memory in networked environments by enabling direct, high-speed, and low-latency communication between processors and memory. [...]

API usage, verbatim:

```python
import torch.distributed._symmetric_memory as symm_mem

t = symm_mem.empty(128, device=torch.device("cuda", rank))
hdl = symm_mem.rendezvous(t, group)
```

> The symm_mem.empty function creates a tensor that is backed by a symmetric memory allocation. The rendezvous function establishes a rendezvous with peers in the group, and returns a handle to the symmetric memory allocation. The handle provides method to access information related to the symmetric memory allocation, such as pointers to symmetric buffer on peer ranks, multicast pointer (if supported), and signal pads.
>
> The empty and rendezvous functions must be called in the same order on all ranks in the group.

```python
# Most SymmMem ops are under the torch.ops.symm_mem namespace
torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group)
```

```text
Access to buffer pointers, multimem pointer, and signal pads is provided via:

hdl.buffer_ptrs
hdl.multicast_ptr
hdl.signal_pad_ptrs
```

> PyTorch Symmetric Memory provides CUDA Graph-compatible synchronization primitives that operate on the signal pad accompanying each symmetric memory allocation.

The reference one-shot allreduce kernel in Triton, verbatim:

```python
@triton.jit
def one_shot_all_reduce_kernel(
    buf_tuple,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasSubsequenceMemAccess=True
    )

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    while block_start < numel:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.bfloat16)

        for i in tl.static_range(world_size):
            buffer_rank = buf_tuple[i]
            x = tl.load(buffer_rank + offsets, mask=mask)
            acc += x

        tl.store(output_ptr + offsets, acc, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True
    )
```

> Synchronizations at the beginning and the end of the kernel above guarantee that all the processes see consistent data. [...] Visit https://github.com/meta-pytorch/kraken/blob/main/kraken to see additional utilities and examples of using symmetric memory to implement common patterns in Triton.

One-sided get:

```python
src = symm_mem.empty(1024, device=device)
hdl = symm_mem.rendezvous(src, group)

if dist.get_rank(group) == 0:
    dst = torch.empty((512,), device=device)
    # Copy the last 512 elements of the peer's allocation into dst.
    symm_mem.get(dst, hdl, peer=1, offset=512)
```

Memory-pool route, which avoids rewriting allocations:

```python
    mempool = symm_mem.get_mem_pool(device)
    ...
    torch.ops.symm_mem.one_shot_all_reduce(x, "sum", group_name)
```

NCCL symmetric kernels:

```text
NCCL Symmetric Kernels

Note

Requires NCCL 2.27 or later and a single NVLink domain (every rank reachable
over direct NVLink).

NCCL 2.27+ added a family of device kernels — "SymK" internally — written
specifically for symmetric, window-registered buffers. Because each rank knows
every peer's buffer address up front, these kernels skip the generic
proxy/ring machinery and instead use LL (low-latency), multimem/NVLS, and TMA
variants. NCCL picks one per call from message size, so the same
dist.all_reduce gets a latency-optimized kernel for small messages and a
bandwidth-optimized one for large ones.
```

```python
# `symm=True` registers each segment with `ncclCommWindowRegister` using
# `NCCL_WIN_COLL_SYMMETRIC`, which is what makes the symmetric kernels
# eligible. The default `symm=False` performs ordinary user-buffer
```

## Relevance
The Triton kernel is a 30-line reference one-shot: sync, load from all `world_size` peer pointers, accumulate, store, sync. That is a template we can mirror in AscendC over `aclshmem_ptr` / `GetPeerPointer`. Note PyTorch's structural split matching our problem: `one_shot_all_reduce` for the small/latency case and `fused_matmul_reduce_scatter` for the fused case, as two different operators — upstream does not try to make one kernel do both. Also note "CUDA Graph-compatible synchronization primitives", which is the property we need for ACLGraph capture of the reduce.

---

### FLUX (ByteDance) — fusing the collective into the GEMM kernel

https://arxiv.org/abs/2406.06858 · https://arxiv.org/html/2406.06858v1 · code: https://github.com/bytedance/flux (mirror https://gitee.com/ByteDance/flux)
Li-Wen Chang, Wenlei Bao, Qi Hou, Chengquan Jiang, Ningxin Zheng, Yinmin Zhong, Xuanrun Zhang, Zuquan Song, Chengji Yao, Ziheng Jiang, Haibin Lin, Xin Jin, Xin Liu (ByteDance / Peking University).

Abstract, verbatim:

> Large deep learning models have demonstrated strong ability to solve many tasks across a wide range of applications. Those large models typically require training and inference to be distributed. Tensor parallelism is a common technique partitioning computation of an operation or layer across devices to overcome the memory capacity limitation of a single processor, and/or to accelerate computation to meet a certain latency requirement. However, this kind of parallelism introduces additional communication that might contribute a significant portion of overall runtime. Thus limits scalability of this technique within a group of devices with high speed interconnects, such as GPUs with NVLinks in a node. This paper proposes a novel method, Flux, to significantly hide communication latencies with dependent computations for GPUs. Flux over-decomposes communication and computation operations into much finer-grained operations and further fuses them into a larger kernel to effectively hide communication without compromising kernel efficiency. Flux can potentially overlap up to 96% of communication given a fused kernel. Overall, it can achieve up to 1.24x speedups for training over Megatron-LM on a cluster of 128 GPUs with various GPU generations and interconnects, and up to 1.66x and 1.30x speedups for prefill and decoding inference over vLLM on a cluster with 8 GPUs with various GPU generations and interconnects.

Mechanism, as reported:

> Each dependent computation and communication tile is mapped into a thread block, with optimizations such as kernel fusion and tile coordinate swizzling.
> ReduceScatter communication is fused into the epilogue of the GEMM kernel.
> The paper also notes that fusing the AlltoAll step into the GEMM epilogue is typically enough to overlap communication, while the reduction fusion gives only marginal gains.
> Existing overlap approaches have trouble with precise control over execution timing of operations on GPUs when using streams, and splitting GEMMs into smaller kernels often underutilizes the GPU's stream processors.
> It is built on NVIDIA CUTLASS, and reports results on A100 and H800 GPUs across PCIe and NVLink interconnects.

## Relevance
"1.30x speedups for decoding inference over vLLM" is the only published decode-phase number for GEMM-epilogue collective fusion, and it is on 8 GPUs. The important caveat in their own findings is that at the MoE AlltoAll the *communication* fusion carries the win while the *reduction* fusion is marginal — consistent with our measurement that the reduction, not the transfer, is the serial cost. Note also the stated reason streams do not work: "existing overlap approaches have trouble with precise control over execution timing of operations on GPUs when using streams" — which matches the Ascend situation where HCCL on a separate stream overlaps but our own kernel on the compute stream does not.

---

### TileLink (MLSys 2025) — generating overlapped compute-communication kernels from tile-centric primitives

https://proceedings.mlsys.org/paper_files/paper/2025/hash/c6ee784cbe46d854843e4c883a3321ef-Abstract-Conference.html · arXiv:2503.20313 · https://hgpu.org/?p=29841
ByteDance Seed; MLSys 2025.

As reported:

> Fusing communication kernels with compute kernels demands significant expertise and is error-prone.
> The system has a frontend and a backend. The frontend separates the design space for communication from that for computation and linking these two parts via tile-centric primitives. The backend lowers those primitives into low-level communication instructions and combines the parts into overlapped execution.
> Each side can be tuned on its own for tile size, tile order, and hardware resource. For example, communication can run on a copy engine while computation runs on SMs.
> Results: a 1.17× to 20.76× speedup to non-overlapping baseline and performance comparable to state-of-the-art overlapping libraries on GPUs.

## Relevance
"communication can run on a copy engine while computation runs on SMs" is the structural principle behind why HCCL is free for us and our SuperKernel is not. On Ascend the analogue of the copy engine is SDMA/xDMA (reachable through `aclshmemx_mte_put_nbi` / `aclshmemx_sdma_*`) and the analogue of "computation on SMs" is AI Core. A reduce that issues SDMA descriptors and then polls, rather than one that occupies vector cores for the whole transfer, is the shape that can overlap.

---

## Part 4 — HCCL's own open-source AIV one-shot AllReduce SuperKernel (Ascend, directly comparable)

### `HCCL_OP_EXPANSION_MODE` — the exact accepted values and what each sets

https://gitcode.com/cann/hccl/blob/master/src/common/alg_env_config.cc
(Huawei / CANN `cann/hccl`, master as of 2026-10-20; Copyright (c) 2025 Huawei Technologies Co., Ltd.)

```cpp
HcclResult ParseOpExpansion()
{
    const std::string& opExpansionModeEnv = GetEnv("HCCL_OP_EXPANSION_MODE");
    g_algEnvConfig.aicpuUnfold = false;
    g_algEnvConfig.aivMode = false;
    g_algEnvConfig.aivOnlyMode = false;
    g_algEnvConfig.ccuMSMode = false;
    g_algEnvConfig.ccuSchedMode = false;

    if (opExpansionModeEnv == "CCU_MS") {
        g_algEnvConfig.ccuMSMode = true;
        HCCL_INFO("HCCL_OP_EXPANSION_MODE set by environment to [CCU_MS]");
        return HCCL_SUCCESS;
    }

    if (opExpansionModeEnv == "CCU_SCHED") {
        g_algEnvConfig.ccuSchedMode = true;
        HCCL_INFO("HCCL_OP_EXPANSION_MODE set by environment to [CCU_SCHED]");
        return HCCL_SUCCESS;
    }

    HcclDevType deviceType;
    CHK_RET(HcclGetDeviceType(deviceType));
    // 910_93默认打开AICPU展开
    if (deviceType == HcclDevType::DEV_TYPE_910_93) {
        g_algEnvConfig.aicpuUnfold = true;
    }

    if (opExpansionModeEnv == "EmptyString") {
        HCCL_INFO(
            "HCCL_OP_EXPANSION_MODE is not set, aicpuUnfold is [%u], aivMode is [%u]", g_algEnvConfig.aicpuUnfold,
            g_algEnvConfig.aivMode);
        return HCCL_SUCCESS;
    }

    if (opExpansionModeEnv == "AI_CPU" || opExpansionModeEnv == "AICPU_TS") {
        if (deviceType == HcclDevType::DEV_TYPE_910) {
            HCCL_WARNING("910 does not support AICPU unfolding.");
        } else {
            g_algEnvConfig.aicpuUnfold = true;
        }

        // A5及后续代际默认开启aicpu task cache
        if (EqualToOrGreaterThan950(deviceType)) {
            g_algEnvConfig.aicpuCacheEnable = 1;
        }
    } else if (opExpansionModeEnv == "AIV") {
        g_algEnvConfig.aivMode = true;
    } else if (opExpansionModeEnv == "HOST") {
        g_algEnvConfig.aivMode = false;
        g_algEnvConfig.aicpuUnfold = false;
    } else if (opExpansionModeEnv == "HOST_TS") {
        if (deviceType == HcclDevType::DEV_TYPE_910B) {
            g_algEnvConfig.enableFfts = false;
        } else {
            HCCL_WARNING("deviceType[%u] does not support HOST_TS", deviceType);
        }
    } else if (opExpansionModeEnv == "AICPU_CacheDisable") {
        if (deviceType == HcclDevType::DEV_TYPE_910) {
            HCCL_WARNING("910 does not support AICPU unfolding.");
        } else {
            g_algEnvConfig.aicpuUnfold = true;
            g_algEnvConfig.aicpuCacheEnable = 0; // Disable aicpu cache
        }
    } else {
        HCCL_ERROR(
            "HCCL_OP_EXPANSION_MODE is set to [%s], which is incorrect. Please check", opExpansionModeEnv.c_str());
        return HCCL_E_PARA;
    }
    HCCL_INFO(
        "environmental variable HCCL_OP_EXPANSION_MODE is [%s], aicpuUnfold[%u], aicpuCacheEnable[%u], "
        "aivMode[%u], enableFfts[%u]",
        opExpansionModeEnv.c_str(), g_algEnvConfig.aicpuUnfold, g_algEnvConfig.aicpuCacheEnable, g_algEnvConfig.aivMode,
        g_algEnvConfig.enableFfts);
    return HCCL_SUCCESS;
}
```

And the config struct it writes into, from https://gitcode.com/cann/hccl/blob/master/src/common/alg_env_config.h :

```cpp
    u32 intraRoceSwitch;             // server内的通信方式 与intraPcieSwitch组合使用，默认为0
    int32_t inconsistentCheckSwitch; // 参数一致性校验开关，默认为0
    u8 hcclDeterministic;
    bool aicpuUnfold;
    uint8_t aicpuCacheEnable;
    bool aivMode;
    bool aivOnlyMode;
    bool ccuMSMode;
    bool ccuSchedMode;
    bool enableFfts;
    bool execTimeOutSet;
    double execTimeout;
    bool multipleDimensionSplitRatioSet;
    double multipleDimensionSplitRatio;
    bool hcclRetryConfig[HCCL_RETRY_ENABLE_LEVEL_NUM];
    bool taskExceptionEnable;
    std::map<HcclCMDType, std::vector<HcclAlgoType>> hcclAlgoConfig;
```

```cpp
        hcclDeterministic = static_cast<u8>(
            DeterministicEnableLevel::
                DETERMINISTIC_DISABLE); // 确定性配置 0：不支持；1：支持确定性不支持规约保序；2：支持确定性&规约保序
```

## Relevance
`DEV_TYPE_910_93` (the A3 class) defaults to AICPU unfolding. Setting `AIV` turns `aivMode` on and leaves `aicpuUnfold` false in that branch — i.e. the two are alternatives, not additive, and `AIV_ONLY` is a separate `aivOnlyMode` flag. Note the undocumented accepted values the error message does not list: `AICPU_TS`, `HOST_TS`, `AICPU_CacheDisable`, `CCU_MS`, `CCU_SCHED`.

---

### HCCL A3 operator support matrix — AIV mode is small-data/inference-only, and excludes zero-copy and retry by design

https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/hcclug/hcclug_000134.html
(static mirror used for extraction: https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/hcclug/hcclug_000134.html — CANN Community Edition 9.0.0, page title "Atlas A3 训练系列产品/Atlas A3 推理系列产品")

Definitions, verbatim:

```text
本节提供Atlas A3 训练系列产品/Atlas A3 推理系列产品的通信算子支持情况。
单算子零拷贝：为了降低内存拷贝开销，使得HCCL可以直接对业务传入的内存进行操作，提升通信性能。
通信算子重执行：网络故障导致通信闪断时，HCCL会尝试重新执行此通信算子，提升通信稳定性。
确定性计算：归约类通信算子在相同的硬件和输入下，多次执行将产生相同的输出。
下面按照通信算子的展开模式进行通信算子支持情况的呈现，未列出的展开模式代表不支持。
本节表格中“√”代表支持，“☓”代表不支持，“NA”代表不涉及。
未列出的算子与网络运行模式代表不支持。
```

The AIV table, verbatim (columns: 算子 | 网络运行模式 | 单算子零拷贝 | 确定性计算 | 重执行 | 节点内通信 | 超节点内通信 | 超节点间通信):

```text
AIV
|算子 | 网络运行模式 | 单算子零拷贝 | 确定性计算 | 重执行 | 节点内通信 | 超节点内通信 | 超节点间通信|
|Broadcast | 单算子模式 | ☓ | NA | ☓ | √ | ☓ | ☓|
|图模式Ascend IR | ☓ | NA | ☓ | ☓ | ☓ | ☓|
|图捕获模式aclgraph | ☓ | NA | ☓ | ☓ | ☓ | ☓|
|AllGather | 单算子模式 | ☓ | NA | ☓ | √ | √ | ☓|
|图模式Ascend IR | ☓ | NA | ☓ | √ | √ | ☓|
|图捕获模式aclgraph | ☓ | NA | ☓ | √ | √ | ☓|
|AllReduce | 单算子模式 | ☓ | √ | ☓ | √ | √ | ☓|
|图模式Ascend IR | ☓ | √ | ☓ | √ | √ | ☓|
|图捕获模式aclgraph | ☓ | √ | ☓ | √ | √ | ☓|
|ReduceScatter | 单算子模式 | ☓ | √ | ☓ | √ | √ | ☓|
|图模式Ascend IR | ☓ | √ | ☓ | √ | √ | ☓|
|图捕获模式aclgraph | ☓ | √ | ☓ | √ | √ | ☓|
|AlltoAll | 单算子模式 | ☓ | NA | ☓ | √ | √ | ☓|
|AlltoAllV | 单算子模式 | ☓ | NA | ☓ | √ | √ | ☓|
|AlltoAllVC | 单算子模式 | ☓ | NA | ☓ | √ | √ | ☓|
|ReduceScatterV | 单算子模式 | ☓ | NA | ☓ | √ | √ | ☓|
```

The footnote, which is the key statement:

```text
AIV模式在小数据量通信时性能较优，主要用于推理场景，此模式下：
单算子零拷贝会引入执行时内存协商，导致通信时延变大，所以当前AIV模式下不支持单算子零拷贝。
重执行特性会增加执行耗时，所以AIV模式不支持算子重执行。
仅支持超节点内通信，不支持超节点间通信。
```

By contrast, the AI CPU table on the same page shows `AllReduce / 单算子模式 / 单算子零拷贝 = √ / 确定性计算 = √ / 重执行 = √`, and all four scopes (节点内/超节点内/超节点间) supported:

```text
AI CPU
|AllReduce | 单算子模式 | √ | √ | √ | √ | √ | √|
|图模式Ascend IR | NA | √ | √ | √ | √ | √|
|图捕获模式aclgraph | NA | √ | √ | √ | √ | √|
```

## Relevance
Huawei's own position: AIV expansion is the small-data, inference-oriented mode, supported for AllReduce in `图捕获模式aclgraph` with deterministic computation, intra-supernode only. Our TP16 on one A3 machine is 超节点内, so the mode is in scope. The two things AIV deliberately gives up — 单算子零拷贝 and 重执行 — are both acceptable for us, and the stated reason for dropping zero-copy is instructive: "单算子零拷贝会引入执行时内存协商，导致通信时延变大" — negotiation at execution time raises latency, so a *pre-registered* symmetric buffer beats a zero-copy handshake at batch=1.

---

### HCCL AllReduce algorithm auto-selection — the exact AIV one-shot/two-shot crossover constants

https://gitcode.com/cann/hccl/blob/master/src/ops/all_reduce/selector/all_reduce_auto_selector.cc
(Huawei / CANN `cann/hccl`, master as of 2026-10-20)

```cpp
namespace ops_hccl {
constexpr u64 RS_MAX_DATA_SIZE = 16 * 1024 * 1024;
constexpr u64 AR_ONESHOT_1D_MAX_DATA_SIZE = 16 * 1024;
constexpr u64 AR_M2M_1D_MAX_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_SMALL_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_MAX_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_CROSS_SMALL_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_64DATATYPE_DATA_SIZE = 8 * 1024 * 1024;
constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4;
constexpr u32 MAX_RANK_NUM_FOR_REDUCE_MS_ALGO = 8;
constexpr u64 AR_FLATTEN_MAX_DATA_SIZE = 512 * 1024;
constexpr u64 AR_CCU_CLOS_1D_SMALL_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_SEQUENCE_DATA_SIZE = 4ULL * 1024 * 1024 * 1024;
constexpr u64 OMNI_PCIE_AR_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 OMNI_UBX_AR_SCHED_DATA_SIZE = 64 * 1024 * 1024;
constexpr u64 OMNI_UBX_AR_MS_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 OMNI_UBX_AR_LARGE_SIZE = 128 * 1024 * 1024;
constexpr u64 OMNI_UBX_AR_16P_LARGE_SIZE = 256 * 1024 * 1024;
constexpr u32 OMNI_UBX_AR_RANK_SIZE = 16;
constexpr u64 AR_AIV_SMALL_DATA_SIZE_IN_BOARD = 128 * 1024;
constexpr u64 AR_AIV_BOARD_SIZE = 8;
constexpr u32 DEVICE_NUM_PER_MODULE_8 = 8;
constexpr u64 AR_2P_DETOUR_DATA_SIZE = 8 * 1024 * 1024;
constexpr u32 AR_MORE_64P_SEQ_MAX_DATA_SIZE = 32 * 1024 * 1024;
constexpr u32 AR_CCU_MAX_RANK_SIZE = 64;
constexpr u64 AR_CCU_SEQ_MAX_DATA_SIZE = 64 * 1024 * 1024;
```

The AIV selection, verbatim:

```cpp
    if (topoInfo->level0Topo != Level0Shape::MESH_1D) {
        selectAlgName = "AivAllReduceSoleMeshTwoShot";
    } else if (topoInfo->userRankSize <= AR_AIV_BOARD_SIZE) {
        // 板内8p场景，按照时延拐点选择算法
        if (dataSize < AR_AIV_SMALL_DATA_SIZE_IN_BOARD) {
            selectAlgName = "AivAllReduceSoleMeshOneShot";
        } else {
            selectAlgName = "AivAllReduceSoleMeshTwoShot";
        }
    } else {
        if (IsSmallData(dataSize)) {
            selectAlgName = "AivAllReduceSoleMeshOneShot";
        } else {
            selectAlgName = "AivAllReduceSoleMeshTwoShot";
        }
    }
```

The guards that reject AIV, verbatim:

```cpp
    // 保序模式不支持AIV，需要回退到AICPU
    CHK_PRT_RET(
        IsNeedStrictModeForOrderPreserved(opParam, topoInfo->userRankSize),
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG,
            "[AllReduceAutoSelector][Algo][AllReduceAutoSelector] DETERMINISTIC_STRICT mode is not supported yet for "
            "AIV mode."),
        SelectorStatus::NOT_MATCH);

    // aiv 模式不支持 PROD
    CHK_PRT_RET(
        opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG, "[Algo][AllReduceAutoSelector] ReduceOp[%d] is not supported yet for aiv mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64
        || opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG, "[Algo][AllReduceAutoSelector] aiv mode does not support UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->userRankSize > MAX_RANK_SIZE) { ... return SelectorStatus::NOT_MATCH; }

    void* cclBufferAddr;
    uint64_t cclBufferSize;
    CHK_PRT_RET(
        HcclGetHcclBuffer(opParam.hcclComm, &cclBufferAddr, &cclBufferSize) != HCCL_SUCCESS, ...);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (opParam.opExecuteConfig != OpExecuteConfig::AIV_ONLY
        && dataSize >= AIV_MAX_PER_RANK_DATA_SIZE * topoInfo->userRankSize) {
        ... return SelectorStatus::NOT_MATCH;
    }
    if (dataSize > cclBufferSize * AIV_MAX_CCL_LOOP_NUM) {
        ... return SelectorStatus::NOT_MATCH;
    }
```

## Relevance
Huawei tuned the AIV one-shot/two-shot crossover for a 1D mesh at **128 KB for ≤8 ranks** ("按照时延拐点选择算法" — chosen at the latency knee), versus vLLM's 256 KB at ≤8 GPUs. Above 8 ranks it falls back to a generic `IsSmallData`. `OMNI_UBX_AR_RANK_SIZE = 16` and `OMNI_UBX_AR_16P_LARGE_SIZE = 256 MB` show 16-rank paths exist but are sized for large messages. Also note `HcclGetHcclBuffer` is a public-ish accessor for the registered CCL buffer — the same symmetric scratch our own kernel would need, and `HCCL_BUFFSIZE` sizes it.

---

### HCCL AIV one-shot AllReduce kernel — complete source, including the `SuperKernel` entry point

https://gitcode.com/cann/hccl/blob/master/src/ops/all_reduce/algorithm/template/aiv/kernel/aiv_all_reduce_mesh_1d_oneshot.h
(Huawei / CANN `cann/hccl`, master as of 2026-10-20; Copyright (c) 2025 Huawei Technologies Co., Ltd.)

```cpp
#include "aiv_communication_base_v2.h"

using namespace AscendC;

template <typename T>

class AivAllReduceMesh1DOneShot : public AivCommBase {
    constexpr static uint64_t CORE_NUMS_PER_STAGE = 16; // 每个阶段提供的最大核数
    constexpr static uint64_t STAGE_NUM = 2;            // 生产者 消费者
    constexpr static uint64_t TAG_FLAG_SIZE = 8;
    constexpr static uint64_t coreNumPerRank = 1;

public:
    __aicore__ inline AivAllReduceMesh1DOneShot() {}

    __aicore__ inline void Producer()
    {
        CpGM2GM((__gm__ T*)outputOffset, (__gm__ T*)input_, len_);
        pipe_barrier(PIPE_ALL);
        Record(targetRank, rank_, curTag_);
    }

    __aicore__ inline void Consumer()
    {
        // 初值：rank0 数据（myGmIn_ 槽 0）就位后覆盖到 output_
        WaitFlag(rank_, 0, curTag_);
        CpGM2GM((__gm__ T*)output_, (__gm__ T*)myGmIn_, len_);
        pipe_barrier(PIPE_ALL);
        // 树形折叠（与 Aicpu 保序 RunLocalReduce 同序）：slot(dst) += slot(src)，dst==0 累加进 output_，dst>=1 累加进
        // myGmIn_ 槽 零存储 flag 等待：源槽仅当 src >= k 时等待（[1,k) 已在作为目的槽被折入时等待过）；
        // 目的槽仅首次被折入时等待（新目的槽只出现在第 0/1 轮）
        uint32_t curBlocks = static_cast<uint32_t>(rankSize_);
        uint32_t m0 = GetLargestPowerOf2LessThan(curBlocks);
        uint32_t firstRoundDstNum = curBlocks - m0;
        uint32_t k = (firstRoundDstNum > (m0 >> 1)) ? firstRoundDstNum : (m0 >> 1);
        uint32_t round = 0;
        while (curBlocks > 1) {
            uint32_t M = GetLargestPowerOf2LessThan(curBlocks);
            if (M == 0) {
                break;
            }
            for (uint32_t src = M; src < curBlocks; ++src) {
                uint32_t dst = src % M;
                if (src >= k) {
                    WaitFlag(rank_, src, curTag_);
                }
                if (round == 0 || (round == 1 && dst >= firstRoundDstNum)) {
                    WaitFlag(rank_, dst, curTag_);
                }
                uint64_t srcOff = reinterpret_cast<uint64_t>(myGmIn_) + src * len_ * sizeof(T);
                if (dst == 0) {
                    CpGM2GM((__gm__ T*)output_, (__gm__ T*)srcOff, len_, reduceOp_);
                } else {
                    CpGM2GM(
                        (__gm__ T*)(reinterpret_cast<uint64_t>(myGmIn_) + dst * len_ * sizeof(T)), (__gm__ T*)srcOff,
                        len_, reduceOp_);
                }
                pipe_barrier(PIPE_ALL);
            }
            curBlocks = M;
            ++round;
        }
    }

    __aicore__ inline void FlagClear()
    {
        uint64_t flag_offset = 0; //(blockIdx_ % coreNumPerRank) * TAG_FLAG_SIZE;
        Record(rank_, flag_offset, 0);
    }

    // aiv core数目大于ranksize+1
    __aicore__ inline void ProcessCoreLargeCase(uint64_t curCount, uint32_t sliceId, uint64_t stride)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        this->curCount = curCount / coreNumPerRank;
        coreNumPerStage = coreNumPerRank * rankSize_;

        if (blockIdx_ < coreNumPerStage) {
            targetRank = blockIdx_;
            uint32_t preSyncOffset = rankSize_ * 2;
            Record(targetRank, preSyncOffset + rank_, curTag_);
            WaitFlag(rank_, preSyncOffset + targetRank, curTag_);
            uint64_t outerOffset = rank_ * this->curCount * sizeof(T);
            outputOffset = reinterpret_cast<uint64_t>(GetGmIn(targetRank)) + outerOffset;
            Producer();
            SyncAll<true>();
        } else if (blockIdx_ < coreNumPerStage + coreNumPerRank) {
            SyncAll<true>();
            Consumer();
        } else {
            SyncAll<true>();
        }
    }

    // aiv core数目小于ranksize
    __aicore__ inline void ProcessCoreSmallCase(uint64_t curCount, uint32_t sliceId, uint64_t stride)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        this->curCount = curCount;

        // SK空闲核（blockIdx >= numBlocks_）跳过生产者搬运，避免与忙核重复写数据和flag
        if (!IsIdleCore()) {
            uint32_t preSyncOffset = rankSize_ * 2;
            for (uint32_t i = 0; blockIdx_ + i * numBlocks_ < rankSize_; i++) {
                targetRank = blockIdx_ + i * numBlocks_;
                Record(targetRank, preSyncOffset + rank_, curTag_);
            }
            pipe_barrier(PIPE_ALL);
            for (uint32_t i = 0; blockIdx_ + i * numBlocks_ < rankSize_; i++) {
                targetRank = blockIdx_ + i * numBlocks_;
                WaitFlag(rank_, preSyncOffset + targetRank, curTag_);
            }
            for (uint32_t i = 0; blockIdx_ + i * numBlocks_ < rankSize_; i++) {
                targetRank = blockIdx_ + i * numBlocks_;
                uint64_t outerOffset = rank_ * this->curCount * sizeof(T);
                outputOffset = reinterpret_cast<uint64_t>(GetGmIn(targetRank)) + outerOffset;
                Producer();
            }
        }
        SyncAll<true>();
        if (blockIdx_ == numBlocks_ - 1) {
            Consumer();
        }
    }

    uint32_t coreNumPerStage;
    uint32_t targetRank;
    GM_ADDR peerMemThisCore;
    uint64_t outputOffset;
    uint64_t curCount;
    uint64_t dataBufferSize;
};

template <typename T>
__aicore__ inline void AivAllReduceV2Mesh1DOneShot(KERNEL_ARGS_DEF)
{
    AivAllReduceMesh1DOneShot<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    if (rankSize + 1 <= op.numBlocks_) {
        op.ProcessCoreLargeCase(len, sliceId, inputSliceStride);
    } else {
        op.ProcessCoreSmallCase(len, sliceId, inputSliceStride);
    }
    op.BarrierAll();
}

template <typename T>
__aicore__ inline void AivAllReduceV2Mesh1DOneShotSuperKernel(SUPERKERNEL_ARGS_DEF)
{
    AivAllReduceMesh1DOneShot<T> op;
    op.Init(SUPERKERNEL_CLASS_INIT);

    uint64_t maxCountPerLoop = op.cclBufferSize_ / UB_ALIGN_SIZE * UB_ALIGN_SIZE / op.rankSize_ / sizeof(T);
    uint64_t countLeft = op.len_;

    int32_t loopTag = op.tag_;

    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        if (op.rankSize_ + 1 <= op.numBlocks_) {
            op.ProcessCoreLargeCase(curCount, loopTag, 1);
        } else {
            op.ProcessCoreSmallCase(curCount, loopTag, 1);
        }
        op.BarrierAll();

        countLeft -= curCount;
        op.input_ += curSize;
        op.output_ += curSize;
        loopTag += curSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
}

__aicore__ inline void sk_ar_mesh_1d_oneshot(SUPERKERNEL_ARGS_DEF)
{
#ifdef HCCL_DTYPE_INT8
    AivAllReduceV2Mesh1DOneShotSuperKernel<int8_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT16
    AivAllReduceV2Mesh1DOneShotSuperKernel<int16_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT32
    AivAllReduceV2Mesh1DOneShotSuperKernel<int32_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP16
    AivAllReduceV2Mesh1DOneShotSuperKernel<half>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP32
    AivAllReduceV2Mesh1DOneShotSuperKernel<float>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_BFP16
    AivAllReduceV2Mesh1DOneShotSuperKernel<bfloat16_t>(SUPERKERNEL_ARGS_CALL);
#else
#endif
}
```

## Relevance
This is HCCL's own one-shot push allreduce, in AscendC, with a `SuperKernel` entry — the direct reference implementation for what we built. Structural points worth comparing against ours: (a) the shape is push-then-tree-fold, *not* push-then-flat-sum — the consumer does `GetLargestPowerOf2LessThan` rounds of `slot(dst) += slot(src)` with `CpGM2GM(..., reduceOp_)`, matched in order to the AICPU order-preserving reduce for bit-identical results; (b) exactly one core does the Consumer (`blockIdx_ == numBlocks_ - 1`) in the small-core case, which serializes the reduction on one AIV; (c) the pre-sync uses a *per-target-rank* flag slot at `preSyncOffset = rankSize_ * 2` so the arrival wait is one flag per peer, no 36×16 matrix; (d) `pipe_barrier(PIPE_ALL)` appears after every copy — a full pipe barrier per peer is a real cost at 16 peers.

---

### HCCL AIV constants and the CCL buffer layout — `aiv_defines.h`

https://gitcode.com/cann/hccl/blob/master/src/ops/op_common/algorithm/template/aiv/aiv_defines.h
(Huawei / CANN `cann/hccl`, master as of 2026-10-20)

```cpp
// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
constexpr uint32_t MAX_RANK_SIZE = 1024; // server内最大卡数
constexpr uint32_t MAX_RANK_SIZE_V = 256;
constexpr uint32_t BR_CTRL_CORE_LIMIT_RANK_SIZE = 64;
constexpr uint64_t BUFFER_OUT_ADDR_OFFSET = 16 * 1024;
constexpr uint64_t LOCAL_FLAG_BUF_LEN = 2560;
constexpr uint64_t AIV_TAG_MOVE_RIGHT_BITS = 16;
constexpr uint64_t LOW_16_BITS = 0xFFFF;
constexpr uint64_t DATA_LIMIT = 512 * 1024;
constexpr uint32_t PING_PONG = 2;

constexpr uint64_t AIV_PING_PONG_FACTOR_TWO = 2;
constexpr uint32_t NUM_BLOCKS_FOUR_PER_RANK_A3 = 4;
constexpr uint32_t MAX_NUM_BLOCKS = 48;

constexpr uint64_t FLAG_SIZE = 128;
constexpr uint64_t UB_ALIGN_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE_4 = UB_FLAG_SIZE * 4;
constexpr uint64_t UB_FLAG_SIZE_8 = UB_FLAG_SIZE * 8;
constexpr uint64_t UB_MAX_DATA_SIZE = 190 * 1024;
constexpr uint64_t UB_DB_DATA_BATCH_SIZE = UB_MAX_DATA_SIZE / 2;
constexpr uint32_t MaxBufferSize = 200 * 1024 * 1024;

constexpr uint64_t ATOMIC_FLAG_SIZE = 512;
constexpr uint64_t FLAG_ONE_OFFSET = 0;
constexpr uint64_t FLAG_TWO_OFFSET = FLAG_SIZE;
constexpr uint64_t FLAG_THREE_OFFSET = FLAG_SIZE * 2;
constexpr uint64_t FLAG_FOUR_OFFSET = FLAG_SIZE * 3;
constexpr uint64_t FLAG_FIVE_OFFSET = FLAG_SIZE * 4;

constexpr uint64_t DOUBLE = 2;
constexpr uint64_t FLAG_BUF_NUM = 3;
constexpr uint64_t TILING_NUM = 4;
constexpr uint64_t CHUNK_SIZE = 2048;

constexpr int32_t TAG_INIT_VALUE = 1;
constexpr int32_t TAG_RESET_COUNT = 1000;
constexpr uint32_t AIV_FLAG_CLEAR_OFFSET = 512 * 1024;
// 相对于GM_OUT，前同步、尾同步使用的同步标记区的偏移，也是普通标记区的大小
constexpr uint32_t FLAG1_OFFSET = 1 * 1024 * 1024;
constexpr uint32_t FLAG2_OFFSET = 5 * 1024 * 1024;
constexpr uint32_t BASE_FLAG_OFFSET = 9 * 1024 * 1024;
constexpr uint32_t AIV_FLAG_EMPTY_OFFSET = 10 * 1024 * 1024;
constexpr uint32_t GM_OUT_PING_OFFSET = 18 * 1024 * 1024;
constexpr uint32_t GM_OUT_PONG_OFFSET = 34 * 1024 * 1024;
constexpr uint64_t PINGPONG_TOTAL_DATA_LIMIT = (GM_OUT_PONG_OFFSET - GM_OUT_PING_OFFSET) / PING_PONG;

#if defined(HCCL_AIV_ACLGRAPH_SK)
// SK_BIND cap 掩码：blockdim_scale_up，表示算子已自行支持核数向上扩展
constexpr uint64_t SK_CAP_BLOCKDIM_SCALE_UP = 16;
#endif

/**
 * ccl buffers        GM_OUT               Tag(大小4)             flag1         flag2             BarrierBase Clear
 * data1              data2 0       |         16K          |         512K          |      1M      |       5M      | 9M
 * |               10M           |   18M             |    34M BUFFER_OUT_ADDR_OFFSET | AIV_FLAG_CLEAR_OFFSET |
 * FLAG1_OFFSET|  FLAG2_OFFSET |  BASE_FLAG_OFFSET |  AIV_FLAG_EMPTY_OFFSET |GM_OUT_PING_OFFSET | GM_OUT_PONG_OFFSET
 */
```

The SuperKernel argument struct and macros:

```cpp
using AivSuperKernelArgs = struct AivSuperKernelArgsDef {
    GM_ADDR buffersIn = nullptr; // 注册的CCLIN地址，所有卡可访问
    uint64_t rank;
    uint64_t rankSize;
    uint64_t len;
    uint64_t dataType;
    uint64_t unitSize;
    uint64_t reduceOp;
    uint64_t numBlocks;
    uint64_t tag; // 第几次调用，定时重置成1
    uint64_t clearEnable;
    uint64_t inputSliceStride;
    uint64_t outputSliceStride;
    uint64_t repeatNum;
    uint64_t inputRepeatStride;
    uint64_t outputRepeatStride;
    uint64_t input;
    uint64_t output;
    uint64_t cclBufferSize;
};

enum class AivNotifyType { ACK, DataSignal, Done };

#define SUPERKERNEL_LITE_ARGS_DEF uint64_t args_offset

#define SUPERKERNEL_LITE_ARGS_EXTRACT                \
    GM_ADDR* param_base = (GM_ADDR*)get_para_base(); \
    GM_ADDR hiddenInput = param_base[args_offset++]; \
    GM_ADDR input = param_base[args_offset++];       \
    GM_ADDR output = param_base[args_offset++]

#define SUPERKERNEL_ARGS_DEF GM_ADDR hiddenInput, GM_ADDR input, GM_ADDR output

#define SUPERKERNEL_ARGS_CALL hiddenInput, input, output

#define SUPERKERNEL_CLASS_INIT hiddenInput, input, output

#define EXPORT_AIV_META_INFO(kernel_name)                               \
    static const struct FunLevelKType kernel_name##_kernel_type_section \
        __attribute__((used, section(".ascend.meta." #kernel_name)))    \
        = {{F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}}
```

## Relevance
`HCCL_AIV_ACLGRAPH_SK` and `SK_CAP_BLOCKDIM_SCALE_UP = 16` confirm HCCL has a supported SuperKernel path intended for ACLGraph capture, with a capability bit meaning "the operator handles core-count scale-up itself". `SUPERKERNEL_ARGS_DEF` is only three pointers (`hiddenInput, input, output`) with everything else read from `get_para_base()` — so a captured SuperKernel's per-call mutable state is three addresses, which is the mechanism that makes graph capture of a collective workable. `MAX_NUM_BLOCKS = 48` and `NUM_BLOCKS_FOUR_PER_RANK_A3 = 4` give the core-budget shape on A3; `UB_MAX_DATA_SIZE = 190 * 1024` with double buffering gives 95 KB per batch, and `FLAG_SIZE = 128` is a whole cache line per flag.

---

### HCCL AIV `Record` / `WaitFlag` / `BarrierAll` — the actual Ascend barrier implementation

https://gitcode.com/cann/hccl/blob/master/src/ops/op_common/algorithm/template/aiv/aiv_communication_base_v2.h
(Huawei / CANN `cann/hccl`, master as of 2026-10-20)

```cpp
__aicore__ inline void AivCommBase::Record(uint32_t targetRank, uint64_t flag_offset, int32_t curTag)
{
    d2hGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(GetGmOut(targetRank) + flag_offset * FLAG_SIZE));
    localTagTensor.SetValue(0, curTag);
    pipe_barrier(PIPE_ALL);
    DataCopyUB2GM(d2hGlobal, localTagTensor, FLAG_SIZE / sizeof(int32_t));
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void AivCommBase::WaitFlag(uint32_t targetRank, uint64_t flag_offset, int32_t curTag)
{
    d2hGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(GetGmOut(targetRank) + flag_offset * FLAG_SIZE));
    while (true) {
        DataCopyGM2UB(localTagTensor, d2hGlobal, FLAG_SIZE / sizeof(int32_t));
        pipe_barrier(PIPE_ALL);
        if (localTagTensor.GetValue(0) == curTag) {
            break;
        }
    }
}
```

```cpp
__aicore__ inline void AivCommBase::BarrierAll()
{
    SyncAll<true>();
    // SK空闲核必须参与本SyncAll（补齐满核同步），之后的flag收发不参与
    if (IsIdleCore()) {
        return;
    }

    // 每个核分配多个rank
    uint32_t perCoreRankNum = rankSize_ / numBlocks_;
    uint32_t remainRankNum = rankSize_ % numBlocks_;
    uint32_t curCoreRankNum = blockIdx_ < remainRankNum ? perCoreRankNum + 1 : perCoreRankNum;
    uint32_t startRank
        = blockIdx_ < remainRankNum ? (perCoreRankNum + 1) * blockIdx_ : perCoreRankNum * blockIdx_ + remainRankNum;
    uint64_t flag_offset = BASE_FLAG_OFFSET - gmOutOffset_ + rank_ * FLAG_SIZE;
    for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
        Record(rank, flag_offset / FLAG_SIZE, 1);
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
        uint64_t flag_offset = BASE_FLAG_OFFSET - gmOutOffset_ + rank * FLAG_SIZE;
        WaitFlag(rank_, flag_offset / FLAG_SIZE, 1);
        Record(rank_, flag_offset / FLAG_SIZE, 0);
    }
}
```

```cpp
__aicore__ inline void AivCommBase::Barrier(uint32_t step)
{
    // 用10个flag
    uint32_t flagOffset = AIV_FLAG_EMPTY_OFFSET - gmOutOffset_ - (step % 2 + 1) * FLAG_SIZE * rankSize_;
    __gm__ int32_t* ctrlFlagsGM;
    if (blockIdx_ == 0) {
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < rankSize_; i++) {
            uint32_t targetRank = (rank_ + i) % rankSize_;
            ctrlFlagsGM = (__gm__ int32_t*)(GetGmOut(targetRank) + flagOffset + rank_ * FLAG_SIZE);
            SetSignalValue(ctrlFlagsGM, localSetTensor, 1);
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < rankSize_; i++) {
            uint32_t targetRank = (rank_ + i) % rankSize_;
            ctrlFlagsGM = (__gm__ int32_t*)(myGmOut_ + flagOffset + targetRank * FLAG_SIZE);
            WaitSignalValue(ctrlFlagsGM, localCheckTensor, 1);
        }
```

```cpp
    // SuperKernel满核启动时，blockIdx超出算子自身核数(numBlocks_)的核为空闲核
    __aicore__ inline bool IsIdleCore() const { return blockIdx_ >= static_cast<uint32_t>(numBlocks_); }

    __aicore__ inline void BarrierForFirstOP()
    {
        BarrierForFirstOPInner(0);
        SyncAll<true>();
        ClearGM();
        SyncAll<true>();
        BarrierForFirstOPInner(1);
        SyncAll<true>();
    }
```

```cpp
__aicore__ inline void AivCommBase::ClearGM()
{
    // 无论pingpong，清零区域始终从FLAG1_OFFSET开始
    // myGmOut_ = base + gmOutOffset_，需减去gmOutOffset_再加FLAG1_OFFSET
    // SK空闲核不参与清零（忙核的均分已覆盖整个清零区域）
    if (IsIdleCore()) {
        return;
    }
    GM_ADDR flagBase = myGmOut_ - gmOutOffset_ + FLAG1_OFFSET;
    uint32_t emptyOffset = AIV_FLAG_EMPTY_OFFSET - FLAG1_OFFSET;
    uint32_t blockCount = (BASE_FLAG_OFFSET - FLAG1_OFFSET) / numBlocks_;
    uint32_t blockOffset = blockCount * blockIdx_;
    CpGM2GM(flagBase + blockOffset, flagBase + blockOffset + emptyOffset, blockCount);
}
```

## Relevance
`Record` and `WaitFlag` are each wrapped in `pipe_barrier(PIPE_ALL)`, and `WaitFlag` re-issues a full `DataCopyGM2UB` of a 128-byte `FLAG_SIZE` plus a `pipe_barrier(PIPE_ALL)` on *every* poll iteration. That polling loop is the Ascend analogue of the >1 us barrier the SoL paper measures, and it is strictly more expensive than a scalar `ld.volatile` spin because each iteration drains all pipes. `BarrierAll` splits ranks across cores (`perCoreRankNum = rankSize_ / numBlocks_`), so at TP16 with few blocks each core does several Record/WaitFlag pairs serially. These three facts together are the concrete, Ascend-specific place to attack a 5.35 ms serial reduce: replace the GM2UB-poll-plus-full-pipe-barrier loop with a cheaper wait, and stop paying `pipe_barrier(PIPE_ALL)` per peer.

---

## Part 5 — Measured small-message numbers and remaining tuning knobs

### MC² 算子性能调优案例 — Huawei's own measured MatmulAllReduce fusion gain and the AllReduce latency fit

https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0043.html
(static mirror used for extraction: https://www.hiascend.com/doc_center/source/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0043.html — CANN commercial 8.5.0, "MC²算子性能调优案例")

```text
MC2通算融合算子的性能收益主要来自于通信、计算的并行执行，即将输入数据切分为多个子块，子块的计算和通信任务形成两条流水线，通过两条流水线上任务的并行执行，实现流水掩盖，从而提升算子性能。如下图所示，MC2算子先做Matmul计算、后通信的场景，输入矩阵沿M轴被切分为两块，第二块数据的Matmul计算和第一块数据的通信可以并行执行，从而达到计算和通信时间相互掩盖的目的。
```

```text
MC2算子性能收益公式为：
融合前算子串行耗时 = 融合前计算算子耗时 + 融合前通信算子耗时
MC2算子收益 = (融合前算子串行耗时 - 融合后MC2算子耗时) / 融合前算子串行耗时
```

```text
因素一：计算和通信的执行时间差异
若计算和通信任务的执行时间相差不大，则融合后，MC2算子的计算和通信并行执行，实现流水掩盖效果，性能收益较大。
若计算和通信任务的执行时间差异较大，则融合后，MC2算子内计算和通信并行执行，能够掩盖的时间较少，算子整体执行耗时与未切分串行时的算子执行耗时接近，此时无法获得较大性能收益。

因素二：数据切分导致的计算或通信的执行时间膨胀
当对输入数据进行切分后，原本的整块数据被切分成若干小数据块，对若干小数据块分别做Matmul计算或者执行通信任务，此时相比切分前，计算或者通信任务的执行时间可能发生膨胀（即执行时间变长）。该膨胀产生的原因包括：切分后的数据块过小导致计算或通信的效率降低、切分的数据块过多导致增加额外的调度开销、并行执行后计算和通信对L2 Cache或device存储内存的访问冲突等。
```

```text
未发生膨胀：
数据切分前，Matmul执行时间为200us，将Matmul的输入均匀切分为两块，假设切分后，每块数据的Matmul执行时间都是100us，通过计算的并行执行，下图实际性能收益为100us。

发生一般程度的膨胀：
数据切分前，Matmul执行时间为200us，将Matmul的输入均匀切分为两块，假设切分后，每块数据的Matmul执行时间都是150us，通过计算的并行执行，下图实际性能收益为50us。

发生严重程度的膨胀：
数据切分前，Matmul执行时间为200us，将Matmul的输入均匀切分为两块，假设切分后，每块数据的Matmul执行时间都是200us，通过计算的并行执行，下图实际性能收益为劣化50us。
```

The M-axis-only constraint, verbatim:

```text
只切分M轴。因为通信任务调用的Hccl API要求分块数据内存连续，若按N轴切分，则每行数据都被切断，导致通信数据的内存不连续，不满足通信要求；若按M轴切分，则每行数据都是内存连续的，满足通信要求。
若A表示长块、B表示短块，只能切出A或B连续排布的形式，例如AAAB、BAAA等情况。
```

The measured AllReduce cost model (8 cards, A2, half, N=8192):

```text
数据量x = m * N * sizeof(dataType)，单位是Bytes。该拟合公式表示为：
x小于8MB：t = -0.9698202 * x * x + 27.0622573 * x + 14.769，单位是us。
x大于等于8MB：t = 13.58491263 * x + 61.508333，单位是us。
```

```text
注意：通信和计算并行执行时，可能出现抢占内存带宽的情况，导致执行时间增加，一般按经验在拟合公式中乘以1.15的系数，用户可以根据实测情况调整该系数。
```

The worked example with hard numbers:

```text
本MatmulAllReduce案例中，给定的输入矩阵Shape为M=4096，K=3072，N=8192，数据类型为half，分核数为8，通过融合前msProf工具采集，得到该输入的Matmul计算执行时间为803us，AllReduce通信的执行时间为1071us，总耗时1874us，属于通信bound场景。
...
最终得到将原始输入矩阵切分为5个数据块，长度分别为：{512，896，896，896，896}。
```

```cpp
MatmulAllReduceCustomTilingData *tiling = context->GetTilingData<MatmulAllReduceCustomTilingData>();
tiling->param.rankDim = 8;
tiling->param.tileM = 512; // 短块大小
tiling->param.tileNum = 1; // 短块个数
tiling->param.tailM = 896; // 长块大小
tiling->param.tailNum = 4; // 长块个数
tiling->param.rankM = 4096;
tiling->param.rankN = 8192;
tiling->param.rankK = 4096;
tiling->param.isTransposeA = 0;
tiling->param.isTransposeB = 0;
tiling->param.cToFloatLen = 0;
tiling->param.nd2NzWorkLen = true;
tiling->param.dataType = static_cast<uint8_t>(HCCL_DATA_TYPE_MAP.at(aType));
```

```text
按该切分策略测试，融合后该算子的执行时间为1262us，则融合算子的性能收益为(1874 - 1262) / 1874 = 32.7%。
```

```text
针对切分膨胀做调整
上文提到，切分数据会引起计算或通信执行时间的膨胀，使实测结果与理论值有偏差。比如切块数量较多时，执行时间的膨胀对性能影响较大，可能导致性能收益变小或者出现性能劣化，因此最终需要根据上述理论切分策略，结合实测，对切分策略做调整。
```

## Relevance
The intercept of the small-message branch of Huawei's own fit is `14.769 us` at x→0 — i.e. an 8-card AllReduce on A2 has a measured fixed cost around 15 us even for zero bytes. That is the per-call floor to beat, and at TP16 on A3 it will be larger. Also note the fusion mechanism only works when compute and comm times are comparable ("计算和通信的执行时间差异"); at batch=1 the GEMM per layer is tiny relative to the collective, so the M-axis pipelining this doc describes has almost nothing to pipeline — which is consistent with Huawei's own rule that MC2 is disabled in 增量 (decode).

---

### NVSHMEM measured intra-node small-message latencies (H200 / NVLink-4)

https://arxiv.org/abs/2606.05951 · https://arxiv.org/html/2606.05951v1
Yijun Ma, Siyuan Shen, Tiancheng Chen, Akhil Langer, Jiri Kraus, Benjamin Glick, Craig Belusar, Jeff Hammond, Torsten Hoefler. Submitted 4 Jun 2026.

Testbed, verbatim:

> All measurements in this section were collected on a CoreWeave H200 cluster. Each node contains eight NVIDIA H200 SXM5 GPUs with 144 GB HBM3e per GPU, connected intra-node by NVLink-4. The cluster notes report 900 GB/s bidirectional NVLink bandwidth per GPU and NVLink-SHARP (NVLS) multicast support. Inter-node communication uses ConnectX-7 InfiniBand with eight NICs per node.

One-sided RMA, bandwidth and the push/pull asymmetry:

> Intra-node bulk put reaches 313 GB/s, while get peaks at 141 GB/s. Scalar p reaches 172 GB/s because many GPU threads issue independent remote stores that can be buffered and overlapped. In contrast, scalar g stays below 9 GB/s because for each remote load, the thread needs the returned value before the operation completes, limiting the number of outstanding operations and preventing deep pipelining. These results remain below the 450 GB/s NVLink reference due to NVSHMEM address translation, synchronization, work partitioning, and protocol overheads. Across nodes with tuned IBGDA settings, bulk put and get both approach the single-rail IB reference at 48.0 GB/s and 48.2 GB/s, respectively. Scalar p improves substantially and peaks at 15.6 GB/s, while scalar g remains lower at 1.28 GB/s.

The small-message latency numbers:

> The latency insets show why small messages are not bandwidth efficient. Intra-node operations take about 1.8–2.5 μs for bulk put/get and 1.3–2.2 μs for scalar p/g. With IBGDA, inter-node put/get take about 9.4–9.5 μs at 256 B and remain near 9.7 μs at 64 KiB, while scalar p/g take about 7.5 μs and 25.3 μs at 256 B. Thus, NVSHMEM is most effective for bulk or aggregated write-style RMA, while scalar operations should be treated as latency/control primitives and batched when bandwidth matters.

Device-initiated (single-CTA) vs host-side (multi-CTA) collectives:

> Figure 5 shows that multi-CTA execution is crucial for NVSHMEM collective performance. Intra-node, the host-side on-stream path reaches 264 GB/s in algorithm bandwidth, outperforming the forced NCCL ring baseline and approaching NCCL's NVLS path at 276 GB/s, while the device-side block-scoped path peaks at only 30 GB/s because it uses a single CTA. For small messages, however, the device path remains latency-competitive, taking about 3.8–7.1 μs up to 64 KiB, compared with roughly 4.7–8.9 μs for NCCL ring and 5.6–5.9 μs for NCCL NVLS. Across nodes, both NVSHMEM variants remain below 0.20 GB/s, while NCCL reaches 180 GB/s with ring and 252 GB/s with NVLS Tree.

DeepEP's low-latency inference path:

> DeepEP's low-latency path targets inference, where batch sizes are small and end-to-end layer latency matters more than peak bandwidth. Accordingly, it removes the intra-node NVLink forwarding stage used by the high-throughput path and relies on RDMA for inter-node delivery.

> During dispatch, the sender warps place tokens into per-peer RDMA ring buffers on the symmetric NVSHMEM heap, and the sender-coordinator warp periodically batches these writes into larger RDMA transfers through nvshmemi_ibgda_put_nbi_warp, updating the remote tail with nvshmemi_ibgda_amo_nonfetch_add.

## Relevance
Two concrete transfers. First, the push/pull asymmetry is not small: scalar put 172 GB/s against scalar get <9 GB/s, because a load blocks on its return value and cannot pipeline. If any part of our reduce reads peer memory rather than being pushed, that is a ~20x structural penalty on issue rate. Second, a *fully device-initiated single-CTA* collective is latency-competitive for small messages (3.8–7.1 us to 64 KiB) but caps at 30 GB/s — exactly the trade a persistent-kernel in-graph reduce makes, and the reason it is the right choice only at batch=1.

---

### vLLM `QuickAllReduce` (AMD) — quantized-in-kernel allreduce and its per-(dtype, world_size) size table

https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/quick_all_reduce.py
https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html (AMD ROCm blog, "QuickReduce: Up to 3x Faster All-reduce for vLLM and SGLang")
https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce-3/README.html (AMD ROCm blog, "QuickReduce INT3 Quantization and Benchmarking on MI355")

The size-threshold table, verbatim from `quick_all_reduce.py`:

```python
MB = 1024 * KB
...
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
...
        (torch.float16, 2): [1 * MB, 2 * MB, 2 * MB, 1 * MB, 1 * MB],
        (torch.float16, 4): [1 * MB, 16 * MB, 4 * MB, 2 * MB, 2 * MB],
        (torch.float16, 8): [16 * MB, 4 * MB, 4 * MB, 2 * MB, 2 * MB],
        (torch.bfloat16, 2): [2 * MB, 8 * MB, 8 * MB, 8 * MB, 8 * MB],
        (torch.bfloat16, 4): [8 * MB, 64 * MB, 64 * MB, 16 * MB, 16 * MB],
        (torch.bfloat16, 8): [
            16 * MB,
            2048 * MB,
            2048 * MB,
            2048 * MB,
            2048 * MB,
        ],
```

```python
        # INT3 is only enabled for TP2 (world_size == 2).
        ...
        if self.qr_quant_level == QuickReduceRegime.INT3 and self.world_size != 2:
            ...
                "only supported for TP2 (world_size == 2), but world_size "
```

```python
        # VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB is specified in MB
        qr_max_size = envs.VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB
        ...
                    "You should not set a max_size smaller than 1MB, which can "
        ...
            qr_max_size = qr_max_size * MB
        ...
        self._ptr = ops.init_custom_qr(self.rank, self.world_size, qr_max_size)
        ...
                "Custom quick allreduce: min size override = %d MB",
                qr_min_size // MB,
        ...
                "Custom quick allreduce: quantization codec threshold = %d KB",
```

As published in the docs and blog:

> Compression and decompression happen inside the thread-level data exchanges of the kernel, not through quantizing all the data up front.
> Q3 (INT3) is restricted to TP2 (world_size == 2) due to poor performance on larger world sizes.
> QuickReduce originally supported only fp16 inputs. We extended it to support bfloat16 (bf16) as well.
> On RocM, bfloat16 kernels are slower than fp16 due to slower match operations.
> Set the environment variable `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` to a level such as FP, INT8, INT6, or INT4. The value NONE disables it.
> Only the ROCm MI300 series is supported for quick allreduce at this time. It is a complement to custom allreduce and has stricter initialization requirements.

## Relevance
This is the one production allreduce that quantizes *inside* the kernel's thread-level exchanges rather than quantizing the tensor first. For a W4A8C8 model where the activations crossing the TP boundary are already low precision, the same idea applies directly: push int8 instead of bf16 and halve the pushed bytes, accepting the same accuracy argument our quantization already accepts. Note the hard caveat that INT3 was restricted to world_size==2 for performance, i.e. aggressive codecs do not survive wide fan-in.

---

### CoCoNet / Centauri / COMET / DistributedGEMM — the rest of the fused-collective family

https://proceedings.mlsys.org/paper_files/paper/2025/hash/e27ea0cd50b798ff8942caf9203f0992-Abstract-Conference.html (COMET, MLSys 2025) · arXiv:2502.19811 · https://github.com/bytedance/flux
https://seed.bytedance.com/en/blog/comet-has-been-deployed-in-large-scale-clusters-saving-millions-of-gpu-hours-moe-communication-optimization-technology-comet-is-now-open-source
Shulai Zhang, Ningxin Zheng, Haibin Lin et al. (ByteDance Seed).

As published:

> Large MoE models spend a lot of time on inter-device communication. One figure cited is 47% of total execution time. Earlier overlap methods split work coarsely, which hurts compute efficiency.
> The system uses data dependency analysis and task rescheduling to overlap communication and computation at a fine grain. It also adjusts workload assignment adaptively.
> Results: a 1.96× speedup for a single MoE layer and a 1.71× average speedup end to end.
> ByteDance says COMET runs in production on clusters with tens of thousands of GPUs and has saved millions of GPU hours.

## Relevance
COMET's 47% figure is for MoE AllToAll in training, not decode TP allreduce; its mechanism (dependency analysis + task rescheduling inside the kernel) is the same family as Flux/TileLink. Recorded here for completeness of the approach family rather than as a transferable number — none of COMET, CoCoNet or Centauri publishes a batch=1 decode allreduce latency.

---

## Part 6 — Barrier elimination, measured: SiFAR and the SGLang/vLLM push protocol

### SiFAR: Synchronization-Free All-Reduce for Low-Latency LLM Inference (the closest published match to our problem)

https://arxiv.org/abs/2607.08973 · https://arxiv.org/html/2607.08973v1
Georgia Institute of Technology, Atlanta, Georgia, USA; NVIDIA, Santa Clara, California, USA.
Keywords: Low-Latency LLM Inference, All-Reduce, In-Switch Reduction.

Abstract, verbatim:

> The rise of reasoning models and agentic systems has made LLM token-generation latency a key bottleneck. Unlike chatbots, where latency gains saturate at human reading speed, these systems generate intermediate reasoning tokens that are not consumed by humans. As a result, per-token latency directly determines end-to-end response time. To reduce latency, inference engines operate with minimal batching, making token generation bandwidth-bound. Tensor Parallelism addresses this bottleneck by sharding model weights across GPUs and loading them in parallel. However, scaling to more GPUs introduces All-Reduce overheads that grow with GPU count. We observe that removing All-Reduce improves token throughput by 43% for Llama-3.1-8B on 8 H200 GPUs.
>
> To address this overhead, we propose Synchronization-Free All-Reduce (SiFAR), which reduces synchronization overheads incurred by All-Reduce during low-latency inference. Existing oneshot and twoshot algorithms incur high synchronization overheads due to barriers before and after communication, accounting for 32–62% of the latency for small payloads we observe in low-batch serving. To tackle this, first, we find that the bottom barrier in oneshot enforces a WAW dependency, and eliminate it by co-designing communication and model execution to enable dual buffering. However, oneshot scales poorly with GPU count. Twoshot performs better at higher TP degrees, but incurs an unavoidable bottom barrier. To overcome this, we leverage in-switch reduction in modern switches, primarily used for reduce-scatter. We propose redundant pull, where each GPU reduces the entire all-reduce payload at the switch. This improves oneshot scalability while retaining its no bottom barrier advantage. Finally, to reduce top barrier overhead, we observe that each token generation step issues multiple All-Reduce operations, keeping GPUs tightly synchronized after the first All-Reduce. We therefore propose speculative reduction, which initiates data transfer without waiting for the top barrier and ensures correctness via a lightweight validation mechanism. SiFAR reduces All-Reduce latency by up to 52% and improves end-to-end throughput by 18.6% for Llama-3.1-8B and 13.1% for Qwen3.5-397B-17B at TP=8.

The motivating measurement — megakernel + TP scaling:

> To quantify this trend, we measure time per output token (TPOT) for Llama 3.1 8B on H200 GPUs using a Megakernel implementation (Spector et al., 2025). As we scale from TP=1 to TP=8, TPOT reduces from 2.73 ms to 1.43 ms, and the fraction of time in All-Reduce increases from 0% to 30% (corresponding to a 43% throughput gain). Thus, All-Reduce becomes a growing contributor to latency and is poised to dominate as other bottlenecks diminish. Despite this, optimizing All-Reduce for low-latency inference remains largely unexplored.

> Recent system optimizations mitigate many of these non-communication bottlenecks. For example, Megakernels (Wu et al., 2025b; Spector et al., 2025) fuse the forward pass into a single kernel to eliminate launch overheads and improve HBM utilization, while Programmatic Dependent Launch (PDL) (NVIDIA, 2025b) enables weights prefetching across kernel boundaries. Thus, as non-communication overheads are mitigated, All-Reduce becomes the dominant bottleneck in low-latency inference.

> For Llama-3.1-8B, removing All-Reduce improves performance by 18% at TP=2 and 43% at TP=8. For Qwen3.5-397B-17B, removing All-Reduce improves performance by 16% at TP=4 and 21% at TP=8.

The barrier breakdown — the headline numbers:

> Figure 7. Breakdown of All-Reduce latency on H200 GPUs. Synchronization overhead accounts for 32-50% of oneshot and 49-62% of twoshot latency for small payloads across all TP configurations.

> We observe that synchronization overhead accounts for 32-50% of oneshot and 49-62% of twoshot latency for small payloads (8-32 KB) across all TP configurations. For example, at TP=4 with 8 KB payload, oneshot takes 4.66 μs with barriers but only 2.33 μs without, a 2× slowdown purely from synchronization. For twoshot at TP=8 with 8 KB payload, barriers account for 3.15 μs out of 5.11 μs total, representing 62% overhead. Twoshot exhibits higher synchronization overhead because its bottom barrier must ensure that broadcast writes are visible before the next operation, which requires a more expensive acquire-release barrier that enforces write ordering. These sizes of 4-8 KB represent the All-Reduce payloads for most leading models (Liu et al., 2024; Guo et al., 2025; Grattafiori et al., 2024; Yang et al., 2025; Agarwal et al., 2025; Team et al., 2025) at batch size 1. Thus, during autoregressive decoding, every All-Reduce operation encounters this synchronization bottleneck, spending more time waiting at barriers than transferring and reducing data.

Insight 1 — dual buffering removes the bottom (WAW) barrier:

> The bottom barrier in oneshot prevents Write-After-Write (WAW) hazards by ensuring all peers finish reading the payload buffer before the owning GPU overwrites it with new data. However, WAW is an avoidable dependency that can be eliminated through redundancy. Despite this, dual buffering is not widely used in current LLM serving systems because communication is handled by application-agnostic libraries like NCCL, which cannot make assumptions about buffer usage patterns and must include a bottom barrier to ensure correctness.

> The bottom barrier in oneshot ensures all peers have finished reading the payload buffer before the owning GPU overwrites it with new data, enforcing a write-after-write dependency. We eliminate this barrier using dual buffering. The inference engine alternates between two payload buffers across successive All-Reduce operations, ensuring that the payload is never overwritten before all peers finish reading. While dual buffering is well known, it is not widely used in current LLM serving systems because communication libraries like NCCL (NVIDIA Corporation, 2025a) are decoupled from model execution and cannot make assumptions about buffer reuse patterns. SiFAR overcomes this by co-designing communication with model execution, enabling dual buffering in systems like vLLM (Kwon et al., 2023) and Megakernels (Spector et al., 2025) without changes to their memory management (Figure 1 (b)).

The TP-scaling numbers for one-shot vs two-shot data transfer:

> However, removing the bottom barrier is insufficient. Oneshot scales poorly with GPU count, making it slower than twoshot beyond TP=2. At TP=8 with an 8 KB payload, oneshot takes 4.01 μs for data transfer compared to 1.96 μs for twoshot. While twoshot has lower data transfer latency, it requires a bottom barrier that cannot be eliminated because it enforces a read-after-write dependency after the broadcast phase of twoshot. Ideally, we want oneshot latency to be comparable to twoshot so that dual buffering is effective.

> In oneshot, each GPU sends and receives K×(N−1) bytes (where K is the payload size and N is the number of GPUs), scaling linearly with GPU count. In contrast, twoshot keeps data transfer constant by decomposing reduction into two steps: reduce-scatter and broadcast. The data transfer during twoshot can be further reduced using in-switch computation capabilities (NVIDIA Corporation, 2025b; Wang et al., 2025) of modern switches. With in-switch computation, across both steps of twoshot, each GPU sends K+K/N and receives K bytes.

Insight 2 — redundant pull via in-switch reduction:

> Figure 10. Redundant Pull: each GPU issues ld_reduce over the entire K bytes. The switch reduces data from all peers and returns the result, reducing data received per GPU from K×(N−1) to K.

> However, we find that ld_reduce can reduce oneshot latency beyond its conventional use in the reduce-scatter phase of twoshot. Reduce-scatter latency remains flat up to 64 KB and grows only slightly up to 256 KB. At TP=8, a 256 KB reduce-scatter means each GPU reduces 32 KB at the switch, which is comparable to the 4-32 KB payloads in low-batch inference.

Insight 3 — speculative reduction removes the top (RAW) barrier:

> The top barrier ensures all GPUs have produced valid partials before communication begins, enforcing a RAW dependency that cannot be simply eliminated. However, we observe that LLM decoding issues multiple All-Reduce operations per token. As a result, GPUs remain tightly synchronized after the first All-Reduce, as they execute identical operations across successive All-Reduces. Thus, the overhead of the top barrier is primarily due to flag exchange, not divergence between GPUs. This leads to our third optimization: speculate and verify. When a GPU enters an All-Reduce, it speculatively begins fetching data from peers, assuming their data is ready, thereby avoiding explicit flag exchange. To ensure correctness, each GPU writes a flag to a validation buffer that is reduced alongside the payload. If the result equals N×flag after reduction, all GPUs had valid data and speculation succeeded. Otherwise, SiFAR reruns the All-Reduce.

> Figure 12. Representative Nsight profiler output: the cost of the top barrier decomposes into divergence (dynamic overhead) and flag exchange (static overhead). Because kernels launch via CUDA Graphs, divergence is near-zero after the first All-Reduce, making flag exchange the dominant cost.

Implementation detail of the validation mechanism:

> The payload buffer is reduced using switch_reduce, which performs reduction in the switch. For the validation buffer, we instead use pull-and-local-reduce to compute valid_out, as it empirically outperforms using switch_reduce for this purpose. Each threadblock runs with one extra warp to process the validation buffer while the remaining warps handle the payload. Before beginning the All-Reduce, each GPU writes the flag value into its validation buffer. After reduction, we check if the result matches ngpus × flag. If validation fails, we re-execute until validation succeeds.

> To maintain correctness across multiple All-Reduce invocations, we increment the flag value after each All-Reduce, ensuring each All-Reduce has a unique validation result. We apply dual buffering to the validation buffer to prevent race conditions where a fast GPU might overwrite its flag before all peers have finished reading it. Since remote loads are cached in L1 (Fusco et al., 2024), upon mis-speculation we use the .cg modifier to load the validation buffer to bypass L1 and fetch fresh data. The payload switch_reduce does not require this, since data is fetched from the switch and not cached locally. The correctness of the verification logic relies on GPU monotonicity; see Section 8.1 for details.

> In Megakernels (Spector et al., 2025), the entire forward pass is fused into a single kernel, and successive operations within the kernel communicate completion via flags. SiFAR reuses these existing operator flags for validation by reading and reducing the flag from all remote GPUs, instead of explicitly writing to a separate validation buffer. This eliminates the overhead of writing to the validation buffer before each All-Reduce, further reducing latency.

Evaluation methodology:

> We evaluate SiFAR by integrating it into the Megakernel framework (Spector et al., 2025). Megakernels fuse the entire forward pass into a single CUDA kernel, eliminating kernel launch overheads and improving HBM utilization. Within the Megakernel, weights of the next operation are prefetched into shared memory whenever possible, including during All-Reduce, further reducing latency. We evaluate on Megakernels rather than serving systems like vLLM (Kwon et al., 2023) because Megakernels represent the state-of-the-art for low-latency inference, where non-communication overheads are already minimized, making the impact of All-Reduce optimization most visible. We extend the original Megakernel implementation to support Llama-3.1-8B and Qwen3.5-397B-17B with multi-GPU tensor parallelism. Our Megakernel baseline without SiFAR already achieves 1.3× lower TPOT than vLLM 0.18.1 for Llama-3.1-8B and 2.2× for Qwen3.5-397B-17B at TP=8, confirming that non-communication overheads are substantially reduced. We run both models on batch size 1 unless otherwise specified.

> Figure 15 compares standalone All-Reduce latency across five implementations: oneshot, twoshot, TRT-LLM, Lamport, and SiFAR for TP configurations of 2, 4, and 8 GPUs with payloads from 8 to 32 KB. The baseline for each configuration is the best of oneshot and twoshot, which is the default in vLLM and SGLang. We execute All-Reduce back-to-back within a Megakernel to avoid measuring kernel launch overheads. The measured latency includes any mis-speculation overheads incurred by SiFAR.

Mis-speculation cost — the critical risk number:

> For Llama-3.1-8B at TP=8, SiFAR achieves a 18.6% improvement. With perfect speculation, the improvement increases to 20.2%, meaning mis-speculation retries cost only 1.6% despite a 32.2% average mis-speculation rate (Figure 24). Removing barriers entirely yields 21.6%, so verification overhead accounts for an additional 1.4%. Together, mis-speculation and verification account for only 3.0% of lost improvement, showing that speculative reduction captures most of the available benefit.

> Figure 24. Mis-speculation rate for Llama-3.1-8B and Qwen3.5-397B-17B. While mis-speculation rate increases with TP degree, it results in at most 1.6% lost performance.

> Figure 25 shows the SiFAR latency for the no-retry case and the added latency from each retry. We measure this by forcing SiFAR to retry a fixed number of times. Even with retries, the average SiFAR latency stays well below the baseline (oneshot or twoshot) across all evaluated TP degrees and payload sizes. Each retry costs less than the first attempt, which pays the one-time startup cost of beginning the All-Reduce.

> Figure 21 reports TPOT percentiles at TP=8, normalized to the baseline p50. SiFAR lowers both median and tail latency: for Llama-3.1-8B it reduces p50 to 81.4% and p99 to 84.0%, and for Qwen3.5-397B-17B it reduces p50 to 86.3% and p99 to 87.8%. The speedup from SiFAR holds through p99.9, so mis-speculation retries do not inflate tail latency.

TP is the only parallelism that helps TPOT:

> Other parallelism strategies are less effective for reducing TPOT. Pipeline parallelism (Huang et al., 2019) does not increase memory bandwidth, as each stage still loads full model weights sequentially. Expert parallelism (Fedus et al., 2022) introduces load imbalance across experts, leading to uneven bandwidth utilization. As a result, neither approach is optimal for reducing TPOT. Thus, in this work, we focus on TP.

> TP shards weight matrices across GPUs, requiring each layer to perform two All-Reduce operations: one after attention and one after the feed-forward network.

## Relevance
This paper is our problem statement, measured on GPUs. Three of its findings map onto our 5.35 ms directly. (1) The payload size regime matches: "These sizes of 4-8 KB represent the All-Reduce payloads for most leading models at batch size 1" — our TP16 batch=1 activations are in the same few-KB range, where synchronization is 32–62% of the cost. (2) The two barriers have *different* dependency types and different fixes: the bottom barrier is WAW and removable by alternating two payload buffers (free, no correctness risk, requires only that the engine not reuse the buffer before the next reduce); the top barrier is RAW and only removable by speculate-and-verify, at a measured 32.2% mis-speculation rate that costs just 1.6% of the win because a retry is cheaper than the first attempt. (3) The specific enabling observation for speculation is ours too: "LLM decoding issues multiple All-Reduce operations per token. As a result, GPUs remain tightly synchronized after the first All-Reduce" — with ~60 reductions a round in our model, after the first one the dies are implicitly in lockstep, so the top barrier is paying for flag exchange rather than for real divergence. Note the one part that does **not** transfer: `redundant pull` depends on in-switch reduction (NVLink SHARP `ld_reduce`), which A3's HCCS fabric does not provide; on Ascend the equivalent lever is `WriteReduceNbi`, and that is UB_CTP/950-only (see Part 2).

---

### vLLM `push_all_reduce` — SGLang's barrier-free push protocol, with a measured batch=1 TP=8 decode TPOT

https://github.com/vllm-project/vllm/commit/e3b4fdaf5db1aed53675c8467bc4862ba53bbfff
Alexander Matveev <amatveev@redhat.com>. Ported from `sglang/jit_kernel/csrc/distributed/custom_all_reduce_push.cuh`, SGLang commit `edb1b3f`.

The full commit message, verbatim:

```text
perf: add push-based allreduce for small tensor reductions

Port SGLang's push-based 2-buffer allreduce protocol into vLLM as a new
communicator backend for small-message reductions. The push protocol
eliminates the two explicit cross-GPU NVLink barrier round-trips used by
the existing barrier-based CustomAllreduce, replacing them with a
sentinel-based data arrival detection mechanism and double-buffered epoch
alternation.

Key advantages over the barrier-based approach:
- Zero barriers: data arrival IS the synchronization (positive-zero sentinel)
- Single NVLink round-trip instead of two barrier exchanges + remote reads
- All SMs active (SM_count CTAs vs 2 CTAs) for higher NVLink bandwidth
- No cudaMemcpy to IPC staging buffer in eager mode
- PDL (griddepcontrol) support for kernel overlap on sm_90+

The new PushAllReduce is inserted in the CudaCommunicator dispatch chain
above the existing CustomAllreduce for messages below a size threshold
(~720 KB at TP=8). Larger messages continue to use the barrier-based
path. The existing CustomAllreduce code is not modified.

Measured results on DeepSeek-V4-Pro (61 layers, TP=8, 8x NVIDIA B200,
BS=1, decode with ISL=4, OSL=33024):
- Throughput: +2.14% (84.06 vs 82.30 tokens/s)
- TPOT: -2.09% (11.90 vs 12.15 ms/token)

Correctness verified via lm_eval gsm8k 5-shot with no regression
(exact_match delta within statistical noise).

The feature can be disabled at runtime via VLLM_DISABLE_PUSH_ALLREDUCE=1
to fall back to the barrier-based path.

Signed-off-by: Alexander Matveev <amatveev@redhat.com>
```

Files changed: `CMakeLists.txt`, `csrc/push_all_reduce.cu`, `csrc/push_all_reduce.cuh`, `csrc/push_all_reduce_kernel.cuh`, `csrc/torch_bindings.cpp`, `tests/distributed/_test_push_ar_worker.py`, `tests/distributed/test_push_all_reduce.py`, `vllm/_custom_ops.py`, `vllm/distributed/device_communicators/cuda_communicator.py`, `vllm/distributed/device_communicators/push_all_reduce.py`, `vllm/distributed/parallel_state.py`.

The header comment of the ported kernel, verbatim:

```cuda
// Push-based 2-buffer allreduce kernel, ported from SGLang's
// all_reduce_one_shot_push_kernel. Uses epoch-based double-buffered
// protocol with positive-zero sentinel for data arrival detection.
//
// Original source:
//   sglang/jit_kernel/csrc/distributed/custom_all_reduce_push.cuh
// Protocol reference: SGLang commit edb1b3f
//
// Changes from SGLang:
//   - All code placed in namespace vllm::push_ar
//   - SGL_DEVICE macros replaced with __device__ __forceinline__
//   - SGL_CUDA_ARCH replaced with __CUDA_ARCH__
//   - std::integral (C++20) replaced with explicit overloads (C++17)
//   - kMaxVecBytes hardcoded to 16 (push kernel uses 16-byte vectors)
//   - TVM/FFI dependencies removed; file is self-contained
```

The sentinel mechanism, verbatim:

```cuda
template <typename T>
struct fp_trait {};

template <>
struct fp_trait<bf16_t> {
  using type = uint16_t;
  [[maybe_unused]] static constexpr uint16_t pos_zero = 0x0000u;
  [[maybe_unused]] static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<fp16_t> {
  using type = uint16_t;
  [[maybe_unused]] static constexpr uint16_t pos_zero = 0x0000u;
  [[maybe_unused]] static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<float> {
  using type = uint32_t;
  [[maybe_unused]] static constexpr uint32_t pos_zero = 0x00000000u;
  [[maybe_unused]] static constexpr uint32_t neg_zero = 0x80000000u;
};

template <typename DType>
__device__ __forceinline__ void clear_pos_zero(DType& val) {
  using Trait = fp_trait<DType>;
  const auto ptr = reinterpret_cast<typename Trait::type*>(&val);
  if (*ptr == Trait::pos_zero) *ptr = Trait::neg_zero;
}

template <typename DType>
__device__ __forceinline__ bool is_pos_zero(const DType& val) {
  using Trait = fp_trait<DType>;
  const auto ptr = reinterpret_cast<const typename Trait::type*>(&val);
  return *ptr == Trait::pos_zero;
}

template <typename DType>
__device__ __forceinline__ DType get_pos_zero() {
  using Trait = fp_trait<DType>;
  const auto value = Trait::pos_zero;
  return *reinterpret_cast<const DType*>(&value);
}
```

The epoch controller and the data structures:

```cuda
static constexpr uint32_t kMaxNumGPU = 8;

struct PushController {
  using SignalType = uint32_t;
  static constexpr int64_t kNumStages = 2;  // double-buffered epochs

  PushController() : m_local_signal(nullptr) {}

  PushController(void* ptr) : m_local_signal(static_cast<SignalType*>(ptr)) {}

  __device__ __forceinline__ SignalType epoch() const {
    return m_local_signal[blockIdx.x];
  }

  __device__ __forceinline__ void exit() const {
    __syncthreads();
    if (threadIdx.x == 0) {
      exit_unsafe(blockIdx.x);
    }
  }

  __device__ __forceinline__ void exit_unsafe(uint32_t which) const {
    auto& signal = m_local_signal[which];
    signal = (signal + 1) % kNumStages;
  }

  SignalType* m_local_signal;
};

struct AllReducePushData {
  void* __restrict__ buffer[kMaxNumGPU];
  const void* input;
  void* output;
  uint32_t rank;
  uint32_t num_items;
  uint32_t buffer_bytes;
  uint32_t epoch_bytes;
};
```

The push half — note `clear_pos_zero` applied to every element before the remote store, so that a genuine `+0.0` in the data can never be mistaken for "not yet arrived":

```cuda
template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ void push_impl(DType* (&push_buf)[kNumGPU],
                                          const void* data,
                                          uint32_t num_items) {
  constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  using Storage = AlignedVector<packed_t<DType>, kVecSize>;

  for (auto i = blockIdx.x;; i += gridDim.x) {
    const auto offset = i * blockDim.x + threadIdx.x;
    if (offset * kVecSize * 2 >= num_items) break;
    Storage vec;
    vec.load(data, offset);
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      clear_pos_zero(vec[j].x);
      clear_pos_zero(vec[j].y);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i) {
      st_global_volatile_16B(vec, push_buf[i], offset);
    }
  }
}
```

The poll half — the whole barrier is replaced by polling the *data* for the sentinel, and the buffer is re-armed with sentinels at the end:

```cuda
template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ void poll_impl(DType* (&poll_buf)[kNumGPU],
                                          void* data, uint32_t num_items) {
  constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  using Storage = AlignedVector<packed_t<DType>, kVecSize>;

  for (auto i = blockIdx.x;; i += gridDim.x) {
    const auto offset = i * blockDim.x + threadIdx.x;
    if (offset * kVecSize * 2 >= num_items) break;
    Storage storage[kNumGPU];

    while (true) {
      bool has_pos_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kNumGPU; ++i) {
        ld_global_volatile_16B(storage[i], poll_buf[i], offset);
#pragma unroll
        for (auto j = 0; j < kVecSize; ++j) {
          has_pos_zero |= is_pos_zero(storage[i][j].x);
          has_pos_zero |= is_pos_zero(storage[i][j].y);
        }
      }
      if (!has_pos_zero) break;
    }

    const Storage result = reduce_impl(storage);
    result.store(data, offset);

    Storage pos_zeros;
    pos_zeros.fill({get_pos_zero<DType>(), get_pos_zero<DType>()});
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i) {
      pos_zeros.store(poll_buf[i], offset);
    }
  }
}
```

The kernel, with PDL for cross-kernel overlap:

```cuda
template <typename DType, uint32_t kNumGPU, bool kUsePDL>
__global__ __launch_bounds__(1024, 1) void all_reduce_one_shot_push_kernel(
    const AllReducePushData __grid_constant__ params,
    const PushController __grid_constant__ ctrl) {
  const auto [buffer, input, output, rank, num_items, buffer_bytes,
              epoch_bytes] = params;

  PDLWaitPrimary<kUsePDL>();

  // Phase 1: Push data from input to all ranks' push buffers
  const auto epoch_offset = ctrl.epoch() * epoch_bytes;
  DType* push_buf[kNumGPU];
#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i) {
    push_buf[i] = static_cast<DType*>(
        ptr_byte_offset(buffer[i], rank * buffer_bytes, epoch_offset));
  }
  push_impl(push_buf, input, num_items);

  PDLTriggerSecondary<kUsePDL>();

  // Phase 2: Poll local buffer, reduce, write output, reset
  DType* poll_buf[kNumGPU];
#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i) {
    poll_buf[i] = static_cast<DType*>(
        ptr_byte_offset(buffer[rank], i * buffer_bytes, epoch_offset));
  }
  poll_impl(poll_buf, output, num_items);
```

```cuda
template <bool kUsePDL>
__device__ __forceinline__ void PDLWaitPrimary() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}

template <bool kUsePDL>
__device__ __forceinline__ void PDLTriggerSecondary() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
#endif
}
```

The size thresholds, from `vllm/distributed/device_communicators/push_all_reduce.py`:

```python
_FEATURE_DESCRIPTION = "Push-based AllReduce"
_DISABLE_ENV_VAR = "VLLM_DISABLE_PUSH_ALLREDUCE"

# Push threshold map: world_size -> buffer_bytes
# From SGLang's tuned thresholds for sm100 (B200)
PUSH_THRESHOLD_SM100 = {
    2: 4 * 1024 * 1024,  # 4 MB
    4: 2 * 1024 * 1024,  # 2 MB
    6: 1 * 1024 * 1024,  # 1 MB
    8: 720 * 1024,  # 720 KB
}

# Conservative default for untuned GPUs
DEFAULT_PUSH_BUFFER = 512 * 1024  # 512 KB


class PushAllReduce:
    """
    Push-based custom allreduce using epoch-based 2-buffer protocol.
    Ported from SGLang's CustomAllReduceV2 push allreduce.
    """
```

```python
        # Prerequisite checks
        if self.world_size not in (2, 4, 6, 8):
        ...
        props = torch.cuda.get_device_properties(device)
        self.num_sm = props.multi_processor_count
        ...
        # Allow env var override (V1 fix: validate 128-byte alignment)
        env_override = os.environ.get("VLLM_PUSH_AR_BUFFER_BYTES")
        ...
            # Round up to 128-byte alignment for kernel volatile stores
```

## Relevance
This is a production, merged, barrier-free one-shot push allreduce with a batch=1 TP=8 decode measurement at **TPOT 11.90 ms** — the same order as our 11.6198 ms — on a 61-layer MoE. Three directly portable design decisions. (1) The sentinel *is* the data: `+0.0` means "not arrived", and the producer rewrites any genuine `+0.0` in its payload to `-0.0` before pushing, which is numerically free and removes both barriers. On Ascend the equivalent is a pre-armed symmetric buffer plus `aclshmem_bfloat16_put_nbi`, with the poll done by AIV reading the data rather than a `FLAG_SIZE`-wide `DataCopyGM2UB` with `pipe_barrier(PIPE_ALL)` per iteration. (2) "All SMs active (SM_count CTAs vs 2 CTAs) for higher NVLink bandwidth" — HCCL's AIV one-shot does the opposite, funnelling the whole Consumer reduce onto `blockIdx_ == numBlocks_ - 1`. (3) `__grid_constant__` params plus `cudaLaunchKernelEx` is how the peer-pointer array is delivered without breaking graph capture; the Ascend analogue is the three-pointer `SUPERKERNEL_ARGS_DEF` plus `get_para_base()`. Note the TP ceiling: SGLang's tuned table stops at world_size 8 and the kernel hardcodes `kMaxNumGPU = 8`, so TP16 is unexplored territory upstream.

---

## Gaps

Looked for and could not find:

- **`aclshmem` measured latency numbers.** The `cann/shmem` repo, its full host/device API surface, the v1.7.0 release notes and the install path are all public and quoted above, but no published microbenchmark gives an `aclshmem_*_put_nbi` or `aclshmem_barrier` latency on A3. The repo ships `multi-QP performance tests` and `UDMA AllReduce examples` per the v1.7.0 notes; those scripts exist in-tree but no results are published. The SHMEM documentation site (https://shmem-doc.pages.dev/) was not reachable for extraction.
- **`LCCL` / `LCOC`.** Searched in English and Chinese, including against the CANN GitCode org listing, the HCCL repo tree, and the ATB/MindIE documentation. No public repository, API reference or measurement for an Ascend "low-latency collective communication library" under that name was found. The nearest real artefacts are `HCCL_OP_EXPANSION_MODE=AIV` (quoted above) and `cann/asc-comm`. The one hit referencing an "lccl backend" was an ATB API page (`ascendtb_01_0051`) whose content could not be extracted. Do not treat LCCL as a usable target without an internal source.
- **Any measured A3 TP16 small-message AllReduce latency.** HCCL's selector constants give the *thresholds* Huawei tuned to (128 KB AIV one-shot/two-shot crossover for ≤8 ranks; `OMNI_UBX_AR_RANK_SIZE = 16`), and the MC2 tuning case gives an 8-card A2 fit with a ~14.769 us zero-size intercept, but no published number exists for a 16-die A3 AllReduce at a few KB.
- **`HCCL_BUFFSIZE` default value and its exact relation to the AIV CCL buffer.** The `aiv_defines.h` layout map (16K/512K/1M/5M/9M/10M/18M/34M offsets) and `HcclGetHcclBuffer` are quoted, and vllm-ascend's GLM-5.2 reference sets `HCCL_BUFFSIZE: "400"`, but the documented default and the unit's precise semantics for the AIV path were not found in an extractable page (hiascend.com renders via JavaScript; the `doc_center/source` mirror worked for `hcclug_000134` and the MC2 case but not for the env-var reference pages).
- **`HCCL_INTRA_ROCE_ENABLE` semantics beyond one sentence.** Only vllm-ascend's one-line description ("Uses RDMA instead of SDMA links between two 8P nodes for mesh interconnect") and the HCCL source note that it is not parsed on A5-and-later (`if (!EqualToOrGreaterThan950(deviceType))`) were retrievable.
- **The TensorRT-LLM `AllReduceBestStrategyTable` contents.** `customAllReduceUtils.h` was downloaded (15,848 bytes) but the generated LUT body is machine-produced and is not quotable in a useful form; the indexing scheme `[sm][tp][fusionOp][log2(hidden)-7][log2(tokens)]` and the `SelectStrategyLP` rule are quoted from the casebook instead.
- **A multimem/NVLS equivalent on Ascend.** No Ascend hardware multicast or in-switch reduction primitive was found; SiFAR's `redundant pull` therefore has no A3 counterpart. `WriteReduceNbi` is the nearest remote-reduce primitive and is explicitly UB_CTP/URMA-only, i.e. Ascend 950 and later.
- **`DistributedGEMM` as a distinct project.** Searches returned TensorRT-LLM's `cutlass_kernels/allreduce_gemm` tree (file list quoted in Part 1's source enumeration: `gemm_universal_allreduce.hpp`, `sm90_allreduce_nvls_warpspecialized.hpp`, `sm100_visitor_allreduce_tma_warpspecialized.hpp`) rather than a separate library; no standalone documentation or measurement was retrieved.
- **Chinese-language first-party analysis of小包 AllReduce on Ascend.** Searched `昇腾 allreduce 小包 延迟`, `昇腾 AllReduce 小数据量 延迟 优化 零拷贝 AIV`, `MC2 融合通信 昇腾`, `通信计算融合`. The returns were either the official hiascend pages already quoted or third-party reposts with no measurements. No Zhihu article with primary numbers was found.
