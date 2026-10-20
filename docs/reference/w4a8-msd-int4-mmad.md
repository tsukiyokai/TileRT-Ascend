# W4A8 matmul without a mixed-width MAD — CANN's MSD decomposition

Source material on the one problem a hand-written W4A8 kernel on dav-2201 cannot avoid: `Mmad`
takes INT4 only when *both* operands are INT4, so an INT8 activation against an INT4 weight has
no single instruction. CANN's answer is a named scheme, MSD, shipped as a three-stage operator
family. Everything below is quoted as published.

Read in this order: §1 is Huawei's own writeup of the scheme and the only place the acronym is
expanded; §2 is the shipped kernel source, which is where the scheme's *actual* numeric
behaviour differs from §1's claim; §3 is the performance and accuracy numbers.

> The single most important line in this file: §1 claims the decomposition is exact
> (「保证结果等价于原始INT8×INT4计算结果」), and §2's source shows the shipped kernel
> declares its matmul output type as `half`. Both cannot be true. See §2.4.

---

## 1. Huawei official: 【昇腾热点算子大解密】CANN极致优化GroupedMatmul量化方案，MoE推理Decode性能提升超30%

URL: https://www.hiascend.com/developer/techArticles/20250717-2
Author: 昇腾社区 官方技术文章 (tags 昇腾CANN / 算子开发 / 性能调优)
Date: 发表于 2025/07/17
Fetched: 2026-10-20 (via `curl`; the `developer/techArticles/` path serves full HTML to a
fetcher, unlike `document/detail/`)

This is the only public document that expands the acronym. It does **not** mean
most-significant-digit or 多位十进制:

> 因此，我们将W4A8量化方案与昇腾硬件微架构特性结合优化，创新性地提出MSD（Mixed-precision
> Split-activation Decomposition，混合精度激活拆分分解）方案，将INT8激活拆分为高、低两路INT4，
> 并直接与INT4权重进行计算，用“高位×16+低位”合成，在保证结果等价于原始INT8×INT4计算结果的同时，
> 大幅降低KV Cache和参数存储需求，提升单卡batch size和上下文的支持长度。

### 1.1 Why the split exists at all — the redundant-dequant motivation

> 然而，仅依赖W4A8方案并不能充分释放低比特运算的潜力。推理流程中不同阶段对计算和访存的依赖有所区别，
> Prefill阶段的activation更长，计算更密集；而Decode阶段activation更短，访存压力更大，两者对硬件资源的
> 需求和优化重点不同。面对Decode阶段的访存瓶颈，传统W4A8方案虽然能降低显存占用，但在实际计算时仍需将
> INT4权重还原为INT8后，再与INT8激活做矩阵乘法，有冗余的反量化计算开销。

And on why GroupedMatmul is the carrier:

> 分组矩阵乘法（GroupedMatmul）是一种将多个独立的小规模矩阵乘法（GEMM）任务分组并行执行的优化技术，
> 通过合并计算拓扑相似的算子为单一内核调用，有效降低核启动开销并减少内存访问碎片化。
>
> MoE模型在Decode阶段每个专家处理的激活数量偏少，导致传统的GEMM操作因单个专家计算规模小而难以充分发挥
> 硬件算力。GroupedMatmul天然适配MoE模型的稀疏专家计算，通过将分散的专家任务合并为分组批量进行计算，
> 提升任务规模以实现接近峰值算力的效率；同时，连续存储专家权重可以进一步优化缓存命中率，显著提升计算性能。

> W4A8量化方案是当前主流的大模型低比特量化方案之一，通常结合per-channel、per-group和per-token等多种
> 量化scale，兼顾精度与效率。通过将权重量化为INT4格式、激活量化为INT8格式，可以极大压缩模型参数的存储
> 需求，W4A8方案的权重占用仅为传统FP16方案的四分之一。其显存优化能力，为大模型在有限硬件资源下的高效
> 部署提供了坚实基础。
>
> 下图展示了GroupedMatmul的非对称/对称量化方案。在推理时，INT4权重通过scale和zero point解码，激活则
> 直接以INT8参与整数GEMM计算；所有量化/反量化操作均在整数域内完成，最后一步统一反量化为浮点。

### 1.2 The three scale levels, and the admission that per-group barely helps

> W4A8量化方案采用三层量化scale设计，实现了精细化的量化控制，其数学公式表达如下：
>
> 其中量化scale分别为：
>
> (1) Per-channel scale：针对weight的通道级量化参数，保证不同通道间的精度平衡
>
> (2) Per-group scale：针对weight的分组级量化参数，提供更细粒度的量化控制
>
> (3) Per-token scale：针对activation的token级量化参数，动态适应不同输入的分布特性
>
> 这种多层级的量化策略能够更好地适应不同数据分布的特点，有效降低量化误差。**实践中发现，per-group量化
> 对模型精度的提升较为轻微，所以实践中一般可以不采用per-group量化。**

(bold ours; the sentence is verbatim)

### 1.3 前处理 — the split, and the `8*weight` offset

> **前处理：完成MSD方案拆分过程**
>
> MSD方案的核心是将单一INT8激活拆分为两个INT4分量，这个拆分过程在前处理中完成。假设原始INT8激活为
> x_int8，那么其拆分过程可以表示为：
>
> 其中A₁[int4]和A₂[int4]分别为activation的高4位和低4位的INT4表示。
>
> 当INT8被拆分为两个INT4时，由于INT4的符号位决定了其范围在[-8, 7]，而低4位的最高有效位（即符号扩展）
> 会带来一个偏移量，具体就是8*weight。为了保证拆分的精度，需要将拆分操作变型为如下过程：
>
> 这个多余项在所有INT8拆分为INT4的操作中是固定的，等价于每个权重通道都引入了一个偏置项。在
> GroupedMatmul的具体实现中，通常将这部分多余项放入bias参数中，通过离线计算提前补偿，从而保证推理精度
> 和性能。下图展示了前处理的计算流程：

(The two displayed formulas and the three figures are images, not text, and did not survive the
fetch. The prose pins the arithmetic down completely anyway, and §2 has the executable version.)

### 1.4 MatMul — the two weight scales are folded offline into one `anti-scale`

> **MatMul过程**
>
> **1. Per-channel和per-group scale的融合**
>
> 权重首先以per-channel FP16 scale进行粗量化，得到中间的INT8权重。然后对INT8权重再做per-group INT4
> 量化，得到最终的INT4权重和per-group scale。
>
> 如上图所示，x与weight被拆分后完成INT4矩阵乘。在实际推理时，为了提升效率，per-channel scale和
> per-group scale会离线融合成一个总scale（图中的anti-scale），这样可以在GEMM输出阶段一次性完成浮点还原，
> 减少多次逐步缩放带来的性能损耗。
>
> 融合后的scale一般为total_scale = perchannel_scale * pergroup_scale。这样，矩阵乘法主环节只需用整数
> 计算，最后输出时统一乘以融合后的scale即可。
>
> **2. GEMM的per-group累加实现**
>
> Per-group scale是针对权重的分组量化，每组权重共用一个scale。在GEMM实现中，通常沿着k轴（即输入通道）
> 切分，每g个通道为一组（g为group size），每组使用独立的scale。
>
> 在做矩阵乘法时，需要将输入矩阵X（已分解为INT4）与权重矩阵W（INT4，已分组）分块处理。**对每个group，
> 通过total scale将INT4权重直接解码为FP16。** 具体而言，每进行一组累加，就应用对应的per-group scale。

(bold ours. 「直接解码为FP16」 is the officially documented FP16 intermediate — it is not an
implementation accident. This is the sentence that §2.4's contradiction turns on.)

### 1.5 后处理 — the three steps, in order

> **后处理：合成并反量化输出结果**
>
> W4A8量化后处理主要包括三个步骤：高低位结果合成、bias补偿、反量化（per-token scale）。如下图，matmul
> 的结果在图中表示为C，经过后处理后输出的结果为y。
>
> 下面详细介绍各环节的原理和实现方式：
>
> (1) 高低位结果合成：在W4A8方案中，INT8激活被拆分为高4位和低4位。为还原INT8原始乘法的效果，需要将高4位
> 的结果左移4位（即乘以16），再与低4位结果相加。这样可以保证拆分计算后，数值与直接INT8×INT4乘法结果一致。
>
> (2) BIAS余量补偿：拆分INT8为两个INT4时，由于符号扩展和量化区间的差异，会引入一个额外的常量项
> （8×W×scale）。这个多余项在量化离线阶段提前计算，并作为bias参数加到最终输出上，补偿拆分带来的偏移，
> 保证数值精度。在实际后处理时，直接将该bias向量加到矩阵乘法的结果上即可，无需额外计算，且不会影响推理效率。
>
> (3) Per-token反量化：W4A8方案通常采用per-token scale对激活进行量化，推理后需反量化还原浮点结果。
> Per-token反量化中包含每个token（即每一行激活）独立的量化尺度，能够动态适应不同输入分布，提升精度。
> 这种逐token的反量化方式，能有效降低量化误差，尤其适合大模型推理中的动态输入场景。

### 1.6 Measured result, and the published entry point

> 至此，W4A8量化与MSD方案的所有主流程均在整数域内完成，充分利用昇腾硬件对低比特整数计算的支持，减少主循环
> 的访存和运算瓶颈，为LLM的高效推理提供了关键支撑，极大提升大模型推理的响应速度和服务能力。**基于W4A8
> 方案的DeepSeek R1单机部署实践表明，其最大BatchSize达到了W8A8方案的3.1倍；同时，W4A8+MSD方案在Decode
> 阶段的性能更是达到了W8A8的1.3~1.7倍。**

> GroupedMatmul算子功能通过CANN软件包使能，社区版资源下载地址：
> https://www.hiascend.com/developer/download/community/result?module=cann
>
> 算子接口定义为：
>
> ```
> aclnnStatus aclnnGroupedMatmulV4GetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight, const aclTensorList *biasOptional, const aclTensorList *scaleOptional,
> const aclTensorList *offsetOptional, const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional, const aclTensorList *perTokenScaleOptional,
> const aclTensor *groupListOptional, const aclTensorList *activationInputOptional, const aclTensorList *activationQuantScaleOptional, const aclTensorList *activationQuantOffsetOptional,
> int64_t splitItem, int64_t groupType, int64_t groupListType, int64_t actType, aclTensorList *out, aclTensorList *activationFeatureOutOptional, aclTensorList *dynQuantScaleOutOptional,
> uint64_t *workspaceSize, aclOpExecutor **executor)
>
> aclnnStatus aclnnGroupedMatmulV4(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
> ```
>
> 详细接口文档可参考昇腾社区API描述：
> https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/aolapi/context/aclnnGroupedMatmulV4.md

### Relevance

Four things we did not have before, all from this one page:

1. **The acronym is not what we assumed.** MSD = "Mixed-precision Split-activation
   Decomposition / 混合精度激活拆分分解". Not "most-significant-digit", not "多位十进制". Correct
   this in our own notes — searching for the wrong expansion is why this page is hard to find.
2. **The `8*sum(w)` fold is Huawei's own published mechanism**, under the name 「BIAS余量补偿」,
   with the offset written as `8*weight` / 「8×W×scale」, 「在量化离线阶段提前计算」 and 「放入
   bias参数中」. Our per-column bias is the vendor's design, not a workaround.
3. **Per-group weight quantisation is officially rated near-worthless** —
   「per-group量化对模型精度的提升较为轻微，所以实践中一般可以不采用per-group量化」. We can set
   `quantGroupNum = 1` (one scale per column over the whole K) and keep the vendor's own accuracy
   story. That collapses the `loopK` loop in §2.2 to a single `Iterate`, which removes the
   fp16 atomic re-accumulation in §2.4 *and* the per-group scale reload. This is the cheapest
   accuracy-and-speed win on the page.
4. **The win is claimed only for Decode, at 1.3-1.7x over W8A8**, i.e. the same regime we are in.
   The 3.1x is batch size, not speed; do not quote it as a latency number.

---

## 2. The shipped kernel source — `ops-transformer/gmm/grouped_matmul_swiglu_quant`

Repository: `https://gitcode.com/cann/ops-transformer` (CANN Open Software License Agreement
Version 2.0; headers carry `Copyright (c) 2025 Huawei Technologies Co., Ltd.`)
Read from a local clone of that remote at commit `d6aaa9fe52300b2fc7138742eb71ba265000196b`
(2026-10-19). A second clone at `e027a008be2766fa5cb6198bebbc7b9dcc08310b` (2026-09-01) carries
the same files byte-for-byte for the paths quoted here.

Paths, verbatim:

```
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant.cpp
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_pre.h
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_mid.h
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_post.h
gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_utils_kernel.h
gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pre.h
gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h
gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h
gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h
gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd.h
gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd_new.h
gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd_pre.h
gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd_pre_nz.h
gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_antiquant_a8w4_msd.h
gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_antiquant_a8w4_msd_pre.h
gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_antiquant_a8w4_msd_l1_opt.h
```

### 2.0 The dispatch — A8W8 gets `int32_t`, A8W4-MSD gets `half`

`grouped_matmul_swiglu_quant.cpp`, both macros side by side, verbatim:

```cpp
#define GMM_CV_SPLIT_IMP(computeClass, dtypeWeightScale, transA, transB, sync) \
    do { \
        using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X, false>; \
        using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, DTYPE_WEIGHT, false>; \
        using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>; \
        using matmulType = MMImplTypeStatic<xType, weightType, yType>; \
        matmulType::MT mm; \
        ...

#define GMM_CV_SPLIT_IMP_A8W4_MSD(computeClass, dtypeWeightScale, transA, transB, sync) \
    do { \
        GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingData, gmmSwigluBaseParams, gmmSwigluBaseParams_, tiling); \
        GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingData, mmTilingData, mmTilingData_, tiling); \
        GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingData, gmmSwiglu, gmmSwiglu_, tiling); \
        using xType = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>; \
        using weightType = MatmulType<TPosition::GM, wFormat, int4b_t, false>; \
        using yType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>; \
        using matmulType = MMImplType<xType, weightType, yType>; \
        matmulType::MT mm; \
        if ASCEND_IS_AIC { \
            mm.SetSubBlockIdx(0); \
            mm.Init(&mmTilingData_); \
        } \
        computeClass<matmulType> op(mm, &gmmSwigluBaseParams_, &gmmSwiglu_, &tPipe); \
        op.Init(x, weight, weightScale, xScale, weightAssistanceMatrix, groupList, y, yScale, user1); \
\
        op.Process(); \
    } while (0)
```

Both operands are `int4b_t` — confirming our finding that `Mmad` on 2201 needs both sides INT4 —
and the destination is `half`, against `int32_t` on the A8W8 path three lines above it.

The kernel entry, with the extra input the MSD path needs and the tiling keys:

```cpp
extern "C" __global__ __aicore__ void grouped_matmul_swiglu_quant(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale,
                                                                  GM_ADDR xScale, GM_ADDR weightAssistanceMatrix,
                                                                  GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale,
                                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR user1 = GetUserWorkspace(workspace);
#if defined(GMM_SWIGLU_QUANT_A8W8)
    if (TILING_KEY_IS(0)) { // antiquant msd
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP(GMMSwigluCompute, ...);
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP(GMMSwigluSplitWorkSpaceCompute, ...);
    }
#elif defined(GMM_SWIGLU_QUANT_A8W4_MSD)
    if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP_A8W4_MSD(GMMSwigluQuantPipelineSchedule, ...);
    }
#endif
}
```

`weightAssistanceMatrix` is a **declared kernel argument**, i.e. the `8*sum(w)` correction is a
first-class operator input, not something computed on device. `KERNEL_TYPE_MIX_AIC_1_2`: one cube,
two vector, which is how pre (AIV) / mid (AIC) / post (AIV) share one launch.

### 2.1 `..._msd_pre.h` — the split, instruction by instruction

The declared tensor set already tells the story — `int4b_t` for both halves, `half` as the
working type, and a 16-bit mask constant:

```cpp
#define BUFFER_NUM_A8W4_PRE 1
constexpr int TWO = 2;
constexpr int EIGHT = 8;
constexpr size_t LEN_128 = 128; // 16bit operator
constexpr int DATA_BLOCK_SIZE_32 = 32;
```

```cpp
    LocalTensor<int8_t> xTensor;
    LocalTensor<half> xHighHalfTensor;
    LocalTensor<float> xHighFloatTensor;
    LocalTensor<half> xLowHalfTensor;
    LocalTensor<half> xLowHalfTensor2;
    LocalTensor<int4b_t> xHighI4Tensor;
    LocalTensor<int4b_t> xLowI4Tensor;
    LocalTensor<int16_t> xLowI16Tensor;
    LocalTensor<int64_t> groupListTensor;
    LocalTensor<float> groupListFTensor;
    LocalTensor<float> xRowSumTensor;
```

The UB budget, verbatim, with the per-K-element cost spelled out:

```cpp
        vK = gmmSwigluBaseParams->K;
        groupNum = static_cast<uint32_t>(gmmSwigluBaseParams->groupNum);
        // M * K * 7B (1B + 0.5B + 0.5B + 2B + 4B) <= UBsize - 256B
        blockDim = GetBlockNum() * GetTaskRation();
```

```cpp
__aicore__ inline void GMMA8W4PreProcess::CustomInitBuffer(TPipe* pipe)
{
    pipe->InitBuffer(vecInQueueX, BUFFER_NUM_A8W4_PRE, vK * sizeof(int8_t));    // K * 1B
    pipe->InitBuffer(vecOutQueueA1, BUFFER_NUM_A8W4_PRE, vK * sizeof(int4b_t)); // K * 0.5B
    pipe->InitBuffer(vecOutQueueA2, BUFFER_NUM_A8W4_PRE, vK * sizeof(int4b_t)); // K * 0.5B
    pipe->InitBuffer(vecOutQueueA3, BUFFER_NUM_A8W4_PRE, vK * sizeof(half));    // K * 2B
    // xLowHalfTensor, xLowHalfTensor2 and xHighFloatTensor share the same buffer
    pipe->InitBuffer(tempBuff, vK * sizeof(float)); // K * 4B
    constexpr int BUFFER_SIZE_256B = 128 * sizeof(int16_t);
    pipe->InitBuffer(vecOutQueue0F, BUFFER_NUM_A8W4_PRE, BUFFER_SIZE_256B); // 256B
}
```

The mask and the magic constant:

```cpp
        Duplicate(xLowI16Tensor, static_cast<int16_t>(0x0F0F), MASK); // get rid of high 4 bits in every int8
        PipeBarrier<PIPE_V>();
        const size_t LEN_VK = (vK / 2) / 128;
        const size_t LAST_LEN_VK = (vK % 256) / 2;
        const half ONE_SIXTEENTH = static_cast<half>(0.0625f);
        // groupList仅支持count
```

And the loop body — this is the whole decomposition, and the only arithmetic in it:

```cpp
        for (uint32_t xloop = 0; xloop < curCoreTaskNum; xloop++) {
            uint64_t relStartAddr = (xloop + curCoreStartOffset) * vK;
            uint64_t absStartAddr = workspaceSplitLoopIdx * workspaceSplitConfig.notLastTaskSize * vK + relStartAddr;
            // 高四位处理开始
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0); // 0
            DataCopy(xTensor, xGM[absStartAddr], vK);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);  // 3
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0); // 3
            Cast(xHighHalfTensor, xTensor, AscendC::RoundMode::CAST_NONE, vK);
            PipeBarrier<PIPE_V>();
            Muls(xHighHalfTensor, xHighHalfTensor, ONE_SIXTEENTH, vK);
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID1); // 2
            Cast(xHighI4Tensor, xHighHalfTensor, AscendC::RoundMode::CAST_FLOOR, vK);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);  // 4
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0); // 4
            DataCopy(yGm[relStartAddr], xHighI4Tensor.ReinterpretCast<int8_t>(), vK / 2);
            // 高四位处理结束

            // 低四位处理开始
            SetFlag<HardEvent::MTE3_V>(EVENT_ID1); // 2
            And(xLowHalfTensor.ReinterpretCast<int16_t>(), xTensor.ReinterpretCast<int16_t>(), xLowI16Tensor, LEN_128,
                LEN_VK, {1, 1, 1, 8, 8, 0});
            if (LAST_LEN_VK > 0) {
                And(xLowHalfTensor[LEN_VK * LEN_128].ReinterpretCast<int16_t>(),
                    xTensor[LEN_VK * LEN_128 * TWO].ReinterpretCast<int16_t>(), xLowI16Tensor, LAST_LEN_VK, 1,
                    {1, 1, 1, 8, 8, 0});
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); // 0
            Cast(xLowHalfTensor2.ReinterpretCast<half>(), xLowHalfTensor.ReinterpretCast<int8_t>(),
                 AscendC::RoundMode::CAST_NONE, vK);
            PipeBarrier<PIPE_V>();
            const half MINUS_EIGHT = static_cast<half>(-8);
            Adds(xHighHalfTensor, xLowHalfTensor2, MINUS_EIGHT, vK);
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0); // 1
            Cast(xLowI4Tensor, xHighHalfTensor.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, vK);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);  // 5
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1); // 5
            DataCopy(yGm[relStartAddr + vK / TWO], xLowI4Tensor.ReinterpretCast<int8_t>(), vK / TWO);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0); // 1
            // 低四位处理结束
        }
```

Decoded, with nothing added:

| | high nibble | low nibble |
| --- | --- | --- |
| extract | `Cast(int8 -> half)`, `Muls(0.0625)`, `Cast(half -> int4b_t, CAST_FLOOR)` | `And(int16, 0x0F0F)`, `Cast(int8 -> half)`, `Adds(-8)`, `Cast(half -> int4b_t, CAST_NONE)` |
| value | `h = floor(x / 16)`, in `[-8, 7]` | `l = (x & 0x0F) - 8`, in `[-8, 7]` |
| GM byte offset within row `r` | `r*vK` | `r*vK + vK/2` |

So the identity the kernel relies on is

```
x_int8 = 16*h + (l + 8)       h = floor(x/16) in [-8,7],  l = (x & 0xF) - 8 in [-8,7]
```

which is exact for every `x` in `[-128, 127]`, including negatives: `floor(x/16)` is an arithmetic
shift, and `x & 0xF` is the unbiased low nibble. Note the split is done **in fp16 arithmetic**
(`Muls` by `0.0625` then `CAST_FLOOR`), not with a shift instruction — every int8 is exactly
representable in fp16, so this particular step is still exact.

### 2.2 `..._msd_mid.h` — one matmul at 2M rows, not two matmuls

This is the line that settles the stacking question. `Process`, verbatim:

```cpp
            int32_t splitValue = (currSplitValue - prevSplitValue) * 2; // 2: int8 has been split in 2 int4
            prevSplitValue = currSplitValue;

            SetMNConfig(splitValue, mnConfig);
            if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
                continue;
            }
            mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
            mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
```

`mnConfig.m` is the per-expert row count **doubled**, and `K`/`N` are untouched. One
`SetOrgShape(2M, N, K)`. The A-matrix offset advances by the doubled M:

```cpp
template <typename mmType>
__aicore__ inline void GMMA8W4MidProcess<mmType>::UpdateMnConfig(MNConfig& mnConfig)
{
    if constexpr (bT::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset +=
            static_cast<int64_t>(AlignUp<16>(mnConfig.k)) * AlignUp<32>(mnConfig.n); // 16: nz format last two dim size
    } else {
        mnConfig.wBaseOffset += static_cast<int64_t>(mnConfig.k) * mnConfig.n;
    }
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += static_cast<int64_t>(mnConfig.m) * mnConfig.k;
    mnConfig.yBaseOffset += static_cast<int64_t>(mnConfig.m) * mnConfig.n;
}
```

The inner compute, including the per-quant-group K loop and the accumulate flag:

```cpp
    mm.SetSingleShape(curSingleM, curSingleN, quantGroupSize); // 8, 256, 512 --> 514us
    GlobalTensor<int4b_t> weightSlice;
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    mnConfig.workSpaceOffset = outOffset + mnConfig.yBaseOffset;
    for (uint32_t loopK = 0; loopK < gmmBaseParams->quantGroupNum; loopK++) {
        mm.SetTensorA(
            xGM[mnConfig.xBaseOffset + mnConfig.mIdx * mnConfig.k * mnConfig.singleM + loopK * quantGroupSize]);
        if constexpr (mmType::BT::format == CubeFormat::NZ) {
            weightSlice = weightGM[weightOffset + loopK * quantGroupSize * 64];
        } else {
            weightSlice = weightGM[weightOffset + loopK * quantGroupSize * gmmBaseParams->N];
        }
        if (mnConfig.blockDimM == 1) {
            weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        mm.SetTensorB(weightSlice);
        mm.SetQuantVector(weightScaleGM[groupIdx * gmmBaseParams->N * gmmBaseParams->quantGroupNum +
                                        loopK * gmmBaseParams->N + tailN]);
        mm.Iterate();
        mm.GetTensorC(mmOutGM[mnConfig.workSpaceOffset], loopK == 0 ? 0 : 1);
    }
```

Note `quantGroupSize = gmmBaseParams->K / gmmBaseParams->quantGroupNum; // 约束为整除关系`, the
`uint64_t` weight-scale tensor fed to `SetQuantVector` (the packed deq-scale format), and the
in-source tiling datum `// 8, 256, 512 --> 514us` — `baseM=8`, `baseN=256`, `quantGroupSize=512`.

### 2.3 `..._msd_post.h` — `16*high + low`, then the auxiliary matrix

The constant, and the recombination, verbatim:

```cpp
#define DOUBLE_BUFFER 2
constexpr float DEFAULT_MUL_SCALE = 16.0f;
```

```cpp
    mmLocal_fp16 = mmOutQueue.DeQue<half>();
    mmLocal_fp32 = mmLocal_fp16.ReinterpretCast<float>();
    const int64_t processNum = 2 * vecConfig.innerLoopNum * gmmSwiglu->tokenLen;
    DataCopyExtParams copyParams_0{1, static_cast<uint32_t>(processNum * sizeof(half)), 0, 0, 0};
    DataCopyPadExtParams<half> padParams_0{false, 0, 0, 0};
    DataCopyPad(mmLocal_fp16[processNum], mmOutGM[vecConfig.curOffset * DOUBLE_ROW], copyParams_0, padParams_0);
    GmmsqSetWaitFlag<HardEvent::MTE2_V>();

    mmOutQueue.EnQue(mmLocal_fp16);
    mmLocal_fp16 = mmOutQueue.DeQue<half>();
    // 1. fp16 -> fp32
    Cast(mmLocal_fp32, mmLocal_fp16[processNum], RoundMode::CAST_NONE, processNum);
    PipeBarrier<PIPE_V>();
    // 2. high_4bit * 16 + low_4bit
    for (uint32_t i = 0; i < vecConfig.innerLoopNum; i++) {
        Muls(mmLocal_fp32[(DOUBLE_ROW * i) * gmmSwiglu->tokenLen], mmLocal_fp32[(DOUBLE_ROW * i) * gmmSwiglu->tokenLen],
             DEFAULT_MUL_SCALE, gmmSwiglu->tokenLen);
        PipeBarrier<PIPE_V>();
        Add(mmLocal_fp32[i * gmmSwiglu->tokenLen], mmLocal_fp32[(DOUBLE_ROW * i) * gmmSwiglu->tokenLen],
            mmLocal_fp32[(DOUBLE_ROW * i + 1) * gmmSwiglu->tokenLen], gmmSwiglu->tokenLen);
        PipeBarrier<PIPE_V>();
        vecConfig.curIdx++;
    }
```

With `constexpr int64_t DOUBLE_ROW = 2;` from the utils header, row `2i` is the high half and row
`2i+1` is the low half of logical row `i`. **The M-stacking is interleaved, not blocked**:
`[h_0, l_0, h_1, l_1, ...]`, with stride `2*tokenLen` between logical rows.

The four post stages, in the order they run:

```cpp
__aicore__ inline void GMMA8W4PostProcess::VectorCompute(uint32_t loopIdx, VecConfig& vecConfig,
                                                         WorkSpaceSplitConfig& workspaceSplitConfig)
{
    // 1.辅助矩阵加回
    MergeAuxiliaryMatrix(loopIdx, vecConfig);
    // 2.perToken反量化
    MulPertokenScale(loopIdx, vecConfig, workspaceSplitConfig);
    // 3.Swiglu
    Swiglu(loopIdx, vecConfig);
    // 4.Quant
    Quant(loopIdx, vecConfig);
}
```

The `8*sum(w)` correction is an `(E, N)` fp32 tensor, added before the per-token scale:

```cpp
        weightAuxiliaryMatrixGM.SetGlobalBuffer((__gm__ float*)gmAddrParams.weightAuxiliaryMatrixGM); // E, N
```

```cpp
__aicore__ inline void GMMA8W4PostProcess::MergeAuxiliaryMatrix(uint32_t loopIdx, VecConfig& vecConfig)
{
    // perChanelScale * perTokenScale
    mmLocal_fp32 = mmOutQueue.DeQue<float>();
    LocalTensor<float> weightAuxiliaryMatrixLocal = weightAuxiliaryMatrixInQueue.DeQue<float>();
    Add(mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen], mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen],
        weightAuxiliaryMatrixLocal, gmmSwiglu->tokenLen);
    vecConfig.nextUpadteInterVal--;
    PipeBarrier<PIPE_V>();
    weightAuxiliaryMatrixInQueue.EnQue(weightAuxiliaryMatrixLocal);
}

__aicore__ inline void GMMA8W4PostProcess::MulPertokenScale(uint32_t loopIdx, VecConfig& vecConfig,
                                                            WorkSpaceSplitConfig& workspaceSplitConfig)
{
    float scale = perTokenScaleGM.GetValue(loopIdx + workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx);
    GmmsqSetWaitFlag<HardEvent::S_V>();
    Muls(mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen], mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen], scale,
         gmmSwiglu->tokenLen);
    PipeBarrier<PIPE_V>();
}
```

The auxiliary matrix is reloaded only on expert change, by walking `groupList`:

```cpp
__aicore__ inline void GMMA8W4PostProcess::UpdateAuxiliaryMatrix(uint32_t loopIdx, VecConfig& vecConfig)
{
    // 更新weightAuxiliaryMatrix
    if (unlikely(vecConfig.nextUpadteInterVal == 0)) {
        int64_t loop = gmmSwiglu->groupListLen - vecConfig.curGroupIdx - 1;
        while (loop--) {
            int64_t curTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
            vecConfig.curGroupIdx++;
            int64_t nextTemp = groupListGM.GetValue(vecConfig.curGroupIdx);
            if (nextTemp != curTemp) {
                vecConfig.nextUpadteInterVal = nextTemp - curTemp;
                break;
            }
        }
        ...
```

And the SwiGLU + per-token-max requant that follows, for completeness (`SWIGLU_REDUCE_FACTOR = 2`,
`BISECT = 2`, `QUANT_SCALE_INT8 = 127.0f`, all from the utils header):

```cpp
__aicore__ inline void GMMA8W4PostProcess::Swiglu(uint32_t loopIdx, VecConfig& vecConfig)
{
    // 高阶API swiglu
    float beta = 1.0f;
    LocalTensor<float> workspaceLocal = reduceWorkspace.Get<float>();
    LocalTensor<float> src0Local =
        mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen + gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR];
    LocalTensor<float> src1Local = mmLocal_fp32[loopIdx * gmmSwiglu->tokenLen];

    SwiGLU<float, false>(workspaceLocal, src0Local, src1Local, beta, gmmSwiglu->tokenLen / SWIGLU_REDUCE_FACTOR);
```

```cpp
    ReduceMaxTemplate(reduceResLocal, workLocal, mmLocal_fp32[preOffset + gmmSwiglu->tokenLen / BISECT], reduceTmpLocal,
                      static_cast<uint32_t>(halfTokenLen));
    GmmsqSetWaitFlag<HardEvent::V_S>();
    float quantScale = reduceResLocal.GetValue(0) / QUANT_SCALE_INT8;
```

The UB arithmetic for post, verbatim, which is the budget to beat:

```cpp
    // 2 * row * n * sizeof(float) + row * n / 2 * sizeof(int8) + alignUp<row, 8> * sizeof(float) + n * sizeof(float) +
    // n / 2 *sizeof(float) + 64 < 191 * 1024
```

### 2.4 The contradiction: §1 says exact, the source says `half`

Lay the two against each other.

§1.5: 「这样可以保证拆分计算后，数值与直接INT8×INT4乘法结果一致」 — numerically identical to a
direct INT8xINT4 multiply. §1 heading sentence: 「在保证结果等价于原始INT8×INT4计算结果的同时」.

§2.0: `using yType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;`
§2.3: `mmOutGM1.SetGlobalBuffer((__gm__ half*)...)`, `Cast(mmLocal_fp32, mmLocal_fp16[...])`,
`Muls(..., DEFAULT_MUL_SCALE /* 16.0f */)`.

The integer identity `x = 16h + (l+8)` is exact; the *implementation* is not, in three places,
none of which is visible from the article:

1. **The two partial products are rounded to fp16 before they are combined.** `SetQuantVector`
   applies the weight scale in Fixpipe and the result leaves the cube as `half` — 10 bits of
   mantissa. The `*16` in post then multiplies the high partial product's fp16 rounding error by
   16 as well. An error of 1 ulp on the high term becomes 16 ulp of the result. The A8W8 path two
   lines above in the same file keeps `int32_t` and has none of this.
2. **Accumulation across quant groups is an fp16 atomic add in GM.**
   `mm.GetTensorC(mmOutGM[...], loopK == 0 ? 0 : 1)` — the second argument is `enAtomic`, so
   groups `1..quantGroupNum-1` accumulate atomically into a `half` buffer. Within one group
   (`SetSingleShape(curSingleM, curSingleN, quantGroupSize)`, K = 512) the MAC is exact int32;
   across groups it is fp16.
3. **The auxiliary matrix is fp32 added to a value reconstructed from fp16 terms**, so it cannot
   recover the lost bits — it only corrects the mean.

Consequence for us, stated as a hypothesis to measure rather than a claim: **we can be strictly
more accurate than CANN's MSD and no slower**, by keeping the cube output `int32_t` (as the A8W8
macro already does), doing `16*h + l` in int32 — exact, since `|16h+l| <= 16*8*K + 8*K` fits int32
for any K we will ever use — and applying the weight scale, the `8*sum(w)` term and the per-token
scale once, in fp32, in the vector unit. The cost is `int32` instead of `half` out of Fixpipe, i.e.
2x the cube→GM traffic on the mmOut workspace. With `quantGroupNum = 1` (which §1.2 licenses) item
2 disappears entirely and there is a single `Iterate`.

### Relevance

- **Our M-stacking is the vendor's own choice**, confirmed by
  `int32_t splitValue = (currSplitValue - prevSplitValue) * 2; // 2: int8 has been split in 2 int4`
  and a single `SetOrgShape(2M, N, K)`. There is no two-matmul variant in this operator. One
  refinement to copy: CANN interleaves (`[h_0, l_0, h_1, l_1, ...]`, `DOUBLE_ROW = 2`) rather than
  concatenating all highs then all lows. Interleaving keeps the two halves of one token in the same
  `baseM` tile (`baseM = 8` per the in-source comment), so the post stage reads one contiguous
  `2*tokenLen` run per token and the recombine is a `Muls`+`Add` with no gather. A blocked layout
  would put the pair `M` rows apart and, at `baseM = 8`, in different cube tiles.
- **`8*sum(w)` is right, and its sign and position are pinned.** CANN subtracts 8 from the
  activation's low nibble (`Adds(xHighHalfTensor, xLowHalfTensor2, MINUS_EIGHT, vK)`), so the
  correction is `+8*sum_k(w_k * w_scale_k)` per `(expert, n)`, applied **after** the `16h+l`
  recombine and **before** the per-token scale. That ordering is forced: the term carries the
  weight scale but not the activation scale. Our fold is the same quantity; check the order.
- **Name to use in our own code and when asking the vendor**: input
  `weightAssistanceMatrix` / member `weightAuxiliaryMatrixGM` / 「权重辅助矩阵」, `(E, N)` fp32.
- **Tiling datum to start from**: `// 8, 256, 512 --> 514us` — `baseM=8`, `baseN=256`,
  `quantGroupSize=512`. `baseM = 8` means 4 logical tokens per tile after doubling, which at
  batch=1 decode is already the right order for us.
- **The cheapest accuracy win available** is the one in §2.4: keep int32 out of the cube. CANN's
  own A8W8 macro in the same file does it; the MSD macro does not, and §1 does not admit it.

---

## 3. The same decomposition in plain `GroupedMatmul` — `gmm/grouped_matmul/`

Repository: `https://gitcode.com/cann/ops-transformer`, paths
`gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd{,_pre,_pre_nz,_new}.h` and
`gmm/grouped_matmul/op_kernel/grouped_matmul.cpp`. Same clone and commits as §2.

This is the operator the §1 article is actually about (`aclnnGroupedMatmulV4`). It carries the
*second*, better-engineered implementation of the same decomposition, plus the asymmetric-weight
path that §2's SwiGLU variant does not have. Read this one, not §2, if you are writing the kernel.

### 3.1 The type block, and the line that names the correction

`grouped_matmul_antiquant_a8w4_msd.h`, verbatim:

```cpp
#ifdef GMM_ANTI_QUANT_A8W4_MSD
namespace GROUPED_MATMUL {
using namespace matmul;
using namespace AscendC;

using DTYPE_PERTOKEN_SCALE_A8W4MSD = float;
using DTYPE_BIAS_A8W4MSD = float;
using DTYPE_OFFSET_A8W4MSD = float;

#ifdef GMM_ANTI_QUANT_A8W4_MSD_OUT_BF16
using DTYPE_Y_A8W4MSD = bfloat16_t;
#else
using DTYPE_Y_A8W4MSD = half;
#endif

using DTYPE_X_DEV_A8W4MSD = int4b_t;
using DTYPE_WEIGHT_DEV_A8W4MSD = int4b_t;
using DTYPE_SCALE_DEV_A8W4MSD = uint64_t;

constexpr uint64_t SYNC_AIV_TO_AIC = 3;
constexpr uint64_t SYNC_AIC_TO_AIV = 5;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t MM_BASE_BLOCK_OFFSET = 32768; // baseM * baseN = 128 * 256
```

And the member declarations — the comment on `biasGm` is the single most useful line in this file:

```cpp
    GlobalTensor<DTYPE_BIAS_A8W4MSD> biasGm; // for 8 * weight
    GlobalTensor<DTYPE_SCALE_DEV_A8W4MSD> scaleGm;
```

```cpp
    GlobalTensor<DTYPE_OFFSET_A8W4MSD> offsetGm;
    GlobalTensor<float> xRowSumGm;
```

`biasGm` is bound from the operator's **public `bias` argument**:

```cpp
    biasGm.SetGlobalBuffer(GetTensorAddr<DTYPE_BIAS_A8W4MSD>(0, bias));
    scaleGm.SetGlobalBuffer(GetTensorAddr<DTYPE_SCALE_DEV_A8W4MSD>(0, scale));
```

```cpp
        biasGm.SetGlobalBuffer(GetTensorAddr<DTYPE_BIAS_A8W4MSD>(groupIdx, biasTensorPtr));
```

So `aclnnGroupedMatmulV4`'s `biasOptional` is, on the A8W4-MSD tiling path, not a model bias at
all — it is the `8*sum(w)` MSD correction, fp32, one vector of length N per expert. That matches
§1.3's 「通常将这部分多余项放入bias参数中」 exactly, and it means the operator has **no spare slot
for a real bias** on this path.

`DataCopyScale` reads `biasGm` under a variable called `scale`, which is confusing on purpose
(「bias queue」 in the comment gives it away):

```cpp
    pipe->InitBuffer(scaleInQueue, BUFFER_NUM, mmTilingData->baseN * sizeof(DTYPE_BIAS_A8W4MSD)); // bias queue
```

```cpp
__aicore__ inline void GMMA8W4MSDCompute<mmType>::DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN,
                                                                uint64_t scaleOffset)
{
    // GM拷贝scale
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams scaleParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    LocalTensor<float> scaleLocal = scaleInQueue.AllocTensor<DTYPE_BIAS_A8W4MSD>();
    DataCopyPad(scaleLocal, biasGm[scaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);
    scaleInUb = scaleInQueue.DeQue<float>();
}
```

### 3.2 The MatmulConfig — `isPartialOutput`, and the `int32_t` bias slot that is left unused

`grouped_matmul.cpp`, verbatim:

```cpp
#if defined(GMM_ANTI_QUANT_A8W4_MSD)
constexpr MatmulConfig A8W4_GMM_CFG_MDL = GetNormalConfig();
constexpr auto GetMmCFG()
{
    auto CFG = CFG_MDL;
    CFG.isPartialOutput = true;
    return CFG;
}
constexpr MatmulConfig A8W4_GMM_CFG_MDL_NEW = GetMmCFG();
#endif
```

```cpp
#define GMM_CV_SPLIT_IMP_A8W4_MSD(computeClass, cfg) \
    do { \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling); \
        if ASCEND_IS_AIV { \
            if (gmmBaseParams_.isA8W4MSDPreNZ == 1) { \
                GMMA8W4PreProcessNZ op1; \
                op1.Init(x, user1, groupList, user1, gmmBaseParams_, &tPipe); \
                op1.Process(); \
            } else { \
                GMMA8W4PreProcess op1; \
                op1.Init(x, x, groupList, user1, gmmBaseParams_, &tPipe); \
                op1.Process(); \
            } \
            tPipe.Reset(); \
            tPipe.Destroy(); \
            tPipe.Init(); \
        } \
        if (gmmBaseParams_.isA8W4MSDPreNZ == 1) { \
            using aT = MatmulType<TPosition::GM, CubeFormat::NZ, DTYPE_X_DEV_A8W4MSD, false>; \
            using bT = MatmulType<TPosition::GM, wFormat, int4b_t, false>; \
            using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>; \
            using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>; \
            using matmulType = MMImplType<aT, bT, cT, biasT, cfg>; \
            ...
        } else { \
            using aT = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X_DEV_A8W4MSD, false>; \
            using bT = MatmulType<TPosition::GM, wFormat, DTYPE_WEIGHT_DEV_A8W4MSD, false>; \
            using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>; \
            using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>; \
            using matmulType = MMImplType<aT, bT, cT, biasT, cfg>; \
            ...
```

Note what this shows: the `Matmul` object **does** have an `int32_t` bias type parameter
(`biasT`), i.e. the hardware/template path for an int32 bias added inside the cube exists — and
the MSD implementation does not use it. The `8*sum(w)` correction is instead applied in fp32 in
the vector unit (§3.4). That is a choice, not a constraint, and it is the choice that costs
accuracy (§2.4), because by the time the correction is applied the partial products have already
been through `half`.

The AIV pre-stage runs inside the same kernel, then tears down and rebuilds the pipe
(`tPipe.Reset(); tPipe.Destroy(); tPipe.Init();`) before the cube stage — a reusable trick for
reclaiming the whole UB between phases of a megakernel.

### 3.3 `*2` again, and the authoritative statement of the A layout

```cpp
        mnConfig.m = static_cast<uint32_t>(m) * 2; // 2: int8 has been split in 2 int4
```

```cpp
    A8W4preM = AlignUp<8>(xRowSumCount) * 2;
```

```cpp
    uint32_t mGlobalOffset = mnConfig.offsetM / 2 + mnConfig.mIdx * curCubeSingleM; // 2: 2 lines int4 to 1 line int8
```

```cpp
    uint64_t perTokenScaleOffset =
        mnConfig.offsetM / 2 + mnConfig.mIdx * mnConfig.singleM / 2 + offsetM; // 2: m方向两行合并为1行
```

Two int4 rows per int8 row, 「m方向两行合并为1行」. The pre writes them adjacent, exactly as in §2.1:

```cpp
        DataCopy(yGm[startAddr], xHighI4Tensor.ReinterpretCast<int8_t>(), vK / 2); // 2: size int8 -> int4
        ...
        DataCopy(yGm[startAddr + vK / TWO], xLowI4Tensor.ReinterpretCast<int8_t>(), vK / TWO);
```

with `uint64_t startAddr = xloop * vK;` and `yGm` typed `int8_t` — so logical row `r` owns bytes
`[r*vK, (r+1)*vK)`, which as `int4b_t` rows of `K` elements is rows `2r` (high) and `2r+1` (low).
**Interleaved M-stacking, confirmed in both operators.**

### 3.4 The de-interleave: a strided `DataCopyPad`, then one `Muls` and one `Add` per tile

This is the part worth copying. The cube output tile has `2*curVecBaseM` rows, interleaved. The
vector stage reads it as two strided planes:

```cpp
            LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
            if constexpr (mmType::BT::format == CubeFormat::ND) {
                DataCopyPad2DA8W4ND(mmOutLocal, mmOutGm[mmOutOffset + offsetM * 2 * curVecBaseN], curVecBaseM,
                                    curVecBaseN, curVecBaseN * 2); // 2: 2 lines int4 to 1 line int8
            } else {
                DataCopyPad2DA8W4(mmOutLocal, mmOutGm[mmOutOffset + offsetM * 2 * curVecBaseN], curVecBaseM,
                                  curVecBaseN, curVecBaseN * 2); // 2: 2 lines int4 to 1 line int8
            }
            uint32_t targetAddr;
            if constexpr (mmType::BT::format == CubeFormat::ND) {
                alignBaseN = (alignBaseN + HALF_ALIGN - 1) / HALF_ALIGN * HALF_ALIGN;
            }
            targetAddr = curVecBaseM * alignBaseN;
            uint64_t lowBitAddr = mmOutOffset + (offsetM * 2 + 1) * curVecBaseN;
            if constexpr (mmType::BT::format == CubeFormat::ND) {
                DataCopyPad2DA8W4ND(mmOutLocal[targetAddr], mmOutGm[lowBitAddr], curVecBaseM, curVecBaseN,
                                    curVecBaseN * 2); // 2: 2 lines int4 to 1 line int8
            } else {
                DataCopyPad2DA8W4(mmOutLocal[targetAddr], mmOutGm[lowBitAddr], curVecBaseM, curVecBaseN,
                                  curVecBaseN * 2); // 2: 2 lines int4 to 1 line int8
            }
            vecInQueue.EnQue(mmOutLocal);
            ComputeDequantAndActivate(mnConfig, curVecBaseM, alignBaseN, curVecBaseN, offsetM);
```

with the two copy helpers, verbatim — note `srcStride` is `(srcDim0 - dim0)` with
`srcDim0 = 2*curVecBaseN` and `dim0 = curVecBaseN`, i.e. skip one row each time, and note the
reason the ND variant pads each row to an even number of 32B blocks:

```cpp
template <typename T>
__aicore__ inline void DataCopyPad2DA8W4(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1,
                                         uint32_t dim0, uint32_t srcDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    // 32: int32 -> float16, 为防止跨行数据进入同一32B block，提前每行按偶数block对齐
    params.dstStride = Ceil(dim0 * sizeof(T), 32) % 2;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4ND(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1,
                                           uint32_t dim0, uint32_t srcDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    params.dstStride = 0;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
    return;
}
```

After which the recombine is two whole-tile instructions, not a per-row loop:

```cpp
    uint32_t computeSize = curVecBaseM * alignBaseN;
    LocalTensor<cT::T> mmOutInUb = vecInQueue.DeQue<cT::T>();
    uint32_t castSize;
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        castSize = computeSize + (computeSize + HALF_ALIGN - 1) / HALF_ALIGN * HALF_ALIGN;
    } else {
        castSize = computeSize * 2;
    }
    Cast(buffer1, mmOutInUb, RoundMode::CAST_NONE, castSize);
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);
    const float RIGHT_MOVE = 16.0f; // right move int4 to int8
    Muls(buffer2, buffer1, RIGHT_MOVE, computeSize);
    PipeBarrier<PIPE_V>();
    uint32_t addStartAddr;
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        addStartAddr = (computeSize + HALF_ALIGN - 1) / HALF_ALIGN * HALF_ALIGN;
    } else {
        addStartAddr = computeSize;
    }
    Add(buffer3, buffer1[addStartAddr], buffer2, computeSize);
    PipeBarrier<PIPE_V>();
```

Then the `8*sum(w)` add, broadcast along M by `BinaryRepeatParams` with a zero src1 block stride
(`param(1, 1, 1, blkStride, blkStride, 0)`) — no materialised broadcast needed:

```cpp
    uint32_t loop = alignBaseN / 64; // 256B为64个float，alignBaseN需约束为64倍数
    uint8_t blkStride = static_cast<uint8_t>(alignBaseN * sizeof(float) / 32); // 32: 单位32B
    BinaryRepeatParams param(1, 1, 1, blkStride, blkStride, 0);
    uint64_t mask = 64;
    uint64_t last = alignBaseN % 64;
    for (uint32_t i = 0; i < loop; i++) {
        uint32_t offset = i * 64; // 每次64个元素
        Add(buffer2[offset], buffer3[offset], scaleInUb[offset], mask, curVecBaseM, param);
    }
    PipeBarrier<PIPE_V>();
    if (unlikely(last > 0)) {
        uint32_t offset = loop * 64;
        Add(buffer2[offset], buffer3[offset], scaleInUb[offset], last, curVecBaseM, param);
    }
    PipeBarrier<PIPE_V>();
```

### 3.5 The asymmetric-weight path — `withOffset`, `xRowSum`, and `offset`

`constexpr uint32_t WITH_OFFSET = 1;` in the pre header. When set, the pre stage additionally
reduces the **original int8 row** (before the `*0.0625`) to a single fp32 and writes it to
workspace:

```cpp
        Cast(xHighHalfTensor, xTensor, AscendC::RoundMode::CAST_NONE, vK);
        PipeBarrier<PIPE_V>();

        if (withOffset == WITH_OFFSET) {
            Cast(xHighFloatTensor, xHighHalfTensor, AscendC::RoundMode::CAST_NONE, vK);
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
            ReduceSum(xRowSumTensor, xHighFloatTensor, workTensor, vK);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

            DataCopyExtParams xRowSumParams{1, static_cast<uint32_t>(1 * sizeof(float)), 0, 0, 0};
            DataCopyPad(xRowSumGm[xloop], xRowSumTensor, xRowSumParams);
            AscendC::PipeBarrier<PIPE_V>();
        }

        Muls(xHighHalfTensor, xHighHalfTensor, one_eight, vK);
```

i.e. `xRowSum[m] = sum_k x_int8[m,k]`, fp32, one value per token. The post stage broadcasts it
along N and multiplies by the per-column `offset`:

```cpp
__aicore__ inline void GMMA8W4MSDCompute<mmType>::DataCopyAndBrcbOfRowSum(MNConfig& mnConfig, uint32_t curBaseM,
                                                                          uint32_t alignBaseN, uint32_t offsetM)
{
    const uint64_t xRowSumOffset =
        mnConfig.offsetM / 2 + mnConfig.mIdx * mnConfig.singleM / 2 + offsetM; // 2: m方向两行合并为1行
    ...
    const uint32_t xRowSumBroadCastDst[2] = {curBaseM, alignBaseN};
    const uint32_t xRowSumBroadCastSrc[2] = {curBaseM, 1};
    BroadCast<float, 2, 1>(buffer3, xRowSumLocal, xRowSumBroadCastDst, xRowSumBroadCastSrc, buffer6);
    xRowSumInQueue.FreeTensor(xRowSumLocal);
}
```

```cpp
    if (withOffset == WITH_OFFSET) {
        DataCopyAndBrcbOfRowSum(mnConfig, curVecBaseM, alignBaseN, offsetM);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < loop; i++) {
            uint32_t offset = i * 64; // 每次64个元素
            Mul(buffer4[offset], buffer3[offset], offsetInUb[offset], mask, curVecBaseM, param);
        }
        PipeBarrier<PIPE_V>();
        if (unlikely(last > 0)) {
            uint32_t offset = loop * 64;
            Mul(buffer4[offset], buffer3[offset], offsetInUb[offset], last, curVecBaseM, param);
        }
        PipeBarrier<PIPE_V>();

        Add(buffer5, buffer4, buffer2, computeSize);
        PipeBarrier<PIPE_V>();
    }

    DataCopyPerTokenScaleAndBrcb(mnConfig, curVecBaseM, alignBaseN, offsetM);
    PipeBarrier<PIPE_V>();

    LocalTensor<DTYPE_OUT> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_OUT>();
    if (withOffset == WITH_OFFSET) {
        Mul(buffer4, buffer5, buffer3, computeSize);
    } else {
        Mul(buffer4, buffer2, buffer3, computeSize);
    }
    PipeBarrier<PIPE_V>();
    Cast(yLocalInUb, buffer4, RoundMode::CAST_RINT, computeSize);
    vecOutQueue.EnQue(yLocalInUb);
```

### 3.6 The complete expression, assembled from §3.4 and §3.5

Reading the buffer traffic end to end, with `h`/`l` the two int4 halves, `s` the fused
`total_scale = perchannel_scale * pergroup_scale` applied in Fixpipe by `SetQuantVector`,
`b` = the `bias` input (`// for 8 * weight`), `z` = the `offset` input, and `t` = per-token scale:

```
P_h[m,n] = sum_k h[m,k] * w[k,n] * s[n]          (fp16 out of the cube)
P_l[m,n] = sum_k l[m,k] * w[k,n] * s[n]          (fp16 out of the cube)

y[m,n] = ( 16*P_h[m,n] + P_l[m,n]  +  b[n]  +  xRowSum[m] * z[n] ) * t[m]
```

evaluated in fp32 in the vector unit, then `Cast(..., CAST_RINT)` to `half` or `bfloat16_t`.
The `withOffset == 0` path drops the `xRowSum*z` term.

### Relevance

- **`bias` is the `8*sum(w)` slot, and the comment says so**: `// for 8 * weight`. Our fold is
  the vendor's, named. If we ever call `aclnnGroupedMatmulV4` on the A8W4-MSD path we must put the
  correction in `biasOptional` and we cannot also pass a model bias there.
- **Interleaved M-stacking, de-interleaved on MTE2.** Both CANN operators write
  `[h_0, l_0, h_1, l_1, ...]` and read it back with a `DataCopyPad` whose `srcStride` skips one
  tile row (`srcDim0 = 2*baseN`, `dim0 = baseN`). That turns the recombine into exactly two
  whole-tile vector instructions — `Muls(buffer2, buffer1, 16.0f, computeSize)` and
  `Add(buffer3, buffer1[addStartAddr], buffer2, computeSize)` — with no per-row loop and no
  gather. §2's SwiGLU variant loops per row; this one does not. Copy this one.
- **The asymmetric-weight correction is `xRowSum[m] * offset[n]`, and `xRowSum` is the sum of the
  *original* int8 row**, taken before the `*0.0625`. If our weights are symmetric
  (zero point 0, which is what msmodelslim's W4A8 recipe does per §1.1's 「所有量化/反量化操作均在
  整数域内完成」) we skip the whole `withOffset` branch and the extra `ReduceSum` per row in pre.
  That is one `ReduceSum` + one `BroadCast` + one `Mul` + one `Add` per tile saved.
- **A real int32 bias slot exists in the `Matmul` template and MSD does not use it**
  (`using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;`). If we want the
  exact-int32 variant from §2.4, that slot is where the `8*sum(w)` term — which *is* an integer,
  `8 * sum_k w[k,n]` before scaling — belongs. Folding it there makes the pre-scale accumulator
  exactly equal to `sum_k x_int8[m,k] * w[k,n]`, which is the whole point.
- **`MM_BASE_BLOCK_OFFSET = 32768; // baseM * baseN = 128 * 256`** is this operator's tile, against
  §2's `// 8, 256, 512 --> 514us`. Two different tilings for the same decomposition; at batch=1
  decode the `baseM = 8` one is ours.
- **`tPipe.Reset(); tPipe.Destroy(); tPipe.Init();`** between the AIV pre stage and the cube stage,
  inside one kernel. Directly useful for our megakernel: it is how CANN reclaims the whole UB
  across phases without a launch boundary.

---

## 4. The official operator specification — `aclnnGroupedMatmulSwigluQuantV2` and `aclnnGroupedMatmulV4/V5`

These are the published `aclnn` docs, shipped **inside the open-source repo** rather than only on
the portal, which is why they are quotable in full. Same clone as §2/§3.

```
gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuant.md
gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md
gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantWeightNzV2.md
gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantWeightNzV3.md
gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md
gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md
gmm/grouped_matmul/docs/aclnnGroupedMatmulWeightNz.md
```

The portal copies of the same pages, for citation:
`https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/aolapi/context/aclnnGroupedMatmulV4.md`
(given as the reference link at the end of §1's article).

### 4.1 The complete MSD formula, verbatim — `aclnnGroupedMatmulSwigluQuantV2.md`

Section header 「MSD场景A8W4（A指激活矩阵，W指权重矩阵，8指INT8数据类型，4指INT4数据类型）」, under
「Atlas A3系列产品、Atlas A2系列产品」. Verbatim, LaTeX and all:

> - **定义**：
>   * **⋅** 表示矩阵乘法。
>   * **⊙** 表示逐元素乘法。
>   * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
>   * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
>   * $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
>   * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
> - **输入**：
>   * $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
>   * $W∈\mathbb{Z_4}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
>   * $weightAssistMatrix∈\mathbb{R}^{E \times N}$：计算矩阵乘时的辅助矩阵（生成辅助矩阵的计算过程见下文）。
>   * $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，K\_group\_num是在K轴维度上的分组数，N是输出维度。
>   * $x\_scale∈\mathbb{R}^{M}$：激活矩阵（左矩阵）的逐token缩放因子，M是总token数。
>   * $grouplist∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
> - **输出**：
>   * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
>   * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
> - **计算过程**
>   - 1.根据groupList[i]确定当前分组的token，$i \in [0,Len(groupList)]$。
>     - 分组逻辑与A8W8相同。
>   - 2.生成辅助矩阵（weightAssistMatrix）的计算过程（请注意weightAssistMatrix部分计算为离线生成作为输入，并非算子内部完成）：
>     - 当为per-channel量化（$w\_scale$为2维）：
>
>       $weightAssistMatrix_{i} = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]$
>
>     - 当为per-group量化（$w\_scale$为3维）：
>
>       $weightAssistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × weightScale[:, ⌊k/num\_per\_group⌋, :])$
>
>       注：$num\_per\_group = K // K\_group\_num$
>
>   - 3.根据分组确定的入参进行如下计算：
>
>     - 3.1.将左矩阵$\mathbb{Z_8}$，转变为高低位两部分的$\mathbb{Z_4}$
>       $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$
>       $X\_low\_4bits_{i} = X_{i} \& 0x0f - 8$
>     - 3.2.做矩阵乘时，开启per-channel或per-group量化
>       per-channel：
>
>       $C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$
>
>       $C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$
>
>       per-group：
>
>       $C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$
>
>       $C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$
>
>     - 3.3.将高低位的矩阵乘结果还原为整体的结果
>
>       $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}$
>
>       $C_{i,act}, gate_{i} = split(C_{i})$
>
>       $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;其中$Swish(x)=\frac{x}{1+e^{-x}}$
>
>   - 3.量化输出结果
>
>     $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$
>
>     $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$

Compare, from the same file, the 「量化场景A8W8」 block — note that its definitions list is
identical **minus `Z_4`**, and that its formula has no intermediate at all:

> - **定义**：
>   * **⋅** 表示矩阵乘法。
>   * **⊙** 表示逐元素乘法。
>   * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
>   * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
>   * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
> ...
>   - 2.根据分组确定的入参进行如下计算：
>
>     $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

Both blocks define `Z_32` and **neither MSD formula ever uses it**: `C_high` and `C_low` are
stated to be in ℝ after the `⊙ w_scale`, which is the doc quietly conceding §2.4 — the per-half
products leave the integer domain before they are combined. §1.6's 「所有主流程均在整数域内完成」
is therefore an overstatement of its own operator.

The `weightAssistMatrix` parameter entry:

> | weightAssistMatrix | 表示计算矩阵乘时的辅助矩阵，公式中的weightAssistMatrix。<br> • 仅A8W4场景生效，其他场景需传空指针。 |

and in V1, which spells out the non-tensorlist signature and the "offline" caveat again:

> * $weightAssistMatrix∈\mathbb{R}^{E \times N}$：计算矩阵乘时的辅助矩阵（生成辅助矩阵的计算过程见下文）。
> - 2.生成辅助矩阵（weightAssistMatrix）的计算过程（请注意weightAssistMatrix部分计算为离线生成作为输入，并非算子内部完成）：
>   $weightAssistMatrix_{i} = 8 × w\_scale × Σ_{k=0}^{K-1} weight[:,k,:]$

> | 权重矩阵，对应公式中的W。 <br> • INT32为适配用途，实际1个INT32会被解释为8个INT4数据。<br> • A8W4场景：shape支持3或5维，分别对应数据格式ND、FRACTAL_NZ。 | INT8、INT4、INT32 |
> | 计算矩阵乘时的辅助矩阵，对应公式中的weightAssistMatrix。<br> 仅A8W4场景生效，A8W8场景需传空指针。 |
> | per-channel非对称反量化的偏移，对应公式中的offset。 |
> | 右矩阵的量化因子，对应公式中的w_scale。 <br> • A8W4场景：shape支持2或3维，数据类型支持UINT64。 |

V2 differs from V1 only in plurality:

> - 参数weight, weightScale, weightAssistMatrix的字段类型变为tensorlist，请根据实际情况选择合适的接口。

### 4.2 `aclnnGroupedMatmulV4.md` — A8W4 is classified 伪量化, and its output can never be INT32

The scenario table, verbatim:

> |场景名|    x    |    weight      |   out | 约束说明|计算公式|
> |---------|---------|----------------|--------|--------|--|
> |非量化|FLOAT32|FLOAT32|FLOAT32|...|
> |非量化|BFLOAT16|BFLOAT16|BFLOAT16|...|
> |非量化|FLOAT16|FLOAT16|FLOAT16|...|
> |全量化-A8W8|INT8|INT8|BFLOAT16/FLOAT16/INT32/INT8|[A8W8场景约束](#a8w8场景约束)|[计算公式](#全量化场景)|
> |全量化-A4W4|INT4|INT4|BFLOAT16/FLOAT16|[A4W4场景约束](#a4w4场景约束)|[计算公式](#全量化场景)|
> |伪量化-A8W4|INT8|INT4|BFLOAT16/FLOAT16|[A8W4场景约束](#a8w4场景约束)|[计算公式](#a8w4伪量化场景)|
> |伪量化-A16W8|BFLOAT16/FLOAT16|INT8|BFLOAT16/FLOAT16|...|
> |伪量化-A16W4|BFLOAT16/FLOAT16|INT4|BFLOAT16/FLOAT16|...|

The four formulas, side by side, verbatim — this is the cleanest statement of the accuracy
difference anywhere in the material:

> - x为INT8，bias为INT32
>
>   $$ y_i=(x_i\times weight_i + bias_i) * scale_i * per\_token\_scale_i $$
>
> - x为INT8，bias为BFLOAT16
>
>   $$ y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i  + bias_i $$
>
> - x为INT4，无bias
>
>   $$ y_i=x_i\times (weight_i * scale_i) * per\_token\_scale_i $$

> <a id="a8w4伪量化场景"></a>
>
> - x为INT8，weight为INT4（仅支持x、weight、y均为单tensor的场景）。其中$bias$为必选参数，是离线计算的辅助结果，且 $bias_i=8\times weight_i  * scale_i$ ，并沿k轴规约。
>
>   $$ y_i=((x_i - 8) \times weight_i * scale_i+bias_i ) * per\_token\_scale_i $$

A8W8 with an INT32 bias puts the bias **inside** the parenthesis, before `scale` — an integer add
in the accumulator. A8W4-MSD puts it **after** `* scale_i` — a float add. Identical to what §3.4's
buffer traffic does.

Note also how the doc writes the split: `(x_i - 8) × weight_i`, as if one shifted matmul. The
split into two is not mentioned in the `GroupedMatmulV4` doc at all; only the SwiGLU variant's doc
(§4.1) spells out `C_high`/`C_low`. Searching the V4 page for "MSD" finds nothing.

### 4.3 `aclnnGroupedMatmulV4.md` — the A8W4 constraints, verbatim

> **数据类型要求**
>
> | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
> |---|---|---|---|---|---|---|---|---|---|---|---|---|
> | INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | null | null | null | FLOAT | INT64 | null | null | null | BFLOAT16|
> | INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | FLOAT/null | null | null | FLOAT | INT64 | null | null | null | FLOAT16 |
>
> **约束说明**
>
> 除[公共约束](#公共约束)外，A8W4场景其余约束如下：
> - 仅支持GroupType=0（M轴分组），actType=0
> - 当前仅支持x、weight、out均为长度为1的TensorList
> - x不支持转置、weight不支持转置
> - x仅支持2维Tensor，Shape为（M，K）
> - weight默认支持3维Tensor，Shape为（E，K，N）
> - Bias为计算过程中离线计算的辅助结果，值要求为$8\times weight \times scale$，并在第1维累加，shape要求为$[E, N]$。
> - 当weight传入数据类型为INT32时，会将每个INT32视为8个INT4。
> - offset为空时
>   - 该场景下仅支持groupListType为1（算子不会检查groupListType的值，会认为groupListType为1），k要求为quantGroupSize的整数倍，且要求k <= 18432。其中quantGroupSize为k方向上pergroup量化长度，当前支持quantGroupSize=256。
>   - 该场景下要求n为8的整数倍。
>   - 该场景下scale为pergroup与perchannel离线融合后的结果，shape要求为$[E, quantGroupNum, N]$，其中$quantGroupNum=k \div quantGroupSize$。
>   - 该场景下，各个专家处理的token数的预期值大于n/4时，即tuningConfigOptional中第一个值大于n/4时，通常会取得更好的性能，此时显存占用会增加$g\times k \times n$字节（其中g为matmul组数即分组数）。
> - offset不为空时
>   - scale为pergroup与perchannel离线融合后的结果，shape要求为$[E, 1, N]$。
>   - 该场景下offsetOptional不为空。非对称量化offsetOptional为计算过程中离线计算辅助结果，即$antiquantOffset \times scale$，shape要求为$[E, 1, N]$，dtype为FLOAT32。
> - tuningConfigOptional数组第二个元素可置1，以开启A8W4场景中weight的特殊格式模板，以优化算子性能(性能优势的shape范围参考:K >= 2048 && N >= 2048)。需要说明的是，该模板要求weight的shape为（E，N，K）,然后再对其进行ND2NZ转换后作为算子输入。

From `aclnnGroupedMatmulV5.md`, the `tuningConfigOptional` triple is spelled out:

> | tuningConfigOptional（aclIntArray *） | 可选输入 | 第一个数代表各个专家处理的 token数的预期值；第二个数代表 A8W4 可选开启weight先转置的NZ格式；第三个数代表允许额外使用的内存空间。 | 兼容历史版本，用户如不使用该参数，不传（即为 nullptr）即可 | INT64 | - | 3 | - |

> - `[0]`：预期各专家处理的token数，tiling 按此优化，适用场景：A8W4/A8W8/A4W4，x/weight/out 单Tensor。
> - `[1]`：是否开启weight亲和格式（先转置再 NZ），适用场景：A8W4。

> - A8W8、A8W4、A4W4：仅支持 groupType=0，且 x 为单 tensor。
> | x不转置；weight可转置（A8W4/A4W4 不可） |

> - 确定性计算：
>   - aclnnGroupedMatmulV4默认确定性实现。

### Relevance

- **Our `8*sum(w)` fold is the published formula, character for character.**
  `weightAssistMatrix = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]`, `(E, N)`, fp32, offline,
  and 「并非算子内部完成」. Our high/low extraction is also the published formula:
  `X_high_4bits = ⌊X/16⌋`, `X_low_4bits = X & 0x0f - 8`. Nothing in our derivation needs changing.
- **`quantGroupSize = 256` is the only value the public operator accepts** (「当前支持
  quantGroupSize=256」), with `k % 256 == 0`, `k <= 18432`, `n % 8 == 0`. The kernel comment in §2.2
  says `512` for the SwiGLU variant's tiling, so the two operators differ — do not assume one
  number. Also note `quantGroupSize = 256` with our K contradicts §1.2's advice to skip per-group
  entirely: the `offset == null` path *forces* per-group. To get one scale per column you must take
  the `offset != null` branch, where 「scale ... shape要求为$[E, 1, N]$」.
- **The tuning hint is pointed the wrong way for us.** 「各个专家处理的token数的预期值大于n/4时
  ... 通常会取得更好的性能」. At batch=1 decode with top-k routing, tokens per expert is 1-8 and
  `n/4` is in the hundreds. We are two orders of magnitude below the threshold where CANN's own
  tiling pays off — which is the quantitative reason to write our own rather than call theirs.
- **`out` for A8W4 is `BFLOAT16` or `FLOAT16` only, never INT32 or INT8**, while A8W8 offers
  `BFLOAT16/FLOAT16/INT32/INT8`. If we want an int32 result out of a W4A8 GEMM, no public CANN
  operator will give it to us. That is the API-level face of §2.4.
- **`weight` may be passed as INT32** — 「实际1个INT32会被解释为8个INT4数据」 — useful if our
  weight loader has no int4 dtype.
- **Weight transpose is forbidden on A8W4/A4W4** (「weight可转置（A8W4/A4W4 不可）」), and the
  affine-format fast path wants `(E, N, K)` then ND2NZ with `tuningConfig[1] = 1`, advantageous at
  `K >= 2048 && N >= 2048`. Our GLM-5.2 expert shapes are in that range.
- **`aclnnGroupedMatmulV4默认确定性实现`** — determinism is claimed by default, which matters when
  we A/B our kernel against it.
