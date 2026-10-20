# Fused MoE Expert Kernels — Source Material

Verbatim research deposit for the GLM-5.2-W4A8C8 / Ascend A3 megakernel effort.
All quoted blocks are reproduced as published. Chinese sources are left in Chinese.
Collected 2026-10-20.

---

## 1. CANN first-party: the op we are already paying for

### GroupedMatmulSwigluQuantV2 — official op README (fusion scope, A8W4 MSD math)

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul_swiglu_quant_v2/README.md
Huawei / CANN `ops-transformer` repo, master branch.

> # GroupedMatmulSwigluQuantV2
>
> ## 产品支持情况
>
> | 产品 | 是否支持 |
> | :---------------------------- | :-----------: |
> | <term>Ascend 950PR&950DT系列产品</term> | √ |
> | <term>Atlas A3系列产品</term> | √ |
> | <term>Atlas A2系列产品</term> | √ |
> | <term>Atlas 200I/500 A2推理产品</term> | × |
> | <term>Atlas推理系列产品</term> | × |
> | <term>Atlas训练系列产品</term> | × |
>
> ## 功能说明
>
> - 算子功能：融合GroupedMatmul 、dequant、swiglu和quant，详细解释见计算公式。

The A8W8 path (what the fusion actually computes):

```
- 1.根据groupList[i]确定当前分组的token ，$i \in [0,Len(groupList)]$。

  >例子：假设groupList=[3,4,4,6]、groupListType=cumsum或groupList=[3,1,0,2]、groupListType=count。
  >
  >注：以上两种不同的分组方式，实际为相同的分组结果。
  >
  >第0个右矩阵`W[0,:,:]`，对应索引位置[0,3)的token`x[0:3]`（共3-0=3个token），对应`x_scale[0:3]`、`w_scale[0]`、`bias[0]`、`offset[0]`、`Q[0:3]`、`Q_scale[0:3]`、`Q_offset[0:3]`；
  >
  >第1个右矩阵`W[1,:,:]`，对应索引位置[3,4)的token`x[3:4]`（共4-3=1个token），对应`x_scale[3:4]`、`w_scale[1]`、`bias[1]`、`offset[1]`、`Q[3:4]`、`Q_scale[3:4]`、`Q_offset[3:4]`；
  >
  >第2个右矩阵`W[2,:,:]`，对应索引位置[4,4)的token`x[4:4]`（共4-4=0个token），对应`x_scale[4:4]`、`w_scale[2]`、`bias[2]`、`offset[2]`、`Q[4:4]`、`Q_scale[4:4]`、`Q_offset[4:4]`；
  >
  >第3个右矩阵`W[3,:,:]`，对应索引位置[4,6)的token`x[4:6]`（共6-4=2个token），对应`x_scale[4:6]`、`w_scale[3]`、`bias[3]`、`offset[3]`、`Q[4:6]`、`Q_scale[4:6]`、`Q_offset[4:6]`；
  >
  >注：grouplist中未指定的部分将不会参与更新。
  >例如当groupList=[12,14,18]、GroupListType=cumsum，X的shape为[30，:]时。
  >
  >则第一个输出Q的shape为[30，:]，其中Q[18:，：]的部分不会进行更新和初始化，其中数据为显存空间申请时的原数据。

- 2.根据分组确定的入参进行如下计算：

  $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}$

  $C_{i,act}, gate_{i} = split(C_{i})$

  $S_{i}=Swish(C_{i,act})\odot gate_{i}$  其中$Swish(x)=\frac{x}{1+e^{-x}}$

- 3.量化输出结果

  $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

  $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
```

The A8W4 **MSD** path — this is the one we run. MSD = the INT8 activation is split into
two INT4 halves and **two** matmuls are issued per group, plus an offline correction matrix:

```
<summary>MSD场景A8W4（A指激活矩阵，W指权重矩阵，8指INT8数据类型，4指INT4数据类型）：</summary>

  - **定义**：
    * **⋅** 表示矩阵乘法。
    * **⊙** 表示逐元素乘法。
    * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
    * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
    * $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
    * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
  - **输入**：
    * $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
    * $W∈\mathbb{Z_4}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
    * $weightAsistMatrix∈\mathbb{R}^{E \times N}$：计算矩阵乘时的辅助矩阵（生成辅助矩阵的计算过程见下文）。
    * $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，K\_group\_num是在K轴维度上的分组数，N是输出维度。
    * $x\_scale∈\mathbb{R}^{M}$：激活矩阵（左矩阵）的逐token缩放因子，M是总token数。
    * $grouplist∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
  - **输出**：
    * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
    * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
  - **计算过程**
    - 1.根据groupList[i]确定当前分组的token，$i \in [0,Len(groupList)]$。
      - 分组逻辑与A8W8相同。
    - 2.生成辅助矩阵（weightAsistMatrix）的计算过程（请注意weightAsistMatrix部分计算为离线生成作为输入，并非算子内部完成）：
      - 当为per-channel量化（$w\_scale$为2维）：

        $weightAsistMatrix_{i} = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]$

      - 当为per-group量化（$w\_scale$为3维）：

        $weightAsistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × weightScale[:, ⌊k/num\_per\_group⌋, :])$

        注：$num\_per\_group = K // K\_group\_num$

    - 3.根据分组确定的入参进行如下计算：

      - 3.1. 将左矩阵$\mathbb{Z}_8$，转变为高低位两部分的$\mathbb{Z}_4$
        $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$
        $X\_low\_4bits_{i} = (X_{i} \& 0x0f) - 8$
      - 3.2. 做矩阵乘时，采用per-channel或per-group量化
        per-channel：

        $C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

        $C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

        per-group：

        $C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

        $C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

      - 3.3.将高低位的矩阵乘结果还原为整体的结果

        $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAsistMatrix_{i}) \odot x\_scale_{i}$

        $C_{i,act}, gate_{i} = split(C_{i})$

        $S_{i}=Swish(C_{i,act})\odot gate_{i}$  其中$Swish(x)=\frac{x}{1+e^{-x}}$
```

A3-specific parameter and shape rules, including the NZ weight layouts:

```
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
    - x仅支持INT8量化数据类型、不支持其他数据类型。
    - weight仅支持非转置，支持INT8、INT4、INT32数据类型，ND格式shape形如{(E, K, N)}，NZ格式下，当weight数据类型是INT8时shape形如{(E, N / 32, K / 16, 16, 32)}，INT4时shape形如{(E, N / 64, K / 16, 16， 64)}，INT32时shape形如{(E, N / 64, K / 16, 16， 8)}。
    - weightScale，A8W8场景支持FLOAT、FLOAT16、BFLOAT16数据类型，shape只支持2维，形如{(E, N)}；A8W4场景支持UINT64数据类型，shape支持2维和3维，其中per-channel的shape形如{(E, N)}，per-group的shape形如{(E, KGroupCount, N)}。
    - 支持dequantMode参数：0表示激活矩阵per-token，权重矩阵per-channel；1表示激活矩阵per-token，权重矩阵per-group。
    - 不支持dequantDtype参数。
    - 不支持quantMode参数。
    - A8W8/A8W4场景，不支持N轴长度超过10240。
    - A8W8场景，不支持x的尾轴长度大于等于65536。
    - A8W4场景，不支持x的尾轴长度大于等于20000。
    - output仅支持数据类型INT8，shape支持2维，形如(M, N / 2)。
    - outputScale仅支持数据类型FLOAT，shape支持1维，形如(M,)。

  - 确定性计算：
      - aclnnGroupedMatmulSwigluQuantV2默认为确定性实现。
```

## Relevance

This is the exact op that costs us 6.793 ms against 2.272 ms of bytes. The MSD A8W4 formula
shows why: per group it issues **two** INT4 MMADs (`C_high` and `C_low`) over the same
`W_i`, then a vector pass for `C_high*16 + C_low + weightAssistMatrix`, then dequant, split,
Swish, multiply, and a per-token max-abs reduction for `Q_scale`. The weight bytes are read
once but the MMAD count and the vector-pass count are both doubled relative to a plain A8W8
grouped GEMM — so at M=8 (batch=1, top-8) the op is entirely fixed-cost, which matches the
3.0x gap we measure. `weightAssistMatrix` is offline, so it is free to carry into a megakernel.

---

### aclnnGroupedMatmulSwigluQuantV2 — exact signature and `groupListType` / `tuningConfig` semantics

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md

```cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(
    const aclTensor     *x,
    const aclTensorList *weight,
    const aclTensorList *weightScale,
    const aclTensorList *weightAssistMatrix,
    const aclTensor     *bias,
    const aclTensor     *xScale,
    const aclTensor     *smoothScale,
    const aclTensor     *groupList,
    int64_t              dequantMode,
    int64_t              dequantDtype,
    int64_t              quantMode,
    int64_t              groupListType,
    const aclIntArray   *tuningConfigOptional,
    aclTensor           *output,
    aclTensor           *outputScale,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor);
```

> | groupListType | 输入 | 表示分组的解释方式，用于确定groupList的语义。 | 0表示cumsum模式，groupList中的每个元素代表当前分组的累计长度。 1表示count模式，groupList中的每个元素代表该分组包含多少元素。 | INT64 |
>
> | tuningConfigOptional | 可选输入 | 用于算子预估M/E的大小，走不同的算子模板，以适配不同场景性能要求。 | 数组，传入的第一个数字表示各个专家处理的token数的预期值，用于优化tiling，A4W4 右矩阵NZ输入时开启，其他输入请传入空指针。 | INT64 |

> - weightAssistMatrix：
>   - 仅A8W4场景生效，其他场景需传空指针。
> - groupList：表示输入和输出在分组轴方向的matmul大小分布，对应公式中的grouplist，具体含义由groupListType确定。
>   - 当groupListType为0时，最后一个值不大于x中tensor的第一维，当groupListType为1时，数值的总和不大于x中tensor的第一维。
>   - groupList中的值约束了输出数据的有效部分，groupList未指定的部分将不会参与更新。
> - weight在A4W4下支持NZ输入转置，其他场景仅支持非转置。INT32为A8W4和A4W4场景下的适配用途，实际1个INT32会被解释为8个INT4数据，A8W8场景不支持ND数据格式。
> - 支持dequantMode参数：A8W4和A4W4场景支持取值0和1，A8W8场景仅支持取值0。
> - weight、weightScale和weightAssistMatrix支持单Tensor场景（tensorlist长度为1）和多Tensor场景（tensorlist长度大于1）。
> - A8W4场景下，不支持N轴长度超过10240，不支持x的尾轴长度大于等于20000。
> - 多tensor场景下，即tensorlist长度大于1时，weight、weightScale和weightAssistMatrix的shape需要按照E的维度展平，例如{(E, K, N)}需要变成{E个(K, N)}。

## Relevance

`tuningConfigOptional` is the only published hook for telling the op "expect this many tokens
per expert", and the doc says it is only honoured for A4W4-NZ — i.e. on our A8W4 path there is
no way to tell the stock op that M=8. That is a direct argument for owning the tiling ourselves.
`groupList未指定的部分将不会参与更新` matters for a megakernel: output tail is uninitialized
garbage, so a fused consumer must not read past `groupList[-1]`.

---

### The A8W4-MSD kernel is already a 3-stage software pipeline with grid-wide barriers

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h

```cpp
template <class mmType>
class GMMSwigluQuantPipelineSchedule {
    ...
    // 前处理GMMA8W4PreProcess类
    GMMA8W4PreProcess preProcess;
    // 中间处理GMMA8W4MidProcess类
    GMMA8W4MidProcess<mmType> midProcess;
    // 后处理GMMA8W4PostProcess类
    GMMA8W4PostProcess postProcess;
```

```cpp
template <class mmType>
__aicore__ inline void GMMSwigluQuantPipelineSchedule<mmType>::Process()
{
    // 1.对每次workspace切分做大循环。
    preProcess.Init(gmAddrParams, gmmSwigluQuantV2BaseParams);
    midProcess.Init(gmAddrParams, gmmSwigluQuantV2BaseParams);
    postProcess.Init(gmAddrParams, gmmSwigluQuantV2BaseParams, gmmSwigluQuantV2);

    // 1.前处理提前下发一次
    preProcess.Process(workspaceSplitConfig, 0, pipe);
    for (int64_t workspaceSplitLoopIdx = 0; workspaceSplitLoopIdx < workspaceSplitConfig.loopCount;
         workspaceSplitLoopIdx++) {
        // 更新workspaceSplitConfig
        UpdateWorkSpaceSplitConfig(workspaceSplitConfig, workspaceSplitLoopIdx);
        if ASCEND_IS_AIV {
            pipe->Reset();
        }

        SyncAll<false>();
        // 2.第n次中处理 && 第n+1次前处理 && 第n-1次后处理 并行
        midProcess.Process(workspaceSplitConfig, workspaceSplitLoopIdx);

        preProcess.Process(workspaceSplitConfig, workspaceSplitLoopIdx + 1, pipe);
        if ASCEND_IS_AIV {
            pipe->Reset();
            SyncAll<true>();
        }
        postProcess.Process(tempWorkspaceSplitConfig, workspaceSplitLoopIdx - 1, pipe);
        // 3.第n-1次后处理需要保留第n次的切分数据
        tempWorkspaceSplitConfig = workspaceSplitConfig;
        // reset
        if ASCEND_IS_AIV {
            pipe->Reset();
        }
        SyncAll<false>();
        // 3.前一次后处理 && 后一次MM 并行
    }
    // reset
    if ASCEND_IS_AIV {
        pipe->Reset();
    }
    SyncAll<false>();
    // // 4.最后一次后处理
    postProcess.Process(workspaceSplitConfig, workspaceSplitConfig.loopCount - 1, pipe);
    if ASCEND_IS_AIV {
        pipe->Destroy();
    }
}
```

## Relevance

This is the megakernel pattern already in production inside one op: `pre(n+1) || mid(n) || post(n-1)`
over workspace tiles, with `SyncAll<>()` as the grid-wide barrier and `ASCEND_IS_AIV` selecting
the role of each core. Our megakernel can copy this skeleton and extend the loop body across
op boundaries rather than restarting it per op. Note the cost it implies at batch=1: four
`SyncAll` per iteration plus `pipe->Reset()`, all of which are fixed and unamortized when M=8.

---

### `group_list` is walked with scalar global-memory reads inside the kernel

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h

```cpp
template <class mmType>
__aicore__ inline void GMMSwigluQuantPipelineSchedule<mmType>::InitWorkSpaceSplitConfig(
    WorkSpaceSplitConfig& workspaceSplitConfig)
{
    if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
        workspaceSplitConfig.M = groupListGM.GetValue(gmmSwigluQuantV2->groupListLen - 1);
    } else {
        int64_t totalTmp = 0;
        for (uint32_t i = 0; i < gmmSwigluQuantV2->groupListLen; i++) {
            totalTmp += groupListGM.GetValue(i);
        }
        workspaceSplitConfig.M = totalTmp;
    }
    workspaceSplitConfig.loopCount = Ceil(workspaceSplitConfig.M, gmmSwigluQuantV2BaseParams->mLimit);
    workspaceSplitConfig.notLastTaskSize = gmmSwigluQuantV2BaseParams->mLimit;
    workspaceSplitConfig.lastLoopTaskSize =
        workspaceSplitConfig.M - (workspaceSplitConfig.loopCount - 1) * gmmSwigluQuantV2BaseParams->mLimit;
    ...
}
```

```cpp
template <class mmType>
__aicore__ inline void GMMSwigluQuantPipelineSchedule<mmType>::UpdateWorkSpaceSplitConfig(
    WorkSpaceSplitConfig& workspaceSplitConfig, int32_t workspaceSplitLoopIdx)
{
    if (workspaceSplitLoopIdx < 0)
        return;
    workspaceSplitConfig.leftMatrixStartIndex = workspaceSplitLoopIdx * gmmSwigluQuantV2BaseParams->mLimit;
    workspaceSplitConfig.rightMatrixExpertStartIndex = workspaceSplitConfig.rightMatrixExpertNextStartIndex;
    workspaceSplitConfig.rightMatrixExpertEndIndex = workspaceSplitConfig.rightMatrixExpertStartIndex;
    // 计算右专家矩阵的终止索引(rightMatrixExpertEndIndex) 和下一次的起始索引(rightMatrixExpertNextStartIndex)
    int32_t curTaskNum = 0;
    int32_t nextTaskNum = 0;
    int32_t curTaskNumTmp = 0;
    int32_t nextTaskNumTmp = 0;
    if (gmmSwigluQuantV2BaseParams->groupListType == 1) {
        for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertEndIndex; i++) {
            curTaskNumTmp += groupListGM.GetValue(i);
        }
        if (workspaceSplitConfig.rightMatrixExpertEndIndex == 0) {
            nextTaskNumTmp = groupListGM.GetValue(0);
        } else {
            for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertEndIndex; i++) {
                nextTaskNumTmp += groupListGM.GetValue(i);
            }
        }
    }
    while (workspaceSplitConfig.rightMatrixExpertEndIndex < gmmSwigluQuantV2->groupListLen) {
        if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
            curTaskNum = groupListGM.GetValue(workspaceSplitConfig.rightMatrixExpertEndIndex) -
                         workspaceSplitConfig.leftMatrixStartIndex;
        } else {
            curTaskNumTmp += groupListGM.GetValue(workspaceSplitConfig.rightMatrixExpertEndIndex);
            curTaskNum = curTaskNumTmp - workspaceSplitConfig.leftMatrixStartIndex;
        }
        int32_t nextTaskIdx = workspaceSplitConfig.rightMatrixExpertEndIndex >= gmmSwigluQuantV2->groupListLen - 1 ?
                                  gmmSwigluQuantV2->groupListLen - 1 :
                                  workspaceSplitConfig.rightMatrixExpertEndIndex + 1;
        if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
            nextTaskNum = groupListGM.GetValue(nextTaskIdx) - workspaceSplitConfig.leftMatrixStartIndex;
        } else {
            if (workspaceSplitConfig.rightMatrixExpertEndIndex < gmmSwigluQuantV2->groupListLen - 1) {
                nextTaskNumTmp += groupListGM.GetValue(nextTaskIdx);
            }
            nextTaskNum = nextTaskNumTmp - workspaceSplitConfig.leftMatrixStartIndex;
        }
        if (curTaskNum > gmmSwigluQuantV2BaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex;
            break;
        } else if (curTaskNum == gmmSwigluQuantV2BaseParams->mLimit &&
                   nextTaskNum > gmmSwigluQuantV2BaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex + 1;
            break;
        } else if (nextTaskNum > gmmSwigluQuantV2BaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertEndIndex++;
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex;
            break;
        }
        workspaceSplitConfig.rightMatrixExpertEndIndex++;
    }
    workspaceSplitConfig.isLastLoop = workspaceSplitLoopIdx == workspaceSplitConfig.loopCount - 1 ? true : false;
    ...
}
```

## Relevance

Both helpers resolve the group layout at runtime with per-element `groupListGM.GetValue()`
scalar global-memory loads on the scalar unit, and the `while` loop advances one expert per
iteration up to `groupListLen`. With E=256 and total M=8 (batch=1, top-8) the loop cannot exit
early on the `mLimit` tests, so it walks the full expert list doing two dependent scalar GM
reads per step — a serial scalar cost that is independent of how little math there is to do.
A batch=1 megakernel can instead carry the 8 active expert ids in registers or UB and skip the
walk entirely.

---

### `GroupedMatmul算子设计介绍` — how CANN schedules tiles across groups (base-block and diagonal partitioning)

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/docs/GroupedMatmul%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E4%BB%8B%E7%BB%8D.md
CANN `ops-transformer`, licensed CC-BY 4.0 per the file header.

> 声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。
>
> # 1 GroupedMatmul融合算子设计介绍
>
> GroupedMatmul算子的功能是进行分组矩阵乘计算，每组矩阵乘的维度大小可以不同。矩阵乘公式如下：
> $$
> y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g
> $$
> 其中g为分组个数，$m_i、k_i、n_i$为对应shape。
> GroupedMatmul算子实现时还需要考虑如下两个方面：
>
> 1. 支持不同的参数、数据类型，如有无bias、不同激活函数类型，非量化、量化、伪量化等不同场景；不同场景对应的计算流程不同，性能优化方法不同，因此实现上划分成了不同的模板，有各自的模板参数；
> 2. 硬件上AiCore内存大小有限，一般完成一个算子的计算需要对数据进行切分，并对数据搬运和计算过程进行流水并行排布，该过程对算子的影响非常大，也是性能优化阶段主要调整对象，而host上的tiling函数即是为完成该切分和流水的参数计算。

> # 2 场景划分
>
> 从功能角度可分为非量化场景、量化场景和伪量化场景，代码层面通过三种方式选择具体模板：
>
> 1. 编译宏：通过x和weight的数据类型编译的宏ORIG_DTYPE_X，ORIG_DTYPE_WEIGHT；
> 2. tilingkey：如x/weight是否转置；
> 3. tilingData：如tiling中isPerTokenQuant表示是否为per token量化。
> 说明：
>
> - 非量化指x、weight、y均为浮点数类型，如float16/bfloat16/float32，非量化为纯cube场景，其计算过程由matmul高阶api实现；
> - 量化指x和weight为低精度整数类型，GroupedMatmul支持A8W8场景，包括per-tensor + per-channel量化和per-token + per-channel量化（简称per-token量化）；
> - 伪量化指x为浮点数类型，weight为低精度整数类型，GroupedMatmul支持A16W8和A16W4场景。
>
> ## 2.1 per token量化
>
> 本章以per token量化场景为例介绍一下GroupedMatmul的算法流程，计算过程如下：
> matmul(int32) -> 反量化(fp32) -> mul(fp32) -> 激活函数(fp32)(可选) -> cast(fp16/bf16)，其中mul(fp32)的输入perTokenScale还需要从shape(m) broadcast成(m，n)。

Grouping axis semantics:

> ## 2.2 分组方式
>
> 针对不同场景，GroupedMatmul可分为m轴分组和k轴分组，又称切M，切K。在正向训练过程对m轴进行分组，在反向计算梯度时就需要对k轴进行分组。
>
> - m轴分组：$k_i$各组相同，$weight_i/y_i$可以在$n_i$上拼接，此时group type = 0;
> m轴分组可用于非量化正向训练场景，量化场景和伪量化场景。

Single/multi tensor shape table:

> | group_type | supported scenario | x shape | weight shape | y shape | optional-dynamic inputs shape if needed | group_list shape if passed | per_token_scale shape if passed |
> | :--------: | :----------------: | :-----: | :----------: | :-----: | :-------------------------------------: | :------------------------: | :-----------------------------: |
> | -1         |        多多多         | [(M1,K1),(M2,K2),...] | [(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | [(N1),(N2),...] | not support | not support |
> | 0          |        单单单         | [(M,K)] | [(G,K,N)] | [(M,N)] | [(G,N)] | (G) | (M) |
> | 0          |        单多单         | [(M,K)] | [(K,N),(K,N),...] | [(M,N)] | [(N),(N),...] | (G) | not support |
> | 0          |        多多单         | [(M1,K1),(M2,K2),...] | [(K1,N),(K2,N),...] | [(M,N)] | [(N),(N),...] | (G) | not support |
>
> 例如在多多多场景，x shape={{4，16}, {12，16}, {16，16}}，weight shape={{16，8}，{16，8}，{16，8}}，y shape={{4，8}，{12，8}，{16，8}}，如果想在单单单场景进行相同的计算，则x shape={32，16}，weight shape= {3，16，8}，y shape={32，8}，groupList={4，12，16}。

Tiling data layout and the reason the kernel cannot instantiate per-group matmul objects:

> ```c++
> BEGIN_TILING_DATA_DEF(GMMTilingData)
>   TILING_DATA_FIELD_DEF_STRUCT(GMMBaseParams, gmmBaseParams);
>   TILING_DATA_FIELD_DEF_STRUCT(GMMArray, gmmArray);
>   TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
> END_TILING_DATA_DEF;
> ```
>
> tilingData主要包含上述结构里的三个部分：
>
> 1. GMMBaseParams: GroupedMatmul的分组数量、Aicore核数、ub tiling参数、matmul分核tiling参数等基础tiling参数，是否有激活函数、量化类型（per token或per tensor）、激活函数类型等功能参数;
> 2. GMMArray: 当输入是多tensor时，通过三个数组记录每组Matmul的shape, kernel通过GlobalTensor::GetValue()的方式获取。当输入为全单tensor时，m/k/n中的一个值在group_list中，另外两个值在所有的group中都相同，此时kernel只需要访问数组中的第一个值。
> 3. TCubeTiling: matmul高阶api对应的tilingData。
>
> kernel中为了避免在栈空间中申请GMMArray中3个数组以及避免拷贝这些数组，采用GET_TILING_DATA_MEMBER接口拷贝除GMMArray之外的结构体。

UB budget, with concrete numbers:

> ## 3.2. UB buffer分配
>
> 在初始化阶段，GroupedMatmul需要确定UB buffer的分配复用情况。非量化、伪量化的计算过程简单，没有UB buffer的复用，而量化场景多、计算复杂，UB buffer需要复用，以提高单次计算的数据量。
> 定义每份buffer分配的字节大小比上处理的数据个数（baseM\*baseN，记为ubCalcSize，此处baseM/baseN为vector计算的参数）为该buffer的份数，以per token量化为例，假设baseM = 24, baseN = 256，则ubCalcSize = baseM * baseN = 6kb。
>
> UB buffer分配如下：
>
> 1. vector计算的输入即matmul的输出的类型为int32，在开启doubleBuffer之后输入需要sizeof(int32) * 2，即8份buffer；
> 2. vector计算的输出为fp16/bf16，其需要sizeof(fp16/bf16) * 2，即4份buffer；
> 3. vector计算的中间计算过程需要申请tmpBuffer，其中broadcast的输出需要sizeof(fp32)块buffer，反量化的输出sizeof(fp32)块buffer，还需申请sharedBuffer进行buffer复用，包含broadcast临时空间、反量化临时空间和Mul输出，大小为8块buffer，共需要申请tmpBuffer的大小为16块buffer。
>
> 总共需要分配的UB buffer为28 * 6kb = 168kb。

The tile scheduling scheme — base blocks, then diagonal walk, with the two inequalities:

> ## 3.3 基本块分核方案
>
> GroupedMatmul实现时需要考虑输入为多个tensor的情况，即每组matmul的shape可能各不相同，而kernel侧不能为每组matmul单独配置对应的matmul高阶api接口实例（tiling结构体和core栈空间大小均不允许）。为了适配不同shape的matmul计算，GroupedMatmul采用基本块方式（横向分核），以baseM、baseN为基本块进行分核计算，此处baseM/baseN为matmul的参数。
>
> ## 3.4 对角线分核方案
>
> 按基本块方案进行分核，容易存在同地址访问的问题，例如当基本块方案中nDim=coreNum时，则同一时间所有的核都在访问左矩阵的相同地址，对性能影响较大。因此当基本块数量超过coreNum时（没超过coreNum时，对角线方案无法解决同地址访问问题），可以采用如下对角线方案，同一时间不同核尽量错开对数据的访问，每个方块代表一个输出的基本块，数字代表基本块遍历顺序（横向分核为原始基本块分核方案）。
>
> 对于适合对角线优化的场景，在基本块方案上输入数据会存在多次访问，当nDim/mDim很大时，“对角线”不能直接任意往下延伸，否则在k值比较大的场景下，对角线上对应的输入数据均为不同的数据，已经加载过一次的数据被新数据从L2 cache中替换掉，后续需要加载时还是从DDR内存中加载，导致性能劣化。因此需要限制对角线范围，以充分利用L2 cache中缓存的数据。
>
> 将对角线遍历按阈值进行分组，为尽量避免同地址访问，两个方向的阈值最好都不小于实际的物理核数，因此有：
>
> $$
> \min(T_m, T_n) \geq numCore
> $$
>
> 为充分利用cache，一个分组块对应的$X$、$Weight$、$Y$的总数据量最好不超过设备的L2 cache大小，因此有:
>
> $$
> sizeof(dtype) \cdot (T_m \cdot singleM \cdot K + K \cdot T_n \cdot singleN + T_m \cdot singleM \cdot T_n \cdot singleN) \leq L2_{size}
> $$
>
> 若两个条件不能同时满足，须根据实际情况做取舍。

## Relevance

The decisive sentence is `kernel侧不能为每组matmul单独配置对应的matmul高阶api接口实例（tiling结构体和core栈空间大小均不允许）` — CANN's grouped GEMM is forced into one shared
`TCubeTiling` for all 256 groups, so the base block `baseM×baseN` is a single global choice.
At batch=1 our active groups have `m_i = 1` while `baseM` is sized for throughput (the doc's
worked example uses `baseM = 24`), so every base block is almost entirely padding. The diagonal
scheme and both inequalities assume base-block count ≫ `coreNum`; with 8 active groups of 1 row
there are far fewer blocks than cores, so neither the diagonal walk nor the L2 constraint is
doing anything for us. A batch=1 megakernel should specialize `baseM = 1..8` and drop the
diagonal machinery.

---
## 2. The closest published prior art: MonoMoE, a weight-major persistent MoE decode megakernel

### MonoMoE: An Efficient Fused Mega-kernel for Quantized MoE Decoding

https://arxiv.org/abs/2609.04244
Yu Gong, Kailash Budhathoki, Taeho Kim, Haipeng Li, Ashish Khetan. Submitted 19 Aug 2026.
HTML: https://arxiv.org/html/2609.04244v1

Abstract, verbatim:

> Mixture-of-Experts (MoE) layers increase model capacity without proportionally increasing arithmetic, but their sparse expert computation is difficult to execute efficiently during autoregressive decode. Existing grouped and batched GEMMs are token-major: they construct expert-local token tiles and obtain parallelism from the token dimension. When few tokens reach each expert, this organization incurs tile padding and preprocessing, launches short-lived grids that underutilize memory bandwidth, and exposes quantization, activation, and reduction as separate stages. We present **MonoMoE**, a weight-major persistent megakernel for block-wise quantized MoE decode. MonoMoE places the complete decode-step token tile on the fine-grained tensor-core $N$ dimension and partitions CTAs over expert-weight tiles, eliminating expert-local token materialization and reducing padded arithmetic. A persistent grid fuses routing, top-$k$ selection, quantization, both expert projections, activation, and reduction in one launch; warp specialization and readiness flags overlap auxiliary work with the dominant expert-weight stream. MonoMoE is integrated with vLLM and supports multiple model shapes through generated kernel specializations and offline schedule tuning. On NVIDIA H200 GPUs, MonoMoE accelerates the complete routed-MoE operator by up to $\mathbf{1.54\times}$ over vLLM Triton Grouped GEMM, is $\mathbf{2.20}$--$\mathbf{3.84\times}$ faster than FlashMoE-FP8 adaptation across various models and batch sizes, and reduces end-to-end time per output token by up to $\mathbf{18.7\%}$ across the evaluated FP8 models, while preserving task accuracy.

Their baseline profiling of a batch-1 MoE layer — the same measurement we made:

> Our profiling puts numbers to this inefficiency. For a single Qwen3.5-35B-A3B-FP8 layer at batch size one on an H200, vLLM spends 23.0 μs in the two expert GEMMs – yet these kernels sustain only 17–22% of peak DRAM bandwidth. Routing, quantization, activation, and reduction add a comparable amount of low-bandwidth work on the same critical path. Across decode token counts from one to eight, our entitlement analysis shows that closing kernel-boundary gaps and recovering memory throughput can cut inter-token latency by 32–36%. The expert-weight traffic itself is unavoidable — but repeatedly interrupting that traffic and serializing auxiliary work around it does not have to be.

The padding arithmetic for the decode regime, with the E=256 / top-8 case worked out:

> The routing distribution affects the two layouts differently. Let R=Bk denote the number of routed token–expert pairs for routing fanout k. For grouped GEMM, flattening mitigates load imbalance across CTAs, but padding is maximized when the pairs are dispersed across as many experts as possible. In the decode regime where R ≤ E, the worst case assigns one pair to each of R experts, yielding M_sched = R · BLOCK_M and a useful-row fraction of only 1/BLOCK_M. For example, B = 8, k = 8, and BLOCK_M = 16 produce 64 useful rows among 1,024 scheduled rows, or 93.75% padding. Batched GEMM suffers the opposite extreme: if all tokens select the same k experts, then M_cap = B while only k of the E expert slices contain data. Its useful-row fraction is therefore k/E — 3.125% for E = 256 and k = 8 — even before any additional tile-alignment padding.

The SwapAB formulation — tokens go on the narrow MMA axis:

> To reduce this underutilization, MonoMoE adopts SwapAB by placing tokens on the small-N side of the MMA. Let X_e of shape B_e × D denote the compact activations selected for expert e, and let X̃_e of shape B × D denote an expert-masked view of the original token tile. For expert weight W_e of shape D × O, the conventional and weight-major formulations are
>
> Y_e = X_e W_e            : (M,N,K)_std    = (B_e, O, D),        (9)
>
> Ỹ_e^⊤ = W_e^⊤ X̃_e^⊤      : (M,N,K)_SwapAB = (O, B, D),
>
> where D is the reduction dimension and O is the expert output width. SwapAB exchanges the logical operand roles and produces a transposed output layout; it does not launch a transpose kernel or move an additional tensor through global memory. For FP8, Hopper WGMMA supports N = 8,…,256 in multiples of eight, and Blackwell's UMMA/tcgen05 path provides analogous small-N shapes. MonoMoE selects the smallest supported N that covers the step's token count.

Persistent grid structure and warp roles:

> MonoMoE persistent-grid mapping. The kernel launches one CTA per SM, so the grid contains N_SM CTAs on a GPU with N_SM streaming multiprocessors. Each CTA contains 12 warps: eight compute warps that issue tensor-core work and four prefetch warps that drive asynchronous data movement. With 32 threads per warp, each CTA has 384 threads.

> Example MonoMoE persistent schedule with three experts and six CTAs. During the up-projection, CTAs form three two-CTA expert groups: CTAs in the same group stream the same expert's weights while computing disjoint output stripes. The groups then advance through experts in round-robin order. The down-projection uses a different grouping, with three CTAs per active expert, but follows the same pattern of expert-local weight streaming and stripe-wise output partitioning.

On-chip buffer aliasing — how they keep one CTA per SM resident:

> Shared-memory reuse. Persistence is only practical if the live state of the fused layer fits within the per-SM shared-memory budget. MonoMoE uses a lifetime-aware shared-memory layout in which buffers with disjoint lifetimes alias the same storage. The routing-window input tile, the up-projection weight tile, and the down-projection weight tile share one region because their lifetimes do not overlap; similarly, epilogue scratch buffers are collapsed into a single reusable region. This reuse keeps the persistent footprint small enough to maintain one CTA per SM at the target tile sizes.

The negative result on fusing collectives — directly relevant to our TP16:

> Folding these collectives into the kernel does not improve single-node, small-batch decode because NVLink transfers the small messages in only a few microseconds, leaving cross-rank synchronization as the dominant exposed cost. A waiting persistent kernel also keeps one CTA resident on every SM, causing the grid to spin while communication traffic competes with the bandwidth-bound expert-weight stream. TP offers no useful overlap window because its all-reduce is terminal, and MonoMoE's atomicAdd-based accumulation does not produce an output element until all contributing expert groups finish. Exposing partial results earlier would require per-group buffers, roughly doubling output traffic and adding a reduction pass, while the receive would still wait for the slowest rank. Separate TP collectives are therefore preferable in MonoMoE's target regime; fusion becomes more attractive at higher TP degrees or during prefill, where larger transfers pro[vide]...

End-to-end results, including GLM-5.2:

> In end-to-end serving, MonoMoE reduces mean time per output token by 9.9–18.7% on Qwen3.5-35B, 9.1–11.5% on Qwen3.5-122B, and 2.9–16.7% on GLM-5.2. DeepSeek-V3.1's eight-way tensor-parallel path dilutes the local-kernel gain: TPOT ranges from a 1.2% regression to a 3.7% reduction, while aggregate output throughput improves by up to 31.8%.

> The decline is most pronounced for GLM-5.2 because its TP=8 serving path includes communication and non-MoE work that the local kernel does not reduce.

## Relevance

This is our problem, on our model, one generation of hardware away. Their baseline numbers
(two expert GEMMs at 17–22% of peak DRAM bandwidth with batch=1) are the same shape as ours
(GroupedMatmulSwigluQuantV2 at 2.272/6.793 = 33%, GroupedMatmul at 1.212/5.596 = 22%). Four
design decisions transfer directly: (1) partition cores over expert-weight output stripes, not
over tokens — at batch=1 the weight matrix is the only thing with enough parallelism; (2) put
the token count on the narrow MMA axis (SwapAB), which on Ascend means `baseM` = token count
and `baseN` = output width rather than the reverse; (3) alias on-chip buffers by lifetime so
the whole fused layer stays resident; (4) do **not** try to fuse the TP16 all-reduce into the
megakernel — their measurement says the terminal all-reduce has no overlap window and a
spinning persistent grid actively competes with the weight stream. Their 93.75% padding figure
for `B=8, k=8, BLOCK_M=16` is the quantitative indictment of the grouped-GEMM layout at our
operating point.

---

## 3. Masked grouped GEMM: the device-side group-count contract

### DeepGEMM README — contiguous vs masked layouts, and Mega MoE

https://github.com/deepseek-ai/DeepGEMM
Raw: https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/README.md — DeepSeek.

> #### Grouped GEMMs (contiguous layout)
>
> Unlike traditional grouped GEMMs in CUTLASS, DeepGEMM groups only the M-axis, while N and K must remain fixed. This design is tailored for scenarios where experts in an MoE model share the same shape. For training forward passes or inference prefilling, where each expert may process a varying number of tokens, we concatenate these tokens into a single tensor, referred to as the "contiguous" layout. Note that each expert segment must be aligned to the GEMM M block size (`get_mk_alignment_for_contiguous_layout()`).  For more information, please refer to the `m_grouped_fp8_gemm_{nt, nn}_contiguous` function documentation.
>
> #### Grouped GEMMs (masked layout)
>
> During the inference decoding phase, when CUDA graph is enabled and the CPU is unaware of the number of tokens each expert receives, we support masked grouped GEMMs. By providing a mask tensor, the kernel computes only the valid portions.
>
> Use `m_grouped_fp8_gemm_nt_masked` for this purpose and consult the relevant documentation. An example usage is to use the output of low-latency kernels from [DeepEP](https://github.com/deepseek-ai/DeepEP) as input.
>
> #### Mega MoE
>
> Mega MoE fuses and overlaps EP dispatch, linear 1 and linear 2 (FP8xFP4 or FP8xFP8), SwiGLU, and EP combine into a single mega-kernel, overlapping NVLink communication and tensor core computation. It requires multi-process launch with symmetric memory.

### `m_grouped_fp8_fp4_gemm_nt_masked` — exact signature and the `expected_m` hint

https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/csrc/apis/gemm.hpp

```cpp
static void m_grouped_fp8_fp4_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& masked_m,
                                             const int& expected_m,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             std::optional<std::tuple<int, int>> recipe_a,
                                             std::optional<std::tuple<int, int>> recipe_b,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast,
                                             const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());
    ...
    const auto num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);
```

```cpp
    m.def("m_grouped_fp8_fp4_gemm_nt_masked", &m_grouped_fp8_fp4_gemm_nt_masked,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"), py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false,
          py::arg("epilogue") = nullptr);
    ...
    m.attr("m_grouped_fp8_gemm_nt_masked") = m.attr("m_grouped_fp8_fp4_gemm_nt_masked");
```

Test-side usage, including the efficiency caveat:

https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/tests/test_fp8_fp4.py

```python
def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    ...
        expected_m = int(expected_m_per_group * 1.2)

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m)
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        ...
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(a_, b_, d_, grouped_layout, expected_m, **common)
```

## Relevance

The masked contract is the correct shape of the problem at batch=1: static `[G, M, K]`, one
device-resident `int32 masked_m[G]`, and a scalar `expected_m` used only for tile scheduling.
Compare this to CANN's `groupList` in `GroupedMatmulSwigluQuantV2`, which the kernel walks with
scalar GM reads (see Section 1 above). The stated reason for the masked design — `the CPU is
unaware of the number of tokens each expert receives` under graph capture — is exactly our
situation, since we must stay inside an aclgraph.

---

### DeepGEMM Mega MoE host API — one kernel for dispatch + L1 + SwiGLU + L2 + combine, with the shared expert folded in

https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/csrc/apis/mega_moe.hpp

```cpp
static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
)
```

The intermediate-activation ring buffer, sized from a closed-form live-block bound instead of
a full `[E, capacity, H]` staging buffer:

```cpp
    // Ring capacity: worst-case live pool blocks over all candidate BLOCK_M; mirrors the kernel assert.
    // TODO: we temporarily assume the SM count is consistent with the runtime value
    const auto num_sms = get_num_sms_for_mega_moe();
    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto num_active_topk = std::min(num_topk, num_experts_per_rank);
    const auto num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk;
    ...
    int num_ring_tokens = 0;
    for (const auto& block_m: layout::kCandidateBlockM) {
        const auto num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
        const auto num_live_pool_blocks = sched::get_num_max_live_pool_blocks(
            num_pool_blocks, num_sms, hidden, intermediate_hidden);
        num_ring_tokens = std::max(num_ring_tokens, num_live_pool_blocks * block_m);
    }
    num_ring_tokens = math::align(num_ring_tokens, layout::kLCMCandidateBlockM);
```

The shared expert is expressed as an integer multiple of the routed expert's intermediate width:

```cpp
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = check_weights_layout_3d(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = check_weights_layout_3d(l2_weights);
    ...
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    ...
        shared_intermediate_hidden = shared_l2_k;
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;
```

## Relevance

`num_shared_experts = shared_intermediate_hidden / intermediate_hidden` is the trick for
putting our W8 shared expert in the same megakernel as the W4A8 routed experts: treat it as
extra expert slices of the same shape rather than as a separate op (we currently pay a separate
`QuantBatchMatmulV3` at 3.810 ms vs 0.780 ms of bytes for it). The ring-buffer sizing shows the
intermediate `[8, 128]` activations never need a full-capacity staging buffer — at batch=1 our
live set is 8 rows of 128 per rank, trivially on-chip.

---

### DeepGEMM MegaMoE in-kernel scheduler — task descriptors, bitmask inter-stage dependency, warp roles

https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh

```cpp
// Computation phase for the current block
enum class BlockPhase : uint32_t {
    None = 0,
    Linear1 = 1,
    Linear2 = 2,
    SharedLinear1 = 3,
    SharedLinear2 = 4
};

template <bool kHasSharedExperts>
struct alignas(16) TaskInfo {
    BlockPhase block_phase;
    uint32_t local_expert_idx;
    uint32_t m_block_idx;
    uint32_t n_cluster_idx;
    uint32_t pool_block_idx;
    uint32_t valid_m;
    uint32_t shape_n;
    uint32_t shape_k;
```

The XOR bitmask that lets Linear2 start per-fed-tile instead of after a layer barrier:

```cpp
// Each finished L1 N block toggles its bit in `l2_full_mask`, so L2 K blocks can start as soon as they are fed
// NOTES: the mask parity per ring generation relies on the ring capacity, see `get_num_max_live_pool_blocks`
template <uint32_t L1_SHAPE_N, uint32_t BLOCK_N, uint32_t BLOCK_K>
struct L2KBlockDependency {
    static constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumL1BlockNsPerL2KBlock = BLOCK_K / (BLOCK_N / 2);
    static constexpr uint64_t kFullMask = kNumL1BlockNs == 64 ? ~0ull : (1ull << kNumL1BlockNs) - 1;
    ...
    CUTLASS_DEVICE static void arrive(const uint64_t* mask_ptr, const uint32_t& n_block_idx) {
        ptx::red_xor_rel(mask_ptr, 1ull << n_block_idx);
    }

    // Re-read the mask only when the K block is not fed yet
    CUTLASS_DEVICE void wait(const uint32_t& k_block_idx) {
        const auto k_block_mask = ((1ull << kNumL1BlockNsPerL2KBlock) - 1) << (k_block_idx * kNumL1BlockNsPerL2KBlock);
        while (pending_mask & k_block_mask)
            pending_mask = ptx::ld_acq_gpu(mask_ptr) ^ expected_mask;
    }
};
```

The locality/work-stealing rule:

```cpp
    // Locality domain affinity. When a queue is drained, we try work stealing to ensure load balancing.
    // But when there are few tasks, we disable work stealing to avoid cross-die latency.
    static constexpr uint32_t kNumMinStealWaves = 2;
```

Expert token counts fetched lane-parallel, spinning on a completion sentinel:

```cpp
    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
```

Warp-role split and the swap-AB MMA config:
https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh

```cpp
    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    ...
    // Prefetch so the TMEM stage is released earlier, keeping the epilogue off the critical path
    constexpr uint32_t kNumEpiPrefetchStages = BLOCK_M == 240 ? 1 : 0;
```

```cpp
    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        ...
        // Dispatch warps
    } else if (warp_idx == kNumDispatchWarps) {
        // GEMM TMA load warp for tokens with SFA
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // GEMM TMA load warp for weights with SF
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // GEMM MMA issue warp (only the leader CTA will run)
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Do mainloop by the leader CTA
        if (is_leader_cta)
            scheduler.mainloop(num_tokens, sm_locality_domains);
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
```

## Relevance

The mechanism set we need, in production form: a 16-byte aligned task descriptor, a 64-bit XOR
bitmask so the down GEMM's K-blocks start as each up-GEMM N-block lands (instead of a full
barrier between the two GEMMs), `UMMA_N = BLOCK_M` as the swap-AB encoding, and a dedicated
scheduler role on one core. On Ascend the bitmask is replaceable with a workspace counter plus
`CrossCoreWaitFlag`. The comment `when there are few tasks, we disable work stealing to avoid
cross-die latency` is a direct warning for a TP16 A3 superpod: at batch=1 there are few tasks,
so static assignment beats any dynamic balancing.

---

## 4. Token-major grouped GEMM at batch=1: the padding tax, measured in framework code

### vLLM `moe_align_block_size` — the padding formula and the worked example

https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/fused_moe/moe_align_block_size.py

```python
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    return_scatter_idx: bool = False,
) -> ...:
    """Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.
    ...
    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    ...
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
```

## Relevance

`max_num_tokens_padded = numel + num_experts * (block_size - 1)` is the token-major layout's
cost in closed form. At our point `numel() = 8` and local `num_experts = 16` at TP16 (256
global), so `numel() < num_experts` and even vLLM falls back to `numel * block_size` — 8
useful rows occupying 8 whole M-tiles. This is the same conclusion MonoMoE reaches analytically.

---

### vLLM Triton `fused_moe_kernel` — the `naive_block_assignment` batch-1 bypass

https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/fused_moe/fused_moe.py

```python
@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, b_bias_ptr, a_scale_ptr, b_scale_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn, stride_asm, stride_ask,
    stride_bse, stride_bsk, stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    group_n: tl.constexpr, group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr, use_int8_w8a8: tl.constexpr, use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr, HAS_BIAS: tl.constexpr, SWAP_AB: tl.constexpr,
    USE_TD: tl.constexpr = False,
):
    """...
    - naive_block_assignment: A boolean flag indicating whether to use naive
        token wise block assignment. If True, each block corresponds to a
        single token.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    ...
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
```

The small-batch invalid-block skip:

```python
        EM = sorted_token_ids.size(0)
        if A.size(0) < config["BLOCK_SIZE_M"]:
            # optimize for small batch_size.
            # We assume that top_ids of each token is unique,
            # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
            # and we can skip some invalid blocks.
            EM = min(
                sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"]
            )
```

## Relevance

vLLM's own escape hatch: at small batch it abandons `sorted_token_ids` and maps one program to
one token. This is permission to drop the permutation/`moe_init_routing` stage entirely at
batch=1 and index the 8 routed expert weights directly — on Ascend that deletes a whole op from
the chain rather than making it faster.

---

### vLLM decode-tile heuristics for block-wise FP8 MoE — the small-M reasoning in comments

https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/fused_moe/fused_moe.py
and https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/fused_moe/utils.py

```python
    if dtype == "fp8_w8a8" and block_shape is not None:
        # Block-wise quant. Use a small M tile for decode-like batches where
        # tokens are spread thin across experts. Larger batches benefit from
        # GROUP_SIZE_M > 1 because the per-block scales add memory traffic
        # that benefits from L2 tile reuse.
        #
        # BLOCK_SIZE_N need not equal block_shape[0]: the kernel indexes block
        # scales per N element (offs_bn // group_n), so any N tile dividing the
        # quant block is valid. At decode a 128-wide N tile leaves the gate-up
        # GEMM SM-bound; a 64-wide tile exposes ~2x the thread blocks, and the
        # swap-AB kernel keeps it efficient on Hopper down to the smallest
        # batches, so prefer N=64 through low batch sizes. CUDA only (validated
        # on NVIDIA); ROCm keeps its prior tile/pipeline sizes.
        if current_platform.is_rocm():
            block_n = block_shape[0]
            num_stages = num_stages_rocm
        elif M <= 8 and block_shape[0] % 64 == 0:
            block_n = 64
            # The smallest batches are memory-latency bound, so a deeper
            # pipeline hides the weight loads; by M=8 it turns occupancy/SMEM
            # bound and the extra stages hurt.
            num_stages = 4 if M <= 4 else 3
        ...
        config = {
            "BLOCK_SIZE_M": 16 if M <= 64 else 64,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 1 if M <= 16 else 32,
            "SPLIT_K": 1,
            "num_warps": 4,
            "num_stages": num_stages,
        }
```

```python
@functools.lru_cache
def enable_swap_ab(BLOCK_SIZE_M: int, BLOCK_SIZE_N: int) -> bool:
    return (
        current_platform.is_device_capability(90)
        and BLOCK_SIZE_M < 64
        and BLOCK_SIZE_N >= 64
    )
```

## Relevance

Two transferable levers stated as comments by people who tuned them: at the smallest batches
increase *pipeline depth* to hide weight loads rather than tile size, and narrow the N tile to
expose more blocks. The swap-AB gate `BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64` is the
numeric threshold at which putting tokens on the narrow axis wins.

---

### TensorRT-LLM "min latency mode" — the batch-1 MoE data layout with no permutation arrays

https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/kernels/moe/cutlass/include/moe_kernels.h

```cpp
struct MoeMinLatencyParams
{
    // All these are allocated on device memory
    // Number of active experts on current node; smaller than or equal to num_experts_per_node
    int* num_active_experts_per_node;
    // The score of each token for each activated expert. 0 if the expert is not chosen by the token.
    // Only the first num_active_experts_per_ rows are valid
    float* experts_to_token_score;
    // The global expert id for each activated expert
    // Only the first num_active_experts_per_ values are valid
    int* active_expert_global_ids;
```

https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/kernels/moe/cutlass/moe_kernels.cu

```cpp
/**
 * Takes the input maps and prepares the expanded maps for min latency
 ...
template <int BLOCK_SIZE>
__global__ void buildMinLatencyActiveExpertMapsKernel(...)
{
    cudaGridDependencySynchronize();
    // Use one block to process the min latency case
```

```cpp
    size_t num_moe_inputs = min_latency_mode ? num_experts_per_node * num_rows : experts_per_token * num_rows;
    ...
    size_t const permuted_row_to_unpermuted_row_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
    size_t const permuted_token_selected_experts_size = min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
```

```cpp
    if (min_latency_mode)
    {
        TLLM_CHECK_WITH_INFO(using_tma_ws_gemm1, "Only TMA warp specialized GEMM is supported in min latency mode.");
        // TODO: as for bias, need to get the correct expert id according to the active expert global ids
        TLLM_CHECK_WITH_INFO(fc1_expert_biases == nullptr, "Min latency mode does not support bias.");
        TLLM_CHECK(use_fp4);
    }
    ...
    if (min_latency_mode)
        return;   // (in gemm2: no separate finalizeMoeRouting pass)
```

## Relevance

NVIDIA's production answer for minimum-latency MoE abandons the permutation arrays entirely
(`permuted_row_to_unpermuted_row_size = 0`), replaces them with a dense
`[num_experts_per_node, num_rows]` score matrix, builds the routing metadata in a single block,
and deletes the separate finalize pass. That is a complete alternative layout for the batch=1
regime, and it is the layout our megakernel should adopt instead of the prefill layout CANN's
`moe_init_routing` + `GroupedMatmul` + `moe_finalize_routing` chain assumes.

---

### TensorRT-LLM hand-written batch-1 MoE kernel — one block per output row, no MMA at all

https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/kernels/llama4MinLatencyKernels/llama4MinLatencyMoEOp.cu

```cpp
#define NUM_EXPERTS 128
#define HIDDEN_SIZE 5120
#define INTER_SIZE 1024
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define TILE_ROW 4
#define VEC_SIZE 8

#define ENABLE_ACQBULK 1
#define ENABLE_PREFETCH 1
#define ENABLE_PREEXIT 1
```

```cpp
// This is the hand-optimized kernel.
// The computation is:
//   C = silu(AxB_gated * in_scale * sigmoid(logit)) * (AxB_linear * in_scale * sigmoid(logit)) * out_scale_inv
// The out_scale_inv cannot be fused with in_scale because silu() is non-linear.
// Also, Llama-4 applies score scaling, which is sigmoid(logit), on tensor A.
__global__ void llama4_moe_fc13_swiglu_fp8_kernel(int num_tokens,
    __nv_fp8_e4m3 const* __restrict__ A,      // Input tensor [num_tokens][HIDDEN_SIZE]
    __nv_fp8_e4m3 const* __restrict__ B,      // Input tensor [num_experts][INTER_SIZE*2][HIDDEN_SIZE]
    __nv_bfloat16 const* __restrict__ logits, // Input tensor logits [num_tokens][num_experts]
    __nv_fp8_e4m3* __restrict__ C,            // Output tensor [num_tokens][INTER_SIZE]
    int* __restrict__ exp_idx,                // Output tensor [num_tokens]
    ...
    int const token_idx = blockIdx.x / INTER_SIZE;
    int const row = blockIdx.x % INTER_SIZE; // Matrix row / Output element index
    ...
    // Preload the scaling factors before ACQBULK.
    __shared__ float in_scales_shared[NUM_EXPERTS];
    in_scales_shared[tid] = in_scales[tid];

    // Logits depends on the previous kernel, so we cannot prefetch anything.
#if ENABLE_ACQBULK
    cudaGridDependencySynchronize();
#endif

    // Perform top1 within the current thread, which processes 4 experts.
    ...
    // Perform top1 across threads using Warp reduction.
    // We pack logit and expert index into an int so that we can use integer max op for reduction.
    int best_result
        = ((int) (__bfloat16_as_short(best_logit) ^ (best_logit < __nv_bfloat16(0.f) ? 0x7fff : 0)) << 16) | best_exp;
    ...
#if ENABLE_PREEXIT
    cudaTriggerProgrammaticLaunchCompletion();
#endif
```

```cpp
    int const grid_size = num_tokens * INTER_SIZE;
    ...
    launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, (void*) llama4_moe_fc13_swiglu_fp8_kernel, args, 8);
```

The second GEMM maps `TILE_ROW = 4` output rows per block:
`int const grid_size = num_tokens * HIDDEN_SIZE / TILE_ROW;`

## Relevance

The fully-specialized extreme, and it beats grouped GEMM at batch=1: no matrix unit at all,
routing + gate + up + SwiGLU + requantize inside one block whose grid is the intermediate
width, all scale factors preloaded *above* the dependency-wait point so the wait costs nothing.
On Ascend the analogue is to compute per-output-stripe on AI_VECTOR_CORE (0.70 us floor, vs
AI_CORE 2.30 us and MIX_AIC 13.58 us) and hoist every scalar/scale load above the
`CrossCoreWaitFlag`. Note also the hard-coded shapes — this kernel is specialized to one model,
which is exactly the licence our brief gives us.

---

### FlashInfer `cutlass_fused_moe` — `min_latency_mode` and finalize fused into the GEMM2 epilogue

https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/flashinfer/fused_moe/core.py

```python
@flashinfer_api(trace=cutlass_fused_moe_trace)
def cutlass_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    fc1_expert_biases: Optional[torch.Tensor] = None,
    fc2_expert_biases: Optional[torch.Tensor] = None,
    input_sf: Optional[torch.Tensor] = None,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    tp_size: int = 1, tp_rank: int = 0,
    ep_size: int = 1, ep_rank: int = 0,
    cluster_size: int = 1, cluster_rank: int = 0,
    output: Optional[torch.Tensor] = None,
    enable_alltoall: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4_group_scaling: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    ...
    enable_pdl: Optional[bool] = None,
    activation_type: ActivationType = ActivationType.Swiglu,
    swizzled_input_sf: bool = True,
    use_fused_finalize: bool = True,
    ...
) -> torch.Tensor:
    """Compute a Mixture of Experts (MoE) layer using CUTLASS backend.

    This function implements a fused MoE layer that combines expert selection, expert computation,
    and output combination into a single operation.
```

```
    min_latency_mode : bool = False
        Whether to use minimum latency mode. Defaults to False.
```

```
    use_fused_finalize : bool = True
        Whether to fuse the top-k expert reduction ("finalize") into the GEMM2 epilogue.
        Defaults to True for best performance. The fused epilogue reduces expert outputs via
        non-associative atomics, so results are not deterministic run-to-run. Set to
        False to use the non-fused, deterministic finalize path.
```

```
    enable_pdl : Optional[bool]
        Whether to launch the kernel with Programmatic Dependent Launch (PDL).
        ``None`` (default) lets the runtime pick a safe value based on the device
        and surrounding stream operations; pass ``True`` to force PDL when every
        adjacent kernel on the stream also supports it, or ``False`` to disable.
```

## Relevance

The top-k combine is folded into the down GEMM's epilogue by default, with the nondeterminism
cost stated explicitly. CANN has the equivalent op already —
`GroupedMatmulFinalizeRouting` ("GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作", see below) — so this fusion boundary is available to us
without writing it. Note our `aclnnGroupedMatmulSwigluQuantV2` is documented as
`默认为确定性实现`, so adopting an atomics-based combine is a deliberate determinism trade.

---

### CANN `GroupedMatmulFinalizeRouting` — the down-GEMM + combine fusion that already exists on A3

https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul_finalize_routing/README.md

> # GroupedMatmulFinalizeRouting
>
> ## 产品支持情况
>
> |产品      | 是否支持 |
> |:----------------------------|:-----------:|
> |<term>Ascend 950PR&950DT系列产品</term>|      √     |
> |<term>Atlas A3系列产品</term>|      √     |
> |<term>Atlas A2系列产品</term>|      √     |
>
> ## 功能说明
>
> - 算子功能：GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作

> | x1 | 输入 | 输入x(左矩阵)。 | INT8、FLOAT8_E5M2<sup>1</sup>、FLOAT8_E4M3FN<sup>1</sup>、HIFLOAT8<sup>1</sup>、FLOAT4_E2M1<sup>1</sup> | ND |
> | x2 | 输入 | 输入weight(右矩阵) | INT4<sup>2</sup>、INT8、INT32<sup>2</sup>、FLOAT8_E5M2<sup>1</sup>、FLOAT8_E4M3FN<sup>1</sup>、HIFLOAT8<sup>1</sup>、FLOAT4_E2M1<sup>1</sup> | ND、NZ |
> | scaleOptional | 输入 | 量化参数中的缩放因子，perchannel量化参数 | INT64<sup>2</sup>、BFLOAT16、FLOAT32、FLOAT8_E8M0<sup>1</sup> | ND |

The kernel directory shows a dedicated A8W4-MSD implementation plus an L1-optimized variant:

```
gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_antiquant_a8w4_msd.h
gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_antiquant_a8w4_msd_l1_opt.h
gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_antiquant_a8w4_msd_pre.h
```

## Relevance

INT4 weight (`x2` accepts `INT4`, superscript 2 = A2/A3 only) with an A8W4-MSD kernel and NZ
weight support — i.e. CANN already ships the down-GEMM + combine fusion for exactly our
quantization on exactly our chip. That op is the natural replacement for our separate
`GroupedMatmul` (5.596 ms vs 1.212 ms) plus whatever combine follows it, and the
`_l1_opt` variant name says someone has already worked the L1 residency of the weight stream.

---
## 5. Huawei's own account of why these ops exist, and what MSD costs

### 昇腾官方技术文章：GroupedMatmul W4A8 量化与 MSD 方案

https://www.hiascend.com/developer/techArticles/20250717-2
昇腾CANN，2025/07/17，《【昇腾热点算子大解密】CANN极致优化GroupedMatmul量化方案，MoE推理Decode性能提升超30%》

> 分组矩阵乘法（GroupedMatmul）是一种将多个独立的小规模矩阵乘法（GEMM）任务分组并行执行的优化技术，通过合并计算拓扑相似的算子为单一内核调用，有效降低核启动开销并减少内存访问碎片化。
>
> MoE模型在Decode阶段每个专家处理的激活数量偏少，导致传统的GEMM操作因单个专家计算规模小而难以充分发挥硬件算力。GroupedMatmul天然适配MoE模型的稀疏专家计算，通过将分散的专家任务合并为分组批量进行计算，提升任务规模以实现接近峰值算力的效率；同时，连续存储专家权重可以进一步优化缓存命中率，显著提升计算性能。

> 然而，仅依赖W4A8方案并不能充分释放低比特运算的潜力。推理流程中不同阶段对计算和访存的依赖有所区别，Prefill阶段的activation更长，计算更密集；而Decode阶段activation更短，访存压力更大，两者对硬件资源的需求和优化重点不同。面对Decode阶段的访存瓶颈，传统W4A8方案虽然能降低显存占用，但在实际计算时仍需将INT4权重还原为INT8后，再与INT8激活做矩阵乘法，有冗余的反量化计算开销。
>
> 因此，我们将W4A8量化方案与昇腾硬件微架构特性结合优化，创新性地提出MSD（Mixed-precision Split-activation Decomposition，混合精度激活拆分分解）方案，将INT8激活拆分为高、低两路INT4，并直接与INT4权重进行计算，用"高位×16+低位"合成，在保证结果等价于原始INT8×INT4计算结果的同时，大幅降低KV Cache和参数存储需求，提升单卡batch size和上下文的支持长度。

So MSD is named **Mixed-precision Split-activation Decomposition** — not "multi-scale".
On the three scale levels and whether per-group is worth it:

> W4A8量化方案采用三层量化scale设计，实现了精细化的量化控制，其数学公式表达如下： 其中量化scale分别为：
> (1) Per-channel scale：针对weight的通道级量化参数，保证不同通道间的精度平衡
> (2) Per-group scale：针对weight的分组级量化参数，提供更细粒度的量化控制
> (3) Per-token scale：针对activation的token级量化参数，动态适应不同输入的分布特性
> 这种多层级的量化策略能够更好地适应不同数据分布的特点，有效降低量化误差。实践中发现，per-group量化对模型精度的提升较为轻微，所以实践中一般可以不采用per-group量化。

On the offline bias term and the scale fusion:

> 当INT8被拆分为两个INT4时，由于INT4的符号位决定了其范围在[-8, 7]，而低4位的最高有效位（即符号扩展）会带来一个偏移量，具体就是8*weight。为了保证拆分的精度，需要将拆分操作变型为如下过程： 这个多余项在所有INT8拆分为INT4的操作中是固定的，等价于每个权重通道都引入了一个偏置项。在GroupedMatmul的具体实现中，通常将这部分多余项放入bias参数中，通过离线计算提前补偿，从而保证推理精度和性能。

> 如上图所示，x与weight被拆分后完成INT4矩阵乘。在实际推理时，为了提升效率，per-channel scale和per-group scale会离线融合成一个总scale（图中的anti-scale），这样可以在GEMM输出阶段一次性完成浮点还原，减少多次逐步缩放带来的性能损耗。
> 融合后的scale一般为total_scale = perchannel_scale * pergroup_scale。这样，矩阵乘法主环节只需用整数计算，最后输出时统一乘以融合后的scale即可。

> Per-group scale是针对权重的分组量化，每组权重共用一个scale。在GEMM实现中，通常沿着k轴（即输入通道）切分，每g个通道为一组（g为group size），每组使用独立的scale。
> 在做矩阵乘法时，需要将输入矩阵X（已分解为INT4）与权重矩阵W（INT4，已分组）分块处理。对每个group，通过total scale将INT4权重直接解码为FP16。具体而言，每进行一组累加，就应用对应的per-group scale。

The post-processing decomposition and the headline numbers:

> W4A8量化后处理主要包括三个步骤：高低位结果合成、bias补偿、反量化（per-token scale）。
>
> (1) 高低位结果合成：在W4A8方案中，INT8激活被拆分为高4位和低4位。为还原INT8原始乘法的效果，需要将高4位的结果左移4位（即乘以16），再与低4位结果相加。这样可以保证拆分计算后，数值与直接INT8×INT4乘法结果一致。
> (2) BIAS余量补偿：拆分INT8为两个INT4时，由于符号扩展和量化区间的差异，会引入一个额外的常量项（8×W×scale）。这个多余项在量化离线阶段提前计算，并作为bias参数加到最终输出上，补偿拆分带来的偏移，保证数值精度。在实际后处理时，直接将该bias向量加到矩阵乘法的结果上即可，无需额外计算，且不会影响推理效率。
> (3) Per-token反量化：W4A8方案通常采用per-token scale对激活进行量化，推理后需反量化还原浮点结果。

> 至此，W4A8量化与MSD方案的所有主流程均在整数域内完成，充分利用昇腾硬件对低比特整数计算的支持，减少主循环的访存和运算瓶颈，为LLM的高效推理提供了关键支撑，极大提升大模型推理的响应速度和服务能力。基于W4A8方案的DeepSeek R1单机部署实践表明，其最大BatchSize达到了W8A8方案的3.1倍；同时，W4A8+MSD方案在Decode阶段的性能更是达到了W8A8的1.3~1.7倍。

## Relevance

Huawei states the purpose of GroupedMatmul as `有效降低核启动开销` — amortizing kernel-launch
overhead across the small per-expert GEMMs — which is the same lever we are pulling one level
up. Two actionable items: `per-group量化对模型精度的提升较为轻微，所以实践中一般可以不采用per-group量化`
means we can likely run `dequantMode=0` (per-channel) and delete the per-group K-axis scale
application from the inner loop; and `per-channel scale和per-group scale会离线融合成一个总scale`
plus the `8×W×scale` bias both being offline means the only in-kernel scale work is the
per-token multiply and the final `max|S|/127`.

---

### 昇腾官方技术文章：通算融合 Dispatch/Combine — 下发时延随 EP 规模线性增长

https://www.hiascend.com/developer/techArticles/20250726-1
昇腾CANN，《【昇腾热点算子大解密】CANN创新通算融合算子破局MoE通信瓶颈，实现推理吞吐50%提升》

> 针对MoE架构，我们的技术团队深入分析后发现，高额的通信开销主要源于两大技术挑战：
> 动态专家选择机制下，每个Token分发的目标专家呈现离散分布特征，导致两个关键问题：
> 数据分发不均匀：不同专家接收的Token长度存在差异，需依赖低效的AllToAllV通信；
> 元数据同步开销：获取收发信息需调用前置AllGather算子收集路由表，并在Host侧完成同步，引入额外通信开销和Stream同步延迟；
> 算子下发延迟：传统Host驱动通信需要构造子图并调度，其下发时延随EP规模线性增长；

> 由上可知，Dispatch/Combine操作本质上是计算与通信的结合。因此，我们创新型地实现了MoeDistributeDispatch和MoeDistributeCombine这两个通算融合算子，将路由计算等Host侧逻辑下沉至Device侧，消除了Host与Device间的同步开销。同时，将Combine操作中所需的参数计算、Token的加权求和等计算操作与AllToAllV通信流水并行，也得以实现计算与通信的耗时掩盖。
>
> 与此同时，AIV将数据在共享内存中的地址、数据长度信息通过GM直接传递给同处Device侧上的AICPU，由AICPU直接驱动RDMA通信，彻底摒弃了传统需要Host侧构造子图和调度RDMA任务的繁琐流程，不仅解决了Host侧处理耗时长的问题，更消除了传统调度方式带来的额外时延。

> 以BS = 16，H = 7168，topK = 8，数据类型FP16场景为例：
> 全互联方案下，总通信量近似为BS * H * topK * sizeof(FP16) * 63/64 = 1.75(MB)（63/64是因为发往本卡的MoE专家无需通过RDMA通信），总通信时间约86.1us。
> 分层方案下，每个Token平均选择4个node，机间通信量BS * H * 4 * sizeof(FP16) * 7/8 = 0.766(MB)，机间通信时间约为38.3us，机内通信量为BS * H * 8 * sizeof(FP16) * 7/8=1.53(MB)，机内通信时间约8us，总通信时间46.3us。

> 虽然实现了计算与通信的初步融合，但受限于AICPU驱动RDMA时WQE串行下发的固有机制，导致计算与通信、机间与机内通信无法实现深度流水并行，且下发时延随下发次数线性增长的问题迫使数据必须按目标rank汇聚。
>
> 为了提高流水并行度，技术团队在AIV+AICPU分层方案的基础上，创新性地实现了AIV直驱RDMA的通路。该方案不再依赖AICPU，实现更低时延，且多个AIV核可以并行下发WQE，WQE下发时延不再是算子耗时的关键，彻底解除了Token数据需按照目标rank汇聚的限制，可以直接以Token为粒度发送；… 最终使得算子平均性能较原有AIV+AICPU的分层方案提升10%+

## Relevance

The communication side of the same argument: `算子下发延迟：…其下发时延随EP规模线性增长` and
`下发时延随下发次数线性增长的问题迫使数据必须按目标rank汇聚`. The AIV-direct-RDMA path removes
the per-rank aggregation constraint, which matters if our megakernel wants to emit tokens at
token granularity rather than batching by destination rank.

---

### SuperKernel 技术综述 — what a megakernel actually recovers on Ascend, decomposed

https://cann.csdn.net/69afcc5554b52172bc603633.html
《大模型推理加速利器SuperKernel技术综述》，2025-11-24

> 该方法将整个网络模型重新编译为一个大算子，从而减少硬件调度开销并优化了Cache和同地址访问，在现有网络优化基础上提升整网性能10-20个百分点。

> 以DeepSeekV3网络中的一层为例，根据Profiling数据分析发现，网络上肉眼可见的Task等待调度开销较为明显。在BatchSize大小为96、Sequence长度3K的场景下，平均每层的Task平均等待的开销约为18us左右。按照61层网络计算，总的调度等待开销在1.1ms左右。这些开销基本是绝对开销，在大的Batch场景下还会进一步扩大。

> 通过详细评估和调试，SuperKernel技术优化后的基本收益模型可以分为以下三部分：首先是两个Task之间的调度开销降低；其次是每个Task结束后会执行Cache Flush操作，将所有修改的Cache内容刷新出去，其中包含一些不必要的栈空间；最后是算子调度后的核启动时间，包括同地址访问的排队延迟、Scalar执行初始化等环节。

> 当前已经支持将DeepSeekV3模型61层网络中的58层融合为1个SuperKernel算子，整网收益平均10-20%左右，远超前面的任务调度分析的开销部分。

> 1）ICache Preload优化
> 在SuperKernel执行过程中，由于运行时系统通常仅预取其入口点的指令，子Kernel的代码段往往无法被硬件预取机制有效捕获，导致指令缓存（ICache）未命中率增高。… 该技术在当前子Kernel执行完毕前，以2KB对齐的方式，提前将下一个即将执行的子Kernel代码段预取到ICache中。
>
> 2）同地址访存优化和亲和部署
> 在多核处理器系统中，当多个计算核心同时执行同一个内核函数时，它们会访问内存中完全相同的指令地址（Text段）。这种对同一地址的并发访问会在共享的L2缓存层面形成一个串行化的访问队列，导致多个核心的请求发生争用，即使数据已缓存在L2中，也会因队列拥堵产生访问延迟，显著削弱多核并行带来的性能提升。… 可将子内核的代码段复制为多份副本，使不同核通过核ID映射到不同的物理地址执行。
>
> 3）Early-start优化
> 在常规的Task调度机制或SK内部同步前后两个算子之间，系统通常需要等待所有硬件线程完成当前操作后才启动下一个算子。但实际上，大多数算子在启动初期会先进行 Scalar计算，随后才执行数据搬入和搬出操作。… 首先，在前一个Kernel的MTE3搬出阶段，设置线程级的同步点（set）；其次，在下一个Kernel的MTE2搬入阶段前，仅对数据依赖部分进行同步等待（wait）； 然后，在Scalar阶段无需等待前一个Kernel完成，可直接启动，从而实现两个Kernel之间的"早启动"流水线重叠。

> 2）支持双流并发融合
> 通过多流实现Cube/Vector并发场景后，在简单转换为SK时，如果不感知依赖关系，仅依据执行顺序进行处理，SK内部的执行将退化为串行，导致整体性能相比融合前有所下降。… 可以将算子按类型进行分类，并在SK内部根据Cube和Vector的不同特性分配执行队列；同时，结合流的属性和Event事件，精准插入同步点，从而实现Cube与Vector的高效并发执行。

## Relevance

The authoritative Ascend-specific decomposition of megakernel gains, and it is **not only
launch count**: (1) inter-task scheduling, (2) the per-task Cache Flush that writes back
unnecessary stack space, (3) core-start time including same-address L2 queueing and Scalar
init. Three concrete techniques to copy: ICache preload of the next sub-kernel at 2 KB
alignment, replicating the text segment so cores map to different physical addresses, and
Early-start (set at the previous kernel's MTE3, wait only on the data-dependent part before
MTE2, start Scalar immediately). The last warning is the trap: naive fusion serializes
Cube/Vector unless you partition execution queues by op type.

---

### vLLM-Ascend 优化实践 — parasitic Transpose/Cumsum around GroupedMatmul, measured

https://cann.csdn.net/69afcc540a2f6a37c59646d7.html
《vLLM-Ascend：大模型推理的优化实践》，2025-11-28

> PyTorch的Eager模式可以通过Ascend Extension for PyTorch（torch_npu）无缝对接昇腾芯片，但推理总耗时易受算子下发调度开销影响，出现调度Bound现象，从而显著影响性能。以DeepSeek-V3的MLA结构为例，下图展示了PyTorch在Eager模式下的调度开销情况：NPU在大部分时间处于空闲状态，CPU侧算子调度开销成为推理耗时的关键因素。

> DeepSeek-V3模型普遍使用大EP的推理部署方式，通过减少MoE部分参与GroupedMatmul计算的路由专家数量，以降低单次Decode耗时。然而，在分析EP256配置下的profiling时发现，GroupedMatmul算子执行前出现了额外的Transpose操作，导致性能反而出现较大劣化。
>
> 上图的profiling信息显示，单层MoE中的Transpose算子总耗时高达175us。经过模型代码排查、对比EP128无Transpose场景的dump图，发现问题由以下原因导致：
> 在EP256单卡1专家的配置下，由于专家权重shape中存在值为1的维度，导致针对权重的Transpose算子触发TorchAir的"enable_view_optimize"优化…而被插入了额外的Reshape算子。
> 新引入的Reshape算子使融合Pass "GroupedMatmulTransFusionPass"无法检测到其预定目标模式，最终导致Transpose算子被保留到图执行阶段。

> 对应的torch_npu接口"npu_grouped_matmul"依赖入参"group_list"获取每个子矩阵运算中实际M轴的大小，并支持以"累加和数列"或"逐维度指定"两种形式提供这一参数。vLLM-Ascend的模型脚本中使用了"累加和数列"模式，因而在MoE层中引入了用于实现累加和的Cumsum算子。
>
> 调整TorchAir的优化功能开关"enable_view_optimize"和torch_npu接口"npu_grouped_matmul"的配置后，MoE阶段关键路径上的Transpose/Cumsum算子即被消除。尽管GroupedMatmul算子耗时有所增加，仍能获得单层158us，整网9ms的Decode性能收益，在3K推理长度下可将rollout总耗时优化超过28s。

## Relevance

The decisive sentence is `尽管GroupedMatmul算子耗时有所增加，仍能获得单层158us，整网9ms的Decode性能收益` —
deleting the parasitic ops paid 158 us/layer and 9 ms whole-network **even though
GroupedMatmul itself got slower**. That is the exact shape of the trade a megakernel makes, and
it is the reason not to judge the megakernel by per-op timings. Also names `group_list` cumsum
mode as the source of an extra Cumsum launch per MoE layer, which is live in our path.

---

## 6. vllm-ascend: the W4A8 call sites, weight layouts, and the fused MC2 megakernel that already exists

### The two measured ops, as the framework calls them

https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/device/device_op.py
`BaseDeviceAdaptor` (A2/A3), lines ~334-425.

```python
    @staticmethod
    def npu_grouped_matmul_swiglu_quant(
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        weight_scale: torch.Tensor,
        x_scale: torch.Tensor,
        bias=None,
        use_mxfp_quant: bool = False,
        act_quant_type: torch.dtype | int = torch.float8_e4m3fn,
        weight_quant_type: torch.dtype | int = torch.float8_e4m3fn,
        swiglu_limit: float = 0.0,
        mxfp_quant_dtype: QuantType | None = None,
    ):
        if use_mxfp_quant:
            raise RuntimeError("MXFP MoE quantization is only supported on Ascend A5.")

        return torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz(
            x=x,
            weight=weight,
            weight_scale=weight_scale,
            x_scale=x_scale,
            group_list=group_list,
            bias=bias,
            swiglu_limit=swiglu_limit,
        )
```

```python
    @classmethod
    def npu_grouped_matmul_gmm2(
        cls,
        *,
        hidden_states: torch.Tensor,
        weight: list[torch.Tensor] | torch.Tensor,
        weight_scale: list[torch.Tensor] | torch.Tensor,
        per_token_scale: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
        ...
    ) -> torch.Tensor:
        ...
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=weight,
            scale=weight_scale,
            bias=bias,
            per_token_scale=[per_token_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=fallback_output_dtype,
        )[0]
```

### `AscendW4A8DynamicFusedMoEMethod` — the full expert chain we are replacing

https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/quantization/methods/w4a8/w4a8.py

```python
        hidden_states, swiglu_out_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
            x=hidden_states,
            weight=w1,
            weight_scale=w1_scale,
            x_scale=pertoken_scale,
            group_list=group_list,
            weight_assist_matrix=bias1,
            dequant_mode=0,
            group_list_type=group_list_type,
            swiglu_limit=mlp_compute_input.swiglu_limit,
        )
        dispose_tensor(mlp_compute_input.hidden_states)
        return hidden_states, swiglu_out_scale
```

```python
    def apply_act_quant(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor):
        return torch_npu.npu_dynamic_quant(hidden_states, dst_type=torch.int8)

    def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
        _, _, w2, w2_scale = self._get_mlp_weights(mlp_compute_input.layer)
        _, bias2 = self._get_mlp_scale_bias(mlp_compute_input.layer)
        group_list, group_list_type = self._maybe_convert_group_list(mlp_compute_input)
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2,
            scale=w2_scale,
            bias=bias2,
            per_token_scale=[act_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.bfloat16,
        )[0]
```

### The exact W4A8 weight/scale byte layout the kernel consumes

Same file, class docstring:

```
    Original MoE layer weights generated by msModelSlim:

    - ``model.layers.L.mlp.experts.E.gate_proj.weight``:
      ``torch.int8``, ``[moe_intermediate_size // 2, hidden_sizes]``.
      Each int8 element stores two packed 4-bit weight values along the
      output dimension.
    ...
    - Each linear also has ``scale_bias``: ``torch.float32``,
      ``[moe_intermediate_size, 1]`` for ``gate_proj`` and ``up_proj``,
      and ``[hidden_sizes, 16 // tp_size]`` for ``down_proj``.
    ...
    After processing, ``apply`` passes these tensors to the fused MoE
    operator:

    - ``w13_weight``: ``torch.int32``,
      ``[num_experts, hidden_sizes, moe_intermediate_size // 4]``.
    - ``w2_weight``: ``torch.int32``,
      ``[num_experts, moe_intermediate_size, hidden_sizes // 8]``.
    - ``w13_scale_bias``: ``torch.float32``,
      ``[num_experts, 2 * moe_intermediate_size]``.
    - ``w2_scale_bias``: ``torch.float32``,
      ``[num_experts, hidden_sizes]``.
    - ``w13_weight_scale``: ``torch.int64``,
      ``[num_experts, 2 * moe_intermediate_size]``.
    - ``w2_weight_scale``: ``torch.int64``,
      ``[num_experts, 1, hidden_sizes]``.
```

And the transforms that produce it:

```python
    def _pack_to_int32(self, weight: torch.Tensor):
        # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
        assert weight.shape[-1] % 4 == 0, (
            f"the last dim of weight needs to be divided by 4 but got shape {weight.shape}"
        )
        return weight.view(torch.int32).contiguous()

    def _process_scale(self, scale):
        scale = scale.transpose(1, 2).contiguous()
        scale_np = scale.cpu().numpy()
        scale_np.dtype = np.uint32
        scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
        return scale_uint64_tensor

    def _pack_int4_to_int8(self, weight: torch.Tensor) -> torch.Tensor:
        shape = weight.shape
        weight = weight.reshape(-1, 2)
        weight0 = weight[:, :1]
        weight1 = weight[:, 1:]
        weight1_4 = torch.bitwise_left_shift(weight1, 4)
        weight2_4 = weight0 & 0b00001111
        weight_add = torch.bitwise_or(weight1_4, weight2_4)
        # The clone() call is used to break the view chain
        return weight_add.reshape(shape[:-1] + (shape[-1] // 2,)).clone()
```

```python
    def process_weights_after_loading_modelslim(self, layer):
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()

        layer.w13_weight_scale.data = self._process_scale(layer.w13_weight_scale.data)
        layer.w2_weight_scale.data = self._process_scale(layer.w2_weight_scale.data)

        layer.w13_scale_bias.data = layer.w13_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        layer.w2_scale_bias.data = layer.w2_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        layer.w13_weight_scale.data = self.maybe_squeeze_per_channel_weight_scale(layer.w13_weight_scale.data)
```

https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/utils.py

```python
ACL_FORMAT_FRACTAL_NZ = 29
```

```python
# - 310P: always convert supported weights to FRACTAL_NZ
# - non-310P: follow additional_config.weight_nz_mode
# - FP32: never convert
# - meta tensor: never convert
def maybe_trans_nz(
    weight: torch.Tensor,
    customize_dtype: torch.dtype | None = None,
    input_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not _should_trans_nz(weight):
        return weight
    kwargs = {}
    if customize_dtype is not None:
        kwargs["customize_dtype"] = customize_dtype
    if input_dtype is not None:
        kwargs["input_dtype"] = input_dtype
    return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ, **kwargs)
```

## Relevance

This pins the byte layout our megakernel must consume without re-converting anything: `w13`
int32 `[E, 6144, 2*128/4]`, `w2` int32 `[E, 128, 6144/8]`, scales as **int64-packed uint32 fp32
bit patterns** (`scale_np.dtype = np.uint32` then `astype(np.int64)`) — so the scale tensor
cannot be reinterpreted as float, and `scale_bias` already summed over the K-group axis to
`[E, 2*moe_intermediate]`. `ACL_FORMAT_FRACTAL_NZ = 29` is the format id.

---

### `DispatchFFNCombineW4A8` — a merged A3 MoE megakernel already in vllm-ascend's csrc

https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_def.cpp

```cpp
class DispatchFFNCombineW4A8 : public OpDef {
 public:
  explicit DispatchFFNCombineW4A8(const char *name) : OpDef(name) {
    ...
    this->Input("w1")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
        .IgnoreContiguous();
    ...
    this->Output("expert_token_nums")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ...})

    this->Attr("group").AttrType(REQUIRED).String();
    this->Attr("M").AttrType(OPTIONAL).Int();
    this->Attr("transB").AttrType(OPTIONAL).Bool(false);
    this->Attr("weightNz").AttrType(OPTIONAL).Bool(false);
    this->Attr("swigluLimit").AttrType(OPTIONAL).Float(0.0f);
    this->Attr("worldSize").AttrType(REQUIRED).Int();

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("jitCompile.flag", "static_false")
        .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
    this->AICore().AddConfig("ascend910_93", aicore_config);
    this->MC2().HcclGroup("group");
  }
};

OP_ADD(DispatchFFNCombineW4A8);
```

https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/tiling_args.h

```cpp
namespace Moe {
constexpr uint64_t COMBINE_STATE_WIN_OFFSET = 3U * 1024UL * 1024UL;
constexpr uint64_t NOTIFY_DISPATCH_WIN_OFFSET = 204U * 1024UL * 1024UL;
}  // namespace Moe
```

https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_tiling.cpp

```cpp
    constexpr int64_t MAX_WORLD_SIZE = 768;
    constexpr uint64_t INIT_TILINGKEY = 1000000;
    constexpr uint64_t TILINGKEY_TRANS_B = 1U;
    constexpr uint64_t TILINGKEY_WEIGHT_NZ = 10;
    constexpr uint32_t BLOCK_NUM = 20;
    constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
...
    uint32_t aivNumInitRouting = 2 * BLOCK_NUM;
```

The host call site, and the weight-layout constraint comment:
https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_comm_method.py

```python
                torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
                    x=fused_experts_input.hidden_states,
                    weight1=weights.w1,
                    weight2=weights.w2,
                    expert_idx=fused_experts_input.topk_ids,
                    scale1=weights.w1_scale,
                    scale2=weights.w2_scale,
                    bias1=weights.w1_scale_bias,
                    bias2=weights.w2_scale_bias,
                    probs=fused_experts_input.topk_weights.to(torch.float32),
                    group=self.token_dispatcher.moe_all_to_all_group_name,
                    max_output_size=get_ascend_config().mega_moe_max_tokens,
                    swiglu_limit=self.swiglu_limit,
                    world_size=self.token_dispatcher.ep_world_size,
                    x_active_mask=fused_experts_input.routing.mc2_mask,
                    out=out,
                    expert_token_nums=self.expert_token_nums,
                )
```

```python
        # A8W4-INT MegaMoe reads N from weight1.storageShape.lastDim treated as int8 (N = lastDim*2)
        # and checks weight2.dim0 == N/2, so the weights MUST be int8-shaped (two int4 per byte), NOT
        # the eight-int4-per-int32 packing (that makes the op read N four times too small and fail
        # CheckWeight2Input). The op prototype also REQUIRES FRACTAL_NZ per expert. The W4A8 quant
        # method therefore builds per-expert int8 + FRACTAL_NZ lists (cann_mega_moe_*_weight_list) and
        # they are passed through as-is here. W8A8 weights are already int8 + FRACTAL_NZ, also as-is.
```

```python
        x_active_mask = None
        # Ascend 950 (A5) MegaMoe only support a null x_active_mask, and it
        # must be passed as None. But on A2/A3 it must be valid.
```

## Relevance

A working, merged, A3-targeted (`ascend910_93`) MoE megakernel registered through MC2: one
kernel swallowing dispatch + gmm1 + SwiGLU + quant + gmm2 + combine, with `BLOCK_NUM = 20` AIC
blocks and `aivNumInitRouting = 2 * BLOCK_NUM` AIV, a 16 MiB system workspace, and
`NOTIFY_DISPATCH_WIN_OFFSET = 204 MiB`. It takes `scale1/scale2` and `bias1/bias2` exactly in
the MSD layout above. Two traps it records: W4A8 MegaMoe needs int8-shaped (2 int4/byte)
FRACTAL_NZ per-expert tensors, **not** the int32 packing the non-mega path uses; and
`x_active_mask` must be valid on A2/A3 but null on A5.

---

### The adversarial datapoint: that megakernel is broken on A3 TP16 W4A8 today

https://github.com/vllm-project/vllm-ascend/issues/16914
`[Bug] MegaMoe on Kimi K3 (A3, PD-disaggregated 4P4D) produces garbled output and vector-core crash (acl error 507035)` — linnea-lin-00638949, 2026-09-18, open.

> When enabling MegaMoe (`enable_fused_mc2: 2` + `enable_prefill_mc2: true`) for Kimi K3 W4A8 on Atlas 800I A3, **every request returns garbled tokens**, and the prefill nodes eventually **crash with an AI-vector-core exception (`aclrtSynchronizeEvent` error 507035)** followed by glibc heap corruption (`corrupted size vs. prev_size`) on shutdown.
>
> Disabling MegaMoe on the exact same build/environment produces correct output — so the problem is isolated to the MegaMoe path.
>
> - Hardware: Atlas 800I A3 (910_93), 8 nodes x 16 NPU
> - Deployment: **PD-disaggregated 4P4D**, both sides DP4/TP16 + DCP16, MooncakeConnectorV1 KV transfer, dspark speculative decoding (draft=7)
> - CANN: 9.3.0 weekly `20260916.01` (the A3-ops package bundles `cann-A3-ops-transformer_9.2.0`)
>
> Note: although `multistream_overlap_shared_expert: true` is passed, vllm-ascend auto-disables it (warning `enable_fused_mc2 and multistream_overlap_shared_expert cannot be enabled at the same time. Setting multistream_overlap_shared_expert to False.` appeared 18x = once per worker), so the effective runtime value is `false` on all MegaMoe ranks.

## Relevance

The counterweight, on our exact configuration: A3 910_93, TP16, W4A8, CANN 9.2.0-class ops. The
shipped fused MoE megakernel produced garbled output plus 507035 vector-core faults, and it is
mutually exclusive with shared-expert multistream overlap. Any plan needs an A/B toggle and
per-rank numeric validation, not just a profile.

---

### A3 MSD GMSQ with direct ND/NZ weights — scratch sizes and a measured NZ-vs-ND ratio

https://github.com/vllm-project/vllm-ascend/pull/17201
`[Feature][MoE] Add A3 MSD GMSQ with direct ND/NZ weights` — grandolantow, 2026-09-22, open.

> Add an A3 fused grouped matmul + SiTU + per-token INT8 quantization operator using exact A8 decomposition and INT4 Matmul. Packed ND and native packed INT8 NZ weights are consumed in their existing storage, without a full W4-to-W8 weight replica.
>
> - Read ND directly as ND and native NZ directly as NZ. Validate the packed carrier's dtype, physical geometry and strides; retain expert views without cat/copy.
> - Reconstruct the integer dot product before the existing FP16 channel-scaling boundary, then apply SiTU and dynamic INT8 quantization.
> - Parse routing counts on device on each invocation/replay. Normalize int64/int32/float32 group lists without host value reads.
> - Keep graph-visible activation/accumulator scratch stable per device/stream/K/N. Its size is independent of expert count and layer count for a shared stream/K/N.
> - Accept runtime E/K/N subject to implementation limits, rather than profiling-shape whitelists: E=1..128, K a positive multiple of 64 up to 16384, N a positive multiple of 256 subject to the UB budget, and at most 256 padded M blocks. Support K packing chunks and N tails.
>
> The static pointer metadata requires eager warmup for each weight/scale configuration. Weights/scales must remain static afterward, and graphs sharing scratch must replay serially.
>
> For K=3584/N=6144, the primary scratch is 54.71875 MiB (packed activation + INT32 accumulation). The weight-replica allocation is zero, compared with E*K*N bytes for a fully expanded W8 representation (2.296875 GiB at E=112). These figures exclude outputs, pointer metadata and framework/system allocations.

> Paired operator measurements below use E=14/K=3584/N=6144, capacity=28672 and the eight fitted routing cases. Each ND/NZ pair used the same card and alternating independent weight sets, 50 graph replays per batch and five batches. Values are median NPU Graph Event time per operator call; they are not an end-to-end throughput comparison or a comparison against the prior W8-cache implementation.
>
> | Case | Valid rows | ND (us) | Native NZ (us) | NZ / ND |
> |---|---:|---:|---:|---:|
> | 1 | 320 | 235.37 | 167.37 | 0.711 |
> | 2 | 260 | 242.50 | 177.28 | 0.731 |
> | 3 | 320 | 260.27 | 187.17 | 0.719 |
> | 4 | 320 | 294.33 | 206.60 | 0.702 |
> | 5 | 400 | 308.68 | 224.35 | 0.727 |
> | 6 | 500 | 325.34 | 234.87 | 0.722 |
> | 7 | 600 | 341.40 | 252.78 | 0.740 |
> | 8 | 800 | 373.88 | 284.01 | 0.760 |
>
> Geometric mean NZ/ND: **0.7263**.

## Relevance

A3-specific, MSD-specific, and quantified: NZ-native weights beat ND by 27% on the same kernel.
The shape envelope (`E=1..128`, K a multiple of 64 up to 16384, N a multiple of 256, at most 256
padded M blocks) admits our K=6144 / N=256 (=2*128). The scratch figure (54.7 MiB for
K=3584/N=6144, independent of E and layer count) sizes the workspace our megakernel will need.
The caveat `graphs sharing scratch must replay serially` is a constraint on multi-stream plans.

---

### An Ascend MoE expert megakernel that already shipped a 1.46x win from launch removal alone

https://github.com/vllm-project/vllm-ascend/pull/10488
`[Feature] Add fused W4A4 INT4 MoE "mega" kernel for Qwen3.x-MoE (Ascend 910B)` — Mocchibird, 2026-06-15, open.

> MoE decode serving on 910B is HBM-bandwidth bound. The standard vllm-ascend W4A4 path ("vendor" throughout this PR) already shrinks the routed-expert weights 4x, but it assembles the expert path from a chain of **separate `torch_npu` `npu_*` op launches** — `npu_moe_init_routing_v2` -> `npu_dynamic_quant` -> `npu_grouped_matmul` (`gate_up`) -> `npu_swiglu` -> `npu_dynamic_quant` -> `npu_grouped_matmul` (`down`) -> `npu_moe_token_unpermute` — each paying dispatch + cross-core sync overhead and round-tripping the activations through HBM between ops.
>
> This kernel folds that whole chain into **one MIX launch**, with the cube matmuls overlapped against the vector stages, so the intermediate activations never leave on-chip memory. The payoff is twofold:
>
> - **vs. standard vllm-ascend W4A4 (same checkpoint, same accuracy):** the fused kernel is the headline win — **1.46x higher decode throughput** purely from removing the per-op dispatch/sync/HBM-round-trip overhead. The quantization is byte-identical (cos 0.997 between the two kernels), so this is a free serving-throughput gain over the vendor path.
> - **vs. bf16:** faster on decode (**1.20x**) *and* the 4x-smaller expert weights free enough HBM to grow KV-cache capacity / max concurrency by **+30%**.
>
> | stage | engine | work |
> |---|---|---|
> | 1 | cube | 64x64 block-diagonal Hadamard on activations (fp16 GEMM) |
> | 2 | vector | fp16 -> INT4 activations + fp32 scale, fused routing scatter |
> | 3 | cube | INT4xINT4 -> INT32 `gate_up` |
> | 4 | vector | dequant + SwiGLU + INT4 requant |
> | 5 | cube | INT4xINT4 -> INT32 `down` |
> | 6 | vector | dequant + scatter-add unpermute + top-k-weight combine |
>
> - Cube matmuls: AscendC `MatmulImpl<int4b_t>`. Vector stages: PTO-ISA. Cross-core scheduling via FFTS so cube/vector overlap across expert-chunks.
>
> | config | GPQA decode (3-seed) | vs bf16 | vs vendor W4A4 |
> |---|---:|:--:|:--:|
> | bf16 | 276 +/- 25 tok/s | — | — |
> | w4a4 vendor — `torch_npu` `npu_*` chain (separated) | 227 +/- 22 tok/s | 0.82x | 1.00x |
> | **w4a4 hybrid-mega (ours, fused single launch)** | **332 +/- 18 tok/s** | **1.20x** | **1.46x** |

## Relevance

The single most directly transferable result: an Ascend MoE expert megakernel, byte-identical
numerics (cos 0.997), 1.46x over the stock `npu_*` chain, attributed entirely to
dispatch/sync/HBM-round-trip removal. Its 6-stage alternating cube/vector schedule with FFTS
cross-core scheduling is a structural template, and the use of `MatmulImpl<int4b_t>` for cube
plus PTO-ISA for vector names the tools.

---

### Upstream's measurement that small kernels at batch=1 under graph capture cost ~18.6 us each

https://github.com/vllm-project/vllm-ascend/pull/15714
`[Performance][MoE] Reuse a persistent pad buffer in the MC2 prepare path` — MirkoDeVita98, 2026-09-03, open.

> This costs two extra kernels per MoE layer per forward whenever `pad_size > 0`, i.e. whenever the token count is not an exact multiple of `tp_size` — **7 of every 8 batch sizes at `tp_size=8`**.
>
> Profiled with `torch_npu.profiler` on Qwen3-30B-A3B, 8x Atlas 800I A2, EP=8, batch 1, ACL graph captured, rank 0:
>
> | kernel | before | after |
> |---|--:|--:|
> | `aclnnConstantPadNd_PadV3AiCore_PadV3` | 945.2 us/step | 0.0 (0 calls) |
> | `aclnnConstantPadNd_PadV3AiCore_MemSet` | 808.4 us/step | 0.0 (0 calls) |
> | all padding kernels | 36.7 us/layer | 0.20 us/layer |
>
> At 18.6 us per call for 32 KB of data, this is launch overhead rather than bandwidth, so the absolute cost is largely independent of how many rows are padded.

> **Decode latency, batch 1** — 32 prompts served one at a time, 128 generated tokens, 3 rounds with the arms rotated within each round to cancel thermal drift:
>
> | round | ITL unpatched | ITL patched | delta |
> |--:|--:|--:|--:|
> | 1 | 18.3972 ms | 17.6319 ms | -4.16% |
> | 2 | 18.4899 ms | 17.5666 ms | -4.99% |

And the companion, with a TP16 calibration:

https://github.com/vllm-project/vllm-ascend/pull/16343
`[Performance][Communicator] Replace per-layer F.pad with cat of a persistent zero block in MoE prepare` — Levi-JQ, 2026-09-11, closed.

> On NPU each `F.pad` lowers to two device kernels (`constant_pad_nd` -> MemSet + PadV3) while the padded data is only a few KB in decode, so the four launches are pure overhead: ~44 us per MoE layer (~5.6% of a decode layer's kernel time in a Kimi K3 TP16 profile). With speculative decoding (`num_speculative_tokens=7`, minimum graph tier of 8 tokens), odd batch sizes always pad 8 rows to 16 and hit this path.

## Relevance

Upstream independently measuring, at batch=1 with ACL graph captured, that a trivially small
kernel costs 18.6 us/call and that this is launch-bound not bandwidth-bound — the same inference
we drew from 6.793 vs 2.272 ms. The TP16 anchor is the useful one: 44 us = 5.6% of a decode
layer's kernel time implies ~786 us per decode layer at TP16, and deleting 2 launches/layer
bought 4-5% ITL.

---

### W4A8 GroupedMatmul per-call cost on A3 TP16, and how little tiling moves it

https://github.com/vllm-project/vllm-ascend/pull/16606
`[Performance] Add tuning_config to w4a8 MoE grouped matmul` — linnea-lin-00638949, 2026-09-15, open.

> Passes `tuning_config=[0, 0, -1]` to `torch_npu.npu_grouped_matmul` in `BaseDeviceAdaptor.npu_grouped_matmul_gmm2` (the w4a8 MoE grouped-matmul path).
>
> Setup: kimi-k3 w4a8 on 4x Atlas 800I A3, vllm-ascend v0.26.0 (b5e8e2561, the call site is identical on main), DP4xTP16 hybrid, DSpark k=7 speculative decoding. Baseline = the same service without the parameter.
>
> **Operator level** (torch_npu profiler, rank 13, identical request, decode path; 11776 GroupedMatmul calls in both runs, deterministic and directly comparable):
>
> | metric | baseline | tuning_config | delta |
> |---|---|---|---|
> | total (ms) | 776.2 | 743.4 | -4.2% |
> | mean (us) | 65.9 | 63.1 | -4.2% |
> | median (us) | 61.6 | 58.4 | -5.2% |
> | p90 (us) | 94.8 | 92.4 | -2.5% |
> | max (us) | 165.6 | 151.2 | -8.7% |
>
> **E2E** (benchmark_serving, 20 requests, 20k-40k input tokens, max-concurrency=1):
>
> | metric | baseline | tuning_config | delta |
> |---|---|---|---|
> | duration (s) | 1242.13 | 1161.33 | -6.5% |
> | output throughput (tok/s) | 27.94 | 30.08 | +7.7% |
> | TPOT mean (ms) | 32.70 | 28.51 | -12.8% |
> | TPOT P99 (ms) | 44.82 | 40.79 | -9.0% |

## Relevance

The per-call decomposition of our 5.596 ms GroupedMatmul aggregate: mean 65.9 us, median 61.6 us
per call on A3 TP16 W4A8 decode. Tiling choice moves it only ~4%, which is the evidence that
the gap to 1.212 ms of bytes is structural rather than a tiling miss.

---

### The current ceiling of launch-overhead workarounds at concurrency 1

https://github.com/vllm-project/vllm-ascend/pull/19033
`V0.23.0: Optimization of TPOT under low concurrency conditions` — zhu1365377615, 2026-10-16, open.

> All changes were developed and validated on Atlas 910C (Ascend 910_93, CANN 9.1.0) with Qwen3.5-35B-A3B (TP=2), covering both plain decode and MTP speculative decoding.
>
> 1. **[Perf][Compilation] Add `enable_super_kernel` option for small-operator fusion**
>    New `ascend_compilation_config` key `enable_super_kernel`; when set, passes `super_kernel_optimize` to the npugraph_ex compiler backend so small operators inside static kernels are fused automatically.
>
> 2. **[Perf][Kernel] Lower MoE weight prefetch token threshold to 1**
>    `MOE_PREFETCH_TOKEN_THRESHOLD` 96 -> 1 so MoE weight prefetch also stays active for small decode batches (previously skipped below 96 tokens), improving TPOT at concurrency 1/4/8 when `weight_prefetch_config` is enabled.
>
> 4. **[Perf][Kernel] Overlap shared experts with routed experts and reduce decode overhead**
>    - Runs shared experts on a side stream concurrently with routed experts when `multistream_overlap_shared_expert` is enabled; the shared-expert AllReduce stays on the side stream so it overlaps with routed expert computation.
>    - Allows `VLLM_ASCEND_ENABLE_FUSED_MC2` and `multistream_overlap_shared_expert` to coexist (removes the previous mutual-exclusion check).
>    - Skips dispatch/combine/gmm2 event recording in the overlap path via a new `skip_dispatch_combine_events` forward-context attribute.
>    - Prefetches the next-layer shared-expert gate_up weight together with the routed gate_up weight so it lands in cache during the current layer's GroupedMatmul.
>
> 8. **[BugFix][Kernel] Make multistream shared-expert overlap capture-safe for MTP**
>    The side-stream overlap path was not capture-safe: during NPU graph capture the shared-experts side stream is not registered with the graph pool, so any side-stream op — or even a `wait_stream()` recording an event on it — fails with `aclrtAllocatorGetByStream` / `rtStreamWaitEvent` errors, crashing startup at graph capture when MTP is combined with the overlap.
>
> | Scenario | Before | After |
> |---|---|---|
> | c=1 TPOT (no MTP) | ~8.2 ms | ~7.6 ms |
> | c=1 TPOT (MTP=4, min_tokens=512) | ~7.0 ms | ~5.1 ms (-27%) |
> | c=4 TPOT (MTP=4, min_tokens=512) | ~8.9 ms | ~9.0 ms (unchanged) |
> | c=8 TPOT (MTP=4, min_tokens=512) | ~12.7 ms | ~12.4 ms |

> reuses a pre-allocated NPU event for the mamba align-mode stream sync (removes per-step event create/destroy, ~10us host each step)

## Relevance

The ceiling of the non-megakernel approach: stacking SuperKernel fusion, weight prefetch at
1 token, side-stream shared-expert overlap and event-record elision moves c=1 TPOT only
8.2 -> 7.6 ms without MTP. Two items are directly reusable regardless: lowering the MoE weight
prefetch threshold to 1 token (it is skipped below 96 by default — at batch=1 we get no
prefetch at all today), and prefetching the next layer's shared-expert gate_up weight during
the current layer's GroupedMatmul.

---

### The upstream RFC for a GLM-5.2 megakernel execution backend

https://github.com/vllm-project/vllm-ascend/issues/15639
`[RFC]: GLM-5.2 MegaKernel execution backend for vLLM-Ascend` — keyi-zz, 2026-09-03, open.

> Decode-phase execution in vLLM-Ascend today dispatches one op per layer per step and synchronizes at tensor granularity. This leaves launch/host overhead per step and cannot overlap data movement with compute across ops. GLM-5.2 (`GlmMoeDsaForCausalLM` / `glm_moe_dsa`) decode is a long chain of small-to-medium ops (MLA attention, MoE routing + grouped matmuls, DSA indexer sparse attention, norms, LM head) and is exactly the workload that benefits from whole-graph, tile-granularity execution.
>
> Huawei's MegaKernel / `blockrt` runtime aggregates the ops of a network into a single "mega" kernel, schedules work at MK-Tile granularity across AICores with an AI CPU coordinator + scoreboard + callbacks, and enables per-core copyIn/compute/copyOut software pipelining. This RFC proposes bringing the same execution path to vLLM-Ascend as an opt-in, decode-scoped execution backend for GLM-5.2, without touching default behavior.
>
> 2. **Decode-only seam**: pure decode batches run on a pre-built MegaKernel DAG(`blockrt` `Glm52MegaKernel`); prefill and chunked-prefill batches keep the native vLLM-Ascend path. Logits (incl. LM head) are produced inside the DAG.
> 4. **One physical KV cache**: force the packed SFA C8 layout for `glm_moe_dsa` under the flag so native prefill writes and mega decode read the same MLA + DSA indexer caches. Persistent per-batch-size buffers + per-step RoPE caches keep all tensor addresses stable so the captured DAG is replayed instead of re-captured.
> 6. **Incidental, gated fixes**: ascend-config sparse-SFA-C8 forcing, A5 meta-registration skip + `MK_ENABLE_CUSTOM_OP` escape hatch, and small W4A8 MoE grouped-matmul scale/offset fixes (candidates to split into separate PRs).
>
> Milestone scope: GLM-5.2 decode on Ascend 950-family (A5), TP=1; prefill and TP>1 remain follow-ups.
>
> - Known caveats recorded in the doc: shared-cache coupling to the SFA C8 layout; indexer scale-cache placeholders must be 2-D; `indexer_rope` registration must be verified before MTP runs; accuracy/perf numbers to be attached before code PRs.

## Relevance

The upstream RFC for our exact target, naming the runtime (`blockrt`, MK-Tile granularity,
AI CPU coordinator + scoreboard + callbacks, per-core copyIn/compute/copyOut pipelining), the
correctness seam (decode-only DAG, prefill stays native), and the address-stability requirement
for DAG replay. Critically, its milestone scope is **A5 and TP=1** — our A3 / TP16 case is
explicitly listed as a follow-up, i.e. new ground.

---

### The semantic hazard in collapsing gmm1 + SwiGLU + quant: the fused kernel's SwiGLU has no clamp

https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/ascend.py

```python
def _uses_fused_gmm1(kernel, config: MoeRunnerConfig) -> bool:
    ...
    # The fused gmm1 kernel (npu_grouped_matmul_swiglu_quant_v2) applies plain
    # SiLU with no clamp. Models whose activation is SiLU-with-clamp
    # (swiglu_limit, e.g. DSV4) must keep the unfused path so the clamp is
    # applied (NPUSwigluMxfp8Quant / NPUSwigluStepAndMul below).
    ...
        # MXFP8 has no unfused gmm1 path at all, so a swiglu_limit
        # checkpoint cannot be served by it.
    ...
        return False  # W4A8MXFP4: fall back to unfused to preserve the clamp
```

```python
            # Fused methods (MXFP8; MXFP4 W4A8 via use_fused_gmm1) fold
            # gate/up + swiglu + requant into gmm1, so there is no separate
            # activation step — run() skips it. Left None on purpose so that
            # reaching for it fails loudly instead of silently applying an
            # unfused swiglu to already-requantised activations. This holds
            # for both dispatchers: ascend_tp gets its activation quant fused
            # into routing, DeepEP dispatches bf16 and gmm1 quantises it
            # itself.
```

## Relevance

A correctness constraint for our fused epilogue: if GLM-5.2's activation carries a
`swiglu_limit`, the clamp must be implemented in our kernel. Note vllm-ascend's A3 path does
plumb `swiglu_limit` into `grouped_matmul_swiglu_quant_weight_nz`, and the V3 WeightNz doc
exposes `clampLimit`/`gluAlpha`/`gluBias` — but the stock A5 `_v2` fp8 branch drops it.

---

### A per-pipe reference for what a well-tuned GMSQ looks like on real silicon

https://github.com/triton-lang/triton-ascend/issues/1533
`【性能问题】手写的grouped_matmul_swiglu_quant triton算子和ascendc对应算子性能差距大` — boywithstripe, 2026-08-13.

```
对比项目                         AscendC              Triton
shape                            x[2048,10240]        x[2048,10240]
                                 W[8,10240,8192]      W[8,10240,8192]
dtype                            int8 (A8W8)          int8 (A8W8)
性能对比（x倍ascendc）           1.0                  6.57
总耗时 (ms)                      0.8348               5.4828
带宽 (GB/s)                      1.01                 10.11
Vector时间 (us)                  61.47                2033.41
Vector占比 (%)                   7.37                 37.09
Scalar时间 (us)                  31.38                2849.66
Scalar占比 (%)                   3.76                 51.98
MTE2时间 (us)                    53.75                1765.74
MTE2占比 (%)                     6.44                 32.21
MTE3时间 (us)                    37.33                10.22
Cube时间 (us)                    495.4                648.53
Cube占比 (%)                     59.40                11.83
fixpipe时间 (us)                 21.98                1086.53
MTE1时间 (us)                    430.35               1550.43
MTE2时间 (us)                    715.88               3415.47
GM→UB带宽 (GB/s)                 1.83                 20.07
UB→GM带宽 (GB/s)                 0.20                 0.15
L2读命中率 (%)                   98.82                99.91
L2写命中率 (%)                   9.84                 14.47
Block Dim                        24                   24
```

> 在大shape情况下， vec耗时明显，整体性能都消耗在vec上面，cv算子整体流水没有组织好

> 硬件：950PR
> CANN：9.0.0
> Triton：3.2.0
> Triton-Ascend：3.2.1

## Relevance

A per-pipe target: the vendor AscendC GMSQ is 59.4% Cube with Vector at only 7.37% and Scalar
at 3.76%, at Block Dim 24. If our megakernel's GMSQ stage does not land near that Cube
fraction, the loss is in Scalar/MTE1 orchestration, not the matmul. The Triton column shows the
failure signature of a badly pipelined CV kernel: Scalar 51.98%, fixpipe 1086 us, 6.57x slower.

---

### The routing ops a megakernel must absorb, and the AICPU-argsort workaround

https://github.com/vllm-project/vllm-ascend/blob/v0.9.1/vllm_ascend/quantization/w8a8_dynamic.py

```python
        if hasattr(torch_npu, "npu_moe_init_routing_quant"):
            quantized_tokens, expanded_row_idx, global_expert_tokens, _, token_scales = torch_npu.npu_moe_init_routing_quant(
                hidden_states,
                expert_idx=topk_ids.to(torch.int32),
                active_num=0,
                expert_capacity=0,
                expert_num=global_num_experts,
                drop_pad_mode=0,
                expert_tokens_num_mode=2,
                expert_tokens_before_capacity_flag=False,
                quant_mode=1,
            )
```

```python
        hidden_states, dynamic_scale, inverse_indices, expert_tokens = torch_npu.npu_moe_re_routing(
            gathered_tokens,
            gather_sizes.view(ep_group.world_size, -1),
            per_token_scales=dynamic_scale)
```

```python
        hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens)

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts)
```

```python
    if expert_map is not None:
        reordered_outputs = torch.index_select(
            hidden_states,
            dim=0,
            # Workaround: Convert to float so that argsort runs on AI Core instead of slower AICPU
            index=inverse_indices.to(torch.float32).argsort().to(torch.int32))
```

```python
        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )
```

And the W4A8 carve-out, same file:

```python
    # `expand_x` will be disposed in the `apply_mlp` function
    if w1_scale_bias is None:
        down_out_list = apply_mlp_decode([expand_x], ...)
    else:
        # w4a8 scene, cannot use apply_mlp_decode because the operator is not supported
        down_out_list = apply_mlp(expand_x, ..., w1_scale_bias=w1_scale_bias, w2_scale_bias=w2_scale_bias)
```

And the group-list cumsum helper that runs per call:

https://github.com/vllm-project/vllm-ascend/blob/v0.11.0rc0/vllm_ascend/ops/moe/moe_mlp.py

```python
def cumsum_group_list(group_list: torch.Tensor,
                      group_list_type: int,
                      active_num: int = 0,
                      expert_num: int = 0) -> torch.Tensor:
    if group_list_type not in [0, 1, 2]:
        raise ValueError(
            f"group_list_type should be in [0, 1, 2], but received {group_list_type}"
        )

    if group_list_type == 0:
        return group_list
    if group_list_type == 1:
        return group_list.cumsum(dim=0)

    experts = pad(group_list[:, 0], (1, 0))
    tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
    cumsum_group_list = torch.full(size=(expert_num, ),
                                   fill_value=active_num,
                                   dtype=group_list.dtype,
                                   device=group_list.device)

    for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
        if end > start:
            cumsum_group_list[start:end] = tokens[i]

    return cumsum_group_list
```

and in the W4A8 path, the cat/diff conversion on the hot path:

```python
        if w1_scale_bias is not None:
            if group_list_type == 0:
                group_list = torch.cat(
                    [group_list[:1],
                     torch.diff(group_list, dim=0)])
                group_list_type = 1
```

## Relevance

The complete op inventory a megakernel must swallow: `npu_moe_init_routing{,_quant,_v2}`,
`npu_moe_re_routing`, `npu_moe_compute_expert_tokens`, `npu_moe_finalize_routing`, plus
`npu_dynamic_quant`, plus the `cat`/`diff`/`cumsum` group-list conversions, plus an
`argsort` that needs a float cast to avoid landing on AICPU. At batch=1 every one of these is a
fixed-cost launch on data of a few hundred bytes. Upstream also records that the decode-tuned
int32-accumulate path (`apply_mlp_decode`) is unavailable once `scale_bias` is present, i.e.
W4A8 never got the decode specialization.

---

### Huawei's own serving stack (omniinfer): the W4 path still runs unfused, with per-call dummy tensors

https://gitee.com/omniai/omniinfer — `omni/layers/moe/fused_moe/fused_moe.py`

```python
    if layer.weight_num_bits == 8:
        if model_extra_config.operator_opt_config.enable_gmm_swiglu_quant:
            gate_up_proj, pertoken_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                sorted_tokens, [layer.w13_weight], [layer.w13_weight_scale], pertoken_scale, expert_tokens)
        else:
            ...
    elif layer.weight_num_bits == 4:
        tuning_config_13 = None
        tuning_config_2 = None
        if not model_extra_config.operator_opt_config.gmm_nz:
            tuning_config_13 = model_extra_config.task_config.decode_gear_list[:1]
            tuning_config_2 = model_extra_config.task_config.decode_gear_list[:1]
        elif model_extra_config.operator_opt_config.new_w4_op:
            tuning_config_13 = [1, 1]
            if model_extra_config.task_config.decode_gear_list[0] >= 32:
                tuning_config_2 = [256]
        elif model_extra_config.task_config.decode_gear_list[0] >= 32:
            tuning_config_13 = [256]
            tuning_config_2 = [256]
        gate_up_proj = torch_npu.npu_grouped_matmul([sorted_tokens], [layer.w13_weight], bias=[layer.w13_weight_bias], scale=[layer.w13_weight_int4_scale],
                                                    ...
                                                    tuning_config=tuning_config_13, output_dtype=torch.bfloat16)[0]

        fake_scale = torch.ones(layer.w13_weight_int4_scale.shape, dtype=torch.float32, device="npu").view(-1, layer.w13_weight_int4_scale.shape[-1])
        pertoken_scale = torch.ones(pertoken_scale.shape, dtype=torch.float32, device="npu")
        gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(gate_up_proj, weight_scale=fake_scale,
                                                                          activation_scale=pertoken_scale,
                                                                          bias=None, quant_scale=None,
                                                                          quant_offset=None, group_index=expert_tokens,
                                                                          activate_left=True, quant_mode=1)
```

Weight prefetch of the *next* layer's attention weight during the MoE stage:

```python
        if model_extra_config.operator_opt_config.attn_prefetch > 0 and next_attention_weights is not None and next_attention_weights['q_a_proj_weight'] is not None:
            attn_prefetch_size = model_extra_config.operator_opt_config.attn_prefetch * 1024 * 1024
            torch_npu.npu_prefetch(next_attention_weights['q_a_proj_weight'], hidden_states_experts, attn_prefetch_size)
```

## Relevance

Huawei's production stack gates the GMSQ fusion behind `enable_gmm_swiglu_quant` and only for
W8; the W4 path still runs gmm1 -> `npu_dequant_swiglu_quant` -> gmm2 and allocates a
`fake_scale` plus a unit `pertoken_scale` **per call** (extra allocations and launches on the
hot path). `tuning_config` is selected from `decode_gear_list` and toggles between `[1, 1]` and
`[256]` — the same knob PR #16606 exploited. `torch_npu.npu_prefetch` with a byte budget is the
documented API for cross-stage weight prefetch, which is what our megakernel must do internally.

---

### MindSpeed: `group_list_type` 0 vs 1 are different compiled ops, and `act_type` already exists

https://gitcode.com/Ascend/MindSpeed — `mindspeed/ops/gmm.py`, `mindspeed/ops/quant_gmm.py`

```python
class GMMFunction(torch.autograd.Function):
    builder = GMMOpBuilder()
    builder2 = GMMV2OpBuilder()

    @staticmethod
    def forward(ctx, original_weight, x, weight, bias, group_args):
        group_list, group_type, gemm_fusion, group_list_type, group_list_data_type = group_args
        ...
        if group_list_type == 0:
            outputs = GMMFunction.builder.load().npu_gmm([x], [weight], bias, group_list, group_type, group_list_type)
        elif group_list_type == 1:
            outputs = GMMFunction.builder2.load().npu_gmm([x], [weight], bias, group_list, group_type, group_list_type)
```

```python
def npu_quant_gmm(x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None,
                  output_dtype=None, act_type=0):
    return _npu_quant_gmm_common(x, weight, scale, offset=offset, per_token_scale=per_token_scale,
                                 bias=bias, group_list=group_list, group_list_type=0, output_dtype=output_dtype,
                                 act_type=act_type)

def npu_quant_gmm_v2(x, weight, scale, *, offset=None, per_token_scale=None, bias=None, group_list=None,
                     output_dtype=None, act_type=0):
    return _npu_quant_gmm_common(x, weight, scale, offset=offset, per_token_scale=per_token_scale,
                                 bias=bias, group_list=group_list, group_list_type=1, output_dtype=output_dtype,
                                 act_type=act_type)
```

## Relevance

`group_list_type` 0 (cumsum) and 1 (count) are separate compiled ops (`GMMOpBuilder` vs
`GMMV2OpBuilder`), so the cumsum/count choice is a build-time specialization, not a runtime
branch — which means a batch=1 megakernel can pick one and drop the other's code path entirely.
`act_type` confirms in-kernel activation selection is an existing axis.

---

### `npu_quant_grouped_matmul_dequant` — the variant that absorbs the input per-token quant

https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/QuantGroupedMatmulDequantNpuOpApi.cpp

```cpp
static bool is_nz_format(const at::Tensor& tensor) {
  const torch_npu::NPUStorageDesc& tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
  return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ || tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ_C0_4 ||
      tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ_C0_16;
}
at::Tensor npu_quant_grouped_matmul_dequant(
    const at::Tensor& x,
    const at::Tensor& quantized_weight,
    const at::Tensor& weight_scale,
    const at::Tensor& group_list,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& x_scale,
    const c10::optional<at::Tensor>& x_offset,
    const c10::optional<at::Tensor>& smooth_scale,
    c10::optional<c10::string_view> quant_mode) {
  if (is_nz_format(quantized_weight)) {
    static const bool is_weight_nz_available = check_aclnn_kernel_available("aclnnQuantGroupedMatmulDequantWeightNZ");
    ...
```

Call site showing the fused input quant:

```python
    def apply_gmm1_act_quant(self, mlp_compute_input: MoEMlpComputeInput):
        """Fused gmm1 (quant + dequant) + swiglu via 310P kernels.

        ``npu_quant_grouped_matmul_dequant`` quantizes the hidden states to
        int8 internally (``quant_mode="pertoken"``) and dequantizes the gmm1
        output, so the activation scale is not needed by ``apply_gmm2``.
        """
```

### And the WeightNZ shape-inference trap

https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/GroupedMatmulSwigluQuantNpuOpapi.cpp

```cpp
    auto x_size = x.sizes();
    // CANN WeightNZ infers logical N from weightScale, not from packed/NZ weight storage shape.
    int n = weight_scale.sizes().back();
    int m = x_size[0];
    int k = x_size[1];

    at::Tensor output = npu_preparation::apply_tensor_without_format({m, n/2}, c10::dtype(c10::ScalarType::Char));
    at::Tensor output_scale = npu_preparation::apply_tensor_without_format(x_scale, {m});
    at::Tensor output_offset = npu_preparation::apply_tensor_without_format({}, c10::dtype(c10::ScalarType::Float));

    EXEC_NPU_CMD(
        aclnnGroupedMatmulSwigluQuantWeightNZ,
        x, weight, bias, offset, weight_scale, x_scale, group_list,
        output, output_scale, output_offset);
```

## Relevance

`quant_mode="pertoken"` absorbing the input quant inside the GEMM is the pattern for deleting
our separate `npu_dynamic_quant` launch. The WeightNZ comment is a trap worth copying: logical
N comes from `weight_scale.sizes().back()`, not from the packed weight storage shape.

---

## 7. The MonoMoE kernel as shipped in FlashInfer — a complete design spec for a batch<=8, E=256 MoE megakernel

### `docs/design_docs/monomoe_kernel.md`

https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/monomoe_kernel.md
Raw: https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/docs/design_docs/monomoe_kernel.md
Source tree: `csrc/fused_moe/monomoe/` (`monomoe_binding.cu`, `monomoe_wrapper.cuh`,
`src/moe.cuh`, `src/moe_up_projection.cuh`, `src/moe_down_projection.cuh`,
`src/moe_routing.cuh`, `src/moe_scale_inputs.cuh`, `src/moe_tma.cu`),
Python: `flashinfer/fused_moe/monomoe.py`, bench: `benchmarks/bench_monomoe.py`.

Scope and fixed shape — note `E=256` and `M <= 8`:

> # MoE Monokernel — Design
>
> A single persistent CUDA kernel that executes an entire FP8 block-wise MoE
> layer for small decode batches (BS ≤ 8) on Hopper (sm_90a): routing → up
> projection (+SiLU+re-quant) → down projection → weighted combine.  One launch,
> no intermediate kernel boundaries, capturable into a CUDA Graph.
>
> Fixed shape (`Dims_BS8_E256_N512_K2048_BlockFP8_WGMMA_TMA`): E=256 experts,
> N=512 (`moe_intermediate_size` per half), K=2048 hidden, M ≤ 8 tokens.
>
> ```text
> GM in:  activations [BS,K] bf16      router_logits [BS,E] bf16
>         w_up   [E,2N,K] fp8 + block scales [E,2N/128,K/128] fp32
>         w_down [E,K,N]  fp8 + block scales [E,K/128,N/128]  fp32
> GM out: activations_out [BS,K] bf16
> ```

The execution model — one block per SM, roles fixed for the whole kernel, phase changes
reassign *what* each block works on rather than relaunching:

> ## Execution model
>
> - `GRID_SIZE` blocks (default 128), `BLOCK_SIZE = 384` threads = 12 warps.
>   The whole grid is launched ONCE and persists through all five phases —
>   no blocks are created or retired between stages; a block changes *role*
>   (which expert / which output tile it works on) at each phase boundary.
> - `__launch_bounds__(BLOCK_SIZE, 1)` + `GRID_SIZE <= SM count` pin one block
>   per SM.  This co-residency invariant is what makes the cross-block
>   flag/sentinel handoffs (below) deadlock-free: every block is scheduled from
>   launch, so a block spinning on another block's flag can never wait on a
>   block that has not yet been scheduled.
> - Warp roles (fixed for the whole kernel; the same physical warps take
>   phase-specific duties, detailed per stage below):
>     - warps 0–7 (threads 0–255): **calc warps**, forming two WGMMA
>     warpgroups — WG0 = warps 0–3, WG1 = warps 4–7.  A WGMMA is issued by
>     all 128 threads of a warpgroup together.
>     - warp 8 lane 0 (thread 256): the single **TMA launcher thread** —
>     issues every `cp.async.bulk.tensor.2d` and arms every mbarrier in the
>     block, in all phases.
>     - warps 8–11 (threads 256–383): **prefetch warps** (PF0–PF3) — scale
>     loads, the deferred up-proj epilogue, the deferred down-proj
>     accumulate, and (warps 1–11) the Phase-2 quantization.

The derived geometry table — every tile count is a closed form of (E, N, K) and five constants:

> The stage geometry is fixed by the shape's `KernelConfig` (`GRID_SIZE = 128`,
> `DOWN_COL_TILE` = DCT = 256, `K_STEP_UP` = KUP = 256, `K_STEP_DOWN` = KDN = 256,
> `UP_W_SLOTS` = SLOTS = 4; `UP_COL_HALVES` = UCH is derived).  All stage geometry
> follows from these and the (E, N, K) dims:
>
> | derived quantity  | formula                                 | value | meaning                             |
> | ----------------- | --------------------------------------- | ----- | ----------------------------------- |
> | `UCH`             | `max(1, 2N·DCT / (128·K))`              | 1     | 128-row M-atoms per up-block        |
> | `UP_GRID`         | `2N / (128·UCH)`                        | 8     | blocks per expert, up-proj          |
> | `UP_GROUPS`       | `GRID_SIZE / UP_GRID`                   | 16    | experts in parallel, up-proj        |
> | `K_TILES_UP`      | `K / KUP`                               | 8     | up-proj outer K iterations          |
> | `K_SUBSTEPS_UP`   | `KUP / 128`                             | 2     | 128-K substeps per iteration        |
> | `UP_ARM_DISTANCE` | `max(1, SLOTS − 2)`                     | 2     | weight-TMA prefetch distance        |
> | `DOWN_GRID`       | `K / DCT`                               | 8     | blocks per expert, down-proj        |
> | `DOWN_GROUPS`     | `GRID_SIZE / DOWN_GRID`                 | 16    | experts in parallel, down-proj      |
> | `DOWN_COL_HALVES` | `DCT / 128`                             | 2     | 128-col WGMMA passes per K-step     |
> | `K_TILES_DOWN`    | `N / KDN`                               | 2     | down-proj outer K iterations        |
> | `K_SUBSTEPS_DOWN` | `KDN / 128`                             | 2     | 128-K substeps per iteration        |
> | `K_BLOCKS_TOTAL`  | `K / 128`                               | 16    | routing-window / quantization atoms |
>
> `UP_GROUPS == DOWN_GROUPS` (16), so the site-#2 producer/consumer sets coincide.
> The routing-window tile is 32 KB (BS=8).  These constraints hold: `GRID_SIZE` is
> a multiple of `UP_GRID` and `DOWN_GRID` and ≤ SM count; `K_TILES_UP` is a
> multiple of SLOTS (cross-expert stitch slot alignment); `K_TILES_DOWN` is even
> (the inter-expert lookahead lands in the slot the next expert reads); SHM fits
> the 224 KB budget.

The five-phase pipeline, with the two barrier-free handoff sites:

> ## Phase pipeline
>
> ```text
> Phase 1  routing (calc warps: topK)              ∥  routing-window TMA:
>                                                     full [BS,K] bf16 tile
>                                                     → SHM bf16_in_full,
>                                                     completion on bar_rwin
> Phase 2  warp 0: prepare_moe_topk            ∥  warps 1..11: wait bar_rwin,
>          (expert tally / prefix sums /              quantize bf16 → fp8_act_full
>           slot assignment)                          + per-128-K act_scale
>          __syncthreads()  — publishes both
> Phase 3  up-projection (WGMMA + weight-TMA pipeline)
>          → spec->temp_fp8 (expert-sorted rows) + temp_act_scale
> Site #2  NO barrier — sentinel handoff: each act-scale cell doubles as the
>          readiness flag for its fp8 payload; the down-proj polls the cells
>          it consumes (per-expert granularity)
> Phase 4  down-projection (WGMMA + weight/activation TMA double-buffer)
>          → atomicAdd into spec->down_partial_out [BS,K] fp32
> Site #3  NO barrier — flag counters: each block bumps its col stripe's
>          arrival counter; only that stripe's Phase-5 writer polls it
> Phase 5  fp32 → bf16 cast + writeback (first DOWN_GRID blocks only)
> ```
>
> Zero-fill of `down_partial_out` happens at kernel entry with no cross-block
> sync: every block runs the full routing+quantize+up-proj pipeline (tens of µs)
> before its first Phase-4 atomicAdd, while the zero-fill retires within the
> launch's first ~µs.  Phase 5 `=`-writes every output element, so no output
> pre-zero pass or extra grid sync exists.

Routing is *replicated* across the grid rather than partitioned:

> ### Phase 1 — routing ∥ input prefetch (all GRID_SIZE blocks, replicated)
>
> Every block executes Phase 1 identically and independently — routing and
> the input tile are needed by every block later, and each block has its own
> SHM, so the work is *replicated* across the grid rather than partitioned
> (the work is tiny: 8 tokens × E logits).  No cross-block communication.
>
> - **Calc warps 0–7 — top-k routing (`topK`)**: one warp per token
>   (warp w handles token w; warps ≥ `batch_size` return immediately).
>   Within a warp, lane t owns experts `{t, t+32, t+64, ...}` in registers
>   (`E/32` per lane — 8 for E=256).  The warp runs `top_k` rounds of
>   warp-reduce-argmax and lane 0 writes the token's ids/weights to
>   `topk_ids_flat` / `topk_weights_flat`.

The up-projection phase, including the deferred epilogue hidden under the *next* expert:

> ### Phase 3 — up-projection (grid partitioned: UP_GROUPS × UP_GRID)
>
> - **Blocks per expert**: `UP_GRID = 2N/(128·UCH)` blocks jointly produce
>   one expert's full `2N` intermediate rows; block `up_block_idx` owns rows
>   `[up_block_idx · 128·UCH, +128·UCH)`.  Here: 8 blocks × 128 rows (64 gate
>   + 64 up features each, interleaved — UCH = 1).
> - **Experts in parallel**: `UP_GROUPS = GRID_SIZE/UP_GRID` = 16 expert groups
>   run concurrently, one active expert per group at a time.
> - **Expert loop**: group g iterates the *active* expert list (built by
>   Phase 2, ascending id, length `expert_count ≤ min(E, BS·top_k)`) as
>   `e = g, g + UP_GROUPS, g + 2·UP_GROUPS, ...`.  Each group therefore
>   visits ≤ `ceil(expert_count / UP_GROUPS)` experts — with BS=8, top_k=8
>   at most 64 are active, so ≤ 4 per group at UP_GROUPS=16.
> - **Per expert, warp duties**:
>     - warp 8 lane 0 (launcher): inside each K-iteration, arms + TMAs the
>     weight slot `UP_ARM_DISTANCE = max(1, SLOTS−2)` iterations ahead; on
>     the last `UP_ARM_DISTANCE` iterations it stitches the *next* expert's
>     first tiles instead, so the pipeline never drains across experts.
>     - warp 8 (lanes 0–31, as PF warp 0): cp.async-prefetches the *next*
>     expert's block-scale tile into the `up_scale` ping-pong during the
>     current K-loop.
>     - warps 8–11 (PF0–PF3): the **deferred epilogue of the previous
>     expert** — one warp per token, `ceil(BS/4)` waves spaced across the
>     first `K_TILES_UP - 1` iterations: read `post_silu_scratch`,
>     warp-reduce max over the block's 64 features,
>     fp8-quantize, store to `temp_fp8[sorted_slot_row]` +
>     `temp_act_scale`.  The last visited expert has no successor and
>     drains inline on calc warps after the loop (one token per warp).

The weight pre-interleave that makes SwiGLU a register-local combine:

> **Interleaved up-proj weights (UCH == 1).**  One 128-row A-tile packs 64 gate
> + 64 up rows in the gate/up *pair layout*.  The tensor must be pre-interleaved
> in Python (`interleave_for_tma_wgmma_up`) so a single 128×128 TMA fetches one
> full WGMMA A-tile.  Each lane then holds gate and up for the same output
> feature, so silu(gate)·up is a register-local combine.

The down projection, with the inter-expert lookahead:

> ### Phase 4 — down-projection (grid re-partitioned: DOWN_GROUPS × DOWN_GRID)
>
> The same 128 blocks re-map: `down_group = blockIdx.x / DOWN_GRID`,
> `down_block_idx = blockIdx.x % DOWN_GRID`,
> `base_col = down_block_idx · DOWN_COL_TILE`.
>
> - **Expert loop**: `e = down_group, down_group + DOWN_GROUPS, ...` over
>   the same active-expert list.  Note the up and down loops visit experts
>   in a different interleaving; correctness needs only that Phase 3
>   finished the expert before Phase 4 reads it, which site #2 guarantees.
> - **Per expert, warp duties**:
>     - warp 8 lane 0 (launcher): prefetches K-step s+1 during step s; on
>     the last step prefetches the *next expert's* step-0 weight +
>     activation tiles instead (inter-expert lookahead).  The activation
>     tile is one bulk TMA per 128-K substep covering all ≤ 8 routed rows
>     of the expert (fetched from the contiguous `temp_fp8` slab).
>     - warp 8 (PF0): loads the expert's per-token activation scales;
>     warp 9 (PF1): loads the expert's weight scales — both once per
>     expert, in parallel, before the K-loop.
>     - warps 8–11 (128 PF threads): the **deferred accumulate of the
>     previous expert** — `out_accum[tok][col] += down_out[col][rank]`,
>     the (tok, col) plane sliced across the first `K_TILES_DOWN − 1`
>     iterations.

The cross-block synchronization section — this is the complete protocol:

> ## Cross-block synchronization (flag/sentinel handoffs)
>
> There are no grid barriers.  The kernel launches via plain
> `cudaLaunchKernel` (CUDA-Graph capturable) and orders its two cross-block
> handoffs through the data path, so consumers wait only on the values they
> actually need.  Both handoffs rely on the one-block-per-SM co-residency
> invariant (a spinning consumer needs its producers scheduled), enforced at
> compile time by `__launch_bounds__(BLOCK_SIZE, 1)` and at runtime by the
> `GRID_SIZE <= SM count` check in the wrapper.
>
> **Site #2 (Phase 3 → 4), sentinel-in-data:** each `temp_act_scale` cell
> doubles as the readiness flag for the fp8 payload segment it covers.  The
> producing warp stores the payload, then `__syncwarp()` +
> `__threadfence()` + `atomicExch` of the scale (`moe_publish_act_scale`),
> clamped to >= FLT_MIN so the sentinel `+0.0f` is never a valid value.  The
> down-projection polls exactly the cells of the expert it is about to
> consume (device-scope atomic reads) before reading scales or issuing the
> activation TMA; the inter-expert lookahead TMA has its own sweep-poll
> (`moe_wait_expert_scales_published`).  This gives per-expert granularity —
> down work for an expert starts as soon as that expert's rows are published.
>
> **Site #3 (Phase 4 → 5), arrival flags:** `down_partial_out` is
> atomicAdd-accumulated, so readiness cannot live in the data (a partial sum
> looks complete).  Each block instead bumps its col stripe's arrival
> counter (`__syncthreads()`, `__threadfence()`, `atomicAdd(+1)`) and runs
> to exit; only the stripe's single Phase-5 writer polls the counter up to
> `DOWN_GROUPS`.
>
> **Reset discipline (both sites):** flag state must never leak across
> launches, so it is double-buffered by launch parity: every block bumps its
> private `launch_flip` word once per launch (all blocks agree on parity
> with no cross-block sync), the current parity's state is consumed, and the
> OTHER parity's state is zero-refilled in the prologue, off the critical
> path.  A `torch.zeros` scratchpad allocation establishes the invariant for
> the first launch — no host-side re-init is ever needed, and the scheme is
> CUDA-Graph-replay safe (parity keeps alternating across replays).

The shared-memory budget, with the lifetime union and the bank-conflict padding:

> ## Shared memory (`MoE_SHM`, ≤ 224 KB)
>
> The dominant space is a union whose members have strictly disjoint lifetimes
> (separated by the Phase-2 trailing sync and the site-#2 barrier):
>
> | view                                   | phase | size (N512_K2048 cfg0) |
> | -------------------------------------- | ----- | --------------- |
> | `bf16_in_full[K/128][BS][128]`         | 1–2   | 32 KB           |
> | `w_wgmma[UP_W_SLOTS][M_total][128]`    | 3     | 64 KB           |
> | `w_down_wgmma[2][DCT·K_SUBSTEPS][128]` | 4     | dominates       |
>
> Other notable fields:
>
> - `fp8_act_full[K/128][8][T_TILE+1][16]` — single-buffer fp8 activations for
>   the whole K range (Phase 3 reads with no slot alternation).  The 9th
>   token row per 16-B chunk is padding: it moves the chunk stride from 128 B
>   to 144 B so the routing-quantize stores and the WGMMA B reads are
>   bank-conflict-free.  The WGMMA B descriptor's `LBO = 144` steps over the
>   pad.
> - `a_down_wgmma[2][K_SUBSTEPS_DOWN][8][8][16]` — down-proj activation
>   double-buffer (SWZ128 atoms).
> - `partial_result` union: `wgmma_out[128][9]` / `down_out[DCT][8]` /
>   `post_silu_scratch[UCH·128][9]` — the +1 column padding makes the
>   `[col][tok]` read pattern bijective over banks (gcd(9,32)=1).
> - mbarriers (`bar_w[UP_W_SLOTS]`, `bar_a[2]`, `bar_rwin`), `alignas(16)`.
> - Routing tables: `expert_slot_start[E]` (u16, `alignas(16)` — Phase B emits
>   packed STS.128 stores), `expert_routed_count[E]` (u8),
>   `sorted_slot[BS·8]` (u8), `down_rank[E][BS]` (u8, `alignas(16)` for the
>   uint4 seed), `up_rank_for_tok[_prev][BS]` + `up_rw_for_tok[BS]` (V2 only).
> - `act_scale[K/128][BS]` — transposed `[blk][tok]` layout for conflict-free
>   scale-apply broadcasts.

The global scratchpad, and the address-stability invariant required for graph replay:

> ## Global scratchpad (`MoEGemmSpec`)
>
> Persistent GM workspace, one per process, zeroed once on first launch.
> `mono_moe` allocates it automatically when the `scratchpad` argument is
> omitted; callers that want to reuse a buffer across launches allocate one
> explicitly with `alloc_scratchpad()`, sized by `get_scratchpad_size_bytes()`
> (the `monomoe_scratchpad_size` binding, i.e. `sizeof(MoEGemmSpec<Dims>)`):
>
> - `temp_fp8[TEMP_ROWS][N]` + `temp_act_scale[TEMP_ROWS][N/DOWN_ACT_BLOCK]` —
>   Phase 3 → Phase 4 handoff, expert-sorted rows.
>   **Layout invariant:** the host computes the device address of `temp_fp8`
>   as `scratchpad + TEMP_FP8_OFFSET` when building the down-activation TMA
>   descriptor, so no field may ever be inserted before `temp_fp8`; new fields
>   (handoff flags) go at the tail.  A `static_assert` in
>   `monomoe_binding.cu` enforces this.
> - `down_partial_out[BS][K]` fp32 — Phase 4 atomicAdd target.
> - sentinel-handoff tail state: `temp_act_scale_alt` (the second scale
>   buffer), `launch_flip[GRID_SIZE]` (per-block private launch counters →
>   buffer parity), `down_ready[2][DOWN_GRID]` (Phase-4→5 arrival counters).

TMA descriptor table:

> | descriptor       | tensor                               | box     | swizzle |
> | ---------------- | ------------------------------------ | ------- | ------- |
> | up weights       | `[E·2N, K]` fp8 (interleaved or raw) | 128×128 | 128B    |
> | activations      | `[BS, K]` bf16                       | 8×128   | none    |
> | down weights     | `[E·K, N]` fp8 (raw)                 | 128×128 | 128B    |
> | down activations | `temp_fp8 [rows, N]` fp8             | 8×128   | 128B    |

## Relevance

This is the closest thing to a specification for what we are building, at E=256 and BS<=8,
and it is in a shipping repository rather than a paper. Seven items transfer directly:

1. **One block per core, roles fixed, phase changes reassign work.** The Ascend analogue is one
   persistent kernel over all AI cores with `ASCEND_IS_AIC`/`ASCEND_IS_AIV` deciding the role
   and the phase deciding which expert stripe the core owns.
2. **Routing is replicated, not partitioned.** `the work is tiny: 8 tokens × E logits` — so
   every core computes top-k itself and no cross-core communication happens. That deletes the
   entire `moe_gating_top_k` / `moe_init_routing` / `moe_compute_expert_tokens` chain from the
   critical path, and it is the opposite of what CANN's `GroupedMatmulSwigluQuantV2` does when
   it walks `groupListGM` with scalar GM reads.
3. **No grid barriers at all.** Both handoffs ride the data path: the activation scale cell
   doubles as the readiness flag for its payload (`clamped to >= FLT_MIN so the sentinel +0.0f
   is never a valid value`), giving per-expert granularity. Compare our measured op, whose
   `pre/mid/post` pipeline does four `SyncAll` per iteration.
4. **The epilogue of expert *i* runs on the prefetch warps during expert *i+1*'s K-loop**, and
   the weight TMA for the next expert is stitched in during the last `SLOTS-2` iterations `so
   the pipeline never drains across experts`. That is the answer to per-group fixed overhead:
   there is no per-group drain.
5. **Gate/up weights pre-interleaved offline** so one tile fetch yields both halves and
   `silu(gate)·up is a register-local combine`. On Ascend this is a `npu_format_cast`-time
   decision, free at inference.
6. **On-chip buffers unioned by lifetime** (`bf16_in_full` / `w_wgmma` / `w_down_wgmma` share
   one region) with explicit `+1` padding to make bank access bijective. Our UB budget is the
   binding constraint; the `168 kb` figure in CANN's own GroupedMatmul design doc is for one
   stage only.
7. **Launch-parity double-buffered flag state**, so nothing leaks between replays and no host
   re-init is needed — the discipline that makes a persistent kernel safe under graph capture,
   which is exactly our aclgraph constraint.

Note the shape mismatch to check: this config is K=2048, N=512; GLM-5.2 at TP16 is K=6144,
N=128 per half. Our N is 4x smaller and K 3x larger, so `UCH = max(1, 2N·DCT/(128·K))` stays 1
but `UP_GRID = 2N/128 = 2` rather than 8 — far fewer cores per expert, far more experts in
parallel. The geometry formulas carry over; the constants do not.

---

### Cross-check: an independent persistent-kernel decode megakernel on AMD

https://github.com/ROCm/iris/pull/541
`MegaKernels in Iris` — ROCm/iris, author `neoblizz`, 2026-06-24, open.

> This pull request introduces the GPT-OSS-120B Megakernel example, providing a comprehensive, high-performance single-GPU implementation of the GPT-OSS-120B model using a persistent Triton kernel. The changes include detailed documentation, benchmarking and accuracy scripts, and a reusable set of device-level Triton operations for attention, quantization, and expert routing. Together, these updates enable end-to-end quantized inference, benchmarking, and accuracy evaluation for this large mixture-of-experts model.

> **2. Benchmarking and Evaluation Scripts:**
> - Introduced `bench_tpot.py` for measuring steady-state decode latency (TPOT) and comparing quantized vs. BF16 inference paths.
> - Added `bench_islosl.py` to benchmark prefill and decode latency across various input/output sequence lengths, reporting throughput and latency metrics.
> - Implemented `acc_eval.py` to compare accuracy between FP8 and BF16 attention (with shared FP4 experts), providing detailed metrics such as top-1 agreement, top-k overlap, KL divergence, and logit cosine similarity.

> **3. Reusable Device Operations:**
> - Created `common/__init__.py` to expose a suite of reusable Triton device ops for attention, quantization, GEMV, routing, and SwiGLU, supporting both single-GPU and future multi-GPU kernels.
> - Added `common/attention.py` containing Triton JIT helpers for RoPE+KV cache appending and per-head flash decode with attention sinks, facilitating both current and future kernel designs.

## Relevance

A second vendor, a third framework, and the same decomposition: one persistent kernel holding
attention + routing + quantization + GEMV + SwiGLU, built from a library of device-level
primitives rather than one monolith, with TPOT and an accuracy-vs-BF16 harness as the two
acceptance gates. The `GEMV` entry in the device-op list is the notable one — at batch=1 the
expert projections are matrix-vector, not matrix-matrix, which matches both the MonoMoE
SwapAB argument and Hazy Research's finding that matvec can beat the matrix unit. Note: a
search snippet attributed a quantized-path TPOT improvement of 29.4 ms to 22.9 ms to this
work; that figure is **not** in the PR body as fetched and is therefore not quoted here.

---

### MonoMoE Python entry point — the full shape and dtype contract

https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/fused_moe/monomoe.py

```python
def mono_moe(
    activations_in: torch.Tensor,
    router_logits: torch.Tensor,
    expert_weights_up: torch.Tensor,
    expert_scales_up: torch.Tensor,
    expert_weights_down: torch.Tensor,
    expert_scales_down: torch.Tensor,
    top_k: int,
    scoring_func: str = "softmax",
    renormalize: bool = True,
    out: Optional[torch.Tensor] = None,
    scratchpad: Optional[torch.Tensor] = None,
    interleave_up: bool = True,
) -> torch.Tensor:
    """Single-kernel block-FP8 top-K MoE (fixed E256/N512/K2048 shape, SM90a).

    The ``(E, N, K)`` shape is derived from the weight tensors and must be the
    fixed E=256/N=512/K=2048 shape the kernel is specialized for; up to 8
    tokens (the BS8 kernel).

    Args:
        activations_in: bf16 input activations ``[M, K]`` (``M <= 8``).
        router_logits: bf16 router logits ``[M, E]``.
        expert_weights_up: fp8_e4m3 up/gate weights ``[E, 2*N, K]``. This
            function applies :func:`interleave_for_tma_wgmma_up` by default;
            pass ``interleave_up=False`` if the tensor is already interleaved.
        expert_scales_up: fp32 block-wise scales ``[E, 2N/128, K/128]``.
        expert_weights_down: fp8_e4m3 down weights ``[E, K, N]`` (raw row-major).
        expert_scales_down: fp32 block-wise scales ``[E, K/128, N/128]``.
        top_k: experts selected per token (1..8).
        scoring_func: ``"sigmoid"`` or ``"softmax"``.
        renormalize: renormalize the top-K weights to sum to 1.
        out: optional bf16 output buffer ``[M, K]``; allocated if omitted.
        scratchpad: optional reusable uint8 scratchpad from
            :func:`alloc_scratchpad`; allocated per-call if omitted.
        interleave_up: apply the gate/up TMA repack to ``expert_weights_up``
            (default True).

    Returns:
        bf16 MoE output ``[M, K]``.
    """
```

```python
    # The kernel reads the up weights in the interleaved Pair_Layout.
    if interleave_up:
        expert_weights_up = interleave_for_tma_wgmma_up(expert_weights_up)
    ...
    mod.monomoe_topk(
        activations_in,
        router_logits,
        expert_weights_up,
        expert_scales_up,
        expert_weights_down,
        expert_scales_down,
        out,
        scratchpad,
        int(top_k),
        int(sf_map[scoring_func]),
        bool(renormalize),
    )
```

## Relevance

The entry point takes **router logits**, not routed expert ids — routing lives inside the
kernel, so the host does no top-k and there is no permutation array on the interface at all.
That is the API shape our megakernel should present. The `scratchpad` argument being optional
but reusable across launches is the graph-replay affordance. The fixed-shape specialization
(`must be the fixed E=256/N=512/K=2048 shape the kernel is specialized for`) is the precedent
for our brief's instruction that generalization is not a goal.

---

## Gaps

What was looked for and not found.

- **No published per-call cost for `GroupedMatmulSwigluQuantV2` at batch=1 on A3.** The closest
  numbers found are: W4A8 `GroupedMatmul` at 65.9 us mean / 61.6 us median per call on A3 TP16
  decode (PR #16606), and the A3 MSD GMSQ at 167-284 us per call for 260-800 valid rows
  (PR #17201) — both far above batch=1 row counts, so neither pins the fixed cost at M=8.
- **The A3 (`arch22`) `grouped_matmul` and `grouped_matmul_swiglu_quant_v2` tiling functions
  were not read.** The MSD pipeline header and the `WorkSpaceSplitConfig` helpers were read, but
  `op_host/*_tiling.cpp` — which is where `mLimit`, `baseM`/`baseN` and the core split are
  actually chosen for A8W4 — was not opened. Without it, the claim that `baseM` is sized for
  throughput at batch=1 rests on the design doc's worked example (`baseM = 24`) rather than on
  the A8W4 code path.
- **No Zhihu or CSDN article found specifically on batch=1 / single-token MoE decode kernels on
  Ascend.** Repeated Chinese-phrased searches returned the English literature (MonoMoE, Cohere,
  MPK) plus Huawei's own op articles, but no community write-up on the batch=1 MoE regime for
  Ascend. The Huawei articles quoted above are the closest, and they address decode generally
  rather than batch=1.
- **No Ascend-side measurement of the MSD double-matmul cost in isolation.** Huawei's article
  gives the end-to-end claim (W4A8+MSD decode is 1.3-1.7x W8A8) but nothing isolates what the
  two INT4 MMADs plus the `C_high*16 + C_low + weightAssistMatrix` vector pass cost relative to
  a single A8W8 MMAD at small M. That number would say whether MSD is worth keeping at batch=1
  or whether expanding W4 to W8 on-chip is cheaper there.
- **`GroupedMatmulFinalizeRouting` has no published A3 performance data**, and no call site for
  it was found in vllm-ascend — so whether the down-GEMM + combine fusion is actually reachable
  from our stack today is unverified.
- **No documentation found for `tuning_config` beyond "expected tokens per expert".** The doc
  says elements after the first are reserved, and that the parameter is only honoured for
  A4W4-NZ; nothing describes what the A4W4 path does with it, which would tell us what template
  switch we would be asking for on A8W4.
- **DeepEP low-latency buffer sizing not captured.** `Buffer.get_low_latency_rdma_size_hint` and
  `num_rdma_bytes` were identified as the relevant API but their values and the V2 buffer-size
  change were not retrieved verbatim. This matters only if we fuse the TP16 collective, which
  MonoMoE's measurement argues against.
- **No source found that measures `CrossCoreSetFlag`/`CrossCoreWaitFlag`/`SyncAll` cost on
  Ascend.** The GPU analogue is quantified (Hazy Research: ~60 ns per CUDA async barrier even in
  the pass state; 40 of 600 us total); without the Ascend number, the barrier budget for a
  78-layer megakernel cannot be estimated.
- **Cohere's megakernel blog (https://cohere.com/blog/megakernels) could not be fetched** from
  this network, so its claims (a 30B MoE decode loop in one persistent kernel, counters in
  global memory instead of launch boundaries) are unverified and not quoted.
- **`MonoMoE`'s GLM-5.2 configuration was not located.** The paper reports GLM-5.2 TPOT results
  but the shipped FlashInfer kernel is specialized to `E256/N512/K2048`; the GLM-5.2
  specialization (which would have N=1536-class and K=6144) was not found in the repo tree.
