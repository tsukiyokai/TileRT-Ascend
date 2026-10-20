# MLA Decode Fusion on Ascend — Source Material

Verbatim research deposit for the GLM-5.2-W4A8C8 / Ascend A3 megakernel effort.
All quoted blocks are reproduced as published. Chinese sources are left in Chinese.
Collected 2026-10-20.

---

## 1. MlaProlog — the one fused op that already works

### `MlaProlog算子设计介绍` — why the op exists, and the CV pipeline pseudo-code

https://gitcode.com/cann/ops-transformer/blob/master/attention/mla_prolog/docs/MlaProlog%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E4%BB%8B%E7%BB%8D.md
Huawei / CANN `ops-transformer`, master branch.

Stated motivation — op-count and launch-overhead reduction is the explicit design goal:

> 在此背景下，DeepSeek团队提出Mla算子需求，需要对算法和硬件进行协同优化，通过算子融合，减少算子调用和头开销时间，在保证模型精度的前提下，降低推理延迟、并减少硬件资源消耗，MlaProlog算子正是Mla算子前处理融合部分。

What it fuses:

> 按照Multi-Head Latent Attention定义的计算流程实现，整体计算流程如下：
>
> 1. 输入序列x经过下采样$W^{DQ}$矩阵进行降秩变换，并进行归一化处理得到$c^Q$
>
> 2. 将$W^{UQ}$和$W^{QR}$矩阵进行拼接，实现对$c^Q$的升秩和映射计算，对运算结果再进行split拆分，分别做Qn的归一化处理和Rope位置编码，最终得到Query和Query Rope
>
> 3. 将$W^{DKV}$和$W^{KR}$矩阵进行拼接，对输入序列x实现降秩和映射计算，对运算结果再进行split拆分，分别做Rope位置编码和$C^{KV}$的归一化处理，最终得到KR Cache和KV Cache

The four matmuls and how each is split across cores:

> MlaProlog算子有多个Matmul运算：
>
> 1、MatmulCq：将BS合轴，作为M轴，HeadSizeCq为N轴，HeadSizeX为K轴，核间对HeadSizeCq按照cube核数进行切分，每个核分配一定的子块
>
> 2、MatmulCkvKr：将BS合轴，作为M轴，HeadSizeCkvKr为N轴，HeadSizeX为K轴，核间对HeadSizeCkvKr进行切分，分为64块
>
> 3、MatmulQcQr：将BS合轴，作为M轴，HeadSizeQcQr为N轴，HeadSizeCq为K轴，核间对HeadSizeQcQr按照cube核数进行切分，每个核分配一定的子块
>
> 4、MatmulQn：将BS合轴，作为M轴，HeadSizeCkv为N轴，dimHeadSizeQc为K轴

The CV (cube/vector) pipeline design and the six sync points:

> ## 流水设计
>
> MlaProlog融合算子包含了Vector计算和Cube计算，Vector侧和Cube侧的计算存在依赖关系，因此需要进行CV流水设计，
> 否则C侧和V侧很有可能是串行流水的效果，不能达到并行计算的目的，无法使得融合算子性能达到最优：
>
> 1、MatmulCq计算的时候，对Hcq进行了分核，单核没有计算一个完整的token，所以在RmsNormCq计算前，
> 需要做AIC与AIV之间的同步控制（SYNC_MMCQ_NOMRCQ）
>
> 2、MatmulCkvKr计算的时候，也是对Hcq进行了分核，单核没有计算一个完整的token，所以在RmsNormCkv计算前，也
> 需要做AIC与AIV之间的同步控制（SYNC_MMCKVKR_NOMRCKV）
>
> 3、由于RmsNormCq按照行切分到不同核上，因此在做MatmulQcQr计算的时候，也需要AIV与AIC的同步控制（SYNC_NOMRCQ_MMQCQR）
>
> 4、由于和RopeQr的分核策略不同，因此RopeQr计算前，也需要AIC和AIV之间的同步控制（SYNC_MMQCQR_ROPEQR）
>
> 5、由于MatmulQcQr和MatmulQn的分核策略不同，MatmulQn依赖于MatmulQcQr的输出，因此需要做cube的全核同步（SYNC_ALL_CUBE）
>
> 6、Vector核运算前，需要做vector的全核同步（SYNC_ALL_VECTOR），确保数据流水搬运

The single-core main loop, verbatim — this is the canonical Ascend intra-kernel
cross-unit dependency mechanism:

```c
// 单核计算伪码
void Process() {
    loops = compute_step_this_core(); //计算当前core的循环次数
    // 
    for (i = 0; i < loops; i++) {
        updateCurrentStepSize(i, stepSize, allTokenSize);  //刷新当前轮的计算的M轴大小
  
        if ASCEND_IS_AIC {
      
            MatmulCq(tokenXOffset, weightDqOffset, cqResOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MMCQ_NORMSCQ_FLG);      //cube与vector同步

            MatmulCkvKr(tokenXOffset, weightDkvKrOffset, ckvKrResOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MMCKVKR_NORMROPE_FLG);   //cube与vector同步

            CrossCoreWaitFlag(SYNC_MMCQ_NORMSCQ_FLG);                    // MatmulQcQr依赖RmsNormCq的输出，需要插入CV核间同步
      
            MatmulQcQr(weightUqQrOffset, qcQrResOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MMQCQR_ROPEQR_FLG);  //cube与vector同步


            // 由于MatmulQn和MatmulQcQr的分核策略不一样，MatmulQn又依赖MatmulQcQr的输出
            // 需要等所有cube核上的MatmulQcQr执行完后才能启动MatmulQn
            CrossCoreSetFlag<0x0, PIPE_FIX>(SYNC_ALL_CUBE_FLG);
            CrossCoreWaitFlag(SYNC_ALL_CUBE_FLG);
      
            MatmulQn(qcOffset, weightUkOffset, qnResOffset, mmQnLoopTime);  // MatmulQn的结果直接输出到queryOut, qnOffset需要按Batch轴偏移

      }
    
       if ASCEND_IS_AIV {
      
            GetSinCos(tokenIndex);

      CrossCoreWaitFlag(SYNC_MMCQ_NORMSCQ_FLG);        // wait MatmulCq
      
      CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_ALL_VECTOR_FLG);    
      CrossCoreWaitFlag(SYNC_ALL_VECTOR_FLG);
      
            RmsNormCq(rmsNormCqOffset);
      
      // 由于RmsNormCq和MatmulQcQr的分核策略不一样，需要等所有vector上的RmsNormCq执行完成后才能启动MatmulQcQr
      // 需要所有vector核上的RmsNormCq执行完成后，才发起MatmulQcQr的执行
      CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_ALL_VECTOR_FLG);
      CrossCoreWaitFlag(SYNC_ALL_VECTOR_FLG);

      CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_MMCQ_NORMSCQ_FLG);    // 保障MatmulQcQr等RmsNormCq


      CrossCoreWaitFlag(SYNC_MMCKVKR_NORMROPE_FLG);           // wait MatmulCkvKr

      CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_ALL_VECTOR_FLG);    
      CrossCoreWaitFlag(SYNC_ALL_VECTOR_FLG);
      RmsNormRopeScatterCkvKr(tokenIndex, rmsNormCkvOffset, ropeKrOffset);

      CrossCoreWaitFlag(SYNC_MMQCQR_ROPEQR_FLG);        // wait MatmulQcQr

      CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_ALL_VECTOR_FLG);    
      CrossCoreWaitFlag(SYNC_ALL_VECTOR_FLG);
      RopeQr(ropeQrOffset, ropeQrResOffset);     
        }
    }
}
```

The complete formula set:

> $$
> q^N = RmsNorm(x \cdot W^{DQ}) \cdot W^{UQ} \cdot W^{UK} \tag{a}
> $$
>
> $$
> q^R = ROPE(RmsNorm(x \cdot W^{DQ}) \cdot W^{QR}) \tag{b}
> $$
>
> $$
> k^R = ROPE(x \cdot W^{KR}) \tag{c}
> $$
>
> $$
> c^{KV} = RmsNorm(x \cdot W^{DKV}) \tag{d}
> $$

The weight-concatenation trick (two GEMMs folded into one by horizontal concat):

> 利用矩阵乘法的性质，$W^{DKV}$和$W^{KR}$矩阵可以横向拼接成一个矩阵$[W^{DKV}|W^{KR}]$来计算，该拼接矩阵对应接口文档的$weightDkvKr$参数。
> $$
> c^{KV}k^R = x \cdot [W^{DKV}|W^{KR}] = [x \cdot W^{DKV}|x \cdot W^{KR}] \tag{7}
> $$

> 同样利用矩阵乘法的性质，$W^{UQ}$和$W^{QR}$矩阵可以横向拼接成一个矩阵$[W^{UQ}|W^{QR}]$来计算，该拼接矩阵对应接口文档的$weightDkvKr$参数。
> $$
> q^Cq^R = c_{norm}^Q \cdot [W^{UQ}|W^{QR}] = [c_{norm}^Q \cdot W^{UQ}|c_{norm}^Q \cdot W^{QR}] \tag{11}
> $$

`MatmulQn` is the absorb step — `q^C · W^UK` — done inside the prolog:

> ### MatmulQn
>
> 对$q^C$矩阵乘上Key的上采样矩阵$W^{UK}$得到最终的Query矩阵$q^N$。
> $$
> q^N = q^C \cdot W^{UK} \tag{16}
> $$

RoPE is implemented in the even/odd-split form so cos/sin are contiguous:

> 为了节省$cos$/$sin$的存储空间并简化内部的实现逻辑，考虑$q^R*k^R$矩阵乘最终要进行Reduce操作：同时调整行和列中元素的位置，不影响最后的累加结果。最终内部$ROPE$实现的计算调整成公式（15）的形式，对输入$x$按奇偶位置拆分成两部分，对应的$cos$和$sin$部分的输入是连续的。

KV cache NZ stride rule:

> - 当前按NZ格式存储KVCache时，是将整个Block存储成NZ格式；小块内是以行为主(Row Major)的排布，形状如Z字型；块与块之间是以列为主的排布，形状如N字形。将完整的H(Hidden States)按32B长度的小块进行分块后，在ND格式下连续存储的小块，在NZ格式下需要跳$32B * BlockSize$存储，即需要设置stride为$32B * BlockSize$。

## Relevance

This is our 2.950 ms vs 2.745 ms op — the only one at the bandwidth bound. The pseudo-code
explains why it wins: four GEMMs, two RmsNorms, two RoPEs and the cache scatter live in a
single kernel launch, coupled only by `CrossCoreSetFlag`/`CrossCoreWaitFlag` and two grid-wide
barriers. Those two primitives plus `ASCEND_IS_AIC`/`ASCEND_IS_AIV` are the complete mechanism
set we need to replicate for the rest of the layer. Note also that `MatmulQn` already performs
`q^C · W^UK` in-kernel, so the absorb multiply is already inside the prolog's launch, not a
separate op.

---

### MlaPrologV3 — TilingKey bit layout, including `SPLIT_M_MODE` and `CV_MODE`

https://gitcode.com/cann/ops-transformer/blob/master/attention/mla_prolog_v3/docs/MlaPrologV3%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E4%BB%8B%E7%BB%8D.md

> ## TilingKey划分
>
> TilingKey为uint64类型，每个模板参数对应TilingKey中的一到数个二进制位，具体实现如下：
>
> |二进制位|变量名|说明|参数列表|
> |-------|------|----|-------|
> |0-3|CACHE_MODE|KVCache的存储格式|0-ND（BSND/TND），1-PA_BSND，2-PA_NZ，3-PA_BLK_BSND，4-PA_BLK_NZ|
> |4-5|SCENARIO|输入场景|0-FP16(预留)，1-BF16，2-量化场景|
> |6-11|QUANT_MODE|量化场景|0-非量化，1-MMQcQr量化，2-MMQcQr量化+KVcache量化，3-MMcqCkvKr量化+MMQcQr量化，4-MMCqCkvkr量化+MMQcQr量化+KVcache量化，5-MMQcQr量化+KVcache pertoken-pergroup量化，6-MMCqCkvkr量化+MMQcQr量化+KVcache pertoken-pergroup量化，7-Mxfp8量化+MMCqCkvkr量化+MMQcQr量化，8-Mxfp8量化+MMCqCkvkr量化+MMQcQr量化+KVcache量化，9-Mxfp8量化+MMCqCkvkr量化+MMQcQr量化+KVcache pertoken-pergroup量化，10-fp8量化+MMcqCkvKr量化+MMQcQr量化，11-fp8量化+MMCqCkvkr量化+MMQcQr量化+KVcache量化，12-hif8量化+MMcqCkvKr量化+MMQcQr量化，13-hif8量化+MMCqCkvkr量化+MMQcQr量化+KVcache量化，14-fp8量化+MMcqCkvKr量化+MMQcQr量化+KVcache pertoken-pergroup量化，15-hif8量化+MMCqCkvkr量化+MMQcQr量化+KVcache pertoken-pergroup量化|
> |12|ENABLE_DEQUANT_OPTIONAL|反量化开启，不能与ENABLE_GROUP_COMPUTE_OPTIONAL一同使用|0-关闭，1-开启|
> |13|ENABLE_GROUP_COMPUTE_OPTIONAL|量化的算力分组优化，不能与ENABLE_DEQUANT_OPTIONAL一同使用|0-关闭，1-开启|
> |14-15|EMPTY_TENSOR_MODE|空tensor场景，用于输入tensor维度为0的情况|0-无空tensor，1-kv_cache/kr_cache为空，2-query为空且不更新cache|
> |16-17|ACTUAL_SEQ_LEN_MODE|actualSeqLen开启场景|0-关闭，1-开启actualSeqLen|
> |18-19|SPLIT_M_MODE|切M模式|0-关闭(切N)，1-开启(切M)|
> |20|ENABLE_ROPE|RoPE开关，仅V4可选关闭（doRope=false时直通写出），V1/V2/V3仅编译开启|0-关闭(passthrough)，1-开启|
> |21-29|CV_MODE|CV分核模式|ASCENDC_TPL_MIX_AIC_1_1(6)：1:1模式，ASCENDC_TPL_MIX_AIC_1_2(7)：1:2模式|

V3's additional outputs beyond V1:

> 4. 在输出Query量化的情况下，会对Query做Rowmax动态量化，最终得到量化后的Query和对应量化参数DequantScaleQNope
>
> 5. 在queryNormFlag=true的情况下，输入$x$乘以$W^{DQ}$进行下采样和RmsNorm后得到压缩结果$c^Q$，作为queryNorm输出
>
> 6. 在queryNormFlag=true且量化场景（weightQuantMode≠0）的情况下，会对$c^Q$做动态量化，最终得到量化后的queryNorm和对应量化参数dequantScaleQNorm

Tiling criteria the op itself states:

> a. 将核心的数量用满，防止部分核闲置。
>
> b. 每一个核心被分配的计算量相对均匀，避免出现某些核计算的数据量过大，其余核空闲的情况。
>
> c. AIC和AIV之间处理的数据量要符合其对应的算力，避免AIC或AIV出现长时间的空闲。

## Relevance

`SPLIT_M_MODE` (bit 18-19) is the directly actionable knob at batch=1: with BS=1 there is no M
to split, so mode 0 (切N) is the only sane setting and any tiling that assumes 切M wastes cores.
`CV_MODE` exposes the AIC:AIV ratio as a compile-time template choice (1:1 vs 1:2) — relevant
because our per-call floors differ by unit (AI_VECTOR_CORE 0.70us vs AI_CORE 2.30us, MIX_AIV
12.04us vs MIX_AIC 13.58us), so which unit owns which stage changes the fixed cost.
`ENABLE_ROPE` passthrough (V4 only) and `EMPTY_TENSOR_MODE` are the hooks for stripping work we
do not need.

---

### aclnnMlaProlog — original signature, and its scheduled deprecation

https://gitcode.com/cann/ops-transformer/blob/master/attention/mla_prolog/op_api/aclnn_mla_prolog.h

```cpp
#ifndef ACLNN_MLA_PROLOG
#define ACLNN_MLA_PROLOG
#warning "aclnn_mla_prolog.h is scheduled to be deprecated in December 2026, " \
    "and will be replaced by the aclnn_mla_prolog_v3_weight_nz.h. " \
    "We apologize for any inconvenience caused " \
    "and appreciate your timely migration to the new interface. "
```

```cpp
__attribute__((visibility("default"))) aclnnStatus
aclnnMlaPrologGetWorkspaceSize(const aclTensor* tokenX, const aclTensor* weightDq, const aclTensor* weightUqQr,
                               const aclTensor* weightUk, const aclTensor* weightDkvKr, const aclTensor* rmsnormGammaCq,
                               const aclTensor* rmsnormGammaCkv, const aclTensor* ropeSin, const aclTensor* ropeCos,
                               const aclTensor* cacheIndex, aclTensor* kvCacheRef, aclTensor* krCacheRef,
                               const aclTensor* dequantScaleXOptional, const aclTensor* dequantScaleWDqOptional,
                               const aclTensor* dequantScaleWUqQrOptional, const aclTensor* dequantScaleWDkvKrOptional,
                               const aclTensor* quantScaleCkvOptional, const aclTensor* quantScaleCkrOptional,
                               const aclTensor* smoothScalesCqOptional, double rmsnormEpsilonCq,
                               double rmsnormEpsilonCkv, char* cacheModeOptional, const aclTensor* queryOut,
                               const aclTensor* queryRopeOut, uint64_t* workspaceSize, aclOpExecutor** executor);
```

## Relevance

Confirms the exact tensor set the fused prolog consumes: the four weight matrices
(`weightDq`, `weightUqQr`, `weightUk`, `weightDkvKr`), both RmsNorm gammas, sin/cos,
`cacheIndex`, and in-place `kvCacheRef`/`krCacheRef`. Any megakernel that subsumes the prolog
must carry exactly this set, and the `weight_nz` successor says the weights want FRACTAL_NZ.

---

### aclnnMlaPrologV3WeightNz — the current (non-deprecated) signature

https://gitcode.com/cann/ops-transformer/blob/master/attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v3_weight_nz.h

```cpp
__attribute__((visibility("default"))) aclnnStatus aclnnMlaPrologV3WeightNzGetWorkspaceSize(
    const aclTensor* tokenX, const aclTensor* weightDq, const aclTensor* weightUqQr, const aclTensor* weightUk,
    const aclTensor* weightDkvKr, const aclTensor* rmsnormGammaCq, const aclTensor* rmsnormGammaCkv,
    const aclTensor* ropeSin, const aclTensor* ropeCos, aclTensor* kvCacheRef, aclTensor* krCacheRef,
    const aclTensor* cacheIndexOptional, const aclTensor* dequantScaleXOptional,
    const aclTensor* dequantScaleWDqOptional, const aclTensor* dequantScaleWUqQrOptional,
    const aclTensor* dequantScaleWDkvKrOptional, const aclTensor* quantScaleCkvOptional,
    const aclTensor* quantScaleCkrOptional, const aclTensor* smoothScalesCqOptional,
    const aclTensor* actualSeqLenOptional, const aclTensor* kNopeClipAlphaOptional, double rmsnormEpsilonCq,
    double rmsnormEpsilonCkv, char* cacheModeOptional, int64_t weightQuantMode, int64_t kvCacheQuantMode,
    int64_t queryQuantMode, int64_t ckvkrRepoMode, int64_t quantScaleRepoMode, int64_t tileSize, double qcQrScale,
    double kcScale, const aclTensor* queryOut, const aclTensor* queryRopeOut,
    const aclTensor* dequantScaleQNopeOutOptional, const aclTensor* queryNormOutOptional,
    const aclTensor* dequantScaleQNormOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor);
```

---

### aclnnMlaPrologV4WeightNz — full parameter documentation (quant modes, cache modes, `doRope`)

https://gitcode.com/cann/ops-transformer/blob/master/attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v4_weight_nz.h

```cpp
/**
 * @brief 计算MlaPrologV4WeightNz所需的workspace大小并创建算子执行器。
 *
 * @param tokenX [IN] 公式中用于计算Query和Key的输入tensor。
 * @param weightDq [IN] 公式中用于计算Query的下采样权重矩阵。
 * @param weightUqQr [IN] 公式中用于计算Query的上采样权重矩阵和位置编码权重矩阵。
 * @param weightUk [IN] 公式中用于计算Key的上采样权重。
 * @param weightDkvKr [IN] 公式中用于计算Key的下采样权重矩阵和位置编码权重矩阵。
 * @param rmsnormGammaCq [IN] 计算c^Q的RmsNorm公式中的γ参数。
 * @param rmsnormGammaCkv [IN] 计算c^KV的RmsNorm公式中的γ参数。
 * @param ropeSin [IN]
 * 用于计算旋转位置编码的正弦参数矩阵。doRope=true时必选，doRope=false时必须与ropeCos同时为空（nullptr）。
 * @param ropeCos [IN]
 * 用于计算旋转位置编码的余弦参数矩阵。doRope=true时必选，doRope=false时必须与ropeSin同时为空（nullptr）。
 * @param kvCacheRef [IN/OUT] 用于cache索引的aclTensor，计算结果原地更新。
 * @param krCacheRef [IN/OUT] 用于key位置编码的cache，计算结果原地更新。
 * @param cacheIndexOptional [IN] 用于存储kvCache和krCache的索引。
 * @param dequantScaleXOptional [IN] tokenX的反量化参数。
 * @param dequantScaleWDqOptional [IN] weightDq的反量化参数。
 * @param dequantScaleWUqQrOptional [IN] 用于MatmulQcQr矩阵乘后反量化操作的perchannel参数。
 * @param dequantScaleWDkvKrOptional [IN] weightDkvKr的反量化参数。
 * @param quantScaleCkvOptional [IN] 用于对kvCache输出数据做量化操作的参数。
 * @param quantScaleCkrOptional [IN] 用于对krCache输出数据做量化操作的参数。
 * @param smoothScalesCqOptional [IN] 用于对RmsNormCq输出做动态量化操作的参数。
 * @param actualSeqLenOptional [IN] 表示每个batch中的序列长度，以前缀和的形式储存。
 * @param kNopeClipAlphaOptional [IN] 表示对kvCache做clip操作时的缩放因子。
 * @param rmsnormEpsilonCq [IN] 计算c^Q的RmsNorm公式中的ε参数。
 * @param rmsnormEpsilonCkv [IN] 计算c^KV的RmsNorm公式中的ε参数。
 * @param cacheModeOptional [IN]
 * 表示kvCache的模式，可选值为"PA_BSND"、"PA_NZ"、"PA_BLK_BSND"、"PA_BLK_NZ"、"BSND"、"TND"。
 * @param weightQuantMode [IN] 表示weightDq、weightUqQr、weightUk、weightDkvKr的量化模式，0表示非量化。
 * @param kvCacheQuantMode [IN]
 * 表示kvCache的量化模式，0表示非量化，1表示pertensor量化，2表示perchannel量化，3表示pertoken-pergroup量化。
 * @param queryQuantMode [IN] 表示query的量化模式，0表示非量化，1表示per-token-head量化。
 * @param ckvkrRepoMode [IN] 表示kvCache和krCache的存储模式，0表示分别存储，1表示合并存储。
 * @param quantScaleRepoMode [IN]
 * 表示量化scale的存储模式，0表示量化scale和数据分别存储，1表示合并存储作为kvCacheRef输出。
 * @param tileSize [IN] 表示pertoken-pergroup量化时每个tile的大小，仅在kvCacheQuantMode为3时有效，默认值为128。
 * @param qcQrScale [IN] 表示Query的尺度矫正系数，用户不特意指定时需要传入1.0。
 * @param kcScale [IN] 表示Key的尺度矫正系数，用户不特意指定时需要传入1.0。
 * @param doRope [IN]
 * 表示是否对queryRopeOut与krCache执行旋转位置编码（RoPE）。默认true：执行RoPE，此时ropeSin/ropeCos必须为非空有效Tensor；false：跳过RoPE并写直通结果，此时ropeSin/ropeCos必须同时为空（nullptr）。
 * @param queryOut [OUT] 公式中Query的输出tensor。
 * @param queryRopeOut [OUT] 公式中Query位置编码的输出tensor。doRope=false时输出直通结果，shape不受影响。
 * @param dequantScaleQNopeOutOptional [OUT] 公式中Query输出的反量化参数。
 * @param queryNormOutOptional [OUT] 公式中tokenX做rmsNorm后的输出tensor。
 * @param dequantScaleQNormOutOptional [OUT] queryNormOutOptional的反量化参数。
 * @param workspaceSize [OUT] 返回需在Device侧申请的workspace大小，单位为字节。
 * @param executor [OUT] 返回包含算子执行流程的执行器。
 * @return aclnnStatus 成功时返回ACLNN_SUCCESS，否则返回对应错误码。
 * @domain aclnn_ops_infer
 */
```

## Relevance

`ckvkrRepoMode=1` (merged kv/kr storage) and `quantScaleRepoMode=1` (scale merged into
`kvCacheRef`) collapse what would otherwise be separate cache tensors and separate scale
writes — fewer discrete writes per decode step, which is the right direction given that our C8
KV cache path is latency- not bandwidth-limited. `queryQuantMode=1` (per-token-head) produces
a quantized query plus `dequantScaleQNopeOut` directly, so the prolog can hand an already
quantized Q to the attention stage without an intervening Cast (we currently spend 2.300 ms on
Cast against 0.076 ms of bytes).

---

## 2. SparseFlashAttention and the selection stage

### aclnnSparseFlashAttention — signature, `attentionMode=2` is MLA-absorb, and the official statement of its weakness

https://gitcode.com/cann/ops-transformer/blob/master/attention/sparse_flash_attention/docs/aclnnSparseFlashAttention.md

> ## 功能说明
>
> - 接口功能：sparse_flash_attention（SFA）是针对大序列长度推理场景的高效注意力计算模块，该模块通过"只计算关键部分"大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。
>
> - 计算公式：
>
> $$
> \text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
> $$
>
> 其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`lightning_indexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度。

```cpp
aclnnStatus aclnnSparseFlashAttentionGetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *value,
    const aclTensor     *sparseIndices,
    const aclTensor     *blockTableOptional,
    const aclTensor     *actualSeqLengthsQueryOptional,
    const aclTensor     *actualSeqLengthsKvOptional,
    const aclTensor     *queryRopeOptional,
    const aclTensor     *keyRopeOptional,
    double              scaleValue,
    int64_t             sparseBlockSizeOptional,
    char                *layoutQueryOptional,
    char                *layoutKvOptional,
    int64_t             sparseMode,
    int64_t             preTokens,
    int64_t             nextTokens,
    int64_t             attentionMode,
    bool                returnSoftmaxLse,
    const aclTensor     *attentionOut,
    const aclTensor     *softmaxMax,
    const aclTensor     *softmaxSum,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

Parameter semantics, extracted from the doc's table:

```
sparseIndices（aclTensor）  输入   离散取kvCache的索引。
      不支持空tensor。
      sparseSize为一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足sparseSize大于0。
   INT32  ND
      layoutQueryOptional为BSND时，shape为(B, S1, N2, sparseSize)。
      layoutQueryOptional为TND时，shape为(T1, N2, sparseSize)。

sparseBlockSizeOptional（int64_t）  输入  代表sparse阶段的block大小。
      sparseBlockSizeOptional为1时，为Token-wise稀疏化场景，将每个token视为独立单元，在计算重要性分数时，评估每个查询token与每个键值token之间的独立关联程度。
      sparseBlockSizeOptional为大于1小于等于128时，为Block-wise稀疏化场景，将token序列划分为固定大小的连续块，以块为单位进行重要性评估，块内token共享相同的稀疏化决策。

sparseMode（int64_t）
      sparseMode为0时，代表全部计算。
      sparseMode为3时，代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。

attentionMode（int64_t）  输入   仅支持传入2，表示MLA-absorb模式。
```

Constraints:

> ## 约束说明
>
> - 确定性计算：aclnnSparseFlashAttention默认确定性实现。
> - 该接口支持推理场景下使用。
> - N2仅支持1。
> - blockSize为一个block的token数，blockSize取值为16的倍数，且最大支持1024。
> - 参数query中的D和key、value的D值相等为512。
> - queryRopeOptional与keyRopeOptional须成对为空指针或成对传入非空tensor：均为空指针时表示不使用rope，仅对512维nope部分做注意力计算；均传入非空tensor时Dr相等为64。不支持空tensor。
> - 参数query、key、value的数据类型必须保持一致。
> - Ascend 950PR&950DT系列产品：
>   - N1支持1~128。
>   - 仅支持sparseBlockSizeOptional为1。
>   - returnSoftmaxLse为True时，支持layoutKvOptional为PA_BSND。
>   - 当layoutKvOptional为PA_BBND时，key、value和keyRopeOptional支持0轴非连续，非连续Tensor的每一维的stride均不能为0。不支持其他非连续。
> - Atlas A3系列产品、Atlas A2系列产品：
>   - N1支持1/2/4/8/16/32/64/128。
>   - sparseBlockSizeOptional支持[1,128]，且要求是2的幂次方，在PageAttention场景下要求sparseBlockSizeOptional整除blockSize。
>   - key、value和keyRopeOptional不支持非连续。
>   - returnSoftmaxLse为True时，layoutKvOptional不支持PA_BSND。

A3 call-site example from the doc, with the concrete shapes and parameter values:

```cpp
    std::vector<int64_t> queryShape = {1, 2, 1, 512};
    std::vector<int64_t> keyShape = {1, 2, 1, 512};
    std::vector<int64_t> valueShape = {1, 2, 1, 512};
    std::vector<int64_t> sparseIndicesShape = {1, 2, 1, 2};
    std::vector<int64_t> attentionOutShape = {1, 2, 1, 512};
    ...
    int64_t sparseBlockSize = 64;
    ...
    int64_t sparseMode = 3;
    ...
    int64_t attentionMode = 2;
```

## Relevance

Huawei's own one-line summary of this op is the explanation of our 6.973 ms against 1.382 ms of
bytes: `通过"只计算关键部分"大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加`
— sparsity cuts the FLOPs and replaces them with scattered gathers. `attentionMode` is locked
to `2` = MLA-absorb, and `D=512`, `Dr=64`, `N2=1` confirm the op is written against the
absorbed latent layout, so the absorb trick is already assumed, not an optimization left on
the table. `sparseIndices` must be pre-partitioned (`每行有效值均在前半部分`), i.e. the op will
not compact for us — the indexer's output layout is part of the contract.

---

### aclnnLightningIndexer — the selection stage, exact formula and signature

https://gitcode.com/cann/ops-transformer/blob/master/attention/lightning_indexer/docs/aclnnLightningIndexer.md

> ## 功能说明
>
> - 接口功能：`lightning_indexer`基于一系列操作得到每一个token对应的Top-$k$个位置。
>
> - 计算公式：
>
> $$
> Indices=\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
> $$
>
> 对于某个token对应的Index Query $Q_{index}\in\R^{g\times d}$，给定上下文Index Key $K_{index}\in\R^{S_{k}\times d},W\in\R^{g\times 1}$，其中$g$为GQA对应的group size，$d$为每一个头的维度，$S_{k}$是上下文的长度。

```cpp
aclnnStatus aclnnLightningIndexerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    char            *layoutQueryOptional,
    char            *layoutKeyOptional,
    int64_t          sparseCount,
    int64_t          sparseMode,
    int64_t          preTokens,
    int64_t          nextTokens,
    bool             returnValues,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

## Relevance

This is the `index_topk 2048` stage. The formula is cheap in FLOPs — one `Q_index @ K_index^T`
of head dim `d`, a ReLU, a weighted sum over `g`, then a Top-k — but the Top-k over `S_k`
(8K context) is a reduction that does not shrink with batch=1, and its output
`sparseIndicesOut` is the input `sparseIndices` of SFA. Fusing indexer and SFA into one kernel
removes one launch and one full round-trip of the index array through HBM.

---

### The CANN `attention` operator inventory (what exists to fuse against)

https://gitcode.com/cann/ops-transformer/tree/master/attention
Directory listing, `cann/ops-transformer` master, retrieved 2026-10-20.

```
attention_update                       lightning_indexer
block_sparse_attention                 lightning_indexer_v2
bsa_select_block_mask                  lightning_indexer_v2_metadata
compressor                             mixed_quant_flash_attn
dense_lightning_indexer_softmax_lse    mixed_quant_sparse_flash_mla
flash_attention_score                  mla_preprocess
flash_attn                             mla_preprocess_v2
flash_attn_metadata                    mla_prolog
flash_mla                              mla_prolog_v2
flash_mla_metadata                     mla_prolog_v3
fused_infer_attention_score            nsa_compress
gather_pa_kv_cache                     nsa_compress_attention
incre_flash_attention                  nsa_compress_attention_infer
indexer_quant_cache                    nsa_compress_with_cache
key_pool                               nsa_selected_attention
kv_compress_epilog                     nsa_selected_attention_infer
kv_quant_sparse_flash_attention        pool_key_indexer
kv_quant_sparse_flash_attention_v2     prompt_flash_attention
quant_flash_mla                        quant_lightning_indexer
quant_sparse_flash_mla                 scatter_pa_cache
sparse_flash_attention                 scatter_pa_kv_cache
sparse_flash_mla                       scatter_pa_kv_cache_with_k_scale
sparse_flash_mla_softmax_l1_norm       swin_attention_score_quant
```

## Relevance

Two entries matter for us and were not on our radar: `kv_quant_sparse_flash_attention` /
`_v2` (a C8-KV-cache-aware SFA, i.e. the quantized-cache variant of the op we run) and
`kv_compress_epilog`. `quant_lightning_indexer` is the quantized indexer. If any of these
already fuse the dequant of the C8 cache into the attention load, they remove part of our
2.300 ms Cast.

---
### CANN `flash_mla` / `flash_mla_metadata` — a separate metadata kernel that pre-computes the AICore/AIVCore split

https://gitcode.com/cann/ops-transformer/blob/master/attention/flash_mla/docs/torchapi_flash_mla.md

> ## 功能说明
>
> - **接口功能**：
>
>   `flash_mla`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于非量化 MLA 注意力计算。输入`q`和`k_cache`已合并512维nope与64维rope，算子不执行KV压缩或RoPE变换。
>
>   `k_cache`同时提供Key和Value：Key使用全部576维，Value使用前512维，不单独传入`v`、`q_rope`或`k_rope`。
>
>   `flash_mla_metadata`是主算子的元数据生成接口，记录AICore/AIVCore任务切分结果。典型调用流程如下：
>
>   1. 准备Query、分页KV缓存、块表和序列长度。
>   2. 调用`flash_mla_metadata`生成`metadata`。
>   3. 将`metadata`和对应输入传入`flash_mla`。

```python
cann_ops_transformer.flash_mla_metadata(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    cu_seqlens_q=None,
    seqused_q=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=None,
    layout_q="BSND",
) -> Tensor
```

```python
cann_ops_transformer.flash_mla(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
) -> (Tensor, Tensor)
```

> | num_heads_q | int | 必选 | Query头数，配套当前主算子时支持64、96 | int32 | - | - |
> | num_heads_kv | int | 必选 | KV头数，配套当前主算子时为1 | int32 | - | - |

Product support (note: A3 is **not** supported):

> - <term>Ascend 950PR&950DT系列产品</term>：支持
> - <term>Atlas A3系列产品</term>：不支持
> - <term>Atlas A2系列产品</term>：不支持

## Relevance

The `metadata` pattern is directly transplantable: a cheap precursor kernel computes the
AICore/AIVCore work split once and hands it to the heavy kernel as a tensor, so the heavy
kernel does no scheduling scalar work. That is the opposite of what
`GroupedMatmulSwigluQuantV2` does (it resolves its own group layout with scalar GM reads in
kernel). For a batch=1 megakernel the equivalent is to bake the whole decode-round schedule
into one metadata/instruction tensor built once per shape, not per step. Note `flash_mla`
itself is 950-only, so on A3 we are on the `sparse_flash_attention` path.

---

## 3. MLA absorb — the canonical statement of the math and its arithmetic-intensity trade

### vLLM `mla_attention.py` — the full absorb derivation, both paths, with dims

https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py
vLLM project, main branch (file retrieved 2026-10-20).

```python
"""# MLA Common Components

This file implements common components for MLA implementations.

First we define:

Sq      as Q sequence length
Skv     as KV sequence length

MLA has two possible ways of computing, a data-movement friendly approach and a
compute friendly approach. We generally want to use the compute friendly
approach for "prefill" (i.e. the ratio Sq / Skv is relatively large, often near
1) and the data-movement friendly approach for "decode" (i.e. the ratio
Sq / Skv is small, often near 0).

NOTE what we deem small and large is currently determined by if it is labelled
prefill or decode by the scheduler, but this is something we should probably
tune.

Main reference: DeepseekV2 paper, and FlashInfer Implementation
(https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

Deepseek's MLA attention works the following way:
* Use a single latent vector to represent the per-token entry of the KV cache.
* For decode (i.e. the memory friendly approach) the attention "simulates" a
multi-head attention, while the compute is similar to multi-query attention.

Below is an example of both paths assuming batch size = 1

## More Extent Definitions:

C           Context length, `Skv - Sq`
H           hidden size
N           number of attention heads
Lq          latent dimension for Q              1536 in DSV3
Lkv         latent dimension for K/V            512 in DSV3
P           nope dimension, no rope.            128 in DSV3
R           rope dimension, goes through rope.  64 in DSV3
V           V head dim.                         128 in DSV3

## Vector/Matrix Definitions

h_t         hidden states (input to attention)  shape [Sq, H]
q_c         latent/compressed Q                 shape [Sq, Lq]
q_nope      uncompressed Q (no-rope)            shape [Sq, N, P]
q_pe        uncompressed Q (rope)               shape [Sq, N, R]
kv_c        latent/compressed KV                shape [Skv, Lkv]
k_pe        decoupled k position embeddings     shape [Skv, R]
new_kv_c    new kv_c from current iter          shape [Sq, Lkv]
new_k_pe    new k_pe from current iter          shape [Sq, R]
cache_kv_c  cached k_c from previous iters      shape [C, Lkv]
cache_k_pe  cached k_pe from previous iters     shape [C, R]
W_DQ        project h_t to q_c                  shape [H, Lq]
W_UQ        project q_c to q_nope               shape [Lq, N * P]
W_QR        project q_c to q_pe                 shape [Lq, N * R]
W_DKV       project h_t to kv_c                 shape [H, Lkv]
W_UK        project kv_c to k_nope              shape [Lkv, N, P]
W_KR        project h_t to k_pe                 shape [H, R]
W_UV        project kv_c to v                   shape [Lkv, N, V]
W_O         project v to h_t                    shape [N * V, H]


## Compute Friendly Approach (i.e. "forward_mha"):

q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(Sq, N, P)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)
k_nope   = (kv_c @ W_UK.view(Lkv, N * P)).view(Skv, N, P)
v        = (kv_c @ W_UV.view(Lkv, N * V)).view(Skv, N, V)

// MHA with QK headdim = P + R
//           V headdim = V
//      sdpa_o shape [Sq, N, V]
sdpa_o = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    v
)
return sdpa_o @ W_O

NOTE: in the actual code,
    `kv_b_proj` is [W_UK; W_UV] concatenated per head
    `q_b_proj` is [W_UQ; W_QR] concatenated per head
    `out_proj` is W_O


## Data-Movement Friendly Approach (i.e. "forward_mqa"):

Runtime
q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(-1, N, P)
ql_nope  = einsum("snh,lnh->snl", q_nope, W_UK)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)

// MQA with QK headdim = Lkv + R
//           V headdim = Lkv
//      sdpa_o shape [Sq, N, Lkv]
// NOTE: this is less compute-friendly since Lkv > P
//       but is more data-movement friendly since its MQA vs MHA
sdpa_o = scaled_dot_product_attention(
    torch.cat([ql_nope, q_pe], dim=-1),
    torch.cat([kv_c, k_pe], dim=-1),
    kv_c
)

o = einsum("snl,lnv->snv", sdpa_o.reshape(-1, N, Lkv), W_UV)
return o.view(-1, N * V) @ W_O


## Chunked Prefill

For chunked prefill we want to use the compute friendly algorithm. We are
assuming sufficiently large Sq / Skv ratio, in the future may want to switch to
the data-movement friendly approach if the chunk (i.e. `Sq`) is small.

However, the compute-friendly approach can potentially run out of memory if Skv
is large due to: `k_nope = (kv_c @ W_UK).view(Skv, N, P)`
```

## Relevance

The absorb is `ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)` on the input side and
`o = einsum("snl,lnv->snv", sdpa_o, W_UV)` on the output side — i.e. it does **not** reduce
work, it moves work from the Skv axis to the Sq axis. At batch=1, Sq=1, so both einsums are
rank-1 against a weight matrix: pure weight streaming, zero reuse, and therefore pure overhead
unless fused into a neighbouring kernel. The comment `less compute-friendly since Lkv > P but
more data-movement friendly since its MQA vs MHA` is the exact statement of why decode uses
absorb and why it does not raise arithmetic intensity at batch=1 — it lowers bytes read from
the cache, not FLOPs per byte of weight.

---

### vLLM DeepSeek V4.1 "mega attention" — one kernel for RoPE + sparse attention + inverse RoPE + FP8 cast, writing into the o_proj input buffer

https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py
vLLM project, main branch (file retrieved 2026-10-20).

```python
"""DeepSeek V4.1 attention on FlashMLA's mega-attention kernel (SM100).

One kernel does Q RoPE, sparse attention, the inverse RoPE of the output and
its FP8 cast, and writes straight into the buffer ``wo_a`` consumes. So the
layer declares ``accepts_unnormed_unroped_query`` -- the kernel RoPEs Q itself,
leaving the fused KV insert only the zero-pad to the kernel's head count -- and
its ``_alloc_attn_out`` / ``_o_proj`` pair speaks QuantizedActivation instead
of bf16, since the output needs no rotation or quantization here.

``wq_b`` rows and ``wo_a`` columns are permuted once at load so the surrounding
GEMMs speak the kernel's chunk-interleaved layouts directly. A step's prefill
and decode segments write disjoint token ranges of one output buffer pair, so
a single ``wo_a`` einsum covers the whole step.
"""
```

Fixed kernel conventions and the quantization group:

```python
_HEAD_DIM_V = 512
_ROPE_DIM = 64
# The kernel always emits per-32 scales, whatever num_per_channels says.
_QUANT_GROUP = 32
# Fixed kernel conventions: GPT-J (non-neox) RoPE, TMA-aligned col-major
# scales, round-to-nearest, packed ue8m0 -- i.e. what DeepGEMM's fp8_einsum
# consumes. Q norm is off: the V4.1 layer norms qr before wq_b.
_ROPE_ARGS = (False, _ROPE_DIM)
_SF_ARGS = (_QUANT_GROUP, True, True, True)
```

The measured mega-kernel vs split-KV table, and the warning about measuring in eager mode:

```python
# Decode always goes through the mega kernel: there is no batch size at which
# falling back to the split-KV decode pays, so there is no minimum-token
# threshold and no fallback path to maintain. What decides whether this layer
# is worth selecting at all is TP, not the batch size.
#
# Measured on one GB300 with benchmarks/kernels/benchmark_dsv41_mega_attn.py,
# CUDA graphs on, counting every per-step op each path runs (64-head model,
# topk_swa=128, topk_extra=512, NVFP4 compressed cache), as mega / split-KV
# microseconds:
#
#     s_q                     1      32     128     256     512
#     TP1 (64 live heads)  29/29   31/31   31/40   53/72   96/140
#     TP2 (32 live)        29/29   31/30   35/35   61/63  110/119
#     TP4 (16 live)        29/29   31/29   34/34   59/58  109/109
#     TP8 (8 live)         29/29   31/29   33/33   59/57  107/104
#
# The kernel absorbs the inverse RoPE and the FP8 cast, whose cost scales with
# *live* heads, while its own cost scales with *padded* heads -- always 64,
# since FlashMLA's FP8 decode takes only h_q in {64, 128}. So it wins at TP1
# (up to 1.45x, and TP1 is also where the fused insert skips the Q pad
# entirely) and from TP2 down the two cancel to within a few percent, with a
# fixed ~2 us penalty at small s_q where the padded-head cost dominates.
# Measure before assuming a win at high TP -- and measure under CUDA graphs:
# in eager mode launch overhead swamps both paths and overstates the fused one
# several-fold.
```

The fused op name and the support gate:

```python
def is_flashmla_mega_attn_supported() -> tuple[bool, str | None]:
    """Whether FlashMLA's fused mega-attention kernels are usable here."""
    is_available, maybe_reason = is_flashmla_sparse_supported()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(100):
        return False, "FlashMLA mega attention requires sm_10x GPUs."
    if not hasattr(torch.ops._flashmla_C, "fused_norm_rope_attn_rope_cast_decode"):
        return False, "vllm._flashmla_C was built without the mega attention ops."
    if not hasattr(torch.ops._flashmla_C, "permute_q_b_proj"):
        return False, "vllm._flashmla_C predates the mega attention weight permutes."
    return True, None
```

The output-buffer allocation, showing the scale layout contract with the consuming GEMM:

```python
def alloc_mega_attn_output(
    num_tokens: int,
    n_wv_group: int,
    device: torch.device,
) -> QuantizedActivation:
    """Allocate the output buffer pair the mega-attention kernels write into.

    One pair per forward step: the prefill and decode segments fill disjoint
    token ranges, so a single ``fp8_einsum`` over all ``num_tokens`` can
    consume the result instead of one call per segment.

    The scale buffer is MN-major -- stride 1 along tokens, head-dim stride
    ``ceil4(num_tokens)`` -- which is both what the kernel requires of a
    caller-provided buffer and what DeepGEMM expects for this N.
    """
```

## Relevance

This is the closest published analogue of what we are building, and it carries two load-bearing
lessons. First, the fusion boundary that paid for them is `norm + rope + attn + inverse-rope +
cast` ending in a buffer the next GEMM reads directly — i.e. the win came from deleting the
epilogue round-trips and the layout conversions (`permute_q_b_proj`, `wo_a` column permute done
once at load), not from merging the attention math. That maps onto our 2.300 ms Cast against
0.076 ms of bytes. Second, the measured table says at `s_q = 1` the mega kernel and the split
path are identical (29/29 us at every TP) — on that hardware batch=1 is already at a floor the
fusion does not touch. Our situation differs because our floors are per-call launch floors
(MIX_AIC 13.58 us, MIX_AIV 12.04 us) rather than kernel-internal, which is precisely what a
megakernel removes; but the warning `in eager mode launch overhead swamps both paths and
overstates the fused one several-fold` means any A/B we run must hold graph capture constant.

---

### GLM-5.x MLA dimensions as configured in vLLM's own attention benchmark

https://github.com/vllm-project/vllm/blob/main/benchmarks/attention_benchmarks/configs/mla_sparse_masked_mha_vs_mqa_glm5.yaml
vLLM project, main branch (file retrieved 2026-10-20).

```yaml
mode: mha_vs_mqa
model:
  name: "glm-5.x"
  num_layers: 78
  num_q_heads: 64
  num_kv_heads: 1
  head_dim: 576
  kv_lora_rank: 512
  qk_nope_head_dim: 192
  qk_rope_head_dim: 64
  v_head_dim: 256
  block_size: 128
  max_model_len: 131072
model_parameter_sweep:
  param_name: "num_q_heads"
  values: [64, 32, 16, 8]
  label_format: "{backend}_{value}h"
backends: ["FLASHMLA_SPARSE", "FLASHINFER_MLA_SPARSE"]
device: "cuda:0"
profile_memory: false
sparse_mla_masked_mha_max_seq_len: 131072
sparse_mla_mha_variants: ["masked_mha", "mqa"]
sparse_mla_topk_pattern: "random"
```

## Relevance

Independent confirmation of the MLA shape we are optimizing: 78 layers, 64 Q heads, 1 KV head,
`kv_lora_rank = 512`, `qk_rope_head_dim = 64`, `head_dim = 576` — which is exactly the
`D = 512` / `Dr = 64` / `N2 = 1` contract `aclnnSparseFlashAttention` enforces. Note
`qk_nope_head_dim = 192` and `v_head_dim = 256`, both larger than DeepSeek's 128/128, so the
unabsorbed `W_UK`/`W_UV` are correspondingly larger and the absorb is more attractive here than
in DSV3. The `num_q_heads` sweep `[64, 32, 16, 8]` is the TP sweep: at our TP16 we have 4 live
Q heads per die.

---
## 4. DeepSeek FlashMLA: the metadata pattern, the arithmetic-intensity threshold, and the fused epilogue

### FlashMLA — API, reusable scheduler metadata, reported throughput

https://github.com/deepseek-ai/FlashMLA
Raw: https://raw.githubusercontent.com/deepseek-ai/FlashMLA/main/flash_mla/flash_mla_interface.py — DeepSeek (Jiashi Li, Shengyu Liu, Yuanhang Sun).

```python
def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = True,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    enable_batch_invariant: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run sparse attention decode forward.

    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim). bfloat16. `head_dim` must be 512 and
                `num_heads_q` must be 64 or 128.
        ...
        tile_scheduler_metadata: FlashMLASchedMeta, returned by get_mla_metadata. You may reuse the same
                `tile_scheduler_metadata` across different invocations, but only when the tensor shapes and the
                values of topk_length and extra_topk_length remain the same. Note that the values are NOT
                checked at runtime: reusing it with different `topk_length` / `extra_topk_length` values
                silently reuses stale split-KV scheduling metadata.
        num_splits: must be None (kept for compatibility with the old interface; the split counts are
                returned inside `tile_scheduler_metadata`).
        ...
        enable_batch_invariant: bool. If True, the split-KV decoding path is disabled so that results do
                not depend on how the batch is partitioned
    """
```

```python
def get_mla_metadata(*args, **kwargs) -> Tuple[FlashMLASchedMeta, None]:
    """
    Returns an empty instance of FlashMLASchedMeta. The actual scheduling metadata will be generated during the first invocation of flash_mla_with_kvcache.
    """
```

README performance:

> It achieves up to 1460 TFlops during prefill and 950 TFlops during decoding on B200 with CUDA 13.3.

Earlier README (commit `ba89a34`):

> The dense MLA decoding kernel achieves up to 3000 GB/s in memory-bound configuration and 660 TFLOPS in computation-bound configuration on H800 SXM5 with CUDA 12.8.

## Relevance

Scheduling metadata computed once and reused across every decoding step and every layer, with
the hazard stated plainly (`stale split-KV scheduling metadata` is silently accepted). That is
the discipline for a replayable megakernel: hoist all split/tile scheduling into a one-time
host computation, keep it in device memory, replay it. Compare with `flash_mla_metadata` in
CANN, which does the same thing for AICore/AIVCore.

### FlashMLA deep dive — the compute/memory ratio formula and the single-accumulator constraint

https://github.com/deepseek-ai/FlashMLA/blob/ba89a34/docs/20250422-new-kernel-deep-dive.md
DeepSeek, 2025-04-22.

> Assume the number of q heads is $h_q$, the number of q tokens per request is $s_q$ (should be 1 if MTP / speculative decoding is disabled), the number of kv tokens per request is $s_k\ (s_k \gg h_q s_q)$, and the head dimensions of K and V are $d_k$ and $d_v$ respectively. The number of FLOPs is roughly $2 (h_q s_q \cdot d_k \cdot s_k + h_q s_q \cdot s_k \cdot d_v) = 2 h_q s_q s_k (d_k+d_v)$, and the memory access volume (in bytes) is $\mathop{\text{sizeof}}(\text{bfloat16}) \times (h_q s_q d_k + s_k d_k + h_q s_q d_v) \approx 2s_k d_k$. Thus, the compute-memory ratio is $h_q s_q \cdot \frac{d_k+d_v}{d_k} \approx 2 h_q s_q$.
>
> An NVIDIA H800 SXM5 GPU has a peak memory bandwidth of 3.35 TB/s and peak FLOPs of 990 TFlops. However, due to throttling (reducing to ~1600 MHz in our case), the practical peak FLOPs drops to ~865 TFlops. Therefore, when $h_qs_q \ge \frac{1}{2} \cdot \frac{865}{3.35} = 128$, the kernel is compute-bound; otherwise, it's memory-bound.

> The output matrix (scaled and accumulated during each mainloop round, similar to FlashAttention's algorithm) must be stored in registers due to WGMMA instruction requirements. Each $64 \times 512$ output matrix occupies 32,768 32-bit registers. With only 65,536 32-bit registers per SM, we can store only one output matrix per SM. This eliminates the possibility of having two output matrices and letting them use CUDA Core and Tensor Core in a interleaved manner.

The "seesaw" schedule across two warpgroups sharing one running max:

> 0. Maintain a running max $m$ (initialized to $-\infty$, shared between the two warpgroups) and output matrices $\vec o_L, \vec o_R$ (initialized to 0).
> 1. [0] Compute $\vec p_0 = \vec q K_0^\intercal / qk\_scale$.
> 2. [1] Compute $\vec p_1 = \vec q K_1^\intercal / qk\_scale$.
> 3. [0] Compute $mp_0 = \max(\vec p_0)$, $m\_new_0 = \max(m, mp_0)$, and $scale_0 = \exp(m\_new_0 - m)$. Update $m \gets m\_new_0$.
> ...
> 11. [0] Update $\vec o_L \gets \vec o_L \cdot scale_1 + \vec p_1 V_{1L}$.

> - Fine-grained TMA copy - GEMM pipelining: For a $64 \times 576$ K block, we launch 9 TMA copies (each moving a $64 \times 64$ block). GEMM operations begin as soon as each TMA copy completes ... improving memory latency tolerance.
> - Cache hints: Using `cute::TMA::CacheHintSm90::EVICT_FIRST` for TMA copies improves L2 cache hit rates, as shown by experiments.

> These optimizations achieve up to 80% Tensor Core utilization (of the throttled theoretical peak) and 3 TB/s memory bandwidth on an H800 SXM5 GPU.
> ...
> - Programmatic Dependent Launch. We use programmatic dependent launch to overlap `splitkv_mla` and `combine` kernels.
> - Tile Scheduler. We implement a tile scheduler to allocate jobs (requests and blocks) to SMs. This ensures a balanced load across SMs.

## Relevance

`ratio ≈ 2·h_q·s_q` is the formula that decides whether MLA decode is overhead-bound at our
operating point. GLM-5.2 has 64 Q heads; at TP16 that is 4 live heads per die, and with `s_q=1`
the ratio is 8 — two orders of magnitude below the 128 threshold at which the kernel would be
compute-bound. So MLA decode at our configuration is deep in the latency/overhead regime, which
is consistent with SparseFlashAttention at 6.973 ms against 1.382 ms of bytes. DeepSeek's reason
for keeping `h_q` at 128 (no attention TP) is the same observation read in the other direction.
The fine-grained `9 x TMA(64x64)` pipelining of a `64 x 576` K block is the pattern for hiding
C8 cache-load latency inside our attention stage.

### FlashMLA's fused norm + RoPE + attn + inverse-RoPE + FP8-cast kernel

https://github.com/deepseek-ai/FlashMLA#fused-norm--rope--attn--rope--cast-kernel
DeepSeek, 2026.09.10 / 2026.09.30 release notes.

> In the DeepSeek-V4.1 release, we also provide a fused kernel that combines Q-norm (only used in V4, not in V4.1), Q-RoPE, core attention, O-RoPE (conjugate) and the cast to FP8 into a single kernel. It removes the extra time spent on these small kernels while keeping the same or even slightly higher TFlops, at the cost of having to permute the Q_b and Wv weights in advance.
>
> In DeepSeek-V4.1 attention, Q (`[hidden_size]`) is first projected to `[q_lora_rank]` (the Q_a projection) and then to `[num_attention_heads, head_dim]` (the Q_b projection). After core attention, the output (`[num_attention_heads, head_dim]`) is reshaped to `[o_groups, num_attention_heads // o_groups * head_dim]`, and each of its rows is projected to `[o_lora_rank]` (the Wv projection), giving an `[o_groups, o_lora_rank]` matrix. That matrix is reshaped to `[o_groups * o_lora_rank]` and finally projected to `[hidden_size]` (the Wo projection). This kernel requires the Q_b and Wv weights to be permuted.

```python
out_fp8, out_sf, lse = fused_norm_rope_attn_rope_cast.decode(
    enable_q_norm, rms_norm_eps,
    token_positions, False, 64, cos_sin_cache,
    n_wv_group, 32, True, True, True,
    q, k_cache, indices_in_kvcache,
    sm_scale=sm_scale, attn_sink=attn_sink, topk_length=topk_length,
    extra_k_cache=extra_k_cache,
    extra_indices_in_kvcache=extra_indices_in_kvcache,
    extra_topk_length=extra_topk_length,
)

# The FP8 output is consumed directly by the Wv projection, using the permuted Wv weight
wv_out = torch.empty((s_q, n_wv_group, wv_proj_out_dim), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_einsum("bhr,hdr->bhd", (out_fp8, out_sf), (wv_proj_fp8, wv_sf), wv_out, recipe=(1, 1, 32))
```

> We also optimize the performance of the fused norm-RoPE-attn-RoPE-cast kernel under decoding settings by around 10% - 15%.

## Relevance

The concrete answer on fusing o_proj into the attention epilogue: DeepSeek deliberately stops
short. The attention kernel absorbs Q-norm, Q-RoPE, O-RoPE and the FP8 cast — so the output
leaves the kernel already quantized with its scale factors — while the Wv/Wo projection remains
a separate einsum consuming that FP8 directly. The price is an offline permutation of Q_b and
Wv. For us: fuse norm/RoPE/cast into attention and hand the projection a pre-quantized
activation (which is what `MlaPrologV3`'s `queryQuantMode=1` + `dequantScaleQNopeOut` already
does on the input side), rather than pulling the projection GEMM inside.

---

### MLA absorption — DeepSeek-V2 paper, and why RoPE obstructs it

https://arxiv.org/abs/2405.04434 (HTML: https://arxiv.org/html/2405.04434v5), Section 2.1.2 and Appendix C.

> c_t^{KV} = W^{DKV} h_t,                         (9)
> k_t^{C}  = W^{UK} c_t^{KV},                     (10)
> v_t^{C}  = W^{UV} c_t^{KV},                     (11)
> where c_t^{KV} ∈ R^{d_c} is the compressed latent vector for keys and values; d_c (≪ d_h n_h) denotes the KV compression dimension; W^{DKV} ∈ R^{d_c×d} is the down-projection matrix; and W^{UK}, W^{UV} ∈ R^{d_h n_h × d_c} are the up-projection matrices for keys and values, respectively.
> During inference, MLA only needs to cache c_t^{KV}, so its KV cache has only d_c l elements, where l denotes the number of layers.
> In addition, during inference, since W^{UK} can be absorbed into W^{Q}, and W^{UV} can be absorbed into W^{O}, we even do not need to compute keys and values out for attention.

> In this way, W^{UK} cannot be absorbed into W^{Q} any more during inference, since a RoPE matrix related to the currently generating token will lie between W^{Q} and W^{UK} and matrix multiplication does not obey a commutative law. As a result, we must recompute the keys for all the prefix tokens during inference, which will significantly hinder the inference efficiency.

Appendix C:

> During inference, the naive formula needs to recover k_t^{C} and v_t^{C} from c_t^{KV} for attention.
> Fortunately, due to the associative law of matrix multiplication, we can absorb W^{UK} into W^{UQ}, and W^{UV} into W^{O}.
> Therefore, we do not need to compute keys and values out for each query.
> Through this optimization, we avoid the computational overhead for recomputing k_t^{C} and v_t^{C} during inference.

## Relevance

The absorption identity and its RoPE obstruction — the reason decoupled RoPE exists, and the
reason an absorbed decode kernel needs separate `nope` and `pe` code paths (which is exactly
what `aclnnSparseFlashAttention`'s paired `queryRopeOptional`/`keyRopeOptional` with `Dr=64`
encodes, and what `MlaProlog` splits via the concatenated `[W^UQ|W^QR]` weight).

---

### MLA absorption as runnable einsums, with a FLOPs/footprint table showing decode is ~1% of naive

https://github.com/flashinfer-ai/flashinfer/pull/551
flashinfer-ai/flashinfer, author `tsu-bin`, 2024-10-23, "feat: support MLA decode".

> ### The mystery Mat Absorb algorithm
> In the DeepSeekV2 paper, there was no specific formulas for how to do param matrixes absorption, but it just vaguely said that
> > Fortunately, due to the associative law of matrix multiplication, we can absorb 𝑊𝑈𝐾 into 𝑊𝑈𝑄, and 𝑊𝑈𝑉 into 𝑊𝑂
> ...
> Here is my conclusion on this topic, Mat Absorb is only suitable for decode, do not use Mat Absorb for prefill, which means MLA decode and prefill should have different computation graph and different set of params, and Mat Absorb should merge param matrixes offline, materialize the merged param matrixes.

```python
# Now we merge W_UQ and W_UK (absorb W_UK into W_UQ)
# q~q_lora_rank  n~num_heads  d~qk_nope_head_dim  l~kv_lora_rank
self.W_UQ_UK = torch.einsum("q n d, l n d -> q n l", W_UQ, W_UK).flatten(start_dim=1) # [1536, 65536]

# Merge W_UV and W_O (absorb W_UV into W_O)
# l~kv_lora_rank  n~num_heads  d~v_head_dim  h~hidden_size
self.W_UV_O = torch.einsum("l n d, h n d -> n l h", W_UV, W_O).flatten(start_dim=0, end_dim=1) # [65536, 5120]
```

> And after Mat Absorb, MLA become a special 128 heads / (512+64) dim MQA algorithm, please note that the compressed_kv is used as both K and V directly without any projection.

```python
def num_float_ops_mat_absorb(q_len, kv_len):
    return ( q_len*1536*(128*64) + # from c_Q to q_pe, corresponding to W_QR
                q_len*1536*(128*512) + # from c_Q to q_nope, corresponding to W_UQUK
                128 * (q_len*64*kv_len + q_len*512*kv_len + q_len*kv_len*512) + # 128 heads MQA
                q_len * (128*512) * 5120 ) # from MHA output to output hidden states, corresponding to W_UV_O
```

> prefill: num_float_ops mat_absorb vs vanilla ratio  ~ 3.3602009088734754
> prefill: mem_footprint mat_absorb vs vanilla ratio  ~ 2.2874373717252205
> decode: num_float_ops mat_absorb vs vanilla ratio  ~ 0.010941137164898957
> decode: mem_footprint mat_absorb vs vanilla ratio  ~ 1.167867978048944

> So we can conclude from the result above, for decode case Mat Absorb version only use about 1% computation compared to Vanilla version, and the memory footprint is at the same level with Vanilla version, but for prefill case, both computation and memory footprint are much higher than Vanilla version, so there is no reason to use Mat Absorb for prefill, but it's worth a try for decode.

> There is one problem that the 128heads / (512+64)dim Q data is too large to fit into one SM's register file or even smem, which means we can't use only one SM/block to process one Q data, we have to tile the num_qo_heads dimension into gridDim.y, which can cause kv-cache data movement from gmem to smem multiple times, though this is inevitable.

> Tensor-core version implementation ... but in my opinion maybe this can have limited performance improvement, because consider the above analysis, the bottle neck is IO bandwidth not the computation intensity.

## Relevance

The absorption written as runnable einsums with offline materialisation, plus the quantified
verdict: decode uses ~1.1% of the naive FLOPs at the same memory footprint. Two consequences
for us. First, `Mat Absorb should merge param matrixes offline, materialize the merged param
matrixes` — but note CANN goes the other way: `MlaProlog` keeps `W^UK` separate and applies it
in-kernel as `MatmulQn`, which is the right choice when `W_UQ_UK` would be `[1536, 65536]`.
Second, the closing judgement — `the bottle neck is IO bandwidth not the computation
intensity`, so the matrix unit buys little on the absorbed path — argues for keeping the
absorbed-path arithmetic on the vector units, where our per-call floor is 0.70 us rather than
13.58 us.

---

### FlashInfer `BatchMLAPagedAttentionWrapper` — the absorbed-formulation contract and graph-replay metadata

https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/flashinfer/mla/_batch_mla/_wrapper.py

```python
class BatchMLAPagedAttentionWrapper:
    r"""Wrapper for MLA PagedAttention on DeepSeek models.

    This wrapper is intended for decode and incremental prefill with the
    Matrix Absorption formulation of MLA, where the query/key and value/output
    projections are absorbed before attention. For the non-absorbed MLA
    prefill path, use the appropriate prefill wrapper instead.

    The planned-wrapper surface owns FA2, FA3, CUTLASS, cuTile, TRTLLM-GEN,
    XQA, and CuTe DSL planning and execution. Call :meth:`plan` once with
    canonical metadata before invoking :meth:`run`; the plan captures the
    supported input/output contract and the concrete backend's metadata
    representation.
    """
```

```python
    def plan(
        self, *,
        metadata: MLAPlanMetadata,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool = False,
        query_layout: Literal["packed", "split"] = "packed",
        kv_cache_layout: Literal["packed", "split"] = "packed",
        lse_mode: Literal["none", "base2", "basee"] = "none",
        output_dtype: Optional[torch.dtype] = None,
        output_scale: Literal["none", "per-tensor"] = "none",
        scale_mode: Literal["default", "kv-per-tensor", "bmm-scalar", "bmm-tensor"] = "default",
        skip_softmax: bool = False,
        enable_pdl: Optional[bool] = None,
        use_sinks: bool = False,
    ) -> None: ...
```

> TRTLLM-GEN, XQA, and CuTe
> DSL CUDA graph plans require supplied contiguous ``int32`` dense
> metadata on the wrapper device. Retain and update those device tensors
> in place for replay; CPU and CSR-only graph metadata are rejected.
> Query offsets and lengths must remain fixed after graph planning.

## Relevance

The split of `head_dim_ckv` from `head_dim_kpe` in the interface is the API-level fingerprint of
the absorbed decode path, matching CANN's `D=512` / `Dr=64`. `Retain and update those device
tensors in place for replay` plus `Query offsets and lengths must remain fixed after graph
planning` is the same address-stability discipline the vllm-ascend GLM-5.2 MegaKernel RFC
states for DAG replay.

---

## 5. Megakernel mechanism: how a persistent kernel schedules a whole decode step

### ThunderMLA — a megakernel for decode attention driven by an instruction tensor

https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla
Benjamin Spector, Aaryan Singhal, Dan Fu, Chris Ré; Mar 4, 2025.

> ThunderMLA is a completely fused "megakernel" for decode that's 20-35% faster than DeepSeek's FlashMLA on diverse workloads. It turns out some simple scheduling tricks can get you pretty far!

> The core issue lies in the substantial overhead created by:
> Running two separate kernel launches
> Tail effects between kernel executions
> Limited batch sizes when running large reasoning models
> ...
> A batch of 4 prompts, of lengths [4641, 45118, 1730, 1696]
> Generating 4 new tokens (e.g., for a speculator)
> 8-way tensor parallel -- that is, 16 heads per GPU for DeepSeek R1
>
> With 4 new tokens, FlashMLA runs in 52 us on an SXM H100, achieving just 144 TFLOPS and 1199 GB/s (out of the 939 TFLOPS / 3300 GB/s advertised by NVIDIA).
> ...
> on the same workload, ThunderMLA runs in just 41 us, achieving 183 TFLOPs and 1520 GB/s.

> Our solution eliminates these kernel launches by creating a simple virtual instruction set on the GPU. Instead of launching separate kernels, we fuse them into a single "megakernel," and then pass this megakernel its instructions at runtime through an instruction tensor, which in turn can dramatically modify its behavior. Within the megakernel, GPU SMs read from this instruction tensor, decide what work to perform next, and then execute it.

> Our implementation adapts a slot attention kernel and token reduction kernel into this interpreter template with just a few hundred lines of device code, and uses a global tensor as a semaphore to synchronize dependencies across instructions. Once both the two constituent kernels ("partial" and "reduction") of the megakernel for FlashMLA are implemented, the megakernel can be instantiated as interpreter::kernel<config, partial_template, reduction_template>. Altogether, this fused megakernel is just 250 lines of device code in ThunderKittens.

> Key benefits include:
> Vastly simplified pipelining within instructions, across loop boundaries
> Asynchronous fetching of the future instructions, minimizing instruction overhead
> Deep pipelines that hide data fetch latencies within and across instructions

## Relevance

The minimal viable megakernel, and the natural first increment for us: fuse only the split-KV
partial and the reduction, driven by a runtime instruction tensor, 250 lines of device code,
20-35% over a hand-tuned kernel. On Ascend the semaphore is a workspace counter plus
`CrossCoreWaitFlag`; the instruction tensor is a device-resident schedule built once per shape
(exactly what CANN's `flash_mla_metadata` already is).

---

### "Look Ma, No Bubbles!" — the full cost accounting of a decode megakernel

https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
Benjamin Spector*, Jordan Juravsky*, Stuart Sul, Owen Dugan, Dylan Lim, Dan Fu, Simran Arora, Chris Ré; May 27, 2025.

Why kernels stall, verbatim:

> First: GPU kernels are launched with a strict ordering, so that a thread block in one kernel can't start until all thread blocks in previous kernels have completely finished. Consequently, every time we start a kernel, we have to wait for all the straggler thread blocks from the prior one to finish. For example, if a kernel runs 512 thread blocks (like our Llama-1B down projection), but we only have 148 streaming multiprocessors (like on a B200), we end up with 80 empty SM's at the end.
>
> Second, as we've previously highlighted, each kernel launch and teardown incurs costs. In principle, NVIDIA's CUDA graphs can help hide costs, but by our measurements they still leave a lot on the table. For a simple dummy kernel (which dumps a start time, sleeps, and dumps an end time) on an H100, we find that running on a CUDA stream incurs a launch cost of about 2.1 microseconds, and with CUDA graphs the launch cost only decreases to around 1.3 microseconds – time spent with the GPU doing no useful work!
>
> Finally, even after we start the next kernel, we still have to wait to load weights and activations before any compute can start. These latencies leave the GPU sitting idle for thousands of cycles! Ideally, we'd start loading the next weights while the previous computations and stores are happening. NVIDIA has also built a mechanism for this called Programmatic Dependent Launch (PDL), which allows the next kernel to start preparing while the previous kernel is running, but we found it still introduces unnecessary stalls because the PDL synchronization mechanism (cudaGridDependencySynchronize) is very coarse. For example, it means we have to wait for all queries, keys, and values to complete in order to start attention, as opposed to starting heads as soon as they are ready.

> To illustrate the magnitude of the problem: for single-sequence generation in 16-bit precision on a single H100, the memory limit is 3.35TB/s / 2.48GB = ~1350 forward passes per second. But with 7 kernel launches per layer, and 16 layers, even with an optimistic 5 us of stalling per kernel (counting stragglers, kernel launch, and memory latencies), generation would run at just ~770 forward passes per second.

The instruction/interpreter model:

> Our interpreter is designed such that each streaming multiprocessor (SM) within the GPU receives a sequence of instructions (each implemented using the same CUDA template) and executes them. We schedule each SM's instruction sequence ahead of time on the Python side, and notably we can reuse each schedule for hundreds of forward passes!
>
> For our end-to-end Llama forwards pass megakernel, we define the following set of instructions:
> A fused RMS norm & QKV & RoPE instruction.
> An attention computation instruction.
> An attention reduction instruction (for ThunderGQA on long sequences).
> An O-projection + residual instruction.
> A fused RMS norm & up-gate & SiLU instruction.
> A down-projection + residual instruction.
> An RMS norm & language modeling head instruction, for computing the final token logits.

Paged shared memory:

> We accomplish this by paging shared memory. We first divide the first 213kB of shared memory on an H100 into 13 16KiB pages, and use remaining shared memory for special purposes, like storing instruction parameters. To use one of these pages, instructions have to explicitly request and release them from the interpreter. The interpreter automatically passes released pages to the next instruction, allowing them to start issuing memory loads as early as shared memory becomes available.

Synchronization by counters, and the chunked-MLP optimization it enables:

> We accomplish this with a simple counter system. Before the megakernel launches, we initialize an array of counters (i.e. integers) in GPU global memory with a starting value of zero. Whenever an instruction completes, it increments one of these counters. Similarly, whenever a new instruction starts, it must wait for some of these counters to reach a target value, indicating that all of its dependencies have finished.
>
> One optimization this enables is in the big multi-layer perceptrons (MLPs) in Llama-1B. In a naive implementation using PDL, one must await completing the whole hidden state before beginning the down projection matrix multiply. We instead produce and consume the intermediate state in four chunks, each with their own counter. This way, an instruction for the down projection only needs to wait for its input chunk to finish.

Results, and the full residual-cost breakdown:

> Doing this achieves brr – on an H100, we use 78% of memory bandwidth and outperform existing systems by over 1.5x.

> To our knowledge, our H100 megakernel represents the first time anyone has run the forward pass for a 16-bit 1B+ parameter language model in under one millisecond on a GPU. Our B200 implementation pushes this even further to under 680 microseconds per forward pass!
> ...
> On an H100, our megakernel runs almost 2.5x faster than vLLM and over 1.5x faster than SGLang.
> On a B200, the gap with vLLM rises to over 3.5x, and we remain more than 1.5x faster than SGLang, too.

> A breakdown of the runtime of our current B200 forward pass (total runtime 600 microseconds):
> 250 microseconds are spent storing activations, awaiting consistency, and loading them. This is about 20% higher than a simple model would suggest: since each instruction has a dependence on the last one, we need to pay two load latencies (check ready, and then load activations) and two store latencies (store activations, then mark ready) per instruction. Using ~500 nanoseconds latency per load / store, this would impose about 200 microseconds of overhead. (We suspect some of the remaining 50 microseconds comes from time spent processing atomics in global memory.)
> 200 microseconds are spent actually running RMS norm and matrix-vector computations. 95% of this portion is devoted to matrix-vector. On Blackwell, we find that using the tensor cores is marginally helpful for this; on Hopper, we find it better to simply run on the CUDA cores. This difference comes from the fact that both GPUs have relatively similar CUDA core performance, but Blackwell tensor cores are much faster.
> 30 microseconds are spent awaiting weights from global memory (pipelining works!) Of these, 40% are spent in the LM head, which is the best-pipelined part of the whole megakernel due to its homogeneity and huge size.
> 40 microseconds are spent on low-level synchronization overhead across warps. A key issue here is that CUDA's asynchronous barriers are relatively slow, even when they're already in the "pass" state, requiring about 60 nanoseconds each time.
> 80 microseconds are on setup and various other overheads (e.g. passing instruction barriers, marking pages as complete, etc.)

## Relevance

The most transferable budget in the literature, and a caution. After fusing everything,
250/600 us is still activation store + ready-flag + load latency — two load and two store
latencies per instruction — and 40 us is barrier cost at ~60 ns per barrier. For us that means
the win is bounded by instruction count and flag round-trips, not launch count alone, so the
megakernel's instruction granularity is the design variable. Two directly applicable findings:
chunk the intermediate state and give each chunk its own counter so the down projection starts
per chunk (the Ascend analogue of DeepGEMM's L2 bitmask, and of our `pre/mid/post` pipeline);
and on Hopper matvec is better on CUDA cores than tensor cores — the Ascend analogue is
AI_VECTOR_CORE at a 0.70 us floor versus MIX_AIC at 13.58 us.

---

### TP megakernel for Llama-70B — and the finding that fusing o_proj under TP is the wrong move

https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main
and https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-intro — same authors, Sep 28, 2025.

> However, we've made one important change to this formulation. One of our targets for operation overlapping is to overlap the O projection matrix multiplication with the subsequent reduce-scatter. However, the tensor-parallel sharded O matrix is too small for us to effectively hide the reduce-scatter communication cost. To solve this, we instead choose to replicate the O projection matrix across each GPU and run the O projection with data parallelism instead of with tensor parallelism. Alongside this change, we eliminate the post-attention reduce-scatter and replace it with a "distributed transpose" operation after attention that repartitions our data from a tensor-parallel configuration to a data-parallel configuration.
>
> When using 8 GPUs, this reduces the network traffic by a factor of 8, which makes it much easier to hide the cost of communication cost by overlapping it with matrix multiplications. Note that the downside of this approach is it reduces the maximum batch size by about 15%, because replicating the O projection weights consumes an additional 9 GB of memory per GPU.

The TP instruction set:

> RMS norm + all-gather.
> A QKV matrix multiply + RoPE.
> Attention + distributed transpose.
> O-projection matrix multiply + residual.
> Gate matrix multiply + SiLU.
> Up-projection matrix multiply + elementwise multiplication with the output of the gate.
> Down matrix multiply + reduce-scatter + residual.
> RMS norm without the all-gather (for the LM head)
> LM head matrix multiply.

On recomputation versus communication, and on dependency granularity:

> Most of our instructions for low-latency centered around matrix-vector multiplication, rather than the matrix-matrix multiplications we do here. ... To avoid extra trips to global memory, for our low-latency megakernel we frequently recomputed results across the GPU rather than communicating them through memory. This allowed us to fuse operations more aggressively than usual. For example, our QKV matrix-vector product fused the RMS norm into its beginning, saving an instruction boundary, but performing identical RMS computations over a hundred times. When focusing on throughput, this recomputation is not worth it.

> Putting many tokens in our batch requires us to expand our inter-instruction signalling scheme to track data dependencies across tokens. This signalling can vary by instruction -- for some operations (e.g. attention, RMS norms), sometimes synchronization is at the granularity of 128 output rows of a matrix; other times it's at the granularity of an individual attention head for an individual token.

Static per-SM queues versus a global atomic work queue:

> With our low-latency megakernel, each SM was assigned its own queue of instructions that are scheduled in advance. Instead, for our high-throughput megakernel, we create a global work queue -- a single list of instructions that defines all the work that needs to run on the GPU. When an SM needs to fetch a new instruction to run, it atomically increments a global instruction counter that keeps track of the next instruction to be assigned. ... This solution wasn't possible for our low-latency megakernel, because the runtime of each instruction was so fast that the latency of this atomic increment would be prohibitive.

> We perform all communication from our dedicated storer threads, allowing loader and compute threads to move onto future work while inter-GPU communication is performed in the background on the same SM.

> Whereas with instruction pipelining enabled, the extra gap between consecutive matrix multiply stages is just 3.4 microseconds, without the pipeline, this gap jumps to 10.2 microseconds -- meaning this optimization alone reduces runtime by over 7% on these instructions.

## Relevance

Directly answers the TP16 question. Fusing o_proj into the attention epilogue is the wrong move
when o_proj is TP-sharded: they instead replicate o_proj, run it data-parallel, and replace the
post-attention reduce-scatter with a distributed transpose folded into the attention
instruction — 8x less wire traffic. Also decisive for our batch=1 case: a global atomic work
queue is too slow when instructions are short, so use static per-core instruction queues.
And the licence for aggressive fusion at low latency: recomputing RMS norm a hundred times to
save one instruction boundary was the right trade.

---

### MPK — a compiler and runtime that mega-kernelizes the whole model, with the launch-count arithmetic

https://arxiv.org/abs/2512.22219
`MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs` — Xinhao Cheng, Zhihao Zhang, Yu Zhou, Jianan Ji, Jinchen Jiang, Zepeng Zhao, Ziruo Xiao, Zihao Ye, Yingyi Huang, Ruihang Lai, Hongyi Jin, Bohan Hou, Mengdi Wu, Yixin Dong, Anthony Yip, Songting Wang, Wenqin Yang, Xupeng Miao, Tianqi Chen, Zhihao Jia. Submitted 22 Dec 2025, v2 10 Jun 2026. OSDI '26: https://www.usenix.org/conference/osdi26/presentation/cheng

> We introduce Mirage Persistent Kernel (MPK), the first compiler and runtime system that automatically transforms multi-GPU model inference into a single high-performance mega-kernel. MPK introduces an SM-level graph representation that captures data dependencies at the granularity of individual streaming multiprocessors (SMs), enabling cross-operator software pipelining, fine-grained overlap of computation and communication, and other optimizations that are infeasible under the conventional kernel-per-operator execution model. The MPK compiler lowers tensor programs into optimized SM-level task graphs and generates fast CUDA implementations for each task, while the MPK in-kernel parallel runtime executes these tasks within a single persistent mega-kernel using decentralized scheduling across SMs. Together, these components provide end-to-end kernel fusion with minimal developer effort, while preserving the flexibility of existing programming models. Our evaluation shows that MPK significantly outperforms existing kernel-per-operator LLM serving systems, achieving up to 1.7x lower end-to-end inference latency and pushing LLM inference performance close to the limits of the underlying hardware.

Task/event model:

> Each task—shown as a blue (or orange) rectangle—denotes a unit of computation (or communication) executed on a single SM. Each event—shown as a green circle—represents synchronization across tasks. Tasks and events alternate in the graph: every task only has outgoing edges to triggering events ... A task is ready for execution when its dependent events are all [activated] and notifies its triggering event upon completion. An event is activated once it has received notifications from all tasks associated with it.

> For any two events e1 and e2 in a tGraph, successor-set fusion applies if and only if OutTasks(e1)=OutTasks(e2). MPK removes e1 and e2 from the tGraph and introduces a fused event e′ with InTasks(e′)=InTasks(e1)∪InTasks(e2) and OutTasks(e′)=OutTasks(e1).

> Normalization bounds the dependency metadata stored for each task. When every task depends on and triggers at most one event, each task descriptor needs to store only one dependent-event identifier and one triggering-event identifier, rather than variable-length lists.

> MPK addresses this challenge using a breadth-first-search-based algorithm ... to linearize a tGraph. The algorithm assigns contiguous indices to all tasks triggered by the same event in the final task ordering. As a result, the fan-out of an event can be encoded compactly using only the first and last task indices, eliminating the need to store an explicit list of dependent tasks while preserving all dependencies.

Runtime structure and the JIT/AOT launch distinction:

> To support this execution model, MPK partitions a GPU's SMs into workers and schedulers. Each worker runs on one physical SM and maintains an independent task queue. ... Schedulers are organized at warp granularity, with each SM hosting four scheduler warps. Each scheduler maintains an event queue and repeatedly polls for newly activated events, dispatching the corresponding tasks to workers. The allocation of workers and schedulers is fixed at kernel-launch time and matches the GPU's physical SM count, avoiding any dynamic role-switching overhead inside the kernel.

> Under JIT launch (Figure 8a), the worker executing T1 notifies a scheduler, which then dequeues event e, launches T2, and enqueues it into a worker's task queue. The receiving worker must then dequeue T2 before execution. This chain requires two synchronization steps (worker→scheduler and scheduler→worker). By contrast, under AOT launch (Figure 8b), T2 has already been enqueued on a pre-assigned worker, which only needs to wait for event e to be activated. Thus, AOT launch requires only one synchronization step through the event trigger, reducing per-task launch latency.

> Operators with data-dependent durations (e.g., attention) are marked as JIT, and their downstream operators remain JIT until execution reaches a global barrier (i.e., an event that must be triggered by all upstream tasks). Such barriers eliminate accumulated imbalance, making subsequent operators suitable for AOT launch. All remaining operators are labeled AOT to minimize dispatch overhead.

> A task may acquire one or more pages based on its shared-memory footprint and must release the pages when they are no longer needed. Once a task releases any page, it is no longer permitted to acquire additional shared-memory pages, enforcing a monotonic usage pattern that simplifies scheduling.

> Every task is associated with a task description that encodes its input tensors, output tensors, and configuration parameters; in our current implementation, each description occupies 352 bytes

Configuration and measured numbers:

> GPU  # SMs  # Workers  # Schedulers
> A100  108  104  16
> H100  132  128  16
> B200  148  144  16
> ...
> We set the shared-memory page size to 32 KB on all GPUs, which yields 5, 7, and 7 shared-memory pages per SM on A100, H100, and B200, respectively.

> For example, on Qwen3-8B running on an A100 GPU, MPK reduces per-token decoding latency from 14.5 ms, achieved by highly optimized systems such as vLLM and SGLang, to 12.5 ms. This approaches the approximate hardware lower bound of 10 ms, estimated by loading 16 GB of model parameters at 1.6 TB/s memory bandwidth.

> MPK executes the entire model with a single kernel launch, whereas a kernel-per-operator execution of Qwen3-8B issues 293 kernel launches per token. On B200, each launch costs 3.8 μs in eager execution, totaling 1.1 ms per token, and 0.8 μs with CUDA Graphs, totaling 0.2 ms per token. MPK avoids this overhead; its in-kernel scheduler accounts for only 0.28% of total runtime

> in Qwen3-8B, 293 operators are decomposed into 13,867 tasks. ... across the evaluated models, the final tGraphs contain only 1,142–2,366 events, yet these events encode 69,000–162,000 producer–consumer task-pair dependencies. This corresponds to a 37–118x reduction in synchronization events

The MoE case study:

> For Qwen3-30B-A3B at batch size one on SGLang, this preprocessing step accounts for up to 11% of total MoE execution time. In MPK, implementing this step as a separate preprocessing task would also introduce additional scheduling overhead. MPK addresses this issue by replacing the TMA-based gather with an asynchronous token-level copy integrated directly into the data-loading phase of the GEMM tasks.

> Cross-task pipelining reduces task runtime by 1.2–1.3x and even outperforms cuBLAS-based compiled kernels.

## Relevance

The complete reusable scheduling architecture, with the one design decision that matters most
at batch=1: JIT dispatch costs two synchronization hops and is only worth it for
data-dependent operators (attention); everything else should be pre-enqueued AOT so the runtime
cost is one event-flag wait. Also directly implementable as an Ascend instruction table:
fixed-size task descriptors (352 bytes there), exactly one dependent and one triggering event
per task after normalization, and contiguous-range event fan-out so no dependent-task lists are
stored. The MoE note — fold the gather into the GEMM task's data-loading phase rather than
making it a separate task — is the same conclusion as deleting our `moe_init_routing` stage.

---

### ForgeMegakernel and Event Tensor — the achievable MBU band, and dynamism as the hard part

https://arxiv.org/abs/2609.12379
`ForgeMegakernel` — Leshan Li, Zhui Zhu, Xianglong Deng, Yaojian Chen, Qingfeng He, Yuxuan Li, Rong Zhao, Xu Han, Zhiyuan Liu; submitted 11 Sep 2026.

> Auto-regressive model decode is bandwidth-bound, since every weight and key/value-cache byte crosses high-bandwidth memory once per token. A megakernel is an ideal solution, but existing automatic megakernel generation approaches cannot achieve both generalization across models and correctness guarantees.
> We present ForgeMegakernel, which generates a per-model high-performance decode megakernel using coding agents. ForgeMegakernel pairs a universal knowledge base of ten progressive milestones with an independent mid-state test oracle. The milestones provide the megakernel's structural properties: a fine-grained instruction stream for each SM, dependency counters replacing the global synchronization, and a shared-memory buffer pool for workload balance across SMs and greater parallelism. The test oracle derives the mid-states of the megakernel and checks the performance, error and precision during the generation process, guaranteeing a correct and trustworthy forged megakernel.
> We evaluated ForgeMegakernel on 14 representative decoding operations across eight model families spanning 0.6B-13B parameters. The generated megakernels achieved 50.5-85.9% MBU and geometric mean speedups of 1.21x over SGLang 0.5.18 and 1.54x over a megakernel compiler under identical configurations.

https://arxiv.org/abs/2604.13327
`Event Tensor` — Hongyi Jin, Bohan Hou, Guanjie Wang, Ruihang Lai, Jinqi Chen, Zihao Ye, Yaxing Cai, Yixin Dong, Xinhao Cheng, Zhihao Zhang, Yilong Zhao, Yingyi Huang, Lijie Yang, Jinchen Jiang, Gabriele Oliaro, Jianan Ji, Xupeng Miao, Vinod Grover, Todd C. Mowry, Zhihao Jia, Tianqi Chen; submitted 14 Apr 2026, v2 21 Apr 2026.

> Modern GPU workloads, especially large language model (LLM) inference, suffer from kernel launch overheads and coarse synchronization that limit inter-kernel parallelism. Recent megakernel techniques fuse multiple operators into a single persistent kernel to eliminate launch gaps and expose inter-kernel parallelism, but struggle to handle dynamic shapes and data-dependent computation in real workloads. We present Event Tensor, a unified compiler abstraction for dynamic megakernels. Event Tensor encodes dependencies between tiled tasks, and enables first-class support for both shape and data-dependent dynamism. Built atop this abstraction, our Event Tensor Compiler (ETC) applies static and dynamic scheduling transformations to generate high-performance persistent kernels. Evaluations show that ETC achieves state-of-the-art LLM serving latency while significantly reducing system warmup overhead.

## Relevance

Independent convergence on the same three structural primitives — per-SM instruction stream,
dependency counters replacing global synchronization, shared-memory buffer pool — and a
realistic ceiling to calibrate the 2.x ms target: 50.5-85.9% MBU. Event Tensor names the part
that is genuinely hard for us: data-dependent dynamism, i.e. MoE routing and the sparse
attention index, not the fusion itself.

---

## 6. `KvQuantSparseFlashAttention` — the C8-aware SFA, with dequant and the scale packed into the cache slot

### KvQuantSparseFlashAttention — supported on A3, explicitly rewritten for the discrete-access problem

https://gitcode.com/cann/ops-transformer/blob/master/attention/kv_quant_sparse_flash_attention/README.md
CANN `ops-transformer`, master.

> # KvQuantSparseFlashAttention
>
> ## 产品支持情况
>
> |产品      | 是否支持 |
> |:----------------------------|:-----------:|
> |<term>Ascend 950PR&950DT系列产品</term>|      √     |
> |<term>Atlas A3系列产品</term>|      √     |
> |<term>Atlas A2系列产品</term>|      √     |
> |<term>Atlas 200I/500 A2推理产品</term>|      ×     |
> |<term>Atlas推理系列产品</term>|      ×     |
> |<term>Atlas训练系列产品</term>|      ×     |
>
> ## 功能说明
>
> - API功能：`kv_quant_sparse_flash_attention`在`sparse_flash_attention`的基础上支持了[Per-Token-Head-Tile-128量化]输入。随着大模型上下文长度的增加，Sparse Attention的重要性与日俱增，这一技术通过"只计算关键部分"大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。
>
> - 计算公式：
>
>     $$
>     Attention=\text{softmax}(\frac{Q @ \text{Dequant}({\tilde{K}^{INT8}},{Scale_K})^T}{\sqrt{d_k}})@\text{Dequant}(\tilde{V}^{INT8},{Scale_V}),
>     $$
>
>     其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`LightningIndexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度，$\text{Dequant}(\cdot,\cdot)$为反量化函数。

The statement of what was optimized:

> 本次公布的`kv_quant_sparse_flash_attention`是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。

The cache layout — note that the dequant scale is packed into the same slot as nope and rope:

> | query | 输入 | attention结构的Q输入，不支持非连续。query由相同数据类型的q_nope和q_rope按D维度拼接得到。layout_query为"BSND"时shape为[B, Q_S, Q_N, Q_D]。layout_query为"TND"时shape为[Q_T, Q_N, Q_D]。其中Q_D值仅支持576，即q_nope+q_rope=512+64；Q_N值支持1/2/4/8/16/32/48/64/128。 | FLOAT16、BFLOAT16 | ND |
>
> | key | 输入 | attention结构的K输入。k_nope、query相同数据类型的k_rope和float32的量化参数按D维度拼接得到。layout_kv为"BSND"时shape为[B, KV_S, KV_N, KV_D]。layout_kv为"TND"时shape为[KV_T, KV_N, KV_D]。layout_kv为"PA_BSND"时shape为[block_num, block_size, KV_N, KV_D]，其中block_num为PageAttention时block总数，block_size为一个block的token数，block_size取值为16的整数倍，最大支持到1024。KV_N仅支持1；KV_D值仅支持656，即nope+rope*2+dequant_scale*4=512+64*2+4*4。TQ4模式（key_quant_mode=3）时改为使用386字节packed槽位。 |

Quant modes:

> | key_quant_mode | ... |
> | value_quant_mode | 代表value的量化模式，传入2代表per_tile量化模式；新增传入3表示TQ4 INT4量化，且要求key_quant_mode同时传入3。 |
>
> | rope_head_dim | 表示MLA架构下的rope_head_dim大小，仅在attention_mode为2时有效，仅支持64。默认值为64。 |

The TQ4 (TurboQuant 4-bit) slot layout:

> ### INT4（TQ4）量化类型
>
> 本算子新增 TurboQuant 4-bit（TQ4）量化类型。当 `key_quant_mode=3` 且 `value_quant_mode=3` 时，key/value 使用 TQ4 packed KV cache；两个量化模式必须同时设置为 3。
>
> TQ4 的每个 KV token 使用一个 386 字节的槽位，布局如下：
>
> - `[0, 256)`：256 字节 packed INT4，每个字节包含两个 4-bit code，对应 512 维 latent/nope 数据；
> - `[256, 384)`：128 字节 BF16 RoPE 数据，对应 64 个 RoPE 元素；
> - `[384, 386)`：2 字节 FP16 token scale。该 scale 是每个 KV token 独立的反量化系数，与属性 `scale_value` 含义不同。
>
> TQ4 使用 16 个 codebook centroid 进行反量化，算子内部通过 byte-LUT 将 packed byte 解码为两个 centroid，再结合 token scale 参与 key/value 计算。TQ4 不需要额外提供 `key_dequant_scale` 和 `value_dequant_scale` 输入，量化参数随 KV cache 槽位存放。
>
> TQ4 当前面向 Ascend 910B 的 MLA-absorb 场景，query 支持 FLOAT16/BFLOAT16，`attention_mode=2`，`rope_head_dim=64`。现有单算子参数集中的 TQ4 用例采用 `layout_kv=PA_BSND`。

Constraints:

> - TQ4 INT4 模式仅在 `key_quant_mode=value_quant_mode=3` 时启用，key/value 的 KV_D 必须为 386；模式 3 不支持与模式 2 混用。
> - TQ4 的 query 头维度为 576（nope 512 + rope 64），`attention_mode=2`，`rope_head_dim=64`，KV_N 仅支持 1。

## Relevance

This is the most actionable item found for our 6.973 ms vs 1.382 ms SparseFlashAttention gap.
Three things. First, it is supported on A3 and it folds the C8 dequant into the attention itself
(`Dequant(K_int8, Scale_K)` inside the softmax argument), which is work we currently pay for in
the 2.300 ms Cast against 0.076 ms of bytes. Second, `KV_D值仅支持656，即
nope+rope*2+dequant_scale*4=512+64*2+4*4` means the per-token dequant scale is stored
*inside* the same cache slot as nope and rope — so each selected token is a single contiguous
656-byte gather rather than a nope read plus a rope read plus a scale read. That is the concrete
mechanism behind `针对离散访存进行了指令缩减及搬运聚合的细致优化`. Third, the TQ4 mode packs a
token into 386 bytes (256 B packed INT4 + 128 B BF16 rope + 2 B FP16 scale) with a 16-centroid
byte-LUT dequant, cutting the gathered bytes per selected token by 41% versus the 656-byte C8
slot — but it is documented as 910B/MLA-absorb only, so A3 applicability needs checking.

---

### SparseFlashAttention kernel: A3 (`arch22`) is a 4-file MLA-specific cube/vector split

https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_attention/op_kernel

```
attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_common.h
attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_kernel_mla.h
attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_cube_mla.h
attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_vector_mla.h
attention/sparse_flash_attention/op_kernel/sparse_flash_attention.cpp
attention/sparse_flash_attention/op_kernel/sparse_flash_attention_template_tiling_key.h
```

The template parameter space, which is the complete list of specializations the op admits:

https://gitcode.com/cann/ops-transformer/blob/master/attention/sparse_flash_attention/op_kernel/sparse_flash_attention_template_tiling_key.h

```cpp
#define SFA_LAYOUT_BSND 0
#define SFA_LAYOUT_TND 1
#define SFA_LAYOUT_PA_BSND 2

#define ASCENDC_TPL_4_BW 4

#define C_TEMPLATE 0
#define V_TEMPLATE 1

// 模板参数支持的范围定义
ASCENDC_TPL_ARGS_DECL(SparseFlashAttention, // 算子OpType
                      ASCENDC_TPL_BOOL_DECL(FLASH_DECODE, 0, 1), ASCENDC_TPL_BOOL_DECL(PAGE_ATTENTION, 0, 1),
                      ASCENDC_TPL_UINT_DECL(LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_BSND,
                                            SFA_LAYOUT_TND),
                      ASCENDC_TPL_UINT_DECL(KV_LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_BSND,
                                            SFA_LAYOUT_TND, SFA_LAYOUT_PA_BSND),
                      ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, C_TEMPLATE,
                                            V_TEMPLATE),
                      ASCENDC_TPL_BOOL_DECL(IS_SPLIT_G, 0, 1), ASCENDC_TPL_BOOL_DECL(HAS_ROPE, 0, 1), );
```

## Relevance

`FLASH_DECODE` is a template boolean, so the split-KV decode path is a compile-time
specialization rather than a runtime branch — the same lever `SPLIT_M_MODE` gives us in
MlaPrologV3. `IS_SPLIT_G` (split over the GQA group) is the knob that matters at TP16, where we
have 4 live Q heads per die. The `C_TEMPLATE` / `V_TEMPLATE` split plus the separate
`service_cube_mla.h` / `service_vector_mla.h` files confirm the op is internally organized as a
cube service and a vector service over a shared kernel driver — the same `ASCEND_IS_AIC` /
`ASCEND_IS_AIV` role structure as MlaProlog, so a megakernel can host both services directly.

---

## 7. `FusedInferAttentionScore` and `IncreFlashAttention` — the general-purpose decode attention ops

### aclnnFusedInferAttentionScoreV5 — one op for both prefill and decode

https://gitcode.com/cann/ops-transformer/blob/master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md

> ## 功能说明
>
> - 接口功能：适配decode & prefill场景的FlashAttention算子，既可以支持prefill计算场景（PromptFlashAttention），也可支持decode计算场景（IncreFlashAttention）。
>
>   相比于FusedInferAttentionScoreV4，本接口新增qStartIdxOptional、kvStartIdxOptional、pseType参数。
>
>   **说明：**
>
>   decode场景下特有KV Cache：KV Cache是大模型推理性能优化的一个常用技术。采样时，Transformer模型会以给定的prompt/context作为初始输入进行推理（可以并行处理），随后逐一生成额外的token来继续完善生成的序列（体现了模型的自回归性质）。在采样过程中，Transformer会执行自注意力操作，为此需要给当前序列中的每个项目（无论是prompt/context还是生成的token）提取键值（KV）向量。这些向量存储在一个矩阵中，通常被称为kv缓存（KV Cache）。

> 本算子中Score函数采用Softmax函数，self-attention计算公式为：
>
> $$
> Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
> $$

> query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> Q_S表示query shape中的S，KV_S表示key和value shape中的S，Q_N表示num_query_heads，KV_N表示num_key_value_heads。P表示Softmax((QK^T) / √d)的计算结果。

## Relevance

`FusedInferAttentionScore` is the unified prefill/decode attention op — it dispatches to the
`PromptFlashAttention` path or the `IncreFlashAttention` path depending on `Q_S`. It is the
general-purpose op and not the one on our critical path (we are on `SparseFlashAttention`
because of the DSA index stage), but it is the reference for what the non-sparse decode
attention contract looks like, and `IncreFlashAttention` is its decode half.

---

## Gaps

What was looked for and not found.

- **No published per-call number for MlaPrologV3/V4 at batch=1 on A3.** The design docs give the
  tiling strategy and the full TilingKey layout but no microsecond figures. The only MLA-adjacent
  per-call A3 TP16 number found anywhere was for W4A8 GroupedMatmul (65.9 us mean, PR #16606),
  not for the prolog.
- **`SparseFlashAttention` `arch22` kernel bodies not read.** The four A3 files were located and
  the template parameter space was read, but `sparse_flash_attention_service_vector_mla.h` and
  `_service_cube_mla.h` were not opened, so *how* the sparse gather is pipelined against the
  cube work on A3 is still unverified. That is the remaining load-bearing unknown behind the
  6.973 vs 1.382 ms gap.
- **No `docs/` entry for `kv_quant_sparse_flash_attention`.** The README (quoted above) gives the
  formula, the 656-byte C8 slot layout and the TQ4 386-byte slot, but there is no
  `aclnnKvQuantSparseFlashAttention.md`, so the exact aclnn signature and parameter order were
  not found. Whether TQ4 is usable on A3 (the README says `TQ4 当前面向 Ascend 910B 的
  MLA-absorb 场景` while the product table marks A3 supported for the op as a whole) is
  unresolved.
- **`MlaPrologV3` kernel source not read**, so the V3 `CrossCoreSetFlag` sequence (as opposed to
  the V1 pseudo-code quoted above) is unverified, including what actually changes under
  `SPLIT_M_MODE=0` vs `1` and under `CV_MODE` 1:1 vs 1:2.
- **No community work found that fuses MLA prolog + attention + o_proj into one kernel on
  Ascend.** The closest is the vllm-ascend GLM-5.2 MegaKernel RFC (#15639) — a plan, scoped to
  A5 and TP=1, with no accuracy or performance numbers attached yet. On the GPU side DeepSeek's
  `fused_norm_rope_attn_rope_cast` deliberately stops at the attention epilogue and leaves the
  projection GEMM outside.
- **No measurement of whether the lightning indexer and SFA can share a kernel.** Both ops exist
  in `cann/ops-transformer`; no published fusion of the two, and no `nsa_*`/`lightning_indexer`
  fused variant was found.
- **No Ascend-published cost for `CrossCoreSetFlag` / `CrossCoreWaitFlag` / `SyncAll`.** The GPU
  analogue is documented (~60 ns per CUDA async barrier even in the pass state, per Hazy
  Research); the Ascend equivalent is the number needed to budget a megakernel's barrier count,
  and it was not found in any source.
- **`blockrt` / `Glm52MegaKernel` could not be located.** Named in the vllm-ascend RFC as
  Huawei's MegaKernel runtime (MK-Tile granularity, AI CPU coordinator + scoreboard +
  callbacks), but no public repository or documentation for it was found.
- **Cohere's megakernel blog (https://cohere.com/blog/megakernels, Sep 2026) could not be
  fetched** from this network (proxy interception), so nothing from it is quoted here.
- **No `aclnnSparseFlashMla` vs `aclnnSparseFlashAttention` comparison found.**
  `attention/sparse_flash_mla` and `attention/quant_sparse_flash_mla` exist with docs, but
  nothing states when the runtime picks one over the other, or whether `sparse_flash_mla` is
  A3-capable (`flash_mla` itself is 950-only).

---

## 8. `SparseFlashAttention` on A3: the host tiling never splits KV or G, and at batch=1 decode it runs on ONE cube core

Clone, not raw-URL fetch (`raw.gitcode.com` answers `429 Limited account:unregisteredUser`):

```
git clone https://gitcode.com/cann/ops-transformer.git
commit 3efc65dad2a97a4fea856c3be319da596a42fb8a
author L_Euler
date   Tue Oct 20 16:03:55 2026 +0800
subject 【BSA】aclnnBlockSparseAttentionV2 文档函数原型 qDequantScale/kDequantScale/vDequantScale 与头文件参数名不一致
```

Read 2026-10-20. All paths below are relative to the repo root; the browsable form is
`https://gitcode.com/cann/ops-transformer/blob/master/<path>`.

### 8.1 `FLASH_DECODE` is a declared template parameter that is NEVER instantiated

`attention/sparse_flash_attention/op_kernel/sparse_flash_attention_template_tiling_key.h` declares
the parameter with both values:

```cpp
ASCENDC_TPL_ARGS_DECL(SparseFlashAttention, // 算子OpType
                      ASCENDC_TPL_BOOL_DECL(FLASH_DECODE, 0, 1), ASCENDC_TPL_BOOL_DECL(PAGE_ATTENTION, 0, 1),
```

but every one of the eight entries in the `ASCENDC_TPL_SEL` combination list — the list the comment
says is what `GET_TPL_TILING_KEY` validates against — pins it to `0`:

```cpp
// 支持的模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(FLASH_DECODE, 0), ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 0),
                         ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_BSND),
                         ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_BSND, SFA_LAYOUT_PA_BSND),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, C_TEMPLATE),
                         ASCENDC_TPL_BOOL_SEL(IS_SPLIT_G, 0, 1), ASCENDC_TPL_BOOL_SEL(HAS_ROPE, 0, 1), ),
    ...
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(FLASH_DECODE, 0), ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 1),
                         ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_TND),
                         ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_TND, SFA_LAYOUT_PA_BSND),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, V_TEMPLATE),
                         ASCENDC_TPL_BOOL_SEL(IS_SPLIT_G, 0, 1), ASCENDC_TPL_BOOL_SEL(HAS_ROPE, 0, 1), ), );
```

(Eight `ASCENDC_TPL_ARGS_SEL` entries = {`PAGE_ATTENTION` 0,1} x {`LAYOUT_T` BSND,TND} x
{`TEMPLATE_MODE` C,V}. `FLASH_DECODE` is `0` in all eight; `1` appears nowhere.)

The host side closes the door with a literal. `op_host/sparse_flash_attention_tiling.cpp`,
`SFAMlaTiling::GenTilingKey()`:

```cpp
void SFAMlaTiling::GenTilingKey()
{
    uint32_t inputQType = static_cast<uint32_t>(sfaInfo_->inputQType);
    uint32_t inputKvType = static_cast<uint32_t>(sfaInfo_->inputKvType);
    uint32_t outputType = static_cast<uint32_t>(sfaInfo_->outputType);
    uint32_t layoutQuery = static_cast<uint32_t>(sfaInfo_->qLayout);
    uint32_t layoutKV = static_cast<uint32_t>(sfaInfo_->kvLayout);
    uint32_t pageAttention = 0U;
    if (sfaInfo_->kvLayout == SFALayout::PA_BSND) {
        pageAttention = 1U;
    }

    tilingKey_ = GET_TPL_TILING_KEY(
        0U, pageAttention, layoutQuery, layoutKV, perfMode_ == SFAPerfMode::V_TEMPLATE_MODE,
        static_cast<uint32_t>(sfaInfo_->gSize > 64),
        static_cast<uint32_t>(sfaInfo_->ropeHeadDim != 0)); // N1 > 64时核间切G; rope 用独立 tiling key 避免污染原路径

    OP_LOGI(sfaInfo_->opName, "SFA tilingKey_: %lu.", tilingKey_);
}
```

The first argument — the `FLASH_DECODE` slot — is the literal `0U`.

And the companion host flag is declared `false` and assigned nowhere in the entire operator
directory. `op_host/sparse_flash_attention_tiling.h:342`:

```cpp
    bool splitKVFlag_ = false;
```

`grep -rn 'splitKVFlag_ *=' attention/sparse_flash_attention/` returns exactly one line — that
declaration. Consequently in `FillTilingSplitKVMla()` the split count is always zeroed:

```cpp
// for flash decode
void SFAMlaTiling::FillTilingSplitKVMla()
{
    tilingData_.splitKVParams.set_s2(kvSplitPart_);

    tilingData_.splitKVParams.set_accumOutSize(aicNum_ * 2U * sfaInfo_->n2Size * mBaseSize_ *
                                               headDimAlign_); // 2:每个核可能有头规约和尾规约，一共两份规约信息
    tilingData_.splitKVParams.set_logSumExpSize(2 * aicNum_ * 2 * sfaInfo_->n2Size *
                                                mBaseSize_ * // 2:每个核可能有头规约和尾规约，一共两份规约信息;sum + max
                                                (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));

    if (!splitKVFlag_) {
        tilingData_.splitKVParams.set_s2(0);
    }
}
```

and the `CalcInnerSize` branch that would halve the S2 base block for decode is unreachable:

```cpp
void SFAMlaTiling::CalcInnerSize(uint32_t sfaS2Size)
{
    sInnerSize_ = 512; // 512:s2默认切分大小
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && sfaInfo_->qLayout != SFALayout::TND) {
        if (sfaS2Size == 256) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = 128;  // 128:sInnerSize_值为s2Size的一半，均分为2份进行计算，
        } else if (sfaS2Size > 256 && sfaS2Size <= sInnerSize_) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = (sInnerSize_ + 1) / 2;                  // 2:减半
        }
    }

    sInnerLoopTimes_ = (sfaS2Size + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = sfaS2Size - (sInnerLoopTimes_ - 1) * sInnerSize_;
```

So `sInnerSize_` is **always 512** on A3: a 512-token S2 base block, the prefill-shaped choice.

### 8.2 The host "split" is not a split: it assigns no work and claims every core

`op_host/sparse_flash_attention_tiling.cpp`:

```cpp
void SFAMlaTiling::SplitBalanced()
{
    CalcInnerSize(sfaInfo_->s2Size);

    InnerSplitParams sfaInnerSplitParams;
    sfaInnerSplitParams.s1GBaseSize = sfaInfo_->gSize;
    sfaInnerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.innerSplitParams.set_mBaseSize(sfaInnerSplitParams.s1GBaseSize);
    tilingData_.innerSplitParams.set_s2BaseSize(sfaInnerSplitParams.s2BaseSize);

    usedCoreNum_ = aicNum_;
}

void SFAMlaTiling::Split()
{
    SplitBalanced();
}
```

`mBaseSize` is set to the whole `gSize`, i.e. one base block covers all Q heads of the group.
`usedCoreNum_ = aicNum_` unconditionally. The real distribution is deferred to the kernel.

Core counts and the MIX ratio, from `GetPlatformInfo()` / `CalcBlockDim()`:

```cpp
    aivNum_ = sfaAscendcPlatform.GetCoreNumAiv();
    aicNum_ = sfaAscendcPlatform.GetCoreNumAic();
```

```cpp
void SFAMlaTiling::InitParams()
{
    if (sfaInfo_->s2Size != 0 && sfaInfo_->sparseBlockSize <= 4) { // 4:当前支持范围
        perfMode_ = SFAPerfMode::V_TEMPLATE_MODE;
    } else {
        perfMode_ = SFAPerfMode::C_TEMPLATE_MODE;
    }

    coreNum_ = aicNum_;
```

```cpp
void SFAMlaTiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
    auto aicNum = usedCoreNum_;
    auto aivNum = 2 * usedCoreNum_;

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(sfaInfo_->opName, "SFA block dim: %u aiv Num: %u aic Num: %u.", blockDim_, aivNum, aicNum);
}
```

```cpp
ge::graphStatus SFAMlaTiling::SetTilingKey(uint64_t tilingKey) const
{
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1); // 1: batchmode模式
    return ge::GRAPH_SUCCESS;
}
```

The AIC core-count ceiling, `op_host/sparse_flash_attention_tiling.h`:

```cpp
const uint32_t SFA_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐
```

and the fixed M base:

```cpp
    uint32_t mBaseSize_ = 128;
```

The kernel is declared MIX 1 AIC : 2 AIV, `op_kernel/sparse_flash_attention.cpp`:

```cpp
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
```

with the matching index arithmetic in `op_kernel/arch22/sparse_flash_attention_kernel_mla.h`:

```cpp
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }
```

The comments name the A3 shape exactly: **cube 0-23, vec 0-47**.

### 8.3 `InitCalcParamsEach` — the base block is (batch x kv-head x q-token), so batch=1 decode = 1 core

`op_kernel/arch22/sparse_flash_attention_kernel_mla.h`, verbatim:

```cpp
template <typename SFAT>
__aicore__ inline void SparseFlashAttentionMla<SFAT>::InitCalcParamsEach()
{
    // 计算总的基本块
    uint32_t actBatchS2 = 1;
    uint32_t totalBaseNum = 0;
    uint32_t s1GBaseSize = sfaKernelConstInfo.gSize;
    uint32_t coreNum = GetBlockNum();
    uint32_t currCoreIdx = aiCoreIdx;
    uint32_t actBatchS1 = 1;
    for (uint32_t bIdx = 0; bIdx < sfaKernelConstInfo.batchSize; bIdx++) {
        uint32_t actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
        if (actBatchS1 < sfaKernelConstInfo.qSeqSize) {
            sfaKernelConstInfo.needInit = true;
        }
        totalBaseNum += actBatchS1 * actBatchS2;
    }
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
    } else {
        usedCoreNum = totalBaseNum;
    }
    if (aiCoreIdx >= usedCoreNum) {
        return;
    }
    // 计算当前核的基本块
    uint32_t accumBaseNum = 0; // 当前累积的基本块数
    uint32_t targetBaseNum = 0;
    uint32_t lastValidBIdx = 0;
    uint32_t lastValidactBatchS1 = 0;
    bool setStart = false;
    targetBaseNum = (currCoreIdx + 1) * avgBaseNum; // 计算当前的目标权重
    uint32_t targetStartBaseNum = targetBaseNum - avgBaseNum;
    for (uint32_t bN2Idx = 0; bN2Idx < sfaKernelConstInfo.batchSize * sfaKernelConstInfo.kvHeadNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / sfaKernelConstInfo.kvHeadNum;
        actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
        for (uint32_t s1GIdx = 0; s1GIdx < actBatchS1; s1GIdx++) {
            accumBaseNum += 1;
            if (!setStart && accumBaseNum >= targetStartBaseNum) {
                sfaKernelConstInfo.bN2Start = bN2Idx;
                sfaKernelConstInfo.gS1Start = s1GIdx;
                setStart = true;
            }
            if (accumBaseNum >= targetBaseNum) {
                // 更新当前核的End分核信息
                sfaKernelConstInfo.bN2End = bN2Idx;
                sfaKernelConstInfo.gS1End = s1GIdx;
                sfaKernelConstInfo.s2End = 0;
                sfaKernelConstInfo.coreStartKVSplitPos = 0;
                if (aiCoreIdx != 0) {
                    GetAxisStartIdx(sfaKernelConstInfo.bN2Start, sfaKernelConstInfo.gS1Start, 0);
                }
                return;
            }
        }
```

Three literals decide everything here:

- `uint32_t actBatchS2 = 1;` — the S2 (KV) axis contributes a factor of exactly one to
  `totalBaseNum`. KV length is not a parallel axis.
- `totalBaseNum += actBatchS1 * actBatchS2;` over `bIdx < batchSize` — so
  `totalBaseNum = sum_b (q_tokens_in_batch_b)`.
- `else { usedCoreNum = totalBaseNum; }` followed by `if (aiCoreIdx >= usedCoreNum) { return; }`
  — when `totalBaseNum <= coreNum`, `usedCoreNum` is *lowered* to `totalBaseNum` and every core
  above that index returns before doing any work.

For MLA, `kvHeadNum == 1`; the per-core work unit is one `(bN2Idx, s1GIdx)` pair, and
`mBaseSize == gSize` means all Q heads of the group live in that single unit. So for
**batch=1, one query token, MLA**: `totalBaseNum = 1` → `usedCoreNum = 1` → cores `1..23` take
the early `return`, and the entire sparse attention for that layer executes on cube core 0 plus
its two paired vector cores.

`IS_SPLIT_G` does not rescue this. From `GenTilingKey()` above it is
`static_cast<uint32_t>(sfaInfo_->gSize > 64)`, with the comment `N1 > 64时核间切G`. Splitting
over heads only switches on above 64 heads *per rank*. At TP16 there is no configuration of any
published GLM/DeepSeek MLA model that leaves more than 64 Q heads on a rank.

### 8.4 The workspace confirms a GM staging round trip for the per-token gather

`op_host/sparse_flash_attention_tiling.cpp`, `GetWorkspaceSize()`, the non-A5 (A3) branch:

```cpp
        workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * sfaMmResElemSize);
        workspaceSize_ +=
            preLoadNum *
            static_cast<size_t>(static_cast<float>(mmResUbSize_ * actCoreNum * sfaVec1ResElemSize) * sfaKvDtypeRatio);
        workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * sfaBmm2ResElemSize;
        workspaceSize_ +=
            preLoadNum * static_cast<size_t>(static_cast<float>(qPreSizeMla_ * actCoreNum * sfaQPreProcResElemSize) *
                                             sfaKvDtypeRatio);
        workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * sfaNUpdateElemSize;
        workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * sfaSoftmaxSumElemSize;
        // topk BlkSize == 1场景, 需要额外空间缓存离散聚合的值
        //              bufNum  s2Base   D   dRope  sizeOf(half)
        // 4:bufNum  512:s2Base  512:D  64:dRope  2:sizeOf(half)
        workspaceSize_ += 4 * 512 * (512 + 64) * 2 * actCoreNum;
        // 缓存有效mte2 size的长度 份数  512B对齐的长度  sizeof(int32_t)   aiv核数
        workspaceSize_ +=
            4 * 128 * 4 * (2 * actCoreNum); // 4:缓存有效mte2 size的长度 128:份数  4:512B对齐的长度  2:aiv核数
```

`topk BlkSize == 1场景, 需要额外空间缓存离散聚合的值` — "the top-k block-size-1 case needs extra
space to cache the discretely-gathered values". 4 buffers x 512 S2 slots x 576 dims x 2 B per
core = 2.25 MB per core of HBM workspace, and the gathered KV is written there and read back.
That buffer is `kvMergeGm_` in the kernel. The second line caches "the length of the valid MTE2
size", 4 copies x 128 entries x 4 B, for `2 * actCoreNum` vector cores.

The matching `PRE_LOAD_NUM`:

```cpp
constexpr uint32_t PRE_LOAD_NUM = 2;
```

### 8.5 The fixed per-core budget: 512-row S2 workspace, 64x576 L1 blocks, 128x128 cube tiles

`op_kernel/arch22/sparse_flash_attention_service_cube_mla.h`:

```cpp
    static constexpr uint32_t M_SPLIT_SIZE = 128;     // m方向切分
    static constexpr uint32_t N_SPLIT_SIZE = 128;     // n方向切分
    static constexpr uint32_t N_WORKSPACE_SIZE = 512; // n方向切分

    static constexpr uint32_t L1_BLOCK_SIZE = (64 * (512 + 64) * sizeof(Q_T));
    static constexpr uint32_t L1_BLOCK_OFFSET = 64 * (512 + 64); // 72K的元素个数

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);
```

```cpp
    constexpr uint32_t MERGE_K_PITCH = 576;
```

`M_SPLIT_SIZE = 128` is the cube M tile. At batch=1 decode the real M is `gSize` — 4 to 8 rows at
TP16 — against a 128-row tile. `L1_BLOCK_SIZE = 64 * 576 * 2 = 72 KiB` per L1 block, `L0C_PP_SIZE
= 64 KiB`.

The cube/vector sync event budget, same file:

```cpp
    static constexpr uint32_t L1_EVENT0 = EVENT_ID2;
    static constexpr uint32_t L1_EVENT1 = EVENT_ID3;
    static constexpr uint32_t L1_EVENT2 = EVENT_ID4;
    static constexpr uint32_t L1_EVENT3 = EVENT_ID5;
    static constexpr uint32_t L1_EVENT4 = EVENT_ID6;
    static constexpr uint32_t L1_EVENT5 = EVENT_ID7;
    static constexpr uint32_t L1_EVENT6 = EVENT_ID1;

    static constexpr uint32_t mte21QPIds[4] = {L1_EVENT0, L1_EVENT1, L1_EVENT2, L1_EVENT3}; // mte12复用
    static constexpr uint32_t sfaMte21KvEvents[3] = {L1_EVENT4, L1_EVENT5, L1_EVENT6};
```

and the cross-core sync IDs that a megakernel would have to reproduce,
`op_kernel/arch22/sparse_flash_attention_common.h`:

```cpp
struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t SFA_SYNC_MODE2 = 2;
    ...
    // CUBE和VEC的核间同步EventID
    uint32_t syncV1NupdateC2 = 0U;
    uint32_t syncV0C1 = 0U;
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C2 = 0U;
    uint32_t syncC2V2 = 0U;
    uint32_t syncC2V1 = 0U;
```

`splitKVNum` is in that struct and documented as the inter-core S2 split count — the field exists,
it is just always fed 0:

```cpp
    uint32_t splitKVNum = 0U;         // S2核间切分的切分份数
    ...
    // FlashDecoding
    uint32_t actualCombineLoopSize = 0U; // FlashDecoding场景, S2在核间切分的最大份数
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;
```

### 8.6 The template mode switch: `sparseBlockSize <= 4` picks `V_TEMPLATE`

```cpp
    if (sfaInfo_->s2Size != 0 && sfaInfo_->sparseBlockSize <= 4) { // 4:当前支持范围
        perfMode_ = SFAPerfMode::V_TEMPLATE_MODE;
    } else {
        perfMode_ = SFAPerfMode::C_TEMPLATE_MODE;
    }
```

with the `sparse_block_size` validity constraints from the same file:

```cpp
    OP_CHECK_IF((npuArch_ == NpuArch::DAV_3510 && *opParamInfo_.sparseBlockSize != 1),
                ...
    OP_CHECK_IF((*opParamInfo_.sparseBlockSize <= 0 || *opParamInfo_.sparseBlockSize > 128 ||
                 (static_cast<uint64_t>(*opParamInfo_.sparseBlockSize) &
                  static_cast<uint64_t>(*opParamInfo_.sparseBlockSize - 1L)) != 0UL),
```

and

```cpp
    OP_CHECK_IF(blockSize_ % sparseBlockSize_ > 0,
                ... ") must be divided by sparse_block_size(" + std::to_string(sparseBlockSize_) +
```

So: `sparse_block_size` must be a power of two in `[1, 128]`; the PageAttention `block_size` must
be a multiple of it; on `DAV_3510` (A5/950) it must be exactly 1; and a DSA-style per-token
selection (`sparse_block_size = 1`) lands in `V_TEMPLATE_MODE`.

## Relevance

This is the answer to the question the earlier Gaps entry left open, and it is a *structural*
answer rather than a data-movement one.

`SparseFlashAttention` on A3 has **no decode specialization at all**. The three things that would
give a batch=1 single-query decode its parallelism are each disabled by a constant:

| lever | state on A3 | where |
|---|---|---|
| split-KV across cores (`FLASH_DECODE`) | `GET_TPL_TILING_KEY(0U, ...)` literal; never in `ASCENDC_TPL_SEL`; `splitKVFlag_` assigned nowhere | `GenTilingKey()`, `template_tiling_key.h` |
| split over Q heads (`IS_SPLIT_G`) | only when `gSize > 64`, i.e. >64 heads per rank | `GenTilingKey()` |
| S2 base block size | `sInnerSize_ = 512` always, decode halving branch unreachable | `CalcInnerSize()` |

and the kernel's own distribution counts base blocks as `sum_b q_tokens_b` with
`actBatchS2 = 1`, then *lowers* `usedCoreNum` to that count. At batch=1, Q_S=1, MLA
(`kvHeadNum=1`) that count is **1**. So the op that costs us 6.973 ms — the largest family in the
round — is running on 1 of 24 cube cores while 23 sit in an early `return`.

This reframes our own decomposition. Of the four candidate causes in the brief, this is (c)
*insufficient parallelism across the AI cores at one query token*, and it is not a soft effect: it
is a hard 1/24. It subsumes (d) as well — the 512-row `s2BaseSize` and the 128-row `M_SPLIT_SIZE`
cube tile are prefill-shaped constants applied to a 1-token query, so the first S2 block is also
the tail block, and the cube runs a 128x128 tile to produce `gSize`(=4..8) useful rows.

Two concrete levers follow, both available to a hand-written engine without touching CANN:

1. **Supply the parallelism the op refuses to find.** Because the base block is
   `(batch, kv_head, q_token)` and `usedCoreNum` is just `totalBaseNum`, presenting the same
   decode work as `totalBaseNum >= 24` — splitting the selected KV list into 24 chunks ourselves
   and reducing the partial `(out, lse)` outside, which is exactly what the dead `FLASH_DECODE`
   path would have done with `accumOutSize` / `logSumExpSize` — recovers the cores. The op even
   exposes `softmaxMax` / `softmaxSum` outputs and a `returnSoftmaxLse` attribute to make the
   external combine possible. The alternative framing is to fake `q_tokens` by packing the
   `gSize` heads along S1 in TND layout, which turns `gSize` into `totalBaseNum`.
2. **Stop paying the MIX launch floor per layer.** `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`
   makes every `SparseFlashAttention` call a MIX_AIC launch — our measured 13.58 us floor, 6.5
   MIX_AIC calls per layer, 11.5 ms of launch floor per round. A one-core kernel paying a
   13.58 us MIX launch is the worst possible trade.

And one number to check against ours: 1/24 of the cores predicts a gap far larger than the 5.0x
we measure (6.973 ms spent vs 1.382 ms of bytes). Either the 1.382 ms roofline is computed
against a bandwidth a single core can nearly saturate, or our call is not shaped the way
`InitCalcParamsEach` sees batch=1 — e.g. we pass TND with the heads packed along T, which would
make `totalBaseNum = gSize` and use `gSize` cores rather than 1. Resolving which is the single
highest-value profiling question: read `usedCoreNum` out of the `OP_LOGI` line
`SFA block dim: %u aiv Num: %u aic Num: %u.` in plog, or count active cores in the Level-2
profile. If it really is 1, there is far more than 5.6 ms available here.

---

## 9. The `V_TEMPLATE` gather: a mandatory HBM round trip, 2 tokens per DMA, 32-row UB staging

Same clone and commit as section 8 (`gitcode.com/cann/ops-transformer`,
`3efc65dad2a97a4fea856c3be319da596a42fb8a`, read 2026-10-20). Files:
`attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_vector_mla.h`
and `..._service_cube_mla.h`.

### 9.1 In `V_TEMPLATE` the cube never touches the KV cache — it reads a merged GM buffer

`sparse_flash_attention_service_cube_mla.h`, `ComputeMm1NoRope`. The `V_TEMPLATE` branch is a
single contiguous `Nd2NzParams` copy out of `kvMergeGm_`; only the `else` (= `C_TEMPLATE`) branch
walks the top-k list:

```cpp
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                Nd2NzParams sfaCubeCopyParams;
                sfaCubeCopyParams.ndNum = 1;
                sfaCubeCopyParams.nValue = mm1NrNL1Size;
                sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDim >> 1;
                sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDim;
                sfaCubeCopyParams.dstNzC0Stride = mm1NrNL1SizeAlign;
                sfaCubeCopyParams.dstNzNStride = 1;
                sfaCubeCopyParams.srcNdMatrixStride = 0;
                sfaCubeCopyParams.dstNzMatrixStride = 0;
                DataCopy(mm1NrBL1Tensor,
                         kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * MERGE_K_PITCH +
                                    mm1NrKL1 * (sfaCubeConstInfo.headDim >> 1) +
                                    mm1NrNL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                         sfaCubeCopyParams);
            } else {
                while (copyFinishRowCnt < mm1NrNL1Size) {
                    CalcTopKBlockInfo(sfaCubeRunInfo, curTopKIdx, curOffsetInSparseBlock, curSeqIdx, copyRowCnt,
                                      idInTopK);
```

`kvMergeGm_` is the HBM workspace that section 8.4 budgets as
`4 * 512 * (512 + 64) * 2 * actCoreNum` with the comment
`topk BlkSize == 1场景, 需要额外空间缓存离散聚合的值`. So in the DSA case
(`sparse_block_size <= 4` → `V_TEMPLATE`) the data path is:

```
KV cache in HBM  --(scattered MTE2, AIV)-->  UB (kvMergUb_)
UB               --(MTE3, AIV)-----------> kvMergeGm_ in HBM     <-- extra write
kvMergeGm_       --(contiguous MTE2, AIC)-> L1                   <-- extra read
```

Every selected KV token is read from HBM once, written back to HBM once, and read from HBM again.
Three HBM transits where a roofline model counts one.

### 9.2 The gather loop: one `topkGm_.GetValue()` scalar GM read per selected token

`sparse_flash_attention_service_vector_mla.h`:

```cpp
template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::GetRealS2Idx(int64_t s2GmOffset, int64_t& realS2Idx,
                                                            int64_t topkGmBaseOffset, const RunInfo& sfaVecRunInfo)
{
    int64_t topkGmIdx =
        (s2GmOffset + sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize) / sfaVecConstInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= sfaVecConstInfo.sparseBlockCount)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(sfaVecConstInfo.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize) %
                                     sfaVecConstInfo.sparseBlockSize);
}
```

and in PageAttention mode a *second* scalar GM read per token, for the block table:

```cpp
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableIdx = realS2Idx / sfaVecConstInfo.kvCacheBlockSize;
        int64_t blkTableOffset = realS2Idx % sfaVecConstInfo.kvCacheBlockSize;
        realKeyGmOffset = blkTableGm_.GetValue(sfaVecRunInfo.bIdx * sfaVecConstInfo.maxBlockNumPerBatch + blkTableIdx) *
                              static_cast<int64_t>(sfaVecConstInfo.kvCacheBlockSize) *
                              static_cast<int64_t>(sfaVecConstInfo.kvHeadNum) +
                          blkTableOffset;
    } else {
```

At `sparse_block_size = 1` this is `2 * top_k` scalar GM loads per (head-group, layer), each on the
AIV scalar pipe, each in the dependency chain ahead of the `DataCopyPad` that uses its result.

### 9.3 `MergeKv`: the main loop advances TWO tokens at a time and flushes every 32 rows

```cpp
// b s1 k
template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::MergeKv(const RunInfo& sfaVecRunInfo)
{
    int64_t s2ProcessSize = sfaVecRunInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * sfaVecConstInfo.sparseBlockSize);
```

```cpp
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * sfaVecConstInfo.sparseBlockSize;
    int64_t s2GmLimit =
        GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * sfaVecConstInfo.sparseBlockSize : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;
         s2GmOffsetArray += 2 * sfaVecConstInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, sfaVecRunInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + sfaVecConstInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset, sfaVecRunInfo);
        CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, sfaVecRunInfo);
        if ((mte2Size - mte3Size + 2 * sfaVecConstInfo.sparseBlockSize > 32) ||
            s2GmOffsetArray + 2 * sfaVecConstInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
    }
```

The loop stride is `2 * sparseBlockSize` — **two selected tokens per iteration**. The UB staging
buffer holds 32 rows (`mte2Size - mte3Size + 2 * sparseBlockSize > 32` triggers the flush), and
the UB offsets confirm the size: `kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + ...]` is a double-buffered
32x512 tile, `ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + ...]` a 32x64 tile. So at
`sparse_block_size = 1` the pipeline is: 16 loop iterations (32 tokens) of MTE2, then one MTE3
flush, with a two-deep `MTE3_MTE2` ping-pong and an explicit `SetFlag`/`WaitFlag` pair in between.

The GM merge buffer rotates over 4 slots:

```cpp
    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
```

### 9.4 `CopyInKv`: the whole optimization is folding exactly 2 scattered tokens into one `DataCopyPad`

```cpp
    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 ||
                 (!PAGE_ATTENTION && (keyRopeSrcStride >= INT32_MAX || keyRopeSrcStride < 0)) ||
                 realS2Idx1 + sfaVecConstInfo.sparseBlockSize >= s2IdLimit ||
                 realS2Idx2 + sfaVecConstInfo.sparseBlockSize >= s2IdLimit)) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, sfaVecRunInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, sfaVecRunInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockLen = sfaVecConstInfo.sparseBlockSize * sfaVecConstInfo.headDim * sizeof(KV_T);
        intriParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyOffset1 > -1 ? keyOffset1 : keyOffset2;
        if (keyOffset2 > -1 && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + (mte2Size - mte3Size) * sfaVecConstInfo.headDim],
                    keyGm_[startGmOffset * sfaVecConstInfo.headDim], intriParams, padParams);

        if constexpr (SFAT::hasRope) {
            intriParams.blockLen = sfaVecConstInfo.sparseBlockSize * sfaVecConstInfo.headDimRope * sizeof(KV_T);
            intriParams.dstStride = 0;
            intriParams.srcStride = keyRopeSrcStride;
            DataCopyPad(ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + (mte2Size - mte3Size) * sfaVecConstInfo.headDimRope],
                        keyRopeGm_[startGmOffset * sfaVecConstInfo.headDimRope], intriParams, padParams);
        }
        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1)) * sfaVecConstInfo.sparseBlockSize;
    }
```

This is the entirety of the `搬运聚合` ("transfer aggregation") that the
`kv_quant_sparse_flash_attention` README advertises: `blockCount = 2`, `srcStride` = the gap
between the two chosen tokens. Two tokens per MTE2 instruction and no more, because `DataCopyPad`
takes a single uniform `srcStride` and three arbitrary top-k indices do not share one.

`blockLen` per instruction at MLA/DSA shape: `1 * 512 * 2 = 1024` bytes for nope, and a separate
instruction of `1 * 64 * 2 = 128` bytes for rope. With `blockCount = 2`: 2 KiB and 256 B per pair.
The rope copy is a **128-byte** burst.

Note also the `还原成2条搬运指令` ("fall back to 2 transfer instructions") path: whenever the gap
between the two chosen tokens is negative in the index order, or either token is near the sequence
end, the pairing is abandoned and it becomes one instruction per token. Since the top-k index list
is not guaranteed sorted ascending, `keySrcStride < 0` is a live condition.

### 9.5 `CopyOutMrgeResult` — the write-back, 512 B + 128 B per row

```cpp
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = mte2Size - mte3Size;
    dataCopyParams.blockLen = sfaVecConstInfo.headDim * sizeof(KV_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % 4 * 512 * 576 + (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDim],
                kvMergUb_[mergeMte3Idx % 2 * 32 * 512], dataCopyParams);

    if constexpr (SFAT::hasRope) {
        dataCopyParams.blockLen = sfaVecConstInfo.headDimRope * sizeof(KV_T);
        DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % 4 * 512 * 576 + 512 * 512 +
                               (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDimRope],
                    ropeMergUb_[mergeMte3Idx % 2 * 32 * 64], dataCopyParams);
    }
```

with a hard MTE2→MTE3 barrier immediately before it:

```cpp
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
```

`SetFlag` immediately followed by `WaitFlag` on the same event is a full serialization point, not
a pipelined handoff: the MTE3 flush cannot overlap the MTE2 fill of the same buffer.

### 9.6 The zero-fill tail: one `DataCopyPad` per *missing* row

When fewer than `s2ProcessSize` tokens were found (the usual case at decode when top-k exceeds
the real sequence length, or when the index list terminates with `-1`), the remainder of the 512-row
merge buffer is zeroed one row at a time:

```cpp
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                   s2GmOffset * sfaVecConstInfo.headDim],
                        kvMergUb_, dataCopyParams);
        }
        if constexpr (SFAT::hasRope) {
            dataCopyParams.blockLen = sfaVecConstInfo.headDimRope * sizeof(KV_T);
            for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
                DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                       512 * sfaVecConstInfo.headDim + s2GmOffset * sfaVecConstInfo.headDimRope],
                            kvMergUb_, dataCopyParams);
            }
        }
```

`dataCopyParams.blockCount = 1` (set just above), so these are **single-row, 1 KiB (then 128 B)
MTE3 instructions, one per padded row** — and the cube still multiplies against those zero rows
because `mm1NrNL1Size` is derived from `actualSingleProcessSInnerSize`, not from `mte2Size`.

### 9.7 The cube serializes its own M pipe whenever the Mmad tile is small

`sparse_flash_attention_service_cube_mla.h`, inside the Mmad loop:

```cpp
                    MmadParams mm1NrMmadParams;
                    mm1NrMmadParams.m = mm1NrML1SizeAlign;
                    mm1NrMmadParams.n = mm1NrNL1SizeAlign;
                    mm1NrMmadParams.k = mm1NrKL0Size;
                    mm1NrMmadParams.cmatrixInitVal = (mm1NrKL1 == 0 && mm1NrKL0 == 0);
                    mm1NrMmadParams.cmatrixSource = false;
                    mm1NrMmadParams.unitFlag = (mm1NrKL1 == 1 && mm1NrKL0 == (mm1NrKL0Loops - 1)) ? 0b11 : 0b10;
                    Mmad(mm1NrCL0Tensor, mm1NrAL0Tensor, mm1NrBL0Tensor, mm1NrMmadParams);
                    if ((mm1NrMmadParams.m / 16) * (mm1NrMmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
```

with the K split fixed at:

```cpp
    constexpr uint32_t mm1NrKL1Loops = 2;
    constexpr uint32_t mm1NrKL0Size = 128;
    constexpr uint32_t mm1NrKL0Loops = 2;
```

At batch=1 decode the M of the first matmul is `gSize` rounded up to 16. With `N_SPLIT_SIZE = 128`:
`(16/16) * (128/16) = 8 < 10`, so `PipeBarrier<PIPE_M>()` fires on **every single `Mmad`** —
4 Mmads per (M-tile, N-tile) from `mm1NrKL1Loops * mm1NrKL0Loops`. The guard exists to work
around a small-tile hazard, and decode shape sits permanently on the wrong side of it. The cube's
M pipe is fully drained between consecutive 16x128x128 Mmads.

## Relevance

This answers (b) and (d) of the four candidates, quantitatively, and it changes which number we
should be comparing against.

**The roofline denominator is wrong by ~3x for the DSA path.** Our 1.382 ms is the time to move
the bytes the algorithm needs. Section 9.1 shows `V_TEMPLATE` moves each selected KV token across
HBM three times — scattered read into UB, MTE3 write to `kvMergeGm_`, contiguous read back into
L1. So the *op as written* has a floor near 3 x 1.382 = 4.15 ms, not 1.382 ms. Our measured
6.973 ms is 1.68x that floor, not 5.0x. That reframes the opportunity: the 5.6 ms we hoped to
recover by "closing to roofline" is not available inside this operator at all. About 2.8 ms of it
is the staging round trip, and only a hand-written kernel that keeps the gathered KV in UB/L1 and
feeds the cube directly can claim it.

**The per-instruction burst lengths are small and fixed.** Nope: 1024 B x `blockCount=2`. Rope:
128 B x `blockCount=2`. The merge write-back: 1024 B and 128 B per row. The zero-pad tail: single
1024 B and 128 B instructions, one per row, no batching. A 128-byte MTE2/MTE3 burst is well under
any plausible efficient DMA length, and the rope stream alone is `top_k` instructions of 128 B
per head-group per layer.

**The aggregation ceiling is 2.** `CopyInKv` pairs exactly two tokens per `DataCopyPad` because
`DataCopyExtParams` carries one `srcStride`. Three or more arbitrary top-k indices cannot share
an instruction. So at `sparse_block_size = 1` the instruction count is `top_k / 2` MTE2 pairs
plus the same again for rope — and halves to `top_k` whenever the `还原成2条搬运指令` fallback
trips, which it does whenever the index list is not ascending. If our indexer emits unsorted
top-k, sorting it on the host or in the indexer kernel is a free 2x on MTE2 instruction count.

**Two hard serializations per 32 tokens plus one per Mmad.** The `SetFlag`/`WaitFlag` pair on
`MTE2_MTE3` in `CopyOutMrgeResult` (9.5) prevents overlap of gather and flush within a buffer;
the `PipeBarrier<PIPE_M>()` in 9.7 fires on every Mmad at decode M. Neither is visible as bytes.
Both show up as `aic_mte2_time` / `aiv_mte2_time` far below the kernel wall time, which is the
signature to look for in our own Level-2 profile: if `mte2_ratio` is low *and* `aic_mac_ratio` is
low, the time is in barriers and scalar address generation, not in either pipe.

**Three levers, in descending value:**
1. Eliminate the `kvMergeGm_` round trip. This is the ~2.8 ms item and it is only reachable by
   owning the kernel: gather scattered KV straight into L1 (the `C_TEMPLATE` branch already does
   this, via `CopyInMm1BToL1` / `DataCopyPA`) instead of through GM. The reason upstream went
   through GM at `sparse_block_size <= 4` is that the cube cannot issue `top_k` separate L1 loads
   without stalling its own MTE2 — but with the KV gather on the AIV and a UB→L1 path, or with
   more than 2 tokens per instruction, that constraint is ours to re-examine.
2. Sort the top-k index list ascending. Keeps `CopyInKv` on the paired path instead of the
   `还原成2条搬运指令` fallback, halving MTE2 instruction count.
3. Do not pad to 512. `s2BaseSize` is fixed at 512 (section 8.1) and the shortfall is zero-filled
   one row at a time, then multiplied. At seq 8K with a top-k below 512 per S2 block, the padding
   rows are pure waste in MTE3 instructions, in HBM traffic, and in cube Mmad.

---

## 10. `IncreFlashAttention` vs `SparseFlashAttention`: same vendor, same A3, 16 cores vs 1

Same clone and commit (`gitcode.com/cann/ops-transformer`,
`3efc65dad2a97a4fea856c3be319da596a42fb8a`, read 2026-10-20).
`attention/incre_flash_attention/op_host/incre_flash_attention_tiling_check.cpp` and
`..._tiling.cpp`.

This is the A/B the brief asked for, and it is in source rather than in a benchmark: the dense
decode attention op on the same chip implements the split-KV flash-decode that
`SparseFlashAttention` leaves stubbed out (section 8.1).

### 10.1 `IFATiling::IsFlashDecode` — the live heuristic

```cpp
bool IFATiling::IsFlashDecode(uint32_t coreNum, IfaPerfMode perfMode)
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        perfMode = IfaPerfMode::BMM_ALL_BY_VEC;
        if (nNumOfQInOneGroup_ * qSeqSize_ > 1U) {
            OP_LOGD(ifaContext_->opName, "flash decode has not been implemented when 310P and GQA > 1.");
            return false;
        }
    }
    if (balanceModeFlag_) {
        uint32_t tndFDCoreArrLen = tilingDataMla_.tndSplitCoreParams.get_tndFDCoreArrLen();
        return tndFDCoreArrLen > 0U;
    }
    if (pageAttentionFlag_ && socVersion_ == IfaSocVersion::SOC_ASCEND_910B &&
        perfMode != IfaPerfMode::CUBE_VIEW_MM_MLA && maxActualseq_ <= 1024U) { // 1024, 经验值
        return false;
    }
    constexpr uint32_t maxQNum = 128U;
    if (ropeFlag_ && (qSeqSize_ > 1U || nNumOfQInOneGroup_ > maxQNum)) {
        return false; // 投机推理 和 G > 64 切G待实现
    }

    if (gqaMtpFlag_) {
        return false; // 投机推理FD待实现
    }

    if (CheckCoreOkFlag(coreNum, perfMode)) {
        return true;
    }

    return false;
}
```

```cpp
bool IFATiling::CheckCoreOkFlag(uint32_t coreNum, IfaPerfMode perfMode) const
{
    bool coreOkFlag = false;
    float flashDecodeBNRatio = static_cast<float>(0.4); // 0.4, 经验值
    if (perfMode == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD || perfMode == IfaPerfMode::CUBE_VIEW_MM ||
        perfMode == IfaPerfMode::CUBE_VIEW_MM_MLA || perfMode == IfaPerfMode::CUBE_VIEW_MM_DD) {
        flashDecodeBNRatio = 0.5; // 0.5, 经验值
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) <= flashDecodeBNRatio * coreNum);
    } else if (perfMode == IfaPerfMode::BMM_ALL_BY_VEC) {
        flashDecodeBNRatio = 0.5; // 0.5, 全V模板可以按0.5切分
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    } else {
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    }

    if (coreOkFlag && (nNumOfQInOneGroup_ == 1U)) {
        OP_LOGD(ifaContext_->opName, "Flash Decode Split kv."); // 非gqa时这里只判断bn是否满足
        return true;
    }

    if (coreOkFlag && (maxActualseq_ >= 2048U)) { // 2048, 在flash decode + gqa时的经验值
        OP_LOGD(ifaContext_->opName, "Flash Decode And GQA Split kv.");
        return true;
    }
```

The trigger condition is stated as a ratio against the core count: flash-decode turns on when
`batch * kv_heads <= 0.5 * coreNum`. The *fewer* the batch-and-head work units, the *more* the op
splits KV. That is the exact inverse of what `SparseFlashAttention`'s `InitCalcParamsEach` does
(section 8.3), which lowers `usedCoreNum` to `batch * q_tokens` and leaves the rest idle.

Also note `IfaPerfMode::CUBE_VIEW_MM_MLA` is in the 0.5-ratio list and that the only MLA-specific
exclusion is speculative decoding (`qSeqSize_ > 1U`) or `G > 128`:

```cpp
enum class IfaPerfMode : uint32_t {
    NORMAL = 0,
    BMM_ALL_BY_VEC,         // 1
    C1_V1,                  // 2
    CUBE_VIEW_MM,           // 3
    CUBE_VIEW_MM_FULL_LOAD, // 4
    CUBE_VIEW_MM_MLA,       // 5
    CUBE_VIEW_MM_DD         // 6
};
```

### 10.2 `IFATiling::SplitBNS` — the split count, and the 512-token-per-core empirical floor

```cpp
ge::graphStatus IFATiling::SplitBNS()
{
    formerCoreNum_ = 0U;
    blockSplitBn2Range_ = 1U;
    tailSplitedBatchRange_ = 1U;

    uint32_t bn = batchSize_ * numKvHeads_;
    kvSplitPart_ = coreNum_ / bn;
    while (((maxActualseq_ / kvSplitPart_) < 512U) && (kvSplitPart_ > 1U)) { // 512, 经验值
        kvSplitPart_--;
    }

    usedCoreNum_ = bn * kvSplitPart_;
    uint32_t computeSeqSize = (seqSize_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    if (pageAttentionFlag_) {
        computeSeqSize = Align(computeSeqSize, blockSize_);
    }
    if (inputKvType_ == ge::DT_INT4) {
        computeSeqSize = Align(computeSeqSize, 2U);
    }
    CalcInnerSize(computeSeqSize);
    return ge::GRAPH_SUCCESS;
}
```

and the flag being set, from `incre_flash_attention_tiling.cpp` (three separate call sites plus
one unconditional one):

```cpp
            OP_LOGI(ifaContext_->opName, "aicNum_(use for usedCoreNum_ and kvSplitPart_):%u\n", aicNum_);
            ...
            splitKVFlag_ = true;
            kvSplit_++;
            kvSplitPart_ = aicNum_;
```

```cpp
    if (IsFlashDecode(coreNum_, perfMode_)) {
        splitKVFlag_ = true;
        kvSplit_++;
        kvSplitPart_ = tilingInfo.maxKvSplitPart;
```

**Worked for our shape.** batch=1, MLA so `numKvHeads_ = 1`, `coreNum_ = 24` AICs, seq 8192:

- `bn = 1 * 1 = 1`
- `kvSplitPart_ = 24 / 1 = 24`
- loop: `8192 / 24 = 341 < 512` → decrement. `8192/23 = 356`, ... `8192/17 = 481 < 512`,
  `8192/16 = 512` → stop. **`kvSplitPart_ = 16`**
- `usedCoreNum_ = 1 * 16 = 16`
- `computeSeqSize = ceil(8192/16) = 512` KV tokens per core

And the gate passes: `CheckCoreOkFlag` with `CUBE_VIEW_MM_MLA` gives `1 <= 0.5 * 24 = 12` → true;
`nNumOfQInOneGroup_ > 1` for MLA so it falls through to `maxActualseq_ >= 2048` → `8192 >= 2048`
→ true. `IsFlashDecode` is not blocked by `ropeFlag_` because `qSeqSize_ == 1` and
`nNumOfQInOneGroup_ <= 128`.

So on the *identical* decode shape: `IncreFlashAttention` uses **16 of 24 cube cores**;
`SparseFlashAttention` uses **1**.

The `512U // 512, 经验值` constant is itself a published number worth having: Huawei's own
empirical floor for the minimum KV chunk worth giving a core in a decode attention split. Below
512 tokens per core they stop splitting. Note this is the same 512 that
`SparseFlashAttention`'s `sInnerSize_ = 512` uses as its *S2 base block*, i.e. the two ops agree
on 512 as the unit — one of them distributes those units across cores, the other does not.

### 10.3 The combine kernel SFA lacks exists for IFA on A3

`attention/incre_flash_attention/op_kernel/arch22/` contains, at 845 lines:

```
service_vector_flashdecode_mla.h
```

with the reduction interface:

```cpp
class ServiceFlashDecode {
...
    __aicore__ inline void InitGlobalTensor(GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm,
                                            GlobalTensor<T> accumOutGm, GlobalTensor<OUT_T> attentionOutGm,
    __aicore__ inline void InitParams(const AttentionCommonFlashDecode::ConstInfoFD& constInfo);
    __aicore__ inline void InitDecodeParams();
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void FlashDecode(FDparams& fd);
...
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T>& accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, int64_t baseOffset, uint32_t cntM);
```

and the per-row rescale primitives it is built from:

```cpp
__aicore__ inline void MatDivsVecFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
__aicore__ inline void RowSubFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
__aicore__ inline void ColMaxFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
__aicore__ inline void ColAddFd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
```

`SparseFlashAttention`'s own kernel already allocates exactly the buffers this consumes —
`accumOutSize`, `logSumExpSize`, `lseSumFdGm`, `lseMaxFdGm` (section 8.1 and
`sparse_flash_attention_kernel_mla.h:503-508`) — behind `if constexpr (FLASH_DECODE)`, which is
never true.

The A3 IFA kernel file inventory, for scale:

```
   1264 ifa_service_matmul_full_quant.h
   4619 incre_flash_attention_preload.h
   4119 incre_flash_attention_preload_dd.h
   4324 incre_flash_attention_preload_mla.h
   4286 incre_flash_attention_split_Bbn2s2_Us2.h
   1987 paged_attention_antiquantkv.h
    845 service_vector_flashdecode_mla.h
  21444 total
```

`incre_flash_attention_split_Bbn2s2_Us2.h` — the name is the split axes: B, bn2, s2, with `Us2`
the KV-split variant. `SparseFlashAttention`'s entire `arch22` kernel is 4140 lines across four
files and has no `s2` split template at all.

### 10.4 A5 (`arch35`) has the same hole — this is not an A3-only gap

`attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_kernel_mla_arch35.h`:

```cpp
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitCalcParamsEach()
{
    // 计算总的基本块
    maxS2LoopCnt = 0; // 所有核中最大累计s2Loop
    uint32_t sfaTotalBaseNum = 0;
    uint32_t s1GBaseSize = sfaConstInfo.gSize;
    uint32_t actBatchS2 = 1;
    uint32_t coreNum = GetBlockNum(); // G128时相邻两个cube核处理一个s1，coreNum减半
    uint32_t sfaCurrCoreIdx = aicIdx;
    if constexpr (IS_SPLIT_G) {
        coreNum = coreNum >> 1;
        sfaCurrCoreIdx = sfaCurrCoreIdx >> 1;
    }
    uint32_t actBatchS1 = 1;
    for (uint32_t bIdx = 0; bIdx < sfaConstInfo.bSize; bIdx++) {
        actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx); // 不切S2，只关注S1
        if (actBatchS1 < sfaConstInfo.s1Size) {
            sfaConstInfo.needInit = true;
        }
        sfaTotalBaseNum += actBatchS1 * actBatchS2;
    }
    uint32_t sfaAvgBaseNum = 1;
    if (sfaTotalBaseNum > coreNum) {
        sfaAvgBaseNum = (sfaTotalBaseNum + coreNum - 1) / coreNum;
        if constexpr (IS_SPLIT_G) {
            usedCoreNum = (sfaTotalBaseNum + sfaAvgBaseNum - 1) / sfaAvgBaseNum << 1;
        }
    } else {
        if constexpr (IS_SPLIT_G) {
            usedCoreNum = sfaTotalBaseNum << 1;
        } else {
            usedCoreNum = sfaTotalBaseNum;
        }
    }
    ...
    if (aicIdx >= usedCoreNum) {
        return;
    }
```

The comment is explicit: `// 不切S2，只关注S1` — "do not split S2, only consider S1". Same
`actBatchS2 = 1`, same `usedCoreNum = sfaTotalBaseNum`, same early `return`. On A5 the only
mitigation is `IS_SPLIT_G`, which doubles the core count — and `GenTilingKey` only sets it when
`gSize > 64`.

One thing A5 does gain is a real top-k length input, so the zero-padding of section 9.6 can be
avoided:

```cpp
    GlobalTensor<int32_t> oriTopkLengthGm;
    bool hasOriTopkLength = false;
```

(present only in `arch35`; `grep -rn 'oriTopkLength' op_kernel/ op_host/` returns these two lines
and nothing in `arch22`.)

### 10.5 The published `SparseFlashAttention` contract, for reference

`attention/sparse_flash_attention/README.md`:

> - API功能：sparse_flash_attention（SFA）是针对大序列长度推理场景的高效注意力计算模块，该模块通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。

> 其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`lightning_indexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度。
> 本次公布的`sparse_flash_attention`是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。

The `sparse_indices` contract — note what it does and does *not* require:

> | sparse_indices | 输入 | 离散取kvCache的索引，不支持空tensor和非连续。sparse_size为一次离散选取的block数，需要保证每行有效值均在前半部分、无效值均在后半部分，且sparse_size大于0。layout_query为"BSND"时，shape为[B, Q_S, KV_N, sparse_size]。layout_query为"TND"时，shape为[Q_T, KV_N, sparse_size]。 | INT32 | ND |

Valid entries must be contiguous at the front of each row; **ascending order is not required**.

> | sparse_block_size | 可选属性 | 代表sparse阶段的block大小。sparse_block_size为1时，为Token-wise稀疏化场景；sparse_block_size大于1且小于等于128时，为Block-wise稀疏化场景，块内token共享相同的稀疏化决策。默认值为1。 | INT64 | - |

> | attention_mode | 可选属性 | 表示attention的模式，仅支持传入2，表示MLA-absorb模式。 | INT64 | - |

> | return_softmax_lse | 可选属性 | 用于表示是否返回softmax_max和softmax_sum。True表示返回，False表示不返回，默认值为False。 | BOOL | - |

Constraints:

> ## 约束说明
>
> - 该接口支持推理场景下使用。
> - 该接口支持图模式。
> - KV_N仅支持1。
> - block_size为一个block的token数，block_size取值为16的倍数，且最大支持1024。
> - 参数query中的Q_D和key、value的KV_D值相等为512，参数query_rope中的Dr和key_rope的Dr值相等为64。
> - 参数query、key、value的数据类型必须保持一致。
> - 当前只支持query_rope和key_rope传入，不支持rope为空。
> - sinks仅支持Ascend 950PR&950DT系列产品。
> - <term>Ascend 950PR&950DT系列产品</term>：
>   - Q_N支持1~128。
>   - sparse_block_size仅支持1。
>   - 仅在layout_kv为PA_BSND时，key、value和key_rope支持0轴非连续。
>   - return_softmax_lse为True时，支持layout_kv为PA_BSND。
> - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
>   - Q_N支持1/2/4/8/16/32/64/128。
>   - 参数sparse_block_size支持[1,128]，且要求是2的幂次方，在PageAttention场景下要求sparse_block_size整除block_size。
>   - 不支持sinks。
>   - key、value和key_rope不支持非连续。
>   - return_softmax_lse为True时，layout_kv不支持PA_BSND。

`Q_N支持1/2/4/8/16/32/64/128` on A3 — Q heads per rank must be a power of two in that set, so a
TP16 split of a 64- or 128-head MLA model lands on `Q_N` = 4 or 8, far below the `gSize > 64`
threshold that would enable `IS_SPLIT_G`.

## Relevance

This is the strongest single fact in this whole investigation, because it is an A/B with the
confound removed: same vendor, same repository, same A3 `arch22` target, same decode shape,
same 512-token empirical unit — and the dense op splits KV across 16 cores while the sparse op
splits across 1. The 5.0x roofline miss on `SparseFlashAttention` is not a property of sparse
attention, of scattered DMA, or of A3. It is a missing tiling mode in one operator.

Three consequences for our 6.973 ms.

**The fix has a reference implementation to copy.** `SplitBNS` + `service_vector_flashdecode_mla.h`
is 845 lines of A3 AscendC that does exactly the partial-`(out, max, sum)` combine our
hand-written kernel needs. `SparseFlashAttention` already allocates the workspace for it
(`accumOutSize`, `logSumExpSize`) and already exposes `softmax_max` / `softmax_sum` as outputs
under `return_softmax_lse`, so the external-combine route is open without any CANN change: call
SFA `kvSplitPart_` times over disjoint slices of the sparse index list with
`return_softmax_lse=true`, then run our own `ColMax`/`ColAdd`/`MatDivs` rescale. The cost is
`kvSplitPart_` extra MIX_AIC launches per layer against our 13.58 us floor — at 16 slices that is
217 us per layer, which is far worse than the 75 us we currently pay. **So the external-combine
route only pays inside a megakernel**, where the 16 slices are 16 core-groups of one launch
rather than 16 launches. That is a direct argument for the megakernel being the enabling vehicle
rather than an optimization on top.

**`IFATiling`'s 512-token floor bounds how far the split can go.** `while ((maxActualseq_ /
kvSplitPart_) < 512U && kvSplitPart_ > 1U) kvSplitPart_--;` means at seq 8192 the maximum useful
split is 16, not 24. If our DSA top-k is 2048 rather than the full 8192, the same rule gives
`2048/512 = 4` — only 4 cores' worth of useful split. This is the number that decides whether
closing the parallelism gap is worth 16x or 4x, and it turns on the top-k, which is why the top-k
value matters more than any other single model parameter here.

**Our 5.0x is smaller than single-core execution predicts, so one of our premises is wrong.**
Section 8.3 says 1 core; section 9.1 says 3 HBM transits. Together they predict a gap far beyond
5.0x. The reconciliation candidates, in order of likelihood: (i) we pass `layout_query = "TND"`
with the `gSize` heads packed along `Q_T`, which makes `totalBaseNum = gSize` and recovers
`gSize` cores — testable by reading `Q_T` at the call site; (ii) our 1.382 ms roofline denominator
is a per-die or per-core bandwidth rather than aggregate HBM; (iii) the op is being called once
per head rather than once per layer, which would multiply the launch count and show up in our
3008-kernels-per-round figure. Determining which costs one plog read of
`SFA block dim: %u aiv Num: %u aic Num: %u.` and is the highest-value next measurement.
