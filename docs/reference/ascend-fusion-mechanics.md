# Ascend-specific fusion mechanics — verbatim source dossier

Research dossier for the GLM-5.2-W4A8C8 megakernel effort on Ascend A3 (910-series, CANN 9.2.0).
Everything below is quoted verbatim from the cited source. Chinese text is left in Chinese.
Measured context this bears on: per-kernel-call launch floors on one die of
AI_VECTOR_CORE 0.70 us, AI_CORE 2.30 us, MIX_AIV 12.04 us, MIX_AIC 13.58 us;
~3008 kernels per decode round; 505 MIX_AIC calls per round (6.5 per layer); 11.5 ms launch floor.

---

## 1. Cross-core and cross-block synchronization primitives

### AscendC `kernel_operator_block_sync_intf.h` — the complete declared sync surface

https://raw.gitcode.com/cann/asc-devkit/raw/master/include/basic_api/kernel_operator_block_sync_intf.h
(also browsable at https://gitcode.com/cann/asc-devkit/blob/master/include/basic_api/kernel_operator_block_sync_intf.h)
Repo: `cann/asc-devkit` on GitCode, branch `master`. Copyright header says 2025, CANN Open Software License Agreement Version 2.0.

```cpp
/*!
 * \file kernel_operator_block_sync_intf.h
 * \brief
 */

#if defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
#error "kernel_operator_block_sync_intf.h cannot be used with compile flag --enable-simt enabled."
#endif

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_BLOCK_SYNC_INTF_H__
#endif
#ifndef ASCENDC_MODULE_OPERATOR_BLOCK_SYNC_INTF_H
#define ASCENDC_MODULE_OPERATOR_BLOCK_SYNC_INTF_H

#include "../../impl/basic_api/kernel_macros.h"
#include "../../impl/basic_api/kernel_event.h"
#include "../../impl/basic_api/kernel_reg.h"
#include "kernel_tensor.h"
#include "../../impl/basic_api/kernel_tensor_base.h"
#include "../../impl/basic_api/utils/kernel_utils_mode.h"

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include <cstdint>
#include "stub_def.h"
#include "stub_fun.h"
#endif

namespace AscendC {

template <HardEvent event>
__aicore__ inline void SetFlag(int32_t eventID);

template <HardEvent event>
__aicore__ inline void WaitFlag(int32_t eventID);

template <pipe_t pipe>
__aicore__ inline void PipeBarrier();

template <MemDsbT arg0>
__aicore__ inline void DataSyncBarrier();

/*
 * @ingroup：IBSet, IBWait
 * @brief：Set the flag bit of a core
 * @param [in] gmWorkspace GlobalTensor to store core state
 * @param [in] ubWorkspce LocalTensor for current core
 * @param [in] blockIdx the idx number waiting for the core
 * @param [in] eventID Set and wait events
 */
template <bool isAIVOnly = true>
__aicore__ inline void IBSet(
    const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, int32_t blockIdx,
    int32_t eventID);

template <bool isAIVOnly = true>
__aicore__ inline void IBWait(
    const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, int32_t blockIdx,
    int32_t eventID);
/*
 * @ingroup：SyncALL
 * @brief：Set flag bits of all cores
 * @param [in] gmWorkspace GlobalTensor to store core state
 * @param [in] ubWorkspce LocalTensor for current core
 */
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll(
    const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, const int32_t usedCores = 0);

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

template <pipe_t src, pipe_t dst>
class TQueSync {
public:
    __aicore__ inline void SetFlag(TEventID id);
    __aicore__ inline void WaitFlag(TEventID id);
};

namespace BlockSync {
enum SyncScope {
    InterBlock,
};

template <SyncScope Scope = SyncScope::InterBlock>
__aicore__ inline void MakeBarrier(uint8_t groupID, uint16_t arriveNum);
} // namespace BlockSync

} // namespace AscendC
#if defined(__NPU_ARCH__)
#include "../../impl/basic_api/kernel_operator_block_sync_intf_impl.h"
#endif
#endif // KERNEL_BLOCK_SYNC_INTF_H
```

## Relevance
This is the exact API surface a megakernel must use to replace the 3008 host-side kernel launches with in-kernel
stage transitions. `CrossCoreSetFlag<modeId, pipe>(uint16_t flagId)` / `CrossCoreWaitFlag` is the AIC<->AIV handshake
that a MIX kernel performs internally — i.e. the thing our 13.58 us MIX_AIC floor is paying for once per launch
instead of once per stage. `SyncAll<isAIVOnly>()` with no workspace argument is the cheap intra-kernel grid barrier;
the `GlobalTensor<int32_t> gmWorkspace` overload is the GM-workspace-based variant. Note `BlockSync::MakeBarrier(groupID, arriveNum)`
— an arrive-count barrier, which is what a persistent multi-stage kernel wants rather than a full `SyncAll`.
`__NPU_ARCH__ == 2201` (our A3) takes the `#else` branch: no `SyncAllConfig` on A3, only the plain `SyncAll<bool>()`.

---

## 2. Architecture generations, `__NPU_ARCH__`, SocVersion mapping

### `asc-devkit` NPU architecture guide (`npu-arch-guide.md`)

https://raw.gitcode.com/cann/asc-devkit/raw/master/.agent/skills/asc-npu-arch/references/npu-arch-guide.md
Repo `cann/asc-devkit`, branch `master`. In-repo agent skill reference.

> # Ascend NPU 架构指南
>
> 本文档说明 Ascend NPU 的架构代际划分，以及在 `asc-devkit` 中进行 AscendC API 设计、实现和验证时需要关注的架构边界。
>
> ## 架构代际概述
>
> ### 核心概念
>
> | 概念 | 说明 |
> |-----|------|
> | **NpuArch** | 芯片架构号，定义指令集和微架构 |
> | **SocVersion** | 片上系统版本，软件命名标识 |
> | **__NPU_ARCH__** | Device 侧编译宏，四位数值，用于条件编译 |
>
> ### 架构代号别名
>
> | 代号 | 对应 SocVersion | 说明 |
> |-----|----------------|------|
> | **A2** | ASCEND910B | Ascend910B1~B4, Ascend910B2C |
> | **A3** | ASCEND910B (含 Ascend910_93) | 训练/推理芯片 |
> | **A5** | ASCEND950 | Ascend950DT / Ascend950PR |
>
> ### 关键细节
>
> - 一个 `NpuArch` 可以对应多个芯片型号或产品系列。
> - `Ascend910_93` 的 SocVersion 字符串在运行时映射到 `SocVersion::ASCEND910B`，NpuArch 同为 `DAV_2201`。
> - 源码中可能存在 `SocVersion::ASCEND910_93` 枚举值，但运行时字符串映射应以本地 `convertMap` 为准。
>
> ## 统一芯片调用名
>
> | 统一调用名 | __NPU_ARCH__ | SocVersion | 产品系列 / 常见芯片 | 说明 |
> |-----------|:------------:|------------|---------------------|------|
> | `ascend910` | 1001 | ASCEND910 | Atlas 训练系列 / Ascend910 | 初代训练架构 |
> | `ascend910b1` | 2201 | ASCEND910B | Atlas A2 / Ascend910B1~B4, Ascend910B2C | 与 A3 共用 2201 |
> | `ascend310p` | 2002 | ASCEND310P | Atlas 推理系列 / Ascend310P1, Ascend310P3 | 与 `ascend610` 共用 2002 |
> | `ascend610` | 2002 | ASCEND310P | Atlas 推理系列 / Ascend610 | 与 `ascend310p` 共用 2002 |
> | `ascend310b1` | 3002 | ASCEND310B | Atlas 200I/500 A2 / Ascend310B1~B4 | 推理架构 |
> | `ascend950pr_9599` | 3510 | ASCEND950 | Atlas A5 推理 / Ascend950PR | 当前 regbase / SIMT 主入口 |
>
> ## 完整映射表
>
> | 产品系列 | SocVersion | __NPU_ARCH__ | 芯片型号 | 说明 |
> |---------|-----------|:---:|---------|------|
> | Atlas A2 训练/推理 | ASCEND910B | 2201 | Ascend910B1~B4, Ascend910B2C | 主流训练芯片 |
> | Atlas A3 训练/推理 | ASCEND910B | 2201 | Ascend910_93 | 训练/推理芯片 |
> | Atlas A5 训练 | ASCEND950 | 3510 | Ascend950DT | Decode |
> | Atlas A5 推理 | ASCEND950 | 3510 | Ascend950PR | Prefill |
> | Atlas 训练系列 | ASCEND910 | 1001 | Ascend910 | 初代训练架构 |
> | Atlas 推理系列 | ASCEND310P | 2002 | Ascend310P1, Ascend310P3, Ascend610 | 推理芯片 |
> | Atlas 200I/500 A2 推理 | ASCEND310B | 3002 | Ascend310B1~B4 | 推理芯片 |
>
> ### 基础 dtype 大小与 32B 对齐
>
> | 数据类型 | sizeof | 32B 对齐元素数 | 说明 |
> |---------|--------|---------------|------|
> | `bool` | 1 字节 | 32 个元素 | 全 0 为 false，否则为 true |
> | `int8_t` | 1 字节 | 32 个元素 | INT8 |
> | `uint8_t` | 1 字节 | 32 个元素 | UINT8 |
> | `fp4x2_e2m1_t` | 1 字节 | 32 个元素 | 两个 FP4 E2M1 值打包 |
> | `fp4x2_e1m2_t` | 1 字节 | 32 个元素 | 两个 FP4 E1M2 值打包 |
> | `hifloat8_t` | 1 字节 | 32 个元素 | HiFloat8 |
> | `fp8_e5m2_t` | 1 字节 | 32 个元素 | FP8 E5M2 |
> | `fp8_e4m3fn_t` | 1 字节 | 32 个元素 | FP8 E4M3FN |
> | `fp8_e8m0_t` | 1 字节 | 32 个元素 | FP8 E8M0 |
> | `int4x2_t` | 1 字节 | 32 个元素 | 两个 4-bit 整数打包，单 lane 范围 `[-8, 7]` |
> | `half` | 2 字节 | 16 个元素 | FP16 |
> | `bfloat16_t` | 2 字节 | 16 个元素 | BF16 |
> | `int16_t` | 2 字节 | 16 个元素 | INT16 |
> | `uint16_t` | 2 字节 | 16 个元素 | UINT16 |
> | `float` | 4 字节 | 8 个元素 | FP32 |
> | `int32_t` | 4 字节 | 8 个元素 | INT32 |
> | `uint32_t` | 4 字节 | 8 个元素 | UINT32 |
> | `complex32` | 4 字节 | 8 个元素 | `AscendC::Complex<half>` |
> | `int64_t` | 8 字节 | 4 个元素 | INT64 |
> | `uint64_t` | 8 字节 | 4 个元素 | UINT64 |
> | `double` | 8 字节 | 4 个元素 | FP64 |
> | `complex64` | 8 字节 | 4 个元素 | `AscendC::Complex<float>` |
>
> 32B 对齐的通用约束为 `count * sizeof(T) % 32 == 0`。
>
> | 产品范围 | 文档声明的支持 dtype |
> |---------|----------------------|
> | Ascend 950PR&950DT系列产品 | `bool`, `int8_t`, `uint8_t`, `fp4x2_e2m1_t`, `fp4x2_e1m2_t`, `hifloat8_t`, `fp8_e5m2_t`, `fp8_e4m3fn_t`, `fp8_e8m0_t`, `int4x2_t`, `int16_t`, `uint16_t`, `half`, `bfloat16_t`, `int32_t`, `uint32_t`, `float`, `complex32`, `int64_t`, `uint64_t`, `double`, `complex64` |
> | Atlas A2 / A3 训练和推理系列 | `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`, `double` |
>
> | 简写 | 类型 |
> |-----|------|
> | `S4` | `int4b_t`，通常通过 `int4x2_t` 打包存储 |
> | `U8` / `S8` | `uint8_t` / `int8_t` |
> | `FP8_E4M3` | `fp8_e4m3fn_t` |
> | `HiF8` | `hifloat8_t` |
> | `FP16` / `BF16` / `FP32` | `half` / `bfloat16_t` / `float` |

## Relevance
Our A3 compiles with `__NPU_ARCH__ == 2201` (DAV_2201), shared with A2/910B. Every conditional-compilation branch
in AscendC that gates a feature on `3510` (A5) is unavailable to us — which rules out the `SyncAllConfig` SyncAll
overload and the regbase/SIMT programming model. Note also the A2/A3 dtype list does **not** include `fp8_*`,
`int4x2_t` or `hifloat8_t`: a W4A8 kernel on A3 must hand-pack int4 rather than use a native 4-bit dtype.

---

## 3. `TPipe` / `TQue` / `TBuf` / `LocalTensor` allocation and limits

### `asc-devkit` LocalTensor memory guide (`local-tensor-memory.md`)

https://raw.gitcode.com/cann/asc-devkit/raw/master/.agent/skills/asc-api-ut-gen/references/foundations/local-tensor-memory.md
Repo `cann/asc-devkit`, branch `master`.

> # LocalTensor 内存申请指南
>
> ## 1. 概述
>
> LocalTensor 是 AscendC 中用于表示片上内存（Unified Buffer、L1 等）的张量类型。获取 LocalTensor 需要先通过 `TPipe` 或 `TBufPool` 分配内存给 `TQue` 或 `TBuf`，再从中获取。
>
> ## 2. 内存分配方式
>
> ### 2.1 TPipe::InitBuffer（推荐）
>
> #### 为 TQue 分配内存
>
> ```cpp
> AscendC::TPipe pipe;
> AscendC::TQue<AscendC::TPosition::VECOUT, 2> que;
>
> // 分配 2 块内存，每块 128 字节
> // num=1 不开启 double buffer；num=2 开启 double buffer
> uint8_t num = 2;
> uint32_t len = 128;
> pipe.InitBuffer(que, num, len);
> ```
>
> #### 为 TBuf 分配内存
>
> ```cpp
> AscendC::TPipe pipe;
> AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
>
> uint32_t len = 128;
> pipe.InitBuffer(tmpBuf, len);
> ```
>
> #### 自定义地址分配
>
> ```cpp
> AscendC::TPipe pipe;
> AscendC::TQue<AscendC::TPosition::VECOUT, 1> que;
>
> // 自定义内存地址信息：[起始地址, 长度]
> auto addr0 = Std::make_tuple(0, 1024);
> auto addr1 = Std::make_tuple(2048, 2048);
> auto addr2 = Std::make_tuple(8192, 4096);
> pipe.InitBuffer(que, addr0, addr1, addr2);
> ```
>
> ### 2.2 TBufPool::InitBuffer
>
> ```cpp
> // TBufPool 分配
> // bufIDSize 默认上限为 4，最大为 16
> TBufPool<TPosition::VECIN, 8> bufPool;  // 最多 8 个 Buffer
> bufPool.InitBuffer(que, num, len);
> bufPool.InitBuffer(buf, len);
> ```
>
> ## 3. 获取 LocalTensor
>
> ### 3.1 从 TQue 获取
>
> ```cpp
> // 申请 Tensor
> LocalTensor<T> tensor = que.AllocTensor<T>();
>
> // 使用后释放
> que.EnQue(tensor);  // 先入队
> que.DeQue<T>();     // 再出队使用
> que.FreeTensor(tensor);  // 最后释放
> ```
>
> ### 3.2 从 TBuf 获取
>
> ```cpp
> // 直接获取（无需手动释放）
> LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();
> ```
>
> ## 4. 重要约束
>
> ### 4.1 内存对齐
>
> | 约束项 | 说明 |
> |--------|------|
> | **32 字节对齐** | `len` 参数非 32Bytes 对齐时，会**自动向上补齐至 32Bytes** |
> | **元素对齐** | 确保 count * sizeof(T) 满足对齐要求 |
>
> ### 4.2 Buffer 数量限制
>
> | 限制类型 | 数量限制 |
> |----------|----------|
> | **TPipe 方式** | 一个 kernel 中所有 Buffer 数量之和**不能超过 64** |
> | **TBufPool 方式** | `bufIDSize` 默认上限为 4，最大为 16 |
>
> ### 4.3 Double Buffer
>
> ```cpp
> // num=1: 不开启 double buffer
> pipe.InitBuffer(que, 1, len);
>
> // num=2: 开启 double buffer，可提高流水线效率
> pipe.InitBuffer(que, 2, len);
> ```
>
> ## 5. 生命周期管理
>
> ### 5.2 重新分配
>
> ```cpp
> // 如需重新分配内存，需先调用 Reset
> pipe.Reset();
> pipe.InitBuffer(que, newNum, newLen);
> ```
>
> ## 6. TPosition 位置选择
>
> `TPosition` 决定了 LocalTensor 所在的物理存储位置：
>
> | 位置 | 适用场景 | 说明 |
> |------|----------|------|
> | **VECIN** | 矢量编程输入 | Vector Core 输入数据 |
> | **VECOUT** | 矢量编程输出 | Vector Core 输出数据 |
> | **VECCALC** | 矢量计算临时空间 | 用于临时变量，别名 LCM |
> | **A1/A2** | 矩阵编程 | A 矩阵 L1 缓冲区 |
> | **B1/B2** | 矩阵编程 | B 矩阵 L1 缓冲区 |
> | **C1/C2** | 矩阵编程 | C 矩阵 L1 缓冲区 |
> | **CO1/CO2** | 矩阵编程 | C 矩阵输出缓冲区 |

## Relevance
Hard ceiling for a megakernel: **all TQue/TBuf buffers in one kernel must total <= 64**. A fused multi-stage kernel
that gives each stage its own queues will hit this. `pipe.Reset()` plus re-`InitBuffer` is the documented way to
reuse UB across stages, and the custom-address `InitBuffer(que, addr0, addr1, ...)` overload lets us hand-place
stage buffers in UB instead of letting TPipe bump-allocate — the mechanism for keeping matmul -> vector -> matmul
resident on-chip without returning to HBM.

---

## 4. MIX core: what MIX_AIC / MIX_AIV actually mean, and `KERNEL_TASK_TYPE`

### CANN 9.0 (CANNCommunityEdition 900) — 设置Kernel类型 / Setting Kernel Type

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0218.html
English mirror: https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0218.html
(The JS-gated viewer URL is https://www.hiascend.com/document/detail/en/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0218.html — use the `doc_center/source` path to get raw HTML.)
Publisher: Huawei / Ascend, CANN Community Edition 9.0 API reference, parent topic 「Kernel Tiling」.

Applicability table (产品支持情况):

> | 产品 | 是否支持 |
> |---|---|
> | Atlas 350 加速卡 | √ |
> | Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |
> | Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ |
> | Atlas 200I/500 A2 推理产品 | x |
> | Atlas 推理系列产品 AI Core | √ |
> | Atlas 推理系列产品 Vector Core | x |
> | Atlas 训练系列产品 | x |

功能说明 (Function Usage) — **this is the documented statement that kernel type exists specifically to cut launch cost**:

> 用于用户自定义设置kernel类型，控制算子执行时只启动该类型的核，避免启动不需要工作的核，缩短核启动开销。

> Customizes the kernel type. Only cores of this type are started during operator execution. This prevents unnecessary cores from being started and shortens the core startup overhead.

函数原型 (Prototype):

> 设置全局默认的kernel type，对所有的tiling key生效。
> 当前支持在自定义算子工程和Kernel直调工程中使用。
> ```
> KERNEL_TASK_TYPE_DEFAULT(value)
> ```
> 设置某一个具体的tiling key对应的kernel type。
> 当前仅支持在自定义算子工程中使用。
> ```
> KERNEL_TASK_TYPE(key, value)
> ```

> Sets the global default kernel type, which applies to all tiling keys.
> Currently, this function can be used in custom operator projects and kernel launch projects.
> Sets the kernel type corresponding to a specific tiling key.
> Currently, this function can be used only in custom operator projects.

参数说明:

> | 参数 | 输入/输出 | 说明 |
> |---|---|---|
> | key | 输入 | tiling key的key值，此参数是正数，表示某个核函数的分支。 |
> | value | 输入 | 设置的kernel类型，可选值范围，kernel类型具体说明请参考表2。不同硬件架构支持的参数取值不同，具体支持的参数取值请参考kernel type取值约束。 |

The enum, verbatim:

```cpp
enum KernelMetaType {
    KERNEL_TYPE_AIV_ONLY,
    KERNEL_TYPE_AIC_ONLY,
    KERNEL_TYPE_MIX_AIV_1_0,
    KERNEL_TYPE_MIX_AIC_1_0,
    KERNEL_TYPE_MIX_AIC_1_1,
    KERNEL_TYPE_MIX_AIC_1_2,
    KERNEL_TYPE_AICORE,
    KERNEL_TYPE_VECTORCORE,
    KERNEL_TYPE_MIX_AICORE,
    KERNEL_TYPE_MIX_VECTOR_CORE,
    KERNEL_TYPE_MAX
};
```

表2 kernel type取值说明 — verbatim, each value:

> | 参数 | 说明 |
> |---|---|
> | KERNEL_TYPE_AIV_ONLY | 算子执行时仅启动AI Core上的Vector核：比如用户在host侧设置numBlocks为10，则会启动10个Vector核。 |
> | KERNEL_TYPE_AIC_ONLY | 算子执行时仅启动AI Core上的Cube核：比如用户在host侧设置numBlocks为10，则会启动10个Cube核。 |
> | KERNEL_TYPE_MIX_AIV_1_0 | AIC、AIV混合场景下，使用了多核控制相关指令时，设置核函数的类型为MIX AIV:AIC 1:0（带有硬同步），算子执行时仅会启动AI Core上的Vector核，比如用户在host侧设置numBlocks为10，则会启动10个Vector核。<br>硬同步的概念解释如下：当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用SyncAll()函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。目前多核同步分为硬同步和软同步，硬同步是利用硬件自带的全核同步指令由硬件保证多核同步。 |
> | KERNEL_TYPE_MIX_AIC_1_0 | AIC、AIV混合场景下，使用了多核控制相关指令时，设置核函数的类型为MIX AIC:AIV 1:0（带有硬同步），算子执行时仅会启动AI Core上的Cube核，比如用户在host侧设置numBlocks为10，则会启动10个Cube核。 |
> | KERNEL_TYPE_MIX_AIC_1_1 | AIC、AIV混合场景下，设置核函数的类型为MIX AIC:AIV 1:1，算子执行时会同时启动AI Core上的Cube核和Vector核，比如用户在host侧设置numBlocks为10，则会启动10个Cube核和10个Vector核。 |
> | KERNEL_TYPE_MIX_AIC_1_2 | AIC、AIV混合场景下，设置核函数的类型为MIX AIC:AIV 1:2，算子执行时会同时启动AI Core上的Cube核和Vector核，比如用户在host侧设置numBlocks为10，则会启动10个Cube核和20个Vector核。 |
> | KERNEL_TYPE_AICORE | 算子执行时仅会启动AI Core，比如用户在host侧设置numBlocks为5，则会启动5个AI Core。 |
> | KERNEL_TYPE_VECTORCORE | 该参数为预留参数，当前版本暂不支持。 |
> | KERNEL_TYPE_MIX_AICORE | 该参数为预留参数，当前版本暂不支持。 |
> | KERNEL_TYPE_MIX_VECTOR_CORE | 基于Ascend C开发的矢量计算相关的算子可以运行在Vector Core上，调用本接口传入该参数用于使能Vector Core。<br>使能Vector Core后，算子执行时会同时启动AI Core和Vector Core，用于并行计算。比如用户在host侧设置numBlocks为10，则会启动总数为10的AI Core和Vector Core。<br>需要注意的是，通过SetBlockDim设置核数时，需要大于AI Core的核数，否则不会启动VectorCore。 |

约束说明 — kernel type 取值约束, verbatim:

> Atlas 350 加速卡，支持KERNEL_TYPE_AIV_ONLY、 KERNEL_TYPE_AIC_ONLY、KERNEL_TYPE_MIX_AIV_1_0、KERNEL_TYPE_MIX_AIC_1_0、KERNEL_TYPE_MIX_AIC_1_1、KERNEL_TYPE_MIX_AIC_1_2。
>
> Atlas A3 训练系列产品 / Atlas A3 推理系列产品，支持KERNEL_TYPE_AIV_ONLY、 KERNEL_TYPE_AIC_ONLY、KERNEL_TYPE_MIX_AIV_1_0、KERNEL_TYPE_MIX_AIC_1_0、KERNEL_TYPE_MIX_AIC_1_1、KERNEL_TYPE_MIX_AIC_1_2。
>
> Atlas A2 训练系列产品 / Atlas A2 推理系列产品，支持KERNEL_TYPE_AIV_ONLY、 KERNEL_TYPE_AIC_ONLY、KERNEL_TYPE_MIX_AIV_1_0、KERNEL_TYPE_MIX_AIC_1_0、KERNEL_TYPE_MIX_AIC_1_1、KERNEL_TYPE_MIX_AIC_1_2。
>
> Atlas 推理系列产品，支持KERNEL_TYPE_AICORE、KERNEL_TYPE_MIX_VECTOR_CORE。
>
> KERNEL_TASK_TYPE优先级高于KERNEL_TASK_TYPE_DEFAULT，同时设置了全局kernel type和某一个tiling key的kernel type，该tiling key的kernel type以KERNEL_TASK_TYPE设置的为准。
>
> 没有设置全局默认kernel type的情况下，如果开发者只为其中的某几个tiling key设置kernel type，即部分tiling key没有设置kernel type，会导致算子kernel编译报错。
>
> 当设置具体的kernel task type时，用户的算子实现需要与kernel type相匹配。比如用户设置kernel type为KERNEL_TYPE_MIX_AIC_1_2，则算子内部实现应与核配比AIC:AIV为1:2相对应；若用户设置kernel type为KERNEL_TYPE_AIC_ONLY， 则算子内部实现应该为纯cube逻辑，不应该存在vector部分的逻辑。其他的kernel type类似。
>
> 当纯cube或者纯vec算子强制设定kernel type为MIX类型时，workspace的大小不能设置为0，需要设置一个大于0的值（比如16、32等）。
>
> 使用Tiling模板编程时，需要通过ASCENDC_TPL_KERNEL_TYPE_SEL设置Kernel类型即可，无需再通过该接口进行设置，本接口不生效。

调用示例 示例一：使能VectorCore样例 — verbatim:

```cpp
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    KernelAdd op;
    op.Init(x, y, z, tilingData.numBlocks, tilingData.totalLength, tilingData.tileNum);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE); // 使能VectorCore
    if (TILING_KEY_IS(1)) {
        op.Process1();
    } else if (TILING_KEY_IS(2)) {
        op.Process2();
    }
    // ...
}
```

> 完成算子host侧Tiling开发时，设置的numBlocks代表的是AI Core和Vector Core的总数，比如用户在host侧设置numBlocks为10，则会启动总数为10的AI Core和Vector Core；为保证启动Vector Core，设置数值应大于AI Core的核数。您可以通过GetCoreNumAic接口获取AI Core的核数，GetCoreNumVector接口获取Vector Core的核数。

```cpp
// 配套的host侧tiling函数示例：
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 使能VectorCore，将numBlocks置为AI Core中vector核数 + Vector Core中的vector核数
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    // ASCENDXXX请替换为实际的版本型号
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCENDXXX) {
       totalCoreNum = totalCoreNum + ascendcPlatform.GetCoreNumVector();
    }
    context->SetBlockDim(totalCoreNum);
}
```

示例二：设置某一个具体的tiling key对应的kernel type。如下代码为伪代码，不可直接运行。

```cpp
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    KernelAdd op;
    op.Init(x, y, z, tilingData.numBlocks, tilingData.totalLength, tilingData.tileNum);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 设置默认的kernel类型为纯AIV类型
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIV_1_0); // 设置tiling key=1对应的kernel类型为MIX AIV 1:0
        op.Process1();
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY); // 设置tiling key=2对应的kernel类型为纯AIV类型
        op.Process2();
    }
    // ...
}
// 配套的host侧tiling函数示例：
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // ...
    if (context->GetInputShape(0) > 10) {
        context->SetTilingKey(1);
    } else if (some condition) {
        context->SetTilingKey(2);
    }
}
```

## Relevance
Direct, documented explanation of our 12.04 us MIX_AIV / 13.58 us MIX_AIC vs 0.70 us AI_VECTOR_CORE floors.
`MIX_AIC_1_1` and `MIX_AIC_1_2` *start both Cube and Vector cores* on every launch — a `numBlocks=10` MIX_AIC_1_2 launch
boots 10 Cube cores and 20 Vector cores, i.e. 30 cores to bring up and tear down versus 10 for AIV_ONLY. The doc
states the purpose of this API in exactly our terms: 「避免启动不需要工作的核，缩短核启动开销」 ("shortens the core
startup overhead"). The named mechanisms behind the MIX penalty are (a) the AIC:AIV core-count multiplier, and
(b) 硬同步 — the hardware all-core sync instruction that MIX_AIV_1_0 / MIX_AIC_1_0 carry ("带有硬同步"). Note that
`MIX_AIV_1_0` and `MIX_AIC_1_0` start only one core class yet still cost MIX: the fee is the hard-sync capability,
not the extra cores. So 505 MIX_AIC calls/round at 13.58 us = 6.86 ms of our 11.6 ms TPOT, and the lever is to
retype stages to `AIV_ONLY` / `AIC_ONLY` where no cross-core sync is needed, or to collapse many MIX launches into one.
Also load-bearing for us: forcing a pure-cube or pure-vector operator to a MIX type requires `workspace != 0`
(「workspace的大小不能设置为0，需要设置一个大于0的值（比如16、32等）」).

---

### CrossCoreSetFlag (ISASI) — the AIC<->AIV handshake, modes 0/1/2/4, flagId range, deadlock constraint

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0273.html
JS-gated mirror: https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0273.html
Publisher: Huawei / Ascend, CANN Community Edition 9.0, parent topic 「核间同步」 (inter-core synchronization).

产品支持情况:

> | 产品 | 是否支持 |
> |---|---|
> | Atlas 350 加速卡 | √ |
> | Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |
> | Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ |
> | Atlas 200I/500 A2 推理产品 | x |
> | Atlas 推理系列产品 AI Core | x |
> | Atlas 推理系列产品 Vector Core | x |
> | Atlas 训练系列产品 | x |

功能说明, verbatim:

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

函数原型:

```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
```

表1 模板参数说明:

> | 参数名 | 描述 |
> |---|---|
> | modeId | 核间同步的模式，取值如下：<br>模式0：AI Core核间的同步控制。<br>模式1：AI Core内部，Vector核（AIV）之间的同步控制。<br>模式2：AI Core内部，Cube核（AIC）与Vector核（AIV）之间的同步控制。<br>模式4：AI Core内部，AIC与AIV之间的同步控制。AIV0与AIV1可单独触发AIC等待。 |
> | pipe | 设置这条指令所在的流水类型，流水类型可参考硬件流水类型。<br>特别地，PIPE_S流水类型仅Atlas 350 加速卡支持。 |

表2 参数说明 — flagId range:

> | 参数名 | 输入/输出 | 描述 |
> |---|---|---|
> | flagId | 输入 | 核间同步的标记。<br>Atlas A2 训练系列产品 / Atlas A2 推理系列产品，取值范围是0-10。<br>Atlas A3 训练系列产品 / Atlas A3 推理系列产品，取值范围是0-10。<br>Atlas 350 加速卡，取值范围如下：<br>AIV0发起的flagId 0-10的CrossCoreSetFlag操作对应AIC CrossCoreWaitFlag中flagId 0-10的操作。<br>AIV1发起的flagId 0-10的CrossCoreSetFlag操作对应AIC CrossCoreWaitFlag中flagId 16-26的操作。<br>AIC发起的flagId 0-10的CrossCoreSetFlag操作对应AIV0 CrossCoreWaitFlag中flagId 0-10的操作。<br>AIC发起的flagId 16-26的CrossCoreSetFlag操作对应AIV1 CrossCoreWaitFlag中flagId 0-10的操作。 |

约束说明, verbatim — the constraints that bind a megakernel:

> 使用该同步接口时，需要按照如下规则设置Kernel类型：
>
> 在纯Vector/Cube场景下，需设置Kernel类型为KERNEL_TYPE_MIX_AIV_1_0或KERNEL_TYPE_MIX_AIC_1_0。
>
> 对于Vector和Cube混合场景，需根据实际情况灵活配置Kernel类型。
>
> 因为Matmul高阶API内部实现中使用了本接口进行核间同步控制，所以不建议开发者同时使用该接口和Matmul高阶API，否则会有flagID冲突的风险。
>
> 同一flagId的计数器最多设置15次。
>
> 使用该接口模式0时，建议开启batchmode模式，使算子独占全部所需核资源，否则可能因满足以下条件导致死锁：
>
> 多流并发场景（≥2条执行流）。
>
> ≥2个算子并发执行。
>
> 所有并发算子的核数总和超过物理核数。
>
> ≥2个并发算子使用了核间同步功能。
>
> 具体而言，在多流场景下，某条流的核间同步算子虽分配到n个物理核，但可能仅有n-m个核先被调度执行，而其余m个核因被其他流的核间同步算子抢占而尚未启动。先启动的n-m个核执行到核间同步时等待剩余m核完成，而剩余m核因被其他流的核间同步算子占用而无法释放，形成死锁。
>
> Kernel直调场景下通过__schedmode__(mode)限定符来设置batchmode模式；工程化算子开发场景下，通过TilingContext的SetScheduleMode接口来设置batchmode模式，具体请参考《基础数据结构和接口》。

调用示例 — verbatim:

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

## Relevance
This is the in-kernel substitute for a host kernel launch between a cube stage and a vector stage, and the
constraints are hard limits on megakernel design: (1) only **flagId 0-10** on A3 — 11 distinct cross-core events
total; (2) **同一flagId的计数器最多设置15次** — a counter per flagId saturates at 15 Sets, so a persistent kernel
looping over 78 layers cannot reuse one flagId per layer without resetting; (3) the Matmul high-level API already
consumes flagIds internally, so hand-written `CrossCoreSetFlag` plus `Matmul` risks flagID collision — which
constrains whether we can use the `Matmul` high-level API inside a hand-synchronized megakernel; (4) mode 0
(all-AIC or all-AIV grid sync) wants batchmode (`__schedmode__(mode)` in kernel-direct-launch, or
`TilingContext::SetScheduleMode`) or the kernel can deadlock when total cores across concurrent streams exceed
physical cores — directly relevant because a megakernel occupying all cores on a die is exactly the batchmode case.
Also note `g_coreType == AscendC::AIC` / `AscendC::AIV` is the documented way one MIX kernel body branches between
the cube and vector halves — the mechanics of "one kernel runs on both AIC and AIV".

---

## 5. One kernel doing matmul -> vector op without returning to HBM: the Mix fusion operator pattern

### CANN 9.0 算子开发指南 — CV融合 / 算子实现 (Matmul+LeakyRelu)

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0050.html
Also published at https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0050.html
and https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0050.html
Publisher: Huawei / Ascend, Ascend C 算子开发指南, parent topic 「CV融合」.

> 下文将以Matmul+LeakyRelu融合算子的实现为例，介绍Mix融合算子的设计和实现流程。该样例仅支持在 Atlas A2 训练系列产品 / Atlas A2 推理系列产品 上运行。

数据流分析, verbatim:

> 进行算子的数据流分析：数据流向为在Cube核上完成Matmul计算后将数据搬运至Vector核进行LeakyRelu计算。根据上述数据流并结合融合算子的编程范式，规划并行的流水任务。如下图所示：
>
> 将输入数据从Global Memory搬运到Cube核。
>
> 进行Matmul内部的计算，计算公式和计算示意图如下：
>
> 将Matmul的计算结果搬运到Vector核。
>
> 进行Vector矢量计算，该样例中进行LeakyReLU计算。
>
> 将输出结果搬运到Global Memory。
>
> 前三步的内容都封装在Matmul高阶API内，本样例中可以简化为3个stage。

Tiling 策略 — the UB constraint on the fused C tile, verbatim:

> 多核切分: 根据当前核数，对输入shape的M, K, N进行多核切分，得到单核内shape大小singleCoreM, singleCoreK, singleCoreN。
>
> 核内切分: 根据Local Memory的大小约束，对单核内的shape大小进一步切分，得到A、B、C矩阵参与一次矩阵乘指令的shape大小baseM, baseN, baseK。切分时需要注意：GetTensorC的结果如果放在LocalMemory（UB）上，需要注意，baseM * baseN的大小不能超出UB的限制。

Kernel-side code framework, verbatim:

```cpp
template<typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Process(){
    uint32_t computeRound = 0;

    // 设置Matmul的输入（包括左矩阵、右矩阵、bias）
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    // 调用matmul iterate获取一块[baseM, baseN]的计算结果
    while (matmulObj.template Iterate<true>())
    {
        MatmulCompute();
        LeakyReluCompute();
        CopyOut(computeRound);
        computeRound++;
    }
    matmulObj.End();
}
// kernel入口函数，mix场景，AIC:AIV=1:2
__global__ __mix__(1, 2) void matmul_leakyrelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                      __kfc_workspace__ GM_ADDR workspace, AscendC::tiling::TCubeTiling tiling)
{
    AscendC::TPipe pipe;
    MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
    matmulLeakyKernel.Init(a, b, bias, c, workspace, tiling, &pipe);
    // Matmul对象初始化
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj, &matmulLeakyKernel.tiling);
    matmulLeakyKernel.Process(&pipe);
}
```

Matmul compute — `GetTensorC` is the cube->vector on-chip handoff, verbatim:

```cpp
template<typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::MatmulCompute(){
    reluOutLocal = reluOutQueue_.AllocTensor<cType>();
    // 调用GetTensorC将Matmul的计算结果搬运到Vector核。
    matmulObj.template GetTensorC<true>(reluOutLocal, false, true);
}
```

```cpp
// 调用LeakyRule接口进行计算
template<typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::LeakyReluCompute(){
    AscendC::LeakyRelu(reluOutLocal, reluOutLocal, (cType)alpha, tiling.baseM * tiling.baseN);
    reluOutQueue_.EnQue(reluOutLocal);
}
```

```cpp
// 将结果搬出到GM
template<typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::CopyOut(uint32_t count){
    reluOutQueue_.DeQue<cType>();
    const uint32_t roundM = tiling.singleCoreM / tiling.baseM;
    const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
    uint32_t startOffset = (count % roundM * tiling.baseM * tiling.N + count / roundM * tiling.baseN);
    AscendC::DataCopyParams copyParam = {(uint16_t)tiling.baseM,
        (uint16_t)(tiling.baseN * sizeof(cType) / DEFAULT_C0_SIZE), 0,
        (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) / DEFAULT_C0_SIZE)};
    AscendC::DataCopy(cGlobal[startOffset], reluOutLocal, copyParam);
    reluOutQueue_.FreeTensor(reluOutLocal);
}
```

Host-side tiling, verbatim:

```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
```

> 设置示例如下，其中TPosition::LCM是Unified Buffer上的逻辑位置，等同于TPosition::VECCALC，关于TPosition的详细内容请参考TPosition。

```cpp
cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
cubeTiling.SetCType(matmul_tiling::TPosition::LCM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
```

```cpp
cubeTiling.SetShape(M, N, K);
cubeTiling.SetOrgShape(M, N, K);
```

> 设置Matmul计算时可用的L1 Buffer/L0C Buffer/Unified Buffer空间大小，-1表示AI处理器对应Buffer的大小。
> ```cpp
> cubeTiling.SetBufferSpace(-1, -1, -1);
> ```

### 分离模式 vs 耦合模式 — WHY a MIX kernel costs what it costs

Verbatim, same page (this is the single most important paragraph for our launch-cost question):

> 特别的对于多核场景，需要通过SetDim接口设置Matmul计算所用的核数，MIX模式（包含矩阵计算和矢量计算）的设置规则如下：
>
> 分离模式：Matmul API都是从AIV侧发起的，调用Iterate计算时在AIV侧只会起到通知的作用，通知AIC去做矩阵计算，计算完成后AIC告知AIV计算完成。这个架构下，SetBlockDim设置为实际计算会用到的AI Core（AIC、AIV组合）的数量，SetDim设置为实际计算会用到的AIV的数量。例如，SetBlockDim时可以设置为20，启动20个AI Core（AIC AIV的组合），SetDim设置为40，表示按照40个AIV进行切分。
>
> 耦合模式：SetBlockDim加载的核数就是Matmul API实际计算会用到的核数，SetDim和SetBlockDim设置的值是一样的。

System workspace requirement, verbatim:

> Matmul高阶API内部实现时需要使用系统workspace，开发者需要：
>
> 在host侧Tiling实现时，设置总的workspace的数值大小（包含用户workspace和系统workspace），workspace空间由框架来申请并管理。系统workspace的空间大小通过GetLibApiWorkSpaceSize获取。

```cpp
size_t userWorkspaceSize = 0;
size_t systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
size_t *currentWorkspace = context->GetWorkspaceSizes(1);
currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
```

> 若算子工程不是自定义算子工程，也不是带有HAVE_WORKSPACE编译宏的Kernel直调算子工程，kernel侧需要在Matmul初始化前，通过SetSysWorkSpace设置系统workspace。

```cpp
// 使用Matmul时必须设置workspace空间
SetSysWorkspace(workspace);
if (GetSysWorkSpacePtr() == nullptr) {
    return;
}
```

### The low-level ("底层编码") route: hand-written AIC/AIV isolation and hand-written sync

Verbatim, same page — this is the documented escape hatch from the framework's KFC messaging:

> 上文介绍的实现方法，AIC侧和AIV侧的代码隔离和核间同步由框架来完成，开发者无需关心。除该方法外，开发者也可以选择底层编码的方式在分离模式下实现融合算子，这种方式将更加灵活。采用底层编码方式时，需要注意以下几点：
>
> 通过ASCEND_IS_AIV和ASCEND_IS_AIC实现AIV和AIC代码之间的隔离。
>
> 自行实现AIC核和AIV核之间的同步：比如Matmul + LeakyRelu算子样例中，需要确保在AIC完成矩阵计算后，AIV再进行LeakyRelu的计算。
>
> 使用高阶API Matmul时需要设置ASCENDC_CUBE_ONLY，表示仅在AIC侧调用Matmul API。
>
> 使用设置Kernel类型接口设置Kernel类型为KERNEL_TYPE_MIX_xxx，同时启用AIV核和AIC核。

```cpp
#define ASCENDC_CUBE_ONLY // 指定Matmul运行在AIC上
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // 设置Kernel类型为KERNEL_TYPE_MIX_xxx
if ASCEND_IS_AIC {
    ...
    // AIC核进行Matmul计算
    // AIC核完成计算后，通过AscendC::CrossCoreSetFlag<modeId, pipe>(flagId)发送同步flag
}
if ASCEND_IS_AIV {
    ...
    // AIV核通过AscendC::CrossCoreWaitFlag(flagId)接收同步flag
    // AIV核进行LeakyRelu计算
}
```

> 完整样例请参考BareMix样例。

## Relevance
This is the template for the megakernel. Three things matter to our numbers.
(1) `__global__ __mix__(1, 2)` plus `__kfc_workspace__ GM_ADDR workspace` is the declaration form that produces a
MIX_AIC kernel — the 13.58 us class. "kfc" = kernel function call: in 分离模式 the Matmul API is *issued from the AIV
side*, and `Iterate` on AIV only sends a notification to AIC, which replies on completion. That request/reply channel
through GM workspace is a plausible named mechanism for the ~11 us gap between MIX_AIC (13.58 us) and AI_CORE (2.30 us).
(2) The `while (Iterate<true>()) { MatmulCompute(); LeakyReluCompute(); CopyOut(); }` loop with `GetTensorC<true>(reluOutLocal, ...)`
writing into a UB `LocalTensor` is the documented way to keep a matmul result on-chip and apply a vector op to it
without a round trip to HBM — exactly the fusion we need for the W4A8 dequant/activation chain.
(3) The 底层编码 route (`ASCENDC_CUBE_ONLY` + `ASCEND_IS_AIC`/`ASCEND_IS_AIV` + explicit `CrossCoreSetFlag`) is the
route that lets us keep the Matmul high-level API but drive the AIC/AIV handshake ourselves — which is what a
persistent multi-stage kernel requires, since the framework's per-`Iterate` KFC messaging is the per-call cost.
Note the constraint collision: section 4's CrossCoreSetFlag doc says mixing hand-written `CrossCoreSetFlag` with
the Matmul high-level API risks flagID conflicts, while this page tells us to do exactly that under `ASCENDC_CUBE_ONLY`.

---

## 6. SuperKernel — CANN's own kernel-binary fusion, and the mechanism behind it

### `cann/graph-autofusion` super_kernel README — 原理介绍 (the mechanism, verbatim)

https://gitcode.com/cann/graph-autofusion/blob/master/super_kernel/README.md
raw: https://raw.gitcode.com/cann/graph-autofusion/raw/master/super_kernel/README.md
Repo: `cann/graph-autofusion` on GitCode (CANN open-source org), branch `master`. Copyright 2025 Huawei Technologies.

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
```

Supported products, from `super_kernel/examples/README.md`
(https://raw.gitcode.com/cann/graph-autofusion/raw/master/super_kernel/examples/README.md):

> ## 环境依赖
>
> 支持如下产品型号：
>
> - Ascend 950PR/Ascend 950DT
> - Atlas A3 训练系列产品/Atlas A3 推理系列产品
> - Atlas A2 训练系列产品/Atlas A2 推理系列产品

> ```text
> examples/
> ├── run_example.sh                                   # 工程测试批量执行脚本
> ├── jit/
> │   ├── example01_super_kernel_base/                 # SuperKernel 基础功能
> │   ├── example02_super_kernel_profiling/            # SuperKernel profiling 对比
> │   └── example03_super_kernel_runtime_ascendc_only/ # AscendC + Runtime 极简样例
> └── aot/
>     ├── scripts/                                     # AOT 样例公共 Bash 函数
>     ├── example01_dual_stream/                       # 双流与 NPU Event 控制边
>     ├── example02_sk_options/                        # SuperKernel options
>     └── example03_kernel_pybind/                     # Pybind 自定义算子融合
> ```

The developer guide (https://raw.gitcode.com/cann/graph-autofusion/raw/master/docs/zh/super_kernel/developer_guide.md)
pins the default test SoC — note it is an A3 part:

> - `soc_version`（`fixtures/config.py`）：指定默认 SoC 信息，当前默认值为 `Ascend910_9391`；如需使用其他版本，可在场景内覆盖或增加新的 fixture。

## Relevance
This is the authoritative account of *why* kernel launch dominates and what the vendor does about it:
「将多个子算子融合成一个 SuperKernel，以节省 N-1 次算子调度开销」 — fuse N sub-operators, save N-1 dispatches.
That is our 3008-kernel / 11.5 ms problem stated by the vendor. Four mechanisms are named, and three bear directly
on our measurements:
- 同步优化 states the MIX cost explicitly: 「对于 Kernel Type 为 Mix 1:2 的混合类型算子，完整的全核同步需等待所有
  Aicore 的 Vector 核与 Cube 核均完成同步」 — a full all-core sync on a Mix 1:2 operator must wait for *both* the
  Vector and the Cube cores of every AI Core. That is the named reason MIX_AIC (13.58 us) costs ~6x AI_CORE (2.30 us)
  and ~19x AI_VECTOR_CORE (0.70 us). The fix SuperKernel applies is to narrow the sync scope per adjacent
  sub-operator kernel type (consecutive vector ops get a vector-only all-core sync).
- ICache Preload: a megakernel's binary is large and the hardware prefetches only the entry instructions, so naive
  fusion trades launch cost for ICache misses. We must plan for explicit code-segment preload.
- 子 Kernel 拆分: all cores executing the same instruction address serialize at the shared L2. SuperKernel replicates
  sub-kernel code so cores map to distinct physical addresses by core ID. A hand-written megakernel on 20+ cores
  will hit this L2 instruction-fetch serialization unless we replicate too.
- Early-Start: insert Set before the predecessor's trailing MTE instructions and Wait after the successor's
  input-independent scalar init, so two sub-operators' instructions overlap across pipes.

---

### SuperKernel public C API: `aclskOptimize`, `aclskScopeBegin/End`, `aclskScopeVerify`, full option enum

https://gitcode.com/cann/graph-autofusion/blob/master/super_kernel/include/super_kernel/super_kernel.h
raw: https://raw.gitcode.com/cann/graph-autofusion/raw/master/super_kernel/include/super_kernel/super_kernel.h
Repo `cann/graph-autofusion`, branch `master`. Copyright (c) 2025 Huawei Technologies Co., Ltd.

```cpp
/*!
 * \file superkernel_opt_options.h
 * \brief
 */

#ifndef ACL_SUPERKERNEL_H
#define ACL_SUPERKERNEL_H

#include <cstdint>
#include <cstddef>
#include "acl/acl.h"

enum class aclskOptionType : uint32_t {
  PRELOAD_CODE = 0,
  SPLIT_MODE = 1,
  STREAM_FUSION = 2,
  DCCI_DISABLE_ON_KERNEL = 3,
  DEBUG_SYNC_ALL = 4,
  KERNEL_MAP = 5,
  AUTO_OP_PARALLEL = 7,
  DCCI_BEFORE_KERNEL_START = 8,
  DEBUG_OP_EXEC_TRACE = 9,
  DEBUG_CROSS_CORE_SYNC_CHECK = 10,
  OPT_EXTEND_OPTION = 11,
  DEBUG_EXTEND_OPTION = 12,
  DCCI_AFTER_KERNEL_END = 13,
  AGGRESSIVE_OPT_STRATEGIES = 14,
  UBUF_LOCK_IGNORE_KERNEL = 15,
  EARLY_START = 16,                // early start global option
  DEBUG_PER_OP_MAX_CORE_NUM = 17,  // debug: per op max core num limit
  SK_OPTION_MAX
};

enum class SkInnerOptionType : uint32_t { MIX_KERNEL_SPLIT = 0, SIMT_OP_SUPPORT = 1, SK_INNER_OPTION_MAX };

typedef struct aclskPreloadOption {
  uint32_t preloadMode;
} aclskPreloadOption;

typedef struct aclskSplitModeOption {
  uint32_t splitCnt;
} aclskSplitModeOption;

typedef struct aclskStreamFusionOption {
  uint32_t streamFusion;
} aclskStreamFusionOption;

typedef struct aclskDcciOption {
  char **kernelNames;
  size_t kernelCnt;
} aclskDcciOption;

typedef struct aclskDebugSyncAllOption {
  uint32_t debugSyncAll;
} aclskDebugSyncAllOption;

typedef struct aclskKernelMap {
  char *globalName;
  char *sknlNames[4];
} aclskKernelMap;

typedef struct aclskKernelMapOption {
  aclskKernelMap *kernelMaps;
  size_t numKernels;
} aclskKernelMapOption;

typedef struct aclskAutoOpParallelOption {
  uint32_t enableAutoOpParallel;
} aclskAutoOpParallelOption;

/**
 * aggressiveOpts carries aggressive fusion strategy switches:
 *  - eventBreakerBypass: reserved for event breaker policy
 *  - taskBreakerBypass: enable default-node bypass when set to 1
 *
 * aggressiveOpts.valueBreakerBypass is a bitmask-style value-memory wait policy:
 *  - ACLSK_VALUE_BREAKER_BYPASS_NONE (0b00): reject notify/wait pairing relation, keep wait unfusible
 *  - ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT (0b01): for paired notify+wait, existing rule must pass or SK exits
 *  - ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT (0b10): for waits without notify after rule check, allow wait fusion
 */
enum aclskValueBreakerBypassFlag : uint32_t {
  ACLSK_VALUE_BREAKER_BYPASS_NONE = 0b00U,
  ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT = 0b01U,
  ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT = 0b10U
};

typedef struct aclskAggressiveOptStrategies {
  uint32_t eventBreakerBypass;
  uint32_t valueBreakerBypass;
  uint32_t taskBreakerBypass;
} aclskAggressiveOptStrategies;

/**
 * Configure kernels that should ignore MIX kernel split by name pattern.
 */
typedef struct aclskUbufLockIgnoreKernelOption {
  uint32_t ubufLockIgnoreKernelCnt;
  char **ubufLockIgnoreKernel;
} aclskUbufLockIgnoreKernelOption;

enum aclskEarlyStartValue : uint32_t {
  ACLSK_EARLY_START_DISABLED = 0U,  // 默认值，表示不启用early start
  ACLSK_EARLY_START_ENABLED = 1U,   // 启用early start
};
typedef struct aclskEarlyStartOption {
  uint32_t enableEarlyStart;  // 0[default]: disable early start, 1: enable early start
} aclskEarlyStartOption;

typedef struct aclskDebugPerOpMaxCoreNumOption {
  uint32_t enableDebugPerOpMaxCoreNum;  // 0[default]: disable, 1: enable
} aclskDebugPerOpMaxCoreNumOption;

struct aclskOption {
  aclskOptionType optionType;
  union {
    aclskPreloadOption preload;
    aclskSplitModeOption splitMode;
    aclskStreamFusionOption streamFusion;
    aclskDcciOption disableKernelDcci;
    aclskDebugSyncAllOption debugSync;
    aclskKernelMapOption kernelMap;
    aclskAutoOpParallelOption autoOpParallel;
    aclskDcciOption dcciBeforeKernelStart;
    aclskDebugOpExecTraceOption debugOpExecTrace;
    aclskDebugCrossCoreSyncCheckOption debugCrossCoreSyncCheck;
    aclskExtendOption optExtend;
    aclskExtendOption debugExtend;
    aclskDcciOption dcciAfterKernelEnd;
    aclskAggressiveOptStrategies aggressiveOpts;
    aclskUbufLockIgnoreKernelOption ubufLockIgnoreKernel;
    aclskEarlyStartOption earlyStart;
    aclskDebugPerOpMaxCoreNumOption debugPerOpMaxCoreNum;
  };
};

typedef struct aclskOptions {
  aclskOption *options;
  size_t numOptions;
} aclskOptions;

typedef enum {
  ACLSK_SCOPE_VERIFY_NODE_WAIT = 0,
  ACLSK_SCOPE_VERIFY_NODE_NOTIFY = 1,
  ACLSK_SCOPE_VERIFY_NODE_COMPUTE = 2,
} aclskScopeVerifyNodeType;

typedef enum {
  ACLSK_SCOPE_VERIFY_KERNEL_NO_AICORE = 0,
  ACLSK_SCOPE_VERIFY_KERNEL_CUBE = 1,    // AI CUBE CORE
  ACLSK_SCOPE_VERIFY_KERNEL_VECTOR = 2,  // AI VECTOR CORE
  ACLSK_SCOPE_VERIFY_KERNEL_MIX = 3,
} aclskScopeVerifyKernelType;

typedef struct aclskScopeVerifyNodeInfo {
  int64_t taskId;
  int64_t streamId;
  int64_t eventId;
  int32_t scopeId;
  aclskScopeVerifyNodeType taskType;
  aclskScopeVerifyKernelType kernelType;
  uint32_t numBlocks;
  uint32_t taskRatio[2];
  int32_t scheMode;
  uint32_t flag;  // bit 0：是否动态，0表示静态，1表示动态；其余bit预留
  int32_t coreLimit[2];
  int32_t extendType;  // 当前必须是0
  void *extendInfo;    // 当前必须是nullptr
} aclskScopeVerifyNodeInfo;

typedef enum {
  ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE = 0,  // scope 在 current node 前断开（全核同步）
  ACLSK_SCOPE_VERIFY_SPLIT_EXCLUDE_NODE = 2,  // scope 在 current node 前断开，且current node不参与scope融合（死锁）
} aclskScopeVerifySplitType;

typedef enum {
  ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED = 0,
  ACLSK_SCOPE_VERIFY_SYNCALL_OP_DROP = 1,
} aclskScopeVerifySplitReason;

/**
 * @ingroup AscendCL
 * @brief Optimize model with super kernel
 *
 * @param model [IN]    Model handle to be optimized
 * @param options [IN]  Pointer to optimization options
 *
 * @retval ACL_ERROR_SUCCESS Optimization succeeded
 * @retval ACL_ERROR_INVALID_PARAM Invalid parameters
 * @retval ACL_ERROR_FAILURE Optimization failed
 *
 * @see aclskOptions
 */
ACL_FUNC_VISIBILITY aclError aclskOptimize(aclmdlRI model, aclskOptions *options);
ACL_FUNC_VISIBILITY aclError aclskScopeBegin(const char *scopeName, aclrtStream stream);
ACL_FUNC_VISIBILITY aclError aclskScopeEnd(const char *scopeName, aclrtStream stream);
/**
 * @ingroup AscendCL
 * @brief
 *
 * @param verifyGraph             传入的图
 * @param maxSplitResultCount     传入的splitResults数组的最大长度
 * @param splitResults            导致scope断开的node列表
 * @param realSplitResultCount    导致scope断开的node数量
 * @retval ACL_SUCCESS 接口调用成功
 * @retval ACL_ERROR_INVALID_PARAM 参数校验失败
 * @retval ACL_ERROR_FAILURE 接口调用失败
 */
ACL_FUNC_VISIBILITY aclError aclskScopeVerify(const aclskScopeVerifyGraphInfo *verifyGraph, size_t maxSplitResultCount,
                                              aclskScopeVerifySplitResult *splitResults, size_t *realSplitResultCount);
```

## Relevance
`aclskOptimize(aclmdlRI model, aclskOptions *options)` takes an `aclmdlRI` — the model runtime instance handle
produced by `aclmdlRICaptureBegin`/`aclmdlRICaptureEnd`. So SuperKernel is applied to an already-captured ACL graph,
i.e. the vendor path is "capture the 3008 launches, then fuse their binaries", not "write one kernel".
`aclskScopeBegin(scopeName, stream)` / `aclskScopeEnd` delimit the fusion range on a stream.
`aclskScopeVerify` returns the nodes that *break* a scope, with reasons `ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED` and
`ACLSK_SCOPE_VERIFY_SYNCALL_OP_DROP` — i.e. a node whose own `SyncAll` would be dropped, or that would deadlock,
forcibly splits the super kernel. `aclskScopeVerifyNodeInfo` carries `kernelType` (NO_AICORE / CUBE / VECTOR / MIX),
`numBlocks`, `taskRatio[2]` (the AIC:AIV ratio), `scheMode` and `coreLimit[2]` — the exact per-node metadata the
fuser uses to decide sync scope. `SkInnerOptionType::MIX_KERNEL_SPLIT` and `UBUF_LOCK_IGNORE_KERNEL`
("Configure kernels that should ignore MIX kernel split by name pattern") confirm MIX kernels are handled by
splitting, and that UB locking is what forces the split.
`EARLY_START`, `PRELOAD_CODE`, `SPLIT_MODE.splitCnt`, `DCCI_BEFORE_KERNEL_START` / `DCCI_AFTER_KERNEL_END` are the
knobs corresponding to the four mechanisms above. vllm-ascend passes exactly
`{"dcci_after_kernel_end": [".*"]}` (see next section), i.e. a DCCI (data cache clean-and-invalidate) after every
fused sub-kernel.

---

## 7. SuperKernel as wired up in vllm-ascend (including a GLM-5.2 A3 config)

### vllm-ascend `docs/source/user_guide/feature_guide/graph_mode.md` — Super Kernel optimization

https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/graph_mode.md
raw: https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/user_guide/feature_guide/graph_mode.md
Project: vllm-project/vllm-ascend, branch `main`, fetched 2026-10-20.

> ### Super Kernel optimization
>
> [Super Kernel](https://www.hiascend.com/document/detail/en/Pytorch/latest/devguide/TorchAir/docs/en/npugraph_ex/advanced/superkernel.md) is an **optional** operator-binary fusion optimization. Unlike source-level fusion, it works on compiled kernel binaries: eligible subgraphs are identified, their child kernels are combined into a larger kernel, and synchronization is inserted according to graph dependencies. Compared with launching operators individually, this can reduce task scheduling waits, launch overhead, and operator-head overhead.
>
> In vLLM Ascend, Super Kernel optimization is applied during ACLGraph capture. It depends on both static kernel and Npugraph_ex:
>
> ```text
> Super Kernel -> static kernel -> Npugraph_ex
> ```
>
> !!! note
>
>     Static kernel is disabled by default, so Super Kernel is also disabled under the default configuration. When `enable_static_kernel` is explicitly set to `true` and `enable_super_kernel` is omitted, Super Kernel follows static kernel and is enabled. Set `enable_super_kernel` explicitly to override this inherited behavior. Super Kernel cannot be enabled when static kernel is disabled.
>
>     Not every operator is eligible for Super Kernel fusion. An unsupported operator may split the fusion range, so the actual coverage and performance benefit depend on the model and operator sequence. Changing the Super Kernel setting also changes the static-kernel compilation configuration and may cause the static-kernel cache to be rebuilt.

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true, "enable_static_kernel":true}}'
```

> To use static kernel without Super Kernel:
>
> ```bash
> vllm serve Qwen/Qwen2-7B-Instruct \
>   --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true, "enable_static_kernel":true, "enable_super_kernel":false}}'
> ```
>
> #### Verifying Super Kernel is active
>
> When Super Kernel optimization is applied during ACLGraph capture, vLLM Ascend emits:
>
> ```text
> Super kernel optimization is enabled for ACL graph capture.
> ```
>
> This message confirms that the optimization step was invoked. To verify the resulting fusion coverage and performance benefit, collect an **Ascend Profiling** trace and inspect `kernel_details.csv`. Compare the kernel/task entries and latency with Super Kernel disabled; the profiler also reports the launch core count for generated Super Kernels in the `Block Dim` field.

Static kernel compilation, verbatim:

> ### Static kernel compilation
>
> Static kernel compilation is an **optional** feature that pre-compiles operator binaries with fixed shapes at compile time, reducing runtime overhead for networks with static or near-static shapes. It is **disabled by default** and must be explicitly enabled.
>
> !!! note
>
>     Enabling static kernel triggers a compilation pass during the graph capture phase at service startup. This may add **several minutes to tens of minutes** to the startup time depending on the number of operators to compile and model complexity. Once completed, subsequent request processing is not affected.
>
> #### Verifying static kernel is active
>
> The recommended way to verify static kernel is in effect is through **Ascend Profiling**:
>
> 1. Collect a profiling trace of your running model using [Ascend PyTorch Profiler](...) (`torch_npu.profiler`).
> 2. Open the generated `op_statistic.csv` file.
> 3. Look for operators whose `op_type` or `name` column contains the keyword **`static_kernel`**. If such entries exist, static kernel compilation has taken effect for those operators.
>
> During the compilation phase, you will see a Python warning (visible by default):
>
> ```text
> Starting static kernel compilation, the build directory is <path>
> ```

Npugraph_ex, verbatim — the kernel-launch-reduction rationale:

> ## Using Npugraph_ex
>
> As introduced in the [RFC](https://github.com/vllm-project/vllm-ascend/issues/4715), Npugraph_ex is a compile-time FX graph optimization layer that works together with ACLGraph. It optimizes the model's FX graph before ACLGraph captures it at runtime. Its performance benefits mainly come from fusing multiple operators into single kernels (e.g., add + rms_norm → npu_add_rms_norm) to reduce kernel launch overhead.

> - Graph mode is currently available on both Model Runner V1/V2.
> - **ACLGraph** (capture/replay via `torch.npu.NPUGraph`) is the runtime graph execution mechanism used by the default graph path on Ascend.
> - **Npugraph_ex** is a compile-time FX graph optimization layer, enabled by default in FULL/FULL_DECODE_ONLY modes. It optimizes the graph before ACLGraph captures it.
> - **XliteGraph** is an optional graph path for selected model families and environments.
> - In context parallel scenarios, `cudagraph_mode="FULL"` is not sufficiently supported yet.

> | `cudagraph_mode` | Compile-time | Runtime | Npugraph_ex |
> |---|---|---|---|
> | FULL_AND_PIECEWISE | Piecewise compilation path | Mixed: PIECEWISE for mixed batches, FULL-capable for uniform decode batches | Disabled |
> | FULL / FULL_DECODE_ONLY | Npugraph_ex FX optimization | ACLGraph capture/replay | Enabled |
> | PIECEWISE | Fusion pass only | ACLGraph capture/replay | Disabled |
> | NONE | None | Eager execution | Disabled |

> Ascend sets `use_inductor=False` and selects its own compilation backend. That backend applies npugraph_ex optimizations or FX fusion passes according to the configuration.

Stream-resource exhaustion on capture, verbatim (relevant because a 3008-kernel graph is large):

> If ACL graph capture fails with the confirmed stream-resource signature in the error text, such as `207008` together with `Stream resources are insufficient` or `Insufficient_Stream_Resources`, vLLM Ascend will re-raise that capture failure with targeted mitigation guidance. In practice, the main levers are: upgrading to a newer HDK/CANN stack, reducing `cudagraph_capture_sizes`, lowering `max_cudagraph_capture_size`, or preferring `FULL` / `FULL_DECODE_ONLY` when the workload is mostly uniform decode.

### vllm-ascend source: how Super Kernel is actually invoked

https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/compilation/acl_graph.py
raw: https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/compilation/acl_graph.py

```python
from ..utils import super_kernel_scope, use_updatable_graph, weak_ref_tensors
...
_STREAM_RESOURCE_ERROR_CODE = "207008"
_STREAM_RESOURCE_ERROR_MARKERS = (
    "insufficient_stream_resources",
    "stream resources are insufficient",
)
_OLD_HDK_CAPTURE_ERROR_MARKERS = ("alloc sq cq fail",)
...
        self.enable_super_kernel = get_ascend_config().ascend_compilation_config.enable_super_kernel
...
                forward_context.capturing = True
                try:
                    with torch.npu.graph(aclgraph, pool=self.graph_pool):
                        # `output` is managed by pytorch's aclgraph pool
                        with super_kernel_scope("full_model", self.enable_super_kernel):
                            output = self.runnable(*args, **kwargs)
...
            if self.enable_super_kernel:
                aclgraph.super_kernel_optimize(
                    optimize_options={
                        "dcci_after_kernel_end": [".*"],
                    },
                )
                logger.info_once("Super kernel optimization is enabled for ACL graph capture.")
```

https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/utils.py

```python
@contextmanager
def super_kernel_scope(scope: str, enabled: bool):
    if not enabled:
        yield
        return

    torch.npu.super_kernel_scope_begin(scope)
    try:
        yield
    finally:
        torch.npu.super_kernel_scope_end(scope)
```

### A real GLM-5.2-W8A8C8 A3 config that turns on static kernel + super kernel

https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/cases/models/configs/GLM/GLM-5.2-W8A8C8-A3_128k-static-super-kernels.yaml
raw: https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/tests/e2e/cases/models/configs/GLM/GLM-5.2-W8A8C8-A3_128k-static-super-kernels.yaml

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
```

The decisive line, verbatim:

```
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
      --kv-cache-dtype int8
      --attention_config.indexer_kv_dtype int8
      --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "ascend_compilation_config": {"fuse_muls_add": true, "enable_npugraph_ex": true, "enable_static_kernel": true, "enable_super_kernel": true}, "multistream_overlap_shared_expert": true,"enable_fused_mc2":1,"enable_mlapo":true}'
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'
```

Benchmark entries attached to that config, verbatim:

```yaml
  perf_128k:
    case_type: performance
    dataset_path: vllm-ascend/GSM8K-in131072-bs500-prefix90-glm51
    request_conf: vllm_api_stream_chat
    dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_str_perf
    num_prompts: 256
    max_out_len: 1024
    batch_size: 64
    request_rate: 0
    baseline: 733.1813
    threshold: 0.97
```

### `additional_config.md` as of v0.11.0 (for comparison — `enable_super_kernel` not yet in this table)

https://raw.githubusercontent.com/vllm-project/vllm-ascend/v0.11.0rc0/docs/source/user_guide/configuration/additional_config.md

> | Name                          | Type | Default | Description                                                                                   |
> |-------------------------------| ---- |------|-----------------------------------------------------------------------------------------------|
> | `torchair_graph_config`       | dict | `{}` | The config options for torchair graph mode                                                    |
> | `ascend_scheduler_config`     | dict | `{}` | The config options for ascend scheduler                                                       |
> | `refresh`                     | bool | `false` | Whether to refresh global ascend config content. This value is usually used by rlhf or ut/e2e test case.     |
> | `expert_map_path`             | str  | `None` | When using expert load balancing for the MOE model, an expert map path needs to be passed in. |
> | `enable_prefetch`     | bool | `False` | Whether to enable weight prefetch. |
> | `kv_cache_dtype`     | str | `None` | When using the kv cache quantization method, kv cache dtype needs to be set, currently only int8 is supported. |
> | `enable_shared_expert_dp`     | bool | `False` | When the shared expert in DP, it has better performance but consumes more memory. Currently only DeepSeek series models are supported to use. |
> | `lmhead_tensor_parallel_size` | int | `None` | The custom tensor parallel size of lmhead. |
> | `oproj_tensor_parallel_size` | int | `None` | The custom tensor parallel size of oproj. |
> | `multistream_overlap_shared_expert`| bool | `False` | Whether to enable multistream shared expert. This option only takes effects on moe models with shared experts. |

> **torchair_graph_config**
>
> | Name | Type | Default | Description |
> | ---- | ---- | ------- | ----------- |
> | `enabled` | bool | `False` | Whether to enable torchair graph mode. Currently only DeepSeek series models and PanguProMoE are supported to use torchair graph mode |
> | `mode` | str | `None` | When using reduce-overhead mode for torchair, mode needs to be set |
> | `enable_multistream_mla`| bool | `False` | Whether to put vector ops of MLA to another stream. This option only takes effects on models using MLA (e.g., DeepSeek). |
> | `enable_view_optimize` | bool | `True` | Whether to enable torchair view optimization |
> | `enable_frozen_parameter` | bool | `True` | Whether to fix the memory address of weights during inference to reduce the input address refresh time during graph execution. |
> | `use_cached_graph` | bool | `False` | Whether to use cached graph |
> | `graph_batch_sizes` | list[int] | `[]` | The batch size for torchair graph cache |
> | `graph_batch_sizes_init` | bool | `False` | Init graph batch size dynamically if `graph_batch_sizes` is empty |
> | `enable_kv_nz`| bool | `False` | Whether to enable kvcache NZ layout. This option only takes effects on models using MLA (e.g., DeepSeek). |

Earlier doc versions (v0.9.1 / v0.11.0 published docs) described the option as:

https://docs.vllm.ai/projects/ascend/en/v0.9.1/user_guide/configuration/additional_config.html

> `enable_super_kernel` | bool | `False` | Whether to enable super kernel

## Relevance
This is a working, vendor-maintained configuration for *our model family on our hardware* (GLM-5.2-W8A8C8, A3, 16 NPUs
per node) that enables exactly the launch-overhead attack we need. It establishes the dependency chain
`Super Kernel -> static kernel -> Npugraph_ex` and the three-flag incantation
`{"ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": true, "enable_super_kernel": true}}`
plus `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`. The quoted benefit is exactly our target:
"reduce task scheduling waits, launch overhead, and operator-head overhead". The actual API call is
`aclgraph.super_kernel_optimize(optimize_options={"dcci_after_kernel_end": [".*"]})` on a `torch.npu.NPUGraph`, plus
`torch.npu.super_kernel_scope_begin/end(scope)` around the forward. Two caveats stated by the vendor bear on our
3008-kernel count: (a) "An unsupported operator may split the fusion range" — coverage is sequence-dependent, so the
505 MIX_AIC calls may each be a potential split point; (b) `Block Dim` in `kernel_details.csv` reports the launch core
count for generated Super Kernels, which is how we would measure whether our MIX stages got fused or split.

---

## 8. Buffer capacities and AI Core architecture on 910B / A2 / A3 (`__NPU_ARCH__` 220x)

### `cann/catlass` `platform_info.h` — concrete default buffer sizes in vendor source

https://gitcode.com/cann/catlass/blob/master/examples/102_dynamic_optimized_matmul/include/platform_info.h
raw: https://raw.gitcode.com/cann/catlass/raw/master/examples/102_dynamic_optimized_matmul/include/platform_info.h
Repo `cann/catlass`, branch `master`. Copyright (c) 2025 Huawei Technologies Co., Ltd.

```cpp
#include "tiling/platform/platform_ascendc.h"

struct PlatformInfo {
    uint32_t coreNum{24};
    uint64_t ubSize{192 * 1024};
    uint64_t l1Size{512 * 1024};
    uint64_t l0ASize{64 * 1024};
    uint64_t l0BSize{64 * 1024};
    uint64_t l0CSize{128 * 1024};

    PlatformInfo()
    {
        coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::UB, ubSize);
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L1, l1Size);
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L0_A, l0ASize);
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L0_B, l0BSize);
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L0_C, l0CSize);
    }
};
```

Summarising the literal numbers in that struct: `coreNum = 24`, `UB = 192 KiB`, `L1 = 512 KiB`,
`L0A = 64 KiB`, `L0B = 64 KiB`, `L0C = 128 KiB`. The constructor immediately overwrites all of them from
`platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(CoreMemType::{UB,L1,L0_A,L0_B,L0_C}, ...)`,
so these are documented-as-defaults, not a hard spec. `GetCoreNumAic()` is the AIC (cube) core count.

### CANN 9.0 算子开发指南 — 基本架构 (Basic Architecture)

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0008.html
English: https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html
Publisher: Huawei / Ascend, parent topic 「硬件实现」.

> AI Core负责执行矩阵、矢量计算密集的任务，其包括以下组成部分：
>
> 计算单元：包括Cube（矩阵）计算单元、Vector（矢量）计算单元和Scalar（标量）计算单元。
>
> 存储单元：包括L1 Buffer、L0A Buffer、L0B Buffer、L0C Buffer、Unified Buffer、BiasTable Buffer、Fixpipe Buffer等专为高效计算设计的存储单元。
>
> 搬运单元：包括MTE1、MTE2、MTE3和FixPipe，用于数据在不同存储单元之间的高效传输。

关键概念和术语, verbatim:

> Core
>
> 拥有独立Scalar计算单元的计算核。Scalar计算单元是核内的调度单元，承担了核内的指令发射等功能。
>
> AI Core
>
> AI处理器的计算核，负责执行矩阵、矢量计算密集的任务。
>
> Cube Core
>
> 矩阵计算核，专注于矩阵计算。由Scalar调度单元、矩阵计算单元、搬运单元等组成，不包括矢量计算单元。
>
> Vector Core
>
> 矢量计算核，专注于矢量计算，由Scalar调度单元、矢量计算单元、搬运单元等组成，不包括矩阵计算单元。
>
> AIC
>
> 在AI Core分离模式下，一组Cube Core和Vector Core组合中的Cube Core。
>
> AIV
>
> 在AI Core分离模式下，一组Cube Core和Vector Core组合中的Vector Core。

AI Core 的工作模式 — 分离模式 vs 耦合模式, verbatim:

> 分离模式
>
> AI Core的一种工作模式，矩阵计算单元、矢量计算单元各自对应独立的Scalar调度单元，分离部署在Cube Core和Vector Core上。将Cube Core和Vector Core按照一定比例（1：N）进行组合，这样的组合视为一个AI Core，AI Core的核数以Cube Core为准。
>
> 耦合模式
>
> AI Core的一种工作模式，矩阵计算单元、矢量计算单元对应同一个Scalar调度单元，部署在一个AI Core上。
>
> Ascend C编程中，不同产品的工作模式如下：
>
> Atlas 推理系列产品：耦合模式
>
> Atlas 训练系列产品：耦合模式
>
> Atlas A2 训练系列产品 / Atlas A2 推理系列产品：分离模式
>
> Atlas A3 训练系列产品 / Atlas A3 推理系列产品：分离模式
>
> Atlas 350 加速卡：分离模式
>
> Atlas 200I/500 A2 推理产品：耦合模式

Cube / Vector / Scalar units, verbatim:

> Cube
>
> Cube计算单元负责执行矩阵运算，以float16数据类型为例，Cube每次执行可完成两个float16类型的16x16矩阵的乘法操作。如下图所示，高亮部分为Cube计算单元及其访问的存储单元，其中L0A存储左矩阵，L0B存储右矩阵，L0C存储矩阵乘的结果和中间结果。
>
> Vector
>
> Vector负责执行向量运算。向量计算单元执行向量指令，类似于传统的单指令多数据（Single Instruction Multiple Data，SIMD）指令，每个向量指令可以完成多个操作数的同一类型运算。向量计算单元可以快速完成两个float16类型的向量相加或者相乘。向量指令支持多次迭代执行，也支持对带有间隔的向量直接进行运算。
>
> 如下图所示，Vector所有计算的源数据以及目标数据都要求存储在Unified Buffer中，Vector指令的首地址和操作长度有对齐要求，通常要求32B对齐，具体对齐要求参考API的约束描述。
>
> Scalar
>
> Scalar负责各类型的标量数据运算和程序的流程控制。功能上可以看做一个小CPU，完成整个程序的循环控制、分支判断、Cube/Vector等指令的地址和参数计算以及基本的算术运算，并且可以通过在事件同步模块中插入同步符的方式来控制AI Core中其他执行单元的流水。相对于Host CPU，AI Core中的Scalar计算能力较弱，重点用于发射指令，所以在实际应用场景中应尽量减少Scalar计算，比如性能调优时尽量减少if/else等分支判断及变量运算。

ICache / DCache / L2 — exact figures as published, verbatim:

> 如下图所示：Scalar执行标量运算指令时，执行标准的ALU(Arithmetic Logic Unit)语句，ALU需要的代码段和数据段（栈空间）都来自于GM，ICache(Instruction Cache)用于缓存代码段，缓存大小与硬件规格相关，比如为16K或32K，以2K为单位加载；DCache(Data Cache)用于缓存数据段，大小也与硬件规格相关，比如为16K，以Cache Line（64Byte）为单位加载。考虑到核内访问效率最高，应尽量保证代码段和数据段被缓存在ICache和DCache，避免核外访问； 同时根据数据加载单位不同，编程时可以考虑单次加载数据大小，来提升加载效率。例如在DCache加载数据时，当数据内存首地址与Cache Line（64Byte）对齐时，加载效率最高。
>
> 硬件提供L2Cache用于缓存访问GM的数据（包括代码段、数据段），以此加快访问速度，提高访问效率。核外L2Cache以Cache Line为单位加载数据，根据硬件规格不同，Cache Line大小不同（128/256/512Byte等）。

存储单元 table, verbatim:

> | 存储单元 | 描述 |
> |---|---|
> | L1 Buffer | L1缓冲区，通用内部存储，是AI Core内比较大的一块数据中转区，可暂存Cube计算单元需要反复使用的一些数据从而减少从总线读写的次数。 |
> | L0A Buffer / L0B Buffer | Cube指令的输入。 |
> | L0C Buffer | Cube指令的输出，但进行累加计算的时候，也是输入的一部分。 |
> | Unified Buffer | 统一缓冲区，向量和标量计算的输入和输出。 |
> | BT Buffer | BiasTable Buffer，存放矩阵计算中的Bias。 |
> | FP Buffer | Fixpipe Buffer，存放量化参数、Relu参数等。 |

搬运单元 table, verbatim:

> | 搬运单元 | 描述 |
> |---|---|
> | MTE1 | 负责如下通路的数据搬运：<br>L1->L0A/L0B<br>L1->BT Buffer |
> | MTE2 | 负责如下通路的数据搬运：<br>GM->{L1, L0A/B}，在该通路下，基于分形大小搬运，搬运时满足Cache Line大小对齐，性能更优。<br>GM->UB，基于Cache Line大小搬运性能更优。 |
> | MTE3 | 负责如下通路的数据搬运：<br>UB -> GM |
> | FixPipe | 负责如下通路的数据搬运，搬运过程中可以完成随路数据格式/类型转换：<br>L0C->{GM/L1}<br>L1->FP Buffer |

> 不同类型的AI处理器，存储单元大小不同，开发者可通过GetCoreMemSize接口获取。
>
> 所有通过搬运单元读写GM的数据都缺省被缓存在L2Cache，以此加快访问速度，提高访问效率。

典型的数据流, verbatim:

> Vector计算典型的数据流如下：
>
> GM → UB → Vector → UB → GM
>
> Cube计算典型的数据流如下：
>
> GM →L1→L0A/L0B →Cube →L0C→FixPipe→GM
>
> GM →L1→L0A/L0B →Cube →L0C→FixPipe→L1

典型的指令流 — pipe parallelism and the two sync APIs, verbatim:

> 多条指令从系统内存通过总线接口进入到ICache(Instruction Cache)中，后续的指令执行过程，根据指令的类型，有两种可能：
>
> 如果指令是Scalar指令，指令会被Scalar单元直接执行。
>
> 其他指令会被Scalar单元调度到独立的分类序列（Vector指令序列、Cube指令序列、MTE1/MTE2/MTE3指令序列等），然后再被对应的执行单元执行。
>
> 同一个指令序列中的指令是按照进入指令序列的顺序执行的，不同指令序列之间可以并行执行，通过多个指令序列的并行执行可以提升整体执行效率。对于并行执行过程中可能出现的数据依赖，通过事件同步模块插入同步指令来控制流水线的同步，提供PipeBarrier、SetFlag/WaitFlag两种API，保证序列内部以及序列之间按照逻辑关系执行。
>
> PipeBarrier本身是一条指令，用于在序列内部约束执行顺序（虽然指令是顺序执行，但并不意味着后一条指令开始执行时前一条指令执行结束）。PipeBarrier指令可以保证前序指令中所有数据读写全部完成，后序指令才开始执行。
>
> SetFlag/WaitFlag为两条指令，在SetFlag/WaitFlag的指令中，可以指定一对指令序列的关系，表示两个序列之间完成一组"锁"机制，其作用方式为：
>
> SetFlag：当前序指令的所有读写操作都完成之后，当前指令开始执行，并将硬件中的对应标志位设置为1。
>
> WaitFlag：当执行到该指令时，如果发现对应标志位为0，该序列的后续指令将一直被阻塞；如果发现对应标志位为1，则将对应标志位设置为0，同时后续指令开始执行。

### CANN 9.0 架构规格 — NPU架构版本220x (this is A2 **and** A3)

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0011.html
Publisher: Huawei / Ascend, parent topic 「架构规格」.

> 本节介绍__NPU_ARCH__版本号为220x的硬件架构和其功能说明，其中220代表IP核编号，x表示同一个IP核的配置版本号。对应的产品型号为：
>
> Atlas A3 训练系列产品 / Atlas A3 推理系列产品
>
> Atlas A2 训练系列产品 / Atlas A2 推理系列产品

The decisive sentence for megakernel data flow, verbatim:

> 如下图所示，本架构中AI Core分为AIC和AIV两个独立的核，分别用于矩阵计算和向量计算。每个核都有自己的Scalar单元，能独立加载自己的代码段。AIV与AIC之间通过Global Memory进行数据传递。

> 本架构中，Cube计算单元和Vector计算单元分别部署在AIC核和AIV核上，每个核都有自己的Scalar单元，能独立加载自己的代码段。
>
> Vector计算单元
>
> Vector计算单元的数据来自于Unified Buffer，要求32字节对齐。
>
> Cube计算单元
>
> Cube计算单元可以访问的存储单元有L0A Buffer、L0B Buffer、L0C Buffer，其中L0A Buffer存储左矩阵，L0B Buffer存储右矩阵，L0C Buffer存储矩阵乘的结果和中间结果。

各存储单元的最小访问粒度（对齐要求）, verbatim:

> | 核 | 存储单元 | 对齐要求 |
> |---|---|---|
> | AIV | Unified Buffer | 32Byte对齐。 |
> | AIC | L1 Buffer | 32Byte对齐。 |
> | AIC | L0A Buffer | 512Byte对齐。 |
> | AIC | L0B Buffer | 512Byte对齐。 |
> | AIC | L0C Buffer | 64Byte对齐。 |
> | AIC | BiasTable Buffer | 64Byte对齐。 |
> | AIC | Fixpipe Buffer | 64Byte对齐。 |

> 各存储单元推荐使用的数据排布格式
>
> L0A Buffer、L0B Buffer和L0C Buffer推荐分别采用以下分形格式：
>
> L0A Buffer：FRACTAL_ZZ
>
> L0B Buffer：FRACTAL_ZN
>
> L0C Buffer：FRACTAL_NZ
>
> 这些格式针对矩阵乘法等计算密集型任务进行优化，可显著提升计算效率。
>
> L1 Buffer缓存推荐使用FRACTAL_NZ格式。当L1 Buffer采用NZ格式时，数据搬运到L0A/L0B Buffer（需分别转换为ZZ和ZN格式）时，可降低格式转换开销。
>
> Unified Buffer对数据格式没有要求。

> 当多个操作尝试同时访问Unified Buffer同一个bank或者bank group时，可能会发生bank冲突，包括读写冲突、写写冲突、读读冲突，这种冲突会导致访问排队，降低性能。可以通过优化bank分配的方式来提升读写性能，具体信息请参考避免Unified Buffer的bank冲突章节。

Fixpipe hardware acceleration on AIC — directly relevant to W4A8 dequant, verbatim:

> 支持Fixpipe硬件化加速
>
> Fixpipe是NPU将典型操作进行硬化的加速模块，位于AIC内部，配合Cube计算单元完成随路计算，主要功能如下：
>
> 量化反量化：包括S322FP16、S322S32、S322S4、S322S8、S322S16、FP322FP16、FP322BF16、FP322S8、FP322S4、FP322FP32。
>
> Relu功能，包括ReLu、PReLu和Leaky ReLu等典型的激活函数。
>
> 数据格式转换，包括：
>
> 通过Channel merge、Channel split可以实现分形大小的转换，保证输出到L1 Buffer/GM的分形满足诉求。
>
> NZ2ND数据格式转换。
>
> Channel merge（S8和U8数据类型）
>
> 对于转换为S8或U8的目标数据类型，分形矩阵通过硬件从16x16转换为16x32，如果输出通道数N是16的偶数倍，则N方向上每2个相邻的16x16分形矩阵将合并为1个16x32分形矩阵。如果N是16的奇数倍，则将通道1到通道（N–16）合并，最后16个通道保持未合并。
>
> Channel merge（S4和U4数据类型）
>
> 对于转换为S4或U4的目标数据类型，分形矩阵通过硬件从16x16转换为16x64，如果输出通道数N是64的倍数，则N方向上每4个相邻的16x16分形矩阵将合并为1个16x64分形矩阵。
>
> 在这种情况下，N的配置必须是64的倍数。
>
> FP32 Channel split：
>
> 对于目标类型为FP32，分形矩阵可以通过硬件从16x16转换为16x8，如果使能Channel split，则每个16x16分形矩阵将被分裂为2个16x8分形矩阵。

核内同步 — EventID discipline, verbatim:

> 上图中，ID1、ID2、ID3、ID4、ID5、ID6表示事件ID（EventID），每个EventID对应一块存储数据的搬运状态，确保数据操作的正确性和一致性。
>
> 需要注意以下几点：
>
> 建议通过AllocEventID或者FetchEventID接口获取EventID，以确保其合法性和有效性。
>
> EventID的数量有限，使用后应立即调用ReleaseEventID释放资源，避免EventID耗尽，影响系统正常运行。
>
> SetFlag和WaitFlag必须成对使用，且SetFlag和WaitFlag的参数必须完全一致（包括模板参数和事件ID）。如果不匹配，可能导致当前核的计算异常，或影响下一个核的算子执行，引发timeout问题。
>
> 例如，SetFlag<HardEvent::S_MTE3>(1)和SetFlag<HardEvent::MTE3_MTE1>(1)设置的不是同一个EventID，因为其模板参数不同。只有当模板参数和事件ID完全一致时，才表示同一个EventID。
>
> 不允许连续设置同一个EventID，因为这可能导致事件状态混乱或未被正确处理。
>
> 不建议手动插入 TEventID，不能手动插入6和7的TEventID，因为它们可能被系统预留或用于特殊用途。

核间同步 — the counter semantics, verbatim (this is more precise than the API reference page):

> CrossCoreSetFlag和CrossCoreWaitFlag接口配合使用。使用时需传入核间同步的标记ID(flagId)，即上图中的ID1，每个ID对应一个初始值为0的计数器。执行CrossCoreSetFlag后ID对应的计数器增加1；执行CrossCoreWaitFlag时如果对应的计数器数值为0则阻塞不执行；如果对应的计数器大于0，则计数器减一，同时后续指令开始执行。flagId取值范围是0-10。
>
> 需要注意以下几点：
>
> 成对使用
>
> CrossCoreSetFlag和CrossCoreWaitFlag必须成对使用，否则可能导致算子超时问题。
>
> 一致性要求
>
> CrossCoreSetFlag 的模板参数和flagId必须与CrossCoreWaitFlag完全一致，否则视为不同的flagId。例如，CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8) 和 CrossCoreSetFlag<0x2, PIPE_FIX>(0x8) 设置的不是同一个flagId。
>
> 避免连续设置
>
> 不允许连续设置同一个flagId，以防止计数器状态混乱。
>
> 与高阶 API 的使用冲突
>
> Matmul高阶API内部实现中使用了本接口进行核间同步控制，所以不建议开发者同时使用该接口和Matmul高阶API，否则会有flagId冲突的风险。
>
> 计数器限制
>
> 同一flagId的计数器最多可以设置15次。
>
> 默认流水类型
>
> CrossCoreWaitFlag不需要显式设置指令所在的流水类型，默认使用PIPE_S。

Also from the same page, cross-card DataCopy on A2 and the example motivating cross-core sync:

> 支持跨卡数据搬运（Hccs物理链路）
>
> 在跨卡通信算子开发场景，DataCopy类接口支持跨卡数据搬运，在 Atlas A2 训练系列产品 / Atlas A2 推理系列产品 设备上，仅支持Hccs物理链路，不支持其他通路；开发者开发过程中，请关注涉及卡间通信的物理通路； 通过npu-smi info -t topo指令查询Hccs 物理通路。

> 例如，在AIC中将L0C的计算结果搬运到GM后，AIV需要将GM的数据搬运到UB。此时，可以使用CrossCoreSetFlag和CrossCoreWaitFlag命令，确保数据从L0C成功搬运到GM后，再从GM搬运到UB。

## Relevance
Three facts here constrain the megakernel more than anything else we found.
(1) On 220x (our A3) 「AIV与AIC之间通过Global Memory进行数据传递」 — the AIC and AIV halves of an AI Core exchange
data **through Global Memory**, not through a shared on-chip buffer. The cube result path is
`L0C -> FixPipe -> GM -> (MTE2) -> UB`. So "matmul -> vector -> matmul without returning to HBM" is only achievable
within one core class, or by relying on L2 to absorb the GM round trip. A fully HBM-free cube->vector->cube chain
on A3 is not available at the hardware level; the realistic goal is L2-resident, not UB-resident.
(2) 「每个核都有自己的Scalar单元，能独立加载自己的代码段」 — AIC and AIV each load their own code segment.
This is why a MIX kernel launch has to bring up two distinct code images, and it is the hardware basis for the
ICache-preload and sub-kernel-split problems SuperKernel addresses. ICache is only 「16K或32K，以2K为单位加载」 —
a megakernel's body will far exceed it, so instruction fetch becomes a first-order cost.
(3) The CrossCore counter is literally a saturating counter: Set increments, Wait decrements-if-nonzero, range
flagId 0-10, max 15 Sets per flagId, no two consecutive Sets of the same flagId, and `CrossCoreWaitFlag` defaults to
`PIPE_S`. A persistent kernel iterating layers must budget flag IDs and interleave Waits so no counter exceeds 15.
Buffer budget for tiling: UB 192 KiB, L1 512 KiB, L0A/L0B 64 KiB each, L0C 128 KiB, with L0A/L0B needing 512-byte
alignment and FRACTAL_ZZ/ZN/NZ layouts. Finally, `Fixpipe` on AIC does S32->S8/S4/FP16/BF16 requantization and
ReLU/PReLU/LeakyReLU in-line with the cube output — for W4A8 that means the dequant step can ride the cube pipeline
instead of costing a separate AIV kernel, which is a direct attack on our MIX call count.

---

## 9. `SyncAll`, `IBSet`/`IBWait` — grid barriers, hard vs soft sync

### SyncAll (CANN 9.0 API reference, 核间同步)

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0204.html
Parent topic 「核间同步」, whose TOC (https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0268.html) is exactly:
`IBSet / IBWait / SyncAll / CrossCoreSetFlag(ISASI) / CrossCoreWaitFlag(ISASI) / InitDetermineComputeWorkspace / NotifyNextBlock / WaitPreBlock`.

产品支持情况 — note the two separate columns, 软同步原型 and 硬同步原型:

> | 产品 | 是否支持（软同步原型） | 是否支持（硬同步原型） |
> |---|---|---|
> | Atlas 350 加速卡 | √ | √ |
> | Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ | √ |
> | Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ | √ |
> | Atlas 200I/500 A2 推理产品 | x | x |
> | Atlas 推理系列产品 AI Core | √ | x |
> | Atlas 推理系列产品 Vector Core | x | x |
> | Atlas 训练系列产品 | √ | x |

功能说明, verbatim:

> 当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。目前多核同步分为硬同步和软同步，硬件同步是利用硬件自带的全核同步指令由硬件保证多核同步，软件同步是使用软件算法模拟实现。

函数原型, verbatim:

```cpp
// 软同步
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, const int32_t usedCores = 0)

// 硬同步
template <bool isAIVOnly = true, const SyncAllConfig& config = DEFAULT_SYNC_ALL_CONFIG>
__aicore__ inline void SyncAll()
```

表1 模板参数说明, verbatim:

> | 参数名 | 描述 |
> |---|---|
> | isAIVOnly | 控制SyncAll作用于纯Vector算子或融合（Cube和Vector融合）算子。可选值：<br>true（默认值）：纯Vector算子的全核同步，仅执行Vector核的全核同步。<br>false：融合算子的全核同步，先分别完成Vector核和Cube核的全核同步，再执行两者之间的同步（软同步接口不支持此功能）。 |
> | config | 控制SyncAll函数的行为 ，在多个AI Core之间进行流水线同步时，指定哪些管道（pipe）用于触发和等待。<br>triggerPipe：指定哪个管道用于"发送触发信号"。<br>waitPipe：指定哪个管道用于"接收等待信号"。<br>默认为SyncAllConfig DEFAULT_SYNC_ALL_CONFIG= {PIPE_ALL, PIPE_ALL} ，使用全部管道来进行触发和等待行为。<br>该参数仅支持如下型号：<br>Atlas 350 加速卡 |

表2 参数说明, verbatim:

> | 参数名称 | 输入/输出 | 含义 |
> |---|---|---|
> | gmWorkspace | 输入 | gmWorkspace为用户定义的全局Global空间，作为所有核共用的缓存，用于保存每个核的状态标记，类型为GlobalTensor，支持的数据类型为int32_t。所需空间大小和使用注意项参见约束说明。硬同步接口不支持该参数。 |
> | ubWorkspace | 输入 | ubWorkspace为用户定义的局部Local空间，每个核单独自用，用于标记当前核的状态。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT，支持的数据类型为int32_t。所需空间大小参见约束说明。硬同步接口不支持该参数。 |
> | usedCores | 输入 | 指定多少个核之间的同步，传入数值不能超过算子调用时指定的逻辑numBlocks。此参数为默认参数，不传此参数表示全核软同步。仅在软同步接口中支持，硬同步接口不支持该参数。 |

约束说明, verbatim:

> gmWorkspace缓存申请的空间大小要求大于等于核数*32Bytes，并且缓存的值需要初始化为0。目前常见的有两种初始化方式：
>
> 通过在host侧进行初始化操作，确保传入该接口时，gmWorkspace缓存已经初始化为0；
>
> 在kernel侧初始化的时候对gmWorkspace缓存初始化，需要注意的是，每个核上都需要初始化全部的gmWorkspace缓存空间。
>
> ubWorkspace申请的空间大小要求大于等于核数*32Bytes。
>
> 使用该接口进行多核控制时，算子调用时指定的逻辑numBlocks必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel"卡死"现象。
>
> 在分离模式下，建议使用硬同步接口而非软同步接口。软同步接口仅适用于纯Vector场景，且性能较低。使用硬同步接口时，需根据场景设置Kernel类型：
>
> 在纯Vector/Cube场景下，需设置Kernel类型为KERNEL_TYPE_MIX_AIV_1_0或KERNEL_TYPE_MIX_AIC_1_0。
>
> 对于Vector和Cube混合场景，需根据实际情况灵活配置Kernel类型。
>
> 使用该接口时，建议开启batchmode模式，使算子独占全部所需核资源，否则可能因满足以下条件导致死锁：
>
> 多流并发场景（≥2条执行流）。
>
> ≥2个算子并发执行。
>
> 所有并发算子的核数总和超过物理核数。
>
> ≥2个并发算子使用了核间同步功能。
>
> Kernel直调场景下通过__schedmode__(mode)限定符来设置batchmode模式；工程化算子开发场景下，通过TilingContext的SetScheduleMode接口来设置batchmode模式。

调用示例, verbatim:

```cpp
// syncGlobal为用户定义的全局Global空间，作为所有核共用的缓存，类型为GlobalTensor；workLocal为用户定义的局部Local空间，每个核单独自用，类型为LocalTensor
int srcDataSize = 256; // 参与计算的元素个数
int32_t blockNum = AscendC::GetBlockNum(); // 获取核总数
int32_t blockIdx = AscendC::GetBlockIdx(); // 获取当前工作的核ID
uint32_t perBlockSize = srcDataSize / blockNum; // 每个核平分处理相同个数
// 当前工作核计算后的数据先保存到外部工作空间，workGlobal为GlobalTensor，dstLocal为LocalTensor
AscendC::DataCopy(workGlobal[blockIdx * perBlockSize], dstLocal, perBlockSize);
// 等待所有核都完成计算
AscendC::SyncAll(syncGlobal, workLocal);
```

### IBSet / IBWait (point-to-point inter-block sync)

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0202.html (IBSet)
https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0203.html (IBWait)

```cpp
__aicore__ inline void IBSet(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, int32_t blockIdx, int32_t eventID)
```

> | 参数名 | 输入/输出 | 描述 |
> |---|---|---|
> | gmWorkspace | 输出 | 外部存储核状态的公共缓存，类型为GlobalTensor。 |
> | ubWorkspace | 输入 | 存储当前核状态的公共缓存。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
> | blockIdx | 输入 | 表示等待核的idx号，取值范围：[0, 核数-1]。 |
> | eventID | 输入 | 用来控制当前核的set、wait事件。 |

约束说明, verbatim — the exact workspace sizing formula:

> gmWorkspace申请的空间最少要求为：核数 * 32Bytes * eventID_max + blockIdx_max * 32Bytes + 32Bytes。（eventID_max和blockIdx_max分别指eventID、blockIdx的最大值 ）；
>
> 注意：如果是AIVOnly模式，核数 = GetBlockNum()；如果是MIX模式，核数 = GetBlockNum() * 2；
>
> ubWorkspace申请的空间最少要求为：32Bytes；
>
> gmWorkspace缓存的值需要初始化为0。
>
> 使用该接口进行多核控制时，算子调用时指定的逻辑numBlocks必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel"卡死"现象。

```cpp
// sync_gm为外部存储核状态的公共缓存，类型为GlobalTensor；sync_buf为存储当前核状态的公共缓存，类型为LocalTensor
int32_t blockIdx = AscendC::GetBlockIdx(); // 获取当前核
if (blockIdx == 1) { // 在核1设置IBWait，将阻塞指令执行直到核0操作完成
    AscendC::IBWait(sync_gm, sync_buf, 0, 0);
}
...
if (blockIdx == 0) { // 在核0设置IBSet，当核0的操作完成后再执行核1的指令
    AscendC::IBSet(sync_gm, sync_buf, 0, 0);
}
```

## Relevance
For a persistent megakernel on A3 (分离模式) the doc is explicit: use the **hard-sync** zero-argument
`SyncAll<isAIVOnly>()`, not the GM-workspace soft variant — 「在分离模式下，建议使用硬同步接口而非软同步接口。
软同步接口仅适用于纯Vector场景，且性能较低」. `isAIVOnly = false` is the fused-operator barrier and it costs three
barriers in sequence: vector all-core sync, then cube all-core sync, then a sync between the two. That tri-phase
barrier is the same thing SuperKernel's 同步优化 narrows, and it is the per-stage price of staying inside one MIX kernel.
The hard-sync prototype requires `KERNEL_TYPE_MIX_AIV_1_0` or `KERNEL_TYPE_MIX_AIC_1_0` even in a pure vector/cube
kernel — so a megakernel that uses `SyncAll` is forced into MIX typing, i.e. it cannot be an `AIV_ONLY` kernel.
The hard constraint 「逻辑numBlocks必须保证不大于实际运行该算子的AI处理器核数」 means a megakernel must be launched with
blockDim <= physical cores or it hangs; multi-round scheduling inserts a bad barrier. `SyncAllConfig{triggerPipe, waitPipe}`
would let us narrow the barrier to specific pipes but is **Atlas 350 only** — not available on A3.
`IBSet`/`IBWait` is the point-to-point alternative, cheap in flags but needing
`核数 * 32B * eventID_max + blockIdx_max * 32B + 32B` of GM, and note that in MIX mode 核数 = `GetBlockNum() * 2`.

---

## 10. `ICachePreLoad` — exact ICache sizes on A3, and the instruction-prefetch API a megakernel needs

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0276.html
Parent topic 「缓存控制」 (sibling: `DataCachePreload`).

> 功能说明
>
> 从指令所在DDR地址预加载指令到ICache中。

```cpp
__aicore__ inline void ICachePreLoad(const int64_t preFetchLen)
```

> | 参数名 | 输入/输出 | 描述 |
> |---|---|---|
> | preFetchLen | 输入 | 预取长度。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品：preFetchLen参数单位为2K Byte, 取值应小于ICache的大小/2K。AIC和AIV的ICache大小分别为32KB和16KB。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品：preFetchLen参数单位为2K Byte, 取值应小于ICache的大小/2K。AIC和AIV的ICache大小分别为32KB和16KB。<br>针对Atlas 推理系列产品AI Core：传入该参数无效，预取长度均为128Byte。<br>针对Atlas 350 加速卡：preFetchLen参数单位为2K Byte, 取值应小于ICache的大小/2K。AIC和AIV的ICache大小分别为32KB和16KB。 |

> 产品支持情况: Atlas 350 加速卡 √ / Atlas A3 训练系列产品/Atlas A3 推理系列产品 √ / Atlas A2 训练系列产品/Atlas A2 推理系列产品 √ / Atlas 200I/500 A2 推理产品 x / Atlas 推理系列产品AI Core √ / Atlas 推理系列产品Vector Core x / Atlas 训练系列产品 x

```cpp
int64_t preFetchLen = 2; // 预取指令长度
AscendC::ICachePreLoad(preFetchLen);
```

## Relevance
This is the primitive behind SuperKernel's ICache Preload mechanism, and it is available on A3. Exact figures:
**AIC ICache = 32 KB, AIV ICache = 16 KB**, prefetch granularity 2 KB, so `preFetchLen < 16` on AIC and `< 8` on AIV.
A megakernel that inlines 78 layers of stages will blow a 16 KB AIV ICache by orders of magnitude, so the design must
either (a) keep each stage's hot loop under ~16 KB and call `ICachePreLoad` ahead of the next stage, or (b) accept
instruction-fetch stalls from L2. This is the quantitative constraint that caps how much we can inline before
ICache misses eat the launch savings we are chasing — our 11.5 ms launch floor is only worth attacking if the fused
body does not reintroduce an equivalent fetch cost.

---

## 11. Matmul high-level API: `ASCENDC_CUBE_ONLY`, `MatmulType`, `TPosition::CO1`, `TPosition::TSCM`, `SetLocalWorkspace`

### CANN 9.0 API reference — Matmul使用说明

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0614.html
Parent topic 「Matmul Kernel侧接口」 (TOC id 0613), itself under 「矩阵计算」 (0612) under 「高阶API」.
The 0613 TOC lists, verbatim: `Matmul使用说明 / Matmul模板参数 / MatmulConfig / MatmulCallBackFunc / MatmulPolicy /
Matmul特性场景 / GetNormalConfig / GetMDLConfig / GetSpecialMDLConfig / GetIBShareNormConfig / GetBasicConfig /
GetMMConfig / REGIST_MATMUL_OBJ / Init / SetTensorA / SetTensorB / SetTensorScaleA / SetTensorScaleB / SetBias /
DisableBias / GetBatchTensorC / Iterate / GetTensorC / IterateAll / WaitIterateAll / IterateBatch /
WaitIterateBatch / IterateNBatch / End`.

The MIX-vs-pure-cube mode statement, verbatim — note the explicit performance claim:

> 默认为MIX模式（包含矩阵计算和矢量计算），该场景下通常不定义ASCENDC_CUBE_ONLY宏，如果在程序中使用了ASCENDC_CUBE_ONLY宏，则必须使用ASCEND_IS_AIC宏和ASCEND_IS_AIV宏将Cube计算和Vector计算隔离开。
>
> 纯Cube模式（只有矩阵计算）场景下，建议在代码中定义ASCENDC_CUBE_ONLY宏，避免额外的性能开销。

```cpp
// 纯cube模式（只有矩阵计算）场景下，需要设置该代码宏，并且必须在#include "lib/matmul_intf.h"之前设置
// #define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
AscendC::Matmul<aType, bType, cType, biasType> mm;
```

`MatmulType` definition, verbatim:

```cpp
template <AscendC::TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false, LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false, TPosition SRCPOS = TPosition::GM> struct MatmulType {
    constexpr static AscendC::TPosition pos = POSITION;
    constexpr static CubeFormat format = FORMAT;
    using T = TYPE;
    constexpr static bool isTrans = ISTRANS;
    constexpr static LayoutMode layout = LAYOUT;
    constexpr static bool ibShare = IBSHARE;
    constexpr static TPosition srcPos = SRCPOS;
};
```

POSITION, for A3 specifically, verbatim:

> 针对 Atlas A3 训练系列产品 / Atlas A3 推理系列产品：
>
> A矩阵可设置为TPosition::GM，TPosition::VECOUT，TPosition::TSCM
>
> B矩阵可设置为TPosition::GM，TPosition::VECOUT，TPosition::TSCM
>
> Bias可设置为TPosition::GM，TPosition::VECOUT
>
> C矩阵可设置为TPosition::GM，TPosition::VECIN, TPosition::CO1
>
> 注意，C矩阵设置为TPosition::CO1时，C矩阵的数据排布格式仅支持CubeFormat::NZ，C矩阵的数据类型仅支持float、int32_t。

> 注意，A矩阵、B矩阵或Bias矩阵设置为TPosition::VECOUT或TPosition::TSCM时，对应矩阵用于单核计算的数据必须全部在Unified Buffer或L1 Buffer上

FORMAT for A3, verbatim:

> A矩阵可设置为CubeFormat::ND，CubeFormat::NZ, CubeFormat::VECTOR
>
> B矩阵可设置为CubeFormat::ND，CubeFormat::NZ
>
> Bias可设置为CubeFormat::ND
>
> C矩阵可设置为CubeFormat::ND，CubeFormat::NZ，CubeFormat::ND_ALIGN

TYPE for A3, verbatim — note `int4b_t` is supported for A and B:

> 针对 Atlas A3 训练系列产品 / Atlas A3 推理系列产品：
>
> A矩阵可设置为half、float、bfloat16_t 、int8_t、int4b_t
>
> B矩阵可设置为half、float、bfloat16_t 、int8_t、int4b_t
>
> Bias可设置为half、float、int32_t
>
> C矩阵可设置为half、float、bfloat16_t、int32_t、int8_t

> 注意：除fp8_e4m3fn_t/fp8_e5m2_t两种数据类型、B矩阵为int8_t数据类型外，A矩阵和B矩阵数据类型需要一致，具体数据类型组合关系请参考表2。A矩阵和B矩阵为int4b_t数据类型时，矩阵内轴的数据个数必须为偶数。例如，A矩阵为int4b_t数据类型且不转置时，singleCoreK必须是偶数。关于int4b_t数据类型的使用样例请参考Int4类型输入的Matmul算子样例。关于B8位宽数据类型的样例请参考B8位宽数据类型的Matmul算子样例。

表2 Matmul输入输出数据类型的支持列表 (relevant rows, verbatim):

> | A矩阵 | B矩阵 | Bias | C矩阵 | 支持平台 |
> |---|---|---|---|---|
> | int8_t | int8_t | int32_t | int32_t/half | Atlas 350 加速卡 / Atlas A3 训练系列产品 / Atlas A3 推理系列产品 / Atlas A2 训练系列产品 / Atlas A2 推理系列产品 / Atlas 推理系列产品 AI Core / Atlas 200I/500 A2 推理产品 |
> | int4b_t | int4b_t | int32_t | int32_t/half | Atlas A3 训练系列产品 / Atlas A3 推理系列产品 / Atlas A2 训练系列产品 / Atlas A2 推理系列产品 |
> | half | half | float | float | ... |
> | bfloat16_t | bfloat16_t | float | float | ... |

L1 budget formula for the transpose case, verbatim:

> 注意，由于L1 Buffer上的矩阵数据有分形对齐的约束，A、B矩阵转置和不转置时所需的L1空间可能不相同，在开启支持矩阵转置功能时，必须保证按照Matmul Tiling参数申请的L1空间不超过L1 Buffer的规格，判断方式为(depthA1*Ceil(baseM/c0Size)*baseK + depthB1*Ceil(baseN/c0Size)*baseK) * db * sizeof(dtype) < L1Size，db表示L1是否开启double buffer，取值1（不开启double buffer）或2（开启double buffer）

IBSHARE (IntraBlock Share) — L1 reuse across Matmul objects, verbatim:

> IBSHARE
>
> 是否使能IBShare（IntraBlock Share）。IBShare的功能是能够复用L1 Buffer上相同的A矩阵或B矩阵数据，复用的矩阵必须在L1 Buffer上全载。A矩阵和B矩阵仅有一个使能IBShare的场景，与IBShare模板配合使用，具体参数设置详见表2。
>
> 注意，A矩阵和B矩阵同时使能IBShare的场景，表示L1 Buffer上的A矩阵和B矩阵同时复用，需要满足：
>
> 同一算子中其它Matmul对象的A矩阵和B矩阵也必须同时使能IBShare；
>
> Atlas A3 训练系列产品 / Atlas A3 推理系列产品，获取矩阵计算结果时，只支持调用IterateAll接口，且只支持输出到GlobalTensor，即计算结果放置于Global Memory的地址。

The three calling conventions, verbatim:

> 初始化操作。
> ```cpp
> REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling); // 初始化matmul对象，参数含义请参考REGIST_MATMUL_OBJ章节
> ```
> 设置左矩阵A、右矩阵B、Bias。
> ```cpp
> mm.SetTensorA(gm_a);    // 设置左矩阵A
> mm.SetTensorB(gm_b);    // 设置右矩阵B
> mm.SetBias(gm_bias);    // 设置Bias
> // Atlas 推理系列产品AI Core上需要额外调用SetLocalWorkspace接口设置计算所需的UB空间
> mm.SetLocalWorkspace(usedUbBufLen);
> ```
> 完成矩阵乘操作。用户可以选择以下三种调用方式之一。
>
> 调用Iterate完成单次迭代计算，叠加while循环完成单核全量数据的计算。Iterate方式，可以自行控制迭代次数，完成所需数据量的计算，方式比较灵活。
> ```cpp
> // API接口内部会进行循环结束条件判断处理
> while (mm.Iterate()) {
>     mm.GetTensorC(gm_c);
> }
> ```
> 调用IterateAll完成单核上所有数据的计算。IterateAll方式，无需循环迭代，使用比较简单。
> ```cpp
> mm.IterateAll(gm_c);
> ```
> 用户申请用于存放矩阵乘结果的逻辑位置CO1内存，调用一次或多次Iterate完成单次或多次迭代计算，在需要搬出计算结果时，调用Fixpipe接口完成CO1上计算结果的搬运，然后释放申请的CO1内存。该方式下，用户可以灵活控制计算和搬运的节奏，根据实际需要，一次计算对应一次结果的搬出，或者将多次计算结果缓存在CO1内存中，再一次性搬出计算结果。
>
> 在此种调用方式下，创建Matmul对象时，必须定义C矩阵的内存逻辑位置为TPosition::CO1、数据排布格式为CubeFormat::NZ、数据类型为float或int32_t。
>
> Atlas 350 加速卡暂不支持该方式。Atlas 推理系列产品 AI Core暂不支持该方式。Atlas 200I/500 A2 推理产品暂不支持该方式。

The CO1-resident pattern, verbatim — this is the on-chip accumulation route:

```cpp
// 定义C矩阵的类型信息
typedef AscendC::MatmulType<AscendC::TPosition::CO1, CubeFormat::NZ, float> cType;
// 创建Matmul对象
AscendC::Matmul<aType, bType, cType, biasType> mm;
// 用户提前申请CO1的内存l0cTensor
TQue<TPosition::CO1, 1> CO1_;
// 128 * 1024为申请的CO1内存大小
GetTPipePtr()->InitBuffer(CO1_, 1, 128 * 1024);
// L0cT为C矩阵的数据类型。
// A矩阵数据类型是int8_t或int4b_t时，C矩阵的数据类型是int32_t。
// A矩阵数据类型是half、float或bfloat16_t时，C矩阵的数据类型是float。
LocalTensor<L0cT> l0cTensor = CO1_.template AllocTensor<L0cT>();
// 将l0cTensor作为入参传入Iterate，矩阵乘结果输出到用户申请的l0cTensor上
mm.Iterate(false, l0cTensor);
// 调用Fixpipe接口将CO1上的计算结果搬运到GM
FixpipeParamsV220 params;
params.nSize = nSize;
params.mSize = mSize;
params.srcStride = srcStride;
params.dstStride = dstStride;
CO1_.EnQue(l0cTensor);
CO1_.template DeQue<L0cT>();
Fixpipe<cType, L0cT, CFG_ROW_MAJOR>(gm[dstOffset], l0cTensor, params);
//释放CO1内存
CO1_.FreeTensor(l0cTensor);
```

```cpp
mm.End();
```

## Relevance
This is the concrete toolkit for a W4A8 cube stage on A3.
- `int4b_t x int4b_t -> int32_t` is a supported A3 Matmul combination, with the constraint that the inner-axis
  element count must be even (「A矩阵为int4b_t数据类型且不转置时，singleCoreK必须是偶数」). So W4 weights can be
  consumed natively by the cube unit rather than unpacked by a separate AIV kernel — removing a whole class of
  per-layer AIV launches. (`int8_t x int8_t -> int32_t` covers the A8 activation side.)
- `TPosition::CO1` + `CubeFormat::NZ` + `Fixpipe` is the documented way to **keep the matmul accumulator in L0C
  across multiple `Iterate` calls** and flush once: 「将多次计算结果缓存在CO1内存中，再一次性搬出计算结果」.
  This is the real "don't return to HBM" mechanism for cube->cube chaining, and the 128 KiB `InitBuffer(CO1_, 1, 128 * 1024)`
  matches the L0C size in section 8. It is unsupported on Atlas 350 / Atlas inference / 200I, but supported on A3.
- `TPosition::TSCM` lets A/B come from L1 rather than GM (data must be fully resident), and `IBSHARE` lets several
  Matmul objects in the same kernel share one L1-resident A or B — directly useful for a megakernel reusing a weight
  tile across stages. Caveat for A3: with IBShare on both A and B, only `IterateAll` to a `GlobalTensor` is supported.
- `ASCENDC_CUBE_ONLY` is documented as the way to 「避免额外的性能开销」 for pure-cube use, and the cost it avoids is
  exactly the MIX-mode AIV-side controller that section 5 describes. Our 505 MIX_AIC calls/round at 13.58 us are the
  price of *not* having set it. `SetLocalWorkspace(usedUbBufLen)` is only required on Atlas inference AI Core, not A3.

---

## 12. Matmul static tiling / `MatmulApiStaticTiling` — the measured Scalar-overhead win

### Best practice: Matmul Constant Operator Performance Optimization (with numbers)

https://www.hiascend.com/doc_center/source/en/canncommercial/800/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_10001.html
Also published at https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_10001.html
Publisher: Huawei / Ascend, CANN 8.0 operator development best practices. (Only the English rendering of this page
was reachable; the `zh` path returns AccessDenied.)

> ## Matmul Constant Operator Performance Optimization
>
> ### Case Study
>
> This case shows how to improve the operator performance by enabling Matmul static tiling when the Matmul high-level API is used for matrix multiplication computation. For details about Matmul static tiling, see Matmul Static Tiling.
>
> Matmul APIs involve a large number of Scalar computations during initialization and iteration. The Scalar computations during Matmul initialization affect the instruction head overhead, and the Scalar computations between Matmul iterations may block the MTE2 pipeline. The most variables involved in the Scalar computations during the two processes are the Tiling parameters singleCoreM/singleCoreN/singleCoreK and baseM/baseN/baseK. In the kernel implementation, if the variables obtained from the tiling parameters are replaced with constants and the related computations are moved forward to the compilation phase, the Scalar computation overhead can be reduced at run time.
>
> The operator specifications are as follows.
>
> | Input | Shape | Data type | Format |
> |---|---|---|---|
> | a | 32, 2048 | float16 | ND |
> | b | 2048, 32 | float16 | ND |
>
> For more examples of enabling Matmul static tiling, see matmul_api_constant.
>
> ### Obtaining Profile Data
>
> Use the msProf tool to obtain the profile data of the operator, focusing on the status of the Scalar pipeline.
>
> ### Analyzing Main Bottlenecks
>
> According to the preceding profile data, the ratio of Scalar time is high, implying that the performance bottleneck lies in the Scalar pipeline. Before performance optimization, the average time required for executing an operator for multiple times is 21.88 μs.
>
> ### Developing Optimization Solutions
>
> Enable Matmul static tiling. Specifically, when creating a Matmul object, use the constant template parameters. The procedure is as follows:
>
> When the GetMMConfig API is called to obtain the MatmulConfig template, set MatmulShapeParams to a constant value to obtain a customized MatmulConfig template with constant parameters.
>
> Call the GetMatmulApiTiling API to replace the tiling information with constants to obtain the constant template parameters, including the Matmul tiling information and MatmulConfig template.
>
> When creating a Matmul object, use the constant template parameters in step 2.
>
> Under these operator specifications, the optimal tiling obtained by the GetTiling API is achieved. The values of singleCoreM, singleCoreN, and singleCoreK are 32, 32, and 2048 respectively, and the values of baseM, baseN, and baseK are 32, 32, and 256 respectively. You only need to modify the part for creating the Matmul object in the original code. The details are as follows:

```cpp
constexpr static MatmulShapeParams shapeParams = {32, 32, 2048, 32, 32, 256};
constexpr static MatmulConfig MM_CFG = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);
constexpr MatmulApiStaticTiling static MM_CFG_CONSTANT = GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmConfig);
MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_CONSTANT> mm;
```

> ### Verifying Optimization Benefits
>
> After the performance optimization, the average time required for executing an operator for multiple times is 17.728 μs, and the Scalar computation time is reduced by 44.4%.
>
> ### Summary
>
> If the shape information (singleCoreM/singleCoreN/singleCoreK and baseM/baseN/baseK) for a single Matmul computation in a single core is determined, you can enable Matmul static tiling to reduce the Scalar computation overhead and improve operator performance.

### `GetMatmulApiTiling` API reference (CANN 9.0, 中文)

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0666.html
Parent topic 「Matmul Kernel侧接口」.

产品支持情况: `Atlas 350 加速卡 √ / Atlas A3 训练系列产品/Atlas A3 推理系列产品 √ / Atlas A2 训练系列产品/Atlas A2 推理系列产品 √ / Atlas 200I/500 A2 推理产品 x / Atlas 推理系列产品AI Core x / Atlas 推理系列产品Vector Core x / Atlas 训练系列产品 x`

功能说明, verbatim — note 部分常量化 vs 全量常量化:

> 本接口用于在编译期间获取常量化的Matmul Tiling参数。
>
> Matmul Tiling常量化功能为在编译期期间获取常量化的Matmul Tiling参数并进行算子编译，从而减少Scalar计算开销，提升算子整体性能。具体为，在获取Matmul模板时，可以确定MatmulConfig的singleCore Shape（MatmulConfig中的singleCoreM/singleCoreN/singleCoreK）和Base Shape（MatmulConfig中的basicM/basicN/basicK）参数，或者只确定Base Shape参数；通过指定获取模板的接口中的singleCore Shape和Base Shape参数，或者只指定Base Shape参数，获取自定义模板；然后调用本接口，得到常量化的Matmul Tiling参数。
>
> 当在调用获取MatmulConfig模板的接口时，只将(baseM, baseN, baseK)设置为常数值时，称为部分常量化，此时(singleCoreM, singleCoreN, singleCoreK)都保持默认值0，部分常量化场景在Kernel侧使用REGIST_MATMUL_OBJ初始化Matmul对象时，仍需要使用Tiling；将(baseM, baseN, baseK, singleCoreM, singleCoreN, singleCoreK)都设置为常数值时，称为全量常量化，这时可以在REGIST_MATMUL_OBJ的入参传递Tiling参数的位置，使用空指针替代。
>
> 经过上述部分常量化或全部常量化后，将得到带有常量化参数的MatmulConfig模板，然后使用本接口将Tiling参数常量化。本接口的返回值包含常量化的Matmul Tiling参数和MatmulConfig模板。

```cpp
template<class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ constexpr MatmulApiStaticTiling GetMatmulApiTiling(const MatmulConfig& mmCFG, int32_t l1Size = Impl::L1_SIZE)
```

> | 参数名 | 输入/输出 | 描述 |
> |---|---|---|
> | mmCFG | 输入 | 获取的MatmulConfig模板。<br>对于Atlas 350 加速卡，支持常量化的为全部模板：Norm, IBShare, MDL模板。<br>对于Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持常量化的模板有：Norm, MDL模板。<br>对于Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持常量化的模板有：Norm, MDL模板。 |
> | l1Size | 输入 | 可用的L1大小，默认值L1_SIZE。 |

约束说明, verbatim:

> 入参mmCFG，在调用获取MatmulConfig模板的接口获取时，需要使用常数值指定(baseM, baseN, baseK)或者指定(baseM, baseN, baseK, singleCoreM, singleCoreN, singleCoreK)，并且指定的参数值需要和Tiling计算的值保持一致。
>
> Batch Matmul场景支持全量常量化，但不支持使用空指针替代REGIST_MATMUL_OBJ的入参Tiling。

调用示例, verbatim:

```cpp
//定义Matmul对象
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
// 这里CFG使用GetNormalConfig接口获取，并指定已知的singleshape信息和baseM,baseN,baseK，指定的数值跟运行时tiling保持一致
constexpr auto staticTiling = GetMatmulApiTiling<aType, bType, cType, biasType>(CFG, 524288); // 该示例L1 Buffer可用大小为512KB
AscendC::Matmul<aType, bType, cType, biasType, staticTiling > mm;
```

## Relevance
This is the one place in the vendor docs with a before/after number on the same axis we are measuring.
A tiny matmul (32x2048 x 2048x32) spends 21.88 us per call, of which the dominant term is **Scalar pipeline**, and
constantising the tiling drops it to 17.728 us with Scalar time down 44.4%. The named cost is 「instruction head
overhead」 during Matmul init plus Scalar work between iterations blocking MTE2. 21.88 us is the same order as our
13.58 us MIX_AIC floor, which supports reading our floor as Scalar/init-dominated rather than data-movement-dominated.
Concretely for us: at batch=1 decode the shapes are fixed, so 全量常量化 applies — set all six of
`(baseM, baseN, baseK, singleCoreM, singleCoreN, singleCoreK)` as compile-time constants and then
`REGIST_MATMUL_OBJ` can be passed a **null pointer** in place of the tiling argument, eliminating the runtime tiling
read entirely. On A3 only the `Norm` and `MDL` templates can be constantised (not `IBShare`), and
`l1Size` defaults to `Impl::L1_SIZE`, which the example states as 524288 = 512 KB — corroborating the L1 figure in
section 8.

---

## 13. ACL graph capture: `aclmdlRICaptureBegin` / `aclmdlRICaptureEnd` / `aclmdlRIExecuteAsync`

### CANN 9.0 应用软件开发指南 — 单流捕获 (Single-Stream Capture)

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/acldevg/runtime_doc_dev_0030.html
English: https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/acldevg/runtime_doc_dev_0030.html
Publisher: Huawei / Ascend, CANN Community Edition 9.0.

基本限制, verbatim:

> 捕获任务到模型中、再执行模型的场景下，存在如下基本限制：
>
> 在进入捕获状态前，Stream上的任务依然是立即执行的。
>
> 在Stream上捕获任务时，只会将任务下沉到Device上，并不会立即执行，因此，对Stream或Event的查询或同步均为非法操作。同样，对Device或Context的查询或同步也是非法的，因为Device和Context中包含了Stream的同步信息。捕获过程中，对Stream、Event、Device、Context的同步或查询，在任何捕获模式下都是非法的。
>
> 在捕获过程中，在ACL_MODEL_RI_CAPTURE_MODE_GLOBAL模式（全局禁止，所有线程都不可以调用非安全函数）下，调用内存同步操作类函数（例如aclrtMemset、aclrtMemcpy、aclrtMemcpy2d）是非法的，会校验报错导致捕获失败。若业务侧确定这些函数的执行不会影响任务捕获，此时，可以通过调用aclmdlRICaptureThreadExchangeMode接口切换当前线程的捕获模式为ACL_MODEL_RI_CAPTURE_MODE_RELAXED，解除调用限制。
>
> 在捕获过程中，下发配置类的任务，例如Profiling配置、Dump配置、溢出检测配置等，可能会返回报错或者对捕获模型不生效。
>
> 若捕获的异步内存复制任务涉及Host内存，则只支持使用acl接口（例如aclrtMallocHost）申请Host锁页内存，否则在捕获过程中将返回报错。
>
> 另外，在捕获过程中，对默认Stream的操作是非法的。
>
> 最后还需要注意的是，任务被捕获后，需要使用者保证模型中任务使用资源的有效性，直至模型被销毁后才能销毁相关资源。

The example's capture block, verbatim:

```cpp
    aclmdlRI modelRI;
    aclrtStream stream;
    aclrtCreateStream(&stream);
    // ========开始捕获任务========
    aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
    // 异步拷贝，将算子self输入的数据从Host侧传到Device侧
    aclrtMemcpyAsync(self_d, size * sizeof(float), self_h, size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    // 切换捕获模式为RELAXED，允许调用aclrtMemcpy函数
    aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    aclmdlRICaptureThreadExchangeMode(&mode);
    // 同步拷贝，将算子other输入的数据从Host侧传到Device侧，仅执行一次
    aclrtMemcpy(other_d, size * sizeof(float), other_h, size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    // 将捕获模式切换回GLOBAL
    aclmdlRICaptureThreadExchangeMode(&mode);
    // 执行aclnnAdd算子
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    // 异步拷贝，将算子输出数据从Device侧传回Host侧
    aclrtMemcpyAsync(self_h, size * sizeof(float), out_d, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    // ========结束捕获任务========
    aclmdlRICaptureEnd(stream, &modelRI);
    // 打印模型信息，维测场景下使用
    const char *jsonPath = "./modelRI.json";
    aclmdlRIDebugJsonPrint(modelRI, jsonPath, 0);
    // 多次执行模型
    for (int i = 0; i < 8; i++) {
        aclmdlRIExecuteAsync(modelRI, stream);
        aclrtSynchronizeStream(stream);
```

Related API pages found in search (JS-gated viewer URLs; the doc_center raw equivalents follow the same path shape):
`aclmdlRICaptureBegin` / `aclmdlRICaptureEnd` —
https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_1849.html ,
https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_1782.html ;
「基于捕获模式构建模型运行实例」 (Building a Model Running Instance Based on the Capture Mode) —
https://www.hiascend.com/document/detail/en/canncommercial/850/appdevg/acldevg/aclcppdevg_000519.html ;
`aclrtLaunchKernelAttr` —
https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_1815.html .

The task-group/update APIs for mutating a captured graph (named in the same volume) are
`aclmdlRICaptureTaskGrpBegin` / `aclmdlRICaptureTaskGrpEnd` and
`aclmdlRICaptureTaskUpdateBegin` / `aclmdlRICaptureTaskUpdateEnd`.

## Relevance
`aclmdlRI` is exactly the handle `aclskOptimize(aclmdlRI model, aclskOptions *options)` in section 6 consumes, so the
full vendor pipeline is: `aclmdlRICaptureBegin` -> issue the 3008 launches -> `aclmdlRICaptureEnd` -> `aclskOptimize`
-> `aclmdlRIExecuteAsync` per decode round. That means the launch cost we measure per kernel call is eliminated at
*replay* time without rewriting any kernel, and SuperKernel then additionally fuses the captured binaries.
`aclmdlRIDebugJsonPrint(modelRI, jsonPath, 0)` dumps the captured task list to JSON — a direct way to enumerate our
3008 tasks and their kernel types before attempting fusion. The constraint that matters for a batch=1 decode loop is
that captured task parameters are immutable unless wrapped in `aclmdlRICaptureTaskGrpBegin/End` and updated inside
`aclmdlRICaptureTaskUpdateBegin/End`.

---

## 14. Kernel launch model: `ACLRT_LAUNCH_KERNEL`, `<<<>>>`, and what `blockDim` means in separated architecture

### CANN 8.0 算子开发指南 — Kernel Launch Operator Development / Overview

https://www.hiascend.com/doc_center/source/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0052.html
(TOC parent https://www.hiascend.com/doc_center/source/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0051.html
lists: `Overview / Kernel Launch / pyBind Calling`.)

> ### Kernel Launch Method
>
> In the current version, obtaining user workspace is not supported.
>
> Use ACLRT_LAUNCH_KERNEL to call the API:
> ```
> ACLRT_LAUNCH_KERNEL(kernel_name)(blockDim, stream, argument list);
> ```
> kernel_name: name of the operator kernel function.
>
> blockDim: number of cores on which the kernel function will be executed. A logical ID, that is, block_idx, is allocated to each core that executes the kernel function, and block_idx can be obtained by calling GetBlockIdx during the kernel function implementation.
>
> stream, aclrtStream type. Streams preserve the order of a stack of asynchronous operations being executed on the device.
>
> argument list: list of arguments, which must be the same as the argument list of the kernel function.

```cpp
//If blockDim is set to 8, the add_custom kernel function is called on eight cores. Each core independently executes the kernel function in parallel. The parameter lists of the kernel function are x, y, and z.
ACLRT_LAUNCH_KERNEL(add_custom)(8, stream, x, y, z)
```

> ### Kernel Launch Symbol
>
> The kernel function can use the kernel launch symbol <<<...>>> to specify the execution configuration of the kernel function.
> ```
> kernel_name<<<blockDim, l2ctrl, stream>>>(argument list);
> ```
> blockDim specifies the number of cores on which a kernel function will be executed. A logical ID, that is, block_idx, is allocated to each core that executes the kernel function, and block_idx can be obtained by calling GetBlockIdx during the kernel function implementation.
>
> blockDim is a concept about logical cores, and its value range is [1, 65535]. To fully utilize hardware resources, set this parameter to the number of physical cores or a multiple of the number of physical cores. For the coupled architecture and separated architecture, the meaning and setting rules of blockDim during running are different. The details are as follows:
>
> Coupled architecture: Because the Vector and Cube Units are integrated, blockDim is used to start multiple AI Core instances, without differentiating Vector Units and Cube Units. The number of AI Cores can be obtained by calling GetCoreNumAiv or GetCoreNumAic.
>
> Separated architecture
>
> For operators that contain only Vector Units, blockDim is used to set the number of vector (AIV) instances to be started. For example, if an AI processor has 40 Vector cores, set blockDim to 40.
>
> For operators that contain only Cube Units, blockDim is used to set the number of cube (AIC) instances to be started. For example, if an AI processor has 20 Cube cores, set blockDim to 20.
>
> Operators for Vector/Cube fusion computing are started by groups of AIVs and AICs. blockDim is used to set the number of groups to be started. For example, if an AI processor has 40 Vector cores and 20 Cube cores, a group consists of two Vector cores and one Cube core. Set the number of groups to 20. In this case, 20 groups are started, including 40 Vector cores and 20 Cube cores. Note: In this scenario, the number of blockDim (logical cores) cannot exceed the number of physical cores (a physical core contains two Vector cores and one Cube core).
>
> The number of AIC and AIV cores can be obtained by calling GetCoreNumAic and GetCoreNumAiv, respectively.
>
> l2ctrl is a reserved parameter and can be ignored. This parameter is fixed at nullptr.

```cpp
//If blockDim is set to 8, the add_custom kernel function is called on eight cores. Each core independently executes the kernel function in parallel. The parameter lists of the kernel function are x, y, and z.
add_custom<<<8, nullptr, stream>>>(x, y, z);
```

> The kernel function is called asynchronously. After the kernel function is called, the control right is returned to the host immediately. You can call aclrtSynchronizeStream to force the host program to wait until all kernel functions are executed.

Compiler architecture flag, from `cann/asc-devkit` `examples/README.md`
(https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/README.md):

> | 产品型号 | npu-arch参数 |
> | ---- | ---- |
> | Ascend 950PR&950DT系列产品 | --npu-arch=dav-3510 |
> | Atlas A3系列产品<br>Atlas A2系列产品 | --npu-arch=dav-2201 |
> | Atlas推理系列产品AI Core | --npu-arch=dav-2002 |

## Relevance
`blockDim` for a MIX kernel counts **groups**, not cores: one group = 1 AIC + 2 AIV, and 「blockDim (logical cores)
cannot exceed the number of physical cores」. A MIX launch therefore boots 3x the core count of the blockDim value.
The vendor's own worked example — 40 Vector cores and 20 Cube cores, blockDim 20 — is the AIC:AIV = 1:2 ratio for a
910B-class die. Our A3 build must use `--npu-arch=dav-2201`.

---

## 15. CV fusion best practice with full A2/A3 profiling: `quant_group_matmul_high_performance`

This is the closest published analogue to our workload: per-token-quantized int8 grouped matmul with dequant +
activation + cast fused into one `__mix__` kernel, with three measured optimization stages on A2/A3 and 950PR.

https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/README.md
raw: https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/quant_group_matmul_high_performance/README.md
Repo `cann/asc-devkit`, branch `master`.

> ## 概述
>
> 本样例实现了per-token量化的分组矩阵乘法（QuantGroupMatmul），展示CV融合时Vector bound场景下的性能调优方法。
>
> ## 本样例支持的产品及CANN软件版本
>
> | 产品 | CANN软件版本 |
> |------|-------------|
> | Ascend 950PR&950DT系列产品 | >= CANN 9.2.0 |
> | Atlas A3系列产品 | >= CANN 9.2.0 |
> | Atlas A2系列产品 | >= CANN 9.2.0 |

Operator spec, verbatim: `x [8192, 1024] int8 ND`, `weight [8, 1024, 8192] int8 NZ`, `group [8] int64 ND`,
`scale [8, 8192] float ND`, `perTokenScale [8192] float ND`, output `y [8192, 8192] float16 ND`,
kernel name `quant_group_matmul_custom`, invoked via `<<<>>>`.

Scenario matrix, verbatim:

> | 场景 | SCENARIO_NUM | __mix__模式 | PIPELINE_DEPTH | 说明 |
> |------|-------------|------------|----------------|------|
> | Scenario 0 | 0 | `__mix__(1, 1)` | 1 | 基准：单AIV，无流水重叠，Cube与Vector串行 |
> | Scenario 1 | 1 | `__mix__(1, 2)` | 1 | 双AIV，无流水重叠，多AIV分摊Vector任务 |
> | Scenario 2 | 2 | `__mix__(1, 2)` | 4 | 双AIV + 4级流水，Cube-Vector细粒度流水重叠 |

```cpp
constexpr uint32_t scenarioNum = SCENARIO_NUM;
// ...
constexpr uint32_t PIPELINE_DEPTH = (scenarioNum >= 2) ? 4 : 1;

template <...>
#if SCENARIO_NUM == 0
__global__ __mix__(1, 1)
#else
__global__ __mix__(1, 2)
#endif
void quant_group_matmul_custom(...)
```

Data-flow path, verbatim — note the cube result goes out to **GM workspace** and back in through MTE2:

```text
存储位置                        数据类型           流水

GM (x, weight)                   int8
  |                                               MTE2
L1                               int8
  |                                               MTE1
L0A / L0B                        int8
  |  Iterate                                     MMAD
L0C                              int32
  |                                               FIXPIPE
GM (workspace)                   int32
  |  DataCopyPad2D                                MTE2
UB vecInQueue                    int32
  |  AscendDequant (×scale)   int32 → float       VECTOR
UB dequantMiddleResult            float
  |  Mul (×perTokenScale)     float → float       VECTOR
UB mulsResultLocal                float
  |  FasterGelu               float → float       VECTOR
UB actResultLocal                 float
  |  Cast                     float → half        VECTOR
UB vecOutQueue                    half
  |  DataCopyPad2D                                MTE3
GM (y)                            half

注：scale[float] 和 perTokenScale[float] 通过 MTE2 独立搬入UB，
    分别参与 AscendDequant 和 Mul 步骤。
```

The AIC/AIV handshake, verbatim:

> 本样例采用`__mix__`核函数模式，AIC（Cube核）和AIV（Vector核）并行运行在同一个物理核上。AIC执行矩阵乘法，AIV执行反量化+激活+类型转换，两者通过`CrossCoreSetFlag`/`CrossCoreWaitFlag`实现同步：

```cpp
// AIC执行路径: MMCompute → Iterate循环 → SetFlag(SYNC_AIC_TO_AIV)
// AIV执行路径: 跳过MMCompute → VectorCompute → WaitFlag(SYNC_AIC_TO_AIV)
if ASCEND_IS_AIC {
    matmulObj.Iterate();    // Cube计算
    CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);  // 通知Vector
}
if ASCEND_IS_AIV {
    CrossCoreWaitFlag<2>(SYNC_AIC_TO_AIV);  // 等待Cube完成
    // 处理数据...
    CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);
}
```

> Case 0中PIPELINE_DEPTH=1，workspace仅1个Slot（即 `CORE_NUM×BASE_M×BASE_N×sizeof(int32_t)` 大小的一段连续空间），Cube每次Iterate后必须等Vector处理完才能复用该Slot，Cube与Vector串行执行。

Tiling constantisation and UnitFlag, verbatim:

> **③ Tiling常量化**
>
> 所有Tiling参数通过`MatmulApiStaticTiling`在编译期确定，运行时无需Scalar动态计算，减少Scalar开销：
>
> ```cpp
> constexpr static auto CONSTANT_CFG = GetCustomConstantCFG();
> matmul::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONSTANT_CFG> matmulObj;
> ```
>
> **④ Matmul UnitFlag优化**
>
> 本样例通过`mmCFG.enUnitFlag = true`开启Matmul的UnitFlag优化，实现Cube计算与搬运的512B细粒度流水同步：
>
> ```cpp
> static constexpr MatmulShapeParams shapeParams = {SINGLE_M, SINGLE_N, K, BASE_M, BASE_N, BASE_K};
> MatmulConfig mmCFG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shapeParams);
> mmCFG.enUnitFlag = true;  // 开启UnitFlag
> auto constantCFG = AscendC::GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG);
> ```
>
> 默认情况下，Cube的MMAD计算和FixPipe搬运是粗粒度同步的——必须等待整个`[BASE_M, BASE_N]`块的计算完成后，FixPipe才能开始搬出数据。开启UnitFlag后，MMAD和FixPipe实现细粒度（512B）流水并行：每当Cube完成一个512B数据结果的计算，FixPipe立即搬出该数据，Cube计算与结果写回流水重叠。

Double buffer sizing, verbatim — with the per-arch BASE_M:

```cpp
constexpr uint32_t BUFFER_NUM = 2; // UB队列Buffer数, 设为2开启double buffer

pipe->InitBuffer(scaleInQueue, BUFFER_NUM, BASE_N * sizeof(float));          // 2 × 1024B
pipe->InitBuffer(perTokenScaleInQueue, BUFFER_NUM, BASE_M * sizeof(float));   // DAV_2201(BASE_M=128): 2 × 512B；DAV_3510(BASE_M=256): 2 × 1024B
pipe->InitBuffer(vecInQueue, BUFFER_NUM, UB_CAL_SIZE * sizeof(int32_t));      // 2 × 24,576B
pipe->InitBuffer(vecOutQueue, BUFFER_NUM, UB_CAL_SIZE * sizeof(half));        // 2 × 12,288B
```

The dual-AIV task split, verbatim:

> `__mix__(1, 2)`模式下，每个物理核包含1个AIC和2个AIV。两个AIV通过`taskRation`轮询机制分摊Vector侧的offsetM迭代任务：
>
> ```cpp
> uint32_t taskRation = AscendC::GetTaskRation();  // __mix__(1,2)下为2
> if (taskRation != 0 && vecCount % taskRation != subBlockIdx) {
>     continue;  // 跳过不属于当前AIV的任务
> }
> ```
>
> 两个AIV共享同一个coreIdx，通过`subBlockIdx`（由`GetSubBlockIdx()`获取）区分身份。

The ring-slot workspace pipeline, verbatim:

> workspace是Cube和Vector之间的中转站，采用环形分槽设计。这里的 Slot 指由 `workSpaceOffset = BASE_N × BASE_M × (coreIdx + (cubeTaskIdx % PIPELINE_DEPTH) × CORE_NUM)` 定位的一段连续空间，单个 core 的 Slot 大小为 `BASE_M×BASE_N×sizeof(int32_t)`，每层 pipeline 的 Slot 总空间为 `CORE_NUM×BASE_M×BASE_N×sizeof(int32_t)`，`PIPELINE_DEPTH` 个 Slot 轮流使用，Cube 写入当前 Slot 的同时，Vector 可以处理之前 Slot 的数据
>
> - 当`cubeTaskIdx < PIPELINE_DEPTH`时，Cube无需等待，直接写入新Slot
> - 当`cubeTaskIdx >= PIPELINE_DEPTH`时，Cube需等待Vector释放对应Slot（`CrossCoreWaitFlag<2>(SYNC_AIV_TO_AIC)`）
> - 同步粒度为Iterate（=BASE_M×BASE_N），而非子块（=SINGLE_M×SINGLE_N）

```
时间 ──────────────────────────────────────────────────────────────>
Cube:  |─Iter0→Slot0─|─Iter1→Slot1─|─Iter2→Slot2─|─Iter3→Slot3─|─Iter4→Slot0─| ...
                      (Wait)                                     (Wait)
Vec:                 |─Slot0处理─|─Slot1处理─|─Slot2处理─|─Slot3处理─| ...
```

### Measured numbers — Atlas A2 training series (DAV_2201, same arch as our A3), verbatim

> | Scenario | Task Type | Task Duration(μs) | Block Num | Mix Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
> |------|-----------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
> | Scenario 0 | MIX_AIC | 583.512 | 24 | 24 | 564.06 | 192.02 | 0.34 | 115.317 | 0.204 | 168.645 | 0.299 | 141.648 | 0.251 | 79.47 | 0.141 | 565.12 | 471.71 | 0.835 | 228.275 | 0.404 | 139.25 | 0.246 | 94.682 | 0.168 |
> | Scenario 1 | MIX_AIC | 353.887 | 24 | 48 | 344.01 | 192.612 | 0.56 | 143.518 | 0.417 | 170.031 | 0.494 | 160.396 | 0.466 | 107.662 | 0.313 | 345.22 | 235.871 | 0.683 | 122.424 | 0.355 | 107.715 | 0.312 | 56.494 | 0.164 |
> | Scenario 2 | MIX_AIC | 299.286 | 24 | 48 | 288.65 | 193.576 | 0.671 | 196.668 | 0.681 | 173.003 | 0.599 | 170.193 | 0.59 | 167.936 | 0.582 | 285.21 | 235.872 | 0.827 | 126.227 | 0.443 | 99.321 | 0.348 | 56.645 | 0.199 |

### Measured numbers — Ascend 950PR (DAV_3510), verbatim

> | Scenario | Task Type | Task Duration(μs) | Block Num | Mix Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | ... | aiv_vec_time(μs) | aiv_vec_ratio | ... |
> |---|---|---|---|---|---|---|---|---|---|---|---|
> | Scenario 0 | MIX_AIC | 282.717 | 32 | 32 | 281.84 | 167.952 | 0.596 | | 229.166 | 0.813 | |
> | Scenario 1 | MIX_AIC | 224.125 | 32 | 64 | 223.27 | 168.201 | 0.753 | | 112.441 | 0.504 | |
> | Scenario 2 | MIX_AIC | 204.129 | 32 | 64 | 203.26 | 180.347 | 0.887 | | 112.218 | 0.552 | |

性能分析总结, verbatim:

> 本样例固定shape为`M=8192, N=8192, K=1024`，分组只改变M轴调度方式，总Cube计算量仍为：
>
> Ops_cube = sum_{i=0}^{G-1} group_i × N × K = M × N × K
>
> 对于int8矩阵乘，Cube单cycle按`16 × 32 × 16`次乘加估算，理论Cube计算耗时为：
>
> T_cube = (M × N × K) / (16 × 32 × 16 × f × CORE_NUM)
>
> | 芯片 | CORE_NUM | 理论Cube耗时(μs) | aic_mac_time(μs) | aic_mac_ratio | aiv_vec_time(μs) | 达到理论峰值比例 | 瓶颈判断 |
> |------|----------|------------------|------------------|---------------|------------------|----------------|----------|
> | Atlas A2/A3（DAV_2201） | 24 | 188.93 | 193.576 | 0.671 | 235.872 | 97.60% | 仍偏Vector侧 |
> | Ascend 950PR（DAV_3510） | 32 | 158.88 | 180.347 | 0.887 | 112.218 | 88.10% | 不再是Vector侧 |
>
> Atlas A2/A3上表现不同：Scenario 2中`aiv_vec_time`为235.872μs，高于`aic_mac_time`的193.576μs，`aiv_vec_ratio`也达到0.827，说明Vector侧的反量化、per-token缩放、FasterGelu和Cast链路仍然占据较大比例，整体仍偏Vector侧瓶颈。与此同时，A2实测Cube计算耗时193.576μs已经接近理论Cube耗时188.93μs，达到理论峰值约97.60%，说明Cube矩阵乘本身已经较充分利用，继续优化端到端性能需要优先关注Vector侧计算链路和Cube-Vector同步开销。
>
> 虽然Atlas A2/A3最终仍偏Vector bound，但三级优化仍然有效：Scenario 1通过双AIV并行将Vector计算耗时从471.710μs降低到235.871μs，端到端耗时从583.512μs降低到353.887μs；Scenario 2在Vector计算耗时基本不变的情况下，通过4级workspace流水提升Cube和Vector的重叠度，将端到端耗时进一步降低到299.286μs，相比Scenario 0总耗时减少48.71%。

Profiling metric definitions, verbatim (useful for reading our own traces):

> | 指标 | 说明 |
> |---|---|
> | Task Duration(μs) | 整个任务执行的总时间，样例执行时间以该参数为准 |
> | Block Num | 使用的核数（Block 数量） |
> | aicore_time(μs) | AI Core 的平均执行时间 |
> | aic_mac_time(μs) | cube 类型指令（矩阵类运算指令）耗时 |
> | aic_mte1_time(μs) | mte1 类型指令（L1 -> L0A/L0B 搬运）耗时 |
> | aic_mte2_time(μs) | mte2 类型指令（GM -> L1 搬运）耗时 |
> | aic_fixpipe_time(μs) | fixpipe 类型指令（L0C -> GM/UB 搬运）耗时 |
> | aiv_time(μs) | Task 在 AI Vector Core 上的执行时间 |
> | aiv_vec_time(μs) | vec 类型指令（向量类运算指令）耗时 |
> | aiv_scalar_time(μs) | scalar 类型指令（标量类运算指令）耗时 |
> | aiv_mte2_time(μs) | mte2 类型指令（GM -> UB 搬运类指令）耗时 |
> | aiv_mte3_time(μs) | mte3 类型指令（UB -> GM 搬运类指令）耗时 |

### Sibling example that does the same for an activation chain

`matmul_gelu_high_performance` (same directory) —
https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/matmul_gelu_high_performance/README.md
Listed in the parent README as: 「[03_fusion_compute](./03_fusion_compute) | 融合计算相关的性能优化样例，介绍QuantGroupMatmul、Matmul+GELU等Cube-Vector融合算子的高性能实现。」
(Fetch of this README failed with a proxy 504 at the time of writing; see Gaps.)

### Sibling example: SSBuffer AIV/AIC message passing — **950PR/DT only, not A3**

https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/ssbuf_aiv_aic_comm/README.md

> # SSBuffer AIV/AIC通信样例
>
> 本样例演示基础API场景下如何在同一个`__mix__(1, 2)`任务中使用SSBuffer实现AIV到AIC的消息通信。任务包含1个AIC和2个AIV，两个AIV分别通过独立的环形消息队列发送消息，AIC接收并检查消息，最后将结果写入GM，由Host侧逐条校验。
>
> | 产品 | CANN软件版本 |
> |------|-------------|
> | Ascend 950PR&950DT系列产品 | >= CANN 9.2.0 |
>
> 本样例固定使用`dav-3510`架构，仅支持NPU和仿真运行模式。
>
> ### SSBuffer空间布局
>
> - SSBuffer总容量为3KB，本样例使用前2KB。
> - AIV0使用`[0, 1024)`，包含8条128B消息。
> - AIV1使用`[1024, 2048)`，包含8条128B消息。
> - `[2048, 3072)`在本样例中不使用。
> - 两条消息队列互不重叠，每个AIV只写自己的消息队列。
>
> ### 1. 获取SSBuffer工作区
>
> 基础API通过`AscendC::GetSsbufBaseAddr()`获取SSBuffer基地址。返回值属于`__ssbuf__`地址空间，转换后的指针和后续消息指针都必须保留该地址空间限定，不能当作普通GM地址使用。
>
> ```cpp
> __aicore__ inline __ssbuf__ SsbufWorkspace* GetWorkspace()
> {
>     return reinterpret_cast<__ssbuf__ SsbufWorkspace*>(AscendC::GetSsbufBaseAddr());
> }
> ```
>
> ### 2. AIC初始化消息队列
>
> SSBuffer初始内容不保证为0，`AscendC::InitSocState()`也不会清空SSBuffer。AIC启动后遍历两个AIV的消息队列，清除全部消息头，再通过模式4核间同步分别通知AIV0和AIV1开始工作。
>
> ```cpp
> InitQueues(workspace);
> AscendC::CrossCoreSetFlag<4, PIPE_S>(GetAicFlagId(0, START_FLAG_ID));
> AscendC::CrossCoreSetFlag<4, PIPE_S>(GetAicFlagId(1, START_FLAG_ID));
> ```
>
> 模式4用于同一AI Core内AIC与单个AIV之间的同步。AIV0和AIV1都使用本地标记ID `0~15`，AIC侧使用`0~15`对应AIV0，使用`16~31`对应AIV1。`GetAicFlagId`按照`sender * 16 + aivFlagId`计算AIC侧标记ID。
>
> ## 使用约束
>
> - AIC和两个AIV必须在同一个`__mix__(1, 2)`任务中访问SSBuffer。
> - 自定义消息布局不能超过3KB，也不能与其他SSBuffer协议重叠。
> - 消息字段的写入顺序、`valid`位和消息头清理顺序不能随意调整。
> - 队列深度需要保持为2的幂，才能使用位与方式计算队列位置。
> - 跨核共享字段需要使用`volatile`，消息发布和空间复用需要使用配对的模式4核间同步通知。
> - AIV侧标记ID范围为`0~15`；AIC侧通过`0~15`和`16~31`分别与AIV0和AIV1配对。

## Relevance
This is the most directly transferable evidence we found.
- `CORE_NUM = 24` on DAV_2201 (A2/A3), 32 on DAV_3510. With `__mix__(1, 2)`, Block Num stays 24 but
  **Mix Block Num becomes 48** — the profiler column that reveals the core multiplier a MIX launch pays.
- The exact handshake we would write in a megakernel is published:
  `CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV)` on the cube side, `CrossCoreWaitFlag<2>(SYNC_AIC_TO_AIV)` then
  `CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC)` on the vector side — mode 2, `PIPE_FIX` for the cube-complete
  signal and `PIPE_MTE2` for the slot-release signal.
- It confirms section 8's conclusion the hard way: even inside a single fused kernel, the cube result travels
  `L0C -> FIXPIPE -> GM (workspace) -> MTE2 -> UB`. There is no UB-resident cube->vector path on this architecture.
  The mitigation is a **ring of PIPELINE_DEPTH slots in GM workspace**, sized
  `CORE_NUM × BASE_M × BASE_N × sizeof(int32_t)` per level, with `workSpaceOffset = BASE_N × BASE_M × (coreIdx + (cubeTaskIdx % PIPELINE_DEPTH) × CORE_NUM)`.
  Depth 4 bought 8.92% over depth 1 on 950PR and 15.4% on A2.
- `mmCFG.enUnitFlag = true` gives 512-byte-granularity MMAD/FixPipe overlap instead of waiting for a whole
  `[BASE_M, BASE_N]` tile — a free lever we should check we have on.
- `AscendC::GetTaskRation()` returns 2 under `__mix__(1,2)`, and `GetSubBlockIdx()` identifies which AIV you are:
  the documented way to split vector work between the two AIVs of one group.
- On A2/A3 the workload stayed **Vector bound** (`aiv_vec_time` 235.872 us > `aic_mac_time` 193.576 us,
  `aiv_vec_ratio` 0.827) while cube hit 97.60% of theoretical peak. For our W4A8C8 decode the dequant/activation
  chain is the same shape, which argues the AIV side (not the cube side) is where a megakernel must economise on A3.
- `BASE_M = 128` on DAV_2201 vs 256 on DAV_3510 is the published per-arch base-tile difference.
- SSBuffer (`__ssbuf__`, `AscendC::GetSsbufBaseAddr()`, 3 KB, mode-4 cross-core sync, flag IDs 0-15 / 16-31) is a
  true on-chip AIV<->AIC mailbox — but it is **dav-3510 only**, so it is not available to us on A3. On A3 the
  mailbox must be GM workspace plus mode-2 `CrossCoreSetFlag`.

---

## 16. `matmul_gelu_high_performance` — the A3 cube->vector data path, proven, with L2 as the real fusion buffer

https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/matmul_gelu_high_performance/README.md
raw: https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/matmul_gelu_high_performance/README.md
Repo `cann/asc-devkit`, branch `master`. Kernel file `mmad_gelu.asc`, kernel name `mmad_vector_custom`.

> ## 概述
>
> 本样例基于静态Tensor编程范式，展示了Cube-Vector（CV）融合的实现方法，将Matmul矩阵乘法与GELU激活函数融合到同一AI Core中并行执行。AIC侧完成Matmul计算后通过Fixpipe将结果输出到GM或UB，AIV侧从GM或UB读取数据完成GELU向量计算。
>
> | 产品 | CANN软件版本 |
> | ------------------------------------------- | ------------- |
> | Ascend 950PR&950DT系列产品 | >= CANN 9.2.0 |
> | Atlas A3系列产品 | >= CANN 9.2.0 |
> | Atlas A2系列产品 | >= CANN 9.2.0 |

The two data paths, verbatim — **this is the definitive statement that A3 cannot do L0C -> UB**:

> 本样例提供两种CV融合场景：
>
> **Scenario 1：CV融合-GM中转（中级性能优化）**（A2/A3/Ascend 950PR均支持）
>
> ```
> GM ──(MTE2)──> L1 ──(MTE1)──> L0A/L0B ──(Cube)──> L0C ──(Fixpipe)──> GM ──(MTE2)──> UB ──(Gelu)──> UB ──(MTE3)──> GM
> ```
>
> **Scenario 2：CV融合-UB直通（高阶极限打磨）**（仅Ascend 950PR支持）
>
> Ascend 950PR新增L0C->UB的通路，A2/A3不支持，芯片能力增强，因此在Ascend 950PR架构中可以不经过GM直接将矩阵乘结果输出到UB进行GELU计算
>
> ```
> GM ──(MTE2)──> L1 ──(MTE1)──> L0A/L0B ──(Cube)──> L0C ──(Fixpipe)──> UB ──(Gelu)──> UB ──(MTE3)──> GM
> ```

Kernel entry and AIC/AIV split, verbatim:

> 本样例使用 `__global__ __mix__(1, 2)` 声明核函数，表示每个逻辑核由1个AIC和2个AIV组成。AIC侧负责Matmul计算，AIV侧负责GELU向量计算，两者在同一AI Core内按(baseM × baseN)块粒度流水并行。

```cpp
__global__ __mix__(1, 2) void mmad_vector_custom(__gm__ uint8_t* xMatrix, __gm__ uint8_t* yMatrix, __gm__ uint8_t* zMatrix)
{
    AscendC::InitSocState();
    KernelMmadVector<...> op;
    op.Init(xMatrix, yMatrix, zMatrix);
    op.Process();
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

```cpp
if ASCEND_IS_AIC {
    // AIC侧：Buffer分配 + 计算循环 + Fixpipe搬出
    ProcessLoopAic(...);
}
if ASCEND_IS_AIV {
    // AIV侧：GELU向量计算 + 搬出至GM
    ProcessLoopAiv();
}
```

Cross-core notification, verbatim:

> AIC侧每完成一个 (baseM × baseN)大小子块的Fixpipe输出后，通过 `CrossCoreSetFlag` 通知AIV侧数据已就绪；AIV侧通过 `CrossCoreWaitFlag` 等待该信号，然后开始GELU计算：

```cpp
// AIC侧 CopyOutAic():
AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);  // 通知AIV数据就绪

// AIV侧 GeluFromGM() / GeluFromUB():
AscendC::CrossCoreWaitFlag<0x2>(0x8);  // 等待AIC侧Fixpipe完成
```

BlockIdx semantics under `__mix__(1,2)`, verbatim:

> `__mix__(1,2)` 模式下，`GetBlockIdx()` 在AIC侧和AIV侧各自独立编号：AIC侧返回0~numAICores-1，AIV侧返回0~numAIVCores-1。由于每个逻辑核对应2个AIV，AIV侧需要将 `GetBlockIdx() / 2` 映射回逻辑核ID，从而与AIC侧使用相同的多核切分策略：

```cpp
uint32_t logicCoreId;
if ASCEND_IS_AIC {
    logicCoreId = AscendC::GetBlockIdx();
} else {
    logicCoreId = AscendC::GetBlockIdx() / 2;  // __mix__(1,2): 每个逻辑核对应2个AIV
}
uint32_t mIterIdx = logicCoreId % mIter;
uint32_t nIterIdx = logicCoreId / mIter;
```

> `__mix__(1,2)` 模式下每个逻辑核有2个AIV，Fixpipe输出到GM的M行数据按M/2行拆分给两个AIV处理。每个AIV通过 `GetSubBlockIdx() % 2` 获取自己的子块索引，独立完成各自M/2行的GELU计算

> 本样例的Matmul部分采用与Matmul基础API最佳实践样例相同的优化手段（L1/L0双缓冲、大包搬运、细粒度流水同步、UnitFlag等），GELU部分在Ascend 950PR架构上采用RegBase方式，在A2/A3架构上采用MemBase方式。

### Measured numbers (msOpProf, median of 5; M=K=N=8192, fp16 in, fp32 out)

> 以下数据基于msOpProf工具测量，取5次中位数。矩阵规格M=8192, K=8192, N=8192，输入数据类型float16，输出数据类型float32。
>
> ##### Atlas A2训练系列芯片
>
> - singleCoreM=2048, singleCoreN=1536，24核
>
> | Scenario | Task Type | Task Duration(μs) | Block Num | Mix Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_mte1_time(μs) | aic_mte2_time(μs) | aic_fixpipe_time(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_mte2_time(μs) | aiv_mte3_time(μs) |
> |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
> | Gelu计算 | AI_VECTOR_CORE | 369.827 | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 358.91 | 236.506 | 0.659 | 8.293 | 316.672 | 150.033 |
> | Matmul计算 | AI_CORE | 4227.605 | 24 | 0 | 3760.55 | 3083.197 | 0.82 | 338.636 | 2516.081 | 3685.713 | 193.704 | 0 | 0 | 0 | 0 | 0 | 0 |
> | Scenario 1 | MIX_AIC | 4236.944 | 24 | 48 | 3770.16 | 3083.235 | 0.818 | 531.971 | 2520.852 | 3687.839 | 193.933 | 3773.6 | 174.327 | 0.046 | 2.016 | 90.315 | 66.373 |
>
> ##### Ascend 950PR芯片
>
> - singleCoreM=2048, singleCoreN=1024，32核
>
> | Scenario | Task Type | Task Duration(μs) | Block Num | Mix Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aiv_vec_time(μs) | aiv_mte2_time(μs) | aiv_mte3_time(μs) |
> |---|---|---|---|---|---|---|---|---|---|---|
> | Gelu计算(RegBase) | AI_VECTOR_CORE | 348.868 | 64 | 0 | 0 | 0 | 0 | 66.277 | 320.543 | 314.547 |
> | Matmul计算 | AI_CORE | 2601.311 | 32 | 0 | 2600.55 | 2593.264 | 0.997 | 0 | 0 | 0 |
> | Scenario 1 | MIX_AIC | 2606.91 | 32 | 64 | 2605.9 | 2597.856 | 0.997 | 59.897 | 57.145 | 50.516 |

性能数据解读, verbatim:

> 以Ascend 950PR芯片数据为例：
>
> - 单独Gelu任务耗时348.868μs，单独Matmul任务耗时2601.311μs
> - Scenario 1融合后总耗时2606.91μs，与Matmul单独执行时间基本一致
>
> 这表明采用CV融合方式编写时，AIV的Vector计算时间基本被AIC的Cube计算时间完全覆盖。由于Cube和Vector可以并行执行，整体任务执行时间不是简单的C+V叠加，而是基本等于Cube计算时间（即Matmul时间），实现了高效的流水并行。

The L2 insight, verbatim — **this is the mechanism that makes GM-transit fusion acceptable on A3**:

> **MTE2时间显著减少的原因：**
>
> 对比数据可以发现：
>
> - 单独Gelu计算的aiv_mte2_time为320.543μs
> - Scenario 1中aiv_mte2_time仅为57.145μs，减少约82.2%
>
> 原因分析：虽然Scenario 1中AIV也需要从GM将数据读取到UB，但AIC的Fixpipe将Matmul计算结果写入GM后，数据会被缓存在L2 Cache中。当AIV发起MTE2操作从GM读取数据时，由于数据已在L2 Cache中命中，实际是从L2 Cache读取而非从GM读取，因此MTE2时间显著减少。这也是CV融合场景下的一个隐式性能优化点。

> Fixpipe L0C → UB（通过 `dualDstCtl=0b01` 开启双目标模式，按M维度拆分给2个AIV）

> - Scenario 2融合后总耗时2573.584μs，相比Scenario 1的2606.91μs进一步减少约33μs
> - Scenario 2中aiv_mte2_time仅为0.005μs，相比Scenario 1的57.145μs几乎降为零

Profiling command, verbatim:

> 流水图基于msOpProf工具上板获取，获取命令：`msopprof --aic-metrics=PipeTimeline ./demo`

> ```
>       ├──MemoryL0.csv               # L0A，L0B，和L0C读写带宽速率
> ```

## Relevance
Settles the central architectural question for our megakernel. On A3 there is **no L0C -> UB path** — that is a
950PR addition ("Ascend 950PR新增L0C->UB的通路，A2/A3不支持"), reached via `dualDstCtl=0b01`. The only fusion path
available to us is `L0C -> Fixpipe -> GM -> MTE2 -> UB`. But the measured cost of that round trip is small, because
the Fixpipe write lands in L2 and the AIV's MTE2 hits L2 rather than HBM: `aiv_mte2_time` fell from 320.543 us
standalone to 57.145 us fused, **-82.2%**. So "don't return to HBM" is achievable as "don't miss L2", and
tile sizes must be chosen so the `BASE_M x BASE_N` staging area stays L2-resident.
The headline result: fusing a 348.868 us GELU into a 2601.311 us Matmul produced a 2606.91 us MIX kernel — the
vector work became free, hidden entirely under cube time. On A2 the same experiment: 369.827 us Gelu +
4227.605 us Matmul -> 4236.944 us fused (+9.3 us). Note on A2 the fused kernel's `aic_scalar_time` rose from
338.636 to 531.971 us (+57%) — the MIX mode's Scalar cost is visible and measurable, which is consistent with our
MIX_AIC launch floor being Scalar/handshake bound.
Also note the `Task Type` column is exactly the taxonomy we measured: `AI_VECTOR_CORE`, `AI_CORE`, `MIX_AIC` —
with `Block Num` / `Mix Block Num` showing 24/0, 24/0, 24/48 respectively on A2. `msopprof --aic-metrics=PipeTimeline`
is the command that produces the pipe timeline, and `MemoryL0.csv` gives L0A/L0B/L0C bandwidth.

---

## 17. `npugraph_ex` RFC — the vendor's stated plan for kernel-launch reduction in vllm-ascend

https://github.com/vllm-project/vllm-ascend/issues/4715
Title: `[RFC]: npugraph_ex backend`. Author: ChenCangtao. Created 2025-12-04T11:59:51Z.

> ### Motivation.
>
> In order to achieve better performance when using aclgraph, we introduced the npugraph_ex backend, which is a simple, easy-to-boundary, and accuracy-concern-free fullgraph aclgraph acceleration solution.
>
> This is an overview of our npugraph_ex, all optimization capabilities are performed after AOT autograd
>
> Using the npugraph_ex backend allows for more NPU-friendly optimizations on fx graphs, and node fusion in fx graphs is easy to extend.
>
> ### Proposed Change.
>
> We propose introducing the npugraph_ex backend, which is based on the vllm Adaptor dispatch mechanism.
>
> - All the work of npugraph_ex is based on fx.graph to enhance the experience of aclgraph
> - Use the API in torch_npu to：
>     -  Match known computational subgraphs that can be fused
>     -  Replace them with fused kernels implemented in .torch_npu
>
> Optimize before executing the model, so that when the model does a dummy run, it's already running the optimized graph.
>
> #### Design Overview
>
> 1. Leverage vLLM’s Existing VllmBackend Path
> - We keep the original VllmBackend execution flow in place.
> - We use the npugraph_ex backend by implementing and introducing AscendAdaptor for NPU-friendly graph optimization and compilation. At the same time, a new configuration item was added in additional_config to control whether to use the npugraph_ex backend.
>
> 2.  Custom Fusion Pass Implementation
> This capability is entirely provided by torch_npu. Two main steps of work are needed
> - Implement a target operator matching method
> - Declare fused operators and parameter passing before and after fusion
> We provided an example for custom operator fusion. [norm quant fusion pass demo](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/compilation/npugraph_ex_passes/graphex_norm_quant_fusion_pass.py)

> 1. How to use it by config?
> ```
> additional_config={
>             "npugraph_ex_config": {
>                 "enable": True
>             }
>         }
> ```

> - **The specific implementation of npugraph_ex and its related wiki：** https://gitcode.com/Ascend/torchair/wiki

> ### The current progress of this RFC
> These PRs have been merged：
> - [#4700](https://github.com/vllm-project/vllm-ascend/pull/4700#issue-3693332280)
> - [#5120](https://github.com/vllm-project/vllm-ascend/pull/5120#issue-3737544253)
> - [#5491](https://github.com/vllm-project/vllm-ascend/pull/5491#issue-3767394993)
> - [#5569](https://github.com/vllm-project/vllm-ascend/pull/5569#issue-3778701756)
>
> These PRs are waiting to be merged：
> - [#5775](https://github.com/vllm-project/vllm-ascend/pull/5775#issue-3799594997)
> - [#5721](https://github.com/vllm-project/vllm-ascend/pull/5721#issue-3791830255)
>
> After #5775 is merged, npugraph_ex can run with the default configuration, and support for more features will require waiting for the upcoming versions of torch_npu and CANN.

> ### Fature Plan
> We have submitted the basic code [[Feature] Support npuhraph_ex backend](https://github.com/vllm-project/vllm-ascend/pull/4700)
> We plan and hope that in the future this backend can become the default optimization for aclgraph.
> These optimization measures will also be included in our plan：
> 1. Memory reuse optimization
> 2. Kernel Performance Optimization
> 3. Redundant kernel optimization processing
>
> These features are also in our plan, but they require users to modify their scripts：
> 1. Support muti-stream
> 2. Support for the Dynamo cache in torch.compile
> 3. Supports weight prefetching
> 4. Weight proprietary format optimization — pre-convert the cube operator's weights to NZ format

## Relevance
Confirms the vendor's two-layer attack on launch count: FX-level node fusion (npugraph_ex) feeding ACLGraph capture,
with SuperKernel on top at binary level. The named fusion target in the shipped demo is **norm + quant** —
`graphex_norm_quant_fusion_pass.py` — which is the same pattern as our W4A8C8 RMSNorm+quant per layer. Also note
item 4 of the future plan: pre-converting cube weights to NZ format, which matches the `CubeFormat::NZ` /
`FRACTAL_ZN` recommendations in sections 8 and 11.

---

## 18. FFTS+ (Flexible Task Scheduling Plus) — the task-scheduling layer that classifies MIX_AIC / MIX_AIV

### `cann/ge` FFTS+ ops kernel builder

https://gitcode.com/cann/ge/blob/master/compiler/engines/ffts_engine/task_builder/fftsplus_ops_kernel_builder.cc
raw: https://raw.gitcode.com/cann/ge/raw/master/compiler/engines/ffts_engine/task_builder/fftsplus_ops_kernel_builder.cc
Repo `cann/ge`, branch `master`. Copyright (c) 2025 Huawei Technologies Co., Ltd.
(GitHub mirror of the same tree: https://github.com/hicann/ge)

The complete FFTS+ context-type taxonomy, verbatim:

```cpp
const std::string kFFTSPlusCoreName = "ffts_plus";
const uint64_t max_preload_context_num = 1000;
constexpr uint32_t kUInt8Max = 255;

const std::unordered_set<uint32_t> SUPPORT_CTX_PATH_TYPE{RT_CTX_TYPE_AICORE, RT_CTX_TYPE_AIV, RT_CTX_TYPE_MIX_AIC,
                                                         RT_CTX_TYPE_MIX_AIV};

static const std::unordered_map<rtFftsPlusContextType_t, std::string> kCtxTypeStrMap = {
    {RT_CTX_TYPE_AICORE, "aicore"},
    {RT_CTX_TYPE_AIV, "aiv"},
    {RT_CTX_TYPE_NOTIFY_WAIT, "notify_wait"},
    {RT_CTX_TYPE_NOTIFY_RECORD, "notify_record"},
    {RT_CTX_TYPE_WRITE_VALUE, "write_value"},
    {RT_CTX_TYPE_MIX_AIC, "mix_aic"},
    {RT_CTX_TYPE_MIX_AIV, "mix_aiv"},
    {RT_CTX_TYPE_SDMA, "sdma"},
    {RT_CTX_TYPE_FLUSH_DATA, "flush_data"},
    {RT_CTX_TYPE_INVALIDATE_DATA, "invalidate_data"},
    {RT_CTX_TYPE_WRITEBACK_DATA, "writeback_data"},
    {RT_CTX_TYPE_AICPU, "aicpu"},
    {RT_CTX_TYPE_COND_SWITCH, "cond_switch"},
    {RT_CTX_TYPE_CASE_SWITCH, "case_switch"},
    {RT_CTX_TYPE_AT_START, "at_start"},
    {RT_CTX_TYPE_AT_END, "at_end"},
    {RT_CTX_TYPE_LABEL, "label"},
    {RT_CTX_TYPE_PERSISTENT_CACHE, "persistent_cache"},
    {RT_CTX_TYPE_DSA, "dsa"}};

REGISTER_OPS_KERNEL_BUILDER(kFFTSPlusCoreName, FFTSPlusOpsKernelBuilder);
```

Subgraph SQE generation — the prefetch and preload knobs, verbatim:

```cpp
Status FFTSPlusOpsKernelBuilder::GenSubGraphSqeDef(domi::TaskDef &task_def, const uint64_t &ready_context_num,
                                                   const ge::ComputeGraph &sgt_graph) const {
  ...
  uint64_t preload_context_num;
  if (ready_context_num > max_preload_context_num) {
    preload_context_num = max_preload_context_num;
  } else {
    preload_context_num = ready_context_num;
  }

  ffts_plus_sqe->set_ready_context_num(ready_context_num);
  ffts_plus_sqe->set_preload_context_num(preload_context_num);
  ffts_plus_sqe->set_total_context_num(gen_ctx_size);
  ffts_plus_sqe->set_prefetch_ost_num(Configuration::Instance().GetPrefetchOstNum());
  ffts_plus_sqe->set_cmaint_ost_num(Configuration::Instance().GetCmaintOstNum());
  ffts_plus_sqe->set_aic_prefetch_lower(Configuration::Instance().GetAicPrefetchLower());
  ffts_plus_sqe->set_aic_prefetch_upper(Configuration::Instance().GetAicPrefetchUpper());
  ffts_plus_sqe->set_aiv_prefetch_lower(Configuration::Instance().GetAivPrefetchLower());
  ffts_plus_sqe->set_aiv_prefetch_upper(Configuration::Instance().GetAivPrefetchUpper());
  ffts_plus_sqe->set_data_split_unit(Configuration::Instance().GetDataSplitUnit());
  return SUCCESS;
}

const std::unordered_set<std::string> FFTS_NO_NEED_GEN_TASK_OP_TYPE = {"Data",  "NetOutput", "Variable",
                                                                       "Const", "Constant",  "PhonyConcat"};
const std::unordered_set<std::string> FFTS_CONTROL_OP_V2_TYPE = {"If", "While", "Case"};
```

Context dependency bookkeeping (dispatch on type), verbatim:

```cpp
Status FFTSPlusOpsKernelBuilder::SetSingleCtxPolicyPri(uint32_t type, domi::FftsPlusCtxDef *ffts_plus_ctx_def,
                                                       FftsPlusContextPath &ctx_path) const {
  switch (type) {
    case RT_CTX_TYPE_AICORE:
    case RT_CTX_TYPE_AIV:
      return FFTSPlusTaskBuilder::set_policy_pri(ctx_path.ctx_id, ctx_path.policy_pri,
                                                 ffts_plus_ctx_def->mutable_aic_aiv_ctx());
    case RT_CTX_TYPE_MIX_AIC:
    case RT_CTX_TYPE_MIX_AIV:
      return FFTSPlusTaskBuilder::set_policy_pri(ctx_path.ctx_id, ctx_path.policy_pri,
                                                 ffts_plus_ctx_def->mutable_mix_aic_aiv_ctx());
    default:
      // other type ctx no need to process
      return SUCCESS;
  }
```

```cpp
      "Ctx_path content: ctx_id:%u, type:%u, pre_cnt:%u, policy_pri:%hu, max_pre_index:%d,"
      "pre_list size:%zu, cmo_list size:%zu, label_list size:%zu, succ_list size:%zu.",
```

### FFTS+ task protobuf — AIC/AIV vs MIX_AIC/AIV context fields

https://github.com/hicann/ge/blob/master/graph_metadef/proto/task.proto
raw: https://raw.githubusercontent.com/hicann/ge/master/graph_metadef/proto/task.proto

```protobuf
    uint32 ready_context_num = 5;
    uint32 preload_context_num = 6;

    uint32 prefetch_ost_num = 7;
    uint32 cmaint_ost_num = 8;

    uint32 aic_prefetch_lower = 9;
    uint32 aic_prefetch_upper = 10;
    uint32 aiv_prefetch_lower = 11;
    uint32 aiv_prefetch_upper = 12;

    uint32 data_split_unit = 13;
}

message StarsSqeHeaderDef {
    uint32 l1_lock = 1;
    uint32 l1_unlock = 2;
    uint32 block_dim = 3;
}

// ffts plus context
message FftsPlusCtxDef {
    enum OpType {
        NORMAL = 0;
        ATOMIC = 1;
    }
    uint32 op_index = 1;
    string uniq_ctx_name = 2;
    uint32 context_type = 3;
    uint32 context_id = 4;
    OpType op_type = 5;

    FftsPlusAicAivCtxDef aic_aiv_ctx = 6;
    FftsPlusMixAicAivCtxDef mix_aic_aiv_ctx = 7;
    FftsPlusSdmaCtxDef sdma_ctx = 8;
    FftsPlusNotifyCtxDef notify_ctx = 9;
    FftsPlusWriteValueCtxDef write_value_ctx = 10;
    FftsPlusAicpuCtxDef aicpu_ctx = 11;
    FftsPlusDataCtxDef data_ctx = 12;
    FftsPlusAtStartCtxDef at_start_ctx = 13;
    FftsPlusAtEndCtxDef at_end_ctx = 14;
    FftsPlusLabelCtxDef label_ctx = 15;
    FftsPlusCaseSwitchCtxDef case_switch_ctx = 16;
    FftsPlusCaseDefaultCtxDef case_default_ctx = 17;
    FftsPlusCondSwitchCtxDef cond_switch_ctx = 18;
    FftsPlusCachePersistCtxDef cache_persist_ctx = 19;
    FftsPlusDsaCtxDef dsa_ctx = 20;
}

// aic/aiv context
message FftsPlusAicAivCtxDef {
    uint32 successor_num = 1;
    uint32 aten = 2;
    uint32 prefetch_config = 3;
    uint32 pred_cnt_init = 4;
    uint32 pred_cnt = 5;
    repeated uint32 successor_list = 6; // 16 bits, len = 26

    uint32 schem = 7;
    uint32 atm = 8;
    uint32 prefetch_enable_bitmap = 9;
    uint32 prefetch_once_bitmap = 10;

    uint32 pmg = 11;
    uint32 ns = 12;
    uint32 part_id = 13;
    uint32 qos = 14;

    uint32 thread_id = 15;
    uint32 thread_dim = 16;

    uint32 non_tail_block_dim = 17;
    uint32 tail_block_dim = 18;

    uint32 task_param_ptr_offset = 19;
    uint32 save_task_addr = 20;
    repeated uint64 task_addr = 21;
    repeated uint64 task_addr_offset = 22;
    uint32 input_output_count = 23;

    repeated string kernel_name = 24;

    repeated uint32 src_slot = 25;  // len = 4, context ID for source data which is out of subgraph
    uint32 policy_pri = 26;
    uint32 thread_window_size = 27;
}

// mix aic/aiv context
message FftsPlusMixAicAivCtxDef {
    uint32 successor_num = 1;
    uint32 aten = 2;
    uint32 prefetch_config = 3;
    uint32 pred_cnt_init = 4;
    uint32 pred_cnt = 5;
    repeated uint32 successor_list = 6; // len = 26

    uint32 schem = 7;
    uint32 atm = 8;
    uint32 prefetch_enable_bitmap = 9;
    uint32 prefetch_once_bitmap = 10;

    uint32 pmg = 11;
    uint32 ns = 12;
    uint32 part_id = 13;
    uint32 qos = 14;

    uint32 non_tail_block_ratio_n = 15;
    uint32 tail_block_ratio_n = 16;

    uint32 thread_id = 17;
    uint32 thread_dim = 18;

    uint32 non_tail_block_dim = 19;
    uint32 tail_block_dim = 20;

    uint32 aic_task_param_ptr_offset = 21;
    uint32 aiv_task_param_ptr_offset = 22;

    repeated string kernel_name = 23;

    repeated uint64 task_addr = 24;
    repeated uint64 task_addr_offset = 25;
    uint32 input_output_count = 26;
    uint32 save_task_addr = 27;
    repeated uint32 src_slot = 28;  // len = 4, context ID for source data which is out of subgraph
    uint32 policy_pri = 29;
    uint32 thread_window_size = 30;
    string args_format = 31;
}
```

Related sources found but not fully read (see Gaps):
`compiler/engines/ffts_engine/task_builder/mixl2_ctx/mix_l2_task_builder.cc`,
`compiler/engines/ffts_engine/task_builder/thread_ctx/mix_aic_aiv_dynamic_task_builder.cc`,
`runtime/v2/engine/ffts_plus/converter/ffts_plus_proto_transfer.h`, all in the `hicann/ge` / `cann/ge` tree.
Also `runtime/v1/hybrid/node_executor/node_executor.cc` maps the FFTS+ kernel name to an RTS executor type.

## Relevance
FFTS+ is the scheduling layer where our `MIX_AIC` / `MIX_AIV` / `AI_CORE` / `AI_VECTOR_CORE` task-type labels come
from: `RT_CTX_TYPE_AICORE`, `RT_CTX_TYPE_AIV`, `RT_CTX_TYPE_MIX_AIC`, `RT_CTX_TYPE_MIX_AIV` are literally the four
context types that participate in dependency paths (`SUPPORT_CTX_PATH_TYPE`). The structural difference between
`FftsPlusAicAivCtxDef` and `FftsPlusMixAicAivCtxDef` explains the MIX launch surcharge concretely:
a MIX context carries **two separate task-parameter pointers** (`aic_task_param_ptr_offset` and
`aiv_task_param_ptr_offset`) plus `non_tail_block_ratio_n` / `tail_block_ratio_n` (the AIC:AIV ratio), where a plain
AIC/AIV context has a single `task_param_ptr_offset` and no ratio field. So one MIX dispatch sets up two kernel
argument blocks and a ratio, against one block for a pure kernel. The SQE-level levers that exist but are
compile-configured rather than user-visible are `preload_context_num` (capped at `max_preload_context_num = 1000`),
`prefetch_ost_num`, `cmaint_ost_num`, and the four `aic/aiv_prefetch_lower/upper` bounds — i.e. FFTS+ already
preloads up to 1000 contexts ahead, which is why a captured graph replays far cheaper than 3008 individual
host launches.

---

## 19. SuperKernel at the AscendC level: `SK_BIND`, `__sk__`, Early-Start, and the exact adaptation constraints

Source repo: `hicann/asc-devkit` on GitHub (the `cann/asc-devkit` doc tree), branch `master`.

### 原理介绍 (principles.md)

https://github.com/hicann/asc-devkit/blob/master/docs/zh/guide/programming_guide/advanced_programming/super_kernel/principles.md
raw: https://raw.githubusercontent.com/hicann/asc-devkit/master/docs/zh/guide/programming_guide/advanced_programming/super_kernel/principles.md

> # 原理介绍
>
> ## 概述
>
> SuperKernel是一种算子的二进制融合技术。与源码融合不同，它聚焦于核函数（Kernel）的二进制调度方案优化，在已编译的二进制代码基础上，将多个子核函数（Kernel）融合创建出一个超级核函数（Kernel）（简称SuperKernel），由该SuperKernel以子函数调用的方式整合多个核函数（Kernel），从而优化计算任务、提升性能和资源利用率。
>
> 与单算子下发相比，SuperKernel技术能够减少Task调度的等待时间和调度开销，同时利用Task间隙资源进一步优化算子头开销，提升推理性能。
>
> 开启SuperKernel融合优化后，系统会自动识别图内可被融合的算子，并生成SuperKernel进行计算。同时框架提供标定SuperKernel范围的能力，支持用户根据实际业务需求对融合范围内的算子进行标记和优化配置。
>
> >[!NOTE]说明
> >SuperKernel适用于如下型号：
> > - Atlas A3系列产品
> > - Atlas A2系列产品
> > - Ascend 950PR&950DT系列产品，该型号下SuperKernel特性是试验特性，在后续版本中会调整或改进，不保证后续兼容性，请开发者在使用过程中关注后续版本更新。
>
> ## 性能收益来源
>
> 相对于单算子下发，SuperKernel主要通过以下方式带来性能收益：
>
> - **减少核函数（Kernel）启动开销**：原本需要多次启动的子核函数（Kernel），被统一通过一次SuperKernel启动完成，减少了任务调度开销。
> - **减少Task调度等待时间**：避免了多个独立Task之间的调度间隙。
> - **算子间流水并行**：通过SuperKernel内提供的任务间同步接口（SetNextTaskStart、WaitPreTaskEnd），可以让子核函数（Kernel）之间的部分指令实现并行执行，进一步压缩端到端耗时。
>
> ## 支持的开启方式
>
> 当前SuperKernel特性通过PyTorch图模式开启，主要有如下两种方式：
>
> | 开启方式 | 说明 |
> | --- | --- |
> | npugraph_ex后端 | 在`torch.compile`的`options`中传入`super_kernel_optimize=True`开启SuperKernel融合；如需控制融合范围，可通过`torch.npu.super_kernel_scope_begin/end`进行标定。具体请参考[SuperKernel功能](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md)。 |
> | GE图模式 | 通过`torchair.scope.super_kernel`作用域标定融合范围，配合TorchAir的CompilerConfig开启。具体请参考[图内标定SuperKernel范围](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/advanced/super_kernel_scope.md)。 |
>
> ## 算子适配说明
>
> CANN包发布的主流模型常用算子已支持SuperKernel融合。用户自定义的Ascend C算子如需支持SuperKernel，需要满足一定适配条件

### 算子适配说明 (operator_adaptation.md) — the fusion-eligibility constraints

https://github.com/hicann/asc-devkit/blob/master/docs/zh/guide/programming_guide/advanced_programming/super_kernel/operator_adaptation.md
raw: https://raw.githubusercontent.com/hicann/asc-devkit/master/docs/zh/guide/programming_guide/advanced_programming/super_kernel/operator_adaptation.md

> 自定义算子要支持融合进SuperKernel，开发流程上与普通算子并无显著差异，但需要遵循以下一系列**特定约束**。这些约束与具体的开启方式（npugraph_ex后端或GE图模式）无关，大部分约束与算子启动核数（block num）以及SuperKernel启动核数之间的关系相关，需要开发者在算子设计阶段就予以关注。
>
> >[!NOTE]说明
> >SuperKernel启动核数为其中所有子核函数（Kernel）的最大启动核数。例如，SuperKernel包含算子a（启动核数为4）和算子b（启动核数为2），则SuperKernel启动核数为4。
>
> ## 全核同步约束
>
> 自定义算子若进行全核同步，需注意该子核函数（Kernel）启动的核数需要与SuperKernel的核数一致，通常需要启动当前硬件满核。若子核函数（Kernel）启动的核数少于SuperKernel的核数，全核同步会等待所有核完成，导致卡住超时。
>
> ## MIX 1:1算子核数比例约束
>
> 若自定义算子的核函数（Kernel）类型设置为`KERNEL_TYPE_MIX_AIC_1_1`，因为SuperKernel会根据启动核数等信息调整SuperKernel的启动比例，此时需特别注意该算子也可以适应SuperKernel的1:2启动比例，确保AIC与AIV之间的硬同步操作正确执行。比如：
>
> - 算子内部使用了AIC与AIV之间的硬同步接口（CrossCoreSetFlag和CrossCoreWaitFlag），不要单独指定某些AIV核调用硬同步接口，要使所有AIV核均调用硬同步接口，防止因为硬同步数量不匹配而导致卡死超时。
> - 使用Matmul高阶API时，算子逻辑应保证仅有一个AIV0核调用Matmul接口，防止启动两个AIV核之后出现AIV1核消息无法接收导致卡死超时。
>
> ## 标量读写与Cache一致性
>
> 在开发自定义算子时，开发者必须确保所有对GM的标量读写操作都按需正确插入DataCacheCleanAndInvalid指令。
>
> - 在单算子编译场景下，毕昇编译器自动在算子末尾添加DataCacheCleanAndInvalid指令，刷新整个DCache（数据缓存）。
> - 在SuperKernel中，子核函数（Kernel）被当作普通函数处理，编译器不会自动插入该指令来确保数据缓存一致性，开发者需要自行保证，避免因容错机制改变而导致错误。
>
> 出于性能考虑，SuperKernel场景下Cache刷新机制如下：
>
> - 如果开发者调用GlobalTensor的`GetValue`和`SetValue`接口对GM进行标量读写，在GE入图及npugraph_ex后端aclnn调用场景，SuperKernel编译时会自动在两个接口内部插入DataCacheCleanAndInvalid指令刷新单个Cache Line，保证一定的数据缓存一致性。不会在子核函数（Kernel）调用前后插入DataCacheCleanAndInvalid。
> - 如果开发者通过GlobalTensor的`()`运算符接口来获取值（读值时会插入DataCacheCleanAndInvalid指令刷新单个Cache Line保证数据缓存一致性），但通过该接口直接改写了GlobalTensor对应位置的值时，则不会自动插入DataCacheCleanAndInvalid指令，开发者需要自行保证数据缓存一致性。例如：
>
>     ```c++
>     AscendC::GlobalTensor<float> xGm;
>     xGm.SetGlobalBuffer((__gm__ float *)(addr), length); // addr为GM地址，length为对应GM长度
>     xGm(0) = (float)(1.0); // 在获取值时，SuperKernel能够保证自动刷新cache line，但是写值时，相当于普通变量直接赋值，SuperKernel无法插入DataCacheCleanAndInvalid，需要用户保证数据缓存的一致性。
>     ```
>
> 对于频繁调用`GetValue`和`SetValue`的算子，建议使用`dcci_before_kernel_start`、`dcci_after_kernel_end`关闭指定算子内`GetValue`/`SetValue`中自动插入的缓存刷新指令，而改为算子调用前后插入整个DCache刷新，避免性能劣化。
>
> ## 核数获取接口约束
>
> 在GE入图及npugraph_ex后端aclnn调用场景，子核函数（Kernel）中调用GetBlockNum接口获取核数时，无论是否融合SuperKernel，获取的核数保持不变，不受SuperKernel启动核数的影响。
>
> 在SuperKernel场景下，如下Ascend C API在算子编译过程中适配了SuperKernel，子算子需要严格按照Ascend C提供的API进行编程，从而无需感知是否开启SuperKernel。例如在获取核数和索引时，**不能调用`block_idx`、`block_num`等底层变量和相关API**，必须使用下表中的Ascend C API：
>
> | API列表 | 功能描述 |
> | --- | --- |
> | AscendC::GetBlockIdx() | 获取当前核的index |
> | AscendC::GetBlockNum() | 获取当前任务配置的核数 |
>
> ## TPipe析构约束
>
> 针对Atlas A2系列产品、Atlas A3系列产品，在GE入图及npugraph_ex后端aclnn调用场景，子算子TPipe对象析构（TPipe::Destroy）接口内部的`AscendC::PipeBarrier<PIPE_ALL>()`指令会被去除，如果算子内部使用了多个TPipe对象或手动调用Destroy函数时开发者需自行保障TPipe对象间流水的同步。
>
> ```c++
> // 场景一：两个TPipe对象通过作用域析构，需在第一个析构后插入PipeAll
> {
>     AscendC::TPipe pipe1;
>     AscendC::TQue<AscendC::TPosition::VECOUT, 2> que1;
>     uint8_t num1 = 2;
>     uint32_t len1 = 128;
>     pipe1.InitBuffer(que1, num1, len1);
>     // ... 使用pipe1进行算子计算 ...
> } // pipe1析构，但Destroy中的PipeBarrier已被去除
>
> // 必须手动插入PipeAll，保证pipe1流水完成后再使用pipe2
> AscendC::PipeBarrier<AscendC::PIPE_ALL>();
> ```
>
> ## 性能优化建议
>
> - **任务间同步**
>
>     开发者在进行核函数（Kernel）侧编程时，可以通过调用SetNextTaskStart和WaitPreTaskEnd两个任务间接口，进一步提升性能。
>
>     - 调用`SetNextTaskStart`后的指令可以和后续其他的子核函数（Kernel）实现并行，提升整体性能。SuperKernel按序调用子核函数（Kernel），为保证子核函数（Kernel）之间数据互不干扰，会在子核函数（Kernel）间插入算子间同步进行保序，子核函数（Kernel）N-1调用该接口后，之后的指令会和后续子核函数（Kernel）N实现并行。
>     - 调用`WaitPreTaskEnd`前的指令可以和前序其他的子核函数（Kernel）实现并行，提升整体性能。
>
> - **二进制复用优化**
>
>     在GE入图Tiling下沉场景下，可以通过`--op_relocatable_kernel_binary`编译选项，开启二进制复用优化，提升编译性能
>
> ## 核函数（Kernel）直调算子的额外适配
>
> 对于使用`<<<>>>`方式开发的Ascend C算子，目前仅支持在npugraph_ex后端融合进SuperKernel。此类算子除需遵循上述通用约束外，还需要在算子核函数（Kernel）入口侧增加SuperKernel入口函数。

### `SK_BIND` — binding a kernel to SK sub-functions (A2/A3 supported, 950 NOT)

https://github.com/hicann/asc-devkit/blob/master/docs/zh/api/Utils-API/SuperKernel/SK_BIND.md
raw: https://raw.githubusercontent.com/hicann/asc-devkit/master/docs/zh/api/Utils-API/SuperKernel/SK_BIND.md

> ## 产品支持情况
>
> - Ascend 950PR&950DT系列产品：不支持
> - Atlas A3系列产品：支持
> - Atlas A2系列产品：支持
> - Atlas 200I/500 A2推理产品：不支持
> - Atlas推理系列产品AI Core：不支持
> - Atlas推理系列产品Vector Core：不支持
> - Atlas训练系列产品：不支持
> - Kirin X90处理器系列产品：不支持
> - Kirin 9030处理器系列产品：不支持
>
> ## 功能说明
>
> 本接口为算子SuperKernel场景提供绑定原核函数（Kernel）和SK子函数的能力。
>
> ## 函数原型
>
> ```cpp
> #define SK_BIND(GF, cap, SK0,...)
> ```
>
> ## 参数说明
>
> | 参数名 | 输入/输出 | 描述 |
> | --- | --- | --- |
> | GF | 输入 | 核函数（Kernel）签名。 |
> | cap | 输入 | `uint64_t`类型，用于标记当前算子的SuperKernel特性。SuperKernel框架根据这些信息调整相应的同步和融合策略。各bit位可按位或组合：<br>&bull; 1：early start wait flag，表示算子内使用了WaitPreTaskEnd接口。<br>&bull; 2：early start set flag，表示算子内使用了SetNextTaskStart接口。<br>&bull; 4：disable_dcci，表示当前算子需要SuperKernel框架关闭调用前后的DCCI ALL指令。<br>&bull; 8：disable_batchmode_check，表示跳过当前算子的`__schedmode__`检查，使该算子可继续参与SuperKernel融合。<br>例如，同时配置early start wait flag和early start set flag时，cap取值为3。 <br>&bull; 16：blockdim_scale_up, 表示算子已自行支持核数向上扩展。融合后，SuperKernel会按自身的核数启动该算子；算子需要自行保证在实际启动核数大于其原始核数时仍能正确执行。|
> | SK0 | 输入 | SK子函数签名。 |
> | ... | 输入 | SK1~SK3。可选，可提供额外的SK子函数签名。SuperKernel可以为不同核指定不同的SK子函数缓解同地址访问开销。|
>
> ## 约束说明
>
> -   一个核函数（Kernel）最多绑定四个SK子函数。
> -   核函数（Kernel）直调算子目前仅支持在npugraph_ex后端进入SuperKernel，不支持GE图模式。

```cpp
#include "kernel_operator.h"

// 原普通核函数（Kernel）保留（用于非SuperKernel场景）
__global__ __vector__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
{
    KernelAdd op;
    op.Init(x, y, z, totalLength);
    op.Process();
}

// 规则3：定义参数结构体（根据原global函数的实际参数定义）
struct AddCustomArgs {
   GM_ADDR x;                      // 对应add_custom的第一个参数
   GM_ADDR y;                      // 对应add_custom的第二个参数
   GM_ADDR z;                      // 对应add_custom的第三个参数
   uint32_t totalLength;           // 对应add_custom的第四个参数
};
// 定义一个带模板参数的SK子函数
// 模板参数仅用于实例化出不同的符号，不影响函数逻辑
template<uint32_t splitNum>
__sk__ __vector__ void add_custom_sk(const AddCustomArgs *args, sk::SkSystemArgs *sysArgs/* 可选添加sysArgs参数 */)
{
    // 从结构体中获取参数
    GM_ADDR x = args->x;
    GM_ADDR y = args->y;
    GM_ADDR z = args->z;
    uint32_t totalLength = args->totalLength;
    // 规则5：逻辑与global函数一致
    KernelAdd op;
    op.Init(x, y, z, totalLength);
    op.Process();
}
// 规则6：使用SK_BIND绑定
// 通过指定模板参数实例化出4个不同的符号
// cap取值4表示配置disable_dcci
SK_BIND(add_custom, 4, add_custom_sk<0>, add_custom_sk<1>, add_custom_sk<2>, add_custom_sk<3>);
```

### Early-Start APIs: `SetNextTaskStart` / `WaitPreTaskEnd` — **effective on A3, no-op on A2**

https://github.com/hicann/asc-devkit/blob/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_task_sync/SetNextTaskStart.md
https://github.com/hicann/asc-devkit/blob/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_task_sync/WaitPreTaskEnd.md

> > [!NOTE]说明
> >本接口为试验接口，在后续版本中可能会调整或改进，不保证后续兼容性。
>
> ## 产品支持情况
>
> - Ascend 950PR&950DT系列产品：支持，该接口生效
> - Atlas A3系列产品：支持，该接口生效
> - Atlas A2系列产品：支持，仅保证编译兼容，实际功能不生效。
> - Atlas 200I/500 A2推理产品：支持，仅保证编译兼容，实际功能不生效。
> - Atlas推理系列产品AI Core：支持，仅保证编译兼容，实际功能不生效。
> - Atlas推理系列产品Vector Core：支持，仅保证编译兼容，实际功能不生效。
> - Atlas训练系列产品：支持，仅保证编译兼容，实际功能不生效。
>
> ## 功能说明
>
> 头文件路径为：`"basic_api/kernel_operator_common_intf.h"`。
>
> 算子层面支持SuperKernel特性Early-Start能力的接口，调用后在TorchAir层面：1.GE图模式下进行默认启用; 2.npugraph_ex后端通过early_start选项进行控制启用。
>
> **开发者需要自行保证调用此接口后的指令不会与后序算子互相干扰而导致精度问题，推荐在整个算子最后一条搬运指令后调用此接口。**

```cpp
// Ascend 950PR&950DT / Atlas A3 / Atlas A2 / Atlas 200I/500 A2 / Kirin X90 / Kirin 9030
template<pipe_t AIV_PIPE = PIPE_MTE3, pipe_t AIC_PIPE = PIPE_FIX, bool FORCE = false>
__aicore__ inline void SetNextTaskStart()

// Atlas 推理系列产品 AI Core / Vector Core / Atlas 训练系列产品
template<pipe_t AIV_PIPE = PIPE_MTE3, pipe_t AIC_PIPE = PIPE_MTE3, bool FORCE = false>
__aicore__ inline void SetNextTaskStart()
```

```cpp
template<bool FORCE = false>
__aicore__ inline void WaitPreTaskEnd()
```

> | 参数名 | 描述 |
> | --- | --- |
> | AIV_PIPE | SetNextTaskStart之后运行的指令，如果位于AIV上的AIV_PIPE流水，可以与后序算子并行。AIV_PIPE的取值范围为PIPE_MTE2、PIPE_MTE3、PIPE_S、PIPE_V。 |
> | AIC_PIPE | SetNextTaskStart之后运行的指令，如果位于AIC上的AIC_PIPE流水，可以与后序算子并行。AIC_PIPE的取值范围为PIPE_MTE1、PIPE_MTE2、PIPE_MTE3、PIPE_FIX、PIPE_M。 |
> | FORCE | 离线编译时是否强制启用Early-Start能力，不影响在线编译。<br>false：离线编译时不强制启用Early-Start能力；<br>true：离线编译时启用Early-Start能力，适用于无在线编译流程的算子（如<<<>>>开发的算子）。 |

> ## 约束说明
>
> - 该接口适用于TorchAir图模式开发场景，且需在启用SuperKernel特性后方可生效。
> - 在算子运行过程中，需要保证此接口在每个核上都被调用，且每个核上仅被调用一次。
> - 若子核函数（Kernel）某个TilingKey分支调用了此接口，则开发者需要保证当前算子可能会运行的所有TilingKey均调用了此接口，否则会出现因同步指令数量不匹配而卡住的现象。
> - 本接口内部实现会占用核间同步flagId，若与CrossCoreSetFlag/CrossCoreWaitFlag同时使用需注意避免flagId冲突

```cpp
AscendC::DataCopy(dstGlobal, dstLocal, TOTAL_LENGTH);
// 在最后一条搬运指令后调用，使之后的指令可与后续子核函数（Kernel）并行执行。
AscendC::SetNextTaskStart();
```

> 完整样例请参考TaskSync样例：`examples/01_simd_cpp_api/03_basic_api/05_sync_control/task_sync`

### `SetScheduleMode` — the batchmode setter

https://www.hiascend.com/doc_center/source/zh/canncommercial/81RC1/apiref/basicdataapi/atlasopapi_07_00248.html
(also https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC2/apiref/opdevgapi/atlasophostapi_07_0072.html)

> SetScheduleMode
>
> 函数功能: 设置算子在NPU上执行时的调度模式。
>
> 函数原型
>
> ```
> ge::graphStatus SetScheduleMode(const uint32_t schedule_mode)
> ```
>
> | 参数 | 输入/输出 | 说明 |
> |---|---|---|
> | schedule_mode | 输入 | 0：普通模式，默认情况下为普通模式。<br>1：batchmode模式，核间同步算子需要设置该模式。 |

```cpp
ge::graphStatus TilingForAdd(TilingContext *context) {
  uint32_t batch_mode = 1U;
  auto ret = context->SetScheduleMode(batch_mode);
  GE_ASSERT_SUCCESS(ret);
  ...
}
```

## Relevance
This is the highest-value operational material in the dossier, because it is the vendor's own answer to our exact
problem on our exact chip, and it changes two of my earlier readings.
- `SK_BIND` is **supported on A2/A3 and NOT on 950PR**, so the kernel-direct-launch (`<<<>>>`) path into SuperKernel
  is available to us. The mechanics: keep the original `__global__` kernel, define an args struct, write a
  `__sk__`-qualified sub-function taking `(const Args*, sk::SkSystemArgs*)`, and `SK_BIND(kernel, cap, sk0, sk1, sk2, sk3)`.
  Up to **four** SK sub-functions per kernel, and the reason for four is explicitly 「SuperKernel可以为不同核指定不同的
  SK子函数缓解同地址访问开销」 — the L2 instruction-address contention from section 6's 子Kernel拆分.
- `cap` bits are the exact fusion-policy switches: 1 = uses `WaitPreTaskEnd`, 2 = uses `SetNextTaskStart`,
  4 = `disable_dcci`, 8 = `disable_batchmode_check` (skip the `__schedmode__` check so the operator can still fuse),
  16 = `blockdim_scale_up` (operator tolerates being launched with more cores than it was written for).
- Early-Start (`SetNextTaskStart` / `WaitPreTaskEnd`) **is effective on A3 and a compile-only no-op on A2**. That is
  an A3-specific lever we have and A2 users do not. Defaults are `AIV_PIPE = PIPE_MTE3`, `AIC_PIPE = PIPE_FIX`, and
  `FORCE = true` is required for `<<<>>>`-developed operators because they have no online-compile step.
- Three constraints will bite a megakernel directly: (a) SuperKernel's launch core count is the **max** over its
  sub-kernels, and any sub-kernel doing a full-core sync must be launched with the same core count or it hangs —
  「通常需要启动当前硬件满核」; (b) a `KERNEL_TYPE_MIX_AIC_1_1` operator must also work at the 1:2 ratio SuperKernel may
  impose, every AIV must call the hard-sync API, and with the Matmul high-level API only AIV0 may call Matmul;
  (c) inside a SuperKernel sub-kernel the compiler no longer emits the trailing `DataCacheCleanAndInvalid`, and
  `TPipe::Destroy`'s internal `PipeBarrier<PIPE_ALL>()` is **removed** on A2/A3 — so multi-`TPipe` kernels must insert
  `AscendC::PipeBarrier<AscendC::PIPE_ALL>()` by hand. Also `block_idx` / `block_num` raw variables are forbidden;
  only `AscendC::GetBlockIdx()` / `GetBlockNum()` are SuperKernel-aware.
- `dcci_before_kernel_start` / `dcci_after_kernel_end` are the per-operator options that trade per-`GetValue`/`SetValue`
  cache-line flushes for one whole-DCache flush around the operator — which is exactly what vllm-ascend passes as
  `{"dcci_after_kernel_end": [".*"]}` in section 7.

---

## 20. The exact cross-core `flagId` allocation table (what is already taken, per architecture)

https://github.com/hicann/asc-devkit/blob/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.md
raw: https://raw.githubusercontent.com/hicann/asc-devkit/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.md

> ## 核间同步flagId占用说明
>
> ### 概述
>
> 开发者使用CrossCoreSetFlag和CrossCoreWaitFlag接口进行核间同步时，需要传入核间同步标记**flagId**，每个flagId对应一个用于控制同步的计数器。
>
> 部分接口的内部实现也会占用一部分flagId。当开发者同时使用这些接口与CrossCoreSetFlag/CrossCoreWaitFlag时，若flagId发生冲突（即相互独立的同步操作复用了同一个flagId），其对应的计数器会被错误地共享和修改，导致核间同步行为异常：可能出现阻塞无法解除而使程序卡死，或同步提前完成、阻塞提前解除而导致计算结果错误。
>
> 因此，当开发者同时调用CrossCoreSetFlag/CrossCoreWaitFlag与这些占用flagId的接口时，应**避开各接口内部已占用的flagId**，选用未被占用的flagId。
>
> 核间同步分为**模式0、1、2**与**模式4**，两者的底层硬件指令不同，flagId相互独立
>
> ### 模式0、1、2的flagId占用
>
> **表1**  模式0、1、2flagId占用情况
>
> | 接口 | 限定条件 | 占用flagId | 各计算核占用情况 | NPU架构版本 |
> | :--- | :--- | :--- | :--- | :--- |
> | SyncAll | `isAIVOnly=true` | 14 | AIV：14 | 2201、3510 |
> | SyncAll | `isAIVOnly=false` | 11、12、13 | AIC：11、12、13；AIV：12、13 | 2201、3510 |
> | SetNextTaskStart | SuperKernel子核函数（Kernel）中调用 | 11、14 | AIC：11；AIV：14 | 2201、3510 |
> | WaitPreTaskEnd | SuperKernel子核函数（Kernel）中调用 | 11、12、13、14 | AIC：11、12、13；AIV：12、13、14 | 2201、3510 |
> | Matmul | 定义N个Matmul对象 | `[0, 2N-1]` | AIV/AIC按对象与subBlock分配 | 2201 |
> | Matmul | SuperKernel融合场景（GetTaskRatio返回2，即1:2配比） | 15 | AIV：15 | 2201 |
>
> ### 模式4的flagId占用
>
> **表2**  模式4flagId占用情况
>
> | 接口 | 限定条件 | 占用flagId | 各计算核占用情况 | NPU架构版本 |
> | :--- | :--- | :--- | :--- | :--- |
> | SyncAll | `isAIVOnly=false`（核函数（Kernel）使用`__mix__(1, 2)`修饰时） | 12、13、28、29 | AIC：12、13、28、29；AIV：12、13 | 3510 |
> | WaitPreTaskEnd | SuperKernel子核函数（Kernel）中调用 | 12、13、28、29 | AIC：12、13、28、29；AIV：12、13 | 3510 |
> | Matmul | 定义N个Matmul对象 | `[0, N+17]` | AIC/AIV按对象、eventType、subBlock分配 | 3510 |
> | Matmul | A矩阵和B矩阵同时开启IBSHARE的场景 | 15 | AIV：15 | 3510 |
> | TSCM | 从UB发起的L1 Buffer（TSCM）队列 | 0~11（从11向下逐个分配） | AIV | 3510 |
> | DataCopy（GM→L1） | 无 | 1、17 | AIC：1、17；AIV：1 | 3510 |
>
> ### 各接口限定条件说明
>
> #### SyncAll
>
> - **isAIVOnly=true**：纯Vector算子的全核同步，占用模式0、1、2的flagId为14（AIV）。
> - **isAIVOnly=false**：Mix（包含Cube和Vector计算）算子的全核同步，占用情况随NPU架构与核函数（Kernel）类型不同：
>     - **2201架构**：占用模式0、1、2的flagId为11、12、13。
>     - **3510架构**：
>         - 核函数（Kernel）使用`__mix__(1, 1)`修饰时，占用模式0、1、2的flagId为11、12、13。
>         - 核函数（Kernel）使用`__mix__(1, 2)`修饰时，占用模式0、1、2的flagId为11，以及模式4的flagId为12、13、28、29。
>
> #### SetNextTaskStart / WaitPreTaskEnd
>
> - 二者为**SuperKernel**场景专用接口，仅在算子以SuperKernel方式融合、且子核函数（Kernel）中调用时才生效。
> - **SetNextTaskStart**：占用模式0、1、2的flagId为11（AIC）、14（AIV）。
> - **WaitPreTaskEnd**：
>     - **2201架构**：占用模式0、1、2的flagId为11、12、13、14。
>     - **3510架构**：占用模式0、1、2的flagId为11、12、13、14，以及模式4的flagId为12、13、28、29。
>
> #### Matmul
>
> - 占用flagId的范围与定义的**Matmul对象数目N**相关（N最大为4），并且随NPU架构不同而不同。
> - **2201架构**：占用模式0、1、2的flagId为[0, 2N-1]（N=4时占用`[0,7]`）；SuperKernel融合（1:2配比，即GetTaskRatio返回2）场景下还占用flagId为15。
> - **3510架构**：占用模式4的flagId为[0, N+17]（N=4时占用`[0,21]`）；A矩阵和B矩阵同时开启IBSHARE的场景下还占用flagId为15。
>
> #### TSCM
>
> - 仅当使用从UB发起的L1 Buffer（TSCM）队列时才触发。
> - 占用模式4的flagId，范围为**0~11**：第1个TSCM队列占用11，第2个占用10，依次向下递减分配，最多可分配到0。

The four modes restated with the pipe-blocking semantics, verbatim — **and the critical A3 caveat**:

> - 模式0：AI Core核间的同步控制。对于AIC全核场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的全部流水或者由模板参数pipe指定的流水（与NPU架构有关）中的指令才会执行；对于AIV全核场景，同步所有的AIV核...
> - 模式1：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了CrossCoreSetFlag，CrossCoreWaitFlag后续的全部流水或者由模板参数pipe指定的流水（与NPU架构有关）中的指令才会执行。
> - 模式2：AI Core内部，AIC与AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后，两个AIV上CrossCoreWaitFlag后续的全部流水或者由模板参数pipe指定的流水（与NPU架构有关）中的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的全部流水或者由模板参数pipe指定的流水（与NPU架构有关）中的指令才能执行。
> - 模式4：AI Core内部，AIC与单个AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后，单个AIV上CrossCoreWaitFlag后续的由模板参数pipe指定的流水中的指令才会继续执行；单个AIV执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的由模板参数pipe指定的流水中的指令才能执行。该模式仅在NPU架构版本3510上支持。
>
> > [!NOTE]说明
> > 模式0、1、2下`CrossCoreWaitFlag`阻塞的流水类型因NPU架构而异：
> > - 在NPU架构版本2201上，`CrossCoreWaitFlag`传入的`pipe`参数无效，无论取值为何都默认阻塞全部流水的后续指令，建议省略`pipe`参数并使用默认值。
> > - 在NPU架构版本3510上，`CrossCoreWaitFlag`传入的`pipe`参数生效，阻塞由模板参数`pipe`指定的流水的后续指令。

> 下述同步特性均以如下场景配置为例：核函数（Kernel）使用`__mix__(1, 2)`修饰，即每个AI Core包含1个AIC和2个AIV，并设置逻辑核数`numBlocks=2`，即共启动2个AI Core，因此存在2个AIC和4个AIV。为便于描述，将2个AIC分别编号为AIC0、AIC1；AI Core0中的2个AIV分别编号为AIV0-0、AIV0-1，AI Core1中的2个AIV分别编号为AIV1-0、AIV1-1。**各核中与`flagId`对应的计数器初始值均为0。**

> ```cpp
> // 进行核间同步时，即使只有AIC参与同步也不能用__cube__修饰核函数（Kernel），具体原因请参考CrossCoreSetFlag的约束说明。
> if (ASCEND_IS_AIC) {
>     // 每个核都应该有类似如下的成对调用CrossCoreSetFlag和CrossCoreWaitFlag。
>     // modeId必须配置为0，flagId要一致；CrossCoreWaitFlag的pipe使用默认值。
>     AscendC::CrossCoreSetFlag<0, PIPE_M>(0);       // 待前置PIPE_M流水任务完成后，通知调度模块。
>     AscendC::CrossCoreWaitFlag<0>(0);              // 直到所有AIC都完成前置PIPE_M流水，调度模块更新flagId=0的计数器后，解除所有流水后续指令的阻塞。
> }
> ```

## Relevance
This is the budget sheet for a hand-synchronized megakernel on A3 (`__NPU_ARCH__ == 2201`). The usable flagId space is
0-10 (section 4), and of that:
- `Matmul` with N objects takes `[0, 2N-1]` on 2201. With the max N=4 that is `[0,7]`, leaving only **8, 9, 10** free.
  With N=1 it takes `[0,1]`, leaving 2-10.
- `SyncAll<false>` takes 11, 12, 13 and `SyncAll<true>` takes 14 — both outside the user range 0-10, so they do not
  collide with hand-written flags. Same for `SetNextTaskStart` (11, 14) and `WaitPreTaskEnd` (11-14).
- Under SuperKernel at 1:2 ratio, `Matmul` additionally takes flagId **15** on 2201.
So: a megakernel using one Matmul object plus Early-Start can safely hand-allocate flagIds 2 through 10 — nine
independent cross-core events, each with a counter that saturates at 15 Sets.
One more A3-specific fact that matters for cost: on 2201 「`CrossCoreWaitFlag`传入的`pipe`参数无效，无论取值为何都默认
阻塞全部流水的后续指令」 — a cross-core wait on A3 stalls **all** pipes, not just the named one. Pipe-selective waits
and mode 4 (AIC to a *single* AIV) are 3510-only. That is a concrete, documented reason the same fused design costs
more on A3 than on 950PR, and it means every AIC->AIV handoff in our kernel is a full-pipe barrier on the waiting core.

---

## 21. Getting a `<<<>>>` kernel into SuperKernel: the complete `__sk__` adaptation recipe

https://github.com/hicann/asc-devkit/blob/master/docs/zh/guide/programming_guide/advanced_programming/super_kernel/kernel_direct_call_adaptation.md
raw: https://raw.githubusercontent.com/hicann/asc-devkit/master/docs/zh/guide/programming_guide/advanced_programming/super_kernel/kernel_direct_call_adaptation.md

> # 核函数（Kernel）直调算子额外适配说明
>
> 对于使用`<<<>>>`方式开发的Ascend C算子，除了遵循算子适配说明中的通用约束外，还需要在算子核函数（Kernel）入口侧增加SuperKernel入口函数。
>
> >[!NOTE]说明
> >核函数（Kernel）直调算子目前仅支持在npugraph_ex后端进入SuperKernel，不支持GE图模式。
>
> ## SuperKernel与普通核函数（Kernel）的区别
>
> ### 普通核函数（Kernel）入口函数
>
> - 函数签名：`__global__ __vector__ void func(参数列表)`
> - 参数直接作为函数参数传递。
> - 直接调用即可。
>
> ### SuperKernel入口函数
>
> - 函数签名：`__sk__ __vector__ void func(const ArgsStruct *args, sk::SkSystemArgs *sysArgs)`
> - 参数通过结构体指针传递，需要用户定义参数结构体并封装入参。
> - 可选添加`sk::SkSystemArgs *sysArgs`参数，用于获取系统信息（如block num）。
> - 需要与普通核函数（Kernel）通过SK_BIND宏绑定。
>
> ### 规则1：函数签名格式
>
> ```cpp
> __sk__ __vector__ void function_name_sk(const ArgsStruct *args, sk::SkSystemArgs *sysArgs)
> ```
>
> > **注意**：只有`__sk__`是强约束，必须出现在函数签名中。
>
> ### 规则2：Kernel Type标识与原global函数保持一致
>
> - 如果原global函数使用`__cube__`，则SK子函数也必须使用`__cube__`。
> - 如果原global函数使用`__vector__`，则SK子函数也必须使用`__vector__`。
> - 如果原global函数是`__mix__(c, v)`，则SK子函数也必须使用`__mix__(c, v)`。

```cpp
// 原global函数为cube类型
__global__ __cube__ void func(GM_ADDR x, GM_ADDR y);
// SK子函数也需要cube标识
__sk__ __cube__ void func_sk(const ArgsStruct *args, sk::SkSystemArgs *sysArgs);

SK_BIND(func, 4, func_sk);

// 原global函数为vector类型
__global__ __vector__ void func_vec(GM_ADDR x, GM_ADDR y);
__sk__ __vector__ void func_vec_sk(const ArgsStruct *args, sk::SkSystemArgs *sysArgs);

SK_BIND(func_vec, 4, func_vec_sk);

// 原global函数为mix类型
__global__ __mix__(1, 2) void func_mix(GM_ADDR x, GM_ADDR y);
__sk__ __mix__(1, 2) void func_mix_sk(const ArgsStruct *args, sk::SkSystemArgs *sysArgs);

SK_BIND(func_mix, 4, func_mix_sk);

// 原global函数为mix(1, 0)类型
__global__ __mix__(1, 0) void func_mix_10(GM_ADDR x, GM_ADDR y);
__sk__ __mix__(1, 0) void func_mix_10_sk(const ArgsStruct *args, sk::SkSystemArgs *sysArgs);

SK_BIND(func_mix_10, 4, func_mix_10_sk);

// 原global函数为mix(0, 1)类型
__global__ __mix__(0, 1) void func_mix_01(GM_ADDR x, GM_ADDR y);
__sk__ __mix__(0, 1) void func_mix_01_sk(const ArgsStruct *args, sk::SkSystemArgs *sysArgs);

SK_BIND(func_mix_01, 4, func_mix_01_sk);
```

> ### 规则3：定义参数结构体

```cpp
// 示例：假设原global函数为
// __global__ __vector__ void my_kernel(GM_ADDR x, GM_ADDR y, uint32_t length, int16_t flag)
// 则对应的参数结构体定义如下
struct MyArgs {
    GM_ADDR x;                      // 第一个参数
    GM_ADDR y;                      // 第二个参数
    uint32_t length;                // 第三个参数
    alignas(4) int16_t flag;        // 第四个参数（小于4字节，需使用alignas(4)）
};
```

> > **注意**：
> > - 结构体必须根据用户实际算子核函数（Kernel）（global函数）的参数进行定义，不要直接复制示例中的字段名。
> > - 结构体字段必须按原函数参数顺序排列。
> > - 对于size小于4字节的类型（如`int16_t`、`int8_t`、`uint16_t`、`uint8_t`），必须使用`alignas(4)`确保内存对齐（ABI要求）。
>
> ### 规则4：可选添加SkSystemArgs参数

```cpp
// 场景一：算子使用了AscendC::GetBlockNum()，需要添加sysArgs参数
template<uint32_t splitNum>
__sk__ __vector__ void func_sk(const ArgsStruct *args, sk::SkSystemArgs *sysArgs)
{
    // 从系统参数获取block num（替代AscendC::GetBlockNum()）
    uint32_t blockNum = sysArgs->skNumBlocks;
    // 或者使用接口获取
    uint32_t blockNum = sysArgs->SkGetNumBlocks();
    ...
}
```

> > - 与`AscendC::GetBlockNum()`的逻辑差异是，MIX 1:2场景中，vector核中的`skBlockNum`为`blockNum * 2`。
>
> ### 规则5：使用TPipe场景下，需要手动调用DestroyWithoutPipeAll接口
>
> 为了优化SK函数之间的同步，若使用了TPipe对象，需要在其自然析构之前调用`DestroyWithoutPipeAll()`接口主动销毁TPipe（不调用PipeBarrierAll）。

```cpp
template<uint32_t splitNum>
__sk__ __vector__ void func_sk(const ArgsStruct *args, sk::SkSystemArgs *sysArgs)
{
    ...
    // 使用TPipe场景
    TPipe pipe;
    ...
    // 手动调用接口销毁TPipe，不依赖默认析构函数
    pipe.DestroyWithoutPipeAll();
}
```

> ### 规则6：逻辑与global函数一致
>
> SK子函数内部的逻辑实现必须与原global函数完全一致，包括数据初始化、计算处理、结果输出等所有后续逻辑。
>
> ### 规则7：使用SK_BIND宏绑定
>
> ```cpp
> SK_BIND(original_func, 4, sk_func<0>, sk_func<1>, sk_func<2>, sk_func<3>);
> ```
>
> | 参数 | 说明 |
> | --- | --- |
> | `original_func` | 原global函数，如果是模板函数需要手动实例化 |
> | `4` | 掩码参数（uint64_t类型），用于配置SuperKernel特性 |
> | `sk_func<0>` ~ `sk_func<3>` | 最多4个SK子函数，如果是模板函数需要手动实例化 |
>
> **掩码参数取值**：
> 1/2/4/8/16分别代表不同bit位上置值的结果，比如8为第4bit置值的结果
>
> | 值 | 含义 |
> | --- | --- |
> | 1 | early start wait flag |
> | 2 | early start set flag |
> | 4 | disable_dcci |
> | 8 | disable_batchmode_check |
> | 16 | blockdim_scale_up |

適配步骤总结, verbatim:

> **步骤1：保留原global函数**
>
> 原有的`__global__ __vector__`函数保持不变，继续用于非SuperKernel场景。
>
> **步骤2：创建SK子函数**
>
> 1. 复制原global函数的代码
> 2. 根据原global函数的实际参数定义参数结构体（对于小于4字节的类型使用`alignas(4)`）
> 3. 修改函数签名为`__sk__ __vector__ void func_sk(const ArgsStruct *args, sk::SkSystemArgs *sysArgs)`（根据原函数的核函数（Kernel）类型添加对应标识）
> 4. 如果算子未使用`AscendC::GetBlockNum()`，可以省略`sk::SkSystemArgs *sysArgs`参数
> 5. 使用模板参数实例化出多个符号（模板参数仅用于实例化出不同的符号，不影响函数逻辑）
> 6. 从结构体中获取参数
> 7. 如果使用了`AscendC::GetBlockNum()`，修改为使用`sysArgs->skNumBlocks`或`sysArgs->SkGetNumBlocks()`
> 8. 使用TPipe场景下，需要手动调用`DestroyWithoutPipeAll`接口销毁TPipe
> 9. 保持函数内部逻辑与原global函数完全一致

> **Q1：为什么要定义参数结构体？**
>
> A：SuperKernel使用结构体传递参数，用户需要根据原global函数的实际参数定义一个结构体来封装所有入参，这样可以实现参数的统一管理和跨split共享。

## Relevance
This is a complete, step-by-step path from our existing `<<<>>>` kernels into SuperKernel on A3, which is the
cheapest route to collapsing 3008 launches without rewriting the kernels themselves. Key facts:
- `__mix__(1, 0)` and `__mix__(0, 1)` exist as kernel qualifiers alongside `__cube__`, `__vector__`, `__mix__(1,2)` —
  the source-level counterparts of `KERNEL_TYPE_MIX_AIC_1_0` / `MIX_AIV_1_0` from section 4.
- Only `__sk__` is a hard requirement in the signature; the kernel-type qualifier must match the original exactly.
- Arguments move from a flat parameter list into a struct passed by pointer, with `alignas(4)` mandatory for any
  sub-4-byte field (ABI). The struct is shared across the up-to-4 instantiated splits ("跨split共享").
- `AscendC::GetBlockNum()` must be replaced by `sysArgs->skNumBlocks` / `sysArgs->SkGetNumBlocks()`, and in MIX 1:2
  the vector core's `skBlockNum` is `blockNum * 2`.
- `TPipe::DestroyWithoutPipeAll()` must be called explicitly, specifically to avoid the `PipeBarrier<PIPE_ALL>` that
  normal destruction would emit between SK sub-functions — the same elision described in section 19.
- GE graph mode is not an option for `<<<>>>` kernels; only the npugraph_ex backend, which matches the
  `enable_npugraph_ex` + `enable_static_kernel` + `enable_super_kernel` triple in section 7.

---

## 22. TorchAir SuperKernel reference: scope semantics and the complete option tables

### `docs/zh/npugraph_ex/advanced/superkernel.md` — SuperKernel融合优化功能

https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md
raw: https://raw.gitcode.com/Ascend/torchair/raw/master/docs/zh/npugraph_ex/advanced/superkernel.md
Repo `Ascend/torchair` on GitCode, branch `master`.

> ## 功能简介
>
> SuperKernel是一种算子二进制融合技术。与源码融合不同，它聚焦于内核函数（Kernel）的二进制调度方案优化，在已编译的二进制代码基础上融合创建一个超级Kernel函数（简称SuperKernel），以子函数调用的方式整合多个内核函数，从而优化计算任务、提升性能和资源利用率。
>
> 与单算子下发相比，SuperKernel技术能够优化任务调度的等待时间和调度开销，并可利用Task间隙资源进一步优化算子头开销。
>
> SuperKernel的实现原理如下：
>
> 基于aclgraph捕获后的Model，SuperKernel模块识别可被融合的连续Task，将其合并为一个SuperKernel Task并替换原有Task，通过Runtime接口更新Model后基于新Model执行。
>
> ## 使用约束
>
> - 本功能支持如下产品：
>
>   - Atlas A3 训练系列产品/Atlas A3 推理系列产品
>   - Atlas A2 训练系列产品/Atlas A2 推理系列产品
>   - Ascend 950PR/Ascend 950DT，该型号下SuperKernel特性是试验特性
>
> - 需要注意的是，SuperKernel融合会按照网络中算子的顺序依次判断是否可被融合。当**识别到不可融合的算子时**，系统会生成第一段SuperKernel，并自动跳过该算子进行第二段SuperKernel融合。
> - 目前支持SuperKernel融合的通信类算子包括AllReduce、ReduceScatter、AllGather、AlltoAll。
> - 开启SuperKernel融合优化时，需要同时开启静态Kernel编译功能。
>
>   >[!NOTE]说明
>   >同一进程内存在多个Model时，若不同Model中存在输入输出Shape完全相同的算子，但SuperKernel相关选项配置不一致，后续相同Shape算子可能复用前一次静态Kernel编译产物，导致SuperKernel融合不符合预期。建议这些Model保持配置一致，或仅在需要SuperKernel的Model上开启静态Kernel编译。
>
> - 开启SuperKernel融合优化后，算子Data Dump功能将会失效。

Scope marking, verbatim:

```python
torch.npu.super_kernel_scope_begin(scope_name: str)
待融合的算子操作
torch.npu.super_kernel_scope_end(scope_name: str)
```

> scope_name：表示该范围内算子融合后的SuperKernel名称，相同的scope\_name代表相同的融合范围，由用户控制。若传入None，则该范围内的算子不进行SuperKernel融合。
>
> >[!NOTE]说明
> >- scope_name名称不超过255字节。
> >- 全局支持的scope_name（不同）数量不超过1024个。
> >- 标定范围不代表SuperKernel的融合结果。
>
> 当多个标定范围有交集时，按照标记集对算子进行划分。当交集不为子集时，op1和op2的标记集是{sk1}，归入一组；op3和op4的标记集是{sk1，sk2}，归入一组；op5、op6和op7的标记集是{sk2}，归入一组。op8、op9没有被标记，不进行融合。
>
> 当交集为子集时，嵌套范围内的算子会同时带有外层和内层scope标记。
>
> 当传入的标记为None，则被框定范围内的算子不进行融合，且该类型的标记具备最高优先级。
>
> 当全局只存在为None的标记时，未被标记的算子都将进入SuperKernel的融合流程。

```python
import torch
import torch_npu

opt_model = torch.compile(model, backend="npugraph_ex", options={"static_kernel_compile": True, "super_kernel_optimize": True, "super_kernel_optimize_options": dict, "super_kernel_debug_options": dict}, dynamic=False, fullgraph=True)
```

> **表 1**  参数说明
>
> |**参数名**|**参数说明**|
> |--|--|
> |super_kernel_optimize|布尔类型，是否开启SuperKernel融合优化。<br>•False（默认值）：默认关闭。<br>•True：开启SuperKernel融合优化。|
> |super_kernel_optimize_options|字典类型，SuperKernel融合优化参数。|
> |super_kernel_debug_options|字典类型，SuperKernel融合调试参数。<br>**说明：super_kernel_debug_options中的选项均会影响SuperKernel融合效果或运行性能，建议仅在功能调试时使用。**|

表 2 `super_kernel_optimize_options`, verbatim:

> |**参数名**|**参数说明**|
> |--|--|
> |dcci_before_kernel_start|通过本选项指定的算子，其内部调用GlobalTensor的GetValue/SetValue时不会自动插入缓存刷新指令，**而在SuperKernel调用该算子前**，会插入DataCacheCleanAndInvalid指令刷新ENTIRE_DATA_CACHE。<br>•当dcci_before_kernel_start、dcci_after_kernel_end和dcci_disable_on_kernel均未配置时，子算子内部调用GlobalTensor的GetValue/SetValue时默认不自动插入缓存刷新指令，SuperKernel默认会在所有子算子调用后插入DataCacheCleanAndInvalid指令刷新ENTIRE_DATA_CACHE。<br>•配置格式形如：`{"dcci_before_kernel_start": [".*op1_type.*", ".*op2_type.*"]}`。仅支持匹配算子符号名而不是op_type，算子符号名为xx_op_type_xxx，带有前缀及后缀，仅支持`.*`格式的正则表达。<br>•若本选项指定的算子为<<<>>>算子或通信类算子，将无法去除算子内部调用GlobalTensor的GetValue/SetValue时插入的缓存刷新指令，仅支持控制SuperKernel调用算子前是否插入。<br>•本选项指定的算子op_type可通过Profiling查看，例如：`{"dcci_before_kernel_start": [".*GroupedMatmul.*", ".*MoeGatingTopK.*"]}`。|
> |dcci_after_kernel_end|通过本选项指定的算子，其内部调用GlobalTensor的GetValue/SetValue时不会自动插入缓存刷新指令，**而在SuperKernel调用该算子后**，会插入DataCacheCleanAndInvalid指令刷新ENTIRE_DATA_CACHE。<br>•配置格式形如：`{"dcci_after_kernel_end": [".*op1_type.*", ".*op2_type.*"]}`。|
> |dcci_disable_on_kernel|通过本选项指定的算子，其内部调用GlobalTensor的GetValue/SetValue会默认自动插入缓存刷新指令，**而在SuperKernel调用该算子前后**，不会插入任何DataCacheCleanAndInvalid指令。<br>•当某个算子同时配置了dcci_disable_on_kernel及dcci_before_kernel_start、dcci_after_kernel_end时，dcci_disable_on_kernel优先级低于其余两者，其余两者的功能将生效。|
> |ubuf_lock_ignore_kernel|通过本选项声明指定算子不依赖UB Buffer的加锁、解锁机制保证数据访问顺序，自动优化融合策略。<br>•仅在Ascend 950PR/Ascend 950DT上生效。默认值为空列表。|
> |auto_op_parallel|多流融合场景下，SuperKernel默认按照算子下发顺序执行，相比单算子执行，可能无法达到最大化算子并行度。配置该选项后，在确保算子间依赖关系不变的前提下，可以将多条流上的算子按照Cube/Vector类型并行排布，获得相对较优的算子执行流水。<br>•0（默认值）：不使能多流自动Cube/Vector自动并行排布。<br>•1：使能多流自动Cube/Vector自动并行排布。|
> |early_start|配置是否启用SuperKernel Early-Start优化。<br>•0：关闭本功能（默认值）。<br>•1：开启本功能。<br>开启后，SuperKernel会识别并生效算子内适配的AscendC::SetNextTaskStart()/AscendC::WaitPreTaskEnd()任务间同步接口，达到算子头部Scalar指令提前执行的优化效果。|
> |aggressive_opt_strategies|字典类型，为保证功能正确性，SuperKernel默认采用保守融合策略，通过本选项可按需开启激进融合策略，实现更深度的优化。配置格式形如：`{"aggressive_opt_strategies": {"task_breaker_bypass": 0, "value_breaker_bypass": 0b00}}`，各子项说明如下：<br>1. task_breaker_bypass：<br>•0（默认值）：SuperKernel融合过程中，遇到非AICore类型Task（如AICPU算子）会自动断开；<br>•1：SuperKernel融合范围内遇到了非AICore类型Task，且该流范围内没有其他AICore算子，则不做断开处理，仅将非AICore算子从融合范围内剔除。<br>2. value_breaker_bypass，比特位组合控制ValueWait节点的融合策略，各比特位含义如下：<br>•bit0（0b01）：已配对的ValueWait节点（存在对应的ValueWrite节点且满足Flag匹配规则），在通过死锁检测后可进行融合。<br>•bit1（0b10）：未配对的ValueWait节点（不存在对应的ValueWrite节点），允许进行融合。<br>取值示例：<br>•0b00（默认值）：ValueWait节点不进行融合；<br>•0b01：仅允许已配对的ValueWait节点融合；<br>•0b10：仅允许未配对的ValueWait节点融合；<br>•0b11：同时允许已配对和未配对的ValueWait节点融合。<br>说明：若ValueWait节点被多个ValueWrite匹配，则将会触发SuperKernel报错，整体网络融合失败。|

表 3 `super_kernel_debug_options`, verbatim:

> |**参数名**|**参数说明**|
> |--|--|
> |debug_sync_all|配置本选项后，SuperKernel内每个算子间会自动插入SyncAll全核同步指令，可辅助定界SuperKernel机制中的算子间同步问题。<br>•0：关闭本功能（默认值），SuperKernel子算子调用后不插入SyncAll全核同步。<br>•1：开启本功能。|
> |debug_op_exec_trace|配置本选项后，启用算子执行轨迹追踪功能。该功能会在Kernel侧记录每个子算子的执行状态（包括SK入口启动、子算子启动、子算子完成、SK入口完成），在异常发生时会在plog中打印所有核的算子执行进度信息（含当前执行到的算子ID及符号信息），辅助定位算子级执行异常。|
> |debug_cross_core_sync_check|配置本选项后，启用子算子Cube&Vector核间同步（参见CrossCoreSetFlag接口mode 2介绍）校验功能，检测同步数量不匹配的问题。开启本功能时会隐式启用debug_op_exec_trace选项，用于辅助定位出错子算子。|
> |debug_per_op_max_core_num|配置本选项后，启用单算子满核验证模式，用于在SuperKernel融合场景下单独验证每个算子在满核条件下的功能正确性。开启后，融合范围内的每个算子将独立成为一个SuperKernel，并以设备最大核数启动运行（Cube&Vector均启动最大核数），从而覆盖核数相关的边界场景。开启本功能时会隐式启用debug_cross_core_sync_check选项。|

Working example, verbatim:

```python
import torch
import torch_npu

torch.random.manual_seed(42)


# 定义模型model
class Model(torch.nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, gating_input, smooth_scales):
        moe_gating_res = torch_npu.npu_moe_gating_top_k_softmax(
            gating_input, None, self.top_k
        )
        dynamic_quant_res = torch_npu.npu_dynamic_quant(
            moe_gating_res[0], smooth_scales=smooth_scales
        )
        return dynamic_quant_res


if __name__ == "__main__":
    token_num = 864
    expert_num = 2048
    top_k = 1024
    gating_input = torch.randn((token_num, expert_num), dtype=torch.bfloat16, device="npu")
    smooth_scales = torch.randn((top_k,), dtype=torch.bfloat16, device="npu")

    # 使用图模式后端编译并优化模型
    print("--------------------\t[SuperKernel][compile and optimize]\t--------------------")
    model_sk = torch.compile(Model(top_k), backend="npugraph_ex",
        options={
            "static_kernel_compile": True,
            "super_kernel_optimize": True, # 使能SuperKernel融合优化
        }
    )
    res_sk = model_sk(gating_input, smooth_scales)
```

### `docs/zh/ascend_ir/features/advanced/super_kernel_scope.md` — the GE-graph-mode options

https://gitcode.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/advanced/super_kernel_scope.md
raw: https://raw.gitcode.com/Ascend/torchair/raw/master/docs/zh/ascend_ir/features/advanced/super_kernel_scope.md

> 实现SuperKernel的原理如下：
>
> 1. 通过SuperKernel融合策略识别可被融合的子图。
> 2. 将子图内的算子按SuperKernel融合规则合并为一个大Kernel，在新Kernel内通过生成一段子Kernel调用代码将子图上所有Kernel入口函数完成一次调用，并基于图的依赖完成同步插入。
>
> ## 使用约束
>
> - 本功能仅适用于GE图模式场景，并且需要在静态图场景下使用。
> - 本功能支持如下产品：
>     - Atlas A5 训练系列产品/Atlas A5 推理系列产品
>     - Atlas A3 训练系列产品/Atlas A3 推理系列产品
>     - Atlas A2 训练系列产品/Atlas A2 推理系列产品

```python
with torchair.scope.super_kernel(scope: str, options: str = ''):
```

> - scope：表示上下文算子被融合的SuperKernel名，相同的scope代表相同的范围，由用户控制。若传入None，表示该范围内的算子不进行SuperKernel融合。
> - options：表示融合SuperKernel的编译选项...同时支持用户自定义组合编译选项，配置格式形如"<option1\>=<value1\>:<option2\>=<value2\>:<option3\>=......"，多个选项时用英文冒号分割。

表 1 编译选项说明, verbatim (the three most load-bearing for us):

> |选项参数|说明|
> |--|--|
> |feed-sync-all|当子算子启动核数小于SuperKernel启动核数且使用了SyncAll全核同步指令，若出现算子执行卡住或超时，可尝试配置本选项解决此问题。<br>0：关闭本功能（默认值），若子算子使用SyncAll全核同步指令，用户需自行保证子算子与SuperKernel启动核数相同。<br>1：开启本功能，系统自动识别SuperKernel内算子是否调用SyncAll全核同步指令，同时判断子算子启动核数是否小于SuperKernel启动核数。若小于SuperKernel启动核数，会在SuperKernel内子算子的其余核中插入SyncAll指令，保证与子算子内调用SyncAll次数匹配，防止卡住超时。<br>SuperKernel启动核数为子算子的最大启动核数。假设SuperKernel包括算子a（启动核数为4）和算子b（启动核数为2），此时SuperKernel启动核数为4。<br>启动核数可通过Profiling采集的性能数据获取，即"kernel_details.csv"文件中"Block Dim"字段<br>子算子启动核数：非SuperKernel场景下开启Profiling，"Block Dim"字段表示每个算子的启动核数。<br>SuperKernel启动核数：SuperKernel场景下开启Profiling，"Block Dim"字段表示SuperKernel的启动核数。|
> |stream-fusion|控制SuperKernel内是否启用多流融合，从而提升算子运行效率，选项取值如下：<br>0（默认值）：关闭多流融合。当标定范围内存在算子分布在多条流上的场景时，将触发编译报错，无法完成融合。<br>1：启用多流融合，表示SuperKernel融合范围的算子可分布在多条流上。<br>多流融合的收益来自减少各流上算子的调度开销，而非增加并行度，融合前后跨流并行能力不变。SuperKernel内部因同步约束可能引入串行化，实际收益取决于算子排布，通常Cube算子和Vector算子多流并行融合可以获得较好收益。|
> |strict-scope-check|本选项用于检查SuperKernel融合的范围是否符合预期。对于断开的SuperKernel、不支持融合的算子可通过本功能查询：<br>bypass：打印C++侧Warning级别日志，忽略该范围的SuperKernel生成。<br>abort：打印C++侧Error级别日志，对该范围的SuperKernel直接报错退出。<br>Warning或Error级别日志信息可在plog文件（文件名为plog-*pid_**.log）中查看，搜索关键字"super_kernel_scope"即可。<br>本功能暂不支持检查SuperKernel融合范围内的集合通信算子，如AllGather、AlltoAll等。|
> |dcci-before-kernel-start|通过本选项指定的算子，其内部调用GlobalTensor的GetValue/SetValue时不会自动插入缓存刷新指令，**而在SuperKernel调用该算子前**会插入DataCacheCleanAndInvalid指令，刷新整个DCache（数据缓存）...配置格式形如：dcci-before-kernel-start=<op1_type>,<op2_type>...若本选项指定的算子为支持Tiling下沉的算子或通信类算子，功能将不生效。...本选项指定的算子op_type可通过Profiling查看，例如：dcci-before-kernel-start=GroupedMatmul,MoeGatingTopK。|
> |dcci-after-kernel-end|通过本选项指定的算子，其内部调用GlobalTensor的GetValue/SetValue时不会自动插入缓存刷新指令，**而在SuperKernel调用该算子后**，会插入DataCacheCleanAndInvalid指令，刷新整个DCache（数据缓存），保证该算子内的数据缓存不会影响后续的算子。<br>配置格式形如：dcci-after-kernel-end=<op1_type>,<op2_type>。<br>其他要求与dcci-before-kernel-start一样。|

## Relevance
Operationally decisive for us.
- The fusion algorithm is a **linear scan with hard breaks**: 「SuperKernel融合会按照网络中算子的顺序依次判断是否可被
  融合。当识别到不可融合的算子时，系统会生成第一段SuperKernel，并自动跳过该算子进行第二段SuperKernel融合」. With 3008 kernels
  per round, the practical question is not "does it fuse" but "how many breaks", and `strict-scope-check=abort` /
  `bypass` plus the plog keyword `super_kernel_scope` is the way to enumerate them. By default a non-AICore task
  (e.g. an AICPU operator) **breaks** fusion; `task_breaker_bypass = 1` relaxes that.
- `early_start = 1` is the switch that activates the `SetNextTaskStart` / `WaitPreTaskEnd` pair, and the stated
  benefit is literally 「算子头部Scalar指令提前执行」 — early execution of the operator's head Scalar instructions. That
  is the same "operator-head overhead" the Matmul static-tiling case study in section 12 measured at ~4 us per call.
- `auto_op_parallel = 1` reorders multi-stream operators by Cube/Vector type inside the SuperKernel; `stream-fusion`
  in GE mode notes the benefit is 「减少各流上算子的调度开销，而非增加并行度」 and that Cube+Vector cross-stream fusion
  usually pays best.
- `feed-sync-all = 1` auto-pads `SyncAll` on the extra cores when a sub-kernel's blockDim is smaller than the
  SuperKernel's — the fix for the hang described in section 19's 全核同步约束.
- Two costs to plan for: SuperKernel **disables operator Data Dump**, and SuperKernel options participate in the
  static-kernel compilation cache key, so mismatched options across models in one process can silently reuse the
  wrong binary.
- Collective ops that can be fused are exactly `AllReduce, ReduceScatter, AllGather, AlltoAll` — relevant to our
  TP16 decode round, where the per-layer allreduce would otherwise break every fusion segment.
- Debug levers worth knowing: `debug_op_exec_trace` logs SK-entry/sub-kernel start/finish per core to plog,
  `debug_cross_core_sync_check` validates mode-2 Cube/Vector sync counts, and `debug_per_op_max_core_num` isolates
  each operator into its own SuperKernel at full core count.

---

## 23. `TPipe` / `TQueBind` / `TQue` / `TBuf` — the declared API surface, verbatim from the header

https://gitcode.com/cann/asc-devkit/blob/master/include/basic_api/kernel_tpipe.h
raw: https://raw.gitcode.com/cann/asc-devkit/raw/master/include/basic_api/kernel_tpipe.h
Repo `cann/asc-devkit`, branch `master`.

```cpp
template <TPosition src, TPosition dst, int32_t depth, auto mask = 0>
class TQueBind {
public:
    __aicore__ inline TQueBind();
    __aicore__ inline void FreeBuffer(TBufHandle buf);
    __aicore__ inline TBuffAddr GetBufferAddr(TBufHandle buf);
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    template <typename T>
    __aicore__ inline __sync_noalias__ LocalTensor<T> AllocTensor();
    template <typename T>
    __aicore__ inline __sync_noalias__ void AllocTensor(LocalTensor<T>& tensor);
#else
    template <typename T>
    __aicore__ inline __sync_alias__ LocalTensor<T> AllocTensor();
    template <typename T>
    __aicore__ inline __sync_alias__ void AllocTensor(LocalTensor<T>& tensor);
#endif
    template <typename T>
    __aicore__ inline void FreeTensor(LocalTensor<T>& tensor);
    template <typename T>
    __aicore__ inline bool EnQue(const LocalTensor<T>& tensor);
    __aicore__ inline bool EnQue(TBufHandle buf);
    template <TPosition srcUserPos, TPosition dstUserPos, typename T>
    __aicore__ inline bool EnQue(const LocalTensor<T>& tensor);
    template <typename T>
    __aicore__ inline void DeQue(LocalTensor<T>& tensor);
    template <typename T>
    __aicore__ inline LocalTensor<T> DeQue();
    __aicore__ inline TBufHandle DeQue();
    template <TPosition srcUserPos, TPosition dstUserPos, typename T>
    __aicore__ inline LocalTensor<T> DeQue();
    __aicore__ inline bool VacantInQue();
    __aicore__ inline bool HasTensorInQue();
    __aicore__ inline int32_t GetTensorCountInQue();
    __aicore__ inline bool HasIdleBuffer();
    __aicore__ inline void FreeAllEvent();
    template <typename T>
    __aicore__ inline TBufState GetState(const LocalTensor<T>& tensor) const;
    __aicore__ inline void InitStartBufHandle(TBufHandle startBufhandle, uint8_t num, uint32_t len);
    template <typename T>
    __aicore__ inline void InitBufHandle(
        T* bufPool, uint32_t index, TBufHandle bufhandle, uint32_t curPoolAddr, uint32_t len);

protected:
    static constexpr TQueConfig config = GetTQueConfig(mask);
    static constexpr bool nd2nz = config.nd2nz;
    static constexpr bool nz2nd = config.nz2nd;
    static constexpr bool scmBlockGroup = config.scmBlockGroup;
    static constexpr bool enableLoopQueue = config.enableLoopQueue;
```

```cpp
template <TPosition pos, int32_t depth, auto mask = 0>
class TQue : public TQueBind<GetBufferLogicPos(pos, true), GetBufferLogicPos(pos, false), depth, mask> {
public:
    __aicore__ inline TQue() = default;
...
};

template <TPosition pos = TPosition::LCM>
class TBuf : public TQueBind<pos, pos, 0, 0> {
public:
    __aicore__ inline TBuf() = default;
    template <typename T>
    __aicore__ inline LocalTensor<T> Get();
    template <typename T>
    __aicore__ inline LocalTensor<T> Get(uint32_t len);
    template <typename T>
    __aicore__ inline LocalTensor<T> GetWithOffset(uint32_t size, uint32_t bufOffset);
    // inheritance function from Class TQueBind
    template <typename T>
    __aicore__ inline void EnQue(const LocalTensor<T>& tensor);
    template <typename T>
    __aicore__ inline LocalTensor<T> DeQue();
    template <typename T>
    __aicore__ inline LocalTensor<T> AllocTensor();
    template <typename T>
    __aicore__ inline void FreeTensor(LocalTensor<T>& tensor);
    template <typename T>
    __aicore__ inline TBufState GetState(const LocalTensor<T>& tensor) const;
    __aicore__ inline bool EnQue(TBufHandle buf);
    __aicore__ inline TBufHandle DeQue();
    __aicore__ inline void FreeBuffer(TBufHandle buf);
    __aicore__ inline TBuffAddr GetBufferAddr(TBufHandle buf);
    __aicore__ inline void InitStartBufHandle(TBufHandle startBufhandle, uint8_t num, uint32_t len);
```

```cpp
class TPipe : public TPipeBase {
public:
#if defined(__NPU_ARCH__)
    __aicore__ inline TPipe();
    __aicore__ inline ~TPipe();
#else
    __aicore__ inline TPipe() {}
    __aicore__ inline ~TPipe() {}
#endif
    __aicore__ inline void Init();
    template <class T>
    __aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len);
    template <class T, class U, class V, class... Addrs>
    __aicore__ inline bool InitBuffer(T& que, const Std::tuple<U, V>& addr0, const Addrs&... addrs);
    template <TPosition pos>
    __aicore__ inline bool InitBuffer(TBuf<pos>& buf, uint32_t len);
    template <class T>
    __aicore__ inline bool InitBufPool(T& bufPool, uint32_t len);
    template <class T, class U>
    __aicore__ inline bool InitBufPool(T& bufPool, uint32_t len, U& shareBuf);
    template <HardEvent evt>
    __aicore__ inline TEventID AllocEventID();
    template <HardEvent evt>
    __aicore__ inline void ReleaseEventID(TEventID id);
    template <HardEvent evt>
    __aicore__ inline TEventID FetchEventID();
    __aicore__ inline TEventID FetchEventID(HardEvent evt);
    // NOTICE: GetAbsAddr has been deprecated and will be removed in the next version. Please do not use it!
    template <TPosition pos, typename T>
    __aicore__ inline LocalTensor<T> GetAbsAddr(int32_t offset, int32_t size) const;
    // NOTICE: GetAbsAddr has been deprecated and will be removed in the next version. Please do not use it!
    template <TPosition pos>
    __aicore__ inline TBuffAddr GetAbsAddr(int32_t offset, int32_t len) const;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 5102 || __NPU_ARCH__ == 5162 || \
                              __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    __aicore__ inline MutexID AllocMutexID();
    __aicore__ inline void ReleaseMutexID(MutexID id);
#endif
    /*
     * brief: these functions are used to use spm buffer;
     * demo case:
     * GlobalTensor<T> workTensor;
     * tpipe.InitSpmBuffer(workTensor, size);
     * LocalTensor<T> calcTensor = tpip.Get<T>(size);
     * // when local buffer is not enough, spill local to spm buffer;
     * tpipe.WriteSpmBuffer(calcTensor, size);
     * // ...
     * // read buffer from spm buffer into local
     * tpipe.ReadSpmBuffer(calcTensor, size);
     */
    template <typename T>
    __aicore__ inline void InitSpmBuffer(const GlobalTensor<T>& workspace, const int32_t bufferSize);
    __aicore__ inline void InitSpmBuffer(const int32_t bufferSize);
    template <typename T>
    __aicore__ inline void WriteSpmBuffer(
        const LocalTensor<T>& writeBuffer, const DataCopyParams& copyParams, int32_t writeOffset = 0);
    template <typename T>
    __aicore__ inline void ReadSpmBuffer(
        const LocalTensor<T>& readBuffer, const DataCopyParams& copyParams, int32_t readOffset = 0);
    template <typename T>
    __aicore__ inline void WriteSpmBuffer(
        const LocalTensor<T>& writeBuffer, const int32_t writeSize, int32_t writeOffset = 0);
    template <typename T>
    __aicore__ inline void ReadSpmBuffer(
        const LocalTensor<T>& readBuffer, const int32_t readSize, int32_t readOffset = 0);
    __aicore__ inline void DestroyWithoutPipeAll();
    __aicore__ inline void Destroy();
    __aicore__ inline void Reset();
```

## Relevance
Three things we need are declared here and nowhere in the prose docs.
(1) `DestroyWithoutPipeAll()` is a real `TPipe` member — the SuperKernel rule in section 21 is satisfiable.
(2) `InitBuffer(T& que, const Std::tuple<U, V>& addr0, const Addrs&... addrs)` is the variadic explicit-address
overload that lets a megakernel hand-place each stage's UB region instead of bump-allocating, and `Reset()` is the
documented way to re-partition UB between stages.
(3) `AllocEventID<HardEvent>` / `ReleaseEventID<HardEvent>` / `FetchEventID<HardEvent>` are the sanctioned way to
obtain intra-core event IDs (section 8 warns IDs are finite and that 6 and 7 are reserved).
Also note the `__NPU_ARCH__` guard: `AllocMutexID` / `ReleaseMutexID` exist only on 3510/5102/5162/9201/9202 — **not**
on our 2201 — and `AllocTensor` is `__sync_alias__` on 2201 versus `__sync_noalias__` on 3510, i.e. the compiler
assumes aliasing between queue buffers on A3 and inserts conservative sync that it skips on A5.
`InitSpmBuffer` / `WriteSpmBuffer` / `ReadSpmBuffer` give a documented UB-spill-to-GM mechanism ("when local buffer is
not enough, spill local to spm buffer") that a multi-stage megakernel can use when 192 KiB of UB is not enough.

---

## 24. `Mmad` / `MmadParams` and the `GET_TILING_DATA` family

### `Mmad` — the raw cube instruction wrapper

https://github.com/hicann/asc-devkit/blob/master/include/basic_api/kernel_operator_mm_intf.h
raw: https://raw.githubusercontent.com/hicann/asc-devkit/master/include/basic_api/kernel_operator_mm_intf.h

```cpp
/* **************************************************************************************************
 * Mmad                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Mmad
 * @brief Matrix multiplication and addition
 * @param [out] dst output LocalTensor
 * @param [in] fm input LocalTensor
 * @param [in] filter input LocalTensor
 * @param [in] mmadParams.m Left matrix row number
 * @param [in] mmadParams.n right matrix column number
 * @param [in] mmadParams.k Left matrix column number m
 * @param [in] mmadParams.unitFlag whether enable unit flag
 * @param [in] mmadParams.kDirectionAlign is the indicator for alignment in L0A/L0B in the K direction
 * @param [in] mmadParams.cmatrixSource indicates the C matrix source, 1: the C matrix is in bias table buffer, 0: the C
 * matrix is in L0C
 * @param [in] mmadParams.cmatrixInitVal indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real
 * number in C matrix
 */

template <typename T, typename U, typename S>
__aicore__ inline void Mmad(
    const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, const MmadParams& mmadParams);

template <typename T, typename U, typename S, typename V>
__aicore__ inline void Mmad(
    const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, const LocalTensor<V>& bias,
    const MmadParams& mmadParams);

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
template <typename T, typename U, typename S>
__aicore__ inline void Mmad(
    const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter,
    const MmadBitModeParams& mmadParams);

template <typename T, typename U, typename S, typename V>
__aicore__ inline void Mmad(
    const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, const LocalTensor<V>& bias,
    const MmadBitModeParams& mmadParams);
#endif
```

Also declared in the same header: `MmadMx` ("Matrix multiplication and addition in microscaling scenario") and
`MmadWithSparse`:

```cpp
template <
    typename T = int32_t, typename U = int8_t,
    typename Std::enable_if<Std::is_same<PrimT<T>, int32_t>::value, bool>::type = true,
    typename Std::enable_if<Std::is_same<PrimT<U>, int8_t>::value, bool>::type = true>
__aicore__ inline void MmadWithSparse(
    const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<U>& filter, const MmadParams& mmadParams);
```

### `MmadParams`

https://github.com/hicann/asc-devkit/blob/master/include/basic_api/kernel_struct_mm.h
raw: https://raw.githubusercontent.com/hicann/asc-devkit/master/include/basic_api/kernel_struct_mm.h

```cpp
struct MmadParams {
    __aicore__ MmadParams() {}

    __aicore__ MmadParams(
        const uint16_t mIn, const uint16_t nIn, const uint16_t kIn, const bool isBiasIn, const int32_t fmOffsetIn,
        const bool enSsparseIn, const bool enWinogradAIn, const bool enWinogradBIn)
        : m(mIn), n(nIn), k(kIn), isBias(isBiasIn), fmOffset(fmOffsetIn), enSsparse(enSsparseIn),
          enWinogradA(enWinogradAIn), enWinogradB(enWinogradBIn)
    {}

    __aicore__ MmadParams(
        const uint16_t mIn, const uint16_t nIn, const uint16_t kIn, const uint8_t unitFlagIn,
        const bool cmatrixSourceIn, const bool cmatrixInitValIn)
        : m(mIn), n(nIn), k(kIn), unitFlag(unitFlagIn), cmatrixSource(cmatrixSourceIn), cmatrixInitVal(cmatrixInitValIn)
    {}

    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;
    // Indicates whether to accumulate the initial matrix, 0: matrix multiplication, 1: matrix multiplication and
    // addition
    bool isBias = false;
    // Left matrix offset
    int32_t fmOffset = 0;
    // Enable the structured sparse feature, default value is false
    bool enSsparse = false;
    // Indicates whether matrix a is generated by winograd_feature_map_transform, default value is false;
    bool enWinogradA = false;
    // Indicates whether matrix b is generated by winograd_feature_map_transform, default value is false;
    bool enWinogradB = false;
    uint8_t unitFlag = 0;
    // also mean gemvCtrl in 3101 and 5102
    bool kDirectionAlign = false;
    // Indicates the C matrix source, 1: the C matrix is in bias table buffer, 0: the C matrix is in L0C
    bool cmatrixSource = false;
    // Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    bool cmatrixInitVal = true;
#if defined(__NPU_ARCH__) && \
    ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    bool disableGemv = false;
#endif
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    bool hif4Amode = false;
    bool hif4Bmode = false;
    bool cScaleCtrl = false;
#endif
#ifdef CANN_ASC_ENABLE_FIX_SHIFT
    uint8_t fixShiftVal = 0;
#endif
};
```

### `GET_TILING_DATA_WITH_STRUCT`

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0215.html

> 功能说明
>
> 使用该接口指定结构体名称，可获取指定的tiling信息，并填入对应的Tiling结构体中，此函数会以宏展开的方式进行编译。与GET_TILING_DATA的区别是：GET_TILING_DATA只能获取默认注册的结构体，该接口可以根据指定的结构体名称获取对应的结构体，常用于针对不同的TilingKey注册了不同结构体的情况下。
>
> 函数原型
>
> ```
> GET_TILING_DATA_WITH_STRUCT(struct_name, tiling_data, tiling_arg)
> ```
>
> | 参数 | 输入/输出 | 说明 |
> |---|---|---|
> | struct_name | 输入 | 指定的结构体名称。 |
> | tiling_data | 输出 | 返回指定Tiling结构体变量。 |
> | tiling_arg | 输入 | 此参数为算子入口函数处传入的tiling参数。 |
>
> 约束说明
>
> 本函数需在算子Kernel代码处使用，并且传入的tiling_data参数不需要声明类型。
>
> 暂不支持Kernel直调工程。

```cpp
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
{
    KernelAdd op;
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(Add_Struct_Special, tilingData, tiling); // 使用算子指定注册的结构体
        op.Init(x, y, z, tilingData.totalLengthSpecial, tilingData.tileNumSpecial);
    } else {
        GET_TILING_DATA(tilingData, tiling);   // 使用算子默认注册的结构体
        op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    }
    if (TILING_KEY_IS(1)) {
        op.Process();
```

### `GET_TILING_DATA_MEMBER`

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0216.html

> 功能说明: 用于获取tiling结构体的成员变量。
>
> ```
> GET_TILING_DATA_MEMBER(struct_name, mem_name, tiling_data, tiling_arg)
> ```
>
> | 参数 | 输入/输出 | 说明 |
> |---|---|---|
> | struct_name | 输入 | 指定的结构体名称。 |
> | mem_name | 输入 | 指定的成员变量名称。 |
> | tiling_data | 输出 | 返回指定Tiling结构体的成员变量。 |
> | tiling_arg | 输入 | 此参数为算子入口函数处传入的tiling参数。 |
>
> 约束说明
>
> 本函数需在算子kernel代码处使用，并且传入的tiling_data参数不需要声明类型。
>
> 暂不支持Kernel直调工程。

Both macros: 产品支持情况 lists `Atlas A3 训练系列产品/Atlas A3 推理系列产品 √`, `Atlas A2 ... √`,
`Atlas 200I/500 A2 推理产品 √`, `Atlas 推理系列产品AI Core √`, `Atlas 推理系列产品Vector Core √`, `Atlas 训练系列产品 x`.

## Relevance
`MmadParams.unitFlag` is the field behind `mmCFG.enUnitFlag = true` from section 15 (512-byte MMAD/FixPipe overlap),
and `cmatrixInitVal` is what lets consecutive `Mmad` calls accumulate into the same L0C tile instead of zeroing it —
the low-level mechanism under the `TPosition::CO1` accumulation pattern in section 11. `isBias` toggles
multiply vs multiply-accumulate; `cmatrixSource` selects bias-table vs L0C as the C source. `MmadWithSparse`
(int8 only, `int32_t` dst) and `enSsparse` expose structured sparsity. Note `disableGemv` and the `hif4*` / `cScaleCtrl`
fields are gated off 2201, so they are unavailable to us.
Both `GET_TILING_DATA_WITH_STRUCT` and `GET_TILING_DATA_MEMBER` are explicitly **暂不支持Kernel直调工程** — not
supported in kernel-direct-launch projects. Since the `__sk__`/`SK_BIND` SuperKernel path in section 21 *is* the
kernel-direct-launch path, a megakernel on that route cannot use the tiling-struct macros and must pass its
parameters through the `SK_BIND` args struct instead. That is consistent with section 12's 全量常量化 advice: make
the tiling compile-time constant and pass a null tiling pointer.

---

## 25. `__schedmode__` / batch scheduling in a production kernel, with measured hang rate

### `hicann/ops-nn` `rnn/lstm_block_cell/op_kernel/arch35/sync_helper.h`

https://github.com/hicann/ops-nn/blob/master/rnn/lstm_block_cell/op_kernel/arch35/sync_helper.h
raw: https://raw.githubusercontent.com/hicann/ops-nn/master/rnn/lstm_block_cell/op_kernel/arch35/sync_helper.h
Repo `hicann/ops-nn`, branch `master`. Copyright (c) 2026 Huawei Technologies Co., Ltd.
Header comment says "Generated By CANNBot". **This file targets Ascend 950PR (dav_3510), not A3** — the mode-4
specifics below do not apply on 2201 (see section 20), but the `__schedmode__` discussion and the pairing discipline do.

```cpp
/* sync_helper.h -- ALL synchronization for the raw Ascend 950PR (dav_3510) kernels, in one place.
 *
 * WHY THIS IS ITS OWN HEADER, against the "split by who issues it" rule the other headers follow:
 *
 *   A HANDSHAKE IS A PROTOCOL, AND A PROTOCOL HAS TWO ENDS. The thing that goes wrong is not one
 *   side or the other -- it is the PAIRING. A dedicated cross-core measurement pins the pairing rule, because the three
 * plausible pairings do not all work (1 set : 1 consume deadlocks, 2 : 2 deadlocks, 2 : 1 runs). Putting the two ends
 * in two different files would separate exactly the thing that must be read together.
 *
 * So cube_helper.h and vec_helper.h are pure dataflow, and every ordering question lives here.
 *
 * TWO KINDS OF ORDERING, DO NOT CONFUSE THEM
 *   intra-core   between PIPES of one core (MTE2 / MTE1 / M / FIX / V / MTE3). Set+Wait on a
 *                typed HardEvent. Cheap, and purely local.
 *   cross-core   between the AIC and its two AIVs. CrossCoreSetFlag / CrossCoreWaitFlag.
 *                Semantics differ BY DIRECTION -- see below, this is the part that bites.
 */
```

The `__schedmode__` comment block, verbatim — **with a measured hang rate**:

```cpp
/* ---------------------------------------------------------------------------------------------
 * BATCH SCHEDULING (`__schedmode__` / `SetScheduleMode`) -- WHAT IT IS, AND WHY IT MATTERS
 *
 * A cross-core wait only holds if the AIC and its two AIVs are dispatched as a BATCH:
 * without batch scheduling the wait can be posted with nobody to answer it. A reference
 * recurrent-operator probe measured 3 hangs in 49 runs (~6%) without it, 0 in 240 with --
 * concentrated on SMALL shapes, where both AIVs reach the barrier at nearly the same moment.
 *
 * TWO SEPARATE LEVERS, AND ONLY ONE OF THEM EXISTS ON EACH PATH:
 *   op framework   `gert::TilingContext::SetScheduleMode(uint32_t)` -- an OUTPUT SLOT of
 *                  tiling, so it only exists when tiling runs. This kernel's TilingFunc
 *                  sets it (batch scheduling: AIC + 2 AIV dispatched together).
 *   raw <<<>>>     `__schedmode__(N)` -- there is no TilingContext on a raw launch, so
 *                  this attribute is the only lever there.
 * ------------------------------------------------------------------------------------------- */
```

Intra-core helpers, verbatim:

```cpp
/* ---------------------------------------------------------------------------------------------
 * Intra-core: order one pipe against another, within a single core.
 *
 * Each is a Set immediately followed by a Wait on the same event id -- a full stop rather than
 * true overlap. That is deliberate: correctness first, and the pipe pair stays legible at the
 * call site. Software-pipelining these (Set early, Wait late, multiple event ids) is a later
 * optimisation, not a cleanup.
 * ------------------------------------------------------------------------------------------- */
template <AscendC::HardEvent E>
__aicore__ inline void PipeWait(event_t id = EVENT_ID0)
{
    AscendC::SetFlag<E>(id);
    AscendC::WaitFlag<E>(id);
}

__aicore__ inline void WaitMte2ToMte1() { PipeWait<AscendC::HardEvent::MTE2_MTE1>(); } // GM->L1 before L1->L0
__aicore__ inline void WaitMte1ToM() { PipeWait<AscendC::HardEvent::MTE1_M>(); }       // L1->L0/BT before Mmad
__aicore__ inline void WaitMToFix() { PipeWait<AscendC::HardEvent::M_FIX>(); }         // Mmad before the drain
__aicore__ inline void WaitFixToM() { PipeWait<AscendC::HardEvent::FIX_M>(); }         // drain read before reusing L0C
```

Cross-core discipline, verbatim (dav_3510 / mode 4):

```cpp
/* ---------------------------------------------------------------------------------------------
 * Cross-core. Both ends of each protocol are adjacent ON PURPOSE.
 *
 * (!) MODE 4 (intra-block) IS USED, NOT MODE 0x2. Mode 0x2 leaks across MIX blocks within
 * a group at blockDim > 1, and the "barrier, not counting semaphore" pairing (both AIVs
 * set, cube waits x1) breaks: a flag set by one block's AIV is consumed by another
 * block's AIC, deadlocking the producer. Mode 4 (`set_intra_block` /
 * `wait_intra_block`) is strictly intra-MIX-block (1 AIC + 2 AIV) and is what every
 * vendor MIX-1:2 op uses (quant_batch_matmul_v3, mat_mul_v3 arch35).
 *
 * Mode-4 pairing rules (measured by vendor + confirmed by dav_c310 impl):
 *   set_intra_block from AIC  -> flag visible to BOTH AIVs (subBlock-scoped: AIV0 sees
 *                               flagId, AIV1 sees flagId + INTRA_NUM=16).
 *   set_intra_block from AIV  -> flag visible to AIC only (subBlock-scoped: AIC distinguishes
 *                               AIV0 by flagId and AIV1 by flagId + INTRA_NUM).
 *   wait_intra_block from AIC -> must wait for BOTH AIVs: flagId AND flagId + INTRA_NUM.
 *   wait_intra_block from AIV -> waits for its own flagId (hardware routes by subBlockId).
 *
 * So the AIC sets/waits TWO flags (one per AIV slot), each AIV sets/waits ONE.
 *
 * (!) MODE-4 FLAGS ARE COUNTERS, NOT ONE-SHOT BARRIERS. A set landing while an earlier
 * set on the same id is still unwaited ACCUMULATES -- it is neither lost nor
 * latched-as-edge (proven on real 3510 by count==0 fast paths pre-firing several
 * consumed slots ahead of their waits). The counter reading does NOT relax the
 * cross-launch pairing discipline: a set left unwaited at launch end stays armed
 * into the NEXT launch and makes its wait return early -- every protocol below
 * keeps one set + one wait per flag slot per launch.
 * ------------------------------------------------------------------------------------------- */
constexpr uint16_t INTRA_NUM = 16; // dav_c310 subBlock flag offset (AIV0: 0..15, AIV1: 16..31)
```

```cpp
constexpr uint16_t V2C_PREPASS = 7;

__aicore__ inline void VecSignalPrepass() { AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(V2C_PREPASS); }
__aicore__ inline void CubeWaitPrepass()
{
    /* (!) PIPE_MTE2, NOT PIPE_M. This wait guards the
     * bias workspace reads, which are MTE2 copies (GM->L1 CopyInFlat /
     * CopyInNd2Nz inside CubeProjectKS). On PIPE_M it stalled the M pipe and
     * left the MTE2 reads ungated -- the AIC could read the fp32 bias
     * workspace before the AIV pre-pass wrote it. Vendor pattern: wait on the
     * consuming pipe (same rule as every other cross-core wait here). */
    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(V2C_PREPASS);
    AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(V2C_PREPASS + INTRA_NUM);
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

```cpp
/* ---------------------------------------------------------------------------------------------
 * K-SPLIT PHASE HANDOFF -- the K-split two-phase structure:
 * the AICs write their partials straight to GM (L0C->GM fixpipe), then every
 * AIV waits for its OWN AIC and only then joins the grid-wide
 * AscendC::SyncAll(). The ledger is one set + one wait per flag slot per
 * launch -- no ring, no reuse.
 *
 * (!) WHY THE AIV WAITS ON PIPE_MTE3: SyncAll() on 3510 is an AIV-only ffts
 * barrier whose ARRIVE instruction rides the MTE3 queue (dav_3510
 * kernel_operator_sync_impl.h: ffts_cross_core_sync(PIPE_MTE3, ...)). Enqueuing
 * this wait on the SAME queue makes the arrive queue-ordered behind the
 * own-AIC flag, so the barrier cannot pass while any AIC is still draining
 * (ops-nn kernel_matmul_streamk.h: CrossCoreWaitFlag<MODE_4, PIPE_MTE3> then
 * SyncAll -- the exact vendor form). After SyncAll() returns, EVERY cluster's
 * partials are GM-visible to every AIV.
 * ------------------------------------------------------------------------------------------- */
constexpr uint16_t KS_DONE = 8; // distinct from V2C_PREPASS; 16 logical ids total

/* AIC: my LAST partial drain is on the FIX queue -- this set rides PIPE_FIX,
 * so it is delivered only after every GM write of the phase is complete. */
__aicore__ inline void CubeSignalKsDone()
{
    AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(KS_DONE);
    AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(KS_DONE + INTRA_NUM);
}
/* AIV: my AIC's partials are in GM. Call BEFORE AscendC::SyncAll(). */
__aicore__ inline void VecWaitKsDone() { AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE3>(KS_DONE); }
```

## Relevance
First, the number we could not find in the docs: without batch scheduling, a MIX cross-core wait **hangs ~6% of the
time** (3/49 runs), concentrated on small shapes where both AIVs hit the barrier simultaneously, and 0/240 with it.
At batch=1 decode our shapes are the small ones, so `__schedmode__(N)` on a `<<<>>>` megakernel is mandatory, not
advisory. The two levers are mutually exclusive by path: `gert::TilingContext::SetScheduleMode(uint32_t)` only exists
when tiling runs; `__schedmode__(N)` is the only lever for a raw `<<<>>>` launch.
Second, the pairing rule discovered empirically — 「1 set : 1 consume deadlocks, 2 : 2 deadlocks, 2 : 1 runs」 — and the
named failure mode of mode 0x2 at `blockDim > 1`: 「Mode 0x2 leaks across MIX blocks within a group... a flag set by
one block's AIV is consumed by another block's AIC, deadlocking the producer」. On 3510 the fix is mode 4. **On our
A3 (2201) mode 4 does not exist** (section 4: 「模式4仅支持Atlas 350 加速卡」; section 20: 「该模式仅在NPU架构版本3510上
支持」), so a multi-block MIX megakernel on A3 must either keep `blockDim == 1` per group for mode-2 handshakes or
route the AIC<->AIV handoff through GM workspace with mode-0 grid barriers instead. This is a concrete A3 hazard
we should design around rather than discover.
Third, the pipe-selection rule: 「wait on the consuming pipe」 — a cross-core wait placed on `PIPE_M` left MTE2 reads
ungated. (On 2201 the `pipe` template argument of `CrossCoreWaitFlag` is ignored and all pipes block, which makes this
class of bug impossible but every wait more expensive — see section 20.)
Fourth, a set left unwaited at the end of one launch **stays armed into the next launch** and makes its wait return
early. For a persistent/looping megakernel that is the dominant correctness hazard: one set + one wait per flag slot
per iteration, no exceptions.

---

## 26. Static kernel compilation and its cache key

https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/basic/static_kernel_compile.md
raw: https://raw.gitcode.com/Ascend/torchair/raw/master/docs/zh/npugraph_ex/basic/static_kernel_compile.md

> # 静态Kernel编译功能
>
> ## 功能简介
>
> 对于纯静态shape网络或者shape变化较少的动态shape网络，如需提升网络执行性能，可通过算子预先静态编译达到目的，该方式简称为静态Kernel编译。它是指在模型编译时指定shape大小，运行时不需要指定shape大小，减少运行时开销，具体优势如下：
>
> - 编译时已知所有Tensor的大小，存储空间利用率高。
> - 编译时可以针对实际的shape大小做针对性优化。
> - AI处理器擅长并行指令运行，不擅长逻辑计算，如果有太多的Scalar操作可能会打断并行指令的运行，从而导致性能下降。静态编译可以在编译时完成标量的计算，一定程度上可以提升性能。
> - 算子编译工具（op_compiler）在编译时知道确切的操作数据大小，不会额外插入同步，不会导致并行执行多个指令变成串行执行，一定程度上可以提升性能。
>
> 开启静态Kernel编译后，系统根据输入的算子信息统计文件，得到确定的shape信息，针对每一个shape都编译出一个算子二进制。
>
> ## 使用约束
>
> - 需注意的是，当Ascend PyTorch Profiler中**"experimental_config"参数**开启算子信息统计功能（即record_op_args=True），且"**schedule**"**参数**设置的预先跳过的step轮数为0时（即skip_first=0），不支持同时使用本功能。
>
> - 多卡场景使用限制：必须配置环境变量**LOCAL_WORLD_SIZE**，其每个节点的LOCAL_WORLD_SIZE必须一致。
>
>     > [!NOTE]说明
>     >- 由于多卡场景在安装静态kernel run包时，会更新算子库公共文件，若没有环境变量LOCAL_WORLD_SIZE的协同，可能会读取到过程态的内容，从而导致未定义行为（undefined behavior）。因此，在多卡场景使用该功能时，必须配置LOCAL_WORLD_SIZE。

```python
import torch
import torch_npu

options = {"static_kernel_compile": True, "disable_static_kernel_compile_cache": False}

opt_model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=False, fullgraph=True)
```

> |**参数名**|**参数说明**|
> |--|--|
> |static_kernel_compile|布尔类型，是否开启静态Kernel编译。False（默认值）：默认关闭。True：开启静态Kernel编译。编译产物路径默认为当前执行脚本的同级目录，当同时启用了模型编译缓存功能但未启用force_eager功能时，编译产物路径为模型缓存文件所在目录。请确保该路径用户具有读、写操作权限。|
> |disable_static_kernel_compile_cache|布尔类型，是否禁用静态Kernel缓存功能。<br>False（默认）：使用缓存功能。二次运行且缓存命中时，将直接复用编译好的kernel run包。<br>True：禁用缓存功能。每次运行都会触发静态Kernel编译过程。<br>仅在 `static_kernel_compile` 为 `True` 时生效。缓存映射关系记录于静态编译产物路径下的 `static_kernel_cache` 目录中。|

> > [!NOTE]说明
> > 在使用静态Kernel缓存功能时，系统会对比当前运行特征与缓存记录。只有当以下所有条件全部满足且缓存中对应的kernel run包存在时，才会判定为缓存命中，跳过静态Kernel编译过程：
> >
> > 1. CANN版本不变、硬件型号不变；
> > 2. 模型中所有参与静态Kernel编译的算子属性（包括算子类型、输入输出Shape、Format、数据类型等）保持一致;
> > 3. 静态编译时的特定选项保持一致，包括：确定性计算配置、`super_kernel_optimize` 配置（含 `super_kernel_optimize_options` 和 `super_kernel_debug_options`）。
> >
> > 若上述任一条件不匹配，则判定缓存未命中，将触发完整的静态Kernel编译流程，并更新 `static_kernel_cache` 目录下对应的缓存记录。

## Relevance
Static kernel compilation is the prerequisite for SuperKernel (section 7's `Super Kernel -> static kernel -> Npugraph_ex`),
and the stated mechanism is the same one section 12 measured: 「静态编译可以在编译时完成标量的计算」 — resolve Scalar work
at compile time, plus 「不会额外插入同步」. At batch=1, seq 8K, TP16 decode our shapes are fixed, which is precisely the
「纯静态shape网络」 case. Two operational facts: the multi-card requirement `LOCAL_WORLD_SIZE` must be set and identical
per node (it appears in the GLM-5.2 A3 config in section 7 as `LOCAL_WORLD_SIZE: "16"`), and the cache key includes
the SuperKernel option dicts — so flipping any SuperKernel option forces a full recompile, which the vendor warns
takes 「several minutes to tens of minutes」.

---

## 27. CV fusion rationale and the logical-position data flow (`CO2 -> VECIN`, `VECOUT -> A1 -> A2`)

### CANN 9.0 算子开发指南 — CV融合 / 基础知识

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0049.html
Publisher: Huawei / Ascend, parent topic 「CV融合」.

> ## CV融合算子
>
> 融合算子由多个独立的小算子融合而成，其功能与多个小算子的功能等价，性能方面通常优于独立的小算子。用户可以根据实际业务场景诉求，按照具体算法自由融合向量（Vector）、矩阵（Cube）算子以达到性能上的收益。融合了Cube计算、Vector计算的算子统称为CV融合算子。
>
> 比如对于LLM大模型中最核心的一个融合算子Flash Attention， 其核心实现如下图。图中的Matmul算子（Cube）、Scale算子（Vector）、Mask算子（Vector）、SoftMax算子（Vector）融合为一个大的算子Flash Attention。
>
> ## 使用场景和优势
>
> 针对有数据依赖的矢量算子和矩阵算子，可以通过融合算子编程的方式，将矢量算子和矩阵算子进行融合，通过一个算子Kernel函数来承载，由此来获得性能上的收益。下图展示了独立矢量算子和矩阵算子、Mix融合算子的执行耗时对比，由此可以看出为什么开发Mix融合算子会带来性能上的收益。
>
> 独立的矢量算子和矩阵算子实现：矩阵计算后的结果需要搬运到Global Memory上，然后由Global Memory搬运到Local Memory，再进行矢量算子的计算，计算和搬运都是串行执行，另外多个算子的调度执行，会增加算子的调度耗时。
>
> 融合算子的实现方法：可以对数据进行切片，再通过流水的设计，使得矢量计算单元和矩阵计算单元实现并行计算；另外相比于不融合的单算子，减少了算子的调度耗时。
>
> 除了有效提升算子性能，充分发挥AI处理器的算力，融合算子还有如下优势：
>
> 减少计算量：融合算子可以将多个算子合并为一个，简化计算过程，减少计算量，提高计算效率。
>
> 减少内存占用：融合算子可以将多个算子的中间结果合并为一个，从而减少内存占用，提高内存利用率。
>
> 优化数据流：融合算子可以优化数据流，减少数据在不同算子之间的传输，从而提高数据处理效率。
>
> 简化代码实现：融合算子可以简化代码实现，减少代码量，提高代码可读性和可维护性。
>
> ## 编程范式
>
> Ascend C提供融合算子的编程范式，方便开发者基于该范式表达融合算子的数据流，快速实现自定义的融合算子。
>
> 融合算子数据流指融合算子的输入输出在各存储位置间的流向。以一个典型的Cube和Vector融合算子为例，逻辑位置间的数据流向如下图所示：
>
> Cube的输出可以作为Vector的输入：CO2->VECIN
>
> Vector的输出可以作为Cube的输入：VECOUT->A1->A2、VECOUT->B1->B2
>
> 基于Matmul高阶API的融合算子编程范式，对上述数据流简化表达如下：
>
> 初始化一个Matmul对象。
>
> 进行Matmul内部的计算。
>
> 将Matmul的计算结果搬运到Vector核上。
>
> 进行Vector矢量计算。
>
> 将输出结果搬运到Global Memory上。

```cpp
template<typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Process()
{
    uint32_t computeRound = 0;
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    while (matmulObj.template Iterate<true>()) { // 步骤2：进行Matmul内部的计算。
        // 步骤3：将Matmul的计算结果搬运到Vector核上。
        reluOutLocal = reluOutQueue_.AllocTensor<cType>();
        matmulObj.template GetTensorC<true>(reluOutLocal, false, true);
       // 步骤4：进行Vector矢量计算。
        AscendC::LeakyRelu(reluOutLocal, reluOutLocal, (cType)alpha, tiling.baseM * tiling.baseN);
        reluOutQueue_.EnQue(reluOutLocal);
        // 步骤5：将输出结果搬运到Global Memory上
        reluOutQueue_.DeQue<cType>();
        ...
        AscendC::DataCopy(cGlobal[startOffset], reluOutLocal, copyParam);
        reluOutQueue_.FreeTensor(reluOutLocal);
        computeRound++;
    }
    matmulObj.End();
}
// kernel入口函数，__mix__(1, 2)表示：mix场景，AIC:AIV=1:2。
__global__ __mix__(1, 2) void matmul_leakyrelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                      __kfc_workspace__ GM_ADDR workspace, AscendC::tiling::TCubeTiling tiling)
{
    AscendC::TPipe pipe;
    MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
    matmulLeakyKernel.Init(a, b, bias, c, workspace, tiling, &pipe);
    // 步骤1：初始化Matmul对象。
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj, &matmulLeakyKernel.tiling);
    matmulLeakyKernel.Process(&pipe);
}
```

### `KERNEL_TASK_TYPE_DEFAULT` in a shipping attention kernel (arch-conditional)

https://github.com/hicann/ops-transformer/blob/master/attention/compressor/op_kernel/compressor.cpp
raw: https://raw.githubusercontent.com/hicann/ops-transformer/master/attention/compressor/op_kernel/compressor.cpp
Repo `hicann/ops-transformer`, branch `master`.

```cpp
                                      __gm__ uint8_t* cmpKvOut, __gm__ uint8_t* stateCacheOut,
                                      __gm__ uint8_t* softmaxScoreOut, __gm__ uint8_t* kvOut, __gm__ uint8_t* workspace,
                                      __gm__ uint8_t* tiling)
{
    REGISTER_TILING_DEFAULT(optiling::CompressorTilingData);
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201))
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    GET_TILING_DATA_WITH_STRUCT(optiling::CompressorTilingData, tilingDataIn, tiling);
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::EMPTY_X) {
        return;
    }
    const optiling::CompressorTilingData* __restrict tilingData = &tilingDataIn;
    TPipe pipe;
```

Other shipping kernels found by code search that call `KERNEL_TASK_TYPE_DEFAULT` / `KERNEL_TASK_TYPE`
(all in `hicann/ops-transformer`, `master`, under `op_kernel/`):
`attention/quant_compressor/op_kernel/quant_compressor.cpp`,
`attention/lightning_indexer_v2/op_kernel/lightning_indexer_v2.cpp`,
`mhc/mhc_pre_sinkhorn/op_kernel/mhc_pre_sinkhorn_apt.cpp`,
`mhc/mhc_pre_sinkhorn_backward/op_kernel/mhc_pre_sinkhorn_backward_apt.cpp`,
`mc2/allto_all_matmul_v2/op_kernel/arch35/allto_all_matmul_v2_apt.cpp`,
`mc2/all_gather_matmul_v3/op_kernel/arch35/all_gather_matmul_v3_apt.cpp`.
Also `sgl-project/sgl-kernel-npu` `csrc/lora/op_kernel/sgemmc_shrink_kernel.cpp` and
`csrc/lora/op_kernel/sgemmc_expand_kernel.cpp`, `xLLM-AI/xllm-ops` `xllm_ops/index_group_matmul/op_kernel/index_group_matmul.cpp`,
and `tile-ai/tilelang-ascend` `src/target/codegen_ascend.cc` (which emits the macro).

## Relevance
The vendor's own stated reason for CV fusion is our reason, in two parts: 「矩阵计算后的结果需要搬运到Global Memory上，
然后由Global Memory搬运到Local Memory，再进行矢量算子的计算，计算和搬运都是串行执行，另外多个算子的调度执行，会增加算子的
调度耗时」 — the serialization *and* the per-operator dispatch cost. Flash Attention is given as the canonical example
(Matmul + Scale + Mask + SoftMax in one kernel), which is the shape of our attention block.
The logical-position data flow is stated as `CO2 -> VECIN` for cube-to-vector and `VECOUT -> A1 -> A2` /
`VECOUT -> B1 -> B2` for vector-to-cube — note this is the *logical* position flow the Matmul high-level API
presents, while the *physical* path on A3 still passes through GM (sections 8 and 16). `VECOUT -> A1 -> A2` is
why `MatmulType<TPosition::VECOUT, ...>` exists for the A and B matrices (section 11): a vector stage's UB output
can feed the next matmul's L1/L0A without the kernel re-reading GM — which is the direction that *is* available
on A3 and is the one worth exploiting for a vector->cube handoff in a megakernel.
The production attention kernel shows the real-world pattern we should copy: select the kernel task type by
`__NPU_ARCH__` at compile time (`MIX_AIC_1_1` on 9201, `MIX_AIC_1_2` otherwise, so A3/2201 takes `MIX_AIC_1_2`).

---

## 28. Measured SuperKernel A/B on A2/A3: 5.61% end-to-end, 21% on the fused segment

https://gitcode.com/cann/graph-autofusion/blob/master/super_kernel/examples/super_kernel_profiling/README.md
raw: https://raw.gitcode.com/cann/graph-autofusion/raw/master/super_kernel/examples/jit/example02_super_kernel_profiling/README.md
(English copy: https://gitcode.com/cann/graph-autofusion/blob/master/super_kernel/examples/super_kernel_profiling/README_en.md)
Repo `cann/graph-autofusion`, branch `master`. Sample script `superkernel_compare.py`.

> # super_kernel 用例演示
>
> ## 用例功能
>
> 模型包含6个sk片段，部分sk相同复用缓存，部分多了可输入bias或option配置不同走在线编译
>
> 编译使用super_kernel和不使用super_kernel的模型，将性能数据输出
>
> ## 融合算子
>
> 使用如下with语句块（super_kernel），语句块内算子均被融合为一个超级Kernel进行计算
> ```python
> with torchair.scope.super_kernel("sk1"):
> ```
>
> ## 环境依赖
>
> 支持如下产品型号：
>
> - Atlas A3 训练系列产品/Atlas A3 推理系列产品
> - Atlas A2 训练系列产品/Atlas A2 推理系列产品

> ## 预期执行结果
>
> 在执行目录生成prof_result文件夹，目录如下，获取数据后对比耗时
> ```text
> prof_result
> ├── sk_model                             # 带superkernel结果
> │  ├── localhost.localdomain_ascend_pt
> │     ├── PROF_*
> │        ├── mindstudio_profiler_output
> │           ├── op_statistic.csv         # profiling数据
> ├── no_sk_model                          # 不带superkernel结果
> │  ├── localhost.localdomain_ascend_pt
> │     ├── PROF_*
> │        ├── mindstudio_profiler_output
> │           ├── op_statistic.csv         # profiling数据
> ```
> 分别从两份op_statistic.csv表中得到如下数据
>
> | OP_Type | Core Type | Total Time(us) |
> |--|--|--|
> | GroupedMatmul | MIX_AIC | 126.26 |
> | Transpose | AI_VECTOR_CORE | 90.02 |
> | MoeGatingTopK | AI_VECTOR_CORE |  68.32|
> |Tile  |  AI_VECTOR_CORE| 24.96 |
> | DequantSwigluQuant |AI_VECTOR_CORE  | 24.18 |
> | ReduceMeanD |  MIX_AIV| 16.36 |
> | ConcatV2D|  AI_VECTOR_CORE| 14.9 |
> | ReduceMeanD | AI_VECTOR_CORE | 14.14 |
> | SplitVD |AI_VECTOR_CORE  | 10.04 |
> | MatMul |AI_CORE  | 6.26 |
> | Cast |  AI_VECTOR_CORE|3.96 |
> | Data |  AI_VECTOR_CORE| 3.3 |
> | StridedSliceD | AI_VECTOR_CORE |3.18  |
> | AutomaticBufferFusionOp |AI_VECTOR_CORE  |1.66  |
> |no_sk_model|总耗时|407.54|
>
> | OP_Type | Core Type | Total Time(us) |
> |--|--|--|
> |SuperKernel | MIX_AIC| 172.4|
> |Transpose |AI_VECTOR_CORE | 92.42|
> |Tile |AI_VECTOR_CORE |24.66 |
> |SuperKernel |MIX_AIV | 18.48|
> |ReduceMeanD |MIX_AIV | 16.34|
> |ConcatV2D |AI_VECTOR_CORE |14.74 |
> |ReduceMeanD | AI_VECTOR_CORE| 14.24|
> |SplitVD |AI_VECTOR_CORE |10.12 |
> |MatMul |AI_CORE | 8.6|
> |Cast |AI_VECTOR_CORE |4.08 |
> |Data |AI_VECTOR_CORE | 3.72|
> |StridedSliceD |AI_VECTOR_CORE |3.1 |
> |AutomaticBufferFusionOp |AI_VECTOR_CORE | 1.76|
> |sk_model|总耗时|384.66|
>
> 对比获得使用super_kernel融合算子的收益是5.61%

## Relevance
The only measured SuperKernel A/B we found, and it is on A2/A3-supported hardware with an MoE-shaped graph
(`GroupedMatmul`, `MoeGatingTopK`, `DequantSwigluQuant`) close to our W4A8 MoE decode. Reading the two tables:
the three fused operators `GroupedMatmul` (MIX_AIC 126.26) + `MoeGatingTopK` (AI_VECTOR_CORE 68.32) +
`DequantSwigluQuant` (AI_VECTOR_CORE 24.18) = 218.76 us collapse into one `SuperKernel` MIX_AIC entry of 172.4 us —
a **21% reduction on the fused segment**. End-to-end it is 407.54 -> 384.66 us, **5.61%**, because most of the
remaining time (`Transpose` 90.02, `Tile` 24.96, `ConcatV2D` 14.9, ...) stayed unfused. Two readings matter for us:
(a) the per-segment win is real and sizeable where fusion actually lands; (b) the end-to-end win is gated entirely by
coverage, and in this sample coverage was poor — `Transpose` alone is 22% of the graph and did not fuse. Our 3008
kernels per round would need broad coverage, not just the MIX_AIC stages, for SuperKernel alone to approach our
11.6 -> 2.x ms target. Note also that SuperKernel produced **two** entries, `SuperKernel MIX_AIC` and
`SuperKernel MIX_AIV` — the fusion segmented by core type, consistent with section 6's 同步优化.
Lastly, the measurement recipe is reusable verbatim: run with and without, read `op_statistic.csv` under
`mindstudio_profiler_output`, compare `Total Time(us)` by `OP_Type` / `Core Type`.

---

## 29. `MatmulCallBackFunc` — user callbacks for CopyA1 / CopyB1 / DataCopyOut, with quant fields

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0618.html
Parent topic 「Matmul Kernel侧接口」.

产品支持情况: `Atlas 350 加速卡 √ / Atlas A3 训练系列产品/Atlas A3 推理系列产品 √ / Atlas A2 训练系列产品/Atlas A2 推理系列产品 √ / Atlas 200I/500 A2 推理产品 x / Atlas 推理系列产品AI Core x / Atlas 推理系列产品Vector Core x / Atlas 训练系列产品 x`

功能说明, verbatim:

> 模板参数MatmulCallBackFunc支持用户定制化Matmul的A矩阵、B矩阵及C矩阵的搬入搬出功能，如非连续搬入或针对搬出设置不同的数据片段间隔等。具体方式为：用户根据实际需要，实现一个或多个自定义的搬运函数，定义Matmul对象时，通过模板参数MatmulCallBackFunc，传入实现的搬运函数的函数指针，传入的函数指针会替换Matmul流程中默认的搬运函数。
>
> MatmulCallBackFunc中包含3个可由用户自定义的回调函数接口，即用户可配置3个函数指针。3个函数指针分别为C矩阵从CO1拷贝到GM、A矩阵从GM拷贝到A1、B矩阵从GM拷贝到B1的回调函数指针。3个函数指针的位置固定，不使用自定义搬运函数的函数指针位置需要设为空指针。每个回调函数实现矩阵搬运中单个基本块（A矩阵基本块baseM * baseK、B矩阵基本块baseK * baseN、C矩阵基本块baseM * baseN）的搬运策略，无法对整块内存空间进行管理。Matmul默认的搬运函数实现单核上单个基本块的搬运，搬运的基本块大小是固定的，在完整的Matmul计算过程中，多次调用搬运函数，对连续排布的基本块按顺序逐个搬运

表1 MatmulCallBackFunc回调函数接口及参数说明, verbatim:

> | 回调函数功能 | 回调函数接口 | 参数说明 |
> |---|---|---|
> | 可自定义设置不同的搬出数据片段数目等参数，实现将Matmul计算结果从CO1搬出到GM的功能 | `void DataCopyOut(const __gm__ void *gm, const LocalTensor<int8_t> &co1Local, const void *dataCopyOutParams, const uint64_t tilingPtr, const uint64_t dataPtr)` | gm：输出的GM地址。<br>co1Local: CO1上的计算结果。<br>dataCopyOutParams：Matmul定义的DataCopyOutParams结构体指针。<br>tilingPtr: 用户使用SetUserDefInfo设置的tiling参数地址。<br>dataPtr: 用户使用SetSelfDefineData设置的计算数据地址。 |
> | 可自定义左矩阵搬入首地址、搬运块位置、搬运块大小，实现左矩阵从GM搬入L1的功能 | `void CopyA1(const LocalTensor<int8_t> &aMatrix, const __gm__ void *gm, int row, int col, int useM, int useK, const uint64_t tilingPtr, const uint64_t dataPtr)` | aMatrix: 目标L1Buffer地址。<br>gm：左矩阵GM首地址。<br>row、col：搬运块在M、K方向的索引，即在M、K方向上搬运块的序号，序号从0开始。<br>useM、useK：搬运块M、K方向大小，单位为元素个数。通过row、col和useM、useK计算出该搬运块左上角在左矩阵中的地址偏移。 |
> | 可自定义右矩阵搬入首地址、搬运块位置、搬运块大小，实现右矩阵从GM搬入L1的功能 | `void CopyB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col, int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)` | bMatrix: 目标L1Buffer地址。<br>gm：右矩阵GM首地址。<br>row、col：搬运块在K、N方向的索引。<br>useK、useN：搬运块K、N方向大小，单位为元素个数。 |

```cpp
struct DataCopyOutParams {
    uint16_t cBurstNum; //传输数据片段数目
    uint16_t burstLen; //连续传输数据片段长度
    uint16_t srcStride;//源tensor相邻连续数据片段间隔
    uint32_t dstStride; // 目的tensor相邻连续数据片段间隔
    uint16_t oriNSize; // NZ转ND时，源tensorN方向大小
    bool enUnitFlag; // 是否使能UnitFlag
    uint64_t quantScalar; // 量化场景下量化Scalar的值
    uint64_t cbufWorkspaceAddr; //量化场景下量化Tensor地址
}
```

调用示例, verbatim:

```cpp
//用户自定义回调函数
void DataCopyOut(const __gm__ void *gm, const LocalTensor<int8_t> &co1Local, const void *dataCopyOutParams, const uint64_t tilingPtr, const uint64_t dataPtr);
void CopyA1(const LocalTensor<int8_t> &aMatrix, const __gm__ void *gm, int row, int col, int useM, int useK, const uint64_t tilingPtr, const uint64_t dataPtr);
void CopyB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col, int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr);
AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM, MatmulCallBackFunc<DataCopyOut, CopyA1, CopyB1>> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
uint64_t tilingPtr = reinterpret_cast<uint64_t>(tiling);
mm.SetUserDefInfo(tilingPtr); // 设置算子tiling地址，用于回调函数使用
```

> 完整的使用样例请参考Matmul回调函数样例。

## Relevance
Supported on A3, and the cleanest documented hook for a W4A8 epilogue: `DataCopyOut` receives the raw
`LocalTensor<int8_t> co1Local` (the L0C accumulator) plus a `DataCopyOutParams` that already carries
`quantScalar` ("量化场景下量化Scalar的值") and `cbufWorkspaceAddr` ("量化场景下量化Tensor地址") and `enUnitFlag`.
That means the requantization of the int32 cube result can be expressed inside the Matmul high-level API's own
drain path, on the AIC side via Fixpipe, rather than as a separate AIV kernel — which attacks our MIX call count
directly. `CopyA1` / `CopyB1` receive `(row, col, useM, useK)` block indices and the GM base, so a custom loader can
pull non-contiguous or differently-packed weights (relevant for int4 packing) without a pre-pass kernel.
Hard limit stated by the doc: each callback handles exactly one base block (`baseM*baseK`, `baseK*baseN`,
`baseM*baseN`) and 「无法对整块内存空间进行管理」 — no whole-tensor orchestration. `SetUserDefInfo(tilingPtr)` and
`SetSelfDefineData` are how per-call context reaches the callbacks.

---

## 30. `BareMix` — the complete low-level MIX fusion kernel, start to finish

The sample referenced by section 5's 「完整样例请参考BareMix样例」. Located in community mirrors of the Ascend samples tree:

https://github.com/XJ19023/ascend_samples/blob/master/operator/ascendc/0_introduction/22_baremix_kernellaunch/BareMixInvocation/baremix_custom.cpp
raw: https://raw.githubusercontent.com/XJ19023/ascend_samples/master/operator/ascendc/0_introduction/22_baremix_kernellaunch/BareMixInvocation/baremix_custom.cpp
(same file also at `c303803919-jpg/transmit` under `samples-master/operator/ascendc/0_introduction/22_baremix_kernellaunch/`)
Header: `Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.`

Verbatim, the whole kernel:

```cpp
/**
 * @file baremix_custom.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

/**
  * @brief  Copy tiling data to TCubeTiling ptr from tiling gm addr.
  */
__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

template <typename aType, typename bType, typename cType, typename biasType> class MatmulLeakyKernel {
public:
    __aicore__ inline MatmulLeakyKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                const TCubeTiling &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process(AscendC::TPipe *pipe);

    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC, int32_t &offsetBias);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, cType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    AscendC::GlobalTensor<biasType> biasGlobal;
    TCubeTiling tiling;
};

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias,
                                                                              GM_ADDR c, GM_ADDR workspace,
                                                                              const TCubeTiling &tiling, AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);

    int32_t offsetA, offsetB, offsetC, offsetBias;
    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias); // Calculate the gm offset based on the blockidx.
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Process(AscendC::TPipe *pipe)
{
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);

    matmulObj.template IterateAll(cGlobal);
    matmulObj.End();
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(3);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                                             int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
                                                             int32_t &offsetBias)
{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;
}

template <typename cType> class LeakyReluKernel {
public:
    __aicore__ inline LeakyReluKernel(){};
    __aicore__ inline void Init(GM_ADDR c, const TCubeTiling &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process(AscendC::TPipe *pipe);

    __aicore__ inline void LeakyReluCopyIn(const TCubeTiling &tiling);
    __aicore__ inline void LeakyReluCompute(const TCubeTiling &tiling);
    __aicore__ inline void LeakyReluCopyOut(const TCubeTiling &tiling);

    AscendC::GlobalTensor<cType> cGlobal;

    AscendC::LocalTensor<cType> reluInLocal;
    AscendC::LocalTensor<cType> reluOutLocal;
    TCubeTiling tiling;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> reluInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> reluOutQueue_;
};

template <typename cType>
__aicore__ inline void LeakyReluKernel<cType>::Init(GM_ADDR c, const TCubeTiling &tiling, AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c) + AscendC::GetBlockIdx() * tiling.M * tiling.N / 2); //c:v = 1:2, split into 2 parts, for vector calculation

    pipe->InitBuffer(reluInQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(cType) /2); // Init input buffer.
    pipe->InitBuffer(reluOutQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(cType)/2); // Init output buffer.
}

template <typename cType>
__aicore__ inline void LeakyReluKernel<cType>::Process(AscendC::TPipe *pipe)
{
    AscendC::CrossCoreWaitFlag(3);
    LeakyReluCopyIn(tiling);
    LeakyReluCompute(tiling);
    LeakyReluCopyOut(tiling);
}
template <typename cType>
__aicore__ inline void LeakyReluKernel<cType>::LeakyReluCopyIn(const TCubeTiling &tiling)
{
    AscendC::LocalTensor<float> reluInLocal = reluInQueue_.AllocTensor<float>();
    AscendC::DataCopy(reluInLocal, cGlobal, tiling.singleCoreM * tiling.singleCoreN / 2);
    reluInQueue_.EnQue<float>(reluInLocal);
}

template <typename cType>
__aicore__ inline void LeakyReluKernel<cType>::LeakyReluCompute(const TCubeTiling &tiling)
{
    AscendC::LocalTensor<float> reluInLocal = reluInQueue_.DeQue<float>();
    AscendC::LocalTensor<float> reluOutLocal = reluOutQueue_.AllocTensor<float>();
    AscendC::LeakyRelu(reluOutLocal, reluInLocal, (float)0.001, tiling.singleCoreM * tiling.singleCoreN /2);
    reluOutQueue_.EnQue<float>(reluOutLocal);
    reluInQueue_.FreeTensor(reluInLocal);
}

template <typename cType>
__aicore__ inline void LeakyReluKernel<cType>::LeakyReluCopyOut(const TCubeTiling &tiling)
{
    AscendC::LocalTensor<float> reluOutLocal = reluOutQueue_.DeQue<float>();
    AscendC::DataCopy(cGlobal, reluOutLocal, tiling.singleCoreM * tiling.singleCoreN / 2);
    reluOutQueue_.FreeTensor(reluOutLocal);
}

/**
  * @brief  baremix kernel function entry
  */
extern "C" __global__ __aicore__ void baremix_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                              GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    if ASCEND_IS_AIC {
        MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
        matmulLeakyKernel.Init(a, b, bias, c, workspace, tiling, &pipe);
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj, &matmulLeakyKernel.tiling); // Initialize the matmul object.
        matmulLeakyKernel.Process(&pipe);
    }
    if ASCEND_IS_AIV {
        LeakyReluKernel<float> leakyReluKernel;
        leakyReluKernel.Init(c, tiling, &pipe);
        leakyReluKernel.Process(&pipe);
    }
}
```

## Relevance
This is the smallest complete, compilable template for the hand-synchronized MIX kernel our megakernel will be —
every element section 5 described in prose, in one file:
- `#define ASCENDC_CUBE_ONLY` **before** `#include "lib/matmul_intf.h"`, so the Matmul high-level API compiles
  AIC-only and does not install the AIV-side KFC controller.
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` at the top of the kernel body, plus
  `if ASCEND_IS_AIC { ... }` / `if ASCEND_IS_AIV { ... }` as the two halves. Note the entry is a plain
  `extern "C" __global__ __aicore__` function — **not** `__global__ __mix__(1,2)` with `__kfc_workspace__`; the kernel
  type macro does the work instead, and `GM_ADDR tilingGm` is copied in by hand (`CopyTiling`) rather than via
  `GET_TILING_DATA` (which section 24 notes is unsupported in kernel-direct-launch projects).
- Exactly one handshake: `AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(3)` after `matmulObj.End()` on the AIC side, and
  a bare `AscendC::CrossCoreWaitFlag(3)` (default `modeId = 0`... note the asymmetry: the Set is mode 2, the Wait
  uses the default template argument) at the top of the AIV `Process`. flagId 3 is inside the user range 0-10 but
  **collides with `Matmul`'s `[0, 2N-1]` allocation** for N >= 2 (section 20) — a live hazard if a megakernel defines
  more than one Matmul object.
- The AIV side reads the cube result back from `cGlobal` in GM (`DataCopy(reluInLocal, cGlobal, ...)`), confirming
  section 16's conclusion: on this architecture the cube->vector handoff is a GM round trip, L2-absorbed.
- `cGlobal` is offset by `GetBlockIdx() * tiling.M * tiling.N / 2` with the comment `//c:v = 1:2, split into 2 parts,
  for vector calculation` — the AIV-side blockIdx-based work split under `MIX_AIC_1_2`, matching section 16's
  `GetBlockIdx() / 2` logic from the other direction.
- `MatmulType<TPosition::VECIN, CubeFormat::ND, cType>` for the C matrix even though `IterateAll(cGlobal)` writes to
  GM — the logical position, per section 27.

---

---

## 31. The A3 cube→vector GM visibility rule: `CrossCoreSetFlag` is NOT enough, `DataCacheCleanAndInvalid` + `DataSyncBarrier<DDR>` on **both** sides is

This section answers the concrete failure "AIC writes INT32 to a GM `partial` buffer, `PipeBarrier<PIPE_ALL>()`,
`CrossCoreSetFlag<0x02, PIPE_FIX>(8)`; AIV does `CrossCoreWaitFlag(8)`, `PipeBarrier<PIPE_ALL>()`, `DataCopy` — and
reads zeros non-deterministically, while the same buffer downloaded to the host is always correct."

### 31.1 The rule, stated by the people who hit the identical bug

https://github.com/flashserve/flash-linear-attention-npu/blob/main/fla/ops/ascendc/cp/pre_process_fwd_kernel_merged/op_kernel/arch22/pre_process_fwd_kernel_merged_cube.h
Repo `flashserve/flash-linear-attention-npu`, branch `main`, file
`fla/ops/ascendc/cp/pre_process_fwd_kernel_merged/op_kernel/arch22/pre_process_fwd_kernel_merged_cube.h`,
`arch22` = our DAV_2201. Retrieved 2026-10-20.

This is the **producer (AIC) side**, verbatim, including its comment. `vTmpF_` and `t1F_` are `GlobalTensor<float>`
holding the cube's `C` output; `AicSetToAiv` is the `CrossCoreSetFlag` wrapper:

```cpp
        // 注意： 写侧也要 clean（写回），只靠读者 DCCI 不够：FIX
        // 写回可能还停在写缓冲里，
        //   此时 AIV 即便 DCCI 也会读到旧值。实测（PPFM_DIAG 指纹）：AIV 读到 vTmp 全 0，
        //   而 AIC 实际写了 -0.013/+0.0092 → v_new 退化成 v，h 半边随机整头崩（m
        // 不受影响）。
#if !PPFM_T1_FIXPIPE_BF16
#if !PPFM_VTMP_UB
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(vTmpF_);
#endif
#endif
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(t1F_);
#endif
#if !PPFM_T1_FIXPIPE_BF16
        // 注意： 正式原语：DDR 数据同步屏障——保证 C 的写回对其他核可见后再抬 flag。
        //   诊断版实验表明竞态是"flag 已到、写回仍在途"的时序窗口（加探针即掩盖）。
#if PPFM_LEGACY_CACHEOPS
        DataSyncBarrier<MemDsbT::DDR>();
#endif
        AicSetToAiv(kFlagHalf1);
#endif
```

Translated for the record, because this is the rule: *the write side must clean too — the reader's DCCI alone is not
enough: the FIX write-back may still be sitting in the write buffer, in which case the AIV reads the old value even
after a DCCI. Measured (PPFM_DIAG fingerprint): the AIV read vTmp as all zeros while the AIC had actually written
-0.013/+0.0092.* And: *the proper primitive is the DDR data-sync barrier — it guarantees C's write-back is visible to
other cores before raising the flag. The diagnostic build showed the race is the timing window "flag already arrived,
write-back still in flight" (adding a probe masks it).*

"AIV 读到 vTmp 全 0" is our symptom exactly, and "加探针即掩盖" is why our host download of `partial` is always correct.

The same file, the **consumer-of-the-other-core's-GM** direction (here AIC reading what AIV wrote — the mirror image,
same rule):

```cpp
        AicWaitFromAiv(kFlagInputs);
        // 注意： 读别的核（AIV）刚写过的 GM 之前必须让本核的 cache 失效：h/m 每 chunk 都被
        //    AIV 重写，若 AIC 命中自己缓存的旧行，mm1/mm3 就会拿到过期的 h/m
        //    （实测表现为概率性的 h 半边大面积错、m 只是略偏，且随调度时好时坏）。
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(wBf_);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(hBf_);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(mBf_);
#endif
```

### 31.2 The AIV side of the same kernel — our exact case, cube product read back by vector

https://github.com/flashserve/flash-linear-attention-npu/blob/main/fla/ops/ascendc/cp/pre_process_fwd_kernel_merged/op_kernel/arch22/pre_process_fwd_kernel_merged_vector.h
Same repo/branch, `arch22` vector half. Retrieved 2026-10-20.

Verbatim — note the cited precedent "CANN `matmul_client.h` does a DCCI before reading a cross-core GM flag", and the
hard-won detail that **every** alias of the buffer must be invalidated, not just the one the even iteration uses:

```cpp
            AivWaitFromAic(kFlagDH);
            // 读别的核（AIC）写的 GM 前必须让本核缓存行失效，否则会读到过期数据
            // （与 CANN matmul_client.h 中"读跨核 GM flag 前先 DCCI"的用法一致）
            // 注意： 原来只失效了**偶数奇偶**的 dHF_/t2F_，而下面按 chunk 奇偶读的是
            //   `dhBuf`/`t2Buf`（两份）⇒ 奇数 chunk 读的 `dHF1_`/`t2F1_` 从来没被失效过，
            //   命中旧行就会把过期 T2 减进 m（表现为 m 半边小范围错、h 正常）。
            //   这个缺口在 A2 上（PPFM_LEGACY_CACHEOPS=1）才暴露得出来；补齐两份。
#if PPFM_KEEP_CACHEOPS_R
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(dHF_);
#endif
#if PPFM_KEEP_CACHEOPS_R
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(dHF1_);
#endif
#if PPFM_KEEP_CACHEOPS_R
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(t2F_);
#endif
#if PPFM_KEEP_CACHEOPS_R
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(t2F1_);
#endif
```

And the AIV reading the cube's `vTmpF_` / `t1F_`, i.e. literally "vector reads the cube's INT32/FP32 product out of
GM" — the invalidate sits immediately before the consuming vector code:

```cpp
#if !PPFM_VTMP_UB
#if PPFM_KEEP_CACHEOPS_R
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(vTmpF_);
#endif
#endif
#if PPFM_KEEP_CACHEOPS_R
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(t1F_);
#endif
        // v_new = (v - vTmp) · dg → bf16（逐行；整块版本会引入 ~0.4% 的 GDN 偏差，待查）
```

### 31.3 The flag wrappers — where the cache ops go relative to the flag, and the `pipe` that matters

https://github.com/flashserve/flash-linear-attention-npu/blob/main/fla/ops/ascendc/cp/pre_process_fwd_kernel_merged/op_kernel/pre_process_fwd_kernel_merged_common.h
Same repo/branch. Retrieved 2026-10-20. All four wrappers, verbatim and complete:

```cpp
// ---------------- CrossCore flag（MIX 内 AIC <-> 2×AIV）----------------
// 两个平台的同步模型不同（见 PPFM_XCORE_MODE / PPFM_ARCH_IS_950）：
//   * 950 用 mode 0x4：flag ID 按 subblock 分槽（第二个子核 = id + 16），AIC 侧显式
//     wait/set 两个 slot。与仓内 arch35 算子的约定一致
//     （见 chunk_fwd_h/op_kernel/chunk_fwd_h_policy.h、kda/chunk_kda_fwd/.../fwd_h.h）。
//     实测反例：用 A2/A3 风格的 `CrossCoreSetFlag<0x2, ...>` 时，950 上先干完的那个
//     子核就会把 AIC 放行，AIC 的 mm1/mm3 读到"只写了一半"的 h/m bf16 状态，
//     表现为 h 半边约一半行数据错、m 基本对（GDN 快路径暴露，KDA 慢路径看不出来）。
//   * 910B/910_93 用 mode 0x2：AIC 与「本 block 的 2 个 AIV」是集合同步 —— AIC 的一次
//     set 对本 block 两个 AIV 同时置起；两个 AIV 都 set 同一个 ID 才算 AIC 侧事件置起。
//     因此 AIC 侧每轮只需一对 set/wait（多 set/多 wait 会让 flag 计数失衡）。
constexpr uint16_t PPFM_SUBFLAG_STRIDE = 16;   // AIV 子核 1 的 slot 偏移
constexpr uint16_t kFlagInputs = 1;   // AIV -> AIC：本 chunk staging 就位
constexpr uint16_t kFlagState = 2;    // AIV -> AIC：h/m 状态就位
constexpr uint16_t kFlagHalf1 = 3;    // AIC -> AIV：vTmp 与 T1 就位
constexpr uint16_t kFlagVNew = 4;     // AIV -> AIC：bf16(v_new) 与 bf16(T1) 就位
constexpr uint16_t kFlagDH = 5;       // AIC -> AIV：dH 与 T2 就位
// （dH 走 L0C->UB）：dH 的 UB 槽「归还」信用（AIV -> AIC）。槽 j 用 id + j，
// AIC 侧按 subblock 各等一次（id 与 id+PPFM_SUBFLAG_STRIDE），
// 与仓内 arch35 的 MATRIX_CV_AIV_TO_AIC_FLAG_BEGIN/CV_SUBBLOCK_FLAG_STRIDE 同构。
constexpr uint16_t kFlagDhFree = 8;
// T2 的 UB 槽归还信用（AIV -> AIC），单槽。
constexpr uint16_t kFlagT2Free = 10;

// AIV 子核：写自己本地 slot（硬件按子核自动映射到 id / id+16）
__aicore__ inline void AivSetToAic(uint16_t id)
{
    // set_intra_block 不保证"之前的搬运已落地"，这里显式排空 MTE3
    PipeBarrier<PIPE_MTE3>();
    // 对称于 AIC 侧：本核写出的 GM（kBf_/wBf_/lBf_/vNewBf_/t1Bf_/状态）也要先对其他核可见
#if PPFM_LEGACY_CACHEOPS
    DataSyncBarrier<MemDsbT::DDR>();
#endif
    CrossCoreSetFlag<PPFM_XCORE_MODE, PIPE_MTE3>(id);
}

__aicore__ inline void AivWaitFromAic(uint16_t id)
{
    // 950：用 PIPE_S 排队 —— 等待指令卡住后续指令的发射（否则后面的 MTE2/V 会先跑）。
    // 910B/910_93：消费方是 MTE2（从 GM 回读 cube 的 C），按 A2 惯用法在 MTE2 上排队
    //   （见仓内 arch22 的 `CrossCoreWaitFlag<0x2, PIPE_MTE2>(..._READY_FLAG)`）。
    //   实测 A2 上用 PIPE_S 等待时，随后的 MTE2 会提前发射，读到"半新半旧"的 C：
    //   表现为 h 半边误差 ~1.5e-2 且逐次小幅跳动（v=0 的探针本应得到 h≡0）。
#if PPFM_ARCH_IS_950
    CrossCoreWaitFlag<PPFM_XCORE_MODE, PIPE_S>(id);
#else
    CrossCoreWaitFlag<PPFM_XCORE_MODE, PIPE_MTE2>(id);
#endif
}
```

and the AIC half (note `PipeBarrier<PIPE_FIX>()` before the set, and the "15 consecutive sets hangs" citation):

```cpp
//   950 (mode 0x4)：flag ID 按 subblock 分槽，两个 AIV 子核各自 set 自己的 ID ⇒ 这里要等两次
//   910B/910_93 (mode 0x2)：AIC 与「本 block 的 2 个 AIV」是**集合同步**——两个 AIV 必须
//     都 set 同一个 ID，事件才会对 AIC 置起 ⇒ 这里只等一次
//   （约定出处：仓内 chunk_fwd_h/op_kernel/chunk_fwd_h_policy.h 的 FwdHAicPeerFlag 注释）
__aicore__ inline void AicWaitFromAiv(uint16_t id)
{
#if PPFM_ARCH_IS_950
    CrossCoreWaitFlag<PPFM_XCORE_MODE, PIPE_S>(id);
    CrossCoreWaitFlag<0x4, PIPE_S>(static_cast<uint16_t>(id + PPFM_SUBFLAG_STRIDE));
#else
    // 910B：AIC 侧消费方同样是 MTE2（把 AIV 写好的 k/w/h/m/vNew 搬进 L1/L0）
    CrossCoreWaitFlag<PPFM_XCORE_MODE, PIPE_MTE2>(id);
#endif
}

// AIC -> AIV：
//   950：两个 slot 都要置位，否则只会唤醒一个子核；
//   910B/910_93：0x2 下一次 set 即对本 block 的两个 AIV 同时置起，只能 set 一次
//   （多 set 会让计数失衡 —— 单条 flag 连续 set 超过 15 次会挂死，见 catlass
// cross_core_sync.hpp）
__aicore__ inline void AicSetToAiv(uint16_t id)
{
    PipeBarrier<PIPE_FIX>();
    CrossCoreSetFlag<PPFM_XCORE_MODE, PIPE_FIX>(id);
#if PPFM_ARCH_IS_950
    CrossCoreSetFlag<PPFM_XCORE_MODE, PIPE_FIX>(static_cast<uint16_t>(id + PPFM_SUBFLAG_STRIDE));
#endif
}
```

https://github.com/flashserve/flash-linear-attention-npu/blob/main/fla/ops/ascendc/cp/pre_process_fwd_kernel_merged/op_kernel/pre_process_fwd_kernel_merged_policy.h
Same repo/branch. The mode and the per-arch default of the cache ops, verbatim:

```cpp
constexpr int32_t PPFM_XCORE_MODE = 0x4;
```
```cpp
constexpr int32_t PPFM_XCORE_MODE = 0x2;
```
```cpp
// 950：实测只用跨核 flag 就够（并且去掉 DCCI 后竞态由 3/6 降到 1/6），默认 0。
// 910B/910_93：实测 AIC 会读到 AIV 尚未对其他核可见的 bf16(h)（chunk0 的 h≡0 探针
//   仍得到 1.6e-2 的 h），故先按 A2 老做法启用 DCCI/DSB；若后续定位到更精确的边，
//   可只保留必要的那一条。
#ifndef PPFM_LEGACY_CACHEOPS
#if PPFM_ARCH_IS_950
#define PPFM_LEGACY_CACHEOPS 0
#else
#define PPFM_LEGACY_CACHEOPS 1
#endif
#endif

// ---- C3 实验（A2 专用；默认 = 现状，950 机器码按构造不变）----
// 动机：A2 的 DCCI/DSB 目前"不分生产方"地全开（AIC 每 chunk 10 次整缓存 DCCI + AIV 6 次
//   + 4 次 DSB），而 950 的实测记录是"去掉 DCCI 后竞态由 3/6 降到 1/6"。
//   把 A2 的缓存操作按**数据生产方**拆成两组，做 4 组 A/B（默认都是 1 = 现状）：
//     W 组 = AIC 抬 flag 前，对**自己算出的 C** 做的 DCCI/DSB（vTmp / t1Bf_ / dH / T2 及末尾 DSB）
//     R 组 = AIV 读 **AIC 算出的 C** 之前做的失效（dHF_/dHF1_/t2F_/t2F1_/vTmpF_/t1F_）
//   保留不动的是"消费 AIV 产物"的那一半（AIC 读 wBf_/kBf_/hBf_/mBf_/lBf_/vNewBf_ 前的 DCCI，
//   以及 AivSetToAic 的 DSB）—— 那一半有明确故障证据（AIC 读到尚未可见的 bf16(h)）。
//   实验矩阵：W1R1（基线）/ W0R1 / W1R0 / W0R0，判据 = PROC S≥40 的失败率 + L1 + perf。
#ifndef PPFM_A2_CACHEOPS_W
//   **结论（2026-10-08，234 上机）**：W0R1（=A2 默认）在 60 进程位级口径下残差 1/60，
//   基线 W1R1 是 9/60 ⇒ 去掉 W 组的 DCCI 没有让竞态变差，反而随 C7/C9 一起变好。
#define PPFM_A2_CACHEOPS_W (PPFM_ARCH_IS_950 ? 1 : 0)
#endif
#ifndef PPFM_A2_CACHEOPS_R
#define PPFM_A2_CACHEOPS_R 1
#endif
```

Read that last block carefully: their final A2/A3 default is **W0R1** — producer-side DCCI *off*, consumer-side DCCI
*on*, with `DataSyncBarrier<MemDsbT::DDR>()` kept on the producer before the flag
(`PPFM_KEEP_CACHEOPS_W` gates both the producer DCCI *and* its DSB, but the DSB inside `AivSetToAic` is gated by
`PPFM_LEGACY_CACHEOPS` alone, which stays 1 on A2/A3). Measured: W1R1 baseline 9/60 processes mismatched, W0R1 1/60.
So the producer-side *cache* op was not what paid; the producer-side *barrier* and the consumer-side *invalidate* did.

### 31.4 A second, independent kernel that hit the same thing — the V→C relay, with a 2000-trial A/B

https://github.com/flashserve/flash-linear-attention-npu/pull/860
Author `weinachuan` (Collaborator), opened 2026-10-10T09:33:09Z, title
「fix(kda): ChunkKdaFwdPrepare 的 V6→C7 relay 交接补保守可见性同步（Arch22，临时验证）」. Draft, 2 commits
(`1a606d6`, `af7a2af`). Retrieved 2026-10-20.

PR body, verbatim excerpt — this is the general statement of the rule, and it is attributed to a repo-wide
"established conclusion":

```
// AIV → AIC 跨核 GM 交接的保守同步档位（Arch22 现场 A/B 用）。
//
// V6 把 K_beta_g/V_beta 写进 workspace relay，随后由 AIC 的 C7 读回并计算
// W/U；这条交接当前只依赖 ready/free 跨核 flag。A2/A3 上"只靠跨核 flag 传递
// GM 数据"不足以稳定保证写侧已经对读侧可见（同 CP/PPFM 的既定结论），这里提供
// 一个可回退的加固开关：
//   0 = 与改动前逐字节一致（基线）；
//   1 = 只在 V6 → C7 这条交接上补"写侧 clean + 读侧失效 + DSB"；
//   2 = 再把 C2/C4 读入 AIV payload 的位置按同样方式加固（默认，现场验证档位 1 无效、
//       档位 2 消除漂移；见 PR 描述）。
// 只影响同步强度，不改变任何计算、地址、输出档位或 tiling 数据。
#ifndef CHUNK_KDA_FWD_PREPARE_RELAY_SYNC
#define CHUNK_KDA_FWD_PREPARE_RELAY_SYNC 2
#endif
```

The producer-side hunk from the same PR's diff (`.../arch22/chunk_kda_fwd_prepare_vec.h`), verbatim, showing the
order **DataCopy → DCCI → DSB → flag**:

```cpp
         AscendC::DataCopy(kBetaRelay, kBetaG, rhsRows * Shape::kHeadDim);
         AscendC::DataCopy(vBetaRelay, vBeta, rhsRows * Shape::kValueDim);
+#if CHUNK_KDA_FWD_PREPARE_RELAY_SYNC >= 1
+        // 写侧加固（保守）：这两份 relay 马上会被 AIC 的 C7 跨核读回。只靠
+        // ready 跨核 flag 时，MTE3 写缓冲/L2 里的数据可能还没对读侧可见，读侧
+        // 就会拿到 relay 的旧内容（现场表现为单个 work item 的 U 整块错、同一
+        // Stage 的 W 正常）。先把两份 relay clean 出去再放行下一步。
+        AscendC::DataCacheCleanAndInvalid<bfloat16_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
+                                          AscendC::DcciDst::CACHELINE_OUT>(kBetaRelay);
+        AscendC::DataCacheCleanAndInvalid<bfloat16_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
+                                          AscendC::DcciDst::CACHELINE_OUT>(vBetaRelay);
+        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
+#endif
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ioFree_[pair]);
```

and the consumer-side hunk (`.../arch22/chunk_kda_fwd_prepare_cube.h`), order **wait → DCCI → DSB → DataCopy**:

```cpp
+#if CHUNK_KDA_FWD_PREPARE_RELAY_SYNC >= 1
+            // 读侧加固（保守）：读 AIV 刚写过的 relay 之前先让本核缓存行失效，
+            // 否则 MTE2 可能命中旧行，把上一代（或空）的 K_beta_g/V_beta 读成
+            // 本代的 RHS（现场表现为同一 Stage 里 W 正常、U 整块错）。
+            AscendC::DataCacheCleanAndInvalid<bfloat16_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
+                                              AscendC::DcciDst::CACHELINE_OUT>(kBetaRelay);
+            AscendC::DataCacheCleanAndInvalid<bfloat16_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
+                                              AscendC::DcciDst::CACHELINE_OUT>(vBetaRelay);
+            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
+#endif
             AscendC::DataCopy(kBetaL1, kBetaRelay, rhsCopy);
             AscendC::DataCopy(vBetaL1, vBetaRelay, rhsCopy);
```

Measured A/B from the PR's commit `af7a2af` message and body: tier 0 (baseline) gave
「~19-22/2000 trial 出现跨 stream 不一致」, tier 1 (that one handoff only) 「仍然 ~19/2000，无改善」, tier 2 (also
hardening the two other read sites) 「漂移消失」. The diagnostic signature, verbatim:

```
现场反馈 KDA 前向在并发场景下 fwd / recompute 输出不一致。离线逐元素分析显示：差异集中在一个
`(value head, chunk)` work item 的 `u` 上，而同一 Stage 的 `w`、更早 Stage 的 `gk / Aqk / Akk / qg / kg`
逐位一致；`u` 的等效右操作数中位幅值比正常小约 4 个数量级、约 1.4% 元素为巨大脏值，指向"读到了尚未写入或已过期的
relay 缓冲"。
```

and the author's own verdict on the cost, verbatim:

```
- 代价是每个 work item 多两次 `ENTIRE_DATA_CACHE` 级 clean/invalidate 与 DSB，属于保守选择，**可能带来性能回退**。
  若现场验证有效，后续应换成 slot 级双缓冲 + 代际号（或 L1 中转）的结构性方案再收口，不建议以本 PR 形态合入。
```

The "tier 1 ineffective, tier 2 effective" result is the single most useful datum here: **it is not enough to harden
the one handoff you suspect.** Every GM buffer that one core writes and another reads needs the treatment, or the
drift survives.

### 31.5 `DataCacheCleanAndInvalid` — the official API page, exact spelling and parameters

https://www.hiascend.com/doc_center/source/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0177.html
CANN Commercial 8.0 API reference, 「缓存处理」. Retrieved 2026-10-20. Verbatim:

```
DataCacheCleanAndInvalid
功能说明
在AI Core内部，Scalar单元和DMA单元都可能对Global Memory进行访问。
图1 DataCache内存层次示意图
如上图所示：
DMA搬运单元读写Global Memory，数据通过DataCopy等接口在UB等Local Memory和Global Memory间交互，没有Cache一致性问题；
Scalar单元访问Global Memory，首先会访问每个核内的Data Cache，因此存在Data Cache与Global Memory的Cache一致性问题。
该接口用来刷新Cache，保证Cache的一致性，使用场景如下：
读取Global Memory的数据，但该数据可能在外部被其余核修改，此时需要使用DataCacheCleanAndInvalid接口，直接访问Global Memory，获取最新数据；
用户通过Scalar单元写Global Memory的数据，希望立刻写出，也需要使用DataCacheCleanAndInvalid接口。
函数原型
template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dstTensor)
template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const LocalTensor<T>& dstTensor)
该原型支持的硬件平台型号：
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas 200I/500 A2 推理产品
template <typename T, CacheLine entireType>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dstTensor)
该原型支持的硬件平台型号：
Atlas 推理系列产品AI Core
参数说明
dstTensor  输入  需要刷新Cache的Tensor。
entireType 输入  指令操作的模式：
SINGLE_CACHE_LINE：只刷新传入地址所在的cacheLine，注意如果该地址非64B对齐，只会操作传入地址到64B对齐的部分。
ENTIRE_DATA_CACHE：此时传入的地址无效，核内会刷新整个Data Cache，但是耗时较大，性能敏感的场景慎用。
dcciDst    输入  表示使用该接口来保证Data Cache与哪一种存储保持一致性，类型为DcciDst枚举类。
CACHELINE_ALL：与CACHELINE_OUT效果一致;
CACHELINE_UB：预留参数，暂未支持;
CACHELINE_OUT：表示通过该接口来保证Data Cache与Global Memory的一致性;
CACHELINE_ATOMIC：预留参数，暂未支持。
```

and its multi-core warning, verbatim:

```
// 示例3：SINGLE_CACHE_LINE 模式，假设mmAddr_为0x40（64B对齐），多核处理场景（本样例仅做示例说明，便于开发者理解使用限制，非正常使用样例）
AscendC::GlobalTensor<uint64_t> global;
global.SetGlobalBuffer((__gm__ uint64_t*)mmAddr_);
global.SetValue(AscendC::GetBlockIdx(), AscendC::GetBlockIdx());
// 算子中多核操作虽然不在同一个地址，但在同一个CacheLine, 会出现数据的随机覆盖，和通用CPU的行为不同
// 调用DataCacheCleanAndInvalid指令后，由于多核操作的时间不一致，最终结果存在随机性，后执行的核会覆盖前面核的结果
AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
```

Note the tension worth carrying forward: the official page asserts 「DMA搬运单元读写Global Memory ... 没有Cache一致性
问题」 — DMA has no coherence problem — while both kernels in §31.1-31.4 are pure-DMA handoffs (FIXPIPE out, MTE2 in)
that measurably needed DCCI and DSB on DAV_2201. Treat the doc's claim as scoped to a single core's own DMA, not to
cross-core publication.

### 31.6 `DataSyncBarrier` — the primitive that actually waits for the write-back

https://www.hiascend.com/doc_center/source/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0272.html
CANN Commercial 8.0 API reference, 「核内同步」. Retrieved 2026-10-20. Verbatim, complete:

```
DataSyncBarrier(ISASI)
功能说明
用于阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可通过参数控制）执行结束。
函数原型
template <MemDsbT arg0>
__aicore__ inline void DataSyncBarrier()
参数说明
arg0  模板参数，表示需要等待的内存位置，类型为MemDsbT，可取值为：
ALL，等待所有内存访问指令。
DDR，等待GM访问指令。
UB，等待UB访问指令。
SEQ，预留参数，暂未启用，为后续的功能扩展做保留。
返回值
无。
支持的型号
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas 200I/500 A2 推理产品
约束说明
无。
调用示例
AscendC::Mmad(...);
AscendC::DataSyncBarrier<MemDsbT::ALL>();
AscendC::Fixpipe(...);
```

This is the semantic distinction our problem turns on. `PipeBarrier<PIPE_ALL>()` is declared as
「阻塞相同流水」 — it orders instruction *issue* between pipes inside one core. `DataSyncBarrier<MemDsbT::DDR>()` is
declared as blocking until the *memory access instructions themselves have finished*. Only the second one can close
the window PPFM named: 「flag 已到、写回仍在途」.

It is declared in the public AscendC header alongside the other sync primitives (see §1):

```cpp
template <MemDsbT arg0>
__aicore__ inline void DataSyncBarrier();
```

### 31.7 `CrossCoreSetFlag` / `CrossCoreWaitFlag` — what they do and do not promise

https://www.hiascend.com/doc_center/source/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0273.html
https://www.hiascend.com/doc_center/source/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0274.html
CANN Commercial 8.0 API reference, 「核间同步」. Retrieved 2026-10-20. `CrossCoreSetFlag`, verbatim:

```
CrossCoreSetFlag(ISASI)
功能说明
面向分离架构的核间同步控制接口。
该接口和CrossCoreWaitFlag接口配合使用。使用时需传入核间同步的标记ID(flagId)，每个ID对应一个初始值为0的计数器。执行CrossCoreSetFlag后ID对应的计数器增加1；执行CrossCoreWaitFlag时如果对应的计数器数值为0则阻塞不执行；如果对应的计数器大于0，则计数器减一，同时后续指令开始执行。
同步控制分为以下几种模式，如图1所示：
模式0：AI Core核间的同步控制。对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行；对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行。
模式1：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了CrossCoreSetFlag，CrossCoreWaitFlag后续的指令才会执行。
模式2：AI Core内部，AIC与AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后， 两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。
函数原型
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
参数说明
modeId  核间同步的模式，取值范围为0~2。
pipe    设置这条指令所在的流水类型，流水类型可参考硬件流水类型。
flagId  输入  核间同步的标记。
Atlas A2 训练系列产品/Atlas 800I A2 推理产品，取值范围是0-10。
返回值
无
支持的型号
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
约束说明
因为Matmul高阶API内部实现中使用了本接口进行核间同步控制，所以不建议开发者同时使用该接口和Matmul高阶API，否则会有flagID冲突的风险。
同一flagId的计数器最多设置16次。
调用示例
// 注意：如果调用高阶API,无需开发者处理AIC和AIV的同步
// AIC侧做完Matmul计算后通知AIV进行后处理
if (g_coreType == AscendC::AIC) {
 // Matmul处理
 AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
}
// AIV侧等待AIC Set消息, 进行Vector后处理
if (g_coreType == AscendC::AIV) {
 AscendC::CrossCoreWaitFlag(0x8);
 // Vector后处理
}
父主题： 核间同步
```

`CrossCoreWaitFlag`, verbatim:

```
CrossCoreWaitFlag(ISASI)
功能说明
面向分离架构的核间同步控制接口。该接口和CrossCoreSetFlag接口配合使用。具体使用方法请参考CrossCoreSetFlag。
函数原型
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
参数说明
modeId  核间同步的模式，取值范围为0~2。
pipe    设置这条指令所在的流水类型，流水类型可参考硬件流水类型。
flagId  输入  核间同步的标记。
Atlas A2 训练系列产品/Atlas 800I A2 推理产品，取值范围是0-10。
约束说明
CrossCoreWaitFlag必须与CrossCoreSetFlag接口配合使用，避免计算核一直处于阻塞阶段。
如果执行CrossCoreWaitFlag时该flagId的计数器的值为0，则CrossCoreWaitFlag之后的所有指令都将被阻塞，直到该flagId的计数器的值不为0。同一个flagId的计数器最多设置16次。
```

Three things to extract. (a) The entire documented semantics are a **counter**: set increments, wait decrements and
blocks at zero. Nothing in either page claims any memory-ordering or cache effect. The flag is a token, not a fence.
(b) 「因为Matmul高阶API内部实现中使用了本接口进行核间同步控制，所以不建议开发者同时使用该接口和Matmul高阶API，否则
会有flagID冲突的风险」 — the vendor explicitly advises against hand-rolling `CrossCoreSetFlag` in the same kernel as
the Matmul high-level API, which is exactly what a `Matmul<...>::IterateAll` + hand-written flag kernel does. See §20
for the per-arch occupancy table. (c) The published example uses flagId `0x8` with a bare, un-templated
`CrossCoreWaitFlag(0x8)` on the AIV side — which, per the header in §1, defaults to `modeId = 0, pipe = PIPE_S`:

```cpp
template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);
```

And flagId 8 is not neutral: `catlass` reserves it (§31.8).

### 31.8 `catlass/arch/cross_core_sync.hpp` — the 15-set limit in vendor source, and the reserved flagIds

https://gitcode.com/cann/catlass/blob/master/include/catlass/arch/cross_core_sync.hpp
raw: https://raw.gitcode.com/cann/catlass/raw/master/include/catlass/arch/cross_core_sync.hpp
Repo `cann/catlass`, branch `master`, copyright 2025 Huawei, CANN Open Software License Agreement Version 2.0.
Retrieved 2026-10-20. Verbatim, the whole substantive part of the file:

```cpp
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
```

This file answers two separate questions at once. For §31 it says flagId **8 is `AIV_INTER_BLOCK_BARRIER`** in vendor
library convention and 9 and 10 are the AIC inter-block and AIV inter-subblock barriers; `FFTS_MAX_FLAG = 7` marks the
boundary of what FFTS+ itself may use. For question 2's "78 layers of handshakes" it is the canonical answer —
`CrossCoreFlagWithReverse` + `CrossCoreSetFlagWithReverse`, which pairs every flag with a `reverseId` and, every
`REVERSE_DEPTH = 15` sets, blocks on the consumer's reverse set. That is how a one-way producer→consumer handshake
survives an unbounded number of stages: back-pressure every 15, not a reset.

### 31.9 `IterateAll` and `End` — what they guarantee

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0640.html
CANN Community Edition 9.0 API reference, 「Matmul Kernel侧接口」 → `IterateAll`. Retrieved 2026-10-20. Verbatim:

```
功能说明
调用一次IterateAll，会计算出singleCoreM * singleCoreN大小的C矩阵。迭代顺序可通过tiling参数iterateOrder调整。
函数原型
template <bool sync = true> __aicore__ inline void IterateAll(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0, bool enSequentialWrite = false, bool waitIterateAll = false, bool fakeMsg = false)
template <bool sync = true> __aicore__ inline void IterateAll(const LocalTensor<DstT>& ubCmatrix, uint8_t enAtomic = 0)
参数说明
sync
获取C矩阵过程分为同步和异步两种模式：
同步：需要同步等待IterateAll执行结束
异步：不需要同步等待IterateAll执行结束
通过该参数设置同步或者异步模式：同步模式设置为true；异步模式设置为false，默认为同步模式。
Atlas 200I/500 A2 推理产品只支持设置为true。
...
waitIterateAll
输入
仅在异步场景下使用，是否需要通过WaitIterateAll接口等待IterateAll执行结束。
true：需要通过WaitIterateAll接口等待IterateAll执行结束。
false：不需要通过WaitIterateAll接口等待IterateAll执行结束，开发者自行处理等待IterateAll执行结束的过程。
fakeMsg
输入
仅在IBShare场景（模板参数中开启了doIBShareNorm开关）和IntraBlockPartSum场景（模板参数中开启了intraBlockPartSum开关）使用。
...
约束说明
传入的C矩阵地址空间大小需要保证不小于singleCoreM * singleCoreN个元素。
调用示例
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
 mm.SetBias(gmBias);
}
mm.IterateAll(gm_c); // 计算
mm.End();
```

https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0645.html
`End`. Retrieved 2026-10-20. Verbatim, complete:

```
End
功能说明
多个Matmul对象之间切换计算时，必须调用一次End函数，用于释放Matmul计算资源，防止多个Matmul对象的计算资源冲突。
函数原型
__aicore__ inline void End()
参数说明
无
返回值说明
无
约束说明
无
调用示例
mm1.IterateAll(gm_c1);
mm1.End();
mm2.IterateAll(gm_c2);
mm2.End();
```

So: `IterateAll<true>` (the default) does mean "wait for IterateAll to finish" at the issuing core, and `End()` is
documented purely as resource release — 「释放Matmul计算资源，防止多个Matmul对象的计算资源冲突」 — and is
**mandatory between two Matmul objects**, which is exactly the two-projection shape. Neither one says anything about
publishing the result to another core. There is no visibility guarantee anywhere in the Matmul API; it has to come
from §31.5/§31.6.

### 31.10 The third witness: scalar-store vs DMA incoherence, with a 12.9M-mismatch A/B

https://github.com/tile-ai/tilelang-ascend/pull/1834
Title 「[Bugfix] Scalar GM stores race MTE3 DMA through an incoherent write-back cache (#1304)」, author
`2Sheldonlaw` (commits authored by `m0_69757319`), opened 2026-09-23, closed 2026-09-28, head
`2Sheldonlaw:fix/copy-ub2gm-scalar-cache-1304-investigation` → base `tile-ai:ascendc_pto`, commits `e187fbe`,
`27b0a78`. Environment stated as Ascend910 A3 (dav-c220), CANN 9.0.0. Retrieved 2026-10-20.

The mechanism, verbatim fragments as published:

```
标量路径「经过写回缓存」：写 4 字节前先从 GM 拉入整条 64B cache line（read-modify-write），标脏，
把该行的写回推迟到逐出时。
MTE3 DMA 则「绕过该缓存直写 GM」。
```
```
t1: row oh 的标量尾触发 RMW → 下一行头部的旧值（0）被夹带进写回缓存（标脏）
t2: row oh+1 的 DMA 把正确新值写进 GM（绕过缓存）
t3: t1 的脏行被逐出 → 整条 64B 写回 GM → 旧值 0 覆盖 t2 的新值 ✗
```
```
DataCacheCleanAndInvalid<ENTIRE_DATA_CACHE, CACHELINE_OUT> before the DMA
```

and the scoping fact that matters for a megakernel, verbatim: both `dcci` and `MTE3_S` are 「本核作用域的」 — this
core's scope only — which is why synchronization alone left ~27% residual mismatches when t1/t3 ran on core A and t2
on core B. Measured: 12,967,510 → 3,496,136 mismatches with commit 1 (−73%), → 0 with commit 2 (which rewrote the
scalar store loop into a UB staging buffer plus one DMA burst). A template-level fix was rejected for ~12% slowdown
on GM-traffic-heavy kernels. `AllocateEventId` was also changed to cycle 1..7 instead of 0..7 because the C++
template `copy_gm_to_ub` in `tl_templates/ascend/common.h` hard-codes event id 0, and a pass-emitted id-0 allocation
on kernels with 8+ sync points collapsed two sets into one and deadlocked the second wait.

### 31.11 SuperKernel confirms the framework normally issues DCCI ALL around every operator

Already quoted in §19 and §22, repeated here because it is the direct confirmation the question asked for.
`cap` bit 4, verbatim:

```
• 4：disable_dcci，表示当前算子需要SuperKernel框架关闭调用前后的DCCI ALL指令。
```

and the TorchAir default, verbatim:

```
•当dcci_before_kernel_start、dcci_after_kernel_end和dcci_disable_on_kernel均未配置时，子算子内部调用GlobalTensor的GetValue/SetValue时默认不自动插入缓存刷新指令，SuperKernel默认会在所有子算子调用后插入DataCacheCleanAndInvalid指令刷新ENTIRE_DATA_CACHE。
```

and the single-operator compile behaviour, verbatim from §19:

```
- 在单算子编译场景下，毕昇编译器自动在算子末尾添加DataCacheCleanAndInvalid指令，刷新整个DCache（数据缓存）。
```
```
在开发自定义算子时，开发者必须确保所有对GM的标量读写操作都按需正确插入DataCacheCleanAndInvalid指令。
```

Confirmed, then: in the ordinary single-operator path the Bisheng compiler appends a whole-DCache
`DataCacheCleanAndInvalid` at the **end of the operator**, and SuperKernel by default inserts a
`ENTIRE_DATA_CACHE` DCCI **after every sub-operator call**. Both are per-operator-boundary flushes. A hand-written
megakernel that fuses 78 layers into one launch crosses zero operator boundaries, therefore receives exactly one
trailing flush for the whole kernel, and must issue every intermediate flush itself.

## Relevance
This is the answer to the blocking `mix_expert_kernel.cpp` failure: 0/0/416/672/896/896 wrong of 1024 across six runs
of one binary, with `partial` downloaded to the host always bit-correct, and wrong values being 0 — AIV saw zeros.
- The matching report is verbatim in §31.1: 「实测（PPFM_DIAG 指纹）：AIV 读到 vTmp 全 0，而 AIC 实际写了
  -0.013/+0.0092」, on `arch22` = DAV_2201, with the race named as 「flag 已到、写回仍在途」 and the note
  「加探针即掩盖」 — which is why our host readback is always clean.
- The ordering to write, both sides, is therefore:
  producer `Matmul::IterateAll(...)` → `End()` → `DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE,
  DcciDst::CACHELINE_OUT>(partialGlobal)` → `DataSyncBarrier<MemDsbT::DDR>()` → `PipeBarrier<PIPE_FIX>()` →
  `CrossCoreSetFlag<0x02, PIPE_FIX>(flag)`; consumer `CrossCoreWaitFlag<0x02, PIPE_MTE2>(flag)` →
  `DataCacheCleanAndInvalid<int32_t, ENTIRE_DATA_CACHE, CACHELINE_OUT>(partialGlobal)` →
  `DataSyncBarrier<MemDsbT::DDR>()` → `DataCopy`. `PipeBarrier<PIPE_ALL>()` cannot substitute: it is documented as
  「阻塞相同流水」, an issue-order barrier, whereas `DataSyncBarrier` blocks until the memory accesses themselves
  retire (§31.6).
- PPFM's own A/B (§31.3) says which half pays: W1R1 → 9/60 failures, W0R1 → 1/60. The producer's **DSB** and the
  consumer's **invalidate** are load-bearing; the producer's DCCI was not. Start with both on, then A/B the producer
  DCCI away — it is an `ENTIRE_DATA_CACHE` flush, the expensive one, and at 2.x ms TPOT we cannot afford one per
  stage per layer.
- Two further suspects in the same kernel, independent of cache coherence. (i) `CrossCoreWaitFlag(8)` with no
  template arguments resolves to `modeId = 0, pipe = PIPE_S` (§31.7 header). Mode 0 is the all-AIV inter-block sync,
  not the AIC↔AIV mode, and PPFM measured that on A2 a `PIPE_S` wait lets the following MTE2 issue early and read
  「半新半旧」 C — they use `CrossCoreWaitFlag<0x2, PIPE_MTE2>` on 2201 for exactly this. (That measurement sits
  against §20's official claim that on 2201 the `pipe` argument is inert and all pipes block; the conflict is
  unresolved, but writing `<0x02, PIPE_MTE2>` costs nothing.) (ii) flagId 8 is `AIV_INTER_BLOCK_BARRIER` in
  `catlass` (§31.8) and the vendor explicitly warns against mixing hand-written `CrossCoreSetFlag` with the Matmul
  high-level API because of flagId collision (§31.7) — and this kernel does both, with two Matmul objects.
- `End()` is **mandatory** between two Matmul objects (§31.9), which is the two-projection shape in
  `mix_expert_kernel.cpp`; but it is resource release only and buys no visibility.
- §31.4's tier-1-vs-tier-2 result is the design lesson for the per-layer kernel: hardening the one handoff you
  suspect left ~19/2000 trials still wrong; only hardening *every* cross-core GM buffer removed the drift. For 78
  fused layers that means the visibility protocol has to be a property of the buffer abstraction, not something
  sprinkled at suspected sites.
- §31.11 closes the question the `disable_dcci` capability bit raised: the Bisheng compiler appends a whole-DCache
  DCCI at the end of each single operator and SuperKernel inserts an `ENTIRE_DATA_CACHE` DCCI after every
  sub-operator by default. Our 3008 kernels/round each got one for free. One fused kernel per layer gets 78 instead
  of 3008, so the flushes we now need inside the kernel are ours to place — and their cost is a new line item
  against the 11.5 ms launch floor we are trying to remove.
- §31.8's `CrossCoreFlagWithReverse` / `MAX_REVERSE_DEPTH = 15` is the pre-answer to question 2.1.


---

## 32. `matmul_fused_manual` — the vendor's own A3 `ASCENDC_CUBE_ONLY` + manual `CrossCoreSetFlag` sample, complete, and what it does *not* do

https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_fused_manual/README.md
raw: https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_fused_manual/matmul_fused_manual.asc
Repo `cann/asc-devkit`, branch `master`. Copyright 2025 Huawei, CANN Open Software License Agreement Version 2.0.
Retrieved 2026-10-20.

This is the nearest official analogue of `tests/glm52/mixbrick/mix_expert_kernel.cpp`: the same `ASCENDC_CUBE_ONLY`
pure-Cube Matmul object, the same `IterateAll` to a `GlobalTensor`, the same hand-written `CrossCoreSetFlag` /
`CrossCoreWaitFlag` handoff, the same `__mix__(1, 2)`, the same `<<<>>>` launch, the same `dav-2201`.

Its own description of itself, verbatim:

```
# 融合编程matmul_fused_manual样例

## 概述

分核AIC、AIV融合编程实现，主要介绍了Matmul高阶API的纯Cube模式，需要调用相关接口实现手动控制AIC、AIV的核间同步。

## 本样例支持的产品及CANN软件版本

| 产品 | CANN软件版本 |
|------|-------------|
| Ascend 950PR&950DT系列产品 | >= CANN 9.1.0 |
| Atlas A3系列产品 | >= CANN 9.0.0 |
| Atlas A2系列产品 | >= CANN 9.0.0 |

## 样例描述

- 样例功能：
  本样例在AIC核创建纯Cube模式的Matmul对象，实现Matmul计算，中间结果输出至GM。AIC核计算完成后，通过调用CrossCoreSetFlag和CrossCoreWaitFlag接口手动控制核间同步，继续在AIV核对GM上的中间结果进行LeakyRelu的计算。

- 样例规格：
  本样例中：M = 128, N = 128, K = 256
```

and its documented handoff, verbatim:

```
  - Kernel关键步骤
    - AIC侧具体步骤：
      - 创建Matmul对象并初始化。
        在#include "lib/matmul_intf.h"前配置ASCENDC_CUBE_ONLY宏，创建纯Cube模式的Matmul对象
          ```cpp
          #define ASCENDC_CUBE_ONLY
          #include "lib/matmul_intf.h"
          ```
      - 完成Matmul计算。
      - 设置核间同步。
        ```cpp
        if ASCEND_IS_AIC {
          AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(3);
        }
        ```
    - AIV侧具体步骤：
      - 创建LeakyRelu对象并初始化。
      - 等待核间同步。
        ```cpp
        if ASCEND_IS_AIV {
          AscendC::CrossCoreWaitFlag<0x2>(3);
        }
        ```
      - 完成LeakyRelu计算。
```

and the stated acceptance criterion, verbatim:

```
- 执行结果

  执行结果如下，说明精度对比成功：
  ```bash
  test pass!
  ```
```

The build line, verbatim — `dav-2201` is the default and A2/A3 both map to it:

```
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;             # 编译工程，默认npu模式
```
```
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201`（默认）、`dav-3510` | NPU 架构：dav-2201 对应 Atlas A2系列产品和 Atlas A3系列产品，dav-3510 对应 Ascend 950PR&950DT系列产品 |
```

### 32.1 The kernel, verbatim

The entry point and the handoff, verbatim and complete:

```cpp
extern "C" __global__ __mix__(1, 2) void matmul_fused_manual_custom(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* bias, __gm__ uint8_t* c,
    __gm__ uint8_t* __kfc_workspace__ workspace, __gm__ uint8_t* tilingGm)
{
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    if ASCEND_IS_AIC {
        MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
        matmulLeakyKernel.Init(a, b, bias, c, workspace, tiling, &pipe);
        REGIST_MATMUL_OBJ(
            &pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj,
            &matmulLeakyKernel.tiling); // Initialize the matmul object.
        matmulLeakyKernel.Process(&pipe);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(3);
    }
    if ASCEND_IS_AIV {
        LeakyReluKernel<float> leakyReluKernel;
        leakyReluKernel.Init(c, tiling, &pipe);
        AscendC::CrossCoreWaitFlag<0x2>(3);
        leakyReluKernel.Process(&pipe);
    }
}
```

The Matmul object's declared types — note the **C matrix is declared `TPosition::VECIN`**, not `GM`, even though
`IterateAll` is handed a `GlobalTensor`:

```cpp
    Matmul<
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
        MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, CType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>>
        matmulObj;
```

`Process` — the whole cube side, verbatim. `End()` is called, and it is called **before** the flag set:

```cpp
template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline void MatmulLeakyKernel<AType, BType, CType, BiasType>::Process(AscendC::TPipe* pipe)
{
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);

    matmulObj.template IterateAll(cGlobal);
    matmulObj.End();
}
```

The consumer side, verbatim — a bare `DataCopy` from the same `c` pointer the cube wrote, with no cache op:

```cpp
template <typename CType>
__aicore__ inline void LeakyReluKernel<CType>::Init(__gm__ uint8_t* c, const TCubeTiling& tiling, AscendC::TPipe* pipe)
{
    this->tiling = tiling;
    // c:v = 1:2, split into 2 parts, for vector calculation.
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(c) + AscendC::GetBlockIdx() * tiling.M * tiling.N / 2);

    // Init input buffer.
    pipe->InitBuffer(reluInQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(CType) / 2);
    // Init output buffer.
    pipe->InitBuffer(reluOutQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(CType) / 2);
}
```
```cpp
template <typename CType>
__aicore__ inline void LeakyReluKernel<CType>::LeakyReluCopyIn(const TCubeTiling& tiling)
{
    AscendC::LocalTensor<float> reluInLocal = reluInQueue_.AllocTensor<float>();
    AscendC::DataCopy(reluInLocal, cGlobal, tiling.singleCoreM * tiling.singleCoreN / 2);
    reluInQueue_.EnQue<float>(reluInLocal);
}
```

The tiling, verbatim — one `baseM = baseN = 128` tile, i.e. a single `Iterate` worth of work, `FIRSTM` traversal, and
the tiling side declares the result position as **GM** while the kernel side declared it `VECIN`:

```cpp
    matmul_tiling::TPosition resultPosition = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat resultFormat = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType resultDtype = matmul_tiling::DataType::DT_FLOAT;
```
```cpp
    int baseM = 128;
    int baseN = 128;
```
```cpp
    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetBias(isBias);
    tilingApi.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM); // Set the matmul traverse is FIRSTM.
    tilingApi.SetFixSplit(baseM, baseN, -1);                      // Set the fixed baseM=128, baseN=128.
    tilingApi.SetBufferSpace(-1, -1, -1);
```

And the launch, verbatim — **one block**:

```cpp
    uint32_t numBlocks = 1;
```
```cpp
    matmul_fused_manual_custom<<<numBlocks, 0, stream>>>(
        inputADevice, inputBDevice, inputBiasDevice, outputCDevice, workspaceDevice, tilingDevice);
```

### 32.2 What the sample omits

Enumerated plainly, because the absences are the finding. The sample contains, across the whole file:
no `DataCacheCleanAndInvalid`, no `DataSyncBarrier`, no `PipeBarrier` of any pipe, and no `SetFlag`/`WaitFlag`.
The entire producer-to-consumer protocol is three lines:

```cpp
    matmulObj.template IterateAll(cGlobal);
    matmulObj.End();
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(3);
```
```cpp
    AscendC::CrossCoreWaitFlag<0x2>(3);
```

Four structural differences from our failing kernel, any of which could be why the sample gets away with it:
`numBlocks = 1` (our brick launches `<<<blocks>>>` with blocks > 1, and §4 records mode 2 as documented to leak
across MIX blocks at `blockDim > 1`); a single `IterateAll` of a single `baseM × baseN = 128 × 128` tile (ours is two
projections); `flagId = 3`, not 8 — 8 is `catlass`'s `AIV_INTER_BLOCK_BARRIER` (§31.8); and the mode template
argument is written out on the wait, `CrossCoreWaitFlag<0x2>(3)`, so only `pipe` takes its default.

## Relevance
This is the "known-good hand-written MIX kernel passing cube results to the vector side through global memory,
documented to be correct" that the brick was looking for, and it is structurally identical to
`tests/glm52/mixbrick/mix_expert_kernel.cpp` down to the `__mix__(1, 2)`, `ASCENDC_CUBE_ONLY`, `IterateAll` to GM and
`CrossCoreSetFlag<0x2, PIPE_FIX>`. Copy its skeleton.
- But read it against §31 rather than instead of §31. The sample's acceptance criterion is a single `test pass!` on
  M=N=128, K=256, one block, one tile. The failures §31 documents are probabilistic at rates of 19/2000 trials,
  9/60 processes, and our own 2-of-6 runs clean — a rate that one verification run of one shape cannot distinguish
  from correct. The sample is evidence that the flag-only protocol is *sufficient at blockDim = 1 on one tile*. It is
  not evidence that it is sufficient at `blockDim > 1` across 78 layers, and the three third-party kernels in §31 all
  broke in exactly the regime the sample does not enter.
- Two free corrections it hands us regardless: `End()` goes **between** `IterateAll` and the flag set, and the
  wait should be written `CrossCoreWaitFlag<0x2>(...)` with the mode spelled out (our bare `CrossCoreWaitFlag(8)`
  takes `modeId = 0`, the all-AIV inter-block mode, per the header in §1). And move off flagId 8.
- The `MatmulType<TPosition::VECIN, ...>` for C while the host tiling says `TPosition::GM` is the vendor's own
  configuration for this pattern — worth copying exactly rather than "fixing", since the Matmul high-level API's
  choice of fixpipe path depends on it.
- `__gm__ uint8_t* __kfc_workspace__ workspace` is the attribute that marks the KFC (kernel function call) workspace
  argument in a `__mix__` kernel; our brick must carry it too, sized from
  `ascendcPlatform->GetLibApiWorkSpaceSize()`.
- The sample's one-tile shape is also a reminder of the economics: at `M = 1` decode our MIX_AIC launch floor is
  13.58 us against a GroupedMatmul family that spends 5.596 ms moving 1.212 ms of bytes. The brick exists to delete
  the launch, not to make the tile faster.


---

## 33. `MixDualMaster` — turning off the AIV→AIC message drive, on A3, with `ASCENDC_CUBE_ONLY`

https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_mixdualmaster/README.md
raw: https://raw.gitcode.com/cann/asc-devkit/raw/master/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_mixdualmaster/matmul_mixdualmaster.asc
Repo `cann/asc-devkit`, branch `master`. Copyright 2025 Huawei, CANN Open Software License Agreement Version 2.0.
Retrieved 2026-10-20.

The README, verbatim and in full for the substantive part:

```
# Matmul 双主模式直调样例
## 概述
使能双主模式（MixDualMaster）的Matmul样例，AIC和AIV独立运行代码，不依赖消息驱动，用于提升性能。

Matmul API在默认的MIX模式下，通过消息机制由AIV驱动AIC运行；而在双主模式下，AIC和AIV独立运行代码，不依赖消息驱动，性能更优。
当满足以下条件之一时，可以开启双主模式：
- 核函数类型为MIX，且AIC核数：AIV核数=1：1
- 核函数类型为MIX，且AIC核数：AIV核数=1：2，且A矩阵和B矩阵同时使能IBSHARE参数

## 本样例支持的产品及CANN软件版本

| 产品 | CANN软件版本 |
|------|-------------|
| Ascend 950PR&950DT系列产品 | >= CANN 9.1.0 |
| Atlas A3系列产品 | >= CANN 9.0.0 |
| Atlas A2系列产品 | >= CANN 9.0.0 |
```

and its acceptance criterion, verbatim: 「执行结果如下，说明精度对比成功：`test pass!`」, built with
`cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201`.

The switch, verbatim — one `MatmulConfig` field off `CFG_NORM`:

```cpp
__aicore__ inline constexpr MatmulConfig GetCustomCFG()
{
    auto mmCfg = CFG_NORM;
    mmCfg.enableMixDualMaster = true;
    return mmCfg;
}
```

The Matmul object, verbatim — note `IBSHARE` is the **sixth** `MatmulType` template argument, `true` on both A and B,
which is what the 1:2 ratio requires:

```cpp
    constexpr static MatmulConfig CUSTOM_CFG = GetCustomCFG();
    // Enable IBSHARE parameters for matrix A and matrix B
    AscendC::Matmul<
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, IS_TRANS_A, LayoutMode::NONE, true>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, IS_TRANS_B, LayoutMode::NONE, true>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CUSTOM_CFG>
        matmulObj;
```

The entry point, verbatim — declared `__cube__`, **not** `__mix__`, with `ASCENDC_CUBE_ONLY` defined before
`lib/matmul_intf.h`, and no `ASCEND_IS_AIC` / `ASCEND_IS_AIV` split at all:

```cpp
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

constexpr uint32_t M = 12288;
constexpr uint32_t N = 256;
constexpr uint32_t K = 128;
constexpr bool IS_TRANS_A = false;
constexpr bool IS_TRANS_B = false;
constexpr bool IS_BIAS = true;
```
```cpp
__global__ __cube__ void matmul_mixdualmaster_custom(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* bias, __gm__ uint8_t* c, __gm__ uint8_t* workspace,
    __gm__ uint8_t* tilingGm)
{
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);
    AscendC::TPipe pipe;

    MatmulKernel<half, half, float, float> matmulKernel;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tiling);
    matmulKernel.Init(a, b, bias, c, tiling);
    matmulKernel.Process();
}
```

The compute body, verbatim — `SetTail` for the ragged edge, then `IterateAll` then `End()`:

```cpp
    // process with tail block
    int tailM = tiling.M - mCoreIndex * tiling.singleCoreM;
    tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
    int tailN = tiling.N - nCoreIndex * tiling.singleCoreN;
    tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
    if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN) {
        matmulObj.SetTail(tailM, tailN);
    }

    matmulObj.SetTensorA(aGlobal, IS_TRANS_A);
    matmulObj.SetTensorB(bGlobal, IS_TRANS_B);
    if (tiling.isBias) {
        matmulObj.SetBias(biasGlobal);
    }
    matmulObj.IterateAll(cGlobal);
    matmulObj.End();
```

and the host side, verbatim — the grid is the AIC core count, taken from the tiling:

```cpp
    tilingApi.SetDim(ascendcPlatform->GetCoreNumAic());
```
```cpp
    uint32_t numBlocks = reinterpret_cast<TCubeTiling*>(tilingBuf)->usedCoreNum;
```

## Relevance
This is the named, A3-supported mechanism for deleting the thing that makes a MIX launch cost 13.58 us instead of
2.30 us: 「Matmul API在默认的MIX模式下，通过消息机制由AIV驱动AIC运行；而在双主模式下，AIC和AIV独立运行代码，不依赖消息
驱动，性能更优」. 505 MIX_AIC calls/round at 13.58 us is 6.86 ms of our 11.6198 ms TPOT, and §15 already showed the
MIX controller as the cost `ASCENDC_CUBE_ONLY` is meant to avoid — `enableMixDualMaster` is the matching
`MatmulConfig` field.
- The eligibility rule is the catch, verbatim: either 1:1 AIC:AIV, or 1:2 **with IBSHARE enabled on both A and B**.
  Our brick is `KERNEL_TYPE_MIX_AIC_1_2`, so the 1:2 branch applies and both operands must be IBShare. §11 records
  the A3 restriction that comes with that: 「Atlas A3 训练系列产品 / Atlas A3 推理系列产品，获取矩阵计算结果时，只支持
  调用IterateAll接口，且只支持输出到GlobalTensor」 — which is already what the brick does, so the two constraints are
  compatible. But IBShare on A *and* B means both operands are L1-resident and shared across Matmul objects, which
  for our W4A8 routed experts (128 columns of `moe_intermediate` per rank) is a real tiling constraint, not a free
  switch.
- Note what the sample is *not*: it is a pure-cube `__global__ __cube__` kernel with no vector half and therefore no
  cross-core handoff. So it demonstrates the config flag and its eligibility rule, not a dual-master CV handoff. The
  combination we actually want — `enableMixDualMaster` plus a hand-written `CrossCoreSetFlag` to an AIV half — is not
  exhibited by any sample found.
- `SetTail(tailM, tailN)` is the documented way to run the ragged last tile without a second Matmul object, which at
  M=1 decode with 128 columns per rank is the normal case rather than the exception.


---

## 34. The official 220x memory model for AIC→AIV — the vendor says the flag *is* the ordering mechanism

https://www.hiascend.com/doc_center/source/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html
CANN Commercial 8.3.RC1 算子开发指南 → 架构规格 → 「NPU架构版本220」. This page covers
「Atlas A3 训练系列产品 / Atlas A3 推理系列产品 / Atlas A2 训练系列产品 / Atlas A2 推理系列产品」, i.e. our
DAV_2201. Retrieved 2026-10-20.

This is the authoritative statement of the question §31 was asked about, and it answers it in the opposite
direction. Read §31 and §34 together; the contradiction is the finding.

The architecture, verbatim:

```
如下图所示，本架构中AI Core分为AIC和AIV两个独立的核，分别用于矩阵计算和向量计算。每个核都有自己的Scalar单元，能独立加载自己的代码段。AIV与AIC之间通过Global Memory进行数据传递。
```

The inter-core section, verbatim and complete:

```
 核间同步 
 当不同核之间操作同一块全局内存时，可能存在读后写、写后读以及写后写等数据依赖问题，需要进行核间同步控制。
 核间同步控制分为以下几种模式，如下图所示：
 模式0：AI Core核间的同步控制。对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行；对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行。
 模式1：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了CrossCoreSetFlag，CrossCoreWaitFlag后续的指令才会执行。
 模式2：AI Core内部，AIC与AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后， 两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。
 例如，在AIC中将L0C的计算结果搬运到GM后，AIV需要将GM的数据搬运到UB。此时，可以使用CrossCoreSetFlag和CrossCoreWaitFlag命令，确保数据从L0C成功搬运到GM后，再从GM搬运到UB，流程如下图所示。
 CrossCoreSetFlag和CrossCoreWaitFlag接口配合使用。使用时需传入核间同步的标记ID(flagId)，即上图中的ID1，每个ID对应一个初始值为0的计数器。执行CrossCoreSetFlag后ID对应的计数器增加1；执行CrossCoreWaitFlag时如果对应的计数器数值为0则阻塞不执行；如果对应的计数器大于0，则计数器减一，同时后续指令开始执行。flagId取值范围是0-10。
 需要注意以下几点：
 成对使用
 CrossCoreSetFlag和CrossCoreWaitFlag必须成对使用，否则可能导致算子超时问题。
 一致性要求
 CrossCoreSetFlag 的模板参数和flagId必须与CrossCoreWaitFlag完全一致，否则视为不同的flagId。例如，CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8) 和 CrossCoreSetFlag<0x2, PIPE_FIX>(0x8) 设置的不是同一个flagId。
 避免连续设置
 不允许连续设置同一个flagId，以防止计数器状态混乱。
 与高阶 API 的使用冲突
 Matmul高阶API内部实现中使用了本接口进行核间同步控制，所以不建议开发者同时使用该接口和Matmul高阶API，否则会有flagId冲突的风险。
 计数器限制
 同一flagId的计数器最多可以设置15次。
 默认流水类型
 CrossCoreWaitFlag不需要显式设置指令所在的流水类型，默认使用PIPE_S。
```

The intra-core section, verbatim — the `SetFlag`/`WaitFlag` identity rule and the reserved event ids:

```
 核内同步 
 由于AI Core内部的执行单元（如MTE2搬运单元、Vector计算单元等）以异步并行的方式运行，在读写Local Memory（如Unified Buffer）时可能存在数据依赖关系。为确保数据一致性及计算正确性，需通过同步控制协调操作时序。
...
 需要注意以下几点： 
 建议通过AllocEventID或者FetchEventID接口获取EventID，以确保其合法性和有效性。
 EventID的数量有限，使用后应立即调用ReleaseEventID释放资源，避免EventID耗尽，影响系统正常运行。
 SetFlag和WaitFlag必须成对使用，且SetFlag和WaitFlag的参数必须完全一致（包括模板参数和事件ID）。如果不匹配，可能导致当前核的计算异常，或影响下一个核的算子执行，引发timeout问题。 
 例如，SetFlag<HardEvent::S_MTE3>(1)和SetFlag<HardEvent::MTE3_MTE1>(1)设置的不是同一个EventID，因为其模板参数不同。只有当模板参数和事件ID完全一致时，才表示同一个EventID。
 不允许连续设置同一个EventID，因为这可能导致事件状态混乱或未被正确处理。
 不建议手动插入 TEventID，不能手动插入6和7的TEventID，因为它们可能被系统预留或用于特殊用途。
```

The Fixpipe capability list, verbatim — this is the quantisation and activation set available *on the way out of
L0C*, i.e. for free, before the result ever reaches GM:

```
 支持Fixpipe硬件化加速
 Fixpipe是NPU将典型操作进行硬化的加速模块，位于AIC内部，配合Cube计算单元完成随路计算，主要功能如下：
 量化反量化：包括S322FP16、S322S32、S322S4、S322S8、S322S16、FP322FP16、FP322BF16、FP322S8、FP322S4、FP322FP32。
 Relu功能，包括ReLu、PReLu和Leaky ReLu等典型的激活函数。
 数据格式转换，包括： 
 通过Channel merge、Channel split可以实现分形大小的转换，保证输出到L1 Buffer/GM的分形满足诉求。
 NZ2ND数据格式转换。
```

and the alignment table, verbatim:

```
 核 存储单元 对齐要求
 AIV Unified Buffer 32Byte对齐。
 AIC L1 Buffer 32Byte对齐。
 L0A Buffer 512Byte对齐。
 L0B Buffer 512Byte对齐。
 L0C Buffer 64Byte对齐。
 BiasTable Buffer 64Byte对齐。
```

and the cross-card note, verbatim — relevant to the TP reduce, since it says `DataCopy` itself can cross cards but
only over HCCS on A2:

```
 支持跨卡数据搬运（Hccs物理链路）
 在跨卡通信算子开发场景，DataCopy类接口支持跨卡数据搬运，在
 Atlas A2 训练系列产品 / Atlas A2 推理系列产品
 设备上，仅支持Hccs物理链路，不支持其他通路；开发者开发过程中，请关注涉及卡间通信的物理通路； 通过npu-smi info -t topo指令查询Hccs 物理通路。
```

## Relevance
This page is the official answer to "what makes an AIC's global write visible to an AIV's `DataCopy`", and the answer
it gives is: the flag, and nothing else. Verbatim, in exactly our scenario:
「例如，在AIC中将L0C的计算结果搬运到GM后，AIV需要将GM的数据搬运到UB。此时，可以使用CrossCoreSetFlag和
CrossCoreWaitFlag命令，**确保数据从L0C成功搬运到GM后，再从GM搬运到UB**」. No cache maintenance is mentioned
anywhere on the page. Combined with §32's vendor sample, which carries no DCCI either, the documented contract is
flag-only.
- Against that stand the three measured third-party failures in §31 on this same DAV_2201, including one whose
  symptom is ours to the letter (AIV reads all zeros where AIC wrote nonzero). The resolution is not available from
  any source found: either the hardware does not honour the documented contract in some regime (multi-block, multi-
  tile, high occupancy), or all three kernels had a different bug that DCCI happened to mask. Treat the DCCI+DSB as
  an empirical workaround with a measured effect, not as the documented rule — and A/B it rather than adopting it.
- Four hard constraints to take at face value, all new or sharper than what §20 and §4 carried:
  (a) 「CrossCoreSetFlag 的模板参数和flagId必须与CrossCoreWaitFlag完全一致，否则视为不同的flagId」 — the Set's
  template arguments must match the Wait's exactly, or they are different flagIds. Our brick sets
  `<0x02, PIPE_FIX>(8)` and waits with a bare `CrossCoreWaitFlag(8)`, which is `<0x0, PIPE_S>`. By the letter of this
  rule those are two different flags and the wait should block forever. It does not — the kernel produces output —
  and the same page's own 「默认流水类型：CrossCoreWaitFlag不需要显式设置指令所在的流水类型，默认使用PIPE_S」 plus
  §20's 「在NPU架构版本2201上，CrossCoreWaitFlag传入的pipe参数无效」 plus the `CrossCoreSetFlag` page's own example
  pairing `<0x2, PIPE_FIX>(0x8)` with a bare `CrossCoreWaitFlag(0x8)` all say the Wait's template arguments are not
  part of the flag's identity on 2201. So the consistency requirement is, in practice, about matching two *Sets* —
  which is how the page's own illustration reads, both of its examples being `CrossCoreSetFlag`. Write
  `CrossCoreWaitFlag<0x02>(...)` anyway, as `matmul_fused_manual` does; it costs nothing and removes the ambiguity.
  (b) 「同一flagId的计数器最多可以设置15次」 — **15** here, against 「同一flagId的计数器最多设置16次」 on both the
  `CrossCoreSetFlag` and `CrossCoreWaitFlag` API pages (§31.7) and 「continuously set up to 15 times」 in `catlass`
  (§31.8). Two of three sources say 15; design to 15.
  (c) 「不允许连续设置同一个flagId」 — restated, and it is the reason §31.3's `AicSetToAiv` sets once under mode 2
  rather than once per AIV.
  (d) 「不建议手动插入 TEventID，不能手动插入6和7的TEventID」 — intra-core event ids 6 and 7 are reserved, which
  narrows the usable `SetFlag`/`WaitFlag` event space to 0-5 per pipe pair, tighter than the "8 ids" figure in
  `ascend-launch-overhead-and-sync.md` §3. Note `tile-ai/tilelang-ascend` PR #1834 (§31.10) independently moved its
  allocator to cycle 1..7 for a different reason (a template hard-coding id 0).
- The Fixpipe list is a direct lever on our measured numbers. `Cast` costs 2.300 ms/round against 0.076 ms of bytes
  moved — a pure overhead family — and Fixpipe does `S322FP16`, `S322S8`, `S322S16`, `FP322BF16`, `FP322S8` and
  ReLU/PReLU/LeakyReLU *in the AIC on the way out of L0C*. For W4A8 routed experts producing INT32, the
  `S322S8`/`S322FP16` conversion and the NZ2ND format change can happen before the data is ever written, which both
  removes the Cast and shrinks the GM traffic the AIV must then read back.
- 「AIV与AIC之间通过Global Memory进行数据传递」 is the official confirmation of README finding 3b: there is no
  L0C→UB path on 220x, so a per-layer fused kernel's cube→vector edge is always a GM round trip, and the only
  question is whether it stays in L2.


---

## 35. `asc.gitcode.com` — the full AscendC API reference as static HTML, and the definitive 2201 cache model it contains

### 35.0 The source itself

https://asc.gitcode.com/api/
A complete mirror of the Ascend C API reference, served as static HTML (VitePress), with a product-version selector
listing 「全量 / Ascend 950 / Atlas A3 / Atlas A2 / Atlas 训练系列 / Atlas 推理系列 / Atlas 200I/500 A2 / Kirin X90 /
Kirin 9030」, and **every page carries an explicit 「产品支持情况」 block naming Atlas A3 by name**. Retrieved
2026-10-20.

This matters operationally: `hiascend.com/document/detail/...` is a JavaScript SPA that returns an empty body to a
fetcher, and `hiascend.com/doc_center/source/...` works but has a sparse, undiscoverable id space (see this file's
Gaps). `asc.gitcode.com` resolves by readable path and contains the sidebar of every page, so one fetch yields the
whole URL map. URL shape:

```
https://asc.gitcode.com/api/SIMD-API/基础API/同步控制/核间同步/SyncAll.html
https://asc.gitcode.com/api/SIMD-API/基础API/同步控制/核内同步/DataSyncBarrier_ISASI.html
https://asc.gitcode.com/api/SIMD-API/基础API/缓存控制/系统缓存概述.html
https://asc.gitcode.com/api/SIMD-API/基础API/缓存控制/DataCacheCleanAndInvalid.html
https://asc.gitcode.com/api/SIMD-API/基础API/标量计算/WriteGmByPassDCache_ISASI.html
https://asc.gitcode.com/api/SIMD-API/基础API/标量计算/ReadGmByPassDCache_ISASI.html
https://asc.gitcode.com/api/SIMD-API/基础API/cube_compute_ISASI/矩阵计算的搬出/Fixpipe_L0CToUB.html
```
(percent-encode the Chinese path segments.) It is also newer than the CANN 8.0 `doc_center` copies quoted in §31:
several pages carry content those do not.

### 35.1 「系统缓存概述」 — the authoritative per-cache coherence table for 2201

https://asc.gitcode.com/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E7%BC%93%E5%AD%98%E6%8E%A7%E5%88%B6/%E7%B3%BB%E7%BB%9F%E7%BC%93%E5%AD%98%E6%A6%82%E8%BF%B0.html
Retrieved 2026-10-20. This is the page `WriteGmByPassDCache` cross-references as 「Cache写策略与Cache一致性问题」.
Verbatim, the parts that settle §31 and §34:

```
表1 Cache类型及功能说明 
Cache类型
说明
L2 Cache
L2 Cache作为二级缓存，专门用于存储频繁访问的数据和指令，以便减少对GM的读写。
•通过MTE2单元读取GM时，优先从L2 Cache中读取数据；当L2 Cache未命中时，再从GM加载数据并更新到L2 Cache中。
•通过Scalar单元读取GM时，优先从DCache中读取数据；当DCache未命中时，再从L2 Cache中读取数据。当L2 Cache也未命中时，再从GM加载数据并更新到L2 Cache和DCache中。
•通过Scalar单元读取GM指令时，优先从ICache中读取指令；当ICache未命中时，再从L2 Cache中读取指令。当L2 Cache也未命中时，再从GM加载指令并更新到L2 Cache和ICache中。
DCache
DCache（Data Cache）用于缓存Scalar单元近期可能被重复访问的数据段。
通过Scalar单元读取GM时，优先从DCache中读取数据；当DCache未命中时，再从L2 Cache中读取数据。当L2 Cache也未命中时，再从GM加载数据并更新到L2 Cache和DCache中。
ICache
ICache（Instruction Cache）用于缓存Scalar单元最近使用或频繁使用的指令。
```

and the architecture inventory, verbatim:

```
以NPU架构版本3510为例，图1展示了AI Core中支持的五类Cache（L2 Cache、DCache、ICache、SIMT DCache、NDDMA Cache）在硬件架构中的位置关系。
以NPU架构版本2201为例，图2展示了AI Core中支持的三类Cache（L2 Cache、DCache和ICache）在硬件架构中的位置关系。
```

the write policies, verbatim:

```
Cache写策略与Cache一致性问题​
当向GM写入数据，并且启用Cache时，存在以下两种Cache写策略：延迟写回（Write-Back）和直写（Write-Through）。
当向GM写数据时采取延迟写回策略，工作原理如下：
开发者预期向GM写入数据（即期望数据直接写入GM），硬件实际行为是：
当数据被修改时，只在缓存中更新数据。
缓存中的数据标记为“脏”。
当该Cache Line被替换时（或通过软件显式执行Clean操作时），数据才会被写回GM，Cache Line何时被替换由硬件决定。
...
在延迟写回策略下，被修改的数据仅更新在本地缓存中，不会立即同步到GM。当其他核读取GM上同一地址时，获取的将是过时数据，从而导致不同核间缓存数据不一致。 Cache一致性是多核中确保数据正确性的核心机制。简单来说，当多个核各自拥有独立的Cache时，它们可能同时缓存相同的数据。如果某个核修改了缓存中的数据，其他核的缓存副本就会变得“过时”，如果不及时同步，就会导致多个核间的数据不一致，进而引发计算错误。
```

and — the single most load-bearing paragraph in this file — the per-cache verdict for **our architecture**, verbatim:

```
针对NPU架构版本2201，支持的三类Cache在多核间数据一致性方面的情况如下： L2 Cache：多核间共享的缓存，因此不需要考虑多核间数据不一致的问题。
DCache：多核独立缓存，因此需要考虑多核间数据一致性的问题。
ICache：只读，因此不需要考虑多核间数据不一致的问题。
```

the four cache operations, verbatim — note that `Invalid` discards dirty lines:

```
表2 Cache操作及功能说明
操作名称
功能说明
Prefetch
硬件根据访问模式自动将预期访问的数据提前加载到缓存，提升后续访问速度。
Preload
软件通过显式指令将指定数据主动加载到缓存，为即将到来的读写操作做准备。
Invalid
•将指定地址范围的Cache Line（缓存的最小操作单元）标记为“无效”，使其从缓存中移除。
•保证下一次访问这些内存地址时，会从GM重新加载数据，而不是使用可能过期的缓存数据。
•注意：Invalid操作不检查缓存行是否为“脏”（dirty，表示该数据已被修改但尚未写回到GM），若存在脏数据将被直接丢弃。
Clean
将缓存中被修改过的数据（脏数据）写回到GM中，避免数据丢失。Clean操作不将缓存行标记为“无效”，缓存行仍保持有效状态。
```

and the control-interface summary, verbatim:

```
表3 缓存控制接口汇总
L2 Cache  SetL2CacheHint  读写GM的数据都默认被缓存在L2 Cache（默认启用L2 Cache），SetL2CacheHint能够设置GlobalTensor是否启用L2 Cache。
DCache  DataCachePreload  DataCachePreload接口从源地址所在的特定GM地址预加载数据到DCache中，每次调用只能预加载一个Cache Line大小的数据。
DCache  DataCacheCleanAndInvalid  当Scalar单元访问GM时，使用该接口刷新Cache，保证Cache的一致性，使用场景如下：
•读取GM的数据，但该数据可能在外部被其余核修改，此时需要使用DataCacheCleanAndInvalid接口，直接访问GM，获取最新数据。
•用户通过Scalar单元写GM的数据，希望立刻写出，也需要使用DataCacheCleanAndInvalid接口。
DCache  ReadGmByPassDCache/WriteGmByPassDCache  不经过DCache向GM地址上读/写数据。使用场景：
•当多个核写入的数据落在同一条Cache Line内时，经过DCache的读写将以64B为粒度，可能引发多核数据随机覆盖问题。使用该接口不经过DCache直接按操作数大小读/写GM，可避免此问题。
•使用该接口不经过DCache直接向GM读/写数据，可避免由DCache缓存引发的多核间数据不一致问题。
ICache  ICachePreLoad  开发者手动调用ICachePreLoad接口，能够从指令所在GM地址预加载指令到ICache中。
ICache  GetICachePreloadStatus  GetICachePreloadStatus为调试接口，在调用ICachePreLoad后调用，用于获取ICache的PreLoad的状态。当返回值为0时，说明ICache的PreLoad已完成；当返回值为1时，说明ICache的PreLoad未完成。
NDDMA Cache  NdDmaDci  在使用DataCopy（GMToUB多维数据搬运NDDMA）接口进行数据搬运前，需要使用NdDmaDci接口刷新NDDMA Cache缓存保证为最新状态。
```

### 35.2 `DataSyncBarrier` — the official cross-core publication idiom, with a complete worked example

https://asc.gitcode.com/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E5%90%8C%E6%AD%A5%E6%8E%A7%E5%88%B6/%E6%A0%B8%E5%86%85%E5%90%8C%E6%AD%A5/DataSyncBarrier_ISASI.html
Retrieved 2026-10-20. This page is materially different from the CANN 8.0 copy in §31.6 — the `MemDsbT::ALL`
definition is narrower and `SEQ` is now live. Verbatim:

```
DataSyncBarrier(ISASI)
产品支持情况​
Ascend 950PR/Ascend 950DT：支持
Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
Atlas 200I/500 A2 推理产品：支持
Atlas 推理系列产品AI Core：不支持
Atlas 推理系列产品Vector Core：不支持
Atlas 训练系列产品：不支持
功能说明​
头文件路径为："basic_api/kernel_operator_block_sync_intf.h"。
阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可通过参数控制）执行结束，用于解决Scalar的内存依赖。
函数原型​
template <MemDsbT arg0>
__aicore__ inline void DataSyncBarrier()
参数说明​
arg0
模板参数，表示需要等待的内存位置，类型为MemDsbT，可取值为：
• ALL，等待GM和UB的访问指令（不包括其他内存位置，例如L0C Buffer、L1 Buffer）。
• DDR，等待GM访问指令。
• UB，等待UB访问指令。
• SEQ，等待SEQ访问指令。
约束说明​
等待的内存位置取值为ALL时，只能等待GM和UB的访问指令，不包括其他内存位置，例如L0C Buffer、L1 Buffer。
```

and its 调用示例, verbatim and complete — this is a producer/consumer handoff between two blocks with **no
`CrossCoreSetFlag` at all**, just a GM flag, `DataSyncBarrier<DDR>` and the bypass-DCache accessors:

```cpp
if (blockIdx == 0) {
 // 先写入被依赖数据，再通过DataSyncBarrier等待GM访问完成。
 AscendC::WriteGmByPassDCache<T>(reinterpret_cast<__gm__ T *>(srcGm) + 1, DATA_VALUE);
 // DataSyncBarrier<DDR>阻塞后续GM写，确保上一条GM写对其他核可见。
 AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
 // 最后写入同步标记，block 1读到该标记后即可安全读取srcGm[1]。
 AscendC::WriteGmByPassDCache<T>(reinterpret_cast<__gm__ T *>(srcGm), SYNC_FLAG);
 }
 if (blockIdx == 1) {
 while (true) {
 __gm__ T *addr = const_cast<__gm__ T *>(srcGlobal.GetPhyAddr());
 // 轮询GM第0个元素，等待block 0写入同步标记。
 T flagValue = AscendC::ReadGmByPassDCache<T>(addr);
 if (flagValue == SYNC_FLAG) {
 // DataSyncBarrier保证同步标记之前的srcGm[1]写入已完成。
 T dataValue = AscendC::ReadGmByPassDCache<T>(addr + 1);
 AscendC::WriteGmByPassDCache<T>(reinterpret_cast<__gm__ T *>(dstGm), 2 * dataValue);
 return;
 }
 }
 }
```

### 35.3 `WriteGmByPassDCache` / `ReadGmByPassDCache` — A3-supported, and the ordering rule between them and DCCI

https://asc.gitcode.com/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E6%A0%87%E9%87%8F%E8%AE%A1%E7%AE%97/WriteGmByPassDCache_ISASI.html
https://asc.gitcode.com/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E6%A0%87%E9%87%8F%E8%AE%A1%E7%AE%97/ReadGmByPassDCache_ISASI.html
Retrieved 2026-10-20. Verbatim:

```
WriteGmByPassDCache(ISASI)
产品支持情况​
Ascend 950PR/Ascend 950DT：支持
Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
Atlas 200I/500 A2 推理产品：不支持
Atlas 推理系列产品AI Core：不支持
Atlas 推理系列产品Vector Core：不支持
Atlas 训练系列产品：不支持
功能说明​
头文件路径为："basic_api/kernel_operator_scalar_intf.h"。
不经过DCache向GM地址上写数据。使用场景：
当多个核写入的数据落在同一条Cache Line内时，经过DCache的读写将以64B为粒度，可能引发多核数据随机覆盖问题（参考DataCacheCleanAndInvalid调用示例3）。使用该接口不经过DCache直接按操作数大小写GM，可避免此问题。
经过DCache写GM时可能导致多核间的数据不一致问题（详细原因请参考Cache写策略与Cache一致性问题），使用该接口不经过DCache直接向GM写数据，可避免此问题。
由于WriteGmByPassDCache接口向GM写数据时不经过DCache，因此开发者需要考虑如下场景：
当GM上地址addr已经在DCache缓存并且其对应的Cache Line标记为“脏”（dirty，表示该数据已被修改但尚未写回到GM）时，开发者应该在WriteGmByPassDCache接口之前调用DataCacheCleanAndInvalid接口将addr对应的Cache Line立即写回GM，否则WriteGmByPassDCache接口写入addr的数据后续可能被DCache写回的脏数据覆盖。
DataCacheCleanAndInvalid接口与WriteGmByPassDCache接口向GM写数据时硬件不能保证两个接口的执行顺序，因此开发者应该在WriteGmByPassDCache接口之前调用DataSyncBarrier接口对这两个接口进行同步，否则WriteGmByPassDCache接口写入addr的数据依然可能被DCache写回的脏数据覆盖。
根据以上的描述，为了简化编程（开发者无需关心addr是否在DCache缓存以及是否被标记为“脏”），建议开发者在使用WriteGmByPassDCache接口时采用如下代码片段：
```
```cpp
AscendC::GlobalTensor<T> global; // global为addr对应的GlobalTensor
AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
// 保证WriteGmByPassDCache接口向addr写入value之前，DCache中的脏数据已经写回GM。
AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
AscendC::WriteGmByPassDCache<T>(addr, value);
```
```
函数原型​
template <typename T>
__aicore__ inline void WriteGmByPassDCache(__gm__ T* addr, T value)
数据类型​
支持的数据类型为int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t。
```
```
ReadGmByPassDCache(ISASI)
功能说明​
头文件路径为："basic_api/kernel_operator_scalar_intf.h"。
不经过DCache从GM地址上读数据。
函数原型​
template <typename T>
__aicore__ inline T ReadGmByPassDCache(__gm__ T* addr)
数据类型​
支持的数据类型为int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t。
```

### 35.4 The A3 cube→vector data-path support matrix, checked page by page

Each of these is the verbatim 「产品支持情况」 line for `Atlas A3 训练系列产品/Atlas A3 推理系列产品`, from the
mirror, retrieved 2026-10-20.

`Fixpipe（L0C Buffer到UB数据搬运）` —
https://asc.gitcode.com/api/SIMD-API/%E5%9F%BA%E7%A1%80API/cube_compute_ISASI/%E7%9F%A9%E9%98%B5%E8%AE%A1%E7%AE%97%E7%9A%84%E6%90%AC%E5%87%BA/Fixpipe_L0CToUB.html
```
Ascend 950PR/Ascend 950DT：支持包含FixpipeParamsArch3510参数的接口。
Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
Atlas 200I/500 A2 推理产品：支持包含FixpipeParamsM300参数的接口。
```

`DataCopy（L0C Buffer到UB数据搬运）` —
https://asc.gitcode.com/api/SIMD-API/%E5%9F%BA%E7%A1%80API/cube_compute_ISASI/%E7%9F%A9%E9%98%B5%E8%AE%A1%E7%AE%97%E7%9A%84%E6%90%AC%E5%87%BA/DataCopy_L0CToUB.html
```
说明： 本接口搬运功能仅在Atlas 推理系列产品AI Core产品型号的L0C Buffer->UB（CO1 -> CO2）通路支持。其他型号和其他通路支持接口调用但功能不生效，功能等同于基础数据搬运。
Ascend 950PR/Ascend 950DT：不支持
Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
```

`DataCopyL1ToUB（L1到UB数据搬运）` —
https://asc.gitcode.com/api/SIMD-API/%E5%9F%BA%E7%A1%80API/cube_compute_ISASI/%E7%9F%A9%E9%98%B5%E8%AE%A1%E7%AE%97%E7%9A%84%E6%90%AC%E5%87%BA/DataCopyL1ToUB.html
```
Ascend 950PR/Ascend 950DT：支持
Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
```
and its constraint, verbatim — worth recording for the 950 port even though A3 lacks it:
```
该接口仅适用于Mix算子场景，且仅支持Cube核数与Vector核数比例为1:2的配置。在该场景下，subBlockId仅支持取值0或1。
如果需要执行多个DataCopyL1ToUB指令，且DataCopyL1ToUB的目的地址存在重叠，需要通过调用PipeBarrier(ISASI)来插入同步指令，保证多个DataCopyL1ToUB指令的串行化，防止出现异常数据。
```
with `subBlockId` documented as 「AI Core上的Vector核ID。根据输入的Vector核ID，将数据搬运至对应核上的UB地址。」

`Fixpipe（L0C到L1数据搬运）` — the one on-chip egress A3 *does* have —
https://asc.gitcode.com/api/SIMD-API/%E5%9F%BA%E7%A1%80API/cube_compute_ISASI/%E7%9F%A9%E9%98%B5%E8%AE%A1%E7%AE%97%E7%9A%84%E6%90%AC%E5%87%BA/Fixpipe_L0CToL1.html
```
Ascend 950PR/Ascend 950DT：支持包含FixpipeParamsV220/FixpipeParamsArch3510参数的接口。
Atlas A3 训练系列产品/Atlas A3 推理系列产品：仅支持包含FixpipeParamsV220参数的接口。
Atlas A2 训练系列产品/Atlas A2 推理系列产品：仅支持包含FixpipeParamsV220参数的接口。
```
with the A3-specific caveat on its `config`, verbatim:
```
  • CFG_ROW_MAJOR（默认取值）：开启NZ2ND，输出数据格式为ND格式。针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，
Atlas A2 训练系列产品/Atlas A2 推理系列产品，
在L0C Buffer -> L1 Buffer通路下不生效。
  • CFG_NZ：NZ2NZ，输出数据格式为NZ格式。
```

### 35.5 `SyncAll` on A3 — a MIX grid barrier that does three syncs in one, and its 2201 flagId bill

https://asc.gitcode.com/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E5%90%8C%E6%AD%A5%E6%8E%A7%E5%88%B6/%E6%A0%B8%E9%97%B4%E5%90%8C%E6%AD%A5/SyncAll.html
Retrieved 2026-10-20. The three prototypes and their A3 support, verbatim:

```
软同步原型​
Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
硬同步（不支持config指定流水）原型​
Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
硬同步（支持config指定流水）原型​
Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
```

the semantics, verbatim — the fused-operator hard sync is three barriers in one instruction:

```
SyncAll是核间同步控制接口，根据不同的函数原型介绍其功能：
硬件同步接口： 用于多个核之间的同步控制，支持以下同步场景：
纯Vector算子的全核同步，可参考CrossCoreSetFlag的模式0。
融合算子的全核同步，一次性实现三个核间同步能力（参考CrossCoreSetFlag的关键特性说明）：首先完成单个AI Core内，AIC与所有AIV同步（AIC等AIV）；接着完成多个AI Core间，所有AIC的同步；最后完成单个AI Core内，所有AIV与AIC同步（AIV等AIC）。
软件同步接口： 仅支持纯Vector算子的多核（参与同步的核数可通过入参指定）同步。每一个AIV都向全局内存对应位置写入信号值1，接着每一个AIV都持续轮询全局内存中所有AIV对应的信号值，直到所有AIV核对应的信号值都变为1。
在纯Vector算子场景中，若所有AIV核默认参与同步，推荐采用性能更优的硬件同步接口；若需指定部分AIV核参与同步，则应使用软件同步接口，并通过入参usedCores完成配置。
```

the hard-sync constraints, verbatim — the kernel-modifier requirement and the `isAIVOnly` traps:

```
硬件同步接口约束：
使用硬同步接口时，需根据场景使用合适的核函数修饰符：
在纯Vector算子，需使用__mix__(0, 1)修饰核函数，而不是使用__vector__，具体原因请参考CrossCoreSetFlag约束说明中的Kernel类型配置说明。
对于Mix（包含Cube和Vector计算）算子，需使用__mix__(1, 1)或__mix__(1, 2)修饰核函数。
对于纯Vector算子，模板参数isAIVOnly必须设置为true，否则会导致程序卡死。
对于Mix（包含Cube和Vector计算）算子，模板参数isAIVOnly设置为true，会导致接口实际只能完成Vector全核同步与预期同步行为不符。
使用该接口时，建议开启batchmode模式，使算子独占全部所需核资源，否则可能因满足以下条件导致死锁：
多流并发场景（≥2条执行流）。
≥2个算子并发执行。
所有并发算子的核数总和超过物理核数。
≥2个并发算子使用了核间同步功能。
具体而言，在多流场景下，某条流的核间同步算子虽分配到n个物理核，但可能仅有n-m个核先被调度执行，而其余m个核因被其他流的核间同步算子抢占而尚未启动。先启动的n-m个核执行到核间同步时等待剩余m核完成，而剩余m核因被其他流的核间同步算子占用而无法释放，形成死锁。 Kernel直调场景下通过__schedmode__(mode)限定符来设置batchmode模式；工程化算子开发场景下，通过TilingContext的SetScheduleMode接口来设置batchmode模式
```

the flagId bill for 2201, verbatim:

```
SyncAll硬件同步接口内部实现中使用了CrossCoreSetFlag和CrossCoreWaitFlag进行核间同步控制，因此开发者同时使用CrossCoreSetFlag（或CrossCoreWaitFlag）和SyncAll硬件同步接口时，需注意避免以下flagId冲突：
表4 SyncAll硬件同步接口flagId占用情况
SyncAll 使用场景
占用flagId汇总
各计算核flagId占用情况
isAIVOnly=true
14
AIV：14
isAIVOnly=false
11、12、13
AIC：11、12、13；AIV：12、13
```
(and for 3510, for comparison: 「isAIVOnly=false（核函数使用__mix__(1, 2)修饰时）| 11、12、13、28、29」.)

the common constraint, verbatim:

```
硬件同步接口和软件同步接口公共约束：使用该接口进行多核控制时，算子调用时指定的逻辑AI Core核数numBlocks必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。
```

and the soft-sync sizing, verbatim:

```
软件同步接口约束：
gmWorkspace缓存申请的空间大小要求大于等于GetBlockNum()*32Bytes，并且缓存的值需要初始化为0。目前常见的有两种初始化方式： 通过在host侧进行初始化操作，确保传入该接口时，gmWorkspace缓存已经初始化为0；
在kernel侧初始化的时候对gmWorkspace缓存初始化，需要注意的是，每个核上都需要初始化全部的gmWorkspace缓存空间。
ubWorkspace申请的空间大小要求大于等于GetBlockNum()*32Bytes。
usedCores传入数值不能超过算子调用时指定的逻辑AI Core中AIV的数量：GetBlockNum()*GetTaskRatio()，不传此参数表示全核软同步。
```

and `SyncAllConfig`, which A3 does **not** get, verbatim:

```
struct SyncAllConfig {
 pipe_t triggerPipe;
 pipe_t waitPipe;
};
// 默认使用全部流水来进行触发和等待行为。
constexpr SyncAllConfig DEFAULT_SYNC_ALL_CONFIG = {PIPE_ALL, PIPE_ALL};
• triggerPipe：指定哪个流水用于“发送触发信号”。仅支持MTE2、MTE3、PIPE_ALL。
• waitPipe：指定哪个流水用于“接收等待信号”。仅支持MTE2、MTE3、PIPE_ALL。
```

## Relevance
§35.1 settles question 1.1, and it revises the recommendation I drew from §31. On NPU architecture 2201, verbatim:
「L2 Cache：多核间共享的缓存，因此不需要考虑多核间数据不一致的问题。DCache：多核独立缓存，因此需要考虑多核间数据
一致性的问题。」 — and 「通过MTE2单元读取GM时，优先从L2 Cache中读取数据」. So:
- **The AIC→FIXPIPE→GM→MTE2→AIV path has no cache-coherence problem on A3.** Both ends go through the shared L2.
  The only incoherent cache on 2201 is the per-core DCache, which only the **Scalar** unit uses. That is exactly what
  the `DataCacheCleanAndInvalid` page means by 「Scalar单元访问Global Memory ... 存在 ... 一致性问题」, and it is why
  §31's producer-side DCCI on a FIXPIPE-written buffer was aimed at the wrong cache — which is also, retrospectively,
  why PPFM's measured A/B found the producer DCCI *unnecessary* (W0R1 1/60 beat W1R1 9/60) while the producer's
  `DataSyncBarrier` stayed in.
- **What `mix_expert_kernel.cpp` is missing is completion ordering, not cache maintenance.** The primitive for it is
  named and its contract is quoted in the vendor's own example comment: 「DataSyncBarrier<DDR>阻塞后续GM写，确保上一条
  GM写对其他核可见」. So the minimal correct producer sequence is `IterateAll` → `End()` →
  `DataSyncBarrier<MemDsbT::DDR>()` → `CrossCoreSetFlag<0x02, PIPE_FIX>(flag)`, and the consumer needs **nothing**
  beyond `CrossCoreWaitFlag<0x02>(flag)` → `DataCopy`, unless it also touches the buffer with Scalar
  (`GetValue`/`SetValue`). One instruction, no `ENTIRE_DATA_CACHE` flush, no per-stage cost worth measuring. Try
  that before the §31 protocol.
- **Caveat on `MemDsbT`:** 「ALL，等待GM和UB的访问指令（不包括其他内存位置，例如L0C Buffer、L1 Buffer）」. `ALL` does
  not cover L0C or L1, so there is no `DataSyncBarrier` flavour that waits on the cube's own L0C staging; the FIXPIPE
  write to GM is what `DDR` covers, which is the edge we need. Note also that the CANN 8.0 page in §31.6 defined
  `ALL` as 「等待所有内存访问指令」 and `SEQ` as reserved; the newer page narrows `ALL` and activates `SEQ`. Use the
  newer wording.
- **For every flag, counter and epoch tag in the megakernel, use `WriteGmByPassDCache` / `ReadGmByPassDCache`
  instead of `GetValue`/`SetValue` + DCCI.** Both are A3-supported, both are in
  `basic_api/kernel_operator_scalar_intf.h`, and §35.2's example is a complete host-barrier-free producer/consumer
  handoff built from nothing but them plus `DataSyncBarrier<DDR>` — no `CrossCoreSetFlag` at all. That is the
  building block for question 2.1 (78 layers of handshakes without burning flagIds), question 2.3 (epoch-tagged
  in-kernel collective) and question 2.4 (GM-counter grid reduction), and it sidesteps the flagId budget entirely.
  The one rule to respect, verbatim: DCCI and `WriteGmByPassDCache` have no hardware-guaranteed order, so a
  `DataSyncBarrier` must separate them — the vendor's own three-line recipe is quoted in §35.3.
- §35.4 **confirms README finding 3b rather than overturning it**, and closes the "go via L1" idea: on A3,
  `Fixpipe(L0C→UB)` 不支持, `DataCopy(L0C→UB)` 不支持, and `DataCopyL1ToUB` 不支持. `Fixpipe(L0C→L1)` *is* supported
  (with `FixpipeParamsV220`), but with no L1→UB egress, L1 is only useful for feeding back into the cube. So the
  cube→vector edge of a per-layer kernel is unavoidably `L0C → Fixpipe → GM → MTE2 → UB` and the design goal stays
  "stay in L2", consistent with the measured `aiv_mte2_time` 320.543 → 57.145 us when fused.
- §35.5 is a grid barrier for a MIX megakernel that we did not have. `SyncAll<false>()` under `__mix__(1, 2)` does
  three barriers in one — AIC waits all its AIVs, then all AICs across blocks, then all AIVs wait their AIC — which
  is exactly the inter-layer boundary a 78-layer fused kernel needs, and it costs flagIds **11, 12, 13** on 2201
  rather than any of the hand-allocatable ones. Two traps to encode now: `isAIVOnly=true` in a MIX kernel silently
  degrades to a Vector-only barrier (「与预期同步行为不符」), and `numBlocks` must not exceed the physical core count
  or the framework's multi-round scheduling inserts a bad sync and hangs. `__schedmode__` is restated as the
  anti-deadlock requirement, with the four co-occurring conditions spelled out — consistent with
  `ascend-fusion-mechanics.md` §25's measured 3-hangs-in-49.
- Finally, §35.0 is a tooling result for this channel: every future AscendC API question should start at
  `asc.gitcode.com`, because it is static, path-addressable, newer than the `doc_center` copies, and states A3
  support explicitly on every page.


## Gaps

What was looked for and not found, or found only partially.

- **No published per-kernel launch-cost figures on A3.** Nothing in the vendor docs gives a microsecond number for
  "cost of one MIX_AIC launch" versus "one AI_VECTOR_CORE launch". The closest quantitative anchors found were
  (a) the Matmul static-tiling case study (21.88 us -> 17.728 us per call, Scalar time -44.4%, section 12) and
  (b) the `__schedmode__` hang statistic (3/49 vs 0/240, section 25). Our measured 0.70 / 2.30 / 12.04 / 13.58 us
  floors have no published counterpart; the documented *mechanisms* (core-count multiplier, 硬同步, two task-param
  pointers per MIX FFTS+ context, full-pipe cross-core waits on 2201) are consistent with them but are not a
  published decomposition.
- **The `super_kernel/src/jit` and `src/aot` implementation sources in `cann/graph-autofusion` were not read.**
  Only the public header, README and docs were retrieved. The actual fusion/segmentation policy, the ICache-preload
  insertion pass, and the sub-kernel-split code generation live there. GitCode's API rate-limited at 10^5 calls/hour
  partway through, so directory listings under `super_kernel/src`, `super_kernel/kernel` and
  `super_kernel/examples/{jit,aot}/*` are unread. Likewise `super_kernel/docs/zh/super_kernel/super_kernel_migration_design.md`.
- **Partially closed: one measured SuperKernel A/B exists (section 28), but none on a transformer decode round.**
  The vendor's own profiling sample reports 5.61% end-to-end on a 6-segment MoE-shaped graph. Neither the TorchAir docs
  nor the vllm-ascend docs publish a before/after latency for Super Kernel on a full model. The GLM-5.2 A3 config in
  section 7 carries `baseline: 733.1813` for a 128k throughput benchmark but no A/B against `enable_super_kernel: false`.
- **The hiascend `llms-content/*.md` endpoint is WAF-blocked** (returns the HuaweiCloudWAF JS challenge), as is the
  `document/detail/...` SPA viewer. Only `doc_center/source/<lang>/<product>/<vol>/<book>/<id>.html` returns raw HTML,
  and the id space is sparse — several probed ids returned S3 `AccessDenied`. Volumes not reachable this way at the
  paths tried: `devaids/opdev/ascendcbestP` (zh), `opdevg/cceintrinsicguide` (all versions tried), and the zh
  rendering of the Matmul-constant best practice (only the en CANN 8.0 copy was obtainable).
- **`matmul_gelu_high_performance` Scenario 2 and the Matmul basic-API / Gelu tuning samples were only partly read.**
  The `01_matrix_compute/matmul_basic_api_high_performance` and `02_reg_compute/gelu_high_performance` READMEs that
  section 16 cross-references were not fetched; they contain the L1/L0 double-buffer and 512B-alignment case studies
  (Case 1 单核Tiling优化 / Case 2 多核切分 2x12 / Case 3 多核切分 4x6) referenced by title only.
- **No `cann-ops-adv` repository found under that name on GitCode.** The CANN operator tree appears to have been
  split into `cann/ops-nn`, `cann/ops-math`, `cann/ops-blas`, `cann/ops-transformer` (and GitHub mirrors
  `hicann/ops-nn`, `hicann/ops-transformer`, `hicann/ops-blas`). No fusion-specific issue or PR with a quantified
  MIX-overhead discussion was located; searches surfaced only PR numbers (`cann/ops-nn#8743`, `#8172`,
  `cann/ops-math#4354`, `cann/graph-autofusion#1871`, `cann/ge#993`) whose bodies were not retrieved.
- **`__schedmode__` has no API-reference page.** It is referenced by name in the `SyncAll` and `CrossCoreSetFlag`
  constraint sections and documented in prose in a production header comment (section 25), but no page giving its
  syntax, legal `N` values, or placement rules was found. `SetScheduleMode(schedule_mode)` with `0 = 普通模式`,
  `1 = batchmode模式` is the only formally documented form.
- **No published L2 cache size for A3.** Section 8's architecture doc says only 「根据硬件规格不同，Cache Line大小不同
  （128/256/512Byte等）」 and that buffer sizes must be queried via `GetCoreMemSize`. `CoreMemType` has no L2 member in
  the `catlass` usage found. HBM bandwidth figures for A3 were not found in any vendor doc reached.
- **`DEVICE_USE_ONLY` and AICPU-dispatched kernels were not covered.** No doc page was located for `DEVICE_USE_ONLY`.
  AICPU appears only indirectly: as `RT_CTX_TYPE_AICPU` in the FFTS+ taxonomy (section 18), as the Tiling-下沉
  Notify/Wait partner in the SuperKernel README (section 6), and as the default `task_breaker_bypass = 0` fusion
  breaker (section 22).
- **`aclrtLaunchKernel` vs taskless dispatch was not resolved.** `ACLRT_LAUNCH_KERNEL` and `<<<>>>` are documented
  (section 14) and `aclrtLaunchKernelAttr` has a page at
  `https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_1815.html`,
  but that page's body (2.8 KB) was not extracted, and no doc comparing launch paths' overhead was found.
- **`SetLocalWorkspace` is documented only as an Atlas-inference-AI-Core requirement** (section 11). No page
  describing it as a general megakernel UB-reservation mechanism was found.
- **Closed for `MatmulCallBackFunc` (section 29); `MatmulPolicy` still unread.** `MatmulPolicy` has a page at
  `https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0619.html`
  and `MatmulConfig` at `..._0616.html`, `GetNormalConfig` at `..._0621.html`; none of those three bodies were extracted,
  so the full `MatmulConfig` field list (including `enUnitFlag`'s siblings) and the `MatmulPolicy` module-replacement
  mechanism remain unread.
- **Closed: the `BareMix` sample is reproduced in section 30.** Found only in community mirrors of the
  `Ascend/samples` tree (`XJ19023/ascend_samples`, `c303803919-jpg/transmit`) at
  `operator/ascendc/0_introduction/22_baremix_kernellaunch/`. The canonical `Ascend/samples` upstream path was not
  confirmed, and the sample's own `README.md` / `main.cpp` / `baremix_custom_tiling.cpp` were not fetched (so the host
  side's `blockDim`, `__schedmode__` setting and workspace sizing are unverified).
- **`matmul_client.h` was not obtained, so the vendor's own cross-core DCCI could not be quoted first-hand.**
  §31.2's comment cites 「与 CANN matmul_client.h 中"读跨核 GM flag 前先 DCCI"的用法一致」, which would be the
  strongest possible evidence (the Matmul high-level API doing it internally). `cann/asc-devkit` does not appear to
  ship the Matmul high-level implementation headers: `include/lib/matmul_intf.h`,
  `include/high_level_api/lib/matmul_intf.h`, `impl/high_level_api/lib/matmul/matmul_client.h` and
  `impl/high_level_api/matmul/matmul_client.h` all return 404, while `include/basic_api/...` resolves. The file
  presumably lives only in the installed CANN toolkit under `.../tikcpp/tikcfw/lib/matmul/matmul_client.h` — read it
  on the A3 and quote it.
- **`WaitIterateAll` has no reachable API page.** It is listed in the 「Matmul Kernel侧接口」 TOC (§11) but the TOC
  link extraction in §31.9 returned pages for `IterateAll` (0640) and `End` (0645) and no `WaitIterateAll` entry;
  the referenced 「IterateAll异步场景样例」 was likewise not located. So the exact asynchronous contract
  (`sync=false` plus `waitIterateAll=true` plus a `WaitIterateAll()` call) is documented only by the parameter
  descriptions quoted there, not by its own page.
- **The `pipe`-argument conflict on 2201 is unresolved.** §20 quotes the official claim 「在NPU架构版本2201上，
  `CrossCoreWaitFlag`传入的`pipe`参数无效，无论取值为何都默认阻塞全部流水的后续指令，建议省略`pipe`参数并使用默认值」;
  §31.3 quotes a measured counter-claim from a shipping A2 kernel that a `PIPE_S` wait lets the following MTE2 issue
  early and read half-stale C, which is why they pass `PIPE_MTE2` on 2201. No third source was found to adjudicate,
  and no erratum or DTS number was located for either statement.
- **No vendor erratum or DTS number was found for the cross-core GM visibility behaviour itself.** The evidence in
  §31 is three independent third-party kernels plus the API pages' silence. Nothing on `hiascend.com`,
  `bbs.huaweicloud.com` or gitcode states the DAV_2201 publication rule as a documented hardware property, and the
  official `DataCacheCleanAndInvalid` page positively asserts the opposite for DMA
  (「DMA搬运单元读写Global Memory ... 没有Cache一致性问题」).
- **`MemDsbT::SEQ` is 「预留参数，暂未启用」** and `DcciDst::CACHELINE_UB` / `CACHELINE_ATOMIC` are both
  「预留参数，暂未支持」 on the pages quoted in §31.5/§31.6, so the only usable combinations on A3 are
  `CacheLine::{SINGLE_CACHE_LINE, ENTIRE_DATA_CACHE}` × `DcciDst::{CACHELINE_ALL, CACHELINE_OUT}` and
  `MemDsbT::{ALL, DDR, UB}`. No cost figure was found for any of them; PPFM's own accounting is only
  「AIC 每 chunk 10 次整缓存 DCCI + AIV 6 次 + 4 次 DSB」 with no per-instruction time.
