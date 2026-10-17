# TileRT Ascend：A3 单用户 2 ms decode

目标：`a3-gs:trt-asc`，单机 8 模组／16 die，GLM-5.2-W4A8C8，batch=1，8K 上下文，TPOT ≤2 ms，即 ≥500 tok/s。vLLM-Ascend 执行 prefill，HIXL 交接 KV，TileRT-Ascend 执行 decode；单机两阶段依次占用设备，decode 计时期间 prefill 静默。

vLLM-Ascend P+D 基线已完成：同一 checkpoint 与 8K token IDs，TP16／DP1、MTP3／graph4，逐请求串行生成 1024 tokens；3 次预热、32 次测量，TPOT p50=16.39 ms、p95=18.29 ms，均值 16.51 ms，即 60.55 tok/s。2 ms 需要将 p50 缩短 8.20 倍；后续用 vLLM prefill、HIXL 与 TileRT M4 重跑相同输入，先验精度，再根据 profiling 压缩关键路径。[原始结果与口径](evidence/ascend-decode/vllm-baseline-20261017/README.md)、[基线启动](../tests/glm52/serve_vllm_baseline.sh)与[测量入口](../tests/glm52/bench_vllm_decode.py)。

本次 P+D 基线在 `a3-gs:trt-asc` 空闲时复用已有镜像与统一 Python 环境。此前 TP16 短序列验证使用 `a3-187:trt-asc`。[环境快照](evidence/ascend-decode/a3-187-environment-20261016.json)、[运行时检查](evidence/ascend-decode/a3-187-runtime-20261016.json)

实施顺序：直接复用 CANN 9.2.0 已有算子，先跑通真实 W4A8C8、batch=1、TP16 decode，并通过逐层精度、路由、索引和缓存检查；随后采集 profiling，建立正确执行路径的算子、搬运、启动与同步基线；再按采集结果调整 tiling、布局、流水与融合范围。只有能力缺失或已有实现难以达到预算时，才用 TileLang Ascend／AscendC 重写。每次替换保留同输入、同精度、同计时范围的 CANN 对照。[已有算子与精度入口](evidence/ascend-decode/cann-operators.md)

精度收敛先重比冻结张量，再将首个分歧的算子或 rank 本地计算片段放到单 die 验证，保持真实 TP16 分片、量化和缓存布局。完整 TP16 运行负责验收跨卡归约、状态提交和独立生成轨迹，按实际 HBM 与通信资源准入；整机空闲用于性能测量。每轮以一个可重放的数值分歧为单位，修复后先重比原输入，再扩大覆盖。

## 开发主线与停止条件

主线固定 M4，M1 保留为精度与性能对照。将设备执行看作串行 greedy decode 的精化。设 prompt 长度为 P，正常轮次提交边界的逻辑输出数为 N，主干有效长度为 L；最后一个输出是尚未被主干消费的 pending token。优化保持以下四个不变量。

| 不变量 | 形式与代码依据 |
|---|---|
| 前缀一致 | `L=P+N−1`，主干 KV 对应 prompt 和前 N−1 个输出；[完整 P 行注入](../tilert/models/glm_5_2_ascend/cann_model.py)后，首个 pending token 在位置 P 被消费。 |
| 接受等价于 greedy | `c[0]=pending`；若连续匹配 k 个 draft，则自然提交 `A=k+1≤M` 个 target predictions，并按输出预算／EOS 裁剪；`L'=L+A`，新 pending 为最后一个提交的 prediction。[接受状态](../tilert/models/glm_5_2_ascend/mtp_state.py) |
| MTP parent 正确 | 有效 MTP 行 p 对应 `(normalized_target_hidden[p], token[p+1])`。chain 行先截掉，再由 target hidden 重算；seed 后每轮提交的 main/MTP 同长。[轮次实现](../tilert/models/glm_5_2_ascend/mtp_round.py) |
| TP／PD 一致 | 16 rank 消费同一请求 generation，并按同一命令序列推进有效长度；本 generation 的全部 KV 写入完成后才能 inject。[TP 控制](../tilert/models/glm_5_2_ascend/tp_runtime.py)、[PD 接收](../tilert/pd_vllm/ascend_receiver.py) |

归纳步骤：设本轮 target 预测为 y，前 k 个 draft 满足 `c[i+1]=y[i]`。因 causal verify 的前 k+1 行与串行输入一致，`y[0:k+1]` 正是后续 greedy 输出；主干消费 `c[0:k+1]` 后仍留下最后一个 y 未消费，前缀不变量保持。由这些行的 target hidden 重算 MTP，再截断到 `L+A`，保持 MTP 不变量。现有[状态穷举](../tests/glm52/check_mtp_state.py)覆盖接受位置、parent 标签和 reset；真实 CANN 数值、因果索引及 HIXL 可见性由设备对照封闭。

性能以请求交付计时 `TPOT=(t_last−t_first)/(N−1)` 为准。稳态每轮只允许花 `2 ms×平均实际提交数`；初次 seed、控制与交付另占请求预算。按第 1 节的流式模型，若有效带宽为 1.12 TB/s、平均 A=3.2，每轮 HBM 字节上限是 7.17 GB；若再给 HBM 之外未被掩盖的工作留 1 ms，则降至 6.05 GB。当前 7.54–8.19 GB 的场景需要减少读取量或提高实际接受数／有效带宽，Graph 的启动收益须与这项差距一起核算。

| 顺序 | 本阶段交付 | 转入下一步的条件 |
|---|---|---|
| 1. 封闭精度 | 复用已有 HF、main、MTP 和 session 入口；主干先短前缀、再一个 8K 输入的 32 步，随后验证同一请求的 MTP 和 session。补齐执行所需的阶段输入接线。 | 所选 backend／width 的 `validate_precision` 通过，greedy、有效 KV 和逐层 logits 满足第 4 节标准，即可采集该输入的短 profiling。 |
| 2. 定标成本 | M1／M4 短 profiling；profiler 关闭后记录轮耗时、实际 A、专家／KV 并集；实测 HBM 和 12／48 KiB HCCL。 | 用实际字节、带宽及暴露的通信／主机时间解释每轮成本，选出支配 2 ms 差距的一项。 |
| 3. 压缩关键路径 | 主机同步／启动占主导时，验证设备 indices 路径及已有 device GMM，再做固定阶段 Graph；流量占主导时减少复制读取并复用多行权重；归约占主导时融合生产端并重叠通信。每次只推进其中一项。 | 同输入精度保持、profiler 关闭的请求耗时下降，且校准后的预算仍能指向 2 ms；随后重测并选择下一项。 |
| 4. 完成验收 | 八个不同 8K 输入全部通过精度，预热后执行 32 个 1024-token 请求，保留实际接受数和交付时间。 | 请求 TPOT 的 p50／p95 均≤2 ms。 |

走读已确认两个候选主机同步点：默认 [MLA](../tilert/models/glm_5_2_ascend/ops/mla.py) 每层读取 indices 到 CPU；默认 [MoE](../tilert/models/glm_5_2_ascend/native_engine.py) 读取 routes 到 CPU 后逐专家执行。现有设备路径需先通过相同输入的数值验收，再以实际暴露耗时判断收益。

当前推进第 1 步：M1 优先复用已运行的 vLLM-Ascend 模型计算层及 CANN fused prolog，由 TileRT 管理完整 P 行 KV 注入与 decode 调用循环；首个输入是 prefill 已产出的 pending token，位置为 P。[薄适配器与验收入口](evidence/ascend-decode/cann-model.md)先单独验证，设备精度通过后接入 PD 服务；已有 Native 算子实现用于同输入差分。M1／M2／M4 已接入逐层 hidden、logits 和实际路由／索引捕获；[宽验证比较器](../tests/glm52/compare_saved_cann_verify.py)逐行直接对照相同前缀与 teacher 的 HF，串行／宽验证的容差不传递。随后验证 CANN MTP 的 seed、chain、target-conditioned 重算与完整 session。后续任务须明确封闭哪个精度不变量，或预计减少多少 ms/round，并在上述阶段条件满足后转入下一步。若固定布局的必要成本连 `A=4`、8 ms/round 都无法满足，就调整读取／同步机制。

精度定位按 `F_N(s_N)−F_H(s_H)=[F_N(s_N)−F_H(s_N)]+[F_H(s_N)−F_H(s_H)]` 分解；s 包含 hidden、有效 KV 和共享 indices。[同输入逐层检查](../tests/glm52/check_saved_layers.py)定位局部算术误差，[独立轨迹重比](../tests/glm52/compare_saved_decode.py)保留误差传播结果。局部已通过的分支收束，整体未通过时检查共同传播机制；真实 8K 精度验收决定能否进入 profiling。生产路径修改后重跑设备检查，reference 修改先重比已有设备张量。

精度基线提前注册并初始化 vLLM SFA vendor。M1 AddRmsNorm 保留 `BF16(normalized) × gamma` 边界；CANN 普通 input norm 与 fused MLA prolog 按各自的 FP32 中间精度建立独立参考，[单 die 同输入证据](evidence/ascend-decode/mla-prolog.md)已封闭首个前处理分歧。各 rank 的 BF16 partial 转为 FP32 做 HCCL、最后转回 BF16；M=1／4 的基线消息为 24／96 KiB。通信优化以 BF16 传输、FP32 累加恢复 12／48 KiB 消息，融合保持上述数值合同。

[W4 MSD 合同](evidence/ascend-decode/w4-msd.md)保留两个 FP16 部分积及 FP32 的 `16H+L+checkpoint_bias` 合成顺序，32/32 同输入 M1／M4 投影逐位通过；[A8 合同](evidence/ascend-decode/a8-rounding.md)按 `Div(127,amax)` 生成倍率，184 个冻结样例逐位通过。[RoPE 合同](evidence/ascend-decode/rope-rounding.md)保留 BF16 系数、FP32 乘加及末尾一次 BF16 舍入，42 个冻结 index key 逐位通过；此前同输入检查仍有 38/1,572 项失败。继续用冻结输入和单 die 差分封闭 CANN 前处理、MLA 与专家链，再以真实 8K HIXL 交接和完整 P 行注入验证续算。[通信修复与设备证据](evidence/ascend-decode/pd-hixl-20261017/README.md)

## 1. 先算速度边界

先取 TP16、MTP 关闭、qA/kvA、router、indexer 复制的简单布局。模型有 78 层，前 3 层 FFN 为 dense，其余 75 层激活 8/256 个路由专家及 1 个共享专家；78 层 attention 均使用 top-2048，21 层重新计算 index。

| 输入 | 数值及依据 |
|---|---|
| A3 HBM | 本机每 die 64 GiB；带宽规格每 die 1.6 TB/s、每模组 3.2 TB/s，整机峰值 25.6 TB/s。[Huawei §3.3](https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1) |
| 权重容量 | 实际主权重 404.69 GB，另有 MTP 使用的 75.5 MB `rot.weight`；均分约 23.56 GiB/die。加复制权重、KV、workspace 后按每 die ≤48 GiB 设计。[header 审计](evidence/a3-model-bytes.json) |
| 运行时权重读取 | 每 rank、单 token 约 3.844 GB。按当前代码计入复制的 FP32 router、W4 的 INT64 scale 与 FP32 MSD 补偿；W8 丢弃已检查为零的 offset，LM head 预转换为 BF16。[可重现计算](../tests/glm52/estimate_traffic.py) |
| 单 rank 读取 | 权重 3.844 GB；MLA `78×2048×576×2=184 MB`；index KV `8192×(16×130+5×256)=27.5 MB`；合计约 4.056 GB。INT8 index 层携带 FP16 scale，其余 5 个主干 indexer 保持 BF16。[精度契约](evidence/ascend-decode/weight-contract.md) |
| AllReduce | 主干每层 attention、FFN 各一次，共 156 次，目标单 token 为 `6144×2=12 KiB` BF16；当前 FP32 精度基线为 24 KiB。TP16 同缓冲区依赖链的图重放 event p50：24 KiB FP32 为 5.028 ms，96 KiB FP32 为 5.074 ms；Python 提交的 24 KiB 链为 26.756 ms。[实测与复现](evidence/ascend-decode/tp-sum.md) |

流式读取模型给出 `T_HBM=4.056 GB/(1.6 TB/s×η)`：峰值带宽下为 2.53 ms，约 394 tok/s；取 70% 带宽为 3.62 ms，约 276 tok/s。加上当前 HCCL 图路径实测的 5.03 ms 归约链，串行预算为 7.56／8.65 ms，即约 132／116 tok/s；理想通算重叠预算为 `max(T_HBM,5.03 ms)`，约 199 tok/s。图链包含重放提交、rank 偏差和依赖调度，不能将总时长除以 156 当作裸 HCCL α；HBM 有效带宽与完整 decode 速度仍待实测。

因此，只消除 Python 提交开销仍不足以达到 2 ms。关键是复制的低秩投影、router、indexer、156 次归约，以及小 M 下的解包和计算效率。2 ms 路线从 MTP 共享权重读取、减少复制读取、tile 流水和生产端融合归约展开。

MTP 实现验证宽度 `M=2/4`：`T_round=T_verify(M)+ΣT_draft+T_accept`，稳态预算为 `ΣT_round/ΣA`；A 为实际提交数，包含 target 修正／bonus token，M=4 时 `1≤A≤4`。稳态 M=4 包含两次链式 draft 和一次 target-conditioned MTP4 重算；若每层专家并集为 32，当前复制布局每轮约读 7.54–8.19 GB，70% 带宽、A=3.2 时，仅 HBM 已需 2.10–2.29 ms/token，主干 96 KiB 归约图链再占 1.59 ms/token。两项串行预算为 3.69–3.87 ms/token，尚未加入 MTP 额外归约、4 次选词 all-gather 和计算。MTP 必须结合读取优化和通算重叠；平均 A=3.2 时，每轮 6.4 ms 是稳态预算上限，初次 seed、采样和交付仍需余量。最终 TPOT 使用每请求首末 token 间耗时／(交付数−1)，计入首 token 后的初次 seed、采样和交付。[场景预算](evidence/ascend-decode/runtime-traffic.json)

## 2. 从真实接口确定工作量

本仓库是闭源 `libtilert_*.so` 的 Python 包装层：Python 管理权重、缓存和 PD，计算、通信及 engine 藏在库中。本任务是重建 NPU 上的执行与调度核心，优先调用已有 CANN 算子；需要自定义 tile 时再选择 TileLang Ascend 或 AscendC。当前 GLM-5.2 库为 ROCm／FP8、TP8，只接受 MTP 0/1/3；TP16、W4A8C8 和 NPU 同步机制需要重新实现。

审计基线为仓库 `f554fa57`、wheel `tilert-0.1.6.post3-cp312-cp312-manylinux_2_28_x86_64.whl`。三个库的 ELF 动态导出表均为空；结合 Torch 注册 schema、注册点反汇编和 Python 调用恢复接口，得到 DSV32 96 项、GLM5 37 项、GLM5.2 48 项，去重后 175 项。完整名称、参数顺序、返回类型和二进制地址见 [schema 清单](evidence/tilert-ascend/schemas.tsv)；shape、dtype、缓冲区副作用及分级见 [GLM-5.2 契约](evidence/tilert-ascend/glm52-contracts.md)。48 项包括 26 项计算／通信、12 项有状态控制、9 项查询、1 项 peer setup，覆盖整层和整网入口；未恢复的 native 内存序等字段在清单中标明。

| 分级 | 实现范围 |
|---|---|
| 可直接复用 | 模型结构、采样语义、Python reference 与 PD 控制逻辑；计算优先检查 CANN 的量化 matmul／GMM、norm、RoPE、cache 更新、indexer、稀疏 attention 与 HCCL。真实权重投影和单专家链已通过；其余实际形状逐项验证。闭源二进制可复用数为 0/48。 |
| 需重写 | 闭源 engine 的生命周期、权重／缓存地址管理及 NPU 调度适配；ROCm swizzle、FP8 cache 与地址表替换为 Ascend 布局。已有算子缺失或实测改造难以达标的计算项，单列依据后重写。 |
| 需新机制 | TP16 tile 调度、AIC/AIV/MTE 协作、计算与归约衔接、epoch 同步、设备侧 MTP 接受／回退，以及 HIXL→Ascend KV 注入。 |

## 3. 最小后端设计

1. 入口与基线：在统一 Python 环境、CANN 9.2.0 上以现有算子串联完整 decode；`glm_5_2_ascend` 与 backend loader 沿用 `prepare → inject → decode → reset` 生命周期。先通过固定 78 层、M=1 的精度验收，再采集该路径的 profiling；M=4 在 MTP 数值通过后采集。需要自定义 tile 时使用已固定的 [tilelang-ascend `ascendc_pto`](https://github.com/tile-ai/tilelang-ascend/tree/1d5daf0385156cbe0c499a16c10935c5190642a4) 或 AscendC。
2. 权重与并行：TP16，每 die 4 个 attention heads、每个专家 intermediate=128；每个 rank 持有所有专家的 TP 分片，直接处理选中专家。先复制 qA/kvA、router、indexer，记录其独占耗时，再比较低秩投影切分加 gather，以及独立 index rank 的净收益。按张量配方处理 scale、offset 与 scale_bias；down 的补偿只取本 rank 的一列。主干直接使用已导出的权重，MTP 在 hnorm 前应用 `rot.weight`；LM head 按 vocabulary 切分。
3. 执行调度：先使用已有算子的预打包权重和预分配缓存，验证 ACL Graph 对固定阶段的收益。重复搬运、独立启动或同步边界进入关键路径时，优先调整 CANN 实现的 tiling、布局、双缓冲和融合范围。需要跨算子的 tile 流水时，再以 AIC/AIV/MTE 和 cross-core flag 组织生产与消费；缓冲区按 `round/layer/tile` epoch 复用。
4. 通信：以 HCCL 给出 TP16 的正确性与时延基线，再比较适用的 AIV 路径。若归约等待主导，改造已有输出投影／expert down 与归约的衔接，并比较 residual/RMSNorm 融合收益；需要时以小型 AscendC extern 接入 device 通信。每项改造验证完成语义与关键路径净收益。HIXL 负责 P→D KV 交接。
5. 稀疏与 MTP：先复用已有 Q/KV 投影、norm／RoPE／cache 更新、indexer 和稀疏 attention，逐项验证 8K 选 2K、4 heads/rank 与 INT8/BF16 混合 index cache；再比较现有 fused prolog 和 SwiGLU／量化接口。共享层复用 indices。MoE 按 M 个位置的专家并集复用权重；draft、verify、accept 按实际接受长度提交 KV／位置并丢弃未接受尾部，每轮向 Python 返回 token 列表。
6. PD：复用 [Transport 接口](../tilert/pd_vllm/transport.py) 接入 HIXL 注册、批量 WRITE 与完成通知；[Ascend 专用 V1 connector／receiver](evidence/ascend-decode/pd-ascend.md) 使用 16 个 rank endpoint，独立于 GPU wire 的 8-rank 默认。开启 `indexer_kv_dtype=int8` 后，本 checkpoint 的 16 个主干 indexer 使用 INT8＋FP16 scale，0/1/2/70/74 及 MTP78 使用 BF16；MLA latent512+RoPE64 仍按 BF16。两端统一 page、逐层 dtype、scale、rotation、position 和 layout version。传主干 P 行、MTP P-1 行，全部 rank 完成同一 generation 才 ready；M1 直接消费 prefill 首 token。MTP 另传 12 KiB 的主干 normalized hidden[P-1]，在 D 端只补一行 MTP seed；[可选交接与捕获](evidence/ascend-decode/mtp-parent-handoff.md)及 [CANN MTP 请求接线](evidence/ascend-decode/cann-mtp.md)已实现，CPU API／HTTP 生命周期检查通过，真实权重执行与设备精度待验证。8K 前缀约 0.78 GB/rank。生成 1024 tokens、M=4 时按 `L≥8192+1024+3` 再向 page 对齐分配；双份权重、KV 与当前阶段 workspace 合计峰值 ≤60 GiB/die，workspace 按阶段释放。

已验证的起点：单 die 四层 BF16 缩小模型已执行，TileLang RMSNorm＋Torch/NPU 计算路径在 CPU、A3 上各通过 1,729 项检查；NPU 最大 NRMSE 为 0.008435，greedy token、索引、专家选择和 reset 重放通过。完整 W4A8C8／TP16／MTP 与 2 ms 性能验收按下表推进。[复现与原始结果](evidence/ascend-decode/README.md)

当前代码已接入 CANN TP16 的 M=1/2/4 验证、按专家分组、KV 截断、分布式 top1、MTP78、完整轮次编排与服务请求循环。[真实 8K 两步续算](evidence/ascend-decode/cann-model.md#frozen-8k-continuation)在全部 rank 上输出 `[382,262]`，与 HF 一致；数值与 reset 验收仍失败。冻结输入已将首个重放分歧定位到 layer 4 的融合 GMM1／SwiGLU／A8，后续 GMM2 的 48 条同输入结果逐位通过，继续修复该融合边界。服务通过 `--mtp-width 1/2/4` 选择串行或 MTP，支持按实际交付裁剪取消请求的 KV；默认保留串行精度基线。MTP 缓存行 p 使用 normalized target hidden[p] 与 token[p+1]；链式 indices 复用，接受后按 target hidden 重算 MTP KV。先完成真实 8K 精度验收，再采集 profiling。[验证入口](evidence/ascend-decode/offline-development.md)

[测量入口](evidence/ascend-decode/measurement.md) 将精度报告与原始 8K prompt、权重、源码和版本绑定；各 rank 在执行线程中采集短 profiling，独立关闭 profiler 的 PD 请求同时记录生产端 callback 与客户端 token 到达时间。2 ms 判定要求实测覆盖八个不同的已验收输入。

## 4. 可独立验收

以下为开发验收预算。microbench 在空闲 `a3-gs:trt-asc` 上运行，固定实际权重分片及 M=1/4，保存命令、版本、字节数与原始结果；预热 100 轮、测量 1000 轮，取 16 rank 最大耗时的 p50/p95。每个优化项先给出 CANN 直接调用基线，再给出修改／融合或新实现的差分结果。HBM 使用超出 L2 的轮换数据；设备事件与整段 wall time 分别记录。HCCL 每批预热 20 次、测量 100 次，在计时窗外收集 16 rank 的批均值并取 MAX，重复 100 批统计分位数；另测 156 次依赖链。[定标入口](evidence/a3-roofline.md#本机定标)

| 验收对象 | 数字与判据 |
|---|---|
| HBM 与容量 | 报告各 die 持续读最低值。当前 M4／32 专家并集／A=3.2 若保持 8.19 GB 读取量，仅 HBM 就要求 ≥1.28 TB/s；若只有 1.12 TB/s，需将轮流量降至 ≤7.17 GB，并另留计算／调度余量。decode 权重＋KV＋workspace ≤48 GiB/die；P/D 共存时单 die 总峰值 ≤60 GiB。 |
| TP16 AllReduce | BF16 传输、FP32 累加，rank-MAX 批均值的 p50：12 KiB≤6 µs、48 KiB≤8 µs；同时记录 24／96 KiB FP32 HCCL 精度基线。测独立算子及 156 次依赖链，12 KiB 链≤1 ms；接入后报告暴露在关键路径上的时间。 |
| W4A8 MoE | 本地 intermediate=128，M=1／8 专家的 gate+up+SiLU+down≤20 µs；M=4／32 个不同专家≤45 µs；包含解包、动态 A8 与 scale_bias，不含 router/shared/跨卡归约。 |
| 稀疏 attention | M=1、4 heads、2K×576 BF16 的 MLA≤8 µs；完整 32-head indexer 在 8K 上投影＋打分＋top-2048≤20 µs。M=4 同时报专家／KV 并集与实际读取量。 |
| LM head | BF16 `[9680,6144]` 分片，M=1 的 norm＋投影＋local top1≤120 µs；跨 rank 选词另计。 |
| 接口覆盖 | 175/175 名称有归属；GLM-5.2 的 48/48 契约标明实现、等价适配或范围外。实际 decode 路径的算子及初始化／reset／采样／MTP 状态接口，样例通过率 100%，报告通过数／应实现数；4 个辅助 MFMA 入口及 DSV32/GLM5 专属入口单列，不计入已实现数。 |
| 数值 parity | HF eager reference 逐层加载并反量化同一 checkpoint，显式模拟 TP16 局部 A8、INT8 index cache、scale/offset/bias、旋转与融合舍入边界；qA/kvA RMS epsilon 显式对齐 Ascend prefill 的 `1e-5`，indexer 使用其交错坐标及 raw head weights。固定 8 个 8K 输入，各 teacher-force 32 步；78 层及 MTP 层逐层比较 hidden states，并以同一 final norm＋LM head 得到逐层 logits：`NRMSE=‖x−ref‖₂/max(‖ref‖₂,1e−12)≤1e−2`、cosine≥0.999；logits 的 max-abs≤0.1、greedy token 100% 一致。路由 top-8、index top-2048、位置／cache 提交亦逐项核对；tie 规则固定。 |
| 2 ms 目标 | 预热后串行运行 32 个请求，各输入 8K、输出 1024 tokens；每请求 `(t_last−t_first)/(N−1)` 的 p50/p95 均≤2 ms，同时报告逐 token 间隔、真实 A 和轮耗时，包含接受、采样和输出交付。另做 10,000 轮 reset、短接受／全拒绝交替：主干有效 KV 对齐非 MTP reference，额外 MTP 层 KV 对齐 MTP teacher-forced reference，使用上述数值容差；提交位置和有效长度完全一致。 |

交付顺序：真实 W4A8C8／TP16 主干已执行 8K 两步续算；先修复融合 GMM1 的同输入数值异常，再完成 32 步及 MTP 精度验收，采集完整 decode profiling，按热点实测 HBM、通信与相关算子，推进修改、融合并完成 2 ms 验收。AgentX 提交使用其正式 workload 与统计规则；这里的 8K TPOT 是开发指标。[AgentX methodology](https://inferencex.semianalysis.com/agentx/methodology)
