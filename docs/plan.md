# TileRT-Ascend：A3 单用户 2 ms 解码

目标：AgentX，A3 单机 8 模组／16 die，GLM-5.2-W4A8C8，batch=1，8K 上下文，TPOT 达到 2.x ms；开发验收继续采用 p50／p95 均≤2 ms。允许针对该形状优化。vLLM-Ascend 负责 prefill，TileRT-Ascend 负责 decode；正式打榜使用 [AgentX workload 与统计规则](https://inferencex.semianalysis.com/agentx/methodology)。

## 当前证据与优先级

截至 2026-10-19，已测最好请求 TPOT 为 11.6198 ms。pinned `seq_lens` 与 pageable 同会话交替，两对完整对照分别节省 3.384／3.791 ms/round；六个 pinned 请求均快于九个 pageable 请求。该组受共租户影响，运行利用率设为 0.78，pageable 为 12.3–12.6 ms/token；绝对最好值与配对差分别保留。[测量记录](probe.md)、[staging 机制](evidence/ascend-decode/metadata-h2d-20261019/README.md)

width-8 MTP 是当前控制组，单一冻结输入的平均实际提交数约 4.507，2 ms 对应稳态轮预算约 9.01 ms。旧相位参考为 verify 39.68、chain 10.27、conditioned 2.08 ms/round；这些来自旧配置，须随新组合重测。[宽度扫描](evidence/ascend-decode/width-sweep-20261019/README.md)、[同步相位](evidence/ascend-decode/phase-synced-20261019/README.md)

| 顺序 | 路线 | 下一项与通过条件 |
|---|---|---|
| 1 | lookup 提议与执行路径 | CPU 状态／串行对照通过；等价增量索引将每请求 CPU 查找约 1.03 s 降至 21 ms。独立开关 shortest2→1 的 width32 重放为 174→147 轮。下一步测 width 8／16／32 的真实接受数和请求 TPOT。该轨迹 A=6.959 对应 2 ms 的轮预算 13.92 ms；设备收益尚未测得。[重放](../tests/glm52/ngram_acceptance.py)、[实现](../tilert/models/glm_5_2_ascend/lookup_draft.py) |
| 2 | 多 query／跨层共享 | 可选 `shared_fia` 支持 M8／M32：完整 KV 跨 query 复用，21 个 producer 构建选择 mask。四层完整 Graph 配对改善14.6%／37.0%，M32 为657.2→416.1 µs，A/B/reset 对重建FIA逐位一致；M1／MTP保持SFA。下一步以冻结teacher／串行等价开路，再与lookup宽度对照测整轮；旧gather_fia M≤6门槛不变。[证据](evidence/ascend-decode/fia-mask-20261020/README.md)。 |
| 3 | 投影与归约流水 | 既有真实 O 投影＋完整归约对照：原装 CANN＋push16 为162.141 µs，融合 publication＋consumer 为162.781 µs，未减少 launch、无净收益。关闭该分支；须提出消除后续 consumer 边界的新机制再重开，停止修补不含 scale／有效 tiling 的 INT8 演示。[完整反证](evidence/ascend-decode/execution-gap-20261018/README.md#o-publication-and-complete-reduction) |

每条路线一次只改一个可归因因素。按新策略实际经过的位置、宽度和专家并集计价；减少 launch 数量、外推行成本或套用 64-expert 数字均不能证明端到端收益。lookup 的精确 CPU 重放及外推撤回依据见[复制提议](evidence/ascend-decode/lookup-drafting-20261019/README.md)，须由与部署一致的重放及设备对照更新。

## 执行不变量

设 prompt 长度为 P，已产生 N 个逻辑输出，主干有效长度为 L；最后一个输出 pending 尚未被主干消费。

| 对象 | 必须保持的状态 |
|---|---|
| 主干前缀 | `L=P+N−1`，有效 KV 对应 `prompt + output[:-1]`；完整 P 行注入后，首个 pending 在位置 P 消费。 |
| 接受与截断 | `c[0]=pending`；连续匹配 k 个提议后，自然接受 `k+1≤M` 个 target predictions，按预算／EOS 裁剪后实际提交 A 个；`L'=L+A`，最后一个已提交 prediction 成为新 pending。只保留正确输入前缀的 KV。 |
| lookup | context 始终为 `prompt + output`，包含 pending；短提议按实际实现补齐。未执行 MTP 时，draft plan／commit 为空，实际 MTP KV 保持注入长度，不得声明其已重算。 |
| MTP | 有效行 p 使用 `(normalized_target_hidden[p], token[p+1])`；链式行在 target-conditioned 重算后截断，正常提交后 main／MTP 同长。 |
| TP／PD | 16 rank 使用同一 request generation 和命令顺序；本 generation 的全部 KV 写入完成后才 ready／inject。 |

状态与因果归纳由[接受状态](../tilert/models/glm_5_2_ascend/mtp_state.py)、[轮次实现](../tilert/models/glm_5_2_ascend/mtp_round.py)、[独立标量检查](../tests/glm52/check_mtp_round.py)和[设备串行对照](../tests/glm52/run_cann_mtp_rounds.py)承接；host 检查不替代真实 CANN 因果性、数值和 HIXL 可见性验证。

混合 lookup／MTP 的两个分支需要按深度设置 RoPE position、按祖先限制 indexer 与 attention，并将选中路径的 KV／indices 提交为连续前缀。只增加 attention mask 不足以保持上述状态。轮次选择 lookup 后再切回 MTP，须先从真实 target hidden／后继 token 补齐缺失的 MTP 行；需跨图保存的 hidden 必须脱离复用输出缓冲区。

离线混合接受率必须沿自身提交边界推进，并取得每个实际到达前缀的 MTP 提议。单次运行仅在原轮次边界记录的 `candidates`／`predictions` 不覆盖新边界；首个不匹配后的 predictions 也不是串行轨迹。接受率确定后，另测新策略的轮成本和实际专家并集。

按该约束实测后，混合树暂缓：把 227 个已记录 MTP 匹配长度按四种顺序发牌，k=31 的合并接受数为 8.317／10.129／11.494／13.286，而相对单独 lookup 的保本比为 1.303（来自 chain 占实测轮 23.3%，与行成本外推无关），发牌区间 1.195–1.909 跨过保本点；边界配对 r=+0.358 为正相关，进一步压低期望最大值。重开条件是能在 chain 未到达位置给出其匹配长度的机制，而非对同一组数字再发一次牌。最大后缀已首次扫描：longest 4–16 持平、≥24 单调变差，保持 8。[混合与后缀证据](evidence/ascend-decode/merge-tree-20261020/README.md)

生产 indexer 输出任意 token IDs。块化只有在每个 query 经因果裁剪后的稳定展开逐位等于原选集时才无损；不能由索引跨层复用频率推导块结构。C1 候选保留原选集与顺序，另核验 kernel 流水排空的索引读取边界。

## 后端与数值合同

保留 `prepare → inject → decode → reset` 生命周期，优先使用 CANN 的 matmul／GMM、norm、RoPE、cache、indexer、稀疏 attention 与 HCCL；实际形状的差分和计时决定自定义 AscendC／TileLang 的范围。本轮实读 a3-84／187 激活 toolkit 为 9.2.0、V100R001C25B114；历史 9.1.0 记录不能代替当前包身份；每次绑定实际包、源码、kernel 和编译缓存产物。

闭源包审计基线为 `f554fa57`／`tilert-0.1.6.post3`：DSV32 96、GLM5 37、GLM-5.2 48 项，去重 175 项；NPU 可直接复用的闭源二进制为 0/48。48 项含计算／通信 26、状态控制 12、查询 9、peer setup 1。[完整 schema](evidence/tilert-ascend/schemas.tsv)、[GLM-5.2 契约](evidence/tilert-ascend/glm52-contracts.md)

- TP16：78 个主干层，前 3 层 dense，其余 75 层 top-8／256 routed experts＋shared expert；每 die 4 attention heads，专家 intermediate=128；当前选集为 top-2048，21 层重算 index。[权重与布局](evidence/ascend-decode/weight-contract.md)
- W4 MSD：两个 FP16 部分积，按 FP32 `16H+L+checkpoint_bias` 合成；A8 倍率使用 `Div(127,amax)`；RoPE 使用 BF16 系数、FP32 乘加、末尾一次 BF16 舍入。[W4](evidence/ascend-decode/w4-msd.md)、[A8](evidence/ascend-decode/a8-rounding.md)、[RoPE](evidence/ascend-decode/rope-rounding.md)
- TP 归约：BF16 partial 转 FP32 累加，再转 BF16。融合允许改变边界，须保持此算术语义；通信候选可传 BF16、累加 FP32。FP32 head 保留真实 `[9680,6144]` 分片的冻结 teacher 对照。[head 验证](evidence/ascend-decode/layer-major-20261018/README.md)
- PD：统一 page、dtype、scale、rotation、position 和 layout version；传主干 P 行、MTP P−1 行与 12 KiB normalized hidden[P−1]。MLA latent512＋RoPE64 为 BF16；16 个主干 indexer 为 INT8＋FP16 scale，0／1／2／70／74 和 MTP78 为 BF16。[交接合同](evidence/ascend-decode/mtp-parent-handoff.md)、[CANN 请求接线](evidence/ascend-decode/cann-mtp.md)
- 容量：主权重 404.69 GB，另有 75.5 MB `rot.weight`；decode 权重＋KV＋workspace≤48 GiB/die，P/D 共存总峰值≤60 GiB/die。完整末轮 verify 也须有空间；当前容量 9344 支持 8K＋1024 输出及 M≤32。单机 P 退出后加载 D，保留接收 KV。[容量审计](evidence/a3-model-bytes.json)、[在线交接](evidence/ascend-decode/pd-live-20261017/README.md)

HF 保持独立同输入诊断。hidden 的 NRMSE／cosine 曲线、logits 极差／top-1 余量、索引重叠／切点间隙用于定位；不得以轨迹差异提高阈值。reference 修正先重比已有设备张量，生产路径修改后重跑设备检查。冻结标签、串行提交等价与有效状态决定验收。[精度定位](evidence/ascend-decode/offline-development.md)

## 验收

microbench 固定实际权重分片与 M=1／4，预热 100 轮、测量 1000 轮，报告 16 rank 最大耗时的 p50／p95；保存直接 CANN 基线与候选差分，分别记录设备事件和 wall time。HBM 使用超出 L2 的轮换数据。HCCL 每批预热 20 次、测量 100 次，在窗外取 16 rank 批均值的 MAX，重复 100 批；另测 156 次依赖链。[定标口径](evidence/a3-roofline.md#本机定标)

| 对象 | 保留的开发验收预算 |
|---|---|
| HBM／容量 | 报告各 die 持续读最低值、模型实际读取与专家并集。历史 M4、A=3.376 的轮预算 6.752 ms：按 die0 流读率，7.80 GB 仅够 HBM，另留 1 ms 时≤6.64 GB；新策略按实际 A 重算。容量上限按前述 48／60 GiB。 |
| TP16 AllReduce | BF16 传输、FP32 累加：rank-MAX 批均值 p50，12 KiB≤6 µs、48 KiB≤8 µs；记录 24／96 KiB FP32 HCCL 基线。12 KiB 的 156 次依赖链≤1 ms，并报告模型关键路径暴露时间。 |
| W4A8 MoE | intermediate=128；M1／8 专家的 gate＋up＋SiLU＋down≤20 µs，M4／32 个不同专家≤45 µs；含解包、动态 A8、scale_bias，不含 router／shared／跨卡归约。 |
| 稀疏 attention | M1、4 heads、2K×576 BF16 MLA≤8 µs；完整 32-head indexer 在 8K 的投影＋打分＋top-2048≤20 µs。M4 报告专家／KV 并集与实际读取。 |
| LM head | BF16 `[9680,6144]` 分片，M1 norm＋投影＋local top1≤120 µs；跨 rank 选词另计。 |
| 接口 | 175/175 名称有归属；48/48 GLM-5.2 契约标明实现、等价适配或范围外。实际 decode、初始化／reset／采样／MTP 状态样例通过率 100%，报告通过数／应实现数；4 个辅助 MFMA 与 DSV32／GLM5 专属入口单列，不计入已实现数。 |
| 数值 | 8 个不同 8K 输入各 teacher-force 32 步，greedy 后继 token 与冻结标签 100% 一致；检查 MTP 接受、有效 KV 长度和提交位置。1024-token 性能轨迹的 teacher-32 段仍与冻结标签一致，扩展段保持串行／MTP 提交等价。HF 逐层数值用于定位，不作放行闸门。 |
| 请求性能 | 预热后串行运行 32 个 8K 输入／1024 输出请求，覆盖八个已验收输入；每请求 `TPOT=(t_last−t_first)/(N−1)` 的 p50／p95 均≤2 ms。报告逐 token 间隔、实际 A 和轮耗时；接受、采样、控制与交付均在请求计时内，关闭 profiler。 |
| 长稳态 | 10,000 轮 reset、短接受／全拒绝交替；主干有效 KV 对齐非 MTP reference，额外 MTP KV 对齐 MTP teacher-forced reference，使用既定数值容差；提交位置与有效长度完全一致。有限值检查不能代替数值等价。 |

稳态成本按 `ΣT_round/ΣA` 计算，请求预算还须容纳首次 seed 和交付。验收记录绑定八个原始 prompt、checkpoint、prefix／parent、源码、包版本与实际加载产物；host 标量检查、单算子通过、短轨迹或单输入最好值均按其覆盖范围报告。[测量入口](evidence/ascend-decode/measurement.md)

## 实验执行

每次从 `lsenv.md` 选择最空闲且栈完整的 A3，在 `trt-asc` 容器执行；候选包括 `a3-84` 与 `a3-187`。每臂重新检查全部 16 die 的 HBM／AICore；机器被占时推进源码、构建和 host 验证。整机发射统一使用 [launch_arm.sh](../tests/glm52/launch_arm.sh)／[chain.sh](../tests/glm52/chain.sh)，每臂独立缓存与 HCCL 端口区间，记录实际加载 kernel。

同一会话交替 A/B 并配对，保留重复控制臂；约 0.2 ms/round 的效应须对照 run 间漂移。采短 trace 与无 profiler 计时分开；按实际 critical path 与逐流计算并集核对相位，重叠工作只计一次。[归属依据](evidence/ascend-decode/phase-attribution-20261019/README.md)

实验结果见 [probe.md](probe.md)，可复用结论见 [lesson.md](lesson.md)，发射与取证步骤见 [workflow.md](workflow.md)。
