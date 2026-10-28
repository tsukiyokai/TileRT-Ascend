# 实施计划

先完成 GLM-5.2 / A3 的完整 decode 路径，对输出质量、请求状态和时延负责。首个目标采用者是已使用 vLLM-Ascend 的服务团队；交付同一个执行核心及其离线、在线入口。跨算子、跨层和跨卡控制权按完整请求收益与维护成本逐步扩大。项目定位见 [readme]，方法见 [lesson]，结果与证据缺口见 [probe]。

## CPU 前沿验收

CPU 接入、状态、计时修复及正负结果见 [10-28 账本](probe.md)；[同源码原生 lookup](evidence/ascend-decode/lookup-native-20261028/README.md)的四类本地总 CPU p50 均改善，编辑稳态仍有尾部退化，默认部署尚未安装扩展。[78 层合成 HF](evidence/ascend-decode/hf-runtime-20261028/README.md)已生成 [A/B/A 各 32 行的完整参考帧](evidence/ascend-decode/graph-frame32-20261028/README.md)，接收器拒绝空参考及身份类型混淆。下一步检验局部索引回收与请求观测成本，闭合实际 compiled/SuperKernel graph→HF 数值收集及正式 CANN 门禁；设备验证按[单 die、少 rank、空闲 A3](evidence/ascend-decode/online-lookup-20261028/README.md)推进，先核对构造副作用。整模型精度与在线 TPOT 仍待验收，当前未宣告 CPU 前沿耗尽。

## 首条验证路径

| 项目 | 范围 |
| -- | -- |
| 模型 | 真实 GLM-5.2-W4A8C8 checkpoint |
| decode 设备 | 一台 A3，16 die，TP16，单请求 |
| 基准输入 | 八个不同的 8192-token 输入 |
| 基准输出 | 每请求 1024 个 token |
| 时延目标 | TPOT 2.x ms |
| 服务目标 | 常驻 decode 接入成熟服务栈 |
| 扩展测量 | 短输入、长会话、EOS、取消、AgentX |

2.x ms 由通过精度与状态检查的完整请求测量验收；字节下界和逐层预算用于选择实验。保留 checkpoint 的稀疏 attention、量化、采样与 KV 语义；改变语义的候选须单独定义质量验收。

已有 [冻结前缀计时][width32]、[真实 P/D 功能链路][pd-live] 和 [原生执行片段检查][native-evidence]。各项结果的配置与范围见 [probe]。固定 8K/1K 测量与 AgentX 回放分别验收，均报告 P90 token 间隔、请求完成时间和每卡吞吐。

## 优先级与依赖

| 阶段 | 交付物 | 完成条件 |
| -- | -- | -- |
| P0 | 基线与支持矩阵 | 精度、状态可重现 |
| P1 | 首个服务接入 | 共用核心，常驻运行 |
| P2 | 更快的完整 decode | 整请求收益可重现 |
| P3 | 更大的执行与采用范围 | 新收益抵偿维护成本 |

局部开发和 CPU 检查可与 P0、P1 并行。进入默认后端的候选依次通过设备正确性、整模型和服务验收。首个外部贡献可以是一个片段；本项目仍对它在完整 decode 中的效果负责。

## P0：固定语义与可信基线

以 [CannModel] 为首条整模型验证路径，复用模型加载、量化和 KV 实现；[NativeDecodeEngine] 的独立检查用于算子验证。每份结果记录实际使用的引擎、入口、开关和加载库。

1. 固定 checkpoint revision、量化配方、全部权重文件清单与校验值，校验运行依赖。记录 CANN、驱动、torch/torch_npu、vLLM-Ascend、编译器、算子 overlay、源码与动态库身份。完整加载 78 层及所需 MTP 权重后建立基线。
2. 冻结八个输入、对应前缀、首 token 和请求标识，按 [输入与精度协议][measurement] 保存全部 rank 回执。加载、前缀恢复、预热和 profiler 分别计时。
3. 建立支持矩阵，登记模型、精度、TP/EP、上下文与容量、验证宽度、起草器、attention、图模式、KV 布局和采样。每个组合分别标注局部、整模型和在线验收状态。
4. 固定串行 CANN 对照，比较层输出、logits、top-k 顺序、路由、main/indexer/MTP KV 和完整输出。逐宽度、逐起草器定位首次分歧；分别检查 CPU 轮状态和设备因果 mask、slot、规约顺序。接受数、重复度和输出多样性只作诊断。

验收：八个输入均有完整 TP16 报告；reset 后重放一致，较短请求不读旧尾部，贪心输出符合固定串行参考。局部算术按登记的逐字节要求或误差阈值检查；整模型允许的差异须先定义误差与质量标准，阈值不随失败结果放宽。长上下文、长生成覆盖累计误差、近平局、重复输出和任务完成质量。在线支持范围以整模型与服务验收为准。

## P1：同一核心接入常驻服务

首个服务接入交付可替换的 decode 核心、支持矩阵、版本要求和验收脚本。离线测量与在线调用使用同一执行实现；接入方式由采用者的模型、缓存和部署约束决定。

| 接入方式 | 复用对象 | 重点验收 |
| -- | -- | -- |
| 框架内调用 | 模型、缓存、调度与服务入口 | 执行区域、图与状态生命周期 |
| P/D 后端 | prefill、路由及现有交接链路 | KV 传输、资源与常驻运行 |

[CANN 服务入口][serve-pd] 已将 `CannModel` 接入 session、TP command loop 和 HTTP，可用于现有 P/D 路径验收。框架内接入复用同一核心；采用者持有权重和缓存分配时，核心仍须明确暂存状态、提交、回滚和完成信号的责任。

先固定首个接入配置。P/D 部署分别记录 prefill、decode、路由的节点和设备；同机部署须覆盖加载与预热峰值。显存预算包含权重及转换副本、KV、通信缓冲、各宽度和 stream 的临时缓冲、图池与算子工作区。连续请求期间保持所需进程和模型就绪，去掉人工触发文件。

两种接入都须验证提交、回滚与完成信号。P/D 的传输与请求状态遵循 [P/D 状态协议][pd-contract] 和 [MTP parent 交接][parent-contract]：

- 接收与可见性：版本、checkpoint、请求、generation、rank 和布局一致。生产者写入完成并收到匹配回执后才能释放源页；16 个 rank 全部完成后才宣布 ready。
- main 与 MTP：输入长度为 P 时，交接主 KV 的 P 行和 MTP 的 P-1 行。首 token 仅交付一次；等待下一步消费的 pending token 与已提交 KV 分开。启用 parent 交接时，携带目标模型 P−1 位置的 final-norm hidden，已归一化且尚未应用 MTP 的 `rot.weight`。
- 投机提交：未接受的候选尾部在全部 rank 回滚。分别记录提出、验证、接受和交付数量，覆盖草稿零接受、部分接受、全部接受、EOS 和输出预算边界。
- 取消与重置：覆盖传输中、轮中、交付中及取消前后。按 pending/EOS 规则核对已交付范围、提交位置和有效 KV 长度；活跃 WRITE 完成前保留缓冲，失联写入隔离槽位，迟到消息不能复活旧请求。
- rank 与故障：命令序号、请求和位置一致。广播后失配时停止复用 TP 组，按明确的整组恢复流程重启；检查所有 rank 的正常关闭与资源释放。
- 服务能力：不支持的配置在分配或执行前拒绝，或按声明的规则路由至通用后端。满载有准入与背压；正常完成取消且写入已收束后，槽位可复用。

当前 Ascend connector 采用 TP16、batch1、完整新输入、128-token 页、顺序 MTP 和贪心采样。前缀复用、增量传输、LoRA、结构化输出及其他采样方式由采用需求推动，每项独立验收；增量传输须同时验证 KV 与投机回滚的生命周期。

验收：采用者通过真实服务入口完成八个冻结输入、长短交替请求和长会话。在线 M1 及声明支持的投机配置通过 P0 检查；服务输出、回调、KV 状态与 TP 回执相符。取消、超时、断连、拒收和关闭后可按协议继续服务。分别报告冷启动、TTFT、P/D 传输和稳态 decode 时间。

## P2：优化完整 decode

在同一基线上采集宿主下发、设备执行、同步、HBM 流量与通信时间线。每个候选说明要减少的等待、数据遍历或资源占用，并在完整请求中比较收益、兼容范围和维护成本。

1. 元数据与小算子：预留固定地址和图输入，批量读取索引与专家元数据，合并可共享读取的 norm、quant、cast 和投影前处理。变化位置、宽度和路由不得产生旧值；在相同地址和内存压力下比较完整路径。
2. 稀疏 attention：保持 top-k、因果 mask 和 cache 格式，比较批量 gather、地址计算、CUBE/AIV 交接与 projection tail。覆盖 INT8/BF16 indexer、分页映射、负 slot、越界输入和全部声明宽度；共享读取保留逐行选择语义。构建与算术要求见 [attention 接口][attention]。
3. MoE：复用权重，减少空组扫描和重复元数据遍历，组合 routed/shared、SwiGLU、requant、down 与 combine。保持实际 W4A8 NZ 布局、原始 top-8 slot、概率舍入、量化及 combine 顺序，覆盖变化专家分布；计时包含 dispatch、恢复和路由。实现要求见 [routed/shared][routed] 和 [完整 FFN][ffn]。
4. 计算与通信：比较生产者发布、传输、规约组合后的 HCCL 与 SHMEM 路径。实际 TP16 须保持选定 HCCL 的算术树和类型，覆盖重复图重放、立即消费者、跨轮变化 payload、取消与特殊数值。依次参考 [TP layer][tp-layer]、[三 rank 诊断][reduce-diagnostic] 和 [树规约整层检查][tree-check]；两 rank 相等不能替代 TP16 验收。
5. 代码生成：用同一真实 shape、布局和数值要求比较 TileLang-Ascend、AscendC 或 PTO 实现。生成代码、静态资源、同步与加载库均可追溯；工具链选择由片段及整模型结果决定。

原生片段保留存储 owner 和调用顺序。当前 Torch stream 的生产者先提交，消费者读取前结果可见；跨 AIC/AIV 与跨 die 完成协议分别验证。捕获前预留各宽度和 stream 的 workspace，图销毁后释放；UB 的 SDK 保留区和通信窗口计入容量。采用窗口 epoch 的协议须在设备重放时推进，跨轮不得读取旧 payload；跨调用保留的输出须独立持有或显式拷贝。adapter 对 KV transfer、权重重载等场景的限制保留到对应生命周期检查通过。

投机与执行联合优化。先用确定性目标检查预算变化下的候选、祖先、mask 与接受路径，再扫真实请求的宽度和起草器。接受数下降须结合目标输出与候选集合定位。分别记录 propose、verify、KV 提交/回滚、采样和交付时间，并保留输出。按通过质量检查的总墙钟时间和实际交付数选宽度，计入自动选择策略的开销。

验收顺序为实际权重局部比较、变化输入图重放、TP16 组合、整模型、P1 服务入口。交替运行对照和候选，固定输入、权重、起草器、宽度、cache、资源与计时范围，记录编译缓存和加载库。性能退化的候选留作实验结果；有新机制或新适用范围时再验证。

## P3：按收益扩大控制权与采用范围

从相邻 indexed/shared-indexer 层试验静态权重预取、workspace 复用、分块发布和计算通信重叠。保留值依赖、归一化、地址产生与可见性要求；用时间线验证提前量，计入 L1/UB/L2 容量、争用和额外同步。比较多 launch 的精细依赖与合并执行，只有完整请求收益抵偿兼容与维护成本，才扩大共同执行区域。

以相同请求和总资源比较 TP/EP 混合、热点专家复制与驻留策略，重新验证 shard、路由和规约。首个采用者通过后，由真实的第二模型、硬件或服务栈需求推动公共接口；仅提取两条已验证路径需要共享的部分。

最终交付包含支持矩阵、可部署环境、复现入口、精度报告和完整服务测量。记录总节点、卡与 die、并发、上下文、精度、实际接受数和计时分母。平均 TPOT 为全部有效输出间隔的总时间除以间隔数，并保留逐请求分布。AgentX 单独记录回放版本、会话、输出长度、任务质量及排队、prefill、decode、网络、工具时间；结合吞吐、资源成本和尾延迟确定采用范围。

[attention]: ../tests/glm52/whole_layer/attention_indexed.md
[cannmodel]: ../tilert/models/glm_5_2_ascend/cann_model.py
[ffn]: ../tests/glm52/whole_layer/ffn_original_shared_gu_static/README.md
[lesson]: lesson.md
[measurement]: evidence/ascend-decode/measurement.md
[native-evidence]: evidence/ascend-decode/megakernel-device-20261021/README.md
[nativedecodeengine]: ../tilert/models/glm_5_2_ascend/native_engine.py
[parent-contract]: evidence/ascend-decode/mtp-parent-handoff.md
[pd-contract]: evidence/ascend-decode/pd-ascend.md
[pd-live]: evidence/ascend-decode/pd-live-20261017/README.md
[probe]: probe.md
[readme]: readme.md
[reduce-diagnostic]: evidence/ascend-decode/megakernel-device-20261021/integration/communication-order-three-rank-v1-a3-194/AUDIT.md
[routed]: ../tests/glm52/routed_int4/README.md
[serve-pd]: ../tests/glm52/serve_cann_pd.py
[tp-layer]: ../tests/glm52/whole_layer/layer_tp_chain/README.md
[tree-check]: evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-tree-v1-a3-194/AUDIT.md
[width32]: evidence/ascend-decode/width-ladder-20261023/w32-lookup.json
