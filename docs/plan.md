# TileRT Ascend：A3 单用户 2 ms decode

目标：`a3-gs:trt-asc`，单机 8 模组／16 die，GLM-5.2-W4A8C8，batch=1，8K 上下文，TPOT ≤2 ms，即 ≥500 tok/s。vLLM-Ascend 执行 prefill，HIXL 交接 KV，TileRT-Ascend 执行 decode；单机两阶段依次占用设备，decode 计时期间 prefill 静默。

vLLM-Ascend P+D 基线已完成：同一 checkpoint 与 8K token IDs，TP16／DP1、MTP3／graph4，逐请求串行生成 1024 tokens；3 次预热、32 次测量，TPOT p50=16.39 ms、p95=18.29 ms，均值 16.51 ms，即 60.55 tok/s。16.39／2=8.20 是该基线到目标的压缩比，不能当作 TileRT 实测收益。同一服务在 temperature=0、seed=0 下仍产生 32 条不同轨迹，使用冻结 teacher-32 作为确定的输入与后继标签，不能把重新请求服务后的分叉归因于 TileRT。[原始结果与口径](evidence/ascend-decode/vllm-baseline-20261017/README.md)、[基线启动](../tests/glm52/serve_vllm_baseline.sh)与[测量入口](../tests/glm52/bench_vllm_decode.py)。

本次 P+D 基线在 `a3-gs:trt-asc` 空闲时复用已有镜像与统一 Python 环境。此前 TP16 短序列验证使用 `a3-187:trt-asc`。[环境快照](evidence/ascend-decode/a3-187-environment-20261016.json)、[运行时检查](evidence/ascend-decode/a3-187-runtime-20261016.json)

实施顺序：直接复用 CANN 9.2.0 已有算子，先跑通真实 W4A8C8、batch=1、TP16 decode；修正 HF 参考并重比五层，随后量完整 78 层曲线，再进入 profiling。逐层指标用于定位，端到端冻结轨迹用于放行。性能优化按实测调整 tiling、布局、流水与融合范围；能力缺失或已有实现难以达到预算时，再用 TileLang Ascend／AscendC 重写。[已有算子与精度入口](evidence/ascend-decode/cann-operators.md)

HF 独立差分保持首要诊断地位：SwiGLU 屏障和 Div 1-ULP 缺陷均由它定位。vLLM-Ascend 的同配置冻结执行可检查接线与状态；运行中的服务不作逐层、逐 token 等价靶子。先修参考再解释误差：冻结 BF16 P／V 的精确有理数点积证明，默认 CPU BF16 matmul 有 417／1024 个舍入错误，关闭 oneDNN 后全部消失。HF 的 prefix 展开、前向及探针投影统一关闭 oneDNN，保留 Div 校准；CANN 保留 SwiGLU 屏障。先重比冻结五层，再重跑 78 层，曲线出来前不调阈值。[参考误差证据](evidence/ascend-decode/mla-8k-differential-20261017.json)

[逐层换入验证](evidence/ascend-decode/streamed-decode.md)已在共享 A3 上执行真实 8K、TP16 的连续 layer 0–4，3,568/3,568 项执行与重建重放检查通过。原 RoPE 表前缀逐位一致地裁至 8,320 行，配合小消息 HCCL 缓冲，两轮结束后 Torch 分配量均为 245.5 MiB/die。此前 HF 的 70／80 采用有误的 CPU matmul 参考，不能用于验收。

[关闭 oneDNN 后的五层重比与 78 层三曲线](evidence/ascend-decode/hf-corrected-20261017/README.md)已完成，HF 执行与重放分别通过 59／59、1,587／1,587 项检查；五层与完整执行的前五层逐位一致。两步的 hidden NRMSE 峰值为 17.09%（L31）／20.66%（L38），末层为 9.04%／7.43%，仍有待定位的离散突跃，不能作为舍入底板。最终 logits 幅值为 23.75／23.125，BF16 间隔为 0.125，撤销固定 0.1 界，改报最大差／logits 极差与 top-1 余量。两步最终 top-1 均相同；这不等于八输入／32 步验收通过。参考修正与完整曲线已满足第 1→2 步条件，进入 profiling 和实际时延测量。

[最初的 M1 同配置执行与实测](evidence/ascend-decode/cann-profile-20261017/README.md)：原生 vLLM 与 TileRT 的 TP16、32 步张量和 reset 逐位一致。关闭 profiler 后，3 个 32 步请求的内部 TPOT 为 230.71／260.73／289.68 ms，中位数距 2 ms 预算约 130.4 倍；不含加载、KV 注入、reset、HTTP／HIXL。两步 trace 的 rank 0 设备跨度约 265 ms，其中归约区间约 215 ms，但 rank 14 的归约仅约 20–23 ms、区间内空隙约 184–185 ms，并在多数归约上最后提交。归约等待须与裸网络传输分开计量；该 M1 短测与 MTP3／graph4 的 1024-token 基线口径不同。

[M1 ACL Graph](evidence/ascend-decode/cann-graph-20261017/README.md)已完成实际执行与计时：16 rank 的 32 步 graph／eager、reset／eager、reset／graph 三组对照各有 100,352 个张量逐位一致；位置、长度、RoPE、int64 slot 与图内共享 indices 均随执行更新。无 profiler 短测中位 TPOT 为 44.68 ms，距 2 ms 仍约 22.34 倍。同版本 eager 对照为 256.84 ms；单独绑核为 246.65 ms，仅约 4% 差异且范围重叠。主机逐算子提交已被图替代，后续按 M4 路径的 trace 优化 CANN 算子与归约。

[M4 target graph](evidence/ascend-decode/cann-graph-20261017/README.md#m4-target-graph-and-session-timing)已通过 16-token 串行／投机／图／dirty reset 对照，三组各 47,088 行新 KV 逐位一致。32-token 短测中，eager 三次请求中位 TPOT 为 152.46 ms；target graph 三十次为 p50=38.70、p95=39.02 ms，每请求实际 A=31/17=1.824。最新[同源 1024-output 诊断](evidence/ascend-decode/cann-graph-20261017/README.md#m4-1024-output-diagnostic)预热三次、测量三次，TPOT 中位数 20.44 ms，距 2 ms 约 10.22 倍；每请求 303 轮，实际 A=1023/303=3.376，轮均值约 69 ms。该下降来自长轨迹接受数，代码未改。内部 callback 计时包含 seed、draft、verify、接受与交付，不含注入、reset、HTTP／HIXL；draft 仍为 eager。

[三轮分阶段 trace](evidence/ascend-decode/cann-mtp-profile-20261017/README.md)的后两轮 target 验证窗口约 49 ms，专家两类 GMM 合计约 13.3 ms；这些是 profiling 数字。原 vLLM 1024-output 基线为 16.39 ms，当前观测中位比为 1.247；其主干经过 torch.compile，我们只录制 eager 模型，归约 dtype、custom OPP 与调度也不同。[仅切 AIV 的长测](evidence/ascend-decode/cann-graph-20261017/README.md#ai_cpu-versus-aiv-control)保持全部 token、提议与接受一致，TPOT 中位数升至 21.47 ms；但[纯通信对照](evidence/ascend-decode/tp-mode-20261017/README.md)的 FP32 96 KiB、156 次图归约链由 AI_CPU 10.93 ms 降至 AIV 5.04 ms。[新 M4 trace](evidence/ascend-decode/cann-mtp-profile-20261017/README.md#aiv-decode-comparison)确认 AIV 实际生效，并在一次归约观察到 rank 14 晚到约 4.21 ms；尚未证明资源竞争。保留 AIV 路线，编译主干先作独立单变量对照。共有 GMM 成本不能当作双方增量差距；长诊断尚未完成 teacher 精度与正式 32 请求验收。

## 开发主线与停止条件

主线固定 M4，M1 保留为精度与性能对照。将设备执行看作串行 greedy decode 的精化。设 prompt 长度为 P，正常轮次提交边界的逻辑输出数为 N，主干有效长度为 L；最后一个输出是尚未被主干消费的 pending token。优化保持以下四个不变量。

| 不变量 | 形式与代码依据 |
|---|---|
| 前缀一致 | `L=P+N−1`，主干 KV 对应 prompt 和前 N−1 个输出；[完整 P 行注入](../tilert/models/glm_5_2_ascend/cann_model.py)后，首个 pending token 在位置 P 被消费。 |
| 接受等价于 greedy | `c[0]=pending`；若连续匹配 k 个 draft，则自然提交 `A=k+1≤M` 个 target predictions，并按输出预算／EOS 裁剪；`L'=L+A`，新 pending 为最后一个提交的 prediction。[接受状态](../tilert/models/glm_5_2_ascend/mtp_state.py) |
| MTP parent 正确 | 有效 MTP 行 p 对应 `(normalized_target_hidden[p], token[p+1])`。chain 行先截掉，再由 target hidden 重算；seed 后每轮提交的 main/MTP 同长。[轮次实现](../tilert/models/glm_5_2_ascend/mtp_round.py) |
| TP／PD 一致 | 16 rank 消费同一请求 generation，并按同一命令序列推进有效长度；本 generation 的全部 KV 写入完成后才能 inject。[TP 控制](../tilert/models/glm_5_2_ascend/tp_runtime.py)、[PD 接收](../tilert/pd_vllm/ascend_receiver.py) |

归纳步骤：设本轮 target 预测为 y，前 k 个 draft 满足 `c[i+1]=y[i]`。因 causal verify 的前 k+1 行与串行输入一致，`y[0:k+1]` 正是后续 greedy 输出；主干消费 `c[0:k+1]` 后仍留下最后一个 y 未消费，前缀不变量保持。由这些行的 target hidden 重算 MTP，再截断到 `L+A`，保持 MTP 不变量。现有[状态穷举](../tests/glm52/check_mtp_state.py)覆盖接受位置、parent 标签和 reset；真实 CANN 数值、因果索引及 HIXL 可见性由设备对照封闭。

性能以请求交付计时 `TPOT=(t_last−t_first)/(N−1)` 为准。稳态每轮只允许花 `2 ms×平均实际提交数`；初次 seed、控制与交付另占请求预算。本次单输入实测 A=3.376，轮预算为 6.752 ms。以 die 0 流式中位 1.155 TB/s 作场景标尺，轮读取上限约 7.80 GB；若再给 HBM 之外未被掩盖的工作留 1 ms，则降至 6.64 GB。[真实路由采集](evidence/ascend-decode/execution-gap-20261018/README.md#observed-expert-unions)的 303 轮中，target 四行专家并集均值 27.121，MTP 为 29.729，16 rank 记录逐字节一致。代入固定 8K 逻辑模型，每轮 6.833–7.481 GB、HBM 情景 1.753–1.919 ms/token，未重叠预算仅 0.247–0.081 ms/token；实际 HBM 读取、cache 复用及模型带宽另测。[表示依据](evidence/ascend-decode/cann-traffic-source-audit-20261017.json)

[执行边界对照](evidence/ascend-decode/execution-gap-20261018/README.md)：小包 top1 长测比同源全词表慢，保留全词表；单层驻留已在 TP16 跑通真实 AscendCompiler、Graph 与 reset。共享环境配对热图 p50：eager 4.112 ms、compiled 3.968 ms；30 对样本的配对差中位数仅 50.25 µs，编译快 20 对。先 profile 剩余层成本，再选融合；完整主干另作整机同源对照。TP16 完整 one-shot push 通过换输入、脏缓冲、延迟消费者及 epoch 复用；单 kernel 的十次依赖 SUM 热图 p50 为 38.059 µs，双 payload／READY 槽按因果关系消除 ACK 后为 35.716 µs，对照 BF16→FP32→HCCL→BF16 为 59.183 µs。原 14.334 µs 只读归约不代表完整协议。独立通信核的 flag／tiling 调优已停止；安装态生产端融合对照见下文，保留 8 µs 空闲定标及实际模型验证。

[现有 CANN attention 复用](evidence/ascend-decode/sfa-reuse-20261018/README.md)：保留连续 KV，去掉页址计算后，真实 layer-0 M4 完整热图的 p50 从 SFA 89.05 µs 降至 INT32 gather＋FIA 69.15 µs。两类内核各自 A/B/reset 的 819,200 个输出元素逐位通过；跨内核 NRMSE 为 0.000628，作诊断。主干和 MTP 已接入 `--attention-kernel gather_fia`，默认仍为 SFA；下一步验冻结 teacher，再做同源 1024 输出对照。该共享 die 结果不是模型 TPOT。已有 `GatherPaKvCache` 显式传零 offset 后正确，但合成 M4 完整图为 858.42 µs，停止该性能分支。

[单层短 trace](evidence/ascend-decode/execution-gap-20261018/resident-profile-audit.json)包含 embedding 和三次归约。共享环境中，eager 第 1 轮的 4.445 ms 跨度包含 3.657 ms 的 O 归约等待，实际 Transit 为 7.62 µs；三个 replay 均在首轮完成前提交。主机下发、实际计算和到达偏斜须分开归因。[安装态 O 投影＋归约对照](evidence/ascend-decode/execution-gap-20261018/README.md#o-publication-and-complete-reduction)否定了相对重编译控制的约 9 µs 收益：融合 push／pull 的配对中位收益为 −0.328／−2.129 µs。停止这条融合及并行发送调优。

[现有 CANN 静态内核＋SuperKernel](evidence/ascend-decode/execution-gap-20261018/README.md#static-kernels-and-superkernel)已在 TP16 M4 单层执行：每轮 6 个 SuperKernel（4 组跨算子）、28 个独立计算任务及 3 次独立归约；同口径计算任务从 44 降至 34，含 embedding。[实际产品代码复验](evidence/ascend-decode/execution-gap-20261018/static-super-product-audit.json)与原静态／普通编译版本五阶段张量逐位一致。精度与计时入口通过 `--target-graph --compile-target --super-kernel` 启用。[MoE layer 3 验证](evidence/ascend-decode/execution-gap-20261018/static-super-moe-audit.json)也已通过：真实 M4 输入及合成换序对照的 5,611,520 个比较元素与普通编译逐位一致，16 rank 均保留修复后的 GMM1。[MoE 短 trace](evidence/ascend-decode/execution-gap-20261018/static-super-moe-profile-audit.json)确认每轮计算任务由 27 降至 14（6 个 SK＋8 个独立任务），其中 3 组跨算子融合；2 次归约与独立 GMM1 保留。接着验证完整冻结轨迹并量同源长测收益。

[逐层驻留的 teacher 验证](evidence/ascend-decode/layer-major-20261018/README.md)保持 M1 与 hidden／residual 分离，逐 token 恢复共享索引。完整 78 层×32 步已执行，16 rank 的 logits 逐位一致，峰值约 897 MiB/die；BF16 head 为 31/32，位置 8211 的 279／64019 同为 16.75。生产 BF16 权重的精确分数为 16.69348／16.75541。现有 CANN BF16 输入／FP32 输出 head 按真实 `[9680,6144]` 分片重建后，M1／M4 均为 32/32；官方 TP16 M1／M4 processor／gather／裁剪也与同形状参考逐位一致，权重保持 BF16。`--fp32-head` 已接入主干及独立 MTP head，默认 BF16；本地 head Graph 已通过；[TP16 head Graph](evidence/ascend-decode/layer-major-20261018/head-tp-graph-audit.json)的 1,280 次变输入／重捕获重放也通过，投影、HCCL gather 与裁剪后保存的 475,791,360 个 FP32 logits 与参考逐位一致。[独立 MTP 自有头](evidence/ascend-decode/layer-major-20261018/README.md#mtp-own-head)也通过真实加载、M1／M4 head Graph 与逐位参考对照。[完整 MTP 前向 Graph](evidence/ascend-decode/layer-major-20261018/mtp-forward-graph-audit.json)的 seed M1、共享索引 chain M1 与 conditioned M4 也已逐桶通过，16 rank 的 64,979,328 个比较输出元素及新 KV 逐位一致。[链式图产品验证](evidence/ascend-decode/layer-major-20261018/mtp-chain-graph-audit.json)已通过真实 recycle 回喂、回退与重捕获；两个入口通过 `--draft-graph` 启用共享索引 M1 链，seed／conditioned M4 保持 eager。[两次链式调用配对计时](evidence/ascend-decode/layer-major-20261018/mtp-chain-timing-audit.json)的 eager／Graph p50 为 29.513／28.885 ms，但配对节省中位数仅 0.468 ms、19/30 胜且 Graph p95 更差，共享负载下未确认稳定加速。960 个样本的 C2 输出及有效 KV 逐位一致。现有 CANN rot／eh_proj 输出分片已通过完整 MTP 图／回退验证，投影微测 M1／M4 为 2.989／2.955→0.562／0.583 ms；[真实两轮链配对](evidence/ascend-decode/layer-major-20261018/README.md#mtp-output-projection-sharding)仅节省中位 0.522 ms、19/30 胜，未确认稳定净收益。跨配置 C2 recycle NRMSE 为 0.080302；[固定 C1](evidence/ascend-decode/layer-major-20261018/mtp-projection-fixed-c1-audit.json)已将起点定位到 eh 输出的四个 1-ULP 变化，精确点积确认这四个新值均正确舍入。当前测 teacher 轨迹提议覆盖，并以短 trace 分解 matmul、gather 与归约等待。[两轮合图原型](evidence/ascend-decode/layer-major-20261018/README.md#two-chain-calls-in-one-graph)的设备候选回喂与动态 metadata 已通过 TP16 A/B/reset，时延另测；完整轨迹保持第 4 节门槛。teacher 与 KV 来自不同配置、不同请求；驻留顺序不量 TPOT。

| 顺序 | 本阶段交付 | 转入下一步的条件 |
|---|---|---|
| 1. 修参考与定位 | 关闭 HF oneDNN，保留 Div 修正和 CANN SwiGLU 屏障，重比五层；随后提交 78 层 hidden NRMSE、cosine、logits 最大绝对差曲线，另报索引诊断与 logits 分辨率。 | 五层重比和完整三曲线完成、执行与 KV 状态正确，即可采集该输入的短 profiling；不要求逐层指标收敛到旧阈值。端到端冻结轨迹验收继续推进。 |
| 2. 定标成本 | M1／M4 短 profiling；profiler 关闭后记录轮耗时、实际 A、专家／KV 并集；实测 HBM 和 12／48 KiB HCCL。 | 用实际字节、带宽及暴露的通信／主机时间解释每轮成本，选出支配 2 ms 差距的一项。 |
| 3. 压缩关键路径 | 主机同步／启动占主导时，验证设备 indices 路径及已有 device GMM，再做固定阶段 Graph；流量占主导时减少复制读取并复用多行权重；归约占主导时融合生产端并重叠通信。每次只推进其中一项。 | 冻结轨迹正确、profiler 关闭的请求耗时下降，且校准后的预算仍能指向 2 ms；随后重测并选择下一项。 |
| 4. 完成验收 | 八个不同 8K 输入全部通过精度，预热后执行 32 个 1024-token 请求，保留实际接受数和交付时间。 | 请求 TPOT 的 p50／p95 均≤2 ms。 |

走读已确认两个候选主机同步点：默认 [MLA](../tilert/models/glm_5_2_ascend/ops/mla.py) 每层读取 indices 到 CPU；默认 [MoE](../tilert/models/glm_5_2_ascend/native_engine.py) 读取 routes 到 CPU 后逐专家执行。现有设备路径需先通过相同输入的数值验收，再以实际暴露耗时判断收益。

当前推进第 2 步：M1 复用 vLLM-Ascend 模型计算层及 CANN fused prolog，由 TileRT 管理完整 P 行 KV 注入与 decode 循环；首个输入是 prefill 已产出的 pending token，位置为 P。[薄适配器](evidence/ascend-decode/cann-model.md)的同配置原生执行与 reset 用于接线诊断，HF 保留独立数值定位。M1／M2／M4 已接入逐层 hidden、logits 和实际路由／索引捕获；[宽验证比较器](../tests/glm52/compare_saved_cann_verify.py)同样将跨配置 HF 差异保存为诊断，不再以旧逐层阈值放行。CANN MTP 的 seed、chain 和 target-conditioned 重算已通过上述 16-token 设备对照，完整 session 与长轨迹继续验证。后续任务须明确封闭哪个精度不变量，或预计减少多少 ms/round；若必要成本连 `A=4`、8 ms/round 都无法满足，就调整读取／同步机制。

精度定位按 `F_N(s_N)−F_H(s_H)=[F_N(s_N)−F_H(s_N)]+[F_H(s_N)−F_H(s_H)]` 分解；s 包含 hidden、有效 KV 和共享 indices。同输入逐层检查定位局部算术误差，独立轨迹保留传播结果。reference 修改先重比已有设备张量，生产路径修改后重跑设备检查。逐层指标指路，冻结轨迹决定端到端验收；完成参考修正和完整曲线后进入 profiling。

精度基线提前注册并初始化 vLLM SFA vendor。M1 AddRmsNorm 保留 `BF16(normalized) × gamma` 边界；CANN 普通 input norm 与 fused MLA prolog 按各自的 FP32 中间精度建立独立参考，[单 die 同输入证据](evidence/ascend-decode/mla-prolog.md)已封闭首个前处理分歧。各 rank 的 BF16 partial 转为 FP32 做 HCCL、最后转回 BF16；M=1／4 的基线消息为 24／96 KiB。通信优化以 BF16 传输、FP32 累加恢复 12／48 KiB 消息，融合保持上述数值合同。

[W4 MSD 合同](evidence/ascend-decode/w4-msd.md)保留两个 FP16 部分积及 FP32 的 `16H+L+checkpoint_bias` 合成顺序，32/32 同输入 M1／M4 投影逐位通过；[A8 合同](evidence/ascend-decode/a8-rounding.md)按 `Div(127,amax)` 生成倍率，184 个冻结样例逐位通过。[RoPE 合同](evidence/ascend-decode/rope-rounding.md)保留 BF16 系数、FP32 乘加及末尾一次 BF16 舍入，42 个冻结 index key 逐位通过；此前同输入检查仍有 38/1,572 项失败。继续用冻结输入和单 die 差分封闭 CANN 前处理、MLA 与专家链；在线 HIXL 交接后的 decode 续算另作端到端验收。[通信修复与设备证据](evidence/ascend-decode/pd-hixl-20261017/README.md)

## 1. 先算速度边界

先取 TP16、MTP 关闭、qA/kvA、router、indexer 复制的简单布局。模型有 78 层，前 3 层 FFN 为 dense，其余 75 层激活 8/256 个路由专家及 1 个共享专家；78 层 attention 均使用 top-2048，21 层重新计算 index。

| 输入 | 数值及依据 |
|---|---|
| A3 HBM | 本机每 die 64 GiB；带宽规格每 die 1.6 TB/s、每模组 3.2 TB/s，整机峰值 25.6 TB/s。[Huawei §3.3](https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1) |
| 权重容量 | 实际主权重 404.69 GB，另有 MTP 使用的 75.5 MB `rot.weight`；均分约 23.56 GiB/die。加复制权重、KV、workspace 后按每 die ≤48 GiB 设计。[header 审计](evidence/a3-model-bytes.json) |
| 运行时权重读取 | 每 rank、单 token 约 3.606 GB。当前 CANN 的复制 router 权重为 BF16、输出为 FP32；W4 保留 INT64 scale carrier 与 FP32 MSD 补偿，普通 W8 scale 为 BF16，Prolog scale 为 FP32，LM head 为 BF16。[可重现计算](../tests/glm52/estimate_traffic.py) |
| 单 rank 读取 | 权重 3.606 GB；MLA `78×2048×576×2=184 MB`；index KV `8192×(16×130+5×256)=27.5 MB`；合计约 3.818 GB。INT8 index 层携带 FP16 scale，其余 5 个主干 indexer 保持 BF16。[精度契约](evidence/ascend-decode/weight-contract.md) |
| AllReduce | 主干每层 attention、FFN 各一次，共 156 次，目标单 token 为 `6144×2=12 KiB` BF16；当前 FP32 精度基线为 24 KiB。同源 TP16 依赖链的图重放 rank-MAX event p50：24／96 KiB FP32 的 AI_CPU 为 10.925／10.931 ms，AIV 为 4.976／5.044 ms。该链包括重放、rank 偏差与依赖调度。[实测与复现](evidence/ascend-decode/tp-sum.md#matched-ai_cpu-and-aiv) |

流式读取模型给出 `T_HBM=3.818 GB/(1.6 TB/s×η)`：峰值带宽下为 2.39 ms，约 419 tok/s；取 70% 带宽为 3.41 ms，约 293 tok/s。加上匹配 AIV 的 M1 FP32 图归约链 4.976 ms，串行场景为 7.36／8.39 ms，即约 136／119 tok/s；理想重叠场景为 `max(T_HBM,4.976 ms)`，约 201 tok/s。这些是场景预算，图链包含重放提交、rank 偏差和依赖调度，不能当作裸 HCCL α 或模型增量耗时。[die 0 流式读取](evidence/a3-hbm-20261017.json)现测得中位 1.155 TB/s：四个 513 MiB 输入轮换，预热 100 次、测量 1000 次，15 组 checksum 通过；代入 3.818 GB 场景约需 3.31 ms。16 die 并行最低带宽及 W4 模型有效带宽另测。

因此，只消除 Python 提交开销仍不足以达到 2 ms。关键是复制的低秩投影、router、indexer、156 次归约，以及小 M 下的解包和计算效率。2 ms 路线从 MTP 共享权重读取、减少复制读取、tile 流水和生产端融合归约展开。

MTP 实现验证宽度 `M=2/4`：`T_round=T_verify(M)+ΣT_draft+T_accept`，稳态预算为 `ΣT_round/ΣA`；A 包含 target 修正／bonus token，M=4 时 `1≤A≤4`。稳态包含两次链式 draft 和一次 target-conditioned MTP4 重算。若每层专家并集为 32，逻辑读取约 7.30–7.94 GB/round；按实测 A=3.376 与 die 0 流读率，仅 HBM 已需 1.87–2.04 ms/token。该估算假设每组权重只读一次、KV 共享或独立读 1／4 次，不计 workspace 与激活搬运，不能替代流量实测。当前还有 166 次 FP32 归约及 4 次完整词表 all-gather；M1／M4 每 rank 的词表消息为 19,360／77,440 B，并非 top1 小包。初次 seed、采样和交付仍须计入请求 TPOT。[场景预算及源码绑定](evidence/ascend-decode/runtime-traffic.json)

## 2. 从真实接口确定工作量

本仓库是闭源 `libtilert_*.so` 的 Python 包装层：Python 管理权重、缓存和 PD，计算、通信及 engine 藏在库中。本任务是重建 NPU 上的执行与调度核心，优先调用已有 CANN 算子；需要自定义 tile 时再选择 TileLang Ascend 或 AscendC。当前 GLM-5.2 库为 ROCm／FP8、TP8，只接受 MTP 0/1/3；TP16、W4A8C8 和 NPU 同步机制需要重新实现。

审计基线为仓库 `f554fa57`、wheel `tilert-0.1.6.post3-cp312-cp312-manylinux_2_28_x86_64.whl`。三个库的 ELF 动态导出表均为空；结合 Torch 注册 schema、注册点反汇编和 Python 调用恢复接口，得到 DSV32 96 项、GLM5 37 项、GLM5.2 48 项，去重后 175 项。完整名称、参数顺序、返回类型和二进制地址见 [schema 清单](evidence/tilert-ascend/schemas.tsv)；shape、dtype、缓冲区副作用及分级见 [GLM-5.2 契约](evidence/tilert-ascend/glm52-contracts.md)。48 项包括 26 项计算／通信、12 项有状态控制、9 项查询、1 项 peer setup，覆盖整层和整网入口；未恢复的 native 内存序等字段在清单中标明。

| 分级 | 实现范围 |
|---|---|
| 可直接复用 | 模型结构、采样语义、Python reference 与 PD 控制逻辑；计算优先检查 CANN 的量化 matmul／GMM、norm、RoPE、cache 更新、indexer、稀疏 attention 与 HCCL。真实权重投影和单专家链已通过；其余实际形状逐项验证。闭源二进制可复用数为 0/48。 |
| 需重写 | 闭源 engine 的生命周期、权重／缓存地址管理及 NPU 调度适配；ROCm swizzle、FP8 cache 与地址表替换为 Ascend 布局。已有算子缺失或实测改造难以达标的计算项，单列依据后重写。 |
| 需新机制 | TP16 tile 调度、AIC/AIV/MTE 协作、计算与归约衔接、epoch 同步、设备侧 MTP 接受／回退，以及 HIXL→Ascend KV 注入。 |

## 3. 最小后端设计

1. 入口与基线：在统一 Python 环境、CANN 9.2.0 上以现有算子串联完整 decode；`glm_5_2_ascend` 与 backend loader 沿用 `prepare → inject → decode → reset` 生命周期。完成参考修正与固定 78 层、M=1 的三曲线后采集该路径 profiling；M=4 在 MTP 执行与状态检查后采集。最终数值放行使用第 4 节冻结轨迹。需要自定义 tile 时使用已固定的 [tilelang-ascend `ascendc_pto`](https://github.com/tile-ai/tilelang-ascend/tree/1d5daf0385156cbe0c499a16c10935c5190642a4) 或 AscendC。
2. 权重与并行：TP16，每 die 4 个 attention heads、每个专家 intermediate=128；每个 rank 持有所有专家的 TP 分片，直接处理选中专家。先复制 qA/kvA、router、indexer，记录其独占耗时，再比较低秩投影切分加 gather，以及独立 index rank 的净收益。按张量配方处理 scale、offset 与 scale_bias；down 的补偿只取本 rank 的一列。主干直接使用已导出的权重，MTP 在 hnorm 前应用 `rot.weight`；LM head 按 vocabulary 切分。
3. 执行调度：先使用已有算子的预打包权重和预分配缓存，验证 ACL Graph 对固定阶段的收益。重复搬运、独立启动或同步边界进入关键路径时，优先调整 CANN 实现的 tiling、布局、双缓冲和融合范围。需要跨算子的 tile 流水时，再以 AIC/AIV/MTE 和 cross-core flag 组织生产与消费；缓冲区按 `round/layer/tile` epoch 复用。
4. 通信：以 HCCL 给出 TP16 的正确性与时延基线，再比较适用的 AIV 路径。若归约等待主导，改造已有输出投影／expert down 与归约的衔接，并比较 residual/RMSNorm 融合收益；需要时以小型 AscendC extern 接入 device 通信。每项改造验证完成语义与关键路径净收益。HIXL 负责 P→D KV 交接。
5. 稀疏与 MTP：先复用已有 Q/KV 投影、norm／RoPE／cache 更新、indexer 和稀疏 attention，逐项验证 8K 选 2K、4 heads/rank 与 INT8/BF16 混合 index cache；再比较现有 fused prolog 和 SwiGLU／量化接口。共享层复用 indices。MoE 按 M 个位置的专家并集复用权重；draft、verify、accept 按实际接受长度提交 KV／位置并丢弃未接受尾部，每轮向 Python 返回 token 列表。
6. PD：复用 [Transport 接口](../tilert/pd_vllm/transport.py) 接入 HIXL 注册、批量 WRITE 与完成通知；[Ascend 专用 V1 connector／receiver](evidence/ascend-decode/pd-ascend.md) 使用 16 个 rank endpoint，独立于 GPU wire 的 8-rank 默认。开启 `indexer_kv_dtype=int8` 后，本 checkpoint 的 16 个主干 indexer 使用 INT8＋FP16 scale，0/1/2/70/74 及 MTP78 使用 BF16；MLA latent512+RoPE64 仍按 BF16。两端统一 page、逐层 dtype、scale、rotation、position 和 layout version。传主干 P 行、MTP P-1 行，全部 rank 完成同一 generation 才 ready；M1 直接消费 prefill 首 token。MTP 另传 12 KiB 的主干 normalized hidden[P-1]，在 D 端只补一行 MTP seed；[可选交接与捕获](evidence/ascend-decode/mtp-parent-handoff.md)及 [CANN MTP 请求接线](evidence/ascend-decode/cann-mtp.md)已实现。真实 8K HIXL 捕获、冻结前缀 M1／M4 与[在线 M1 HTTP 续算](evidence/ascend-decode/pd-live-20261017/README.md)已通过：同一请求 16 卡 WRITE 后执行 7 步，返回 8 tokens。单机保持接收 KV 驻留，P 退出后在待处理 inject 中加载 D；双模型共驻在 P 预热阶段实测 OOM。在线 MTP 另验。8K 前缀约 0.78 GB/rank；生成 1024 tokens、M=4 时按 `L≥8192+1024+3` 再向 page 对齐分配，权重与 workspace 按阶段释放。

已验证的起点：单 die 四层 BF16 缩小模型已执行，TileLang RMSNorm＋Torch/NPU 计算路径在 CPU、A3 上各通过 1,729 项检查；NPU 最大 NRMSE 为 0.008435，greedy token、索引、专家选择和 reset 重放通过。完整 W4A8C8／TP16／MTP 与 2 ms 性能验收按下表推进。[复现与原始结果](evidence/ascend-decode/README.md)

当前代码已接入 CANN TP16 的 M=1/2/4 验证、按专家分组、KV 截断、分布式 top1、MTP78、完整轮次编排与服务请求循环。layer 4 首个重放分歧已定位到 SwiGLU 写回后缺少向量流水屏障；[一行修复](evidence/ascend-decode/swiglu-barrier.md)随后在全部 16 rank 的[完整 8K 两步重放](evidence/ascend-decode/cann-full8k-barrier-20261017.json)中通过 38,048 项 reset 检查，输出 `[382,262]`。完整 vendor 仅改三个文件，保留原 host 库；[原生 Div 校准](evidence/ascend-decode/dynamic-quant-div.md)覆盖 31,744 个 BF16 amax。HF CPU matmul 已修正，下一步用实际时延和 trace 选择性能主线，保留曲线突跃供同输入定位。服务通过 `--mtp-width 1/2/4` 选择串行或 MTP，支持按实际交付裁剪取消请求的 KV；默认保留串行精度基线。MTP 缓存行 p 使用 normalized target hidden[p] 与 token[p+1]；链式 indices 复用，接受后按 target hidden 重算 MTP KV。[验证入口](evidence/ascend-decode/offline-development.md)

[测量入口](evidence/ascend-decode/measurement.md) 将精度报告与原始 8K prompt、权重、源码和版本绑定；各 rank 在执行线程中采集短 profiling，独立关闭 profiler 的 PD 请求同时记录生产端 callback 与客户端 token 到达时间。2 ms 判定要求实测覆盖八个不同的已验收输入。

## 4. 可独立验收

以下为开发验收预算。microbench 在空闲 `a3-gs:trt-asc` 上运行，固定实际权重分片及 M=1/4，保存命令、版本、字节数与原始结果；预热 100 轮、测量 1000 轮，取 16 rank 最大耗时的 p50/p95。每个优化项先给出 CANN 直接调用基线，再给出修改／融合或新实现的差分结果。HBM 使用超出 L2 的轮换数据；设备事件与整段 wall time 分别记录。HCCL 每批预热 20 次、测量 100 次，在计时窗外收集 16 rank 的批均值并取 MAX，重复 100 批统计分位数；另测 156 次依赖链。[定标入口](evidence/a3-roofline.md#本机定标)

| 验收对象 | 数字与判据 |
|---|---|
| HBM 与容量 | 报告各 die 持续读最低值和实际专家并集。M4／A=3.376 的总轮预算 6.752 ms；以已测 die 0 流读率换算，7.80 GB 仅够 HBM，另留 1 ms 后需≤6.64 GB。模型有效带宽、重复读取与激活流量另测。decode 权重＋KV＋workspace ≤48 GiB/die；P/D 共存时单 die 总峰值 ≤60 GiB。 |
| TP16 AllReduce | BF16 传输、FP32 累加，rank-MAX 批均值的 p50：12 KiB≤6 µs、48 KiB≤8 µs；同时记录 24／96 KiB FP32 HCCL 精度基线。测独立算子及 156 次依赖链，12 KiB 链≤1 ms；接入后报告暴露在关键路径上的时间。 |
| W4A8 MoE | 本地 intermediate=128，M=1／8 专家的 gate+up+SiLU+down≤20 µs；M=4／32 个不同专家≤45 µs；包含解包、动态 A8 与 scale_bias，不含 router/shared/跨卡归约。 |
| 稀疏 attention | M=1、4 heads、2K×576 BF16 的 MLA≤8 µs；完整 32-head indexer 在 8K 上投影＋打分＋top-2048≤20 µs。M=4 同时报专家／KV 并集与实际读取量。 |
| LM head | BF16 `[9680,6144]` 分片，M=1 的 norm＋投影＋local top1≤120 µs；跨 rank 选词另计。 |
| 接口覆盖 | 175/175 名称有归属；GLM-5.2 的 48/48 契约标明实现、等价适配或范围外。实际 decode 路径的算子及初始化／reset／采样／MTP 状态接口，样例通过率 100%，报告通过数／应实现数；4 个辅助 MFMA 入口及 DSV32/GLM5 专属入口单列，不计入已实现数。 |
| 数值 parity | 固定 8 个不同的 8K 输入，各 teacher-force 32 步，greedy 后继 token 与冻结标签 100% 一致，并检查 MTP 接受、KV 有效长度与提交位置。性能验收的 1024-token 轨迹，其 teacher-32 覆盖段须与冻结标签一致；扩展段保持串行／MTP 提交等价。HF 为首要诊断工具，逐层数值用于定位，不作放行闸门。 |
| 2 ms 目标 | 预热后串行运行 32 个请求，各输入 8K、输出 1024 tokens；每请求 `(t_last−t_first)/(N−1)` 的 p50/p95 均≤2 ms，同时报告逐 token 间隔、真实 A 和轮耗时，包含接受、采样和输出交付。另做 10,000 轮 reset、短接受／全拒绝交替：主干有效 KV 对齐非 MTP reference，额外 MTP 层 KV 对齐 MTP teacher-forced reference，使用上述数值容差；提交位置和有效长度完全一致。 |

逐层诊断按下表解释。修正后的 78 层曲线已完成，峰值仍远高于 L0 的 0.003595／0.004917；轨迹差异不能充当舍入底板，当前不抬高阈值来覆盖峰值。BF16 的 `u=2^-8` 是相对舍入尺度，不能直接等同任意张量 NRMSE 或逐元素 ULP 数。

| 诊断量 | 改判与依据 |
|---|---|
| 索引 | 取消集合相等的通过／失败判定。报告 top-2048 重叠数、切点分数间隙、错配槽位的实际注意力权重质量。[layer 1 反事实](evidence/ascend-decode/layer1-index-input-20261017.json)仅替换进入 norm 的实际输入，就由 2046 回到 2048／2048；索引变化须与输入轨迹分开定位。 |
| logits | 已测最终 BF16 间隔 0.125，大于旧 0.1 界；撤销该绝对判据，报告最大差／参考 logits 极差及 top-1 余量。两步最终相对差为 0.031716／0.025571，最终 top-1 相同。中间层 final norm＋LM head 是探针，其绝对差没有跨层不变的含义。 |
| hidden | 保留 NRMSE、cosine 曲线和相邻层增量用于定位；旧 1e−2／0.999 不作放行闸门，也不按当前 17–21% 的轨迹峰值拟合新阈值。先区分突跃与局部舍入底板，端到端按冻结标签验收。 |

修参考、五层重比、78 层三曲线、M1 profiling、M1／M4 target graph 数值对照与内部计时已交付；在线 HIXL→M1 decode→HTTP、M4 分阶段 trace、1024 输出诊断及 AIV 对照已完成。单层真实编译与 Graph 已验证并计时，FP32 head 已通过 TP16 M1／M4；独立 MTP 自有头也已通过；MoE 静态执行与完整 MTP 单桶图已通过数值验证；链式图真实回喂与 MoE 融合覆盖已验证；链式图与复制投影分片均未确认稳定整链收益；固定输入已定位 eh 的四个正确舍入变化及其传播；当前量 teacher 轨迹提议覆盖并用短 trace 找实际关键路径，整机空闲后测同源长轨迹。GMM1 baseN=128 的真实 M4 对照未见收益，停止该 tiling 分支；O 发布融合无净收益，已停止；完整双槽归约保留作空闲定标与模型对照。端到端八输入／32 步和长 MTP 轨迹继续验收；按热点实测 HBM、通信与相关算子，再推进修改、融合和 2 ms 验收。AgentX 提交使用其正式 workload 与统计规则；这里的 8K TPOT 是开发指标。[AgentX methodology](https://inferencex.semianalysis.com/agentx/methodology)

[实验日志](probe.md)记录最新测量与下一项实验；[经验](lesson.md)保留可复用的执行与数值结论。
