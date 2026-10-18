# TileRT Ascend：A3 单用户 2 ms decode

目标：A3 单机 8 模组／16 die，GLM-5.2-W4A8C8，batch=1，8K 上下文，TPOT ≤2 ms，即 ≥500 tok/s。当前实验机是 `a3-84:trt-asc`（镜像 `trt-asc:a3-84-unified-20261018`），`a3-gs` 已被外部作业占满。两台都是共享机：整机计时只在空闲时做，被占时推进单 die 与编译类工作。vLLM-Ascend 执行 prefill，HIXL 交接 KV，TileRT-Ascend 执行 decode；单机两阶段依次占用设备，decode 计时期间 prefill 静默。

vLLM-Ascend P+D 基线已完成：同一 checkpoint 与 8K token IDs，TP16／DP1、MTP3／graph4，逐请求串行生成 1024 tokens；3 次预热、32 次测量，TPOT p50=16.39 ms、p95=18.29 ms，均值 16.51 ms，即 60.55 tok/s。16.39／2=8.20 是该基线到目标的压缩比，不能当作 TileRT 实测收益。同一服务在 temperature=0、seed=0 下仍产生 32 条不同轨迹，使用冻结 teacher-32 作为确定的输入与后继标签，不能把重新请求服务后的分叉归因于 TileRT。[原始结果与口径](evidence/ascend-decode/vllm-baseline-20261017/README.md)、[基线启动](../tests/glm52/serve_vllm_baseline.sh)与[测量入口](../tests/glm52/bench_vllm_decode.py)。

同配置在空闲 `a3-84` 的复测给出 TPOT 中位 21.210 ms，对 `a3-gs` 的 20.441 ms 慢 3.8%，所以 20 ms 这一量级不是被他人占用造成的。在同一台空闲机上背靠背加开关：五个非编译开关（`--draft-graph --draft-projection-tp --attention-kernel gather_fia --fp32-head`）到 19.496 ms，再加 `--compile-target --super-kernel` 到 16.640 ms；在此之上把跨卡求和的展开方式设为 `HCCL_OP_EXPANSION_MODE=AIV` 得到 15.978 ms；再把 vLLM-Ascend 的 rope 派生缓冲改为连续（`rotary_embedding.py` 的 `chunk` 留下行跨度 128 的非连续视图，使 `_cos_cache[positions]` 每次构造 metadata 都物化整表、每轮搬 7.4 GB，比全部权重还多）得到 **14.540 ms（68.8 tok/s）**，距 2 ms 7.27 倍；该改动轮数与接受数不变（298／3.433），是纯速度收益。[rope 连续化](evidence/ascend-decode/rope-contiguous-20261018/README.md)该对照三组轮数与接受数全同（298／3.433），仅此一个变量不同，AIV 比 AI_CPU 快 0.463 ms、每轮快 1.59 ms；不设该变量等于 AI_CPU。这推翻了 10-17 在默认配置下「AIV 慢 5.05%、保留 AI_CPU」的结论。[展开方式对照](evidence/ascend-decode/expansion-20261018/README.md)，比默认降 21.5%，每轮 71.61→62.52→57.12 ms 降 20.2%；[留一法归因](evidence/ascend-decode/ablate-20261018/README.md)把五个开关分成两类：真加速只有 `--super-kernel` 1.807 ms 与 `--draft-graph` 1.681 ms（两者 A 与轮数均锁定在 3.433／298），这是 SuperKernel 首个实测时延收益；`--fp32-head` 1.317 ms、`gather_fia` 0.216 ms、`--draft-projection-tp` 0.100 ms 全是数值变更——去掉它们每轮反而更快 0.60／0.95／0.23 ms，收益百分之百来自接受数（分别掉 0.285／0.101／0.034）。即 4.570 ms 里约三分之一来自猜得更准，而 M=4 的接受数上限是 4、现为 3.433，这条路只剩 16%。`gather_fia` 还与选中它的筛选相矛盾：筛选测 89.05→69.15 µs，全模型里它使每轮慢 0.95 ms，另一次合成测量（ND 布局 137 对 88 µs）同向。对 vLLM-Ascend 基线 p50 16.390 ms 的比值为 1.015（默认配置时是 1.247），距 2 ms 由 10.61 倍收到 8.32 倍。`--compile-target` 此前不可达的原因是 `AOTAutogradCache` 要求 `OutputCode` 而 npugraph FX 编译器返回 `_CompiledFxGraph`，需 `TORCHINDUCTOR_AUTOGRAD_CACHE=0` 绕开。三组接受数分别为 3.376／3.207／3.433，轨迹已变，`precision_verified` 均为假：该组合仍欠第 4 节的冻结 teacher 与八输入验收。[三组对照](evidence/ascend-decode/compose-20261018/README.md)

本次 P+D 基线在 `a3-gs:trt-asc` 空闲时复用已有镜像与统一 Python 环境。此前 TP16 短序列验证使用 `a3-187:trt-asc`。[环境快照](evidence/ascend-decode/a3-187-environment-20261016.json)、[运行时检查](evidence/ascend-decode/a3-187-runtime-20261016.json)

实施顺序：直接复用 CANN 9.2.0 已有算子，先跑通真实 W4A8C8、batch=1、TP16 decode；修正 HF 参考并重比五层，随后量完整 78 层曲线，再进入 profiling。逐层指标用于定位，端到端冻结轨迹用于放行。性能优化按实测调整 tiling、布局、流水与融合范围；能力缺失或已有实现难以达到预算时，再用 TileLang Ascend／AscendC 重写。[已有算子与精度入口](evidence/ascend-decode/cann-operators.md)

融合的靶子已于 2026-10-18 修正，原来那个（把算子融在一起省开销）被自己的测量否掉。bf16 `torch.matmul` 在 `[4,2048]×[2048,6144]` 上已达 1117.3 GB/s，是实测上限的 93.8%；1 到 32 个背靠背的量化矩阵乘每次成本恒定（25.25 对 25.78 µs），下发早已被前一个 kernel 的执行掩住。所以融合在带宽和下发两个轴上都无可赚，而「AIC 高出流式下限 16.8 ms」这个归因建立在 ND 权重的 bench 上，生产路径的权重经 `maybe_trans_nz` 本来就是 FRACTAL_NZ，该结论不成立。[流式速率与首个手写 kernel](evidence/ascend-decode/matmul-streaming-20261018/README.md)

融合该瞄的是矩阵乘**周围**三件事：把激活留在片上、不再经每轮约 637 次 L2／GM 交接；把 TP 归约做成生产者的收尾（ROCm 那边 29 个 kernel 里有 6 族如此，而 A3 上已证没有厂商算子可用——`MatmulAllReduce` 无 ascend910_93 二进制、`MatmulReduceScatter` 要求 M 被 world_size 整除而 M=4／TP16 不满足）；以及按 FRACTAL_NZ 读权重。首个 TileLang kernel 一次就 bit-exact 但只有厂商的 0.18 倍，12 组分块全平在 190–198 GB/s，根因是行主序 `T.copy` 每段连续仅 256 到 1024 字节。[融合通信算子清点](evidence/ascend-decode/fused-collective-ops-20261018/README.md)

HF 独立差分保持首要诊断地位：SwiGLU 屏障和 Div 1-ULP 缺陷均由它定位。vLLM-Ascend 的同配置冻结执行可检查接线与状态；运行中的服务不作逐层、逐 token 等价靶子。先修参考再解释误差：冻结 BF16 P／V 的精确有理数点积证明，默认 CPU BF16 matmul 有 417／1024 个舍入错误，关闭 oneDNN 后全部消失。HF 的 prefix 展开、前向及探针投影统一关闭 oneDNN，保留 Div 校准；CANN 保留 SwiGLU 屏障。先重比冻结五层，再重跑 78 层，曲线出来前不调阈值。[参考误差证据](evidence/ascend-decode/mla-8k-differential-20261017.json)

[逐层换入验证](evidence/ascend-decode/streamed-decode.md)已在共享 A3 上执行真实 8K、TP16 的连续 layer 0–4，3,568/3,568 项执行与重建重放检查通过。原 RoPE 表前缀逐位一致地裁至 8,320 行，配合小消息 HCCL 缓冲，两轮结束后 Torch 分配量均为 245.5 MiB/die。此前 HF 的 70／80 采用有误的 CPU matmul 参考，不能用于验收。

[关闭 oneDNN 后的五层重比与 78 层三曲线](evidence/ascend-decode/hf-corrected-20261017/README.md)已完成，HF 执行与重放分别通过 59／59、1,587／1,587 项检查；五层与完整执行的前五层逐位一致。两步的 hidden NRMSE 峰值为 17.09%（L31）／20.66%（L38），末层为 9.04%／7.43%，仍有待定位的离散突跃，不能作为舍入底板。最终 logits 幅值为 23.75／23.125，BF16 间隔为 0.125，撤销固定 0.1 界，改报最大差／logits 极差与 top-1 余量。两步最终 top-1 均相同；这不等于八输入／32 步验收通过。参考修正与完整曲线已满足第 1→2 步条件，进入 profiling 和实际时延测量。

[最初的 M1 同配置执行与实测](evidence/ascend-decode/cann-profile-20261017/README.md)：原生 vLLM 与 TileRT 的 TP16、32 步张量和 reset 逐位一致。关闭 profiler 后，3 个 32 步请求的内部 TPOT 为 230.71／260.73／289.68 ms，中位数距 2 ms 预算约 130.4 倍；不含加载、KV 注入、reset、HTTP／HIXL。两步 trace 的 rank 0 设备跨度约 265 ms，其中归约区间约 215 ms，但 rank 14 的归约仅约 20–23 ms、区间内空隙约 184–185 ms，并在多数归约上最后提交。归约等待须与裸网络传输分开计量；该 M1 短测与 MTP3／graph4 的 1024-token 基线口径不同。

[M1 ACL Graph](evidence/ascend-decode/cann-graph-20261017/README.md)已完成实际执行与计时：16 rank 的 32 步 graph／eager、reset／eager、reset／graph 三组对照各有 100,352 个张量逐位一致；位置、长度、RoPE、int64 slot 与图内共享 indices 均随执行更新。无 profiler 短测中位 TPOT 为 44.68 ms，距 2 ms 仍约 22.34 倍。同版本 eager 对照为 256.84 ms；单独绑核为 246.65 ms，仅约 4% 差异且范围重叠。主机逐算子提交已被图替代，后续按 M4 路径的 trace 优化 CANN 算子与归约。

[M4 target graph](evidence/ascend-decode/cann-graph-20261017/README.md#m4-target-graph-and-session-timing)已通过 16-token 串行／投机／图／dirty reset 对照，三组各 47,088 行新 KV 逐位一致。32-token 短测中，eager 三次请求中位 TPOT 为 152.46 ms；target graph 三十次为 p50=38.70、p95=39.02 ms，每请求实际 A=31/17=1.824。最新[同源 1024-output 诊断](evidence/ascend-decode/cann-graph-20261017/README.md#m4-1024-output-diagnostic)预热三次、测量三次，TPOT 中位数 20.44 ms，距 2 ms 约 10.22 倍；每请求 303 轮，实际 A=1023/303=3.376，轮均值约 69 ms。该下降来自长轨迹接受数，代码未改。内部 callback 计时包含 seed、draft、verify、接受与交付，不含注入、reset、HTTP／HIXL；draft 仍为 eager。

修 rope 之后的[轮构成](evidence/ascend-decode/profile-rope-20261018/README.md)：每轮 SuperKernel 的 AIC 组 22.80 ms／351 次、AIV 组 11.73 ms／286 次、attention 3.04 ms、`Cast` 1.54 ms，算子合计约 42.5 ms 对轮长 49.92 ms，约 7.4 ms 不在任何算子里。SuperKernel 把每轮调用从约 2400 降到 637 组，但它不跨 cube/vector 边界（351 加 286 意味着每轮仍有约 637 次经 L2／GM 的交接）、不触及编译区域之外（`Cast` 仍 556 次／轮）、也不能重排数据流去预取下一层权重。那个「AIV 组双峰」的待定项已定案：`trace_view.json` 给出名字 `sk_1277_full_model_start_aiv_all_reduce_float_end_aiv_all_reduce_float`，而 637 个组里带 `all_reduce` 的 157 个全部同名同形、序号 1277 到 1589 间距一律 2。所以 `sk_1277` 不是形状特殊，它是**每轮的第一次归约**：16 个 rank 进轮时错开约 4.4 ms，第一次集合通信吸收掉这笔错位（每轮 4.36 ms，整轮 8.7%），其余 156 次因已对齐各只花 36 µs。根源是主机驱动轮次，与 draft 侧同一机理，算子融合碰不到它。[轮构成](evidence/ascend-decode/profile-rope-20261018/README.md#one-grouping-boundary-costs-436-ms-per-round)

[draft 侧集合通信](evidence/ascend-decode/draft-collectives-20261018/README.md)是当前最大的已定机理目标。原始逐 rank 分析把阶段跨度四分为计算、集合包络、AICPU 残差与真空闲，无需相减：round 2 为 72.732 = 48.037 + 16.210 + 1.528 + 6.958 ms，verify 的真空闲只有 1.014 ms（相减法会给出 7.855 ms，把三样东西糊在一起）。每个 draft 阶段仅 4 次集合（3 AllReduce + 1 AllGather）却耗 3.5–3.9 ms，逐次包络 0.723／1.248／1.656 ms 递增，`hccl_wait` 占 95.9%、transit 仅 0.6%；而 verify 的 157 次归约均摊 33.8 µs、等待约 26 µs。即 draft 里一次归约的等待是 verify 的 46 倍，每轮 9 次合计约 11 ms 只搬 72 µs 数据，占整轮 15% 且不是工作。机理是 verify 的归约被每层 0.26 ms 计算流水锁步，而 draft 的三次归约紧跟一次 host Gloo 广播、中间没有东西维持 16 rank 同步，递增则是同一落后者加累积延迟的签名；这也是 `--draft-graph` 值每轮 5.78 ms 的机理。候选修法：把单层 MTP 由 TP 切分改为复制，消掉全部 9 次 draft 归约，代价是每 rank 多读该层完整权重（专家部分约 151 MB，注意力份额与每 die 常驻字节待算，故标为待验证）。

[三轮分阶段 trace](evidence/ascend-decode/cann-mtp-profile-20261017/README.md)的后两轮 target 验证窗口约 49 ms，专家两类 GMM 合计约 13.3 ms；这些是 profiling 数字。原 vLLM 1024-output 基线为 16.39 ms，当前观测中位比为 1.247；其主干经过 torch.compile，我们只录制 eager 模型，归约 dtype、custom OPP 与调度也不同。[仅切 AIV 的长测](evidence/ascend-decode/cann-graph-20261017/README.md#ai_cpu-versus-aiv-control)保持全部 token、提议与接受一致，TPOT 中位数升至 21.47 ms；但[纯通信对照](evidence/ascend-decode/tp-mode-20261017/README.md)的 FP32 96 KiB、156 次图归约链由 AI_CPU 10.93 ms 降至 AIV 5.04 ms。[新 M4 trace](evidence/ascend-decode/cann-mtp-profile-20261017/README.md#aiv-decode-comparison)确认 AIV 实际生效，并在一次归约观察到 rank 14 晚到约 4.21 ms；尚未证明资源竞争。保留 AIV 路线，编译主干先作独立单变量对照。共有 GMM 成本不能当作双方增量差距；长诊断尚未完成 teacher 精度与正式 32 请求验收。

## 2 ms 的算术，以及为什么只剩一条路（2026-10-19 定）

一轮在 A=3.433 下可用 6.866 ms。三项各自的诚实值：

| 项 | ms/轮 | 修法 |
|---|---|---|
| 权重流 | 3.886（die 边际 1540 GB/s）… 5.873（wall 中位 1019） | 砍 1.449 GB 可到 2.939（乐观口径） |
| 157 次归约，单边 | 3.70（对 HCCL 5.08） | 已实测，但见下方正确性地板 |
| 主机下发路径 | 5.861 | 只有捕图 |

**任意两项已超预算，三项相加 13.4 ms（乐观端）。没有任何增量组合能到 6.866。** 所以路线必须同时做三件事：整条拿掉主机路径、把通信藏进计算而非仅仅缩短它、让权重流跑到 die 的边际速率。而这三件是一个改动——工作不再被切成分别下发的小块——它就是 megakernel。[骨架](megakernel-skeleton.md)

这比先前那个被夸大的集合通信定价是更强的论证：当时说集合通信值 8–10 ms/轮，对抗审计实测它只值 1.4–2.0；纲领没变，但理由从「最大单项」换成了「别无他途」。

## 主机下发（2026-10-18 夜定）

穷尽分解把每一轮拆到闭合后，**射程之外的主机地板是 5.861 ms/轮**——522 次下发类 API 的 5.503，加仅剩那一次 token 回读的 0.335，加 `advance` 的 Python 控制流 0.023。按 A=3.433 的 6.87 ms 轮预算，这是 85.3%，比 3.56 ms 的权重流更大；其中至少 0.404 ms 是严格暴露的（eager 区 154 个任务设备时间 1.220 ms，对自己的下发成本 1.624 ms，盖不住）。修法是把 eager 区捕进图，这是主机侧改造，再好的 kernel 也碰不到。[分解](evidence/ascend-decode/round-accounting-20261018/README.md)

同时三项旧结论作废（另见[对抗审计](evidence/ascend-decode/onesided-audit-20261018/README.md)推翻的两条）：不存在 49.92 ms 的 profiled round（开 profiler 是 133.93 ms，那个 7.4 ms 残差的分子分母不同源，重算是 10.1–10.4）；`aclrtSynchronizeStream` 的 24.255 ms 不是损失（13 次大调用全部在设备仍忙时返回，下一个设备任务 2.164–22.766 µs 内开始）；算子表里 4.32 ms/轮 是 `sk_1277` 吸收 rank 错位却被记成计算（156 个同名同族均值 36.0 µs）。

设备侧真正能碰的是：156 次组内归约现 5.635 ms。但按对抗审计的图内直测（32.340 µs/次），单边换下来只省 1.4–2.0 ms/轮，不是先前估的 4.1——那个估计用了被离群组污染的基线。奖品真实但比想象的小，而这恰好把重心推向 GMM 的 5.6 倍余量（约 7.3 ms/轮）与主机的 5.861 ms。

## 单边语义集合通信（2026-10-18 夜，最大单项）

权重预算重算后每轮 3.56 ms、占 2 ms/token 预算 6.87 ms 的 52%，余量 3.31 ms；而集合通信每轮 11.29 ms、与计算流重叠实测 0.000000 ms。所以这是棋盘上最大的单项，且它不是带宽问题：float32 `all_reduce` 在 4 字节要 157.43 µs、6 MiB 要 109.72 µs，跨 150 万倍不涨，**成本百分之百是会合**。[会合实证](evidence/ascend-decode/reduce-floor-20261018/README.md)

厂商路径全部封死：`MatmulAllReduce` 无 ascend910_93 二进制、`MatmulReduceScatter` 要求 M 被 world_size 整除而 M=4，而独立算子即使存在也无法与计算重叠。[清点](evidence/ascend-decode/fused-collective-ops-20261018/README.md)

CANN 的 SHMEM 是独立组件（gitcode `cann/shmem` v1.7.0），不在 CANN 装包里；拉下来 10 秒编完，16 PE 建对称域一次成功——这同时推翻「TP16 对称内存死于 507015」。线速实测**同模块 0.31 µs、跨模块 0.84 µs**，即 HCCL 的每次开销是其承载内容的 100 到 300 倍，而且它把真实存在的拓扑结构埋掉了。[安装与线速](evidence/ascend-decode/shmem-install-20261018/README.md)

★ **图捕获闸门已开**：一侧式对称内存操作在 A3 的 ACL graph capture 与 replay 下完整存活，证据是绕掉 capture 后故障一模一样（仅包装状态码从 507011 变 507015），以及修好后 400 次重放里 800 个含 put/get/signal/sync 的核逐轮 bit 正确。生产 decode 全程在图里，这曾是整条路线唯一的致命未知。

**一个 ABI 缺陷曾让这件事看起来不可能**：AscendC 核签名里 8 字节参数排在 4 字节之后会被设备从越界偏移读出，而库自己在 `aclshmemi_barrier_on_stream_kernel` 上犯了它——于是 `aclshmem_barrier_all()` 在这台机器上从未工作过（每次记 507015 但返回 void 吞掉）。我整晚的死锁就是它，且穿过坏 barrier 的所有测量都测了别的东西。给我们自己代码的规则：8 字节核参数一律排在前面。[ABI](evidence/ascend-decode/shmem-abi-20261018/README.md)

实测结果：裸 kernel **22.3 µs**（400 轮摊销、三次重复离散 0.6%、带「各 rank 贡献序号故和必为 120」的校验），对 HCCL 63.5 µs 图内 **2.84 倍**；16 核最优也是下界。两半分别为推送加同步 17.523 µs、本地求和 3.421 µs。而只同步不搬数据的地板是 **5.078 µs**，故推送侧有 12.58 µs 真实传输，RS+AG 把字节减到八分之一、代价多一次同步，预测约 16.5 µs（再快 26%）。[全归约](evidence/ascend-decode/onesided-allreduce-20261018/README.md)

torch 算子化后 TP16 实测 **28.181 µs 对 HCCL eager 142.850 µs，5.07 倍**，`wrong 0` 且与 HCCL **逐元素零差异**——后者才是允许换进「逐位比较 token」那条轨迹门的依据。接入点无需新管线：`cann_model.py` 早就把 157 次归约全部收束在 `_tp_sum_group._all_reduce_out_place` 这一个覆写上。[torch 算子](evidence/ascend-decode/torch-allreduce-20261018/README.md)

★ 2026-10-18 深夜续：照官方 `examples/shmem_mega_moe` 重写后再快 2.02 倍——库的 `mte_put_nbi` 换成 `aclshmem_ptr` 加普通 `DataCopy`（22.4→14.4），`PipeBarrier<PIPE_ALL>` 换成具名事件加双缓冲（→12.5），归约侧做成软件流水（→11.2），逐次 `probe_failures 0`。官方那个 4896 行 megakernel 全程只用三个单边调用，且通信与计算的重叠靠**按 sub-block 劈核**（一半常驻通信、一半驱动 cube），不是在同一个核上时分复用。[直接存储](evidence/ascend-decode/direct-store-push-20261018/README.md)

拿掉最后那道全核栅栏（改逐对端 `signal_op` + `signal_wait_until`）仍挂，已交由 agent 定位机制；候选是 `core_ratio(0,1)` 下 16 块各向不同对端发信号是否合法，以及 `quiet` 能否为 `DataCopy` 式远程写定序。

★ 2026-10-19 对抗审计定案：HCCL 图内实测 32.340 µs/次（边际 31.291，AIV 展开、production 旗标、一张图 156 次相互依赖的原地归约），所以倍数是 **1.44 倍**不是 2.84，单边收益 **1.4–2.0 ms/轮**。63.5 µs 是被单个 4.356 ms 离群组抬起来的均值。

而 11.2 µs 不是可交付的数：它测的 kernel 两处欠同步——写侧没有 MTE3 排空（`aclshmem_quiet` 在单机 HCCS 上是空操作、`sync_vec_all` 明说不完成未决通信），且双平面的奇偶在图重放下冻结（轮次令牌是宿主自增的 kernel 立即数，重放不跑宿主代码；draft 图三次归约是奇数，相邻两轮会共用平面）。正确性地板改为**单平面双同步**，奇偶随之消失；多一次 rendezvous（16 卡 5.016 µs）。校验也必须换：各 rank 贡献序号时全部 16! 种顺序都给 120.0，且 payload 四百轮不变对过期与串扰全盲。

端到端尚未报出，在它报出之前不计入收益。

开放路线登记在 [approach-registry.md](approach-registry.md)：A 通信消除、B 每轮 token 数、C 权重字节与驻留、D 轮次结构与主机税、E 注意力块、F 对抗审计，另有 G 并行布局（已 blocked：EP16 实测每轮 +1.09 ms、加融合再 +1.41，离散 0.252→0.756 ms 是 batch=1 尾部签名）与 H 为融合而融合（已收窄：bf16 矩阵乘已达峰值 93.8%、32 连也不省,只剩图内 8 µs 算子地板）。A、B、C 攻击同一个不等式的不同项，可以同时成立，不因另一个漂亮而放弃。

## decode 接管与融合的分层账（2026-10-18 定）

参照物是本项目 AMD 侧的 `libtilert_glm52_rocm.so`：459 个 kernel、29 个基名，形态是层粒度融合加「归约做成生产者收尾」加形状特化，而非单个大 kernel。注意它也不是「一层一个算子」——`pure_mla_moe_layer_kernel` 是 MLA 加 MoE 合体（48 变体），但 `sparse_index`（63+57 变体，全库最大）与 `topk_select` 自成一族，真实形态是每层两到三个 kernel、层间无裸露。[kernel 清单](evidence/tilert-ascend/libtilert_glm52_rocm-elf.txt)

按额分四级，逐级独立可验：

一、轮间主机税。每轮原有 4 次设备同步加 1 次 Gloo 广播；token 已改为驻留设备、命令可按块下发。隔离测得每轮主机命令值约 1.9 ms（32 轮一命令比 1 轮一命令快 0.88 ms/token）。[别名缺陷与测量](evidence/ascend-decode/devtok-aliasing-20261018/README.md)

二、MoE 块。TP16 下每层约 200 µs：两次 GMM 共 160 µs（其中 124 µs 是固定开销，且 k=128 的切片打烂 cube 效率）、permute/unpermute、36 µs 裸归约。EP16 加 `enable_fused_mc2=1` 把整串换成一个 mega_moe 融合算子，intermediate 回到 2048 同时解开 `finalize_routing` 的 k>128 封印。实验已就绪未测。注意终局反向约束：batch=1 无统计平滑，27.1 个激活专家撒 16 个桶，最忙桶期望 4.5–5 个完整专家，combine 是每层栅栏，临界路径为 TP16 的约 2.8 倍，每轮多约 3.7 ms。故终局候选是 EP4×TP4（尾部约 1.5 倍，intermediate/rank=512 恰踩 mega_moe 下限）或 batch=1 专属的热专家复制加两选一路由（HBM 余量约 44 GB/卡，复制一份 +22.6 GB/卡）。

三、注意力块的裸露部分。厂商已做块内融合（`MlaPrologV3`、`FusedInferAttentionScore`、`lightning_indexer`），裸露的是三样：每轮约 637 次经 L2／GM 的 cube/vector 交接、o_proj 之后 78 次没有厂商算子肯融的归约（`MatmulAllReduce` 无 ascend910_93 二进制，实测钉死）、以及缝隙里 556 次 `Cast`。这是手写 MIX kernel 的地盘：o_proj 矩阵乘加归约收尾加残差归一化，一个 kernel。归约收尾的配方已从 HyperMPMD 读出（UB 乒乓、MTE3 排空、fence、远端 signal-add，发送端零 DCCI，等待只自旋本地 flag）。[解剖](evidence/ascend-decode/hyper-mpmd-20261018/README.md)

四、层缝合。把注意力块与 MoE 块并成 ROCm 那种每层两三个 kernel。

四级全部兑现只清掉结构税；2 ms 终局还要压权重流式的下限，那是量化与驻留的另一仗。

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

[逐层驻留的 teacher 验证](evidence/ascend-decode/layer-major-20261018/README.md)已执行 78 层×32 步，保持 M1、hidden／residual 分离及逐 token 共享索引，峰值约 897 MiB/die。BF16 head 为 31/32；CANN BF16 权重／FP32 输出 head 在真实 `[9680,6144]` 分片上恢复 M1／M4 的 32/32，`--fp32-head` 已接入主干和独立 MTP 自有头。[TP16 head Graph](evidence/ascend-decode/layer-major-20261018/head-tp-graph-audit.json)通过 processor、gather、裁剪及 1,280 次变输入／重捕获重放。teacher 与 KV 来自不同配置、不同请求；驻留执行不量 TPOT。

[完整 MTP Graph](evidence/ascend-decode/layer-major-20261018/mtp-forward-graph-audit.json)的 seed M1、共享索引 M1、conditioned M4 已逐桶通过 eager／Graph 和有效 KV 对照。[产品链式图](evidence/ascend-decode/layer-major-20261018/mtp-chain-graph-audit.json)通过真实 recycle 回喂、回退和重捕获；`--draft-graph` 只启用共享索引 M1，seed／conditioned M4 保持 eager。两次链式调用的 eager／Graph p50 为 29.513／28.885 ms，但配对中位仅省 0.468 ms、19/30 胜且 Graph p95 更差，未确认稳定加速。

[MTP 投影分片](evidence/ascend-decode/layer-major-20261018/README.md#mtp-output-projection-sharding)的 M1／M4 微测为 2.989／2.955→0.562／0.583 ms；完整两轮链配对仅省中位 0.522 ms、19/30 胜。跨配置 C2 recycle NRMSE 达 0.080302；固定 C1 和精确点积已确认起点是 eh 四个正确舍入的 1-ULP 变化。[30 个 teacher 窗口](evidence/ascend-decode/layer-major-20261018/mtp-teacher-proposal-audit.json)中，五个提议槽位变化，但每个窗口的连续覆盖长度均保持不变；实际会话接受数另验。

[完整链短 trace](evidence/ascend-decode/layer-major-20261018/mtp-chain-profile-summary.json)已审计 TP16 的 128 个窗口、640 次 collective：head／router／MLA 存在毫秒级设备任务长尾，head 跨 rank 入口偏斜的保守下界仍超过 2.9 ms；尚未确定长尾原因。通信入口紧随本地 producer，跨卡时钟包络不足以将结束偏差解释为纯传输时延。[两轮合图原型](evidence/ascend-decode/layer-major-20261018/README.md#two-chain-calls-in-one-graph)已通过设备候选回喂、独立动态 metadata 及 TP16 A/B/reset；[合图配对计时](evidence/ascend-decode/layer-major-20261018/mtp-two-chain-timing-audit.json)仅省中位 0.089 ms、15/30 胜，停止默认集成；[PMU 复核](evidence/ascend-decode/layer-major-20261018/mtp-chain-pmu-summary.json)已匹配全部 7,936 个计算任务，但 10.52 µs 任务携带 4.81 ms 换算计数的反例使长尾归因仍未闭合。当前验证 GMM1 单分块同步补丁的完整 MoE M1／M4，再于整机空闲时测同源长轨迹。完整验收保持第 4 节口径。

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

单 die 的[形状扫描](evidence/ascend-decode/sfa-shape-20261018/README.md)把 attention 的成本拆成两项：SFA 在 top-k 256–4096 上拟合 `29.4 µs + 27.9 ns/行`，并且对 query head 不敏感（4→64 head、16 倍计算量只多 8%），所以它既非 compute-bound，也不因 TP16 的 4 head 而缺活。斜率折合 41.3 GB/s，仅为 1.155 TB/s 的 3.6%；同样的行用 `index_select` 读是平的（256→4096 行 23.4→24.0 µs，4096 行达 196 GB/s 未饱和），所以瓶颈是融合 kernel 内部的取数循环，不是散读本身。按 27.9 ns/行与约 600 ns 访存延迟估算，在飞请求只有约 20 个。另测 eager 连发的[每算子地板](evidence/ascend-decode/sfa-shape-20261018/README.md#per-kernel-floor)：`add` 从 1 元素到 49 KiB 恒为 10.09–10.25 µs，纯分配 4.90 µs，单行 `npu_rms_norm` 24.86 µs；图内 SFA 的 90.3 µs 与此处 88 µs 吻合，故其截距属设备侧，约 19 µs/次、78 层合 1.5 ms/轮。图内地板随后单独量出：[串行链回放](evidence/ascend-decode/graph-chain-20261018/README.md)在 8–256 长度上给出 1.079 µs/kernel（16 起边际稳定在 1.01–1.10 µs），对 eager 的 10.09 µs 少约 9 µs，正是 M1 开图 256.84→44.68 ms 的机理。按 profile 的族表，一次 M4 verify 实为 **2,416 次调用**（150 grouped GMM、234 MLA prolog/attention、255 量化投影、941 cast/quant/norm、225 MoE routing/combine、548 其它 traced compute、21 索引、42 RoPE，加总正好 40.698 ms 的 compute union），而非前五算子类型的 561 次；乘地板是 2.61 ms，占轮预算 6.866 ms 的 38%。所以下发数量不是把轮耗时钉在 57 ms 的原因，但它是 2 ms 预算里的一阶项，必须一起降；图内 SFA 的 90.3 µs 是该地板的 84 倍，成本在 kernel 内部。融合要以局部性与流量立论，不是省下发。同法扫 W4A8 grouped matmul，但首次扫描被 L2 污染并已作废（单个 256 组 bank 恰为 201,326,592 B，正是 die 的 192 MiB L2；其 64→128 边际 1261.2 GB/s 超过单 die 实测上限 1190.77 GB/s）。[轮转五个 bank 的重测](evidence/ascend-decode/gmm-l2-20261018/README.md)：`gate_up` 收敛到 1138 GB/s（上限的 95.6%）、固定成本 60–63 µs；`down` 618–705 GB/s、固定约 53 µs；≤32 组的点仍在 L2 内，而模型实测并集 27.121 正落在该污染区，故该形状下的逐调用拆分视为未测。按 60 µs 固定对约 19 µs 流式，减 MoE 字节在固定成本未降之前收益仍小。固定成本不在下发（图内地板 1.079 µs），而在算子内部：`loopCount==1` 时 `a8w4_msd_pipeline.h` 仍有 3 处设备级 `SyncAll`，其中两处之间只隔一个 AIV `pipe->Reset()`，且 `preProcess` 仍把激活的两个 int4 半边经 GM 交给 `midProcess`——该往返只为给 cube 引擎喂 int4 操作数，而 1–4 行不需要 cube。

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

参考修正、78 层曲线、在线 HIXL→M1→HTTP、M4 Graph／长测及短 trace 已完成。静态／SuperKernel、FP32 head、MTP 图与投影分片已逐项验证；新组合仍需完整冻结轨迹与同源长测。两轮 draft 合图未见稳定净收益，停止默认集成；PMU 已完整采集，计数不能独立归因长尾；当前验证 GMM1 单分块同步补丁的完整 MoE，随后依据无 profiler 对照选择下一项；GMM1 baseN=128 与 O 发布融合无净收益，停止对应分支；双槽归约保留作空闲定标与模型对照。端到端八输入／32 步及 32 个长请求均按上表验收。AgentX 提交使用其正式 workload 与统计规则；这里的 8K TPOT 是开发指标。[AgentX methodology](https://inferencex.semianalysis.com/agentx/methodology)

[实验日志](probe.md)记录最新测量与下一项实验；[经验](lesson.md)保留可复用的执行与数值结论。
