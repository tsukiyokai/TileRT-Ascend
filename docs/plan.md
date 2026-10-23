# TileRT-Ascend：A3 单用户 2 ms 解码

目标：AgentX，A3 单机 16 die，GLM-5.2-W4A8C8，batch=1，8K 上下文，TPOT 2.x ms。
vLLM-Ascend 负责 prefill，TileRT-Ascend 负责 decode。

TPOT = 一轮时间 ÷ 一轮接受数。只有两根杠杆，而两者都已被量到边界。

## 一轮是设备时间

宿主把整轮 1848 个 kernel 下发完只用 2.14 ms（约 1.1 µs 一个），随后在唯一一次
同步上等 50.45 ms。一轮的分相：propose 0.037、verify 2.138、host_tokens 50.454、
resolve 0.090 ms，各一次标记，合 52.720 对实测 53.9。

所以"固定项是宿主下发次数，减 kernel 数即可"这条读法作废；28.8 µs 一个 kernel
是设备侧小 kernel 的地板，不是下发费。
[证据](evidence/round-is-device-20261023)

## 一轮 53 ms 的绝对账单：55% 是标量

2026-10-23 第一份稳态解码轮的 profile（宽度 12、自动机、sfa、51 轮、
2579 次下发一轮即 33.1 次一层、260 个不同 kernel）：

    Duration            53.052 ms
    aic_mac              1.158    2.2%   cube 真正在乘的时间
    aic_mte2             8.236   16%     权重读
    aic_scalar          15.480   29%
    aiv_vec              1.752    3.3%
    aiv_scalar          13.471   25%

标量合计 28.95 ms（55%），算术合计 2.91 ms（5.5%）。三个最大 kernel 占全轮
42%，而它们的 mac 占比只有 2–6%、`aicore_time` 占 85–90%——核在忙，不在乘：

     ms/轮   核    mac   mte2  标量  向量标量  kernel
     8.840  7.471  0.212  1.286  4.855  2.036  GroupedMatmulSwigluQuant（gate_up）
     7.089  3.895  0.392  1.703  1.402  2.575  SparseFlashAttention
     6.566      -      -      -      -  1.399  157 个 SuperKernel（规约）
     6.554  5.890  0.106  2.934  5.089  3.670  GroupedMatmul（down）
     3.942  3.178  0.133  0.805  2.042  0.515  QuantMatmul，255 次一轮
     3.055  2.748  0.169  0.964  0.784  1.109  MlaPrologV3
     2.968      -      -      -      -  0.139  InplaceCopy，531 次一轮
     2.288      -      -      -      -  0.040  Index/Slice，18 次一轮

两个 MoE grouped matmul 自己就是 15.65 ms 标量，占一轮 30%。batch=1 下每层
路由碰约 81 个专家而总共只有 12 行，所以组循环为几乎不存在的算术付了 81 份
标量开销——这也解释了为什么"专家库收窄"按字节算是零收益却仍然贵：代价按
**看过的组数**计，不按读过的字节计。

若标量归零，地板是 mac + mte2 约 8–9 ms 一轮，按 2.59 接受数即 3.4 ms，
换更好的起草器就进 2.x。**所以 2.x 只能靠砍标量拿到。**

"一层 = 12 µs × pass 数 + 72.7 µs" 这个模型不作废但降级：12 µs 的 pass 地板
本身主要是标量启动开销，所以"数 pass"是"数标量"的影子。实测一层 680.2 µs，
对模型给的 650.4 µs。

## 接受数已接近天花板

同一条真实 token 流上离线比较，排序在每个宽度都是 lookup < 自动机 < 树：

    宽度 12   lookup 4.900   自动机 5.565   树 6.059
    宽度 32   lookup 5.689   自动机 6.872   树 8.258

该口径比引擎实测保守约 10%，排序稳定。复制天花板（含起草器自己已提交的输出）
为宽度 8/12/32 上 5.652/6.959/8.819。加宽本身不解决问题：接受数次线性涨而每行
代价线性涨，两者抵消；渐近线 6.332。

所以 2.9 ms 要求轮 ≈ 20 ms 且起草器打满，缺口全在轮上。

## 缺口的位置已定位到五个阶段

native 分项里每一项都有一块既不是字节也不是浮点的时间：

    SFA 窗口        76.97 µs   gather 26，算术约 2，余 49 未解释
    indexer front   70.37
    projection tail 56.55
    selector/cache  54.84
    Prolog          42.38

下一步是在这个口径内再下一层，而那需要 kernel 级 profile，不是又一个形状探针。
`--profile-last` 把 profile 限定在稳态解码轮上，项目至今三份设备 profile 全部跑在
`max_tokens: 6, mtp_width: 4`，不是稳态。

## 已测、可叠加、尚未进引擎

    FFN 两半流水（专家 e 的 activate 对 e+1 的 gate_up）  42 µs/层 = 3.3 ms/轮
    整层一次下发 对 三次 native 下发                      10.00 µs/层
    两次规约进同一下发                                    11.50
    Prolog 两投影拼成一次调用（int8）                     26.87
    MoE gate_up 与 down 融成一次调用                      4.63 ms/轮

MoE 固定开销可分离：`gate_up = 59.37 µs + 0.709/专家`、`down = 63.31 + 0.578`。
固定那份是每次调用一份、在单专家处就已满额，所以融合抹掉两份里的一份。

机制同一条：一个操作折进已有载入值 2.35 µs，自带载入值 24.30 µs，差 10.3 倍。
load 一次、算几样、store 一次。

## 已证伪

    专家库收窄        厂商 grouped matmul 每片只读一次，这笔钱从未花出去
    权重驻留          单银行对轮换银行差 3–7% 且符号不定
    KV 连续化         gather 已带宽受限 1094 GB/s，与 run 长度无关
    宽度 32 本身      最优在 24，只值 0.45 ms，是调参不是机制
    rope 刷新入图     实测 0.14 ms，不是 handoff 记的 7.28
    行选择最优化      贪心最小并集值 19–24%，不是需要的四倍
    树（shared_fia）  结构已正确，17.044 ms／4.263 接受，不值它更宽的轮

## 标量的归属已分开，而 vendor 源码在盘上可改

单 die、gate_up、每次调用（µs，20 次调用的 Level1 trace）：

    活跃专家  duration  mac  mte2  cube标量  向量核标量
      27        71       3    24     30        28
      44        87       5    37     40        28
      81       117       8    71     61        31

`mte2` 严格正比于活跃专家（0.87 µs 一个，截距 0.5），就是权重读。
`aic_scalar` 八成跟活跃专家走（0.574 µs 一个，每调用剩 14.5）。
`aiv_scalar` 完全按调用计价，恒 28 µs，而同一 kernel 的 `aiv_vec` 只有 0.001 ms。

那 28 µs 在 vendor 源码里找到了：`grouped_matmul_swiglu_quant_v2` 的
`InitWorkSpaceSplitConfig` 与 `UpdateWorkSpaceSplitConfig` 在
`groupListType == 1` 分支里用 `groupListGM.GetValue(i)` 逐项做标量 GM 读、
把 256 项的前缀和反复重算（一遍 O(n)，再在 while 里嵌两遍），而
`groupListType == 0` 分支取最后一项即可，是 O(1)。引擎传的是 type 1。

但 type 0 不是免费的 host 改动：实测它被算子接受、随后在设备上 507015
（AI core 执行异常），`gate_up` 与 `down` 在 81 与 27 个活跃专家上四次全挂。
所以引擎传 type 1 不是疏忽——W4A8 这条路只在 type 1 下验证过。那 28 µs
是否就是前缀和，仍未证实。

这个探针的噪声底：27 个活跃专家处三次独立运行 77.274 / 78.435 / 83.583 µs，
81 处四次 119.665 / 119.785 / 120.911 / 122.305，所以斜率 0.76 ± 0.05 µs
一个专家、截距约 59 µs。

MoE 的 15.3 ms 里字节只要求 6.2 ms（127 MB/层 ÷ 1607 GB/s × 78），约 9 ms 可取：
按调用计价的 42.5 µs × 2 × 78 = 6.6 ms，按活跃专家计价的
0.574 × 81 × 2 × 78 = 7.3 ms。

CANN 的 AscendC op 源码在
`sdk/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/`，
这个 kernel 3255 行，W4A8 走 `a8w4_msd_*`。穿刺阶段允许无限制修改 vendor。

vendor 的参数规则也关掉了一条路：`aclnnGroupedMatmulV5` 在 groupType 0 下要求
`size(groupList) == weight.shape[0]`，所以"host 侧递短列表"不可表达，除非连权重
一起聚拢（每调用 63.7 MB，远超要省的）。

## 测量完整性已判定：起草器不改输出，宽度改

同一 prompt、1024 token、sfa、同一容器内连跑：

    w12-auto    tpot 23.86/23.82/23.80   接受 2.570  sha 2cb99464  139 distinct  多样性 0.249
    w12-lookup  tpot 22.15/22.16/22.15   接受 2.590  sha 2cb99464  139 distinct  0.249
    w4-auto     tpot 23.99/24.02/23.96   接受 2.325  sha f06b1373  136 distinct  0.233

换起草器摘要不变，换宽度摘要变。宿主侧编排已在 CPU 上判定为宽度不变
（`check_round_invariance.py`，54 种组合同一条串行流，两个故意做坏的对照臂
都会改变输出），所以宽度依赖在设备上。而换起草器会完全换掉候选 token 却不改
输出，说明**逐行掩码没有泄漏未来行**——否则候选变了预测就得变。剩下的解释只有
数值：不同捕获宽度改变分块与规约顺序，移动 logits 并翻转贪心近平局。

两条多样性 0.249 对 0.233、distinct 139 对 136，所以两条流同样健康，只是近平局
的不同分支。**结论：同多样性下跨宽度比较 TPOT 是成立的，夜里那条"多样性随宽度
单调上升"的怀疑作废**——它混淆了注意力 kernel 与宽度。

另一项附带更正：lookup 在 TPOT 上胜自动机（22.15 对 23.86）而接受数几乎相同
（2.590 对 2.570），因为自动机的 propose 更贵、而 propose 在轮的关键路径上。
离线排序只数了接受数，没数代价。

轮时间 = tpot × 接受数：w12 为 57.4 ms、w4 为 55.8 ms，斜率约 0.2 ms 一行，
不是拟合给的 0.814——那个拟合的臂是混淆的。

## 悬着的前提：9.4 与 14.743 的输出是循环

9.4 与 14.743 两条最快读数的输出都是循环——1023 个二元组里 113 与 82 个不同，
而 prompt 有 4059 个不同二元组、记录里单 token 重复 80 次。接受数随重复度涨，
所以 TPOT 只在输出多样性相同时可比。目前唯一可信基线是 22.135 ms（sfa w12，
多样性 0.249）。

宿主这一半已判定：`check_round_invariance.py` 在 CPU 上让三个确定性目标、三个
真实起草器、宽度 2/4/8/12/16/32 共 54 种组合解出同一条串行流，两个对照臂（读到
未来的掩码、未归并的接受路径）都会改变输出。剩下的只能是设备：逐行掩码、规约
顺序，或模型自己。判据是同一 prompt 下三条臂的 `output_sha256` 必须相同。

## 机器

A3 devserver 池拒绝 key 认证，密码正确也报 Permission denied，必须
`-o PreferredAuthentications=password -o PubkeyAuthentication=no`。
启动失败先用 `npu-smi info -t proc-mem` 看进程名判断是否被抢占，再怀疑自己的残留。
kbenv 的"占用"先看进程名再看 HBM，且 npu-smi 没报到的 die 算被占，所以一个空闲的 vllm 进程能把 16 die 全空的机器判成占用——
7.150.10.78 就是这样被我连跳三台。用 `scripts/free_dies.sh` 逐卡读 HBM。
