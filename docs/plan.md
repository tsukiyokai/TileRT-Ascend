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

但这条只对了一半，而且不值钱。读源码后：`InitWorkSpaceSplitConfig`
（pipeline.h:144-164）确实是 type 1 走 256 次标量 GM 读、type 0 读一项；
`UpdateWorkSpaceSplitConfig`（166-233）方向相反，两种类型每访问一组都读两次，
而 type 1 独有的前缀和前言在 `loopCount == 1` 时成本为零。按 256 组、81 非空，
每核整次调用的 groupList 标量读是 type 0 的 1025 对 type 1 的 1280，差 255 次，
几乎全在 `Init`。

实测 type 0 也走不通：被算子接受、随后设备 507015，`gate_up` 与 `down` 在 81 与
27 个活跃专家上四次全挂。引擎传 type 1 不是疏忽。

更要紧的更正：55.7 µs 的固定项不是走组。256 次循环体每核约 12000 条标量指令、
1000–1300 次串行标量 GM 读，量级 10 µs。余下落在每次 workspace 迭代的 4 个
`SyncAll` 加 4 个 `pipe->Reset`、`mm.Init`、以及每次调用的 `InitBuffer`
（pre.h:94-101、post.h:287-297），尚未测量。所以**把 gate_up 与 down 融成一次
调用**才是这笔钱的正解（约 55 µs × 78 = 4.3 ms 一轮），而它省掉的是这些
per-call 设置，不是抽象的"固定开销"。

有名有姓的结构杠杆：`groupListType == 2`（`GROUP_LIST_TYPE_SPARSE`）在平凡
`grouped_matmul` 里已实现（`GMMGroupMSparseProcess`，kernel.h:339/358-369），
groupList 是 `[e,2]` 的 (expertIdx, count) 前置紧排对，只访问活跃项、遇首个
非正计数即 break。两道闸门关着它：`SET_GMM_A8W4_TPL_ARGS`
（grouped_matmul_tiling_key.h:50-54）只给 `GROUP_LIST_TYPE` 放行 COUNT，
int8×int4 没实例化稀疏模板；`grouped_matmul_swiglu_quant_v2` 自己只接受 0/1。
允许改 vendor，所以这是一条具体改动。

部署前提：算子以预编译 `.o` 发布
（`GroupedMatmulSwigluQuantV2_ed7aa5e41344a2216aa977694f4ab9b3.o`，tilingKey 2），
改头文件不生效，须重建替换或以自定义算子另名发布；项目已在用
`ASCEND_CUSTOM_OPP_PATH` overlay，机制现成。

顺带在 vendor 里发现两个缺陷（当前 shape 不触发）：`UpdateAuxiliaryMatrix`
（post.h:345-346）恒按单张量布局索引 `weightAuxiliaryMatrixGM[curGroupIdx * tokenLen]`，
而 `PreLoadAuxiliaryMatrix`（306-313）按 `isSingleTensor` 分支，所以
`isSingleTensor == 0` 且某个 AIV 核的行跨两组时越界——12 行分到 12 核不会发生，
提高每核行数或减少核数就会；pipeline.h:184-190 的 `else` 臂重算了 `if` 臂
已经算好的和。

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

## 注意力：97% 的 fixpipe 时间是串行，不是搬字节

`SparseFlashAttention` 一轮 7.089 ms、一次调用 90.88 µs。读 vendor 源码
（`vllm-ascend-8237/csrc/attention/sparse_flash_attention/op_kernel/arch22/
sparse_flash_attention_service_cube_mla.h`）后：

真正的乘累加 0.59 µs，占 AIC 活跃时间 1.2%。而 1920 次 mmad 每次占 M 管道
116 周期、名义只需 54.4，因为那道守卫
`if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) PipeBarrier<PIPE_M>();`
在这个 shape 上算出 8、恒为真，于是每次调用排空 M 管道 1920 次。

Fixpipe：528 次 FIX 操作（384 次 `Fixpipe` 加 144 次
`CrossCoreSetFlag<..., PIPE_FIX>`）花 881685 周期，约 1670 周期一次，而它搬的
3.0 MB 按 v220 速率只需 24576 周期——**97% 是延迟与串行**。三个串行化源：
`fixParams.unitFlag = 0b11`（:1018, :1289）把每次 fixpipe 钉在其链最后一次
mmad 之后；192 次 MM2 fixpipe 全是 `SetAtomicAdd`(:1281) 与
`SetAtomicNone`(:1294) 夹住的 fp32 原子加，彼此无法重叠；144 个跨核 flag
在每个 `ComputeMm1`/`ComputeMm2` 末尾强制 FIX 排空。

两个一行改动：`fixParams.mSize` 从 `mL1SizeAlign`(16) 改为 `mL1Size`(4)
（:1014, :1285，`srcStride` 留 16），停止每次 MM2 fixpipe 把 14 行垃圾原子加进
没人读的 `mm2ResGm`；把 4 个 nL1 分块累进同一个 L0C 缓冲（16×512 fp32 = 32 KB，
占 `L0C_PP_SIZE` 64 KB 的 12.5%），MM1 从每 S2 分块四次 fixpipe 变一次，
384 降到 96。

GM 往返：AIV 每次调用写 27.6 MB、cube 读回 52.2 MB，1536 次 `DataCopy`。
MM1 把 nope 当 K（d 在 k 轴）、MM2 再当 V（s2 在 k 轴），所以每核每个 S2 分块读
1088 KB。L1 fractal 布局其实兼容，MM2 那 512 KB 原则上可免，但要重构
`ComputeMm1`/`ComputeMm2` 相隔两轮的二级预取流水。

这个实例化里 cube 完全不碰分页缓存与 block table：`keyGm`/`kRopeGm`/
`valueGm`/`blockTableGm` 与 `DataCopyPA`(:73-113) 只有 C_TEMPLATE 的 else
分支能到。

AIV 侧更集中，而且那里的标量是真发射不是停顿。比值
（aiv_scalar 0.660、aiv_mte2 0.522、aiv_vec 0.0190、aiv_mte3 0.1135）全归一个
循环：`MergeKv`（service_vector_mla.h:994-1088，热点 :1021），每次调用迭代
12288 次、处理 24576 个选中位置。每个位置两次相依的标量 GM 读
（:829 `topkGm_.GetValue` 后接 :844 `blkTableGm_.GetValue`）、四次 64 位整除取模
（:824 :830 :842 :843）、一次 1024 B 键加 128 B rope 的 `DataCopyPad`（:951 :958）。
一次调用 49152 次标量 GM 读、98304 次 64 位除法、24576 次 MTE2 发射，
约 203 周期一次迭代。

而那两个除数编译期已知：`sparseBlockSize == 1` 是恒等、`kvCacheBlockSize == 128`
是移位加掩码，但它们以运行时 `int64_t` 从 `constInfo` 传入
（kernel_mla.h:213-214, :221-222），所以发射了真除法。向量算术只占 1.9%，因为
`SoftmaxFlashV2` 每次调用只跑 48 次 4×512（:675-676），且每对里的奇 AIV
`vecDealM == 0`（:1104-1106）直接在 :780 返回。

没有任何东西随 8192/9344 增长：kernel_mla.h:343-345 夹到
`min(sparseBlockCount*sparseBlockSize, threshold)` = 2048，所以 `s2SplitNum = 4`。
但这个结论挂在 `sparse_block_size == 1` 上（字面量在
`vllm_ascend/device/device_op.py:570`）；若是 64，同一个夹子给 8192，kernel
就会走全缓存。

随 12 行增长的是核划分：kernel_mla.h:528-549 每个查询行一个基块，
`usedCoreNum = totalBaseNum = 12`，S2 轴从不跨核切分（所有编译出的 tiling key
里 `FLASH_DECODE` 均为 0）。所以 24 个 cube 核有 12 个闲着。

indexer 的账对上了：一轮 21 次（16 次量化加 5 次 bf16）对 78 次注意力，恰是
`index_topk_freq = 4` 与 `index_skip_topk_offset = 3` 推出的层集合
{0,1,2} ∪ {6,10,…,74}，并由 trace 独立验证（一轮 177 个算子，`LayerNormV3`
恰好 1071 = 21×51）。那 5 次走 bf16 是因为 checkpoint 只给 16 层标了
`.indexer.quant_type: INT8_DYNAMIC`，其余 5 层只有 `W8A8_DYNAMIC` 的
`wq_b.weight`，被 `ascend_config.py:1220` 拒绝，于是 `device_op.py:481` 对 :505
分叉到 bf16 算子与两倍大的索引缓存——71.21 µs 对 42.61 µs，一轮 143 µs。

按置信度排序的改动（µs 一次调用）：

    MergeKv 地址算术          20–35   机制高，量级中   约 20 行，一个函数内
    fixpipe mSize + L0C 累加  15–30   中
    去掉恒真的 PipeBarrier     5–6    高              一行
    S2 轴跨 12 个空闲核       35–40   低中            被 usedCoreNum 挡着
    量化 5 个 bf16 indexer   一轮 143  —              checkpoint 改动

确认口径（代理给的）：单算子重 profile 同一 shape，`aiv_scalar_ratio` 须从
0.660 落到 0.35–0.40 且 `aiv_total_cycles` 同比例下降、Duration 随之下降；若比值
降而 Duration 不降，关键路径就是 `aiv_mte2`（0.522，24576 次发射约 92 周期），
下一步改为 fixpipe 加索引表的 run-length 合并。正确性门：索引表里放一个 `-1`
与一个越界项，以触发 :825 与 :1026。

执行的那份源码在 7.150.10.78 的
`/root/trt-assets/kernel-overlay/vendors/custom_transformer/op_impl/ai_core/tbe/
custom_transformer_impl/ascendc/sparse_flash_attention/`（容器内
`/opt/kernel-overlay/...`，`ASCEND_CUSTOM_OPP_PATH` 已指向），带 `op_host` 的
同字节副本在同机 `/home/shanshan/tilert-megakernel-20261021/vendor-hook-pinned/
vllm-ascend-8237/csrc/attention/sparse_flash_attention/`。CANN 9.2 自带的那份
不参与这个算子；两个 indexer 与 MlaPrologV3 才来自 CANN。

所以处方要改口径：不是"执行更少的标量指令"这种抽象说法，而是两件具体事——
停止串行化（管道排空、原子加全序、把对齐填充的行也写回去），以及把编译期
已知的除数真的当常量用。

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
