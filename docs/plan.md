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

## 一轮 61.22 ms，而真正在乘的是 4.5%

2026-10-23 的第一份稳态解码轮 profile（宽度 12、自动机、sfa、51 轮、
2579 次下发一轮即 33.1 次一层、260 个不同 kernel），经对抗审查后的读法：

一轮是 **61.22 ms ± 0.94**，由 AllGather 周期定（n=50），并由 Stage/51 = 61.266
与 52.394 + 8.799 = 61.193 两路印证。kernel duration 之和 53.052 ms 占 86.7%，
其余是 8222 µs 设备空闲，其中单独一个 5.26 ms 的宿主换手空洞占 63.6%，
另有约 2573 个亚 5 µs 的下发气泡合 1954 µs。宿主 86.7% 的时间在阻塞而只做了
2.3% 的下发工作，所以这是设备受限，不是宿主受限。

计数器语义（精确确立，非推断）：`*_time = total_cycles /(Block Num × 1.85 GHz)`，
是按核平均的活跃时间；比值的分母是 `aicore_time` 而非 `Duration`
（|aic_mac_ratio − mac/aicore_time| 中位 0.000239，对 Duration 是 0.005750）。

因此不可相加：管道不划分核，`(mac+scalar+mte1+mte2+fixpipe)/aicore_time`
中位 0.8518、p90 1.3244、最大 1.574，27.5% 的行超过 1.0。AIV 反向失败，
`(vec+scalar+mte2+mte3)/aiv_time` 中位 0.3808——**向量核活跃时间的 62%
不落在任何被计数的管道里**，一轮 11.7 ms，gate_up 上约 5.6 ms，很可能是
AIC/AIV 的 MSD 握手，是全轮最大的未解释桶。AIC 与 AIV 是物理分离的核，
`(aicore+aiv)/Duration` 在 25857 个 MIX_AIC 行上超过 1.0 最高 1.939，而
`max(aicore,aiv)/Duration` 从不超过 1.0。

站得住的两条。一，**不是算术受限**：cube 单元占 cube 核活跃周期 4.50%，
向量单元占向量核活跃周期 4.28%（`aic_mac_time` 在 20 次相同调用上方差为零）。
二，矩阵乘是标量占用受限，按 `aic_scalar/aicore` 计：全局 60.15%，
`GroupedMatmul` 86.4%、`GroupedMatmulSwigluQuantV2` 65.0%、
`QuantBatchMatmulV3` 64.2%、`SparseFlashAttention` 在 AIV 上 66.1%；
两个 MoE GMM 占 cube 核标量时间的 64%（15.480 中的 9.944 ms）。

唯一站得住的墙钟说法：`Σ max(aic_scalar, aiv_scalar)/51 = 20.387 ms`，
占一轮 33.3%。对照 `Σ max(aicore, aiv)/51 = 43.861 ms`，占 71.6%。

已作废：「标量 28.95 ms 占 55%」、「算术 2.91 ms 占 5.5%」、
「地板是 mac+mte2 的 8–9 ms」、「只能靠砍标量」。

## 地板在 2.x 上，不在 8–9 ms

按字节：每专家每 die gate_up 786432 B 加 down 393216 B，75 个 MoE 层给
0.0885N GB，非 MoE 约 0.77 GB；按 1.421 TB/s 每 die 得 `0.0623N + 0.542 ms`，
即 N=27 时 2.25、N=44 时 3.28 ms。另一条与 N 无关的界：2579 个 kernel ×
1.04 µs 最小实测时长 = 2.68 ms。两条都落在 2.x 目标上。
24 核 × 8.236 ms × 59.2 GB/s 是 11.7 GB 的 MTE2 容量，用来搬约 3.4 GB。

所以 2.x 不受物理限制。TPOT 2.9 要求轮 7.5 ms（今天 2.59 接受数）或 20.2 ms
（宽度 12 的复制上界 6.96），而今天是 61.22 ms。

## 标量的主项是内存延迟停在标量管道上

`compiler/tikcpp/tikcfw/impl/kernel_tensor_impl.h:1257-1266`：

    __aicore__ inline __inout_pipe__(S) ... GetValue(const uint64_t offset)
        DcciReadCache(this->oriAddress_ + offset);
        return this->oriAddress_[offset];

`__inout_pipe__(S)` 把它记在标量管道上，而 `DcciReadCache` 无条件执行，所以
每次读都先失效缓存行再走完整 L2/HBM 往返，编译器无法缓存。约 125 周期一次，
与直接测到的 108（AIC）/197（AIV）相符。

两条独立证据说这是等待而非发射：20 次逐位相同的调用里 `aiv_scalar_time`
相对散布 0.44–0.67、与 Duration 相关 r = +0.85…0.97，而那一次快调用
（64.3 µs 对均值 87.4）里 `aiv_scalar` 从 28.1 掉到 15.2——固定指令流不可能
少执行 46% 条指令。另外只有 `aic_mte1_time` 被厂商注明「不包括搬运等待时间」，
标量那一项厂商拒绝这么注明。

用轮自己的 3825 次 gate_up 调用对 N 回归（比单 die 探针更可信，见下）：

    aic_mte1    = -1.206 + 0.5896 N   r=+0.998   正比，真工作
    aic_mte2    = -3.560 + 0.7534 N   r=+0.982   正比，真工作
    aic_fixpipe = -0.557 + 0.1092 N   r=+0.938   正比，真工作
    aic_scalar  = 53.948 + 0.3927 N   r=+0.316   83% 固定
    aiv_scalar  = 25.537 + 0.0586 N   r=+0.060   94% 固定
    Duration    = 91.935 + 0.9438 N   r=+0.604   78% 固定

最大那个 kernel 78% 的时长与活跃专家数无关，而三条正比的管道才是真工作。
所以「融合两次调用」胜过「压组数」，这与读 MoE 源码得到的结论独立一致。

修复是把批量数据一次搬进 UB，不是指令减肥：MoE 侧约 26 µs 一次调用 ×
150 次 MoE GMM = 3.9 ms 一轮，占 6.4%；而 `Overlapped = 0.0`，所以这部分
无法靠调度隐藏。注意力侧 `MergeKv` 的改法是同一机制（512 个索引与 73 项
block table 一次 `DataCopyPad` 进 UB，替掉 49152 次相依标量 GM 读）。
两条互不知情的代码路径指向同一个修复。

## 单 die 探针测的不是轮里那个算子

探针测的是平凡 `GroupedMatmul`、ND 权重、宽 256 未量化输出，而轮里从不下发
这个算子：它是 gate_up 的未融合替身（K 对、N 对、每专家 768 KiB 对，但
kernel、布局与 epilogue 都不对），而轮里的 `GroupedMatmul` 是 down 投影。
唯一命中真算子的那个探针高估而非低估：104.488 µs 对生产的 87.392。

探针还有三处缺陷：它自己的元数据写着「one die, one rank, shared with another
tenant」；N=44 上调用间 Duration 散布 1.48 倍（64.28–95.38 µs），所以绝对值
约 ±20%；`aic_icache_miss_rate` 在六份探针 trace 里精确为 0，而生产是 0.0179。
所以轮自己的 3825 次调用是更好且无争用的扫描。

L2 的那一半仍然可信：25 banks × 192 MiB，每次调用 34603008 B，重读间隔
792 MiB 对 l2_size 201326592。边际 0.83 µs 一个专家是 768 KiB 跑在推导出的
1421 GB/s 上界的 88%，所以这个 kernel 随 N 变化的那一半已近屋脊线，没得赚。

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
