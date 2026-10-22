# TileRT-Ascend：A3 单用户 2 ms 解码

目标：AgentX，A3 单机 16 die，GLM-5.2-W4A8C8，batch=1，8K 上下文，TPOT 2.x ms。
vLLM-Ascend 负责 prefill，TileRT-Ascend 负责 decode。

## 实测位置（2026-10-20）

最好 **9.3794 ms**：**`LookupDrafter`** @ 宽度 12（记录里 `drafter: "lookup"`；
目录名 `auto12` 曾被误读为起草器）。自动机从未进引擎。

在引擎自己的 token 流上（8K 提示 + 真实生成的 1024 token）离线比较，排序在每个
宽度都是 lookup < 自动机 < 树，而最优宽度是 24–32 而非 12：

    宽度 12   lookup 4.900 / 10.98 ms   自动机 5.565 / 9.67   树 6.059 / 8.88
    宽度 32   lookup 5.689 / 11.78      自动机 6.872 / 9.75   树 8.258 / 8.11

该口径比引擎实测保守约 10%（lookup@12 给 4.900 对实测 5.441），但排序稳定。
[真实流](evidence/ascend-decode/real-stream-drafters-20261021/README.md)

一轮的代价模型，四个宽度拟合、残差几百微秒：

    round = 45.88 ms + 0.814 ms 一行

固定项不是地板，是**每层下发次数 × 28.8 µs**：12 个元素的一次标量乘就要
28.824 µs，一层约 25 个 kernel，25 × 28.8 = 720 µs 对实测 710 µs/层。所以

    TPOT = 每层 kernel 数 × 28.8 µs × 78 / 接受数

25 个、接受 5.44 给出 10.3 ms（实测 9.3794）。到 2.x ms 需降到约 5 个/层。
[地板](evidence/ascend-decode/kernel-count-floor-20261021/README.md)
[代价模型](evidence/ascend-decode/round-cost-model-20261020/README.md)

一轮 55.424 ms 的构成，逐项实测（**两处口径警告**：各项皆为单 die 或少 die 的
算子级测量，且多半在热缓存下取得——同一矩阵乘重复测 31 µs、轮换权重 62 µs，
差一倍；而项目的三份设备 profile 全部跑在 `max_tokens: 6, mtp_width: 4`，不是
稳态解码，所以**从未有过可用的稳态设备侧分解**。拿到机器后第一件事是补这个
capture）：

| 项 | ms | 口径 |
|---|---|---|
| 注意力 | 7.0 | SFA 89.4 µs/层 × 78 |
| MoE 矩阵乘 | 12.3 | vendor 157.2 µs/层 × 78 |
| TP 规约 | 9.9 | 63.5 µs × 2 × 78 |
| 路由 | 12.1 | 六算子实测，gating 已被引擎融合 |
| 宿主下发 | 6.6 | 空 kernel 3.384 µs × 25 × 78 |
| norm 与残差 | ≤5.5 | 裸算子口径，引擎有编译期融合 pass |

合计约 53 ms，未归属 2 ms。地板是 HBM 1.011 ms/轮，所以目标不受物理限制，
问题全在固定开销。[地板](evidence/ascend-decode/round-roofline-20261020/README.md)

## 头号待测：每行 0.814 ms 是实现代价还是物理代价

单 die 实测一整层从 8 到 32 token 只多 8.07 µs，即每行 **0.026 ms/轮**，而代价
模型拟出 **0.814** ——差 31 倍。拟合本身可靠（四宽度、残差几百微秒、w8 到 w32
相差 19.3 ms），起草器也已排除（宿主侧 0.1–0.3 µs/轮）。所以那笔钱在单 die 碰不到
的地方：十六卡 TP 规约，或图的静态 token 分桶。

若是后者（可消除），树的节点就不再贵，固定预算树的 14.510 接受变成最优配置：

    边界合并后固定项   25.58 ms
    64 节点 × 0.026     1.66 ms
    一轮               27.25 ms
    接受               14.510
    TPOT                1.88 ms   ← 落进目标

判别法：宽度 12 对 13。同桶则差约 0.03 ms，跨桶则差一个桶。需整机一对臂。
[证据](evidence/ascend-decode/row-cost-mismatch-20261021/README.md)

## 已排除的杠杆：checkpoint 自带的 MTP 层

`num_hidden_layers: 78` 而权重索引含 **0..78 共 79 层**，第 78 层就是
`num_nextn_predict_layers: 1` 那个训练过的草稿头。引擎从未加载它
（`cann_model.py:152` 传 `speculative_config=None`，263 行直接拒绝 drafter），
所以至今所有接受率都来自宿主侧后缀自动机——它只能复述见过的内容，改写段必盲。
代价是每轮多一层（78 的 1.3%）。
[证据](evidence/ascend-decode/mtp-layer-unused-20261021/README.md)

公开实测把它的上限钉死：k=1 接受 78–85%、MAL 约 1.85；k=3–5 累积 50–65%、
MAL 2.6–3.9；且 vLLM-Ascend 文档说 DeepSeek 只有单个 MTP 层，MTP>1 不保证。
而自适应树已测 9.288（编辑率 0.02）/4.731（0.05）——**MTP 比现有起草器差一个
数量级**。原因是结构性的：MTP 是通用预测器，自动机利用的是"输出大量复制输入"
这一负载性质，而在后缀匹配通常正确的分布上，回忆胜过预测。

    一轮 35.1 ms（边界合并后）
    接受 9.288（自适应树）        TPOT 4.9 ms  ← 当前最好路径
    MTP 的 MAL 上限 2–4           远不如上一行

## 已建成、未进引擎

| 杠杆 | ms | 状态 |
|---|---|---|
| 单边规约 | 未知 | 十六卡正确；capture 显示它替换的流量占 17.56 ms/轮 |
| MoE 单 kernel | 8.54 | 冷测；四次下发压成一次，省的是中间结果往返 |
| 算子派发回落 | 1.4 | CANN 自带 grouped matmul 快 38 µs/层，净值已扣 norm/gating 损失 |

合计 16.4 ms，折算 TPOT 约 6.9 ms。前两项共用同一个交付点：
`AscendMoERunner._forward_entry`（`fused_moe.py:135`，实例属性），它收未置换的
hidden states 与 router logits、产出 `(shared_out, fused_out)`，所以 gating、
置换、两组专家、合并、共享路径的 reduce-scatter 全在界内。
替换未写，只有[捕获钩子](../tilert/models/glm_5_2_ascend/moe_capture.py)。

## MoE kernel 这条路的真实形状

它慢的原因是结构性的：vendor 吃 fractal-NZ 权重、在 L0C 内经 fixpipe 反量化，
我读 ND、在向量侧反量化，每个专家多付一次 UB 往返。调 tiling 补不上，要么改成
NZ + fixpipe（等于重写一遍 vendor 的核心），要么放弃替换矩阵乘、只保留它作为
跨层融合的载体——后者才是它对 2 ms 目标的真正价值：一次下发覆盖多层，省的是
启动与同步，不是矩阵乘。
[翻转](evidence/ascend-decode/kernel-timing-void-20261021/README.md)

## 2026-10-21 撤回的

判据"有限、非零、行数对"无法因算错而失败，而今天全部 kernel 验证都跑在
`absmax 2.4e-17` 的配置里。以它为据的结论作废：变长 group list、吸收后的索引
检验（含阴性对照）、256 专家形状、以及重排背后的正确性。更要紧的是计时也随之
失真——cube 半边从 UB 读 count（该核无 UB 访问权）而跳过全部专家，于是
79.206 µs 测的是空转，1.98x 与 5.8 ms 一并作废。
[判据](evidence/ascend-decode/validation-void-20261020/README.md)
[裁判](evidence/ascend-decode/vendor-verdict-20261021/README.md)
[二分](evidence/ascend-decode/kernel-bisect-20261021/README.md)

尚存一个缺陷：constant 下 0.47461 对 vendor 0.45117，每元素不等，早于重排。

## 更早证伪的

- 残差是宿主下发：实为 36 ms 中的 6.6
- 紧凑专家列表：空载耗时不随专家数增长，扫描在噪声里
- 共享专家按"16 个 1 行"折叠：权重被重读 16 次，71.6 µs 对分开的 51.1
- 融合 gating / add-rms-norm：引擎已有，分别省 0 与 0
- 树状起草是近期杠杆：固定成本不降时最优宽度就小，树没有预算
- 成对旗标是 5 轮上限的成因：换网格 barrier 后界未动，真因是 `Matmul::Init`
  每轮从同一个 `TPipe` 取 scratch

## 2026-10-21 测出的边界

目标 2.9 ms 配实测接受数 5.441 要求一轮 15.8 ms，即**每层 203 µs 含通信**。
而十六卡单次集合通信的地板是 190 µs，与消息大小无关（12 KB 到 576 KB 差
5%，两卡已付 193 µs），也不随 rank 数走。所以一层只要规约一次，预算就用
尽，而 TP16 的层规约两次。

一层直接测得（十六卡、12 行、AIV 开）：

    仅计算 493.22 µs ／ 含两次 all_reduce 870.63 ／ 通信 377.42
    按 78 层：一轮 67.91 ms（计算 38.47 ＋ 通信 29.44）

该总数是上界——引擎融成 637 个 SuperKernel 并多流重叠，实测轮 50.710。
按普查的 16.4 ms SuperKernel 口径联合定价：

    每层 2 次规约 → 12.52 ms ／ 1 次 → 9.79 ／ 0 次 → 7.07（朴素计算）
    一次规约配融合计算 → 5.74

所以三项必须同时达标：规约次数、融合后的计算、起草上界。任一不动都不够。

层内分解按 profiler 口径（设备时间，宿主计时会多算约 120 µs 的 FIA 参数
处理）：一层 228.67 µs = GroupedMatmul 127.17（56%）＋ FIA 50.28（22%）
＋ 四个 Matmul 44.63（20%）＋ RmsNorm 3.33 ＋ DynamicQuant 3.27。一轮
17.84 ms，与 kernel 普查的 16.4 ms 独立吻合 8%。

所以最大项是 MoE 而非注意力，而专家库收窄正对准它。按 checkpoint 真实宽度
（gate/up `[1024,6144]`、down `[3072,2048]`，输出轴 nibble 打包，
moe_intermediate=2048 而非 1536，一个专家 18.9 MB）重测，6 行、profiler
口径：

    16 专家 @1024  80.21 µs   1255 GB/s
    16 专家 @2048 160.53 µs   1254 GB/s
     2 专家 @1024  16.65      756
     2 专家 @2048  17.69     1423

16 专家两个宽度都稳在 1254–1255 GB/s（带宽受限已确认），2 专家恒 17 µs
（与库无关的地板）。一层跑两次 grouped matmul（gate_up 与 down），所以
满库约 320 µs/层、收窄后约 35 µs，**省 285 µs/层 = 22.2 ms/轮**，是本会话
最大的杠杆。此前 7.57 ms 的定价用了错的 inner 宽度且只算一次调用。
[一层](evidence/ascend-decode/tp16-layer-20261021/README.md)
[通信](evidence/ascend-decode/tp16-collectives-20261021/README.md)

通信成本的五个出口已由测量关闭：压小消息（平坦）、显式重叠（更慢
33.69 µs）、减少次数（SuperKernel 已吸收至 0.26 次/层独立调用）、降 rank
数（省 58 µs 付 120 µs 的专家库）、手搭点对点（跨 die 拷贝 218 µs 比 HCCL
的 193 还贵，而同样字节到锁页宿主只 37 µs，故地板是同步不是传输）。

## 2026-10-21 轮的账，按口径分开

计时装置本身有 26.58 µs 地板（空图回放＋同步），**每次回放一份**，不是每个
图内算子一份。所以回放对回放的比值可用，而分开回放的和不可用——两图"重叠"
与层链"气泡"两处结论因此撤回。

    设备计算，八算子真实宽度（profiler）   27.38 ms
    MoE 管路：路由 gather／合并／量化        3.03
    集合通信：156 次在一张图里              5.14
    和                                     35.55
    实测轮                                 50.73
    未归属                                 15.18   195 µs/层

杠杆，各按其被测的口径：

    专家库收窄                             0 ms    ← 2026-10-22 撤销，见下节
    rope 刷新入图（回放对 eager）           7.28
    AIV（图内、一轮规模）                   约 0

集合通信的三次重定价：torch 单次 190 µs → 29.64 ms／一张图四个 66.41 µs →
10.36／一轮 156 次 32.77 µs → **5.11**。只有最后一个对应引擎（图内 1848 个
kernel）。曲线收敛到约 33 µs，与把同样负载搬出 die 的 37 µs 同阶，所以图内
一个集合通信基本只剩传输。

## 2026-10-22 层的 650 µs 里约 500 是等待,而倾斜调度够到目标

两个阶段有 per-core trace,形状相同:

                 cube工作  cube等待  vector工作  vector等待   窗口
    路由 FFN 体    71.25     67.52      23.40      111.74    136.40
    MLA 尾         13.02     19.84      12.32       19.52     约32.8

尾里逐阶段可见交替:`output_matmul` 9.41 cube 时 AIV 等 10.10,
`output_dequant` 10.10 vector 时 AIC 等 11.06。

因为走查即读取,cube 的工作就是层的读取量 73 µs;两份 trace 把 vector 半边
放在 cube 的 0.33 与 0.95 之间,所以真实工作约 146 µs,而层是 650——
**约 500 µs 是栅栏与空转**。

完美倾斜后层趋近 `max(cube, vector)` 加栅栏,按 2.29 µs/次:

    10 次栅栏   层  95.9 µs   轮  7.48 ms   tpot 1.375
    20 次       层 118.8      轮  9.27      tpot 1.703
    30 次       层 141.7      轮 11.05      tpot 2.031
    今天        层 650.4      轮 50.73      tpot 9.324

所以 2.x ms 在每层约三十次栅栏之内是算术可达的,而从 9.324 到那里的全部距离
就是等待。这是上界不是计划:两份 trace 口径不同(native FFN 体与独立 MLA 尾,
都带instrument;650 是 vendor 轮除以 78),vector/cube 比只有两个样本且差三倍,
而调度本身未建——跨半交接试了两次都挂,能跑的范式是在同一栅栏区间内共置
独立工作(`ROUTED_SHARED_PIPELINE` 以此省 32 µs)。

## 2026-10-22 第二次更正:矩阵乘调用没有固定费,所以地板是读取量

固定 2624 列、把 K 从 256 扫到 6144,最小二乘:

    fused         = 15.04 + 1.96 µs / 1000 K
    多一次调用    = −1.58 + 4.41 µs / 1000 K

多一次调用的截距是零,全部代价随 K 走(K=256 预测 1.13 实测 1.29,
K=6144 预测 27.1 实测 25.68)。所以那 15.04 是**整个核**的固定费——下发、
`Matmul::Init`、tiling 读取、收尾栅栏——发多少次调用都只付一次。

于是成本是 `15.04 + (K 走查次数) × 约 2.1·K/1000`,而一次走查就是把那片
权重流进来。**走查即读取。** 这把 FFN 那两个看似矛盾的读法统一了:cube 半边
71.25 µs 既是 95.6 MB 按 1341 GB/s,也是每核三四个专家两个相位、K=6144 约
13 µs 加 K=2048 约 4.3 µs 的 60.6——同一个量被数了两遍。

**由此撤销上一节的结论。** 我说"一层 12–15 次调用 × 11.6 µs 固定 = 211–264,
已吃掉 235 预算,所以 2 ms 要改模型结构",前提是错的:没有那笔固定费。

真正的地板是必须读的字节:116.9 MB/卡/层,按 1607 GB/s 是 **73 µs/层**。
预算 235,所以余量 3.2 倍;今天 650,是地板的 8.9 倍。那 577 µs 的差额是
**等待**,而两半轮流空转已经量出其中一部分(FFN 体里 cube 等 67.52、
vector 等 111.74)。

## 2026-10-22 更正:每卡持有全部 256 个专家,一轮碰 81 个

trace 的 `measured_routing` 数组有 256 项、81 个非零。所以 TP16 是把每个专家
沿中间维切片,而非把专家分给各卡——每卡看到全部 256 个,一轮碰到 81 个。
先前记的"每卡 16 个、碰 5 个"是错的,`a5` 那套 EP 直觉套到了 TP 上。

受影响的数:

    专家字节      302 MB × 81/256 = 95.6 MB/层/卡   (原记 37.8)
    流量地板      11.9 + 9.4 + 95.6 = 116.9 MB → 81 µs/层  (原记 41)
    预算          235 µs/层,所以是地板的 2.9 倍  (原记 5.7 倍)

而 FFN 的 cube 半边 71.25 µs 读完那 95.6 MB,折合 1341 GB/s——对实测 HBM
1607 是 83%。所以它已在带宽上,不读更少字节就下不去,71.25 就是 FFN 体的地板。

不受影响的:专家库收窄仍为零(厂商跳空组这个机制与计数无关),而宽度扫描的
结论方向不变(接受数渐近 6.332 仍封顶)。

## 2026-10-22 最大的未取项:FFN 两半轮流空转

路由体自己的 stage trace 里已经有答案。宽度 12、24 cube 核 48 vector 核,
按核取中位:

               工作      栅栏等待    闲置
    cube      71.25 µs    67.52      49%
    vector    23.40      111.74      82%

窗口中位 136.40,而两半各自的"工作加栅栏"都对得上它(138.77 与 135.14)。
真实工作合计只有 94.65 µs。

等在哪里也清楚:AIV 在两次矩阵乘期间白等 87.46(gate_up 51.31、down 36.15),
AIC 在向量阶段白等 39.77,剩下 27.75 是 cube 等 cube——专家循环的不均衡。

单个专家内部是严格链(metadata → gate_up → activate → down → store),所以
向量工作藏不进它自己产出的那段 cube。但一轮每卡碰到约 5 个专家,而
**专家 e 的 activate 与专家 e+1 的 gate_up 无关**。把循环做成流水,两半就能
互相覆盖。cube 的 71.25 是地板,但流水自己要付栅栏:`SyncAll` 实测 **2.29 µs/次**,
而 81 个活跃专家分到 24 核、每核三四个,倾斜 gate_up/activate 约五步、
down/store 约五步,十次栅栏 22.9 µs。所以落点约 94 µs 对现在的 136.40——
**省约 42 µs/层 = 3.3 ms/轮**,仍是桌上最大单项。更浅的流水(每相位两步)
栅栏 13.7 但藏得少,落点相近,所以深度不是杠杆。

共享专家那边已经示范过:`ROUTED_SHARED_PIPELINE` 把共享 cube 投影与路由向量
阶段并排,M=12 上 187.17 对 219.24,省 32 µs。

这也给"cube 阶段免费藏 10.5 µs 向量工作"定了量级:FFN 整个向量半边只有
23.40 µs,两个 cube 阶段的容量就够装下全部。

## 2026-10-22 megakernel 实测可叠加的部分

每一项都在设备上测过,且都确认过厂商没有先拿走:

    整层一次下发 对 三次 native 下发    10.00 µs/层   另一代理, 60/60 全胜
    两次规约进同一下发                  11.50         今日图回放关已过, 60/60
    Prolog 两投影拼成一次调用(int8)     26.87         今日已建, 精确相等
    合计                                48.37 µs/层 = 3.77 ms/轮

    轮 50.73 → 46.96    tpot 9.324 → 8.63

拼接那一项的边界也量清了。决定性的变量是**归约深度而非调用次数**:

    K=6144 省一次调用(Prolog, int8)    +26.87 µs   fused 23.91 对 split 51.07
    K=6144 省三次调用(四件分组, bf16)   +42.90      但引擎用不上, 见下
    K=2048 省一次调用(query_norm, int8) −1.19       fused 28.03 对 split 27.10

引擎用不上四件分组的原因是读源码读出来的:Prolog 从 int8 的 `token_x` projects
`weight_dq` 与 `weight_dkv_kr`(这两个可拼),而 indexer 的 key 读 **bf16** 的
`hidden`、dtype 不同拼不进去;indexer 的 query 与 q_b 读 `query_norm` 属第二级;
router gate 在 `ffn_bridge::Forward` 里、在 attention 之后。

而厂商没拿走这一项,也是读源码读出来的:
`mla_prolog/op_kernel/arch22/kernel_mla_prolog_split_n_arch22.h` 的 796 与 804
行是两次独立矩阵乘,各带独立的 `mmCqParam_` / `mmCkvKrParam_` 与各自的
`needSetOrgShape`,而两者读同一个 `tokenXGm_[tokenXOffset]`。

还有一个白得的:权重行跨步必须按 512 字节对齐。四件分组 3040 列(6080 字节/行)
是 48.12 µs,补到 3072(6144 字节)是 38.43——补 32 列零省五分之一,而 38.43
正好追平 `torch.matmul` 同形状的 39.12。

## 2026-10-22 收盘：路径剩下什么

今天撤销了两个最大项，其中一个是路径上唯一"已测"的首步。剩下的全部兑现后：

    今天                       650.4 µs/层   tpot 9.324
    路由元数据融成一个 kernel  −25.0  625.4        8.965   估
    专家合并折进 epilogue      −20.0  605.4        8.679   估
    四个投影融合               −35.0  570.4        8.177   估
    每层只剩一次规约           −34.0  536.4        7.689   已测
    两次规约进同一下发         −11.5  524.9        7.525   已测（今日新增）

撤销的：

    专家库收窄       −164.4  厂商 kernel 跳过空组，这笔钱从未花出去
    KV 连续化         −23.0  gather 已带宽受限 1094 GB/s，与 run 长度无关
    宽度 32           −13.0  最优在 24，只值 0.45 ms，是调参不是机制

而接受数的渐近线是 6.332，所以 2.9 ms 要求 **235 µs/层**。全部兑现后是 525，
缺口 **289 µs/层 = 22.6 ms/轮**，没有机制对应。流量地板 41 µs/层，所以 235 是
地板的 5.7 倍——物理上有，机制上还没找到。

缺口最集中的地方已定位：native 分项里 SFA 窗口 76.97 µs，其中 gather 26、
算术约 2，**49 µs 既不是字节也不是浮点**。另四个阶段（Prolog 42.38、
indexer front 70.37、selector/cache 54.84、projection tail 56.55）同理未拆。
下一步是在这个口径内再下一层，而那需要 kernel 级 profile，不是又一个形状探针。

## 2026-10-22 引擎环境可重建，但端到端卡在 NAS 带宽

交接里把"引擎环境随容器被删"列为最硬的阻塞。它已解除：`bench_cann_mtp.py --help`
在重建的环境里正常跑起来（a3-194，单 die，无 checkpoint）。配方是

    镜像      sha256:1910b1ac…（torch 2.12 / torch_npu 2.12）
    CANN      宿主 /home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0
              → /opt/ascend/cann-9.2.0:ro，容器内 source set_env.sh
    PYTHONPATH 仓库树 + vendor-hook-runtime212 的
              vendor-hook-pinned/{vllm-a982-py311-compatible,ascend-python-build}
              + vendor-hook-deps/{transformers519,python}
    OPP       vendor-hook-deps/python/fla_npu/opp{,/vendors/fla_npu_transformer}
    env       VLLM_PLUGINS=ascend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
              HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true
              env -u ASCEND_DOCKER_RUNTIME -u ASCEND_RT_VISIBLE_DEVICES

镜像自带的 omniinfer 会在插件注册时抛一个 vllm.config 循环导入的 traceback，
非致命，三个包都正常导入。`/workspace/environment/activate.sh` 不需要重建——
它设的 ASCEND_CUSTOM_OPP_PATH 把两个自定义包排在 CANN 前面，而启动器自己的注释
说 CANN 原生 grouped matmul 在 decode 形状上是四个实现里最快的（66.813 µs 对
97.362／104.929／123.985），所以不装它们等于一直开着 STOCK_OPP。

而 checkpoint 这条仍堵着：`GLM-5.2-w4a8c8` 只在 NAS `7.242.104.142:/model/` 上，
107 个文件无 `.partial`，**755 GB**。实测该链路单流 45.3 MB/s，16 并流 82 MB/s
（瓶颈在链路不在流数），所以一遍要 2.5 小时；而 a3-194 根分区只剩 41 GB、
a3-162 剩 94 GB，落不下。只读 NFS 卷的配方可用：

    docker volume create --driver local --opt type=nfs \
      --opt o=addr=7.242.104.142,ro,nfsvers=3,soft,timeo=50,retrans=1,nolock,nosuid,nodev,noexec \
      --opt device=:/ <name>

8K 前缀快照已落宿主：`checkpoint-owned/prefix8k-v3/case-00`，12 GB、
`status: capture_complete`、seq_len 8192、16 rank。

所以端到端臂的阻塞从"环境没了"变成"755 GB 过 82 MB/s 的链路且无处缓存"。

## 2026-10-22 撤销：专家库收窄这个杠杆不存在

厂商 grouped matmul 对一个 count 为零的专家不收费。同一张 16 专家权重张量，
只给 5 个非零 group，代价 87.23 µs；物理切到那 5 个专家，87.10 µs。差 0.13 µs
（重复间噪声约 0.6）。而 16 个全非零是 250.53。七次重复、单 die、NPU event。

所以引擎本来就只付被碰到那几个专家的代价，收窄要省的那笔钱从未花出去。
原探针之所以看起来有杠杆：它靠切张量来改专家数，同时让剩下的每个专家都非零，
于是把"张量大小"与"活跃组数"一起改了，再把全部效应记给前者。

源码同向：引擎自己的 gmm1 在 `SetOrgShape`／`MMCompute` 之前
`if (mnConfig.m <= 0 ...) continue;`，a8w4 族另有三处、通用 kernel 一处，
本项目手写的 routed kernel 也有 `if (counts.GetValue(expert) == 0) continue;`。

这也把上一节那个"巧合"解释掉了：native 574.15 与 vendor 575.22 之所以相等，
是因为两者在这一项上是同一个机制，谁都不可能领先。

实现与开关已删（`narrow_bank.py`、`narrow_bank_check.py`、`--narrow-bank`）。
到 2ms 那条路因此没有了"已测"的第一步，剩下的都是估算或小项。

## 2026-10-22 native 整层与 vendor 整层在同一口径上对上了

两条独立路径在十二行、单 die、引擎形状下给出同一个每层数：

    vendor 口径  一层 21 kernel 一次回放 497.03 ＋ 链里两次规约 78.19 ＝ 575.22 µs/层
    native 口径  三 rank 计时控制臂 1148.30 µs/两层对 ÷ 2            ＝ 574.15 µs/层

差 0.19%。两份测量不共用代码、fixture 与归属方法，所以这是对轮账里"每层
约 575 µs"那一项的交叉检验，而不是同一个数的两次抄写。

但两者规约不同类型，这一项不可直接相减：native 控制臂走
`BF16 → FP32 → HCCL SUM → BF16`，载荷是引擎 BF16 规约的两倍。规约在这个尺寸
上由对齐抖动支配而非带宽（孤立 32.77 对链里 78.19），所以 FP32 的代价既不是
两倍也无法从现有数据推出。能说的是：native 的计算那一半不比 vendor 贵，偏差
的方向对 native 有利，而确切的分解需要在同一 dtype 下再测一次。

`vendor-hook-tp-timing-m12-v1-a3-194` 把两次规约折进层级下发省 23.05／23.26 µs
每两层对（60/60 全胜，std 1.3–1.7），即约 11.5 µs/层、0.9 ms/轮。fixture 自己
声明这一项**混合了下发融合与 SHMEM 替代 HCCL 两个机制,未隔离**。

## 机器上的两种被杀

容器整体 `Exited (137)`：另一租户抢占，`docker start trt-asc` 可恢复，工作
目录仍在。

单个 rank `SIGKILL`（torchrun 报 exitcode -9，而外层 `timeout` 的退出码仍是
0，所以循环会误记成成功）：宿主管道页软限。`/proc/sys/fs/pipe-user-pages-soft`
是 16384，而 dmesg 里 `pid=... cmd=python user_bufs=16400
too_many_pipe_buffers_soft` 正好越界——十六个 rank 各开管道，叠加其他租户的
进程就撞上。NPU 显存此时仍空（剩 58 GiB），所以别去查显存。

绕法：让每 rank 的输出不经 torchrun 的管道（`--redirects` 写各自文件），或让
非 0 rank 静默。而循环里判成功要看记录文件是否生成，不能只看退出码。

第三种：`[Set][Options]OpCompileProcessor init failed`，算子在线编译进程起不
来。记录写出来了但 `requests` 为空、`status: error`。与旗标无关，和管道限额
同源（编译要开子进程）。2026-10-21 三臂连续撞上三种不同失败，共同点只有一
个：另一租户同机。

判据：臂成功 = 记录里有非 warmup 请求。三种失败的退出码都是 0。

第四种，也是最贵的：**容器被删除而非停止**。`docker ps -a` 里连条目都没有
（`No such container: trt-asc`），而另一租户的 `lan_d_d0` 同时在跑。所以
`/workspace/experiments/` 不是持久的——2026-10-21 这一轮臂的代码树、日志、
记录全部随容器消失，只有 git 里的东西活下来。

所以在 a3-84 上做整机实验的方式要改：每臂一出记录就立刻 `scp` 回本地并提交，
不要等一批跑完；代码树从 git 重建而不是依赖容器里的副本；而守候脚本本身也
会消失，所以它不能是唯一的调度者。

2026-10-21 对 a3-84 宿主 `/data/models`、`/sfs_green_wuhu_ci/models`、
`/efs_guiyang`、`/mnt`、`/home` 的搜索只找到 `GLM-5.2-w8a8c16`；这不能
证明 w4a8c8 只有已删除容器中的一份。2026-10-22 的
[恢复验证](evidence/ascend-decode/megakernel-device-20261021/integration/checkpoint-restore-v1-a3-194/README.md)
确认 NAS `7.242.104.142:/model/GLM-5.2-w4a8c8` 仍有原清单中的 108 个文件。
缺少的 `optional/quarot.safetensors` 已从固定版本补齐，保存在 a3-194 的
任务目录。实际容器内的 109 个文件路径与大小全部匹配，配置和补件的 SHA-256
匹配；本次未重读、重算 95 个权重分片的哈希。

checkpoint 可通过只读 NAS 和宿主补件挂载恢复，下一步是重建整模型运行环境并
复测稳态基线。模型和实验记录应保留在容器之外。

重建试过一次，结论如下（2026-10-21，之后按指示停手未再动环境）：

- 镜像 `ai-infra-infer-1.2.1-...-cann9.1.0-...-202610122142` 里 CANN 在
  `/usr/local/Ascend/cann-9.1.0`，不在原容器的 `/opt/ascend/cann-9.2.0`；也
  没有 `/opt/trt-asc` 这个 venv。所以原 trt-asc 是在镜像之上另装的，镜像不是
  它的等价物。
- `torch_npu` 需先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，否则
  `Failed to load the backend extension`。
- 挂载配方可从兄弟容器抄：`docker inspect lan_d_d0 --format
  "{{range .Mounts}}{{.Source}}:{{.Destination}} {{end}}"`。
- 但新起的容器活不过一分钟：一个只 `sleep infinity` 的探针容器在 t+20s 还
  `Up`，t+30s 就 `Exited (1)`。两次独立尝试都如此。所以这台机器上有东西在主动
  终止非白名单容器，而不是租户偶然清理。

因此在 a3-84 上的整机实验不能依赖"自己起容器"。要么用已存在的 trt-asc（现已
不存在），要么先弄清容器被终止的机制。

## 机器

a3-84 容器耐久、16 die 常被租户占；187 die 常空但容器约四分钟被回收
（docker 事件流证实，租户容器同样被销毁），宿主 `/tmp` 也会被清。
整机臂只能在 a3-84；187 只做能在一分钟内跑完的单 die 探针。
[回收器](evidence/ascend-decode/container-reaper-20261020/README.md)
