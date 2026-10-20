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
[一层](evidence/ascend-decode/tp16-layer-20261021/README.md)
[通信](evidence/ascend-decode/tp16-collectives-20261021/README.md)

通信成本的五个出口已由测量关闭：压小消息（平坦）、显式重叠（更慢
33.69 µs）、减少次数（SuperKernel 已吸收至 0.26 次/层独立调用）、降 rank
数（省 58 µs 付 120 µs 的专家库）、手搭点对点（跨 die 拷贝 218 µs 比 HCCL
的 193 还贵，而同样字节到锁页宿主只 37 µs，故地板是同步不是传输）。

## 机器

a3-84 容器耐久、16 die 常被租户占；187 die 常空但容器约四分钟被回收
（docker 事件流证实，租户容器同样被销毁），宿主 `/tmp` 也会被清。
整机臂只能在 a3-84；187 只做能在一分钟内跑完的单 die 探针。
[回收器](evidence/ascend-decode/container-reaper-20261020/README.md)
