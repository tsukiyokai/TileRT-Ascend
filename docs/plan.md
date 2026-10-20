# TileRT-Ascend：A3 单用户 2 ms 解码

目标：AgentX，A3 单机 16 die，GLM-5.2-W4A8C8，batch=1，8K 上下文，TPOT 2.x ms。
vLLM-Ascend 负责 prefill，TileRT-Ascend 负责 decode。

## 实测位置（2026-10-20）

最好 **9.3794 ms**：自动机起草器 @ 宽度 12，同机对照基线 11.2312。
拆分过：约 12 个百分点来自宽度，3–5 来自起草器（`chain8 @16` = 9.8913 是那个对照）。

一轮的代价模型，四个宽度拟合、残差几百微秒：

    round = 45.88 ms + 0.814 ms 一行

固定项与宽度无关，单独决定可达性：**即使每轮全收**，宽度 12 的 TPOT 地板是
4.637 ms、宽度 32 是 2.248。所以起草器不是主线，固定成本才是。
[代价模型](evidence/ascend-decode/round-cost-model-20261020/README.md)

一轮 55.424 ms 的构成，逐项实测：

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

## 已建成、未进引擎

| 杠杆 | ms | 状态 |
|---|---|---|
| 路由置换吸收 | 7.1 | 替掉 init_routing 45.772 + finalize_routing 55.979 |
| 单边规约 | 6.5 | 十六卡正确，32.488 µs 对 HCCL 201.995 |
| MoE 单 kernel | −16.3 | 256 专家 365.887 µs 对 vendor 157.176，慢 2.33x |

前两项合计 13.6 ms，折算 TPOT 约 8.0 ms。两项共用同一个交付点：
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

## 机器

a3-84 容器耐久、16 die 常被租户占；187 die 常空但容器约四分钟被回收
（docker 事件流证实，租户容器同样被销毁），宿主 `/tmp` 也会被清。
整机臂只能在 a3-84；187 只做能在一分钟内跑完的单 die 探针。
[回收器](evidence/ascend-decode/container-reaper-20261020/README.md)
