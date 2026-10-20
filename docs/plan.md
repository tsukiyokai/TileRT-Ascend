# TileRT-Ascend：A3 单用户 2 ms 解码

目标：AgentX，A3 单机 8 模组／16 die，GLM-5.2-W4A8C8，batch=1，8K 上下文，TPOT 达到 2.x ms；开发验收继续采用 p50／p95 均≤2 ms。允许针对该形状优化。vLLM-Ascend 负责 prefill，TileRT-Ascend 负责 decode；正式打榜使用 [AgentX workload 与统计规则](https://inferencex.semianalysis.com/agentx/methodology)。

状态：按要求暂停；10-20 的78层融合因果实验及审计已完成，无运行中的实验。恢复时先重查环境，再执行已准备的关闭融合完整编译请求，验证33个冻结输出；通过后测同路径TPOT。

## 当前证据与优先级

截至 2026-10-19，已测最好请求 TPOT 为 11.6198 ms。pinned `seq_lens` 与 pageable 同会话交替，两对完整对照分别节省 3.384／3.791 ms/round；六个 pinned 请求均快于九个 pageable 请求。该组受共租户影响，运行利用率设为 0.78，pageable 为 12.3–12.6 ms/token；绝对最好值与配对差分别保留。[测量记录](probe.md)、[staging 机制](evidence/ascend-decode/metadata-h2d-20261019/README.md)

width-8 MTP 是当前计时控制组，单一冻结输入的平均实际提交数约 4.507，2 ms 对应稳态轮预算约 9.01 ms。旧相位参考为 verify 39.68、chain 10.27、conditioned 2.08 ms/round；这些来自旧配置，须随新组合重测。[宽度扫描](evidence/ascend-decode/width-sweep-20261019/README.md)、[同步相位](evidence/ascend-decode/phase-synced-20261019/README.md)

10-20 复核：旧 `chain-w8` 三条轨迹均在输出 offset4 与冻结 teacher 分叉；prompt IDs 相同，hash 差异仅来自序列化。147 轮／A=6.959 只用于旧未验收轨迹的策略筛选。可信 teacher33 的独立 CPU 回放中，width8／16／32均为28轮、新增32个输出，计划验证行数224／448／896；这段窗口加宽无接受收益；相同30个新增输出，冻结MTP4候选为14轮、lookup8为26轮，完整轮价比须低于1.857才由MTP胜出。[短窗口回放](evidence/ascend-decode/lookup-contract-20261019/teacher33/README.md)。同一前缀的完整M32/SFA已通过32个冻结标签；shared-FIA完整对照为30/32，先定位其数值差异及旧链首个分叉，再核33输出的串行／M32／Graph／reset与完整请求。[证据](evidence/ascend-decode/shared-fia-model-20261020/layer-major/m32/full/README.md)

2026-10-20 发射地板定调优先级:把实测 per-call 地板(AI_VECTOR_CORE 0.70、AI_CORE 2.30、MIX_AIV 12.04、MIX_AIC 13.58 µs)乘以一轮真实调用数,3008 次调用给出 11.512 ms;而 2 ms/token 在 A=6.959 下只允许 13.92 ms 的轮。地板占预算 83%,所以目标要的是更少的发射而不是更快的 kernel。地板的 82% 是 505 次 MIX_AIC,每层 6.5 个。同一捕获内的对照:MlaPrologV3 作为一个融合算子跑在声明字节的 1.1 倍,所有未融合邻居是 3 到 91 倍。两个 grouped matmul 合计 12.388 ms 对 3.484 ms 的 W4 权重(并集 44.35 时每轮 4.023 GB),3.56 倍。[发射地板与分族预算](evidence/ascend-decode/critical-path-20261020/README.md)

TileRT 自己在 ROCm 上对同一个模型选的粒度已经拆出来了,它直接回答了"整层融合"这一问."pure_mla_moe_layer_kernel" 一个 kernel 覆盖 norm→MLA→O 投影→TP all-reduce→router→routed experts→shared expert→SwiGLU→第二次 all-reduce→bf16 尾巴,48 个特化、30–34 个 s_barrier,可证明就是两个半边(各自也单独发布)的拼接。但它不是 persistent megakernel:每层退出,无工作队列、无 grid-wide barrier、零 atomics。决定性的资源事实是 LDS 恰为 95232＋13840×tokens,而融合层 kernel 在三个宽度上与只做 attention 的 kernel 相等——把 MoE 半边融进来不需要任何额外片上状态,因为它复用 attention 的 arena。TP 归约在 kernel 内部,每层两次,靠对 32 位字高半部那个 16 位 epoch 标记自旋等待。KV 选择仍是独立发射,因为 topk 要 158896 B 的 LDS(总共 160 KiB)。另外它没有 W4:权重是 uint8 FP8 E4M3＋f32 scale,反量化是 MFMA 前紧挨着的四条 v_cvt;投机宽度(1/2/4)是编译期模板参数。[ROCm 结构](reference/tilert-rocm-glm52-structure.md)

手写 megakernel 的调研已把几个比当前砖块更大的杠杆钉出来,按量级排:`enableMixDualMaster`——MIX 默认由 AIV 通过消息驱动 AIC,双主模式下两者独立执行不依赖消息,对着每轮 505 次 MIX_AIC × 13.58 µs = 6.86 ms;前提是 1:1,或 1:2 且 A、B 两侧都开 IBSHARE(第六个 MatmulType 模板参数),对每 rank 128 列的 moe_intermediate 是真约束不是免费开关。Fixpipe 硬件自带 S322S8／S322FP16／FP322BF16 等量化反量化和 NZ2ND,在数据写出 L0C 之前完成,直接对着 `Cast` 一族每轮 2.300 ms 搬 0.076 ms 字节的纯开销。层间屏障用 `SyncAll<false>()`:MIX 下一条指令做三件事(AIC 等所有 AIV、所有 AIC 互等、所有 AIV 等 AIC),用 flagId 11/12/13 不占手工分配区;两个坑是 MIX 里写 `isAIVOnly=true` 会静默退化成纯向量屏障,以及 numBlocks 超过物理核数会让框架插入错误同步而挂死。TP 边沿不能同时用 `SyncAll` 和 `aclshmem_barrier_all`(「Barrier APIs conflict with SyncAll」),所以 TP 侧要自己做 epoch 标记的 flag;而 LCCL 的做法是 flag 只走 MTE(`DataCopy` 加 `MTE2_S`／`S_MTE3`)从不走标量存取,因此完全不需要 dcci——这和 2201 上「只有 DCache 不一致、且只有标量单元用它」对得上。A3 只有 `atomic_add`,没有 atomic set／CAS,epoch 只能来自标量入参、单一属主槽位或加法。[参考](reference/ascend-fusion-mechanics.md)、[TP 归约](reference/low-latency-tp-reduce.md)

| 顺序 | 路线 | 下一项与通过条件 |
|---|---|---|
| 1 | lookup 提议与执行路径 | CPU 状态／串行对照通过；等价增量索引将每请求 CPU 查找约 1.03 s 降至 21 ms。独立开关 shortest2→1 的 width32 重放为 174→147 轮。下一步测 width 8／16／32 的真实接受数和请求 TPOT。该轨迹 A=6.959 对应 2 ms 的轮预算 13.92 ms；设备收益尚未测得。[重放](../tests/glm52/ngram_acceptance.py)、[实现](../tilert/models/glm_5_2_ascend/lookup_draft.py) |
| 2 | 多 query／跨层共享 | 可选 `shared_fia` 支持 M8／M16／M32：完整 KV 跨 query 复用，21 个 producer 构建选择 mask。各自四层 Graph 配对改善14.6%／11.9%／37.0%；M16为373.365→328.620µs，29/30对胜出，36项共享／Graph／reset检查逐位一致。[M16](evidence/ascend-decode/fia-mask-20261020/joint/m16-screen/README.md)。M1／MTP保持SFA。下一步以冻结teacher／串行等价开路，再与lookup宽度对照测整轮；旧gather_fia M≤6门槛不变。[证据](evidence/ascend-decode/fia-mask-20261020/README.md)。 |
| 3 | 投影与归约流水 | 既有真实 O 投影＋完整归约对照：原装 CANN＋push16 为162.141 µs，融合 publication＋consumer 为162.781 µs，未减少 launch、无净收益。该反证仍然有效，但它测的是 publication＋consumer 这条边界。10-20 独占时间复核换了边界并重开本路线：两轮完整对照中 TP 归约独占 5.35 ms/round，而全部 HCCL 流独占 0.000 ms；其 FP32 cast 外壳 146 站共 1.99 ms/round；AI_VECTOR_CORE 用 8.71 ms 搬运 0.35 ms 的字节。三个具名 brick，按代价从小到大：新增可选 `--plain-moe-combine`，把 `fused_moe.py:236` 的单元素 `_foreach_add` 换成直接相加——同一捕获内同形状配对为 1.820 对 21.282 µs，且与行数无关故全是结构，75 层约 1.4 ms/round，同为 BF16 逐元素加法故逐位一致；把 widen／narrow 折进按 rank 定序的 one-sided reducer（同一结合序，故逐位一致，并把线上字节减半）；再融合其余 residual combine（每层 Muls 8.5 µs）。通过条件照旧：单 die 同 ELF 配对对照与冻结 teacher 逐位一致，再测整轮。停止修补不含 scale／有效 tiling 的 INT8 演示。[完整反证](evidence/ascend-decode/execution-gap-20261018/README.md#o-publication-and-complete-reduction)、[独占时间](evidence/ascend-decode/critical-path-20261020/README.md) |
| 4 | MoE 活跃专家表 | GMM1／GMM2均有可选 `--compact-experts` overlay，保持NZ权重、MSD算术和tile分配。单die同ELF：GMM1的M1／M8／M32随机90.90→49.36／111.56→83.91／163.28→141.16µs；GMM2的M1／M8为62.42→34.86／91.27→72.61µs，M32随机无收益、shared8为64.59→41.40µs。各自24组跨二进制和4次真实权重检查逐位一致。可选 `--adaptive-tiles` 初版在每专家≤16行时用32×1024；初版正式ABBA的M32随机143.90–146.98→106.18–107.86µs，M1／shared8分别退化1.18–1.82／5.79–7.01µs，46组Graph与4次真实权重逐位一致。[动态tiling](evidence/ascend-decode/gmm2-compact-20261020/adaptive/README.md)。默认关闭；按真实lookup专家并集选择，再分别核整模型teacher和轮耗时。[GMM1](evidence/ascend-decode/gmm-compact-20261020/README.md)、[GMM2](evidence/ascend-decode/gmm2-compact-20261020/README.md) |
| 5 | C8缓存写入合并 | 现有CANN `ScatterPaKvCache`一次写INT8 key与FP16 scale原始字节。真实dynamic_quant→write_cache路径18组整缓存Graph/reset检查一致；M8 28.97→10.34µs，10/10对胜出。13行补丁仅在隔离模块验证，下一次整模型teacher与compiled Graph分别开关。[证据](evidence/ascend-decode/index-cache-pair-20261020/README.md) |

每条路线一次只改一个可归因因素。按新策略实际经过的位置、宽度和专家并集计价；减少 launch 数量、外推行成本或套用 64-expert 数字均不能证明端到端收益。lookup 的精确 CPU 重放及外推撤回依据见[复制提议](evidence/ascend-decode/lookup-drafting-20261019/README.md)，须由与部署一致的重放及设备对照更新。

已通过teacher的M32选集支持并集压紧研究：21个producer、4 heads、假设16×128 tile，union+mask工作量为全cache的45.60%，精确级联为49.42%；两者均完整保留选集。先核动态并集长度如何留在设备上，再测构建＋gather＋attention整段，不将较小工作集直接称为HBM收益。[结构计数](evidence/ascend-decode/shared-fia-model-20261020/exact-sharing/README.md)

GMM1 的 full-A、N 拆分与 K512 筛选均未改善 M32 随机路由的原始对照，关闭该调参分支。[K512 结果](evidence/ascend-decode/gmm-compact-20261020/k512/README.md)。每层每新增一个专家，每 rank 增加1,256,448 B逻辑权重读取；下一项先采实际lookup的75层专家并集与接受数，再定宽度和kernel策略。[预算与边界](evidence/ascend-decode/shared-fia-model-20261020/routes/budget.md)

同一旧轨迹的M64／128／256复制重放均为144轮，对照M32的147轮；扩宽只允许平均轮耗时增加2.083%，设备比较保持8／16／32。[宽度上限筛选](evidence/ascend-decode/lookup-contract-20261019/wide-frontier/README.md)。原生vLLM局部top-1虽减少词表通信，实际TP2 Graph仍退化14–30µs，暂不接入。[head对照](evidence/ascend-decode/reduced-head-20261020/README.md)

固定32行的纯复制树，4条续段共享前缀后在同一旧轨迹为132轮／A=7.75；最多容许完整轮成本增加11.36%。A3 C8 indexer须在top-k前排除兄弟分支，现有有序输出不足以保证前缀／尾部合并的同分次序；先用已验收输入与实测轮预算确认价值，再改树选集与KV提交。[复制树证据与能力边界](evidence/ascend-decode/lookup-contract-20261019/copy-tree/README.md)

单层驻留的M32/SFA现已完成TP16全部78层：FP32 head的32个冻结标签在16 rank全部通过，最终logits跨rank逐位相同，峰值899MiB/die。旧M1比较中582/660对边界／cache相同；hidden／residual到层67相同，层68/query13首次不同，final norm NRMSE0.02398。该诊断与标签通过分别保留。同源shared-FIA完整对照仅30/32，query3／21失败，final norm NRMSE0.15993；固定Q/KV/mask的8层单die对照中，FIA行无效修正0→2的524,288个输出逐位不变，关闭该参数分支。SFA在旧链首差前缀的top1余量仅0.002617，冻结两轮重放同时检查算术与拒绝尾部覆盖。随后仍须token-major Graph/reset、串行提交与真实轮耗时。2GiB/die配额用于共享机器上的功能验证，不产生TPOT结论。[完整证据](evidence/ascend-decode/shared-fia-model-20261020/layer-major/m32/full/README.md)

单die补充：层66–73的M32真实attention已完成，两组C8／BF16 producer及consumer的19份新cache与旧M1逐位一致，索引选集全同；共享／逐层重建FIA输出逐位一致，FIA对SFA的pre-O NRMSE为0.00224–0.00319，峰值147MiB。该截面止于O投影前，下一项仍是完整teacher传播。[后段attention](evidence/ascend-decode/shared-fia-model-20261020/late-attention/README.md)

冻结residual重建的75层teacher路由，M8／16／32平均专家并集44.35／68.07／99.36。GMM2新增条件：M32、每专家≤32行、活跃数≥64、超过16行者占比≤1/8；原小组条件保留，覆盖40→75层。正式新旧ABBA的三组teacher分布均改善11.18–22.12µs；58次Graph对CPU MSD逐位一致。同路径控制差值反转，开销未定；默认仍关闭，下一项核整模型teacher与实际lookup轮成本。这些统计不含拒绝尾部。[路由](evidence/ascend-decode/shared-fia-model-20261020/teacher-routes/README.md)、[发布对照](evidence/ascend-decode/gmm2-compact-20261020/guard-release/README.md)

两轮冻结M8已通过TP16全部78层、4/4可信标签；对应hidden／residual／新KV及最终FP32 logits与已验收M32逐位一致，旧首差位置输出2524。[两轮证据](evidence/ascend-decode/shared-fia-model-20261020/two-round-replay/README.md)。历史编译图有3处稠密层norm＋量化融合。已关闭该开关并核实际IR；实际ACLGraph／SuperKernel的135项原始张量检查通过。TP16全部78层中，仅开启这3处融合，即使4/4可信标签变为3/4，并在所有rank复现2524→6500的旧首差；1,264份模型产物与48份norm快照已核验。下一项是关闭融合的完整编译请求，核33个冻结输出后再测同路径TPOT。[融合对照](evidence/ascend-decode/shared-fia-model-20261020/norm-quant-fusion/README.md)

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
- W4 MSD：Fixpipe先清零scale的FP32低13位，再生成两个FP16部分积，按FP32 `16H+L+checkpoint_bias` 合成；A8倍率使用 `Div(127,amax)`；RoPE使用BF16系数、FP32乘加、末尾一次BF16舍入。[W4](evidence/ascend-decode/w4-msd.md)、[Fixpipe合同](evidence/ascend-decode/gmm2-compact-20261020/fixpipe-official-receipt.json)、[A8](evidence/ascend-decode/a8-rounding.md)、[RoPE](evidence/ascend-decode/rope-rounding.md)
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

按需求分层，先跑能决断的最便宜那层:单 die 数百 MiB 的配对算子微基准(满载也能跑,秒级)；16 die 无模型的集合通信／协议筛选(分钟级)；16 die 载模型的接受数、专家并集与请求 TPOT(只有这层要排队)。闸门拒绝时把等待限在几十分钟而不是一整天,拒绝的时间花在第一层上——2026-10-20 两个 driver 重试 163 次、空等 11.5 小时,全程零产出;同一天单 die 微基准几秒就定了 MoE combine。

需要整机的队列保持短且按决断力排序:先 `--plain-moe-combine` 配对(纯轮耗时、逐位一致、无接受数混淆)，再 unigram 配对(接受数与专家并集)，再宽度 8／16，最后 one-sided 归约首次端到端(可能直接在 aclshmem 初始化失败,故排最后)。每臂只回答一个问题,其余删掉。

实验结果见 [probe.md](probe.md)，可复用结论见 [lesson.md](lesson.md)，发射与取证步骤见 [workflow.md](workflow.md)。
