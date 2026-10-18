# TileRT-Ascend → Claude Code 交接

交接时间：2026-10-18 10:35，Asia/Shanghai。当前状态：环境准备完成，研究任务已按用户要求暂停。此次交接未启动新的整模型实验；恢复开发以用户后续指令为准。

## 1. 背景与目标

本地仓库 `~/repo/tile-ai/TileRT-Ascend`，分支 `main`，远端 `https://github.com/tsukiyokai/TileRT-Ascend.git`。下文相对路径均以此仓库为根。先读 `docs/plan.md`、`docs/lesson.md`、`docs/probe.md`；它们分别承载蓝图、经验和实验索引，不需要重新从聊天记录拼接设计。

唯一性能目标：AgentX，A3 单机 8 模组／16 die，GLM-5.2-W4A8C8，batch=1，8K 上下文，TPOT ≤2 ms（≥500 tok/s）。vLLM-Ascend 做 prefill，HIXL 交接 KV，TileRT-Ascend 做 decode。允许固定形状特化，以及修改所需算子与框架；当前不考虑 A5 或泛化。

上游 TileRT 是闭源 `libtilert_*.so` 的 Python 包装层；本项目在昇腾上重建执行运行时。目前可执行主线复用 vLLM-Ascend 模型层与 CANN 算子，再由本仓管理 KV、MTP 状态、Graph 和服务接线。完整常驻 tile engine 尚未实现。实现选择以 2 ms 预算为尺度：先复用已有 CANN，再修改与融合，能力缺失时才用 TileLang Ascend／AscendC 新写。

## 2. 已验证里程碑

| 里程碑 | 已证明的范围 | 证据入口 |
|---|---|---|
| vLLM-Ascend 基线 | 同一 checkpoint／8K IDs，32 个 1024-token 请求；TPOT p50=16.39、p95=18.29 ms。相同 temperature=0、seed=0 仍得到 32 条不同轨迹，后续使用冻结 teacher。 | `docs/evidence/ascend-decode/vllm-baseline-20261017/README.md` |
| 真实在线 PD | a3-gs 上新鲜 8K prefill→16 rank HIXL WRITE→TileRT M1 decode→HTTP 200，返回 8 tokens。P 退出后 D 加载，收到的 KV 始终驻留设备。 | `docs/evidence/ascend-decode/pd-live-20261017/README.md` |
| 独立数值诊断 | HF 差分定位 Div 1-ULP 与 SwiGLU 同步缺陷；精确点积又证伪 CPU 默认 oneDNN BF16 matmul。修正参考后已跑满 78 层曲线。 | `docs/evidence/ascend-decode/hf-corrected-20261017/README.md`、`docs/lesson.md` |
| Graph／MTP 接线 | M1 Graph 短测约 256.84→44.68 ms；M4 验证、接受／回退、KV 截断和 reset 已有设备对照。旧 M4 长诊断中位约 20.44 ms。 | `docs/evidence/ascend-decode/cann-graph-20261017/README.md` |
| teacher 精度推进 | 78 层×32 步驻留执行，BF16 head 为 31/32；对保存的 hidden 使用 BF16 权重／FP32 输出 head，M1／M4 均恢复 32/32。head-only 结果尚未替代完整新配置会话验收。 | `docs/evidence/ascend-decode/layer-major-20261018/README.md` |
| a3-84 环境就绪 | 统一环境、checkpoint、前缀、HIXL bridge 已到位；16 rank 变输入 BF16 MatMul Graph 与 12 KiB FP32 AllReduce 正确。尚未在该机跑整模型。 | 第 3 节的服务器验收记录 |

最新同源全词表 1024-output 对照的 TPOT 中位数是 20.674 ms，约 48 tok/s，见 `docs/evidence/ascend-decode/execution-gap-20261018/README.md`。这是旧配置的实测；最新优化组合没有长测数字。一个请求 303 轮、实际平均提交数 A=1023/303≈3.376，因此 2 ms 对应约 6.752 ms/round，当前约需压缩十倍。

已有局部收益及其验证边界统一见 `docs/plan.md` 的开发主线：gather＋FIA 的真实 M4 attention 89.05→69.15 µs；静态 CANN＋SuperKernel 将单层计算任务 dense 44→34、MoE 27→14；MTP 投影 TP 分片的孤立投影约 2.99→0.56 ms。它们尚未合成为已证明的整模型 TPOT 收益。

## 3. 当前可用环境：a3-84:trt-asc

2026-10-18 10:35 实时只读检查：16 die 健康、AICore 0%、无 NPU 进程；容器运行中。此为空闲快照，接手时重新检查本机，勿扫描其他机器或终止他人任务。

```bash
ssh a3-84
docker exec -it trt-asc bash
source /workspace/environment/activate.sh
cd /workspace/src/TileRT-Ascend
```

镜像 `trt-asc:a3-84-unified-20261018`，ID `sha256:40b2371a47be409554fa77d5a2f7bda684368eb5b52d0134ec86cc36feca294a`。容器当前只保持待机。

| 组件 | 实际版本／位置 |
|---|---|
| 驱动 | 25.5.5 |
| CANN | 9.2.0，V100R001C25B114，`/opt/ascend/cann-9.2.0` |
| Python | 3.11.12，唯一活动环境 `/opt/trt-asc/venv` |
| torch／torch_npu | 2.12.0+cpu／2.12.0；torch 的包名不代表没有 NPU 支持 |
| vLLM | 0.30.1rc1.dev447+gc4df37dfc.d20261015.empty；`/workspace/src/vllm-unified` |
| vLLM-Ascend | 0.19.1rc2.dev2828+gf558b58c1.d20261015；`/workspace/src/vllm-ascend-unified` |
| Transformers | 5.14.1 |
| Triton Ascend／TileLang Ascend | 3.2.2+trtasc.1；TileLang 源码位于 `/workspace/src/tilelang-ascend`，未取得独立版本号 |

关键输入与加载约定：

- checkpoint：`/workspace/models/GLM-5.2-w4a8c8`，96 个权重分片，78 层、hidden=6144。tokenizer 长度 154856，勿与 padded vocabulary 混同。
- 已冻结真实 8K 前缀：`/workspace/offline/prefix8k-allocation-20261017/capture/case-00`；teacher：仓库 `docs/evidence/ascend-decode/vllm-baseline-20261017/teacher-32.json`。
- HIXL：`/workspace/experiments/hixl-build/build-sync-contract/libtilert_hixl.so`，加载与五个 bridge 导出检查通过；a3-84 尚未进行新的在线 KV 传输。
- activation 已配置仓库 PYTHONPATH、HIXL 地址及已验证 SwiGLU 控制 overlay。TileRT 的展示型 pyproject 没有构建系统，不要为导入而盲目 `pip install -e .`。
- 默认 GMM1 控制来自 `/workspace/offline/cann-workspace-owner-20261017/kernel-overlay/vendors/custom_transformer`；新候选在 `/workspace/offline/cann-gmm1-product-build-20261018-03/overlay`。Fla-NPU 导入会前置自己的 OPP 路径；84 尚未验证实际加权 GMM 的最终 ELF 选择，整模型实验需记录真实加载来源。
- 权重由共享 NFS 承载；NFS 禁止 rename，完整载荷保留为 `.partial`，本地文件名通过相对软链提供。保留容器 `/workspace/models` 与 `/workspace/shared` 两个挂载及链接，勿把这些完整载荷当作失败残片删除。已核全量文件大小、非权重文件字节和 safetensors 头，未重算全量权重 SHA。

验收原件均在容器 `/workspace/migration/a3-gs-20261018-01/`：`environment.json`、`setup-exit.json`、`smoke-exit.json`、`smoke-rank-{0..15}.json`。smoke 退出 0，16 rank 全通过，AllReduce SUM=136，峰值 Torch reserved 6 MiB/rank；这是环境功能验收，不是性能测量。

a3-gs 原始实验与证据仍保留，迁移不包含所有历史大文件。此前该机有外部 `run_test_npu.py` 占用全机，不去干预；需要历史 raw 时按证据 README 的 `/workspace/offline/...` 路径读取。

## 4. 代码、Git 与未完成工作

本地 `HEAD == origin/main == GitHub main` 已在交接时核对：`8aa63fc7aeadfe6a5ac86178ee6862b89d634426`，提交 `perf(cann): isolate single-chunk GMM1 kernel rebuild`。a3-84 仓库干净但仍是 `e062d50cbc7e4cb6fb0cb76a4ef3448603060e5f`；恢复后先同步已发布提交，网络不便时可用小型 Git bundle。不要将服务器旧 HEAD 记成最新版本。

产品入口：`tilert/models/glm_5_2_ascend/` 下的 `cann_model.py`、`mtp_round.py`、`mtp_state.py`、`tp_runtime.py` 及 Graph 实现；PD 在 `tilert/pd_vllm/`；实验驱动在 `tests/glm52/`。实验 CANN overlay 有本仓外的 vLLM-Ascend custom-op 源码与构建物，实际 ELF、host/API/tiling 来源必须随实验保存。

交接时未提交的本轮文档／审计材料：`docs/plan.md`、`docs/lesson.md`、`docs/probe.md`、`docs/evidence/ascend-decode/layer-major-20261018/README.md`，以及该目录新文件 `mtp-chain-pmu-summary.json`、`mtp-two-chain-timing-audit.json`。这些结果可直接读，但尚未推送。a3-84 新环境事实在本交接与服务器记录中，尚未写入 plan/probe；plan 开头仍指定 a3-gs，恢复时更新。

另有独立工作必须原样保留：`docs/chat.md`、`docs/evidence/ascend-decode/streamed-decode.md`、`docs/evidence/ascend-decode/layer4-input-20261017.json`、`tests/glm52/audit_cann_ranks.py`、`tests/glm52/check_cann_offload.py`，以及被忽略的日志／pyc。不要 `git add -A`、reset 或清理。上次提交范围及保护文件哈希记录在本地 `/tmp/tilert-gmm1-build-commit-scope-20261018.json`。

最近两项结论：两次 draft 合成一个 Graph 的 30 对计时仅中位省 0.089 ms、15/30 胜，停止默认集成；PMU 已匹配全部 7936 个计算任务，但存在 TASK 10.52 µs 对应计数换算 4.81 ms 的反例，长尾原因仍未确定。不要将计数关联写成计算、网络或抢占的已证根因。证据都在上述 layer-major README 与待提交 JSON。

小项待结：GMM1 单分块跳过两个冗余屏障，微测约省 3.679 µs；隔离重建已验证仅 19.82 s，完整 MoE 控制路径 13,249,536 元素逐位通过，候选完整 MoE 尚未跑。a3-gs 的 `/workspace/offline/cann-gmm1-singlechunk-moe-20261018-01/continue-candidate.sh` 已准备；原退出 1 是 ELF 收据误把验证器哈希读取算成运行时加载，修正后的控制收据通过。恢复前读 `docs/evidence/ascend-decode/gmm1-tiling-20261018/README.md`；候选续跑脚本尚未迁到 84。此小项不应挤掉整模型实验。

## 5. 恢复后最重要的下一步

1. 在 a3-84 同步源代码、确认实际 OPP／head／dtype／通信配置，完成新配置的冻结 teacher 会话验证。先用已验证 SwiGLU 控制与 `--fp32-head`，逐项加入优化。HF 独立差分继续诊断 kernel；同 kernel／dtype／layout 配置要求 bit-exact 或 ≤1 ULP，跨配置记录分布与索引诊断，最终以冻结轨迹验收。具体标准遵循 `docs/plan.md` 第 4 节。
2. 在这台空闲整机做同输入、同精度、profiler 关闭的整模型对照，量出最新组合实际 TPOT、每轮耗时、接受数 A 与专家并集。先 3 个长请求筛选，再做正式 32 请求；一次只改变一个待归因因素。对照入口 `tests/glm52/bench_cann_mtp.py`，参数含 `--checkpoint`、`--capture-case`、`--output`、真实 `--source-commit`，长测用 `--max-tokens 1024 --capacity 9344 --warmup 3 --requests 3`。产品开关已有 `--target-graph --draft-graph --draft-projection-tp --compile-target --super-kernel --attention-kernel gather_fia --fp32-head`，组合结果尚待验证。
3. 依据整模型差距采 3–5 轮短 trace，分开 target M4、MTP draft、head、算子执行和 rank 到达偏斜，选一条支配路径做优化。过去共享环境约 49 ms target verify、约 16.8 ms 两次 draft 只能提供线索。逻辑 HBM 预算已逼近 2 ms；不能预设消掉启动／同步就一定达标，见 plan 的实际专家并集预算。

正式验收仍是八个固定 8K 输入、各 teacher-force 32 步 greedy 100% 一致，以及 1024-token 轨迹一致性；预热后 32 个长请求 TPOT p50／p95 均≤2 ms。在线 MTP、最新组合完整轨迹、a3-84 整模型性能与最终验收都未完成。逐层曲线只用于定位，不能复活旧 NRMSE／绝对 logits 阈值来替代端到端标准。

## 6. 工作约定

重型构建、下载、模型执行均放服务器；检查短小有用，满机时推进 CPU 分析。保持 plan／lesson／probe 简洁且与证据一致；显著且已验证的改动按 `~/note/inbox/git-conv.md` 定范围提交并推送，保留无关工作；该文件含敏感内容，勿整份输出。编码遵循 `~/note/inbox/clean-code.md` 与 `~/note/inbox/pep-0008.rst`。

已停止的支线及原因见 plan／probe：独立 one-shot 归约仍约 35.716 µs、超预算；O 投影融合相对真实安装态控制无收益；GMM1 baseN=128 无收益；两次 draft 合图无稳定收益。重新开启须有新机制与可证伪实验。若启用子代理，先让其强化问题定义，用户指定 `gpt-6.1-sol`、`max`，最多 3 个并发（含主代理）。

接手的首要产物是空闲 a3-84 上可信的整模型精度与同源长测。让这组数字决定下一项实现，而不是继续积累无法兑换为 TPOT 的局部收益。
