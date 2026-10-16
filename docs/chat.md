目标：打榜AgentX,A3,GLM-5.2-W4A8C8,单机,batch=1,seq_len=8K,目标tpot=2ms
A5的话则支持GLM-5.2-W4A4，当前不考虑A5。
实验环境：a3-gs:trt-asc
以达成topt目标为前提，允许修改、新开发glm5.2所需的一切算子、框架、配置。
kv传输功能由HIXL提供，prefill由vllm接管，decode由tileRT-ascned接管，只关注tpot这一个指标，因此主要聚焦对TileRT-Ascend的开发和优化，但不限制对基础cann包里的蒜子进行有价值的最小修改。
允许针对batch=1等条件进行特化，现阶段不考虑泛化。
请基于 `/Users/shanshan/repo/tile-ai/TileRT-Ascend` 的实际代码，在 ~/repo/tile-ai/TileRT-Ascend/docs/plan.md 写一份简短的 TileRT 昇腾后端的设计，直觉上正确即可，禁止防御式写作和过度设计。先用实测或可查证的昇腾 A3 HBM 带宽、单卡权重容量、HCCL allreduce 延迟算出 single-user decode 的 TPOT roofline，给出「移植后最快能到多少 tok/s、瓶颈在哪」，让后续所有设计从这个数推导。明确写清本仓库只是闭源 libtilert_*.so 的 Python 包装层，因此工作性质是在 TileLang Ascend 后端上重建 tile 级 runtime，不是移植。从 wheel 的导出符号表提取全部 torch.ops.tilert.* 算子契约，以此清点真实工作量并分级（可直接复用 / 需重写 / 需新机制）。设计要给出可独立验证的验收标准：microbench 数字、算子清单覆盖率、与 HF reference 的逐层 logits parity。