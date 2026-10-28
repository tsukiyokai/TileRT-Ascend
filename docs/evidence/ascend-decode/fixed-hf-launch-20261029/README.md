# 修正运行包后的整模型尝试

7.150.11.49、9382、TP16、CANN9.2、固定镜像`3892a691…`，源码`3fcfd8b`。SFA设备消费者为block1版本`b7062ab6…`，OPP显式保持overlay优先；同一冻结8K前缀、32个teacher输入及已完成的独立FP32-head HF参考。[启动脚本](launch.sh)、[源码摘要](launch.sha256)、[即时16die空闲记录](before.txt)与实际接收的runtime/SDK/prefix字节摘要均保留。

[最后观察](interruption.json)为68/96分片、加载进度16:59；rank报告与`.pt`工件均为0。[Docker事件](docker-events.jsonl)记录启动后1160秒收到SIGKILL 9，随后die 137、destroy；删除发起方未知。之后16die被另一个VLLM作业占用，本方未在该节点重建作业。它是环境中断记录，没有模型数值或新增TPOT结果。

174节点的[后续准备](checkpoint-174.json)先以[CPU顺序读取](hash-model-view.py)核验109文件、405008283962B，再做设备占用检查。核验用时123.310秒且SHA全部匹配；这是数据与页缓存准备，不是A3算子或TPOT收益。整模型复验仍由新作业实际结果验收。
