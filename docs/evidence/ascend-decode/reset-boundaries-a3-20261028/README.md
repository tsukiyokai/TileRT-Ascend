# 重复前向的 MoE 首差

9382、TP16、CANN9.2、同一冻结8K前缀与32个teacher输入。使用[边界采集器](../../../../experiments/cann-precision/reset_boundaries.py)的`7877b063120b2ab13fd32af06ded77cfdfffcc83`版本，在32步M1后依次执行清尾部M32、保留尾部M32、再清尾部M32。前向中只作设备端独立复制，整轮完成后才读CPU；无计时或精度通过声明。

[16rank实际读回](boundaries-v1.json.gz)包含各rank原始报告SHA、每次65个边界、工件SHA、真实退出0及镜像/源码身份。三次均在位置8224结束，最终reset到0。归一化、SFA与attention的已采边界均重复一致；首个已采差异落在零基层3的MoE。

rank1、rank2在两次重放的TP输入已经变化，首次坐标分别为`[15,0]`、`[9,0]`，最大绝对差分别为`5.7220458984375e-6`、`1.1444091796875e-5`。再次清尾部时rank3也在TP输入变化，从`[16,0]`开始。其他rank先在TP输出看见差异。仅凭rank0的TP输入相同不能归因于HCCL；当前首差位于部分rank的本地MoE生产路径，共享/路由专家与本地合并继续细分。清尾部未消除该现象。

48份原始Tensor与执行源码、启动器、失败的时间解析guard及修正后guard均保留在共享NFS `/trt-a3-evidence-20261028/reset-boundaries-v1-source7877b06.tar.zst`；1408002890B，SHA256 `98b4bb5c7f48ea3abcc401d2c22535495e56e91c882c07b2e89cdc127e7a1494`。归档已完整解码并核对48个`.pt`成员。此采集改变了观测指令，不替代先前无此细分采集的失败记录，也未通过HF、八输入或在线TPOT验收。
