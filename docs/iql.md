# 固定 IQL 与开放候选的算法边界

## 固定 IQL

基线位于 `tasks/iql/trainer.py`，保留原来的计算顺序与默认配置：

\[
L_V=\mathbb{E}_{(s,a)\sim D}[|\tau-\mathbf{1}(Q_{\bar\theta}(s,a)-V(s)<0)|
(Q_{\bar\theta}(s,a)-V(s))^2],
\]
\[
y=r+\gamma(1-d)V(s'),\qquad
L_Q=\tfrac12\mathbb{E}[(Q_1(s,a)-y)^2+(Q_2(s,a)-y)^2],
\]
\[
L_\pi=-\mathbb{E}[\min(\exp(\beta A(s,a)),w_{\max})\log\pi(a\mid s)].
\]

actor 是均值经 tanh 的未压缩高斯，推理取均值；不是 tanh 变换后的概率分布。
这是固定、可复验的参照，而不是所有 in-sample 方法必须遵循的结构。
IQL 原论文：[Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)。
实现细节同时参考 [CORL IQL](https://github.com/corl-team/CORL/blob/main/algorithms/offline/iql.py)。

## 本项目采用的 in-sample 定义

Q/V 的更新不得依赖数据外动作的价值估计。检查状态—动作配对，不能把别的状态出现过的动作
视为当前状态已经观测过的动作。生成动作估值进入 target、regularizer 或额外价值梯度均不合法。
`no_grad`/`detach` 只改变梯度，不改变该估值来自哪里。

policy 更新与部署可以生成动作，并查询只读 Q；这些值不能反馈进价值学习。
因此本版准确描述为 **in-sample value learning + 可开放的 policy extraction**，
而不是“所有阶段从不查询新动作”。

## 交付的开放源码起点

`ensemble-mixture`：三成员 Q、LayerNorm/SiLU、混合高斯策略、真实数据多步目标、
Huber Q 损失和不同策略更新频率。`diffusion-extraction`：条件 DDPM 去噪、
可选 actor-Q 梯度指导、多候选动作的冻结 Q 选择。

IDQL 说明了 in-sample critic 可以结合更灵活的生成式策略提取：
[IDQL 原论文](https://arxiv.org/abs/2304.10573)。这里的模板只是开放接口的研究起点，
与论文目标、实现和协议不完全相同，**不称为 IDQL 复现，也不借用它的实验成绩**。
扩散模板使用短 cosine 噪声日程，正式实验应审阅日程、网络规模与推理预算。

模板同时改变多个因素。评价效果时需分别比较架构、目标、调度与提取方法，
同时报告优化器步数、训练成本和采样推理开销。

## 有符号目标

\(L-c\) 与 \(L\) 的梯度相同。因此非负不能成为通用的候选合法条件。
本版保留有限值、形状、梯度与模块归属检查，撤去 Q/V 损失非负限制。
旧接口的 actor importance weights 仍按其既有非负加权似然语义处理；
新 Python 接口不要求使用 importance weights 或对数似然。

## 多步与终止

导入时记录真实 raw-row successor，先处理 timeout 的丢弃语义再建立索引。
不能跨 terminal 或不同 episode 拼接数据。真实 terminal 不 bootstrap；
缺失后继/数据边界时，可从该转移真实 next observation 的 V bootstrap。
没有链接的旧文件不被自动推断成一个长轨迹。

## 不把扩展误当作验证

运行时检查建立在候选诚实登记与使用接口的前提上；源码仍需审查。
测试中的五维 Q 输出是接口覆盖，不是某篇分布式 RL 算法的复现。
目前 D4RL、GPU 和真实模型服务是否执行，以当次 evidence 为准。
