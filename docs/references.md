# 来源与实现定位

以下核对日期保留原项目记录；本轮仅按源码修正文档，没有重新检索网页。原核对日期：2026-09-17。以下是论文和项目的一手资料。主分支链接用于解释实现依据；
实际基准复现实验必须另行固定真实 commit、数据哈希、依赖版本和运行协议。
本仓库没有伪造未下载源码的 commit，也不把参考仓库的成绩作为自己的实验结果。

1. Ilya Kostrikov, Ashvin Nair, Sergey Levine. **Offline Reinforcement Learning with Implicit Q-Learning**.
   https://arxiv.org/abs/2110.06169
   依据：expectile V、数据动作上的 Q 回归、advantage-weighted behavioral cloning。
   “in-sample”针对训练时 Q 的动作查询，不是评价阶段不能输出新动作。
2. 作者 IQL 实现：https://github.com/ikostrikov/implicit_q_learning
   PyTorch 参考：https://github.com/gwthomas/IQL-PyTorch
   CORL IQL：https://github.com/corl-team/CORL/blob/main/algorithms/offline/iql.py
   本例重新实现训练与工程，不直接附带上述项目源码；Gaussian actor、双 Q、
   target update、数值范围和更新次序进行了源码核对，未声称数值逐位复现论文结果。
3. D4RL 官方仓库：https://github.com/Farama-Foundation/D4RL
   转换逻辑：https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/__init__.py
   参考分数：https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/infos.py
   依赖声明：https://github.com/Farama-Foundation/D4RL/blob/master/setup.py
   环境注册：https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/gym_mujoco/__init__.py
   底层数据与环境适配支持 12 个 legacy locomotion-v2 任务，学生主协议选其中不含 random 的九项。数字常数来源在代码中注明；
   不附带 D4RL 数据、MuJoCo、第三方源码或它们的再分发许可。
4. Karpathy autoresearch：https://github.com/karpathy/autoresearch
   参考“修改—真实运行—记录—保留/回退”的基本实验循环。本项目不是其代码分支，
   没有复用外部编码智能体运行时，也没有完成同成本性能优势比较。
5. 结构化模型输出接口的官方资料：
   https://developers.openai.com/api/docs/guides/structured-outputs
   Anthropic Messages API：https://platform.claude.com/docs/en/api/messages
   本项目的 compatible/anthropic 适配器有协议级 HTTP 模拟测试，
   本次没有用实际模型服务凭据进行在线验收。
6. 用户提供的《大模型智能体》章节：规划、记忆、工具使用、反馈、驾驭工程、
   运行评价、能力更新和递归自我改进部分。
   本教材范例对应规划、记忆、工具、反馈与驾驭工程；自我提升／RSI 仅属于章节的更广内容，不是本实现的功能。具体 IQL 与实验系统属于本项目实现。
   章节中的外部成绩与前沿产品描述不作为本项目已验证结论。

## 当前工程语义与沿用的调研接口来源

固定 IQL 保留；候选允许完整 Python，而不是只允许损失表达式。固定数据、最终 checkpoint、
独立评价和真实轨迹链接继续保留。v0.6 自适应研究在第一次运行前固定完整学习率日程，兼容累计端点通过检查点续训；独立 train 未指定 schedule_steps 时仍按本次请求长度初始化。
奖励、checkpoint 和评估协议可能不同于论文，参考论文成绩不得当作本项目成绩。RSI 已移除。

以下官方接口文档用于核对适配器，读取于 2026-09-20；网页可读不表示容器客户端已成功联网。

- arXiv API / Atom / 分页 / 版本：https://info.arxiv.org/help/api/user-manual.html
- arXiv API 使用条款：https://info.arxiv.org/help/api/tou.html
- Crossref 元数据 REST：https://www.crossref.org/documentation/retrieve-metadata/rest-api/
- GitHub Git Trees：https://docs.github.com/en/rest/git/trees
- GitHub 仓库内容：https://docs.github.com/en/rest/repos/contents
- Brave Web Search：https://api-dashboard.search.brave.com/app/documentation/web-search/get-started

用户教材的上下文选择、按需装配和压缩负责概念组织；具体接口实现、字符限额和代码测试属于本项目工程选择。
未验证任何真实模型上的研究效率提升。
