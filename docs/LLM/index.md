

## 全新的可编程抽象层

Agents、Sub-agents、Prompts、Contexts、Memory、Modes、Permissions、Tools、Plugins、Skills、Hooks、MCP、LSP、Slash Commands、Workflows、IDE Integrations



agent 智能体

subagent 子智能体

prompt 提示词

context 上下午

memory 记忆

MCP

LSP

mode(如Plan Mode)

你需要掌握的不再是语法细节、算法实现、框架特性，而是：

- 如何设计和使用AI代理（Agents）

- 如何拆解任务给不同的子智能体（Sub-agents）

- 如何给AI提供恰当的上下文（Context）

- 如何让AI记住项目的历史和决策（Memory）

- 如何编排AI的协作流程（Workflows）

- 如何与MCP、LSP等新协议打交道



Karpathy的原话一针见血：

> 我们需要构建一个全局心智模型，以驾驭那些本质上具有随机性、易出错、难解释且持续演变的实体——它们突然与传统严谨的工程实践交织在一起。



**Step 0：立即接入****AI****代码审查**

第一步是最简单、风险最低的：在你的代码库中接入AI驱动的代码审查工具。Graptile、CodeRabbit这些工具会在PR阶段自动检查代码质量、发现潜在Bug。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/UicQ7HgWiaUb1L99MvDEXtNGhuZ1lp01IpyHldKYW2NTBeVLxleXZicQ4A9gXoVzPVgysD1iaYLuLn81g1eusjS63w/640?wx_fmt=png&from=appmsg&watermark=1)

**零成本、零****风险****、立竿见影。**

**Step 1：测试****AI****的极限**

找一个你过去花了一周时间完成的任务，尝试用AI在几分钟内完成。不要期待完美，重点是建立对AI能力边界的直觉。

Theo的建议很直接：**如果你没有感到哪怕一点点不适，说明你还不够努力。**

**Step 2：学会阅读****AI****的思考过程**

使用Plan Mode观察AI如何分析代码库、制定计划、拆解任务。这就像看棋手复盘，你不仅要知道结果，还要理解每一步的考量。

**Step 3：建立agent.md体系**

这是最关键的一步。在你的代码库中创建并维护一个agent.md文件，每当你手动修改AI代码时，就往这个文件里加一条规则。

效果是指数级的：

第一周：AI准确率从60%提升到75%

第一个月：AI准确率提升到85%

三个月后：AI准确率接近95%

**你的工作从写代码逐渐变成了提需求。**

**Step 4：学会编排多个Agent**

最后一步是终极目标：让多个AI Agent协同工作，像交响乐团一样。

这是一个全新的技能树，而且这个技能树还在快速生长。



## MCP

MCP：ai的外部触手，让ai能够动起来，获取外部资源；  

需要调用外部、第三方软件、系统数据，获取动态实时的数据用mcp

## Agent

Agent：ai工作流程，用工作流、prompt的形式，让ai按照要求完成一件事；

## Skill

Skills：ai的能力库，可复用，给ai配置上，能让ai更好的完成工作；

固定的一套技能，数据都是静态的，让ai 跑脚本

## Prompt

prompt：人与llm沟通的根本。agent、skills内部都有prompt工程。


