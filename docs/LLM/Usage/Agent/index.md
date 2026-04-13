# Agent

**大模型负责理解与决策，Agent 负责执行与调度** 

## [cloud.google-What is an AI agent?](https://cloud.google.com/discover/what-are-ai-agents)

**AI agents** are software systems that use AI to pursue goals and complete tasks on behalf of users. They show reasoning, planning, and memory and have a level of autonomy(自主性) to make decisions, learn, and adapt.

Their capabilities are made possible in large part by the multimodal capacity of generative AI and AI foundation models. AI agents can process multimodal information like text, voice, video, audio, code, and more simultaneously; can converse, reason, learn, and make decisions. They can learn over time and facilitate transactions and business processes. Agents can work with other agents to coordinate and perform more complex workflows.

翻译: 人工智能智能体是一类依托人工智能技术、**代表用户追求目标并完成任务**的软件系统。它们具备推理、规划与记忆能力，并拥有一定程度的自主性，可自主决策、学习与适配环境。

这类系统的能力在很大程度上得益于生成式人工智能与人工智能基础大模型的多模态特性。人工智能智能体能够同时处理文本、语音、视频、音频、代码等多模态信息，可开展对话、推理、学习与决策；能够持续迭代学习，并助力各类交易与业务流程的执行。智能体之间还可相互协作，协同完成更为复杂的工作流程。

### Key features of an AI agent

As explained above, while the key features of an AI agent are reasoning and acting (as described in [ReAct Framework](https://arxiv.org/pdf/2210.03629)) more features have evolved over time.

- **Reasoning:** This core cognitive process involves using logic and available information to draw conclusions, make inferences, and solve problems. AI agents with strong reasoning capabilities can analyze data, identify patterns, and make informed decisions based on evidence and context.
- **Acting**: The ability to take action or perform tasks based on decisions, plans, or external input is crucial for AI agents to interact with their environment and achieve goals. This can include physical actions in the case of **embodied AI(具身智能)**, or digital actions like sending messages, updating data, or triggering other processes.
- **Observing**: Gathering information about the environment or situation through perception or sensing is essential for AI agents to understand their context and make informed decisions. This can involve various forms of perception, such as computer vision, natural language processing, or sensor data analysis.
- **Planning**: Developing a strategic plan to achieve goals is a key aspect of intelligent behavior. AI agents with planning capabilities can identify the necessary steps, evaluate potential actions, and choose the best course of action based on available information and desired outcomes. This often involves anticipating future states and considering potential obstacles.
- **Collaborating**: Working effectively with others, whether humans or other AI agents, to achieve a common goal is increasingly important in complex and dynamic environments. Collaboration requires communication, coordination, and the ability to understand and respect the perspectives of others.
- **Self-refining**: The capacity for self-improvement and adaptation is a hallmark of advanced AI systems. AI agents with self-refining capabilities can learn from experience, adjust their behavior based on feedback, and continuously enhance their performance and capabilities over time. This can involve machine learning techniques, optimization algorithms, or other forms of self-modification.

翻译: 如上所述，尽管人工智能智能体的核心特征是**推理**与**执行**（如 ReAct 框架所述），但随着发展，其功能特征已不断丰富完善。

**推理**：这一核心认知过程，是指运用逻辑与现有信息得出结论、做出推断并解决问题。具备强大推理能力的 AI 智能体，能够分析数据、识别规律，并依据证据与上下文做出明智决策。

**执行**：基于决策、规划或外部输入采取行动、完成任务的能力，对 AI 智能体与环境交互并实现目标至关重要。这既包括具身智能的物理动作，也涵盖发送消息、更新数据、触发其他流程等数字操作。

**感知**：通过感知手段获取环境或场景信息，是 AI 智能体理解上下文、做出合理决策的基础。感知形式多样，包括计算机视觉、自然语言处理、传感器数据分析等。

**规划**：制定实现目标的策略方案是智能行为的关键环节。具备规划能力的 AI 智能体能够明确必要步骤、评估可行行动，并根据现有信息与预期目标选择最优方案，通常还会预判未来状态并考虑潜在障碍。

**协作**：在复杂多变的环境中，与人类或其他 AI 智能体高效配合以达成共同目标，正变得愈发重要。协作需要沟通、协调，以及理解并尊重他人视角的能力。

**自我优化**：自我完善与自适应能力是高级人工智能系统的典型标志。具备自我优化能力的 AI 智能体能够从经验中学习，根据反馈调整行为，并随时间持续提升性能与能力，这一过程可借助机器学习技术、优化算法或其他自我修正方式实现。

### What is the difference between AI agents, AI assistants, and bots?


