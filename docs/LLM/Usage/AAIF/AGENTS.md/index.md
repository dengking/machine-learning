# [AGENTS.md](https://agents.md/)





Think of AGENTS.md as a **README for agents**: a dedicated, predictable place to provide the context(上下文) and instructions to help AI coding agents work on your project.

翻译: 你可以将 `AGENTS.md` 理解为**面向 AI 智能体的 README**：一个固定、可预期的专属文件，用于提供上下文与操作指引，帮助 AI 编码智能体在你的项目中高效开展工作。

## Why AGENTS.md?

README.md files are for humans: quick starts, project descriptions, and contribution guidelines.

AGENTS.md complements this by containing the extra, sometimes detailed context coding agents need: build steps, tests, and conventions that might clutter a README or aren’t relevant to human contributors.

We intentionally kept it separate to:

Give agents a clear, predictable place for instructions.

Keep READMEs concise and focused on human contributors.

Provide precise, agent-focused guidance that complements existing README and docs.

Rather than introducing another proprietary file, we chose a name and format that could work for anyone. If you’re building or using coding agents and find this helpful, feel free to adopt it.

翻译: `README.md` 文件面向人类用户：用于快速入门、项目说明以及贡献指南。

`AGENTS.md` 则与之互补，它包含了 AI 编码智能体所需的额外、有时甚至是详尽的上下文信息：比如构建步骤、测试流程以及编码规范 —— 这些内容若放入 README 会显得杂乱，或与人类贡献者无关。

我们刻意将二者分离，目的是：

- 为智能体提供一个清晰、可预期的专属指令空间。
- 保持 README 的简洁，使其专注于服务人类贡献者。
- 提供精准、以智能体为核心的指引，作为现有 README 和文档的有益补充。

我们没有引入新的专属文件格式，而是选择了一个适用于所有人的名称与规范。如果你正在开发或使用编码智能体，并认为这一模式有所助益，欢迎采纳。

## One AGENTS.md works across many agents

Your agent definitions are compatible with a growing ecosystem of AI coding agents and tools:

## How to use AGENTS.md?

### 1. Add AGENTS.md

Create an AGENTS.md file at the root of the repository. Most coding agents can even scaffold(脚手架) one for you if you ask nicely.

### 2. Cover what matters

Add sections that help an agent work effectively with your project. Popular choices:

- Project overview
- Build and test commands
- Code style guidelines
- Testing instructions
- Security considerations

### 3. Add extra instructions

Commit messages or pull request guidelines, security gotchas, large datasets, deployment steps: anything you’d tell a new teammate belongs here too.

翻译: 提交信息规范、拉取请求指南、安全注意事项、大型数据集说明、部署步骤 —— 任何你会告知新入职同事的内容，也都适合放在这里。

### 4. Large monorepo? Use nested AGENTS.md files for subprojects

Place another AGENTS.md inside each package. Agents automatically read the nearest file in the directory tree, so the closest one takes precedence and every subproject can ship tailored instructions. For example, at time of writing the main OpenAI repo has 88 AGENTS.md files.

翻译: **大型单体仓库？为子项目使用嵌套式 AGENTS.md 文件**

在每个包目录下再放置一份 `AGENTS.md`。AI 智能体会自动读取目录树中**距离最近的文件**，因此最贴近当前目录的文件优先级最高，每个子项目都可以提供量身定制的专属指引。例如，截至本文撰写时，OpenAI 主仓库中共有 88 个 `AGENTS.md` 文件。

## AGENTS.md是谁提出的

AGENTS.md 由 OpenAI 主导提出，并联合 Google、Cursor、Factory 等多家公司与社区共同协作制定。

### 关键时间线与背景

- **2025 年 5 月**：OpenAI 在推出 **Codex CLI** 时首次正式引入 AGENTS.md 格式，作为 AI 编码智能体的标准化指令文件。
- **2025 年**：由 OpenAI、Google、Cursor、Factory、Sourcegraph 等组成的行业工作组，共同完善并推动其成为跨工具的通用标准。
- **2025 年 12 月 9 日**：OpenAI 与 Anthropic 将 AGENTS.md 捐赠给 **Linux Foundation 旗下的 Agentic AI Foundation (AAIF)**，转为社区中立治理。
- 截至 2025 年底，已有 **60,000+ 开源项目**采用，被 Cursor、GitHub Copilot、Gemini CLI 等主流 AI 编码工具支持。

### 核心定位

它是**面向 AI 编码智能体的标准化 Markdown 配置文件**，用于提供项目专属的构建、测试、规范等上下文，解决 AI 工具配置碎片化问题。

需要我帮你整理一份 AGENTS.md 的**官方规范与常用字段清单**，方便你直接在项目中使用吗？


