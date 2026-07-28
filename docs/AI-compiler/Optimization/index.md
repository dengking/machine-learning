# Optimization in AI compiler

AI 编译器是连接上层深度学习框架与底层硬件（GPU / [ASIC](https://en.wikipedia.org/wiki/Application-specific_integrated_circuit)）的核心工具链，通过多级中间表示（IR）完成**计算图化简**、算子重构、内存优化、硬件指令映射，核心目标是**破解访存瓶颈、提升算力利用率**，尤其适配 Transformer、MoE 等大模型场景的算力需求。

## 全景分类

AI 编译器优化按编译器标准层级可划分为**四大通用方向 + 一个大模型专项方向**：

| 分类          | 层级          | 优化对象                     | 核心目标               | 详见                                                                  |
| ----------- | ----------- | ------------------------ | ------------------ | ------------------------------------------------------------------- |
| **图级优化**    | 中端高层（IR 层面） | 完整计算图                    | 减少访存 & kernel 启动开销 | [Graph-Level-Optimization](Graph-Level-Optimization/index.md)       |
| **算子级优化**   | 中端底层（循环层面）  | 单个算子内核                   | 最大化单 SM 利用率        | [Operator-Level-Optimization](Operator-Level-Optimization/index.md) |
| **内存优化**    | 贯穿全编译栈      | 张量生命周期与显存                | 降低峰值内存、提升带宽利用率     | [Memory-Optimization](Memory-Optimization/index.md)                 |
| **指令优化**    | 后端代码生成      | 硬件指令 & 流水线               | 隐藏延迟、打满硬件流水线       | [Instruction-Optimization](Instruction-Optimization/index.md)       |
| **大模型专项优化** | 跨层级组合       | Transformer / MoE / 长上下文 | 针对大模型场景的定制化编译      | [LLM-Specific-Optimization](LLM-Specific-Optimization/index.md)     |

> **核心动机**：现代硬件算力增长 >> 内存带宽增长，很多算子是"访存密集型（Memory-Bound）"，瓶颈在于反复读写中间结果而非计算本身。因此上表中的绝大多数优化，本质都是围绕**"减少访存 / 隐藏延迟 / 提升硬件利用率"**这三条主线展开。

## 各类别一图速览

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Compiler Optimization                  │
├─────────────────────────────────────────────────────────────┤
│ ① 图级优化（Graph-Level）                                    │
│    算子融合 · 代数化简 · 冗余消除 · 布局变换 · 算子等价替换   │
├─────────────────────────────────────────────────────────────┤
│ ② 算子级优化（Operator-Level）                               │
│    循环分块 · 循环重排 · 向量化 · 并行化 · 自动调优           │
├─────────────────────────────────────────────────────────────┤
│ ③ 内存优化（Memory）                                         │
│    内存复用与池化 · 静态内存规划 · 访存对齐与合并             │
├─────────────────────────────────────────────────────────────┤
│ ④ 指令优化（Instruction / Code-Gen）                        │
│    硬件原语映射 · 指令调度与流水线 · 多 SM 负载均衡           │
├─────────────────────────────────────────────────────────────┤
│ ⑤ 大模型专项（LLM-Specific）                                 │
│    FlashAttention · PagedAttention · Prefix Cache            │
│    MoE 路由通信流水化 · 稀疏计算 · KV 缓存量化               │
│    分块流水线并行 · 稀疏注意力 · 混合精度 & 量化融合          │
└─────────────────────────────────────────────────────────────┘
```

## 主流技术栈对应

- **通用全栈编译器**：Apache TVM、MLIR（LLVM 生态）、XLA（Google）
- **NVIDIA 生态专属**：TensorRT、TensorRT-LLM、Triton DSL
- **大模型推理编译框架**：vLLM、TensorRT-LLM、SGLang

## 阅读建议

- 想快速建立系统性认知 → 先看四大通用方向的 `index.md`
- 关注大模型场景（面试/工程实践的热点） → 直接进入 [LLM-Specific-Optimization](LLM-Specific-Optimization/index.md)
- 关注传统"打地基"式的优化 → 从 [Graph-Level-Optimization](Graph-Level-Optimization/index.md) 开始
