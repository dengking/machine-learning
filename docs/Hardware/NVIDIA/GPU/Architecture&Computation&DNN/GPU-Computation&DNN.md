# GPU Computation & Neural Networks（已合并）

> 本篇原有内容已与本目录下的 [`index.md`](./index.md) 融合，统一整理为一篇按
> **GPU 架构 → 计算特性 → 优势 → 深度学习为何离不开 GPU** 四部分组织的完整文章。
>
> 请阅读：[GPU Architecture & Computation & DNN](./index.md)

## 内容索引（对应新文档章节）

- **第一部分：GPU 架构** —— GPU vs CPU 设计哲学、Grid/SM/Warp/Thread 执行层次、SM 内部结构、存储层次与近几代 NVIDIA GPU 参数
- **第二部分：GPU 的计算特性** —— SIMT 执行模型、Warp Divergence / Memory Coalescing / Occupancy、GPU 擅长与不擅长的计算、GEMM / Element-wise / Reduction / Convolution
- **第三部分：GPU 的优势** —— 相对 CPU 的量化加速直觉与异构计算分工
- **第四部分：为什么深度学习离不开 GPU** —— 前向 / 反向传播的矩阵运算本质、Tensor Core、CNN 与 Transformer 为何适配 GPU / Tensor Core 流水线，以及 Capsule 动态路由为何不适配
