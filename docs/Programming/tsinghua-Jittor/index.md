# [计图（Jittor）](https://cg.cs.tsinghua.edu.cn/jittor/)

[Jittor](https://github.com/Jittor)/**[jittor](https://github.com/Jittor/jittor)**

https://mp.weixin.qq.com/mp/appmsgalbum?action=getalbum&__biz=MzA3OTE4OTkxMw==&scene=1&album_id=1363066409391276032&count=3&version=4.1.3.90694&platform=mac#wechat_redirect



## 基础定位

**Jittor（中文名：计图）** 是**清华大学计算机系图形学实验室（CG组，胡事民团队）** 2020年3月开源的**国产自主深度学习框架**，核心创新：**元算子 + JIT即时编译 + 统一计算图**，兼顾PyTorch动态图的灵活与TensorFlow静态图的高性能优化。

## 三大核心创新

### 1. 元算子（Meta-operator）

主流框架有上千种算子，维护、优化、移植成本极高；Jittor仅定义**20余种基础元算子**，所有卷积、池化、BN、激活等复杂算子都由元算子组合生成，且满足**反向传播闭包**（元算子求导结果仍是元算子）：

- 三类基础元算子：**元素级、广播/重索引、聚合**
- 自动算子融合：运行时合并细碎计算，大幅减少内存读写与GPU调度开销
- 自定义算子极简：支持Python内联C++/CUDA代码，几行实现高性能算子

### 2. 统一计算图（Unified Graph）

解决传统框架**动态图灵活但优化弱、静态图性能强但调试麻烦**的痛点：

- 前端完全动态编程（和PyTorch写法几乎一致，逐行执行、即时调试）
- 后端运行时自动拆分可优化子图、JIT编译为C++/CUDA机器码，无需手动转静态图
- 全程自动梯度、自动混合精度、自动并行优化

### 3. 全链路JIT即时编译

框架所有代码（算子、模型、梯度逻辑）均运行时编译，内置LLVM/CUDA编译器：

- 不用预编译算子库，新模型自动生成定制化优化代码
- 支持算子特化、常量折叠、内存复用、多流并行等深度优化
- 视觉任务训练速度普遍优于PyTorch、TensorFlow

## 核心优势

1. **国产自主，全栈自研**：从计算内核、编译器到上层模型库完整自研，适配国产硬件（华为昇腾910、海光DCU、ROCm），代码无需修改即可跨英伟达/国产芯片迁移
2. **上手成本极低**：API高度对标PyTorch，`jt.Tensor`、`jt.nn.Module`、自动梯度逻辑几乎一一对应，迁移代码工作量小
3. **速度更快**：元算子融合消除大量访存开销，图像分类、检测、生成、3D视觉任务训练/推理加速明显
4. **生态完整**：内置丰富模型库：CNN骨干、GAN、Transformer、分割、检测、3D几何、情感计算、遥感AI等
5. **跨平台全硬件支持**：Linux/macOS/Windows；CPU、NVIDIA GPU、华为昇腾、AMD ROCm、TPU、OpenCL

# 