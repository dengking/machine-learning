# [Design and Architecture](https://tvm.apache.org/docs/arch/index.html)

> NOTE: 这篇文字的问题:
> 
> 1. **tirx::PrimFunc**、TensorIR、TIR 混用

This guide provides a few complementary views of the architecture. First, we review a single **end-to-end compilation flow** and discuss the key data structures and the transformations. This runtime-based view focuses on the interactions of each components when running the compiler. Then we will review the logical modules of the codebase and their relationship. This part provides a static overarching view of the design.

翻译: 本指南从多个互补视角介绍整体架构。首先，我们梳理一条完整的端到端编译流程，讲解核心数据结构与程序变换操作。该基于运行时的视角，着重剖析编译器运行时各个组件之间的交互关系。随后介绍代码库内的各个逻辑模块以及模块间的关联；此板块提供一份静态、宏观的架构设计概述。

## Overall Flow

In this guide, we will study an example **compilation flow** in the compiler. The figure below shows the flow. At a high-level, it contains several steps:

- **Model Creation**: Create the IRModule to be optimized and compiled, which contains a collection of functions that internally represent the model. Users can manually construct IRModule via `NNModule`, **TVMScript**, or import a **pre-trained model** from **Relax frontend**.

- **Transformation**: The compiler transforms an IRModule to another functionally equivalent or approximately equivalent(e.g. in the case of **quantization(量化)**) IRModule. Many of the transformations are **target (backend) independent**. We also allow target to affect the configuration of the transformation pipeline.

- **Target Translation**: The compiler translates(**codegen**) the IRModule to an executable format specified by the target. The target translation result is encapsulated as a **runtime.Module** that can be exported, loaded, and executed on the **target runtime environment**.

- **Runtime Execution**: the user loads back a **runtime.Module** and runs the compiled functions in the supported runtime environment.
  
  - 翻译: 用户加载回运行时模块，并在兼容的运行环境当中执行已经编译完成的函数

### Key data structures

One of the best ways to design and understand a complex system is to identify the **key data structures** and APIs that manipulate (transform) these **data structures**. Once we identified the **key data structures**, we can then breakdown a system into logical components that either define a collection of key data structures or transformations among the data structures.

**IRModule** is the primary data structure used across the entire stack. An IRModule (intermediate representation module) contains a collection of **functions**. Currently, we support two primary variants of functions.

- **relax::Function** is a high-level functional program representation. A **relax.Function** represents high-level graph structure, usually corresponds to an end-to-end model or a sub-graph of the overall model. You can view a **relax.Function** as a computational graph with additional support for control-flow, and complex data structures.

- **tirx::PrimFunc** is a low-level program representation that contains elements including loop-nest choices, multi-dimensional load/store, threading, and vector/tensor instructions. It is usually used to represent an **operator program** that executes a (possibly-fused(融合) layer in a model.



During the **compilation** and **transformation**, all **relax operators** are lowered to `tirx::PrimFunc` or `TVM PackedFunc`, which can be executed directly on the target device, while the calls to **relax operators** are lowered to calls to low-level functions (e.g. `R.call_tir` or `R.call_dps_packed`).



### Transformations

Now that we have covered the **key data structures**, let us talk about the transformations. Each transformation could serve one of the following purposes:

- optimization: transform a program to an equivalent, possibly more optimized version.

- lowering: transform a program to a lower-level representation that is closer to the target.



#### relax transformations

relax transformations contain a collection of passes that apply to relax functions. The optimizations include common graph-level optimizations such as constant folding and dead-code elimination for operators, and backend-specific optimizations such as library dispatch.

#### TensorIR transformations

- **TensorIR schedule**: **TensorIR schedules** are designed to optimize the **TensorIR functions** for a specific target, with user-guided instructions and control how the target code is generated. For CPU targets, a TensorIR PrimFunc can generate valid code and execute on the target device without schedule but with very-low performance. However, for GPU targets, the schedule is essential for generating valid code with thread bindings. For more details, please refer to the [TensorIR Transformation](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_transformation.html#tirx-transform) section. Additionally, we provides `MetaSchedule` to automate the search of TensorIR schedule.

- **Lowering Passes**: These passes usually perform after the schedule is applied, transforming a TensorIR PrimFunc into another functionally equivalent PrimFunc, but closer to the target-specific representation. For example, there are passes to flatten multi-dimensional access to one-dimensional pointer access, to expand the intrinsics into target-specific ones, and to decorate the function entry to meet the runtime calling convention.

Many **low-level optimizations** can be handled in the target phase by the LLVM, CUDA C, and other target compilers. As a result, we leave **low-level optimizations** such as **register allocation** to the downstream compilers and only focus on optimizations that are not covered by them.

#### cross-level transformations

Apache TVM enables cross-level optimization of end-to-end models. As the IRModule includes both Relax and TensorIR functions, the cross-level transformations are designed to mutate the IRModule by applying different transformations to these two types of functions.

For example, `relax.LegalizeOps` pass mutates the IRModule by lowering relax operators, adding corresponding TensorIR PrimFunc into the IRModule, and replacing the relax operators with calls to the lowered TensorIR PrimFunc. Another example is the operator fusion pipeline (`relax.FuseOps` + `relax.FuseTIR`), which fuses multiple consecutive tensor operations into a single kernel. See [Operator Fusion](https://tvm.apache.org/docs/arch/fusion.html#fusion-arch) for a detailed explanation of the fusion algorithm, operator pattern classification, and pattern-based fusion for external backends.
