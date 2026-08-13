# [Design and Architecture](https://tvm.apache.org/docs/arch/index.html)

This guide provides a few complementary views of the architecture. First, we review a single **end-to-end compilation flow** and discuss the key data structures and the transformations. This runtime-based view focuses on the interactions of each components when running the compiler. Then we will review the logical modules of the codebase and their relationship. This part provides a static overarching view of the design.

翻译: 本指南从多个互补视角介绍整体架构。首先，我们梳理一条完整的端到端编译流程，讲解核心数据结构与程序变换操作。该基于运行时的视角，着重剖析编译器运行时各个组件之间的交互关系。随后介绍代码库内的各个逻辑模块以及模块间的关联；此板块提供一份静态、宏观的架构设计概述。

## Overall Flow


