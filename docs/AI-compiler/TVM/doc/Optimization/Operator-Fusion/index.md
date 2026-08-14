# [Operator Fusion](https://tvm.apache.org/docs/arch/fusion.html)

**Operator fusion** is one of the most impactful optimizations in TVM. Instead of launching one **kernel** per **operator** (e.g., conv2d, bias_add, relu), fusion merges multiple operators into a single kernel, eliminating intermediate memory allocations and kernel launch overhead.

TVM provides two complementary fusion mechanisms:

- **Automatic fusion** (`FuseOps` + `FuseTIR`): groups operators based on their computational patterns using a **post-dominator analysis algorithm**.

- **Pattern-based fusion** (`FuseOpsByPattern`): groups operators that match user-defined dataflow patterns, typically for offloading to external backends (cuBLAS, CUTLASS, DNNL, etc.).
  
  - 翻译: 将匹配用户自定义数据流模式的算子进行归组，通常用于将对应计算任务卸载到外部后端（如 cuBLAS、CUTLASS、DNNL 等）执行。

Both produce the same output: Relax functions marked with `Primitive=True` that are later lowered to fused **TIR kernels** or dispatched to external libraries.

翻译: 两种方式的最终产出一致：都会生成标记了 `Primitive=True` 的 Relax 函数，这类函数后续会被降级为融合 TIR 算子内核，或调度至外部加速库执行。

## Overview

Fusion involves three passes:

```python
IRModule (after LegalizeOps)
     │
     ▼  AnnotateTIROpPattern        ← label each op (elementwise, reduce, etc.)
IRModule (annotated)
     │
     ▼  FuseOps                     ← group ops into fused Relax functions
IRModule (with fused functions marked Primitive=True)
     │
     ▼  FuseTIR                     ← merge TIR PrimFuncs inside each group
IRModule (fused TIR kernels)
```

In the compilation pipeline, these passes appear in the backend-specific `legalize_passes` phase. For example, the **CUDA pipeline** (`python/tvm/relax/backend/cuda/pipeline.py`) runs:

```python
LegalizeOps()          # lower Relax ops to call_tir
AnnotateTIROpPattern() # annotate pattern kinds
FoldConstant()
FuseOps()              # group ops
FuseTIR()              # merge TIR functions
```

## Operator Pattern Classification

Before fusion, `AnnotateTIROpPattern` analyzes each **TIR function** in the module and assigns an `OpPatternKind`. The fusion algorithm uses these pattern kinds to decide which operators can be fused together.


