# Computational Graph Simplification(计算图化简)

图化简技术可分为五大类：

```
┌────────────────────────────────────────────┐
│ 1. 算子融合 (Operator Fusion)     ★最重要     │
│    · 垂直融合、水平融合                       │
├────────────────────────────────────────────┤
│ 2. 代数化简 (Algebraic Simplification)       │
│    · 常量折叠、代数恒等式、强度削减           │
├────────────────────────────────────────────┤
│ 3. 冗余消除 (Redundancy Elimination)         │
│    · CSE、死代码消除、恒等算子消除           │
├────────────────────────────────────────────┤
│ 4. 布局优化 (Layout Optimization)            │
│    · NCHW↔NHWC、消除冗余 transpose/reshape   │
├────────────────────────────────────────────┤
│ 5. 算子替换/规范化 (Canonicalization)        │
│    · 用高效算子替换、算子分解与合并           │
└────────────────────────────────────────────┘
```

---

## 四、算子融合：最重要的化简

### 4.1 垂直融合（Vertical / Producer-Consumer Fusion）

沿**数据流方向**，把"生产者-消费者"链上的算子合并成一个 kernel。

```
融合前（3 个 kernel，2 次中间结果往返内存）：
    Conv ──▶ [写内存] ──▶ BatchNorm ──▶ [写内存] ──▶ ReLU

融合后（1 个 kernel，中间结果留在寄存器/片上）：
    ┌──────────────────────────┐
    │  Conv + BN + ReLU (fused) │
    └──────────────────────────┘
```

**最经典的融合模式**：

```
Conv + BatchNorm + ReLU        → CBR 融合（CV 领域标配）
MatMul + BiasAdd + Activation  → 全连接层融合
Elementwise 链 (Add→Mul→ReLU)  → 逐元素算子链融合
```

> **特例：Conv + BN 折叠（BN Folding）**
> 推理时 BN 的参数是常量，可直接**数学上合并进 Conv 的权重和偏置**，让 BN 彻底消失——这是常量折叠 + 融合的结合。

### 4.2 水平融合（Horizontal Fusion）

把**共享同一输入**、彼此无依赖的**并行算子**合并。

```
融合前：           融合后：
    ┌─▶ Conv1        ┌─▶ [Conv1|Conv2|Conv3
x ──┼─▶ Conv2   →  x ─┤   一次性并行计算]
    └─▶ Conv3        └─▶ 提高并行度、减少启动
```

**典型场景**：Multi-Head Attention 的多个 Q/K/V 投影、Inception 模块的并行分支。

### 4.3 融合的约束条件

```
不是所有算子都能融合，需考虑：
  ✅ 数据依赖关系是否允许
  ✅ 硬件资源（寄存器/共享内存）是否够用
  ✅ 融合后是否反而降低并行度
  ✅ 算子类型是否兼容（如 reduce 类算子融合受限）
```

---

## 五、代数化简与常量折叠

### 5.1 常量折叠（Constant Folding）

编译期就把**只依赖常量**的计算算出来。

```
优化前：                    优化后：
    a = 2 * 3               a = 6         （编译期算好）
    y = x + a               y = x + 6
```

在深度学习中，**权重、BN 参数、shape 计算**等常量子图都可提前折叠。

### 5.2 代数恒等式化简（Algebraic Simplification）

利用数学恒等式删除无意义运算：

```
x + 0        → x
x * 1        → x
x * 0        → 0
x - x        → 0
transpose(transpose(x)) → x
reshape(reshape(x, s1), s2) → reshape(x, s2)
concat(split(x)) → x         （逆操作抵消）
```

### 5.3 强度削减（Strength Reduction）

用**低成本运算**替换高成本运算：

```
x / 常量  → x * (1/常量)      （除法 → 乘法）
x ^ 2     → x * x             （幂 → 乘法）
x * 2     → x << 1            （乘 → 移位，整数场景）
```

---

## 六、冗余消除类优化

### 6.1 公共子表达式消除（CSE）

复用重复计算的子图（详见"可用表达式分析"）：

```
优化前：                     优化后：
    t1 = a + b                t1 = a + b
    t2 = a + b        →       t2 = t1     （复用）
```

在计算图中，若两个节点**算子相同 + 输入相同**，则可合并为一个。

### 6.2 死代码消除（Dead Code Elimination, DCE）

删除**结果未被使用**的节点：

```
    x ──▶ [OpA] ──▶ y  （y 被输出使用，保留）
    x ──▶ [OpB] ──▶ z  （z 无人使用 → 删除整个 OpB）
```

### 6.3 恒等算子消除（Identity Elimination）

删除**不改变数据**的算子：

```
- Identity / Dropout(推理时) / 无效的 Cast
- reshape 到相同 shape
- scale=1 的缩放
```

---

## 七、布局与数据流优化

### 7.1 布局转换与传播（Layout Transformation）

不同硬件偏好不同数据布局：

```
CPU 常偏好 NCHW，GPU/TensorCore 常偏好 NHWC
    → 编译器插入 transpose 适配
    → 但要通过"布局传播"消除连续冗余的 transpose
```

### 7.2 冗余 Transpose / Reshape 消除

```
优化前：                              优化后：
    x → transpose → transpose → y  →   x → y
    （两次转置互相抵消）
```

这是深度学习图化简中**极其常见**的优化——框架导出的模型往往含大量冗余的 reshape/transpose。

---

## 八、实现机制：Pattern Rewrite

现代 AI 编译器（尤其是 MLIR）用**模式重写（Pattern Rewrite）**统一实现这些化简。

### 核心思想

```
定义一系列 "匹配模式 → 替换模式" 的规则：

    match:    A → B → C  (某个子图模式)
    rewrite:  fused_ABC   (替换为化简后的形式)

编译器反复扫描图，匹配到就替换，直到无法再化简（收敛）
```

### MLIR 中的实现

```
- Canonicalization Pass：规范化 + 基础代数化简
- DRR (Declarative Rewrite Rules)：声明式重写规则
- PDL (Pattern Descriptor Language)：模式描述语言
- 各 Dialect 自带 canonicalize / fold 方法
```

**示例（概念性）**：

```cpp
// 匹配 add(x, 0) 并化简为 x
struct SimplifyAddZero : public OpRewritePattern<AddOp> {
  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const {
    if (isZero(op.getRhs())) {
      rewriter.replaceOp(op, op.getLhs());  // add(x,0) → x
      return success();
    }
    return failure();
  }
};
```

### 关键：迭代直到不动点

```
化简规则之间会互相触发：
  常量折叠 → 产生新常量 → 又触发新的代数化简 → …

因此需要迭代应用，直到图不再变化（Fixed-Point）
```

---

## 九、工程实践与面试要点

### 各 AI 编译器中的对应实现

| 编译器          | 图化简实现                                                            |
| ------------ | ---------------------------------------------------------------- |
| **TVM**      | Relay/Relax 的 Pass（FuseOps、FoldConstant、EliminateCommonSubexpr…） |
| **MLIR**     | Canonicalization、CSE Pass、各 Dialect 的 fold                       |
| **XLA**      | HLO 层的 Algebraic Simplifier、Fusion                               |
| **TensorRT** | Layer Fusion（垂直/水平融合，闭源但文档丰富）                                    |
| **PyTorch**  | torch.fx 图变换、Inductor 的融合                                        |

### 面试高频问题

| 问题                | 要点                                                 |
| ----------------- | -------------------------------------------------- |
| **算子融合为什么能提速？**   | 减少中间结果的内存往返（访存瓶颈）+ 减少 kernel 启动开销                  |
| **垂直融合 vs 水平融合？** | 垂直=生产者消费者链；水平=共享输入的并行算子                            |
| **Conv+BN 折叠原理？** | BN 推理时参数为常量，可数学合并进 Conv 权重/偏置                      |
| **什么算子难融合？**      | reduce 类、改变数据依赖的算子、资源超限时                           |
| **图化简如何保证正确性？**   | 每条重写规则必须是"语义等价"变换                                  |
| **为什么要迭代到不动点？**   | 化简规则互相触发，需反复应用直到收敛                                 |
| **在 MLIR 里怎么实现？** | Pattern Rewrite（DRR/PDL）、Canonicalization、CSE Pass |

### 与整体优化流程的关系

```
图化简（高层，硬件无关）
      ↓ Lowering
算子/循环优化（中层：tiling、向量化）  ← 图化简后的图更利于此层优化
      ↓ Lowering
后端代码生成（低层：寄存器分配、指令调度）
```

> **重要认知**：图化简是**"打地基"**的优化。图化简得越干净，后续中低层优化的效果越好——一个含大量冗余 transpose 的脏图，会严重拖累后续所有优化。

---

## 总结

```
计算图化简 = 在 DAG 层面做"语义等价"的结构精简

五大武器：
  ① 算子融合    ── 减少访存与启动开销（最重要）
  ② 代数化简    ── 常量折叠、恒等式、强度削减
  ③ 冗余消除    ── CSE、DCE、恒等算子消除
  ④ 布局优化    ── 消除冗余 transpose/reshape
  ⑤ 规范化      ── 算子替换与合并

实现机制：Pattern Rewrite，迭代至不动点

核心价值：深度学习瓶颈在访存，图化简（尤其融合）
         通过减少内存往返，直接命中性能要害
```

> 📌 **一句话总结**：计算图化简是 AI 编译器高层优化的**主战场**，其中**算子融合**是皇冠上的明珠——它抓住了"深度学习是访存密集型"这个本质，通过减少中间结果的内存读写来大幅提速。理解它，是理解 AI 编译器"为什么快"的关键。

```

```
