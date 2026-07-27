# Computational-graph&Back-Propagation(反向传播)

The overall network is a combination of [function composition](https://en.wikipedia.org/wiki/Function_composition "Function composition") and [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication "Matrix multiplication"):

$$
g(x) := f^L\Big(W^L f^{L-1}\Big(W^{L-1}\Big(\cdots f^1\big(W^1 x + b^1\big)\cdots\Big)+b^{L-1}\Big)+b^L\Big)

$$

深度学习模型在编译器中被表示为**有向无环图（DAG）**：

```
- 节点（Node）  = 算子/操作（Conv、MatMul、Add、ReLU…）
- 边（Edge）    = 张量（Tensor），表示数据流动
```

**示例**：一个简单的 `y = ReLU(Conv(x, w) + b)`

```
    x ──┐
        ├──▶ [Conv] ──▶ [Add] ──▶ [ReLU] ──▶ y
    w ──┘            ▲
                 b ──┘
```





## 素材

- 简单易懂的入门读物：zhihu [如何直观地解释 backpropagation 算法？](https://www.zhihu.com/question/27239198?rf=24827633)

- wikipedia [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) 

- Jorge-Nocedal-Stephen-J-Wright-Numerical-Optimization # 8.2-AUTOMATIC-DIFFERENTIATION

## Computational graph&Syntax tree

在wikipedia [Computer algebra](https://en.wikipedia.org/wiki/Computer_algebra) “`Computer science aspects#Expressions`”段中，让我想起了在compiler principle中描述的对expression的表示：syntax tree、grammar tree，显然computational graph也是一种表达方式；显然在计算代数中，非常重要的一个课题就是如何来表示computation，显然computational graph是一种非常强大的工具；各种各样的问题，如果要使用computer来进行解决，那么一个非常重要的课题就是：如何来表示？显然这是各种data structure排上用场的时候了。

symbolic computation: computational graph是就是一种典型的symbolic computation，它在6.5.5 Symbol-to-Symbol Derivatives、6.5.4 Back-Propagation Computation in Fully-Connected MLP 中有描述
