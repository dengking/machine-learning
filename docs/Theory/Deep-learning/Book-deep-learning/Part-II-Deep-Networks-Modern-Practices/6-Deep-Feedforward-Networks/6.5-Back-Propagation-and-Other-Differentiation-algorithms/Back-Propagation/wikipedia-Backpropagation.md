#  维基百科[Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)

The backpropagation algorithm works by computing the gradient of the loss function with respect to each weight by the [chain rule](https://en.wikipedia.org/wiki/Chain_rule), computing the gradient one layer at a time, [iterating](https://en.wikipedia.org/wiki/Iteration) backward from the last layer to avoid redundant calculations of intermediate terms in the chain rule; this is an example of [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming).[[3\]](https://en.wikipedia.org/wiki/Backpropagation#cite_note-FOOTNOTEGoodfellowBengioCourville2016[httpswwwdeeplearningbookorgcontentsmlphtmlpf33_214]-3)

> NOTE: [chain rule](https://en.wikipedia.org/wiki/Chain_rule) and [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming)，与此类似的例子还有：[Markov chain](https://en.wikipedia.org/wiki/Markov_chain)

Backpropagation requires the derivatives of activation functions to be known at network design time. [Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) is a technique that can automatically and analytically provide the derivatives to the training algorithm. In the context of learning, backpropagation is commonly used by the [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) optimization algorithm to adjust the weight of neurons by calculating the [gradient](https://en.wikipedia.org/wiki/Gradient) of the [loss function](https://en.wikipedia.org/wiki/Loss_function); backpropagation computes the gradient(s), whereas (stochastic) gradient descent uses the gradients for training the model (via optimization).

> NOTE: backpropagation 和gradient descent之间的关系



## Overview

Backpropagation computes the [gradient](https://en.wikipedia.org/wiki/Gradient) in [weight space](https://en.wikipedia.org/wiki/Parameter_space) of a feedforward neural network, with respect to a [loss function](https://en.wikipedia.org/wiki/Loss_function). Denote:

$x$

input (vector of features)

$y$

target output

For classification, output will be a vector of class probabilities (e.g., $ (0.1,0.7,0.2)$ ), and target output is a specific class, encoded by the [one-hot](https://en.wikipedia.org/wiki/One-hot)/[dummy variable](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)) (e.g., $(0,1,0)$).

$C$

[loss function](https://en.wikipedia.org/wiki/Loss_function) or "cost function"

For classification, this is usually [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) (XC, [log loss](https://en.wikipedia.org/wiki/Log_loss)), while for regression it is usually [squared error loss](https://en.wikipedia.org/wiki/Squared_error_loss) (SEL).

$L$

the number of layers

$W^{l}=(w_{jk}^{l})$:

the weights between layer $l-1$and $l,$ where $w_{jk}^{l}$ is the weight between the $k$-th node in layer $l-1$ and the $j$-th node in layer $l$ [b\]](https://en.wikipedia.org/wiki/Backpropagation#cite_note-9)

$ f^{l}$

[activation functions](https://en.wikipedia.org/wiki/Activation_function) at layer $l$

For classification the last layer is usually the [logistic function](https://en.wikipedia.org/wiki/Logistic_function) for binary classification, and [softmax](https://en.wikipedia.org/wiki/Softmax_function) (softargmax) for multi-class classification, while for the hidden layers this was traditionally a [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) (logistic function or others) on each node (coordinate), but today is more varied, with [rectifier](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) ([ramp](https://en.wikipedia.org/wiki/Ramp_function), [ReLU](https://en.wikipedia.org/wiki/ReLU)) being common.



The overall network is a combination of [function composition](https://en.wikipedia.org/wiki/Function_composition) and [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication):
$$
g(x):=f^{L}(W^{L}f^{L-1}(W^{L-1}\cdots f^{1}(W^{1}x)\cdots ))
$$

> NOTE: 数学的简洁



For a training set there will be a set of input–output pairs, $\left\{(x_{i},y_{i})\right\}$. For each input–output pair $ (x_{i},y_{i})$ in the training set, the loss of the model on that pair is the cost of the difference between the predicted output $g(x_{i})$ and the target output $y_{i}$

$$
C(y_{i},g(x_{i}))
$$

Note the distinction: 

During **model evaluation**, the weights are fixed, while the inputs vary (and the target output may be unknown), and the network ends with the output layer (it does not include the loss function). 

During **model training**, the input–output pair is fixed, while the weights vary, and the network ends with the loss function.

> NOTE: evaluation阶段和training阶段的对比



Backpropagation computes the gradient for a fixed input–output pair $(x_{i},y_{i})$, where the weights $w_{jk}^{l}$ can vary. Each individual component of the gradient, $\partial C/\partial w_{jk}^{l}$  can be computed by the chain rule; however, doing this separately for each weight is inefficient. Backpropagation efficiently computes the gradient by avoiding duplicate calculations and not computing unnecessary intermediate values, by computing the gradient of each layer – specifically, the gradient of the weighted input of each layer, denoted by $\delta ^{l}$– from back to front.

> NOTE:  $\delta ^{l}$表示的是第$l$层的gradient 

> NOTE: 为什么是front back to front？下面这一段对此进行了直观的解释，如下是我结合整体的理解：
>
> 整个模型可以表示为：$g(x):=f^{L}(W^{L}f^{L-1}(W^{L-1}\cdots f^{1}(W^{1}x)\cdots ))$，显然逐层的，可以认为它是一个nesting结构，显然它是一个线性的结构，这样的结构就决定了：
>
> the only way a weight in $W^{l}$ affects the loss is through its effect on the next layer, and it does so linearly
>
> 这让我想起来：结构决定属性（参见`discrete-math\docs\Guide\Relation-structure-computation`）
>
> 显然，在最后一层，它的weight即$W^{l}$直接“affects the loss”，即它无需通过中间层而直接作用于loss，所以$\delta ^{l}$ 可以直接计算得到。在计算得到了$\delta ^{l}$后，就可以计算得到直接依赖于它的$\delta ^{l-1}$；依次可以递归进行直到第一层，从而可以计算得到所有的gradient；

Informally, the key point is that since the only way a weight in $W^{l}$ affects the loss is through its effect on the next layer, and it does so linearly, $\delta ^{l}$ are the only data you need to compute the gradients of the weights at layer $l$, and then you can compute the previous layer $\delta ^{l-1}$ and repeat recursively. 

This avoids inefficiency in two ways. Firstly, it avoids duplication because when computing the gradient at layer $l$, you do not need to recompute all the derivatives on later layers $l+1,l+2,\ldots $ each time. Secondly, it avoids unnecessary intermediate calculations because at each stage it directly computes the gradient of the weights with respect to the **ultimate output** (the loss), rather than unnecessarily computing the derivatives of the values of hidden layers with respect to changes in weights $\partial a_{j'}^{l'}/\partial w_{jk}^{l}.$

> NOTE: 上面介绍了backpropagation高效性的原因

Backpropagation can be expressed for simple feedforward networks in terms of [matrix multiplication](https://en.wikipedia.org/wiki/Backpropagation#Matrix_multiplication), or more generally in terms of the [adjoint graph](https://en.wikipedia.org/wiki/Backpropagation#Adjoint_graph).

## Matrix multiplication

Given an input–output pair $ (x,y) $, the loss is:
$$
C(y,f^{L}(W^{L}f^{L-1}(W^{L-1}\cdots f^{1}(W^{1}x)\cdots )))
$$
To compute this, one starts with the input $x$ and works forward; denote the weighted input of each layer as $z^{l}$ and the output of layer $l$ as the activation $a^{l}$. For backpropagation, the activation $a^{l}$ as well as the derivatives $(f^{l})'$ (evaluated at $z^{l}$) must be cached for use during the backwards pass.

> NOTE: 所谓$z^{l}$既weighted input，其实就是它的input dot product weight；所谓$a^{l}$，其实就是这一层的output。



The derivative of the loss in terms of the inputs is given by the chain rule; note that each term is a [total derivative](https://en.wikipedia.org/wiki/Total_derivative), evaluated at the value of the network (at each node) on the input $x$
$$
{\displaystyle {\frac {dC}{da^{L}}}\cdot {\frac {da^{L}}{dz^{L}}}\cdot {\frac {dz^{L}}{da^{L-1}}}\cdot {\frac {da^{L-1}}{dz^{L-1}}}\cdot {\frac {dz^{L-1}}{da^{L-2}}}\cdots {\frac {da^{1}}{dz^{1}}}\cdot {\frac {\partial z^{1}}{\partial x}}.}
$$
These terms are: the derivative of the loss function;[[d\]](https://en.wikipedia.org/wiki/Backpropagation#cite_note-11) the derivatives of the activation functions;[[e\]](https://en.wikipedia.org/wiki/Backpropagation#cite_note-12) and the matrices of weights:[[f\]](https://en.wikipedia.org/wiki/Backpropagation#cite_note-13)
$$
{\displaystyle {\frac {dC}{da^{L}}}\cdot (f^{L})'\cdot W^{L}\cdot (f^{L-1})'\cdot W^{L-1}\cdots (f^{1})'\cdot W^{1}.}
$$
The gradient ${\displaystyle \nabla }$ is the [transpose](https://en.wikipedia.org/wiki/Transpose)（转置） of the derivative of the output in terms of the input, so the matrices are transposed and the order of multiplication is reversed, but the entries are the same:
$$
{\displaystyle \nabla _{x}C=(W^{1})^{T}\cdot (f^{1})'\cdots \cdot (W^{L-1})^{T}\cdot (f^{L-1})'\cdot (W^{L})^{T}\cdot (f^{L})'\cdot \nabla _{a^{L}}C.}
$$
Backpropagation then consists essentially of evaluating this expression from right to left (equivalently, multiplying the previous expression for the derivative from left to right), computing the gradient at each layer on the way; there is an added step, because the gradient of the weights isn't just a subexpression: there's an extra multiplication.