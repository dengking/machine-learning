# Back Propagation(反向传播)

The overall network is a combination of [function composition](https://en.wikipedia.org/wiki/Function_composition "Function composition") and [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication "Matrix multiplication"):

$$
g(x) := f^L\Big(W^L f^{L-1}\Big(W^{L-1}\Big(\cdots f^1\big(W^1 x + b^1\big)\cdots\Big)+b^{L-1}\Big)+b^L\Big)

$$



“back propagation”即“方向传播”是machine learning的主要技术，本文对它进行说明。网络上对讲述它的文章非常多，下面是我参考的文章：

简单易懂的入门读物：zhihu [如何直观地解释 backpropagation 算法？](https://www.zhihu.com/question/27239198?rf=24827633)

深入浅出的文章：维基百科[Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)


