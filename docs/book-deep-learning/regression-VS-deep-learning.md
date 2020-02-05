# regression VS deep learning
本书Chapter 5 Machine Learning Basics中用regression导引介绍了machine learning的基础知识，在后面学习完了deep learning的理论后，对比来看Chapter 5 Machine Learning Basics中关于regression的介绍，有如下发现：

deep learning和regression之间有着本质的差别：

regression正如chapter 5.2 Capacity, Overfitting and Underfitting中所描述的：它会限制model的**hypothesis space**

> One way to control the capacity of a learning algorithm is by choosing its hypothesis space, the set of functions that the learning algorithm is allowed to select as being the solution. For example, the linear regression algorithm has the
set of all linear functions of its input as its hypothesis space. We can generalize linear regression to include polynomials, rather than just linear functions, in its hypothesis space. Doing so increases the model’s capacity.

> A polynomial of degree one gives us the linear regression model with which we
are already familiar, with prediction
$ y=b+wx.$

显然在linear regression中，需要学习的参数是`b`和`w`

但是deep learning则不同，它并不会限制model的**hypothesis space**，而是去approximate，也就是**universal approximation theorem**中所描述的那样；

deep learning和regression之间的共同点：
本质上来说deep learning和regression需要学习的都是parameter；不管是多么负责的model（MLP，CNN，RNN，LSTM等等），它最终都是由一系列的parameter来决定；


