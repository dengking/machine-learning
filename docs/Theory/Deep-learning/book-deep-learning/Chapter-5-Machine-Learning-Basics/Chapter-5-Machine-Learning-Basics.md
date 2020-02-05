# [Chapter 5 Machine Learning Basics](http://www.deeplearningbook.org/contents/ml.html)

每章的开头作者都会抛出一系列的问题，本章作者所抛出的问题可以说是machine learning中最最本质的问题：

- What a learning algorithm is
- How the challenge of fitting the training data differs from the challenge of finding patterns that generalize to new data
- How to set hyperparameters



Machine learning is essentially a form of applied statistics with increased emphasis on the use of
computers to statistically estimate complicated functions and a decreased emphasis on proving confidence intervals around these functions; we therefore present the two central approaches to statistics: frequentist estimators and Bayesian inference.

这段话如何理解？

本质上来说，“machine learning”属“applied statistics”。所以要理解上面这段话需要对[Statistics](https://en.wikipedia.org/wiki/Statistic)的研究分支有一些了解了。上面这段话中的frequentist estimators 和 Bayesian inference都属于[Statistical inference](https://en.wikipedia.org/wiki/Statistical_inference)范轴，按照[Statistical inference](https://en.wikipedia.org/wiki/Statistical_inference)的说法，“frequentist estimators ”对应的是[Frequentist inference](https://en.wikipedia.org/wiki/Frequentist_inference)，“Bayesian inference”对应的是[Bayesian inference](https://en.wikipedia.org/wiki/Bayesian_inference)，它们是[Frequentist inference](https://en.wikipedia.org/wiki/Frequentist_inference)的两个学派（或paradigm），关于[Frequentist inference](https://en.wikipedia.org/wiki/Frequentist_inference)的学派，参见[Paradigms for inference](https://en.wikipedia.org/wiki/Statistical_inference#Paradigms_for_inference)。需要注意的是：

> These schools—or "paradigms"—are not mutually exclusive, and methods that work well under one paradigm often have attractive interpretations under other paradigms.

“confidence intervals”的中文意思是：置信区间，参见[Confidence Interval](https://en.wikipedia.org/wiki/Confidence_interval)。

有这些认知对于理解后续章节的内容是比较重要的。



We describe how to combine various **algorithm components** such as an **optimization algorithm**, a **cost function**, a **model**, and a **dataset** to build a **machine learning algorithm**.

我觉得这段话所传达的思想是比较好的：如果将machine learning algorithm看做是一个机器的，那么它有如下零件（component）组成：

- **optimization algorithm**
- **model**
- **dataset**

那读者看到会提出这样的问题：

- 这些component分别表示的是什么？
- 它们之间是如何组装、协助来构成一个完整的machine learning algorithm？
- 每个component有哪些可供选择的option？

这些问题在本书的后续章节会专门进行介绍。



Finally, in section , we describe some of the 5.11 factors that have limited the **ability** of traditional machine learning to generalize. These challenges have motivated the development of deep learning algorithms that
overcome these obstacles.

上述ability的含义是什么？deep learning algorithm较traditional machine learning algorithm的优势何在？

## 5.1 Learning Algorithms

本段中关于“learning”的定义是引用的如下书籍：

[Machine Learning](http://www.cs.cmu.edu/~tom/mlbook.html), [Tom Mitchell](http://www.cs.cmu.edu/~tom), McGraw Hill, 1997.



### 5.1.1 The Task, T

本节作者所要表达的主要思想简单描述如下：

人类开发computer program来解决形形色色的problem，这些problem就是program所要执行的task，但是我们知道，program并非万能的，还是有非常非常多的problem是无法使用program来解决的。随着科技的发展，program能够解决的problem也越来越多了，也就是program的能力越来越强了。machine learning algorithm就是一种在一类task中

思考：machine learning algorithm VS 普通algorithm？

machine learning algorithm是一种全新的算法范式，它使program能够“learning”（“learning”在5.1 Learning Algorithms中给出定义）

Learning is our means of attaining the ability to perform the task.



