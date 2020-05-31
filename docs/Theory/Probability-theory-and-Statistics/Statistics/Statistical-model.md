# Statistical model

本文是阅读维基百科[Statistical model](https://en.wikipedia.org/wiki/Statistical_model)的笔记，本文的内容涉及了：

- 维基百科[Statistical inference](https://en.wikipedia.org/wiki/Statistical_inference)
- Deep learning book的Chapter [5.2 Capacity, Overfitting and Underfitting](https://dengking.github.io/machine-learning/Theory/Deep-learning/Book-deep-learning/Chapter-5-Machine-Learning-Basics/5.2-Capacity-Overfitting-and-Underfitting/)

> A **statistical model** is a [mathematical model](https://en.wikipedia.org/wiki/Mathematical_model) that embodies a set of [statistical assumptions](https://en.wikipedia.org/wiki/Statistical_assumptions) concerning the generation of [sample data](https://en.wikipedia.org/wiki/Sample_(statistics)) (and similar data from a larger [population](https://en.wikipedia.org/wiki/Statistical_population)). A statistical model represents, often in considerably idealized form, the **data-generating process**.

上面这段话的中文意思：

“统计模型是一种数学模型，它包含了一组关于样本数据生成的统计假设。统计模型通常以相当理想化的形式表示数据生成过程。”

> A statistical model is usually specified as a mathematical relationship between one or more [random variables](https://en.wikipedia.org/wiki/Random_variables) and other non-random variables.

这是原文中，最清楚、简洁的解释，简单理解是：statistical model其实一个函数，这个函数是关于random variable的。Statistical model的最最典型的例子就是 [linear regression](https://en.wikipedia.org/wiki/Linear_regression) 。

## [Formal definition](https://en.wikipedia.org/wiki/Statistical_model#Formal_definition)

In mathematical terms, a statistical model is usually thought of as a pair (S, P), where S is the set of possible observations, i.e. the [sample space](https://en.wikipedia.org/wiki/Sample_space), and P is a set of [probability distributions](https://en.wikipedia.org/wiki/Probability_distributions) on S.

The intuition behind this definition is as follows. It is assumed that there is a "true" probability distribution induced（诱发） by the process that generates the observed data. We choose P to represent a set (of distributions) which contains a distribution that adequately approximates the true distribution.

这个定义背后的直觉是这样的。假设在生成观测数据的过程中，存在一个“真实的”概率分布。上面这段话中的“process”的含义是“过程”。

Note that we do not require that P contains the true distribution, and in practice that is rarely the case. Indeed, as Burnham & Anderson state, "A model is a simplification or approximation of reality and hence will not reflect all of reality"—whence the saying "[all models are wrong](https://en.wikipedia.org/wiki/All_models_are_wrong)".

The set $\mathcal {P}$ is almost always parameterized: $\mathcal {P}=\{P_{\theta }:\theta \in \Theta \}$. The set $\Theta $ defines the parameters of the model. A parameterization is generally required to have distinct parameter values give rise to distinct distributions, i.e. $ P_{\theta _{1}}=P_{\theta _{2}}\Rightarrow \theta _{1}=\theta _{2}$ must hold (in other words, it must be injective). A parameterization that meets the requirement is said to be  *[identifiable](https://en.wikipedia.org/wiki/Identifiability)*.



## Example of statistical model

| Statistical model                                            | [Statistical assumptions](https://en.wikipedia.org/wiki/Statistical_assumptions) | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) | [Markov assumption](https://en.wikipedia.org/wiki/Markov_property) | 这是我在阅读[Sequence labeling](https://en.wikipedia.org/wiki/Sequence_labeling)发现的，在[Markov property](https://en.wikipedia.org/wiki/Markov_property)中也提及了此 |
| [Linear regression](https://en.wikipedia.org/wiki/Linear_regression) | [Assumptions](https://en.wikipedia.org/wiki/Linear_regression#Assumptions) | 这是最最常见的统计模型                                       |

