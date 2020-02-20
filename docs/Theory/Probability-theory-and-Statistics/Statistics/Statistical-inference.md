# Statistical inference

本文是阅读维基百科[Statistical inference](https://en.wikipedia.org/wiki/Statistical_inference)的笔记，本文的内容涉及了：

- 维基百科[Statistical model](https://en.wikipedia.org/wiki/Statistical_model)
- Deep learning book的Chapter [5.2 Capacity, Overfitting and Underfitting](https://dengking.github.io/machine-learning/Theory/Deep-learning/Book-deep-learning/Chapter-5-Machine-Learning-Basics/5.2-Capacity-Overfitting-and-Underfitting/)

要充分理解本文以及后续内容，需要首先搞清楚数学家们为什么创建了“Statistical inference”这个学科，也就是这个学科能够解决什么问题：

统计学认为，数据的生成（generate）是有一定规律的，数学中使用[概率分布](https://en.wikipedia.org/wiki/Probability_distributions)来进行描述这个规律。现实世界中，一条数据生成完全可以使用掷骰子来进行类比，掷一次骰子就生成一个数据。我们可以肯定的是，现实世界中，肯定存在着一个[概率分布](https://en.wikipedia.org/wiki/Probability_distributions)，它控制着数据的生成，它就像是上帝的手，在各种文章中，它可能被称为

- “underlying [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution)”（在维基百科[Statistical inference](https://en.wikipedia.org/wiki/Statistical_inference)中使用的这种描述）
- “underlying distribution“（在Deep learning book的Chapter [5.2 Capacity, Overfitting and Underfitting](https://dengking.github.io/machine-learning/Theory/Deep-learning/Book-deep-learning/Chapter-5-Machine-Learning-Basics/5.2-Capacity-Overfitting-and-Underfitting/)中使用的这种描述）
- “the true distribution”

需要注意的是，在后面的内容中常常会出现“data generating process”这个词，“data-generating process”的字面意思是“数据生成过程”，它其实和“probability distribution”意思相同。参见[Data generating process](https://en.wikipedia.org/wiki/Data_generating_process)、Deep learning book的Chapter [5.2 Capacity, Overfitting and Underfitting](https://dengking.github.io/machine-learning/Theory/Deep-learning/Book-deep-learning/Chapter-5-Machine-Learning-Basics/5.2-Capacity-Overfitting-and-Underfitting/)。

那这就引出了一个问题：我们如何得到“underlying [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution)”？ 这就是[Statistical inference](https://en.wikipedia.org/wiki/Statistical_inference)的研究内容了。其名称中的“inference”蕴含着：以目前人类的科技水平，我们无法准确地描述这个“underlying [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution)”（只有上帝才知道它），只能够去deduce（infer）这个“underlying [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution)”的性质，我们只能够去approximate这个“underlying [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution)”。



有了这些基本的认知，读者应该能够更好地理解本文和后续的内容了。

**Statistical inference** is the process of using [data analysis](https://en.wikipedia.org/wiki/Data_analysis) to deduce（推断） properties of an underlying [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution). It is assumed that the observed data set is [sampled](https://en.wikipedia.org/wiki/Sampling_(statistics)) from a larger population.

这段话回答了statistical inference所“infer”的：properties of an underlying [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution)。我们假设研究的问题是存在着一个distribution的，我们通过[Statistical inference](https://en.wikipedia.org/wiki/Statistical_inference)可以学习到这个distribution。



## Introduction

*Main articles:* [Statistical model](https://en.wikipedia.org/wiki/Statistical_model) *and* [Statistical assumptions](https://en.wikipedia.org/wiki/Statistical_assumptions)

Statistical inference makes propositions about a population, using data drawn from the population with some form of [sampling](https://en.wikipedia.org/wiki/Sampling_(statistics)). Given a hypothesis about a population, for which we wish to draw inferences, statistical inference consists of 

- (first) [selecting](https://en.wikipedia.org/wiki/Model_selection) a [statistical model](https://en.wikipedia.org/wiki/Statistical_model) of the process that generates the data 
- (second) deducing propositions from the model.

"The majority of the problems in statistical inference can be considered to be problems related to statistical modeling"



## Paradigms for inference

### [Frequentist inference](https://en.wikipedia.org/wiki/Frequentist_inference)



### [Bayesian Inference](https://en.wikipedia.org/wiki/Bayesian_Inference)



### Likelihood-based inference

*Main article:* [Likelihoodism](https://en.wikipedia.org/wiki/Likelihoodism)