[TOC]



# [What does “end to end” mean in deep learning methods?](https://stats.stackexchange.com/questions/224118/what-does-end-to-end-mean-in-deep-learning-methods)

I want to know what it is, and how it is any different from ensembling?

Suppose, I want to achieve high accuracy in classification and segmentation, for a specific task, if I use different networks, such as CNN, RNN, etc to achieve this, is this called an end to end model? (architecture?) or not?

## [A](https://stats.stackexchange.com/a/224120)

- end-to-end = all parameters are trained jointly (vs. [step-by-step](http://www.hangli-hl.com/uploads/3/4/4/6/34465961/naacl_tutorial_version2.2.pdf))
- ensembling = several classifiers are trained independently, each classifier makes a prediction, and all predictions are combined into one using some strategy (e.g., take the most common prediction across all classifiers).





# [Limits of End-to-End Learning](http://proceedings.mlr.press/v77/glasmachers17a/glasmachers17a.pdf)

## Abstract

**End-to-end learning** refers to training a possibly complex learning system by applying **gradient-based learning** to the system as a whole. **End-to-end learning systems** are specifically designed so that all modules are differentiable（可微的）. In effect, not only a **central learning machine**, but also all “peripheral” modules like **representation learning** and **memory formation** are covered by a holistic（整体的） learning process. The power of end-to-end learning has been demonstrated on many tasks, like playing a whole array of Atari video games with a single architecture. While pushing for solutions to more challenging tasks, network architectures keep growing more and more complex. In this paper we ask the question whether and to what extent **end-to-end learning** is a future-proof technique in the sense of scaling to complex and diverse data processing architectures. We point out potential inefficiencies, and we argue in particular that end-to-end learning does not make optimal use of the modular design of present neural networks. Our surprisingly simple experiments demonstrate these inefficiencies, up to the complete
breakdown of learning.

Keywords: end-to-end machine learning



## Introduction

We are today in the position to train rather deep and complex neural networks in an end-to-end (e2e) fashion, by gradient descent. In a nutshell, this amounts to（相当于） scaling up the good old backpropagation algorithm (see Schmidhuber, 2015 and references therein) to immensely rich and complex models. However, the end-to-end learning philosophy goes one step further: carefully ensuring that all modules of a learning systems are differentiable with respect to all adjustable parameters (weights) and training this system as a whole are lifted to the status of principles.

This elegant although straightforward and somewhat brute-force technique has been popularized in the context of deep learning. It is a seemingly natural consequence of deep neural architectures blurring the classic boundaries between learning machine and other processing components by casting a possibly complex processing pipeline into the coherent and flexible modeling language of neural networks.1 The approach yields state-of-the-art results (Collobert et al., 2011; Krizhevsky et al., 2012; Mnih et al., 2015). Its appeal is a unified training scheme that makes most of the available information by taking labels (supervised learning) and rewards (reinforcement learning) into account, instead of relying only on the input distribution (unsupervised pre-training). Excellent recent examples of studies