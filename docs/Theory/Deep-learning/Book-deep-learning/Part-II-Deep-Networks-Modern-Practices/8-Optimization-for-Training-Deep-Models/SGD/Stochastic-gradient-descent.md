[TOC]



# [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

**Stochastic gradient descent** (often abbreviated **SGD**) is an [iterative method](https://en.wikipedia.org/wiki/Iterative_method) for [optimizing](https://en.wikipedia.org/wiki/Mathematical_optimization) an [objective function](https://en.wikipedia.org/wiki/Objective_function) with suitable smoothness properties (e.g. [differentiable](https://en.wikipedia.org/wiki/Differentiable_function) or [subdifferentiable](https://en.wikipedia.org/wiki/Subgradient_method)). It is called **stochastic** because the method uses randomly selected (or shuffled) samples to evaluate the gradients, hence SGD can be regarded as a [stochastic approximation](https://en.wikipedia.org/wiki/Stochastic_approximation) of [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) optimization. The ideas can be traced back[[1\]](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#cite_note-1) at least to the 1951 article titled "A Stochastic Approximation Method" by [Herbert Robbins](https://en.wikipedia.org/wiki/Herbert_Robbins) and [Sutton Monro](https://en.wikipedia.org/w/index.php?title=Sutton_Monro&action=edit&redlink=1), who proposed with detailed analysis a root-finding method now called the [Robbins–Monro algorithm](https://en.wikipedia.org/wiki/Stochastic_approximation).

## Background

Main article: [M-estimation](https://en.wikipedia.org/wiki/M-estimation)

See also: [Estimating equation](https://en.wikipedia.org/wiki/Estimating_equation)

Both [statistical](https://en.wikipedia.org/wiki/Statistics) [estimation](https://en.wikipedia.org/wiki/M-estimation) and [machine learning](https://en.wikipedia.org/wiki/Machine_learning) consider the problem of [minimizing](https://en.wikipedia.org/wiki/Mathematical_optimization) an [objective function](https://en.wikipedia.org/wiki/Objective_function) that has the form of a sum:

$ Q(w)={\frac {1}{n}}\sum _{i=1}^{n}Q_{i}(w), $

where the [parameter](https://en.wikipedia.org/wiki/Parametric_statistics) $ w $ that minimizes $ Q(w) $ is to be [estimated](https://en.wikipedia.org/wiki/Estimator). Each summand function $ Q_{i} $ is typically associated with the $ i $-th [observation](https://en.wikipedia.org/wiki/Observation_(statistics)) in the [data set](https://en.wikipedia.org/wiki/Data_set) (used for training).

In classical statistics, sum-minimization problems arise in [least squares](https://en.wikipedia.org/wiki/Least_squares) and in [maximum-likelihood estimation](https://en.wikipedia.org/wiki/Maximum-likelihood_estimation) (for independent observations). The general class of estimators that arise as minimizers of sums are called [M-estimators](https://en.wikipedia.org/wiki/M-estimator). However, in statistics, it has been long recognized that requiring even local minimization is too restrictive for some problems of maximum-likelihood estimation.[[2\]](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#cite_note-2) Therefore, contemporary statistical theorists often consider [stationary points](https://en.wikipedia.org/wiki/Stationary_point) of the [likelihood function](https://en.wikipedia.org/wiki/Likelihood_function) (or zeros of its derivative, the [score function](https://en.wikipedia.org/wiki/Score_(statistics)), and other [estimating equations](https://en.wikipedia.org/wiki/Estimating_equations)).

The sum-minimization problem also arises for [empirical risk minimization](https://en.wikipedia.org/wiki/Empirical_risk_minimization). In this case, $ Q_{i}(w) $ is the value of the [loss function](https://en.wikipedia.org/wiki/Loss_function) at $ i $-th example, and $ Q(w) $ is the empirical risk.

When used to minimize the above function, a standard (or "batch") [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) method would perform the following iterations :



where $ \eta $ is a step size (sometimes called the *learning rate* in machine learning).

In many cases, the summand functions have a simple form that enables inexpensive evaluations of the sum-function and the sum gradient. For example, in statistics, [one-parameter exponential families](https://en.wikipedia.org/wiki/Exponential_families) allow economical function-evaluations and gradient-evaluations.

However, in other cases, evaluating the sum-gradient may require expensive evaluations of the gradients from all summand functions. When the training set is enormous and no simple formulas exist, evaluating the sums of gradients becomes very expensive, because evaluating the gradient requires evaluating all the summand functions' gradients. To economize on the computational cost at every iteration, stochastic gradient descent [samples](https://en.wikipedia.org/wiki/Sampling_(statistics)) a subset of summand functions at every step. This is very effective in the case of large-scale machine learning problems.[[3\]](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#cite_note-3)

