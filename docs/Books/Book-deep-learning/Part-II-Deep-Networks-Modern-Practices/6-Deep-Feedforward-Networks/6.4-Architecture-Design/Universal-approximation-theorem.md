# [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)

In the [mathematical](https://en.wikipedia.org/wiki/Mathematics) theory of [artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_networks), the **universal approximation theorem** states[[1\]](https://en.wikipedia.org/wiki/Universal_approximation_theorem#cite_note-1) that a [feed-forward](https://en.wikipedia.org/wiki/Feedforward_neural_network) network with a single hidden layer containing a finite number of [neurons](https://en.wikipedia.org/wiki/Neuron) can approximate [continuous functions](https://en.wikipedia.org/wiki/Continuous_function) on [compact subsets](https://en.wikipedia.org/wiki/Compact_space) of [**R***n*](https://en.wikipedia.org/wiki/Euclidean_space), under mild assumptions on the activation function. The theorem thus states that simple neural networks can *represent* a wide variety of interesting functions when given appropriate parameters; however, it does not touch upon the algorithmic [learnability](https://en.wikipedia.org/wiki/Computational_learning_theory) of those parameters.

One of the first versions of the [theorem](https://en.wikipedia.org/wiki/Theorem) was proved by [George Cybenko](https://en.wikipedia.org/wiki/George_Cybenko) in 1989 for [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation functions.[[2\]](https://en.wikipedia.org/wiki/Universal_approximation_theorem#cite_note-cyb-2)

Kurt Hornik showed in 1991[[3\]](https://en.wikipedia.org/wiki/Universal_approximation_theorem#cite_note-horn-3) that it is not the specific choice of the activation function, but rather the multilayer feedforward architecture itself which gives neural networks the potential of being universal approximators. The output units are always assumed to be linear.

Although [feed-forward networks](https://en.wikipedia.org/wiki/Feed-forward_network) with a single hidden layer are universal approximators, the width of such networks has to be exponentially large. In 2017 Lu et al.[[4\]](https://en.wikipedia.org/wiki/Universal_approximation_theorem#cite_note-ZhouLu-4) proved universal approximation theorem for width-bounded [deep neural networks](https://en.wikipedia.org/wiki/Deep_neural_network). In particular, they showed that width-*n+4* networks with [ReLU](https://en.wikipedia.org/wiki/ReLU) activation functions can approximate any [Lebesgue integrable function](https://en.wikipedia.org/wiki/Lebesgue_integration) on *n*-dimensional input space with respect to $ L^{1} $distance if network depth is allowed to grow. They also showed the limited expressive power if the width is less than or equal to *n*. All [Lebesgue integrable functions](https://en.wikipedia.org/wiki/Lebesgue_integration) except for a zero measure set cannot be approximated by width-*n* [ReLU](https://en.wikipedia.org/wiki/ReLU) networks.

Later Hanin improved the earlier result,[[4\]](https://en.wikipedia.org/wiki/Universal_approximation_theorem#cite_note-ZhouLu-4) showing that [ReLU](https://en.wikipedia.org/wiki/ReLU) networks with width *n+1* is sufficient to approximate any [continuous](https://en.wikipedia.org/wiki/Continuous_function) [convex function](https://en.wikipedia.org/wiki/Convex_function) of *n*-dimensional input variables.[[5\]](https://en.wikipedia.org/wiki/Universal_approximation_theorem#cite_note-5)







# Universal approximation theorem and representational capacity and effective capacity

在chapter 5.2中提出了representational capacity 和 effective capacity的概念，这两个概念和universal approximation theorem非常密切；