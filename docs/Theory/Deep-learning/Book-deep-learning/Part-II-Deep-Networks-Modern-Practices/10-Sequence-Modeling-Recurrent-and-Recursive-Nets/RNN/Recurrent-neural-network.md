# [Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network)

周期神经网络

A **recurrent neural network** (**RNN**) is a class of [artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) where connections between nodes form a [directed graph](https://en.wikipedia.org/wiki/Directed_graph) along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Unlike [feedforward neural networks](https://en.wikipedia.org/wiki/Feedforward_neural_networks), RNNs can use their internal state (memory) to process sequences of inputs. This makes them applicable to tasks such as unsegmented, connected [handwriting recognition](https://en.wikipedia.org/wiki/Handwriting_recognition)[[1\]](https://en.wikipedia.org/wiki/Recurrent_neural_network#cite_note-1) or [speech recognition](https://en.wikipedia.org/wiki/Speech_recognition).[[2\]](https://en.wikipedia.org/wiki/Recurrent_neural_network#cite_note-sak2014-2)[[3\]](https://en.wikipedia.org/wiki/Recurrent_neural_network#cite_note-liwu2015-3)

The term "recurrent neural network" is used indiscriminately to refer to two broad classes of networks with a similar general structure, where one is [finite impulse](https://en.wikipedia.org/wiki/Finite_impulse_response) and the other is [infinite impulse](https://en.wikipedia.org/wiki/Infinite_impulse_response). Both classes of networks exhibit temporal [dynamic behavior](https://en.wikipedia.org/wiki/Dynamic_system).[[4\]](https://en.wikipedia.org/wiki/Recurrent_neural_network#cite_note-4) A finite impulse recurrent network is a [directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) that can be unrolled（展开） and replaced with a strictly feedforward neural network, while an infinite impulse recurrent network is a [directed cyclic graph](https://en.wikipedia.org/wiki/Directed_cyclic_graph) that can not be unrolled.

Both finite impulse and infinite impulse recurrent networks can have additional stored state, and the storage can be under direct control by the neural network. The storage can also be replaced by another network or graph, if that incorporates time delays or has feedback loops. Such controlled states are referred to as gated state or gated memory, and are part of [long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory) networks (LSTMs) and [gated recurrent units](https://en.wikipedia.org/wiki/Gated_recurrent_unit).



## Architectures



RNNs come in many variants.

### Fully recurrent

Basic RNNs are a network of [neuron-like](https://en.wikipedia.org/wiki/Artificial_neuron) nodes organized into successive "layers." Each node in a given layer is connected with a [directed (one-way) connection](https://en.wikipedia.org/wiki/Directed_graph) to every other node in the next successive layer.[*citation needed*] Each node (neuron) has a time-varying real-valued activation. Each connection (synapse) has a modifiable real-valued [weight](https://en.wikipedia.org/wiki/Weighting). Nodes are either input nodes (receiving data from outside the network), output nodes (yielding results), or hidden nodes (that modify the data *en route* from input to output).

For [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) in discrete time settings, sequences of real-valued input vectors arrive at the input nodes, one vector at a time. At any given time step, each non-input unit computes its current activation (result) as a nonlinear function of the weighted sum of the activations of all units that connect to it. Supervisor-given target activations can be supplied for some output units at certain time steps. For example, if the input sequence is a speech signal corresponding to a spoken digit, the final target output at the end of the sequence may be a label classifying the digit.

In [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) settings, no teacher provides target signals. Instead a [fitness function](https://en.wikipedia.org/wiki/Fitness_function) or [reward function](https://en.wikipedia.org/wiki/Reward_function) is occasionally used to evaluate the RNN's performance, which influences its input stream through output units connected to actuators that affect the environment. This might be used to play a game in which progress is measured with the number of points won.

Each sequence produces an error as the sum of the deviations of all target signals from the corresponding activations computed by the network. For a training set of numerous sequences, the total error is the sum of the errors of all individual sequences.

