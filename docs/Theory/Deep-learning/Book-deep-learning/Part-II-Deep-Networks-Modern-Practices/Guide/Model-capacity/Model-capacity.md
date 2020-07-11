# Model capacity

在5.2 Capacity, Overfitting and Underfitting中介绍了capacity的概念，现在开始介绍deep learning model的capacity。





## [How to Control Neural Network Model Capacity With Nodes and Layers](https://machinelearningmastery.com/how-to-control-neural-network-model-capacity-with-nodes-and-layers/)

The **capacity** of a **deep learning neural network model** controls the scope of the types of **mapping functions** that it is able to learn.

A model with too little capacity cannot learn the training dataset meaning it will **underfit**, whereas a model with too much capacity may memorize the training dataset, meaning it will **overfit** or may get stuck or lost during the optimization process.

The capacity of a neural network model is defined by configuring **the number of nodes** and **the number of layers**.

In this tutorial, you will discover how to control the capacity of a neural network model and how capacity impacts what a model is capable of learning.

After completing this tutorial, you will know:

- Neural network model capacity is controlled both by the number of nodes and the number of layers in the model.
- A model with a single hidden layer and sufficient number of nodes has the capability of learning any mapping function, but the chosen learning algorithm may or may not be able to realize this capability.
- Increasing the number of layers provides a short-cut to increasing the capacity of the model with fewer resources, and modern techniques allow learning algorithms to successfully train deep models.

Discover how to train faster, reduce overfitting, and make better predictions with deep learning models [in my new book](https://machinelearningmastery.com/better-deep-learning/), with 26 step-by-step tutorials and full source code.

### Tutorial Overview

his tutorial is divided into five parts; they are:

1. Controlling Neural Network Model Capacity
2. Configure Nodes and Layers in Keras
3. Multi-Class Classification Problem
4. Change Model Capacity With Nodes
5. Change Model Capacity With Layers



### Controlling Neural Network Model Capacity

The goal of a neural network is to learn how to map input examples to output examples.

Neural networks learn mapping functions. The capacity of a network refers to the range or scope of the types of functions that the model can approximate.

> Informally, a model’s capacity is its ability to fit a wide variety of functions.

— Pages 111-112, [Deep Learning](https://amzn.to/2IXzUIY), 2016.

A model with less capacity may not be able to sufficiently learn the training dataset. A model with more capacity can model more different types of functions and may be able to learn a function to sufficiently map inputs to outputs in the training dataset. Whereas a model with too much capacity may memorize the training dataset and fail to generalize or get lost or stuck in the search for a suitable mapping function.

Generally, we can think of model capacity as a control over whether the model is likely to underfit or overfit a training dataset.

> We can control whether a model is more likely to overfit or underfit by altering its capacity.

— Pages 111, [Deep Learning](https://amzn.to/2IXzUIY), 2016.

The capacity of a neural network can be controlled by two aspects of the model:

- Number of Nodes.
- Number of Layers.

A model with more nodes or more layers has a greater capacity and, in turn, is potentially capable of learning a larger set of mapping functions.

> A model with more layers and more hidden units per layer has higher **representational capacity** — it is capable of representing more complicated functions.

— Pages 428, [Deep Learning](https://amzn.to/2IXzUIY), 2016.

The number of nodes in a layer is referred to as the **width**.

Developing wide networks with one layer and many nodes was relatively straightforward. In theory, a network with enough nodes in the single hidden layer can learn to approximate any mapping function, although in practice, we don’t know how many nodes are sufficient or how to train such a model.

The number of layers in a model is referred to as its **depth**.

Increasing the depth increases the capacity of the model. Training deep models, e.g. those with many hidden layers, can be computationally more efficient than training a single layer network with a vast number of nodes.

> Modern deep learning provides a very powerful framework for supervised learning. By adding more layers and more units within a layer, a deep network can represent functions of increasing complexity.

— Pages 167, [Deep Learning](https://amzn.to/2IXzUIY), 2016.

Traditionally, it has been challenging to train neural network models with more than a few layers due to problems such as vanishing gradients（梯度消失）. More recently, modern methods have allowed the training of deep network models, allowing the developing of models of surprising depth that are capable of achieving impressive performance on challenging problems in a wide range of domains.



## [How to decide the number of hidden layers and nodes in a hidden layer?](https://www.researchgate.net/post/How_to_decide_the_number_of_hidden_layers_and_nodes_in_a_hidden_layer)

I have 18 input features for a prediction network, so how many hidden layers should I take and what number of nodes are there in those hidden layers? Is there any formula for deciding this, or it is trial and error?