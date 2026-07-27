# [How to Code a Neural Network with Backpropagation In Python](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)



The **backpropagation algorithm** is used in the classical feed-forward（前馈的） artificial neural network.

It is the technique still used to train large [deep learning](http://machinelearningmastery.com/what-is-deep-learning/) networks.

In this tutorial, you will discover how to implement the backpropagation algorithm for a neural network from scratch with Python.

After completing this tutorial, you will know:

- How to forward-propagate an input to calculate an output.
- How to back-propagate error and train a network.
- How to apply the backpropagation algorithm to a real-world predictive modeling problem.

***SUMMARY*** : forward-propagate和back-Propagate

Discover how to code ML algorithms from scratch including [kNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm), decision trees, neural nets, ensembles and much more [in my new book](https://machinelearningmastery.com/machine-learning-algorithms-from-scratch/), with full Python code and no fancy libraries.

Let’s get started.

- **Update Nov/2016**: Fixed a bug in the activate() function. Thanks Alex!
- **Update Jan/2017**: Fixes issues with Python 3.
- **Update Jan/2017**: Updated small bug in update_weights(). Thanks Tomasz!
- **Update Apr/2018**: Added direct link to CSV dataset.
- **Update Aug/2018**: Tested and updated to work with Python 3.6.

## Description

This section provides a brief introduction to the Backpropagation Algorithm and the Wheat Seeds dataset that we will be using in this tutorial.

### Backpropagation Algorithm

The Backpropagation algorithm is a supervised learning method for **multilayer feed-forward networks **from the field of Artificial Neural Networks.

**Feed-forward neural networks** are inspired by the information processing of one or more neural cells, called a ***neuron***. A neuron accepts input signals via its dendrites（树突）, which pass the electrical signal down to the cell body. The axon（轴突） carries the signal out to synapses, which are the connections of a cell’s axon to other cell’s dendrites.

前馈神经网络受到一个或多个神经细胞（称为神经元）的信息处理的启发。 神经元通过其树突接受输入信号，树突将电信号传递到细胞体。 轴突将信号传递给突触，突触是细胞轴突与其他细胞树突的连接。

The principle of the backpropagation approach is to model a given function by modifying **internal weightings** of **input signals** to produce an expected **output signal**. The system is trained using a supervised learning method, where the error between the system’s output and a known expected output is presented to the system and used to modify its **internal state**.

Technically, the backpropagation algorithm is a method for training the weights in a **multilayer feed-forward neural network**. As such, it requires a network structure to be defined of one or more layers where one layer is fully connected to the next layer. A standard network structure is one input layer, one hidden layer, and one output layer.

Backpropagation can be used for both classification and regression problems, but we will focus on classification in this tutorial.

In classification problems, best results are achieved when the network has one neuron in the output layer for each class value. For example, a 2-class or binary classification problem with the class values of A and B. These expected outputs would have to be transformed into binary vectors with one column for each class value. Such as [1, 0] and [0, 1] for A and B respectively. This is called a one hot encoding.

### Wheat Seeds Dataset

The seeds dataset involves the prediction of species given measurements seeds from different varieties of wheat.

There are 201 records and 7 numerical input variables. It is a classification problem with 3 output classes. The scale for each numeric input value vary, so some data normalization may be required for use with algorithms that weight inputs like the backpropagation algorithm.

Below is a sample of the first 5 rows of the dataset.

```
15.26,14.84,0.871,5.763,3.312,2.221,5.22,1
14.88,14.57,0.8811,5.554,3.333,1.018,4.956,1
14.29,14.09,0.905,5.291,3.337,2.699,4.825,1
13.84,13.94,0.8955,5.324,3.379,2.259,4.805,1
16.14,14.99,0.9034,5.658,3.562,1.355,5.175,1
```



Using the Zero Rule algorithm that predicts the most common class value, the baseline accuracy for the problem is 28.095%.

You can learn more and download the seeds dataset from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/seeds).

Download the seeds dataset and place it into your current working directory with the filename **seeds_dataset.csv**.

The dataset is in tab-separated format, so you must convert it to CSV using a text editor or a spreadsheet program.

Update, download the dataset in CSV format directly:

- [Download Wheat Seeds Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv)

## Tutorial

This tutorial is broken down into 6 parts:

1. Initialize Network.
2. Forward Propagate.
3. Back Propagate Error.
4. Train Network.
5. Predict.
6. Seeds Dataset Case Study.

These steps will provide the foundation that you need to implement the **backpropagation algorithm** from scratch and apply it to your own predictive modeling problems.

### 1. Initialize Network

Let’s start with something easy, the creation of a new network ready for training.

Each neuron has a set of weights that need to be maintained. One weight for each input connection and an additional weight for the bias. We will need to store additional properties for a **neuron** during training, therefore we will use a dictionary to represent each neuron and store properties by names such as ‘**weights**‘ for the weights.

A network is organized into layers. The input layer is really just a row from our training dataset. The first real layer is the hidden layer. This is followed by the output layer that has one neuron for each class value.

We will organize layers as arrays of dictionaries and treat the whole network as an array of layers.

It is good practice to initialize the network weights to small random numbers. In this case, will we use random numbers in the range of 0 to 1.

Below is a function named **initialize_network()** that creates a new neural network ready for training. It accepts three parameters, the number of inputs, the number of neurons to have in the hidden layer and the number of outputs.

You can see that for the hidden layer we create **n_hidden** neurons and each neuron in the hidden layer has **n_inputs + 1** weights, one for each input column in a dataset and an additional one for the bias.

You can also see that the output layer that connects to the hidden layer has **n_outputs**neurons, each with **n_hidden + 1** weights. This means that each neuron in the output layer connects to (has a weight for) each neuron in the hidden layer.

