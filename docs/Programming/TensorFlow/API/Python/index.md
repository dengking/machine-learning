# TensorFlow Python reference documentation



## haosdent [TensorFlow Python reference documentation](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/)

### [Building Graphs](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/framework.html)  

Classes and functions for building TensorFlow graphs.



### [Asserts and boolean checks](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/check_ops.html)  

assertion



### [Constants, Sequences, and Random Values](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/constant_op.html)  

Constant: 生成Constant Value Tensors的API 

Sequences: 生成sequence的API 

Random Tensors: 生成random tensors with different distributions的API



### [Variables](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/state_ops.html)



### [Tensor Transformations](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/array_ops.html)  

对Tensor进行操作



### [Math](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/math_ops.html)  

basic arithmetic operators



### [Strings](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/string_ops.html)



### [Histograms](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/histogram_ops.html)  

直方图



### [Control Flow](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/control_flow_ops.html)  

TensorFlow provides several operations and classes that you can use to control the execution of operations and add conditional dependencies to your graph.



### [Higher Order Functions](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/functional_ops.html)  

TensorFlow provides several higher order operators to simplify the common **map-reduce programming patterns**



### [TensorArray Operations](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/tensor_array_ops.html)  

`class tf.TensorArray`



### [Tensor Handle Operations](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/session_ops.html)  

TensorFlow provides several operators that allows the user to keep tensors "in-place" across run calls(允许用户在运行调用时将张量保持在“原位”)



### [Images](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/image.html)  

图像进行操作的API

### [Sparse Tensors](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/sparse_ops.html)



### [Inputs and Readers](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/io_ops.html)



### [Data IO (Python functions)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/python_io.html)  

feed data的API



### [Neural Network](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/nn.html)  

`tf.nn` 
- Activation Functions 
- Convolution 
- Pooling 
- Morphological filtering 
- Normalization 
- Losses 
- Classification 
- Embeddings 
- Recurrent Neural Networks 
- Connectionist Temporal Classification (CTC) 
- Evaluation 
- Candidate Sampling 
- Other Functions and Classes 

### [Neural Network RNN Cells](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/rnn_cell.html)





### [Running Graphs](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/client.html)

This library contains classes for launching graphs and executing operations.

The [basic usage](https://haosdent.gitbooks.io/tensorflow-document/content/get_started/#basic-usage) guide has examples of how a graph is launched in a [`tf.Session`](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/client.html#Session).

### [Training](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/train.html)

This library provides a set of classes and functions that helps train models.



### [Wraps python functions](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/script_ops.html)

TensorFlow provides allows you to wrap python/numpy functions as TensorFlow operators.

### [Summary Operations](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/summary.html)

This module contains ops for generating summaries.

### [Testing](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/test.html)

TensorFlow provides a convenience class inheriting from `unittest.TestCase` which adds methods relevant to TensorFlow tests. 

### [BayesFlow Entropy (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.bayesflow.entropy.html)



### [BayesFlow Stochastic Tensors (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.bayesflow.stochastic_tensor.html)



### [BayesFlow Variational Inference (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.bayesflow.variational_inference.html)

Variational(变分法) inference.

### [CRF (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.crf.html)

Linear-chain CRF layer.

### [Statistical distributions (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.distributions.html)

Classes representing statistical distributions and ops for working with them.



### [FFmpeg (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.ffmpeg.html)

Encoding and decoding audio using FFmpeg

### [Framework (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.framework.html)



### [Graph Editor (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.graph_editor.html)

The TensorFlow Graph Editor library allows for modification of an existing `tf.Graph` instance in-place.

### [Layers (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.layers.html)

Ops for building neural network layers, regularizers, summaries, etc.

### [Learn (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.learn.html)

High level API for learning with TensorFlow.

### [Monitors (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.learn.monitors.html)

Monitors allow user instrumentation of the training process.

### [Losses (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.losses.html)

Ops for building neural network losses.

### [RNN (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.rnn.html)

Additional RNN operations and cells.

### [Metrics (contrib)](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.metrics.html)

Ops for evaluation metrics and summary statistics