# Core graph data structures



## `class tf.Graph`

A TensorFlow computation, represented as a dataflow graph.

A `Graph` contains a set of [`Operation`](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/framework.html#Operation) objects, which represent units of computation; and [`Tensor`](https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/framework.html#Tensor) objects, which represent the units of data that flow between operations.

## `class tf.Operation`

Represents a graph node that performs computation on tensors.

## `class tf.Tensor`