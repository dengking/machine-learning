# 资源

## [TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](http://download.tensorflow.org/paper/whitepaper2015.pdf)

> A TensorFlow computation is described by a directed ***graph***, which is composed of a set of ***nodes***. The graph represents a **dataflow computation**, with extensions for allowing some kinds of nodes to maintain and update **persistent state** and for **branching** and **looping** control structures within the graph in a manner similar to Naiad [[36]](http://research.microsoft.com:8082/pubs/201100/naiad_sosp2013.pdf). Clients typically construct a **computational graph** using one of the supported front end languages (`C++` or Python). An example fragment to construct and then execute a TensorFlow graph using the Python front end is shown in Figure 1, and the resulting computation graph
> in Figure 2.

上面段中的

> with extensions for allowing some kinds of nodes to maintain and update **persistent state** 

所指的其实就`tf.Variable`，其实所谓的**persistent state** 应该就是一个表示**参数**的值，**参数**所指的是需要模型进行学习的参数，比如weight和bias；正如上述所介绍的，`tf.Variable`对应的是node，即它是一个operation，正如`tf.constant`也是一个node，这个node的出边是它的值，所以`tf.Variable`这个node的出边就是它的值；

在这篇论文的Figure 2: Corresponding computation graph for Figure 1中，将`b`（`tf.Variable`）、`W`（`tf.Variable`）、`X`（`tf.placeholder`）都画出了node；

## [Graphs and Sessions](https://www.tensorflow.org/guide/graphs)

>  Executing `v = tf.Variable(0)` adds to the graph a [`tf.Operation`](https://www.tensorflow.org/api_docs/python/tf/Operation) that will store a writeable tensor value that persists between [`tf.Session.run`](https://www.tensorflow.org/api_docs/python/tf/Session#run) calls. The [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) object wraps this operation, and can be used [like a tensor](https://www.tensorflow.org/guide/graphs#tensor-like-objects), which will read the current value of the stored value. The [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) object also has methods such as[`tf.Variable.assign`](https://www.tensorflow.org/api_docs/python/tf/Variable#assign) and [`tf.Variable.assign_add`](https://www.tensorflow.org/api_docs/python/tf/Variable#assign_add) that create [`tf.Operation`](https://www.tensorflow.org/api_docs/python/tf/Operation) objects that, when executed, update the stored value. (See [Variables](https://www.tensorflow.org/guide/variables) for more information about variables.) 

其实这段话对于理解为什么使用 [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) 来作为参数提供了一个非常好的解释，它有着如下的特性： [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) can be used [like a tensor](https://www.tensorflow.org/guide/graphs#tensor-like-objects)，当它作为一个tensor来使用的时候，那么它就会读取the current value of the stored value，这样它就有值了，显然这种情况往往是出现在前馈的时候；如果从dataflow graph的角度来看的话，这种情况就相当于从node中流出tensor值；

The [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) object also has methods such as[`tf.Variable.assign`](https://www.tensorflow.org/api_docs/python/tf/Variable#assign) and [`tf.Variable.assign_add`](https://www.tensorflow.org/api_docs/python/tf/Variable#assign_add) that create [`tf.Operation`](https://www.tensorflow.org/api_docs/python/tf/Operation) objects that, when executed, update the stored value.显然这是在反馈的时候基于梯度来学习参数的时候要使用的；如果从dataflow graph的角度来看的话，这种情况就相当于tensor值流入到node中；



## [Variables](https://www.tensorflow.org/guide/variables)

> A TensorFlow **variable** is the best way to represent **shared**, **persistent state** manipulated by your program.
>
> **Variables** are manipulated via the [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) class. A [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) represents a tensor whose **value** can be changed by running **ops** on it. Unlike [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) objects, a [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) exists **outside** the context of a single `session.run` call.
>
> Internally, a [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) stores a **persistent tensor**. Specific ops allow you to read and modify the values of this tensor. These modifications are visible across multiple [`tf.Session`](https://www.tensorflow.org/api_docs/python/tf/Session)s, so multiple workers can see the same values for a [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable).
>
> ***SUMMARY\*** ： 显然`tf.Varialbe`的这种特性是非常适合来作为模型的参数，权重的；因为训练的过程就是不断地调整参数，并且训练的过程往往不是在一个`run`中就完成的，而是需要经过多次`run`，因此`tf.Varaible`的persist特性就非常重要了；

## [Tensors](https://www.tensorflow.org/guide/tensors)

> Some types of tensors are special, and these will be covered in other units of the TensorFlow guide. The main ones are:
>
> - [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable)
> - [`tf.constant`](https://www.tensorflow.org/api_docs/python/tf/constant)
> - [`tf.placeholder`](https://www.tensorflow.org/api_docs/python/tf/placeholder)
> - [`tf.SparseTensor`](https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor)
>
> With the exception of [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable), the value of a tensor is **immutable**, which means that in the context of a single execution tensors only have a single value. However, evaluating the same tensor twice can return different values; for example that tensor can be the result of reading data from disk, or generating a random number.

***SUMMARY*** : 从computation graph的角度来看待：

- [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable)
- [`tf.constant`](https://www.tensorflow.org/api_docs/python/tf/constant)
- [`tf.placeholder`](https://www.tensorflow.org/api_docs/python/tf/placeholder)

它们都node， 但是它们都有一个出边，这个出边是它所维护的tensor的值；

## SUMMARY

其实从不同的角度来对[`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable)进行分析，会得出不同的结论；

