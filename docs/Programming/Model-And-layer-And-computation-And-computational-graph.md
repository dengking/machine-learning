[TOC]

# 前言

# model

目前已经使用了三种model：

- MLP
- CNN
- RNN

每种model都表示一种computation（这是model的本质），在论文中，也是通过computation（各种各样的公式）来描述一个模型的；而在各种blog和tutorial中，往往喜欢使用model的topology来介绍它；显然model的topology可以看做是对computation的形式化地表示，它们能够帮助学习者更快地学习model，帮助学习者理解model的computation；

在deep learning book的6.5.1 Computational Graphs中介绍，我们可以computationaln graph来formalizing computation as graph。在TensorFlow的low API中指出：TensorFlow也是使用[computational graph](https://www.tensorflow.org/guide/low_level_intro)来实现底层计算的描述的（ [A TensorFlow computation, represented as a dataflow graph.](https://www.tensorflow.org/api_docs/python/tf/Graph) tensorflow中是使用的dataflow graph来表示computation的）

要想完整地掌握一个模型，需要从computation，model topology，computation graph这三个方面入手；其实computation才是本质所在，model topology，computation graph都是对computation的形式化地展示；除此之外，目前大多数deep learning framework的实现都是采用的基于tensor的数据表示、基于tensor的computation，从整体来看这个model可以看做是一个函数`f(input_tensor)= outout_tensor`，`input_tensor`流经model得到`output_tensor`，所以掌握tensor在model中的流动过程（流动过程中tensor的shape的变化等）也是掌握model的computation的一个捷径；更加准确地说他们是使用[dataflow graph](https://www.tensorflow.org/api_docs/python/tf/Graph)来表示computation，比如tensorflow；

在模型设计中所设计地各种思想，如parameter sharing等，都是有对应的computation的；

[Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)



## layer

按照目前的大多数深度学习库，如tensorflow，Keras，都是将model抽象成由多个layer叠加而成的，所以在进行实现的时候，往往是先从实现层入手；并且很多的论文中也是这样描述的；从deep learning book的chapter 6中，也将这个model描述为一个复合函数，复合函数中的每一个都对应了一层；

# computational graph

其实computation graph并没有严格的定义，在deep learning book中的computation graph就和TensorFlow的computation graph是定义就是不同的；

deep learning book中的computation graph侧重点在于对computation的形式化地展示，它的规则如下：
-  node in the graph to indicate a variable



TensorFlow的[computational graph](https://www.tensorflow.org/guide/low_level_intro)的定义如下：

- [`tf.Operation`](https://www.tensorflow.org/api_docs/python/tf/Operation) (or "ops"): The nodes of the graph. Operations describe calculations that consume and produce tensors.
- [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor): The edges in the graph. These represent the values that will flow through the graph. Most TensorFlow functions return `tf.Tensors`.

显然，TensorFlow中的定义和deep learning book中的定义是不同的，直观是觉得TensorFlow中[computational graph](https://www.tensorflow.org/guide/low_level_intro)的定义是比较适合于编码实现的，而deep learning book中的定义是便于对computation的形式化地展示（当然TensorFlow的computational graph也能够达到这个目的）；

在下面的描述中，computational graph的描述都是按照TensorFlow中的定义；

# MLP

MLP的model topology：

- full connected（输入层与隐藏层之间，隐藏层与隐藏层之间，隐藏层与输出层之间，都是按照这种方式连接的）

MLP的computation：

- matrix multiplication



下面的代码是摘自[Getting started with the Keras Sequential model](https://keras.io/getting-started/sequential-model-guide/)的Multilayer Perceptron (MLP) for multi-class softmax classification:

```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```

思考：上述代码所构建的模型的权重矩阵是什么？

答：一般**输入矩阵**的shape是`[batch_size, feature_num]`，则第一隐藏层的权重矩阵的shape是`[feature_num, hidden_layer_node_num_1]`，即第一隐藏层的权重矩阵的shape是和`batch_size`无关的，需要注意的是，这是一个非常好的特性：这样我们就可以在不知道`batch_size`的情况下就可以构建模型了，无论对于什么模型，模型中每一层的**节点数**和**特征的个数**、`batch_size`都是没有关联的（当涉及LSTM的时候，LSTM中每一层的neuron的个数和`time_step`也是没有关联的），**特征的个数**会影响neuron中的参数个数有关；

由于MLP要求full connected，使用**矩阵乘法**是能够非常好地实现这种需求的，下面是一个简单的示例：

```
[
[1,1,1],
[1,1,1],
[1,1,1],
[1,1,1],
[1,1,1],
]

4*3      第一隐藏层有10个node，则它的权重矩阵是[3 * 10]

[
[2,2,2,2,2,2,2,2,2,2],
[2,2,2,2,2,2,2,2,2,2],
[2,2,2,2,2,2,2,2,2,2],
]

每一列表示的是
```

第二隐藏层的权重矩阵的shape是：`[hidden_layer_node_num_1, hidden_layer_node_num_2]`，依次类推，所以最终最后一层即输出层的与前一层之间的权重矩阵`[hidden_layer_node_num_-1, n_class]`（`-1`表示最后一层）。

所以，一个batch_size的数据流经MLP之后，最终得到的数据的shape是`[batch_size, n_classes]`。

其实从这个数学关系也可以看出为什么要将label以one-hot的方式表示了；



下面的代码是TensorFlow中构建MLP的一个demo：

```python
        with tf.name_scope('input'):
            self.x_in = tf.placeholder(tf.float32, [None, self.feature_num], name='x_in')
            # self.y_in = tf.placeholder(tf.float32, [None, self.time_step, self.n_classes], name='y_in')
            # 每个sequence都取最后一条记录的target来作为这个sequence的target
            self.y_in = tf.placeholder(tf.float32, [None, self.n_classes], name='y_in')
            self.keep_prob = tf.placeholder(tf.float32, name='dropout_in')

        with tf.name_scope('layer1'):
            w_in = self.__weight_variable__([self.feature_num, self.num_lstm_units])
            b_in = self.__bias_variable__([self.num_lstm_units])

            lstm_input_layer = tf.reshape(self.x_in, [-1, self.feature_num])  # input layer
            lstm_input_layer = tf.nn.relu(tf.matmul(lstm_input_layer, w_in) + b_in)
```



一般，我们在阅读书籍的时候，书中所描述的流程都是一次输入一条记录，这种做法是理论上的，实际上如果真滴一次仅仅喂入一条数据的话，会非常缓慢；实际的实现是一次喂入一个batch的，即是上面所描述的**输入矩阵**，现代的GPU处理矩阵运算的速度非常快；其实一次喂入一条记录也可以套用上面的矩阵的表示方式，即`batch_size=1`；

从TensorFlow的代码可以看出，输入矩阵和第一层的权重矩阵执行矩阵乘法，根据矩阵**乘法原理**可以知道每一条数据会流入到第一隐藏层中的每一个节点，一条记录流入一个节点产生的输出其实是一个标量；其实这也是full connected的含义所在；









# CNN

关于CNN的computation和model topology参考下面两篇文章，其中给出了非常好的解释：

- [Convolutional Neural Networks (CNNs / ConvNets)](http://cs231n.github.io/convolutional-networks/)
- [A Beginner's Guide To Understanding Convolutional Neural Networks](https://adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)



全连接对应的是矩阵乘法，CNN中的filter则对应的卷积运算，卷积层中的神经元只会和input的一部分进行连接，而不是全连接；

# RNN



## LSTM

[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)



## Bi-RNN



## encoder-decoder/seq2seq



## encoder-align model-decoder

paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)就是采用的这种架构，它的model topology在blog [Attention mechanism](https://lab.heuritech.com/attention-mechanism)这给出了，两者结合起来能够更加深刻理解它的本质；	