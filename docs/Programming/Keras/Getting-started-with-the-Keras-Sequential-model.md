# [Getting started with the Keras Sequential model](https://keras.io/getting-started/sequential-model-guide/)





### Multilayer Perceptron (MLP) for multi-class softmax classification:

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

> NOTE: 思考：上述权重矩阵是什么？一般输入矩阵的shape是`[batch_size, feature_num]`，则第一隐藏层的权重矩阵的shape是`[feature_num,  hidden_layer_node_num_1]`；
>
> 显然，第一隐藏层的权重矩阵的shape是和`batch_size`无关的；
>
> 一般，我们在阅读书籍的时候，书中所描述的流程都是一次输入一条记录，这种做法是理论上的，实际上如果真滴一次仅仅喂入一条数据的话，会非常缓慢；实际的实现是一次喂入一个batch的，即是上面所描述的输入矩阵，现代的GPU处理矩阵运算的速度非常快；其实一次喂入一条记录也可以套用上面的矩阵的表示方式，即`batch_size=1`；根据矩阵的乘法原理可以从知道每一条数据会流入到第一隐藏层中的每一个节点，一条记录流入一个节点产生的输出其实是一个标量；
>
> 无论对于什么模型，上述原理都是通用的；所以，模型中每一层的节点数和特征的个数是没有关联的；下面是一个简单的示例
>
> ```
> [
> [1,1,1],
> [1,1,1],
> [1,1,1],
> [1,1,1],
> [1,1,1],
> ]
> 
> 4*3      第一隐藏层有10个node，则它的权重矩阵是[3 * 10]
> 
> [
> [2,2,2,2,2,2,2,2,2,2],
> [2,2,2,2,2,2,2,2,2,2],
> [2,2,2,2,2,2,2,2,2,2],
> ]
> 
> 每一列表示的是
> ```
>
> 第二隐藏层的权重矩阵的shape是：`[hidden_layer_node_num_1, hidden_layer_node_num_2]`，依次类推，所以最终最后一层即输出层的与前一层之间的权重矩阵`[hidden_layer_node_num_-1, n_class]`（`-1`表示最后一层）。
>
> 所以，最终一个batch_size的数据流经MLP之后，得到的数据的shape是`[batch_size, n_classes]`。
>
> 其实从这个数学关系也可以看出为什么要将label以one-hot的方式表示了；



