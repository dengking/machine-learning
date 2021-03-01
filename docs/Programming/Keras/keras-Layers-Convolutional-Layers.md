# [Convolutional Layers](https://keras.io/layers/convolutional/)

### Conv1D

```python
keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

```

1D convolution layer (e.g. **temporal convolution**).

This layer creates a **convolution kernel** that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. If `use_bias` is True, a bias vector is created and added to the outputs. Finally, if `activation` is not `None`, it is applied to the outputs as well.

When using this layer as the first layer in a model, provide an `input_shape` argument (tuple of integers or `None`, does not include the **batch** axis), e.g. `input_shape=(10, 128)` for time series sequences of 10 time steps with 128 features per step in `data_format="channels_last"`, or `(None, 128)` for variable-length sequences with 128 features per step.



 **Arguments** 

- **filters**: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
- **kernel_size**: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
- **strides**: An integer or tuple/list of a single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any `dilation_rate` value != 1.
-  **data_format**: A string, one of `"channels_last"` (default) or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch, steps, channels)` (default format for temporal data in Keras) while `"channels_first"` corresponds to inputs with shape `(batch, channels, steps)`. 



> NOTE: 关于**filters**和**kernel_size**，参见下面这篇文章：[Keras conv1d layer parameters: filters and kernel_size](https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size)



> NOTE : 关于channel-last和channel-first，参见这篇文章：[A Gentle Introduction to Channels-First and Channels-Last Image Formats](https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/)



#### Input  shape

3D tensor with shape: `(batch, steps, channels)`

> NOTE: 要想理解这段话中`steps`、`channels`的含义，首先需要仔细阅读上面的第三段，其中已经给出了一个example；这里我再补充一个例子：

如果以[Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf)中的sentence为例，那么`steps`则表示的sentence的长度，即sentence中word的个数；`channels`则表示word embedding+position embedding的长度；

#### Output shape

3D tensor with shape: `(batch, new_steps, filters)` `steps` value might have changed due to padding or strides.

> NOTE: 上述output shape和[Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf)中描述的不同；



> NOTE: 一般在讲解model的原理时候都是不会涉及到`batch_size`的，而是仅仅一一条输入数据为例来进行说明，但是实现库中则必须要涉及到`batch_size`，这里便是这样；其实我觉得应该这样来理解：Conv1D肯定会对输入的`batch_size`条记录中的每一条都执行系统的卷积过程；



