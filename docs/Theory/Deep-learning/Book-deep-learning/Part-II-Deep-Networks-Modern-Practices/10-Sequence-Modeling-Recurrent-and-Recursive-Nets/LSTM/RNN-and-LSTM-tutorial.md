# [Recurrent neural networks and LSTM tutorial in Python and TensorFlow](https://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/)

In the deep learning journey so far on this website, I’ve introduced [dense neural networks](https://adventuresinmachinelearning.com/neural-networks-tutorial/) and [convolutional neural networks](https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/) (CNNs) which explain how to perform classification tasks on static images.  We’ve seen good results, especially with CNN’s. However, what happens if we want to analyze dynamic data? What about videos, voice recognition or sequences of text? There are ways to do some of this using CNN’s, but the most popular method of performing classification and other analysis on *sequences* of data is recurrent neural networks.  This tutorial will be a very comprehensive introduction to recurrent neural networks and a subset of such networks – long-short term memory networks (or LSTM networks). I’ll also show you how to implement such networks in TensorFlow – including the data preparation step. It’s going to be a long one, so settle in and enjoy these pivotal networks in deep learning – at the end of this post, you’ll have a very solid understanding of recurrent neural networks and LSTMs. By the way, if you’d like to learn how to build LSTM networks in Keras, see [this tutorial](https://adventuresinmachinelearning.com/keras-lstm-tutorial/).

# An introduction to recurrent neural networks

A recurrent neural network, at its most fundamental level, is simply a type of densely connected neural network (for an introduction to such networks, [see my tutorial](https://adventuresinmachinelearning.com/neural-networks-tutorial/)). However, the key difference to normal feed forward networks is the introduction of *time* – in particular, the output of the hidden layer in a **recurrent neural network** is *fed back* *into itself*. Diagrams help here, so observe:

![Recurrent LSTM tutorial - RNN diagram with nodes](https://i0.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/09/Explicit-RNN.jpg?resize=363%2C229&ssl=1)

Recurrent neural network diagram with nodes shown

In the diagram above, we have a simple recurrent neural network with three input nodes.  These input nodes are fed into a hidden layer, with sigmoid activations, as per any normal [densely connected neural network](https://adventuresinmachinelearning.com/neural-networks-tutorial/). What happens next is what is interesting – the output of the hidden layer is then *fed back* into the same hidden layer. As you can see the hidden layer outputs are passed through a conceptual *delay* block to allow the input of $\textbf{h}^{t-1}$ into the hidden layer.  What is the point of this? Simply, the point is that we can now model *time* or sequence-dependent data.

***SUMMARY*** ： 输入数据是按照time序列而组织的多条记录，这些数据会按照**时间次序**输入到分多次输入到model中呢还是一次输入多条记录，即输入是一个向量而不是是一个标量；



A particularly good example of this is predicting text sequences.  Consider the following text string: “A girl walked into a bar, and she said ‘Can I have a drink please?’.  The bartender said ‘Certainly {}”. There are many options for what could fill in the {} symbol in the above string, for instance, “miss”, “ma’am” and so on. However, other words could also fit, such as “sir”, “Mister” etc. In order to get the correct gender of the noun, the neural network needs to “recall” that two previous words designating the likely gender (i.e. “girl” and “she”) were used. This type of flow of information through time (or sequence) in a **recurrent neural network** is shown in the diagram below, which *unrolls* the sequence:

![Recurrent LSTM tutorial - unrolled RNN](https://i2.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/09/Recurrent-neural-network.png?resize=555%2C181&ssl=1)

Unrolled recurrent neural network

***SUMMARY*** : `t`表示的是time step

On the left-hand side of the above diagram, we have basically the same diagram as the first (the one which shows all the nodes explicitly). What the previous diagram neglected to show explicitly was that we in fact only ever supply finite length sequences to such networks – therefore we can *unroll* the network as shown on the right-hand side of the diagram above. This unrolled network shows how we can supply a stream of data to the recurrent neural network. For instance, first, we supply the word vector for “A” (more about word vectors later) to the network *F* – the output of the nodes in *F* are fed into the “next” network and also act as a stand-alone output ($h_0$).  The next network (though it is really the same network) *F* at time *t=1* takes the next word vector for “girl” and the previous output $h_0$ into its hidden nodes, producing the next output $h_1$ and so on.



# Creating an LSTM network in TensorFlow

We are now going to create an LSTM network in TensorFlow. The code will loosely follow the TensorFlow team tutorial found [here](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb), but with updates and my own substantial modifications. The text dataset that will be used and is a common benchmarking corpus is the [Penn Tree Bank](https://catalog.ldc.upenn.edu/ldc99t42) (PTB) dataset. As usual, all the code for this post can be found on the [AdventuresinML Github site](https://github.com/adventuresinML/adventures-in-ml-code). To run this code, you’ll first have to download and extract the .tgz file from [here](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz). First off, we’ll go through the data preparation part of the code.



## Preparing the data

This code will use, verbatim（逐字的）, the following functions from the [previously mentioned TensorFlow tutorial](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb): *read_words, build_vocab* and *file_to_word_ids.* I won’t go into these functions in detail, but basically, they first split the given text file into separate words and sentence based characters (i.e. end-of-sentence <eos>). Then, each **unique word** is identified and assigned a unique integer. Finally, the original text file is converted into a list of these unique integers, where each word is substituted with its new integer identifier. This allows the text data to be consumed in the neural network.



## Creating an input data pipeline

As discussed in my [TensorFlow queues and threads](https://adventuresinmachinelearning.com/introduction-tensorflow-queuing/) tutorial, the use of a feed dictionary to supply data to your model during training, while common in tutorials, is not efficient – as can be read [here](https://www.tensorflow.org/performance/performance_guide#input_pipeline_optimization) on the TensorFlow site. Rather, it is more efficient to use TensorFlow queues and threading. Note, that there is a new way of doing things, using the Dataset API, which won’t be used in this tutorial, but I will perhaps update it in the future to include this new way of doing things. I’ve packaged up this code in a function called *batch_producer* – this function extracts batches of *x, y* training data – the *x* batch is formatted as the **time stepped** text data. The y batch is the same data, except delayed one **time step**. So, for instance, a single *x, y* sample in a batch, with the number of time steps being 8, looks like:

正如在我的TensorFlow队列和线程教程中所讨论的，在训练期间使用feed字典向模型提供数据，虽然在教程中很常见，但并不有效——可以在TensorFlow站点上阅读。相反，使用TensorFlow队列和线程更有效。注意，有一种使用Dataset API的新方法，在本教程中不会用到，但我可能会在将来更新它，以包括这种新方法。我将这段代码打包到一个名为batch_producer的函数中——这个函数提取成批的x、y训练数据——x批数据被格式化为时间步长文本数据。y批处理是相同的数据，只是延迟了一个时间步长。例如，批次中的单个x, y样本，时间步长为8，看起来像:

- *x =* “A girl walked into a bar, and she”
- y = “girl walked into a bar, and she said”

Remember that *x* and *y* will be batches of integer data, with the size (*batch_size*, *num_steps*), not text as shown above – however, I have shown the above *x* and *y* sample in text form to aid understanding. So, as demonstrated in the **model architecture diagram** above, we are producing a many-to-many LSTM model, where the model will be trained to predict the very next word in the sequence *for each* word in the number of time steps.

***SUMMARY*** : 需要将文本数据分成*batch_size*批数据，每一批中包含*num_steps*个单词；

Here’s what the code looks like:

```python
def batch_producer(raw_data, batch_size, num_steps):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size # len of every batch
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y
```

In the code above, first, the raw text data is converted into an *int32* tensor. Next, the length of the full data set is calculated and stored in *data_len* and this is then divided by the batch size in an *integer division (//)* to get the number of full batches of data available within the dataset. The next line reshapes the *raw_data* tensor (restricted in size to the number of full batches of data i.e. 0 to *batch_size \* batch_len*) into a (*batch_size, batch_len*) shape. The next line sets the number of iterations in each epoch – usually, this is set so that all the training data is passed through the algorithm in each epoch. This is what occurs here – the number of batches in the data (*batch_len*) is integer divided by the number of time steps – this gives the number of time-step-sized batches that are available to be iterated through in a single epoch.



## Creating the model

In this code example, in order to have nice encapsulation and better-looking code, I’ll be building the model in [Python classes](https://docs.python.org/3/tutorial/classes.html). The first class is a simple class that contains the input data:

```python
class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)
```



We pass this object important input data information such as **batch size**, **the number of recurrent time steps** and finally the raw data file we wish to extract batch data from. The previously explained *batch_producer* function, when called, will return our input data batch *x* and the associated time step + 1 target data batch, *y*.