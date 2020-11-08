# `tf.placeholder`-VS-`tf.Variable`

从symbolic programming的角度来看，`tf.placeholder`其实就是symbol。

## stackoverflow [What's the difference between tf.placeholder and tf.Variable?](https://stackoverflow.com/questions/36693740/whats-the-difference-between-tf-placeholder-and-tf-variable)

I'm a newbie to TensorFlow. I'm confused about the difference between `tf.placeholder` and `tf.Variable`. In my view, `tf.placeholder` is used for input data, and `tf.Variable` is used to store the state of data. This is all what I know.

Could someone explain to me more in detail about their differences? In particular, when to use `tf.Variable` and when to use `tf.placeholder`?

***COMMENTS*** :

 Intuitively, you'll want gradients with respect to `Variable`s, but not `placeholder`s (whose values must always be provided). – [Yibo Yang](https://stackoverflow.com/users/4115369/yibo-yang) [Jun 6 '17 at 3:56](https://stackoverflow.com/questions/36693740/whats-the-difference-between-tf-placeholder-and-tf-variable#comment75762535_36693740) 



### [A](https://stackoverflow.com/a/36694819)

In short, you use `tf.Variable` for **trainable variables**（也就是需要计算梯度的） such as weights (W) and biases (B) for your model.

```python
weights = tf.Variable(
    tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                    stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name='weights')

biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
```

`tf.placeholder` is used to feed actual training examples.

```py
images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
```

This is how you feed the training examples during the training:

```py
for step in xrange(FLAGS.max_steps):
    feed_dict = {
       images_placeholder: images_feed,
       labels_placeholder: labels_feed,
     }
    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
```

Your `tf.variables` will be trained (modified) as the result of this training.

See more at https://www.tensorflow.org/versions/r0.7/tutorials/mnist/tf/index.html. (Examples are taken from the web page.)



### [A](https://stackoverflow.com/a/36703529)

The difference is that with `tf.Variable` you have to provide an initial value when you declare it. With `tf.placeholder` you don't have to provide an initial value and you can specify it at run time with the `feed_dict` argument inside `Session.run` 



### [A](https://stackoverflow.com/a/39177244)

Since **Tensor computations** compose of [graphs](https://www.tensorflow.org/api_docs/python/tf/Graph) then it's better to interpret the two in terms of **graphs**.

Take for example the simple linear regression WX+B=Y(where W and B stand for the **weights** and **bias** and **X** for the observations' inputs and Y for the observations' outputs). Obviously X and Y are of the same nature(**manifest variables**) which differ from that of W and B(**latent variables**). X and Y are values of the samples(observations) and hence need a **place to be filled**, while W and B are the weights and bias, *Variables*(the previous value affects the later) in the graph which should be trained using different X and Y pairs. We place different samples to the *Placeholders* to train the *Variables*.

We can and only need to **save or restore** the *Variables* (in checkpoints) to save or rebuild the graph with the code. *Placeholders* are mostly holders for the different **datasets** (for example training data or test data) but *Variables* are trained in the training process and remain the same(to predict the outcome of the input or map the inputs and outputs[labels] of the samples) later until you retrain or fine-tune the model(using different or the same samples to fill into the *Placeholders* often through the dict, for instance `session.run(a_graph, dict={a_placeholder_name: sample_values})`, *Placeholders* are also passed as parameters to set models).

If you change **placeholders**(add or delete or change the shape and etc) of a model in the middle of training, you still can reload the **checkpoint** without any other modifications. But if the variables of a saved model are changed you should adjust the **checkpoint** accordingly to reload it and continue the training(all variables defined in the graph should be available in the checkpoint).

To sum up, if the values are from the samples(observations you already have) you safely make a **placeholder** to hold them, while if you need a parameter to be trained harness a *Variable*(simply put, set the *Variables* for the values you want to get using TF automatically).

In some interesting models, like [a style transfer model](https://github.com/anishathalye/neural-style/blob/master/stylize.py#L104), the input pixes are going to be optimized and the normally-called model variables are fixed, then we should make the input(usually initialized randomly) as a variable as implemented in that link.

在一些有趣的模型中，比如一个风格转换模型，输入像素将被优化，通常被调用的模型变量是固定的，然后我们应该把输入(通常是随机初始化的)作为一个变量在链接中实现。

For more information please infer to this [simple and illustrating doc](https://www.tensorflow.org/get_started/get_started).



### [A](https://stackoverflow.com/a/43693704)

The most obvious difference between the [tf.Variable](https://www.tensorflow.org/api_docs/python/tf/Variable) and the [tf.placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) is that

------

> you use variables to hold and update parameters. Variables are in-memory buffers containing tensors. They must be explicitly initialized and can be saved to disk during and after training. You can later restore saved values to exercise or analyze the model.

Initialization of the variables is done with `sess.run(tf.global_variables_initializer())`. Also while creating a variable, you need to pass a Tensor as its initial value to the `Variable()` constructor and when you create a variable you always know its shape.

------

On the other hand, you can't update the placeholder. They also should not be initialized, but because they are a promise to have a tensor, you need to feed the value into them `sess.run(, {a: })`. And at last, in comparison to a variable, placeholder might not know the shape. You can either provide parts of the dimensions or provide nothing at all.

------

**There other differences:**

- the values inside the variable can be updated during optimizations
- variables can be [shared](https://www.tensorflow.org/programmers_guide/variable_scope), and can be [non-trainable](https://stackoverflow.com/q/40736859/1090562)
- the values inside the variable can be stored after training
- when the variable is created, [3 ops are added to a graph](https://www.tensorflow.org/programmers_guide/variables#creation) (variable op, initializer op, ops for the initial value)
- [placeholder is a function, Variable is a class](https://stackoverflow.com/a/43536220/1090562) (hence an uppercase)
- when you use TF in a distributed environment, variables are stored in a special place ([parameter server](https://www.tensorflow.org/deploy/distributed)) and are shared between the workers.

Interesting part is that not only placeholders can be fed. You can feed the value to a Variable and even to a constant.