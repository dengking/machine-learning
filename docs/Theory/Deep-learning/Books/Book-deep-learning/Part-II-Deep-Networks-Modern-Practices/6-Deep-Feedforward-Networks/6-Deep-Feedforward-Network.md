# Deep Feedforward Networks

*Deep feedforward networks* , also often called *feedforward neural networks*, or *multi-layer perceptrons* （MLPs）, are the quintessential（典型的） deep learning models. The goal of a **feedforward network** is to **approximate** some function $f ^∗$. A **feedforward networkd** defines a mapping $y = f ( x ; θ )$ and **learns** the value of the parameters $θ$ that result in the best **function approximation**.

When **feedforward neural networks** are extended to include **feedback connections**, they are called **recurrent neural networks**, presented in Chapter 10.

**Feedforward neural networks** are called because they are typically represented by composing together many different **functions**. The model is associated with a **directed acyclic graph** describing how the functions are composed together.For example, we might have three functions $f ^{(1)} $, $f ^{(2)}$ , and $f ^{(3)} $connected in a **chain**, to form $f ( x ) = f ^{(3)} ( f^{(2)} ( f^{(1)} ( x )))$. These **chain structures** are the most commonly used structures of neural networks. In this case, $f^{(1)}$ is called the first layer of the network, $f^{(2)}$ is called the second layer, and so on. The overall length of the chain gives the ***depth*** of the model.It is from this terminology that the name “**deep learning**” arises. The final **layer** of a **feedforward network** is called the
**output layer**. 

Because the **training data** does not show the desired output for each of these layers, these layers are called ***hidden layers***.

 Each **hidden layer** of the network is typically **vector-valued**. The **dimensionality** of these hidden layers determines the ***width*** of the model.Each element of the **vector** may be interpreted as playing a role analogous to a **neuron**.Rather than thinking of the layer as representing a single **vector-to-vector function**, we can also think of the layer as consisting of many ***units*** that act in parallel, each representing a **vector-to-scalar function**. Each **unit** resembles a neuron in the sense that it receives input from many other units and computes its own **activation value**. 

***SUMMARY*** : input-layer和output-layer的维度往往是由所需要解决的问题决定的，比如对于MNIST问题，output layer的是`[10,1]`，因为需要识别的是10个数字；input layer的维度由输入数据的维度决定；

***SUMMARY*** : 从函数的角度来理解layer，neuron；上述使用vector-to-scalar function来描述neuron，这个scalar就是**activation value**；

It is best to think of **feedforward networks** as **function approximation machines** that are designed to achieve **statistical generalization**, occasionally drawing some insights from what we know about the brain, rather than as models of brain function.



This general principle of improving models by **learning features** extends beyond the **feedforward networks** described in this chapter.

***SUMMARY*** : 这是指导得到feedforward neural network的思想；

First, training a **feedforward network** requires making many of the same design decisions as are necessary for a linear model: choosing the **optimizer**, the **cost function**, and the form of the output units.



 **Feedforward networks** have introduced the concept of a **hidden layer**, and this requires us to choose the ***activation functions*** that will be activation functions used to compute the **hidden layer values**.

***SUMMARY*** : 此处提出了hidden layer value的概念，其实在前面介绍***unit***（neuron）的时候也提及了activation value的概念；

We must also design the architecture of the network, including how many layers the network should contain, how these networks should be connected to each other, and how many units should be in 
each layer.





Clearly, we must use a nonlinear function to describe the features. Most neural networks do so using an **affine transformation** controlled by learned parameters, followed by a fixed, nonlinear function called an **activation function**. We use that strategy here, by defining $h = g ( W^{T} x+ c )$, where $W$ provides the weights of a linear transformation and $c$ the biases.Previously, to describe a **linear regression model**, we used a vector of weights and a scalar bias parameter to describe an **affine transformation** from an **input vector** to an **output scalar**. Now, we describe an **affine transformation** from a vector $x$ to vector $h$, so an entire vector of bias parameters is needed.  The **activation function**  $g$ is typically chosen to be a function  that is applied **element-wise**, with $h_i = g(x^{T}W_{:,i} + c_i)$.. In modern neural networks,the default recommendation is to use the rectified linear unit or **ReLU** defined by the activation function





## 补充

[A Beginner's Guide to Multilayer Perceptrons (MLP)](https://skymind.com/wiki/multilayer-perceptron)

[MULTI LAYER PERCEPTRON](http://neuroph.sourceforge.net/tutorials/MultiLayerPerceptron.html)