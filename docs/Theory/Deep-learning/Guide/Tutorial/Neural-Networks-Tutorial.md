# [Neural Networks Tutorial – A Pathway to Deep Learning](https://adventuresinmachinelearning.com/neural-networks-tutorial/)

Chances are, if you are searching for a tutorial on artificial neural networks (ANN) you already have some idea of what they are, and what they are capable of doing.  But did you know that neural networks are the foundation of the new and exciting field of deep learning?  Deep learning is the field of machine learning that is making many state-of-the-art advancements, from beating players at [Go](http://www.sciencemag.org/news/2016/01/huge-leap-forward-computer-mimics-human-brain-beats-professional-game-go) and [Poker](http://www.sciencemag.org/news/2017/03/artificial-intelligence-goes-deep-beat-humans-poker) ([reinforcement learning](https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/)), to speeding up [drug discovery](http://www.nanalyze.com/2016/01/4-companies-using-deep-learning-for-drug-discovery/) and [assisting self-driving cars](http://spectrum.ieee.org/cars-that-think/transportation/self-driving/driveai-brings-deep-learning-to-selfdriving-cars).  If these types of cutting edge applications excite you like they excite me, then you will be interesting in learning as much as you can about deep learning.  However, that requires you to know quite a bit about how neural networks work.  This tutorial article is designed to help you get up to speed in neural networks as quickly as possible.

In this tutorial I’ll be presenting some concepts, code and maths that will enable you to build *and understand* a simple neural network.  Some tutorials focus only on the code and skip the maths – but this impedes understanding. I’ll take things as slowly as possible, but it might help to brush up on your [matrices](https://www.khanacademy.org/math/precalculus/precalc-matrices) and [differentiation](https://www.khanacademy.org/math/differential-calculus) if you need to. The code will be in Python, so it will be beneficial if you have a basic understanding of how Python works.  You’ll pretty much get away with knowing about Python functions, loops and the basics of the [numpy](http://www.numpy.org/) library.  By the end of this neural networks tutorial you’ll be able to build an ANN in Python that will correctly classify handwritten digits in images with a fair degree of accuracy.

Once you’re done with this tutorial, you can dive a little deeper with the following posts:

[Python TensorFlow Tutorial – Build a Neural Network](https://adventuresinmachinelearning.com/python-tensorflow-tutorial/)
[Improve your neural networks – Part 1 [TIPS AND TRICKS]](https://adventuresinmachinelearning.com/improve-neural-networks-part-1/)
[Stochastic Gradient Descent – Mini-batch and more](https://adventuresinmachinelearning.com/stochastic-gradient-descent/)

All of the relevant code in this tutorial can be found [here](https://github.com/adventuresinML/adventures-in-ml-code).

 

Here’s an outline of the tutorial, with links, so you can easily navigate to the parts you want:

[1 What are artificial neural networks?](https://adventuresinmachinelearning.com/neural-networks-tutorial/#what-are-anns)
[2 The structure of an ANN](https://adventuresinmachinelearning.com/neural-networks-tutorial/#structure-ann)
[2.1 The artificial neuron](https://adventuresinmachinelearning.com/neural-networks-tutorial/#the-artificial-neuron)
[2.2 Nodes](https://adventuresinmachinelearning.com/neural-networks-tutorial/#nodes)
[2.3 The bias](https://adventuresinmachinelearning.com/neural-networks-tutorial/#the-bias)
[2.4 Putting together the structure](https://adventuresinmachinelearning.com/neural-networks-tutorial/#putting-together-the-structure)
[2.5 The notation](https://adventuresinmachinelearning.com/neural-networks-tutorial/#the-notation)
**3 The feed-forward pass**
[3.1 A feed-forward example](https://adventuresinmachinelearning.com/neural-networks-tutorial/#the-feed-forward-pass)
[3.2 Our first attempt at a feed-forward function](https://adventuresinmachinelearning.com/neural-networks-tutorial/#first-attempt-feed-forward)
[3.3 A more efficient implementation](https://adventuresinmachinelearning.com/neural-networks-tutorial/#more-efficient-implementation)
[3.4 Vectorisation in neural networks](https://adventuresinmachinelearning.com/neural-networks-tutorial/#vectorisation)
[3.5 Matrix multiplication](https://adventuresinmachinelearning.com/neural-networks-tutorial/#matrix-mult)
**4 Gradient descent and optimisation**
[4.1 A simple example in code](https://adventuresinmachinelearning.com/neural-networks-tutorial/#simple-example)
[4.2 The cost function](https://adventuresinmachinelearning.com/neural-networks-tutorial/#the-cost-function)
[4.3 Gradient descent in neural networks](https://adventuresinmachinelearning.com/neural-networks-tutorial/#gradient-descent-in-nn)
[4.4 A two dimensional gradient descent example](https://adventuresinmachinelearning.com/neural-networks-tutorial/#two-dimensional)
[4.5 Backpropagation in depth](https://adventuresinmachinelearning.com/neural-networks-tutorial/#backprop-in-depth)
[4.6 Propagating into the hidden layers](https://adventuresinmachinelearning.com/neural-networks-tutorial/#prop-in-hidden-layers)
[4.7 Vectorisation of backpropagation](https://adventuresinmachinelearning.com/neural-networks-tutorial/#vector-backprop)
[4.8 Implementing the gradient descent step](https://adventuresinmachinelearning.com/neural-networks-tutorial/#imp-gradient-desc)
[4.9 The final gradient descent algorithm](https://adventuresinmachinelearning.com/neural-networks-tutorial/#final-gradient-desc-algo)
**5 Implementing the neural network in Python**
[5.1 Scaling data](https://adventuresinmachinelearning.com/neural-networks-tutorial/#scaling-data)
[5.2 Creating test and training datasets](https://adventuresinmachinelearning.com/neural-networks-tutorial/#test-and-train)
[5.3 Setting up the output layer](https://adventuresinmachinelearning.com/neural-networks-tutorial/#setting-up-output)
[5.4 Creating the neural network](https://adventuresinmachinelearning.com/neural-networks-tutorial/#creating-nn)
[5.5 Assessing the accuracy of the trained model](https://adventuresinmachinelearning.com/neural-networks-tutorial/#creating-nn)

## 1 What are artificial neural networks?

Artificial neural networks (ANNs) are software implementations of the neuronal structure of our brains.  We don’t need to talk about the complex biology of our brain structures, but suffice to say, the brain contains ***neurons*** which are kind of like **organic switches**.  These can change their output state depending on the strength of their electrical or chemical input.  The neural network in a person’s brain is a hugely interconnected network of **neurons**, where the output of any given neuron may be the input to thousands of other neurons.  Learning occurs by repeatedly **activating** certain neural connections over others, and this reinforces（加强） those connections.  This makes them more likely to produce a desired outcome given a specified input.  This learning involves *feedback* – when the desired outcome occurs, the neural connections causing that outcome become strengthened.

*Artificial* neural networks attempt to simplify and mimic（模仿） this brain behaviour.  They can be trained in a *supervised* or *unsupervised* manner.  In a *supervised* ANN, the network is trained by providing matched input and output data samples, with the intention of getting the ANN to provide a desired output for a given input.  An example is an e-mail spam filter – the input training data could be the count of various words in the body of the e-mail, and the output training data would be a classification of whether the e-mail was truly spam or not.  If many examples of e-mails are passed through the neural network this allows the network to *learn* what input data makes it likely that an e-mail is spam or not.  This learning takes place be adjusting the *weights* of the ANN connections, but this will be discussed further in the next section.

*Unsupervised* learning in an ANN is an attempt to get the ANN to “understand” the structure of the provided input data “on its own”.  This type of ANN will not be discussed in this post.

## 2 The structure of an ANN

### 2.1 The artificial neuron

The **biological neuron** is simulated in an ANN by an ***activation function***. In classification tasks (e.g. identifying spam e-mails) this **activation function** has to have a “switch on” characteristic – in other words, once the input is greater than a certain value, the output should change state i.e. from 0 to 1, from -1 to 1 or from 0 to >0. This simulates the “turning on” of a **biological neuron**. A common **activation function** that is used is the sigmoid function:
$$
\begin{equation*} 
f(z) = \frac{1}{1+exp(-z)} 
\end{equation*}
$$
Which looks like this:

![img](https://i0.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/03/sigmoid.png?resize=300%2C210&ssl=1)



As can be seen in the figure above, the function is “activated” i.e. it moves from 0 to 1 when the input *x* is greater than a certain value. The sigmoid function isn’t a step function however, the edge is “soft”, and the output doesn’t change instantaneously. This means that there is a derivative（导数） of the function and this is important for the training algorithm which is discussed more in [Section 4](https://adventuresinmachinelearning.com/neural-networks-tutorial/#gradient-desc-opt).

### 2.2 Nodes

As mentioned previously, **biological neurons** are **connected hierarchical networks**, with the outputs of some neurons being the inputs to others. We can represent these networks as connected layers of *nodes.* Each node takes multiple **weighted inputs**, applies the *activation function* to the **summation** of these inputs, and in doing so generates an output. I’ll break this down further, but to help things along, consider the diagram below:

![img](https://i0.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/03/Node-diagram.jpg?resize=300%2C125&ssl=1)

Figure 2. Node with inputs

The circle in the image above represents the node. The node is the “seat” of the **activation function**, and takes the weighted inputs, sums them, then inputs them to the activation function. The output of the activation function is shown as *h* in the above diagram. Note: a *node* as I have shown above is also called a *perceptron*（感知机） in some literature.

What about this “weight” idea that has been mentioned? The **weights** are real valued numbers (i.e. not binary 1s or 0s), which are multiplied by the inputs and then summed up in the node. So, in other words, the weighted input to the node above would be:
$$
\begin{equation*} 
x_1w_1 + x_2w_2 + x_3w_3 + b 
\end{equation*}
$$
Here the $w_i$ values are weights (ignore the $b$ for the moment).  What are these weights all about?  Well, they are the variables that are changed during the learning process, and, along with the input, determine the output of the node.  The $b$ is the weight of the +1 *bias* element – the inclusion of this **bias** enhances the flexibility of the node, which is best demonstrated in an example.

### 2.3 The bias

Let’s take an extremely simple node, with only one input and one output:




![img](https://i0.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/03/Simple-node.jpg?resize=300%2C51&ssl=1)

Figure 2. Simple node

The input to the activation function of the node in this case is simply $x_1w_1$.  What does changing $w_1$ do in this simple network?

```python
w1 = 0.5
w2 = 1.0
w3 = 2.0
l1 = 'w = 0.5'
l2 = 'w = 1.0'
l3 = 'w = 2.0'
for w, l in [(w1, l1), (w2, l2), (w3, l3)]:
    f = 1 / (1 + np.exp(-x*w))
    plt.plot(x, f, label=l)
plt.xlabel('x')
plt.ylabel('h_w(x)')
plt.legend(loc=2)
plt.show()
```



![img](https://i2.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/03/Weight-adjustment-example-300x210.png?resize=300%2C210&ssl=1)

Figure 4. Effect of adjusting weights

Here we can see that changing the weight changes the slope（斜率） of the output of the **sigmoid activation function**, which is obviously useful if we want to model different strengths of relationships between the input and output variables.  However, what if we only want the output to change when x is greater than 1?  This is where the bias comes in – let’s consider the same network with a bias input:

![img](https://i2.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/03/Simple-node-with-bias.png?resize=300%2C126&ssl=1)

Figure 5. Effect of bias

 

```python
w = 5.0
b1 = -8.0
b2 = 0.0
b3 = 8.0
l1 = 'b = -8.0'
l2 = 'b = 0.0'
l3 = 'b = 8.0'
for b, l in [(b1, l1), (b2, l2), (b3, l3)]:
    f = 1 / (1 + np.exp(-(x*w+b)))
    plt.plot(x, f, label=l)
plt.xlabel('x')
plt.ylabel('h_wb(x)')
plt.legend(loc=2)
plt.show()
```



![img](https://i2.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/03/Bias-adjustment-example-300x210.png?resize=300%2C210&ssl=1)

Figure 6. Effect of bias adjusments

In this case, the $w_1$ has been increased to simulate a more defined “turn on” function.  As you can see, by varying the bias “weight” $b$, you can change when the node activates.  Therefore, by adding a bias term, you can make the node simulate a generic **if** function, i.e. *if (x > z) then 1 else 0*.  Without a bias term, you are unable to vary the *z* in that if statement, it will be always stuck around 0.  This is obviously very useful if you are trying to simulate **conditional relationships**.



### 2.4 Putting together the structure

Hopefully the previous explanations have given you a good overview of how a given node/neuron/perceptron in a neural network operates.  However, as you are probably aware, there are many such interconnected nodes in a fully fledged（成熟的） neural network.  These structures can come in a myriad of different forms, but the most common simple neural network structure consists of an *input layer*, a *hidden layer* and an *output layer*.  An example of such a structure can be seen below:

![img](https://i0.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/03/Three-layer-network.png?resize=300%2C158&ssl=1)

Figure 10. Three layer neural network

The three layers of the network can be seen in the above figure – Layer 1 represents the **input layer**, where the external input data enters the network. Layer 2 is called the **hidden layer** as this layer is not part of the input or output. Note: neural networks can have many hidden layers, but in this case for simplicity I have just included one. Finally, Layer 3 is the **output layer**. You can observe the many connections between the layers, in particular between Layer 1 (L1) and Layer 2 (L2). As can be seen, each node in L1 has a connection to all the nodes in L2. Likewise for the nodes in L2 to the single output node L3. Each of these connections will have an associated weight.

### 2.5 The notation

The maths below requires some fairly precise notation so that we know what we are talking about.  The notation I am using here is similar to that used in the Stanford deep learning tutorial.  In the upcoming equations, each of these weights are identified with the following notation: ${w_{ij}}^{(l)}$. $i$ refers to the node number of the connection in layer $l+1$ and $j$ refers to the node number of the connection in layer $l$. Take special note of this order. So, for the connection between node 1 in layer 1 and node 2 in layer 2, the weight notation would be ${w_{21}}^{(1)}$. This notation may seem a bit odd, as you would expect the *i* and *j* to refer the node numbers in layers $l$ and $l+1$ respectively (i.e. in the direction of input to output), rather than the opposite. However, this notation makes more sense when you add the bias.



As you can observe in the figure above – the (+1) bias is connected to each of the nodes in the subsequent layer. So the bias in layer 1 is connected to the all the nodes in layer two. Because the bias is not a true node with an **activation function**, it has no inputs (it always outputs the value +1). The notation of the bias weight is ${b_i}^{(l)}$, where *i* is the node number in the layer $l+1$ – the same as used for the normal weight notation ${w_{21}}^{(1)}$. So, the weight on the connection between the bias in layer 1 and the second node in layer 2 is given by ${b_2}^{(1)}$.

Remember, these values – ${w_{ji}}^{(1)}$ and ${b_i}^{(l)}$ – all need to be calculated in the training phase of the ANN.

Finally, the node output notation is ${h_j}^{(l)}$, where $j$ denotes the node number in layer $l$ of the network. As can be observed in the three layer network above, the output of node 2 in layer 2 has the notation of ${h_2}^{(2)}$.

Now that we have the notation all sorted out, it is now time to look at how you calculate the output of the network when the input and the weights are known. The process of calculating the output of the neural network given these values is called the *feed-forward* pass or process.

## 3 The feed-forward pass

To demonstrate how to calculate the output from the input in neural networks, let’s start with the specific case of the three layer neural network that was presented above. Below it is presented in equation form, then it will be demonstrated with a concrete example and some Python code:
$$
\begin{align} 
h_1^{(2)} &= f(w_{11}^{(1)}x_1 + w_{12}^{(1)} x_2 + w_{13}^{(1)} x_3 + b_1^{(1)}) \\ 
h_2^{(2)} &= f(w_{21}^{(1)}x_1 + w_{22}^{(1)} x_2 + w_{23}^{(1)} x_3 + b_2^{(1)}) \\ 
h_3^{(2)} &= f(w_{31}^{(1)}x_1 + w_{32}^{(1)} x_2 + w_{33}^{(1)} x_3 + b_3^{(1)}) \\ 
h_{W,b}(x) &= h_1^{(3)} = f(w_{11}^{(2)}h_1^{(2)} + w_{12}^{(2)} h_2^{(2)} + w_{13}^{(2)} h_3^{(2)} + b_1^{(2)}) 
\end{align}
$$
***SUMMARY*** : 
$$
\begin{align} 
h_1^{(2)} &= f(w_{11}^{(1)}x_1 + w_{12}^{(1)} x_2 + w_{13}^{(1)} x_3 + b_1^{(1)}) \\ 

\end{align}
$$
第一层节点1到第二层节点1，第一层节点2到第二层节点1，第一层节点3到第二层节点1；





In the equation above $f(\bullet)$ refers to the **node activation function**, in this case the sigmoid function. The first line, ${h_1}^{(2)}$ is the output of the first node in the second layer, and its inputs are $w_{11}^{(1)} x_1$  $w_{12}^{(1)} x_2$ $w_{13}^{(1)} x_3$  and $b_1^{(1)}$. These inputs can be traced in the three-layer connection diagram above. They are simply summed and then passed through the **activation function** to calculate the output of the first node. Likewise, for the other two nodes in the second layer.

The final line is the output of the only node in the third and final layer, which is ultimate output of the neural network. As can be observed, rather than taking the weighted input variables ($x_1, x_2, x_3$), the final node takes as input the weighted output of the nodes of the second layer ($h_{1}^{(1)}$, $h_{2}^{(2)}$, $h_{3}^{(3)}$), plus the weighted bias. Therefore, you can see in equation form the hierarchical nature of artificial neural networks.

### 3.1 A feed-forward example

Now, let’s do a simple first example of the output of this neural network in Python. First things first, notice that the weights between layer 1 and 2 ($w_{11}^{(1)}$, $w_{12}^{(1)}$,…) are ideally suited to matrix representation? Observe:
$$
\begin{equation} 
W^{(1)} = 
\begin{pmatrix} 
w_{11}^{(1)} & w_{12}^{(1)} & w_{13}^{(1)} \\ 
w_{21}^{(1)} & w_{22}^{(1)} & w_{23}^{(1)} \\ 
w_{31}^{(1)} & w_{32}^{(1)} & w_{33}^{(1)} \\ 
\end{pmatrix} 
\end{equation}
$$
This matrix can be easily represented using numpy arrays:

```python
import numpy as np
w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
```

Here I have just filled up the layer 1 weight array with some example weights. We can do the same for the layer 2 weight array:
$$
\begin{equation} 
W^{(2)} = 
\begin{pmatrix} 
w_{11}^{(2)} & w_{12}^{(2)} & w_{13}^{(2)} 
\end{pmatrix} 
\end{equation}
$$

```python
w2 = np.zeros((1, 3))
w2[0,:] = np.array([0.5, 0.5, 0.5])
```

We can also setup some dummy values in the layer 1 **bias weight** array/vector, and the layer 2 **bias weight** (which is only a single value in this neural network structure – i.e. a scalar):

```python
b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])
```



Finally, before we write the main program to calculate the output from the neural network, it’s handy to setup a separate Python function for the **activation function**:

```python
def f(x):
    return 1 / (1 + np.exp(-x))
```



### 3.2 Our first attempt at a feed-forward function

Below is a simple way of calculating the output of the neural network, using nested loops in python.  We’ll look at more efficient ways of calculating the output shortly.

```python
def simple_looped_nn_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        #Setup the input array which the weights will be multiplied by for each layer
        #If it's the first layer, the input array will be the x input vector
        #If it's not the first layer, the input to the next layer will be the 
        #output of the previous layer
        if l == 0:
            node_in = x
        else:
            node_in = h
        #Setup the output array for the nodes in layer l + 1
        h = np.zeros((w[l].shape[0],))
        #loop through the rows of the weight array
        for i in range(w[l].shape[0]):
            #setup the sum inside the activation function
            f_sum = 0
            #loop through the columns of the weight array
            for j in range(w[l].shape[1]):
                f_sum += w[l][i][j] * node_in[j]
            #add the bias
            f_sum += b[l][i]
            #finally use the activation function to calculate the
            #i-th output i.e. h1, h2, h3
            h[i] = f(f_sum)
    return h
```



This function takes as input the number of layers in the neural network, the x input array/vector, then Python tuples or lists of the weights and bias weights of the network, with each element in the tuple/list representing a layer ll in the network.  In other words, the inputs are setup in the following:

```python
w = [w1, w2]
b = [b1, b2]
#a dummy x input vector
x = [1.5, 2.0, 3.0]
```

The function first checks what the input is to the layer of nodes/weights being considered. If we are looking at the first layer, the input to the second layer nodes is the input vector xx multiplied by the relevant weights. After the first layer though, the inputs to subsequent layers are the output of the previous layers. Finally, there is a nested loop through the relevant ii and jj values of the weight vectors and the bias. The function uses the dimensions of the weights for each layer to figure out the number of nodes and therefore the structure of the network.

Calling the function:

```python
simple_looped_nn_calc(3, x, w, b)
```



gives the output of 0.8354.  We can confirm this results by manually performing the calculations in the original equations:
$$
\begin{align} 
h_1^{(2)} &= f(0.2*1.5 + 0.2*2.0 + 0.2*3.0 + 0.8) = 0.8909 \\ 
h_2^{(2)} &= f(0.4*1.5 + 0.4*2.0 + 0.4*3.0 + 0.8) = 0.9677 \\ 
h_3^{(2)} &= f(0.6*1.5 + 0.6*2.0 + 0.6*3.0 + 0.8) = 0.9909 \\ 
h_{W,b}(x) &= h_1^{(3)} = f(0.5*0.8909 + 0.5*0.9677 + 0.5*0.9909 + 0.2) = 0.8354 
\end{align}
$$

### 3.3 A more efficient implementation

As was stated earlier – using loops isn’t the most efficient way of calculating the feed forward step in Python. This is because the loops in Python are notoriously slow. An alternative, more efficient mechanism of doing the feed forward step in Python and numpy will be discussed shortly. We can benchmark how efficient the algorithm is by using the `%timeit` function in IPython, which runs the function a number of times and returns the average time that the function takes to run:

```python
%timeit simple_looped_nn_calc(3, x, w, b)
```



Running this tells us that the looped feed forward takes $40\mu s$. A result in the tens of microseconds sounds very fast, but when applied to very large practical NNs with 100s of nodes per layer, this speed will become prohibitive, especially when training the network, as will become clear later in this tutorial.  If we try a four layer neural network using the same code, we get significantly worse performance – $70\mu s$ in fact.

### 3.4 Vectorisation in neural networks

There is a way to write the equations even more compactly, and to calculate the feed forward process in neural networks more efficiently, from a computational perspective.  Firstly, we can introduce a new variable $z_{i}^{(l)}$ which is the summated input into node $i$ of layer $l$, including the bias term.  So in the case of the first node in layer 2, $z$ is equal to:
$$
z_{1}^{(2)} = w_{11}^{(1)}x_1 + w_{12}^{(1)} x_2 + w_{13}^{(1)} x_3 + b_1^{(1)} = \sum_{j=1}^{n} w_{ij}^{(1)}x_i + b_{i}^{(1)}
$$
where `n` is the number of nodes in layer 1.  Using this notation, the unwieldy previous set of equations for the example three layer network can be reduced to:

$$
\begin{align} 
z^{(2)} &= W^{(1)} x + b^{(1)} \\ 
h^{(2)} &= f(z^{(2)}) \\ 
z^{(3)} &= W^{(2)} h^{(2)} + b^{(2)} \\ 
h_{W,b}(x) &= h^{(3)} = f(z^{(3)}) 
\end{align}
$$






## 4 Gradient descent and optimisation

### 4.1 A simple example in code

Below is an example of a simple Python implementation of gradient descent for solving the minimum of the equation $f(x) = x^4 – 3x^3 + 2$ taken from [Wikipedia.](https://en.wikipedia.org/wiki/Gradient_descent)  The gradient of this function is able to be calculated analytically (i.e. we can do it easily using calculus, which we can’t do with many real world applications) and is $f'(x) = 4x^3 – 9x^2$. This means at every value of xx, we can calculate the gradient of the function by using a simple equation. Again, using calculus we can know that the exact minimum of this equation is $x=2.25$.



Previously, we’ve talked about iteratively minimising the error of the output of the neural network by varying the **weights** in **gradient descent**. However, as it turns out, there is a mathematically more generalised way of looking at things that allows us to reduce the error while also preventing things like *overfitting* (this will be discussed more in later articles). This more general optimisation formulation revolves around minimising what’s called the **cost function**. The equivalent cost function of a single training pair ($x^z$, $y^z$) in a neural network is:

```python
x_old = 0 # The value does not matter as long as abs(x_new - x_old) > precision
x_new = 6 # The algorithm starts at x=6
gamma = 0.01 # step size
precision = 0.00001

def df(x):
    y = 4 * x**3 - 9 * x**2
    return y

while abs(x_new - x_old) > precision:
    x_old = x_new
    x_new += -gamma * df(x_old)

print("The local minimum occurs at %f" % x_new)
```



This function prints “The local minimum occurs at 2.249965”, which agrees with the exact solution within the precision.  This code implements the **weight adjustment algorithm** that I showed above, and can be seen to find the minimum of the function correctly within the given precision. This is a very simple example of **gradient descent**, and finding the gradient works quite differently when training **neural networks**. However, the main idea remains – we figure out the **gradient** of the neural network then **adjust** the **weights** in a step to try to get closer to the **minimum error** that we are trying to find. Another difference between this toy example of gradient descent is that the **weight vector** is multi-dimensional, and therefore the gradient descent method must search a multi-dimensional space for the minimum point.



The way we figure out the gradient of a neural network is via the famous ***backpropagation*** method, which will be discussed shortly. First however, we have to look at the error function more closely.



### 4.2 The cost function



### 4.3 Gradient descent in neural networks

Gradient descent for every weight $w_{(ij)}^{(l)}$ and every bias $b_i^{(l)}$ in the neural network looks like the following:
$$
\begin{align} 
w_{ij}^{(l)} &= w_{ij}^{(l)} – \alpha \frac{\partial}{\partial w_{ij}^{(l)}} J(w,b) \\ 
b_{i}^{(l)} &= b_{i}^{(l)} – \alpha \frac{\partial}{\partial b_{i}^{(l)}} J(w,b) 
\end{align}
$$
Basically, the equation above is similiar to the previously shown gradient descent algorithm: $w_{new} = w_{old} – \alpha * \nabla error$. The new and old subscripts are missing, but the values on the left side of the equation are *new* and the values on the right side are *old*. Again, we have an iterative process whereby the weights are updated in each iteration, this time based on the cost function $J(w,b)$.

The values $\frac{\partial}{\partial w_{ij}^{(l)}}$ and $\frac{\partial}{\partial b_{i}^{(l)}}$ are the *partial derivatives* of the single sample cost function based on the weight values. What does this mean? Recall that for the simple gradient descent example mentioned previously, each step depends on the *slope* of the error/cost term with respect to the weights. Another word for slope or gradient is the *derivative*. A normal derivative has the notation $\frac{d}{dx}$. If $x$ in this instance is a vector, then such a derivative will also be a vector, displaying the gradient in all the dimensions of $x$.

### 4.4 A two dimensional gradient descent example





### 4.5 Backpropagation in depth



In this section, I’m going to delve into the maths a little. If you’re wary of the maths of how **backpropagation** works, then it may be best to skip this section.  The next [section](https://adventuresinmachinelearning.com/neural-networks-tutorial/#implementing-nn) will show you how to implement **backpropagation** in code – so if you want to skip straight on to using this method, feel free to skip the rest of this section. However, if you don’t mind a little bit of maths, I encourage you to push on to the end of this section as it will give you a good depth of understanding in **training** neural networks. This will be invaluable to understanding some of the key **ideas** in deep learning, rather than just being a code cruncher who doesn’t really understand how the code works.

***SUMMARY*** : neural network的training；

First let’s recall some of the foundational equations from [Section 3](https://adventuresinmachinelearning.com/neural-networks-tutorial/#the-feed-forward-pass) for the following three layer neural network:

![img](https://i0.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/03/Three-layer-network.png?resize=300%2C158&ssl=1)

Figure 10. Three layer neural network (again)

The output of this neural network can be calculated by:
$$
\begin{equation} 
h_{W,b}(x) = h_1^{(3)} = f(w_{11}^{(2)}h_1^{(2)} + w_{12}^{(2)} h_2^{(2)} + w_{13}^{(2)} h_3^{(2)} + b_1^{(2)}) 
\end{equation}
$$
We can also simplify the above to $h_1^{(3)} = f(z_1^{(2)})$ by defining $z_1^{(2)}$ as:
$$
z_{1}^{(2)} = w_{11}^{(2)}h_1^{(2)} + w_{12}^{(2)} h_2^{(2)} + w_{13}^{(2)} h_3^{(2)} + b_1^{(2)}
$$
***SUMMARY*** : $z_1^{(2)}$表示的是第2层输入到第2+1层的第1个节点的sum；

Let’s say we want to find out how much a **change** in the weight $w_{12}^{(2)}$ has on the **cost function** $J$. This is to evaluate $\frac {\partial J}{\partial w_{12}^{(2)}}$. To do so, we have to use something called the chain function:
$$
\frac {\partial J}{\partial w_{12}^{(2)}} = \frac {\partial J}{\partial h_1^{(3)}} \frac {\partial h_1^{(3)}}{\partial z_1^{(2)}} \frac {\partial z_1^{(2)}}{\partial w_{12}^{(2)}}
$$
***SUMMARY*** : 如果有多层隐藏层的话，上述公式是否依然适用？



If you look at the terms on the right – the numerators “cancel out” the denominators, in the same way that $\frac {2}{5} \frac {5}{2} = \frac {2}{2} = 1$. Therefore we can construct $\frac {\partial J}{\partial w_{12}^{(2)}}$ by stringing together a few **partial derivatives** (which are quite easy, thankfully). Let’s start with $\frac {\partial z_1^{(2)}}{\partial w_{12}^{(2)}}$
$$
\begin{align} 
\frac {\partial z_1^{(2)}}{\partial w_{12}^{(2)}} &= \frac {\partial}{\partial w_{12}^{(2)}} (w_{11}^{(1)}h_1^{(2)} + w_{12}^{(1)} h_2^{(2)} + w_{13}^{(1)} h_3^{(2)} + b_1^{(1)})\\ 
&= \frac {\partial}{\partial w_{12}^{(2)}} (w_{12}^{(1)} h_2^{(2)})\\ 
&= h_2^{(2)} 
\end{align}
$$
The partial derivative of $z_1^{(2)}$ with respect $w_{12}^{(2)}$ only operates on one term within the parentheses, $w_{12}^{(1)} h_2^{(2)}$, as all the other terms don’t vary at all when $w_{12}^{(2)}$ does. The derivative of a constant is 1, therefore $\frac {\partial}{\partial w_{12}^{(2)}} (w_{12}^{(1)} h_2^{(2)})$ collapses to just $h_2^{(2)}$, which is simply the output of the second node in layer 2.

The next partial derivative in the chain is $\frac {\partial h_1^{(3)}}{\partial z_1^{(2)}}$, which is the **partial derivative** of the **activation function** of the $h_1^{(3)}$ **output node**. Because of the requirement to be able to derive this derivative, the activation functions in neural networks need to be *differentiable*. For the common sigmoid activation function (shown in [Section 2.1](https://adventuresinmachinelearning.com/neural-networks-tutorial/#the-artificial-neuron)), the derivative is:
$$
\frac {\partial h}{\partial z} = f'(z) = f(z)(1-f(z))
$$
Where $f(z)$ is the activation function. So far so good – now we have to work out how to deal with the first term $\frac {\partial J}{\partial h_1^{(3)}}$. Remember that $J(w,b,x,y)$ is the mean squared error loss function, which looks like (for our case):
$$
J(w,b,x,y) = \frac{1}{2} \parallel y_1 – h_1^{(3)}(z_1^{(2)}) \parallel ^2
$$
Here $y_1$ is the training target for the **output node**. Again using the chain rule:
$$
\begin{align} 
&Let\ u = \parallel y_1 – h_1^{(3)}(z_1^{(2)}) \parallel\ and\ J = \frac {1}{2} u^2\\ 
&Using\ \frac {\partial J}{\partial h} = \frac {\partial J}{\partial u} \frac {\partial u}{\partial h}:\\ 
&\frac {\partial J}{\partial h} = -(y_1 – h_1^{(3)}) 
\end{align}
$$
***SUMMARY*** : 上述推导并没有搞清楚；

So we’ve now figured out how to calculate $\frac {\partial J}{\partial w_{12}^{(2)}}$, at least for the **weights** connecting the **output layer**. 



Before we move to any **hidden layers** (i.e. layer 2 in our example case), let’s introduce some simplifications to tighten up our notation and introduce $\delta$:
$$
\delta_i^{(n_l)} = -(y_i – h_i^{(n_l)})\cdot f^\prime(z_i^{(n_l)})
$$
Where $i$ is the node number of the **output layer**. 

***SUMMARY*** : $\delta_i^{(n_l)}$中包含$z_i^{(n_l)}$

In our selected example there is only one such layer, therefore $i=1$ always in this case. Now we can write the complete cost function derivative as:
$$
\begin{align} 
\frac{\partial}{\partial W_{ij}^{(l)}} J(W,b,x, y) &= h^{(l)}_j \delta_i^{(l+1)} \\ 
\end{align}
$$


Where, for the output layer in our case, $l$ = 2 and $i$ remains the node number.



***SUMMARY*** : 这是神经网络中的链式法则的通用形式；以开头所提出的问题：Let’s say we want to find out how much a **change** in the weight $w_{12}^{(2)}$ has on the **cost function** $J$. This is to evaluate $\frac {\partial J}{\partial w_{12}^{(2)}}$. 

使用上述公式来看的话，$\frac {\partial J}{\partial w_{12}^{(2)}}= h^{(2)}_2 \delta_1^{(2+1)}$. $h^{(2)}_2$表示第2层第2个节点的输出值；$\delta_1^{(2+1)}$，$w_{12}^{(2)}$是第2层的节点2与第三层的节点1之间的连接权重；



### 4.6 Propagating into the hidden layers

What about for weights feeding into any **hidden layers** (layer 2 in our case)? For the **weights** connecting the **output layer**, the $\frac {\partial J}{\partial h} = -(y_i – h_i^{(n_l)})$ derivative made sense, as the **cost function** can be directly calculated by comparing the **output layer** to the **training data**. The output of the hidden nodes, however, have no such direct reference, rather, they are connected to the **cost function** only through mediating **weights** and potentially other layers of nodes. How can we find the variation in the **cost function** from changes to **weights** embedded deep within the neural network? As mentioned previously, we use the *backpropagation* method.

***SUMMARY*** : 上面这段话的意思是如何来计算调整输入到hidden layer的weight对cost function的影响；如果以上述三层神经网络为例的话，此处的weight就是input layer到hidden layer的weight；对于这种weigh而言，因为它们并不是直接连接到输出层，所以无法直接使用comparing the **output layer** to the **training data**. 那如何来计算那些改变那些不是直接连接到输出层而是嵌入地比较深的层的节点的权重对neural network的影响呢？此处就需要使用backpropagation；



***SUMMARY*** : 上述给出了backpropagation的价值所在

Now that we’ve done the hard work using the **chain rule**, we’ll now take a more graphical approach. The term that needs to propagate back through the network is the $\delta_i^{(n_l)}$ term, as this is the network’s ultimate connection to the **cost function**. What about node `j` in the second layer (hidden layer)? How does it contribute to $\delta_i^{(n_l)}$ in our test network? It contributes via the weight $w_{ij}^{(2)}$ – see the diagram below for the case of $j=1$ and $i=1$.

![img](https://i0.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/03/Backpropagation-illustration.jpg?resize=300%2C204&ssl=1)

Figure 11. Simple backpropagation illustration

***SUMMARY*** : 上图中，比较模糊的公式是：
$$
\delta_i^{(2)} = \delta_i^{(3)} w_{11}^{(2)}
$$
As can be observed from above, the output layer $δ$ is *communicated* to the hidden node by the weight of the connection. In the case where there is only one output layer node, the generalised hidden layer $δ$ is defined as:
$$
\delta_j^{(l)} = \delta_1^{(l+1)} w_{1j}^{(l)}\ f^\prime(z_j)^{(l)}
$$
***SUMMARY*** : 这是一个递归关系，问题是这个递归关系是如何得到的？

Where $j$ is the node number in layer $l$. What about the case where there are multiple output nodes? In this case, the weighted sum of all the communicated errors are taken to calculate $\delta_j^{(l)}$, as shown in the diagram below:

![img](https://i0.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/03/Backpropagation-illustration-with-multiple-outputs.jpg?resize=300%2C242&ssl=1)

Figure 12. Backpropagation illustration with multiple outputs

***SUMMARY*** : 上图中，比较模糊的公式是：
$$
\delta_1^{(2)} = (\sum_{i=1}^{3} w_{i1}^{(2)} \delta_i^{(3)})\ f^\prime(z_i^{(2)})
$$
As can be observed from the above, each $\delta$ value from the output layer is included in the sum used to calculate $\delta_1^{(2)}$, but each output  $\delta$  is weighted according to the appropriate $w_{i1}^{(2)}$ value. In other words, node 1 in layer 2 contributes to the error of three output nodes, therefore the **measured error** (or **cost function value**) at each of these nodes has to be “**passed back**”（传回） to the $\delta$ value for this node. Now we can develop a generalised expression for the  $\delta$  values for nodes in the hidden layers:
$$
\delta_j^{(l)} = (\sum_{i=1}^{s_{(l+1)}} w_{ij}^{(l)} \delta_i^{(l+1)})\ f^\prime(z_j^{(l)})
$$
Where $j$ is the node number in layer $l$ and $i$ is the node number in layer $l+1$(which is the same notation we have used from the start). The value $s_{(l+1)}$ is the number of nodes in layer $(l+1)$.

***SUMMARY*** : 上式是递归公式，在Wikipedia的[Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)有相应的介绍；

So we now know how to calculate:
$$
\frac{\partial}{\partial W_{ij}^{(l)}} J(W,b,x, y) = h^{(l)}_j \delta_i^{(l+1)}
$$
***SUMMARY*** : 根据该公式就可以得到调整任意的$W_{ij}^{(l)}$对cost function的影响；

as shown previously. What about the **bias weights**? I’m not going to derive them as I did with the normal weights in the interest of saving time / space. However, the reader shouldn’t have too many issues following the same steps, using the chain rule, to arrive at:
$$
\frac{\partial}{\partial b_{i}^{(l)}} J(W,b,x, y) = \delta_i^{(l+1)}
$$
Great – so we now know how to perform our original **gradient descent problem** for neural networks:
$$
\begin{align} 
w_{ij}^{(l)} &= w_{ij}^{(l)} – \alpha \frac{\partial}{\partial w_{ij}^{(l)}} J(w,b) \\ 
b_{i}^{(l)} &= b_{i}^{(l)} – \alpha \frac{\partial}{\partial b_{i}^{(l)}} J(w,b) 
\end{align}
$$
However, to perform this gradient descent training of the weights, we would have to resort to loops within loops. As previously shown in [Section 3.4](https://adventuresinmachinelearning.com/neural-networks-tutorial/#vectorisation) of this neural network tutorial, performing such calculations in Python using loops is slow for large networks. Therefore, we need to figure out how to vectorise such calculations, which the next section will show.



### 4.8 Implementing the gradient descent step





### 4.9 The final gradient descent algorithm



## 5 Implementing the neural network in Python