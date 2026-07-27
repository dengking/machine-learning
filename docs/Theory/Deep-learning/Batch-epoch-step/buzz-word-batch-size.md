# stackexchange [What is batch size in neural network?](https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network)

I'm using `Python Keras package` for neural network. This is the [link](http://keras.io/models/). Is `batch_size` equals to number of test samples? From Wikipedia we have [this](http://en.wikipedia.org/wiki/Stochastic_gradient_descent#cite_note-2) information:

> However, in other cases, evaluating the sum-gradient may require expensive evaluations of the gradients from all summand functions. When the training set is enormous and no simple formulas exist, evaluating the sums of gradients becomes very expensive, because evaluating the gradient requires evaluating all the summand functions' gradients. To economize on the computational cost at every iteration, stochastic gradient descent samples a subset of summand functions at every step. This is very effective in the case of large-scale machine learning problems.

Above information is describing test data? Is this same as `batch_size` in keras (Number of samples per gradient update)?



## [A](https://stats.stackexchange.com/a/153535)

The **batch size** defines the number of samples that will be propagated through the network.

For instance, let's say you have 1050 training samples and you want to set up a `batch_size` equal to 100. The algorithm takes the first 100 samples (from 1st to 100th) from the training dataset and trains the network. Next, it takes the second 100 samples (from 101st to 200th) and trains the network again. We can keep doing this procedure until we have propagated all samples through of the network. Problem might happen with the last set of samples. In our example, we've used 1050 which is not divisible by 100 without remainder. The simplest solution is just to get the final 50 samples and train the network.

Advantages of using a batch size < number of all samples:

- It requires less memory. Since you train the network using fewer samples, the overall training procedure requires less memory. That's especially important if you are not able to fit the whole dataset in your machine's memory.
- Typically networks train faster with mini-batches. That's because we update the weights after each propagation. In our example we've propagated 11 batches (10 of them had 100 samples and 1 had 50 samples) and after each of them we've updated our network's parameters. If we used all samples during propagation we would make only 1 update for the network's parameter.

Disadvantages of using a batch size < number of all samples:

- The smaller the batch the less accurate the estimate of the gradient will be. In the figure below, you can see that the direction of the mini-batch gradient (green color) fluctuates much more in comparison to the direction of the full batch gradient (blue color).

[![Gradient directions for different batch setups](https://i.stack.imgur.com/lU3sx.png)](https://i.stack.imgur.com/lU3sx.png)

Stochastic is just a mini-batch with `batch_size` equal to 1. In that case, the gradient changes its direction even more often than a mini-batch gradient.