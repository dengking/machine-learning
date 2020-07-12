convolutional network  grid of values X such as an image

recurrent neural network is a neural network that is specialized for processing a sequence of values x (1) ,...,x ( )

***SUMMARY***:专注于不同领域的模型

***thinking***: 那不同的领域要选择怎样的cost function才合适呢？

**sharing parameters** across different parts of a model. **Parameter sharing** makes it possible to extend and apply the model to examples of different forms (different lengths, here) and generalize across them. If we had separate parameters for each value of the time index（这是MLP的做法）, we could not generalize to sequence lengths not seen during training, nor share statistical strength across different sequence lengths and across different positions in time. 

Such sharing is particularly important when a specific piece of information can occur at multiple positions within the sequence. For example, consider the two sentences “I went to Nepal in 2009” and “In 2009, I went to Nepal.” If we ask a machine learning model to read each sentence and extract the year in which the narrator went to Nepal, we would like it to recognize the year 2009 as the relevant piece of information, whether it appears in the sixth word or the second word of the sentence. Suppose that we trained a feedforward network that processes sentences of fixed length. A traditional fully connected feedforward network would have separate parameters for each input feature, so it would need to learn all of the rules of the language separately at each position in the sentence. By comparison, a recurrent neural network shares the same weights across several time steps.



***SUMMARY***: RNN的一个重要思想，以及为什么这样做；

***SUMMARY*** : RNN与传统的feedforward network之间的差异；



**Recurrent networks** share parameters in a different way. Each member of the output is a function of the previous members of the output. Each member of the output is produced using the same update rule applied to the previous outputs. This recurrent formulation results in the sharing of parameters through a very deep computational graph.

This chapter extends the idea of a **computational graph** to include cycles. These cycles represent the influence of the present value of a variable on its own value at a future time step. Such computational graphs allow us to define recurrent neural networks. We then describe many different ways to construct, train, and use recurrent neural networks.

