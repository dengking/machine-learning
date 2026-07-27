# 领域知识

在[Convolutional Neural Networks (CNNs / ConvNets)](http://cs231n.github.io/convolutional-networks/)中提及：

> ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the **forward function** more efficient to implement and vastly reduce the amount of **parameters** in the network.

利用领域知识来对模型进行改进；无论是CNN还是RNN，都运用了领域相关的知识对模型进行了改进，因此它们能够非常好的解决它的目标领域的问题；

# parameter sharing在neural network中的运用

- CNN

  参见[Convolutional Neural Networks (CNNs / ConvNets)](http://cs231n.github.io/convolutional-networks/)

- RNN

  参见deep learning book：Chapter 10 Sequence Modeling: Recurrent and Recursive Nets

在deep learning book：Chapter 10 Sequence Modeling: Recurrent and Recursive Nets的导语中对比了CNN和RNN在parameter sharing中的差异；

> A related idea is the use of convolution across a 1-D temporal sequence. This convolutional approach is the basis for time-delay neural networks (Lang and Hinton 1988 Waibel 1989 Lang 1990 , ; et al., ; et al., ). The convolution operation allows a network to share parameters across time, but is shallow. The output of convolution is a sequence where each member of the output is a function of a small number of neighboring members of the input. The idea of parameter sharing manifests in the application of the same convolution kernel at each time step. Recurrent networks share parameters in a different way. Each member of the output is a function of the previous members of the output. Each member of the output is produced using the same update rule applied to the previous outputs. This recurrent formulation results in the sharing of parameters through a very deep computational graph.

