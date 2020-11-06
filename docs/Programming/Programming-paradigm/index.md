# Programming model/paradigm



## Symbolic VS imperative

在`./Symbolic-and-imperative`中进行了总结。



## TensorFlow Programming Model

在[TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](http://download.tensorflow.org/paper/whitepaper2015.pdf)中对TensorFlow的programming model进行了描述。



## 经验总结



### Rule one: separation of model and dataset

将model和dataset分隔开来

### Use trainer to connect dataset and model







## Feedforward and feedback
前馈过程：计算得到loss

反馈过程：backpropagation to compute gradient and then adjust parameter；

