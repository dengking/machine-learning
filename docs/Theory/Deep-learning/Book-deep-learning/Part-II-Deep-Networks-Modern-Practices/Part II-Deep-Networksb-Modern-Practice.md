# Part II Deep Networks: Modern Practices

These less-developed branches of deep learning appear in the final part of the book. This part focuses only on those approaches that are essentially working technologies that are already used heavily in industry.

> NOTE: Part II介绍的technology是相对比较成熟的，而part III则是less-developed branches 

Modern deep learning provides a very powerful framework for **supervised learning**. By adding more layers and more units within a layer, a deep network can represent **functions** of increasing complexity. Most tasks that consist of mapping an **input vector** to an **output vector**, and that are easy for a person to do rapidly, can be accomplished via **deep learning**, given sufficiently large models and sufficiently large datasets of labeled training examples. Other tasks, that can not be described as associating one vector to another, or that are difficult enough that a person would require time to think and reflect in order to accomplish the task, remain beyond the scope of deep learning for now.

This part of the book describes the core **parametric function approximation**（参数化函数近似）technology that is behind nearly all modern practical applications of deep learning. 

> NOTE: function、mapping an **input vector** to an **output vector**、**supervised learning** 、statistical model，它们是同义词。
>
> 从更高的角度来看，我们所学习的deep learning技术，本质上是：**parametric function approximation**

Next, we present advanced techniques for **regularization** and **optimization** of such models. 

Scaling these models to large inputs such as high resolution images or long temporal sequences requires **specialization**. We introduce the **convolutional network** for scaling to large images and the **recurrent neural network** for processing temporal sequences. 

> NOTE: 随着deep learning技术的发展，出现了越来越多的specialization。

Finally, we present general guidelines for the practical methodology involved in designing, building, and configuring an application involving deep learning, and review some of the applications of deep learning.