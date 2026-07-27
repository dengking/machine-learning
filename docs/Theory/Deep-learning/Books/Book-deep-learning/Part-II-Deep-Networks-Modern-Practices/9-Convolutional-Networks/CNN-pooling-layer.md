https://stackoverflow.com/questions/53743729/when-to-use-globalaveragepooling1d-and-when-to-use-globalmaxpooling1d-while-usin





# [Max-pooling / Pooling](https://computersciencewiki.org/index.php/Max-pooling_/_Pooling#cite_note-2)

## Introduction

 Max pooling is a **sample-based discretization process**（基于样例的降采样过程）. The objective is to **down-sample** an input representation (image, hidden-layer output matrix, etc.), reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned.[[2\]](https://computersciencewiki.org/index.php/Max-pooling_/_Pooling#cite_note-2) 

## How does it work and why

This is done to in part to help over-fitting by providing an abstracted form of the representation. As well, it reduces the computational cost by reducing the number of parameters to learn and provides basic **translation invariance** to the internal representation.

***SUMMARY*** : **translation invariance**是CNN的一个非常重要的特性

**Max pooling** is done by applying a max filter to (usually) non-overlapping subregions of the initial representation.

## Examples

Let's say we have a 4x4 matrix representing our initial input. Let's say, as well, that we have a 2x2 filter that we'll run over our input. We'll have a **stride**（跨度） of 2 (meaning the (dx, dy) for stepping over our input will be (2, 2)) and won't overlap regions.

For each of the regions represented by the filter, we will take the **max** of that region and create a new, output matrix where each element is the max of a region in the original input.

Pictorial representation: [![MaxpoolSample2.png](https://computersciencewiki.org/images/8/8a/MaxpoolSample2.png)](https://computersciencewiki.org/index.php/File:MaxpoolSample2.png)

Real-life example: [![MaxpoolSample.png](https://computersciencewiki.org/images/9/9e/MaxpoolSample.png)](https://computersciencewiki.org/index.php/File:MaxpoolSample.png)

## References

1.  http://www.flaticon.com/
2. [Jump up↑](https://computersciencewiki.org/index.php/Max-pooling_/_Pooling#cite_ref-2) https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks



## [A Gentle Introduction to Pooling Layers for Convolutional Neural Networks](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)



[Convolutional layers](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/) in a convolutional neural network summarize the presence of features in an input image.

A problem with the **output feature maps** is that they are sensitive to the **location** of the features in the input. One approach to address this **sensitivity** is to **down sample** the **feature maps**. This has the effect of making the resulting **down sampled feature maps** more robust to changes in the position of the feature in the image, referred to by the technical phrase “*local translation invariance*.”

Pooling layers provide an approach to down sampling feature maps by summarizing the presence of features in patches of the feature map. Two common pooling methods are **average pooling** and **max pooling** that summarize the **average presence of a feature** and the **most activated presence of a feature** respectively.

In this tutorial, you will discover how the **pooling operation** works and how to implement it in **convolutional neural networks**.

After completing this tutorial, you will know:

- Pooling is required to down sample the detection of features in feature maps.
- How to calculate and implement average and maximum pooling in a convolutional neural network.
- How to use **global pooling** in a convolutional neural network.

Discover how to build models for photo classification, object detection, face recognition, and more [in my new computer vision book](https://machinelearningmastery.com/deep-learning-for-computer-vision/), with 30 step-by-step tutorials and full source code.

Let’s get started.

