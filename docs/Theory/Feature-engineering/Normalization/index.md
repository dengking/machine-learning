# Normalization

normalization: 归一化、正规化、标准化

读物:

- medium [Feature Scaling- Why it is required?](https://rahul-saini.medium.com/feature-scaling-why-it-is-required-8a93df1af310) 

- csdn [Normalization](https://blog.csdn.net/gwplovekimi/article/details/84647354) 

## wikipedia [Normalization (machine learning)](https://en.wikipedia.org/wiki/Normalization_(machine_learning))

In [machine learning](https://en.wikipedia.org/wiki/Machine_learning "Machine learning"), **normalization** is a statistical technique with various applications. There are two main forms of normalization, namely *data normalization* and *activation normalization*. 

Data normalization (or [feature scaling](https://en.wikipedia.org/wiki/Feature_scaling "Feature scaling")) includes methods that rescale input data so that the [features](https://en.wikipedia.org/wiki/Feature_(machine_learning) "Feature (machine learning)") have the same range, mean, variance, or other statistical properties. For instance, a popular choice of feature scaling method is [min-max normalization](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization) "Feature scaling"), where each feature is transformed to have the same range (typically $[0,1]$ or $[-1,1]$.  This solves the problem of different features having vastly different scales(单位), for example if one feature is measured in kilometers and another in nanometers.

Activation normalization, on the other hand, is specific to [deep learning](https://en.wikipedia.org/wiki/Deep_learning "Deep learning"), and includes methods that rescale the activation of [hidden neurons](https://en.wikipedia.org/wiki/Hidden_layer "Hidden layer") inside [neural networks](https://en.wikipedia.org/wiki/Neural_network_(machine_learning) "Neural network (machine learning)").

Normalization is often used to:

- increase the speed of training convergence,
- reduce sensitivity to variations and feature scales in input data,
- reduce [overfitting](https://en.wikipedia.org/wiki/Overfitting "Overfitting"),
- and produce better model generalization to unseen data.

Normalization techniques are often theoretically justified as reducing covariance shift, smoothing optimization landscapes, and increasing [regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics) "Regularization (mathematics)"), though they are mainly justified by empirical success.

> **翻译**: 归一化技术在理论上常被认为可减少协方差偏移、平滑优化地形并增强正则化效果，尽管其合理性主要源于实证上的成功。
> 
> **解析**：
> 
> - **Normalization techniques**：归一化技术（如 Batch Normalization、Layer Normalization 等）。
> - **covariance shift**：协方差偏移，指训练和测试数据的特征分布差异，归一化可通过标准化数据缓解该问题。
> - **smoothing optimization landscapes**：平滑优化地形，即改善目标函数的曲面形态，减少局部极小值和鞍点对收敛的阻碍。
> - **empirical success**：实证成功，指在实际应用中通过实验验证效果，而非仅依赖理论推导。  
>   该句强调归一化技术虽有理论支撑，但更关键的是其在实际任务中的有效性。

## wikipedia [Normalization (statistics)](https://en.wikipedia.org/wiki/Normalization_(statistics))

## wikipedia [Feature scaling](https://en.wikipedia.org/wiki/Feature_scaling)

**Feature scaling** is a method used to normalize(正规化，标准化) the range of independent variables or features of data. In [data processing](https://en.wikipedia.org/wiki/Data_processing), it is also known as **data normalization** and is generally performed during the data preprocessing step.

### Motivation

Since the range of values of raw data varies widely, in some [machine learning](https://en.wikipedia.org/wiki/Machine_learning) algorithms, **objective functions** will not work properly without [normalization](https://en.wikipedia.org/wiki/Normalization_(statistics)). For example, many [classifiers](https://en.wikipedia.org/wiki/Statistical_classification) calculate the distance between two points by the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance). If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.

Another reason why **feature scaling** is applied is that [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) converges much faster with feature scaling than without it.[[1\]](https://en.wikipedia.org/wiki/Feature_scaling#cite_note-1)

### Methods

#### Rescaling (min-max normalization)

Also known as min-max scaling or min-max normalization, is the simplest method and consists in rescaling the range of features to scale the range in [0, 1] or [−1, 1]. Selecting the target range depends on the nature of the data. The general formula for a min-max of [0, 1] is given as:

$ x'={\frac {x-{\text{min}}(x)}{{\text{max}}(x)-{\text{min}}(x)}} $

where $ x $ is an original value, $ x' $ is the normalized value. For example, suppose that we have the students' weight data, and the students' weights span [160 pounds, 200 pounds]. To rescale this data, we first subtract 160 from each student's weight and divide the result by 40 (the difference between the maximum and minimum weights).

To rescale a range between an arbitrary set of values [a, b], the formula becomes:

$ x'=a+{\frac {(x-{\text{min}}(x))(b-a)}{{\text{max}}(x)-{\text{min}}(x)}} $

where $ a,b $ are the min-max values.

#### Mean normalization

$ x'={\frac {x-{\text{average}}(x)}{{\text{max}}(x)-{\text{min}}(x)}} $

where $ x $ is an original value, $ x' $ is the normalized value. There is another form of the mean normalization which is when we divide by the standard deviation which is also called standardization.

#### Standardization (Z-score Normalization)

In machine learning, we can handle various types of data, e.g. audio signals and pixel values for image data, and this data can include multiple [dimensions](https://en.wikipedia.org/wiki/Dimensions). Feature standardization makes the values of each feature in the data have **zero-mean** (when subtracting the mean in the numerator) and **unit-variance**. This method is widely used for normalization in many machine learning algorithms (e.g., [support vector machines](https://en.wikipedia.org/wiki/Support_vector_machine), [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression), and [artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network))[[2\]](https://en.wikipedia.org/wiki/Feature_scaling#cite_note-:0-2)[*citation needed*]. The general method of calculation is to determine the distribution [mean](https://en.wikipedia.org/wiki/Mean) and [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation) for each feature. Next we subtract the mean from each feature. Then we divide the values (mean is already subtracted) of each feature by its standard deviation（标准差）.

$ x'={\frac {x-{\bar {x}}}{\sigma }} $

Where $ x $ is the original feature vector, $ {\bar {x}}={\text{average}}(x) $ is the mean of that feature vector, and $ \sigma $ is its standard deviation.

#### Scaling to unit length

Another option that is widely used in machine-learning is to scale the components of a feature vector such that the complete vector has length one. This usually means dividing each component by the [Euclidean length](https://en.wikipedia.org/wiki/Euclidean_length) of the vector:

$ x'={\frac {x}{\left\|{x}\right\|}} $

In some applications (e.g. Histogram features) it can be more practical to use the L1 norm (i.e. Manhattan Distance, City-Block Length or [Taxicab Geometry](https://en.wikipedia.org/wiki/Taxicab_Geometry)) of the feature vector. This is especially important if in the following learning steps the Scalar Metric is used as a distance measure.

### Application

In stochastic [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), feature scaling can sometimes improve the convergence speed of the algorithm[[2\]](https://en.wikipedia.org/wiki/Feature_scaling#cite_note-:0-2)[*citation needed*]. In support vector machines,[[3\]](https://en.wikipedia.org/wiki/Feature_scaling#cite_note-3) it can reduce the time to find support vectors. Note that **feature scaling** changes the SVM result[*citation needed*].
