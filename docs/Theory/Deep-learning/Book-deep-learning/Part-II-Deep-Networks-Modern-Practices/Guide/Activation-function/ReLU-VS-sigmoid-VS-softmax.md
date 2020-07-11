# [Softmax vs Sigmoid function in Logistic classifier?](https://stats.stackexchange.com/questions/233658/softmax-vs-sigmoid-function-in-logistic-classifier)

What decides the choice of function ( Softmax vs Sigmoid ) in a Logistic classifier ?

Suppose there are 4 output classes . Each of the above function gives the probabilities of each class being the correct output . So which one to take for a classifier ?



## [A](https://stats.stackexchange.com/a/254071)

The [sigmoid function](https://en.wikipedia.org/wiki/Logistic_function) is used for the **two-class logistic regression**, whereas the [softmax function](https://en.wikipedia.org/wiki/Softmax_function) is used for the **multiclass logistic regression** (a.k.a. MaxEnt, multinomial logistic regression, softmax Regression, Maximum Entropy Classifier).

In the two-class **logistic regression**, the predicted probablies are as follows, using the sigmoid function:
$$
\begin{align}
\Pr(Y_i=0) &= \frac{e^{-\boldsymbol\beta_ \cdot \mathbf{X}_i}} {1 +e^{-\boldsymbol\beta_0 \cdot \mathbf{X}_i}} \, \\
\Pr(Y_i=1) &= 1 - \Pr(Y_i=0) = \frac{1} {1 +e^{-\boldsymbol\beta_ \cdot \mathbf{X}_i}}
\end{align}
$$
In the multiclass logistic regression, with KK classes, the predicted probabilities are as follows, using the softmax function:
$$
\begin{align}
\Pr(Y_i=k) &= \frac{e^{\boldsymbol\beta_k \cdot \mathbf{X}_i}} {~\sum_{0 \leq c \leq K}^{}{e^{\boldsymbol\beta_c \cdot \mathbf{X}_i}}} \, \\
\end{align}
$$
One can observe that the `softmax` function is an extension of the `sigmoid` function to the multiclass case, as explained below. Let's look at the multiclass logistic regression, with $K=2$ classes:


$$
\begin{align}
\Pr(Y_i=0) &= \frac{e^{\boldsymbol\beta_0 \cdot \mathbf{X}_i}} {~\sum_{0 \leq c \leq K}^{}{e^{\boldsymbol\beta_c \cdot \mathbf{X}_i}}} = \frac{e^{\boldsymbol\beta_0 \cdot \mathbf{X}_i}}{e^{\boldsymbol\beta_0 \cdot \mathbf{X}_i} + e^{\boldsymbol\beta_1 \cdot \mathbf{X}_i}} = \frac{e^{(\boldsymbol\beta_0 - \boldsymbol\beta_1) \cdot \mathbf{X}_i}}{e^{(\boldsymbol\beta_0 - \boldsymbol\beta_1) \cdot \mathbf{X}_i} + 1}  = \frac{e^{-\boldsymbol\beta_ \cdot \mathbf{X}_i}} {1 +e^{-\boldsymbol\beta \cdot \mathbf{X}_i}} \\ \, \\
\Pr(Y_i=1) &= \frac{e^{\boldsymbol\beta_1 \cdot \mathbf{X}_i}} {~\sum_{0 \leq c \leq K}^{}{e^{\boldsymbol\beta_c \cdot \mathbf{X}_i}}} = \frac{e^{\boldsymbol\beta_1 \cdot \mathbf{X}_i}}{e^{\boldsymbol\beta_0 \cdot \mathbf{X}_i} + e^{\boldsymbol\beta_1 \cdot \mathbf{X}_i}} = \frac{1}{e^{(\boldsymbol\beta_0-\boldsymbol\beta_1) \cdot \mathbf{X}_i} + 1} = \frac{1} {1 +e^{-\boldsymbol\beta_ \cdot \mathbf{X}_i}}  \, \\
\end{align}
$$
with $\boldsymbol\beta = - (\boldsymbol\beta_0 - \boldsymbol\beta_1)$ We see that we obtain the same probabilities as in the two-class logistic regression using the sigmoid function. [Wikipedia](https://en.wikipedia.org/w/index.php?title=Logistic_regression&oldid=755697139#As_a_.22log-linear.22_model) expands a bit more on that.


# 深度学习常用激活函数之— Sigmoid & ReLU & Softmax
https://blog.csdn.net/zahuopuboss/article/details/70056231

