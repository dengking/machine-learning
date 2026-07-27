# Backpropagation&Automatic-differentiation

wikipedia [Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation):

>  In contrast, reverse accumulation requires the evaluated partial functions for the partial derivatives. Reverse accumulation therefore evaluates the function first and calculates the derivatives with respect to all independent variables in an additional pass.



Q: 训练时为什么要保存"中间激活值"（activations）？

A:

- 前向传播：计算 loss，并【保存每层的中间激活】，这些"中间激活"就是 "evaluated partial functions" 

- 反向传播：用保存的激活值计算梯度，"requires the evaluated ... for the partial derivatives" 



这也是为什么训练比推理更耗显存：必须存下所有中间激活，供反向传播使用！


