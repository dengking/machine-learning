[TOC]



# [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)

**Gradient descent** is a [first-order](https://en.wikipedia.org/wiki/Category:First_order_methods) [iterative](https://en.wikipedia.org/wiki/Iterative_algorithm) [optimization](https://en.wikipedia.org/wiki/Mathematical_optimization) [algorithm](https://en.wikipedia.org/wiki/Algorithm) for finding the minimum of a function. To find a [local minimum](https://en.wikipedia.org/wiki/Local_minimum) of a function using gradient descent, one takes steps proportional to the *negative* of the [gradient](https://en.wikipedia.org/wiki/Gradient) (or approximate gradient) of the function at the current point. If, instead, one takes steps proportional to the *positive* of the gradient, one approaches a [local maximum](https://en.wikipedia.org/wiki/Local_maximum) of that function; the procedure is then known as **gradient ascent**.

***SUMMARY*** : 梯度上升与梯度下降

Gradient descent is also known as **steepest descent**（极速下降）. However, gradient descent should not be confused with the [method of steepest descent](https://en.wikipedia.org/wiki/Method_of_steepest_descent) for approximating integrals.

## Description

Gradient descent is based on the observation that if the [multi-variable function](https://en.wikipedia.org/wiki/Multi-variable_function) $ F(\mathbf {x} ) $ is [defined](https://en.wikipedia.org/wiki/Defined_and_undefined) and [differentiable](https://en.wikipedia.org/wiki/Differentiable_function) in a neighborhood of a point $ \mathbf {a} $, then $ F(\mathbf {x} ) $ decreases *fastest* if one goes from $ \mathbf {a} $ in the direction of the **negative gradient** of $ F $ at $ \mathbf {a} ,-\nabla F(\mathbf {a} ) $. It follows that, if

$ \mathbf {a} _{n+1}=\mathbf {a} _{n}-\gamma \nabla F(\mathbf {a} _{n}) $

for $ \gamma \in \mathbb {R} _{+} $ small enough, then $ F(\mathbf {a_{n}} )\geq F(\mathbf {a_{n+1}} ) $. In other words, the term $ \gamma \nabla F(\mathbf {a} ) $ is subtracted from $ \mathbf {a} $ because we want to move against the gradient, toward the minimum. With this observation in mind, one starts with a guess $ \mathbf {x} _{0} $ for a local minimum of $ F $, and considers the sequence $ \mathbf {x} _{0},\mathbf {x} _{1},\mathbf {x} _{2},\ldots $ such that

$ \mathbf {x} _{n+1}=\mathbf {x} _{n}-\gamma _{n}\nabla F(\mathbf {x} _{n}),\ n\geq 0. $

We have a [monotonic](https://en.wikipedia.org/wiki/Monotonic_function) sequence

$ F(\mathbf {x} _{0})\geq F(\mathbf {x} _{1})\geq F(\mathbf {x} _{2})\geq \cdots , $

so hopefully the sequence $ (\mathbf {x} _{n}) $ converges to the desired local minimum. Note that the value of the *step size* $ \gamma $ is allowed to change at every iteration. With certain assumptions on the function $ F $ (for example, $ F $ [convex](https://en.wikipedia.org/wiki/Convex_function) and $ \nabla F $ [Lipschitz](https://en.wikipedia.org/wiki/Lipschitz_continuity)) and particular choices of $ \gamma $ (e.g., chosen either via a [line search](https://en.wikipedia.org/wiki/Line_search) that satisfies the [Wolfe conditions](https://en.wikipedia.org/wiki/Wolfe_conditions) or the Barzilai-Borwein[[1\]](https://en.wikipedia.org/wiki/Gradient_descent#cite_note-1) method shown as following),

$ \gamma _{n}={\frac {\left|\left(\mathbf {x} _{n}-\mathbf {x} _{n-1}\right)^{T}\left[\nabla F(\mathbf {x} _{n})-\nabla F(\mathbf {x} _{n-1})\right]\right|}{\left\|\nabla F(\mathbf {x} _{n})-\nabla F(\mathbf {x} _{n-1})\right\|^{2}}} $

[convergence](https://en.wikipedia.org/wiki/Convergent_series) to a local minimum can be guaranteed. When the function $ F $ is [convex](https://en.wikipedia.org/wiki/Convex_function), all **local minima** are also **global minima**, so in this case gradient descent can converge to the global solution.

This process is illustrated in the adjacent picture. Here $ F $ is assumed to be defined on the plane, and that its graph has a [bowl](https://en.wikipedia.org/wiki/Bowl_(vessel)) shape. The blue curves are the [contour lines](https://en.wikipedia.org/wiki/Contour_line), that is, the regions on which the value of $ F $ is constant. A red arrow originating at a point shows the direction of the negative gradient at that point. Note that the (negative) gradient at a point is [orthogonal](https://en.wikipedia.org/wiki/Orthogonal) to the contour line going through that point. We see that gradient *descent* leads us to the bottom of the bowl, that is, to the point where the value of the function $ F $ is minimal.

[![img](https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Gradient_descent.svg/350px-Gradient_descent.svg.png)](https://en.wikipedia.org/wiki/File:Gradient_descent.svg)Illustration of gradient descent on a series of level sets.



### Examples

Gradient descent has problems with pathological（病态） functions such as the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) shown here.

$ f(x_{1},x_{2})=(1-x_{1})^{2}+100(x_{2}-{x_{1}}^{2})^{2}. $

The Rosenbrock function has a narrow curved valley which contains the minimum. The bottom of the valley is very flat. Because of the curved flat valley the optimization is zigzagging slowly with small step sizes towards the minimum.

[![Banana-SteepDesc.gif](https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Banana-SteepDesc.gif/400px-Banana-SteepDesc.gif)](https://en.wikipedia.org/wiki/File:Banana-SteepDesc.gif)



The zigzagging nature of the method is also evident below, where the gradient descent method is applied to

$ F(x,y)=\sin \left({\frac {1}{2}}x^{2}-{\frac {1}{4}}y^{2}+3\right)\cos \left(2x+1-e^{y}\right). $

[![The gradient descent algorithm in action. (1: contour)](https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/Gradient_ascent_%28contour%29.png/350px-Gradient_ascent_%28contour%29.png)](https://en.wikipedia.org/wiki/File:Gradient_ascent_(contour).png)[![The gradient descent algorithm in action. (2: surface)](https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Gradient_ascent_%28surface%29.png/450px-Gradient_ascent_%28surface%29.png)](https://en.wikipedia.org/wiki/File:Gradient_ascent_(surface).png)





