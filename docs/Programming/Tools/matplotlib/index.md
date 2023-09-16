# Matplotlib



## [Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html) 

> NOTE:
>
> 一、介绍Matplotlib的基本概念



### Types of inputs to plotting functions

Plotting functions expect [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html#numpy.array) or [`numpy.ma.masked_array`](https://numpy.org/doc/stable/reference/generated/numpy.ma.masked_array.html#numpy.ma.masked_array) as input, or objects that can be passed to [`numpy.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html#numpy.asarray). 

> NOTE:
>
> 一、Matplotlib是基于 [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html#numpy.array) 而设计的





## [Plot types](https://matplotlib.org/stable/plot_types/index.html)

> NOTE:
>
> 一、 matplotlib能够画哪些类型的图，这类进行了非常好的说明

### Pairwise data



### [scatter(x, y)](https://matplotlib.org/stable/plot_types/basic/scatter_plot.html)

> NOTE:
>
> 一、"scatter"即"散点图"，这种图的一些例子:
>
> 1、[LeetCode-587. Erect the Fence](https://leetcode.cn/problems/erect-the-fence/) 



```python
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make the data
np.random.seed(3)
x = 4 + np.random.normal(0, 2, 24)
y = 4 + np.random.normal(0, 2, len(x))
# size and color:(设置每个点的size、color)
sizes = np.random.uniform(15, 80, len(x))
colors = np.random.uniform(15, 80, len(x))

# plot
fig, ax = plt.subplots()

ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
```



### Statistical distributions

### Gridded data:

### Irregularly gridded data

### 3D and volumetric data