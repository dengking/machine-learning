# codinglabs [PCA的数学原理](http://blog.codinglabs.org/articles/pca-tutorial.html) 

PCA（Principal Component Analysis）是一种常用的数据分析方法。PCA通过**线性变换**将原始数据变换为一组各维度线性无关的表示，可用于提取数据的主要**特征分量**，常用于**高维数据**的降维。网上关于PCA的文章有很多，但是大多数只描述了PCA的分析过程，而没有讲述其中的原理。这篇文章的目的是介绍PCA的基本数学原理，帮助读者了解PCA的工作机制是什么。 

当然我并不打算把文章写成纯数学文章，而是希望用直观和易懂的方式叙述PCA的*数学原理*，所以整个文章不会引入严格的数学推导。希望读者在看完这篇文章后能更好的明白PCA的工作原理。 

## 数据的向量表示及降维问题

一般情况下，在数据挖掘和机器学习中，数据被表示为向量。例如某个淘宝店2012年全年的流量及交易情况可以看成一组记录的集合，其中每一天的数据是一条记录，格式如下：

(日期, 浏览量, 访客数, 下单数, 成交数, 成交金额)

其中“日期”是一个记录标志而非度量值，而数据挖掘关心的大多是**度量值**，因此如果我们忽略日期这个字段后，我们得到一组记录，每条记录可以被表示为一个五维向量，其中一条看起来大约是这个样子：
$$
(500,240,25,13,2312.15)^\mathsf{T}
$$
注意这里我用了**转置**，因为习惯上使用列向量表示一条记录（后面会看到原因），本文后面也会遵循这个准则。不过为了方便有时我会省略转置符号，但我们说到向量默认都是指**列向量**。 

我们当然可以对这一组五维向量进行分析和挖掘，不过我们知道，很多机器学习算法的复杂度和数据的维数有着密切关系，甚至与维数呈指数级关联。当然，这里区区五维的数据，也许还无所谓，但是实际机器学习中处理成千上万甚至几十万维的情况也并不罕见，在这种情况下，机器学习的资源消耗是不可接受的，因此我们必须对数据进行**降维**。 

<!--这里提到了对数据进行降维，我昨天在学习核函数的时候，里面提到了对数据进行升维，两者之间的关联是什么，需要搞清楚-->

<!--根据作者的这番描述，他对数据进行降维的目的是降低资源消耗；那在实际中，我们进行降维的目的都是因为资源消耗过大吗？-->

**降维**当然意味着信息的丢失，不过鉴于实际数据本身常常存在的相关性，我们可以想办法在降维的同时将信息的损失尽量降低。 

举个例子，假如某学籍数据有两列M和F，其中M列的取值是如何此学生为男性取值1，为女性取值0；而F列是学生为女性取值1，男性取值0。此时如果我们统计全部学籍数据，会发现对于任何一条记录来说，当M为1时F必定为0，反之当M为0时F必定为1。在这种情况下，我们将M或F去掉实际上没有任何信息的损失，因为只要保留一列就可以完全还原另一列。 

当然上面是一个极端的情况，在现实中也许不会出现，不过类似的情况还是很常见的。例如上面淘宝店铺的数据，从经验我们可以知道，“浏览量”和“访客数”往往具有较强的**相关关系**，而“下单数”和“成交数”也具有较强的**相关关系**。这里我们非正式的使用“相关关系”这个词，可以直观理解为“当某一天这个店铺的浏览量较高（或较低）时，我们应该很大程度上认为这天的访客数也较高（或较低）”。后面的章节中我们会给出**相关性**的严格数学定义。 

这种情况表明，如果我们删除浏览量或访客数其中一个指标，我们应该期待并不会丢失太多信息。因此我们可以删除一个，以降低机器学习算法的**复杂度**。

上面给出的是降维的朴素思想描述，可以有助于直观理解降维的动机和可行性，但并不具有操作指导意义。例如，我们到底删除哪一列损失的信息才最小？亦或根本不是单纯删除几列，而是通过某些**变换**将**原始数据**变为更少的列但又使得丢失的信息最小？到底如何度量丢失信息的多少？如何根据原始数据决定具体的降维操作步骤？

要回答上面的问题，就要对降维问题进行数学化和形式化的讨论。而PCA是一种具有严格数学基础并且已被广泛采用的**降维方法**。下面我不会直接描述PCA，而是通过逐步分析问题，让我们一起重新“发明”一遍PCA。

## 向量的表示及基变换

既然我们面对的数据被抽象为一组向量，那么下面有必要研究一些向量的数学性质。而这些数学性质将成为后续导出PCA的理论基础。 

### 内积与投影

下面先来看一个高中就学过的向量运算：内积。两个维数相同的向量的内积被定义为： 
$$
(a_1,a_2,\cdots,a_n)^\mathsf{T}\cdot (b_1,b_2,\cdots,b_n)^\mathsf{T}=a_1b_1+a_2b_2+\cdots+a_nb_n
$$
内积运算将两个向量映射为一个实数。其计算方式非常容易理解，但是其意义并不明显。下面我们分析内积的几何意义。假设A和B是两个n维向量，我们知道n维向量可以等价表示为n维空间中的一条从原点发射的有向线段，为了简单起见我们假设A和B均为二维向量，则 $A=(x_1,y_1)$,$B=(x_2,y_2)$。则在二维平面上A和B可以用两条发自原点的有向线段表示，见下图： 

![img](http://blog.codinglabs.org/uploads/pictures/pca-tutorial/01.png)

好，现在我们从A点向B所在直线引一条垂线。我们知道垂线与B的交点叫做A在B上的投影，再设A与B的夹角是a，则投影的矢量长度为 $|A|cos(a)$，其中$|A|=\sqrt{x_1^2+y_1^2}$,是向量A的**模**，也就是A线段的**标量长度**。

注意这里我们专门区分了**矢量长度**和**标量长度**，**标量长度**总是大于等于0，值就是线段的长度；而**矢量长度**可能为负，其绝对值是线段长度，而符号取决于其方向与标准方向相同或相反。

到这里还是看不出**内积**和这东西有什么关系，不过如果我们将内积表示为另一种我们熟悉的形式： 
$$
A\cdot B=|A||B|cos(a)
$$
现在事情似乎是有点眉目了：A与B的内积等于A到B的投影长度乘以B的模。再进一步，如果我们假设B的模为1，即让 $|B|=1 $,那么就变成了：  
$$
A\cdot B=|A|cos(a)
$$
也就是说，**设向量B的模为1，则A与B的内积值等于A向B所在直线投影的矢量长度**！这就是内积的一种几何解释，也是我们得到的第一个重要结论。在后面的推导中，将反复使用这个结论。 

### 基

下面我们继续在二维空间内讨论向量。上文说过，一个**二维向量**可以对应二维笛卡尔直角坐标系中从原点出发的一个**有向线段**。例如下面这个向量：

![img](http://blog.codinglabs.org/uploads/pictures/pca-tutorial/02.png)

在代数表示方面，我们经常用线段终点的点坐标表示向量，例如上面的向量可以表示为(3,2)，这是我们再熟悉不过的向量表示。

不过我们常常忽略，**只有一个(3,2)本身是不能够精确表示一个向量的**。我们仔细看一下，这里的3实际表示的是向量在x轴上的投影值是3，在y轴上的投影值是2。也就是说我们其实**隐式**引入了一个定义：以x轴和y轴上*正方向*长度为1的向量为标准。那么一个向量(3,2)实际是说在x轴投影为3而y轴的投影为2。注意**投影**是一个**矢量**，所以可以为负。 

更正式的说，向量(x,y)实际上表示线性组合： 
$$
x(1,0)^\mathsf{T}+y(0,1)^\mathsf{T}
$$
不难证明所有二维向量都可以表示为这样的线性组合。此处(1,0)和(0,1)叫做二维空间中的一组**基**。 

<!--经过作者的这些描述，可以知道，线性组合和投影是有关联的-->

![img](http://blog.codinglabs.org/uploads/pictures/pca-tutorial/03.png)

所以，**要准确描述向量，首先要确定一组基，然后给出在基所在的各个直线上的投影值，就可以了**<!--该向量就可以描述为这些投影的线性组合，这就是上面提及的投影和线性组合之间的关联-->。只不过我们经常省略第一步，而默认以(1,0)和(0,1)为基。

我们之所以默认选择(1,0)和(0,1)为基，当然是比较方便，因为它们分别是x和y轴正方向上的单位向量，因此就使得二维平面上点坐标和向量一一对应，非常方便。但实际上任何两个线性无关的二维向量都可以成为一组基，所谓线性无关在二维平面内可以直观认为是两个不在一条直线上的向量。

例如，(1,1)和(-1,1)也可以成为一组基。一般来说，我们希望基的模是1，因为从内积的意义可以看到，如果基的模是1，那么就可以方便的用向量点乘基而直接获得其在新基上的坐标了！实际上，对应任何一个向量我们总可以找到其同方向上模为1的向量，只要让两个分量分别除以模就好了。例如，上面的基可以变为$(\frac{1}{\sqrt{2}},\frac{1}{\sqrt{2}})$和$(-\frac{1}{\sqrt{2}},\frac{1}{\sqrt{2}})$。

现在，我们想获得(3,2)在新基上的坐标，即在两个方向上的投影矢量值，那么根据内积的几何意义，我们只要分别计算(3,2)和两个基的内积，不难得到新的坐标为  $(\frac{5}{\sqrt{2}},-\frac{1}{\sqrt{2}})$.下图给出了新的基以及(3,2)在新基上坐标值的示意图：

![img](http://blog.codinglabs.org/uploads/pictures/pca-tutorial/05.png)

另外这里要注意的是，我们列举的例子中基是正交的（即内积为0，或直观说相互垂直），但可以成为一组基的唯一要求就是线性无关，非正交的基也是可以的。不过因为正交基有较好的性质，所以一般使用的基都是正交的。

### 基变换的矩阵表示

下面我们找一种简便的方式来表示基变换。还是拿上面的例子，想一下，将(3,2)变换为新基上的坐标，就是用(3,2)与第一个基做内积运算，作为第一个新的坐标分量，然后用(3,2)与第二个基做内积运算，作为第二个新坐标的分量。实际上，我们可以用矩阵相乘的形式简洁的表示这个变换： 
$$
\begin{pmatrix}
  1/\sqrt{2}  & 1/\sqrt{2} \\
  -1/\sqrt{2} & 1/\sqrt{2}
\end{pmatrix}
\begin{pmatrix}
  3 \\
  2
\end{pmatrix}
=
\begin{pmatrix}
  5/\sqrt{2} \\
  -1/\sqrt{2}
\end{pmatrix}
$$
太漂亮了！其中矩阵的两行分别为两个基，乘以原向量，其结果刚好为新基的坐标。可以稍微推广一下，如果我们有m个二维向量，只要将二维向量**按列**排成一个两行m列矩阵，然后用“**基矩阵**”乘以这个矩阵，就得到了所有这些向量在新基下的值。例如(1,1)，(2,2)，(3,3)，想变换到刚才那组基上，则可以这样表示： 
$$
\begin{pmatrix}
  1/\sqrt{2}  & 1/\sqrt{2} \\
  -1/\sqrt{2} & 1/\sqrt{2}
\end{pmatrix}
\begin{pmatrix}
  1 & 2 & 3 \\
  1 & 2 & 3
\end{pmatrix}
=
\begin{pmatrix}
  2/\sqrt{2} & 4/\sqrt{2} & 6/\sqrt{2} \\
  0           & 0           & 0
\end{pmatrix}
$$
于是一组向量的基变换被干净的表示为矩阵的相乘。 

**一般的，如果我们有M个N维向量，想将其变换为由R个N维向量表示的新空间中，那么首先将R个基按行组成矩阵A，然后将向量按列组成矩阵B，那么两矩阵的乘积AB就是变换结果，其中AB的第m列为A中第m列变换后的结果**。

<!--需要注意的是，这些向量的都是N维的，这里我就有一个疑问：R个N维向量表示的新空间，如果R和N不相等，即这个R维空间的基是N维的向量，这样可以吗？-->

数学表示为：
$$
\begin{pmatrix}
  p_1 \\
  p_2 \\
  \vdots \\
  p_R
\end{pmatrix}
\begin{pmatrix}
  a_1 & a_2 & \cdots & a_M
\end{pmatrix}
=
\begin{pmatrix}
  p_1a_1 & p_1a_2 & \cdots & p_1a_M \\
  p_2a_1 & p_2a_2 & \cdots & p_2a_M \\
  \vdots  & \vdots  & \ddots & \vdots \\
  p_Ra_1 & p_Ra_2 & \cdots & p_Ra_M
\end{pmatrix}
$$
其中$p_i$是一个行向量，表示第i个基，$a_j$是一个列向量，表示第j个原始数据记录。 

特别要注意的是，这里R可以小于N，而R决定了变换后数据的**维数**。也就是说，我们可以将一N维数据变换到更低维度的空间中去，变换后的维度取决于基的数量。因此这种矩阵相乘的表示也可以表示**降维变换**。 



最后，上述分析同时给矩阵相乘找到了一种物理解释：**两个矩阵相乘的意义是将右边矩阵中的每一列列向量变换到左边矩阵中每一行行向量为基所表示的空间中去**。更抽象的说，一个矩阵可以表示一种**线性变换**。很多同学在学线性代数时对矩阵相乘的方法感到奇怪，但是如果明白了矩阵相乘的物理意义，其合理性就一目了然了。 

<!--矩阵相乘的物理意义直到刚才才明白-->

## 协方差矩阵及优化目标

上面我们讨论了选择不同的基可以对同样一组数据给出不同的表示，而且如果基的数量少于向量本身的维数，则可以达到降维的效果。但是我们还没有回答一个最最关键的问题：如何选择基才是最优的。或者说，如果我们有一组N维向量，现在要将其降到K维（K小于N），那么我们应该如何选择K个基才能最大程度保留原有的信息？ 

要完全数学化这个问题非常繁杂，这里我们用一种非形式化的直观方法来看这个问题。 

为了避免过于抽象的讨论，我们仍以一个具体的例子展开。假设我们的数据由五条记录组成，将它们表示成矩阵形式： 
$$
\begin{pmatrix}
  1 & 1 & 2 & 4 & 2 \\
  1 & 3 & 3 & 4 & 4
\end{pmatrix}
$$
其中每一列为一条数据记录，而一行为一个字段。为了后续处理方便，我们首先将每个字段内所有值都减去字段均值，其结果是将每个字段都变为均值为0（这样做的道理和好处后面会看到）。 

我们看上面的数据，第一个字段均值为2，第二个字段均值为3，所以变换后： 
$$
\begin{pmatrix}
  -1 & -1 & 0 & 2 & 0 \\
  -2 & 0 & 0 & 1 & 1
\end{pmatrix}
$$
我们可以看下五条数据在平面直角坐标系内的样子：

![img](http://blog.codinglabs.org/uploads/pictures/pca-tutorial/06.png)

现在问题来了：如果我们必须使用一维来表示这些数据，又希望尽量保留原始的信息，你要如何选择？

通过上一节对基变换的讨论我们知道，这个问题实际上是要在二维平面中选择一个方向，将所有数据都投影到这个方向所在直线上，用**投影值**表示**原始记录**。这是一个实际的二维降到一维的问题。

那么如何选择这个方向（或者说基）才能尽量保留最多的原始信息呢？一种直观的看法是：希望投影后的**投影值**尽可能**分散**。

<!--这里的分散让我想到了之前看到的KL散度-->

以上图为例，可以看出如果向x轴投影，那么最左边的两个点会重叠在一起，中间的两个点也会重叠在一起，于是本身四个各不相同的二维点投影后只剩下两个不同的值了，这是一种严重的信息丢失，同理，如果向y轴投影最上面的两个点和分布在x轴上的两个点也会重叠。所以看来x和y轴都不是最好的投影选择。我们直观目测，如果向通过第一象限和第三象限的斜线投影，则五个点在投影后还是可以区分的。 

下面，我们用数学方法表述这个问题。 

### 方差

上文说到，我们希望投影后**投影值**尽可能**分散**，而这种**分散程度**，可以用数学上的方差来表述。此处，一个字段的方差可以看做是每个元素与字段均值的差的平方和的均值，即：
$$
Var(a)=\frac{1}{m}\sum_{i=1}^m{(a_i-\mu)^2}
$$
由于上面我们*已经将每个字段的均值都化为0了*<!--这个在前面已经做了-->，因此方差可以直接用每个元素的平方和除以元素个数表示： 
$$
Var(a)=\frac{1}{m}\sum_{i=1}^m{a_i^2}
$$
于是上面的问题被形式化表述为：寻找一个一维基，使得所有数据变换为这个基上的坐标表示后，方差值最大。 

### 协方差

对于上面二维降成一维的问题来说，找到那个使得**方差**最大的方向就可以了。不过对于更高维，还有一个问题需要解决。考虑三维降到二维问题。与之前相同，首先我们希望找到一个方向使得投影后**方差**最大，这样就完成了第一个方向的选择，继而我们选择第二个投影方向。

如果我们还是单纯只选择方差最大的方向，很明显，这个方向与第一个方向应该是“几乎重合在一起”，显然这样的维度是没有用的，因此，应该有其他约束条件。从直观上说，让两个字段尽可能表示更多的原始信息，我们是不希望它们之间存在（线性）**相关性**的，因为相关性意味着两个字段不是完全独立，必然存在重复表示的信息。

数学上可以用两个字段的协方差表示其相关性，由于已经让每个字段均值为0，则： 
$$
Cov(a,b)=\frac{1}{m}\sum_{i=1}^m{a_ib_i}
$$
可以看到，在字段均值为0的情况下，两个字段的协方差简洁的表示为其内积除以元素数m。

当协方差为0时，表示两个字段完全独立。为了让协方差为0，我们选择第二个基时只能在与第一个基正交的方向上选择。因此最终选择的两个方向一定是正交的。

至此，我们得到了降维问题的优化目标：**将一组N维向量降为K维（K大于0，小于N），其目标是选择K个单位（模为1）正交基，使得原始数据变换到这组基上后，各字段两两间协方差为0，而字段的方差则尽可能大（在正交的约束下，取最大的K个方差）**。 

### 协方差矩阵

上面我们导出了优化目标，但是这个目标似乎不能直接作为操作指南（或者说算法），因为它只说要什么，但根本没有说怎么做。所以我们要继续在数学上研究计算方案。

我们看到，最终要达到的目的与字段内方差及字段间协方差有密切关系。因此我们希望能将两者统一表示，仔细观察发现，两者均可以表示为内积的形式，而内积又与矩阵相乘密切相关。于是我们来了灵感：

假设我们只有a和b两个字段，那么我们将它们按行组成矩阵X：
$$
X=\begin{pmatrix}
  a_1 & a_2 & \cdots & a_m \\
  b_1 & b_2 & \cdots & b_m
\end{pmatrix}
$$
然后我们用X乘以X的转置，并乘上系数1/m： 
$$
\frac{1}{m}XX^\mathsf{T}=\begin{pmatrix}
  \frac{1}{m}\sum_{i=1}^m{a_i^2}   & \frac{1}{m}\sum_{i=1}^m{a_ib_i} \\
  \frac{1}{m}\sum_{i=1}^m{a_ib_i} & \frac{1}{m}\sum_{i=1}^m{b_i^2}
\end{pmatrix}
$$
奇迹出现了！这个矩阵对角线上的两个元素分别是两个字段的方差，而其它元素是a和b的协方差。两者被统一到了一个矩阵的。

根据矩阵相乘的运算法则，这个结论很容易被推广到一般情况：

**设我们有m个n维数据记录，将其按列排成n乘m的矩阵X，设** $C=\frac{1}{m}XX^\mathsf{T}$,**则C是一个对称矩阵，其对角线分别个各个字段的方差，而第i行j列和j行i列元素相同，表示i和j两个字段的协方差**。 

### 协方差矩阵对角化

<!--我们最终的目的是对数据进行降维，并且在前面还探讨了评价降维好坏的方法，即前面的协方差矩阵；现在想想，我要对数据进行降维，其实需要根据已有的数据构造出一个新的向量空间，这个向量空间的构建是基于当前所有数据的协方差矩阵；那如何来根据协方差矩阵来构建这个新的向量空间的基向量呢？-->

根据上述推导，我们发现要达到优化目标，等价于将**协方差矩阵对角化**：即除**对角线**外的其它元素化为0，并且在**对角线**上将元素按大小从上到下排列，这样我们就达到了优化目的<!--这个优化目标说白了就是让协方差矩阵对角线上的元素，即方差，尽可能的大，让方对角线上的元素，即协方差，尽可能的小-->。这样说可能还不是很明晰，我们进一步看下**原矩阵**与基变换后矩阵协方差矩阵的关系： 

设**原始数据矩阵X**对应的**协方差矩阵**为C，而P是一组基按行组成的矩阵，设Y=PX，则Y为X对P做**基变换**后的数据。设Y的**协方差矩阵**为D，我们推导一下D与C的关系<!--两个协方差矩阵之间的关系-->： 
$$
\begin{array}{l l l}
  D & = & \frac{1}{m}YY^\mathsf{T} \\
    & = & \frac{1}{m}(PX)(PX)^\mathsf{T} \\
    & = & \frac{1}{m}PXX^\mathsf{T}P^\mathsf{T} \\
    & = & P(\frac{1}{m}XX^\mathsf{T})P^\mathsf{T} \\
    & = & PCP^\mathsf{T}
\end{array}
$$
现在事情很明白了！我们要找的P不是别的，而是能让原始协方差矩阵对角化的P。换句话说，优化目标变成了**寻找一个矩阵P，满足$PCP^\mathsf{T}$是一个对角矩阵，并且对角元素按从大到小依次排列，那么P的前K行就是要寻找的基，用P的前K行组成的矩阵乘以X就使得X从N维降到了K维并满足上述优化条件**。 

至此，我们离“发明”PCA还有仅一步之遥！

现在所有焦点都聚焦在了**协方差矩阵对角化**问题上，有时，我们真应该感谢数学家的先行，因为**矩阵对角化**在**线性代数**领域已经属于被玩烂了的东西，所以这在数学上根本不是问题。

由上文知道，**协方差矩阵C**是一个是**对称矩阵**，在线性代数上，**实对称矩阵**有一系列非常好的性质：

1）**实对称矩阵**不同**特征值**对应的**特征向量**必然**正交**。

2）设特征向量$\lambda$**重数**为r，则必然存在r个**线性无关**的**特征向量**对应于$\lambda$，因此可以将这r个特征向量**单位正交化**。

由上面两条可知，一个n行n列的**实对称矩阵**一定可以找到n个单位正交特征向量，设这n个特征向量为$e_1,e_2,\cdots,e_n$，我们将其按列组成矩阵：
$$
E=\begin{pmatrix}
  e_1 & e_2 & \cdots & e_n
\end{pmatrix}
$$
则对**协方差矩阵C**有如下结论： 
$$
E^\mathsf{T}CE=\Lambda=\begin{pmatrix}
  \lambda_1 &             &         & \\
              & \lambda_2 &         & \\
              &             & \ddots & \\
              &             &         & \lambda_n
\end{pmatrix}
$$
其中  $\Lambda$为对角矩阵，其**对角元素**为各**特征向量**对应的**特征值**（可能有重复）。 

以上结论不再给出严格的数学证明，对证明感兴趣的朋友可以参考线性代数书籍关于“实对称矩阵对角化”的内容。

到这里，我们发现我们已经找到了需要的矩阵P：
$$
P=E^\mathsf{T}
$$
P是**协方差矩阵**的*特征向量*单位化后按行排列出的矩阵，其中每一行都是C的一个**特征向量**。如果设P按照$\lambda$中特征值的从大到小，将特征向量从上到下排列，则用P的前K行组成的矩阵乘以原始数据矩阵X，就得到了我们需要的降维后的数据矩阵Y。

至此我们完成了整个PCA的数学原理讨论。在下面的一节，我们将给出PCA的一个实例。

## 算法及实例

为了巩固上面的理论，我们在这一节给出一个具体的PCA实例。

### PCA算法

总结一下PCA的算法步骤：

设有m条n维数据。

1）将原始数据按列组成n行m列矩阵X

2）将X的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值

3）求出协方差矩阵 $C=\frac{1}{m}XX^\mathsf{T}$

4）求出**协方差矩阵**的**特征值**及对应的**特征向量**

5）将**特征向量**按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P

6） $Y=PX$即为降维到k维后的数据 

### 实例

这里以上文提到的 
$$
\begin{pmatrix}
  -1 & -1 & 0 & 2 & 0 \\
  -2 & 0 & 0 & 1 & 1
\end{pmatrix}
$$
为例，我们用PCA方法将这组二维数据其降到一维。 

因为这个矩阵的每行已经是零均值，这里我们直接求协方差矩阵： 
$$
C=\frac{1}{5}\begin{pmatrix}
  -1 & -1 & 0 & 2 & 0 \\
  -2 & 0 & 0 & 1 & 1
\end{pmatrix}\begin{pmatrix}
  -1 & -2 \\
  -1 & 0  \\
  0  & 0  \\
  2  & 1  \\
  0  & 1
\end{pmatrix}=\begin{pmatrix}
  \frac{6}{5} & \frac{4}{5} \\
  \frac{4}{5} & \frac{6}{5}
\end{pmatrix}
$$
然后求其特征值和特征向量，具体求解方法不再详述，可以参考相关资料。求解后特征值为： 
$$
\lambda_1=2,\lambda_2=2/5
$$
其对应的特征向量分别是： 
$$
c_1\begin{pmatrix}
  1 \\
  1
\end{pmatrix},c_2\begin{pmatrix}
  -1 \\
  1
\end{pmatrix}
$$
其中对应的特征向量分别是一个通解，c1c1和c2c2可取任意实数。那么标准化后的特征向量为： 
$$
\begin{pmatrix}
  1/\sqrt{2} \\
  1/\sqrt{2}
\end{pmatrix},\begin{pmatrix}
  -1/\sqrt{2} \\
  1/\sqrt{2}
\end{pmatrix}
$$
因此我们的矩阵P是： 
$$
P=\begin{pmatrix}
  1/\sqrt{2}  & 1/\sqrt{2}  \\
  -1/\sqrt{2} & 1/\sqrt{2}
\end{pmatrix}
$$
可以验证协方差矩阵C的对角化： 
$$
PCP^\mathsf{T}=\begin{pmatrix}
  1/\sqrt{2}  & 1/\sqrt{2}  \\
  -1/\sqrt{2} & 1/\sqrt{2}
\end{pmatrix}\begin{pmatrix}
  6/5 & 4/5 \\
  4/5 & 6/5
\end{pmatrix}\begin{pmatrix}
  1/\sqrt{2} & -1/\sqrt{2}  \\
  1/\sqrt{2} & 1/\sqrt{2}
\end{pmatrix}=\begin{pmatrix}
  2 & 0  \\
  0 & 2/5
\end{pmatrix}
$$
最后我们用P的第一行乘以数据矩阵，就得到了降维后的表示： 
$$
Y=\begin{pmatrix}
  1/\sqrt{2} & 1/\sqrt{2}
\end{pmatrix}\begin{pmatrix}
  -1 & -1 & 0 & 2 & 0 \\
  -2 & 0 & 0 & 1 & 1
\end{pmatrix}=\begin{pmatrix}
  -3/\sqrt{2} & -1/\sqrt{2} & 0 & 3/\sqrt{2} & -1/\sqrt{2}
\end{pmatrix}
$$
降维投影结果如下图： 

## 进一步讨论

根据上面对PCA的数学原理的解释，我们可以了解到一些PCA的能力和限制。PCA本质上是将**方差最大的方向**作为**主要特征**，并且在各个正交方向上将数据“离相关”，也就是让它们在不同正交方向上没有**相关性**。

因此，PCA也存在一些限制，例如它可以很好的解除**线性相关**<!--为什么PCA可以解除线性相关性-->，但是对于**高阶相关性**就没有办法了，对于存在**高阶相关性**的数据，可以考虑Kernel PCA，通过Kernel函数<!--即核函数-->将**非线性相关**转为**线性相关**，关于这点就不展开讨论了。另外，PCA假设数据各主特征是分布在正交方向上，如果在非正交方向上存在几个方差较大的方向，PCA的效果就大打折扣了。

最后需要说明的是，PCA是一种**无参数技术**，也就是说面对同样的数据，如果不考虑清洗，谁来做结果都一样，没有主观参数的介入，所以PCA便于通用实现，但是本身无法个性化的优化。

希望这篇文章能帮助朋友们了解PCA的数学理论基础和实现原理，借此了解PCA的适用场景和限制，从而更好的使用这个算法。

