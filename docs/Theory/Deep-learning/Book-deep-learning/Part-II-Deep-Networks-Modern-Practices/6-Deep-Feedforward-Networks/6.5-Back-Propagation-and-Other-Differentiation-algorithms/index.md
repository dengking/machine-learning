# 关于本章

原书的6.5节《Back-Propagation and Other Differentiation Algorithms》描述Back-Propagation algorithm，这个算法是不容易理解，为了便于理解，可以先阅读科普性的读物：[Back-Propagation](./Back-Propagation/index.md)；在初步了解该算法后，正式进入[6.5-Back-Propagation-and-Other-Differentiation](./6.5-Back-Propagation-and-Other-Differentiation.md)，其中的描述是非常严谨、专业的；掌握该算法后，我们需要考虑如何实现back propagation：[Implementation](./Implementation.md)。



## computational-graph and chain-of-calculus and back-propagation

computational-graph是对math function（可以看做是一个expression）的structure representation；

chain of calculus是计算math function的公式；

### 思考: 如何根据函数表达式来构造computational-graph

这个过程跟compiler parse program是类似的；

math expression和program是context free language，都是遵循context free grammar；

compiler根据grammar（往往使用production的方式来表达）使用grammar tree来表示我们的program；

相应的，我们可以根据math expression的grammar构造出它的computational graph；

其实computational graph非常类似于abstract syntax tree的；

computational graph的构造构成是可以参考parsing的过程的。



#### Computation graph VS parse tree

> NOTE: parse tree在工程compiler-principle[#](https://dengking.github.io/compiler-principle/#compiler-principle)中介绍

两者都需要遵循grammar：

computation graph的构建需要遵循的各种operation的grammar；

parse tree的构建需要遵循的是context free grammar；

在工程compiler-principle[#](https://dengking.github.io/compiler-principle/#compiler-principle)中，parse tree的构建可以是top-down、bottom-up的。

在tensorflow whitepaper 2015中，computation graph的构建是bottom-up的。

#### Function composition and CFG

复合函数是显然的containing/nesting关系，这和CFG是一致的，这就决定了它们可以使用类似的方式来进行parsing。programming language、math都是CFG。

### 思考：为什么使用computational graph？

简而言之：使用这种representation，可以方便实现back-propagation。

从“结构化思维”的角度来分析：computational-graph 和 chain-of-calculus 有着相似的结构：hierarchy、containing关系，rewrite rule，recursion。两者结构的相似，决定了当沿着computational-graph的edge，可以非常自然地使用chain-of-calculus，进而非常容易地使用back-propagation。由此可以看到，采用合适的representation对于解决各种computational problem的意义。

#### Structure of chain-of-calculus

chain-of-calculus本质上就是一个formula，和普通的math expression类似。



显然，computational-graph对于实现back-propagation的重要意义，决定了tensorflow、torch的底层都使用computational graph来实现的原因。

back-propagation是对chain-of-calculus的运用，在



