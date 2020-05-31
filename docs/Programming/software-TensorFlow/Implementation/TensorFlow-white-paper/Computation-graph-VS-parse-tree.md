# Computation graph VS parse tree

> NOTE: parse tree在工程compiler-principle[#](https://dengking.github.io/compiler-principle/#compiler-principle)中介绍

两者都需要遵循grammar：

computation graph的构建需要遵循的各种operation的grammar；

parse tree的构建需要遵循的是context free grammar；

在工程compiler-principle[#](https://dengking.github.io/compiler-principle/#compiler-principle)中，parse tree的构建可以是top-down、bottom-up的。

在[whitepaper2015](./whitepaper2015.md)中，computation graph的构建是bottom-up的。