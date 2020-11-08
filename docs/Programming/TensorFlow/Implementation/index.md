# 关于本章

本章探讨tensorflow的实现，主要参考的有：

1) [whitepaper2015](http://download.tensorflow.org/paper/whitepaper2015.pdf)

2) [图解tensorflow 源码](https://github.com/yao62995/tensorflow)

## TensorFlow is a parallel numeric processing system

本节标题的含义是: TensorFlow是一个parallel numeric processing system，关于parallel numeric processing system，参见工程parallel computing的`Application\Parallel-numeric-processing-system`章节。

## TensorFlow VS compiler

从某种程度上来说，TensorFlow和compiler是有些类似之处的，可以进行比较:

### Front end/Interface

让programmer使用**symbol expression**来描述computation（symbolic programming），TensorFlow的front end使用computation graph的来进行**结构化表示**。如果使用compiler来进行类比的话，computation graph其实和AST非常类似，如果从结构化表示的角度来看: 两者本质上是相同的，都是对symbol expression的**结构化表示**。但是，由于两者是不同领域的，所以需要考虑各自领域中的特定 问题，在下面的章节中会讨论TensorFlow computation graph需要考虑的问题。

### Back end/Computation engine/Core

Back end实现computation graph表示的computation。与compiler中，将AST转换为三地址码、然后转换为instruction进而实现语义理解不同的是，TensorFlow back end并不会将computation graph转换为instruction的方式，TensorFlow back end会将computation graph完整地保存，然后基于computation graph来安排计算，TensorFlow back end:

1) dataflow programming paradigm

2) node对应operator

3) distributed: TensorFlow允许用户将node指定到不同的computer，这些computer之间通过network来进行communicate，显然整体来看，它们还是对应的用户定义的computation graph

4) abstraction: heterogeneous机器的抽象

5) 借鉴了microsoft [Naiad](https://www.microsoft.com/en-us/research/project/naiad/)



> NOTE: 关于上述讨论，在infogalactic [Dataflow programming#Properties of dataflow programming languages](https://infogalactic.com/info/Dataflow_programming#Properties_of_dataflow_programming_languages)中也进行了讨论。