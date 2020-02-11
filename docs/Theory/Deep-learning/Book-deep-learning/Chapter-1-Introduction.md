# Chapter 1 Introduction

In the early days of artificial intelligence, the field rapidly tackled and solved problems that are intellectually difficult for human beings but relatively straightforward for computers—**problems that can be described by a list of formal, mathematical rules**. The true challenge to artificial intelligence proved to be solving the tasks that are easy for people to perform but hard for people to describe **formally**—problems that we solve intuitively, that feel automatic, like recognizing spoken words or faces in images.

需要注意的是，在本章中，作者频繁的使用了**formal**这个词语，所以理解这个词语对于理解本文是至关重要的，读者可以将它简单的理解为可以规则化的，可以使用数学公式进行描述的。有经验的软件工程师一定能够想到，**problems that can be described by a list of formal, mathematical rules**是比较容易用程序来进行实现的。

计算机和人类所擅长的是不同的：计算机所擅长的是解决如下类型的问题：

> problems that can be described by a list of formal, mathematical rules

对于这类问题，传统的计算机算法基本能够实现。

人类所擅长的是解决的是:

>  problems that are intuitive and hard for people to describe formally

对于这类问题，传统的计算机算法是非常难以实现的。这类问题正是本书所描述的deep learning技术志于解决的问题，解决思想如下：

This book is about a solution to these more **intuitive problems**. This solution is to **allow computers to learn from experience and understand the world in terms of a hierarchy of concepts, with each concept defined in terms of its relation to simpler concepts**. By gathering knowledge from experience, this approach avoids the need for human operators to formally specify all of the knowledge that the computer needs. The hierarchy of concepts allows the computer to learn complicated concepts by building them out of simpler ones. If we draw a graph showing how these concepts are built on top of each other, the graph is deep, with many layers. For this reason, we call this approach to AI **deep learning**.

上面这段话描述了当前**machine learning**的核心思想，这个思想的对于理解后面章节的内容是非常重要的，或者说，后面章节就是在告诉读者如何来实现machine learning。



原书的这一节的后续内容基本上就是围绕着我上面所总结的内容展开的。



## 案例：Formal language and natural language

关于计算机和人类所擅长解决的问题的一个典型案例就是：formal language和natural language。计算机能够轻松理解formal language（因为formal language的grammar和semantic都能够形式化地进行描述），但是对于natural language的理解则是非常可能的（natural language的grammar和semantic是非常复杂的，无法进行形式化的描述）。而人类基本相反。