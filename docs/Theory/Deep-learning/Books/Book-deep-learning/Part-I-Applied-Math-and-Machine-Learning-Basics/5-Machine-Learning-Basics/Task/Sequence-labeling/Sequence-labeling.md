# Sequence labeling

本文是阅读维基百科[Sequence labeling](https://en.wikipedia.org/wiki/Sequence_labeling)的笔记。

In [machine learning](https://en.wikipedia.org/wiki/Machine_learning), **sequence labeling** is a type of [pattern recognition](https://en.wikipedia.org/wiki/Pattern_recognition) task that involves the algorithmic assignment of a [categorical](https://en.wikipedia.org/wiki/Categorical_data) label to each member of a sequence of observed values. 

显然，sequence labeling所处理的是sequence结构的数据，显然，它的输出也是一个sequence。

生活中哪些数据是具备sequence结构的呢？

- sentence，与此相关的有[part of speech tagging](https://en.wikipedia.org/wiki/Part_of_speech_tagging)
- speech，参见 [speech](https://en.wikipedia.org/wiki/Speech_recognition)



Sequence labeling can be treated as a set of independent [classification](https://en.wikipedia.org/wiki/Classification_(machine_learning)) tasks, one per member of the sequence. However, accuracy is generally improved by making the optimal label for a given element dependent on the choices of nearby elements, using special algorithms to choose the *globally* best set of labels for the entire sequence at once.

这段话其实描述了解决sequence label问题的两种思路：

- independent，即“labeling one item at a time”
- dependent，即“finding the globally best label sequence”

原文的下一段结合part-of-speech tagging对这两种方式进行了比较。

Most **sequence labeling algorithms** are [probabilistic](https://en.wikipedia.org/wiki/Probability_theory) in nature, relying on [statistical inference](https://en.wikipedia.org/wiki/Statistical_inference) to find the best sequence. The most common **statistical models** in use for sequence labeling make a **Markov assumption**, i.e. that the choice of label for a particular word is directly dependent only on the immediately adjacent labels; hence the set of labels forms a [Markov chain](https://en.wikipedia.org/wiki/Markov_chain). This leads naturally to the [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMM), one of the most common statistical models used for sequence labeling. Other common models in use are the [maximum entropy Markov model](https://en.wikipedia.org/wiki/Maximum_entropy_Markov_model) and [conditional random field](https://en.wikipedia.org/wiki/Conditional_random_field).

上面这段话中的**statistical models**参见[Statistical model](https://en.wikipedia.org/wiki/Statistical_model)，**Markov assumption**参见[Markov property](https://en.wikipedia.org/wiki/Markov_property)。	