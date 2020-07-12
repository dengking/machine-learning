# Part of speech

本文的内容基于：

- 维基百科[Part of speech](https://en.wikipedia.org/wiki/Part_of_speech)
- [Natural Language Processing with Python](http://www.nltk.org/book/)的[5. Categorizing and Tagging Words](http://www.nltk.org/book/ch05.html)

## What is “part of speech”？

"part of speech"即词性，如果以machine learning的角度来看的话，“part of speech”非常类似于“label”，即它表示的是词语的**类别**（后文中会出现”word class“这个词，显然这个词是更加能够体现它的含义的），这一点在[Natural Language Processing with Python](http://www.nltk.org/book/)的[5. Categorizing and Tagging Words](http://www.nltk.org/book/ch05.html)的名称中的“Categorizing"（分类）中体现出来了。显然，既然能够进行分类，那么各种part of speech（word class）肯定有着显著的特征。

有了这些认知，就能够理解维基百科的[Part of speech](https://en.wikipedia.org/wiki/Part_of_speech)中的定义了：

> In [traditional grammar](https://en.wikipedia.org/wiki/Traditional_grammar), a **part of speech** (abbreviated form: **PoS** or **POS**) is a category of words (or, more generally, of [lexical items](https://en.wikipedia.org/wiki/Lexical_item)) that have similar [grammatical](https://en.wikipedia.org/wiki/Grammar) properties. Words that are assigned to the same part of speech generally display similar [syntactic](https://en.wikipedia.org/wiki/Syntax) behavior—they play similar roles within the grammatical structure of sentences—and sometimes similar [morphology](https://en.wikipedia.org/wiki/Morphology_(linguistics)) in that they undergo [inflection](https://en.wikipedia.org/wiki/Inflection) for similar properties.

下面给出了part of speech的一个例子：

> Commonly listed [English](https://en.wikipedia.org/wiki/English_language) parts of speech are [noun](https://en.wikipedia.org/wiki/Noun), [verb](https://en.wikipedia.org/wiki/Verb), [adjective](https://en.wikipedia.org/wiki/Adjective), [adverb](https://en.wikipedia.org/wiki/Adverb), [pronoun](https://en.wikipedia.org/wiki/Pronoun), [preposition](https://en.wikipedia.org/wiki/Preposition), [conjunction](https://en.wikipedia.org/wiki/Conjunction_(grammar)), [interjection](https://en.wikipedia.org/wiki/Interjection), and sometimes [numeral](https://en.wikipedia.org/wiki/Numeral_(linguistics)), [article](https://en.wikipedia.org/wiki/Article_(grammar)), or [determiner](https://en.wikipedia.org/wiki/Determiner). 

Other terms than *part of speech*—particularly in modern [linguistic](https://en.wikipedia.org/wiki/Linguistics) classifications, which often make more precise distinctions than the traditional scheme does—include **word class**, **lexical class**, and **lexical category**. 

需要注意的是，不同的语言有着不同的part of speech

Because of such variation in the number of categories and their identifying properties, analysis of parts of speech must be done for each individual language.



## 如何定义（创造）part of speech？

一门语言的可用使用的Part of speech不是固定的，研究人员是可以根据需求来创造适合于特定问题的part of speech（其实就是定义类别，但是显然在进行定义的时候，应该是需要考虑语言学的理论的），这些part of speech称为“[Tag sets](https://en.wikipedia.org/wiki/Part-of-speech_tagging#Tag_sets)”。那如何来进行创造呢？下面给出了一些有参考价值的内容：

- [Functional classification](https://en.wikipedia.org/wiki/Part_of_speech#Functional_classification)
- [7   How to Determine the Category of a Word](http://www.nltk.org/book/ch05.html)



## 如何让计算机来自动地进行“分类”？

我们已经了解了part of speech的含义，知道它本质上就是类别。给定一段话，我们人类是可以非常轻松地指出哪些词是属于哪种part of speech，那如何让computer也获得这种能力呢？这就是本节标题的所提出的问题,，这个问题是NLP领域的经典问题，叫做[Part-of-speech tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)，在下一章节的[Part-of-speech-tagging](../NLP/Part-of-speech-tagging/index.md)中对此进行专门讨论。

