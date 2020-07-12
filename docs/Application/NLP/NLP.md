# NLP



## 维基百科[Natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing)

**Natural language processing** (**NLP**) is a subfield of [linguistics](https://en.wikipedia.org/wiki/Linguistics), [computer science](https://en.wikipedia.org/wiki/Computer_science), [information engineering](https://en.wikipedia.org/wiki/Information_engineering_(field)), and [artificial intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence) concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of [natural language](https://en.wikipedia.org/wiki/Natural_language) data.

> NOTE: 综合学科



### History

Up to the 1980s, most natural language processing systems were based on complex sets of hand-written **rules**. Starting in the late 1980s, however, there was a revolution in natural language processing with the introduction of [machine learning](https://en.wikipedia.org/wiki/Machine_learning) algorithms for language processing. This was due to both the steady increase in computational power (see [Moore's law](https://en.wikipedia.org/wiki/Moore's_law)) and the gradual lessening of the dominance of [Chomskyan](https://en.wikipedia.org/wiki/Noam_Chomsky) theories of linguistics (e.g. [transformational grammar](https://en.wikipedia.org/wiki/Transformational_grammar)), whose theoretical underpinnings（基础） discouraged the sort of [corpus linguistics](https://en.wikipedia.org/wiki/Corpus_linguistics) that underlies the machine-learning approach to language processing.[[3\]](https://en.wikipedia.org/wiki/Natural_language_processing#cite_note-3) Some of the earliest-used machine learning algorithms, such as [decision trees](https://en.wikipedia.org/wiki/Decision_tree), produced systems of hard if-then rules similar to existing hand-written rules. However, [part-of-speech tagging](https://en.wikipedia.org/wiki/Part_of_speech_tagging)（词性标注） introduced the use of [hidden Markov models](https://en.wikipedia.org/wiki/Hidden_Markov_models) to natural language processing, and increasingly, research has focused on [statistical models](https://en.wikipedia.org/wiki/Statistical_models), which make soft, [probabilistic](https://en.wikipedia.org/wiki/Probabilistic) decisions based on attaching [real-valued](https://en.wikipedia.org/wiki/Real-valued) weights to the features making up the input data. The [cache language models](https://en.wikipedia.org/wiki/Cache_language_model) upon which many [speech recognition](https://en.wikipedia.org/wiki/Speech_recognition) systems now rely are examples of such statistical models. Such models are generally more robust when given unfamiliar input, especially input that contains errors (as is very common for real-world data), and produce more reliable results when integrated into a larger system comprising multiple subtasks.

> NOTE: 上述rule的含义是什么？grammar rule？应该就是grammar rule，比如在python的Language Reference[¶](https://docs.python.org/3/reference/index.html#the-python-language-reference)中一般会使用[BNF](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form)来描述这种语言的grammar；

> NOTE: 乔姆斯基语言学的理论基础阻碍了[corpus linguistics](https://en.wikipedia.org/wiki/Corpus_linguistics)（语料库语言学），[corpus linguistics](https://en.wikipedia.org/wiki/Corpus_linguistics)是machine-learning构成语言处理方法的基础。



### Rule-based vs. statistical NLP

In the early days, many language-processing systems were designed by hand-coding a set of rules:[[9\]](https://en.wikipedia.org/wiki/Natural_language_processing#cite_note-winograd:shrdlu71-9)[[10\]](https://en.wikipedia.org/wiki/Natural_language_processing#cite_note-schank77-10) such as by writing grammars or devising heuristic rules for [stemming](https://en.wikipedia.org/wiki/Stemming).

Since the so-called "statistical revolution"[[11\]](https://en.wikipedia.org/wiki/Natural_language_processing#cite_note-johnson:eacl:ilcl09-11)[[12\]](https://en.wikipedia.org/wiki/Natural_language_processing#cite_note-resnik:langlog11-12) in the late 1980s and mid 1990s, much natural language processing research has relied heavily on [machine learning](https://en.wikipedia.org/wiki/Machine_learning).

The machine-learning paradigm calls instead for using [statistical inference](https://en.wikipedia.org/wiki/Statistical_inference) to automatically learn such rules through the analysis of large *corpora* of typical real-world examples (a *corpus* (plural, "corpora") is a set of documents, possibly with human or computer annotations).

> NOTE: 机器学习范式要求使用统计推理，通过对典型的真实世界示例的大型语料库(语料库(复数，“语料库”)是一组文档，可能带有人工或计算机注释)的分析来自动学习这些规则。

Many different classes of machine-learning algorithms have been applied to natural-language-processing tasks. These algorithms take as input a large set of "features" that are generated from the input data. Some of the earliest-used algorithms, such as [decision trees](https://en.wikipedia.org/wiki/Decision_tree), produced systems of hard if-then rules similar to the systems of handwritten rules that were then common. Increasingly, however, research has focused on [statistical models](https://en.wikipedia.org/wiki/Statistical_models), which make soft, [probabilistic](https://en.wikipedia.org/wiki/Probabilistic) decisions based on attaching [real-valued](https://en.wikipedia.org/wiki/Real-valued) weights to each input feature. Such models have the advantage that they can express the relative certainty of many different possible answers rather than only one, producing more reliable results when such a model is included as a component of a larger system.

Systems based on machine-learning algorithms have many advantages over handproduced rules:

- The learning procedures used during machine learning automatically focus on the most common cases, whereas when writing rules by hand it is often not at all obvious where the effort should be directed.
- Automatic learning procedures can make use of statistical-inference algorithms to produce models that are robust to unfamiliar input (e.g. containing words or structures that have not been seen before) and to erroneous input (e.g. with misspelled words or words accidentally omitted). Generally, handling such input gracefully with handwritten rules, or, more generally, creating systems of handwritten rules that make soft decisions, is extremely difficult, error-prone and time-consuming.
- Systems based on automatically learning the rules can be made more accurate simply by supplying more input data. However, systems based on handwritten rules can only be made more accurate by increasing the complexity of the rules, which is a much more difficult task. In particular, there is a limit to the complexity of systems based on handcrafted rules, beyond which the systems become more and more unmanageable. However, creating more data to input to machine-learning systems simply requires a corresponding increase in the number of man-hours worked, generally without significant increases in the complexity of the annotation process.



### Major evaluations and tasks

> NOTE: 在`NLP\NLP-progress`章节对这部分内容进行了介绍。



## 维基百科[Corpus linguistics](https://en.wikipedia.org/wiki/Corpus_linguistics)

> NOTE: 使用machine learning技术来解决NLP问题，就是属于这个流派

**Corpus linguistics** is the [study of language](https://en.wikipedia.org/wiki/Study_of_language) as expressed in *corpora* (samples) of "real world" text. Corpus linguistics proposes that reliable language analysis is more feasible with corpora collected in the field in its natural context ("realia"), and with minimal experimental-interference.

The field of corpus linguistics features divergent views about the value of corpus annotation. These views range from [John McHardy Sinclair](https://en.wikipedia.org/wiki/John_McHardy_Sinclair), who advocates minimal annotation so texts speak for themselves,[[1\]](https://en.wikipedia.org/wiki/Corpus_linguistics#cite_note-1) to the [Survey of English Usage](https://en.wikipedia.org/wiki/Survey_of_English_Usage) team ([University College, London](https://en.wikipedia.org/wiki/University_College,_London)), who advocate annotation as allowing greater linguistic understanding through rigorous recording.[[2\]](https://en.wikipedia.org/wiki/Corpus_linguistics#cite_note-2)

The **text-corpus method** is a digestive approach that derives（导出） a set of abstract rules that govern a [natural language](https://en.wikipedia.org/wiki/Natural_language) from texts in that language, and explores how that language relates to other languages. Originally derived manually, corpora now are automatically derived from source texts.

In addition to linguistics research, assembled corpora have been used to compile [dictionaries](https://en.wikipedia.org/wiki/Dictionaries) (starting with *The American Heritage Dictionary of the English Language* in 1969) and grammar guides, such as *A Comprehensive Grammar of the English Language*, published in 1985.



## 维基百科[Language model](https://en.wikipedia.org/wiki/Language_model)