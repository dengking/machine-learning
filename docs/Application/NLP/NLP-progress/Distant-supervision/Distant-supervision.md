# [what is distant supervision?](https://stackoverflow.com/questions/29575784/what-is-distant-supervision)



According to my understanding, **Distant Supervisio**n is the process of specifying the **concept** which the individual words of a passage, usually a sentence, are trying to convey.

For example, a database maintains the structured relationship `concerns( NLP, this sentence).`

Our distant supervision system would take as input the sentence: `"This is a sentence about NLP."`

Based on this sentence it would recognize the entities, since as a pre-processing step the sentence would have been passed through a named-entity recognizer, `NLP` & `this sentence`.

Since our database has it that `NLP` and `this sentence` are related by the bond of `concern(s)` it would identify the input sentence as expressing the relationship `Concerns(NLP, this sentence)`.

My questions is two fold:

1) What is the use of that? Is it that later our system might see a sentence in "the wild" such as `That sentence is about OPP` and realize that it's seen something similar to that before and thereby realize the novel relationship such that `concerns(OPP, that sentence).`, based only on the words/ individual tokens?

2) Does it take into account the actual words of the sentence? The verb 'is' and the adverb 'about' for instance, realizing (through WordNet or some other hyponymy system) that this is somehow similar to the higher-order concept "concerns"?

Does anyone have some code used to generate a distant supervision system that I could look at, i.e. a system that cross references a KB, such as Freebase, and a corpus, such as the NYTimes, and produces a distant supervision database? I think that would go a long way in clarifying my conception of distant supervision.



# [Distant supervision: supervised, semi-supervised, or both?](https://stats.stackexchange.com/questions/46685/distant-supervision-supervised-semi-supervised-or-both)

"Distant supervision" is a learning scheme in which a classifier is learned given a weakly labeled training set (training data is labeled automatically based on **heuristics / rules**). I think that both supervised learning, and semi-supervised learning can include such "distant supervision" if their labeled data is heuristically/automatically labeled. However, in [this page](http://www.gabormelli.com/RKB/Distant-Supervision_Learning_Algorithm), "distant supervision" is defined as "semi-supervised learning" (i.e., limited to "semi-supervision").

So my question is, **does "distant supervision" exclusively refer to semi-supervision?** In my opinion it can be applied to both supervised and semi-supervised learning. Please provide any reliable references if any.



## [A](https://stats.stackexchange.com/a/47036)

A Distant supervision algorithm usually has the following steps: 
1] It may have some labeled training data 
2] It "has" access to a pool of unlabeled data 
3] It has an operator that allows it to sample from this unlabeled data and label them and this operator is expected to be noisy in its labels 
4] The algorithm then collectively utilizes the original labeled training data if it had and this new noisily labeled data to give the final output.

Now, to answer your question, you as well as the site both are correct. You are looking at the 4th step of the algorithm and notice that at the 4th step one can use any algorithm that the user has access to. Hence your point, **"it can be applied to both supervised and semi-supervised learning"**.

Whereas the site is looking at all the steps 1-4 collectively and notices that the noisily labeled data is obtained from a pool of unlabeled data (with or without the use of some pre-existing labeled training data) and this process of obtaining noisy labels is an essential component for any distant supervision algorithm, hence it *is* a semi-supervised algorithm.





# [Distant supervision](http://deepdive.stanford.edu/distant_supervision)

Most **machine learning techniques** require a set of **training data**. A traditional approach for collecting training data is to have humans label a set of documents. For example, for the marriage relation, human annotators may label the pair "Bill Clinton" and "Hillary Clinton" as a positive training example. This approach is expensive in terms of both time and money, and if our corpus is large, will not yield enough data for our algorithms to work with. And because humans make errors, the resulting training data will most likely be noisy.

An alternative approach to generating training data is **distant supervision**. In distant supervision, we make use of an **already existing database**, such as [Freebase](http://www.freebase.com/) or a domain-specific database, to collect examples for the relation we want to extract. We then use these examples to automatically generate our training data. For example, Freebase contains the fact that Barack Obama and Michelle Obama are married. We take this fact, and then label each pair of "Barack Obama" and "Michelle Obama" that appear in the same sentence as a positive example for our marriage relation. This way we can easily generate a large amount of (possibly noisy) training data. Applying distant supervision to get positive examples for a particular relation is easy, but [generating negative examples](http://deepdive.stanford.edu/generating_negative_examples) is more of an art than a science.








# [远程监督浅谈](https://blog.csdn.net/lzw17750614592/article/details/88908018)

关系抽取是NER基础上的一个任务，就是抽取一个句子中实体对之间的关系。想要训练一个关系抽取器，给它一个句子俩实体，首先它需要知道给这俩实体间的关系打个什么标签，模型不可能自己给关系取名字，所以肯定需要人用标注好的语料告诉他，这俩实体间的关系叫啥。然后模型训练好了，再遇到哪个句子里有这种实体对，他就会知道是这个关系并抽出来。

 

那么问题来了。人工标注好的语料哪里去找。这是NLP方向一个巨大的挑战。自己标费时费力，而且数量实在有限，数据规模大大限制了模型训练。

 

这个时候mintz大佬首次提出了不依赖人工标注的关系抽取，也就是把远程监督应用到了关系抽取上。那么到底什么叫做远程监督？它既不是单纯的传统意义上的监督语料，当然也不是无监督。它是一种用KB去对齐朴素文本的标注方法。

 

KB中已经有关系名和实体对的三元组，只需要把这**三元组**付给**朴素文本**中相应的句子就可以了，那按照什么原则付？这时候z大佬就提出了一个非常大的假设：如果一个句子中含有一个关系涉及的实体对，那这个句子就是描述的这个关系。也就是说，报纸里所有含有中国和北京的句子，全都假设说的是北京是中国的首都。然后把这些句子全都提取出来作为首都这个关系的训练语料，直接批量打个**标签**，**实体识别**和**标注**一举两得。然后把一个**关系**对应的所有句子打个包，称作一个**bag**,干脆一个**bag**一个**标签**。这就是后来又有的工作，被叫做**多示例学习**。

 

说到这就会发现这个方法有很多不严谨的地方，一是找句子时候，谁说含有中国和北京的句子全都是首都的关系，比如我说中国的面积比北京的面积大，就不是。在举个通用的例子，乔布斯是苹果的创始人，和乔布斯吃了一个苹果，表达的完全不是一个关系。这就说明远程监督的数据里存在大量的噪声，我们把真正含有指定关系的句子叫做real instance ，实际上不含任何关系的句子叫NA，其余的就都是反例。这个噪声问题被叫做wrong label 问题。这是远程监督方法第一个需要解决的大问题。

其次，给bag打标签时候根据什么打？按那个关系抽的句子就冠那个关系的名？那万一里面只有一个real sentence 其余全是反例或者NA咋办？所以给包打标签的策略也有很多种，最经典的就是当里面所有句子都是NA的时候，包也NA，里面只要有带关系的句子，就把占比最大的关系打给这个bag。然后拿去作为训练语料。其实也可以对bag里的句子做一些预处理来提高训练效果，比如两个实体在句子里的距离超过一个阈值就删掉，实体竟然是其他实体的子字符串，那不是瞎对应吗，也删掉。等等。

 

当然远程监督还有一些其他问题，比如你找不到你想做的领域的合适的KB。那这种情况我也不知道能有什么办法了，如果解决实际问题还需要具体问题具体分析，DS也不能处处适用。

那么综上，大家知道了什么是关系抽取 ，什么是远程监督。接下来就要面对刚才提到的那些挑战了，为了找到更优的关系抽取器。之后的工作主要也集中在对数据的降噪，对模型的改造上。

近些年陆续提出了很多优秀的方法，具体介绍在review里。

