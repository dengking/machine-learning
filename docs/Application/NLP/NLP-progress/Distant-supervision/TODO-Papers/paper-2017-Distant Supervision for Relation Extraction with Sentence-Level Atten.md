#  [Distant Supervision for Relation Extraction with Sentence-level Attention and Entity Descriptions](https://pdfs.semanticscholar.org/b8da/823ad81e3b8e5b80d82f86129fdb1d9132e7.pdf?_ga=2.78898385.436512546.1571998669-788894272.1569305268) 

## Abstract

**Distant supervision** for **relation extraction** is an efficient method to scale **relation extraction** to very large corpora which contains thousands of relations. However, the existing approaches have flaws on selecting **valid instances** and lack of background knowledge about the **entities**. In this paper, we propose a **sentence-level attention model** to select the **valid instances**, which makes full use of the **supervision information** from **knowledge bases**. And we extract **entity descriptions** from Freebase and Wikipedia pages to supplement background knowledge for our task. The background knowledge not only provides more information for predicting relations, but also brings better entity representations for the attention module. We conduct three experiments on a widely used dataset and the experimental results show that our approach outperforms all the baseline systems significantly.

***SUMMARY*** : 提出两个想法：

- a **sentence-level attention model** to select the **valid instances**
- extract **entity descriptions** from Freebase and Wikipedia pages to supplement background knowledge for our task

***SUMMARY*** : 如何理解**supervision information**？其实他就是后面所属的valid sentence，也就是正确描述了关系的句子；

## Introduction

Relation extraction (RE) under **distant supervision** aims to predict **semantic relations** between pairs of entities in texts supervised by **knowledge bases** (KBs). It heuristically（启发式地） **aligns** entities in texts to a given KB and uses this **alignment** to learn a **relation extractor**. The training data are labelled automatically as follows: for a triplet $r(e_1 ,e_2 )^1$ in the KB, all sentences that mention both entities $e_1$ and $e_2$ are regarded as the training instances of relation $r$. Figure 1 shows the training instances of triplet `/location/location/contains (Nevada, Las Vegas)`. The sentences from $S_1$ to $S_4$ all mention entities Nevada and Las Vegas, so they are all **training instances** of the **relation** `/location/location/contains`. The task is crucial for many Natural Language Processing (NLP) applications such as automatic knowledge completion and question-answering.

***SUMMARY*** : 在attention中，align是也经常被提及的词；

> 

Figure 1: Training instances of the triplet /location/location/contains (Nevada, Las Vegas). The low
part shows the descriptions of Nevada and Las Vegas.



**Distant supervision strategy** is an effective method of automatically labeling training data, however, it is plagued by the **wrong label problem** (Riedel,Yao,and McCallum 2010). A sentence that mentions two entities may not express the **relation** which links them in a KB. It is possible that the two entities may just appear in the same sentence because they are related to the same topic.  For example, in Figure 1, sentences $S_2$ and $S_4$ both mention **Nevada** and **Las Vegas**, but they do not express the relation `/location/location/contains`. Mintz et al., (2009) **ignored** the problem and extracted **features** from all（所有的，无论它们是否表达了这种关系） the sentences to feed a **relation classifier**. Riedel, Yao, and McCallum, (2010) proposed the $expressed-
at-least-once^2$（ If two entities participate in a **relation**, at least one sentence that mentions these two entities might express that relation.） assumption, and used an undirected graphical model to predict which sentences express the **relation**.  Based on the Multi-Instance Learning (Dietterich, Lathrop, and Lozano-Pérez 1997), Hoffmann et al., (2011) and Surdeanu et al., (2012) also used a probabilistic, graphical model to select sentences and added overlapping relations to their relation extraction systems.  Zeng et al., (2015) combined **multi-instance learning** (MIL) and **piecewise convolutional neuraln etworks** (PCNNs) to choose the most likely valid sentence and predict relations, which achieved state-of-the-art performance on the dataset developed by (Riedel, Yao, and McCallum 2010).



In **multi-instance learning** paradigm, for the triplet $r(e_1 ,e_2 )$, all the sentences which mention both $e_1$ and $e_2$ constitute a **bag** and the relation $r$ is the **label** of the bag. Although the above approaches have achieved high performance on RE under **distant supervision**, they have two main flaws. More specifically, 

(1) A bag may contain **multiple valid sentences**. For example, in Figure 1, sentences $S_1$ and $S_3$ both express the relation `/location/location/contains`. The probabilistic, graphical models (Riedel, Yao, and McCallum 2010; Hoffmann et al. 2011; Surdeanu et al. 2012) had considered the observation, but the features they designed to choose **valid sentences** are often derived from preexisting NLP tools which suffer from **error propagation** and accumulation (Bach and Badaskar 2007).  Zeng et al., (2015) extracted sentence features by **PCNNs** instead of relying on the traditional NLP tools and achieved state-of-the-art performance. However, in the learning process, its MIL module only selected one sentence which has the maximum probability to be a **valid candidate**. This strategy doesn’t make full use of the **supervision information**. Therefore, integrating the merits（优点） (considering multiple valid sentences and extracting features by neural networks) of the two approaches may be promising;

(2) The **entity descriptions**, which can provide helpful **background knowledge**, are useful resources
for our task. For example, in Figure 1, it’s difficult to decide which **relation** the sentence $S_1$ expresses without the information that Nevada is a state and Las Vegas is a city. When lacking the **background knowledge**, Nevada may be a government official’s name and  $S_1$  doesn’t express the relation
`/location/location/contains`. Therefore, the descriptions are beneficial for the task. Unfortunately, none of the existing work uses them for RE under **distant supervision**.

To select multiple valid sentences, we propose a **sentence-level attention model** based on **PCNNs** (denoted by **APCNNs**), which extracts **sentence features** using **PCNNs** and learns the weights of sentences by the **attention module**. We hope that the **attention mechanism** is able to selectively focus on the relevant sentences through assigning higher weights for **valid sentences** and lower weights for the **invalid ones**. In this way, **APCNNs** could recognize multiple **valid sentences** in a **bag**. Concretely, motivated by TransE (Bordes et al. 2013) which modeled a triplet $r(e_1 ,e_2 )$ with $e_1 + r  \approx  e_2$ (the bold, italic letters represent **vectors**), we use $(e_1 − e_2 )$ to represent the **relation** between $e_1$ and $e_2$ in sentences (we will show more explanations later). For a  **bag**, we first use **PCNNs** to extract each sentence’s feature vector $v_ {sen}$ , then compute the **attention weight** for each sentence through a **hidden layer** with the **concatenation（串联） way** $[ v_{sen} ;e_1 − e_2 ]$ (Luong, Pham, and Manning 2015). At last, the weighted sum of all sentence feature vectors is the bag’s features. In addition, to encode more **background knowledge** into our **model**, we use **convolutional neural networks**(CNNs)  to extract entity descriptions’ feature vectors and let them be close to the corresponding **entity vectors** via adding constraints on the **objective function** of **APCNNs** (called APCNNs+D, where “D” refers to descriptions). The **background knowledge** not only provides more information for predicting **relations**, but also brings better **entity representations** for the **attention module**.

Therefore, our main contributions in this paper are: 

(1)We introduce a **sentence-level attention model** to select multiple valid sentences in a bag. This strategy makes full use of the supervision information; 

(2) We use **entity descriptions** to provide **background knowledge** for predicting relations and improving entity representations; 

(3) We conduct experiments on a widely used $dataset^3$ and achieve state-of-the-art performance.



### Task Definition

In **multi-instance learning paradigm**, all sentences labeled by a triplet constitute a bag and each sentence is called an instance. Suppose that there are $N$ bags ${B_1 ,B_2 ,··· , B_N }$ in the training set and that the $i$-th bag contains $q_i$ instances $B_i = {b_1^i ,b_2^i  ,··· ,b_{q_i}^i } (i = 1,··· ,N)$. The objective of multi-instance learning is to predict the labels of the unseen **bags**. We need to learn a **relation extractor** based on the training data and then use it to predict relations for test set. Specifically, for a **bag**  in $B_j = {b_1^j ,b_2^j  ,··· ,b_{q_j}^j } (j = 1,··· ,N)$ training set, we need to extract features from the bag (from one or several valid instances) and then use them to train a classifier. For a **bag** in **test set**, we also need to extract features in the same way and use the classifier to predict the relation between the given entity pair.







### Methodology

In this section, we present the main innovative solutions including **sentence-level attention** and **entity descriptions**. **Sentence-level attention** makes our model be able to select multiple **valid instances** for training, so that we can make full use of the **supervision information**. **Entity descriptions** provide more **background knowledge** about the **entities**, which could improve the performance of our model and bring better **entity representations** for **attention module**. Figure 2 shows the neural network architecture of our model **APCNNs**. It consists of two parts: **PCNNs Module** and **Sentence-level Attention Module**. 

**PCNNs Module** includes 

- Vector Representation, 
- Convolution
- Piecewise Max-pooling. 

**Sentence-level Attention Module** is composed of 

- Attention Layer 
- Softmax Classifier.

We describe these parts in details below.

#### PCNNs Module

This module is used to extract **feature vector** of an instance (sentence) in a bag.



##### Vector Representation 



###### Word Embeddings



###### Position Embeddings







##### Convolution



##### Piecewise Max-pooling



#### Sentence-level Attention Module







### Experiments



#### Dataset





#### Evaluation Metrics



#### Experimental Results and Analysis





### Conclusions and Future Work