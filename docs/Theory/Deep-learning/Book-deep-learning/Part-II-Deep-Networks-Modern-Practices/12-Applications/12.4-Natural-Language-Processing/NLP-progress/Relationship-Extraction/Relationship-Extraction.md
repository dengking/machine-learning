# [Relationship Extraction](http://nlpprogress.com/english/relationship_extraction.html)

**Relationship extraction** is the task of extracting **semantic relationships** from a text. Extracted relationships usually occur between two or more **entities** of a certain type (e.g. Person, Organisation, Location) and fall into a number of semantic categories (e.g. married to, employed by, lives in). 

***SUMMARY*** : entity是否需要特别地进行标注？

### Capturing discriminative attributes (SemEval 2018 Task 10)

捕捉有区别性的属性

**Capturing discriminative attributes (SemEval 2018 Task 10)** is a binary classification task where participants were asked to identify whether an attribute could help discriminate between two concepts. Unlike other word similarity prediction tasks, this task focuses on the semantic differences between words. 

 e.g. red(attribute) can be used to discriminate apple (concept1) from banana (concept2) -> label 1 

### FewRel

The Few-Shot Relation Classification Dataset (FewRel) is a different setting from the previous datasets. This dataset consists of 70K sentences expressing 100 relations annotated by crowdworkers on Wikipedia corpus. The few-shot learning task follows the N-way K-shot meta learning setting. It is both the largest supervised relation classification dataset as well as the largest few-shot learning dataset till now.

The public leaderboard is available on the [FewRel website](http://www.zhuhao.me/fewrel/).





### Multi-Way Classification of Semantic Relations Between Pairs of Nominals (SemEval 2010 Task 8)

[SemEval-2010](http://www.aclweb.org/anthology/S10-1006) introduced ‘Task 8 - Multi-Way Classification of Semantic Relations Between Pairs of Nominals’. The task is, given a sentence and two tagged nominals, to predict the relation between those nominals *and* the direction of the relation. The dataset contains nine general semantic relations together with a tenth ‘OTHER’ relation.

Example:

> There were apples, **pears** and oranges in the **bowl**.

```
(content-container, pears, bowl)
```

The main evaluation metric used is macro-averaged F1, averaged across the nine proper relationships (i.e. excluding the OTHER relation), taking directionality of the relation into account.

Several papers have used additional data (e.g. pre-trained word embeddings, WordNet) to improve performance. The figures reported here are the highest achieved by the model using any external resources.

#### End-to-End Models

| Model                                                 | F1                                                           | Paper / Source                                               | Code                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| *BERT-based Models*                                   |                                                              |                                                              |                                                              |
| Matching-the-Blanks (Baldini Soares et al., 2019)     | **89.5**                                                     | [Matching the Blanks: Distributional Similarity for Relation Learning](https://www.aclweb.org/anthology/P19-1279) |                                                              |
| R-BERT (Wu et al. 2019)                               | **89.25**                                                    | [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284) |                                                              |
| *CNN-based Models*                                    |                                                              |                                                              |                                                              |
| Multi-Attention CNN (Wang et al. 2016)                | **88.0**                                                     | [Relation Classification via Multi-Level Attention CNNs](http://aclweb.org/anthology/P16-1123) | [lawlietAi’s Reimplementation](https://github.com/lawlietAi/relation-classification-via-attention-model) |
| Attention CNN (Huang and Y Shen, 2016)                | 84.3 85.9[*](http://nlpprogress.com/english/relationship_extraction.html#footnote) | [Attention-Based Convolutional Neural Network for Semantic Relation Extraction](http://www.aclweb.org/anthology/C16-1238) |                                                              |
| CR-CNN (dos Santos et al., 2015)                      | 84.1                                                         | [Classifying Relations by Ranking with Convolutional Neural Network](https://www.aclweb.org/anthology/P15-1061) | [pratapbhanu’s Reimplementation](https://github.com/pratapbhanu/CRCNN) |
| CNN (Zeng et al., 2014)                               | 82.7                                                         | [Relation Classification via Convolutional Deep Neural Network](http://www.aclweb.org/anthology/C14-1220) | [roomylee’s Reimplementation](https://github.com/roomylee/cnn-relation-extraction) |
| *RNN-based Models*                                    |                                                              |                                                              |                                                              |
| Entity Attention Bi-LSTM (Lee et al., 2019)           | **85.2**                                                     | [Semantic Relation Classification via Bidirectional LSTM Networks with Entity-aware Attention using Latent Entity Typing](https://arxiv.org/abs/1901.08163) | [Official](https://github.com/roomylee/entity-aware-relation-classification) |
| Hierarchical Attention Bi-LSTM (Xiao and C Liu, 2016) | 84.3                                                         | [Semantic Relation Classification via Hierarchical Recurrent Neural Network with Attention](http://www.aclweb.org/anthology/C16-1119) |                                                              |
| Attention Bi-LSTM (Zhou et al., 2016)                 | 84.0                                                         | [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://www.aclweb.org/anthology/P16-2034) | [SeoSangwoo’s Reimplementation](https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction) |
| Bi-LSTM (Zhang et al., 2015)                          | 82.7 84.3[*](http://nlpprogress.com/english/relationship_extraction.html#footnote) | [Bidirectional long short-term memory networks for relation classification](http://www.aclweb.org/anthology/Y15-1009) |                                                              |

 *: It uses external lexical resources, such as WordNet, part-of-speech tags, dependency tags, and named entity tags. 