

# Introduction
Organize all carefully read papers, all papers are sorted by subject and time.

## RNN

### GRU

| 编号 | 时间   | 论文                                                         | 作者                                                         | 领域 | 评价 | code |
| ---- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- | ---- | ---- |
| 1412 | 201412 | [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555) | [Junyoung Chung](https://arxiv.org/search/cs?searchtype=author&query=Chung%2C+J)，<br/>[Bengio, Yoshua](http://xueshu.baidu.com/s?wd=author%3A(Bengio%2C Yoshua) &tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_f_para=sc_hilight%3Dperson) |      |      |      |

### Encoder–Decoder architecture

| 编号 | 时间   | 论文                                                         | 作者                                                         | 领域                    | 评价                                                         | code |
| ---- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------- | ------------------------------------------------------------ | ---- |
| 1406 | 201406 | [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) | [Kyunghyun Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+K)，<br/>[Cho, Kyunghyun](http://xueshu.baidu.com/s?wd=author%3A(Cho%2C Kyunghyun) &tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_f_para=sc_hilight%3Dperson)，<br/>[Bengio, Yoshua](http://xueshu.baidu.com/s?wd=author%3A(Bengio%2C Yoshua) &tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_f_para=sc_hilight%3Dperson) | NLP-machine translation | 首次提出了RNN Encoder–Decoder；<br/>补充：<br/> - [Neural machine translation by jointly learning to align and translate](https://kobiso.github.io/research/research-multi-neural-machine-translation/) <br/>- employing attention in machine translation<br/>- soft alignment |      |
| 1409 | 201409 | [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) | Google                                                       | NLP-machine translation | 这篇论文参考了[Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)，所不同的是，它使用的是LSTM； |      |
### blog

- [Encoder-Decoder Recurrent Neural Network Models for Neural Machine Translation](https://machinelearningmastery.com/encoder-decoder-recurrent-neural-network-models-neural-machine-translation/)

## attention



| 编号 | 时间   | 论文                                                         | 作者                                                         | 领域                    | 评价                                                         | code                                                         |
| ---- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1409 | 201409 | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473v7) | [Bahdanau, Dzmitry](http://xueshu.baidu.com/s?wd=author%3A(Bahdanau%2C Dzmitry) &tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_f_para=sc_hilight%3Dperson)，<br/>[Cho, Kyunghyun](http://xueshu.baidu.com/s?wd=author%3A(Cho%2C Kyunghyun) &tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_f_para=sc_hilight%3Dperson)，<br/>[Bengio, Yoshua](http://xueshu.baidu.com/s?wd=author%3A(Bengio%2C Yoshua) &tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_f_para=sc_hilight%3Dperson) | NLP-machine translation | 首次提出了attention mechanism；<br/>补出：<br/> - [Neural machine translation by jointly learning to align and translate](https://kobiso.github.io/research/research-multi-neural-machine-translation/) <br/>- employing attention in machine translation<br/>- soft alignment | [text_classification](https://github.com/brightmart/text_classification)/[a06_Seq2seqWithAttention](https://github.com/brightmart/text_classification/tree/master/a06_Seq2seqWithAttention)/ |
| 1412 | 201412 | [MULTIPLE OBJECT RECOGNITION WITH VISUAL ATTENTION](https://arxiv.org/abs/1412.7755) | Google DeepMind <br/>University of Toronto                   |                         | - employing attention in OBJECT RECOGNITION                  |                                                              |
| 1502 | 201502 | [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf?) | [Kelvin Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+K), <br/>[Jimmy Ba](https://arxiv.org/search/cs?searchtype=author&query=Ba%2C+J), [Ryan Kiros](https://arxiv.org/search/cs?searchtype=author&query=Kiros%2C+R), [Kyunghyun Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+K), [Aaron Courville](https://arxiv.org/search/cs?searchtype=author&query=Courville%2C+A), [Ruslan Salakhutdinov](https://arxiv.org/search/cs?searchtype=author&query=Salakhutdinov%2C+R), [Richard Zemel](https://arxiv.org/search/cs?searchtype=author&query=Zemel%2C+R), <br/>[Yoshua Bengio](https://arxiv.org/search/cs?searchtype=author&query=Bengio%2C+Y) | image caption           | 比较容易阅读，容易理解，可以作为了解attention的入门读物<br/> - 受1409和1412的启发，将attention mechanism应用于generating its caption<br/>- 提出了soft attention和hard attention | [kelvinxu](https://github.com/kelvinxu)/**[arctic-captions](https://github.com/kelvinxu/arctic-captions)** |
| 1601 | 201601 | [Long Short-Term Memory-Networks for Machine Reading](https://arxiv.org/pdf/1601.06733.pdf) | Jianpeng Cheng<br/>University of Edinburgh                   |                         | Self-Attention                                               |                                                              |
| 1706 | 201706 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Google Brain                                                 |                         | 首次提出了transformer--一种新的model architecture<br/>补充：<br/>- [《attention is all you need》解读](https://zhuanlan.zhihu.com/p/34781297) | - [huggingface](https://github.com/huggingface)/[transformers](https://github.com/huggingface/transformers)<br/>- [models](https://github.com/tensorflow/models)/[official](https://github.com/tensorflow/models/tree/master/official)/**transformer**/ |
|      |        |                                                              |                                                              |                         |                                                              |                                                              |



## NLP

| 编号 | 时间   | 论文                                                         | 作者         | 评价                                                         |
| ---- | ------ | ------------------------------------------------------------ | ------------ | ------------------------------------------------------------ |
| 1810 | 201810 | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) | Google Brain | 补出<br/>- [TensorFlow code and pre-trained models for BERT](https://github.com/google-research/bert) |
|      |        |                                                              |              |                                                              |
|      |        |                                                              |              |                                                              |

### distant supervision for relation extraction

| 标号 | 年份   | 论文                                                         | 优点                                                         | 弱点                                                         |
| ---- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 0911 | 200911 | [Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) | 很多，比如：无需人工标注、成本低、大容量数据集、能够规避一些困扰监督学习的问题 | 1. 噪声大，性能不够强<br/>2. 需要人为设计特征                |
| ...  | ...    | ...                                                          | ...                                                          |                                                              |
| 15   | 2015   | [Distant supervisionfor relation extraction via piecewise convolutional neural networks.](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf) | 1. 使用**PCNNs**s神经网络选择最大概率为valid instance的句子来从中提取特征，不依赖于传统的NLP工具 | 1. 每个bag中仅仅选择一个句子（最大概率）作为valid instance，导致它未能充分利用bag中的信息 |
| 17   | 2017   | [Distant Supervision for Relation Extraction with Sentence-level Attention and Entity Descriptions](https://pdfs.semanticscholar.org/b8da/823ad81e3b8e5b80d82f86129fdb1d9132e7.pdf?_ga=2.78898385.436512546.1571998669-788894272.1569305268) | 1. bag中会考虑多个 valid Instance<br/>2. 由神经网络来提取特征 <br/>3. 提出**entity descriptions**思路 |                                                              |
| 1904 | 201904 | [Distant Supervision Relation Extraction with Intra-Bag and Inter-Bag Attentions](https://arxiv.org/pdf/1904.00143.pdf) | 1. 除了intra-bag（包内） attentions，还添加了inter-bag（包间） attentions |                                                              |



## memory network

| 编号 | 时间   | 论文                                                         | 作者                                                         | 评价                 |
| ---- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------- |
| 1410 | 201410 | [Memory Networks](https://arxiv.org/abs/1410.3916)           | [Jason Weston](https://arxiv.org/search/cs?searchtype=author&query=Weston%2C+J), [Sumit Chopra](https://arxiv.org/search/cs?searchtype=author&query=Chopra%2C+S), [Antoine Bordes](https://arxiv.org/search/cs?searchtype=author&query=Bordes%2C+A) | 首次提出memory model |
| 1503 | 201503 | [End-To-End Memory Networks](https://arxiv.org/abs/1503.08895) | [Sainbayar Sukhbaatar](https://arxiv.org/search/cs?searchtype=author&query=Sukhbaatar%2C+S), [Arthur Szlam](https://arxiv.org/search/cs?searchtype=author&query=Szlam%2C+A), [Jason Weston](https://arxiv.org/search/cs?searchtype=author&query=Weston%2C+J), [Rob Fergus](https://arxiv.org/search/cs?searchtype=author&query=Fergus%2C+R) |                      |
|      |        |                                                              |                                                              |                      |