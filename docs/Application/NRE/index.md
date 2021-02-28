# Neural Relation Extraction (NRE)

This is my repository about **neural relation extraction**.

## distant supervision for relation extraction

### papers

| No | Time   | Paper                                                         | Pros                                                         | Cons                                                         |
| ---- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 0911 | 200911 | [Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) | 很多，比如：无需人工标注、成本低、大容量数据集、能够规避一些困扰监督学习的问题 | 1. 噪声大，性能不够强<br/>2. 需要人为设计特征                |
| ...  | ...    | ...                                                          | ...                                                          |                                                              |
| 15   | 2015   | [Distant supervisionfor relation extraction via piecewise convolutional neural networks.](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf) | 1. 使用**PCNNs**s神经网络选择最大概率为valid instance的句子来从中提取特征，不依赖于传统的NLP工具 | 1. 每个bag中仅仅选择一个句子（最大概率）作为valid instance，导致它未能充分利用bag中的信息 |
| 17   | 2017   | [Distant Supervision for Relation Extraction with Sentence-level Attention and Entity Descriptions](https://pdfs.semanticscholar.org/b8da/823ad81e3b8e5b80d82f86129fdb1d9132e7.pdf?_ga=2.78898385.436512546.1571998669-788894272.1569305268) | 1. bag中会考虑多个 valid Instance<br/>2. 由神经网络来提取特征 <br/>3. 提出**entity descriptions**思路 |                                                              |
| 1904 | 201904 | [Distant Supervision Relation Extraction with Intra-Bag and Inter-Bag Attentions](https://arxiv.org/pdf/1904.00143.pdf) | 1. 除了intra-bag（包内） attentions，还添加了inter-bag（包间） attentions |     
