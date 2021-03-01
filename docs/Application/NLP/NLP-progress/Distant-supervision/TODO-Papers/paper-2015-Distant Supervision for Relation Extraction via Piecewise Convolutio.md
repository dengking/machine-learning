# [Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf) 

##  Abstract 

Two problems arise when using **distant supervision** for **relation extraction**. First, in this method, an already existing knowledge base is heuristically aligned to texts, and the alignment results are treated as labeled data. However, the heuristic alignment can fail, resulting in wrong label problem. In addition, in previous approaches, statistical models have typically been applied to ad hoc（特定的） features. The noise that originates from the feature extraction process can cause poor performance. In this paper, we propose a novel model dubbed the **Piecewise Convolutional Neural Networks** (PCNNs) with **multi-instance learning** to address these two problems. To solve the first problem, distant supervised relation extraction is treated as a **multi-instance problem** in which the uncertainty of **instance labels** is taken into account. To address the latter problem, we avoid **feature engineering** and instead adopt **convolutional architecture with piecewise max pooling** to automatically learn relevant features. Experiments show that our method is effective and outperforms several competitive baseline methods. 

## 1 Introduction  

In relation extraction, one challenge that is faced when building a machine learning system is the generation of training examples. One common technique for coping with this difficulty is distant supervision (Mintz et al., 2009) which assumes that if two entities have a relationship in a known knowledge base, then all sentences that mention these two entities will express that relationship in some way. Figure 1 shows an example of the automatic labeling of data through distant supervision. In this example, Apple and Steve Jobs are two related entities in Freebase 1 . All sentences that contain these two entities are selected as training instances. The distant supervision strategy is an effective method of automatically labeling training data. However, it has two major shortcomings when used for relation extraction.

First, the **distant supervision assumption** is too strong and causes the wrong label problem. A sentence that mentions two entities does not necessarily express their relation in a knowledge base. It is possible that these two entities may simply share the same topic. For instance, the upper sentence indeed expresses the “company/founders” relation in Figure1. The lower sentence, however, does not express this relation but is still selected as a training instance. This will hinder the performance of a model trained on such noisy data.

Second, previous methods (Mintz et al., 2009; Riedel et al., 2010; Hoffmann et al., 2011) have typically applied supervised models to elaborately designed features when obtained the labeled data through distant supervision. These features are often derived from preexisting Natural Language Processing (NLP) tools. Since errors inevitably（不可避免的） exist in NLP tools, the use of traditional features leads to error propagation or accumulation. Distant supervised relation extraction generally addresses corpora from the Web, including many informal texts. Figure 2 shows the sentence length distribution of a benchmark distant supervision dataset that was developed by Riedel et al. (2010). Approximately half of the sentences are longer than 40 words. McDonald and Nivre (2007) showed that the accuracy of syntactic parsing decreases significantly with increasing sentence length. Therefore, when using traditional features, the problem of error propagation or accumulation will not only exist, it will grow more serious.

In this paper, we propose a novel model dubbed **Piecewise Convolutional Neural Networks** (PCNNs) with multi-instance learning to address the two problems described above. To address the first problem, **distant supervised relation extraction** is treated as a **multi-instance problem** similar to previous studies (Riedel et al., 2010; Hoffmann et al., 2011; Surdeanu et al., 2012). In **multi-instance problem**, the training set consists of many **bags**, and each contains many instances. The labels of the bags are known; however, the labels of the **instances** in the **bags** are unknown. We design an **objective function** at the **bag level**. In the learning process, the uncertainty of instance labels can be taken into account; this alleviates（减轻） the **wrong label problem**.

To address the second problem, we adopt **convolutional architecture** to automatically learn relevant features without complicated NLP preprocessing inspired by Zeng et al. (2014). Our proposal is an extension of Zeng et al. (2014), in which a single **max pooling** operation is utilized to determine the **most significant features**. Although this operation has been shown to be effective for textual feature representation (Collobert et al., 2011; Kim, 2014), it reduces the size of the **hidden layers** too rapidly and cannot capture the structural information between two entities (Graham, 2014). For example, to identify the relation between Steve Jobs and Applein Figure1, we need to specify the entities and extract the structural features between them. Several approaches have employed manually crafted features that attempt to model such **structural information**. These approaches usually consider both **internal** and **external contexts**. A sentence is inherently divided into **three segments** according to the two given entities. The **internal context** includes the characters inside the two entities, and the **external context** involves the characters around the two entities (Zhang et al., 2006). Clearly, **single max pooling** is not sufficient to capture such **structural information**. To capture structural and other latent（隐藏的，潜在的） information, we divide the convolution results into **three segments** based on the positions of the two given entities and devise a **piecewise max pooling layer** instead of the single max pooling layer. The piecewise max pooling procedure returns the maximum value in each segment instead of a single maximum value over the entire sentence. Thus, it is expected to exhibit superior performance compared with traditional methods.

The contributions of this paper can be summarized as follows.

- We explore the feasibility of performing **distant supervised relation extraction** without hand-designed features. **PCNNS** are proposed to automatically learn features without complicated NLP preprocessing.
- To address the wrong label problem, we develop innovative solutions that incorporate **multi-instance learning** into the PCNNS for **distant supervised relation extraction**.
-  In the proposed network, we devise a **piecewise max pooling layer**, which aims to capture structural information between two entities.

## 2 Related Work

**Relation extraction** is one of the most important topics in NLP. Many approaches to relation extraction have been developed, such as bootstrapping, unsupervised relation discovery and supervised classification. Supervised approaches are the most commonly used methods for relation extraction and yield relatively high performance (Bunescu and Mooney, 2006; Zelenko et al., 2003; Zhou et al., 2005). In the **supervised paradigm**, **relation extraction** is considered to be a **multi-class classification problem** and may suffer from a lack of labeled data for training. To address this problem, Mintz et al. (2009) adopted Freebase to perform distant supervision. As described in Section 1, the algorithm for training data generation is sometimes faced with the wrong label problem. To address this shortcoming, (Riedel et al., 2010; Hoffmann et al., 2011; Surdeanu et al., 2012) developed the relaxed distant supervision assumption for **multi-instance learning**. The term ‘multi-instance learning was coined by (Dietterich et al., 1997) while investigating the problem of predicting drug activity. In multi-instance learning, the uncertainty of instance labels can be taken into account. The focus of **multi-instance learning** is to discriminate among the bags.

These methods have been shown to be effective for relation extraction. However, their performance depends strongly on the quality of the designed features. Most existing studies have concentrated on extracting features to identify the relations between two entities. Previous methods can be generally categorized into two types: **feature-based methods** and **kernel-based methods**. In feature-based methods, a diverse set of strategies is exploited to convert classification clues (e.g., sequences, parse trees) into feature vectors (Kambhatla, 2004; Suchanek et al., 2006). Feature-based methods suffer from the necessity of selecting a suitable feature set when converting structured representations into feature vectors. Kernel-based methods provide a natural alternative to exploit rich representations of input classification clues, such as syntactic parse trees. Kernel-based methods enable the use of a large set of features without needing to extract them explicitly. Several kernels have been proposed, such as the convolution tree kernel(Qianetal., 2008), the subsequence kernel (Bunescu and Mooney, 2006) and the dependency tree kernel (Bunescu and Mooney, 2005).

Nevertheless, as mentioned in Section 1, it is difficult to design high-quality features using existing NLP tools. With the recent revival（复兴） of interest in neural networks, many researchers have investigated the possibility of using neural networks to automatically learn **features** (Socher et al., 2012; Zeng et al., 2014). Inspired by Zeng et al. (2014), we propose the use of **PCNNs** with **multi-instance learning** to automatically learn features for **distant supervised relation extraction**. Dietterich et al. (1997) suggested that the design of multi-instance modifications for neural networks is a particularly interesting topic. Zhang and Zhou (2006) successfully incorporated multi-instance learning into traditional Backpropagation (BP) and Radial Basis Function (RBF) networks and optimized these networks by minimizing a sum-of-squares error function. In contrast to their method, we define the **objective function** based on the cross-entropy principle.

## 3 Methodology

**Distant supervised relation extraction** is formulated as multi-instance problem. In this section, we present innovative solutions that incorporate **multi-instance learning** into a **convolutional neural network** to fulfill this task. **PCNNs** are proposed for the automatic learning of features without complicated NLP preprocessing. Figure 3 shows our **neural network architecture** for distant supervised relation extraction. It illustrates the procedure that handles one instance of a bag. This procedure includes four main parts: Vector Representation, Convolution, Piecewise Max Pooling and Softmax Output. We describe these parts in detail below.



![]()

Figure 3: The architecture of PCNNs (better viewed in color) used for distant supervised relation extrac-
tion, illustrating the procedure for handling one instance of a bag and predicting the relation between
Kojo Annan and Kofi Annan.





### 3.1 Vector Representation

The inputs of our network are raw word tokens. When using **neural networks**, we typically transform word tokens into **low-dimensional vectors**. In our method, each input word token is transformed into a vector by looking up **pre-trained word embeddings**. Moreover, we use position features (PFs) to specify entity pairs, which are also transformed into vectors by looking up **position embeddings**.

#### 3.1.1 Word Embeddings

**Word embeddings** are distributed representations of words that map each word in a text to a ‘k’-
dimensional real-valued vector. They have recently been shown to capture both semantic（语义） and syntactic（句法） information about words very well, setting performance records in several word similarity tasks (Mikolov et al., 2013; Pennington et al., 2014). Using **word embeddings** that have been trained a priori（先验） has become common practice for enhancing many other NLP tasks (Parikh et al., 2014; Huang et al., 2014).

 

A common method of training a neural network is to randomly initialize all parameters and then optimize them using an optimization algorithm. Recent research (Erhan et al., 2010) has shown that neural networks can converge to better **local minima** when they are initialized with **word embeddings**. **Word embeddings** are typically learned in an entirely unsupervised manner by exploiting the co-occurrence（同现） structure of words in unlabeled text. Researchers have proposed several methods of training **word embeddings** (Bengio et al., 2003; Collobert et al., 2011; Mikolov et al., 2013). In this paper, we use the Skip-gram model (Mikolov et al., 2013) to train **word embeddings**.

#### 3.1.2 Position Embeddings

In relation extraction, we focus on assigning labels to **entity pairs**. Similar to Zeng et al. (2014), we use **PFs** to specify entity pairs. A **PF** is defined as the combination of the relative distances from the current word to $e_1$ and $e_2$ . For instance, in the following example, the relative distances from son to $e_1$ (Kojo Annan) and $e_2$ (Kofi Annan) are 3 and -2, respectively.

Two **position embedding matrixes** ($PF_1$ and $PF_2$  ) are randomly initialized. We then transform the **relative distances** into real valued vectors by looking up the **position embedding matrixes**. In the example shown in Figure 3, it is assumed that the size of the word embedding is $d_w = 4$ and that the size of the **position embedding** is $d_p = 1$. In combined **word embeddings** and **position embeddings**, the **vector representation** part transforms an **instance** into a matrix $S \in  \mathbb R^{ s \times d}$ , where $s$ is the sentence length and $d = d_w + d_p ∗ 2$. The matrix $S$ is subsequently fed into the convolution part.

***SUMMARY*** : 将一个sentence转换为一个matrix；

### 3.2 Convolution

In relation extraction, an input sentence that is marked as containing the **target entities** corresponds only to a **relation type**; it does not predict **labels** for each word. Thus, it might be necessary to utilize all **local features** and perform this prediction globally. When using a neural network, the convolution approach is a natural means of merging all these features (Collobert et al., 2011).

**Convolution** is an operation between a vector of weights, **w**, and a vector of inputs that is treated as a sequence **q**. The weights matrix **w** is regarded as the **filter** for the convolution. In the example shown in Figure 3, we assume that the length of the **filter** is $w (w = 3)$; thus, **w**$ \in    \mathbb R^m (m = w \times d)$（$w \times d$表示的是$w$行$d$列，它表示$w$是一个矩阵）. We consider **S** to be a sequence $\{ \textbf q_1 , \textbf q_2 ,··· , \textbf q_s \}$, where $ \textbf q_i \in  \mathbb R^d$ . In general, let $q_{i:j}$ refer to the concatenation of $q_i$ to $q_j$ . The convolution operation involves taking the **dot product** of **w** with each $w$-gram in the sequence **q** to obtain another sequence $\textbf c \in  \mathbb R^{s+w−1}$
$$
c_j = \textbf w \textbf q_{j−w+1:j} \tag 1\\
$$
where the index $j$ ranges from 1 to $s+w−1$. Out-of-range input values $q_i$ , where $i < 1$ or $i > s$, are taken to be zero.

***SUMMARY*** : dot produce意味着结果是一个张量；

***SUMMARY*** :  注意，$j - (j - w + 1) = w$

***SUMMARY*** :  **filter**的长度$w$表示的是它一次扫描$w$个词

***SUMMARY*** : **c**是一个向量，它的长度是$s+w−1$，我猜测它的含义是filter沿着**q**进行滑动，第一次它仅仅吃下一个word，第二次吃下两个word，第三次吃下3个word，后面每一次都吃下3个word。

The ability to capture **different features** typically requires the use of multiple filters (or **feature maps**) in the **convolution**. Under the assumption that we use n filters ($\textbf W = \{\textbf  w_1 ,\textbf  w_2 ,··· ,\textbf w_n \} $), the convolution operation can be expressed as follows:
$$
c_{ij} = \textbf w_i \textbf q_{j−w+1:j} \space \space \space 1 ≤ i ≤ n \tag 2
$$
The convolution result is a matrix $C = \{\textbf c_1 ,\textbf c_2 ,··· ,\textbf c_n \} \in \mathbb R^{n \times (s+w−1)}$ . Figure 3 shows an example in which we use 3 different filters in the convolution procedure.

***SUMMARY*** : 这就是典型的[Conv1D](https://keras.io/layers/convolutional/)

### 3.3 Piecewise Max Pooling

The size of the convolution output matrix  $C \in \mathbb R^{n \times (s+w−1)}$ depends on the number of tokens $s$ in the sentence that is fed into the network.  To apply subsequent layers, the **features** that are extracted by the convolution layer must be **combined** such that they are independent of the **sentence length**. In traditional Convolution Neural Networks (CNNs), **max pooling operations** are often applied for this purpose (Collobert et al., 2011; Zeng et al., 2014). This type of pooling scheme naturally addresses variable sentence lengths. The idea is to capture the most significant features (with the highest values) in each **feature map**.

However, despite the widespread use of **single max pooling**, this approach is insufficient for **relation extraction**. As described in the first section, **single max pooling** reduces the size of the **hidden layers** too rapidly and is too coarse to capture fine-grained features for relation extraction. In addition, **single max pooling** is not sufficient to capture the **structural information** between two entities. In relation extraction, an input sentence can be divided into three segments based on the two selected entities. Therefore, we propose a **piecewise max pooling** procedure that returns the **maximum value** in each **segment** instead of a single maximum value. As shown in Figure 3, the output of each convolutional filter $c_i$ is divided into three segments ${c_{i1} ,c_{i2} ,c_{i3} }$ by Kojo Annan and KofiA nnan. The **piecewise max pooling** procedure can be expressed as follows:

$$
p_{ij}=max(c_{ij}) \space \space 1 ≤ i ≤ n  \space \space 1 ≤ j ≤ 3	\tag 3
$$
***SUMMARY*** : $p_{ij}$是一个张量

***SUMMARY*** : 实现上的一个问题，$C = {\textbf c_1 ,\textbf c_2 ,··· ,\textbf c_n } \in \mathbb R^{n \times (s+w−1)}$，显然$p_{ij} \in \mathbb R^{(s+w−1)}$，它的维度和输入的句子的维度并不相同，那它如何实现each convolutional filter $c_i$ is divided into three segments ${c_{i1} ,c_{i2} ,c_{i3} }$ by Kojo Annan and KofiA nnan？

For the output of each **convolutional filter**, we can obtain a 3-dimensional vector $ \textbf p_i =
\{p_{i1} ,p_{i2} ,p_{i3} \}$. We then concatenate all vectors $\textbf p_{1:n}$ and apply a non-linear function, such as the hyperbolic tangent. Finally, the piecewise max pooling procedure outputs a vector:
$$
\textbf g = tanh(\textbf p_{1:n}) \tag 4
$$
where $g \in R^{3n}$ . The size of $g$ is fixed and is no longer related to the sentence length.

***SUMMARY*** : $g \in R^{3n}$ 中3的含义是三段，$n$表示的是filter的个数；



### 3.4 Softmax Output

To compute the **confidence** of each relation, the **feature vector** $g$ is fed into a softmax classifier.
$$
\textbf o = \textbf W_1 \textbf g + b \tag 5
$$
$\textbf W_1 \in \mathbb R^{n1 \times 3n}$ is the transformation matrix, and $\textbf o \in  \mathbb R^{n1}$ is the **final output** of the network, where
$n_1$ is equal to the number of possible relation types for the relation extraction system.

We employ **dropout** (Hinton et al., 2012) on the penultimate（倒数第二） layer for regularization. Dropout prevents the co-adaptation of **hidden units** by randomly dropping out a proportion $p$ of the hidden units during forward computing. We first apply a “masking” operation $(\textbf g \circ \textbf 	r)$ on **g**, where **r** is a vector of Bernoulli random variables with probability $p$ of being 1. Eq.(5) becomes:
$$
\textbf o = \textbf W_1 (\textbf g \circ \textbf 	r)	 + b \tag 6
$$
Each output can then be interpreted as the **confidence score** of the corresponding relation. This score can be interpreted as a conditional probability by applying a softmax operation (see Section 3.5). In the test procedure, the learned weight vectors are scaled by $p$ such that $  \hat{\textbf  W_1} = p \textbf W_1$ and are used (without dropout) to score unseen instances.



### 3.5 Multi-instance Learning

In order to alleviate the wrong label problem, we use **multi-instance learning** for PCNNs. The PCNNs-based relation extraction can be stated as a quintuple $\theta = (\textbf E, \textbf {PF_1} , \textbf {PF_2} , \textbf W, \textbf W_1 )$（$\textbf E$ represents the word embeddings） . The input to the network is a **bag**. Suppose that there are T  bags ${M_1 ,M_2 ,··· ,M_T }$ and that the $i$-th bag contains $q_i$ instances $M_i = \{m_i^1 ,m_i^2  ,··· ,m_i^{q_i}\}$. The objective of multi-instance learning is to predict the labels of the unseen bags. In this paper, all instances in a bag are considered independently. Given an input instance $m_i^j$, the network with the parameter $\theta$ outputs a vector $\textbf o$, where the $r$-th component $o_r$ corresponds to the **score** associated with relation $r$.  To obtain the conditional probability $p(r \mid m,\theta)$, we apply a softmax operation over all relation types:
$$
p(r \mid m_i^j;\theta)=\frac {e^{o_r}} {\sum_{k=1}^{n1} e^{o_k}} \tag 7
$$
The objective of multi-instance learning is to discriminate bags rather than instances. To do so, we must define the objective function on the bags. Given all ($T$) training bags $(M_i ,y_i )$, we can define the **objective function** using cross-entropy at the bag level as follows:


$$
J(\theta)=\sum_{i=1}^T \log {p(y_i \mid m_i^j; \theta)} \tag 8
$$


where $j$ is constrained as follows:
$$
j^\star = \arg \max \limits_{w_1} p(y_i \mid m_i^j ; \theta) \space \space 1 ≤ j ≤ q_i \tag 9
$$

Using this defined objective function, we maximize $J(\theta)$ through stochastic gradient descent over shuffled mini-batches with the Adadelta (Zeiler, 2012) update rule. The entire training procedure is described in Algorithm 1.

>---
>
>Algorithm 1 Multi-instance learning
>
>---
>
>1: Initialize $\theta$. Partition the bags into mini-batches of size $b_s$ .
>2: Randomly choose a mini-batch, and feed the bags into the network one by one.
>3: Find the $j$-th instance $m_i^j \space (1 ≤ i ≤ b_s ) $ in each bag according to Eq. (9).
>4: Update $\theta$ based on the gradients of $m_i^j (1 ≤i ≤ b_s )$ via Adadelta.
>5: Repeat steps 2-4 until either convergence or the maximum number of epochs is reached.
>
>---

From the introduction presented above, we know that the traditional backpropagation algorithm modifies a network in accordance with all training instances, whereas backpropagation with multi-instance learning modifies a network based on bags. Thus, our method captures the nature of distant supervised relation extraction, in which some training instances will inevitably be incorrectly labeled. When a trained **PCNN** is used for prediction, a bag is positively labeled if and only if the output of the network on at least one of its
instances is assigned a positive label.

***SUMMARY*** : 在模型进行使用的时候，难道也是按照bag的方式来进行输入的？然后从每个bag中选择出一个valid instance；

## 4 Experiments

Our experiments are intended to provide evidence that supports the following hypothesis: automat-
ically learning features using PCNNs with multiinstance learning can lead to an increase in performance. To this end, we first introduce the dataset and evaluation metrics used. Next, we test several variants via cross-validation to determine the parameters to be used in our experiments. We then compare the performance of our method to those of several traditional methods. Finally, we evaluate the effects of piecewise max pooling and multi-instance learning$^3$ .

> With regard to the position feature, our experiments yield the same positive results described in Zeng et al. (2014). Because the position feature is not the main contribution of this paper, we do not present the results without the position feature.