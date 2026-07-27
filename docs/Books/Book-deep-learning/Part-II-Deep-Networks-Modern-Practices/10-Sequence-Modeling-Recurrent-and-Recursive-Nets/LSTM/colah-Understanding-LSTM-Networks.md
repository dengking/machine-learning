# [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Recurrent Neural Networks

Humans don’t start their thinking from scratch every second. As you read this essay, you understand each word based on your understanding of previous words. You don’t throw everything away and start thinking from scratch again. Your thoughts have persistence.

Traditional neural networks can’t do this, and it seems like a major shortcoming. For example, imagine you want to classify what kind of event is happening at every point in a movie. It’s unclear how a traditional neural network could use its reasoning about previous events in the film to inform later ones.

**Recurrent neural networks** address this issue. They are networks with **loops** in them, allowing information to persist.

![img](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png)

**Recurrent Neural Networks have loops.**

In the above diagram, a chunk of neural network, $A$, looks at some input $x_t$ and outputs a value $h_t$. A loop allows information to be passed from one step of the network to the next.

These loops make **recurrent neural networks** seem kind of mysterious. However, if you think a bit more, it turns out that they aren’t all that different than a normal neural network. A **recurrent neural network** can be thought of as multiple copies of the same network, each passing a message to a successor. Consider what happens if we unroll the loop:


![An unrolled recurrent neural network.](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

**An unrolled recurrent neural network.**

This chain-like nature reveals that **recurrent neural networks** are intimately（亲切的） related to sequences and lists. They’re the natural architecture of neural network to use for such data.

And they certainly are used! In the last few years, there have been incredible success applying RNNs to a variety of problems: speech recognition, language modeling, translation, image captioning… The list goes on. I’ll leave discussion of the amazing feats（功勋） one can achieve with RNNs to Andrej Karpathy’s excellent blog post, [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). But they really are pretty amazing.

Essential to these successes is the use of “LSTMs,” a very special kind of recurrent neural network which works, for many tasks, much much better than the standard version. Almost all exciting results based on recurrent neural networks are achieved with them. It’s these LSTMs that this essay will explore.



## The Problem of Long-Term Dependencies

One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task, such as using previous video frames might inform the understanding of the present frame. If RNNs could do this, they’d be extremely useful. But can they? It depends.

Sometimes, we only need to look at recent information to perform the **present task**. For example, consider a language model trying to predict the next word based on the previous ones. If we are trying to predict the last word in “the clouds are in the *sky*,” we don’t need any further context – it’s pretty obvious the next word is going to be sky. In such cases, where the **gap** between the relevant information and the place that it’s needed is small, RNNs can learn to use the past information.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-shorttermdepdencies.png)

***SUMMARY*** : $[X_0, X_1, X_2, X_3, X_4]$就是一个time_step的数据，显然LSTM cell中的参数还是和它特征有关的；

But there are also cases where we need more context. Consider trying to predict the last word in the text “I grew up in France… I speak fluent *French*.” Recent information suggests that the next word is probably the name of a language, but if we want to narrow down which language, we need the context of France, from further back. It’s entirely possible for the **gap** between the relevant information and the point where it is needed to become very large.

Unfortunately, as that **gap** grows, RNNs become unable to learn to connect the information.

![Neural networks struggle with long term dependencies.](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-longtermdependencies.png)

In theory, RNNs are absolutely capable of handling such “long-term dependencies.” A human could carefully pick parameters for them to solve toy problems of this form. Sadly, in practice, RNNs don’t seem to be able to learn them. The problem was explored in depth by [Hochreiter (1991) [German\]](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf) and [Bengio, et al. (1994)](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf), who found some pretty fundamental reasons why it might be difficult.

***SUMMARY*** : 所谓的long-term dependency其实就是gap太长了。

Thankfully, LSTMs don’t have this problem!



## LSTM Networks

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by [Hochreiter & Schmidhuber (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf), and were refined（改善） and popularized by many people in following work.[1](http://colah.github.io/posts/2015-08-Understanding-LSTMs/#fn1) They work tremendously（非常的） well on a large variety of problems, and are now widely used.

LSTMs are explicitly designed to avoid the **long-term dependency problem**. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single **`tanh` layer**.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)The repeating module in a standard RNN contains a single layer.

LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are **four**, interacting in a very special way.

![A LSTM neural network.](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

The repeating module in an LSTM contains four interacting layers.

Don’t worry about the details of what’s going on. We’ll walk through the LSTM diagram step by step later. For now, let’s just try to get comfortable with the notation we’ll be using.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM2-notation.png)



In the above diagram, each line carries（携带） an entire vector, from the output of one node to the inputs of others. The pink circles represent pointwise operations, like vector addition, while the yellow boxes are **learned neural network layers**. Lines merging denote concatenation, while a line forking denote its content being copied and the copies going to different locations.

***SUMMARY*** : 上图其实是非常类似于TensorFlow中的computational graph的：使用节点来表示operation，使用边来表示tensor的流动；

***SUMMARY*** : 上面这段话中提及： the yellow boxes are **learned neural network layers**. 即上述yellow box是模型需要学习的，在MLP中，model需要学习的是weight和bias，那在LSTM中，是否也有weight和bias呢？这些是可以有的，参看deep learning book chapter 10.2节中的图10.3可以看出，是存在weight的；关于这个问题，还可以参看如下内容：

- [How to interpret weights in a LSTM layer in Keras [closed\]](https://stackoverflow.com/questions/42861460/how-to-interpret-weights-in-a-lstm-layer-in-keras)
- [How can calculate number of weights in LSTM](https://stats.stackexchange.com/questions/226593/how-can-calculate-number-of-weights-in-lstm)
- [Reading between the layers (LSTM Network)](https://towardsdatascience.com/reading-between-the-layers-lstm-network-7956ad192e58)
- https://www.sciencedirect.com/science/article/pii/S187705091831439X/pdf?md5=dc105bed63a63592d9b0a0ff6af56553&pid=1-s2.0-S187705091831439X-main.pdf
- https://fairyonice.github.io/Extract-weights-from-Keras's-LSTM-and-calcualte-hidden-and-cell-states.html

deep learning book chapter 10.10中给出的公式也是包含有weight和bias的；

## The Core Idea Behind LSTMs

The key to LSTMs is the ***cell state***, the horizontal line running through the top of the diagram.

***SUMMARY*** : 在[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf?)中，将cell state称之为memory，其实这应该就是和LSTM中M对应的；

The **cell state** is kind of like a conveyor belt（传送带）. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png)

SUMMARY：**cell state**即上图中的$C$

The LSTM does have the ability to remove or add information to the **cell state**, carefully regulated（调整） by structures called **gates**.

**Gates** are a way to optionally let information through. They are composed out of a **sigmoid neural net layer** and a pointwise multiplication operation.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-gate.png)

The **sigmoid layer** outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”

***SUMMARY*** : 上面所描述的是gate的实现方式；

An LSTM has three of these gates, to protect and control the **cell state**.



## Step-by-Step LSTM Walk Through



### forget gate layer

The first step in our LSTM is to decide what information we’re going to throw away from the **cell state**. This decision is made by a **sigmoid layer** called the “**forget gate layer**.” It looks at $h_{t-1}$ and $x_t$, and outputs a number between 0 and  1 for each number in the cell state $C_{t−1}$. A  `1` represents “completely keep this” while a `0` represents “completely get rid of this.”

Let’s go back to our example of a language model trying to predict the next word based on all the previous ones. In such a problem, the **cell state** might include the gender of the present subject, so that the correct pronouns can be used. When we see a new subject, we want to forget the gender of the old subject.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png)

$$
f_t=\sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$


### input gate layer

The next step is to decide what new information we’re going to store in the **cell state**. This has two parts. First, a **sigmoid layer** called the “**input gate layer**” decides which values we’ll update. Next, a `tanh` layer creates a vector of new candidate values, $\tilde{C}_t$, that could be added to the state. In the next step, we’ll combine these two to create an update to the state（表示的是$\tilde{C}_t$）.

In the example of our language model, we’d want to add the gender of the new subject to the **cell state**, to replace the old one we’re forgetting.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png)

### update cell state

It’s now time to update the old cell state, $C_{t−1}$, into the new cell state $C_t$. The previous steps already decided what to do, we just need to actually do it.

We multiply the old state by $f_t$, forgetting the things we decided to forget earlier. Then we add $i_t*\tilde{C}_t$. This is the new candidate values, scaled by how much we decided to update each state value.

In the case of the language model, this is where we’d actually drop the information about the old subject’s gender and add the new information, as we decided in the previous steps.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png)



$$
C_t = f_t * C_{t-1} + i_t*\tilde{C}_t
$$


### output

Finally, we need to decide what we’re going to output. This output will be based on our **cell state**, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the **cell state** we’re going to output. Then, we put the cell state through  $tanh$ (to push the values to be between  `1`) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.

For the language model example, since it just saw a subject, it might want to output information relevant to a verb, in case that’s what is coming next. For example, it might output whether the subject is singular or plural, so that we know what form a verb should be conjugated into if that’s what follows next.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png)





***SUMMARY*** : 上述表示非常类似于TensorFlow中的computational graph：使用节点来表示operation，使用edge来表示flow of  tensor；

## Variants on Long Short Term Memory

What I’ve described so far is a pretty normal LSTM. But not all LSTMs are the same as the above. In fact, it seems like almost every paper involving LSTMs uses a slightly different version. The differences are minor, but it’s worth mentioning some of them.

One popular LSTM variant, introduced by [Gers & Schmidhuber (2000)](ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf), is adding “peephole connections.” This means that we let the gate layers look at the cell state.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-peepholes.png)

The above diagram adds peepholes to all the gates, but many papers will give some peepholes and not others.

Another variation is to use coupled forget and input gates. Instead of separately deciding what to forget and what we should add new information to, we make those decisions together. We only forget when we’re going to input something in its place. We only input new values to the state when we forget something older.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-tied.png)

A slightly more dramatic variation on the LSTM is the Gated Recurrent Unit, or GRU, introduced by [Cho, et al. (2014)](http://arxiv.org/pdf/1406.1078v3.pdf). It combines the forget and input gates into a single “update gate.” It also merges the cell state and hidden state, and makes some other changes. The resulting model is simpler than standard LSTM models, and has been growing increasingly popular.

![A gated recurrent unit neural network.](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png)

These are only a few of the most notable LSTM variants. There are lots of others, like Depth Gated RNNs by [Yao, et al. (2015)](http://arxiv.org/pdf/1508.03790v2.pdf). There’s also some completely different approach to tackling long-term dependencies, like Clockwork RNNs by [Koutnik, et al. (2014)](http://arxiv.org/pdf/1402.3511v1.pdf).

Which of these variants is best? Do the differences matter? [Greff, et al. (2015)](http://arxiv.org/pdf/1503.04069.pdf) do a nice comparison of popular variants, finding that they’re all about the same. [Jozefowicz, et al. (2015)](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)tested more than ten thousand RNN architectures, finding some that worked better than LSTMs on certain tasks.

## Conclusion

Earlier, I mentioned the remarkable results people are achieving with RNNs. Essentially all of these are achieved using LSTMs. They really work a lot better for most tasks!

Written down as a set of equations, LSTMs look pretty intimidating. Hopefully, walking through them step by step in this essay has made them a bit more approachable.

LSTMs were a big step in what we can accomplish with RNNs. It’s natural to wonder: is there another big step? A common opinion among researchers is: “Yes! There is a next step and it’s attention!” The idea is to let every step of an RNN pick information to look at from some larger collection of information. For example, if you are using an RNN to create a caption describing an image, it might pick a part of the image to look at for every word it outputs. In fact, [Xu, *et al.*(2015)](http://arxiv.org/pdf/1502.03044v2.pdf) do exactly this – it might be a fun starting point if you want to explore attention! There’s been a number of really exciting results using attention, and it seems like a lot more are around the corner…

Attention isn’t the only exciting thread in RNN research. For example, Grid LSTMs by [Kalchbrenner, *et al.* (2015)](http://arxiv.org/pdf/1507.01526v1.pdf) seem extremely promising. Work using RNNs in generative models – such as [Gregor, *et al.* (2015)](http://arxiv.org/pdf/1502.04623.pdf), [Chung, *et al.* (2015)](http://arxiv.org/pdf/1506.02216v3.pdf), or [Bayer & Osendorfer (2015)](http://arxiv.org/pdf/1411.7610v3.pdf) – also seems very interesting. The last few years have been an exciting time for recurrent neural networks, and the coming ones promise to only be more so!