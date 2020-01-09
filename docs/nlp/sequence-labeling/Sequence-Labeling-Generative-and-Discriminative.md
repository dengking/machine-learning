[TOC]



# [Sequence Labeling: Generative and Discriminative Approaches](http://www.icmla-conference.org/icmla10/CFP_Tutorial_files/hakan.pdf)

## Sequence labeling problem definition I

Given a sequence of **observations/feature** vectors, determine an appropriate **label/state** for each **observation**

We will assume the observations can be discrete or continuous, scalar or vector

We assume the labels/states are discrete and from a finite set

Try to reduce errors by considering the **relations** between the **observations** and **states** (observation-state) and the **relation** between **neighboring states** (state-state)	

## Sequence labeling applications

- Speech recognition 
- Part-of-speech tagging 
- Shallow parsing 
- Handwriting recognition 
- Protein secondary structure prediction 
- Video analysis 
- Facial expression dynamic modeling



## Urns and balls example

Assume there are two urns with black and white balls [Rabiner, 1989]. One urn has more black than white (90% vs 10%) and vice versa. Someone pulls out one ball at a time and shows us without revealing which urn he uses and puts it back into the urn. He is more likely to use the same urn (90% chance) once he starts using one We are looking only at the sequence of balls and recording them.

## Questions about the urns and balls example

Questions of interest: 

1. Can we predict which urn is used at a given time?
2. What is the probability of observing the sequence of balls shown to us?
3. Can we estimate/learn the ratio of balls in each urn by looking at a long sequence of balls if we did not know the ratios beforehand?



## Jason Eisner’s ice-cream example

Try to guess whether the weather was hot or cold by observing only how many ice-creams (0, 1, 2 or 3+) Jason ate each day in a sequence of 30 days. Two states and observations with 4 distinct values (discrete observations). Question: Can we determine if a day was hot or cold given the sequence of ice-creams consumed by Jason? Example excel sheet online (illustrates forward backward algorithm). Example also adopted in [Jurafsky and Martin, 2008]



## Approach, notation and variables

We will first analyze binary and multi-class classification with linear models. **Multi-class classification** will be the basis for understanding the **sequence labeling problem**. Then, we will introduce HMM, CRF, and structured SVM approaches for sequence labeling.

Notation:

$x$ is an observed feature vector, $x_t$ a feature vector at sequence position $t$, $x_{1:T}$ a sequence of feature vectors. $y$ is a discrete label (or state), $y \in Y$ where $Y = {−1, +1}$ for binary classification, $Y = [M] = {1, 2, . . . , M}$ for multi-class classification. $y_t$ is the label/state at sequence position $t$, $y_{1:T}$ is a sequence of labels/states $w$ and w˜ are parameter vectors,$w_j$ is the $j$th component. $F(x_{1:T} , y_{1:T} )$ is a feature vector for CRF and structured SVM