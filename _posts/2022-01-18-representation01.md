---
layout: page
title: Memorization capacity
categories: representation
date: 2022-01-18
---

Let us begin by trying to mathematically frame a very simple question:

~~~
How many neurons are enough?
~~~

Of course, this question is not well-posed. Enough for what? For good performance? On what task? Do we even know if this answer is well-defined -- perhaps it depends on some hard-to-estimate quantity related to the learning problem?

Let us begin simple. Again, as before, suppose all we have at our disposal is a bunch of training data points:
\\[ X = \{(x_i, y_i)\}_{i=1}^n \subset \mathbb{R}^d \times \mathbb{R} \\]
and our goal will be to **exactly** fit a neural network to $X$. That is, we will lean a function $f$ that, when evaluated on every data point $x_i$ in the training data, returns $y_i$. Thus, $f$ learns to *memorize* the data. Equivalently, if we define empirical risk via the squared error loss:

$$
l(y,\hat{y}) = 0.5(y - \hat{y})^2
$$

then we seek an $f$ that achieves *zero* risk.

But why should memorization be interesting? After all, machine learning folks are taught to be wary of [overfitting](https://en.wikipedia.org/wiki/Overfitting) to the training set. In introductory ML courses we spend several hours (and homework sets) covering the [bias-variance](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) tradeoff, the importance of adding a regularizer to decrease variance (at the expense of incurring extra "bias"), etc etc.

Unfortunately, deep learning practice throws this classical way of ML thinking out of the window. We seldom use explicit regularizers, instead relying on standard losses. We typically train deep neural networks to achieve 100\% (train) accuracy. Later, we will try to understand why networks trained to *perfectly* interpolate the training data still generalize well, but for now let's focus on just achieving a representation that enables perfect memorization.

## Basics

First, we define what we mean by a "neural network".

A neural network is composed of several primitive "units", each of which we will call a *neuron*. Given an input vector $x \in \mathbb{R}^d$, a neuron transforms the input according to the following functional form:

$$
z = \psi(\sum_{j=1}^d w_j x_j + b) .
$$

Here, $w_j$ are the weights, $b$ is a scalar called the *bias*, and $\psi$ is a nonlinear scalar transformation called the *activation function*.

![Structure of a deep neural network](/fodl/assets/nn.png)

A neural network is a *feedforward composition* of several neurons, typically arranged in the form of *layers*. So if we imagine several neurons participating at the $l^{\text{th}}$ layer, we can stack up their weights (row-wise) in the form of a *weight matrix* $W^{(l)}$. The output of neurons forming each layer forms the corresponding input to all of the neurons in the next layer. So a neural network with two layers of neurons would have

$$
\begin{aligned}
z_1 &= \sigma(W^{1} x + b^{1}), \\
z_2 &= \sigma(W^{2} z_1 + b^{2}), \\
y &= W^{3} z_2 + b^{3} .
\end{aligned}
$$

Analogously, one can extend this definition to $L$ layers for any $L \geq 1$. The nomenclature is a bit funny sometimes. The above example is either called a "3-layer network" or "2-hidden-layer network"; the output $y$ is considered as its own layer and not considered as "hidden" (and )


## Memorization capacity

## Optimal capacity bounds

## Robust interpolation
