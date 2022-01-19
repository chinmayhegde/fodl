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

Let us begin simple. As before, suppose all we have is a bunch of training data points:
\\[ X = \{(x_i, y_i)\}_{i=1}^n \subset \mathbb{R}^d \times \mathbb{R} \\]
and our goal will be to a network that *exactly* memorizes $X$. That is, we will learn a function $f$ that, when evaluated on every data point $x_i$ in the training data, returns $y_i$. Equivalently, if we define empirical risk via the squared error loss:

$$
l(y,\hat{y}) = 0.5(y - \hat{y})^2, \quad R(f) = \sum_i \frac{1}{n} l(y_i, \hat{y_i})
$$

then $f$ achieves *zero empirical risk*.

But why should memorization be interesting? After all, machine learning folks are taught to be wary of [overfitting](https://en.wikipedia.org/wiki/Overfitting) to the training set. In introductory ML courses we spend several hours (and homework sets) covering the [bias-variance](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) tradeoff, the importance of adding a regularizer to decrease variance (at the expense of incurring extra "bias"), etc etc.

Unfortunately, deep learning practice throws this classical way of ML thinking out of the window. We seldom use explicit regularizers, instead relying on standard losses. We typically train deep neural networks to achieve 100\% (train) accuracy. Later, we will try to understand why networks trained to *perfectly* interpolate the training data still generalize well, but for now let's focus on just achieving a representation that enables perfect memorization.

## Basics

First, we define what we mean by a "neural network".

A neural network is composed of several primitive "units", each of which we will call a *neuron*. Given an input vector $x \in \mathbb{R}^d$, a neuron transforms the input according to the following functional form:

$$
z = \psi(\sum_{j=1}^d w_j x_j + b) .
$$

Here, $w_j$ are the weights, $b$ is a scalar called the *bias*, and $\psi$ is a nonlinear scalar transformation called the *activation function*; a typical activation function is the "ReLU function" $\psi(z) = \max(0,z)$ but we will also consider others.

![Structure of a deep neural network](/fodl/assets/nn.png)

A neural network is a *feedforward composition* of several neurons, typically arranged in the form of *layers*. So if we imagine several neurons participating at the $l^{\text{th}}$ layer, we can stack up their weights (row-wise) in the form of a *weight matrix* $W^{(l)}$. The output of neurons forming each layer forms the corresponding input to all of the neurons in the next layer. So a neural network with two layers of neurons would have

$$
\begin{aligned}
z_{1} &= \psi(W_{1} x + b_{1}), \\
z_{2} &= \psi(W_{2} z_{1} + b_{2}), \\
y &= W_{3} z_{2} + b_{3} .
\end{aligned}
$$

Analogously, one can extend this definition to $L$ layers for any $L \geq 1$. The nomenclature is a bit funny sometimes. The above example is either called a "3-layer network" or "2-hidden-layer network"; the output $y$ is considered as its own layer and not considered as "hidden" (and notice that it doesn't have any activation).

## Memorization capacity of shallow networks

<script>
macros["\\f"] = "\\mathscr{F}"
</script>

Let us focus our attention on the ability of *two-layer* networks (or one-hidden-layer networks) with ReLU activations to memorize data. That is, if there are $m$ hidden neurons and if $\psi$ is the ReLU then our hypothesis class $\f_m$ comprises all functions $f$ such that:

\\[ f(x) = \sum_{i=1}^m \alpha_i \psi(\langle w_i, x \rangle + b_i) . \\]

**Theorem**{:.label #MemorizationBasic}
  Let $f$ be a two-layer ReLU network with $m$ hidden neurons. Then, for any _arbitrary_ dataset $X = \{(x_i, y_i)_{i=1}^n\} \subset \mathbb{R}^d \times \mathbb{R}$ where $x_i$ are in general position, the weights and biases of $f$ can be chosen such that $f$ exactly interpolates $X$.
{:.theorem}

**Proof**{:.label #MemorizationBasicProof1}
  This result is non-constructive and seems to be folklore, dating back to at least Baum[^baum]. For modern versions of this proof, see Bach[^bach] or Bubeck et al.[^bubeck1].

  Define the space of *arbitrary width* two-layer networks:
  \\[\f = \bigcup_{m \geq 0} \f_m . \\]
  The high level idea is that $\f$ forms a *vector space*. This is easy to see, since it is closed under additions and scalar multiplications. Formally, fix $x$ and consider the element $\psi_{w,b}: x \mapsto \psi(\langle w, x \rangle + b)$. Then, $\text{span}(\psi_{w,b})$ forms a vector space.  Now, consider the linear *pointwise* evaluation operator $\Psi : V \rightarrow \mathbb{R}^n$:
  \\[\Psi(f) = (f(x_1), f(x_2), \ldots, f(x_n)) .\\]
  We know from classical universal approximation (Chapter 2) that *every vector* in $\mathbb{R}^n$ can be written as *some* (possibly infinite)  combination of neurons. Therefore, $\text{Range}(\Psi) = \mathbb{R}^n$, or $\text{dim(Range}(\Psi)) = n$. Therefore, there *exists* some basis of size $n$ with the same span! Call this basis $\{\psi_1, \ldots,\psi_n\}$. This basis can be used to express any set of labels by choosing appropriate coefficients in a standard basis representation $y = \sum_{i=1}^n \alpha_i \psi_i$.
  The result follows.
{:.proof}

In fact, the above result holds for any activation function $\Psi$ that is not a polynomial. Really, we didn't do much here. Since the "information content" in $n$ labels has dimension $n$, we can extract any arbitrary basis (written in the form of neurons) and write down the expansion of the labels in terms of this basis. Since this approach may be a bit abstract, let's give an alternate *constructive* proof.

**Proof (Alternate.)**{:label #MemorizationBasicProof2}
  This proof can be attributed to Zhang et al.[^zhang]. **COMPLETE**.
{:.proof}




## Optimal capacity bounds

## Robust interpolation

---

[^baum]:
    E. Baum, [*On the capabilities of multilayer perceptrons*](https://www.sciencedirect.com/science/article/pii/0885064X88900209), 1989.

[^bach]:
    F. Bach, [*Breaking the Curse of Dimensionality with Convex Neural Networks*](https://jmlr.org/papers/v18/14-546.html), 2017.

[^bubeck1]:
    S. Bubeck, R. Eldan, Y. Lee, D. Mikulincer, [Network size and weights size for memorization with two-layers neural networks](https://arxiv.org/abs/2006.02855), 2020.

[^zhang]:
    C. Zhang, S. Bengio, M. Hardt, B. Recht, O. Vinyals, [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530), 2017.
