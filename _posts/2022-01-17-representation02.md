---
layout: page
title: Universal approximators
categories: representation
date: 2022-01-17
---

Previously, we visited several results that showed how (shallow) neural networks can effectively memorize training data. However, memorization of a finite dataset may not the end goal[^fn1]. In the ideal case, we would like to our network to simulate a (possibly complicated) prediction function that works well on most input data points. Is a given architecture good enough?

In this note we will describe the representation power of (shallow) neural networks in terms of their ability to approximate continuous functions. This line of work has a long and rich history (and function approximation itself, independent of the context of neural networks, is a vast body of work which we only barely touch upon). See here[^devore] For a recent (and fantastic) survey of this area.

As before, intuition tells us that an infinite number of neurons should be good enough for approximating pretty much anything. Therefore, our guiding principle will be to achieve as *succinct* a neural representation as possible. Moreover, if there is an *efficient computational* routine that gives this representation, that would put the icing on the cake.

## Warmup: Function approximation
{:.label}

We will keep things simple. This time, we don't have any training data to work with; let's just assume we seek some (purported) prediction function $g(x)$. To approximate $g$, we have a candidate hypothesis class of shallow (two-layer) neural networks of the form:

\\[ f(x) = \sum_{i=1}^m \alpha_i \psi(\langle w_i, x \rangle + b_i) . \\]

Our goal is to get reasonable bounds on how large $m$ needs to be in terms of various parameters of $g$. We have to be clear about what "approximate" means here. It is typical to measure approximation in terms of $p$-norms between measurable functions; for example, in the case of $L_2$-norms we will try to control

$$
\int_{\text{dom}(g)} |f -g|^2 d \mu
$$

where $\mu$ is some measure defined over $\text{dom}(g)$. Likewise for the $L_\infty$- (or the sup-)norm, and so on.

Let us first define a useful property to characterize univariate functions.

**Definition**{:.label #Lipschitz}
  A function $g : \R \rightarrow \R$ is $L$-Lipschitz if for all $u,v \in \R$, we have that $|f(u) - f(v) | \leq L |u - v|$.
{:.definition}

Any smooth function with bounded derivative is Lipschitz; actually, certain non-smooth functions (such as the ReLU) are also Lipschitz. Lipschitz-ness does not quite capture everything we care about (e.g. discontinuous functions are not Lipschitz, which can be somewhat problematic if there are "jumps" in the label space) but it serves as a large enough class of functions to prove interesting results.

If our target function $f$ is Lipschitz continuous with small $L$, then we can easily show that it can be well-approximated by a two-layer network with threshold activations: $\psi(z) = \mathbb{I}(z \geq 0)$.

**Theorem**{:.label #univariatesimple}
  Let $g : [0,1] \rightarrow \R$ be $L$-Lipschitz. Then, it can be  $\varepsilon$-approximated in the sup-norm by a two-layer network with $O(L/\varepsilon)$ hidden threshold neurons.
{:.theorem}

**Proof**{:label #univariatesimpleproof}
  This is an example where a picture suffices to show the proof. The high level idea is to 
{:.proof}


## Universal approximation theorems
{:.label}

## The method of Barron
{:.label}


---

[^fn1]:
    Although: exactly interpolating training labels seems standard in modern deep networks; see [here](https://paperswithcode.com/sota/image-classification-on-cifar-10) and Fig 1a  of [this paper](https://arxiv.org/pdf/1611.03530.pdf).

[^devore]:
    R. DeVore, B. Hanin, G. Petrova, [Neural network approximation](https://arxiv.org/pdf/2012.14501.pdf).
