---
layout: page
title: Chapter 4 - Optimization basics
categories: optimization
date: 2022-01-15
---

In the first few chapters, we covered several results related to the representation power of (both shallow and deep) neural networks. We obtained estimates --- sometimes tight ones --- for the sizes of neural networks needed to memorize a given dataset, or to simulate a given function.

However, most of our theoretical results used somewhat funny-looking constructions of neural networks. Our hypothetical networks are all either too wide or too narrow.

But even beyond the issue of size, none of the (theoretically attractive) techniques that we used to memorize datasets resemble deep learning practice. When folks refer to "training models" they almost always are talking about fitting datasets via incremental, greedy, first-order algorithms such as gradient descent (GD), or stochastic variations (like SGD), or accelerations (like Adam), or some other similar approach.

In the new few chapters, we address the question:

```
Do practical approaches for model training work well?
```

This question is once again somewhat ill-posed; we will have to be precise about what "work" and "well" mean in the above question. (An overwhelming amount of empirical evidence seems to suggest that they work just fine, as long as certain tricks/hacks are applied.)

While there is a very large variety of techniques that have been proposed for training deep networks, we will focus on a handful of the most canonical settings, analyze them, and derive precise bounds on their behavior. The hope is that such analysis can illuminate differences/tradeoffs between different choices and provide useful thumb rules to guide further practice.

## Setup
{:.label}

Most (all?) popular approaches in deep learning do the following:

- Write down the loss (or empirical risk) in terms of the training data points and the weights. (Usually the loss is decomposable across the data points). So if the data is $(x_i,y_i)_{i=1}^n$ then the loss is something like:

    $$
    L(w) = \frac{1}{n} \sum_{i=1}^n l(y_i,\hat{y_i}), \quad \hat{y_i} = f_w(x_i),
    $$

    where $l(\cdot,\cdot) : \R \times \R \rightarrow \R_{\geq 0}$ is a non-negative measure of fit, and $f_w$ is the function represented by the neural network with weight/bias parameters $w$. (*We are abusing notation here since we previously used $R$ for the risk, but let's just use $L$ to denote loss.*)
- Then, we seek the weights/biases $w$ that optimize the loss:

    $$
    \hat{w} = \arg \min_w L(w) .
    $$

    Sometimes, we throw in an extra regularization term defined on $w$ for kicks, but let's just stay simple for now.
- Importantly, invariably, the above optimization is carried out using a *simple, iterative, first-order method*. For example, if gradient descent (GD) is the method of choice, then discovering $\hat{w}$ amounts to iterating the following recursion:

    $$
    w \leftarrow w - \eta \nabla_w L(w)
    $$

    some number of times. Or, if SGD is the method of choice, then discovering $\hat{w}$ amounts to iterating the following recursion:

    $$
    w \leftarrow w - \eta g_w
    $$

    where $g_w$ is some (properly defined) stochastic approximation of $\nabla L(w)$. Or if Adam[^adam] is used, then discovering $\hat{w}$ amounts to iterating..a slightly more [complicated recursion](https://chinmayhegde.github.io/dl-notes/notes/lecture03/), but still a first-order method involving gradients and nothing more. You get the picture.

In this chapter, let us focus on GD and SGD. The following questions come to mind:

* Do GD and SGD converge.
* If so, do they converge to the set (or *a* set?) of globally optimal weights.
* How many steps do they take.
* How to pick step size, or (in the case of SGD) batch size, or other parameters.

among many others.

Before diving in to the details, let us qualitatively address the first couple of questions.  

Intuition tells us that convergence to global minimizers seems unlikely. It is easy to prove that for anything beyond the simplest neural networks (single-layer networks with linear activations, which are really just linear models), the loss function $L(w)$ is *extremely non-convex*. Therefore the "loss landscape" of $L(w)$, viewed as a function of $w$, has many peaks, valleys, and ridges, and a myopic first-order approach such as GD may likely get stuck in local optima. A fantastic paper[^losslandscape] by Xu et al. proposes creative ways of visualizing these high-dimensional landscapes.

![Loss landscapes of ResNet-56 (left) without skip connections (right) with skip connections.](/fodl/assets/losslandscape.png)

Somewhat fascinatingly, however, it turns out that this intuition is incorrect. GD/SGD *can* be used to train models all the way down to *zero* train error (at least, this is common for deep networks used in classification.) This fact seems to have been folklore, but was systematically demonstrated in a series of interesting experiments by Zhang et al.[^zhang].

We will revisit this fact in the next two chapters. But for now, we will limit ourselves to establishing the more modest claim:

> (S)GD converges to (near) stationary points, provided $L$ is smooth.

The last caveat --- that $L$ is required to be smooth --- excludes several widely used architectures (such as ReLU networks). This should not deter us. The analysis is still very interesting and useful; it is still applicable to other widely used architectures; and extensions to ReLUs can be achieved with a bit more technical heavy lifting (which we won't cover here).

## Gradient descent
{:.label}

## Stochastic gradient descent
{:.label}

## Extensions
{:.label}

---

[^adam]:
    D. Kingma and J. Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), 2014.

[^losslandscape]:
    H. Li, Z. Xu, G. Taylor, C. Studer, T. Goldstein, [Visualizing the Loss Landscape of Neural Nets](https://proceedings.neurips.cc/paper/2018/file/a41b3bb3e6b050b6c9067c67f663b915-Paper.pdf), 2018.

[^zhang]:
    C. Zhang, S. Bengio, M. Hardt, B. Recht, O. Vinyals, [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530), 2017.
