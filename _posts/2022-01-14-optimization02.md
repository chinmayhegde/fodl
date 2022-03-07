---
layout: page
title: Chapter 5 - Optimizing wide networks
categories: optimization
date: 2022-01-14
---

In the previous chaper, we proved that (S)GD converges locally for training neural networks with smooth activations. (The case for ReLU networks is harder but still doable with additional algebraic heavy lifting.)

However, in practice, somewhat puzzlingly we encounter very deep networks that are regularly optimized to *zero training loss*, i.e., they exactly learn to interpolate the given training data and labels. Since loss functions are typically non-negative, this means that (S)GD have achieved convergence to the *global* optimum.

In this chapter, we ask the question:

```
When and why does (S)GD give zero train loss?
```

An answer to this question in full generality is not yet available. However, we will prove a somewhat surprising fact: for *very wide* networks that are *randomly initialized*, this is indeed the case. The road to proving this will give us several surprising insights, and connections to classical ML (such as kernel methods) will become clear along the way.

## Local versus global minima
{:.label}

Let us bring up this picture from Xu et al.[^losslandscape] again:

![Loss landscapes of ResNet-56. Adding skips significantly influences ease of optimization, since the landscape is much better behaved.](/fodl/assets/losslandscape.png)

The picture on the left shows the loss landscape of a feedforward convolutional network. The picture on the right shows the loss landscape of the same network with residual skip connections.

Why would (S)GD navigate its way to the global minimum of this jagged-y, corrugated loss landscape? There could be one of many explanations.

* *Explanation 1*: We got lucky and initialized very well (close enough to the global optimum for GD to work). But if there is only one global minimum then "getting lucky" is an event that happens with exponentially small probability. So there has to be another reason.

* *Explanation 2*: The skips (clearly) matter, so maybe most real-world networks in fact look like the ones on the right. There is a grain of truth here: in modern (very) deep networks, there exist all kinds of bells and whistles to "make things work". Residual (skip) connections are common. So are other tweaks such as batch normalization, dropout, learning rate scheduling, etc etc. All[^resnet] of these[^batchnorm] tweaks[^explearn] significantly influence the nature of the optimization problem.

* *Explanation 3*: Since modern deep networks are so **heavily over-parameterized**, there may be tons of minima that exactly interpolate the data. In fact, with high probability, a random initialization lies "close" to a interpolating set of weights. Therefore, gradient descent starting from this location successfully trains to zero loss.

We will pursue Explanation 3, and derive (mathematically) fairly precise justifications of the above claims. As we will argue in the next chapter, this is by no means the full picture. But the exercise is still going to be very useful.

## The Neural Tangent Kernel
{:.label}

## Extensions
{:.label}

---

[^losslandscape]:
    H. Li, Z. Xu, G. Taylor, C. Studer, T. Goldstein, [Visualizing the Loss Landscape of Neural Nets](https://proceedings.neurips.cc/paper/2018/file/a41b3bb3e6b050b6c9067c67f663b915-Paper.pdf), 2018.

[^resnet]:
    M. Hardt and T. Ma, [Identity Matters in Deep Learning](https://openreview.net/forum?id=ryxB0Rtxx), 2017.

[^batchnorm]:
    H Daneshmand, A Joudaki, F Bach, [Batch normalization orthogonalizes representations in deep random networks](https://proceedings.neurips.cc/paper/2021/file/26cd8ecadce0d4efd6cc8a8725cbd1f8-Paper.pdf), 2021.

[^explearn]:
    Z. Li and S. Arora, [An exponential learning rate schedule for deep learning](https://arxiv.org/pdf/1910.07454.pdf), 2019.
