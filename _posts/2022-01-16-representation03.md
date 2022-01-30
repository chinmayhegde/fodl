---
layout: page
title: Chapter 3 - The role of depth
categories: representation
date: 2022-01-16
---

For *shallow networks*, we now know upper bounds on the number of neurons required to "represent" data, where representation is measured either in the sense of exact interpolation (or memorization), or in the sense of universal approximation. We will continue to revisit these bounds when other important questions such as optimization/learning and out-of-sample generalization arise. In some cases, these bounds are even *tight*, meaning that we could not hope to do any better.

Which leads us to the natural question:

~~~
Does depth buy us anything at all?
~~~

If we have already gotten (some) tight results using shallow models, should there be any *theoretical* benefits in pursuing analysis of deep networks? Two reasons why answering this question is important:

One, after all, this is a course on "deep" learning theory, so we cannot avoid this question.

But two, for the last decade or so, a lot of folks have been trying very hard to replicate the success of deep networks with *highly* tuned shallow models (such as kernel machines), but so far have come up short. Understanding precisely why and where shallow models fall short (while deep models succeed) is therefore of importance.

A full answer to the question that we posed above will remain elusive. See, for example, the last paragraph of Belkin's monograph[^belkin] for some speculation. To quote a recent paper by Shankar et al.[^shankar]:

> "...the question remains open whether the performance gap between kernels and neural networks indicates a fundamental limitation of kernel methods or merely an engineering hurdle that can be overcome."  

Nonetheless: in this note, we will derive several interesting results that highlight the importance of depth in the representation power of neural network architectures. Let us focus on "reasonable-width" networks of depth $L > 2$ (because we already know from universal approximation that exponential-width two-layer networks can represent pretty much anything we like.) There are two angles of inquiry:

* Approach 1: prove that there exist datasets of some large enough size size that can *only* be memorized by networks of depth $\Omega(L)$, but not by networks of depth $o(L)$.

* Approach 2: prove that there exist classes of functions that can be $\varepsilon$-approximated *only* by networks of depth $\Omega(L)$, but not by networks of depth $o(L)$.

Results of either type can be called *depth separation* results. Let us start with the latter.

## Depth separation in function approximation
{:.label}

We will first prove a depth-separation result for dense feed-forward networks with ReLU activations and univariate (scalar) inputs. This result will generally hold for other families of activations.

We will explicitly construct a (univariate) function $g$ that can be exactly represented by a deep neural network (with error $\varepsilon = 0$) but is provably inapproximable by a much shallower network. This result is by Telgarsky[^telgarsky].

**Theorem**{:.label #DepthSeparation}
  There exists a function $g : [0,1] \rightarrow \R$ that is exactly realized by a ReLU network of constant width and depth $O(L^2)$, but for *any* neural network $f$ with depth $\leq L$ and sub-exponential number of units, $\leq 2^{L^\delta}$, $f$ is at least $\varepsilon$-far from $g$, i.e.:
  $$
  \int_0^1 |f(x) - g(x)| dx \geq \varepsilon
  $$
{:.theorem}
for constants $\varepsilon > \frac{1}{32}$ and $0 < \delta < 1$.

**Proof sketch**{:.label #DepthSeparationProof}
  High level idea: (a) observe that any ReLU network $g$ simulates a piecewise linear function. (b) prove that the number of pieces in the range of $g$ grows polynomially with width but exponentially in depth.
  **_(COMPLETE)_.**
{:.proof}

**Remark**{:.label #DepthSepRem1}
  The "hard" example function constructed in the above Theorem was for scalar inputs. What happens for the general case of $d$-variate inputs? Eldan and Shamir[^eldan] showed that there exist 3-layer ReLU networks (and $\text{poly}(d)$ width) that cannot be $\varepsilon$-approximated by any two-layer ReLU network unless they have $\Omega(2^d)$ hidden nodes. Therefore, there is already a separation between depth=2 and depth=3 in the high-dimensional case.
{:.remark}

**Remark**{:.label #DepthSepRem1}
  In the general $d$-variate case, can we get depth separation results for networks 
{:.remark}

## Depth-width tradeoffs in memorization
{:.label}


---

[^belkin]:
    Mikhail Belkin, [Fit without Fear](https://arxiv.org/pdf/2105.14368.pdf), 2021.

[^shankar]:
    V. Shankar, A. Fang, W. Guo, S. Fridovich-Keil, L. Schmidt, J. Ragan-Kelley, B. Recht, [Neural Kernels without Tangents](https://arxiv.org/abs/2003.02237), 2020.

[^telgarsky]:
    M. Telgarsky, [Benefits of depth in neural networks](http://proceedings.mlr.press/v49/telgarsky16.pdf), 2016.

[^eldan]:
    R. Eldan and O. Shamir, [The Power of Depth for Feedforward Neural Networks](http://proceedings.mlr.press/v49/eldan16.pdf), 2016.
