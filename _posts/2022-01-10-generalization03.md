---
layout: page
title: Chapter 9 - Generalization bounds via stability
categories: generalization
date: 2022-01-10
---

In the last two lectures, we have investigated generalization through the lens of

* margin-based regularization
* model parsimony

Each had their shortfalls, especially in the context of deep learning:

* Regularization very much depends on model architecture and/or training procedure. Implicit bias of GD shows up mysteriously in several places.
* Model parsimony is hard to measure; interacts with the data distribution in hard-to-quantify ways; and anyway in the last chapter we saw that uniform convergence seems to be an insufficient tool for getting non-trivial bounds, especially in the over-parameterized setting.

Let us now briefly a *third* lens to understand generalization:

* Algorithmic stability.

Our central hypothesis in this chapter is:

```
Stable learning implies generalization.
```

The high level idea is that if the model produced by a training algorithm is not sensitive to any single training data point, then the model generalizes. This idea has been around for some time (and in fact, methods such as bagging[^breiman] explicitly were developed in order to make classifiers stable to individual points.)

(Notice that the definition of stability involves specifying the training procedure; once again, the role of the training algorithm is more clearly illuminated here.)

The plus side is that we *won't* need to appeal to uniform convergence, so over-parameterization is not a concern. The minus side is that *convexity* of the loss seems to be the key tool here, which makes application to deep learning difficult.

On the other hand, our tour of optimization (Chapter 4) has given us tools to deal with (some) non-convex losses. All these will be prove to be fruitful.

## Setup
{:.label}

Stability as a tool for generalization dates back to an influential paper by Bousquet and Elisseef[^bousquet02], who introduced the notion of *uniform stability*.

Further refined in the context of empirical risk minimization by Shalev-Schwartz et al.[^ermstability].

## Algorithmic stability of (S)GD for empirical risk minimization
{:.label}

### Beyond convexity

Stability under smoothness + PL-condition. Key paper: 

## Connections to differential privacy
{:.label}



---

[^breiman]:
    L. Breiman, [Bagging Predictors](https://www.stat.berkeley.edu/~breiman/bagging.pdf), 1994.

[^bousquet02]:
    O. Bousquet and A. Elisseef, [Stability and Generalization](https://www.jmlr.org/papers/volume2/bousquet02a/bousquet02a.pdf), 2002.

[^ermstability]:
    S. Shalev-Schwartz, O. Shamir, K. Sridharan, N. Srebro, [Learnability, Stability and Uniform Convergence](https://jmlr.csail.mit.edu/papers/volume11/shalev-shwartz10a/shalev-shwartz10a.pdf), 2010.
