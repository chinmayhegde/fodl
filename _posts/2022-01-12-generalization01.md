---
layout: page
title: Chapter 7 - Implicit regularization
categories: generalization
date: 2022-01-12
---

We spent the last few chapters understanding the *optimization* aspect of deep learning. Namely, given a dataset and a family of network architectures, how well do standard training procedures (such as gradient descent or SGD) work in terms of fitting the data? As we witnessed, an answer to this question is far from complete, but a picture (involving tools from optimization theory and kernel-based learning) is beginning to emerge.


However, training is not the end goal. Recall in the Introduction that we want to discover prediction functions that perform "well" on "most" possible data points. In other words, we not only want our model to fit a training dataset well, but also would like our model to exhibit *low generalization error* when evaluated on hitherto unseen inputs.

Over the next few lectures, we address the central question:

```
When (and why) do highly over-parameterized networks trained using (S)GD generalize?  
```

As we work through different aspects of this question, we will find that

* classical statistical learning theory fails to give useful (non-vacuous) generalization error bounds.
* in fact, lots of other weirdness happens.
* architecture, optimization, and generalization in deep learning interact in a non-trivial manner, and should be studied simultaneously.

## Classical models and regularization
{:.label}

Statistical learning theory offers [many](https://en.wikipedia.org/wiki/Regularization_(mathematics)), [many](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory) lenses through which we can [study](https://en.wikipedia.org/wiki/Probably_approximately_correct_learning) [generalization](https://en.wikipedia.org/wiki/Stability_(learning_theory)).

## Implicit bias of GD/SGD for linear models
{:.label}

Under construction.

## Nonlinear models and incremental learning
{:.label}

## Implicit bias of ReLU networks
{:.label}
