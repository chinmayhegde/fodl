---
layout: page
title: Introduction
categories: intro
date: 2022-01-19
---

Our goal is to learn to reason about (deep) neural networks from the lens of *theory*.

Unlike other scientific fields, there is currently a *very* wide gap between what the best available tools in theoretical computer science can tell us about modern neural networks, and the actual ways in which they are deployed. A major motivation behind these course notes is to identify the landscape of how wide exactly these gaps are at present. By asking carefully crafted (but precise) questions, the hope is that one can shed light on why certain aspects of neural networks work (or don't).

This is by no means the first attempt to do so. Other excellent courses/lecture notes include:

* [Fit without Fear](https://arxiv.org/pdf/2105.14368.pdf) by Misha Belkin (UCSD).
* [Mathematics of Deep Learning](https://www.notion.so/Mathematics-of-Deep-Learning-05cd9255f03842489083ec7cbb6338d5) by Joan Bruna (NYU).
* [Deep Learning Theory](https://mjt.cs.illinois.edu/dlt/) by Matus Telgarsky (UIUC).

## Setup
{:.label}

As typical in (supervised) machine learning, our starting point is a list of $n$ example data-label pairs:
\\[ X = \{(x_i, y_i)\}_{i=1}^n \subset \mathbb{R}^d \times \mathbb{R} \\]
which we will call the *training set*. This dataset is assumed to acquired via *iid sampling* with respect to some underlying probability measure $\mu$ defined over $\mathbb{R}^d \times \mathbb{R}$.

Our goal will be to *predict* the label $y \in \mathbb{R}$ associated with some (hitherto unseen) data point $x \in \mathbb{R}^d$. In order to do so, we will seek a prediction function $f$ that can be expressed as a *neural network* and that performs "well" on "most" input data points. Let us agree to measure "goodness" of performance via a *loss function* $l(\cdot,\cdot) : \mathbb{R} \times \mathbb{R} \rightarrow \mathbb{R}_{\geq 0}$ takes in a predicted label and compares with the truth. Then, quantitatively, our prediction function should be such that the *population risk*:
\\[ R(f) = \mathbb{E}_{(x,y) \sim \mu} l(y, f(x)) \\]
is small.

This immediately poses **Major Challenge**, since the population risk $R(f)$ is not an easy object to study. For starters, the probability measure $\mu$ may not be be known. Even if magically $\mu$ is available, calculating the expected value with respect to $\mu$ can be difficult. However, we do have training data lying around. Therefore, instead of directly dealing with $R(f)$, we will instead use a proxy quantity called the *empirical risk*:
\\[ \hat{R}(f) = \frac{1}{n} \sum_{i=1}^n l(y_i, f(x_i)) \\]
and seek an $f$ that makes this small.

<script>
macros["\\f"] = "\\mathscr{F}"
</script>

Now, reducing the empirical risk $\hat{R}(f)$ to as small as possible is akin to function optimization. To make this numerically tractable, we will first cook up a hypothesis class $\f$. In deep learning, this can be thought of as the set of all neural network models that obey a certain architecture[^fn1]. This now poses another **Major Challenge**: what's a good family of architectures? How do we know whether a certain architecture is rich enough to solve our prediction problem? Can it go the other way (i.e., could we somehow pick a network architecture that is far too rich for our purposes?)

Let us set aside such troubling questions for now. Once the network architecture optimization over $\f$ boils down to tuning the weights and biases of $f$ such that $\hat{R}(f)$ is as small as possible. In other words, we will wish to solve for $f_b$, the "best model" in the hypothesis class $\f$:
\\[
f_b = \arg \min_{f \in \f} \hat{R}(f) .
\\]  

This optimization problem hides yet another **Major Challenge**, tuning weights and biases *to optimality* is extremely difficult, except in the simplest of cases (such as linear models). In practice, we never solve this problem exactly, but rather just run some kind of incremental "training" procedure for some number of steps that iteratively decreases $\hat{R}(f)$ until everyone is satisfied with the results.

Let the final result of this training procedure be called $\hat{f}$; this network is what we end up using to perform all future predictions. Our hope is that $\hat{f}$ performs "well" on "most" data points. Quantitatively, we would like to ensure that the population risk
\\[R(\hat{f}) \\]
is small. But can we *prove* that this is the case? Again, the classical theory of supervised learning breaks this down into three components:
\\[
\begin{aligned}
R(\hat{f}) = &R(\hat{f}) - \hat{R}(\hat{f})
          &+ \hat{R}(\hat{f}) - \hat{R}(f_b)
          &+ \hat{R}(f_b) .
\end{aligned}
\\]
If all three components are on the right hand side are *provably* small, then we are in the clear. Let us parse these three terms in reverse order.

First, if
\\[ \hat{R}(f_b) \\]
is small then this means our network architecture $\f$ is rich enough for our purposes (or at least, rich enough to fit the training data well). We call this the **representation error** in deep learning, and a conclusive proof showing that this quantity is small would address the middle Major Challenge identified above.

Second, if
\\[ \hat{R}(\hat{f}) - \hat{R}(f_b) \\]
is small then our numerical training procedure used to find $\hat{f}$ has been reasonably successful. We call this the **optimization error** in deep learning, and a conclusive proof showing that this quantity is small would address the last Major Challenge identified above.

Third, if
\\[R(\hat{f}) - \hat{R}(\hat{f}) \\]
is small then $R$ and $\hat{R}$ are not too different for $\hat{f}$. In other words, we need to prove that the empirical risk is a *good proxy* for the population risk. we call this the **generalization problem** in deep learning, and a decisive solution to this problem would address the first Major Challenge identified above.

### Outline
{:.label}

The above narrative is all rather classical, and can be found in any introductory text on statistical machine learning. For simple cases (such as linear models) fairly precise bounds can be derived for all three quantities.   

The main difference lies in how we address these questions in the context of deep learning. Somewhat troublingly, clean answers to each of the above foundational problems do not seem to (yet) exist for deep neural network models. It is likely that these problems cannot be studied in isolation, and that positive answers to these problems interact in mysterious ways[^fn2].

Still, for the purposes of organizing the existing literature with some degree of coherence, we will follow this classical narrative. We will cover:

* **Representation**: Making as few assumptions as possible, we will derive bounds on the number of neurons (and the shapes of neural networks) required to achieve low representation error. These are in the form of "universal approximation" theorems. We will explore both the finite setting (where we are trying to memorize a finite training dataset) and the infinite setting (where we are trying to approximate continuous functions). We will see how trading off *depth* versus *width* leads to interesting behaviors.

* **Optimization**: We will derive bounds on the number of training 

[^fn1]:
    We will instantiate $\f$ with concrete examples in specific instances.

[^fn2]:
    For example, networks that exhibit low optimization gap *also* sometimes lead to good generalization.
