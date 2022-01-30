---
layout: page
title: Introduction
categories: Notes
date: 2022-01-19
---


Our goal is to reason about (deep) neural networks from the lens of *theory*.

Unlike many other scientific fields, there currently exists a wide gap between:
- what the best available tools in theoretical computer science can tell us about modern neural networks, and
- the actual ways in which modern neural network models work in practice.

This gap between theory and practice is, in my view, unsatisfactory. Yann LeCun invokes the ["streetlight effect"](https://www.youtube.com/watch?v=gG5NCkMerHU&t=3210s): we search for lost keys where we can, not where they really lie.

But here is why (I think) theory matters: by asking carefully crafted (but precise) questions about deep networks, one can precisely answer why certain aspects of neural networks work (or don't), and what is in the realm of the possible (versus what isn't). A major motivation behind these course notes is to identify the landscape of how wide exactly these gaps are at present, and how to bridge them.

These notes are by no means the first attempt to do so. Other excellent courses/lecture notes/surveys include:

* [Fit without Fear](https://arxiv.org/pdf/2105.14368.pdf) by Misha Belkin (UCSD).
* [Mathematics of Deep Learning](https://www.notion.so/Mathematics-of-Deep-Learning-05cd9255f03842489083ec7cbb6338d5) by Joan Bruna (NYU).
* [Deep Learning Theory](https://mjt.cs.illinois.edu/dlt/) by Matus Telgarsky (UIUC).
* [Expository videos](https://blogs.princeton.edu/imabandit/2020/10/13/2020/) by Sebastian Bubeck (Microsoft).


## Setup
{:.label}

As typical in (supervised) machine learning, our starting point is a list of $n$ example data-label pairs:
\\[ X = \{(x_i, y_i)\}_{i=1}^n \subset \mathbb{R}^d \times \mathbb{R} \\]
which we will call the *training set*. This dataset is assumed to acquired via *iid sampling* with respect to some underlying probability measure $\mu$ defined over $\mathbb{R}^d \times \mathbb{R}$.

Our goal will be to *predict* the label $y \in \mathbb{R}$ associated with some (hitherto unseen) data point $x \in \mathbb{R}^d$. In order to do so, we will seek a prediction function $f$ that can be expressed as a *neural network*. Later we will more precisely define neural networks, but the familiar [picture](https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Neural_network_example.svg/640px-Neural_network_example.svg.png) is appropriate for now. The key point is that we want an $f$ that performs "well" on "most" input data points.

Let us first agree to measure "goodness" of performance via a two-argument loss function

$$
l(\cdot,\cdot) : \mathbb{R} \times \mathbb{R} \rightarrow \mathbb{R}_{\geq 0}
$$

which takes in a predicted label, compares with the truth, and spits out a non-negative cost of fit (smaller is better). Then, quantitatively, our prediction function should be such that the population risk:

$$
R(f) = \mathbb{E}_{(x,y) \sim \mu} l(y, f(x))
$$

is small.

This immediately poses a **Major Challenge**, since the population risk $R(f)$ is not an easy object to study. For starters, the probability measure $\mu$ may not be be known. Even if magically $\mu$ is available, calculating the expected value with respect to $\mu$ can be difficult. However, we do have training data lying around. Therefore, instead of directly dealing with $R(f)$, we will instead use a proxy quantity called the *empirical risk*:

$$
\hat{R}(f) = \frac{1}{n} \sum_{i=1}^n l(y_i, f(x_i))
$$

and seek an $f$ that makes this small.

<script>
macros["\\f"] = "\\mathscr{F}"
</script>

Now, reducing the empirical risk $\hat{R}(f)$ to as small as possible is akin to function optimization. To make this numerically tractable, we will first cook up a hypothesis class $\f$. In deep learning, this can be thought of as the set of all neural network models that obey a certain architecture[^fn1].

But: this approach now poses another **Major Challenge**. What's a good family of architectures? How do we know whether a certain architecture is rich enough to solve our prediction problem? Can it go the other way (i.e., could we somehow pick a network architecture that is far too rich for our purposes?)

Let us set aside such troubling questions for now. Once the network architecture optimization over $\f$ boils down to tuning the weights and biases of $f$ such that $\hat{R}(f)$ is as small as possible. In other words, we will wish to solve for $f_b$, the "best model" in the hypothesis class $\f$:

$$
f_b = \arg \min_{f \in \f} \hat{R}(f) .
$$  

Alas, yet another **Major Challenge**. Tuning weights and biases *to optimality* is extremely difficult, except in the simplest of hypothesis classes (such as linear/affine models). In practice, we never solve this optimization problem, but rather just run some kind of incremental "training" procedure for some number of steps that iteratively decreases $\hat{R}(f)$ until everyone is satisfied. Let us assume that we are somehow able to get a decent answer. Let the final result of this training procedure be called $\hat{f}$.

So, to recap: we have introduced two definitions of risk ($R, \hat{R}$), and defined two models ($f_b, \hat{f}$). This final network $\hat{f}$ is what we end up using to perform all future predictions. Our hope is that $\hat{f}$ performs "well" on "most" future data points. Quantitatively, we would like to ensure that the population risk
\\[R(\hat{f}) \\]
is small.

But can we *prove* that this is the case? Again, the classical theory of supervised learning breaks this down into three components:

$$
\begin{aligned}
R(\hat{f}) = & \quad R(\hat{f}) - \hat{R}(\hat{f}) \\
          &+ \hat{R}(\hat{f}) - \hat{R}(f_b) \\
          &+ \hat{R}(f_b) .
\end{aligned}
$$

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

## Outline
{:.label}

The above narrative is all very classical and can be found in any introductory text on statistical machine learning. For simple cases (such as linear models), good bounds can be derived for all three quantities.   

For us, the main difference lies in how we address these questions in the context of deep learning. Somewhat troublingly, clean answers to each of the above foundational problems do not seem to (yet) exist for deep neural network models. It is likely that these problems cannot be studied in isolation, and that theoretical results on these problems interact in mysterious ways[^fn2].

Still, for the purposes of organizing the existing literature with some degree of coherence, we will follow this classical narrative. Within the context of (deep) neural network learning, we will cover:

* **Representation**: Making as few assumptions as possible, we will derive bounds on the **number of neurons** (and the shapes of neural networks) required to achieve low representation error. These are in the form of **universal approximation** theorems. We will explore both the finite setting (where we are trying to memorize a finite training dataset) and the infinite setting (where we are trying to approximate continuous functions). We will see how trading off **depth versus width** leads to interesting behaviors.

* **Optimization**: We will study natural first-order algorithms for neural network training, and derive bounds on the number of training steps required to achieve low optimization error. In some cases, our results will show that the solution achieved by these training procedures are **locally optimal**. In other cases, we will prove that our training procedures achieve **global** optimality. Of particular interest to us are interesting connections to classical kernel-learning called the **Neural Tangent Kernel** (NTK).

* **Generalization**: We will study ways to obtain bounds on the number of training examples required to achieve low generalization error. In many cases, some existing techniques from classical learning theory may lead to vacuous bounds, while other techniques are more successful; our focus will be to get **non-vacuous** generalization guarantees. We will also study **double-descent** phenomena that reveal a curious relationships between the number of parameters and the generalization error.  

* **Miscellaneous** topics:  And finally, to round things off, we will analyze other aspects of neural networks of relevance to practice beyond just achieving good prediction. Questions surrounding the **robustness** of neural networks have already emerged. Issues such as **privacy and security** of data/models are paramount. Beyond just label prediction, neural networks are increasingly being used to solve more challenging tasks, including  **synthesis and generation** of new examples.

---

[^fn1]:
    We will instantiate $\f$ with specific instances as needed.

[^fn2]:
    For example, funnily enough, networks that exhibit very low optimization gap *also* sometimes lead to excellent generalization, in contradiction to what we would expect from classical learning theory.
