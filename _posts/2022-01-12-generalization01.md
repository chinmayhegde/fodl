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

Let us work backwards. In this chapter, we will see how the choice of training algorithm (which, in the case of deep learning, is almost always GD or SGD or some variant thereof) affects generalization.

## Classical models, margins, and regularization
{:.label}

Statistical learning theory offers [many](https://en.wikipedia.org/wiki/Regularization_(mathematics)), [many](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory) lenses through which we can [study](https://en.wikipedia.org/wiki/Probably_approximately_correct_learning) [generalization](https://en.wikipedia.org/wiki/Stability_(learning_theory)). One approach (classical, and fairly intuitive) is to think in terms of *margins*.

Let us be concrete. Say that we are doing binary classification using linear models.  We have a dataset $X = \lbrace (x_i,y_i) \subset \R^d \times \pm 1 \rbrace$. For simplicity let us ignore the bias and assume that everything is centered around the origin. The problem is to find a linear classifier that generalizes well. The following picture (in $d=2$) comes to mind.

![Margins of linear classifiers.](/fodl/assets/margins.png)

Assume that the data is linearly separable. Then, we can write down the desired classifer as the solution to the following optimization problem (and in classical ML, we call such a model a *perceptron*.)

$$
\text{Find}~w,~\text{such that}~\text{sign}\lang w,x_i \rang = y_i~~\forall~i \in [n].
$$

or, equivalently,

$$
\text{Find}~w,~\text{such that}~ y_i \lang w,x_i \rang \geq 0~~\forall~i \in [n].
$$

However, as is the case in the picture above, there could be several linear models that exactly interpolate the labels, which means that the above optimization problem may not have unique solutions. Which solution, then, should we pick?

One way to resolve this issue is by the *Large Margin Hypothesis*, which asserts that:

```
Larger margins --> better generalization.
```

Suppose we associate each linear classifier with the corresponding separating hyperplane (marked in red above). Since we want "most" points in the data space to be correctly classified, a reasonable choice is to find the hyperplane that is furthest away from either class. An easy calculation (shown, for example, [here](https://en.wikipedia.org/wiki/Support-vector_machine#Linear_SVM)) shows that for any hyperplane parameterized by $w$, the margin equals $\frac{2}{\lVert w \rVert}$. Therefore, maximizing the margin is the same as solving the the optimization problem:

$$
\begin{aligned}
&\min_w \frac{1}{2} \lVert w \rVert^2  \\
\text{s.t.~}& y_i \lang w, x_i \rang \geq 1,~\forall i \in [n] .
\end{aligned}
$$

The solution to this problem is called the *support vector machine* (SVM). This is fine, as long as the data is linearly separable. If not, the constraint set is empty and the optimization problem is ill-posed. A remedy is to relax the (hard) constraints using Lagrange. We get the *soft-margin* SVM as a solution to the (relaxed) problem:

$$
\min_w \frac{1}{2} \lVert w \rVert^2 + C \sum_{i=1}^n l(y_i \lang w, x_i \rang)
$$

where $C > 0$ is some weighting parameter, and $l : \R \rightarrow \R$ is some (scalar) loss function that penalizes large negative inputs and leaves inputs bigger than 1 untouched. The $\exp$ loss ($l(z) = e^{-z}$), or the *Hinge* loss ($l(z) = \max(0,1-z)$) are examples.

Notice that the relaxed problem resembles the standard setup in statistical inference. There is a training loss function $L(w) = \sum_{i=1}^n l(y_i \lang w, x_i \rang)$ and a *regularizer* $R(w) = \lVert w \rVert^2$, and we seek a model that tries to minimize both. The $L_2$-regularizer used above (in deep learning jargon) is sometimes called "weight decay" but others (such as the $L_1$-regularizer) are also popular.

An alternative way to arrive at this standard setup is by invoking [Occam's Razor](https://en.wikipedia.org/wiki/Occam%27s_razor): among the many explanations to the data, we want the one that is of "minimum complexity". This also motivates a formulation for linear regression which is the same except we use the least-squares loss. Setting $X \in \R^{n \times d}$ as the data matrix, we get:

$$
\min_w \frac{1}{2} \lVert w \rVert^2 + \mu \lVert y - Xw \rVert^2
$$

which is sometimes called *Tikhonov regularization*.

[There is a long history here which we won't recount. See here[^esl] for a nice introduction to the area, and here[^schapire] which establishes formal generalization bounds in terms of the margin.

**However**.

In deep learning, anecdotally we seem to be doing just fine even without any regularizers. While regularization may help in some cases, several careful experiments[^zhang] show that regularization is neither necessary nor sufficient for good generalization. Why is that the case?

To resolve this, a large body of recent work (which we introduce below) studies what I will call the *Implicit Bias Hypothesis*. This hypothesis asserts that the *dynamics* of gradient-based training itself introduces a regularization effect, therefore biasing the choice of solution towards minimum complexity solutions.

## Implicit bias of GD/SGD for linear models
{:.label}

Under construction.

## Nonlinear models and incremental learning
{:.label}

## Implicit bias of ReLU networks
{:.label}

---

[^esl]:
    T. Hastie, J. Friedman, R. Tibshirani, [Elements of Statistical Learning](https://hastie.su.domains/Papers/ESLII.pdf).

[^schapire]:
    R. Schapire, Y. Freund, P. Bartlett, W. Sun Lee, [Boosting the margin: A new explanation for the effectiveness of voting methods](https://faculty.cc.gatech.edu/~isbell/tutorials/boostingmargins.pdf), 1997.

[^zhang]:
    C. Zhang, S. Bengio, M. Hardt, B. Recht, O. Vinyals, [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530), 2017.
