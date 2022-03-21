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
When (and why) do networks trained using (S)GD generalize?  
```

As we work through different aspects of this question, we will find that

* classical statistical learning theory fails to give useful (non-vacuous) generalization error bounds.
* in fact, lots of other weirdness happens.
* architecture, optimization, and generalization in deep learning interact in a non-trivial manner, and should be studied simultaneously.

Let us work backwards. In this chapter, we will see how the choice of training algorithm (which, in the case of deep learning, is almost always GD or SGD or some variant thereof) affects generalization.

## Classical models, margins, and regularization
{:.label}

Statistical learning theory offers [many](https://en.wikipedia.org/wiki/Regularization_(mathematics)), [many](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory) lenses through which we can [study](https://en.wikipedia.org/wiki/Probably_approximately_correct_learning) [generalization](https://en.wikipedia.org/wiki/Stability_(learning_theory)). One approach (classical, and fairly intuitive) is to think in terms of *margins*.

Let us be concrete. Say that we are doing binary classification using linear models.  We have a dataset $X = \lbrace (x_i,y_i)_{i=1}^n \subset \R^d \times \pm 1 \rbrace$. For simplicity let us ignore the bias and assume that everything is centered around the origin. The problem is to find a linear classifier that generalizes well. The following picture (in $d=2$) comes to mind.

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

Notice that the relaxed problem resembles the standard setup in statistical inference. There is a training loss function $L(w) = \sum_{i=1}^n l(y_i \lang w, x_i \rang)$ and a *regularizer* $R(w) = \lVert w \rVert^2$, and we seek a model that tries to minimize both. The $\ell_2$-regularizer used above (in deep learning jargon) is sometimes called "weight decay" but others (such as the $\ell_1$-regularizer) are also popular.

An alternative way to arrive at this standard setup is by invoking [Occam's Razor](https://en.wikipedia.org/wiki/Occam%27s_razor): among the many explanations to the data, we want the one that is of "minimum complexity". This also motivates a formulation for linear regression which is the same except we use the least-squares loss. Setting $X \in \R^{n \times d}$ as the data matrix, we get:

$$
\min_w \frac{1}{2} \lVert w \rVert^2 + \mu \lVert y - Xw \rVert^2
$$

which is sometimes called *Tikhonov regularization*. Different choices of $\mu$ lead to different solutions: a large $\mu$ reduces the data fit error, while a small $\mu$ encourages lower-norm solutions.

[There is a long history here which we won't recount. See here[^esl] for a nice introduction to the area, and here[^schapire] which establishes formal generalization bounds in terms of the margin.

**However**.

In deep learning, anecdotally we seem to be doing just fine even without any regularizers. While regularization may help in some cases, several careful experiments[^zhang] show that regularization is neither necessary nor sufficient for good generalization. Why is that the case?

To resolve this, a large body of recent work (which we introduce below) studies what I will call the *Algorithmic Bias Hypothesis*. This hypothesis asserts that the *dynamics* of the training procedure (specifically, gradient descent) *implicitly* introduces a regularization effect. That is, the algorithm itself is biased towards the choice of solution towards minimum complexity solutions, and therefore the explicit imposition of a regularizer is unnecessary.

**Remark**{:.label #GDvsSGD}
  The results we will prove below hold for both GD as well as SGD, so there is nothing inherently special about the *stochasticity* of the training procedure. The literature is split on this issue, and I am not aware of compelling theoretical benefits of SGD beyond a per-iteration computational speedup. Indeed, even empirical evidence seems to suggest that stochastic training does not provide any particular benefits from the generalization perspective[^gieping].
{:.remark}


## Implicit bias of GD/SGD for linear models
{:.label}

Let's stick with linear models for a bit longer, since the math will be instructive. We start with the setting of linear regression. We are given a dataset $X = \lbrace (x_i,y_i)_{i=1}^n \subset \R^d \times \R \rbrace$. We assume that the model is *over-parameterized*, so the number of parameters is greater than the number of examples $(n < d)$. We would like to fit a linear model $u = \langle w, x \rangle$ that *exactly* interpolates the training data. If we focus on the squared error as the loss function of choice, we seek a (global) minimizer of the loss function:

$$
L(w) = \frac{1}{2}\lVert y - Xw \rVert^2 .
$$

But note that due to over-parameterization, the data matrix is short and wide, and there are many, many candidate models that achieve zero-loss (indeed, an infinity of them, assuming that the data is in general position.) This is where classical statistics would invoke Occam's Razor, or margin theory, or such and start introducing regularizers.

Let us blissfully ignore such suggestions, and instead, starting from $w_0 = 0$, *run gradient descent on the un-regularized loss* $L(w)$. (To be precise, we will run gradient flow, i.e., with infinitesimally small step size -- but similar conclusions can be derived for finite but small step sizes, or for stochastic minibatches, with extra algebra.) Therefore, at any time $t$, we have the gradient flow (GF) ordinary differential equation:

$$
\frac{dw(t)}{dt} = - \nabla L(w(t)) .
$$

We have the following:

**Lemma**{:.label #GFL2}
  Gradient flow over the squared error loss converges to the minimum $\ell_2$-norm interpolator:
  $$
  \begin{aligned}
  w^* = \arg \min_w \lVert w \rVert^2,~\text{s.t.}~y = Xw.
  \end{aligned}
  $$
  In other words, GF provides an implicit bias towards an $\ell_2$-regularized solution.
{:.lemma}

**Proof**{:.label #GFL2Proof}
  This fact is probably folklore, but a proof can be found in Zhang et al.[^zhang]. We only need one key fact: because we are using the squared error loss, at any time $t$ the gradient looks like:

  $$
  -\nabla L(w(t)) = X^T (y - Xw(t)) = \sum_{i=1}^n x_i e_i(t)
  $$

  which is always a vector in the *span* of the data points. This means that if $w(0) = 0$, then the weights at any time $t$ continues to remain in this span. (Geometrically, the span of the data forms an $n$-dimensional subspace in the $d$-dimensional weight space, so the invariance being preserved here is that the path of gradient descent always lies within this subspace.)

  Now, *assume* that GF converges to a global minimizer, $w^* $. We need to be a bit careful and actually prove that global convergence happens. But we already know that GF/GD minimizes convex loss functions, so let us not repeat the calculations from Chapter 4. Therefore, $w^* = X^T \alpha$ for some set of coefficients $\alpha$. But we also know that since $w^* $ is a global minimizer, $Xw^* = y$, which means that

  $$
  XX^T \alpha = y, \quad \text{or} \quad \alpha = (XX^T)^{-1} y.
  $$

  where the inverse exists since the data is in general position. Therefore,

  $$
  w^* =  X^T (XX^T)^{-1} y
  $$
  which is the [Moore-Penrose inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse), or pseudo-inverse, of $X$ applied to $y$. From basic linear algebra we know that the pseudo-inverse gives the minimum-norm solution to the linear system $y = Xw$.
{:.proof}

**Remark**{:.label #flowlines}
  The above proof is algebraic, but there is a simple geometric intuition. Gradient flowlines are [orthogonal](https://dsgissin.github.io/blog/assets/images/implicit_regularization/line_identity_param_grad_bw.png) to the subspace of all possible solutions. (Complete this.)
{:.remark}

Great! For the squared error loss, the path of gradient descent always leads to the lowest-norm solution, regardless of the training dataset.

Does this intuition carry over to more complex settings? The answer is *yes*. Let us now describe a similar result (proved here[^soudry] and here[^jitelg]) for the setting of linear classification. Given a linearly separable training dataset, we discover an interpolating classifier using gradient descent over the $\exp$ loss. (This result also holds for the logistic loss.) Somewhat surprisingly, we show that the limit of gradient descent (approximately) produces the min-norm interpolator, which is nothing but the (hard margin) SVM!

The exact theorem statement needs some motivation. For that, let's first examine the $\exp$ loss, which is somewhat of a strange beast. As defined above, this looks like:

$$
L(w) = \sum_{i=1}^n \exp(- y_i \lang w, x_i \rang) .
$$

A bit of reflection will suggest that the infimum of $L$ is indeed 0, but 'zero-loss' is attained only when the norm of $w$ diverges to infinity. (**Exercise**: Prove this.) Therefore, we will need to define 'convergence' a bit more carefully. The right way to think about it is to look at convergence *in direction*; for linear classification, the norm of $w$ does not really matter --- so we can examine the properties of $\lim w(t)/\lVert w(t) \rVert$ as $t \rightarrow \infty$. We obtain:

**Theorem**{:.label #GFexp}
  Let $w^*$ be the hard-margin SVM solution, or the minimum $\ell_2$-norm interpolator:

  $$
  \begin{aligned}
  w^* = \arg \min_w \lVert w \rVert^2,~\text{s.t.}~y_i \lang w, x_i \rang \geq 1~\forall i \in [n].
  \end{aligned}
  $$

  Then, gradient flow over the $\exp$ loss converges to:

  $$
  w(t) \rightarrow w^* \log t + O(\log \log t).
  $$

  Moreover,

  $$
  \frac{w(t)}{\lVert w(t) \rVert} \rightarrow  \frac{w^*}{\lVert w^* \rVert} .
  $$

  In other words, GF directionally converges towards the $\ell_2$-max margin solution.
{:.theorem}

**Proof sketch**{:.label #GFL2Proof}
  (Complete.)
{:.proof}

**Remark**{:.label #convergence rate}
  Convergence happens, but at rate $\frac{1}{\log (t)}$, which is way slower than all the rates we showed in Chapter 4. (Prove this?)
{:.remark}

Therefore, we have seen that in linear models, gradient flow induces implicit $\ell_2$-bias for:

* the squared error loss, and
* the $\exp$ loss (as well as the logistic loss[^soudry].)

What about other choices of loss? Ji et al.[^ji2020] showed a similar directional convergence result for general *exponentially tailed* loss functions, while clarifying that the limit direction may differ across different losses.

More pertinently, what about other choices of *architecture* (beyond linear models)? We examine this next.

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

[^gieping]:
    J. Geiping, M. Goldblum, P. Pope, M. Moeller, T. Goldstein, [Stochastic Training is Not Necessary for Generalization](https://arxiv.org/abs/2109.14119), 2021.

[^soudry]:
    D. Soudry, E. Hoffer, M. Spiegel Nacson, S. Gunasekar, N. Srebro, [The Implicit Bias of Gradient Descent on Separable Data](https://arxiv.org/pdf/1710.10345.pdf), 2018.

[^jitelg]:
    Z. Ji, M. Telgarsky, [Directional convergence and alignment in deep learning](https://arxiv.org/pdf/2006.06657.pdf), 2020.

[^ji2020]:
    Z. Ji, M. Dudik, R. Schapire, M. Telgarsky, [Gradient descent follows the regularization path for general losses](http://proceedings.mlr.press/v125/ji20a/ji20a.pdf), 2020.
