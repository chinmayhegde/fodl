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

  Now, *assume* that GF converges to a global minimizer, $\bar{w}$. We need to be a bit careful and actually prove that global convergence happens. But we already know that GF/GD minimizes convex loss functions, so let us not repeat the calculations from Chapter 4. Therefore, $\bar{w} = X^T \alpha$ for some set of coefficients $\alpha$. But we also know that since $\bar{w}$ is a global minimizer, $X\bar{w} = y$, which means that

  $$
  XX^T \alpha = y, \quad \text{or} \quad \alpha = (XX^T)^{-1} y.
  $$

  where the inverse exists since the data is in general position. Therefore,

  $$
  \bar{w} =  X^T (XX^T)^{-1} y
  $$
  which is the [Moore-Penrose inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse), or pseudo-inverse, of $X$ applied to $y$. From basic linear algebra we know that the pseudo-inverse gives the minimum-norm solution to the linear system $y = Xw$.
{:.proof}

**Remark**{:.label #flowlines}
  The above proof is algebraic, but there is a simple geometric intuition. Gradient flowlines are [orthogonal](https://dsgissin.github.io/blog/assets/images/implicit_regularization/line_identity_param_grad_bw.png) to the subspace of all possible solutions. Therefore, if we start gradient flow from the origin, then the shortest path to the set of feasible solutions is the straight line segment orthogonal to this subspace, which means that the final answer is the interpolator with minimum $L_2$ norm.
{:.remark}

Great! For the squared error loss, the path of gradient descent always leads to the lowest-norm solution, regardless of the training dataset.

Does this intuition carry over to more complex settings? The answer is *yes*. Let us now describe a similar result (proved here[^soudry] and here[^jitelg]) for the setting of linear classification. Given a linearly separable training dataset, we discover an interpolating classifier using gradient descent over the $\exp$ loss. (This result also holds for the logistic loss.) Somewhat surprisingly, we show that the limit of gradient descent (approximately) produces the min-norm interpolator, which is nothing but the (hard margin) SVM!

The exact theorem statement needs some motivation. For that, let's first examine the $\exp$ loss, which is somewhat of a strange beast. As defined above, this looks like:

$$
L(w) = \sum_{i=1}^n \exp(- y_i \lang w, x_i \rang) .
$$

A bit of reflection will suggest that the infimum of $L$ is indeed 0, but 'zero-loss' is attained only when the norm of $w$ diverges to infinity. (**Exercise**: Prove this.) Therefore, we will need to define 'convergence' a bit more carefully. The right way to think about it is to look at convergence *in direction*; for linear classification, the norm of $w$ does not really matter --- so we can examine the properties of $\lim w(t)/\lVert w(t) \rVert$ as $t \rightarrow \infty$. We obtain the following:

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

## Implicit bias in multilinear models
{:.label}

The picture becomes much more murky (and, frankly, really fascinating) when we move beyond linear models. As we will show below the architecture plays a fundamental role in the gradient dynamics, which further goes to show that *both* representation and optimization method play a crucial role in inducing algorithmic bias.

A linear model of the form $u = \lang w, x \rang$ can be viewed as a *single* neuron with linear activation. Let us persist with linear activations for some more time, but graduate to *multiple layers of neurons*. One such model is called a *two-layer diagonal linear network model*:

$$
u = \sum_{j=1}^d v_j u_j x_j ,
$$

where the weights $(u_j, v_j)$ are trainable. This model, of course, is merely a re-parameterization of a single neuron and can only express linear functions. However, the weights interact multiplicatively, and therefore the output is a *nonlinear function of the weights*. This re-parameterization makes a world of difference in the context of gradient dynamics, and leads to a *very* different form of algorithmic bias, as we will show below.

### Diagonal linear models
{:.label}

For simplicity, let's just assume that the *weights are tied* across layers, i.e., $u_i = v_i$ for $i \in [d]$. I don't believe that this assumption is important; similar conclusions should hold even if no weight tying occurs.

Then, the prediction for any data point $x \in \R^d$ becomes

$$
y = \sum_{j=1}^d x_j u_j^2
$$

or if we stack up the training data then

$$
y = X (u \circ u)
$$

where $\circ$ denotes the element-wise (Hadamard) product. This is called a "two-layer diagonal linear network", and the extension to $L$-layers is analogous. Notice that by using this re-parameterization, we have not done very much in terms of expressiveness.

(*Actually, the last statement is not very precise; square reparameterization only expresses linear models $y = Xw$, but with the added restriction that $w \geq 0$. But this can be fixed easily as follows: introduce 2 sets of variables $u$ and $v$, and write $y = X(u\circ u - v\circ v)$.*)

Let's stick with the squared error loss:

$$
L(u) = \frac{1}{2} \lVert y - X(u \circ u) \rVert^2
$$

and suppose we perform gradient flow over this loss. Since $L$ is a fourth-order polynomial (quartic) function of $u$, the ODE governing the gradient at any time instant is a bit more complicated:

$$
\frac{du}{dt} = - \nabla_u L(u) = 2 u \circ X^T (y - X (u \circ u)) .
$$

Notice now that the gradient in fact is now a *cubic* function of $u$. Suppose that we initialize $u = \alpha \mathbf{1}$, where $\mathbf{1}$ is the all-ones vector and $\alpha > 0$ is a scalar. (This is fine since in the end we will only be interested in models with positive coefficients.)

Somewhat curiously, we can prove that if $\alpha$ is small enough then the algorithmic bias corresponding to gradient flow corresponds to $\ell_1$-regularization (i.e., we recover the familiar *basis pursuit* or [Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics)) formulation. Formally, we get the following:

**Theorem**{:.label #GFLasso}
  Let $w^*$ be the (non-negative) interpolator with minimum $\ell_1$-norm:

  $$
  \begin{aligned}
  w^* = \arg \min_w \lVert w \rVert_1,~\text{s.t.}~y = Xw,~w \geq 0.
  \end{aligned}
  $$

  Suppose that gradient flow for diagonal linear networks converges to an estimate $\bar{u}$. Then $\bar{w} = \bar{u} \circ \bar{u}$ converges to $w^*$ as $\alpha \rightarrow 0$.
{:.theorem}

**Proof sketch**{:.label #GFLassoProof}
  This proof is by Woodworth et al.[^woodworth] Let us give high level intuition first. It will be helpful to understand two types of dynamics: one in the original (canonical) linear parameterization in terms of $w$, and the other in the square re-parameterization (in terms of $u$). By definition, we have $w_j = u_j^2$, and therefore

  $$
  \frac{dw_j}{dt} = 2u_j \frac{du_j}{dt} .
  $$

  By plugging in the definition of gradient flow, we get:

  $$
  \begin{aligned}
  \frac{dw}{dt} &= - 2 u(t) \circ \nabla_u L(u(t)) \\
  & =2 u(t) \circ 2 u(t) \circ X^T(y - X(u(t) \circ u(t))) \\
  & = 4 w(t) \circ X^T(y - Xw(t)).
  \end{aligned}
  $$

  This resembles gradient flow over the standard $\ell_2$-loss, *except* that each coordinate is multiplied with $w_i(t)$. Intuitively, this induces a "rich-get-richer" phenomenon: a bigger coefficient in $w(t)$ is assigned a larger gradient update (relative to the other coefficients) and therefore "moves" towards its final destination faster.

  Gissin et al.[^gissin] call this *incremental learning*: large coefficients are learned (approximately) in decreasing order of their magnitude. (See their paper/website for some terrific visualizations of the dynamics.)

  ---

  To rigorously argue that the bias indeed is the $\ell_1$-regularizer, let $e(t) = y - X w(t)$ denote the prediction error at any time instant. Then, as shown above, we have the ODE:

  $$
  \frac{du}{dt} = 2 u(t) \circ X^T e(t) .
  $$

  Solving this equation coordinate-wise, we get the closed form solution:

  $$
  u(t) = u(0) \circ \exp\left( 2 X^T \int_0^t e(s) ds \right) .
  $$

  Plugging in the initialization $u(0) = \alpha \mathbf{1}$ and $w(t) = u(t) \circ u(t)$, we get:

  $$
  w(t) = \alpha^2 \mathbf{1} \circ \exp\left( 4 X^T \int_0^t e(s) ds \right).
  $$

  Therefore, *assuming* the dynamics converges (technically we need to prove this, but let's sweep that under the rug for now), we get that the limit of gradient flow gives:

  $$
  \bar{w} = f(X^T \lambda)
  $$

  where $f$ is a coordinate-wise function such that $f(z):= \alpha^2 \exp(z)$ and $\lambda := 4 \int_0^\infty e(s) ds$. Since $\alpha > 0$, $f$ is bijective and therefore

  $$
  X^T \lambda = f^{-1}(\bar{w}) = \log \frac{\bar{w}}{\alpha^2} .
  $$

  Now, *assume* that the algorithmic bias of GF can be expressed by some unique regularizer $Q(w)$. (Again, we are making a major assumption here that the bias is expressible in the form of a well-defined function of $w$, but let's do more rug+sweeping here.) We will show that $Q(w)$ approximately equals the $\ell_1$-norm of $w$. We get:

  $$
  \bar{w} = \arg \min_w Q(w), \quad y = Xw.
  $$

  KKT optimality tells us that the optimum ($\bar{w}$) should satisfy:

  $$
  y = X\bar{w}, \quad \nabla_w Q(\bar{w}) = X^T \lambda.
  $$

  for some vector $\lambda$. Therefore, we can reverse-engineer $Q$ by solving the differential equation:

  $$
  \nabla_w Q(w) = \frac{1}{\alpha^2} \log w .
  $$

  This (again) has the closed form solution:

  $$
  \begin{aligned}
  Q(w) &= \sum_{i=1}^d \int_{0}^{w_i} \log \frac{w_i}{\alpha^2} dw_i \\
  &= \sum_{i=1}^d w_i \log \frac{w_i}{\alpha^2} - w_i \\
  &= \sum_{i=1}^d 2w_i \log \frac{1}{\alpha} +  \sum_{i=1}^d (w_i \log w_i - w_i) .
  \end{aligned}
  $$

  Therefore, as $\alpha \rightarrow 0$, the first term starts to (significantly) dominate the second, and we can therefore write:

  $$
  Q(w) \approx C_\alpha \sum_{i=1}^d w_i \propto \lVert w \rVert_1 ,
  $$

  due to the fact that $w$ is non-negative by definition.
{:.proof}


**Remark**{:.label #kernelvsrich}
  The above proof shows (again) that the algorithmic bias of gradient descent *highly* depends on the initialization; the fact that $\alpha \rightarrow 0$ was crucial in establishing $\ell_1$-bias. On the other hand, a *different* initialization with $\alpha \rightarrow \infty$ leads to a completely different bias: GF automatically provides $\ell_2$-regularization.[^woodworth] Woodworth et al. categorize these two distinct behaviors of gradient descent as respectively operating in the "rich" and "kernel" regimes.
{:.remark}

**Remark**{:.label #shape}
  The fact that $u(0)$ was initialized with a constant vector (with magnitude $\alpha$) is unimportant; one can impose different forms of bias with varying choices (or "shapes") of the initialization[^woodworth2].
{:.remark}

**Remark**{:.label #shape}
  At its core, the net effect of square reparameterization is to alter the *optimization geometry* of gradient descent. This can be connected to classical optimization theory in general Banach spaces, and indeed the dynamics of diagonal linear networks can be viewed as a form of *mirror descent* (MD). See here[^optgeom] and here[^mirror] for a more thorough treatment.
{:.remark}

**Remark**{:.label #noise}
  All of the above assume that the limit of gradient flow admits an exact interpolation. In the presence of *noise* in the labels the situation is a bit more subtle, and a more careful analysis of the ODEs is necessary; in some cases, for consistent generalization, *early stopping* may be necessary. See our paper for details[^li].
{:.remark}

**Remark**{:.label #matrixfactorization}
  (Two-layer dense nets and nuclear norm regularization) (**complete**).
{:.remark}

## Implicit bias of gradient descent in nonlinear networks
{:.label}

As we saw above, changing the architecture fundamentally changes the gradient dynamics (and therefore the implicit regularization induced during training.) But our discussion above focused on networks with linear activations. How does the picture change with more standard networks (such as those with ReLU activations?)

Not surprisingly, life becomes harder when dealing with nonlinearities. Yet, the tools above pave a way to understand algorithmic bias for such networks. Let's start simple and work our way up.

### Single neurons
{:.label}

Consider a network consisting of a single neuron with nonlinear activation $\psi : \R \rightarrow \R$. The model becomes:

$$
u = \psi(\langle w,x \rangle),
$$

and we will use the squared error loss to train it:

$$
L(w) = \frac{1}{2} \lVert y - \psi(Xw) \rVert^2
$$

where $X$ is the data matrix. We train the above (single-neuron) network using gradient flow. The following proof is by Vardi and Shamir[^vardi3].

**Theorem**{:.label #GFNeuron}
  Suppose that $\psi : \R \rightarrow \R$ is a monotonic function. If gradient flow over $L(w)$ converges to zero loss, then the limit $\bar{w}$ automatically satisfies:

  $$
  \begin{aligned}
  w^* = \arg \min_w \lVert w \rVert^2,~\text{s.t.}~y = \psi(Xw).
  \end{aligned}
  $$

  In other words, GF provides an implicit bias towards an $\ell_2$-regularized solution.  
{:.theorem}
**Proof**{:.label #GFNeuronProof}
  Reduce to linear case using monotonicity. **(complete)**
{:.proof}

Activation functions such as leaky-ReLU, sigmoid etc do satisfy monotonicity. (For sigmoid, convergence to zero loss is a bit tricky because of non-convexity.) What about the regular ReLU? This is more difficult, and there does not appear to be a clean "norm"-based regularizer.  

### Multiple layers
{:.label}

Two-layer nets. **complete**.

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

[^gissin]:
    D. Gissin, S. Shalev-Shwartz, A. Daniely, [The Implicit Bias of Depth: How Incremental Learning Drives Generalization](https://openreview.net/pdf?id=H1lj0nNFwB), 2020.

[^woodworth]:
    B. Woodworth, S. Gunasekar, J. Lee, E. Moroshko, P. Savarse, I. Golan, D. Soudry, N. Srebro, [Kernel and Rich Regimes in Overparametrized Models](http://proceedings.mlr.press/v125/woodworth20a/woodworth20a.pdf), 2020.

[^woodworth2]:
    S. Azulay, E. Moroshko, M. Naczon, N. Srebro, B. Woodworth, A. Globerson, D. Soudry, [On the Implicit Bias of Initialization Shape: Beyond Infinitesimal Mirror Descent](http://proceedings.mlr.press/v139/azulay21a/azulay21a.pdf), 2021.

[^optgeom]:
    S. Gunasekar, J. Lee, D. Soudry, N. Srebro, [Characterizing Implicit Bias in Terms of Optimization Geometry](http://proceedings.mlr.press/v80/gunasekar18a/gunasekar18a.pdf), 2018.

[^mirror]:
    S. Gunasekar, B. Woodworth, N. Srebro, [Mirrorless Mirror Descent: A Natural Derivation of Mirror Descent](http://proceedings.mlr.press/v130/gunasekar21a/gunasekar21a.pdf), 2021.

[^li]:
    J. Li, T. Nguyen, C. Hegde, R. Wong, [Implicit Sparse Regularization: The Impact of Depth and Early Stopping](https://arxiv.org/pdf/2108.05574.pdf), 2021.

[^vardi3]:
    G. Vardi and O. Shamir, [Implicit Regularization in ReLU Networks with the Square Loss](https://arxiv.org/pdf/2012.05156.pdf), 2020.
