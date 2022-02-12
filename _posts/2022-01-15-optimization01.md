---
layout: page
title: Chapter 4 - Optimization primer
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

> (S)GD converges to (near) stationary points, provided $L(w)$ is smooth.

The last caveat --- that $L(w)$ is required to be smooth --- excludes several widely used architectures (such as ReLU networks). This should not deter us. The analysis is still very interesting and useful; it is still applicable to other widely used architectures; and extensions to ReLUs can be achieved with a bit more technical heavy lifting (which we won't cover here).

## Gradient descent
{:.label}

We start with analyzing gradient descent for smooth loss functions. (As asides, we will also discuss application of our results to *convex* functions, but these functions are not common in deep learning and therefore not our focus.)

Let us first define smoothness. (All norms below refer to the 2-norm, and all gradients are with respect to the parameters.)

**Definition**{:.label #Smoothness}
  $L$ is said to be $\beta$-smooth if $L$ has $\beta$-Lipschitz gradients:

  $$
  \lVert \nabla L(w) - \nabla L(u) \rVert \leq \beta \lVert w - u \rVert .
  $$

{:.definition}

With [some algebra](https://xingyuzhou.org/blog/notes/Lipschitz-gradient), one can arrive at the following lemma.

**Lemma**{:.label #Quadratic}
  If $L$ is twice-differentiable and $\beta$-smooth, then the eigenvalues of its Hessian are less than $\beta$:

  $$
  \lVert \nabla^2 L(w) \rVert \preceq \beta I.
  $$

  Equivalently, $L(w)$ is *upper-bounded* by a quadratic function:

  $$
  L(w) \leq L(u) + \langle \nabla L(u), w-u \rangle + \frac{\beta}{2} \lVert w-u \rVert^2 .
  $$

{:.lemma}

Basically, the smoothness condition (or its implications in the above [Lemma](#Quadratic)) says that if $\beta$ is reasonable, then the gradients of $L(w)$ are rather well-behaved.

It is intuitive why something like this condition is needed to analyze GD: if the gradient was not well-behaved, then first-order methods such as GD are not likely to be very informative.

There is a *second* natural reason why this definition is relevant to GD. Imagine, for a moment, not minimizing $L(w)$, but rather minimizing the *upper bound*:

$$
B(w) := L(u) + \langle \nabla L(u), w-u \rangle + \frac{\beta}{2} \lVert w-u \rVert^2 .
$$

This is a convex (in fact, quadratic) function of $w$. Therefore, it can be optimized very easily by setting $\nabla B(w)$ to zero and solving for $w$:

$$
\begin{aligned}
\nabla B(w) &= 0, \\
\nabla L(u) &+ \beta (w - u) = 0, \qquad \text{and thus} \\
w &= u - \frac{1}{\beta} \nabla L(u) .
\end{aligned}
$$

This is precisely the same as a *single* step of gradient descent starting from $u$ (with step size inversely proportional to the smoothness parameter). In other words, gradient descent is nothing but the successive optimization of a Lipschitz upper bound of *in every iteration*[^oco].

We are now ready to prove our first result.

**Theorem**{:.label #GDLipchitz}
  If $L$ is $\beta$-smooth, then GD with fixed step size converges to stationary points.
{:.theorem}

**Proof**{:.label #GDLipchitzProof}
  Consider any iteration of GD (with a step size $\eta$ which we will specify later):

  $$
  w = u - \eta \nabla L(u) .
  $$

  This means that:

  $$
  w - u = \eta \nabla L(u).
  $$

  Plug this value of $w-u$ into the quadratic upper bound to get:

  $$
  \begin{aligned}
  L(w) &\leq L(u) - \eta \lVert \nabla L(u) \rVert^2 + \frac{\beta \eta^2}{2} \lVert \nabla L(u) \rVert^2, \quad \text{or} \\
  L(w) &\leq L(u) - \eta \left( 1 - \frac{\beta \eta}{2} \right) \lVert \nabla L(u) \rVert^2 .
  \end{aligned}
  $$

  This inequality already gives a proof of convergence. Suppose that the step size is small enough such that $\eta < \frac{2}{\beta}$. We are in one of two situations:

  1. Either $\nabla L(u) = 0$, in which case we are done since $u$ is a stationary point.

  2. Or $\nabla L(u) \neq 0$, in which case $\lVert \nabla L(u) \rVert > 0$ and the second term in the right hand side is strictly positive. Therefore GD makes progress (and decreases $L$ in the next step).

  Since $L$ is lower bounded by 0 (since we have assumed a non-negative loss), we get [convergence](https://en.wikipedia.org/wiki/Monotone_convergence_theorem).

  This argument does not quite tell us how *many* iterations are needed by GD. For that, let's just set $\eta = \frac{1}{\beta}$. This is not precisely necessary and some wiggle room is okay, but the algebra becomes simpler. Let's just rename $w_i := u$ and $w_{i+1} := w$. Then, the last inequality becomes:

  $$
  \frac{1}{2\beta} \lVert \nabla L(w_i) \rVert^2 \leq L(w_i) - L(w_{i+1}).
  $$

  Telescoping from $w_0, w_1, \ldots, w_t$, we get:

  $$
  \begin{aligned}
  \frac{1}{2\beta} \sum_{i = 0}^{t-1} \lVert \nabla L(w_i) \rVert^2 &\leq L(w_0) - L(w_t) \\
  &\leq L(w_0) - L_{\text{opt}}
  \end{aligned}
  $$

  (since $L_\text{opt}$ is the smallest achievable loss.) Therefore, if we pick $i = \arg \min_{i < t} \nabla \lVert L(w_i) \rVert^2$ as the estimate with lowest gradient norm and set $\hat{w} = w_{i}$, then we get:

  $$
  \frac{t}{2\beta} \lVert L(w_t) \rVert^2 \leq L_0 - L_{\text{opt}},
  $$

  which implies that if $L_0$ is bounded (i.e.: we start somewhere reasonable) then GD finds a point $\hat{w}$ within $t$ iterations whose gradient norm is

  $$
  \lesssim \sqrt{\frac{\beta}{t}}
  $$

  at most. Alternatively, to find an $\varepsilon$-stationary point, GD needs:

  $$
  O\left( \frac{\beta}{\varepsilon^2} \right)
  $$

  iterations.
{:.proof}

This proof is very simple; we didn't use anything much beyond the definition of Lipschitz smoothness. But it already reveals a lot.

First, step sizes in standard GD can be constant, and should be chosen inversely proportional to $\beta$. This makes intuitive sense: if $\beta$ is large then gradients are wiggling around, and therefore it is prudent to take small steps.

Second, we only get convergence in the "neighborhood" sense (in that there is some point along the trajectory which is close to the stationary point). It is harder to prove "last-iterate" convergence results. In fact, one can even show that GD can go near a stationary point, spend a very long time near this point, but then bounce away later[^leegd].

Third, we get $\frac{1}{\sqrt{t}}$ error after $t$ iterations. The terminology to describe this error rate is not very consistent in the optimization literature, but one might call this a "sub-linear" rate of convergence.

---
Aside: $L$ smooth *and convex*? Linear ($\frac{1}{t}$) rate of convergence.

---

Aside: $L$ smooth and *strongly convex*? Exponential ($e^{-t}$) rate of convergence.

---

So the hierarchy of convergence rates is as follows:

* GD assuming Lipschitz smoothness: $\frac{1}{\sqrt{t}}$ rate

* GD assuming Lipschitz smoothness + convexity: $\frac{1}{t}$ rate

* Momentum accelerated GD: $\frac{1}{t^2}$ rate. Remarkably, this is the *best possible* one can do with first-order methods such as GD.

* GD assuming Lipschitz and strong convexity: $\exp(-t)$ rate .


### The PL condition
{:.label}

Above, we saw how leveraging smoothness, along with strong convexity, of the loss results in exponential convergence of GD. However (strong) convexity is not that relevant in the context of deep learning. This is because losses are very rarely convex in their parameters.

However, there is a different characterization of functions (other than convexity) that also implies fast convergence rates of GD. This property was introduced by Polyak[^polyak] in 1963, but has somehow not been very widely publicized. It was re-introduced to the ML optimization literature by Karimi et al.[^karimi] and its relevance (particularly in the context of deep learning) is slowly becoming apparent.

**Definition**{:.label #PLCondition}
  A function $L$ (whose optimum is $L_{\text{opt}}$) is said to satisfy the Polyak-Lojasiewicz (PL) condition with parameter $\alpha$ if:

  $$
  \lVert \nabla L(u) \rVert^2 \geq 2 \alpha (L(u) - L_{\text{opt}} )
  $$

  for all $u$ in its domain.
{:.definition}

The reason for the "2" sitting before $\alpha$ will become clear. Intuitively, the PL condition says that if $L(u) \gg  L_{\text{opt}}$ then $\lVert \nabla L(u) \rVert$ is also large. More precisely, the norm of the gradient at any point grows at least as the square root of the (functional) distance to the optimum.

Notice that there is no requirement of convexity in the definition of the PL condition. For example, the function:

$$
L(x) = x^2 + 3 \sin^2 x
$$

looks like

![Plot of $L(x)$](/fodl/assets/pl-example.png)

which is non-convex upon inspection, but nonetheless satisfies the PL condition with constant $\alpha = \frac{1}{32}$.

(The converse is true. Strong convexity implies the PL condition, but PL is far more general. We return to this in Chapter 6 in the context of neural nets.)

We immediately get the following result.

**Theorem**{:.label #GDPL}
  If $L$ is $\beta$-smooth and satisfies the PL condition with parameter $\alpha$, then GD exponentially converges to the optimum.
{:.theorem}

**Proof**
  Proof follows trivially. Let $\eta = \frac{1}{\beta}$. Then

  $$
  w_{t+1} = w_t - \frac{1}{\beta} \nabla L(w_t) .
  $$

  Plug $w_{t+1} - w_t$ into the smoothness upper bound. We get:

  $$
  L(w_{t+1}) \leq L(w_t) - \frac{1}{2\beta} \lVert \nabla L(w_t) \rVert^2 .
  $$

  But since $L$ satisfies PL, we get:

  $$
  L(w_{t+1}) \leq L(w_t) - \frac{\alpha}{\beta} \left( L(w_{t+1}) - L(w_\text{opt}) \right).
  $$

  Simplifying notation, we get:

  $$
  L_{t+1} - L_{\text{opt}} \leq \left(1 - \frac{\alpha}{\beta} \right) \left( L_{t} - L_{\text{opt}} \right) .
  $$

  which implies that GD converges at $\exp(-\frac{\alpha}{\beta}t)$ rate.
{:.proof}

## Stochastic gradient descent
{:.label}

How does the picture change with inexact (stochastic) gradients?

Need to define "convergence" more carefully/liberally.

Hierarchy:

* SGD assuming Lipschitz smoothness: $\frac{1}{t^{1/4}}$

* SGD assuming Lipschitz smoothness + convexity: $\frac{1}{\sqrt{t}}$

Other rates?

## Extensions
{:.label}

* Nesterov momemtum



---

[^adam]:
    D. Kingma and J. Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), 2014.

[^losslandscape]:
    H. Li, Z. Xu, G. Taylor, C. Studer, T. Goldstein, [Visualizing the Loss Landscape of Neural Nets](https://proceedings.neurips.cc/paper/2018/file/a41b3bb3e6b050b6c9067c67f663b915-Paper.pdf), 2018.

[^zhang]:
    C. Zhang, S. Bengio, M. Hardt, B. Recht, O. Vinyals, [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530), 2017.

[^oco]:
    There is an entire literature on online optimization that uses this interpretation of gradient descent, but they call it the "Follow-the-leader" strategy. See [here](https://courses.cs.washington.edu/courses/cse599s/14sp/scribes/lecture3/lecture3.pdf) for an explanation.

[^leegd]:
    S. Du, C. Jin, J. Lee, M. Jordan, B. Poczos, A. Singh, [Gradient Descent Can Take Exponential Time to Escape Saddle Points](https://arxiv.org/abs/1705.10412), 2017.

[^polyak]:
    B. Polyak, [ГРАДИЕНТНЫЕ МЕТОДЫ МИНИМИЗАЦИИ ФУНКЦИОНАЛОВ (Gradient methods for minimizing functionals)](http://www.mathnet.ru/links/b7971e9e07cc44b5d0e9fb6354f11988/zvmmf7813.pdf), 1963.

[^karimi]:
    H. Karimi, J. Nutini, M. Schmidt, [Linear Convergence of Gradient and Proximal-Gradient Methods Under the Polyak-Lojasiewicz Condition](https://arxiv.org/pdf/1608.04636.pdf), 2016.
