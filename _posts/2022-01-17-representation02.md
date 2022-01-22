---
layout: page
title: Universal approximators
categories: representation
date: 2022-01-17
---

Previously, we visited several results that showed how (shallow) neural networks can effectively memorize training data. However, memorization of a finite dataset may not the end goal[^fn1]. In the ideal case, we would like to our network to simulate a (possibly complicated) prediction function that works well on most input data points. Is a given architecture good enough?

In this note we will describe the representation power of (shallow) neural networks in terms of their ability to approximate continuous functions. This line of work has a long and rich history (and function approximation itself, independent of the context of neural networks, is a vast body of work which we only barely touch upon). See here[^devore] for a recent (and fantastic) survey of this area.

As before, intuition tells us that an infinite number of neurons should be good enough for approximating pretty much anything. Therefore, our guiding principle will be to achieve as *succinct* a neural representation as possible. Moreover, if there is an *efficient computational* routine that gives this representation, that would put the icing on the cake.

## Warmup: Function approximation
{:.label}

<script>
macros["\\f"] = "\\mathscr{F}"
</script>

Let's start simple. This time, we don't have any training data to work with; let's just assume we seek some (purported) prediction function $g(x)$. To approximate $g$, we have a candidate hypothesis class $\f$ of shallow (two-layer) neural networks of the form:

\\[ f(x) = \sum_{i=1}^m \alpha_i \psi(\langle w_i, x \rangle + b_i) . \\]

Our goal is to get reasonable bounds on how large $m$ needs to be in terms of various parameters of $g$. We have to be clear about what "approximate" means here. It is typical to measure approximation in terms of $p$-norms between measurable functions; for example, in the case of $L_2$-norms we will try to control

$$
\int_{\text{dom}(g)} |f -g|^2 d \mu
$$

where $\mu$ is some measure defined over $\text{dom}(g)$. Likewise for the $L_\infty$- (or the sup-)norm, and so on.

Let us first define a useful property to characterize univariate functions.

**Definition (univariate Lipschitz)**{:.label #Lipschitz}
  A function $g : \R \rightarrow \R$ is $L$-Lipschitz if for all $u,v \in \R$, we have that $|f(u) - f(v) | \leq L |u - v|$.
{:.definition}

Why is this an interesting property? Any smooth function with bounded derivative is Lipschitz; actually, certain non-smooth functions (such as the ReLU) are also Lipschitz. Lipschitz-ness does not quite capture everything we care about (e.g. discontinuous functions are not Lipschitz, which can be somewhat problematic if there are "jumps" in the label space) but it serves as a large enough class of functions to prove interesting results.

An additional benefit is because of approximability. If our target function $f$ is Lipschitz continuous with small $L$, then we can easily show that it can be well-approximated by a two-layer network with threshold activations: $\psi(z) = \mathbb{I}(z \geq 0)$.

**Theorem**{:.label #univariatesimple}
  Let $g : [0,1] \rightarrow \R$ be $L$-Lipschitz. Then, it can be  $\varepsilon$-approximated in the sup-norm by a two-layer network with $O(\frac{L}{\varepsilon})$ hidden threshold neurons.
{:.theorem}

**Proof**{:label #univariatesimpleproof}
  A more careful derivation of this fact (and the next one below) can be found in Telgarsky[^mjt]. The proof follows from the same picture we might have seen while first learning about integrals and Riemann sums. The high level idea is to tile the interval $[0,1]$ using "buildings" of appropriate height. Since the derivatives are bounded (due to Lipschitzness), the top of each "building" cannot be too far away from the corresponding function value.

  Specifically, partition $[0,1]$ into equal intervals of size $\varepsilon/L$. Let the $i$-th interval be $[u_i,u_{i+1})$. Define a sequence of functions $f_i(x)$; each $f$ is zero everywhere, except that within this interval it attains the value $g(u_i)$. Then $f_i$ is the difference of two threshold functions:

  $$
  f_i(x) = g(u_i) \left(\psi(x - u_i) - \psi(x - u_{i+1})\right).
  $$

  Our network will be the sum of all the $f_i$'s (there are $L/\varepsilon$ of them). Moreover, it is clear that for any $x \in [0,1]$, if $u_i$ is the left end of the interval corresponding to $x$, then we have:

  $$
  \begin{aligned}
  |f(x) - g(x)| &= |g(x) - g(u_i)| \\
  &\leq L |x - u_i | \qquad \text{(Lipschitzness)} \\
  &\leq L \frac{\varepsilon}{L} = \varepsilon,
  \end{aligned}
  $$

  Taking the supremum over all $x \in [0,1]$ completes the proof.
{:.proof}

**Remark**{:.label #UnivarianteRem1}
So we can approximate $L$-Lipschitz functions with $O(L/\varepsilon)$ threshold neurons. Would the answer change if we used ReLU activations? (Hint: no, up to constants; prove this.)
{:.remark}

Of course, in deep learning we rarely care about univariate functions (i.e., where the input is 1-dimensional). We can ask a similar question in the more general case. Suppose we have $L$-Lipschitz functions over $d$ input variables and we want to approximate it using shallow neural networks. How many neurons do we need?

We answer this question using two approaches. First, we give a construction using standard real analysis that uses *two* hidden layers of neurons. Then, with some more mathematical powerful machinery we will get better (and much more general) results with only one hidden layer (i.e., using the hypothesis class $\f$).

First, we have to define Lipschitzness for $d$-variate functions.

**Definition (multivariate Lipschitz)**{:.label #Lipschitz}
  A function $g : \R^d \rightarrow \R$ is $L$-Lipschitz if for all $u,v \in \R$, we have that $|f(u) - f(v) | \leq L \|u - v\|_\infty$.
{:.definition}


**Theorem**{:.label #multivariatesimple}
  Let $g : [0,1]^d \rightarrow \R$ be $L$-Lipschitz. Then, $g$ can be  $\varepsilon$-approximated in the $L_1$-norm by a three-layer network $f$ with $O(\frac{L}{\varepsilon^d})$ hidden threshold neurons.
{:.theorem}

**Proof sketch**{:label #multivariateproof}
  The proof follows the above construction for univariate functions. We will tile $[0,1]^d$ with equally spaced multidimensional rectangles; there are $O(\frac{1}{\varepsilon}^d)$ of them. The value of the function $f$ within each rectangle will be held constant (and due to the definition of Lipschitzness, the error with respect to $g$ cannot be too large). If we can figure out how to approximate $g$ within each rectangle, then we are done.

  The key idea is to figure out how to realize "indicator functions" for every rectangle. We have seen that in the univariate case, indicators can be implemented using the difference of two threshold neurons. In the $d$-variate case, an indicator over a rectangle is the *Cartesian product* over the $d$ axis. however, Boolean/Cartesian products can be implemented by a layer of threshold activations *on top* of these differences.

  Formally, consider any arbitrary piece with $[u_j,v_j], j=1,2,\ldots,d$ as sides. The domain can be written as the Cartesian product:

  $$
  S = \times_{j=1}^d (u_j, v_j).
  $$

  Therefore, we can realize an indicator function over this domain as follows. Localize within each coordinate by the "difference-of-threshold neurons":

  $$
  h_j(z) = \psi(z-v_j) - \psi(z - u_j)
  $$

  and implement the entire rectangle is implemented via a "Boolean AND":

  $$
  h(x) = \psi(\sum_{j=1}^d h_j(x_j) - (d-1)),
  $$
  where $x_j$ is the $j$-th coordinate of $x$. There is one such $h$ for every rectangle, and the output edge from this neuron is assigned a constant value approximating $g$ within that rectangle. This completes the proof.
{:.proof}


## Universal approximation theorems
{:.label}

## The method of Barron
{:.label}


---

[^fn1]:
    Although: exactly interpolating training labels seems standard in modern deep networks; see [here](https://paperswithcode.com/sota/image-classification-on-cifar-10) and Fig 1a  of [this paper](https://arxiv.org/pdf/1611.03530.pdf).

[^devore]:
    R. DeVore, B. Hanin, G. Petrova, [Neural network approximation](https://arxiv.org/pdf/2012.14501.pdf), 2021.

[^mjt]:
    M. Telgarsky, [Deep Learning Theory](https://mjt.cs.illinois.edu/dlt/), 2021.
