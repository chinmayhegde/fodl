---
layout: page
title: Chapter 8 - PAC learning primer and error bounds
categories: generalization
date: 2022-01-11
---

OK, so implicit regularization (via the choice of training algorithm) may be part of the reason for generalization, but almost surely not the entire picture.

It may be useful to revisit classical ML here. Very broadly, the conventional thinking has been:

```
Parsimony enables generalization.
```

The challenge is to precisely define what "parsimony" means here, since there are a myriad different ways of doing this. The bulk of work in generalization theory explores more and more refined complexity measures of ML models, but as we will see below, most existing classical approaches lead to *vacuous* bounds for deep networks. Getting non-vacuous generalization guarantees is a major challenge.

## Warmup: Finite hypothesis classes
{:.label}

Below, $R(f)$ is the population risk, $\hat{R}(f)$ is the empirical risk, and we would like $R(f) - \hat{R}(f)$ to be (a) small, and (b) decreasing as a function of sample size.
**Complete.**

<script>
macros["\\f"] = "\\mathscr{F}"
</script>

**Theorem**{:.label #HoeffdingBound}
  Consider the setting of binary classification, and let $\f$ be a hypothesis class with finitely many elements. If $n \geq O\left( \frac{1}{\varepsilon^2} \log \frac{|\f|}{\delta}\right)$ then with probability at least $1-\delta$, we have that for every $f \in \f$, $|R(f) - \hat{R}(f)| \leq \varepsilon$.
{:.theorem}
**Proof**
  Hoeffding + union bound.
{:.proof}

**Remark**{:.label #GenRemark1}
  Observe that this holds for all $f \in \f$, not just the optimal predictor. Such a bound is called a *uniform convergence* result.
{:.remark}

**Remark**{:.label #GenRemark2}
  This works irrespective of the choice of $\f$ or the data distribution, which could be a weakness of the technique. Even in incompatible cases (for example, the data is highly highly nonlinear, but we only use linear classifiers), one can use the above result that the generalization error is small. The lesson is that we need to control both train and generalization error.
{:.remark}

**Remark**{:.label #GenRemark1}
  Rearranging terms, we get that the generalization error $\lesssim \frac{1}{\sqrt{n}}$, which is very typical. Classical theory tells us that this is a natural stopping point for optimization -- this is the "noise floor" so we don't have to optimize the train error below this level. (*Note: unfortunately, deep learning doesn't obey this, and optimizing to zero is indeed beneficial; this is the "double descent" phenomenon.*)
{:.remark}

**Remark**{:.label #GenRemark1}
  The scaling of the sample complexity looks like $n \approx \log |\f|$, which basically is the number of "bits" needed to encode any particular hypothesis.
{:.remark}


## Complexity measures
{:.label}

Long list of ways to extend the above reasoning. The major drawback is that the above bound is meaningful only for hypothesis classes $\f$ with finite cardinality.  Alternatives:

* Covering number
* VC-dimension
* Pseudo-dimension

### Agnostic (PAC) learning

### Data-dependent bounds

Definition of Rademacher complexity, upper bounds for NN.

### PAC-Bayes bounds

Possibly first approach to produce "non-vacuous" bounds, at least for small networks. Key results: basic approach[^mcallester], application to DL[^dzuigateroy18].

## Error bounds for (deep) neural networks
{:.label}

Key results: here[^bartlett97], here[^bartlett98], here[^bartlett19].

## Possible roadblocks?
{:.label}

All of the above bounds lead to generalization bounds of the form:

$$
\text{Test error} \leq \text{train error}~+~O \left(\frac{\text{complexity measure}}{\sqrt{n}}\right),
$$

and progress has been focused on defining better and better complexity measures. However, two issues with this:

* Usually, the complexity measure in the numerator is far too large anyway, leading to "vacuous" bounds. For example, in [^bartlett19] it reduces to $\text{depth} \times \text{width}$, which is too large in the overparameterized setting.

* This also (seemingly) means that error bounds should decrease with dataset size, for a fixed model class. Not the case :( .


A recent result provides evidence[^nagarajan19] to show that *any* result that uses uniform convergence may suffer from this kind of looseness. We likely need alternate techniques, which we will do next.

---

[^bartlett97]:
    P. Bartlett, [For Valid Generalization the Size of the Weights is More Important than the Size of the Network](https://proceedings.neurips.cc/paper/1996/file/fb2fcd534b0ff3bbed73cc51df620323-Paper.pdf), 1997.

[^bartlett98]:
    P. Bartlett, V. Maiorov, R. Meir, [Almost Linear VC Dimension Bounds for Piecewise Polynomial Networks](https://proceedings.neurips.cc/paper/1998/file/bc7316929fe1545bf0b98d114ee3ecb8-Paper.pdf), 1998.

[^bartlett19]:
    P. Bartlett, N. Harvey, C. Liaw, A. Mehrabian,  [Nearly-tight VC-dimension and Pseudodimension Bounds for Piecewise Linear Neural Networks](https://www.jmlr.org/papers/volume20/17-612/17-612.pdf), 2019.

[^dzuigateroy18]:
    G. K. Dziugaite, D. Roy, [Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data](https://arxiv.org/pdf/1703.11008.pdf), 2017.

[^mcallester]:
    D. Mcallester, [Some PAC-Bayesian Theorems](https://link.springer.com/article/10.1023/A:1007618624809), 1999.

[^nagarajan19]:
    V. Nagarajan, Z. Kolter, [Uniform convergence may be unable to explain generalization in deep learning](https://arxiv.org/pdf/1902.04742.pdf), 2019.
