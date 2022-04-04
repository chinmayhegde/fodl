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

### Agnostic (PAC) learning

### Data-dependent bounds

## Error bounds for deep networks
{:.label}

## PAC-Bayes
{:.label}
