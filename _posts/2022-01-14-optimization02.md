---
layout: page
title: Chapter 5 - Optimizing wide networks
categories: optimization
date: 2022-01-14
---

We have proved that (S)GD converges locally for training neural networks with smooth activations (ReLU nets are harder but doable.)

However, in practice, somewhat puzzlingly we regularly see that we can train networks to zero loss. Since losses are non-negative, this means that we have achieved convergence to the *global* optimum.

In this chapter, we ask the question:

```
When and why does (S)GD give zero train loss?
```

A full answer is not yet available. However, we will prove that for *very wide* networks that are *randomly initialized*, this is indeed the case. Proving this will give us several surprising insights, and connections to classical ML (such as kernel methods).

## Local versus global minima
{:.label}


## The Neural Tangent Kernel
{:.label}

## Extensions
{:.label}
