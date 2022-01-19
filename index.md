---
layout: default
title: Foundations of Deep Learning
---

# Overview

The impact of deep neural networks in numerous application areas of science, engineering, and technology has never been higher than right now.

Still, progress in practical applications of deep learning has considerably outpaced our understanding of its foundations. Many fundamental questions remain unanswered. Why are we able to train neural networks so efficiently? Why do they perform so well on unseen data? Is there any benefit of one network architecture over another?

These lecture notes are an attempt to sample a growing body of work in theoretical machine learning research that address some of these questions. They supplement a graduate level course taught by [me](https://chinmayhegde.github.io/) in the Spring of 2022.

All pages are under construction. Corrections, pointers to omitted results, and other feedback are welcome: just email me, or open a Github pull request at this [repository](https://github.com/chinmayhegde/fodl).

---

## Table of contents
{% for item in site.posts %}
* [{{ item.title }}]({{ site.baseurl }}{{ item.url }}){% if item.author %} by {{ item.author }}{% endif %}{% endfor %}
