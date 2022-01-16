---
layout: default
title: Proof
---

# hi.
To start publishing, please see [the manual]({{ site.baseurl }}/manual/).

---

## posts.
{% for item in site.posts %}
* [{{ item.title }}]({{ site.baseurl }}{{ item.url }}){% if item.author %} by {{ item.author }}{% endif %}{% endfor %}
