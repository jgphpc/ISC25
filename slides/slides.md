---
#addons:
# - slidev-addon-python-runner
theme: ./slidev-theme-cscs
# theme: seriph
# background: images/title-bg3.png
title: 'CSCS: bla bla '
info: |
  Presentation slides for the CSCS webinar.

  Sources available at https://github.com/jfavre/
favicon: /images/cscs.ico
# apply unocss classes to the current slide
# class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
# python runner config
# python:
#   installs: []
# fonts:
#   sans: Arial
#   seriph: Arial
#   mono: Victor Mono
# lineNumbers: true

hideInToc: true
# background: https://cover.sli.dev
# background: /scorep/logo.png
---

# Titre principal

Swiss National Supercomputing Centre (CSCS)<br/>
15 June 2025<br/>


---
layout: two-cols
layoutClass: gap-16
hideInToc: true
---

# Outline

<br>
<br>
<div class="flex justify-left">
    <img src="/scorep/workflow.png" class="h-65" border="0px">
</div>


::right::

<br>
<br>
<Toc text-sm minDepth="1" maxDepth="1" column="2"/>

---
src: ./src/scorep/scorep.md
hide: false
---

# the end
---
