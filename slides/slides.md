---
#addons:
# - slidev-addon-python-runner
theme: ./slidev-theme-cscs
# theme: seriph
# background: images/title-bg3.png
title: 'ISC 2025 in-situ visualization workshop'
info: |
  https://woiv.gitlab.io/woiv25/#program

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

### Issues and challenges of deploying in-situ visualization for SPH codes

Jean M. Favre and Jean-Guillaume Piccinali<br>
Swiss National Supercomputing Centre (CSCS)<br>

WOIV\'25, 9th International Workshop on In Situ Visualization,
 https://woiv.gitlab.io/woiv25<br>
13th June 2025

---

# Outline

* Smoothed Particle Hydrodynamics (SPH)
* Testing multiple visualization algorithms in-situ
* DummySPH: testing three visualization backends
* Execution mode
* Describing data (sharing memory pointers)
* Device-resident data support (CUDA)
   * The special case of NVIDIA Grace-Hopper
* Special topic: Derived quantities
* Production runs
* Summary & Future work
* Questions

---
info: false
src: ./src/defs/defs.md
hide: false
---

