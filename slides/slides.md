---
#addons:
# - slidev-addon-python-runner
theme: ./slidev-theme-cscs
# theme: seriph
# background: images/title-bg3.png
title: '19th SPHERIC World Conference'
info: |
  https://spheric2025.upc.edu

  Sources available at https://github.com/jgphpc/
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

## In-situ Visualization for SPH Simulations

Jean-Guillaume Piccinali$^{\propto}$, Jean M. Favre$^{\propto}$, Rubén Cabezón$^\rho$<br>
<small>Swiss National Supercomputing Centre$^{\propto}$, University of Basel$^\rho$</small><br>

19th SPHERIC World Conference, https://spheric2025.upc.edu<br>
<small>18th June 2025</small>

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
src: ./src/spheric25.md
---

<!-- 
info: false
src: ./src/defs/defs.md
hide: false
-->
