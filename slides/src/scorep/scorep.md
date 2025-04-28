---
layout: section
---

# Getting Started with Score-P/Scalasca

<div class="flex justify-center">
    <img src="/scorep/logo_scorep_small.png" class="h-45" alt="logo">
</div>

<!-- {{{ VIHPS -->

---

## 

<div class="flex justify-left">
    <img src="/scorep/logo_vi-hps.png" class="h-10" alt="vi-hps" border="0px">
<!--    <img src="/scorep/logo_scorep_small.png" class="h-10" alt="scorep" border="0px"> -->
</div>
<br>

- Performance analysis is the process of <span v-mark.highlight.yellow>measuring the performance</span> of an application with a tool

- The [`Virtual Institute for High Productivity Supercomputing`](https://www.vi-hps.org) [(www.vi-hps.org)](https://www.vi-hps.org)<br>
develops an open source Scalable Performance Measurement Infrastructure for Parallel Codes

<img src="/scorep/logo_scorep_small.png" class="h-10" alt="scorep" border="0px">

- [`Score-P`](https://www.vi-hps.org/tools/score-p.html) is one of the
<span v-mark.highlight.yellow>performance analysis tool</span> supported by VI-HPS

    - <span v-mark.highlight.yellow>Profiling</span> gives a statistical overview of the measured performance
        - It can help to quickly <span v-mark.underline.yellow>identify</span> hotspots or performance bottlenecks

    - <span v-mark.highlight.yellow>Tracing</span> records a detailed timeline of events
        - It can help to <span v-mark.underline.yellow>understand</span> performance issues with a higher level of details (and overhead)


<!-- }}} -->

<!-- {{{ Compiling -->

---

## Score-P: Compiling your code with the tool

<br>

- Load the <span v-mark.highlight.yellow>uenv</span> (in /user-tools/):
```bash
uenv pull scorep/9.0:v1 ; uenv start scorep/9.0:v1 --view default
export PATH=/user-tools/env/default/bin:$PATH
find /user-tools/ -name scorep.pdf
```

<br>

- Invoke `cmake` <span v-mark.highlight.yellow>without instrumentation</span>:
&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;
<span v-mark.underline.yellow>then</span> start building <span v-mark.highlight.yellow>with instrumentation</span>:
<div class="grid grid-cols-[48%_50%] gap-2">
<div>
```bash
SCOREP_WRAPPER=OFF cmake \
-DCMAKE_CXX_COMPILER=scorep-mpic++ \
-DCMAKE_C_COMPILER=scorep-mpicc \
-DCMAKE_CUDA_COMPILER=scorep-nvcc \
-S src -B build
```
</div>

<div>
```bash
SCOREP_WRAPPER=ON cmake --build build
```
</div>
</div>

<br>

- <span v-mark.highlight.yellow>Select the report type</span> before running the executable:
```bash
export SCOREP_ENABLE_PROFILING=true # Call-path profiling: CUBE4 data format (profile.cubex)
export SCOREP_ENABLE_TRACING=true # Event-based tracing: OTF2 data format (traces.otf2)
```

<Transform :scale="0.4">
- /user-tools/linux-sles15-neoverse_v2/gcc-12.4.0/scorep-9.0-gwjsoyrgrgz75ngqjkzvdu7so6qs6wke/share/doc/scorep/pdf/scorep.pdf
</Transform>

<!-- }}} -->

---

<br>
<br>
<br>
<br>
<br>
<br>

## Questions?

