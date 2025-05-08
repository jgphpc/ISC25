---
layout: section
---

# Testing multiple visualization filters/algorithm in-situ

---

## Definitions

- rendering: making an image of a [subset] of the particle cloud
- thresholding: selecting all particles given a threshold value on a variable. If thresholding against a coordinate component => spatial clipping
- compositing: making a vector-variable (e.g. velocity) from independent components (e.g. vx, vy, vz)
- histsampling: selecting (sub-sampling) a particle cloud, while retaining regions of higher entropy
- binning: doing queries such as min(), max(), etc on a set of variables


---

### DummySPH: a mini-app with three in-situ visualization backends
https://github.com/jfavre/DummySPH
<br>

- LLNL <span v-mark.highlight.yellow>Ascent</span>
```bash
https://github.com/Alpine-DAV/ascent
https://ascent.readthedocs.io/en/latest/index.html
```

<br>

- Kitware <span v-mark.highlight.yellow>ParaView Catalyst</span>:
&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;
```bash
https://docs.paraview.org/en/latest/Catalyst/index.html
```

<br>

- <span v-mark.highlight.yellow>VTK-m:</span> 
Accelerating the Visualization Toolkit for Massively Threaded
Architectures
```bash
https://gitlab.kitware.com/vtk/vtk-m
https://vtk-m.readthedocs.io/en/stable/index.html
```


---

<br>
<br>
<br>
<br>
<br>
<br>

## Questions?

