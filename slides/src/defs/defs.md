---
layout: section

---

# Smoothed Particle Hydrodynamics (SPH)

- Based on a purely Lagrangian technique,
SPH particles are free to move with fluids or deforming solid structures.

- A stimulating field of research to simulate free-surface flows, solid
mechanics, multi-phase, fluid-structure interaction and astrophysics.

- Supporting two SPH codes in production mode at CSCS:

        - SPH-EXA: https://github.com/sphexa-org/sphexa
        - PKDGRAV3: https://bitbucket.org/dpotter/pkdgrav3

Example: [The birth of Jupiter: a first visualization computed on ‘Alps’](https://www.cscs.ch/science/computer-science-hpc/2024/the-birth-of-jupiter-a-first-visualization-computed-on-alps)

--- 

## To be AOS or not to be (SOA)

<br>
<div class="grid grid-cols-[50%_50%] gap-2">
<div> <!-- #left -->

```cpp
Array-of-Structures (AOS)

struct tipsySph {
    float mass;
    float positions[3];
    float velocities[3];
    float density;
    float temperature;
    float phi;
}
std::vector<tipsySph> scalarsAOS;
int NbofScalarfields = sizeof(tipsySph)/sizeof(float);
```
</div>

<div> <!-- #right -->
```cpp
Structure-of-Arrays (SOA)

{
    std::vector<float> mass;
    std::vector<float> posx, posy, posz;
    std::vector<float> velx, vely, velz;
    std::vector<float> density;
    std::vector<float> temperature;
    std::vector<float> phi;
}
```
</div>
</div>

---

# Testing multiple visualization algorithms in-situ

<div class="grid grid-cols-[50%_50%]">
<div> <!-- #left -->
<img src="/src/images/left.png" style="width: 5vw; min-width: 200px;">
<img src="/src/images/density_minmax.png" style="width: 10vw; min-width: 300px;">
</div>
<div> <!-- #right -->
<img src="/src/images/HistSampling.png" style="width: 15vw; min-width: 350px;">
<img src="/src/images/pdf.png" style="width: 10vw; min-width: 300px;">
</div>
</div>
<br>
---

## Definitions
<br>

- rendering: making an image of a [subset] of the particle cloud
<br>

- thresholding: selecting all particles given a threshold value on a variable. If thresholding against a coordinate component => spatial clipping
- compositing: making a vector-variable (e.g. velocity) from independent components (e.g. vx, vy, vz)
- histsampling: selecting (sub-sampling) a particle cloud, while retaining regions of higher entropy
- binning: doing queries such as min(), max(), etc on a set of variables

---
layout: section
---
# DummySPH: testing three visualization backends

https://github.com/jfavre/DummySPH
<br>

- <div class="flex items-center gap-0">LLNL <span v-mark.highlight.yellow>Ascent</span> <img src="/src/images/ascent-logo.png" class="h-10 ml-1 mr-2"> </div>
```bash
https://ascent.readthedocs.io/en/latest/index.html
```

<br>

- <div class="flex items-center gap-0">Kitware <span v-mark.highlight.yellow>ParaView Catalyst</span> <img src="/src/images/pvcatalyst-logo.png" class="h-6 ml-1 mr-2"> </div>
```bash
https://kitware.github.io/paraview-catalyst/
```

<br>

- <div class="flex items-center gap-0"><span v-mark.highlight.yellow>VTK-m:</span> <img src="/src/images/vtkm-logo.svg" class="h-6 ml-1 mr-2"> Accelerating the Visualization Toolkit for Massively Threaded Architectures</div>
```bash
https://vtk-m.readthedocs.io/en/stable/index.html
```
<br>
---

# [DummySPH](https://github.com/jfavre/DummySPH): How to

First release! v0.1

To-Do (you, the user)
* compile, install, validate your Ascent/VTK-m/Catalyst
* compile DummySPH and choose one backend
```
set(INSITU None CACHE STRING "Enable in-situ support")
set_property(CACHE INSITU PROPERTY STRINGS None Catalyst Ascent VTK-m)
```
Examples:
```
drwxrwxr-x  7 jfavre jfavre    4096 May 26 13:50 buildAscent/
drwxrwxr-x  5 jfavre jfavre    4096 May 26 13:58 buildAscentCuda/
drwxrwxr-x  6 jfavre jfavre    4096 Jun  4 14:45 buildAscentStrided/
drwxrwxr-x  5 jfavre jfavre    4096 Apr 30 13:12 buildAscentStrided-Cuda/
drwxrwxr-x  5 jfavre jfavre    4096 Jun 11 08:57 buildCatalyst2/
drwxrwxr-x  5 jfavre jfavre    4096 Apr 17 11:00 buildCatalyst2Strided/
drwxrwxr-x  4 jfavre jfavre    4096 Jun  4 08:46 buildVTKm/
drwxrwxr-x  4 jfavre jfavre    4096 May 23 13:59 buildVTKmCuda/
drwxrwxr-x  4 jfavre jfavre    4096 Jun  4 11:44 buildVTKmStrided/
drwxrwxr-x  4 jfavre jfavre    4096 Jun  4 12:04 buildVTKmStridedCuda/
```
---

# Execution mode
<br> <br>

- <strong>Application-aware</strong> coupling
- <strong>On Node</strong> memory access
- <strong>Time Division</strong>
- <strong>Automatic and Adaptive</strong> control

<br>
<br>


<em>Childs, H., et al.: A terminology for in situ visualization and analysis systems.
The International Journal of High Performance Computing Applications, 34(6),
676–691. https://doi.org/10.1177/1094342020935991</em>
---

# Describing data (sharing memory pointers)

- [Conduit](https://llnl-conduit.readthedocs.io/en/latest/blueprint.html): A light-weight (mostly zero-copy pointers) description of the in-memory data, that serves as a Common Denominator to both the Ascent & ParaView Catalyst backends

```cpp
#if defined(USE_CATALYST) || defined(USE_ASCENT)
template<typename T>
void addField(ConduitNode& mesh, const std::string& name, T* field, const size_t N)
{
    mesh["fields/" + name + "/association"] = "vertex";
    mesh["fields/" + name + "/topology"]    = "mesh";
    mesh["fields/" + name + "/values"].set_external(field, N);
    mesh["fields/" + name + "/volume_dependent"].set("false");
}

ConduitNode  mesh;
addField(mesh, "rho", sim->rho.data(), sim->n);     // use the raw memory pointer to the individual scalar field
```

- Custom (application specific data pointers) for VTK-m
```cpp
vtkm::cont::DataSetBuilderExplicit dataSetBuilder;
auto aos7 = vtkm::cont::make_ArrayHandle<T>(sim->rho, vtkm::CopyFlag::Off);
dataSet.AddPointField("rho",  aos7);
```
<br>

---

## What works and what doesn't work (SOA) SPH-EXA
<br>

<style scoped>
table {
  font-size: 13px;
}
</style>
| <strong>SPHEXA</strong> | <strong>VTK-m</strong> | <strong>Ascent</strong> | <strong>ParaView Catalyst</strong> |
| -------- | :-------: | :--------: | :-------: |
| in-situ rendering | ✅ yes, but parallel image compositing missing | ✅ yes, with parallel image compositing | idem |
| Geometric clipping | ✅ yes | ✅ yes | ✅ yes, but with a VTK to VTK-m dataset conversion |
| composing vectors | ✅ yes | ✅ yes | ✅ yes, but  "vtkSOADataArrayTemplate: GetVoidPointer called. This is very expensive for non-array-of-structs subclasses" |
| Data binning | ⛔️ n.a. | float64 ✅ OK <br> float32 ❌ not OK | ⛔️ n.a. |
| Histogram sampling | ✅ yes | ✅ yes | ⛔️ n.a. |

---

## What works and what doesn't work (AOS) PKDGRAV3
<br>

<style scoped>
table {
  font-size: 13px;
}
</style>
| <strong>PKDGRAV3</strong> | <strong>VTK-m</strong> | <strong>Ascent</strong> | <strong>ParaView Catalyst</strong> |
| -------- | :-------: | :--------: | :-------: |
| in-situ rendering | ✅ yes | ❌ failing, although Data saving works | stride is not correct|
| Geometric clipping | ✅ yes | ✅ yes | idem as above |
| composing vectors | ✅ yes | ❌ "composite_vector" failing | idem as above |
| Data binning | ⛔️ n.a. | float64 ✅ OK <br> float32 ❌ not OK | ⛔️ n.a. |
| Histogram sampling | ✅ yes | ❌ failing | ⛔️ n.a. |

---

# Device-resident data support (CUDA)

- DummySPH does a cudaMalloc() + cudaMemcpy(..., cudaMemcpyHostToDevice) after reading its initialization data

<p class="text-sm ...">
```cpp
void
device_move(conduit::Node &data, int data_nbytes)
{
  void *device_ptr = device_alloc(data_nbytes);
  copy_from_host_to_device(device_ptr, data.data_ptr(), data_nbytes);
  conduit::DataType dtype = data.dtype();
  data.set_external(dtype,device_ptr);
}

addField(mesh, "rho", sim->rho.data(), sim->n);
device_move(mesh["fields/rho/values"], data_nbytes);
```
</p>

- SPH-EXA initializes its data directly on the GPU
<br>
---

### Device-resident support. Your mileage will vary
<br>

- VTK-m does the best job 🥇

- Ascent does slightly less 

- Catalyst is at the [proof-of-concept level](https://www.kitware.com/catalyst2-gpu-resident-workflows/)

    - A hardware-tuned solution at CSCS has been validated with the CUDA Summer School exercises

---
layout: section
---

# Going further: the special case of NVIDIA Grace-Hopper

---

### Device-resident support: the special case of NVIDIA Grace-Hopper

<div class="flex justify-center">
<img src="/src/images/superchip_logic_overview_1.png" class="h-79 ml-1 mr-2">
</div>

<div class="flex justify-center">
All CPUs and GPUs on the GH200 share a unified address space, and<br>
support transparent fine grained access to all main memory on the system.
</div>

---

### Device-resident support: the special case of NVIDIA Grace-Hopper
<br>

- NVIDIA\'s NVLink-C2C interconnect enables fast, low latency, and <span v-mark.highlight.yellow>cache coherent interaction</span> between different chiplets
- Every Processing Unit (PU) has complete access to all main memory
- Each GH200 is composed of two NUMA nodes
- Memory allocated with malloc(), new() and mmap() <span v-mark.highlight.yellow>can be accessed by all CPUs and GPUs </span> in the compute node
- Memory allocated with <span v-mark.highlight.yellow>cudaMalloc() cannot be accessed from the CPU and by other GPUs</span> on the compute node
- Placement of a memory page is <span v-mark.highlight.yellow>decided by the NUMA node of the thread that first writes
to it</span>, not by the thread that allocates it.
---

### Application: source code changes to run with ParaView Catalyst
<br>
<div class="grid grid-cols-[55%_45%]">
<div> <!-- #left -->
- allocate memory on the host <br>
- first touch on the GPU-side <br>
- no need to copy from GPU to host when we trigger the in-situ visualizations
<br>
<br>
- we instrumented a CUDA simulator for the 2D heat diffusion equation
</div>

<div> <!-- #right -->

<img src="/src/images/catalyst_temperature040000.png" style="width: 35vw; min-width: 300px;">
<br>
</div>
</div>
---

### Application: source code changes to run with ParaView Catalyst
<br>

- Successfully transformed a CUDA-enabled mini-app to use ParaView Catalyst
```cpp

-    double *x_host = malloc_host<double>(buffer_size);
-    double *x0     = malloc_device<double>(buffer_size);
-    double *x1     = malloc_device<double>(buffer_size);
+    //double *x_host = malloc_host<double>(buffer_size);
+    double *x0     = malloc_host<double>(buffer_size);
+    double *x1     = malloc_host<double>(buffer_size);
```
```cpp
#ifdef USE_CATALYST
     // must copy data to host since we cannot use a CUDA-enabled Catalyst at this time
-    copy_to_host<double>(x1, x_host, buffer_size); // use x1 with most recent result
+    //copy_to_host<double>(x1, x_host, buffer_size); // use x1 with most recent result
     CatalystAdaptor::Execute(step, dt);
 #endif
```
```cpp
 template <typename T>
 T* malloc_host(size_t N, T value=T()) {
     T* ptr = (T*)(malloc(N*sizeof(T)));
-    std::fill(ptr, ptr+N, value);
+    //std::fill(ptr, ptr+N, value);
     return ptr;
 }

```
---
layout: section
---

# Special topic: Derived quantities

---

## Example

Calculate a Probability Density Function, e.g. evaluate radius = sqrt(x^2 + y^2 + z^2)

- Ascent uses OCCA
- ParaView uses numpy
<div class="flex justify-center">
<img src="/src/images/pdf.png" style="width: 23vw; min-width: 250px;">
<br>
</div>

---

###

<div class="grid grid-cols-[50%_50%] gap-1">
<div> <!-- #left -->

### OpenMP code generation
<br>

<Transform :scale="0.9">
```cpp
#include <occa.hpp>

using namespace std;
using namespace occa;

extern "C" void map(double * output,
                    const double * z,
                    const double * x,
                    const double * y,
                    const int & entries) {
#pragma omp parallel for
  for (int group = 0; group < entries; group += 128) {
    for (int item = group; item < (group + 128); ++item) {
      if (item < entries) {
        output[item] = sqrt(
          (((x[item] * x[item]) + \
            (y[item] * y[item])) + \
            (z[item] * z[item]))
        );
      }
    }
  }
}
```
</Transform>
</div>

<div> <!-- #right -->

### CUDA code generation
<br>

<Transform :scale="0.9">
```cpp
extern "C" __global__ void _occa_map_0(double * output,
                                         const double * z,
                                         const double * x,
                                         const double * y,
                                         const int entries) {
  {
    int group = 0 + (128 * blockIdx.x);
    {
      int item = group + threadIdx.x;
      if (item < entries) {
        output[item] = sqrt(
          (((x[item] * x[item])  + \
            (y[item] * y[item]))  + \
            (z[item] * z[item]))
        );
      }
    }
  }
}

```

</Transform>
</div>
</div>

---

###

<div class="grid grid-cols-[50%_50%] gap-1">
<div> <!-- #left -->

### Ascent query/action

<Transform :scale="0.9">
```cpp
conduit::Node queries;
queries["q1/params/expression"] = "field('kx') * \
                                   field('m') / \
                                   field('xm')";
queries["q1/params/name"] = "density";

queries["q2/params/expression"] = \
  "sqrt(field('x')*field('x') + field('y')*field('y') + \
                                field('z')*field('z'))";
queries["q2/params/name"] = "radius";

queries["q3/params/expression"] = \
"binning('', 'pdf', [axis('radius'), axis('density'])";
queries["q3/params/name"] = "pdf1";
```
</Transform>
</div>

<div> <!-- #right -->

### ParaView with numpy expressions
<Transform :scale="0.9">
```py
pythonCalculator1 = PythonCalculator(Input=grid)
pythonCalculator1.Set(
    Expression='sqrt(x*x + y*y + z*z)',
    ArrayName='radius',
)
```
</Transform>
</div>
</div>

---

# Production runs with SPH-EXA
<br>

<div class="flex justify-left">
  <img src="/src/images/density301.01000.png" class="h-42 ml-1">
  <img src="/src/images/density301.02500.png" class="h-42 ml-1">
  <img src="/src/images/density301.03500.png" class="h-42 ml-1">
  <!-- <img src="/src/images/density301.01000.png" style="width: 10vw; min-width: 250px;"> -->
  <!-- <img src="/src/images/density301.02500.png" style="width: 10vw; min-width: 250px;"> -->
  <!-- <img src="/src/images/density301.03500.png" style="width: 10vw; min-width: 250px;"> -->
</div>
<br>

- Testing at scale with DummySPH and SPH-EXA ( with CUDA )
- Testing two classes of GPU (A100 and H100)
- Memory consumption checked with NVIDIA\'s nsys

<!--
The wind-cloud collision test shows the interactions between a supersonic wind
and a dense, cold spherical cloud. 

It is a challenging test for SPH, involving strong shocks and mixing due to the
Kelvin-Helmholtz instability in a two-phase medium with a large density
contrast.

The figures show the time evolution of the particle distribution of the density
in a thin slice. Instabilities are able to develop, mix and eventually destroy
the cloud.
-->


---

## SPH-EXA

<div class="flex justify-left">
  <img src="/src/images/sphexa_baseline_hdf5_ascent.png" class="h" border="1px">
</div>

<div class="flex justify-center">
  <img src="/src/images/sphexa_cuda_memcpy_ascent.png" class="h-65" border="1px">
</div>

<!--
We simulated this test with a total number of 55 billion global particles, 
arranged in four blocks of 2400^3 particles each, with one block containing a
cavity for the high-density cloud.

<br>
<br>
<br>

- https://www.aanda.org/articles/aa/full_html/2022/03/aa41877-21/aa41877-21.html
-->

---
layout: two-cols-header
---

## SPHEXA without/with Ascent: GPU memory usage
<br>

`thresholding`: 48 cn, 10 iterations, 1 H100/95GB <del>120GB</del>, 55 billion particles, 87% of peak memory (95GB)
<!-- 83 GB / 95 GB = 87% -->
<!-- <small>$20.10^6$ particles, 20 iterations, insitu every 5 iteration</small> -->

::left::

<div class="flex justify-left">
  <img src="/src/images/sphexa_nsys_gpu_memory+ascent.png" class="h" border="1px">
</div>

::right::

<div class="flex justify-left">
  <img src="/src/images/sphexa_nsys_gpu_memory-ascent.png" class="h ml-1" border="1px">
</div>

<br>
<br>

---
layout: two-cols-header
---

## DummySPH with Ascent: GPU memory usage
<br>

`thresholding`: single GPU (A100/80GB, H100/120GB), 
<small>$20.10^6$ particles, 20 iterations, insitu every 5 iteration</small>

<!-- 
- OFF ON OFF ON
- A100-SXM4-80GB, GH200/120GB=95GB
- NsightSystems/2025.3.1.90
infile1 = 'n052+ascent/3.csv' "1/8 million particles (max=%.2g bytes)" , STATS_max)
infile2 = 'n066+ascent/3.csv' "1/4 million particles (max=%.2g bytes)" , STATS_max)
infile3 = 'n083+ascent/3.csv' "1/2 million particles (max=%.2g bytes)" , STATS_max)
infile4 = 'n105+ascent/3.csv' "1 million particles (max=%.2g bytes)" , STATS_max)
infile5 = 'n132+ascent/3.csv' "2 million particles (max=%.2g bytes)" , STATS_max)
infile6 = 'n167+ascent/3.csv' "4 million particles (max=%.2g bytes)" , STATS_max)
infile7 = 'n210+ascent/3.csv' "10 million particles (max=%.2g bytes)" , STATS_max)
infile8 = 'n265+ascent/3.csv' "20 million particles (max=%.2g bytes)" , STATS_max)

./sqlite3 nsys_log.sqlite "SELECT * FROM ENUM_CUDA_MEM_KIND" # <-- memKind
0|CUDA_MEMOPR_MEMORY_KIND_PAGEABLE|Pageable
1|CUDA_MEMOPR_MEMORY_KIND_PINNED|Pinned
2|CUDA_MEMOPR_MEMORY_KIND_DEVICE|Device
3|CUDA_MEMOPR_MEMORY_KIND_ARRAY|Array
4|CUDA_MEMOPR_MEMORY_KIND_MANAGED|Managed
5|CUDA_MEMOPR_MEMORY_KIND_DEVICE_STATIC|Device Static
6|CUDA_MEMOPR_MEMORY_KIND_MANAGED_STATIC|Managed Static
7|CUDA_MEMOPR_MEMORY_KIND_UNKNOWN|Unknown
-->

::left::

<!-- - <small><span v-mark.underline.yellow>Stop on CUDA kernel launch</span>:<br></small> -->

<div class="flex justify-left">
  <img src="/src/images/dummysph_nsys_gpu_memory_A100.svg" class="h" border="1px">
</div>

::right::

<div class="flex justify-left">
  <img src="/src/images/dummysph_nsys_gpu_memory_H100.svg" class="h ml-1" border="1px">
</div>

<br>
<br>
---

# Summary

* SPH-EXA runs in production mode with CUDA-enabled ascent (GB submission coming up)
* WIP (Ascent open issue(s))
* WIP (~~VTK-m~~ Viskores open issue(s))
* WIP (Catalyst open issue(s))

---

# Future work

* Viskores has replaced VTK-m
* Instrumenting another SPH code: DualPhysics
* Take advantage of Grace (CPU) and Hopper (GPU) close proximity for concurrent processing
* Test ROCm

---

# Questions?

* Can I try DummySPH?
* others?
