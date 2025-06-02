---
layout: section
---

# Smoothed Particle Hydrodynamics (SPH)


- Based on a purely Lagrangian technique,
SPH particles are free to move with fluids or deforming solid structures.

- A stimulating field of research to simulate free-surface flows, solid
mechanics, multi-phase, fluid-structure interaction and astrophysics.
--- 

## To be AOS or not to be (SOA)

<br>
<div class="grid grid-cols-[45%_55%] gap-2">
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

- rendering: making an image of a [subset] of the particle cloud
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
https://docs.paraview.org/en/latest/Catalyst/index.html
```

<br>

- <div class="flex items-center gap-0"><span v-mark.highlight.yellow>VTK-m:</span> <img src="/src/images/vtkm-logo.svg" class="h-6 ml-1 mr-2"> Accelerating the Visualization Toolkit for Massively Threaded Architectures</div>
```bash
https://vtk-m.readthedocs.io/en/stable/index.html
```
---

# Execution mode

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

- Conduit: A light-weight (mostly zero-copy pointers) description of the in-memory data, that serves as a Common Denominator to both the Ascent & ParaView Catalyst backends

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
<style scoped>
table {
  font-size: 13px;
}
</style>
| <strong>PKDGRAV3</strong> | <strong>VTK-m</strong> | <strong>Ascent</strong> | <strong>ParaView Catalyst</strong> |
| -------- | :-------: | :--------: | :-------: |
| in-situ rendering | yes, but parallel image compositing missing | yes, with parallel image compositing | idem|
| Geometric clipping | yes |yes | yes, but with a VTK to VTK-m dataset conversion |
| composing vectors | yes |yes | doable but  "very expensive" |
| Data binning | n.a. | float64 OK <br> float32 not OK | n.a. |
| Histogram sampling | yes | yes | n.a. |

---

## What works and what doesn't work (AOS) PKDGRAV3

<style scoped>
table {
  font-size: 13px;
}
</style>
| <strong>PKDGRAV3</strong> | <strong>VTK-m</strong> | <strong>Ascent</strong> | <strong>ParaView Catalyst</strong> |
| -------- | :-------: | :--------: | :-------: |
| in-situ rendering | yes | failing, although Data saving works | stride is not correct|
| Geometric clipping | yes | yes | idem as above |
| composing vectors | yes |"composite_vector" failing | idem as above |
| Data binning | n.a. | float64 OK <br> float32 not OK | n.a. |
| Histogram sampling | yes | failing | n.a. |

---

# Device-resident data support (CUDA)

- DummySPH does a cudaMalloc() and cudaMemcpy(..., cudaMemcpyHostToDevice) after reading its initialization data

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

- VTK-m does the best job
- Ascent does slightly less 
- Catalyst is at the proof-of-concept level, but there is one specific solution at CSCS...

---
layout: section
---

## Going further beyond the topics presented in the paper
---
---
### Device-resident support: the special case of NVIDIA Grace-Hopper

<div class="flex items-center gap-0">
<img src="/src/images/superchip_logic_overview_1.png" class="h-79 ml-1 mr-2">
</div>
All CPUs and GPUs on the GH200 share a unified address space and support transparent fine grained access to all main memory on the system.
---

### Device-resident support: the special case of NVIDIA Grace-Hopper

- NVIDIA's NVLink-C2C interconnect enables fast, low latency, and <span v-mark.highlight.yellow>cache coherent interaction</span> between different chiplets
- Every Processing Unit (PU) has complete access to all main memory
- Each GH200 is composed of two NUMA nodes
- Memory allocated with malloc(), new() and mmap() <span v-mark.highlight.yellow>can be accessed by all CPUs and GPUs </span> in the compute node
- Memory allocated with <span v-mark.highlight.yellow>cudaMalloc() cannot be accessed from the CPU and by other GPUs</span> on the compute node
- Placement of a memory page is <span v-mark.highlight.yellow>decided by the NUMA node of the thread that first writes
to it</span>, not by the thread that allocates it.
---

### Application: source code changes to run with ParaView Catalyst
- allocate memory on the host
- first touch on the GPU-side
- no need to copy from GPU to host when we trigger the in-situ visualizations
---

### Application: source code changes to run with ParaView Catalyst

- We successfully transformed a CUDA-enabled mini-app to be able to use ParaView Catalyst

```cpp

-    double *x_host = malloc_host<double>(buffer_size);
-    double *x0     = malloc_device<double>(buffer_size);
-    double *x1     = malloc_device<double>(buffer_size);
+    //double *x_host = malloc_host<double>(buffer_size);
+    double *x0     = malloc_host<double>(buffer_size);
+    double *x1     = malloc_host<double>(buffer_size);
```
###################################################
```cpp
#ifdef USE_CATALYST
           // must copy data to host since we're not using a CUDA-enabled Catalyst at this time
-          copy_to_host<double>(x1, x_host, buffer_size); // use x1 with most recent result
+          //copy_to_host<double>(x1, x_host, buffer_size); // use x1 with most recent result
           CatalystAdaptor::Execute(step, dt);
 #endif
```
###################################################
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
---
## Derived quantities
Example: Calculate a Probability Density Function, e.g. evaluate radius = sqrt(x^2 + y^2 + z^2)

- Ascent uses OCCA
- ParaView uses numpy
<img src="/src/images/pdf.png" style="width: 25vw; min-width: 300px;">
<br>

---

###

<div class="grid grid-cols-[50%_50%] gap-1">
<div> <!-- #left -->

### OpenMP code generation

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

## Questions?

<!-- {{{ sphexa: baseline, hdf5, ascent -->

---

## SPHEXA

<div class="flex justify-left">
  <img src="/src/images/sphexa_baseline_hdf5_ascent.png" class="h" border="1px">
</div>

<div class="flex justify-center">
  <img src="/src/images/sphexa_cuda_memcpy_ascent.png" class="h-65" border="1px">
</div>

<!-- }}} -->
<!-- {{{ nsys_gpu_memory sphexa -->

---
layout: two-cols-header
---

## SPHEXA without/with Ascent: GPU memory usage
<br>

`thresholding`: 48 cn, 10 iterations, -n2400 (H100/120GB), 55 billion particles, 87% of peak memory (95GB)
<!--
83 GB / 95 GB = 87%
-->
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

<!-- }}} -->
<!-- {{{ nsys_gpu_memory dummysph -->

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


<!-- }}} -->
