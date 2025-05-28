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

<!-- {{{ AOS vs SOA -->

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

<!-- }}} -->

---

### DummySPH: a mini-app with three in-situ visualization backends

https://github.com/jfavre/DummySPH
<br>

- <div class="flex items-center gap-0">LLNL <span v-mark.highlight.yellow>Ascent</span> <img src="/src/images/ascent-logo.png" class="h-10 ml-1 mr-2"> </div>
```bash
https://github.com/Alpine-DAV/ascent
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
https://gitlab.kitware.com/vtk/vtk-m
https://vtk-m.readthedocs.io/en/stable/index.html
```
---

# Passing data (sharing memory pointers)

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
addField(mesh, "rho", sim->rho.data(), sim->n);
```
- Custom (application specific data pointers) for VTK-m
```cpp
vtkm::cont::DataSetBuilderExplicit dataSetBuilder;
auto aos7 = vtkm::cont::make_ArrayHandle<T>(sim->rho, vtkm::CopyFlag::Off);
dataSet.AddPointField("rho",  aos7);
```
<br>

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

### Device-resident support: the special case of NVIDIA Grace-Hopper

<div class="flex items-center gap-0">
<img src="/src/images/superchip_logic_overview_1.png" class="h-79 ml-1 mr-2">
</div>
All CPUs and GPUs on the GH200 share a unified address space and support transparent fine grained access to all main memory on the system.
---

### Device-resident support: the special case of NVIDIA Grace-Hopper

- NVIDIA's NVLink-C2C interconnect enables fast, low latency, and cache coherent interaction between different chiplets
- Every Processing Unit (PU) has complete access to all main memory
- Each GH200 is composed of two NUMA nodes

- Memory allocated with host APIs like malloc(), new() and mmap() can be accessed by
all CPUs and GPUs in the compute node
- Memory allocated with cudaMalloc() cannot be accessed from the CPU and by other GPUs on the compute node

- Placement of a memory page is decided by the NUMA node of the thread that first writes
to it, not by the thread that allocates it.
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
layout: section
---

# Some new material
Derived quantities, e.g. evaluate radius = sqrt(x^2 + y^2 + z2)

- Ascent uses OCCA
- ParaView uses numpy
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
                                         const float * m,
                                         const float * kx,
                                         const float * xm,
                                         const int entries) {
  {
    int group = 0 + (128 * blockIdx.x);
    {
      int item = group + threadIdx.x;
      if (item < entries) {
        output[item] = ((kx[item] * m[item]) / xm[item]);
      }
    }
  }
}
```
</Transform>
</div>
</div>


---

## Questions?
---

