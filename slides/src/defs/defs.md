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
layout: two-cols-header
columns: is-two-quarters
---
<! need padding between the 2 block ->
# To be AOS or not to be (SOA)
::left::
```
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
::right::
```
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
---

# DummySPH: a mini-app with three in-situ visualization backends
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

# Passing data (sharing memory pointers)

- Conduit: A light-weight (mostly zero-copy pointers) description of the in-memory data, that serves as a Common Denominator to both the Ascent & ParaView Catalyst backends

```
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
```
vtkm::cont::DataSetBuilderExplicit dataSetBuilder;
auto aos7 = vtkm::cont::make_ArrayHandle<T>(sim->rho, vtkm::CopyFlag::Off);
dataSet.AddPointField("rho",  aos7);
```
<br>

---

# Device-resident data support (CUDA)

- DummySPH does a cudaMalloc() and cudaMemcpy(..., cudaMemcpyHostToDevice) after reading its initialization data

<p class="text-sm ...">
```ts {2,10,12}
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
- Catalyst is at the proof-of-concept level
---
layout: image-left

image: ./src/images/superchip_logic_overview_1.png
---
<! need to shrink image ->
### Device-resident support: The special case of NVIDIA Grace-Hopper

---
layout: section
---

# Some new material
Derived quantities, e.g. evaluate radius = sqrt(x^2 + y^2 + z2)
- Ascent uses OCCA
- ParaView uses numpy
<br>
---
layout: two-cols-header
---
::left::
# OpenMP code generation

```
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
          (((x[item] * x[item]) + (y[item] * y[item])) + (z[item] * z[item]))
        );
      }
    }
  }
}
```

::right::

# CUDA code generation
```
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
---

## Questions?
---

