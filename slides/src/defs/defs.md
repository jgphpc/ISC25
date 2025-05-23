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

### Passing data (sharing memory pointers)

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

### Device-resident support (CUDA)

- DummySPH does a cudaMalloc() and cudaMemcpy(..., cudaMemcpyHostToDevice) after reading its initialization data
```
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

- SPH-EXA initializes its data directly on the GPU
<br>
---

### Device-resident support. Your mileage will vary
- VTK-m does the best job
- Ascent does slightly less 
- Catalyst is at the proof-of-concept level
---

### Device-resident support: The special case of NVIDIA Grace-Hopper
---


## Questions?

