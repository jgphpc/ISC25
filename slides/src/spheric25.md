<!-- {{{ Why ? -->
# Back-of-the-envelope SPH

<br>

<Transform :scale="1.3">
1 checkpoint file: 1'000'000'000'000 particles (2000 GPUs, half billion particles per GPU)<br><br>
</Transform>

<div class="flex justify-center">
<Transform :scale="1.3">

$\times$ 76 bytes per particle<br>
~ 76 Petabytes 💾 (85% of CSCS filesystem)<br><br>
$\times$ Average write speed at 200 GB/s<br>
~ 4.5 days per checkpoint file ⏳️<br><br>
$\times$ Long term storage at 60 EUR / TB / year<br>
~ 4.5 million 💶 (per year)<br><br>
In-situ Visualization to the rescue 🛟<br>

</Transform>
</div>

<!-- 
El Capitan = \#1 = 43,808 AMD Instinct MI300A GPUs
Frontier = \#2 =   37,632 AMD Instinct MI250X GPUs
Aurora = \#3 =     63,744 Intel Max Series GPUs
Jupiter B = \#4 = ~24,000 Hopper H100 GPU
etc...

- gain insights into their simulations as early as possible
- faster with flash drives (ssd), amr, compression, AI, etc...
- https://io500.org/list/sc24/ten-production -> average=94, \#2=200 GiB/s, \#1=734 GiB/s
- existing insitu solutions are sometimes hard to adapt and reuse
https://sli.dev/features/latex
https://sli.dev/demo/starter/11

<v-click>* 76 bytes per particle</v-click><br>
<v-click>~ 76 Petabytes 💾 (85% of CSCS filesystem)</v-click><br><br>
<v-click>* Average write speed at 200 GB/s</v-click><br>
<v-click>~ 4.5 days per checkpoint file ⏳️</v-click><br><br>
<v-click>* Long term storage at 60 💶 / TB / year</v-click><br>
<v-click>~ 4.5 million 💰️ per checkpoint file (per year)</v-click><br><br>
<v-click>In Situ Visualization to the rescue 🛟</v-click><br>
-->

<!-- }}} -->

<!-- {{{ Backends -->
---

# In Situ Visualization

<div> <Transform :scale=".9">

- `Post Hoc`: Simulation data is processed after the simulation (using distinct resources)
- `In Situ`: Simulation data is processed while it is generated (sharing resources, in memory, zero-copy)

### Use cases: Render Pictures, Transform Data, and Capture Data

- `rendering`: make an image of a subset of a cloud of particles
- `thresholding`: exclude particles by applying a threshold value to a specified variable 
- `compositing`: construct a vector field (e.g velocity magnitude) from independent components (vx, vy, vz)
- `histogram sampling`: selectively subsample particles while retaining regions of higher entropy
- `binning`: group or classify sets of particles into a smaller number of bins

### Backends

<div class="flex items-center gap-0"><img src="/src/images/vtk-logo.png" class="h-10 ml-5 mr-1">: Kitware Visualization Toolkit and
<img src="/src/images/vtkm-logo.svg" class="h-6 ml-1 mr-1">: VTK for Massively Threaded Architectures</div>

<div class="flex items-center gap-0"><img src="/src/images/viskores-logo-white.png" class="h-6 ml-5 mr-1">Viskores: Successor to VTK-m being discontinued (Kitware, ORNL, LANL, Sandia, UT-Batelle)</div>

<div class="flex items-center gap-0"><img src="/src/images/ascent-logo.png" class="h-10 ml-5 mr-1">: Easier-to-use Flyweight In-situ library for HPC simulations (NNSA/LLNL)</div>

<div class="flex items-center gap-0"><img src="/src/images/conduit-logo.png" class="h-6 ml-5 mr-1">: Simplified Data Exchange library for HPC Simulations (LLNL)</div>

<div class="flex items-center gap-0"><img src="/src/images/pvcatalyst-logo.png" class="h-6 ml-5 mr-1">: ParaView implementation of the Catalyst API (Kitware, Sandia, LANL)</div>

</Transform></div>

<!-- 
- render: facilitating clearer insights into specific data regions
- thresholding: focus on the most relevant data points
- histsampl: areas with greater information content are retained, enhancing the overall analytical depth
- binning: simplify data helps in the identification of patterns and trends
such as
- Conduit is the underlying interface for passing information to Ascent
    - Blueprint is the API used to pass simulation data (in-memory) to Ascent
    - Actions is the API used to instruct Ascent how to carry out actions
Actions API consists of five calls:
• open initializes Ascent. It can optionally take arguments, for passing along infor-
mation such as the MPI communicator.
• publish is the method that enables a simulation code to pass (“publish”) its data
to Ascent.
• execute specifies which Ascent actions (see Sect. 2) to perform.
• info is the mechanism for getting data out from Ascent into the simulation.
• close directs Ascent to finalize execution.
Ascent usage typically consists of only four calls: open, publish, execute, and
close. These calls
-->

<!-- }}} -->

<!-- {{{ SPH-EXA test -->
---

# Wind-Cloud collision test

This test$^{[1]}$ simulates a spherical cloud of cold gas,
initially at rest, swept by a low-density stream of gas (wind) moving supersonically.

<div class="flex justify-left">
  <img src="/src/images/SPH-EXA_logo.png" class="h-6 ml-5 mr-1">
  <img src="/src/images/density301.01000.png" class="h-55 ml-1">
  <img src="/src/images/density301.03500.png" class="h-55 ml-1">
</div>

<small>
```
             The figures show the time evolution of density in a thin slice of the domain,
             Kelvin–Helmholtz instabilities are able to develop, mix and eventually destroy the cloud.
             This simulation was run with the SPH-EXA code on CSCS Alps system.
```
</small>

<div class="absolute bottom-0 left-0 p-12 w-full text-sm text-gray-500">
  <small>[1] García-Senz D., Cabezón R. and Jose A. Escartín J. A.,
  Conservative, density-based smoothed particle hydrodynamics with improved
  partition of the unity and better estimation of gradients, in Astronomy &
  Astrophysics, 10.1051/0004-6361/202141877</small>
</div>

<!-- Large scale production test is an astrophysical problem that has been
extensively studied in recent years -->

<!-- }}} -->
<!-- {{{ MESH -->

---

## How to pass SPH simulation mesh data to Ascent ?

<div class="flex justify-right">
<Transform :scale=".8">
```cpp
ascent::Ascent a;

void Execute(DataType& d, long startIndex, long endIndex) {
  conduit::Node mesh;
  mesh["coordsets/coords/type"] = "explicit";
  mesh["coordsets/coords/values/x"].set_external(get<"x">(d).data() + startIndex, endIndex - startIndex);
  mesh["coordsets/coords/values/y"].set_external(get<"y">(d).data() + startIndex, endIndex - startIndex);
  mesh["coordsets/coords/values/z"].set_external(get<"z">(d).data() + startIndex, endIndex - startIndex);

  mesh["topologies/mesh/type"] = "unstructured";
  std::vector<conduit_int32> conn(endIndex - startIndex); // CONNECTIVITY_LIST
  std::iota(conn.begin(), conn.end(), 0);
  mesh["topologies/mesh/elements/connectivity"].set(conn);
  mesh["topologies/mesh/elements/shape"] = "point";
  mesh["topologies/mesh/coordset"] = "coords";

  addField(mesh, "x", get<"x">(d).data(), startIndex, endIndex);
  addField(mesh, "y", get<"y">(d).data(), startIndex, endIndex);
  addField(mesh, "z", get<"z">(d).data(), startIndex, endIndex);
  addField(mesh, "Density", get<"rho">(d).data(), startIndex, endIndex);

  a.publish(mesh);
  
void addField(conduit::Node& mesh, const std::string& name, FieldType* field, size_t start, size_t end)
{
    mesh["fields/" + name + "/association"] = "vertex";
    mesh["fields/" + name + "/topology"]    = "mesh";
    mesh["fields/" + name + "/values"].set_external(field + start, end - start);
    mesh["fields/" + name + "/volume_dependent"].set("false");
}
```
</Transform>
</div>

<!--
Conduit Mesh Blueprint provides a strategy to describe and adapt mesh data between a wide range of APIs
Ascent uses Conduit as a shared interface to describe and accept simulation mesh data
Ascent accepts Conduit Mesh Blueprint data
-->

<!-- }}} -->
<!-- {{{ ACTIONS -->

---

## How to pass SPH Actions to Ascent ?

<div class="flex justify-left">
<Transform :scale=".75">
```yaml
-
  action: "add_triggers"
  triggers:
    t1:
      params:
        condition: "cycle() % 5 == 0"
        actions:
          -
            action: "add_pipelines"
            pipelines:
              pl_threshold_thin_clip_z:     pl_threshold_thin_clip_y:   renders:
                f1:                           f1:                         r1:
                  type: "threshold"             type: "threshold"           image_prefix: "datasets/Temperature.%05d"
                  params:                       params:                     image_width: 1920
                    field: "z"                    field: "y"                image_height: 1080
                    min_value: 0.12425            min_value: 0.12425        camera:
                    max_value: 0.12575            max_value: 0.12575          look_at: [0.5, 0.125, 0.125]
                                                                              position: [0.5, 0.125, 3.0]
          -                                                                   up: [0.0, 1.0, 0.0]
            action: "add_scenes"                                              azimuth: -35.0
            scenes:                                                           elevation: 25.0
              s1:                                                             zoom: 5.25
                plots:                                                                      
                  p1:                                       p2:                             
                    type: "pseudocolor"                       type: "pseudocolor"           
                    field: "Temperature"                      field: "Temperature"          
                    pipeline: "pl_threshold_thin_clip_z"      pipeline: "pl_threshold_thin_clip_y"
                    min_value: 1                              min_value: 1
                    max_value: 10                             max_value: 10
                    color_table:                              color_table:
                      name: "Yellow - Gray - Blue"              name: "Yellow - Gray - Blue"
                      annotation: "false"                       annotation: "true"
                    points:                                   points:
                      radius: 0.002                             radius: 0.002                         
```
</Transform>
</div>

<!--
 
-->

<!-- }}} -->

<!-- {{{ DUALSPH test -->
---

# 3-D dam break test

This test simulates a 3-D dam break flow impacting on a structure

<div class="flex justify-center">
  <img src="/src/images/dualsph-logo.png" class="h-10 ml-5 mr-1">
  <img src="/src/images/dualsphysics_dambreak.png" class="h-90 ml-1">
</div>

<!-- }}} -->

<!-- {{{ At scale -->

---

## Large scale In Situ Visualisation (SPH-EXA)

<div class="flex justify-left">
  <img src="/src/images/sphexa_baseline_hdf5_ascent.png" class="h-50" border="1px">
</div>

```
    thresholding: 192 H100 GPUs, $55.10^9$ particles, insitu every iteration, 87% of GPU peak memory (95GB)
```

<!--
We simulated this test with a total number of 55 billion global particles,
arranged in four blocks of 2400^3 particles each, with one block containing a
cavity for the high-density cloud.
- https://www.aanda.org/articles/aa/full_html/2022/03/aa41877-21/aa41877-21.html
-->

- Ascent GPU memory usage

<!-- 83 GB / 95 GB = 87% -->

<div class="flex justify-center">
  <img src="/src/images/sphexa_nsys_gpu_memory-ascent.png" class="h-50 ml-1" border="1px">
</div>

<!-- }}} -->

<!-- {{{ Summary of tests -->
---

### `DummySPH`: a mini-app to test In Situ Visualization libraries for SPH

<div class="flex justify-center">
<img src="/src/images/dummysph_summary.png" class="h-99 ml-1 mr-1">
</div>

<!--
SOA: Struct of Arrays (SPH-EXA),
AOS: Array of Structs (DUALSPHYSICS, PKDGRAV3)
-->

<div class="absolute bottom-0 right-0 p-8 w-full text-sm text-gray-500">

  [https://github.com/jfavre/DummySPH.git](https://github.com/jfavre/DummySPH.git)
</div>

<!-- }}} -->

<!-- {{{ Conclusion -->

---

<br> <br> <br>

## Conclusion

- Open issues remain to be fixed but production runs are possible

<br>

## Next steps

* Viskores instead of VTK-m
* Contine tests with DualPhysics
* ROCm support

<!-- }}} -->
