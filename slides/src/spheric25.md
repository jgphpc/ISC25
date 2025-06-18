<!-- {{{ Why ? -->
# Back-of-the-envelope SPH

<!-- 
55e9 * 18x = 10^12

El Capitan = \#1 = 43'808 AMD Instinct MI300A GPUs
Frontier = \#2 =   37'632 AMD Instinct MI250X GPUs
Aurora = \#3 =     63'744 Intel Max Series GPUs
150 000 GPUs
Jupiter B = \#4 = ~24'000 Hopper H100 GPUs
LUMI-G = 2978 * 4 = 11\'912 MI250x GPUs
LUMI-G hardware partition consists of 2978 nodes with 4 AMD MI250x GPUs
CSCS = 10752
24000+11912+10752 = 46 664
etc...
-->

<Transform :scale="1.2">

🇺🇸LLNL+ORNL+ANL = 150 000 GPUs, 🇪🇺LUMI+JSC+CSCS = 50 000 GPUs<br>
$10000^3$ particles simulation is within reach<br>
(SPH-EXA, 2 000 GPUs, 1/2 billion particles per GPU)<br>
Saving 50 checkpoint files per simulation:

</Transform>

<div class="flex justify-center">
<Transform :scale="1.3">

$10^{12}$ particles $\times$ 50 $\times$ 76 bytes per particle<br>
~ 3.5 PB 💾 (<5% of CSCS 91 PB scratch filesystem)<br>

$\times$ Average write speed at 200 GB/s<br>
~ almost 5 hours writing 50 output files ⏳️ ("wasting" 10 000 GPU hours)<br>

$\times$ Long term storage at 60 EUR / TB / year<br>
~ 210 000 💶 (per year)<br>
$\rightarrow$ In-situ Visualization to the rescue 🛟

</Transform>
</div>


<!-- 
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

<div class="flex items-center gap-0"><img src="/src/images/viskores-logo-white.png" class="h-6 ml-20 mr-1">Viskores: Successor to VTK-m being discontinued (Kitware, ORNL, LANL, Sandia, UT-Batelle)</div>

<div class="flex items-center gap-0"><img src="/src/images/ascent-logo.png" class="h-10 ml-5 mr-1">: Easier-to-use Flyweight In-situ library for HPC simulations (NNSA/LLNL)</div>

<div class="flex items-center gap-0"><img src="/src/images/pvcatalyst-logo.png" class="h-6 ml-5 mr-1">: ParaView implementation of the Catalyst API (Kitware, Sandia, LANL)</div>

<div class="flex items-center gap-0"><img src="/src/images/conduit-logo.png" class="h-6 ml-30 mr-1">: Simplified Data Exchange library for HPC Simulations (LLNL)</div>

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

<!-- {{{ DUALSPH test -->

---

# 3-D dam break test

This test simulates a 3-D dam break flow impacting on a structure (dp=0.0045, $10^6$ particles)

<!-- 
YES! ffmpeg -pattern_type glob -i '*.png' -vcodec libx264 -s 640x360 -pix_fmt yuv420p -y eff.mp4
NO! ffmpeg -r 25 -i density-000%03d.png -vb 20M eff.mpg
-->

<div class="flex justify-center">
  <img src="/src/images/dualsph-logo.png" class="h-20 ml-5 mr-1">
<video controls>
  <source src="/src/videos/dualsph-density.mp4" type="video/mp4">
</video>
</div>

---

# 3-D dam break test (failover)

This test simulates a 3-D dam break flow impacting on a structure (dp=0.0045, $10^6$ particles)

<div class="flex justify-center">
  <img src="/src/images/dualsph-logo.png" class="h-20 ml-5 mr-1">
  <img src="/src/images/dualsphysics_dambreak.png" class="h-90 ml-1">
</div>


<!-- }}} -->
<!-- {{{ MESH -->

---

## DualSPHysics: How to pass simulation mesh data to Ascent ?

<div class="flex justify-right">
<!-- <Transform :scale=".75"> -->

````md magic-move {lines: true}

```cpp {1-9|10-17|18-21|*}
        ascent::Ascent myascent; conduit::Node mymesh;
        mymesh["coordsets/coords/type"] = "explicit";
        mymesh["topologies/mesh/coordset"] = "coords";
        // CONNECTIVITY_LIST
        mymesh["topologies/mesh/type"] = "unstructured";
        std::vector<conduit_int32> conn(array2_count);
        std::iota(conn.begin(), conn.end(), 0);
        mymesh["topologies/mesh/elements/connectivity"].set(conn);
        mymesh["topologies/mesh/elements/shape"] = "point";
        // Pos3_vec
        const tfloat3* pos3_ptr = reinterpret_cast<const tfloat3*>(arrays2.Arrays[0].ptr);
        std::vector<tfloat3> pos3_vec(pos3_ptr, pos3_ptr + array2_count);
        // Pos3_vec.x
        std::vector<float> pos3_vec_x(pos3_vec.size());
        std::transform(pos3_vec.begin(), pos3_vec.end(), pos3_vec_x.begin(),
                       [](const tfloat3& pos) { return pos.x; });
        // [...] replicate code for Pos3_vec.y, Pos3_vec.z and rho (arrays2.Arrays[3].ptr)
        // coordinates                                          // density (do the same for x,y,z )
        mymesh["coordsets/coords/values/x"].set(pos3_vec_x);    mymesh["fields/rhop/association"] = "vertex";       
        mymesh["coordsets/coords/values/y"].set(pos3_vec_y);    mymesh["fields/rhop/topology"] = "mesh";            
        mymesh["coordsets/coords/values/z"].set(pos3_vec_z);    mymesh["fields/rhop/values"].set(rhop_vec);       
        //
        myascent.publish(mymesh);
```

````

<!-- </Transform> -->
</div>

<!--
Conduit Mesh Blueprint provides a strategy to describe and adapt mesh data between a wide range of APIs
Ascent uses Conduit as a shared interface to describe and accept simulation mesh data
Ascent accepts Conduit Mesh Blueprint data.

The Execute function in src/source/ascent_adaptor.h is responsible for
preparing and publishing particle simulation data to the Ascent in situ
visualization library.

- It takes particle data arrays (positions and densities) and a timestep.
- It extracts and organizes the particle positions (x, y, z) and density (rhop) into vectors.
- It builds a Conduit mesh node containing:
    The current simulation state and time,
    Coordinates for all particles,
    Mesh topology as a set of points,
    Field data for x, y, z, and rhop (density).
- It verifies the mesh structure with Conduit’s mesh blueprint.
- If the mesh is valid, it publishes the mesh to Ascent and executes any pre-staged visualization actions.
In summary:
Execute sends the current state of your simulation (positions and densities of
particles) to Ascent for visualization or data extraction at a given timestep.
-->

<!-- }}} -->
<!-- {{{ ACTIONS -->

---

## DualSPHysics: How to pass Actions to Ascent ?

<div class="flex justify-right">

<!-- <Transform :scale=".8"> -->

````md magic-move {lines: true}

```yaml
- action: "add_pipelines"      
  pipelines:                   
    pl_threshold_thin_clip_y:  
      f1:                      
        type: "threshold"      
        params:                
          field: "y"           
          min_value: 0.01      
          max_value: 1000      
```

```yaml
- action: "add_pipelines"      - action: "add_scenes"
  pipelines:                     scenes: 
    pl_threshold_thin_clip_y:      s1: 
      f1:                            plots: 
        type: "threshold"              p2: 
        params:                          type: "pseudocolor"
          field: "y"                     field: "rhop"
          min_value: 0.01                pipeline: "pl_threshold_thin_clip_y"
          max_value: 1000                min_value: 0
                                         max_value: 2
                                         color_table: 
                                           name: "Yellow - Gray - Blue"
                                           annotation: "true"
                                         points: 
                                           radius: 0.002
```

```yaml
- action: "add_pipelines"      - action: "add_scenes"
  pipelines:                     scenes: 
    pl_threshold_thin_clip_y:      s1: 
      f1:                            plots: 
        type: "threshold"              p2: 
        params:                          type: "pseudocolor"
          field: "y"                     field: "rhop"
          min_value: 0.01                pipeline: "pl_threshold_thin_clip_y"
          max_value: 1000                min_value: 0
                                         max_value: 2
                                         color_table: 
                                           name: "Yellow - Gray - Blue"
                                           annotation: "true"
                                         points: 
                                           radius: 0.002

                                     renders: 
                                       r1: 
                                         image_prefix: "ascent_out/density."
                                         ... plus camera settings ...
```
````

<!-- </Transform> -->

</div>

<!-- /Users/piccinal/CSCS/OO/dualsph/density/CaseDambreak_4.0_0.0045+ascent/simple_trigger_actions.yaml -->

<!-- }}} -->

<!-- {{{ SPH-EXA test -->

---

### Wind-Cloud collision test

<div class="flex justify-center">
  <img src="/src/images/SPH-EXA_logo.png" class="h-6 ml-5 mr-1">
 <video width="640" height="360" controls>
  <source src="/src/videos/sphexa-density.mp4" type="video/mp4">
</video>
</div>

<div class="absolute bottom-0 left-0 p-12 w-full text-sm text-gray-500">
  <small> Time evolution of density in a thin slice of the domain:
  A spherical cloud of cold gas, initially at rest, 
  is swept by a low-density stream of gas (wind) moving supersonically.
  Kelvin–Helmholtz instabilities are able to develop, mix and eventually destroy the cloud.<br>
  This simulation was run with the SPH-EXA code on CSCS Alps system.
  García-Senz D., Cabezón R. and Jose A. Escartín J. A., 10.1051/0004-6361/202141877
  </small>
</div>
<!-- }}}-->
<!-- {{{ SPH-EXA test failover -->

---

# Wind-Cloud collision test (failover)

This test$^{[1]}$ simulates a spherical cloud of cold gas,
initially at rest, swept by a low-density stream of gas (wind) moving supersonically.

<div class="flex justify-left">
  <img src="/src/images/SPH-EXA_logo.png" class="h-6 ml-5 mr-1">
  <img src="/src/images/density301.01000.png" class="h-55 ml-1">
  <img src="/src/images/density301.03500.png" class="h-55 ml-1">
</div>

<small>
```
             Time evolution of density in a thin slice of the domain,
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

## SPH-EXA: How to pass simulation mesh data to Ascent ?

<div class="flex justify-right">
<Transform :scale=".8">
````md magic-move {lines: true}

```cpp {1-13|14-25|*}
void Execute(DataType& d, long startIndex, long endIndex) {
  conduit::Node mymesh;
  mymesh["coordsets/coords/type"] = "explicit";
  mymesh["coordsets/coords/values/x"].set_external(get<"x">(d).data() + startIndex, endIndex - startIndex);
  mymesh["coordsets/coords/values/y"].set_external(get<"y">(d).data() + startIndex, endIndex - startIndex);
  mymesh["coordsets/coords/values/z"].set_external(get<"z">(d).data() + startIndex, endIndex - startIndex);

  mymesh["topologies/mesh/type"] = "unstructured";
  std::vector<conduit_int32> conn(endIndex - startIndex); // CONNECTIVITY_LIST
  std::iota(conn.begin(), conn.end(), 0);
  mymesh["topologies/mesh/elements/connectivity"].set(conn);
  mymesh["topologies/mesh/elements/shape"] = "point";
  mymesh["topologies/mesh/coordset"] = "coords";

  addField(mymesh, "x", get<"x">(d).data(), startIndex, endIndex);
  addField(mymesh, "y", get<"y">(d).data(), startIndex, endIndex);
  addField(mymesh, "z", get<"z">(d).data(), startIndex, endIndex);
  addField(mymesh, "Density", get<"rho">(d).data(), startIndex, endIndex);
  
void addField(conduit::Node& mymesh, const std::string& name, FieldType* field, size_t start, size_t end)
{
    mymesh["fields/" + name + "/association"] = "vertex";
    mymesh["fields/" + name + "/topology"]    = "mesh";
    mymesh["fields/" + name + "/values"].set_external(field + start, end - start);
    mymesh["fields/" + name + "/volume_dependent"].set("false");
}

  ascent::Ascent myactions;
  myactions.publish(mymesh);
```
````

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

## SPH-EXA: How to pass Actions to Ascent ?

<div class="flex justify-left">
<Transform :scale=".75">

````md magic-move {lines: true}

```yaml {1-5|6-16|*}
- action: "add_triggers"
  triggers:
    t1:
      params:
        condition: "cycle() % 5 == 0"
        actions:
          - action: "add_pipelines"
            pipelines:
              pl_threshold_thin_clip_z:     pl_threshold_thin_clip_y:   
                f1:                           f1:                       
                  type: "threshold"             type: "threshold"       
                  params:                       params:                 
                    field: "z"                    field: "y"            
                    min_value: 0.12425            min_value: 0.12425    
                    max_value: 0.12575            max_value: 0.12575    
                                                                        
          - action: "add_scenes"                                        
            scenes:                                                     
              s1:                                                       
                plots:                                                                      
                  p1:                                       p2:                                   renders:
                    type: "pseudocolor"                       type: "pseudocolor"                   r1:
                    field: "Density"                      field: "Density"                    image_prefix: "datasets/Density.%05d"
                    pipeline: "pl_threshold_thin_clip_z"      pipeline: "pl_threshold_thin_clip_y"    image_width: 1920
                    min_value: 1                              min_value: 1                            image_height: 1080
                    max_value: 10                             max_value: 10                           camera:
                    color_table:                              color_table:                              look_at: [0.5, 0.125, 0.125]
                      name: "Yellow - Gray - Blue"              name: "Yellow - Gray - Blue"            position: [0.5, 0.125, 3.0]
                      annotation: "false"                       annotation: "true"                      up: [0.0, 1.0, 0.0]
                    points:                                   points:                                   azimuth: -35.0
                      radius: 0.002                             radius: 0.002                           elevation: 25.0
                                                                                                        zoom: 5.25
```
````
</Transform>
</div>

<!--
/Users/piccinal/git/CSCS/KEEP_TODO/SPHERIC25/DummySPH.git/Ascent_yaml/sphexa.yaml 
-->

<!-- }}} -->

<!-- {{{ At scale -->

---

## Large scale In Situ Visualisation

<div class="flex justify-left">
  <img src="/src/images/SPH-EXA_logo.png" class="h-6 ml-5 mr-1">
  <img src="/src/images/sphexa_baseline_hdf5_ascent.png" class="h-50" border="1px">
</div>

<!--
We simulated this test with a total number of 55 billion global particles,
arranged in four blocks of 2400^3 particles each, with one block containing a
cavity for the high-density cloud.
- https://www.aanda.org/articles/aa/full_html/2022/03/aa41877-21/aa41877-21.html
 83 GB / 95 GB = 87%
-->

#### Ascent GPU memory usage (thresholding, 87% of GPU peak memory (95GB))

<!-- <small>192 H100 GPUs, 55 billion particles, insitu every 5 iteration, 87% of GPU peak memory (95GB)</small> -->

<div class="flex justify-center">
  <img src="/src/images/sphexa_nsys_gpu_memory-ascent.png" class="h-60 ml-1" border="1px">
</div>

<!-- }}} -->
<!-- {{{ DummySPH -->
---

### `DummySPH`: a mini-app to test In Situ Visualization libraries for SPH

<div class="flex justify-center">
    <img src="/src/images/aos.png" class="h-55 ml-1 mr-1"><br>
</div>

<div class="flex justify-center">
    <img src="/src/images/soa.png" class="h-45 ml-1 mr-1">
    <!-- <img src="/src/images/dummysph_summary.png" class="h-99 ml-1 mr-1"> -->
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

#### Conclusion

- Open issues remain to be fixed but production runs are possible

<div class="flex justify-left">
<small>https://www.cscs.ch/science/computer-science-hpc/</small>
</div>
<div class="flex justify-center">
<video width="480" height="260" controls>
  <source src="/src/videos/jupiter.mp4" type="video/mp4">
</video>
<img src="/src/images/thankyou.png" class="h-35 ml-20 mr-1">
</div>


#### Next steps

* Viskores instead of VTK-m
* Continue testing with SPH-EXA and DualPhysics (1 out of 150 examples tested)
* ROCm AMD GPUs support

<!-- }}} -->
<!-- {{{ References -->

---

## References

<br>

- https://vtk.org, https://vtk-m.readthedocs.io, https://viskores.readthedocs.io
- https://ascent.readthedocs.io
- https://kitware.github.io/paraview-catalyst/ 
- https://github.com/llnl/conduit.git

- https://github.com/sphexa-org/sphexa.git
- https://github.com/DualSPHysics/DualSPHysics.git
- https://github.com/jfavre/DummySPH.git 


<!-- 
https://github.com/Kitware/VTK 
https://github.com/Viskores/viskores

https://github.com/Alpine-DAV/ascent/releases
-->

<!-- 
ascent: a collaborative effort of the U.S. Department of Energy Office of
Science and the National Nuclear Security Administration, Lawrence Livermore
National
Laboratory

vtkm: Copyright Kitware Inc., National Technology & Engineering Solutions of
Sandia LLC, UT-Battelle LLC, Los Alamos National Security LLC

Viskores: This research was funded by the U.S. Department of Energy, including
Oak Ridge, Los Alamos, and Sandia National Laboratories. It utilized resources
from the Oak Ridge and Argonne Leadership Computing Facilities
This research was funded by the U.S. Department of Energy, including Oak Ridge, Los Alamos, and Sandia National Laboratories. It utilized resources from the Oak Ridge and Argonne Leadership Computing Facilities.

Accelerating the Visualization Toolkit for Massively Threaded Architectures
is a toolkit of scientific visualization algorithms for emerging
processor architectures. VTK-m supports the fine-grained concurrency for
data analysis and visualization algorithms required to drive extreme scale
computing by providing abstract models for data and execution that can be
applied to a variety of algorithms across many different processor
architectures.
 VTK-m is being discontinued, Viskores is its successor.
- Viskores: the visualization toolkit for multi/many-core architectures (ORNL, LANL, Sandia)

-->

<!-- }}} -->
