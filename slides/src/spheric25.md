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
$\times$ Long term storage at 60 💶 / TB / year<br>
~ 4.5 million (per year)<br><br>
In-situ Visualization to the rescue 🛟<br>

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
<v-click>In-situ Visualization to the rescue 🛟</v-click><br>
-->

<!-- }}} -->

<!-- {{{ Backends -->
---

# In-situ Visualization

<div class="flex items-center gap-0"> 
<img src="/src/images/vtk-logo.png" class="h-10 ml-1 mr-1">Kitware Visualization Toolkit<br>
<img src="/src/images/vtkm-logo.svg" class="h-6 ml-1 mr-1">Visualization Toolkit for multi/many-core architectures<br>
<img src="/src/images/viskores-logo.png" class="h-5 ml-1 mr-1">Successor to VTK-m being discontinued<br>
(Kitware, ORNL, LANL, Sandia, UT-Batelle)
</div>

<div class="flex items-center gap-0">
<img src="/src/images/ascent-logo.png" class="h-15 ml-1 mr-1">Flyweight In-situ library<br>
for HPC simulations<br>(NNSA/LLNL)

<img src="/src/images/pvcatalyst-logo.png" class="h-7 ml-1 mr-1">ParaView implementation<br>of the Catalyst API<br>(Kitware, Sandia, LANL)

</div>


<!-- }}} -->

---

https://github.com/jfavre/DummySPH
