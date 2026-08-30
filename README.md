# JAXSIM

<p align="center">
  <img src="assets/drop_trimmed.gif" width="320" />
</p>

jaxsim is a JAX rewrite of [gradSim](https://github.com/gradsim/gradsim): differentiable rendering and physics for parameter estimation and control from video. Rigid bodies, deformable solids, cloth. All differentiable through simulation and rendering.

gradSim was PyTorch + NVIDIA warp/dflex. jaxsim replaces that with JAX: autodiff, JIT and physics integration, all native, no CUDA kernel compilation.

---

## Modules

| Module | Description |
|--------|-------------|
| `jaxsim.dflex` | Cloth, FEM, rigid body physics. (Semi-implicit Euler) Ported from NVIDIA dflex |
| `jaxsim.renderutils` | Soft rasterizer (Liu et al. 2019) and DIBR renderer |
| `jaxsim.bodies` | Rigid body definitions: inertia, center of mass |
| `jaxsim.engines` | Euler integrator |
| `jaxsim.utils` | Mesh utilities, quaternion math, GIF/image logging |

---

## Benchmarks

Comparing this implementation to the original PyTorch based codebase.

### CPU

**DFlex spring-mass chain** (jit-able functional state)

| Particles | torch fwd | jax fwd (jit) | JAX throughput | torch bwd | jax bwd (jit) | JAX throughput |
|---|---|---|---|---|---|---|
| 10 | 1.88 ms | 0.10 ms | 19x | 1.88 ms | 0.07 ms | 27x |
| 50 | 2.14 ms | 0.04 ms | 54x | 2.19 ms | 0.03 ms | 73x |
| 200 | 3.18 ms | 0.06 ms | 53x | 3.35 ms | 0.03 ms | 112x |
| 1000 | 8.42 ms | n/a | — | 9.79 ms | n/a | — |

n/a means the unrolled loop was too large to compile in reasonable time. Eager mode (no jit) runs about 150 ms flat, which is 20-70x slower than torch, and compilation itself costs 150-440 ms.

**RigidBody `Simulator`** (mutates state each step and not jit-able)

| Bodies | torch fwd | JAX fwd | torch bwd | JAX bwd |
|---|---|---|---|---|
| 1 | 14.07 ms | 139.32 ms | 7.52 ms | 77.59 ms |
| 5 | 69.95 ms | 692.16 ms | 35.52 ms | 384.51 ms |
| 20 | 282.98 ms | 2754.13 ms | 149.43 ms | n/a (>7s/trial) |

torch has ~10x faster throughout. Neither side JIT-compiles here because JIT compilation mutates Python attributes each step.

**Optimization loop** (30 Adam iterations, 50-particle chain, 20 sim-steps/iter)

| | torch | JAX (jit) | × |
|---|---|---|---|
| total, 30 iters | 79 ms | 1195 ms (incl. ~1.2s compile) | torch 15.2x |
| steady-state / iter | 1.92 ms | 0.23 ms | jax 8.5x |

Breakeven is about 690 iterations, below which the total wall-clock favors torch (compile cost dominates), and above which jax is faster.

### GPU (RTX 4090)

**DFlex spring-mass chain**

| Particles | torch fwd | jax fwd (jit) | JAX throughput | torch bwd | jax bwd (jit) | JAX throughput |
|---|---|---|---|---|---|---|
| 10 | 2.12 ms | 0.13 ms | 17x | 2.94 ms | 0.22 ms | 13x |
| 50 | 2.12 ms | 0.12 ms | 18x | 3.17 ms | 0.17 ms | 19x |
| 200 | 2.12 ms | 0.13 ms | 17x | 2.85 ms | 0.21 ms | 14x |
| 1000 | 2.16 ms | n/a | — | 3.00 ms | n/a | — |

Both stay flat across particle count here, because this problem size is kernel-launch bound rather than compute bound.

**RigidBody `Simulator`**

| Bodies | torch fwd | JAX fwd | torch bwd | JAX bwd |
|---|---|---|---|---|
| 1 | 34.3 ms | 319.7 ms | 18.1 ms | 73.2 ms |
| 5 | 170.2 ms | 1615.3 ms | 88.3 ms | 373.0 ms |
| 20 | 803.5 ms | 6577.0 ms | 414.9 ms | 1495.7 ms |
| 100 | 4001.3 ms | 33305.2 ms | 2014.4 ms | 7724.2 ms |

torch is about 8-9x faster on the forward pass and about 4x faster on the gradient.

**Optimization loop** (same task as CPU)

| | torch | JAX (jit) | × |
|---|---|---|---|
| total, 30 iters | 166 ms | 2182 ms (incl. ~2.2s compile) | torch 13.2x |
| steady-state / iter | 3.31 ms | 0.58 ms | jax 5.7x |

Breakeven is about 770 iterations, below which the total wall-clock favors torch, and above which jax is faster.

torch and jax were both slower on GPU than on CPU across every benchmark above, because these problem sizes (tens to low hundreds of particles or bodies) are too small to hide GPU launch and dispatch latency behind compute. Larger batches or bigger meshes would be needed to see the GPU actually pay off.


## Installation

Requires Python ≥ 3.9.
### 1. Create a virtual environment

```bash
python -m venv .venv && source .venv/bin/activate
```

### 2. Install JAX (CPU)
```bash
pip install jax jaxlib
```

### For GPU (CUDA 12):
```bash
pip install -U "jax[cuda12]"
```

### 3. Install jaxsim
```bash
pip install -e .
```


## Examples

Run any example from the repo root:

```bash
cd examples
../../jaxenv/bin/python3 <script>.py
```

### Physics demos

| Script | Description |
|--------|-------------|
| `hellodflex.py` | Spring-mass smoketest — 9-particle chain |
| `demo_pendulum.py` | Simple pendulum with parameter optimization |
| `demo_double_pendulum.py` | Double pendulum |
| `demo_bounce2d.py` | 2D bouncing ball, optimize restitution |
| `demo_tablecloth.py` | Flat cloth dropping onto a ground plane |
| `demo_cloth_sphere.py` | Cloth draped over a static sphere (GIF output) |

### Parameter estimation

| Script | Description |
|--------|-------------|
| `demo_mass_known_shape.py` | Estimate mass from rendered video |
| `demo_fem.py` | FEM material parameter optimization |
| `demo_cloth.py` | Cloth velocity optimization from images |

### Visuomotor control

| Script | Description |
|--------|-------------|
| `control_walker.py` | Deformable walker locomotion |
| `control_cloth.py` | Cloth manipulation |
| `control_fem.py` | Deformable gear control |

### Rendering

| Script | Description |
|--------|-------------|
| `softras_simple_render.py` | Soft rasterization forward pass |
| `softras_texture_optimization.py` | Optimize texture from target image |
| `dibr_forward_render.py` | DIBR forward render |

---

