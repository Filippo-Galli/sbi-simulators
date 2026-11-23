# sbi-simulator

A Python library of benchmark simulation models for Simulation-Based Inference (SBI).

## Features

- **Standard benchmark tasks**:
  - `AR1`
  - `BimodalGaussians`
  - `DoubleWell`
  - `GaussianGaussian`
  - `LinearSDE`
  - `LotkaVolterra`
  - `SIR`
  - `SLCP`
  - `SpikeSlab`
  - `TwoMoons`
  - `UniformGaussian`
- **Standardized API**:
  - Base `Simulator` and `DynamicalSimulator` classes with `simulate(parameters)`, `get_prior()`, and `sample_prior(num_samples)`.
  - Analytic or JAX-accelerated reference `posterior_samples(x_obs, num_samples)` for ground-truth evaluation.
  - Trajectory summary statistics (`summary="stats"`) and compositional transition pairs (`summary="two_step"` / `simulate_two_step()`).
  - Batched simulation, cross-instance JAX JIT caching, and configurable dimensions where supported.
- **Nix-ready**: Packaged with Nix flake and `default.nix` for declarative builds and environments.

## Installation

### With pip
```bash
pip install .
```

### With Nix Flakes
Add to your `flake.nix`:
```nix
inputs = {
  sbi-simulator.url = "git+https://github.com/Filippo-Galli/sbi-simulators";
};
```
Or run directly:
```bash
nix run git+https://github.com/Filippo-Galli/sbi-simulators
```

## Usage

```python
from sbi_simulator import SIMULATORS

# Access via dictionary mapping
sim = SIMULATORS["TwoMoons"](seed=42)

# Or import directly
from sbi_simulator.TwoMoons import TwoMoons
sim = TwoMoons(seed=42)

# Sample prior and simulate observations
thetas = sim.sample_prior(10)
x = sim.simulate(thetas)
print("Simulated shapes:", x.shape)
```

## Adding a New Simulator

1. Create a subpackage under `src/sbi_simulator/<YourSim>/` with `__init__.py` and `simulator.py`.
2. Subclass:
   - `Simulator` (from `sbi_simulator.base`) for static/analytic models: implement `get_prior()`, `simulate(parameters)`, and `posterior_samples(x_obs=None, num_samples=1000, **kwargs)` using `self._prepare_parameters(parameters)`.
   - `DynamicalSimulator` (from `sbi_simulator.dynamical`) for SDE/ODE time-series models: attach a cached JAX `_jax_solver_batch`, implement `get_prior()`, `posterior_samples(...)`, and optionally `get_prior_x0()`, `stepper()`, and `compute_summary_stats()` to get `simulate()`, summary modes, PRNG advancement, and compositional `simulate_two_step()` automatically.
3. Register the class in `SIMULATORS` and `__all__` in `src/sbi_simulator/__init__.py`.

## Running Tests

Using pytest:
```bash
pytest
```

Using Nix:
```bash
nix flake check
```

## Development

Enter the development shell with all dependencies (`torch`, `jax`, `pyro-ppl`, `numpy`, `scipy`, `pytest`, `ruff`, `ty`):
```bash
nix develop
```
