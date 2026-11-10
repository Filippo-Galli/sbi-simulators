# sbi-simulator

A Python library of benchmark simulation models for Simulation-Based Inference (SBI).

## Features

- **Standard benchmark tasks**:
  - `TwoMoons`
  - `BimodalGaussians`
  - `GaussianGaussian`
  - `SpikeSlab`
  - `SIR` (JAX RK4 solver & MCMC posterior sampling)
  - `LotkaVolterra` (JAX RK4 solver & MCMC posterior sampling)
  - `UniformGaussian`
- **Standardized API**:
  - Base `Simulator` class with `simulate(parameters)` and `get_prior()`.
  - Analytic or reference `posterior_samples(x_obs, num_samples)` for ground truth evaluation.
  - Batched simulation and configurable dimensions where supported.
- **Lazy loading**: Simulators are dynamically loaded on access to prevent heavy initialization overhead.
- **Nix-ready**: Packaged with Nix flake and `package.nix` for declarative builds and environments.

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
import torch
from sbi_simulator import SIMULATORS, Simulator

# Access via dictionary mapping
sim_cls = SIMULATORS["TwoMoons"]
sim = sim_cls(seed=42)

# Or import directly
from sbi_simulator.TwoMoons import TwoMoons
sim = TwoMoons(seed=42)

# Sample prior
prior = sim.get_prior()
thetas = prior.sample((10,))

# Simulate observations
x = sim.simulate(thetas)
print("Simulated shapes:", x.shape)
```

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

Enter the development shell with all dependencies (`torch`, `jax`, `pyro-ppl`, `numpy`, `scipy`, `pytest`, `ruff`):
```bash
nix develop
```
