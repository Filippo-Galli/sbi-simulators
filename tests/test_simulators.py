"""Tests for sbi-simulator package."""

import pytest
import torch
from sbi_simulator import SIMULATORS, Simulator

EXPECTED_SIMULATORS = [
    "TwoMoons",
    "BimodalGaussians",
    "GaussianGaussian",
    "SpikeSlab",
    "SIR",
    "LotkaVolterra",
    "UniformGaussian",
]


@pytest.mark.parametrize("sim_name", EXPECTED_SIMULATORS)
def test_simulators_registration(sim_name):
    """Verify that all expected simulators are registered and subclass Simulator."""
    assert sim_name in SIMULATORS
    sim_cls = SIMULATORS[sim_name]
    assert issubclass(sim_cls, Simulator)


@pytest.mark.parametrize("sim_name", EXPECTED_SIMULATORS)
def test_simulators_subpackage_import(sim_name):
    """Verify that all expected simulators can be imported from their subpackage."""
    import importlib

    mod = importlib.import_module(f"sbi_simulator.{sim_name}")
    sim_cls = getattr(mod, sim_name)
    assert issubclass(sim_cls, Simulator)


@pytest.mark.parametrize("sim_name", EXPECTED_SIMULATORS)
def test_simulator_prior_and_dimensions(sim_name):
    """Verify prior generation and dimension consistency for all simulators."""
    sim = SIMULATORS[sim_name](seed=123)
    assert sim.dim_parameters > 0
    assert sim.dim_data > 0

    prior = sim.get_prior()
    samples = prior.sample((5,))
    assert samples.shape == (5, sim.dim_parameters)
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("sim_name", EXPECTED_SIMULATORS)
def test_simulator_simulate_1d_input(sim_name):
    """Verify simulate() with a 1D parameter vector produces valid output."""
    sim = SIMULATORS[sim_name](seed=123)
    prior = sim.get_prior()
    theta = prior.sample().flatten()

    x = sim.simulate(theta)
    x_flat = x.cpu().flatten()
    assert x_flat.shape[0] == sim.dim_data
    assert torch.isfinite(x_flat).all()


@pytest.mark.parametrize("sim_name", EXPECTED_SIMULATORS)
def test_simulator_simulate_batch(sim_name):
    """Verify batched simulation for all simulators supporting 2D input tensors."""
    sim = SIMULATORS[sim_name](seed=42)
    prior = sim.get_prior()
    batch_size = 4
    thetas = prior.sample((batch_size,))
    if thetas.ndim == 3:
        thetas = thetas.squeeze(1)

    x = sim.simulate(thetas)
    assert x.shape[0] == batch_size
    assert x.shape[1] == sim.dim_data
    assert torch.isfinite(x).all()


def test_custom_dimensions():
    """Verify simulators that support configurable dimensions."""
    # GaussianGaussian with dim=5
    gg = SIMULATORS["GaussianGaussian"](dim_parameters=5, seed=42)
    assert gg.dim_parameters == 5
    assert gg.dim_data == 5
    theta_gg = gg.get_prior().sample()
    assert theta_gg.shape == (5,)
    x_gg = gg.simulate(theta_gg)
    assert x_gg.shape[0] == 5

    # UniformGaussian with dim=7
    ug = SIMULATORS["UniformGaussian"](dim_parameters=7, seed=42)
    assert ug.dim_parameters == 7
    assert ug.dim_data == 7
    theta_ug = ug.get_prior().sample()
    assert theta_ug.shape == (7,)
    x_ug = ug.simulate(theta_ug)
    assert x_ug.shape[-1] == 7


def test_analytic_posterior_samplers():
    """Verify simulators implementing posterior_samples() have valid shapes and finite values."""
    num_samples = 20

    # GaussianGaussian
    gg = SIMULATORS["GaussianGaussian"](dim_parameters=4, seed=42)
    x_obs_gg = torch.zeros(4)
    samples_gg = gg.posterior_samples(x_obs_gg, num_samples=num_samples)
    assert samples_gg.shape == (num_samples, 4)
    assert torch.isfinite(samples_gg).all()

    # SpikeSlab
    ss = SIMULATORS["SpikeSlab"](dim_parameters=2, seed=42)
    x_obs_ss = torch.zeros(2).numpy()
    samples_ss = ss.posterior_samples(x_obs_ss, num_samples=num_samples)
    assert samples_ss.shape == (num_samples, 2)
    assert torch.isfinite(samples_ss).all()

    # TwoMoons
    tm = SIMULATORS["TwoMoons"](seed=42)
    x_obs_tm = torch.tensor([-0.6397, 0.1623])
    samples_tm = tm.posterior_samples(x_obs_tm, num_samples=num_samples)
    assert samples_tm.shape == (num_samples, 2)
    assert torch.isfinite(samples_tm).all()

    # BimodalGaussians
    bg = SIMULATORS["BimodalGaussians"](dim_parameters=2, seed=42)
    x_obs_bg = torch.zeros(2)
    samples_bg = bg.posterior_samples(x_obs_bg, num_samples=num_samples)
    assert samples_bg.shape == (num_samples, 2)
    assert torch.isfinite(samples_bg).all()

    # UniformGaussian
    ug = SIMULATORS["UniformGaussian"](dim_parameters=5, seed=42)
    x_obs_ug = torch.zeros(5)
    samples_ug = ug.posterior_samples(x_obs_ug, num_samples=num_samples)
    assert samples_ug.shape == (num_samples, 5)
    assert torch.isfinite(samples_ug).all()

    # SIR
    sir = SIMULATORS["SIR"](seed=42)
    samples_sir = sir.posterior_samples(num_samples=num_samples)
    assert samples_sir.shape == (num_samples, 2)
    assert torch.isfinite(samples_sir).all()

    # LotkaVolterra
    lv = SIMULATORS["LotkaVolterra"](seed=42)
    samples_lv = lv.posterior_samples(num_samples=num_samples)
    assert samples_lv.shape == (num_samples, 4)
    assert torch.isfinite(samples_lv).all()


def test_posterior_distribution_accuracy():
    """Verify that posterior samples accurately reflect true posterior distributions."""
    # 1. GaussianGaussian exact analytical posterior mean and variance
    gg = SIMULATORS["GaussianGaussian"](dim_parameters=3, seed=42)
    x_obs = torch.tensor([1.0, 2.0, 3.0])
    samples_gg = gg.posterior_samples(x_obs, num_samples=5000)
    diag_sigma = gg._diag_sigma(3)
    expected_var = 1.0 / (1.0 + 1.0 / diag_sigma)
    expected_mean = expected_var / diag_sigma * x_obs
    assert torch.allclose(samples_gg.mean(dim=0), expected_mean, atol=0.04)
    assert torch.allclose(samples_gg.var(dim=0), expected_var, atol=0.04)

    # 2. TwoMoons symmetry (both moons explored with ~50/50 balance)
    tm = SIMULATORS["TwoMoons"](seed=42)
    samples_tm = tm.posterior_samples(torch.tensor([0.0, 0.0]), num_samples=2000)
    p_pos = (samples_tm[:, 0] > 0).float().mean().item()
    assert 0.40 <= p_pos <= 0.60

    # 3. BimodalGaussians symmetry (both modes at +3 and -3 sampled equally)
    bg = SIMULATORS["BimodalGaussians"](dim_parameters=2, seed=42)
    samples_bg = bg.posterior_samples(torch.tensor([0.0, 0.0]), num_samples=2000)
    p_bg_pos = (samples_bg[:, 0] > 0).float().mean().item()
    assert 0.40 <= p_bg_pos <= 0.60

    # 4. UniformGaussian within bounds and centered
    ug = SIMULATORS["UniformGaussian"](dim_parameters=3, seed=42)
    samples_ug = ug.posterior_samples(torch.tensor([0.5, 0.5, 0.5]), num_samples=2000)
    assert torch.allclose(samples_ug.mean(dim=0), torch.tensor([0.5, 0.5, 0.5]), atol=0.05)
    assert (samples_ug >= -ug.L).all() and (samples_ug <= ug.L).all()

    # 5. SIR posterior accuracy on benchmark observation
    sir = SIMULATORS["SIR"](seed=42)
    x_obs_sir = torch.tensor([0.0, 0.0, 3.0, 120.0, 264.0, 69.0, 14.0, 5.0, 0.0, 0.0])
    samples_sir = sir.posterior_samples(x_obs_sir, num_samples=500)
    # Reference benchmark mean is [0.349, 0.114]
    assert torch.allclose(samples_sir.mean(dim=0), torch.tensor([0.349, 0.114]), atol=0.03)

    # 6. LotkaVolterra posterior accuracy on benchmark observation 1
    lv = SIMULATORS["LotkaVolterra"](seed=42)
    samples_lv = lv.posterior_samples(num_samples=500)
    # Reference benchmark mean is [0.401, 0.084, 0.328, 0.068]
    assert torch.allclose(samples_lv.mean(dim=0), torch.tensor([0.401, 0.084, 0.328, 0.068]), atol=0.03)


def test_simulator_invalid_inputs():
    """Verify simulators raise proper exceptions on invalid inputs."""
    gg = SIMULATORS["GaussianGaussian"]()
    with pytest.raises(ValueError, match="at least one parameter"):
        gg.simulate(torch.tensor([]))

    ss = SIMULATORS["SpikeSlab"]()
    with pytest.raises(ValueError, match="at least one parameter"):
        ss.simulate(torch.tensor([]))
