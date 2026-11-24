"""Tests for sbi-simulator package."""

import importlib
import math

import jax
import jax.numpy as jnp
import pyro.distributions as pdist
import pytest
import torch

from sbi_simulator import (
    AR1,
    SIMULATORS,
    SIR,
    SLCP,
    BimodalGaussians,
    DoubleWell,
    GaussianGaussian,
    GaussianGaussianPreset,
    LinearSDE,
    LogNormalPrior,
    Lorenz63,
    LotkaVolterra,
    LVPreset,
    NormalPrior,
    Simulator,
    SIRPreset,
    SpikeSlab,
    TwoMoons,
    UniformGaussian,
    UniformGaussianPreset,
    UniformPrior,
    as_prior_spec,
    extract_two_step_pairs,
)
from sbi_simulator.LinearSDE.simulator import matrix_fraction_decomposition
from sbi_simulator.mcmc import build_mh_mcmc_sampler

EXPECTED_SIMULATORS = [
    "AR1",
    "TwoMoons",
    "BimodalGaussians",
    "DoubleWell",
    "GaussianGaussian",
    "LinearSDE",
    "Lorenz63",
    "SpikeSlab",
    "SIR",
    "SLCP",
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
    gg = GaussianGaussian(dim_parameters=5, seed=42)
    assert gg.dim_parameters == 5
    assert gg.dim_data == 5
    theta_gg = gg.get_prior().sample()
    assert theta_gg.shape == (5,)
    x_gg = gg.simulate(theta_gg)
    assert x_gg.shape[0] == 5

    ug = UniformGaussian(dim_parameters=7, seed=42)
    assert ug.dim_parameters == 7
    assert ug.dim_data == 7
    theta_ug = ug.get_prior().sample()
    assert theta_ug.shape == (7,)
    x_ug = ug.simulate(theta_ug)
    assert x_ug.shape[-1] == 7


def test_analytic_posterior_samplers():
    """Verify simulators implementing posterior_samples() have valid shapes and finite values."""
    num_samples = 20

    ar1 = AR1(seed=42)
    samples_ar1 = ar1.posterior_samples(ar1.simulate(torch.tensor([0.2])), num_samples=num_samples)
    assert samples_ar1.shape == (num_samples, 1)
    assert torch.isfinite(samples_ar1).all()

    gg = GaussianGaussian(dim_parameters=4, seed=42)
    samples_gg = gg.posterior_samples(torch.zeros(4), num_samples=num_samples)
    assert samples_gg.shape == (num_samples, 4)
    assert torch.isfinite(samples_gg).all()

    ss = SpikeSlab(dim_parameters=2, seed=42)
    samples_ss = ss.posterior_samples(torch.zeros(2).numpy(), num_samples=num_samples)
    assert samples_ss.shape == (num_samples, 2)
    assert torch.isfinite(samples_ss).all()

    tm = TwoMoons(seed=42)
    samples_tm = tm.posterior_samples(
        torch.tensor([-0.6397, 0.1623]), num_samples=num_samples
    )
    assert samples_tm.shape == (num_samples, 2)
    assert torch.isfinite(samples_tm).all()

    bg = BimodalGaussians(dim_parameters=2, seed=42)
    samples_bg = bg.posterior_samples(torch.zeros(2), num_samples=num_samples)
    assert samples_bg.shape == (num_samples, 2)
    assert torch.isfinite(samples_bg).all()

    ug = UniformGaussian(dim_parameters=5, seed=42)
    samples_ug = ug.posterior_samples(torch.zeros(5), num_samples=num_samples)
    assert samples_ug.shape == (num_samples, 5)
    assert torch.isfinite(samples_ug).all()

    sir = SIR(seed=42)
    samples_sir = sir.posterior_samples(num_samples=num_samples)
    assert samples_sir.shape == (num_samples, 2)
    assert torch.isfinite(samples_sir).all()

    lv = LotkaVolterra(seed=42)
    samples_lv = lv.posterior_samples(num_samples=num_samples)
    assert samples_lv.shape == (num_samples, 4)
    assert torch.isfinite(samples_lv).all()

    dw = DoubleWell(seed=42)
    samples_dw = dw.posterior_samples(
        num_samples=num_samples, n_chains=2, burn_in=10, inner_thin=1, n_particles=50
    )
    assert samples_dw.shape == (num_samples, 2)
    assert torch.isfinite(samples_dw).all()

    lsde = LinearSDE(seed=42)
    samples_lsde = lsde.posterior_samples(
        num_samples=num_samples, n_chains=2, burn_in=10
    )
    assert samples_lsde.shape == (num_samples, 2)
    assert torch.isfinite(samples_lsde).all()

    slcp = SLCP(seed=42)
    samples_slcp = slcp.posterior_samples(num_samples=num_samples)
    assert samples_slcp.shape == (num_samples, 5)
    assert torch.isfinite(samples_slcp).all()
    assert (samples_slcp >= -3.0).all() and (samples_slcp <= 3.0).all()


def test_sde_simulators_with_1d_simulated_x_obs():
    """Verify multi-dimensional SDE posterior samplers accept 1D x_obs from simulate()."""
    num_samples = 10

    lv = LotkaVolterra(seed=42)
    th_lv = torch.zeros(4)
    x_1d_lv = lv.simulate(th_lv)
    assert x_1d_lv.ndim == 1
    ll_lv = lv.log_likelihood(th_lv, x_1d_lv)
    assert torch.isfinite(ll_lv).all() and ll_lv.item() > -100.0
    samples_lv = lv.posterior_samples(
        x_obs=x_1d_lv,
        num_samples=num_samples,
        n_chains=2,
        burn_in=10,
        inner_thin=1,
        n_particles=50,
    )
    assert samples_lv.shape == (num_samples, 4)
    assert torch.isfinite(samples_lv).all()

    sir = SIR(seed=42)
    th_sir = torch.zeros(2)
    x_1d_sir = sir.simulate(th_sir)
    assert x_1d_sir.ndim == 1
    ll_sir = sir.log_likelihood(th_sir, x_1d_sir)
    assert torch.isfinite(ll_sir).all() and ll_sir.item() > -100.0
    samples_sir = sir.posterior_samples(
        x_obs=x_1d_sir,
        num_samples=num_samples,
        n_chains=2,
        burn_in=10,
        inner_thin=1,
        n_particles=50,
    )
    assert samples_sir.shape == (num_samples, 2)
    assert torch.isfinite(samples_sir).all()

    dw = DoubleWell(seed=42)
    th_dw = torch.zeros(2)
    x_1d_dw = dw.simulate(th_dw)
    assert x_1d_dw.ndim == 1
    ll_dw = dw.log_likelihood(th_dw, x_1d_dw)
    assert torch.isfinite(ll_dw).all() and ll_dw.item() > -100.0
    samples_dw = dw.posterior_samples(
        x_obs=x_1d_dw,
        num_samples=num_samples,
        n_chains=2,
        burn_in=10,
        inner_thin=1,
        n_particles=50,
    )
    assert samples_dw.shape == (num_samples, 2)
    assert torch.isfinite(samples_dw).all()

    lsde = LinearSDE(seed=42)
    th_lsde = torch.zeros(2)
    x_1d_lsde = lsde.simulate(th_lsde)
    assert x_1d_lsde.ndim == 1
    ll_lsde = lsde.log_likelihood(th_lsde, x_1d_lsde)
    assert torch.isfinite(ll_lsde).all() and ll_lsde.item() > -100.0
    samples_lsde = lsde.posterior_samples(
        x_obs=x_1d_lsde, num_samples=num_samples, n_chains=2, burn_in=10
    )
    assert samples_lsde.shape == (num_samples, 2)
    assert torch.isfinite(samples_lsde).all()


def test_posterior_distribution_accuracy():
    """Verify that posterior samples accurately reflect true posterior distributions."""
    # 1. GaussianGaussian exact analytical posterior mean and variance
    gg = GaussianGaussian(dim_parameters=3, seed=42)
    x_obs = torch.tensor([1.0, 2.0, 3.0])
    samples_gg = gg.posterior_samples(x_obs, num_samples=5000)
    diag_sigma = gg._diag_sigma(3)
    expected_var = 1.0 / (1.0 + 1.0 / diag_sigma)
    expected_mean = expected_var / diag_sigma * x_obs
    assert torch.allclose(samples_gg.mean(dim=0), expected_mean, atol=0.04)
    assert torch.allclose(samples_gg.var(dim=0), expected_var, atol=0.04)

    # 2. AR1 exact conjugate posterior mean and variance
    ar1 = AR1(seed=42)
    x_obs_ar1 = ar1.simulate(torch.tensor([0.3]))
    samples_ar1 = ar1.posterior_samples(x_obs_ar1, num_samples=4000)
    y_prev = x_obs_ar1[:-1]
    y_curr = x_obs_ar1[1:]
    var_noise = ar1.sigma**2
    prec_post = 1.0 / (ar1.prior_std**2) + (y_prev**2).sum() / var_noise
    expected_ar1_var = 1.0 / prec_post
    expected_ar1_mean = expected_ar1_var * (y_curr * y_prev).sum() / var_noise
    assert torch.allclose(
        samples_ar1.mean(dim=0), expected_ar1_mean.unsqueeze(0), atol=0.04
    )
    assert torch.allclose(
        samples_ar1.var(dim=0), expected_ar1_var.unsqueeze(0), atol=0.04
    )

    # 3. TwoMoons symmetry (both moons explored with ~50/50 balance)
    tm = TwoMoons(seed=42)
    samples_tm = tm.posterior_samples(torch.tensor([0.0, 0.0]), num_samples=2000)
    p_pos = (samples_tm[:, 0] > 0).float().mean().item()
    assert 0.40 <= p_pos <= 0.60

    # 4. BimodalGaussians symmetry (both modes at +1 and -1 sampled equally)
    bg = BimodalGaussians(dim_parameters=2, seed=42)
    samples_bg = bg.posterior_samples(torch.tensor([0.0, 0.0]), num_samples=2000)
    p_bg_pos = (samples_bg[:, 0] > 0).float().mean().item()
    assert 0.40 <= p_bg_pos <= 0.60

    # 5. UniformGaussian within bounds and centered
    ug = UniformGaussian(dim_parameters=3, seed=42)
    samples_ug = ug.posterior_samples(torch.tensor([0.5, 0.5, 0.5]), num_samples=2000)
    assert torch.allclose(
        samples_ug.mean(dim=0), torch.tensor([0.5, 0.5, 0.5]), atol=0.05
    )
    assert (samples_ug >= -ug.L).all() and (samples_ug <= ug.L).all()

    # 6. SIR SDE posterior accuracy on benchmark observation
    sir = SIR(seed=42)
    samples_sir = sir.posterior_samples(num_samples=2000)
    expected_sir_mean = torch.tensor([-0.0034, 0.0000])
    assert torch.allclose(samples_sir.mean(dim=0), expected_sir_mean, atol=0.05)

    # 7. LotkaVolterra SDE posterior accuracy on benchmark observation
    lv = LotkaVolterra(seed=42)
    samples_lv = lv.posterior_samples(num_samples=2000)
    expected_lv_mean = torch.tensor([-0.3713, -0.7897, -0.1286, 0.1767])
    assert torch.allclose(samples_lv.mean(dim=0), expected_lv_mean, atol=0.10)

    # 8. SLCP posterior invariants and multimodal coverage on benchmark observation 1
    slcp = SLCP(seed=42)
    samples_slcp = slcp.posterior_samples(num_samples=2000)
    assert (samples_slcp >= -3.0).all() and (samples_slcp <= 3.0).all()
    assert -0.8 <= samples_slcp[:, 0].mean().item() <= 1.2
    assert -0.4 <= samples_slcp[:, 1].mean().item() <= 0.5
    assert torch.isclose(samples_slcp[:, 2].abs().mean(), torch.tensor(2.58), atol=0.18)
    assert torch.isclose(samples_slcp[:, 3].abs().mean(), torch.tensor(1.10), atol=0.15)
    assert torch.isclose(samples_slcp[:, 4].mean(), torch.tensor(2.32), atol=0.18)
    p_t2_pos = (samples_slcp[:, 2] > 0).float().mean().item()
    p_t3_pos = (samples_slcp[:, 3] > 0).float().mean().item()
    assert 0.10 <= p_t2_pos <= 0.90
    assert 0.10 <= p_t3_pos <= 0.90


def test_simulator_invalid_inputs():
    """Verify simulators raise proper exceptions on invalid inputs."""
    gg = GaussianGaussian()
    with pytest.raises(ValueError, match="at least one parameter"):
        gg.simulate(torch.tensor([]))

    ss = SpikeSlab()
    with pytest.raises(ValueError, match="at least one parameter"):
        ss.simulate(torch.tensor([]))


def test_ar1_summary_and_two_step():
    """Verify AR1 summary statistics, two-step compositional mode, and dim validation."""
    with pytest.raises(ValueError, match="dim_parameters=1"):
        AR1(dim_parameters=2)

    ar1_stats = AR1(summary="stats", seed=42)
    assert ar1_stats.dim_data == 4
    th_1d = ar1_stats.get_prior().sample()
    out_1d = ar1_stats.simulate(th_1d)
    assert out_1d.shape == (4,)
    assert torch.isfinite(out_1d).all()

    thetas = ar1_stats.get_prior().sample((5,))
    out_batch = ar1_stats.simulate(thetas)
    assert out_batch.shape == (5, 4)
    assert torch.isfinite(out_batch).all()

    ar1_ts = AR1(summary="two_step", seed=42)
    assert ar1_ts.dim_data == 2
    assert ar1_ts.simulate(th_1d).shape == (2,)
    assert ar1_ts.simulate(thetas).shape == (5, 2)

    pairs = ar1_ts.simulate_two_step(thetas, informed=True, return_dict=False)
    assert pairs.shape == (5, 2, 1)
    assert torch.isfinite(pairs).all()

    data_dict = ar1_ts.simulate_two_step(thetas, informed=False, return_dict=True)
    assert {"thetas", "xs", "ts", "x0s"} <= set(data_dict.keys())
    assert data_dict["xs"].shape == (5, 2, 1)
    assert data_dict["ts"].shape == (5,)
    assert (data_dict["ts"] == 0).all()
    assert data_dict["x0s"].shape == (5, 1)

    x_next = ar1_ts.stepper(torch.zeros(2, 1), thetas[:2])
    assert x_next.shape == (2, 1)
    assert torch.isfinite(x_next).all()


def test_lotka_volterra_summary_stats():
    """Verify LotkaVolterra with summary stats mode."""
    lv = LotkaVolterra(summary="stats", seed=42)
    assert lv.dim_parameters == 4
    assert lv.dim_data == 9

    prior = lv.get_prior()
    theta_1d = prior.sample()
    out_1d = lv.simulate(theta_1d)
    assert out_1d.shape == (9,)
    assert torch.isfinite(out_1d).all()

    thetas_batch = prior.sample((4,))
    out_batch = lv.simulate(thetas_batch)
    assert out_batch.shape == (4, 9)
    assert torch.isfinite(out_batch).all()


def test_lotka_volterra_two_step_generator():
    """Verify LotkaVolterra as time series two step generator like in markovsbi."""
    lv = LotkaVolterra(summary="two_step", seed=42)
    assert lv.dim_parameters == 4
    assert lv.dim_data == 4

    prior = lv.get_prior()
    thetas = prior.sample((5,))

    x = lv.simulate(thetas)
    assert x.shape == (5, 4)
    assert torch.isfinite(x).all()

    x_1d = lv.simulate(thetas[0])
    assert x_1d.shape == (4,)
    assert torch.isfinite(x_1d).all()

    pairs = lv.simulate_two_step(thetas, informed=True, return_dict=False)
    assert pairs.shape == (5, 2, 2)
    assert torch.isfinite(pairs).all()
    assert (pairs >= 0.0).all()

    data_dict = lv.simulate_two_step(thetas, informed=True, return_dict=True)
    assert {"thetas", "xs", "ts", "x0s"} <= set(data_dict.keys())
    assert data_dict["xs"].shape == (5, 2, 2)
    assert data_dict["ts"].shape == (5,)
    assert data_dict["x0s"].shape == (5, 2)

    x_curr = torch.tensor([[30.0, 1.0], [25.0, 2.0]])
    x_next = lv.stepper(x_curr, thetas[:2], dt=0.1)
    assert x_next.shape == (2, 2)
    assert torch.isfinite(x_next).all()
    assert (x_next >= 0.0).all()

    prior_x0 = lv.get_prior_x0()
    x0_samples = prior_x0.sample((10,))
    assert x0_samples.shape == (10, 2)
    assert (x0_samples > 0.0).all()


def test_sir_summary_stats():
    """Verify SIR with summary stats mode."""
    sir = SIR(summary="stats", seed=42)
    assert sir.dim_parameters == 2
    assert sir.dim_data == 15

    prior = sir.get_prior()
    theta_1d = prior.sample()
    out_1d = sir.simulate(theta_1d)
    assert out_1d.shape == (15,)
    assert torch.isfinite(out_1d).all()

    thetas_batch = prior.sample((4,))
    out_batch = sir.simulate(thetas_batch)
    assert out_batch.shape == (4, 15)
    assert torch.isfinite(out_batch).all()


def test_sir_two_step_generator():
    """Verify SIR as time series two step generator like in markovsbi."""
    sir = SIR(summary="two_step", seed=42)
    assert sir.dim_parameters == 2
    assert sir.dim_data == 6

    prior = sir.get_prior()
    thetas = prior.sample((5,))

    x = sir.simulate(thetas)
    assert x.shape == (5, 6)
    assert torch.isfinite(x).all()

    x_1d = sir.simulate(thetas[0])
    assert x_1d.shape == (6,)
    assert torch.isfinite(x_1d).all()

    pairs = sir.simulate_two_step(thetas, informed=True, return_dict=False)
    assert pairs.shape == (5, 2, 3)
    assert torch.isfinite(pairs).all()
    assert (pairs >= 0.0).all()

    data_dict = sir.simulate_two_step(thetas, informed=True, return_dict=True)
    assert {"thetas", "xs", "ts", "x0s"} <= set(data_dict.keys())
    assert data_dict["xs"].shape == (5, 2, 3)
    assert data_dict["ts"].shape == (5,)
    assert data_dict["x0s"].shape == (5, 3)

    x_curr = torch.tensor([[4.0, 1.0, 0.0], [3.5, 1.5, 0.1]])
    x_next = sir.stepper(x_curr, thetas[:2], dt=0.05)
    assert x_next.shape == (2, 3)
    assert torch.isfinite(x_next).all()
    assert (x_next >= 0.0).all()

    prior_x0 = sir.get_prior_x0()
    x0_samples = prior_x0.sample((10,))
    assert x0_samples.shape == (10, 3)
    assert (x0_samples >= 0.0).all()


def test_slcp_summary_stats():
    """Verify SLCP with summary stats mode."""
    slcp = SLCP(summary="stats", seed=42)
    assert slcp.dim_parameters == 5
    assert slcp.dim_data == 5

    prior = slcp.get_prior()
    theta_1d = prior.sample()
    out_1d = slcp.simulate(theta_1d)
    assert out_1d.shape == (5,)
    assert torch.isfinite(out_1d).all()

    thetas_batch = prior.sample((4,))
    out_batch = slcp.simulate(thetas_batch)
    assert out_batch.shape == (4, 5)
    assert torch.isfinite(out_batch).all()


def test_slcp_two_step_generator():
    """Verify SLCP as time series two step generator."""
    slcp = SLCP(summary="two_step", seed=42)
    assert slcp.dim_parameters == 5
    assert slcp.dim_data == 4

    prior = slcp.get_prior()
    thetas = prior.sample((5,))

    x = slcp.simulate(thetas)
    assert x.shape == (5, 4)
    assert torch.isfinite(x).all()

    x_1d = slcp.simulate(thetas[0])
    assert x_1d.shape == (4,)
    assert torch.isfinite(x_1d).all()

    pairs = slcp.simulate_two_step(thetas, informed=True, return_dict=False)
    assert pairs.shape == (5, 2, 2)
    assert torch.isfinite(pairs).all()

    data_dict = slcp.simulate_two_step(thetas, informed=True, return_dict=True)
    assert {"thetas", "xs", "ts", "x0s"} <= set(data_dict.keys())
    assert data_dict["xs"].shape == (5, 2, 2)
    assert data_dict["ts"].shape == (5,)

    x_curr = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    x_next = slcp.stepper(x_curr, thetas[:2])
    assert x_next.shape == (2, 2)
    assert torch.isfinite(x_next).all()


def test_double_well_summary_stats():
    """Verify DoubleWell with summary='stats' computes 16 statistics."""
    dw = DoubleWell(summary="stats", seed=42)
    assert dw.dim_parameters == 2
    assert dw.dim_data == 16

    prior = dw.get_prior()
    theta_1d = prior.sample()
    out_1d = dw.simulate(theta_1d)
    assert out_1d.shape == (16,)
    assert torch.isfinite(out_1d).all()

    thetas_batch = prior.sample((4,))
    out_batch = dw.simulate(thetas_batch)
    assert out_batch.shape == (4, 16)
    assert torch.isfinite(out_batch).all()


def test_double_well_two_step_generator():
    """Verify DoubleWell as time series two step generator."""
    dw = DoubleWell(summary="two_step", seed=42)
    assert dw.dim_parameters == 2
    assert dw.dim_data == 8

    prior = dw.get_prior()
    thetas = prior.sample((5,))

    x = dw.simulate(thetas)
    assert x.shape == (5, 8)
    assert torch.isfinite(x).all()

    x_1d = dw.simulate(thetas[0])
    assert x_1d.shape == (8,)
    assert torch.isfinite(x_1d).all()

    pairs = dw.simulate_two_step(thetas, informed=True, return_dict=False)
    assert pairs.shape == (5, 2, 4)
    assert torch.isfinite(pairs).all()

    data_dict = dw.simulate_two_step(thetas, informed=True, return_dict=True)
    assert {"thetas", "xs", "ts", "x0s"} <= set(data_dict.keys())
    assert data_dict["xs"].shape == (5, 2, 4)
    assert data_dict["ts"].shape == (5,)

    x_curr = torch.zeros(2, 4)
    x_next = dw.stepper(x_curr, thetas[:2])
    assert x_next.shape == (2, 4)
    assert torch.isfinite(x_next).all()


def test_linear_sde_summary_stats():
    """Verify LinearSDE with summary='stats' computes 9 statistics."""
    lsde = LinearSDE(summary="stats", seed=42)
    assert lsde.dim_parameters == 2
    assert lsde.dim_data == 9

    prior = lsde.get_prior()
    theta_1d = prior.sample()
    out_1d = lsde.simulate(theta_1d)
    assert out_1d.shape == (9,)
    assert torch.isfinite(out_1d).all()

    thetas_batch = prior.sample((4,))
    out_batch = lsde.simulate(thetas_batch)
    assert out_batch.shape == (4, 9)
    assert torch.isfinite(out_batch).all()


def test_linear_sde_two_step_generator():
    """Verify LinearSDE as time series two step generator."""
    lsde = LinearSDE(summary="two_step", seed=42)
    assert lsde.dim_parameters == 2
    assert lsde.dim_data == 4

    prior = lsde.get_prior()
    thetas = prior.sample((5,))

    x = lsde.simulate(thetas)
    assert x.shape == (5, 4)
    assert torch.isfinite(x).all()

    x_1d = lsde.simulate(thetas[0])
    assert x_1d.shape == (4,)
    assert torch.isfinite(x_1d).all()

    pairs = lsde.simulate_two_step(thetas, informed=True, return_dict=False)
    assert pairs.shape == (5, 2, 2)
    assert torch.isfinite(pairs).all()

    data_dict = lsde.simulate_two_step(thetas, informed=True, return_dict=True)
    assert {"thetas", "xs", "ts", "x0s"} <= set(data_dict.keys())
    assert data_dict["xs"].shape == (5, 2, 2)
    assert data_dict["ts"].shape == (5,)

    x_curr = torch.zeros(2, 2)
    x_next = lsde.stepper(x_curr, thetas[:2])
    assert x_next.shape == (2, 2)
    assert torch.isfinite(x_next).all()


def test_linear_sde_variants():
    """Verify LinearSDE variants ('periodic' with dim=2, 'general' with dim=8)."""
    p_sde = LinearSDE(variant="periodic", seed=42)
    assert p_sde.dim_parameters == 2
    assert p_sde.dim_data == 22
    th_p = p_sde.get_prior().sample((3,))
    x_p = p_sde.simulate(th_p)
    assert x_p.shape == (3, 22)

    g_sde = LinearSDE(variant="general", dim_parameters=8, seed=42)
    assert g_sde.dim_parameters == 8
    assert g_sde.dim_data == 22
    th_g = g_sde.get_prior().sample((3,))
    x_g = g_sde.simulate(th_g)
    assert x_g.shape == (3, 22)


def test_sir_mode_and_prng():
    """Verify SIR ODE and SDE modes and PRNG key state advancement."""
    sir_sde = SIR(seed=42, mode="sde")
    assert sir_sde.mode == "sde"
    th = torch.tensor([0.33, 0.12])
    x1 = sir_sde.simulate(th)
    x2 = sir_sde.simulate(th)
    assert not torch.allclose(x1, x2)

    sir_sde.reset_rng(42)
    x1_reset = sir_sde.simulate(th)
    assert torch.allclose(x1, x1_reset)

    sir_ode = SIR(seed=42, mode="ode")
    assert sir_ode.mode == "ode"
    xo1 = sir_ode.simulate(th)
    xo2 = sir_ode.simulate(th)
    assert torch.allclose(xo1, xo2)

    ll_ode = sir_ode.log_likelihood(th, xo1)
    assert torch.isfinite(ll_ode).all()
    samples_ode = sir_ode.posterior_samples(xo1, num_samples=50, n_chains=2, burn_in=20)
    assert samples_ode.shape == (50, 2)
    assert torch.isfinite(samples_ode).all()

    x_curr = torch.tensor([[4.0, 1.0, 0.0]])
    x_next_ode = sir_ode.stepper(x_curr, th)
    assert x_next_ode.shape == (1, 3)
    assert torch.isfinite(x_next_ode).all()


def test_lotka_volterra_mode_and_prng():
    """Verify LotkaVolterra ODE and SDE modes and PRNG key state advancement."""
    lv_sde = LotkaVolterra(seed=42, mode="sde")
    assert lv_sde.mode == "sde"
    th = torch.zeros(4)
    x1 = lv_sde.simulate(th)
    x2 = lv_sde.simulate(th)
    assert not torch.allclose(x1, x2)

    lv_sde.reset_rng(42)
    x1_reset = lv_sde.simulate(th)
    assert torch.allclose(x1, x1_reset)

    lv_ode = LotkaVolterra(seed=42, mode="ode")
    assert lv_ode.mode == "ode"
    xo1 = lv_ode.simulate(th)
    xo2 = lv_ode.simulate(th)
    assert torch.allclose(xo1, xo2)

    ll_ode = lv_ode.log_likelihood(th, xo1)
    assert torch.isfinite(ll_ode).all()
    samples_ode = lv_ode.posterior_samples(xo1, num_samples=50, n_chains=2, burn_in=20)
    assert samples_ode.shape == (50, 4)
    assert torch.isfinite(samples_ode).all()

    x_curr = torch.tensor([[1.0, 0.5]])
    x_next_ode = lv_ode.stepper(x_curr, th)
    assert x_next_ode.shape == (1, 2)
    assert torch.isfinite(x_next_ode).all()


def test_lotka_volterra_presets_and_kwargs():
    """Verify LotkaVolterra presets and Simulator kwargs passing."""
    lv_custom = LotkaVolterra(
        preset="sbibm", custom_tag="experiment_1", note="test_kwargs"
    )
    assert lv_custom.kwargs.get("custom_tag") == "experiment_1"
    assert lv_custom.kwargs.get("note") == "test_kwargs"

    lv_m = LotkaVolterra(preset="markovsbi")
    assert lv_m.preset == "markovsbi"
    assert lv_m.num_steps == 11
    assert lv_m.dim_data == 22
    assert lv_m.summary is None
    assert lv_m.param_transform == "sigmoid"
    assert torch.allclose(lv_m.u0, torch.tensor([1.0, 0.5]))
    assert lv_m.mode == "sde"

    lv_p = LotkaVolterra(preset="sbibm")
    assert lv_p.preset == "sbibm"
    assert lv_p.num_steps == 10
    assert lv_p.dim_data == 20
    assert lv_p.summary == "lognormal"
    assert lv_p.param_transform == "identity"
    assert lv_p.param_order == "alphabetical"
    assert torch.allclose(lv_p.u0, torch.tensor([30.0, 1.0]))
    assert lv_p.mode == "ode"

    th = lv_p.get_prior().sample((10,))
    assert th.shape == (10, 4)
    assert (th > 0.0).all()

    out_paper = lv_p.simulate(lv_p.canonical_theta)
    assert out_paper.shape == (20,)
    assert torch.isfinite(out_paper).all()
    assert (out_paper > 0.0).all()

    samples_paper = lv_p.posterior_samples(
        out_paper, num_samples=20, n_chains=2, burn_in=10, inner_thin=1
    )
    assert samples_paper.shape == (20, 4)
    assert torch.isfinite(samples_paper).all()
    assert (samples_paper > 0.0).all()

    lv_snl = LotkaVolterra(preset="papamakarios")
    assert lv_snl.preset == "papamakarios"
    assert lv_snl.num_steps == 151
    assert lv_snl.dim_data == 9
    assert lv_snl.summary == "stats"
    assert lv_snl.param_transform == "exp"
    assert lv_snl.param_order == "paper"
    assert torch.allclose(lv_snl.u0, torch.tensor([100.0, 50.0]))
    assert lv_snl.mode == "ode"

    th_snl = lv_snl.get_prior().sample((10,))
    assert th_snl.shape == (10, 4)
    assert (th_snl >= -5.0).all() and (th_snl <= 2.0).all()

    out_stats = lv_snl.simulate(lv_snl.canonical_theta)
    assert out_stats.shape == (9,)
    assert torch.isfinite(out_stats).all()

    lv_snl_raw = LotkaVolterra(preset="papamakarios", summary=None)
    assert lv_snl_raw.dim_data == 302
    out_raw = lv_snl_raw.simulate(lv_snl.canonical_theta)
    assert out_raw.shape == (302,)
    assert torch.isfinite(out_raw).all()

    x_obs = lv_snl_raw.simulate(lv_snl.canonical_theta)
    samples = lv_snl_raw.posterior_samples(
        x_obs, num_samples=20, n_chains=2, burn_in=10, inner_thin=1
    )
    assert samples.shape == (20, 4)
    assert torch.isfinite(samples).all()


def test_sir_and_gaussian_paper_presets():
    """Verify SIR, GaussianGaussian, and UniformGaussian 'sbibm' presets."""
    sir_p = SIR(preset="sbibm", seed=42)
    assert sir_p.preset == "sbibm"
    assert sir_p.num_steps == 10
    assert sir_p.dim_data == 10
    assert sir_p.summary == "binomial"
    assert sir_p.param_transform == "identity"
    assert sir_p.param_order == "paper"
    assert sir_p.total_population == 1_000_000.0
    assert torch.allclose(sir_p.u0, torch.tensor([999999.0, 1.0, 0.0]))
    assert sir_p.mode == "ode"

    th_sir = sir_p.get_prior().sample((10,))
    assert th_sir.shape == (10, 2)
    assert (th_sir > 0.0).all()

    out_sir = sir_p.simulate(sir_p.canonical_theta)
    assert out_sir.shape == (10,)
    assert torch.isfinite(out_sir).all()
    assert (out_sir >= 0.0).all() and (out_sir <= 1000.0).all()

    samples_sir = sir_p.posterior_samples(
        out_sir, num_samples=20, n_chains=2, burn_in=10, inner_thin=1
    )
    assert samples_sir.shape == (20, 2)
    assert torch.isfinite(samples_sir).all()
    assert (samples_sir > 0.0).all()

    gg_p = GaussianGaussian(preset="sbibm", seed=42)
    assert gg_p.preset == "sbibm"
    assert gg_p.dim_parameters == 10
    assert torch.allclose(gg_p._diag_sigma(10), torch.full((10,), 0.1))
    samples_gg = gg_p.posterior_samples(torch.zeros(10), num_samples=50)
    assert samples_gg.shape == (50, 10)
    assert torch.isfinite(samples_gg).all()

    ug_p = UniformGaussian(preset="sbibm", seed=42)
    assert ug_p.preset == "sbibm"
    assert ug_p.dim_parameters == 10
    assert ug_p.L == 1.0
    assert abs(ug_p.sigma**2 - 0.1) < 1e-6
    samples_ug = ug_p.posterior_samples(torch.zeros(10), num_samples=50)
    assert samples_ug.shape == (50, 10)
    assert (samples_ug >= -1.0).all() and (samples_ug <= 1.0).all()


def test_double_well_and_linear_sde_prng_and_edge_cases():
    """Verify PRNG advancement on DoubleWell/LinearSDE and edge-case dimension handling."""
    dw = DoubleWell(seed=42)
    th_dw = torch.zeros(2)
    x1 = dw.simulate(th_dw)
    x2 = dw.simulate(th_dw)
    assert not torch.allclose(x1, x2)
    dw.reset_rng(42)
    assert torch.allclose(x1, dw.simulate(th_dw))

    dw_odd = DoubleWell(D=3, seed=42)
    assert dw_odd.u0.shape == (3,)
    assert dw_odd.simulate(th_dw).shape == (33,)

    lsde = LinearSDE(seed=42)
    th_ls = torch.zeros(2)
    y1 = lsde.simulate(th_ls)
    y2 = lsde.simulate(th_ls)
    assert not torch.allclose(y1, y2)
    lsde.reset_rng(42)
    assert torch.allclose(y1, lsde.simulate(th_ls))

    with pytest.raises(ValueError, match="state dimension d=2"):
        LinearSDE(variant="general", d=3, dim_parameters=18, summary="stats")


def test_ode_stepper_matches_solve_rk4():
    """Verify ODE stepper uses RK4 matching _solve()."""
    sir_ode = SIR(mode="ode", seed=42)
    th_sir = torch.tensor([0.3, -0.2])
    traj_sir = sir_ode._solve(th_sir)
    step_sir = sir_ode.stepper(sir_ode.u0, th_sir)
    assert torch.allclose(step_sir, traj_sir[0, :, 1], atol=1e-4)

    lv_ode = LotkaVolterra(mode="ode", seed=42)
    th_lv = torch.tensor([0.1, -0.2, 0.3, -0.1])
    traj_lv = lv_ode._solve(th_lv)
    step_lv = lv_ode.stepper(lv_ode.u0, th_lv)
    assert torch.allclose(step_lv, traj_lv[0, :, 1], atol=1e-4)


def test_lorenz63_trajectory_and_transitions():
    from scipy.integrate import solve_ivp

    sim = Lorenz63(num_steps=4, spinup_steps=5)
    theta = sim.canonical_theta

    def rhs(_, state):
        x, y, z = state
        return 10 * (y - x), x * (28 - z) - y, x * y - (8 / 3) * z

    times = [0.10, 0.12, 0.14, 0.16]
    reference = solve_ivp(
        rhs, (0.0, times[-1]), [1.0, 1.0, 1.0], t_eval=times, rtol=1e-10, atol=1e-12
    )
    trajectory = sim.simulate(theta.numpy()).reshape(3, sim.num_steps)
    torch.testing.assert_close(
        trajectory, torch.tensor(reference.y).float(), rtol=2e-5, atol=2e-5
    )

    informed = sim.simulate_two_step(theta.unsqueeze(0), x0=sim.u0, t=1)
    torch.testing.assert_close(informed[0], trajectory[:, 1:3].T)
    local = sim.simulate_two_step(theta, x0=trajectory[:, 1], informed=False)
    torch.testing.assert_close(local, informed[0])

    sim.reset_rng()
    states = sim.get_prior_x0(num_sims=3, num_steps=20).sample((5,))
    assert states.shape == (5, 3) and torch.isfinite(states).all()
    sim.reset_rng()
    torch.testing.assert_close(
        sim.get_prior_x0(num_sims=3, num_steps=20).sample((5,)), states
    )


@pytest.mark.parametrize("summary", [None, "subsample", "stats", "summary_stats", "two_step"])
def test_lorenz63_reference_posterior(summary):
    sim = Lorenz63(num_steps=5, spinup_steps=5, summary=summary)
    theta = sim.canonical_theta
    observation = sim.simulate(theta)
    assert observation.shape == (sim.dim_data,)

    ll = sim.log_likelihood(torch.stack((theta, theta + 0.2)), observation.numpy())
    assert torch.isfinite(ll).all() and ll[0] > ll[1]

    samples = sim.posterior_samples(
        observation, num_samples=8, n_chains=2, burn_in=10, inner_thin=1, n_sir=32
    )
    assert samples.shape == (8, 3) and torch.isfinite(samples).all()


def test_compositional_two_step_edge_cases():
    """Verify explicit time_axis handling when state_dim >= num_steps and clamped ts."""
    # Shape (batch=3, state_dim=4, num_steps=2) where state_dim > num_steps
    traj = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    out = extract_two_step_pairs(traj, t=99, return_dict=True, time_axis=2)
    assert out["xs"].shape == (3, 2, 4)
    assert (out["ts"] == 0).all()
    assert torch.allclose(out["xs"][:, 0, :], traj[:, :, 0])
    assert torch.allclose(out["xs"][:, 1, :], traj[:, :, 1])


def test_bimodal_gaussians_exact_multi_obs_posterior():
    """Verify exact multi-observation posterior sampling for BimodalGaussians."""
    bg = BimodalGaussians(dim_parameters=2, seed=42)
    # 1. Symmetric N=4 (exact 2^N mixture)
    samples_sym = bg.posterior_samples(torch.zeros(4, 2), num_samples=1000)
    assert (samples_sym >= -10.0).all() and (samples_sym <= 10.0).all()
    assert torch.allclose(samples_sym.mean(dim=0), torch.zeros(2), atol=0.15)

    # 2. Informative mixed-sign observations at theta_true=[2.0, -1.0] for N=6 (exact) & N=20 (Gibbs)
    theta_true = torch.tensor([2.0, -1.0])
    base_6 = torch.cat(
        [
            theta_true.unsqueeze(0) + bg.shift,
            theta_true.unsqueeze(0) + bg.shift,
            theta_true.unsqueeze(0) + bg.shift,
            theta_true.unsqueeze(0) - bg.shift,
            theta_true.unsqueeze(0) - bg.shift,
            theta_true.unsqueeze(0) - bg.shift,
        ],
        dim=0,
    )
    samples_6 = bg.posterior_samples(base_6, num_samples=1500)
    assert torch.allclose(samples_6.mean(dim=0), theta_true, atol=0.05)

    obs_20 = bg.simulate(theta_true.unsqueeze(0).expand(20, -1))
    samples_20 = bg.posterior_samples(obs_20, num_samples=1500)
    assert torch.allclose(samples_20.mean(dim=0), theta_true, atol=0.15)


def test_gaussian_gaussian_high_dim_posterior():
    """Verify O(D) diagonal posterior sampling in GaussianGaussian for high D."""
    dim = 100
    gg = GaussianGaussian(dim_parameters=dim, seed=42)
    x_obs = torch.linspace(-1.0, 1.0, dim)
    samples = gg.posterior_samples(x_obs, num_samples=4000)
    assert samples.shape == (4000, dim)
    diag_sigma = gg._diag_sigma(dim)
    expected_var = 1.0 / (1.0 + 1.0 / diag_sigma)
    expected_mean = expected_var / diag_sigma * x_obs
    assert torch.allclose(samples.mean(dim=0), expected_mean, atol=0.06)
    assert torch.allclose(samples.var(dim=0), expected_var, atol=0.06)


def test_two_moons_1d_shape():
    """Verify TwoMoons matches the 1D->1D / 2D->2D convention."""
    tm = TwoMoons(seed=42)
    assert tm.simulate(torch.zeros(2)).shape == (2,)
    assert tm.simulate(torch.zeros(3, 2)).shape == (3, 2)


def test_jax_jit_cache_reuse():
    """Verify multiple instances share compiled JAX callables across seeds."""
    sir1, sir2 = SIR(seed=1), SIR(seed=2)
    assert sir1._jax_solver_batch is sir2._jax_solver_batch
    assert sir1._jax_mcmc is sir2._jax_mcmc

    lv1, lv2 = LotkaVolterra(seed=1), LotkaVolterra(seed=2)
    assert lv1._jax_solver_batch is lv2._jax_solver_batch
    assert lv1._jax_mcmc is lv2._jax_mcmc

    dw1, dw2 = DoubleWell(seed=1), DoubleWell(seed=2)
    assert dw1._jax_solver_batch is dw2._jax_solver_batch
    assert dw1._jax_mcmc is dw2._jax_mcmc

    ls1, ls2 = LinearSDE(seed=1), LinearSDE(seed=2)
    assert ls1._jax_solver_batch is ls2._jax_solver_batch
    assert ls1._jax_step_batch is ls2._jax_step_batch

    sl1, sl2 = SLCP(seed=1), SLCP(seed=2)
    assert sl1._v_ll_fn is sl2._v_ll_fn
    assert sl1._jax_mcmc is sl2._jax_mcmc


def test_linear_sde_kalman_exactness():
    """Verify single-Cholesky Kalman filter log-likelihood matches naive reference."""
    lsde = LinearSDE(seed=42)
    th = torch.tensor([0.5, -0.7])
    x_obs = lsde.simulate(th)
    ll_fast = lsde.log_likelihood(th, x_obs).item()

    # Naive reference implementation using direct matrix inverse and slogdet
    x_2d = jnp.asarray(x_obs.reshape(2, lsde.num_steps).T.numpy())
    th_j = jnp.asarray(th.numpy())
    A = jnp.array([[0.0, -(th_j[0] ** 2)], [th_j[1] ** 2, 0.0]])
    B = jnp.array([[lsde.sigma1, 0.0], [0.0, lsde.sigma2]])
    Phi, Q = matrix_fraction_decomposition(lsde.step_dt, A, B)
    R = (lsde.obs_sigma**2) * jnp.eye(2)
    m = x_2d[0]
    P = 1e-5 * jnp.eye(2)
    ll_ref = 0.0
    for t in range(1, lsde.num_steps):
        m_p = Phi @ m
        P_p = Phi @ P @ Phi.T + Q
        S = P_p + R
        S_inv = jnp.linalg.inv(S)
        r = x_2d[t] - m_p
        _, logdet = jnp.linalg.slogdet(S)
        ll_ref += float(-0.5 * (2 * math.log(2 * math.pi) + logdet + r @ S_inv @ r))
        K = P_p @ S_inv
        m = m_p + K @ r
        P = P_p - K @ S @ K.T

    assert jax.numpy.isfinite(ll_fast)
    assert abs(ll_fast - ll_ref) < 1e-4


def test_prior_spec_pyro_roundtrip():
    """Verify PriorSpec <-> Pyro conversion, hashing, and log-prob parity."""
    specs = [
        NormalPrior(loc=(0.5, -1.0, 2.0), scale=(0.25, 1.5, 0.8)),
        UniformPrior(low=(-2.0, 0.1, -5.0), high=(1.0, 3.0, 0.0)),
        LogNormalPrior(loc=(-0.5, 0.2), scale=(0.4, 0.7)),
    ]
    for spec in specs:
        pyro_dist = spec.to_pyro()
        roundtrip = as_prior_spec(pyro_dist)
        assert type(roundtrip) is type(spec)
        assert roundtrip.dim == spec.dim

        # Float32 round-trip through Pyro tensors is idempotent and hash-consistent
        second = as_prior_spec(roundtrip.to_pyro())
        assert second == roundtrip
        assert hash(second) == hash(roundtrip)

        if isinstance(spec, UniformPrior) and isinstance(roundtrip, UniformPrior):
            assert torch.allclose(torch.tensor(roundtrip.low), torch.tensor(spec.low), atol=1e-6)
            assert torch.allclose(torch.tensor(roundtrip.high), torch.tensor(spec.high), atol=1e-6)
        elif isinstance(spec, (NormalPrior, LogNormalPrior)) and isinstance(
            roundtrip, (NormalPrior, LogNormalPrior)
        ):
            assert torch.allclose(torch.tensor(roundtrip.loc), torch.tensor(spec.loc), atol=1e-6)
            assert torch.allclose(torch.tensor(roundtrip.scale), torch.tensor(spec.scale), atol=1e-6)

        sample = pyro_dist.sample()
        lp_pyro = float(pyro_dist.log_prob(sample).item())
        lp_jax = float(spec.jax_log_prob(jnp.asarray(sample.numpy())))
        assert abs(lp_pyro - lp_jax) < 1e-4


def test_custom_data_driven_presets():
    """Verify passing custom Preset dataclasses works with zero simulator edits."""
    custom_lv = LVPreset(
        dt=0.02,
        inner_steps=5,
        sigma1=0.0,
        sigma2=0.0,
        num_steps=11,
        obs_sigma=0.05,
        mode="ode",
        u0=(2.0, 1.0),
        summary=None,
        param_transform="identity",
        param_order="alphabetical",
        prior=UniformPrior(low=(0.4, 0.1, 0.4, 0.1), high=(1.5, 0.8, 1.5, 0.8)),
        prior_x0=UniformPrior.box(2, 0.5, 5.0),
        canonical_theta=(0.8, 0.3, 0.9, 0.2),
        default_step_size=0.05,
    )
    lv = LotkaVolterra(preset=custom_lv, seed=42)
    assert lv.preset == "custom"
    th_lv = lv.sample_prior(3)
    assert th_lv.shape == (3, 4)
    obs_lv = lv.simulate(lv.canonical_theta)
    assert obs_lv.shape == (22,)
    post_lv = lv.posterior_samples(
        obs_lv, num_samples=20, n_chains=2, burn_in=10, inner_thin=1
    )
    assert post_lv.shape == (20, 4)
    assert (post_lv[:, 0] >= 0.4).all() and (post_lv[:, 0] <= 1.5).all()
    assert (post_lv[:, 1] >= 0.1).all() and (post_lv[:, 1] <= 0.8).all()

    custom_sir = SIRPreset(
        dt=0.01,
        inner_steps=5,
        sigma1=0.0,
        num_steps=11,
        obs_sigma=0.05,
        mode="ode",
        total_population=10.0,
        u0=(8.0, 2.0, 0.0),
        summary=None,
        param_transform="identity",
        param_order="paper",
        prior=UniformPrior(low=(0.2, 0.05), high=(1.2, 0.6)),
        prior_x0=UniformPrior.box(3, 0.0, 10.0),
        canonical_theta=(0.6, 0.2),
        default_step_size=0.05,
    )
    sir = SIR(preset=custom_sir, seed=42)
    assert sir.preset == "custom"
    obs_sir = sir.simulate(sir.canonical_theta)
    assert obs_sir.shape == (33,)
    post_sir = sir.posterior_samples(
        obs_sir, num_samples=20, n_chains=2, burn_in=10, inner_thin=1
    )
    assert post_sir.shape == (20, 2)
    assert (post_sir[:, 0] >= 0.2).all() and (post_sir[:, 0] <= 1.2).all()
    assert (post_sir[:, 1] >= 0.05).all() and (post_sir[:, 1] <= 0.6).all()

    custom_gg = GaussianGaussianPreset(prior_var=0.25, lik_var_range=(0.2, 0.8))
    gg = GaussianGaussian(dim_parameters=4, preset=custom_gg, seed=42)
    assert gg.preset == "custom"
    diag_sig = gg._diag_sigma(4)
    assert torch.allclose(diag_sig, torch.linspace(0.2, 0.8, steps=4))
    x_obs_gg = torch.tensor([1.0, -1.0, 0.5, 2.0])
    post_gg = gg.posterior_samples(x_obs_gg, num_samples=4000)
    exp_var = 1.0 / (1.0 / 0.25 + 1.0 / diag_sig)
    exp_mean = exp_var * (x_obs_gg / diag_sig)
    assert torch.allclose(post_gg.mean(dim=0), exp_mean, atol=0.05)
    assert torch.allclose(post_gg.var(dim=0), exp_var, atol=0.05)

    custom_ug = UniformGaussianPreset(sigma=0.2, L=1.5)
    ug = UniformGaussian(dim_parameters=4, preset=custom_ug, seed=42)
    assert ug.preset == "custom"
    assert ug.sigma == 0.2
    assert ug.L == 1.5
    post_ug = ug.posterior_samples(torch.zeros(4), num_samples=50)
    assert (post_ug >= -1.5).all() and (post_ug <= 1.5).all()


def test_custom_pyro_prior_syncs_with_jax_mcmc():
    """Verify custom Pyro distributions sync across get_prior() and JAX MCMC."""
    custom_sir_prior = pdist.Uniform(
        torch.tensor([0.2, 0.05]), torch.tensor([0.6, 0.25])
    ).to_event(1)
    sir = SIR(preset="sbibm", prior=custom_sir_prior, seed=42)
    obs = sir.simulate(sir.canonical_theta)
    samples = sir.posterior_samples(
        obs, num_samples=40, n_chains=2, burn_in=20, inner_thin=1
    )
    assert samples.shape == (40, 2)
    assert torch.isfinite(samples).all()
    assert (samples[:, 0] >= 0.2).all() and (samples[:, 0] <= 0.6).all()
    assert (samples[:, 1] >= 0.05).all() and (samples[:, 1] <= 0.25).all()

    custom_lv_prior = NormalPrior(
        loc=(0.5, -0.5, 0.2, -0.2), scale=(0.3, 0.3, 0.3, 0.3)
    )
    lv = LotkaVolterra(mode="sde", prior=custom_lv_prior, seed=42)
    obs_lv = lv.simulate(lv.canonical_theta)
    samples_lv = lv.posterior_samples(
        obs_lv, num_samples=20, n_chains=2, burn_in=10, inner_thin=1, n_particles=50
    )
    assert samples_lv.shape == (20, 4)
    assert torch.isfinite(samples_lv).all()


def test_ode_mcmc_custom_x_obs_convergence():
    """Verify ODE MCMC converges and mixes well on non-canonical prior-drawn x_obs."""
    cases = [
        (LotkaVolterra(preset="sbibm", seed=10), 2),
        (SIR(preset="sbibm", seed=11), 2),
        (LotkaVolterra(preset="papamakarios", seed=12), 1),
    ]
    for sim, n_draws in cases:
        torch.manual_seed(7)
        thetas_true = sim.sample_prior(n_draws)
        for th_true in thetas_true:
            x_obs = sim.simulate(th_true)
            lp_true = float(
                sim.log_likelihood(th_true, x_obs).item()
                + sim.get_prior().log_prob(th_true).item()
            )
            samples = sim.posterior_samples(x_obs, num_samples=2000)
            assert samples.shape == (2000, sim.dim_parameters)
            assert torch.isfinite(samples).all()

            # High unique sample fraction confirms adaptive warmup avoids stuck steps
            unique_ratio = torch.unique(samples, dim=0).shape[0] / samples.shape[0]
            assert unique_ratio > 0.75, f"Low unique ratio {unique_ratio:.3f} on {sim}"

            # Evaluate posterior log-joint on a subsample: 10th percentile must be near mode
            sub = samples[:400]
            sub_lp = sim.log_likelihood(sub, x_obs) + sim.get_prior().log_prob(sub)
            assert torch.isfinite(sub_lp).all()
            q10_lp = float(torch.quantile(sub_lp, 0.10).item())
            assert q10_lp > lp_true - 25.0, (
                f"Stuck chains detected on {sim}: q10_lp={q10_lp:.2f}, lp_true={lp_true:.2f}"
            )


def test_ode_mcmc_self_consistency():
    """Verify two independent ODE MCMC runs agree closely on canonical and custom x_obs."""
    for sim_cls in (LotkaVolterra, SIR):
        sim_a = sim_cls(preset="sbibm", seed=101)
        sim_b = sim_cls(preset="sbibm", seed=202)
        torch.manual_seed(13)
        custom_theta = sim_a.sample_prior(1)[0]
        for th in (sim_a.canonical_theta, custom_theta):
            x_obs = sim_a.simulate(th)
            s1 = sim_a.posterior_samples(x_obs, num_samples=3000)
            s2 = sim_b.posterior_samples(x_obs, num_samples=3000)
            mean_diff = (s1.mean(dim=0) - s2.mean(dim=0)).abs() / (s1.std(dim=0) + 1e-8)
            std_ratio = s1.std(dim=0) / (s2.std(dim=0) + 1e-8)
            assert float(mean_diff.max().item()) < 0.15
            assert (std_ratio > 0.80).all() and (std_ratio < 1.25).all()


def test_mh_mcmc_adapt_warmup_edge_cases():
    """Verify sir_pool, adaptive warmup edge cases, and multimodal SLCP coverage."""
    # Strongly correlated 2D Gaussian target: N(mu, Sigma) with rho=0.95
    mu_target = jnp.array([2.5, -1.8], dtype=jnp.float32)
    cov_target = jnp.array([[0.04, 0.038], [0.038, 0.04]], dtype=jnp.float32)
    inv_cov = jnp.linalg.inv(cov_target)
    prior = NormalPrior.isotropic(2)

    # Cancel prior in likelihood so exact posterior is N(mu_target, cov_target)
    def ll_fn(theta, _x_obs):
        diff = theta - mu_target
        target_lp = -0.5 * jnp.dot(diff, inv_cov @ diff)
        return target_lp - prior.jax_log_prob(theta)

    sampler = build_mh_mcmc_sampler(
        ll_fn,
        prior=prior,
        default_init_theta=(0.0, 0.0),
        init_strategy="sir_pool",
        rescue_stuck_chains=True,
    )
    dummy_obs = jnp.zeros(1, dtype=jnp.float32)

    # Edge case: n_sir < n_chains with full 3-phase adaptive warmup
    samples = sampler(
        dummy_obs,
        jax.random.PRNGKey(0),
        n_chains=16,
        num_samples_per_chain=200,
        burn_in=240,
        inner_thin=4,
        step_size=0.5,
        n_sir=4,
        adapt_warmup=True,
    )
    assert samples.shape == (3200, 2)
    assert jnp.isfinite(samples).all()
    assert jnp.allclose(jnp.mean(samples, axis=0), mu_target, atol=0.05)
    assert jnp.allclose(jnp.cov(samples.T), cov_target, atol=0.015)

    # Edge cases: burn_in=0, small burn_in=10 (below cov threshold), adapt_warmup=False
    for burn_in, adapt in [(0, True), (10, True), (20, False)]:
        out = sampler(
            dummy_obs,
            jax.random.PRNGKey(1),
            n_chains=4,
            num_samples_per_chain=10,
            burn_in=burn_in,
            inner_thin=2,
            step_size=0.1,
            n_sir=16,
            adapt_warmup=adapt,
        )
        assert out.shape == (40, 2)
        assert jnp.isfinite(out).all()

    # Multimodal preservation on SLCP under adaptive per-chain warmup
    slcp = SLCP(seed=42)
    slcp_samples = slcp.posterior_samples(num_samples=2000)
    assert slcp_samples.shape == (2000, 5)
    assert torch.isfinite(slcp_samples).all()
    assert (slcp_samples[:, 2] > 0.2).any() and (slcp_samples[:, 2] < -0.2).any()
    assert (slcp_samples[:, 3] > 0.2).any() and (slcp_samples[:, 3] < -0.2).any()
    assert torch.unique(slcp_samples, dim=0).shape[0] / slcp_samples.shape[0] > 0.50
