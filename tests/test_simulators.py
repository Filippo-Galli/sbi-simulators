"""Tests for sbi-simulator package."""

import pytest
import torch

from sbi_simulator import SIMULATORS, Simulator

EXPECTED_SIMULATORS = [
    "TwoMoons",
    "BimodalGaussians",
    "DoubleWell",
    "GaussianGaussian",
    "LinearSDE",
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

    # DoubleWell
    dw = SIMULATORS["DoubleWell"](seed=42)
    samples_dw = dw.posterior_samples(
        num_samples=num_samples, n_chains=2, burn_in=10, inner_thin=1, n_particles=50
    )
    assert samples_dw.shape == (num_samples, 2)
    assert torch.isfinite(samples_dw).all()

    # LinearSDE
    lsde = SIMULATORS["LinearSDE"](seed=42)
    samples_lsde = lsde.posterior_samples(
        num_samples=num_samples, n_chains=2, burn_in=10
    )
    assert samples_lsde.shape == (num_samples, 2)
    assert torch.isfinite(samples_lsde).all()

    # SLCP
    slcp = SIMULATORS["SLCP"](seed=42)
    samples_slcp = slcp.posterior_samples(num_samples=num_samples)
    assert samples_slcp.shape == (num_samples, 5)
    assert torch.isfinite(samples_slcp).all()
    assert (samples_slcp >= -3.0).all() and (samples_slcp <= 3.0).all()


def test_sde_simulators_with_1d_simulated_x_obs():
    """Verify that multi-dimensional SDE posterior samplers work when passed 1D x_obs from simulate()."""
    num_samples = 10

    # LotkaVolterra (dim=4, states=2, num_steps=11)
    lv = SIMULATORS["LotkaVolterra"](seed=42)
    th_lv = torch.zeros(4)
    x_1d_lv = lv.simulate(th_lv)
    assert x_1d_lv.ndim == 1
    ll_lv = lv.log_likelihood(th_lv, x_1d_lv)
    assert torch.isfinite(ll_lv).all() and ll_lv.item() > -100.0
    samples_lv = lv.posterior_samples(
        x_obs=x_1d_lv, num_samples=num_samples, n_chains=2, burn_in=10, inner_thin=1, n_particles=50
    )
    assert samples_lv.shape == (num_samples, 4)
    assert torch.isfinite(samples_lv).all()

    # SIR (dim=2, states=3, num_steps=11)
    sir = SIMULATORS["SIR"](seed=42)
    th_sir = torch.zeros(2)
    x_1d_sir = sir.simulate(th_sir)
    assert x_1d_sir.ndim == 1
    ll_sir = sir.log_likelihood(th_sir, x_1d_sir)
    assert torch.isfinite(ll_sir).all() and ll_sir.item() > -100.0
    samples_sir = sir.posterior_samples(
        x_obs=x_1d_sir, num_samples=num_samples, n_chains=2, burn_in=10, inner_thin=1, n_particles=50
    )
    assert samples_sir.shape == (num_samples, 2)
    assert torch.isfinite(samples_sir).all()

    # DoubleWell (dim=2, states=4, num_steps=11)
    dw = SIMULATORS["DoubleWell"](seed=42)
    th_dw = torch.zeros(2)
    x_1d_dw = dw.simulate(th_dw)
    assert x_1d_dw.ndim == 1
    ll_dw = dw.log_likelihood(th_dw, x_1d_dw)
    assert torch.isfinite(ll_dw).all() and ll_dw.item() > -100.0
    samples_dw = dw.posterior_samples(
        x_obs=x_1d_dw, num_samples=num_samples, n_chains=2, burn_in=10, inner_thin=1, n_particles=50
    )
    assert samples_dw.shape == (num_samples, 2)
    assert torch.isfinite(samples_dw).all()

    # LinearSDE (dim=2, states=2, num_steps=11)
    lsde = SIMULATORS["LinearSDE"](seed=42)
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
    assert torch.allclose(
        samples_ug.mean(dim=0), torch.tensor([0.5, 0.5, 0.5]), atol=0.05
    )
    assert (samples_ug >= -ug.L).all() and (samples_ug <= ug.L).all()

    # 5. SIR SDE posterior accuracy on benchmark observation
    sir = SIMULATORS["SIR"](seed=42)
    samples_sir = sir.posterior_samples(num_samples=2000)
    expected_sir_mean = torch.tensor([-0.0034, 0.0000])
    assert torch.allclose(samples_sir.mean(dim=0), expected_sir_mean, atol=0.05)

    # 6. LotkaVolterra SDE posterior accuracy on benchmark observation
    lv = SIMULATORS["LotkaVolterra"](seed=42)
    samples_lv = lv.posterior_samples(num_samples=2000)
    expected_lv_mean = torch.tensor([-0.3667, -0.7719, -0.2398, 0.1093])
    assert torch.allclose(samples_lv.mean(dim=0), expected_lv_mean, atol=0.08)

    # 7. SLCP posterior accuracy on benchmark observation 1
    slcp = SIMULATORS["SLCP"](seed=42)
    samples_slcp = slcp.posterior_samples(num_samples=2000)
    expected_slcp_mean = torch.tensor([0.5022, 0.0757, 0.9482, -0.1026, 2.3475])
    assert torch.allclose(samples_slcp.mean(dim=0), expected_slcp_mean, atol=0.08)


def test_simulator_invalid_inputs():
    """Verify simulators raise proper exceptions on invalid inputs."""
    gg = SIMULATORS["GaussianGaussian"]()
    with pytest.raises(ValueError, match="at least one parameter"):
        gg.simulate(torch.tensor([]))

    ss = SIMULATORS["SpikeSlab"]()
    with pytest.raises(ValueError, match="at least one parameter"):
        ss.simulate(torch.tensor([]))


def test_lotka_volterra_summary_stats():
    """Verify LotkaVolterra with summary stats mode."""
    lv = SIMULATORS["LotkaVolterra"](summary="stats", seed=42)
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
    lv = SIMULATORS["LotkaVolterra"](summary="two_step", seed=42)
    assert lv.dim_parameters == 4
    assert lv.dim_data == 4

    prior = lv.get_prior()
    thetas = prior.sample((5,))

    # 1. Standard simulate() returning flattened (5, 4)
    x = lv.simulate(thetas)
    assert x.shape == (5, 4)
    assert torch.isfinite(x).all()

    # 2. 1D input returning (4,)
    x_1d = lv.simulate(thetas[0])
    assert x_1d.shape == (4,)
    assert torch.isfinite(x_1d).all()

    # 3. simulate_two_step returning (5, 2, 2)
    pairs = lv.simulate_two_step(thetas, informed=True, return_dict=False)
    assert pairs.shape == (5, 2, 2)
    assert torch.isfinite(pairs).all()
    # Population sizes must be non-negative
    assert (pairs >= 0.0).all()

    # 4. simulate_two_step with return_dict=True (markovsbi interface)
    data_dict = lv.simulate_two_step(thetas, informed=True, return_dict=True)
    assert (
        "thetas" in data_dict
        and "xs" in data_dict
        and "ts" in data_dict
        and "x0s" in data_dict
    )
    assert data_dict["xs"].shape == (5, 2, 2)
    assert data_dict["ts"].shape == (5,)
    assert data_dict["x0s"].shape == (5, 2)

    # 5. Stepper method
    x_curr = torch.tensor([[30.0, 1.0], [25.0, 2.0]])
    params_stepper = thetas[:2]
    x_next = lv.stepper(x_curr, params_stepper, dt=0.1)
    assert x_next.shape == (2, 2)
    assert torch.isfinite(x_next).all()
    assert (x_next >= 0.0).all()

    # 6. Prior over x0
    prior_x0 = lv.get_prior_x0()
    x0_samples = prior_x0.sample((10,))
    assert x0_samples.shape == (10, 2)
    assert (x0_samples > 0.0).all()


def test_sir_summary_stats():
    """Verify SIR with summary stats mode."""
    sir = SIMULATORS["SIR"](summary="stats", seed=42)
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
    sir = SIMULATORS["SIR"](summary="two_step", seed=42)
    assert sir.dim_parameters == 2
    assert sir.dim_data == 6

    prior = sir.get_prior()
    thetas = prior.sample((5,))

    # 1. Standard simulate() returning flattened (5, 6)
    x = sir.simulate(thetas)
    assert x.shape == (5, 6)
    assert torch.isfinite(x).all()

    # 2. 1D input returning (6,)
    x_1d = sir.simulate(thetas[0])
    assert x_1d.shape == (6,)
    assert torch.isfinite(x_1d).all()

    # 3. simulate_two_step returning (5, 2, 3)
    pairs = sir.simulate_two_step(thetas, informed=True, return_dict=False)
    assert pairs.shape == (5, 2, 3)
    assert torch.isfinite(pairs).all()
    # SIR populations must be non-negative
    assert (pairs >= 0.0).all()

    # 4. simulate_two_step with return_dict=True (markovsbi interface)
    data_dict = sir.simulate_two_step(thetas, informed=True, return_dict=True)
    assert (
        "thetas" in data_dict
        and "xs" in data_dict
        and "ts" in data_dict
        and "x0s" in data_dict
    )
    assert data_dict["xs"].shape == (5, 2, 3)
    assert data_dict["ts"].shape == (5,)
    assert data_dict["x0s"].shape == (5, 3)

    # 5. Stepper method
    x_curr = torch.tensor([[4.0, 1.0, 0.0], [3.5, 1.5, 0.1]])
    params_stepper = thetas[:2]
    x_next = sir.stepper(x_curr, params_stepper, dt=0.05)
    assert x_next.shape == (2, 3)
    assert torch.isfinite(x_next).all()
    assert (x_next >= 0.0).all()

    # 6. Prior over x0
    prior_x0 = sir.get_prior_x0()
    x0_samples = prior_x0.sample((10,))
    assert x0_samples.shape == (10, 3)
    assert (x0_samples >= 0.0).all()


def test_slcp_summary_stats():
    """Verify SLCP with summary stats mode."""
    slcp = SIMULATORS["SLCP"](summary="stats", seed=42)
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
    slcp = SIMULATORS["SLCP"](summary="two_step", seed=42)
    assert slcp.dim_parameters == 5
    assert slcp.dim_data == 4

    prior = slcp.get_prior()
    thetas = prior.sample((5,))

    # 1. Standard simulate() returning flattened (5, 4)
    x = slcp.simulate(thetas)
    assert x.shape == (5, 4)
    assert torch.isfinite(x).all()

    # 2. 1D input returning (4,)
    x_1d = slcp.simulate(thetas[0])
    assert x_1d.shape == (4,)
    assert torch.isfinite(x_1d).all()

    # 3. simulate_two_step returning (5, 2, 2)
    pairs = slcp.simulate_two_step(thetas, informed=True, return_dict=False)
    assert pairs.shape == (5, 2, 2)
    assert torch.isfinite(pairs).all()

    # 4. simulate_two_step with return_dict=True
    data_dict = slcp.simulate_two_step(thetas, informed=True, return_dict=True)
    assert (
        "thetas" in data_dict
        and "xs" in data_dict
        and "ts" in data_dict
        and "x0s" in data_dict
    )
    assert data_dict["xs"].shape == (5, 2, 2)
    assert data_dict["ts"].shape == (5,)

    # 5. Stepper method
    x_curr = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    params_stepper = thetas[:2]
    x_next = slcp.stepper(x_curr, params_stepper)
    assert x_next.shape == (2, 2)
    assert torch.isfinite(x_next).all()


def test_double_well_summary_stats():
    """Verify DoubleWell with summary='stats' computes 16 statistics."""
    dw = SIMULATORS["DoubleWell"](summary="stats", seed=42)
    assert dw.dim_parameters == 2
    assert dw.dim_data == 16  # 4 stats * 4 dims

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
    dw = SIMULATORS["DoubleWell"](summary="two_step", seed=42)
    assert dw.dim_parameters == 2
    assert dw.dim_data == 8  # 2 * D

    prior = dw.get_prior()
    thetas = prior.sample((5,))

    # 1. Standard simulate() returning flattened (5, 8)
    x = dw.simulate(thetas)
    assert x.shape == (5, 8)
    assert torch.isfinite(x).all()

    # 2. 1D input returning (8,)
    x_1d = dw.simulate(thetas[0])
    assert x_1d.shape == (8,)
    assert torch.isfinite(x_1d).all()

    # 3. simulate_two_step returning (5, 2, 4)
    pairs = dw.simulate_two_step(thetas, informed=True, return_dict=False)
    assert pairs.shape == (5, 2, 4)
    assert torch.isfinite(pairs).all()

    # 4. simulate_two_step with return_dict=True
    data_dict = dw.simulate_two_step(thetas, informed=True, return_dict=True)
    assert (
        "thetas" in data_dict
        and "xs" in data_dict
        and "ts" in data_dict
        and "x0s" in data_dict
    )
    assert data_dict["xs"].shape == (5, 2, 4)
    assert data_dict["ts"].shape == (5,)

    # 5. Stepper method
    x_curr = torch.zeros(2, 4)
    params_stepper = thetas[:2]
    x_next = dw.stepper(x_curr, params_stepper)
    assert x_next.shape == (2, 4)
    assert torch.isfinite(x_next).all()


def test_linear_sde_summary_stats():
    """Verify LinearSDE with summary='stats' computes 9 statistics."""
    lsde = SIMULATORS["LinearSDE"](summary="stats", seed=42)
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
    lsde = SIMULATORS["LinearSDE"](summary="two_step", seed=42)
    assert lsde.dim_parameters == 2
    assert lsde.dim_data == 4

    prior = lsde.get_prior()
    thetas = prior.sample((5,))

    # 1. Standard simulate() returning flattened (5, 4)
    x = lsde.simulate(thetas)
    assert x.shape == (5, 4)
    assert torch.isfinite(x).all()

    # 2. 1D input returning (4,)
    x_1d = lsde.simulate(thetas[0])
    assert x_1d.shape == (4,)
    assert torch.isfinite(x_1d).all()

    # 3. simulate_two_step returning (5, 2, 2)
    pairs = lsde.simulate_two_step(thetas, informed=True, return_dict=False)
    assert pairs.shape == (5, 2, 2)
    assert torch.isfinite(pairs).all()

    # 4. simulate_two_step with return_dict=True
    data_dict = lsde.simulate_two_step(thetas, informed=True, return_dict=True)
    assert (
        "thetas" in data_dict
        and "xs" in data_dict
        and "ts" in data_dict
        and "x0s" in data_dict
    )
    assert data_dict["xs"].shape == (5, 2, 2)
    assert data_dict["ts"].shape == (5,)

    # 5. Stepper method
    x_curr = torch.zeros(2, 2)
    params_stepper = thetas[:2]
    x_next = lsde.stepper(x_curr, params_stepper)
    assert x_next.shape == (2, 2)
    assert torch.isfinite(x_next).all()


def test_linear_sde_variants():
    """Verify LinearSDE variants ('periodic' with dim=2, 'general' with dim=8)."""
    # Periodic
    p_sde = SIMULATORS["LinearSDE"](variant="periodic", seed=42)
    assert p_sde.dim_parameters == 2
    assert p_sde.dim_data == 22
    th_p = p_sde.get_prior().sample((3,))
    x_p = p_sde.simulate(th_p)
    assert x_p.shape == (3, 22)

    # General
    g_sde = SIMULATORS["LinearSDE"](variant="general", dim_parameters=8, seed=42)
    assert g_sde.dim_parameters == 8
    assert g_sde.dim_data == 22
    th_g = g_sde.get_prior().sample((3,))
    x_g = g_sde.simulate(th_g)
    assert x_g.shape == (3, 22)


def test_sir_mode_and_prng():
    """Verify SIR ODE and SDE modes and PRNG key state advancement."""
    # SDE mode should advance PRNG key across consecutive simulate() calls
    sir_sde = SIMULATORS["SIR"](seed=42, mode="sde")
    assert sir_sde.mode == "sde"
    th = torch.tensor([0.33, 0.12])
    x1 = sir_sde.simulate(th)
    x2 = sir_sde.simulate(th)
    assert not torch.allclose(x1, x2), "Consecutive simulate calls in SDE mode must be stochastic"

    # reset_rng restores exact seed sequence
    sir_sde.reset_rng(42)
    x1_reset = sir_sde.simulate(th)
    assert torch.allclose(x1, x1_reset), "reset_rng must restore initial reproducible state"

    # ODE mode must be deterministic
    sir_ode = SIMULATORS["SIR"](seed=42, mode="ode")
    assert sir_ode.mode == "ode"
    xo1 = sir_ode.simulate(th)
    xo2 = sir_ode.simulate(th)
    assert torch.allclose(xo1, xo2), "ODE simulate calls must be deterministic"

    # Log likelihood and posterior samples in ODE mode
    ll_ode = sir_ode.log_likelihood(th, xo1)
    assert torch.isfinite(ll_ode).all()
    samples_ode = sir_ode.posterior_samples(xo1, num_samples=50, n_chains=2, burn_in=20)
    assert samples_ode.shape == (50, 2)
    assert torch.isfinite(samples_ode).all()

    # Stepper in ODE mode
    x_curr = torch.tensor([[4.0, 1.0, 0.0]])
    x_next_ode = sir_ode.stepper(x_curr, th)
    assert x_next_ode.shape == (1, 3)
    assert torch.isfinite(x_next_ode).all()


def test_lotka_volterra_mode_and_prng():
    """Verify LotkaVolterra ODE and SDE modes and PRNG key state advancement."""
    # SDE mode should advance PRNG key across consecutive simulate() calls
    lv_sde = SIMULATORS["LotkaVolterra"](seed=42, mode="sde")
    assert lv_sde.mode == "sde"
    th = torch.zeros(4)
    x1 = lv_sde.simulate(th)
    x2 = lv_sde.simulate(th)
    assert not torch.allclose(x1, x2), "Consecutive simulate calls in SDE mode must be stochastic"

    # reset_rng restores exact seed sequence
    lv_sde.reset_rng(42)
    x1_reset = lv_sde.simulate(th)
    assert torch.allclose(x1, x1_reset), "reset_rng must restore initial reproducible state"

    # ODE mode must be deterministic
    lv_ode = SIMULATORS["LotkaVolterra"](seed=42, mode="ode")
    assert lv_ode.mode == "ode"
    xo1 = lv_ode.simulate(th)
    xo2 = lv_ode.simulate(th)
    assert torch.allclose(xo1, xo2), "ODE simulate calls must be deterministic"

    # Log likelihood and posterior samples in ODE mode
    ll_ode = lv_ode.log_likelihood(th, xo1)
    assert torch.isfinite(ll_ode).all()
    samples_ode = lv_ode.posterior_samples(xo1, num_samples=50, n_chains=2, burn_in=20)
    assert samples_ode.shape == (50, 4)
    assert torch.isfinite(samples_ode).all()

    # Stepper in ODE mode
    x_curr = torch.tensor([[1.0, 0.5]])
    x_next_ode = lv_ode.stepper(x_curr, th)
    assert x_next_ode.shape == (1, 2)
    assert torch.isfinite(x_next_ode).all()


def test_lotka_volterra_presets_and_kwargs():
    """Verify LotkaVolterra presets ('markovsbi' and 'paper') and Simulator kwargs passing."""
    # 1. Custom kwargs in base Simulator class
    lv_custom = SIMULATORS["LotkaVolterra"](
        preset="sbibm", custom_tag="experiment_1", note="test_kwargs"
    )
    assert lv_custom.kwargs.get("custom_tag") == "experiment_1"
    assert lv_custom.kwargs.get("note") == "test_kwargs"

    # 2. Preset "markovsbi" (default)
    lv_m = SIMULATORS["LotkaVolterra"](preset="markovsbi")
    assert lv_m.preset == "markovsbi"
    assert lv_m.num_steps == 11
    assert lv_m.dim_data == 22
    assert lv_m.summary is None
    assert lv_m.param_transform == "sigmoid"
    assert torch.allclose(lv_m.u0, torch.tensor([1.0, 0.5]))
    assert lv_m.mode == "sde"

    # 3. Preset "sbibm" (Lueckmann et al., 2021, arXiv:2101.04653)
    lv_p = SIMULATORS["LotkaVolterra"](preset="sbibm")
    assert lv_p.preset == "sbibm"
    assert lv_p.num_steps == 10
    assert lv_p.dim_data == 20  # 10 time points x 2 species
    assert lv_p.summary == "lognormal"
    assert lv_p.param_transform == "identity"
    assert lv_p.param_order == "alphabetical"
    assert torch.allclose(lv_p.u0, torch.tensor([30.0, 1.0]))
    assert lv_p.mode == "ode"

    # Prior is LogNormal([-0.125, -3.0, -0.125, -3.0], 0.5)
    th = lv_p.get_prior().sample((10,))
    assert th.shape == (10, 4)
    assert (th > 0.0).all()

    # Simulate canonical theta*
    out_paper = lv_p.simulate(lv_p.canonical_theta)
    assert out_paper.shape == (20,)
    assert torch.isfinite(out_paper).all()
    assert (out_paper > 0.0).all()

    # Posterior samples with paper preset in ODE mode
    samples_paper = lv_p.posterior_samples(
        out_paper, num_samples=20, n_chains=2, burn_in=10, inner_thin=1
    )
    assert samples_paper.shape == (20, 4)
    assert torch.isfinite(samples_paper).all()
    assert (samples_paper > 0.0).all()

    # 4. Preset "papamakarios" (Papamakarios et al. 2019)
    lv_snl = SIMULATORS["LotkaVolterra"](preset="papamakarios")
    assert lv_snl.preset == "papamakarios"
    assert lv_snl.num_steps == 151
    assert lv_snl.dim_data == 9  # default summary="stats"
    assert lv_snl.summary == "stats"
    assert lv_snl.param_transform == "exp"
    assert lv_snl.param_order == "paper"
    assert torch.allclose(lv_snl.u0, torch.tensor([100.0, 50.0]))
    assert lv_snl.mode == "ode"

    # Prior is Uniform(-5, 2)^4
    th_snl = lv_snl.get_prior().sample((10,))
    assert th_snl.shape == (10, 4)
    assert (th_snl >= -5.0).all() and (th_snl <= 2.0).all()

    # Simulate canonical theta* (papamakarios ground truth)
    out_stats = lv_snl.simulate(lv_snl.canonical_theta)
    assert out_stats.shape == (9,)
    assert torch.isfinite(out_stats).all()

    # Papamakarios preset with raw trajectory (summary=None)
    lv_snl_raw = SIMULATORS["LotkaVolterra"](preset="papamakarios", summary=None)
    assert lv_snl_raw.dim_data == 302
    out_raw = lv_snl_raw.simulate(lv_snl.canonical_theta)
    assert out_raw.shape == (302,)
    assert torch.isfinite(out_raw).all()

    # Posterior samples with papamakarios preset in ODE mode
    x_obs = lv_snl_raw.simulate(lv_snl.canonical_theta)
    samples = lv_snl_raw.posterior_samples(
        x_obs, num_samples=20, n_chains=2, burn_in=10, inner_thin=1
    )
    assert samples.shape == (20, 4)
    assert torch.isfinite(samples).all()


def test_sir_and_gaussian_paper_presets():
    """Verify SIR, GaussianGaussian, and UniformGaussian 'paper' presets (arXiv:2101.04653)."""
    # 1. SIR sbibm preset
    sir_p = SIMULATORS["SIR"](preset="sbibm", seed=42)
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

    # 2. GaussianGaussian sbibm preset (gaussian_linear)
    gg_p = SIMULATORS["GaussianGaussian"](preset="sbibm", seed=42)
    assert gg_p.preset == "sbibm"
    assert gg_p.dim_parameters == 10
    assert torch.allclose(gg_p._diag_sigma(10), torch.full((10,), 0.1))
    samples_gg = gg_p.posterior_samples(torch.zeros(10), num_samples=50)
    assert samples_gg.shape == (50, 10)
    assert torch.isfinite(samples_gg).all()

    # 3. UniformGaussian sbibm preset (gaussian_linear_uniform)
    ug_p = SIMULATORS["UniformGaussian"](preset="sbibm", seed=42)
    assert ug_p.preset == "sbibm"
    assert ug_p.dim_parameters == 10
    assert ug_p.L == 1.0
    assert abs(ug_p.sigma**2 - 0.1) < 1e-6
    samples_ug = ug_p.posterior_samples(torch.zeros(10), num_samples=50)
    assert samples_ug.shape == (50, 10)
    assert (samples_ug >= -1.0).all() and (samples_ug <= 1.0).all()


