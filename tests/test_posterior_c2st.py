"""Self-C2ST posterior inference tests across all simulators and MCMC modes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sbi_simulator import (
    SIMULATORS,
    SIR,
    SLCP,
    DoubleWell,
    LinearSDE,
    LotkaVolterra,
)


def compute_self_c2st(
    samples_a: torch.Tensor,
    samples_b: torch.Tensor,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """Compute 5-fold MLP C2ST accuracy matching ``sbi.utils.metrics.c2st``."""
    x_a = samples_a.detach().cpu().numpy()
    x_b = samples_b.detach().cpu().numpy()
    if x_a.ndim == 1:
        x_a = x_a[:, None]
    if x_b.ndim == 1:
        x_b = x_b[:, None]

    n = min(len(x_a), len(x_b))
    x = np.concatenate([x_a[:n], x_b[:n]], axis=0)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

    dim = x.shape[1]
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(10 * dim, 10 * dim),
            max_iter=200,
            random_state=seed,
        ),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, x, y, cv=cv, scoring="accuracy")
    return float(np.mean(scores))


def _assert_self_c2st_passes(
    factory,
    *,
    num_samples: int = 2000,
    max_c2st: float = 0.56,
    min_unique_ratio: float = 0.90,
    obs_seed: int = 42,
):
    sim_obs = factory(obs_seed)
    torch.manual_seed(obs_seed)
    theta_true = sim_obs.get_prior().sample().flatten()
    x_obs = sim_obs.simulate(theta_true)

    s1 = factory(101).posterior_samples(x_obs=x_obs, num_samples=num_samples)
    s2 = factory(202).posterior_samples(x_obs=x_obs, num_samples=num_samples)

    assert s1.shape == (num_samples, sim_obs.dim_parameters)
    assert s2.shape == (num_samples, sim_obs.dim_parameters)
    assert torch.isfinite(s1).all() and torch.isfinite(s2).all()

    u1 = torch.unique(s1, dim=0).shape[0] / num_samples
    u2 = torch.unique(s2, dim=0).shape[0] / num_samples
    assert u1 >= min_unique_ratio, f"Low unique ratio in run 1: {u1:.3f}"
    assert u2 >= min_unique_ratio, f"Low unique ratio in run 2: {u2:.3f}"

    score = compute_self_c2st(s1, s2)
    assert score <= max_c2st, (
        f"Self-C2ST too high ({score:.4f} > {max_c2st}): "
        f"uniq={u1:.2%}/{u2:.2%}, mean1={s1.mean(0).tolist()}, mean2={s2.mean(0).tolist()}"
    )


@pytest.mark.parametrize(
    "sim_name",
    [
        "AR1",
        "TwoMoons",
        "BimodalGaussians",
        "GaussianGaussian",
        "SpikeSlab",
        "UniformGaussian",
    ],
)
def test_analytic_simulators_self_c2st(sim_name: str):
    """All analytic posterior samplers must be self-consistent (self-C2ST ~ 0.50)."""
    _assert_self_c2st_passes(
        lambda s: SIMULATORS[sim_name](seed=s),
        num_samples=2000,
        max_c2st=0.56,
        min_unique_ratio=0.99,
    )


@pytest.mark.parametrize(
    "name,factory",
    [
        ("SLCP", lambda s: SLCP(seed=s)),
        ("LinearSDE_raw", lambda s: LinearSDE(seed=s)),
        ("LinearSDE_subsample", lambda s: LinearSDE(seed=s, summary="subsample")),
        ("SIR_sbibm_ode", lambda s: SIR(seed=s, preset="sbibm")),
        ("LV_sbibm_ode", lambda s: LotkaVolterra(seed=s, preset="sbibm")),
        ("LV_papamakarios_ode", lambda s: LotkaVolterra(seed=s, preset="papamakarios")),
        ("DoubleWell_raw", lambda s: DoubleWell(seed=s)),
    ],
)
def test_exact_mh_simulators_self_c2st(name: str, factory):
    """Exact-likelihood MH MCMC simulators must mix well and pass self-C2ST."""
    _assert_self_c2st_passes(
        factory,
        num_samples=2000,
        max_c2st=0.56,
        min_unique_ratio=0.90,
    )


@pytest.mark.parametrize(
    "name,factory",
    [
        ("SIR_markovsbi_sde", lambda s: SIR(seed=s, preset="markovsbi")),
        ("LV_markovsbi_sde", lambda s: LotkaVolterra(seed=s, preset="markovsbi")),
        ("DoubleWell_subsample", lambda s: DoubleWell(seed=s, summary="subsample")),
    ],
)
def test_pmmh_sde_simulators_self_c2st(name: str, factory):
    """PMMH SDE posterior samplers must mix well and pass self-C2ST."""
    _assert_self_c2st_passes(
        factory,
        num_samples=1500,
        max_c2st=0.56,
        min_unique_ratio=0.90,
    )
