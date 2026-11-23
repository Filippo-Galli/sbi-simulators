"""Simulator for the SIR (Susceptible-Infected-Recovered) SDE/ODE model."""

import functools
import math
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pyro.distributions as pdist
import torch

from ..dynamical import DynamicalSimulator
from ..mcmc import (
    build_batched_ll,
    build_gibbs_pmmh_sampler,
    build_jax_trajectory_solver,
    build_mh_mcmc_sampler,
    build_ode_ll,
    build_particle_filter_ll,
)
from ..priors import (
    LogNormalPrior,
    NormalPrior,
    PriorSpec,
    UniformPrior,
    as_prior_spec,
)
from ..summary_stats import trajectory_stats_3d_sir


class _UnsetType:
    pass


_UNSET = _UnsetType()

_MARKOVSBI_DEFAULT_X_OBS: tuple[tuple[float, float, float], ...] = (
    (4.0000000, 1.0000000, 0.0000000),
    (3.7881465, 1.1751375, 0.0533900),
    (3.5549567, 1.3938990, 0.1153643),
    (3.2974174, 1.5862643, 0.1903032),
    (3.0364633, 1.7333066, 0.2691698),
    (2.7711196, 1.9520751, 0.3586609),
    (2.5065825, 2.1024864, 0.4649960),
    (2.2470863, 2.2430851, 0.5757274),
    (1.9985336, 2.4134316, 0.6894845),
    (1.7609777, 2.6007223, 0.8086434),
    (1.5459836, 2.6992354, 0.9370170),
)


@dataclass(frozen=True)
class SIRPreset:
    """Declarative configuration for an SIR benchmark variant."""

    dt: float
    inner_steps: int
    sigma1: float
    num_steps: int
    obs_sigma: float
    mode: str
    total_population: float
    u0: tuple[float, float, float]
    summary: str | None
    param_transform: str
    param_order: str
    prior: PriorSpec
    prior_x0: PriorSpec
    canonical_theta: tuple[float, float]
    default_step_size: float
    default_x_obs: tuple[tuple[float, float, float], ...] | None = None


SIR_PRESETS: dict[str, SIRPreset] = {
    "markovsbi": SIRPreset(
        dt=0.01,
        inner_steps=5,
        sigma1=0.1,
        num_steps=11,
        obs_sigma=0.01,
        mode="sde",
        total_population=1.0,
        u0=(4.0, 1.0, 0.0),
        summary=None,
        param_transform="sigmoid",
        param_order="alphabetical",
        prior=NormalPrior.isotropic(2),
        prior_x0=UniformPrior.box(3, 0.0, 5.0),
        canonical_theta=(0.0, 0.0),
        default_step_size=0.1,
        default_x_obs=_MARKOVSBI_DEFAULT_X_OBS,
    ),
    "sbibm": SIRPreset(
        dt=160.0 / 9.0 / 200.0,
        inner_steps=200,
        sigma1=0.0,
        num_steps=10,
        obs_sigma=0.01,
        mode="ode",
        total_population=1_000_000.0,
        u0=(999999.0, 1.0, 0.0),
        summary="binomial",
        param_transform="identity",
        param_order="paper",
        prior=LogNormalPrior(
            loc=(math.log(0.4), math.log(1.0 / 8.0)),
            scale=(0.5, 0.2),
        ),
        prior_x0=UniformPrior.box(3, 0.0, 5.0),
        canonical_theta=(0.4, 1.0 / 8.0),
        default_step_size=0.1,
    ),
}


def _build_sir_step_fn(
    dt: float = 0.01,
    inner_steps: int = 5,
    sigma1: float = 0.1,
    mode: str = "sde",
    total_population: float = 1.0,
    param_transform: str = "sigmoid",
    param_order: str = "alphabetical",
):
    """Build single observation step transition function in JAX for SIR."""

    def drift(x, gamma, beta):
        S, I = x[0], x[1]
        inf_rate = beta * S * I / total_population
        return jnp.stack([-inf_rate, inf_rate - gamma * I, gamma * I])

    def extract_rates(theta):
        rates = theta if param_transform == "identity" else jax.nn.sigmoid(theta) * 2.0
        if param_order == "paper":
            return rates[1], rates[0]  # (beta, gamma) -> (gamma, beta)
        return rates[0], rates[1]

    def integrate_step(curr_x, key, gamma, beta):
        curr_x = jnp.clip(curr_x, 0.0)
        if mode == "ode":
            k1 = drift(curr_x, gamma, beta)
            k2 = drift(jnp.clip(curr_x + 0.5 * dt * k1, 0.0), gamma, beta)
            k3 = drift(jnp.clip(curr_x + 0.5 * dt * k2, 0.0), gamma, beta)
            k4 = drift(jnp.clip(curr_x + dt * k3, 0.0), gamma, beta)
            curr_x = curr_x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            xdt = drift(curr_x, gamma, beta)
            d_wt = jax.random.normal(key, shape=(3,)) * jnp.sqrt(dt)
            diff_scale = jnp.array([0.01, curr_x[1] * sigma1, 0.01])
            curr_x = curr_x + xdt * dt + diff_scale * d_wt
        return jnp.clip(curr_x, 0.0)

    def step_one(x, key, theta):
        gamma, beta = extract_rates(theta)
        if mode == "ode":

            def inner_loop(_i, curr_x):
                return integrate_step(curr_x, key, gamma, beta)
        else:
            keys = jax.random.split(key, inner_steps)

            def inner_loop(i, curr_x):
                return integrate_step(curr_x, keys[i], gamma, beta)

        return jax.lax.fori_loop(0, inner_steps, inner_loop, x)

    return step_one


def _build_sir_binomial_ll(
    jax_solver_single,
    default_x0: jax.Array,
    total_population: float = 1_000_000.0,
    binomial_n: int = 1000,
):
    """Build a JIT-compiled Binomial observation log-likelihood for SIR ODE."""
    dummy_key = jax.random.PRNGKey(0)
    log_n_fact = jax.scipy.special.gammaln(binomial_n + 1.0)

    def binomial_ll(theta, x_obs):
        traj = jax_solver_single(dummy_key, theta, default_x0)  # (3, num_steps)
        p = jnp.clip(traj[1] / total_population, 1e-12, 1.0 - 1e-12)
        log_comb = (
            log_n_fact
            - jax.scipy.special.gammaln(x_obs + 1.0)
            - jax.scipy.special.gammaln(binomial_n - x_obs + 1.0)
        )
        ll = jnp.sum(
            log_comb + x_obs * jnp.log(p) + (binomial_n - x_obs) * jnp.log(1.0 - p)
        )
        return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)

    return jax.jit(binomial_ll)


@functools.lru_cache(maxsize=32)
def _get_sir_solver_bundle(
    dt: float,
    inner_steps: int,
    sigma1: float,
    num_steps: int,
    mode: str,
    total_population: float,
    param_transform: str,
    param_order: str,
    default_x0: tuple[float, float, float],
):
    """Build and cache SIR JAX step function and trajectory solvers."""
    step_fn = _build_sir_step_fn(
        dt=dt,
        inner_steps=inner_steps,
        sigma1=sigma1,
        mode=mode,
        total_population=total_population,
        param_transform=param_transform,
        param_order=param_order,
    )
    solve_batch, solve_single = build_jax_trajectory_solver(
        step_fn,
        num_steps=num_steps,
        default_x0=default_x0,
        clip_non_negative=True,
    )
    return step_fn, solve_batch, solve_single


@functools.lru_cache(maxsize=32)
def _get_sir_ode_ll_bundle(
    dt: float,
    inner_steps: int,
    sigma1: float,
    num_steps: int,
    total_population: float,
    param_transform: str,
    param_order: str,
    default_x0: tuple[float, float, float],
    obs_sigma: float,
    binomial_n: int,
    ll_kind: str,
):
    """Lazily build and cache a specific SIR ODE log-likelihood and batched vmap."""
    _, _, solve_single = _get_sir_solver_bundle(
        dt,
        inner_steps,
        sigma1,
        num_steps,
        "ode",
        total_population,
        param_transform,
        param_order,
        default_x0,
    )
    if ll_kind == "binomial":
        ll_fn = _build_sir_binomial_ll(
            solve_single,
            default_x0=jnp.asarray(default_x0, dtype=jnp.float32),
            total_population=total_population,
            binomial_n=binomial_n,
        )
    else:
        ll_fn = build_ode_ll(solve_single, obs_dim=3, obs_sigma=obs_sigma)
    return ll_fn, build_batched_ll(ll_fn, stochastic=False)


@functools.lru_cache(maxsize=32)
def _get_sir_mcmc_bundle(
    dt: float,
    inner_steps: int,
    sigma1: float,
    num_steps: int,
    obs_sigma: float,
    mode: str,
    total_population: float,
    binomial_n: int,
    param_transform: str,
    param_order: str,
    default_x0: tuple[float, float, float],
    ll_kind: str,
    prior_spec: PriorSpec,
    default_init_theta: tuple[float, ...],
):
    """Build and cache SIR JAX MCMC sampler and batched likelihood."""
    if mode == "ode":
        ll_fn, v_ll_fn = _get_sir_ode_ll_bundle(
            dt,
            inner_steps,
            sigma1,
            num_steps,
            total_population,
            param_transform,
            param_order,
            default_x0,
            obs_sigma,
            binomial_n,
            ll_kind,
        )
        jax_mcmc = build_mh_mcmc_sampler(
            ll_fn,
            prior=prior_spec,
            default_init_theta=default_init_theta,
        )
        return v_ll_fn, jax_mcmc

    step_fn, _, _ = _get_sir_solver_bundle(
        dt,
        inner_steps,
        sigma1,
        num_steps,
        mode,
        total_population,
        param_transform,
        param_order,
        default_x0,
    )
    pf_ll_fn = build_particle_filter_ll(step_fn, obs_dim=3, obs_sigma=obs_sigma)
    v_ll_fn = build_batched_ll(pf_ll_fn, stochastic=True)
    jax_mcmc = build_gibbs_pmmh_sampler(pf_ll_fn, prior=prior_spec)
    return v_ll_fn, jax_mcmc


class SIR(DynamicalSimulator):
    r"""SIR (Susceptible-Infected-Recovered) Epidemic Simulator (SDE and ODE modes).

    Presets may be passed as a registered name (``"markovsbi"``, ``"sbibm"``) or
    directly as an :class:`SIRPreset` data object.
    """

    def __init__(
        self,
        dim_parameters: int = 2,
        seed: int = 42,
        dt: float | None = None,
        inner_steps: int | None = None,
        sigma1: float | None = None,
        num_steps: int | None = None,
        obs_sigma: float | None = None,
        summary: str | None | _UnsetType = _UNSET,
        mode: str | None = None,
        preset: str | SIRPreset = "markovsbi",
        u0: torch.Tensor | None = None,
        prior: PriorSpec | pdist.Distribution | None = None,
        param_transform: str | None = None,
        param_order: str | None = None,
        total_population: float | None = None,
        binomial_n: int = 1000,
        **kwargs,
    ):
        if dim_parameters != 2:
            raise ValueError(
                f"SIR requires dim_parameters=2 (gamma, beta), got {dim_parameters}."
            )

        if isinstance(preset, SIRPreset):
            cfg = preset
            self.preset = "custom"
        else:
            preset_key = preset.lower()
            if preset_key not in SIR_PRESETS:
                raise ValueError(
                    f"Unknown preset '{preset}'. Supported presets: {', '.join(SIR_PRESETS)}."
                )
            cfg = SIR_PRESETS[preset_key]
            self.preset = preset_key
        self.preset_config = cfg

        self.dt = float(cfg.dt if dt is None else dt)
        self.inner_steps = int(cfg.inner_steps if inner_steps is None else inner_steps)
        self.sigma1 = float(cfg.sigma1 if sigma1 is None else sigma1)
        resolved_num_steps = int(cfg.num_steps if num_steps is None else num_steps)
        resolved_obs_sigma = float(cfg.obs_sigma if obs_sigma is None else obs_sigma)
        self.total_population = float(
            cfg.total_population if total_population is None else total_population
        )
        self.binomial_n = int(binomial_n)

        mode_str = (cfg.mode if mode is None else mode).lower()
        if mode_str not in ("sde", "ode"):
            raise ValueError(f"mode must be 'sde' or 'ode', got '{mode_str}'.")
        self.mode = mode_str

        self.param_transform = (
            cfg.param_transform
            if param_transform is None
            else param_transform.lower()
        )
        self.param_order = (
            cfg.param_order if param_order is None else param_order.lower()
        )

        resolved_u0 = (
            torch.tensor(cfg.u0, dtype=torch.float32)
            if u0 is None
            else torch.as_tensor(u0, dtype=torch.float32)
        )
        u0_tuple = (
            float(resolved_u0[0].item()),
            float(resolved_u0[1].item()),
            float(resolved_u0[2].item()),
        )

        resolved_summary = cfg.summary if isinstance(summary, _UnsetType) else summary
        dim_raw = int(3 * resolved_num_steps)
        dim_data = self._resolve_summary_dim(
            resolved_summary,
            dim_raw=dim_raw,
            dim_stats=15,
            dim_two_step=6,
            extra_modes={"binomial": resolved_num_steps},
        )

        super().__init__(
            seed=seed,
            dim_parameters=dim_parameters,
            dim_data=dim_data,
            state_dim=3,
            num_steps=resolved_num_steps,
            obs_sigma=resolved_obs_sigma,
            summary=resolved_summary,
            u0=resolved_u0,
            **kwargs,
        )

        prior_spec = cfg.prior if prior is None else as_prior_spec(prior)
        if prior_spec.dim != dim_parameters:
            raise ValueError(
                f"Expected prior of dimension {dim_parameters}, got {prior_spec.dim}."
            )
        self.prior_spec = prior_spec
        self.prior_dist = prior_spec.to_pyro()
        self._prior_x0_dist = cfg.prior_x0.to_pyro()
        self.default_step_size = float(cfg.default_step_size)
        self._default_x_obs = cfg.default_x_obs

        if prior is not None:
            self.canonical_theta = torch.tensor(
                prior_spec.canonical_point(), dtype=torch.float32
            )
        elif self.param_order != cfg.param_order:
            self.canonical_theta = torch.tensor(
                (cfg.canonical_theta[1], cfg.canonical_theta[0]), dtype=torch.float32
            )
        else:
            self.canonical_theta = torch.tensor(
                cfg.canonical_theta, dtype=torch.float32
            )

        self._u0_tuple = u0_tuple
        _, self._jax_solver_batch, _ = _get_sir_solver_bundle(
            self.dt,
            self.inner_steps,
            self.sigma1,
            self.num_steps,
            self.mode,
            self.total_population,
            self.param_transform,
            self.param_order,
            self._u0_tuple,
        )

        ll_kind = "binomial" if self.summary == "binomial" else "ode"
        self._default_ll_kind = ll_kind

        canon_tuple = tuple(float(v) for v in self.canonical_theta.tolist())
        self._v_ll_fn, self._jax_mcmc = _get_sir_mcmc_bundle(
            self.dt,
            self.inner_steps,
            self.sigma1,
            self.num_steps,
            self.obs_sigma,
            self.mode,
            self.total_population,
            self.binomial_n,
            self.param_transform,
            self.param_order,
            self._u0_tuple,
            ll_kind,
            self.prior_spec,
            canon_tuple,
        )

    def get_prior_x0(self) -> pdist.Distribution:
        """Prior over initial condition ``x0 = [S, I, R]``."""
        return self._prior_x0_dist

    @staticmethod
    @override
    def compute_summary_stats(us: torch.Tensor) -> torch.Tensor:
        """Compute 15 summary statistics from ``(num_samples, 3, T)`` trajectory tensor."""
        return trajectory_stats_3d_sir(us)

    def _extract_rates_torch(
        self, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rates = (
            params
            if self.param_transform == "identity"
            else torch.sigmoid(params) * 2.0
        )
        if self.param_order == "paper":
            return rates[:, 1], rates[:, 0]
        return rates[:, 0], rates[:, 1]

    def _drift_torch(
        self,
        u: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        S, I = u[:, 0], u[:, 1]
        inf_rate = beta * S * I / self.total_population
        return torch.stack([-inf_rate, inf_rate - gamma * I, gamma * I], dim=-1)

    def stepper(
        self,
        x: torch.Tensor,
        parameters: torch.Tensor,
        dt: float | None = None,
        sigma1: float | None = None,
    ) -> torch.Tensor:
        """Simulate a single observation-step transition for SIR."""
        x_t, params, is_1d = self._broadcast_stepper_inputs(x, parameters)
        sig1 = float(self.sigma1 if sigma1 is None else sigma1)
        gamma, beta = self._extract_rates_torch(params)

        step_dt = float(self.dt * self.inner_steps if dt is None else dt)
        n_substeps = max(round(step_dt / self.dt), 1)
        actual_h = step_dt / n_substeps
        sqrt_h = math.sqrt(actual_h)

        curr_u = torch.clamp(x_t.clone(), min=0.0)
        for _ in range(n_substeps):
            if self.mode == "ode":
                k1 = self._drift_torch(curr_u, gamma, beta)
                k2 = self._drift_torch(
                    torch.clamp(curr_u + 0.5 * actual_h * k1, min=0.0), gamma, beta
                )
                k3 = self._drift_torch(
                    torch.clamp(curr_u + 0.5 * actual_h * k2, min=0.0), gamma, beta
                )
                k4 = self._drift_torch(
                    torch.clamp(curr_u + actual_h * k3, min=0.0), gamma, beta
                )
                curr_u = torch.clamp(
                    curr_u + (actual_h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
                    min=0.0,
                )
            else:
                drift_step = self._drift_torch(curr_u, gamma, beta)
                d_w = torch.randn_like(curr_u) * sqrt_h
                diff_scale = torch.stack(
                    [
                        torch.full_like(curr_u[:, 0], 0.01),
                        curr_u[:, 1] * sig1,
                        torch.full_like(curr_u[:, 2], 0.01),
                    ],
                    dim=-1,
                )
                curr_u = torch.clamp(
                    curr_u + drift_step * actual_h + diff_scale * d_w, min=0.0
                )

        return curr_u.squeeze(0) if is_1d else curr_u

    @override
    def _apply_summary(self, us_valid: torch.Tensor) -> torch.Tensor:
        if self.summary == "binomial":
            i_traj = us_valid[:, 1, :]
            probs = (i_traj / self.total_population).clamp(0.0, 1.0)
            return torch.distributions.Binomial(
                total_count=self.binomial_n, probs=probs
            ).sample()
        return super()._apply_summary(us_valid)

    @override
    def get_prior(self) -> pdist.Distribution:
        return self.prior_dist

    def _prepare_ode_obs(self, x_obs: torch.Tensor | np.ndarray) -> tuple[jax.Array, str]:
        """Determine whether ``x_obs`` represents Binomial counts or full trajectory."""
        x_obs_t = torch.as_tensor(
            np.array(x_obs, copy=True) if isinstance(x_obs, np.ndarray) else x_obs,
            dtype=torch.float32,
        )
        if x_obs_t.ndim == 2 and x_obs_t.shape[0] == 1:
            x_obs_t = x_obs_t.squeeze(0)

        if (
            x_obs_t.numel() == self.num_steps
            and self.summary == "binomial"
            and self.mode == "ode"
        ):
            return jnp.asarray(x_obs_t.flatten().detach().cpu().numpy()), "binomial"

        return self._format_trajectory_obs(x_obs_t), "ode"

    def log_likelihood(
        self,
        parameters: torch.Tensor | np.ndarray,
        x_obs: torch.Tensor | np.ndarray,
        n_particles: int = 150,
    ) -> torch.Tensor:
        """Evaluate marginal log-likelihood ``log p(y_{1:T} | theta)``."""
        params, _ = self._prepare_parameters(parameters)
        params_jax = jnp.asarray(params.detach().cpu().numpy(), dtype=jnp.float32)

        if self.mode == "ode":
            x_jax, ll_kind = self._prepare_ode_obs(x_obs)
            if ll_kind == self._default_ll_kind:
                v_ll_fn = self._v_ll_fn
            else:
                _, v_ll_fn = _get_sir_ode_ll_bundle(
                    self.dt,
                    self.inner_steps,
                    self.sigma1,
                    self.num_steps,
                    self.total_population,
                    self.param_transform,
                    self.param_order,
                    self._u0_tuple,
                    self.obs_sigma,
                    self.binomial_n,
                    ll_kind,
                )
            lls_jax = v_ll_fn(params_jax, x_jax)
            return torch.from_numpy(np.array(lls_jax, copy=True))

        x_jax = self._format_trajectory_obs(x_obs)
        keys = jax.random.split(self._next_rng_key(), params.shape[0])
        lls_jax = self._v_ll_fn(params_jax, x_jax, keys, n_particles=n_particles)
        return torch.from_numpy(np.array(lls_jax, copy=True))

    @override
    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        n_chains: int = 20,
        burn_in: int = 200,
        inner_thin: int = 4,
        n_particles: int = 150,
        step_size: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from ``p(theta | y_{1:T})`` using JAX MCMC / Gibbs PMMH."""
        if x_obs is None:
            if self._default_x_obs is not None:
                x_obs = torch.tensor(self._default_x_obs, dtype=torch.float32)
            elif self.summary == "binomial":
                x_obs = self.simulate(self.canonical_theta)
            else:
                x_obs = self._solve(
                    self.canonical_theta.unsqueeze(0), u0=self.u0.unsqueeze(0)
                )[0].T

        if self.mode == "ode":
            x_jax, _ = self._prepare_ode_obs(x_obs)
        else:
            x_jax = self._format_trajectory_obs(x_obs)

        rng_key = self._next_rng_key()
        num_samples_per_chain = math.ceil(num_samples / n_chains)
        eff_step_size = (
            self.default_step_size if step_size is None else float(step_size)
        )

        if self.mode == "ode":
            samples_jax = self._jax_mcmc(
                x_jax,
                rng_key,
                n_chains=n_chains,
                num_samples_per_chain=num_samples_per_chain,
                burn_in=burn_in,
                inner_thin=inner_thin,
                step_size=eff_step_size,
            )
        else:
            samples_jax = self._jax_mcmc(
                x_jax,
                rng_key,
                n_chains=n_chains,
                num_samples_per_chain=num_samples_per_chain,
                burn_in=burn_in,
                inner_thin=inner_thin,
                n_particles=n_particles,
                step_size=eff_step_size,
            )
        return self._finalize_mcmc_samples(samples_jax, num_samples)
