"""Simulator for the Lotka-Volterra SDE/ODE model."""

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
from ..summary_stats import trajectory_stats_2d_lv


class _UnsetType:
    pass


_UNSET = _UnsetType()


@dataclass(frozen=True)
class _LVPreset:
    dt: float
    inner_steps: int
    sigma1: float
    sigma2: float
    num_steps: int
    obs_sigma: float
    mode: str
    u0: tuple[float, float]
    summary: str | None
    param_transform: str
    param_order: str
    prior_type: str
    prior_loc: tuple[float, ...] | None = None
    prior_scale: tuple[float, ...] | None = None
    prior_low: float | None = None
    prior_high: float | None = None


_LV_PRESETS: dict[str, _LVPreset] = {
    "markovsbi": _LVPreset(
        dt=0.02,
        inner_steps=5,
        sigma1=0.1,
        sigma2=0.1,
        num_steps=11,
        obs_sigma=0.05,
        mode="sde",
        u0=(1.0, 0.5),
        summary=None,
        param_transform="sigmoid",
        param_order="alphabetical",
        prior_type="gaussian",
    ),
    "sbibm": _LVPreset(
        dt=20.0 / 9.0 / 200.0,
        inner_steps=200,
        sigma1=0.0,
        sigma2=0.0,
        num_steps=10,
        obs_sigma=0.1,
        mode="ode",
        u0=(30.0, 1.0),
        summary="lognormal",
        param_transform="identity",
        param_order="alphabetical",
        prior_type="lognormal",
        prior_loc=(-0.125, -3.0, -0.125, -3.0),
        prior_scale=(0.5, 0.5, 0.5, 0.5),
    ),
    "papamakarios": _LVPreset(
        dt=0.01,
        inner_steps=20,
        sigma1=0.0,
        sigma2=0.0,
        num_steps=151,
        obs_sigma=0.05,
        mode="ode",
        u0=(100.0, 50.0),
        summary="stats",
        param_transform="exp",
        param_order="paper",
        prior_type="uniform",
        prior_low=-5.0,
        prior_high=2.0,
    ),
}


def _build_lv_step_fn(
    dt: float = 0.02,
    inner_steps: int = 5,
    sigma1: float = 0.1,
    sigma2: float = 0.1,
    mode: str = "sde",
    param_transform: str = "sigmoid",
    param_order: str = "alphabetical",
):
    """Build single observation step transition function in JAX for Lotka-Volterra."""

    def drift(x, alpha, beta, gamma, delta):
        dx0 = alpha * x[0] - beta * x[0] * x[1]
        dx1 = -gamma * x[1] + delta * x[0] * x[1]
        return jnp.stack([dx0, dx1])

    def integrate_step(x, key, alpha, beta, gamma, delta):
        x = jnp.clip(x, 0.0)
        if mode == "ode":
            k1 = drift(x, alpha, beta, gamma, delta)
            k2 = drift(jnp.clip(x + 0.5 * dt * k1, 0.0), alpha, beta, gamma, delta)
            k3 = drift(jnp.clip(x + 0.5 * dt * k2, 0.0), alpha, beta, gamma, delta)
            k4 = drift(jnp.clip(x + dt * k3, 0.0), alpha, beta, gamma, delta)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            xdt = drift(x, alpha, beta, gamma, delta)
            d_wt = jax.random.normal(key, shape=(2,)) * jnp.sqrt(dt)
            diffusion = jnp.array([sigma1 * x[0] * x[1], sigma2 * x[1] * x[0]])
            x = x + xdt * dt + diffusion * d_wt
        return jnp.clip(x, 0.0)

    def extract_rates(theta):
        if param_transform == "identity":
            rates = theta
        elif param_transform == "exp":
            rates = jnp.exp(theta)
        else:
            rates = jax.nn.sigmoid(theta) * 2.0 + 1.0
        if param_order == "paper":
            # theta = (delta, gamma, alpha, beta) -> (alpha, beta, gamma, delta)
            return rates[2], rates[3], rates[1], rates[0]
        return rates[0], rates[1], rates[2], rates[3]

    def step_one(x, key, theta):
        alpha, beta, gamma, delta = extract_rates(theta)
        if mode == "ode":

            def inner_loop(_i, curr_x):
                return integrate_step(curr_x, key, alpha, beta, gamma, delta)
        else:
            keys = jax.random.split(key, inner_steps)

            def inner_loop(i, curr_x):
                return integrate_step(curr_x, keys[i], alpha, beta, gamma, delta)

        return jax.lax.fori_loop(0, inner_steps, inner_loop, x)

    return step_one


def _build_lv_stats_ll(
    jax_solver_single,
    default_x0: jax.Array,
    scales: tuple[float, ...] = (
        1.25,
        2.5,
        0.125,
        0.125,
        0.0125,
        0.0125,
        0.0125,
        0.0125,
        0.025,
    ),
):
    """Build a JIT-compiled log-likelihood on 9 summary stats for Lotka-Volterra ODE."""
    dummy_key = jax.random.PRNGKey(0)
    scale_arr = jnp.asarray(scales, dtype=jnp.float32)

    def stats_from_traj(traj):
        x = traj[0]
        y = traj[1]
        eps = 1e-6
        mean_x = jnp.mean(x)
        mean_y = jnp.mean(y)
        dx = x - mean_x
        dy = y - mean_y
        var_x = jnp.mean(dx**2)
        var_y = jnp.mean(dy**2)
        denom_x = jnp.maximum(var_x, eps)
        denom_y = jnp.maximum(var_y, eps)
        log_var_x = jnp.log(denom_x)
        log_var_y = jnp.log(denom_y)
        ac1_x = jnp.mean(dx[1:] * dx[:-1]) / denom_x
        ac1_y = jnp.mean(dy[1:] * dy[:-1]) / denom_y
        ac2_x = jnp.mean(dx[2:] * dx[:-2]) / denom_x
        ac2_y = jnp.mean(dy[2:] * dy[:-2]) / denom_y
        denom_xy = jnp.sqrt(denom_x * denom_y)
        cc_xy = jnp.mean(dx * dy) / denom_xy
        return jnp.array(
            [
                mean_x,
                mean_y,
                log_var_x,
                log_var_y,
                jnp.clip(ac1_x, -1.0, 1.0),
                jnp.clip(ac2_x, -1.0, 1.0),
                jnp.clip(ac1_y, -1.0, 1.0),
                jnp.clip(ac2_y, -1.0, 1.0),
                jnp.clip(cc_xy, -1.0, 1.0),
            ]
        )

    def stats_ll(theta, s_obs):
        traj = jax_solver_single(dummy_key, theta, default_x0)
        s_pred = stats_from_traj(traj)
        diff = (s_pred - s_obs) / scale_arr
        ll = -0.5 * jnp.sum(diff**2)
        return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)

    return jax.jit(stats_ll)


def _build_lv_lognormal_ll(
    jax_solver_single,
    default_x0: jax.Array,
    obs_sigma: float = 0.1,
):
    """Build a JIT-compiled LogNormal observation log-likelihood for Lotka-Volterra ODE."""
    dummy_key = jax.random.PRNGKey(0)
    log_norm = jnp.log(obs_sigma * jnp.sqrt(2.0 * jnp.pi))

    def lognormal_ll(theta, x_obs):
        traj = jax_solver_single(dummy_key, theta, default_x0).T  # (num_steps, 2)
        valid = jnp.all(x_obs > 0.0) & jnp.all(traj > 0.0)
        log_y = jnp.log(jnp.maximum(x_obs, 1e-12))
        log_mu = jnp.log(jnp.maximum(traj, 1e-12))
        diff = (log_y - log_mu) / obs_sigma
        ll = jnp.sum(-0.5 * (diff**2) - log_y - log_norm)
        return jnp.where(valid & jnp.isfinite(ll), ll, -jnp.inf)

    return jax.jit(lognormal_ll)


@functools.lru_cache(maxsize=32)
def _get_lv_solver_bundle(
    dt: float,
    inner_steps: int,
    sigma1: float,
    sigma2: float,
    num_steps: int,
    mode: str,
    param_transform: str,
    param_order: str,
    default_x0: tuple[float, float],
):
    """Build and cache Lotka-Volterra JAX step function and trajectory solvers."""
    step_fn = _build_lv_step_fn(
        dt=dt,
        inner_steps=inner_steps,
        sigma1=sigma1,
        sigma2=sigma2,
        mode=mode,
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
def _get_lv_ode_ll_bundle(
    dt: float,
    inner_steps: int,
    sigma1: float,
    sigma2: float,
    num_steps: int,
    param_transform: str,
    param_order: str,
    default_x0: tuple[float, float],
    obs_sigma: float,
    ll_kind: str,
):
    """Lazily build and cache a specific Lotka-Volterra ODE log-likelihood and batched vmap."""
    _, _, solve_single = _get_lv_solver_bundle(
        dt,
        inner_steps,
        sigma1,
        sigma2,
        num_steps,
        "ode",
        param_transform,
        param_order,
        default_x0,
    )
    base_u0 = jnp.asarray(default_x0, dtype=jnp.float32)
    if ll_kind == "stats":
        ll_fn = _build_lv_stats_ll(solve_single, default_x0=base_u0)
    elif ll_kind == "lognormal":
        ll_fn = _build_lv_lognormal_ll(
            solve_single, default_x0=base_u0, obs_sigma=obs_sigma
        )
    else:
        ll_fn = build_ode_ll(solve_single, obs_dim=2, obs_sigma=obs_sigma)
    return ll_fn, build_batched_ll(ll_fn, stochastic=False)


@functools.lru_cache(maxsize=32)
def _get_lv_mcmc_bundle(
    dt: float,
    inner_steps: int,
    sigma1: float,
    sigma2: float,
    num_steps: int,
    obs_sigma: float,
    mode: str,
    param_transform: str,
    param_order: str,
    default_x0: tuple[float, float],
    ll_kind: str,
    prior_type: str,
    prior_low: float,
    prior_high: float,
    prior_loc: tuple[float, ...] | None,
    prior_scale: tuple[float, ...] | None,
    default_init_theta: tuple[float, ...],
):
    """Build and cache Lotka-Volterra JAX MCMC sampler and batched likelihood."""
    if mode == "ode":
        ll_fn, v_ll_fn = _get_lv_ode_ll_bundle(
            dt,
            inner_steps,
            sigma1,
            sigma2,
            num_steps,
            param_transform,
            param_order,
            default_x0,
            obs_sigma,
            ll_kind,
        )
        jax_mcmc = build_mh_mcmc_sampler(
            ll_fn,
            dim_parameters=4,
            prior_type=prior_type,
            prior_low=prior_low,
            prior_high=prior_high,
            prior_loc=prior_loc,
            prior_scale=prior_scale,
            default_init_theta=default_init_theta,
        )
        return v_ll_fn, jax_mcmc

    step_fn, _, _ = _get_lv_solver_bundle(
        dt,
        inner_steps,
        sigma1,
        sigma2,
        num_steps,
        mode,
        param_transform,
        param_order,
        default_x0,
    )
    pf_ll_fn = build_particle_filter_ll(step_fn, obs_dim=2, obs_sigma=obs_sigma)
    v_ll_fn = build_batched_ll(pf_ll_fn, stochastic=True)
    jax_mcmc = build_gibbs_pmmh_sampler(pf_ll_fn, dim_parameters=4)
    return v_ll_fn, jax_mcmc


class LotkaVolterra(DynamicalSimulator):
    r"""Lotka-Volterra Predator-Prey Simulator (SDE and ODE modes, with presets).

    Presets:
    --------
    - ``"markovsbi"`` (default): SDE mode with shifted-sigmoid rates in ``(1, 3)`` and ``N(0, I_4)`` prior.
    - ``"sbibm"``: ODE mode matching Lueckmann et al. (2021) with LogNormal prior and ``"lognormal"`` obs.
    - ``"papamakarios"`` (alias ``"snl"``): ODE mode matching Papamakarios et al. (2019) with Uniform(-5, 2)^4 prior and 9 summary stats.
    """

    def __init__(
        self,
        dim_parameters: int = 4,
        seed: int = 42,
        dt: float | None = None,
        inner_steps: int | None = None,
        sigma1: float | None = None,
        sigma2: float | None = None,
        num_steps: int | None = None,
        obs_sigma: float | None = None,
        summary: str | None | _UnsetType = _UNSET,
        mode: str | None = None,
        preset: str = "markovsbi",
        u0: torch.Tensor | None = None,
        prior: pdist.Distribution | None = None,
        param_transform: str | None = None,
        param_order: str | None = None,
        **kwargs,
    ):
        if dim_parameters != 4:
            raise ValueError(
                f"LotkaVolterra requires dim_parameters=4 (alpha, beta, "
                f"gamma, delta), got {dim_parameters}."
            )

        preset_key = preset.lower()
        if preset_key == "snl":
            preset_key = "papamakarios"
        if preset_key not in _LV_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Supported presets are 'markovsbi', 'sbibm', and 'papamakarios'."
            )
        cfg = _LV_PRESETS[preset_key]
        self.preset = preset_key

        self.dt = float(cfg.dt if dt is None else dt)
        self.inner_steps = int(cfg.inner_steps if inner_steps is None else inner_steps)
        self.sigma1 = float(cfg.sigma1 if sigma1 is None else sigma1)
        self.sigma2 = float(cfg.sigma2 if sigma2 is None else sigma2)
        resolved_num_steps = int(cfg.num_steps if num_steps is None else num_steps)
        resolved_obs_sigma = float(cfg.obs_sigma if obs_sigma is None else obs_sigma)

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
        u0_tuple = (float(resolved_u0[0].item()), float(resolved_u0[1].item()))

        resolved_summary = cfg.summary if isinstance(summary, _UnsetType) else summary
        dim_raw = int(2 * resolved_num_steps)
        dim_data = self._resolve_summary_dim(
            resolved_summary,
            dim_raw=dim_raw,
            dim_stats=9,
            dim_two_step=4,
            extra_modes={"lognormal": dim_raw},
        )

        super().__init__(
            seed=seed,
            dim_parameters=dim_parameters,
            dim_data=dim_data,
            state_dim=2,
            num_steps=resolved_num_steps,
            obs_sigma=resolved_obs_sigma,
            summary=resolved_summary,
            u0=resolved_u0,
            **kwargs,
        )

        if prior is not None:
            self.prior_dist = prior
            self.canonical_theta = torch.zeros(4, dtype=torch.float32)
        elif self.preset == "sbibm":
            assert cfg.prior_loc is not None and cfg.prior_scale is not None
            self.prior_dist = pdist.LogNormal(
                loc=torch.tensor(cfg.prior_loc, dtype=torch.float32),
                scale=torch.tensor(cfg.prior_scale, dtype=torch.float32),
            ).to_event(1)
            self.canonical_theta = torch.exp(
                torch.tensor(cfg.prior_loc, dtype=torch.float32)
            )
        elif self.preset == "papamakarios":
            self.prior_dist = pdist.Uniform(
                low=torch.full((4,), -5.0, dtype=torch.float32),
                high=torch.full((4,), 2.0, dtype=torch.float32),
            ).to_event(1)
            if self.param_order == "paper":
                self.canonical_theta = torch.tensor(
                    [-4.605170, -0.693147, 0.0, -4.605170], dtype=torch.float32
                )
            else:
                self.canonical_theta = torch.tensor(
                    [0.0, -4.605170, -0.693147, -4.605170], dtype=torch.float32
                )
        else:
            self.prior_dist = pdist.Normal(
                loc=torch.zeros(4, dtype=torch.float32),
                scale=torch.ones(4, dtype=torch.float32),
            ).to_event(1)
            self.canonical_theta = torch.zeros(4, dtype=torch.float32)

        self._u0_tuple = u0_tuple
        _, self._jax_solver_batch, _ = _get_lv_solver_bundle(
            self.dt,
            self.inner_steps,
            self.sigma1,
            self.sigma2,
            self.num_steps,
            self.mode,
            self.param_transform,
            self.param_order,
            self._u0_tuple,
        )

        if self.summary in ("stats", "summary_stats"):
            ll_kind = "stats"
        elif self.summary == "lognormal":
            ll_kind = "lognormal"
        else:
            ll_kind = "ode"
        self._default_ll_kind = ll_kind

        canon_tuple = tuple(float(v) for v in self.canonical_theta.tolist())
        self._v_ll_fn, self._jax_mcmc = _get_lv_mcmc_bundle(
            self.dt,
            self.inner_steps,
            self.sigma1,
            self.sigma2,
            self.num_steps,
            self.obs_sigma,
            self.mode,
            self.param_transform,
            self.param_order,
            self._u0_tuple,
            ll_kind,
            cfg.prior_type,
            cfg.prior_low if cfg.prior_low is not None else -5.0,
            cfg.prior_high if cfg.prior_high is not None else 2.0,
            cfg.prior_loc,
            cfg.prior_scale,
            canon_tuple,
        )

    def get_prior_x0(self) -> pdist.Distribution:
        """Prior over initial condition ``x0 = [prey, predator]``."""
        high = 200.0 if self.preset in ("sbibm", "papamakarios") else 10.0
        return pdist.Uniform(
            low=torch.zeros(2, dtype=torch.float32),
            high=torch.full((2,), high, dtype=torch.float32),
        ).to_event(1)

    @staticmethod
    @override
    def compute_summary_stats(us: torch.Tensor) -> torch.Tensor:
        """Compute 9 summary statistics from ``(num_samples, 2, T)`` trajectory tensor."""
        return trajectory_stats_2d_lv(us)

    def _extract_rates_torch(
        self, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.param_transform == "identity":
            rates = params
        elif self.param_transform == "exp":
            rates = torch.exp(params)
        else:
            rates = torch.sigmoid(params) * 2.0 + 1.0
        if self.param_order == "paper":
            return rates[:, 2], rates[:, 3], rates[:, 1], rates[:, 0]
        return rates[:, 0], rates[:, 1], rates[:, 2], rates[:, 3]

    @staticmethod
    def _drift_torch(
        u: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        p0, p1 = u[:, 0], u[:, 1]
        dx0 = alpha * p0 - beta * p0 * p1
        dx1 = -gamma * p1 + delta * p0 * p1
        return torch.stack([dx0, dx1], dim=-1)

    def stepper(
        self,
        x: torch.Tensor,
        parameters: torch.Tensor,
        dt: float | None = None,
        sigma: float | None = None,
    ) -> torch.Tensor:
        """Simulate a single observation-step transition for Lotka-Volterra."""
        x_t, params, is_1d = self._broadcast_stepper_inputs(x, parameters)
        sig1 = float(self.sigma1 if sigma is None else sigma)
        sig2 = float(self.sigma2 if sigma is None else sigma)
        alpha, beta, gamma, delta = self._extract_rates_torch(params)

        step_dt = float(self.dt * self.inner_steps if dt is None else dt)
        n_substeps = max(round(step_dt / self.dt), 1)
        actual_h = step_dt / n_substeps
        sqrt_h = math.sqrt(actual_h)

        curr_u = torch.clamp(x_t.clone(), min=0.0)
        for _ in range(n_substeps):
            if self.mode == "ode":
                k1 = self._drift_torch(curr_u, alpha, beta, gamma, delta)
                k2 = self._drift_torch(
                    torch.clamp(curr_u + 0.5 * actual_h * k1, min=0.0),
                    alpha,
                    beta,
                    gamma,
                    delta,
                )
                k3 = self._drift_torch(
                    torch.clamp(curr_u + 0.5 * actual_h * k2, min=0.0),
                    alpha,
                    beta,
                    gamma,
                    delta,
                )
                k4 = self._drift_torch(
                    torch.clamp(curr_u + actual_h * k3, min=0.0),
                    alpha,
                    beta,
                    gamma,
                    delta,
                )
                curr_u = torch.clamp(
                    curr_u + (actual_h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
                    min=0.0,
                )
            else:
                p0, p1 = curr_u[:, 0], curr_u[:, 1]
                drift_step = self._drift_torch(curr_u, alpha, beta, gamma, delta)
                d_w = torch.randn_like(curr_u) * sqrt_h
                diff_step = torch.stack(
                    [sig1 * p0 * p1 * d_w[:, 0], sig2 * p1 * p0 * d_w[:, 1]], dim=-1
                )
                curr_u = torch.clamp(
                    curr_u + drift_step * actual_h + diff_step, min=0.0
                )

        return curr_u.squeeze(0) if is_1d else curr_u

    @override
    def _apply_summary(self, us_valid: torch.Tensor) -> torch.Tensor:
        if self.summary == "lognormal":
            raw_data = us_valid.reshape(us_valid.shape[0], -1)
            noise = torch.exp(torch.randn_like(raw_data) * self.obs_sigma)
            return raw_data * noise
        return super()._apply_summary(us_valid)

    @override
    def get_prior(self) -> pdist.Distribution:
        return self.prior_dist

    def _prepare_ode_obs(self, x_obs: torch.Tensor | np.ndarray) -> tuple[jax.Array, str]:
        """Determine whether ``x_obs`` represents summary stats or trajectory and convert."""
        x_obs_t = torch.as_tensor(
            np.array(x_obs, copy=True) if isinstance(x_obs, np.ndarray) else x_obs,
            dtype=torch.float32,
        )
        if x_obs_t.ndim == 2 and x_obs_t.shape[0] == 1:
            x_obs_t = x_obs_t.squeeze(0)

        if (
            x_obs_t.numel() == 9
            and self.summary in ("stats", "summary_stats")
            and self.mode == "ode"
        ):
            return jnp.asarray(x_obs_t.flatten().detach().cpu().numpy()), "stats"

        ll_kind = "lognormal" if self.summary == "lognormal" else "ode"
        return self._format_trajectory_obs(x_obs_t), ll_kind

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
                _, v_ll_fn = _get_lv_ode_ll_bundle(
                    self.dt,
                    self.inner_steps,
                    self.sigma1,
                    self.sigma2,
                    self.num_steps,
                    self.param_transform,
                    self.param_order,
                    self._u0_tuple,
                    self.obs_sigma,
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
        step_size: float = 0.3,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from ``p(theta | y_{1:T})`` using JAX MCMC / Gibbs PMMH."""
        is_default_obs = x_obs is None
        if x_obs is None:
            if self.preset in ("sbibm", "papamakarios"):
                th_star = self.canonical_theta
                if self.summary in ("stats", "summary_stats", "lognormal"):
                    x_obs = self.simulate(th_star)
                else:
                    x_obs = self._solve(
                        th_star.unsqueeze(0), u0=self.u0.unsqueeze(0)
                    )[0].T
            else:
                # Default observation generated from theta = 0 with seed 42 (markovsbi)
                x_obs = torch.tensor(
                    [
                        [1.0000, 0.5000],
                        [1.0762, 0.5163],
                        [1.1869, 0.5604],
                        [1.2694, 0.5928],
                        [1.4117, 0.6073],
                        [1.5249, 0.6987],
                        [1.6904, 0.7705],
                        [1.7563, 0.8746],
                        [1.7599, 1.0564],
                        [1.7599, 1.3159],
                        [1.7551, 1.5372],
                    ],
                    dtype=torch.float32,
                )

        if self.mode == "ode":
            x_jax, _ = self._prepare_ode_obs(x_obs)
        else:
            x_jax = self._format_trajectory_obs(x_obs)

        rng_key = (
            jax.random.PRNGKey(self.seed)
            if (self.preset == "markovsbi" and is_default_obs)
            else self._next_rng_key()
        )
        num_samples_per_chain = math.ceil(num_samples / n_chains)

        if self.preset == "papamakarios" and step_size == 0.3:
            eff_step_size = 0.015
        elif self.preset == "sbibm" and step_size == 0.3:
            eff_step_size = 0.05
        else:
            eff_step_size = step_size

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
