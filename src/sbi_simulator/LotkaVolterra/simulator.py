"""Simulator for the Lotka-Volterra SDE model."""

import math
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pyro
import pyro.distributions as pdist
import torch

from ..base import Simulator
from ..compositional import sample_two_step
from ..mcmc import (
    build_gibbs_pmmh_sampler,
    build_mh_mcmc_sampler,
    build_ode_ll,
    build_particle_filter_ll,
)


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
            dWt = jax.random.normal(key, shape=(2,)) * jnp.sqrt(dt)
            B = jnp.array([sigma1 * x[0] * x[1], sigma2 * x[1] * x[0]])
            x = x + xdt * dt + B * dWt
        return jnp.clip(x, 0.0)

    def extract_rates(theta):
        if param_transform == "identity":
            if param_order == "paper":
                delta, gamma, alpha, beta = theta[0], theta[1], theta[2], theta[3]
            else:
                alpha, beta, gamma, delta = theta[0], theta[1], theta[2], theta[3]
        elif param_transform == "exp":
            if param_order == "paper":
                # theta = (delta, gamma, alpha, beta) as in Papamakarios et al. (2019)
                delta = jnp.exp(theta[0])
                gamma = jnp.exp(theta[1])
                alpha = jnp.exp(theta[2])
                beta = jnp.exp(theta[3])
            else:
                # alphabetical: (alpha, beta, gamma, delta)
                alpha = jnp.exp(theta[0])
                beta = jnp.exp(theta[1])
                gamma = jnp.exp(theta[2])
                delta = jnp.exp(theta[3])
        else:
            # shifted sigmoid into (1.0, 3.0)
            if param_order == "paper":
                delta = jax.nn.sigmoid(theta[0]) * 2.0 + 1.0
                gamma = jax.nn.sigmoid(theta[1]) * 2.0 + 1.0
                alpha = jax.nn.sigmoid(theta[2]) * 2.0 + 1.0
                beta = jax.nn.sigmoid(theta[3]) * 2.0 + 1.0
            else:
                alpha = jax.nn.sigmoid(theta[0]) * 2.0 + 1.0
                beta = jax.nn.sigmoid(theta[1]) * 2.0 + 1.0
                gamma = jax.nn.sigmoid(theta[2]) * 2.0 + 1.0
                delta = jax.nn.sigmoid(theta[3]) * 2.0 + 1.0
        return alpha, beta, gamma, delta

    def step_one(x, key, theta):
        alpha, beta, gamma, delta = extract_rates(theta)

        keys = jax.random.split(key, inner_steps)

        def inner_loop(i, curr_x):
            return integrate_step(curr_x, keys[i], alpha, beta, gamma, delta)

        return jax.lax.fori_loop(0, inner_steps, inner_loop, x)

    return step_one


def _build_lv_sde_jax_solver(
    step_one_fn, num_steps: int = 11, default_x0: jnp.ndarray | None = None
):
    """Build a JIT-compiled batched Euler-Maruyama SDE solver in JAX for Lotka-Volterra."""
    base_x0 = (
        jnp.array([1.0, 0.5], dtype=jnp.float32)
        if default_x0 is None
        else jnp.asarray(default_x0, dtype=jnp.float32)
    )

    def solve_single(rng_key, theta, x0=None):
        u0 = base_x0 if x0 is None else jnp.clip(x0, 0.0)

        def one_step(x, key):
            x_next = step_one_fn(x, key, theta)
            return x_next, x_next

        keys = jax.random.split(rng_key, num_steps - 1)
        _, xs = jax.lax.scan(one_step, u0, keys)
        full_traj = jnp.concatenate([u0[None, :], xs], axis=0)  # (num_steps, 2)
        return full_traj.T  # (2, num_steps)

    solve_batch_fn = jax.vmap(solve_single, in_axes=(0, 0, 0))

    def solve_batch(rng_keys, thetas, x0s=None):
        if x0s is None:
            x0s = jnp.broadcast_to(base_x0, (thetas.shape[0], 2))
        return solve_batch_fn(rng_keys, thetas, x0s)

    return jax.jit(solve_batch), jax.jit(solve_single)


def _build_lv_stats_ll(
    jax_solver_single,
    default_x0: jnp.ndarray,
    scales: jnp.ndarray | None = None,
):
    """Build a JIT-compiled log-likelihood function on 9 summary statistics for Lotka-Volterra ODE."""
    dummy_key = jax.random.PRNGKey(0)
    if scales is None:
        scales = jnp.array(
            [1.25, 2.5, 0.125, 0.125, 0.0125, 0.0125, 0.0125, 0.0125, 0.025],
            dtype=jnp.float32,
        )

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
        log_var_x = jnp.log(jnp.maximum(var_x, eps))
        log_var_y = jnp.log(jnp.maximum(var_y, eps))
        denom_x = jnp.maximum(var_x, eps)
        denom_y = jnp.maximum(var_y, eps)
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
        diff = (s_pred - s_obs) / scales
        ll = -0.5 * jnp.sum(diff**2)
        has_nan = jnp.isnan(ll)
        return jnp.where(has_nan, -1e9, ll)

    return jax.jit(stats_ll)


def _build_lv_lognormal_ll(
    jax_solver_single,
    default_x0: jnp.ndarray,
    obs_sigma: float = 0.1,
):
    """Build a JIT-compiled LogNormal observation log-likelihood for Lotka-Volterra ODE."""
    dummy_key = jax.random.PRNGKey(0)
    log_norm = jnp.log(obs_sigma * jnp.sqrt(2.0 * jnp.pi))

    def lognormal_ll(theta, x_obs):
        # x_obs: (num_steps, 2)
        traj = jax_solver_single(dummy_key, theta, default_x0).T  # (num_steps, 2)
        valid = jnp.all(x_obs > 0.0) & jnp.all(traj > 0.0)
        log_y = jnp.log(jnp.maximum(x_obs, 1e-12))
        log_mu = jnp.log(jnp.maximum(traj, 1e-12))
        diff = (log_y - log_mu) / obs_sigma
        ll = jnp.sum(-0.5 * (diff**2) - log_y - log_norm)
        has_nan = jnp.isnan(ll)
        return jnp.where(valid & (~has_nan), ll, -1e9)

    return jax.jit(lognormal_ll)


_SENTINEL = object()


class LotkaVolterra(Simulator):
    r"""Lotka-Volterra Predator-Prey Simulator (SDE and ODE modes, with presets).

    Presets:
    --------
    - "markovsbi" (default):
      Configuration matching markovsbi task:
      * Initial state: $u_0 = [1.0, 0.5]^T$ (normalized population densities)
      * Parameter transform: shifted sigmoid rates in $(1.0, 3.0)$ via $\text{rate} = 2 \cdot \sigma(\theta) + 1$
      * Parameter ordering: $(\alpha, \beta, \gamma, \delta)$
      * Prior: Standard Normal $\boldsymbol{\theta} \sim \mathcal{N}(0, I_4)$
      * Time discretization: $dt = 0.02, \text{inner\_steps} = 5 \implies \Delta t_{\text{obs}} = 0.10$
      * Trajectory length: $T = 1.0$ with `num_steps = 11`
      * Default mode: `"sde"` with diffusion noise $\sigma_1 = \sigma_2 = 0.1$
      * Default summary: `None` (raw flattened trajectory of shape $(22,)$)

    - "sbibm" (Lueckmann et al., 2021, arXiv:2101.04653, Task 'lotka_volterra'):
      * Initial state: $u_0 = [30.0, 1.0]^T$ (Prey $X(0) = 30$, Predator $Y(0) = 1$)
      * Parameter transform: identity (direct positive rates $(\alpha, \beta, \gamma, \delta)$)
      * Prior: $\alpha \sim \text{LogNormal}(-0.125, 0.5)$, $\beta \sim \text{LogNormal}(-3.0, 0.5)$,
        $\gamma \sim \text{LogNormal}(-0.125, 0.5)$, $\delta \sim \text{LogNormal}(-3.0, 0.5)$
      * Time discretization: $T = 20.0$ evaluated at 10 evenly-spaced points (`num_steps = 10`, $\Delta t_{\text{obs}} = 20/9$)
      * Default mode: `"ode"` (deterministic dynamical system, RK4 integration)
      * Default summary: `"lognormal"` ($y_{k,i} \sim \text{LogNormal}(\log X_k(t_i), 0.1)$, shape $(20,)$)

    - "papamakarios" (alias: "snl"):
      Configuration matching Papamakarios et al. (2019) (arXiv:1805.07226, Appendix A.3 & B.3):
      * Initial state: $u_0 = [100.0, 50.0]^T$ (Prey $Y_0 = 100$, Predator $X_0 = 50$)
      * Parameter transform: exponential log-rates $\text{rate} = \exp(\theta)$
      * Parameter ordering: $\boldsymbol{\theta} = (\delta, \gamma, \alpha, \beta)$ (predator birth, death, prey birth, predation)
      * Prior: Uniform broad prior $\boldsymbol{\theta} \sim \mathcal{U}(-5, 2)^4$
      * Time discretization: $dt = 0.01, \text{inner\_steps} = 20 \implies \Delta t_{\text{obs}} = 0.20$
      * Trajectory length: $T = 30.0$ with `num_steps = 151`
      * Default mode: `"ode"` (deterministic dynamical system)
      * Default summary: `"stats"` (9 summary statistics)
      * Ground truth: $\boldsymbol{\theta}^* = (\log 0.01, \log 0.5, \log 1, \log 0.01) \approx (-4.605, -0.693, 0.0, -4.605)$

    Mathematical Model:
    -------------------
    The state is a 2D vector $x(t) = [x_0(t), x_1(t)]^T$ representing the prey ($x_0$)
    and predator ($x_1$) populations over continuous time $t \ge 0$.

    - SDE Mode (`mode="sde"`):
      Stochastic differential equation with state-dependent multiplicative diffusion:
          $$dx_0 = (\alpha x_0 - \beta x_0 x_1) \, dt + \sigma_1 x_0 x_1 \, dW_0$$
          $$dx_1 = (-\gamma x_1 + \delta x_0 x_1) \, dt + \sigma_2 x_1 x_0 \, dW_1$$
    - ODE Mode (`mode="ode"`):
      Deterministic dynamical system:
          $$\frac{dx_0}{dt} = \alpha x_0 - \beta x_0 x_1$$
          $$\frac{dx_1}{dt} = -\gamma x_1 + \delta x_0 x_1$$
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
        summary: str | None = _SENTINEL,
        mode: str | None = None,
        preset: str = "markovsbi",
        u0: torch.Tensor | None = None,
        prior: pdist.Distribution | None = None,
        param_transform: str | None = None,
        param_order: str | None = None,
        **kwargs,
    ):
        self.dim_parameters = dim_parameters
        if dim_parameters != 4:
            raise ValueError(
                f"LotkaVolterra requires dim_parameters=4 (alpha, beta, "
                f"gamma, delta), got {dim_parameters}."
            )

        preset_key = preset.lower()
        if preset_key == "sbibm":
            self.preset = "sbibm"
            default_dt = 20.0 / 9.0 / 200.0
            default_inner_steps = 200
            default_sigma1 = 0.0
            default_sigma2 = 0.0
            default_num_steps = 10
            default_obs_sigma = 0.1
            default_mode = "ode"
            default_u0 = torch.tensor([30.0, 1.0], dtype=torch.float32)
            default_summary = "lognormal"
            default_transform = "identity"
            default_order = "alphabetical"
        elif preset_key in ("papamakarios", "snl"):
            self.preset = "papamakarios"
            default_dt = 0.01
            default_inner_steps = 20
            default_sigma1 = 0.0
            default_sigma2 = 0.0
            default_num_steps = 151
            default_obs_sigma = 0.05
            default_mode = "ode"
            default_u0 = torch.tensor([100.0, 50.0], dtype=torch.float32)
            default_summary = "stats"
            default_transform = "exp"
            default_order = "paper"
        elif preset_key == "markovsbi":
            self.preset = "markovsbi"
            default_dt = 0.02
            default_inner_steps = 5
            default_sigma1 = 0.1
            default_sigma2 = 0.1
            default_num_steps = 11
            default_obs_sigma = 0.05
            default_mode = "sde"
            default_u0 = torch.tensor([1.0, 0.5], dtype=torch.float32)
            default_summary = None
            default_transform = "sigmoid"
            default_order = "alphabetical"
        else:
            raise ValueError(
                f"Unknown preset '{preset}'. Supported presets are 'markovsbi', 'sbibm', and 'papamakarios'."
            )

        self.dt = default_dt if dt is None else dt
        self.inner_steps = default_inner_steps if inner_steps is None else inner_steps
        self.sigma1 = default_sigma1 if sigma1 is None else sigma1
        self.sigma2 = default_sigma2 if sigma2 is None else sigma2
        self.num_steps = default_num_steps if num_steps is None else num_steps
        self.obs_sigma = default_obs_sigma if obs_sigma is None else obs_sigma

        mode_str = (default_mode if mode is None else mode).lower()
        if mode_str not in ("sde", "ode"):
            raise ValueError(f"mode must be 'sde' or 'ode', got '{mode_str}'.")
        self.mode = mode_str

        self.param_transform = (
            default_transform if param_transform is None else param_transform.lower()
        )
        self.param_order = default_order if param_order is None else param_order.lower()
        self.u0 = default_u0 if u0 is None else torch.as_tensor(u0, dtype=torch.float32)

        self.dim_data_raw = int(2 * self.num_steps)

        resolved_summary = default_summary if summary is _SENTINEL else summary
        if resolved_summary is None or resolved_summary in ("subsample", "lognormal"):
            self.dim_data = self.dim_data_raw
        elif resolved_summary in ("stats", "summary_stats"):
            self.dim_data = 9
        elif resolved_summary == "two_step":
            self.dim_data = 4
        else:
            raise NotImplementedError(f"Unknown summary mode: {resolved_summary}")
        self.summary = resolved_summary

        super().__init__(seed, self.dim_parameters, self.dim_data, **kwargs)

        if prior is not None:
            self.prior_dist = prior
            self.canonical_theta = torch.zeros(4, dtype=torch.float32)
        elif self.preset == "sbibm":
            # Prior from Lueckmann et al. (2021) (arXiv:2101.04653):
            # alpha, gamma ~ LogNormal(-0.125, 0.5), beta, delta ~ LogNormal(-3.0, 0.5)
            self.prior_loc = [-0.125, -3.0, -0.125, -3.0]
            self.prior_scale = [0.5, 0.5, 0.5, 0.5]
            self.prior_dist = pdist.LogNormal(
                loc=torch.tensor(self.prior_loc, dtype=torch.float32),
                scale=torch.tensor(self.prior_scale, dtype=torch.float32),
            ).to_event(1)
            self.canonical_theta = torch.exp(
                torch.tensor(self.prior_loc, dtype=torch.float32)
            )
        elif self.preset == "papamakarios":
            # Broad prior from Papamakarios et al. (2019): Uniform(-5, 2)^4
            self.prior_dist = pdist.Uniform(
                low=torch.full((4,), -5.0, dtype=torch.float32),
                high=torch.full((4,), 2.0, dtype=torch.float32),
            ).to_event(1)
            # Ground truth parameters from paper: (log 0.01, log 0.5, log 1, log 0.01)
            if self.param_order == "paper":
                self.canonical_theta = torch.tensor(
                    [-4.605170, -0.693147, 0.0, -4.605170], dtype=torch.float32
                )
            else:
                self.canonical_theta = torch.tensor(
                    [0.0, -4.605170, -0.693147, -4.605170], dtype=torch.float32
                )
        else:
            # Standard Normal N(0, I_4) matching markovsbi
            self.prior_loc = [0.0, 0.0, 0.0, 0.0]
            self.prior_scale = [1.0, 1.0, 1.0, 1.0]
            self.prior_dist = pdist.Normal(
                loc=torch.zeros(4, dtype=torch.float32),
                scale=torch.ones(4, dtype=torch.float32),
            ).to_event(1)
            self.canonical_theta = torch.zeros(4, dtype=torch.float32)

        # PRNG state management
        self._rng_key = jax.random.PRNGKey(self.seed)

        # Build JAX solver
        self._step_fn = _build_lv_step_fn(
            dt=self.dt,
            inner_steps=self.inner_steps,
            sigma1=self.sigma1,
            sigma2=self.sigma2,
            mode=self.mode,
            param_transform=self.param_transform,
            param_order=self.param_order,
        )
        self._jax_solver_batch, self._jax_solver_single = _build_lv_sde_jax_solver(
            self._step_fn,
            num_steps=self.num_steps,
            default_x0=jnp.asarray(self.u0.detach().cpu().numpy()),
        )

        if self.preset == "sbibm":
            prior_type = "lognormal"
        elif self.preset == "papamakarios":
            prior_type = "uniform"
        else:
            prior_type = "gaussian"

        base_u0_jax = jnp.asarray(self.u0.detach().cpu().numpy())

        if self.mode == "ode":
            self._stats_ll_fn = _build_lv_stats_ll(
                self._jax_solver_single,
                default_x0=base_u0_jax,
            )
            self._lognormal_ll_fn = _build_lv_lognormal_ll(
                self._jax_solver_single,
                default_x0=base_u0_jax,
                obs_sigma=self.obs_sigma,
            )
            self._ode_ll_fn = build_ode_ll(
                self._jax_solver_single,
                obs_dim=2,
                obs_sigma=self.obs_sigma,
            )
            if self.summary in ("stats", "summary_stats"):
                active_ll_fn = self._stats_ll_fn
            elif self.summary == "lognormal":
                active_ll_fn = self._lognormal_ll_fn
            else:
                active_ll_fn = self._ode_ll_fn
            self._jax_mcmc = build_mh_mcmc_sampler(
                active_ll_fn,
                dim_parameters=self.dim_parameters,
                prior_type=prior_type,
                prior_low=-5.0,
                prior_high=2.0,
                prior_loc=tuple(self.prior_loc) if self.preset == "sbibm" else None,
                prior_scale=tuple(self.prior_scale) if self.preset == "sbibm" else None,
                default_init_theta=tuple(float(v) for v in self.canonical_theta.tolist()),
            )
        else:
            self._pf_ll_fn = build_particle_filter_ll(
                self._step_fn,
                obs_dim=2,
                obs_sigma=self.obs_sigma,
            )
            self._jax_mcmc = build_gibbs_pmmh_sampler(
                self._pf_ll_fn,
                dim_parameters=self.dim_parameters,
            )

    def _next_rng_key(self) -> jax.Array:
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return subkey

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset internal PRNG state for reproducibility."""
        if seed is not None:
            self.seed = seed
        self._rng_key = jax.random.PRNGKey(self.seed)
        pyro.set_rng_seed(self.seed)
        torch.manual_seed(self.seed)

    def _solve(
        self,
        parameters: torch.Tensor,
        u0: torch.Tensor | None = None,
        rng_key: jax.Array | None = None,
    ) -> torch.Tensor:
        """Batched simulation accelerated via JAX.

        parameters: (num_samples, 4) tensor of raw thetas
        u0: optional (num_samples, 2) tensor of initial states
        returns: (num_samples, 2, num_steps) tensor
        """
        params_np = parameters.detach().cpu().numpy().astype(np.float32)
        params_jax = jnp.asarray(params_np)
        batch_size = params_jax.shape[0]

        if rng_key is None:
            rng_key = self._next_rng_key()
        keys = jax.random.split(rng_key, batch_size)

        if u0 is not None:
            u0_np = u0.detach().cpu().numpy().astype(np.float32)
            u0_jax = jnp.asarray(u0_np)
            res_jax = self._jax_solver_batch(keys, params_jax, u0_jax)
        else:
            res_jax = self._jax_solver_batch(keys, params_jax)

        res_np = np.array(res_jax, copy=True)
        return torch.from_numpy(res_np)

    def get_prior_x0(self) -> pdist.Distribution:
        """Prior over initial condition x0 = [prey, predator]."""
        if self.preset in ("sbibm", "papamakarios"):
            return pdist.Uniform(
                low=torch.tensor([0.0, 0.0], dtype=torch.float32),
                high=torch.tensor([200.0, 200.0], dtype=torch.float32),
            ).to_event(1)
        return pdist.Uniform(
            low=torch.tensor([0.0, 0.0], dtype=torch.float32),
            high=torch.tensor([10.0, 10.0], dtype=torch.float32),
        ).to_event(1)

    @staticmethod
    def compute_summary_stats(us: torch.Tensor) -> torch.Tensor:
        """Compute 9 summary statistics from (num_samples, 2, T) trajectory tensor."""
        is_2d_traj = us.ndim == 2
        if is_2d_traj:
            us = us.unsqueeze(0)

        eps = 1e-6
        x = us[:, 0, :]
        y = us[:, 1, :]

        mean_x = x.mean(dim=-1, keepdim=True)
        mean_y = y.mean(dim=-1, keepdim=True)

        dx = x - mean_x
        dy = y - mean_y

        var_x = (dx**2).mean(dim=-1, keepdim=True)
        var_y = (dy**2).mean(dim=-1, keepdim=True)

        log_var_x = torch.log(var_x.clamp_min(eps))
        log_var_y = torch.log(var_y.clamp_min(eps))

        T = us.shape[-1]
        denom_x = var_x.clamp_min(eps)
        denom_y = var_y.clamp_min(eps)

        if T >= 2:
            ac1_x = (dx[:, 1:] * dx[:, :-1]).mean(dim=-1, keepdim=True) / denom_x
            ac1_y = (dy[:, 1:] * dy[:, :-1]).mean(dim=-1, keepdim=True) / denom_y
        else:
            ac1_x = torch.zeros_like(mean_x)
            ac1_y = torch.zeros_like(mean_y)

        if T >= 3:
            ac2_x = (dx[:, 2:] * dx[:, :-2]).mean(dim=-1, keepdim=True) / denom_x
            ac2_y = (dy[:, 2:] * dy[:, :-2]).mean(dim=-1, keepdim=True) / denom_y
        else:
            ac2_x = torch.zeros_like(mean_x)
            ac2_y = torch.zeros_like(mean_y)

        denom_xy = torch.sqrt(denom_x * denom_y)
        cc_xy = (dx * dy).mean(dim=-1, keepdim=True) / denom_xy

        ac1_x = torch.nan_to_num(ac1_x.clamp(-1.0, 1.0), nan=0.0)
        ac2_x = torch.nan_to_num(ac2_x.clamp(-1.0, 1.0), nan=0.0)
        ac1_y = torch.nan_to_num(ac1_y.clamp(-1.0, 1.0), nan=0.0)
        ac2_y = torch.nan_to_num(ac2_y.clamp(-1.0, 1.0), nan=0.0)
        cc_xy = torch.nan_to_num(cc_xy.clamp(-1.0, 1.0), nan=0.0)

        stats = torch.cat(
            [
                mean_x,
                mean_y,
                log_var_x,
                log_var_y,
                ac1_x,
                ac2_x,
                ac1_y,
                ac2_y,
                cc_xy,
            ],
            dim=-1,
        )
        return stats.squeeze(0) if is_2d_traj else stats

    def stepper(
        self,
        x: torch.Tensor,
        parameters: torch.Tensor,
        dt: float | None = None,
        sigma: float | None = None,
    ) -> torch.Tensor:
        """Simulate a single time-series step transition for Lotka-Volterra SDE.

        Parameters
        ----------
        x : torch.Tensor
            Current state [x0, x1], shape (2,) or (batch_size, 2).
        parameters : torch.Tensor
            Lotka-Volterra parameters (raw theta in R^4 or rates),
            shape (4,) or (batch_size, 4).
        dt : float, optional
            Step time interval (default: self.dt * self.inner_steps = 0.1).
        sigma : float, optional
            Diffusion level (default: self.sigma1 = 0.1).

        Returns
        -------
        torch.Tensor
            Next state [x0, x1] of shape (2,) or (batch_size, 2).
        """
        x_t = torch.as_tensor(x, dtype=torch.float32)
        params = torch.as_tensor(parameters, dtype=torch.float32)

        is_1d = x_t.ndim == 1
        if is_1d:
            x_t = x_t.unsqueeze(0)
        if params.ndim == 1:
            params = params.unsqueeze(0)
        if params.shape[0] == 1 and x_t.shape[0] > 1:
            params = params.expand(x_t.shape[0], -1)
        elif x_t.shape[0] == 1 and params.shape[0] > 1:
            x_t = x_t.expand(params.shape[0], -1)

        sig1 = float(self.sigma1 if sigma is None else sigma)
        sig2 = float(self.sigma2 if sigma is None else sigma)

        # Parameter mapping
        if self.param_transform == "identity":
            if self.param_order == "paper":
                delta, gamma, alpha, beta = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
            else:
                alpha, beta, gamma, delta = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        elif self.param_transform == "exp":
            if self.param_order == "paper":
                delta = torch.exp(params[:, 0])
                gamma = torch.exp(params[:, 1])
                alpha = torch.exp(params[:, 2])
                beta = torch.exp(params[:, 3])
            else:
                alpha = torch.exp(params[:, 0])
                beta = torch.exp(params[:, 1])
                gamma = torch.exp(params[:, 2])
                delta = torch.exp(params[:, 3])
        else:
            if self.param_order == "paper":
                delta = torch.sigmoid(params[:, 0]) * 2.0 + 1.0
                gamma = torch.sigmoid(params[:, 1]) * 2.0 + 1.0
                alpha = torch.sigmoid(params[:, 2]) * 2.0 + 1.0
                beta = torch.sigmoid(params[:, 3]) * 2.0 + 1.0
            else:
                alpha = torch.sigmoid(params[:, 0]) * 2.0 + 1.0
                beta = torch.sigmoid(params[:, 1]) * 2.0 + 1.0
                gamma = torch.sigmoid(params[:, 2]) * 2.0 + 1.0
                delta = torch.sigmoid(params[:, 3]) * 2.0 + 1.0

        step_dt = float(self.dt * self.inner_steps if dt is None else dt)
        sub_dt = float(self.dt)
        n_substeps = max(round(step_dt / sub_dt), 1)
        actual_h = step_dt / n_substeps

        curr_u = torch.clamp(x_t.clone(), min=0.0)

        for _ in range(n_substeps):
            p0 = curr_u[:, 0]
            p1 = curr_u[:, 1]
            dx0 = alpha * p0 - beta * p0 * p1
            dx1 = -gamma * p1 + delta * p0 * p1
            drift_step = torch.stack([dx0, dx1], dim=-1)

            if self.mode == "ode":
                curr_u = torch.clamp(curr_u + drift_step * actual_h, min=0.0)
            else:
                dW = torch.randn_like(curr_u) * math.sqrt(actual_h)
                diff_0 = sig1 * p0 * p1 * dW[:, 0]
                diff_1 = sig2 * p1 * p0 * dW[:, 1]
                diff_step = torch.stack([diff_0, diff_1], dim=-1)
                curr_u = torch.clamp(
                    curr_u + drift_step * actual_h + diff_step, min=0.0
                )

        return curr_u.squeeze(0) if is_1d else curr_u

    def simulate_two_step(
        self,
        parameters: torch.Tensor,
        x0: torch.Tensor | None = None,
        t: int | torch.Tensor | None = None,
        informed: bool = True,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Generate two-step transition pairs (x_t, x_{t+1}) like in markovsbi."""
        return sample_two_step(
            self,
            parameters=parameters,
            x0=x0,
            t=t,
            informed=informed,
            return_dict=return_dict,
        )

    @override
    def simulate(self, parameters: torch.Tensor | np.ndarray) -> torch.Tensor:
        parameters = torch.as_tensor(parameters, dtype=torch.float32)

        if parameters.numel() == 0:
            raise ValueError("Expected at least one parameter.")

        is_1d = parameters.ndim == 1
        if is_1d:
            parameters = parameters.unsqueeze(0)

        num_samples = parameters.shape[0]

        if self.summary == "two_step":
            pairs = self.simulate_two_step(parameters, return_dict=False)
            pairs_flat = pairs.reshape(num_samples, 4)
            return pairs_flat.squeeze(0) if is_1d else pairs_flat

        us = self._solve(parameters)  # (num_samples, 2, num_steps)
        us = us.float()

        has_nan = torch.isnan(us.reshape(num_samples, -1)).any(dim=1)
        idx_contains_nan = torch.where(has_nan)[0]
        idx_contains_no_nan = torch.where(~has_nan)[0]

        if self.summary is None:
            # Flatten to (num_samples, dim_data)
            traj_flat = us.reshape(num_samples, -1)
            return traj_flat.squeeze(0) if is_1d else traj_flat

        if self.summary in ("subsample", "lognormal"):
            data = torch.full(
                (num_samples, self.dim_data),
                float("nan"),
                dtype=torch.float32,
            )

            if len(idx_contains_nan) == num_samples:
                return data.squeeze(0) if is_1d else data

            raw_data = us[idx_contains_no_nan, :, :].reshape(
                len(idx_contains_no_nan), -1
            )
            if self.summary == "lognormal":
                noise = torch.exp(torch.randn_like(raw_data) * self.obs_sigma)
                sampled = raw_data * noise
            else:
                noise = torch.randn_like(raw_data) * self.obs_sigma
                sampled = raw_data + noise

            data[idx_contains_no_nan, :] = sampled
            return data.squeeze(0) if is_1d else data

        if self.summary in ("stats", "summary_stats"):
            data = torch.full(
                (num_samples, self.dim_data),
                float("nan"),
                dtype=torch.float32,
            )

            if len(idx_contains_nan) == num_samples:
                return data.squeeze(0) if is_1d else data

            stats = self.compute_summary_stats(us[idx_contains_no_nan, :, :])
            data[idx_contains_no_nan, :] = stats
            return data.squeeze(0) if is_1d else data

        raise ValueError(f"Unknown summary mode: {self.summary}")

    @override
    def get_prior(self) -> pdist.Distribution:
        return self.prior_dist

    def log_likelihood(
        self, parameters: torch.Tensor, x_obs: torch.Tensor, n_particles: int = 150
    ) -> torch.Tensor:
        r"""Evaluate marginal log-likelihood $\log p(y_{1:T} \mid \theta)$.

        Mathematical Formulation:
        -------------------------
        - ODE Mode: Exact log-likelihood under deterministic trajectory $x_t(\theta)$ with $x_0 = y_0$:
              $$\log p(y_{1:T} \mid \theta) = -\frac{1}{2 \sigma_{\text{obs}}^2} \sum_{t=1}^{T-1} \| y_t - x_t(\theta) \|_2^2 - (T-1) \left( 2 \log \sigma_{\text{obs}} + \log(2\pi) \right)$$
        - SDE Mode: Marginal log-likelihood integrating out the unobserved diffusion path:
              $$\log p(y_{1:T} \mid \theta) = \log \int p(y_{1:T}, x_{1:T} \mid \theta) \, dx_{1:T}$$
          estimated unbiasedly using a JAX-accelerated Sequential Monte Carlo (SMC) bootstrap
          particle filter with $N = \text{n\_particles}$ particles and systematic resampling.

        Parameters
        ----------
        parameters : torch.Tensor
            Latent parameter tensor $\theta \in \mathbb{R}^4$, shape `(4,)` or `(batch_size, 4)`.
        x_obs : torch.Tensor
            Observed trajectory $y_{1:T}$, shape `(2 * num_steps,)`, `(2, num_steps)`,
            or `(num_steps, 2)`.
        n_particles : int, optional
            Number of particles for the SMC particle filter in SDE mode (default: 150).

        Returns
        -------
        torch.Tensor
            Log-likelihood values of shape `(batch_size,)`.
        """
        parameters = torch.as_tensor(parameters, dtype=torch.float32)
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(0)

        num_samples = parameters.shape[0]
        if isinstance(x_obs, np.ndarray):
            x_obs = np.array(x_obs, copy=True)
        x_obs_tensor = torch.as_tensor(x_obs, dtype=torch.float32)
        if x_obs_tensor.ndim == 2 and x_obs_tensor.shape[0] == 1:
            x_obs_tensor = x_obs_tensor.squeeze(0)

        # Support 9 summary statistics in ODE mode
        if (
            x_obs_tensor.numel() == 9
            and self.summary in ("stats", "summary_stats")
            and self.mode == "ode"
        ):
            x_jax = jnp.asarray(x_obs_tensor.flatten().detach().cpu().numpy())
            params_jax = jnp.asarray(parameters.detach().cpu().numpy())
            v_ll = jax.vmap(lambda th: self._stats_ll_fn(th, x_jax))
            lls_jax = v_ll(params_jax)
            return torch.from_numpy(np.array(lls_jax, copy=True))

        if x_obs_tensor.ndim == 1:
            x_obs_2d = x_obs_tensor.reshape(2, -1).T
        elif x_obs_tensor.ndim == 2 and x_obs_tensor.shape == (2, self.num_steps):
            x_obs_2d = x_obs_tensor.T
        else:
            x_obs_2d = x_obs_tensor

        x_jax = jnp.asarray(x_obs_2d.detach().cpu().numpy())
        params_jax = jnp.asarray(parameters.detach().cpu().numpy())

        if self.mode == "ode":
            if self.summary == "lognormal":
                v_ll = jax.vmap(lambda th: self._lognormal_ll_fn(th, x_jax))
            else:
                v_ll = jax.vmap(lambda th: self._ode_ll_fn(th, x_jax))
            lls_jax = v_ll(params_jax)
            return torch.from_numpy(np.array(lls_jax, copy=True))

        key = self._next_rng_key()
        keys = jax.random.split(key, num_samples)

        v_ll = jax.vmap(
            lambda th, k: self._pf_ll_fn(th, x_jax, k, n_particles=n_particles)
        )
        lls_jax = v_ll(params_jax, keys)
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
        r"""Sample from the true posterior $p(\theta \mid y_{1:T})$ using JAX MCMC.

        Mathematical Formulation:
        -------------------------
        The target posterior distribution is given by Bayes' rule:
            $$p(\theta \mid y_{1:T}) \propto p(\theta) \, p(y_{1:T} \mid \theta)$$

        Inference Algorithms:
        ---------------------
        - ODE Mode: Metropolis-Hastings MCMC with Gaussian random walk proposals,
          evaluated on either the deterministic summary statistics or trajectory likelihood.
        - SDE Mode: Gibbs Particle Marginal Metropolis-Hastings (PMMH) sampler.
        """
        is_default_obs = x_obs is None
        if x_obs is None:
            if self.preset in ("sbibm", "papamakarios"):
                th_star = self.canonical_theta
                if self.summary in ("stats", "summary_stats", "lognormal"):
                    x_obs = self.simulate(th_star)
                else:
                    x_ref = self._solve(th_star.unsqueeze(0), u0=self.u0.unsqueeze(0))[0].T
                    x_obs = x_ref
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

        if isinstance(x_obs, np.ndarray):
            x_obs = np.array(x_obs, copy=True)
        x_obs_tensor = torch.as_tensor(x_obs, dtype=torch.float32)
        if x_obs_tensor.ndim == 2 and x_obs_tensor.shape[0] == 1:
            x_obs_tensor = x_obs_tensor.squeeze(0)

        # Check if x_obs is 9 summary statistics in ODE mode
        if (
            x_obs_tensor.numel() == 9
            and self.summary in ("stats", "summary_stats")
            and self.mode == "ode"
        ):
            x_jax = jnp.asarray(x_obs_tensor.flatten().detach().cpu().numpy())
        else:
            if x_obs_tensor.ndim == 1:
                x_obs_2d = x_obs_tensor.reshape(2, -1).T
            elif x_obs_tensor.ndim == 2 and x_obs_tensor.shape == (2, self.num_steps):
                x_obs_2d = x_obs_tensor.T
            else:
                x_obs_2d = x_obs_tensor
            x_jax = jnp.asarray(x_obs_2d.detach().cpu().numpy())
        if self.preset == "markovsbi" and is_default_obs:
            rng_key = jax.random.PRNGKey(self.seed)
        else:
            rng_key = self._next_rng_key()
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
        samples_np = np.array(samples_jax, copy=True)
        np.random.shuffle(samples_np)
        samples_torch = torch.from_numpy(samples_np)
        return samples_torch[:num_samples]
