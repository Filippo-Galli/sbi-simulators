"""Simulator for the SIR (Susceptible-Infected-Recovered) SDE model."""

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
        dS = -inf_rate
        dI = inf_rate - gamma * I
        dR = gamma * I
        return jnp.stack([dS, dI, dR])

    def diffusion(x):
        return jnp.array(
            [[0.01, 0.0, 0.0], [0.0, x[1] * sigma1, 0.0], [0.0, 0.0, 0.01]]
        )

    def extract_rates(theta):
        if param_transform == "identity":
            if param_order == "paper":
                beta, gamma = theta[0], theta[1]
            else:
                gamma, beta = theta[0], theta[1]
        else:
            if param_order == "paper":
                beta = jax.nn.sigmoid(theta[0]) * 2.0
                gamma = jax.nn.sigmoid(theta[1]) * 2.0
            else:
                gamma = jax.nn.sigmoid(theta[0]) * 2.0
                beta = jax.nn.sigmoid(theta[1]) * 2.0
        return gamma, beta

    def step_one(x, key, theta):
        gamma, beta = extract_rates(theta)
        keys = jax.random.split(key, inner_steps)

        def inner_loop(i, curr_x):
            curr_x = jnp.clip(curr_x, 0.0)
            if mode == "ode":
                k1 = drift(curr_x, gamma, beta)
                k2 = drift(jnp.clip(curr_x + 0.5 * dt * k1, 0.0), gamma, beta)
                k3 = drift(jnp.clip(curr_x + 0.5 * dt * k2, 0.0), gamma, beta)
                k4 = drift(jnp.clip(curr_x + dt * k3, 0.0), gamma, beta)
                curr_x = curr_x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:
                xdt = drift(curr_x, gamma, beta)
                dWt = jax.random.normal(keys[i], shape=(3,)) * jnp.sqrt(dt)
                B = diffusion(curr_x)
                curr_x = curr_x + xdt * dt + jnp.dot(B, dWt)
            return jnp.clip(curr_x, 0.0)

        return jax.lax.fori_loop(0, inner_steps, inner_loop, x)

    return step_one


def _build_sir_sde_jax_solver(
    step_one_fn, num_steps: int = 11, default_x0: jnp.ndarray | None = None
):
    """Build a JIT-compiled batched Euler-Maruyama SDE solver in JAX for SIR."""
    base_x0 = (
        jnp.array([4.0, 1.0, 0.0], dtype=jnp.float32)
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
        full_traj = jnp.concatenate([u0[None, :], xs], axis=0)  # (num_steps, 3)
        return full_traj.T  # (3, num_steps)

    solve_batch_fn = jax.vmap(solve_single, in_axes=(0, 0, 0))

    def solve_batch(rng_keys, thetas, x0s=None):
        if x0s is None:
            x0s = jnp.broadcast_to(base_x0, (thetas.shape[0], 3))
        return solve_batch_fn(rng_keys, thetas, x0s)

    return jax.jit(solve_batch), jax.jit(solve_single)


def _build_sir_binomial_ll(
    jax_solver_single,
    default_x0: jnp.ndarray,
    total_population: float = 1_000_000.0,
    binomial_n: int = 1000,
):
    """Build a JIT-compiled Binomial observation log-likelihood for SIR ODE."""
    dummy_key = jax.random.PRNGKey(0)
    log_n_fact = jax.scipy.special.gammaln(binomial_n + 1.0)

    def binomial_ll(theta, x_obs):
        # x_obs: (num_steps,)
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
        has_nan = jnp.isnan(ll)
        return jnp.where(has_nan, -1e9, ll)

    return jax.jit(binomial_ll)


_SENTINEL = object()


class SIR(Simulator):
    r"""SIR (Susceptible-Infected-Recovered) Epidemic Simulator (SDE and ODE modes).

    Mathematical Model:
    -------------------
    The state is a 3D vector $x(t) = [S(t), I(t), R(t)]^T$ representing the susceptible ($S$),
    infected ($I$), and recovered ($R$) populations over continuous time $t \ge 0$.

    1. Prior Distribution:
       The parameters are represented in an unconstrained latent space $\theta \in \mathbb{R}^2$:
           $$\theta = [\theta_0, \theta_1]^T \sim \mathcal{N}(0, I_2)$$

    2. Parameter Transformation:
       The latent parameters map to positive recovery ($\gamma$) and transmission ($\beta$) rates via a scaled sigmoid:
           $$\gamma = 2.0 \cdot \sigma(\theta_0) \in (0.0, 2.0) \quad \text{(recovery rate)}$$
           $$\beta  = 2.0 \cdot \sigma(\theta_1) \in (0.0, 2.0) \quad \text{(transmission rate)}$$
       where $\sigma(z) = \frac{1}{1 + e^{-z}}$.

    3. Continuous Dynamics:
       - SDE Mode (`mode="sde"`, default):
         Stochastic differential equation with diffusion driven by independent Brownian motions:
             $$dS = -\beta S I \, dt + 0.01 \, dW_0$$
             $$dI = (\beta S I - \gamma I) \, dt + \sigma_1 I \, dW_1$$
             $$dR = \gamma I \, dt + 0.01 \, dW_2$$
         where $W_0, W_1, W_2$ are standard independent Wiener processes, and $\sigma_1 = 0.1$ by default.
       - ODE Mode (`mode="ode"`):
         Deterministic epidemic dynamical system:
             $$\frac{dS}{dt} = -\beta S I$$
             $$\frac{dI}{dt} = \beta S I - \gamma I$$
             $$\frac{dR}{dt} = \gamma I$$

    4. Initial Condition:
       Fixed default initial state $u_0 = [S(0), I(0), R(0)]^T = [4.0, 1.0, 0.0]^T$.

    5. Numerical Integration & Observation Model:
       Integrated using Euler-Maruyama (or Euler) integration with internal step $dt = 0.01$
       and `inner_steps` = 5 sub-steps per observation interval:
           $$\Delta t_{\text{obs}} = dt \times \text{inner\_steps} = 0.05$$
       Over $T = \text{num\_steps} = 11$, total simulated duration is $t \in [0, 0.50]$.
       Additive Gaussian observation noise with standard deviation $\sigma_{\text{obs}} = 0.01$:
           $$y_t \sim \mathcal{N}(x_t, \sigma_{\text{obs}}^2 I_3), \quad t = 0, \dots, T-1$$

    6. Summary Modes:
       - `None`: Raw state trajectory of shape $(3 \times \text{num\_steps},) = (33,)$.
       - `"subsample"`: Trajectory with additive Gaussian observation noise $\sigma_{\text{obs}}$, shape $(33,)$.
       - `"stats"`: 15 summary statistics (mean, log-variance, ac1, ac2, cross-correlation per channel).
       - `"two_step"`: Transition pairs $(x_t, x_{t+1})$ of shape $(6,)$ for Markov SBI.

    Posterior Distribution & Reference Sampling:
    ---------------------------------------------
    The posterior distribution is conditioned on the full trajectory observations $y_{1:T}$:
        $$p(\theta \mid y_{1:T}) \propto p(\theta) \, p(y_{1:T} \mid \theta)$$

    - ODE Mode: $p(y_{1:T} \mid \theta) = \prod_{t=1}^{T-1} \mathcal{N}(y_t \mid x_t(\theta), \sigma_{\text{obs}}^2 I_3)$.
      Sampled via JAX-accelerated Metropolis-Hastings MCMC.
    - SDE Mode: Marginal likelihood $p(y_{1:T} \mid \theta) = \int p(y_{1:T}, x_{1:T} \mid \theta) \, dx_{1:T}$
      is estimated unbiasedly using a bootstrap Particle Filter ($N=150$ particles).
      Sampled via JAX-accelerated Gibbs Particle Marginal Metropolis-Hastings (PMMH).
    - Returned posterior samples are in the unconstrained latent parameter space $\theta \in \mathbb{R}^2$.
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
        summary: str | None = _SENTINEL,
        mode: str | None = None,
        preset: str = "markovsbi",
        u0: torch.Tensor | None = None,
        prior: pdist.Distribution | None = None,
        param_transform: str | None = None,
        param_order: str | None = None,
        total_population: float | None = None,
        binomial_n: int = 1000,
        **kwargs,
    ):
        self.dim_parameters = dim_parameters
        if dim_parameters != 2:
            raise ValueError(
                f"SIR requires dim_parameters=2 (gamma, beta), got {dim_parameters}."
            )

        preset_key = preset.lower()
        if preset_key == "sbibm":
            self.preset = "sbibm"
            default_dt = 160.0 / 9.0 / 200.0
            default_inner_steps = 200
            default_sigma1 = 0.0
            default_num_steps = 10
            default_obs_sigma = 0.01
            default_mode = "ode"
            default_total_population = 1_000_000.0
            default_u0 = torch.tensor([999999.0, 1.0, 0.0], dtype=torch.float32)
            default_summary = "binomial"
            default_transform = "identity"
            default_order = "paper"
        elif preset_key == "markovsbi":
            self.preset = "markovsbi"
            default_dt = 0.01
            default_inner_steps = 5
            default_sigma1 = 0.1
            default_num_steps = 11
            default_obs_sigma = 0.01
            default_mode = "sde"
            default_total_population = 1.0
            default_u0 = torch.tensor([4.0, 1.0, 0.0], dtype=torch.float32)
            default_summary = None
            default_transform = "sigmoid"
            default_order = "alphabetical"
        else:
            raise ValueError(
                f"Unknown preset '{preset}'. Supported presets are 'markovsbi' and 'sbibm'."
            )

        self.dt = default_dt if dt is None else dt
        self.inner_steps = default_inner_steps if inner_steps is None else inner_steps
        self.sigma1 = default_sigma1 if sigma1 is None else sigma1
        self.num_steps = default_num_steps if num_steps is None else num_steps
        self.obs_sigma = default_obs_sigma if obs_sigma is None else obs_sigma
        self.total_population = (
            default_total_population
            if total_population is None
            else float(total_population)
        )
        self.binomial_n = int(binomial_n)

        mode_str = (default_mode if mode is None else mode).lower()
        if mode_str not in ("sde", "ode"):
            raise ValueError(f"mode must be 'sde' or 'ode', got '{mode_str}'.")
        self.mode = mode_str

        self.param_transform = (
            default_transform if param_transform is None else param_transform.lower()
        )
        self.param_order = default_order if param_order is None else param_order.lower()
        self.u0 = default_u0 if u0 is None else torch.as_tensor(u0, dtype=torch.float32)

        self.dim_data_raw = int(3 * self.num_steps)

        resolved_summary = default_summary if summary is _SENTINEL else summary
        if resolved_summary is None or resolved_summary == "subsample":
            self.dim_data = self.dim_data_raw
        elif resolved_summary == "binomial":
            self.dim_data = self.num_steps
        elif resolved_summary in ("stats", "summary_stats"):
            self.dim_data = 15
        elif resolved_summary == "two_step":
            self.dim_data = 6
        else:
            raise NotImplementedError(f"Unknown summary mode: {resolved_summary}")
        self.summary = resolved_summary

        super().__init__(seed, self.dim_parameters, self.dim_data, **kwargs)

        if prior is not None:
            self.prior_dist = prior
            self.canonical_theta = torch.zeros(2, dtype=torch.float32)
        elif self.preset == "sbibm":
            # Prior from Lueckmann et al. (2021) (arXiv:2101.04653):
            # beta ~ LogNormal(log(0.4), 0.5), gamma ~ LogNormal(log(1/8), 0.2)
            self.prior_loc = [math.log(0.4), math.log(1.0 / 8.0)]
            self.prior_scale = [0.5, 0.2]
            self.prior_dist = pdist.LogNormal(
                loc=torch.tensor(self.prior_loc, dtype=torch.float32),
                scale=torch.tensor(self.prior_scale, dtype=torch.float32),
            ).to_event(1)
            self.canonical_theta = torch.tensor([0.4, 1.0 / 8.0], dtype=torch.float32)
        else:
            # Standard Normal N(0, I_2) matching markovsbi
            self.prior_loc = [0.0, 0.0]
            self.prior_scale = [1.0, 1.0]
            self.prior_dist = pdist.Normal(
                loc=torch.zeros(2, dtype=torch.float32),
                scale=torch.ones(2, dtype=torch.float32),
            ).to_event(1)
            self.canonical_theta = torch.zeros(2, dtype=torch.float32)

        # PRNG state management
        self._rng_key = jax.random.PRNGKey(self.seed)

        # Build JAX solver
        self._step_fn = _build_sir_step_fn(
            dt=self.dt,
            inner_steps=self.inner_steps,
            sigma1=self.sigma1,
            mode=self.mode,
            total_population=self.total_population,
            param_transform=self.param_transform,
            param_order=self.param_order,
        )
        base_u0_jax = jnp.asarray(self.u0.detach().cpu().numpy())
        self._jax_solver_batch, self._jax_solver_single = _build_sir_sde_jax_solver(
            self._step_fn,
            num_steps=self.num_steps,
            default_x0=base_u0_jax,
        )

        prior_type = "lognormal" if self.preset == "sbibm" else "gaussian"

        if self.mode == "ode":
            self._binomial_ll_fn = _build_sir_binomial_ll(
                self._jax_solver_single,
                default_x0=base_u0_jax,
                total_population=self.total_population,
                binomial_n=self.binomial_n,
            )
            self._ode_ll_fn = build_ode_ll(
                self._jax_solver_single,
                obs_dim=3,
                obs_sigma=self.obs_sigma,
            )
            active_ll_fn = (
                self._binomial_ll_fn
                if self.summary == "binomial"
                else self._ode_ll_fn
            )
            self._jax_mcmc = build_mh_mcmc_sampler(
                active_ll_fn,
                dim_parameters=self.dim_parameters,
                prior_type=prior_type,
                prior_loc=tuple(self.prior_loc) if self.preset == "sbibm" else None,
                prior_scale=tuple(self.prior_scale) if self.preset == "sbibm" else None,
                default_init_theta=tuple(float(v) for v in self.canonical_theta.tolist()),
            )
        else:
            self._pf_ll_fn = build_particle_filter_ll(
                self._step_fn,
                obs_dim=3,
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

        parameters: (num_samples, 2) tensor of raw thetas
        u0: optional (num_samples, 3) tensor of initial states
        returns: (num_samples, 3, num_steps) tensor
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
        """Prior over initial condition x0 = [S, I, R]."""
        return pdist.Uniform(
            low=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            high=torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32),
        ).to_event(1)

    @staticmethod
    def compute_summary_stats(us: torch.Tensor) -> torch.Tensor:
        """Compute 15 summary statistics from (num_samples, 3, T) trajectory tensor.

        Statistics:
            - Mean of S, I, R (3)
            - Log-variance of S, I, R (3)
            - Lag-1 autocorrelation of S, I, R (3)
            - Lag-2 autocorrelation of S, I, R (3)
            - Pairwise cross-correlations (S-I, I-R, S-R) (3)
        """
        is_2d_traj = us.ndim == 2
        if is_2d_traj:
            us = us.unsqueeze(0)

        eps = 1e-6
        s = us[:, 0, :]
        i = us[:, 1, :]
        r = us[:, 2, :]

        mean_s = s.mean(dim=-1, keepdim=True)
        mean_i = i.mean(dim=-1, keepdim=True)
        mean_r = r.mean(dim=-1, keepdim=True)

        ds = s - mean_s
        di = i - mean_i
        dr = r - mean_r

        var_s = (ds**2).mean(dim=-1, keepdim=True)
        var_i = (di**2).mean(dim=-1, keepdim=True)
        var_r = (dr**2).mean(dim=-1, keepdim=True)

        log_var_s = torch.log(var_s.clamp_min(eps))
        log_var_i = torch.log(var_i.clamp_min(eps))
        log_var_r = torch.log(var_r.clamp_min(eps))

        T = us.shape[-1]
        denom_s = var_s.clamp_min(eps)
        denom_i = var_i.clamp_min(eps)
        denom_r = var_r.clamp_min(eps)

        if T >= 2:
            ac1_s = (ds[:, 1:] * ds[:, :-1]).mean(dim=-1, keepdim=True) / denom_s
            ac1_i = (di[:, 1:] * di[:, :-1]).mean(dim=-1, keepdim=True) / denom_i
            ac1_r = (dr[:, 1:] * dr[:, :-1]).mean(dim=-1, keepdim=True) / denom_r
        else:
            ac1_s = torch.zeros_like(mean_s)
            ac1_i = torch.zeros_like(mean_i)
            ac1_r = torch.zeros_like(mean_r)

        if T >= 3:
            ac2_s = (ds[:, 2:] * ds[:, :-2]).mean(dim=-1, keepdim=True) / denom_s
            ac2_i = (di[:, 2:] * di[:, :-2]).mean(dim=-1, keepdim=True) / denom_i
            ac2_r = (dr[:, 2:] * dr[:, :-2]).mean(dim=-1, keepdim=True) / denom_r
        else:
            ac2_s = torch.zeros_like(mean_s)
            ac2_i = torch.zeros_like(mean_i)
            ac2_r = torch.zeros_like(mean_r)

        denom_si = torch.sqrt(denom_s * denom_i)
        denom_ir = torch.sqrt(denom_i * denom_r)
        denom_sr = torch.sqrt(denom_s * denom_r)

        cc_si = (ds * di).mean(dim=-1, keepdim=True) / denom_si
        cc_ir = (di * dr).mean(dim=-1, keepdim=True) / denom_ir
        cc_sr = (ds * dr).mean(dim=-1, keepdim=True) / denom_sr

        ac1_s = torch.nan_to_num(ac1_s.clamp(-1.0, 1.0), nan=0.0)
        ac2_s = torch.nan_to_num(ac2_s.clamp(-1.0, 1.0), nan=0.0)
        ac1_i = torch.nan_to_num(ac1_i.clamp(-1.0, 1.0), nan=0.0)
        ac2_i = torch.nan_to_num(ac2_i.clamp(-1.0, 1.0), nan=0.0)
        ac1_r = torch.nan_to_num(ac1_r.clamp(-1.0, 1.0), nan=0.0)
        ac2_r = torch.nan_to_num(ac2_r.clamp(-1.0, 1.0), nan=0.0)
        cc_si = torch.nan_to_num(cc_si.clamp(-1.0, 1.0), nan=0.0)
        cc_ir = torch.nan_to_num(cc_ir.clamp(-1.0, 1.0), nan=0.0)
        cc_sr = torch.nan_to_num(cc_sr.clamp(-1.0, 1.0), nan=0.0)

        stats = torch.cat(
            [
                mean_s,
                mean_i,
                mean_r,
                log_var_s,
                log_var_i,
                log_var_r,
                ac1_s,
                ac2_s,
                ac1_i,
                ac2_i,
                ac1_r,
                ac2_r,
                cc_si,
                cc_ir,
                cc_sr,
            ],
            dim=-1,
        )
        return stats.squeeze(0) if is_2d_traj else stats

    def stepper(
        self,
        x: torch.Tensor,
        parameters: torch.Tensor,
        dt: float | None = None,
        sigma1: float | None = None,
    ) -> torch.Tensor:
        """Simulate a single time-series step transition for SIR SDE.

        Parameters
        ----------
        x : torch.Tensor
            Current state [S, I, R], shape (3,) or (batch_size, 3).
        parameters : torch.Tensor
            SIR parameters [gamma, beta] in R^2 (raw theta or mapped rates),
            shape (2,) or (batch_size, 2).
        dt : float, optional
            Step time interval (default: self.dt * self.inner_steps = 0.05).
        sigma1 : float, optional
            Infected diffusion level (default: self.sigma1 = 0.1).

        Returns
        -------
        torch.Tensor
            Next state [S, I, R] of shape (3,) or (batch_size, 3).
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

        sig1 = float(self.sigma1 if sigma1 is None else sigma1)

        # Parameter mapping
        if self.param_transform == "identity":
            if self.param_order == "paper":
                beta, gamma = params[:, 0], params[:, 1]
            else:
                gamma, beta = params[:, 0], params[:, 1]
        else:
            if self.param_order == "paper":
                beta = torch.sigmoid(params[:, 0]) * 2.0
                gamma = torch.sigmoid(params[:, 1]) * 2.0
            else:
                gamma = torch.sigmoid(params[:, 0]) * 2.0
                beta = torch.sigmoid(params[:, 1]) * 2.0

        step_dt = float(self.dt * self.inner_steps if dt is None else dt)
        sub_dt = float(self.dt)
        n_substeps = max(round(step_dt / sub_dt), 1)
        actual_h = step_dt / n_substeps

        curr_u = torch.clamp(x_t.clone(), min=0.0)

        for _ in range(n_substeps):
            S = curr_u[:, 0]
            I = curr_u[:, 1]

            inf_rate = beta * S * I / self.total_population
            dS = -inf_rate
            dI = inf_rate - gamma * I
            dR = gamma * I
            drift_step = torch.stack([dS, dI, dR], dim=-1)

            if self.mode == "ode":
                curr_u = torch.clamp(curr_u + drift_step * actual_h, min=0.0)
            else:
                dW = torch.randn_like(curr_u) * math.sqrt(actual_h)
                diff_0 = 0.01 * dW[:, 0]
                diff_1 = I * sig1 * dW[:, 1]
                diff_2 = 0.01 * dW[:, 2]
                diff_step = torch.stack([diff_0, diff_1, diff_2], dim=-1)
                curr_u = torch.clamp(curr_u + drift_step * actual_h + diff_step, min=0.0)

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
            pairs_flat = pairs.reshape(num_samples, 6)
            return pairs_flat.squeeze(0) if is_1d else pairs_flat

        us = self._solve(parameters)  # (num_samples, 3, num_steps)
        us = us.float()

        has_nan = torch.isnan(us.reshape(num_samples, -1)).any(dim=1)
        idx_contains_nan = torch.where(has_nan)[0]
        idx_contains_no_nan = torch.where(~has_nan)[0]

        if self.summary is None:
            # Flatten to (num_samples, dim_data)
            traj_flat = us.reshape(num_samples, -1)
            return traj_flat.squeeze(0) if is_1d else traj_flat

        if self.summary == "subsample":
            data = torch.full(
                (num_samples, self.dim_data),
                float("nan"),
                dtype=torch.float32,
            )

            if len(idx_contains_nan) == num_samples:
                return data.squeeze(0) if is_1d else data

            # Additive Gaussian observation noise
            raw_data = us[idx_contains_no_nan, :, :].reshape(
                len(idx_contains_no_nan), -1
            )
            noise = torch.randn_like(raw_data) * self.obs_sigma
            sampled = raw_data + noise

            data[idx_contains_no_nan, :] = sampled
            return data.squeeze(0) if is_1d else data

        if self.summary == "binomial":
            data = torch.full(
                (num_samples, self.dim_data),
                float("nan"),
                dtype=torch.float32,
            )

            if len(idx_contains_nan) == num_samples:
                return data.squeeze(0) if is_1d else data

            I_traj = us[idx_contains_no_nan, 1, :]
            probs = (I_traj / self.total_population).clamp(0.0, 1.0)
            sampled = torch.distributions.Binomial(
                total_count=self.binomial_n, probs=probs
            ).sample()
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
              $$\log p(y_{1:T} \mid \theta) = -\frac{1}{2 \sigma_{\text{obs}}^2} \sum_{t=1}^{T-1} \| y_t - x_t(\theta) \|_2^2 - (T-1) \left( 3 \log \sigma_{\text{obs}} + \frac{3}{2} \log(2\pi) \right)$$
        - SDE Mode: Marginal log-likelihood integrating out the unobserved diffusion path:
              $$\log p(y_{1:T} \mid \theta) = \log \int p(y_{1:T}, x_{1:T} \mid \theta) \, dx_{1:T}$$
          estimated unbiasedly using a JAX-accelerated Sequential Monte Carlo (SMC) bootstrap
          particle filter with $N = \text{n\_particles}$ particles and systematic resampling.

        Parameters
        ----------
        parameters : torch.Tensor
            Latent parameter tensor $\theta \in \mathbb{R}^2$, shape `(2,)` or `(batch_size, 2)`.
        x_obs : torch.Tensor
            Observed trajectory $y_{1:T}$, shape `(3 * num_steps,)`, `(3, num_steps)`,
            or `(num_steps, 3)`.
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
        if (
            x_obs_tensor.numel() == self.num_steps
            and self.summary == "binomial"
            and self.mode == "ode"
        ):
            x_jax = jnp.asarray(x_obs_tensor.flatten().detach().cpu().numpy())
            params_jax = jnp.asarray(parameters.detach().cpu().numpy())
            v_ll = jax.vmap(lambda th: self._binomial_ll_fn(th, x_jax))
            lls_jax = v_ll(params_jax)
            return torch.from_numpy(np.array(lls_jax, copy=True))

        if x_obs_tensor.ndim == 1:
            x_obs_2d = x_obs_tensor.reshape(3, -1).T
        elif x_obs_tensor.ndim == 2 and x_obs_tensor.shape == (3, self.num_steps):
            x_obs_2d = x_obs_tensor.T
        else:
            x_obs_2d = x_obs_tensor

        x_jax = jnp.asarray(x_obs_2d.detach().cpu().numpy())
        params_jax = jnp.asarray(parameters.detach().cpu().numpy())

        if self.mode == "ode":
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
        step_size: float = 0.1,
        **kwargs,
    ) -> torch.Tensor:
        r"""Sample from the true posterior $p(\theta \mid y_{1:T})$ using JAX MCMC.

        Mathematical Formulation:
        -------------------------
        The target posterior distribution is given by Bayes' rule:
            $$p(\theta \mid y_{1:T}) \propto p(\theta) \, p(y_{1:T} \mid \theta)$$
        where:
        - Prior: $\theta \sim \mathcal{N}(0, I_2)$ on the latent parameters.
        - Parameters: The 2 latent dimensions map to positive interaction rates via
          $\gamma = 2.0 \cdot \sigma(\theta_0) \in (0.0, 2.0)$ and $\beta = 2.0 \cdot \sigma(\theta_1) \in (0.0, 2.0)$.
        - Conditioning Observation: The likelihood conditions on the raw state trajectory
          observations $y_{1:T}$ (or subsampled observations), NOT on summary statistics.

        Inference Algorithms:
        ---------------------
        - ODE Mode: Metropolis-Hastings MCMC with Gaussian random walk proposals,
          evaluated on the deterministic trajectory likelihood.
        - SDE Mode: Gibbs Particle Marginal Metropolis-Hastings (PMMH) sampler:
          1. Alternates between parameter random-walk updates (with frozen particle key)
             and resampling of the particle filter random seeds.
          2. The particle filter provides an unbiased marginal likelihood estimate $\hat{p}(y_{1:T} \mid \theta)$,
             guaranteeing convergence to the exact stationary posterior distribution $p(\theta \mid y_{1:T})$
             (Andrieu et al., 2010).

        Parameters
        ----------
        x_obs : torch.Tensor or np.ndarray, optional
            Observed trajectory $y_{1:T}$. If None, defaults to the canonical reference
            trajectory generated at $\theta = 0$ with seed 42.
        num_samples : int, optional
            Total number of posterior parameter samples to return (default: 1000).
        n_chains : int, optional
            Number of parallel MCMC chains run concurrently in JAX (default: 20).
        burn_in : int, optional
            Number of initial warmup steps discarded per chain (default: 200).
        inner_thin : int, optional
            Sub-sampling thinning factor between saved MCMC states (default: 4).
        n_particles : int, optional
            Number of particles used in the SMC particle filter for SDE mode (default: 150).
        step_size : float, optional
            Random-walk proposal standard deviation per parameter dimension (default: 0.1).

        Returns
        -------
        torch.Tensor
            Posterior samples of latent parameters $\theta \in \mathbb{R}^2$ with shape `(num_samples, 2)`.
        """
        if x_obs is None:
            if self.preset == "sbibm":
                th_star = self.canonical_theta
                if self.summary == "binomial":
                    x_obs = self.simulate(th_star)
                else:
                    x_obs = self._solve(th_star.unsqueeze(0), u0=self.u0.unsqueeze(0))[0].T
            else:
                # Default observation generated from theta = 0 with seed 42
                x_obs = torch.tensor(
                    [
                        [4.0000000, 1.0000000, 0.0000000],
                        [3.7881465, 1.1751375, 0.0533900],
                        [3.5549567, 1.3938990, 0.1153643],
                        [3.2974174, 1.5862643, 0.1903032],
                        [3.0364633, 1.7333066, 0.2691698],
                        [2.7711196, 1.9520751, 0.3586609],
                        [2.5065825, 2.1024864, 0.4649960],
                        [2.2470863, 2.2430851, 0.5757274],
                        [1.9985336, 2.4134316, 0.6894845],
                        [1.7609777, 2.6007223, 0.8086434],
                        [1.5459836, 2.6992354, 0.9370170],
                    ],
                    dtype=torch.float32,
                )

        if isinstance(x_obs, np.ndarray):
            x_obs = np.array(x_obs, copy=True)
        x_obs_tensor = torch.as_tensor(x_obs, dtype=torch.float32)
        if x_obs_tensor.ndim == 2 and x_obs_tensor.shape[0] == 1:
            x_obs_tensor = x_obs_tensor.squeeze(0)

        if (
            x_obs_tensor.numel() == self.num_steps
            and self.summary == "binomial"
            and self.mode == "ode"
        ):
            x_jax = jnp.asarray(x_obs_tensor.flatten().detach().cpu().numpy())
        else:
            if x_obs_tensor.ndim == 1:
                x_obs_2d = x_obs_tensor.reshape(3, -1).T
            elif x_obs_tensor.ndim == 2 and x_obs_tensor.shape == (3, self.num_steps):
                x_obs_2d = x_obs_tensor.T
            else:
                x_obs_2d = x_obs_tensor
            x_jax = jnp.asarray(x_obs_2d.detach().cpu().numpy())
        rng_key = self._next_rng_key()

        num_samples_per_chain = math.ceil(num_samples / n_chains)

        if self.mode == "ode":
            samples_jax = self._jax_mcmc(
                x_jax,
                rng_key,
                n_chains=n_chains,
                num_samples_per_chain=num_samples_per_chain,
                burn_in=burn_in,
                inner_thin=inner_thin,
                step_size=step_size,
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
                step_size=step_size,
            )
        samples_np = np.array(samples_jax, copy=True)
        np.random.shuffle(samples_np)
        samples_torch = torch.from_numpy(samples_np)
        return samples_torch[:num_samples]
