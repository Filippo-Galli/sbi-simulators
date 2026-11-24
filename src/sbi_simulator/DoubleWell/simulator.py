"""Simulator for the Double-Well SDE model."""

import functools
import math
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
    build_particle_filter_ll,
)
from ..priors import NormalPrior
from ..summary_stats import trajectory_stats_interleaved


def _build_double_well_step_fn(
    dt: float = 0.005,
    inner_steps: int = 10,
    dim: int = 4,
):
    """Build single observation step transition function in JAX for Double-Well."""

    def drift(x, theta1, theta2):
        return theta1 * x + theta2 * (x**3)

    def integrate_step(x, key, theta1, theta2):
        xdt = drift(x, theta1, theta2)
        d_wt = jax.random.normal(key, shape=(dim,)) * jnp.sqrt(dt)
        return x + xdt * dt + d_wt

    def step_one(x, key, theta):
        theta1 = jnp.exp(0.5 * theta[0])
        theta2 = -jnp.exp(0.5 * theta[1])
        keys = jax.random.split(key, inner_steps)

        def inner_loop(i, curr_x):
            return integrate_step(curr_x, keys[i], theta1, theta2)

        return jax.lax.fori_loop(0, inner_steps, inner_loop, x)

    return step_one


def _default_double_well_u0(dim: int) -> tuple[float, ...]:
    return tuple(([-0.5, 0.5] * ((dim + 1) // 2))[:dim])


def _build_double_well_factorized_pf_ll(
    dt: float,
    inner_steps: int,
    dim: int,
    obs_sigma: float,
):
    """Build a factorized 1D Bootstrap PF log-likelihood over ``dim`` independent coords."""
    step_1d = _build_double_well_step_fn(dt=dt, inner_steps=inner_steps, dim=1)
    pf_1d = build_particle_filter_ll(step_1d, obs_dim=1, obs_sigma=obs_sigma)

    def factorized_pf_ll(theta, x_obs, key, n_particles: int = 150):
        # x_obs has shape (num_steps, dim) -> (dim, num_steps, 1)
        x_per_dim = x_obs.T[:, :, None]
        keys = jax.random.split(key, dim)
        lls = jax.vmap(lambda xd, k: pf_1d(theta, xd, k, n_particles=n_particles))(
            x_per_dim, keys
        )
        total = jnp.sum(lls)
        return jnp.where(jnp.isfinite(total), total, -jnp.inf)

    return jax.jit(factorized_pf_ll, static_argnames=["n_particles"])


def _build_double_well_bridge_ll(
    dt: float,
    inner_steps: int,
    n_bridges: int = 64,
):
    """Build deterministic 1D Brownian-bridge log-likelihood for noiseless trajectories."""
    k_steps = int(inner_steps)
    total_var = k_steps * dt
    log_norm_bm = -0.5 * jnp.log(2.0 * jnp.pi * total_var)
    log_n = jnp.log(float(n_bridges))
    # Antithetic standard normal draws shared across transitions for ultra-low variance
    half_b = max(n_bridges // 2, 1)
    base_key = jax.random.PRNGKey(20250101)

    def bridge_transition_ll(x_start, x_end, theta, z_draws):
        theta1 = jnp.exp(0.5 * theta[0])
        theta2 = -jnp.exp(0.5 * theta[1])

        if k_steps == 1:
            mu = theta1 * x_start + theta2 * (x_start**3)
            mean = x_start + mu * dt
            return -0.5 * ((x_end - mean) ** 2) / dt - 0.5 * jnp.log(2.0 * jnp.pi * dt)

        log_bm = -0.5 * ((x_end - x_start) ** 2) / total_var + log_norm_bm

        def one_bridge(z_seq):
            def body(carry, inp):
                z, log_radon = carry
                k_idx, eps = inp
                rem = k_steps - k_idx
                mean_q = z + (x_end - z) / rem
                std_q = jnp.sqrt(dt * (rem - 1.0) / rem)
                z_next = mean_q + std_q * eps
                mu = theta1 * z + theta2 * (z**3)
                dz = z_next - z
                lr = mu * dz - 0.5 * (mu**2) * dt
                return (z_next, log_radon + lr), None

            (z_last, log_radon), _ = jax.lax.scan(
                body, (x_start, 0.0), (jnp.arange(k_steps - 1), z_seq)
            )
            mu_last = theta1 * z_last + theta2 * (z_last**3)
            dz_last = x_end - z_last
            return log_radon + (mu_last * dz_last - 0.5 * (mu_last**2) * dt)

        lrs = jax.vmap(one_bridge)(z_draws)
        lrs = jnp.where(jnp.isfinite(lrs), lrs, -jnp.inf)
        return log_bm + jax.scipy.special.logsumexp(lrs) - log_n

    def noiseless_ll(theta, x_obs):
        starts = x_obs[:-1].reshape(-1)
        ends = x_obs[1:].reshape(-1)
        n_trans = starts.shape[0]
        if k_steps == 1:
            z_all = jnp.zeros((n_trans, 1, 1), dtype=jnp.float32)
        else:
            z_half = jax.random.normal(
                base_key, shape=(n_trans, half_b, k_steps - 1), dtype=jnp.float32
            )
            z_all = jnp.concatenate([z_half, -z_half], axis=1)[:, :n_bridges, :]
        lls = jax.vmap(
            lambda s, e, zd: bridge_transition_ll(s, e, theta, zd)
        )(starts, ends, z_all)
        total = jnp.sum(lls)
        return jnp.where(jnp.isfinite(total), total, -jnp.inf)

    return jax.jit(noiseless_ll)


@functools.lru_cache(maxsize=32)
def _get_double_well_jax_bundle(
    dt: float,
    inner_steps: int,
    dim: int,
    num_steps: int,
    obs_sigma: float,
    use_noiseless_ll: bool = False,
):
    """Build and cache compiled JAX solver, likelihood, and MCMC/PMMH sampler."""
    step_fn = _build_double_well_step_fn(dt=dt, inner_steps=inner_steps, dim=dim)
    solve_batch, _ = build_jax_trajectory_solver(
        step_fn,
        num_steps=num_steps,
        default_x0=_default_double_well_u0(dim),
        clip_non_negative=False,
    )
    if use_noiseless_ll:
        ll_fn = _build_double_well_bridge_ll(dt=dt, inner_steps=inner_steps)
        v_ll_fn = build_batched_ll(ll_fn, stochastic=False)
        jax_mcmc = build_mh_mcmc_sampler(
            ll_fn,
            prior=NormalPrior.isotropic(2),
            init_strategy="sir_pool",
            rescue_stuck_chains=True,
        )
        return solve_batch, v_ll_fn, jax_mcmc

    pf_ll_fn = _build_double_well_factorized_pf_ll(
        dt=dt,
        inner_steps=inner_steps,
        dim=dim,
        obs_sigma=obs_sigma,
    )
    v_pf_ll_fn = build_batched_ll(pf_ll_fn, stochastic=True)
    jax_mcmc = build_gibbs_pmmh_sampler(pf_ll_fn, NormalPrior.isotropic(2))
    return solve_batch, v_pf_ll_fn, jax_mcmc


class DoubleWell(DynamicalSimulator):
    r"""Double-Well Potential Stochastic Differential Equation (SDE) Simulator.

    Mathematical Model:
    -------------------
    The state is a $D$-dimensional vector $x(t) \in \mathbb{R}^D$ (default $D=4$), representing
    particles moving in a bistable symmetric potential well with Brownian noise excitation.

    1. Prior Distribution:
       The parameters are represented in an unconstrained latent space $\tilde{\theta} \in \mathbb{R}^2$:
           $$\tilde{\theta} = [\tilde{\theta}_0, \tilde{\theta}_1]^T \sim \mathcal{N}(0, I_2)$$

    2. Parameter Transformation:
       $$\theta_1 = \exp(0.5 \cdot \tilde{\theta}_0) > 0, \qquad \theta_2 = -\exp(0.5 \cdot \tilde{\theta}_1) < 0$$
       defining the bistable potential $V(x) = -\frac{\theta_1}{2} x^2 - \frac{\theta_2}{4} x^4$.

    3. Continuous Dynamics:
       $$dx_t = (\theta_1 x_t + \theta_2 x_t^3) \, dt + dW_t$$

    4. Summary Modes:
       - `None`: Raw trajectory of shape $(D \times \text{num\_steps},)$.
       - `"subsample"`: Trajectory with additive Gaussian observation noise $\sigma_{\text{obs}}$.
       - `"stats"` / `"summary_stats"`: $4D$ summary statistics (mean, log-var, ac1, ac2 per dim).
       - `"two_step"`: Transition pairs $(x_t, x_{t+1})$ of shape $(2D,)$ for Markov SBI.
    """

    def __init__(
        self,
        dim_parameters: int = 2,
        seed: int = 42,
        D: int = 4,
        dt: float = 0.005,
        inner_steps: int = 10,
        num_steps: int = 11,
        obs_sigma: float = 0.05,
        summary: str | None = None,
        **kwargs,
    ):
        if dim_parameters != 2:
            raise ValueError(
                f"DoubleWell requires dim_parameters=2, got {dim_parameters}."
            )
        if D < 1:
            raise ValueError(f"DoubleWell requires D >= 1, got {D}.")

        self.D = D
        self.dt = float(dt)
        self.inner_steps = int(inner_steps)

        u0 = torch.tensor(_default_double_well_u0(D), dtype=torch.float32)
        dim_data = self._resolve_summary_dim(
            summary,
            dim_raw=int(D * num_steps),
            dim_stats=4 * D,
            dim_two_step=2 * D,
        )

        super().__init__(
            seed=seed,
            dim_parameters=dim_parameters,
            dim_data=dim_data,
            state_dim=D,
            num_steps=num_steps,
            obs_sigma=obs_sigma,
            summary=summary,
            u0=u0,
            **kwargs,
        )

        self.prior_spec = NormalPrior.isotropic(dim_parameters)
        self.prior_dist = self.prior_spec.to_pyro()
        self._use_noiseless_ll = self.summary is None

        self._jax_solver_batch, self._v_pf_ll_fn, self._jax_mcmc = (
            _get_double_well_jax_bundle(
                self.dt,
                self.inner_steps,
                self.D,
                self.num_steps,
                self.obs_sigma,
                self._use_noiseless_ll,
            )
        )

    def get_prior_x0(self) -> pdist.Distribution:
        """Prior over initial condition ``x0`` in ``R^D``."""
        return pdist.Uniform(
            low=torch.full((self.D,), -2.5, dtype=torch.float32),
            high=torch.full((self.D,), 2.5, dtype=torch.float32),
        ).to_event(1)

    @staticmethod
    @override
    def compute_summary_stats(us: torch.Tensor) -> torch.Tensor:
        """Compute ``4 * D`` summary statistics from ``(num_samples, D, T)`` trajectories."""
        return trajectory_stats_interleaved(us)

    def stepper(
        self,
        x: torch.Tensor,
        parameters: torch.Tensor,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Simulate a single observation-step transition for the Double-Well SDE."""
        x_t, params, is_1d = self._broadcast_stepper_inputs(x, parameters)

        theta1 = torch.exp(0.5 * params[:, 0:1])
        theta2 = -torch.exp(0.5 * params[:, 1:2])

        step_dt = float(self.dt * self.inner_steps if dt is None else dt)
        n_substeps = max(round(step_dt / self.dt), 1)
        actual_h = step_dt / n_substeps
        sqrt_h = math.sqrt(actual_h)

        curr_x = x_t.clone()
        for _ in range(n_substeps):
            xdt = theta1 * curr_x + theta2 * (curr_x**3)
            curr_x = curr_x + xdt * actual_h + torch.randn_like(curr_x) * sqrt_h

        return curr_x.squeeze(0) if is_1d else curr_x

    @override
    def get_prior(self) -> pdist.Distribution:
        return self.prior_dist

    def log_likelihood(
        self,
        parameters: torch.Tensor | np.ndarray,
        x_obs: torch.Tensor | np.ndarray,
        n_particles: int = 150,
    ) -> torch.Tensor:
        """Evaluate marginal log-likelihood ``log p(y_{1:T} | theta)``.

        Uses exact 1D Brownian-bridge importance sampling when ``summary is None``
        (noiseless trajectories), and a factorized 1D bootstrap particle filter when
        observation noise ``obs_sigma`` is present (e.g. ``summary="subsample"``).
        """
        params, _ = self._prepare_parameters(parameters)
        x_jax = self._format_trajectory_obs(x_obs)
        params_jax = jnp.asarray(params.detach().cpu().numpy(), dtype=jnp.float32)

        if self._use_noiseless_ll:
            lls_jax = self._v_pf_ll_fn(params_jax, x_jax)
        else:
            keys = jax.random.split(self._next_rng_key(), params.shape[0])
            lls_jax = self._v_pf_ll_fn(params_jax, x_jax, keys, n_particles=n_particles)
        return torch.from_numpy(np.array(lls_jax, copy=True))

    @override
    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        n_chains: int | None = None,
        burn_in: int | None = None,
        inner_thin: int | None = None,
        n_particles: int = 150,
        step_size: float | None = None,
        n_sir: int | None = None,
        adapt_warmup: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from ``p(theta | y_{1:T})`` via JAX MH MCMC (noiseless) or factorized 1D PMMH."""
        if x_obs is None:
            # Default observation generated from theta = [0.0, 0.0] with seed 42
            x_obs = torch.tensor(
                [
                    [-0.5000, 0.5000, -0.5000, 0.5000],
                    [-0.6590, 0.7088, -0.3071, 0.5831],
                    [-0.5456, 0.7847, -0.6824, 0.5861],
                    [-0.7785, 0.7998, -0.5668, 0.9425],
                    [-0.7169, 0.6416, -0.8609, 0.9332],
                    [-0.9724, 0.8284, -1.0258, 0.5080],
                    [-0.3580, 0.8494, -0.4605, 0.3421],
                    [-0.1946, 0.7874, -0.2469, 0.1780],
                    [-0.1787, 0.7089, -0.3305, 0.1740],
                    [0.0391, 0.6627, -1.0107, 0.1377],
                    [0.2567, 0.3869, -0.9646, 0.4792],
                ],
                dtype=torch.float32,
            )

        x_obs_jax = self._format_trajectory_obs(x_obs)

        if self._use_noiseless_ll:
            eff_n_chains = 40 if n_chains is None else int(n_chains)
            eff_burn_in = 400 if burn_in is None else int(burn_in)
            eff_inner_thin = 20 if inner_thin is None else int(inner_thin)
            eff_step_size = 0.2 if step_size is None else float(step_size)
            eff_n_sir = 512 if n_sir is None else int(n_sir)
            num_samples_per_chain = math.ceil(num_samples / eff_n_chains)

            samples_jax = self._jax_mcmc(
                x_obs_jax,
                self._next_rng_key(),
                n_chains=eff_n_chains,
                num_samples_per_chain=num_samples_per_chain,
                burn_in=eff_burn_in,
                inner_thin=eff_inner_thin,
                step_size=eff_step_size,
                n_sir=eff_n_sir,
                adapt_warmup=adapt_warmup,
            )
            return self._finalize_mcmc_samples(samples_jax, num_samples)

        eff_n_chains = 32 if n_chains is None else int(n_chains)
        eff_burn_in = 120 if burn_in is None else int(burn_in)
        eff_inner_thin = 10 if inner_thin is None else int(inner_thin)
        eff_step_size = 0.35 if step_size is None else float(step_size)
        eff_n_sir = 50 if n_sir is None else int(n_sir)
        num_samples_per_chain = math.ceil(num_samples / eff_n_chains)

        samples_jax = self._jax_mcmc(
            x_obs=x_obs_jax,
            rng_key=self._next_rng_key(),
            n_chains=eff_n_chains,
            num_samples_per_chain=num_samples_per_chain,
            burn_in=eff_burn_in,
            inner_thin=eff_inner_thin,
            n_particles=n_particles,
            step_size=eff_step_size,
            n_sir=eff_n_sir,
            adapt_warmup=adapt_warmup,
        )
        return self._finalize_mcmc_samples(samples_jax, num_samples)
