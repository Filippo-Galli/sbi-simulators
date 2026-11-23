"""Simulator for the Linear Stochastic Differential Equation (SDE) model."""

import functools
import math
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pyro.distributions as pdist
import torch
from jax.scipy.linalg import expm

from ..dynamical import DynamicalSimulator
from ..mcmc import build_batched_ll, build_mh_mcmc_sampler
from ..priors import NormalPrior
from ..summary_stats import trajectory_stats_2d_linear


def matrix_fraction_decomposition(dt: float, A: jax.Array, B: jax.Array):
    r"""Van Loan (1978) matrix fraction decomposition for linear SDEs.

    Computes transition matrix ``Phi = exp(A * dt)`` and diffusion covariance
    ``Q = \int_0^{dt} exp(A * s) B B^T exp(A^T * s) ds`` in closed form.
    """
    d = A.shape[-1]
    blockmatrix = jnp.block([[A, jnp.dot(B, B.T)], [jnp.zeros((d, d)), -A.T]])
    M = expm(blockmatrix * dt)
    Phi = M[:d, :d]
    Q = jnp.dot(M[:d, d:], Phi.T)
    Q = 0.5 * (Q + Q.T) + 1e-6 * jnp.eye(d)
    return Phi, Q


def _build_linear_sde_system(
    variant: str = "periodic",
    sigma1: float = 0.1,
    sigma2: float = 0.1,
    d: int = 2,
):
    """Build parameter-to-matrix mappings ``(A, B)`` for LinearSDE."""
    if variant == "periodic":

        def get_AB(theta):
            A = jnp.array([[0.0, -(theta[0] ** 2)], [theta[1] ** 2, 0.0]])
            B = jnp.array([[sigma1, 0.0], [0.0, sigma2]])
            return A, B

    elif variant == "general":
        eye_d = jnp.eye(d)

        def get_AB(theta):
            A = 0.5 * theta[: d**2].reshape((d, d)) - 2.0 * eye_d
            B = 0.5 * theta[d**2 :].reshape((d, d)) + 0.5 * eye_d
            return A, B

    else:
        raise ValueError(f"Unknown LinearSDE variant: {variant}")

    return get_AB


def _build_kalman_log_likelihood(
    get_AB,
    dt: float = 0.1,
    d: int = 2,
    obs_sigma: float = 0.01,
):
    """Build exact single-Cholesky JAX Kalman filter log-likelihood function."""
    R = (obs_sigma**2) * jnp.eye(d)
    init_cov = 1e-5 * jnp.eye(d)
    log_2pi = d * jnp.log(2.0 * jnp.pi)

    def single_log_likelihood(theta, x_obs):
        A, B = get_AB(theta)
        Phi, Q = matrix_fraction_decomposition(dt, A, B)
        mu0 = x_obs[0]

        def kf_step(carry, y):
            m, P = carry
            m_pred = jnp.dot(Phi, m)
            P_pred = jnp.dot(Phi, jnp.dot(P, Phi.T)) + Q
            P_pred = 0.5 * (P_pred + P_pred.T)

            S = P_pred + R
            S = 0.5 * (S + S.T)
            r = y - m_pred

            L_S = jnp.linalg.cholesky(S)
            V = jax.scipy.linalg.solve_triangular(L_S, P_pred, lower=True)
            sol = jax.scipy.linalg.solve_triangular(L_S, r, lower=True)

            m_upd = m_pred + jnp.dot(V.T, sol)
            P_upd = P_pred - jnp.dot(V.T, V)
            P_upd = 0.5 * (P_upd + P_upd.T)

            logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L_S)))
            ll_step = -0.5 * (log_2pi + logdet + jnp.sum(sol**2))
            return (m_upd, P_upd), ll_step

        _, lls = jax.lax.scan(kf_step, (mu0, init_cov), x_obs[1:])
        total_ll = jnp.sum(lls)
        return jnp.where(jnp.isfinite(total_ll), total_ll, -jnp.inf)

    return jax.jit(single_log_likelihood)


def _build_linear_sde_jax_solvers(
    get_AB,
    dt: float,
    d: int,
    num_steps: int,
    default_x0: tuple[float, ...],
):
    """Build JIT-compiled batched trajectory solver and batched single-step stepper."""
    base_x0 = jnp.asarray(default_x0, dtype=jnp.float32)

    def solve_single(rng_key, theta, x0=None):
        u0 = base_x0 if x0 is None else x0
        A, B = get_AB(theta)
        Phi, Q = matrix_fraction_decomposition(dt, A, B)
        L = jnp.linalg.cholesky(Q)

        def step_fn(x, k):
            noise = jax.random.normal(k, shape=(d,))
            x_next = jnp.dot(Phi, x) + jnp.dot(L, noise)
            return x_next, x_next

        keys = jax.random.split(rng_key, num_steps - 1)
        _, xs = jax.lax.scan(step_fn, u0, keys)
        full_traj = jnp.concatenate([u0[None, :], xs], axis=0)  # (num_steps, d)
        return full_traj.T  # (d, num_steps)

    solve_batch_fn = jax.vmap(solve_single, in_axes=(0, 0, 0))

    def solve_batch(rng_keys, thetas, x0s=None):
        if x0s is None:
            x0s = jnp.broadcast_to(base_x0, (thetas.shape[0], d))
        return solve_batch_fn(rng_keys, thetas, x0s)

    def single_step(th, curr_x, k, step_dt):
        A, B = get_AB(th)
        Phi, Q = matrix_fraction_decomposition(step_dt, A, B)
        L = jnp.linalg.cholesky(Q)
        noise = jax.random.normal(k, shape=(d,))
        return jnp.dot(Phi, curr_x) + jnp.dot(L, noise)

    step_batch_jit = jax.jit(jax.vmap(single_step, in_axes=(0, 0, 0, None)))
    return jax.jit(solve_batch), step_batch_jit


@functools.lru_cache(maxsize=32)
def _get_linear_sde_jax_bundle(
    variant: str,
    d: int,
    step_dt: float,
    num_steps: int,
    sigma1: float,
    sigma2: float,
    obs_sigma: float,
    default_x0: tuple[float, ...],
    dim_parameters: int,
):
    """Build and cache compiled LinearSDE JAX solver, stepper, Kalman LL, and MCMC."""
    get_AB = _build_linear_sde_system(
        variant=variant,
        sigma1=sigma1,
        sigma2=sigma2,
        d=d,
    )
    solve_batch, step_batch_jit = _build_linear_sde_jax_solvers(
        get_AB,
        dt=step_dt,
        d=d,
        num_steps=num_steps,
        default_x0=default_x0,
    )
    kf_ll_fn = _build_kalman_log_likelihood(
        get_AB,
        dt=step_dt,
        d=d,
        obs_sigma=obs_sigma,
    )
    v_kf_ll_fn = build_batched_ll(kf_ll_fn, stochastic=False)
    jax_mcmc = build_mh_mcmc_sampler(
        kf_ll_fn,
        prior=NormalPrior.isotropic(dim_parameters),
        init_strategy="prior",
    )
    return solve_batch, step_batch_jit, v_kf_ll_fn, jax_mcmc


class LinearSDE(DynamicalSimulator):
    r"""Linear Gaussian Stochastic Differential Equation (SDE) Simulator.

    Mathematical Model:
    -------------------
    The state is a $d$-dimensional vector $x(t) \in \mathbb{R}^d$ ($d=2$ by default) evolving
    under a linear continuous-time stochastic differential equation:
        $$dx_t = A(\theta) x_t \, dt + B(\theta) \, dW_t$$

    1. Prior Distribution & Variants:
       - `"periodic"`: $\theta \in \mathbb{R}^2 \sim \mathcal{N}(0, I_2)$.
         $$A(\theta) = \begin{pmatrix} 0 & -\theta_0^2 \\ \theta_1^2 & 0 \end{pmatrix}, \quad B = \text{diag}(\sigma_1, \sigma_2)$$
       - `"general"`: $\theta \in \mathbb{R}^{2d^2} \sim \mathcal{N}(0, I_{2d^2})$.
         $$A(\theta) = 0.5 \cdot \text{reshape}(\theta_{0:d^2}, (d, d)) - 2 I_d$$
         $$B(\theta) = 0.5 \cdot \text{reshape}(\theta_{d^2:2d^2}, (d, d)) + 0.5 I_d$$

    2. Exact Continuous-to-Discrete Discretization:
       Computed in closed form via Van Loan matrix fraction decomposition over $\Delta t_{\text{obs}}$.

    3. Posterior Distribution:
       Exact marginal likelihood $p(y_{1:T} \mid \theta)$ via the Kalman Filter,
       sampled via parallel Metropolis-Hastings MCMC in JAX.
    """

    def __init__(
        self,
        dim_parameters: int = 2,
        seed: int = 42,
        variant: str = "periodic",
        d: int = 2,
        dt: float = 0.005,
        inner_steps: int = 20,
        num_steps: int = 11,
        sigma1: float = 0.1,
        sigma2: float = 0.1,
        obs_sigma: float = 0.01,
        summary: str | None = None,
        **kwargs,
    ):
        self.variant = variant
        self.d = int(d)
        self.dt = float(dt)
        self.inner_steps = int(inner_steps)
        self.step_dt = float(dt * inner_steps)
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)

        if variant == "periodic":
            expected_dim = 2
        elif variant == "general":
            expected_dim = 2 * (self.d**2)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        if dim_parameters != expected_dim:
            if dim_parameters in (2, 2 * (self.d**2)):
                self.variant = "periodic" if dim_parameters == 2 else "general"
            else:
                raise ValueError(
                    f"LinearSDE variant='{self.variant}' expects dim_parameters={expected_dim}, "
                    f"got {dim_parameters}."
                )

        if summary in ("stats", "summary_stats") and self.d != 2:
            raise ValueError(
                f"LinearSDE summary='{summary}' requires state dimension d=2, got d={self.d}."
            )

        u0_tuple = (-0.5, 0.5) if self.d == 2 else (0.0,) * self.d
        u0 = torch.tensor(u0_tuple, dtype=torch.float32)
        dim_data = self._resolve_summary_dim(
            summary,
            dim_raw=int(self.d * num_steps),
            dim_stats=9,
            dim_two_step=2 * self.d,
        )

        super().__init__(
            seed=seed,
            dim_parameters=dim_parameters,
            dim_data=dim_data,
            state_dim=self.d,
            num_steps=num_steps,
            obs_sigma=obs_sigma,
            summary=summary,
            u0=u0,
            **kwargs,
        )

        self.prior_spec = NormalPrior.isotropic(self.dim_parameters)
        self.prior_dist = self.prior_spec.to_pyro()

        (
            self._jax_solver_batch,
            self._jax_step_batch,
            self._v_kf_ll_fn,
            self._jax_mcmc,
        ) = _get_linear_sde_jax_bundle(
            self.variant,
            self.d,
            self.step_dt,
            self.num_steps,
            self.sigma1,
            self.sigma2,
            self.obs_sigma,
            u0_tuple,
            self.dim_parameters,
        )

    def get_prior_x0(self) -> pdist.Distribution:
        """Prior over initial condition ``x0`` in ``R^d``."""
        return pdist.Uniform(
            low=torch.full((self.d,), -2.0, dtype=torch.float32),
            high=torch.full((self.d,), 2.0, dtype=torch.float32),
        ).to_event(1)

    @staticmethod
    @override
    def compute_summary_stats(us: torch.Tensor) -> torch.Tensor:
        """Compute 9 summary statistics from ``(num_samples, 2, T)`` trajectory tensor."""
        return trajectory_stats_2d_linear(us)

    def stepper(
        self,
        x: torch.Tensor,
        parameters: torch.Tensor,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Simulate a single observation-step transition for LinearSDE."""
        x_t, params, is_1d = self._broadcast_stepper_inputs(x, parameters)

        params_jax = jnp.asarray(params.detach().cpu().numpy(), dtype=jnp.float32)
        x_jax = jnp.asarray(x_t.detach().cpu().numpy(), dtype=jnp.float32)
        step_dt = float(self.step_dt if dt is None else dt)

        keys = jax.random.split(self._next_rng_key(), params.shape[0])
        next_x_jax = self._jax_step_batch(params_jax, x_jax, keys, step_dt)
        next_x = torch.from_numpy(np.array(next_x_jax, copy=True))
        return next_x.squeeze(0) if is_1d else next_x

    @override
    def get_prior(self) -> pdist.Distribution:
        return self.prior_dist

    def log_likelihood(
        self,
        parameters: torch.Tensor | np.ndarray,
        x_obs: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Evaluate exact marginal log-likelihood ``log p(y_{1:T} | theta)`` via Kalman Filter."""
        params, _ = self._prepare_parameters(parameters)
        x_jax = self._format_trajectory_obs(x_obs)
        params_jax = jnp.asarray(params.detach().cpu().numpy(), dtype=jnp.float32)

        lls_jax = self._v_kf_ll_fn(params_jax, x_jax)
        return torch.from_numpy(np.array(lls_jax, copy=True))

    @override
    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        n_chains: int = 20,
        burn_in: int = 200,
        step_size: float = 0.1,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the exact posterior ``p(theta | y_{1:T})`` via JAX MCMC."""
        if x_obs is None:
            # Default observation generated from theta = [1.0, 1.0] with seed 42
            x_obs = torch.tensor(
                [
                    [-0.5000, 0.5000],
                    [-0.5450, 0.4322],
                    [-0.5663, 0.4009],
                    [-0.5898, 0.3610],
                    [-0.6318, 0.2570],
                    [-0.6336, 0.1981],
                    [-0.6570, 0.0710],
                    [-0.6689, -0.0152],
                    [-0.6549, -0.0822],
                    [-0.6389, -0.1520],
                    [-0.6631, -0.1975],
                ],
                dtype=torch.float32,
            )

        x_obs_jax = self._format_trajectory_obs(x_obs)
        num_samples_per_chain = math.ceil(num_samples / n_chains)

        samples_jax = self._jax_mcmc(
            x_obs=x_obs_jax,
            rng_key=self._next_rng_key(),
            n_chains=n_chains,
            num_samples_per_chain=num_samples_per_chain,
            burn_in=burn_in,
            inner_thin=1,
            step_size=step_size,
        )
        return self._finalize_mcmc_samples(samples_jax, num_samples)
