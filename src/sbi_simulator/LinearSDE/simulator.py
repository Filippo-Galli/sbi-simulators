"""Simulator for the Linear Stochastic Differential Equation (SDE) model."""

import math
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pyro.distributions as pdist
import torch
from jax.scipy.linalg import expm

from ..compositional import sample_two_step
from ..Simulator import Simulator


def matrix_fraction_decomposition(dt: float, A: jax.Array, B: jax.Array):
    r"""Van Loan (1978) matrix fraction decomposition for linear SDEs.

    Computes transition matrix Phi = exp(A * dt) and diffusion covariance
    Q = \int_0^{dt} exp(A * s) B B^T exp(A^T * s) ds in closed form.
    """
    d = A.shape[-1]
    blockmatrix = jnp.block([[A, jnp.dot(B, B.T)], [jnp.zeros((d, d)), -A.T]])
    M = expm(blockmatrix * dt)
    Phi = M[:d, :d]
    Q = jnp.dot(M[:d, d:], Phi.T)
    Q = 0.5 * (Q + Q.T) + 1e-6 * jnp.eye(d)
    return Phi, Q


def _build_linear_sde_system(variant: str = "periodic", sigma1: float = 0.1, sigma2: float = 0.1, d: int = 2):
    """Build parameter-to-matrix mappings (A, B) for LinearSDE."""
    if variant == "periodic":

        def get_AB(theta):
            A = jnp.array([[0.0, -(theta[0] ** 2)], [theta[1] ** 2, 0.0]])
            B = jnp.array([[sigma1, 0.0], [0.0, sigma2]])
            return A, B

    elif variant == "general":

        def get_AB(theta):
            theta_sub = 0.5 * theta[: d**2].reshape((d, d)) - 2.0 * jnp.eye(d)
            B_sub = 0.5 * theta[d**2 :].reshape((d, d)) + 0.5 * jnp.eye(d)
            return theta_sub, B_sub

    else:
        raise ValueError(f"Unknown LinearSDE variant: {variant}")

    return get_AB


def _build_kalman_log_likelihood(get_AB, dt: float = 0.1, d: int = 2, obs_sigma: float = 0.01):
    """Build exact JAX Kalman filter log-likelihood function."""
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

            S = P_pred + R
            S = 0.5 * (S + S.T)
            r = y - m_pred
            K = jnp.linalg.solve(S.T, P_pred.T).T
            m_upd = m_pred + jnp.dot(K, r)
            P_upd = P_pred - jnp.dot(K, jnp.dot(S, K.T))
            P_upd = 0.5 * (P_upd + P_upd.T)

            L_S = jnp.linalg.cholesky(S)
            logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L_S)))
            sol = jnp.linalg.solve(L_S, r)
            quad = jnp.sum(sol**2)
            ll_step = -0.5 * (log_2pi + logdet + quad)
            return (m_upd, P_upd), ll_step

        _, lls = jax.lax.scan(kf_step, (mu0, init_cov), x_obs[1:])
        total_ll = jnp.sum(lls)
        return jnp.where(jnp.isnan(total_ll), -1e9, total_ll)

    return jax.jit(single_log_likelihood)


def _build_linear_sde_sampler(kalman_ll_fn, dim_parameters: int):
    """Build fast JAX MCMC sampler using exact Kalman filter marginal likelihood."""
    log_prior_norm = (dim_parameters / 2.0) * jnp.log(2.0 * jnp.pi)

    def log_joint(theta, x_obs):
        prior_lp = -0.5 * jnp.sum(theta**2) - log_prior_norm
        return prior_lp + kalman_ll_fn(theta, x_obs)

    v_log_joint = jax.vmap(log_joint, in_axes=(0, None))

    def run_mcmc(
        x_obs,
        rng_key,
        n_chains: int = 20,
        num_samples_per_chain: int = 500,
        burn_in: int = 200,
        step_size: float = 0.1,
    ):
        k_init, k_mcmc = jax.random.split(rng_key)
        curr_thetas = jax.random.normal(k_init, shape=(n_chains, dim_parameters))
        curr_lp = v_log_joint(curr_thetas, x_obs)

        def step_fn(carry, subkey):
            th, lp = carry
            k_prop, k_acc = jax.random.split(subkey)
            prop_th = th + jax.random.normal(k_prop, shape=th.shape) * step_size
            prop_lp = v_log_joint(prop_th, x_obs)
            alpha = jnp.minimum(1.0, jnp.exp(prop_lp - lp))
            acc = jax.random.uniform(k_acc, shape=(n_chains,)) < alpha
            new_th = jnp.where(acc[:, None], prop_th, th)
            new_lp = jnp.where(acc, prop_lp, lp)
            return (new_th, new_lp), new_th

        total_steps = burn_in + num_samples_per_chain
        keys = jax.random.split(k_mcmc, total_steps)
        _, all_samples = jax.lax.scan(step_fn, (curr_thetas, curr_lp), keys)
        post_burnin = all_samples[burn_in:]
        return post_burnin.transpose(1, 0, 2).reshape(-1, dim_parameters)

    return jax.jit(
        run_mcmc,
        static_argnames=["n_chains", "num_samples_per_chain", "burn_in"],
    )


class LinearSDE(Simulator):
    r"""Linear Gaussian Stochastic Differential Equation (SDE) simulator.

    Formulation:
        dx = A(\theta) x dt + B(\theta) dW

    Variants:
        - "periodic": 2D state, 2D parameter vector \theta \in R^2.
          A(\theta) = [[0, -\theta_0^2], [\theta_1^2, 0]], B = diag(\sigma_1, \sigma_2).
          Represents a stochastic harmonic oscillator with rotation frequency determined by \theta.
        - "general": 2D state, 8D parameter vector \theta \in R^8 with unconstrained linear drift and diffusion.

    The model supports exact closed-form matrix fraction decomposition and exact Kalman filter log-likelihood.
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
    ):
        self.variant = variant
        self.d = d
        self.dt = dt
        self.inner_steps = inner_steps
        self.step_dt = float(dt * inner_steps)  # Effective observation interval (default 0.1)
        self.num_steps = num_steps
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.obs_sigma = obs_sigma

        if variant == "periodic":
            expected_dim = 2
        elif variant == "general":
            expected_dim = 2 * (d**2)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.dim_parameters = dim_parameters
        if dim_parameters != expected_dim:
            # Allow flexible dim_parameters if user passed explicit expected_dim
            if dim_parameters in (2, 2 * (d**2)):
                if dim_parameters == 2:
                    self.variant = "periodic"
                else:
                    self.variant = "general"
            else:
                raise ValueError(
                    f"LinearSDE variant='{self.variant}' expects dim_parameters={expected_dim}, "
                    f"got {dim_parameters}."
                )

        self.dim_data_raw = int(d * num_steps)

        if summary is None or summary == "subsample":
            self.dim_data = self.dim_data_raw
        elif summary in ("stats", "summary_stats"):
            self.dim_data = 9
        elif summary == "two_step":
            self.dim_data = 2 * d
        else:
            raise NotImplementedError(f"Unknown summary mode: {summary}")
        self.summary = summary

        super().__init__(seed, self.dim_parameters, self.dim_data)

        self.u0 = torch.tensor([-0.5, 0.5] if d == 2 else [0.0] * d, dtype=torch.float32)

        # Prior setup: Standard Normal N(0, I_D)
        self.prior_loc = [0.0] * self.dim_parameters
        self.prior_scale = [1.0] * self.dim_parameters
        self.prior_dist = pdist.Normal(
            loc=torch.zeros(self.dim_parameters, dtype=torch.float32),
            scale=torch.ones(self.dim_parameters, dtype=torch.float32),
        ).to_event(1)

        # Build JAX operators
        self._get_AB = _build_linear_sde_system(
            variant=self.variant,
            sigma1=self.sigma1,
            sigma2=self.sigma2,
            d=self.d,
        )
        self._kf_ll_fn = _build_kalman_log_likelihood(
            self._get_AB,
            dt=self.step_dt,
            d=self.d,
            obs_sigma=self.obs_sigma,
        )
        self._jax_mcmc = _build_linear_sde_sampler(
            self._kf_ll_fn,
            dim_parameters=self.dim_parameters,
        )

        # Build JAX trajectory solver
        self._jax_solver_batch = self._build_jax_solver()

    def _build_jax_solver(self):
        get_AB = self._get_AB
        dt = self.step_dt
        d = self.d
        num_steps = self.num_steps
        default_x0 = jnp.asarray(self.u0.numpy())

        def solve_single(rng_key, theta, x0=None):
            u0 = default_x0 if x0 is None else x0
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
                x0s = jnp.broadcast_to(default_x0, (thetas.shape[0], d))
            return solve_batch_fn(rng_keys, thetas, x0s)

        return jax.jit(solve_batch)

    def _solve(
        self,
        parameters: torch.Tensor,
        u0: torch.Tensor | None = None,
        rng_key: jax.Array | None = None,
    ) -> torch.Tensor:
        """Batched exact discrete simulation accelerated via JAX Matrix Fraction Decomposition.

        parameters: (num_samples, dim_parameters)
        u0: optional (num_samples, d)
        returns: (num_samples, d, num_steps)
        """
        params_np = parameters.detach().cpu().numpy().astype(np.float32)
        params_jax = jnp.asarray(params_np)
        batch_size = params_jax.shape[0]

        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
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
        """Prior over initial condition x0 in R^d."""
        return pdist.Uniform(
            low=torch.full((self.d,), -2.0, dtype=torch.float32),
            high=torch.full((self.d,), 2.0, dtype=torch.float32),
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

        t_steps = us.shape[-1]
        denom_x = var_x.clamp_min(eps)
        denom_y = var_y.clamp_min(eps)

        if t_steps >= 2:
            ac1_x = (dx[:, 1:] * dx[:, :-1]).mean(dim=-1, keepdim=True) / denom_x
            ac1_y = (dy[:, 1:] * dy[:, :-1]).mean(dim=-1, keepdim=True) / denom_y
        else:
            ac1_x = torch.zeros_like(mean_x)
            ac1_y = torch.zeros_like(mean_y)

        if t_steps >= 3:
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
            [mean_x, mean_y, log_var_x, log_var_y, ac1_x, ac1_y, ac2_x, ac2_y, cc_xy],
            dim=-1,
        )
        return stats.squeeze(0) if is_2d_traj else stats

    def stepper(
        self,
        x: torch.Tensor,
        parameters: torch.Tensor,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Simulate a single time-series step transition for LinearSDE."""
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

        params_jax = jnp.asarray(params.detach().cpu().numpy())
        x_jax = jnp.asarray(x_t.detach().cpu().numpy())
        step_dt = self.step_dt if dt is None else dt

        def single_step(th, curr_x, k):
            A, B = self._get_AB(th)
            Phi, Q = matrix_fraction_decomposition(step_dt, A, B)
            L = jnp.linalg.cholesky(Q)
            noise = jax.random.normal(k, shape=(self.d,))
            return jnp.dot(Phi, curr_x) + jnp.dot(L, noise)

        keys = jax.random.split(jax.random.PRNGKey(self.seed), params.shape[0])
        next_x_jax = jax.vmap(single_step)(params_jax, x_jax, keys)
        next_x = torch.from_numpy(np.array(next_x_jax, copy=True))
        return next_x.squeeze(0) if is_1d else next_x

    def simulate_two_step(
        self,
        parameters: torch.Tensor,
        x0: torch.Tensor | None = None,
        t: int | torch.Tensor | None = None,
        informed: bool = True,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Generate two-step transition pairs (x_t, x_{t+1}) for compositional SBI."""
        return sample_two_step(
            self,
            parameters=parameters,
            x0=x0,
            t=t,
            informed=informed,
            return_dict=return_dict,
        )

    @override
    def simulate(self, parameters: torch.Tensor) -> torch.Tensor:
        """Simulate trajectories given parameter tensor."""
        parameters = torch.as_tensor(parameters, dtype=torch.float32)

        if parameters.numel() == 0:
            raise ValueError("Expected at least one parameter.")

        is_1d = parameters.ndim == 1
        if is_1d:
            parameters = parameters.unsqueeze(0)

        num_samples = parameters.shape[0]

        if self.summary == "two_step":
            pairs = self.simulate_two_step(parameters, return_dict=False)
            pairs_flat = pairs.reshape(num_samples, 2 * self.d)
            return pairs_flat.squeeze(0) if is_1d else pairs_flat

        us = self._solve(parameters)  # (num_samples, d, num_steps)
        us = us.float()

        has_nan = torch.isnan(us.reshape(num_samples, -1)).any(dim=1)
        idx_contains_nan = torch.where(has_nan)[0]
        idx_contains_no_nan = torch.where(~has_nan)[0]

        if self.summary is None:
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

            raw_data = us[idx_contains_no_nan, :, :].reshape(
                len(idx_contains_no_nan), -1
            )
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

    def log_likelihood(self, parameters: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
        """Evaluate exact marginal log-likelihood using JAX Kalman Filter."""
        parameters = torch.as_tensor(parameters, dtype=torch.float32)
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(0)

        x_obs_flat = torch.as_tensor(x_obs, dtype=torch.float32).flatten()
        x_obs_2d = x_obs_flat.reshape(-1, self.d)

        x_jax = jnp.asarray(x_obs_2d.detach().cpu().numpy())
        params_jax = jnp.asarray(parameters.detach().cpu().numpy())

        v_ll = jax.vmap(lambda th: self._kf_ll_fn(th, x_jax))
        lls_jax = v_ll(params_jax)
        return torch.from_numpy(np.array(lls_jax, copy=True))

    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        n_chains: int = 20,
        burn_in: int = 200,
        step_size: float = 0.1,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from posterior p(theta | x_obs) using JAX MCMC with exact Kalman likelihood."""
        if (
            x_obs is None
            and "num_samples" in kwargs
            and isinstance(num_samples, (torch.Tensor, np.ndarray))
        ):
            x_obs, num_samples = num_samples, kwargs.get("num_samples", 1000)

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

        if isinstance(x_obs, np.ndarray):
            x_obs = torch.from_numpy(x_obs).float()
        x_obs = x_obs.flatten().reshape(-1, self.d)
        x_obs_jax = jnp.asarray(x_obs.detach().cpu().numpy().astype(np.float32))

        num_samples_per_chain = math.ceil(num_samples / n_chains)

        rng_key = jax.random.PRNGKey(self.seed)
        samples_jax = self._jax_mcmc(
            x_obs=x_obs_jax,
            rng_key=rng_key,
            n_chains=n_chains,
            num_samples_per_chain=num_samples_per_chain,
            burn_in=burn_in,
            step_size=step_size,
        )
        samples_np = np.array(samples_jax[:num_samples], copy=True)
        return torch.from_numpy(samples_np)
