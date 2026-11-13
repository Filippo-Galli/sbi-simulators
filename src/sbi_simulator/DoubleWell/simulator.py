"""Simulator for the Double-Well SDE model."""

import math
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pyro.distributions as pdist
import torch

from ..compositional import sample_two_step
from ..mcmc import build_gibbs_pmmh_sampler, build_particle_filter_ll
from ..Simulator import Simulator


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
        dWt = jax.random.normal(key, shape=(dim,)) * jnp.sqrt(dt)
        return x + xdt * dt + dWt

    def step_one(x, key, theta):
        theta1 = jnp.exp(0.5 * theta[0])
        theta2 = -jnp.exp(0.5 * theta[1])

        keys = jax.random.split(key, inner_steps)

        def inner_loop(i, curr_x):
            return integrate_step(curr_x, keys[i], theta1, theta2)

        return jax.lax.fori_loop(0, inner_steps, inner_loop, x)

    return step_one


def _build_double_well_sde_jax_solver(step_one_fn, num_steps: int = 11, dim: int = 4):
    """Build a JIT-compiled batched Euler-Maruyama SDE solver in JAX for Double-Well."""
    default_x0 = jnp.array([-0.5, 0.5] * (dim // 2), dtype=jnp.float32)

    def solve_single(rng_key, theta, x0=None):
        u0 = default_x0 if x0 is None else x0

        def one_step(x, key):
            x_next = step_one_fn(x, key, theta)
            return x_next, x_next

        keys = jax.random.split(rng_key, num_steps - 1)
        _, xs = jax.lax.scan(one_step, u0, keys)
        full_traj = jnp.concatenate([u0[None, :], xs], axis=0)  # (num_steps, dim)
        return full_traj.T  # (dim, num_steps)

    solve_batch_fn = jax.vmap(solve_single, in_axes=(0, 0, 0))

    def solve_batch(rng_keys, thetas, x0s=None):
        if x0s is None:
            x0s = jnp.broadcast_to(default_x0, (thetas.shape[0], dim))
        return solve_batch_fn(rng_keys, thetas, x0s)

    return jax.jit(solve_batch), jax.jit(solve_single)


class DoubleWell(Simulator):
    r"""Double-Well Stochastic Differential Equation (SDE) simulator.

    Formulation:
        dx = (theta_1 * x + theta_2 * x^3) dt + dW

    where parameters are mapped from unconstrained space via:
        theta_1 = exp(0.5 * \tilde{\theta}_0)
        theta_2 = -exp(0.5 * \tilde{\theta}_1)

    Parameters:
        theta: 2D vector in R^2 with standard normal prior N(0, I_2).
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
    ):
        self.dim_parameters = dim_parameters
        if dim_parameters != 2:
            raise ValueError(
                f"DoubleWell requires dim_parameters=2, got {dim_parameters}."
            )

        self.D = D
        self.dt = dt
        self.inner_steps = inner_steps
        self.num_steps = num_steps
        self.obs_sigma = obs_sigma
        self.dim_data_raw = int(D * num_steps)

        if summary is None or summary == "subsample":
            self.dim_data = self.dim_data_raw
        elif summary in ("stats", "summary_stats"):
            # 4 summary statistics per dimension: mean, log-var, ac1, ac2
            self.dim_data = 4 * D
        elif summary == "two_step":
            self.dim_data = 2 * D
        else:
            raise NotImplementedError(f"Unknown summary mode: {summary}")
        self.summary = summary

        super().__init__(seed, self.dim_parameters, self.dim_data)

        self.u0 = torch.tensor([-0.5, 0.5] * (D // 2), dtype=torch.float32)

        # Prior setup: Standard Normal N(0, I_2)
        self.prior_loc = [0.0] * dim_parameters
        self.prior_scale = [1.0] * dim_parameters
        self.prior_dist = pdist.Normal(
            loc=torch.zeros(dim_parameters, dtype=torch.float32),
            scale=torch.ones(dim_parameters, dtype=torch.float32),
        ).to_event(1)

        # Build JAX SDE solver and Particle Filter MCMC sampler
        self._step_fn = _build_double_well_step_fn(
            dt=self.dt,
            inner_steps=self.inner_steps,
            dim=self.D,
        )
        self._jax_solver_batch, self._jax_solver_single = (
            _build_double_well_sde_jax_solver(
                self._step_fn,
                num_steps=self.num_steps,
                dim=self.D,
            )
        )
        self._pf_ll_fn = build_particle_filter_ll(
            self._step_fn,
            obs_dim=self.D,
            obs_sigma=self.obs_sigma,
        )
        self._jax_mcmc = build_gibbs_pmmh_sampler(
            self._pf_ll_fn,
            dim_parameters=self.dim_parameters,
        )

    def _solve(
        self,
        parameters: torch.Tensor,
        u0: torch.Tensor | None = None,
        rng_key: jax.Array | None = None,
    ) -> torch.Tensor:
        """Batched SDE Euler-Maruyama simulation accelerated via JAX.

        parameters: (num_samples, 2) tensor of raw thetas
        u0: optional (num_samples, D) tensor of initial states
        returns: (num_samples, D, num_steps) tensor
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
        """Prior over initial condition x0 in R^D."""
        return pdist.Uniform(
            low=torch.full((self.D,), -2.5, dtype=torch.float32),
            high=torch.full((self.D,), 2.5, dtype=torch.float32),
        ).to_event(1)

    @staticmethod
    def compute_summary_stats(us: torch.Tensor) -> torch.Tensor:
        """Compute 4*D summary statistics from (num_samples, D, T) trajectory tensor.

        Statistics per dimension d:
            - Mean
            - Log-variance
            - Lag-1 autocorrelation
            - Lag-2 autocorrelation
        """
        is_2d_traj = us.ndim == 2
        if is_2d_traj:
            us = us.unsqueeze(0)

        eps = 1e-6
        num_samples, _, t_steps = us.shape

        means = us.mean(dim=-1, keepdim=True)  # (N, D, 1)
        dus = us - means
        vars_ = (dus**2).mean(dim=-1, keepdim=True)  # (N, D, 1)
        log_vars = torch.log(vars_.clamp_min(eps))
        denom = vars_.clamp_min(eps)

        if t_steps >= 2:
            ac1 = (dus[:, :, 1:] * dus[:, :, :-1]).mean(dim=-1, keepdim=True) / denom
        else:
            ac1 = torch.zeros_like(means)

        if t_steps >= 3:
            ac2 = (dus[:, :, 2:] * dus[:, :, :-2]).mean(dim=-1, keepdim=True) / denom
        else:
            ac2 = torch.zeros_like(means)

        ac1 = torch.nan_to_num(ac1.clamp(-1.0, 1.0), nan=0.0)
        ac2 = torch.nan_to_num(ac2.clamp(-1.0, 1.0), nan=0.0)

        # Shape: (N, 4, D) -> transpose & flatten to (N, 4*D)
        stats = torch.cat([means, log_vars, ac1, ac2], dim=-1)  # (N, D, 4)
        stats = stats.reshape(num_samples, -1)
        return stats.squeeze(0) if is_2d_traj else stats

    def stepper(
        self,
        x: torch.Tensor,
        parameters: torch.Tensor,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Simulate a single time-series step transition for Double-Well SDE.

        Parameters
        ----------
        x : torch.Tensor
            Current state in R^D, shape (D,) or (batch_size, D).
        parameters : torch.Tensor
            Double-Well parameters (raw theta in R^2), shape (2,) or (batch_size, 2).
        dt : float, optional
            Step time interval (default: self.dt * self.inner_steps = 0.05).

        Returns
        -------
        torch.Tensor
            Next state in R^D of shape (D,) or (batch_size, D).
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

        theta1 = torch.exp(0.5 * params[:, 0:1])
        theta2 = -torch.exp(0.5 * params[:, 1:2])

        step_dt = float(self.dt * self.inner_steps if dt is None else dt)
        sub_dt = float(self.dt)
        n_substeps = max(round(step_dt / sub_dt), 1)
        actual_h = step_dt / n_substeps

        curr_x = x_t.clone()

        for _ in range(n_substeps):
            xdt = theta1 * curr_x + theta2 * (curr_x**3)
            dW = torch.randn_like(curr_x) * math.sqrt(actual_h)
            curr_x = curr_x + xdt * actual_h + dW

        return curr_x.squeeze(0) if is_1d else curr_x

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
            pairs_flat = pairs.reshape(num_samples, 2 * self.D)
            return pairs_flat.squeeze(0) if is_1d else pairs_flat

        us = self._solve(parameters)  # (num_samples, D, num_steps)
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

    def log_likelihood(
        self, parameters: torch.Tensor, x_obs: torch.Tensor, n_particles: int = 150
    ) -> torch.Tensor:
        """Evaluate marginal log-likelihood using JAX Particle Filter."""
        parameters = torch.as_tensor(parameters, dtype=torch.float32)
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(0)

        num_samples = parameters.shape[0]
        x_obs_flat = torch.as_tensor(x_obs, dtype=torch.float32).flatten()
        x_obs_2d = x_obs_flat.reshape(-1, self.D)

        x_jax = jnp.asarray(x_obs_2d.detach().cpu().numpy())
        params_jax = jnp.asarray(parameters.detach().cpu().numpy())

        key = jax.random.PRNGKey(self.seed)
        keys = jax.random.split(key, num_samples)

        v_ll = jax.vmap(
            lambda th, k: self._pf_ll_fn(th, x_jax, k, n_particles=n_particles)
        )
        lls_jax = v_ll(params_jax, keys)
        return torch.from_numpy(np.array(lls_jax, copy=True))

    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        n_chains: int = 20,
        burn_in: int = 200,
        inner_thin: int = 4,
        n_particles: int = 150,
        step_size: float = 0.2,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from posterior p(theta | x_obs) using JAX Gibbs PMMH."""
        if (
            x_obs is None
            and "num_samples" in kwargs
            and isinstance(num_samples, (torch.Tensor, np.ndarray))
        ):
            x_obs, num_samples = num_samples, kwargs.get("num_samples", 1000)

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

        if isinstance(x_obs, np.ndarray):
            x_obs = torch.from_numpy(x_obs).float()
        x_obs = x_obs.flatten().reshape(-1, self.D)
        x_obs_jax = jnp.asarray(x_obs.detach().cpu().numpy().astype(np.float32))

        num_samples_per_chain = math.ceil(num_samples / n_chains)

        rng_key = jax.random.PRNGKey(self.seed)
        samples_jax = self._jax_mcmc(
            x_obs=x_obs_jax,
            rng_key=rng_key,
            n_chains=n_chains,
            num_samples_per_chain=num_samples_per_chain,
            burn_in=burn_in,
            inner_thin=inner_thin,
            n_particles=n_particles,
            step_size=step_size,
        )
        samples_np = np.array(samples_jax[:num_samples], copy=True)
        return torch.from_numpy(samples_np)
