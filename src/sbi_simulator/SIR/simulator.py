"""Simulator for the SIR (Susceptible-Infected-Recovered) SDE model."""

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


def _build_sir_step_fn(
    dt: float = 0.01,
    inner_steps: int = 5,
    sigma1: float = 0.1,
):
    """Build single observation step transition function in JAX for SIR."""

    def drift(x, gamma, beta):
        S, I = x[0], x[1]
        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I
        return jnp.stack([dS, dI, dR])

    def diffusion(x):
        return jnp.array(
            [[0.01, 0.0, 0.0], [0.0, x[1] * sigma1, 0.0], [0.0, 0.0, 0.01]]
        )

    def step_one(x, key, theta):
        gamma = jax.nn.sigmoid(theta[0]) * 2.0
        beta = jax.nn.sigmoid(theta[1]) * 2.0
        keys = jax.random.split(key, inner_steps)

        def inner_loop(i, curr_x):
            curr_x = jnp.clip(curr_x, 0.0)
            xdt = drift(curr_x, gamma, beta)
            dWt = jax.random.normal(keys[i], shape=(3,)) * jnp.sqrt(dt)
            B = diffusion(curr_x)
            curr_x = curr_x + xdt * dt + jnp.dot(B, dWt)
            return jnp.clip(curr_x, 0.0)

        return jax.lax.fori_loop(0, inner_steps, inner_loop, x)

    return step_one


def _build_sir_sde_jax_solver(step_one_fn, num_steps: int = 11):
    """Build a JIT-compiled batched Euler-Maruyama SDE solver in JAX for SIR."""
    default_x0 = jnp.array([4.0, 1.0, 0.0], dtype=jnp.float32)

    def solve_single(rng_key, theta, x0=None):
        u0 = default_x0 if x0 is None else jnp.clip(x0, 0.0)

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
            x0s = jnp.broadcast_to(default_x0, (thetas.shape[0], 3))
        return solve_batch_fn(rng_keys, thetas, x0s)

    return jax.jit(solve_batch), jax.jit(solve_single)


class SIR(Simulator):
    """SIR (Susceptible-Infected-Recovered) Stochastic Differential Equation (SDE) simulator.

    Formulation:
        dS = -beta * S * I * dt + 0.01 * dW0
        dI = (beta * S * I - gamma * I) * dt + sigma1 * I * dW1
        dR = gamma * I * dt + 0.01 * dW2

    Parameters:
        theta: 2D vector [gamma, beta] in R^2 with standard normal prior N(0, I_2).
        Mapped to positive transmission and recovery rates via:
            gamma = 2.0 * sigmoid(theta[0]) in (0, 2.0)
            beta  = 2.0 * sigmoid(theta[1]) in (0, 2.0)
    """

    def __init__(
        self,
        dim_parameters: int = 2,
        seed: int = 42,
        dt: float = 0.01,
        inner_steps: int = 5,
        sigma1: float = 0.1,
        num_steps: int = 11,
        obs_sigma: float = 0.01,
        summary: str | None = None,
    ):
        self.dim_parameters = dim_parameters
        if dim_parameters != 2:
            raise ValueError(
                f"SIR requires dim_parameters=2 (gamma, beta), got {dim_parameters}."
            )

        self.dt = dt
        self.inner_steps = inner_steps
        self.sigma1 = sigma1
        self.num_steps = num_steps
        self.obs_sigma = obs_sigma
        self.dim_data_raw = int(3 * num_steps)

        if summary is None or summary == "subsample":
            self.dim_data = self.dim_data_raw
        elif summary in ("stats", "summary_stats"):
            self.dim_data = 15
        elif summary == "two_step":
            self.dim_data = 6
        else:
            raise NotImplementedError(f"Unknown summary mode: {summary}")
        self.summary = summary

        super().__init__(seed, self.dim_parameters, self.dim_data)

        self.u0 = torch.tensor([4.0, 1.0, 0.0], dtype=torch.float32)

        # Prior setup: Standard Normal N(0, I_2) matching markovsbi
        self.prior_loc = [0.0, 0.0]
        self.prior_scale = [1.0, 1.0]
        self.prior_dist = pdist.Normal(
            loc=torch.zeros(2, dtype=torch.float32),
            scale=torch.ones(2, dtype=torch.float32),
        ).to_event(1)

        # Build JAX SDE solver and Particle Filter MCMC sampler
        self._step_fn = _build_sir_step_fn(
            dt=self.dt,
            inner_steps=self.inner_steps,
            sigma1=self.sigma1,
        )
        self._jax_solver_batch, self._jax_solver_single = _build_sir_sde_jax_solver(
            self._step_fn,
            num_steps=self.num_steps,
        )
        self._pf_ll_fn = build_particle_filter_ll(
            self._step_fn,
            obs_dim=3,
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
        u0: optional (num_samples, 3) tensor of initial states
        returns: (num_samples, 3, num_steps) tensor
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

            dS = -beta * S * I
            dI = beta * S * I - gamma * I
            dR = gamma * I
            drift_step = torch.stack([dS, dI, dR], dim=-1)

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
    def simulate(self, parameters: torch.Tensor) -> torch.Tensor:
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
        x_obs_2d = x_obs_flat.reshape(-1, 3)

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
        step_size: float = 0.1,
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

        x_obs_tensor = torch.as_tensor(x_obs, dtype=torch.float32)
        if x_obs_tensor.ndim == 1:
            x_obs_2d = x_obs_tensor.reshape(-1, 3)
        else:
            x_obs_2d = x_obs_tensor

        x_jax = jnp.asarray(x_obs_2d.detach().cpu().numpy())
        rng_key = jax.random.PRNGKey(self.seed)

        num_samples_per_chain = math.ceil(num_samples / n_chains)

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
