"""Simulator for the deterministic Lorenz63 ODE."""

import math
from functools import lru_cache
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pyro.distributions as pdist
import torch

from ..dynamical import DynamicalSimulator
from ..mcmc import build_batched_ll, build_jax_trajectory_solver, build_mh_mcmc_sampler
from ..priors import NormalPrior
from ..summary_stats import trajectory_stats_interleaved


_STATE_DIM = 3
_PRIOR_MEAN = (math.log(28.0), math.log(8.0 / 3.0), math.log(10.0))
_PRIOR_STD = (0.35, 0.25, 0.25)
_STATS_MODES = ("stats", "summary_stats")


@lru_cache(maxsize=32)
def _get_step_fn(dt: float, inner_steps: int):
    def step(state, _key, theta):
        r, b, sigma = jnp.exp(theta)

        def rhs(u):
            x, y, z = u
            return jnp.stack((sigma * (y - x), x * (r - z) - y, x * y - b * z))

        def rk4_step(_, u):
            k1 = rhs(u)
            k2 = rhs(u + 0.5 * dt * k1)
            k3 = rhs(u + 0.5 * dt * k2)
            k4 = rhs(u + dt * k3)
            return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return jax.lax.fori_loop(0, inner_steps, rk4_step, state)

    return step


@lru_cache(maxsize=32)
def _get_step_batch(dt: float, inner_steps: int):
    return jax.jit(jax.vmap(_get_step_fn(dt, inner_steps), in_axes=(0, None, 0)))


@lru_cache(maxsize=32)
def _get_solver(
    dt: float,
    inner_steps: int,
    num_steps: int,
    spinup_steps: int,
    u0: tuple[float, ...],
):
    solve_batch, solve_single = build_jax_trajectory_solver(
        _get_step_fn(dt, inner_steps), num_steps + spinup_steps, u0
    )
    if spinup_steps == 0:
        return solve_batch, solve_single

    @jax.jit
    def spun_batch(keys, thetas, x0s=None):
        return solve_batch(keys, thetas, x0s)[..., spinup_steps:]

    @jax.jit
    def spun_single(key, theta, x0=None):
        return solve_single(key, theta, x0)[..., spinup_steps:]

    return spun_batch, spun_single


def _jax_trajectory_stats(trajectory: jax.Array) -> jax.Array:
    """JAX equivalent of ``trajectory_stats_interleaved`` for a single trajectory."""
    means = trajectory.mean(axis=-1)
    centered = trajectory - means[:, None]
    variance = jnp.maximum((centered**2).mean(axis=-1), 1e-6)

    def autocorrelation(lag):
        if trajectory.shape[-1] <= lag:
            return jnp.zeros_like(means)
        cov = (centered[:, lag:] * centered[:, :-lag]).mean(axis=-1)
        return jnp.clip(cov / variance, -1.0, 1.0)

    return jnp.stack(
        (means, jnp.log(variance), autocorrelation(1), autocorrelation(2)), axis=-1
    ).reshape(-1)


@lru_cache(maxsize=32)
def _get_inference(
    dt: float,
    inner_steps: int,
    num_steps: int,
    spinup_steps: int,
    u0: tuple[float, ...],
    summary: str | None,
    obs_sigma: float,
    prior: NormalPrior,
):
    _, solve_single = _get_solver(dt, inner_steps, num_steps, spinup_steps, u0)
    step = _get_step_fn(dt, inner_steps)
    dummy_key = jax.random.PRNGKey(0)
    log_normalizer = math.log(obs_sigma) + 0.5 * math.log(2.0 * math.pi)

    def log_likelihood(theta, x_obs):
        if summary == "two_step":
            x_t, expected = x_obs
            predicted = step(x_t, None, theta)
        else:
            expected = x_obs
            trajectory = solve_single(dummy_key, theta)
            predicted = (
                _jax_trajectory_stats(trajectory)
                if summary in _STATS_MODES
                else trajectory.T
            )

        residual = (predicted - expected) / obs_sigma
        ll = -0.5 * jnp.sum(residual**2) - residual.size * log_normalizer
        return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)

    sampler = build_mh_mcmc_sampler(
        log_likelihood,
        prior,
        default_init_theta=prior.loc,
        init_strategy="sir_pool",
        rescue_stuck_chains=True,
    )
    return build_batched_ll(log_likelihood), sampler


class Lorenz63(DynamicalSimulator):
    r"""Lorenz63 with parameters ``(log(r), log(b), log(sigma))`` and state ``(x, y, z)``.

    The log parameters have independent Normal priors. The ODE uses their
    exponentials as the physical parameters ``(r, b, sigma)``.

    Reference
    ---------
    Lorenz, E. N. (1963). Deterministic Nonperiodic Flow.
    Journal of the Atmospheric Sciences, 20(2), 130–141.
    https://doi.org/10.1175/1520-0469%281963%29020%3C0130%3ADNF%3E2.0.CO%3B2

    Parameters
    ----------
    dim_parameters : int, optional
        Number of log parameters. Must be 3.
    seed : int, optional
        Random seed. Defaults to 42.
    dt : float, optional
        RK4 integration step size. Defaults to 0.001.
    inner_steps : int, optional
        RK4 steps per observation interval. Defaults to 20, so consecutive
        observations are separated by ``dt * inner_steps`` time units.
    num_steps : int, optional
        Number of recorded states, including the first state after spin-up.
        Defaults to 11.
    spinup_steps : int, optional
        Number of observation intervals to integrate before recording the
        first state. Defaults to 0.
    u0 : torch.Tensor, numpy.ndarray, or tuple of float, optional
        Initial state before spin-up. Defaults to ``(1.0, 1.0, 1.0)``.
    prior_mean : torch.Tensor, numpy.ndarray, tuple of float, or None, optional
        Means in log-parameter order. ``None`` uses ``log([28, 8/3, 10])``.
    prior_std : torch.Tensor, numpy.ndarray, tuple of float, or None, optional
        Standard deviations in log-parameter order. ``None`` uses
        ``(0.35, 0.25, 0.25)``.
    obs_sigma : float, optional
        Positive standard deviation of the Gaussian observation model.
        Defaults to 0.05.
    summary : str or None, optional
        ``None`` (the default) returns a deterministic trajectory flattened in
        state-major order. ``"subsample"`` adds independent Gaussian noise to
        each entry. ``"stats"`` and ``"summary_stats"`` return the mean,
        log-variance, and lag-1 and lag-2 autocorrelations for each coordinate.
        ``"two_step"`` returns a flattened transition pair in time order and
        requires ``num_steps >= 2``.
    **kwargs
        Additional options forwarded to the base simulator.

    Likelihood
    ----------
    ``log_likelihood()`` and ``posterior_samples()`` use independent Gaussian
    errors around the model prediction ``g(theta)`` with standard deviation
    ``obs_sigma``.

    .. math::
        y_{\mathrm{obs}} \mid \theta \sim
        \mathcal{N}(g(\theta), \mathrm{obs\_sigma}^2 I)

    For trajectories, the prediction comes from integrating the ODE from
    ``u0`` with the configured spin-up. For summary statistics, the prediction
    is computed from that trajectory. For transition pairs, the likelihood
    conditions on the observed first state and uses one observation interval
    of integration to predict the second state. The Gaussian model is used
    for reference inference even when ``simulate()`` produces deterministic
    outputs.
    """

    def __init__(
        self,
        dim_parameters: int = 3,
        seed: int = 42,
        *,
        dt: float = 0.001,
        inner_steps: int = 20,
        num_steps: int = 11,
        spinup_steps: int = 0,
        u0: torch.Tensor | np.ndarray | tuple[float, ...] = (1.0, 1.0, 1.0),
        prior_mean: torch.Tensor | np.ndarray | tuple[float, ...] | None = None,
        prior_std: torch.Tensor | np.ndarray | tuple[float, ...] | None = None,
        obs_sigma: float = 0.05,
        summary: str | None = None,
        **kwargs,
    ):
        if dim_parameters != _STATE_DIM:
            raise ValueError("Lorenz63 requires dim_parameters=3.")
        if not math.isfinite(dt) or dt <= 0.0 or inner_steps < 1:
            raise ValueError("Lorenz63 requires positive dt and inner_steps.")
        if num_steps < 1 or spinup_steps < 0:
            raise ValueError("Lorenz63 requires num_steps >= 1 and spinup_steps >= 0.")
        if summary == "two_step" and num_steps < 2:
            raise ValueError("The two_step summary requires num_steps >= 2.")
        if not math.isfinite(obs_sigma) or obs_sigma <= 0.0:
            raise ValueError("Lorenz63 requires a positive obs_sigma.")

        initial = torch.as_tensor(u0, dtype=torch.float32)
        if initial.shape != (_STATE_DIM,) or not torch.isfinite(initial).all():
            raise ValueError(
                "Lorenz63 requires three finite initial state coordinates."
            )

        loc = _PRIOR_MEAN if prior_mean is None else tuple(map(float, prior_mean))
        scale = _PRIOR_STD if prior_std is None else tuple(map(float, prior_std))
        if len(loc) != dim_parameters or len(scale) != dim_parameters:
            raise ValueError(
                "Lorenz63 requires three prior means and standard deviations."
            )
        if not all(map(math.isfinite, (*loc, *scale))):
            raise ValueError(
                "Lorenz63 requires finite prior means and standard deviations."
            )
        self.prior_spec = NormalPrior(loc=loc, scale=scale)
        self.prior_dist = self.prior_spec.to_pyro()
        self.canonical_theta = torch.tensor(loc, dtype=torch.float32)

        self.dt = float(dt)
        self.inner_steps = inner_steps
        self.spinup_steps = spinup_steps
        self._u0_tuple = tuple(initial.tolist())
        dim_data = self._resolve_summary_dim(
            summary, dim_raw=_STATE_DIM * num_steps, dim_stats=12, dim_two_step=6
        )
        super().__init__(
            seed=seed,
            dim_parameters=dim_parameters,
            dim_data=dim_data,
            state_dim=_STATE_DIM,
            num_steps=num_steps,
            obs_sigma=obs_sigma,
            summary=summary,
            u0=initial,
            **kwargs,
        )

        self._jax_solver_batch, _ = _get_solver(
            self.dt, self.inner_steps, self.num_steps, self.spinup_steps, self._u0_tuple
        )
        self._v_ll_fn, self._jax_mcmc = _get_inference(
            self.dt,
            self.inner_steps,
            self.num_steps,
            self.spinup_steps,
            self._u0_tuple,
            self.summary,
            self.obs_sigma,
            self.prior_spec,
        )

    @override
    def get_prior(self) -> pdist.Distribution:
        return self.prior_dist

    def get_prior_x0(
        self, num_sims: int = 100, num_steps: int = 200
    ) -> pdist.Distribution:
        """Build a prior-predictive state proposal with Gaussian jitter for local transitions."""
        if num_sims < 1 or num_steps < 1:
            raise ValueError(
                "The state proposal requires positive num_sims and num_steps."
            )

        key_prior, key_sim = jax.random.split(self._next_rng_key())
        thetas = self.prior_spec.jax_sample_unconstrained(key_prior, num_sims)
        solve_batch, _ = _get_solver(
            self.dt, self.inner_steps, num_steps, self.spinup_steps, self._u0_tuple
        )
        trajectories = solve_batch(jax.random.split(key_sim, num_sims), thetas)
        states = trajectories.transpose(0, 2, 1).reshape(-1, self.state_dim)
        states = torch.from_numpy(np.array(states, copy=True))
        states = states[torch.isfinite(states).all(dim=1)]
        if states.shape[0] == 0:
            raise ValueError(
                "The state proposal requires at least one finite simulated state."
            )

        return pdist.MixtureOfDiagNormals(
            states, torch.full_like(states, 0.05), states.new_zeros(states.shape[0])
        )

    @staticmethod
    @override
    def compute_summary_stats(us: torch.Tensor) -> torch.Tensor:
        return trajectory_stats_interleaved(us)

    def stepper(
        self,
        x: torch.Tensor | np.ndarray,
        parameters: torch.Tensor | np.ndarray,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Advance one observation interval from a supplied state."""
        states, params, is_1d = self._broadcast_stepper_inputs(x, parameters)
        step_size, inner_steps = self.dt, self.inner_steps
        if dt is not None:
            if not math.isfinite(dt) or dt <= 0.0:
                raise ValueError("The step duration must be positive.")
            inner_steps = max(round(dt / self.dt), 1)
            step_size = dt / inner_steps

        step_batch = _get_step_batch(step_size, inner_steps)
        next_state = step_batch(
            jnp.asarray(states.detach().cpu().numpy()),
            None,
            jnp.asarray(params.detach().cpu().numpy()),
        )
        result = torch.from_numpy(np.array(next_state, copy=True)).to(states.device)
        return result.squeeze(0) if is_1d else result

    def _prepare_observation(self, x_obs: torch.Tensor | np.ndarray) -> jax.Array:
        if self.summary == "two_step" or self.summary in _STATS_MODES:
            obs = torch.as_tensor(x_obs, dtype=torch.float32)
            if obs.numel() != self.dim_data:
                raise ValueError(f"Expected {self.dim_data} values in x_obs.")
            if self.summary == "two_step":
                if obs.shape[-2:] == (self.state_dim, 2):
                    obs = obs.transpose(-2, -1)
                obs = obs.reshape(2, self.state_dim)
            else:
                obs = obs.flatten()
            return jnp.asarray(obs.detach().cpu().numpy())

        obs = self._format_trajectory_obs(x_obs)
        if obs.shape != (self.num_steps, self.state_dim):
            raise ValueError(
                f"Expected one trajectory with {self.num_steps} states of dimension {self.state_dim}."
            )
        return obs

    def log_likelihood(
        self, parameters: torch.Tensor | np.ndarray, x_obs: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        """Evaluate the Gaussian observation model used by the reference posterior."""
        params, _ = self._prepare_parameters(parameters)
        params_jax = jnp.asarray(params.detach().cpu().numpy())
        lls = self._v_ll_fn(params_jax, self._prepare_observation(x_obs))
        return torch.from_numpy(np.array(lls, copy=True))

    @override
    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        n_chains: int = 20,
        burn_in: int = 300,
        inner_thin: int = 4,
        step_size: float = 0.05,
        n_sir: int = 512,
        adapt_warmup: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Sample a reference posterior using JAX Metropolis-Hastings.

        The likelihood uses independent Gaussian errors with scale ``obs_sigma``.
        Trajectories and statistics use the configured initial state and spin-up.
        For a pair, the likelihood conditions on the first state and applies
        the error model to the second state.
        """
        if x_obs is None:
            if self.summary == "two_step":
                x_obs = self.simulate_two_step(self.canonical_theta, x0=self.u0, t=0)
            else:
                x_obs = self.simulate(self.canonical_theta)

        samples = self._jax_mcmc(
            self._prepare_observation(x_obs),
            self._next_rng_key(),
            n_chains=n_chains,
            num_samples_per_chain=math.ceil(num_samples / n_chains),
            burn_in=burn_in,
            inner_thin=inner_thin,
            step_size=step_size,
            n_sir=n_sir,
            adapt_warmup=adapt_warmup,
        )
        return self._finalize_mcmc_samples(samples, num_samples)
