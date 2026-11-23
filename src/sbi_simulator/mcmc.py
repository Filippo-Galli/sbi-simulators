"""Shared JAX sampling, trajectory solver, and likelihood utilities."""

import functools

import jax
import jax.numpy as jnp


def systematic_resample(weights: jax.Array, key: jax.Array) -> jax.Array:
    """Systematic resampling of particles in JAX.

    Args:
        weights: Normalized array of particle weights of shape ``(n_particles,)``.
        key: JAX random key.

    Returns:
        Array of resampled particle indices of shape ``(n_particles,)``.
    """
    n = weights.shape[0]
    u = jax.random.uniform(key, shape=()) / n
    points = u + jnp.arange(n) / n
    cumsum = jnp.cumsum(weights)
    idx = jnp.searchsorted(cumsum, points)
    return jnp.clip(idx, 0, n - 1)


def build_jax_trajectory_solver(
    step_one_fn,
    num_steps: int,
    default_x0: tuple[float, ...],
    clip_non_negative: bool = False,
):
    """Build JIT-compiled batched and single-trajectory JAX solvers.

    Parameters
    ----------
    step_one_fn : callable
        Transition function ``(x, key, theta) -> next_x`` advancing one observation step.
    num_steps : int
        Total number of time steps (including initial state at ``t=0``).
    default_x0 : tuple[float, ...]
        Default initial state vector of length ``state_dim``.
    clip_non_negative : bool, default False
        Whether to clamp initial conditions ``x0`` at zero (for population models).

    Returns
    -------
    tuple[callable, callable]
        ``(solve_batch, solve_single)``, both JIT-compiled.
    """
    base_x0 = jnp.asarray(default_x0, dtype=jnp.float32)
    state_dim = len(default_x0)

    def solve_single(rng_key, theta, x0=None):
        if x0 is None:
            u0 = base_x0
        elif clip_non_negative:
            u0 = jnp.clip(x0, 0.0)
        else:
            u0 = x0

        def one_step(x, key):
            x_next = step_one_fn(x, key, theta)
            return x_next, x_next

        keys = jax.random.split(rng_key, num_steps - 1)
        _, xs = jax.lax.scan(one_step, u0, keys)
        full_traj = jnp.concatenate([u0[None, :], xs], axis=0)  # (num_steps, state_dim)
        return full_traj.T  # (state_dim, num_steps)

    solve_batch_fn = jax.vmap(solve_single, in_axes=(0, 0, 0))

    def solve_batch(rng_keys, thetas, x0s=None):
        if x0s is None:
            x0s = jnp.broadcast_to(base_x0, (thetas.shape[0], state_dim))
        return solve_batch_fn(rng_keys, thetas, x0s)

    return jax.jit(solve_batch), jax.jit(solve_single)


def build_batched_ll(ll_fn, stochastic: bool = False):
    """Build a pre-JITted batched log-likelihood function via ``jax.vmap``.

    Parameters
    ----------
    ll_fn : callable
        Scalar log-likelihood function:
        - ``(theta, x_obs)`` if ``stochastic=False``
        - ``(theta, x_obs, key, n_particles=...)`` if ``stochastic=True``.
    stochastic : bool, default False
        Whether ``ll_fn`` takes a per-sample PRNG key and static ``n_particles``.
    """
    if stochastic:
        v_ll = jax.jit(jax.vmap(ll_fn, in_axes=(0, None, 0, None)), static_argnums=(3,))
        return lambda thetas, x_obs, keys, n_particles=150: v_ll(
            thetas, x_obs, keys, n_particles
        )
    return jax.jit(jax.vmap(ll_fn, in_axes=(0, None)))


def build_particle_filter_ll(
    step_one_fn,
    obs_dim: int,
    obs_sigma: float = 0.05,
):
    """Build a JIT-compiled Bootstrap Particle Filter for marginal log-likelihood.

    Parameters
    ----------
    step_one_fn : callable
        Function ``(curr_x, subkey, theta)`` advancing a single particle state by one
        observation time-step (e.g. across inner integration sub-steps).
    obs_dim : int
        Dimensionality of the state / observation vector.
    obs_sigma : float, default 0.05
        Standard deviation of additive Gaussian observation noise.
    """
    v_step = jax.vmap(step_one_fn, in_axes=(0, 0, None))
    log_norm_const = obs_dim * jnp.log(obs_sigma) + (obs_dim / 2.0) * jnp.log(
        2.0 * jnp.pi
    )

    def particle_filter_ll(theta, x_obs, key, n_particles: int = 150):
        init_particles = jnp.broadcast_to(x_obs[0], (n_particles, obs_dim))
        log_n = jnp.log(n_particles)

        def scan_body(carry, t):
            particles, k = carry
            k_trans, k_resamp, next_k = jax.random.split(k, 3)
            trans_keys = jax.random.split(k_trans, n_particles)
            new_particles = v_step(particles, trans_keys, theta)

            y_t = x_obs[t]
            diff = (new_particles - y_t) / obs_sigma
            log_w = -0.5 * jnp.sum(diff**2, axis=-1) - log_norm_const

            lse = jax.scipy.special.logsumexp(log_w)
            log_z_t = lse - log_n
            norm_w = jnp.exp(log_w - lse)

            resamp_idx = systematic_resample(norm_w, k_resamp)
            resampled_particles = new_particles[resamp_idx]
            return (resampled_particles, next_k), log_z_t

        ts = jnp.arange(1, x_obs.shape[0])
        _, log_zs = jax.lax.scan(scan_body, (init_particles, key), ts)
        ll = jnp.sum(log_zs)
        return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)

    return jax.jit(particle_filter_ll, static_argnames=["n_particles"])


def build_gibbs_pmmh_sampler(
    pf_ll_fn,
    dim_parameters: int,
):
    """Build a JIT-compiled Gibbs Particle Marginal Metropolis-Hastings (PMMH) sampler in JAX.

    Features:
    - Per-chain independent SIR (Sequential Importance Resampling) mode discovery from prior
    - Gibbs updates alternating between ``inner_thin`` parameter steps (frozen particle key)
      and an independent MH step on the particle filter key
    - Standard Normal prior ``N(0, I_D)`` on theta
    """
    log_prior_norm = (dim_parameters / 2.0) * jnp.log(2.0 * jnp.pi)

    def log_joint(theta, x_obs, key, n_particles: int = 150):
        lp_prior = -0.5 * jnp.sum(theta**2) - log_prior_norm
        ll = pf_ll_fn(theta, x_obs, key, n_particles=n_particles)
        total = lp_prior + ll
        return jnp.where(jnp.isfinite(total), total, -jnp.inf)

    v_log_joint = jax.vmap(log_joint, in_axes=(0, None, 0, None))

    @functools.partial(
        jax.jit,
        static_argnames=[
            "n_chains",
            "num_samples_per_chain",
            "burn_in",
            "inner_thin",
            "n_particles",
            "n_sir",
        ],
    )
    def run_mcmc(
        x_obs,
        rng_key,
        n_chains: int = 20,
        num_samples_per_chain: int = 500,
        burn_in: int = 200,
        inner_thin: int = 4,
        n_particles: int = 150,
        step_size: float = 0.1,
        n_sir: int = 50,
    ):
        k_init, k_mcmc = jax.random.split(rng_key)

        def init_single_chain(k):
            k1, k2, k3 = jax.random.split(k, 3)
            thetas_sir = jax.random.normal(k1, shape=(n_sir, dim_parameters))
            keys_sir = jax.random.split(k2, n_sir)
            lps = jax.vmap(lambda th, k_pf: log_joint(th, x_obs, k_pf, n_particles))(
                thetas_sir, keys_sir
            )
            safe_lps = jnp.where(jnp.isfinite(lps), lps, -jnp.inf)
            best_th = thetas_sir[jnp.argmax(safe_lps)]
            init_lp = log_joint(best_th, x_obs, k3, n_particles)
            return best_th, k3, init_lp

        init_keys = jax.random.split(k_init, n_chains)
        curr_thetas, curr_pf_keys, curr_lp = jax.vmap(init_single_chain)(init_keys)

        def scan_step(state, subkey):
            k_scan_th, k_key, k_acc_k = jax.random.split(subkey, 3)
            keys_th = jax.random.split(k_scan_th, inner_thin)

            def theta_step(carry, subkey_th):
                th, k_pf, lp = carry
                k_p, k_a = jax.random.split(subkey_th)
                prop_th = th + jax.random.normal(k_p, shape=th.shape) * step_size
                prop_lp = v_log_joint(prop_th, x_obs, k_pf, n_particles)
                log_u = jnp.log(jax.random.uniform(k_a, shape=(n_chains,)))
                acc = jnp.isfinite(prop_lp) & (log_u < prop_lp - lp)
                new_th = jnp.where(acc[:, None], prop_th, th)
                new_lp = jnp.where(acc, prop_lp, lp)
                return (new_th, k_pf, new_lp), None

            (curr_thetas, curr_pf_keys, curr_lp), _ = jax.lax.scan(
                theta_step, state, keys_th
            )

            # 1 step on particle filter key
            prop_pf_keys = jax.random.split(k_key, n_chains)
            prop_lp_key = v_log_joint(curr_thetas, x_obs, prop_pf_keys, n_particles)
            log_u_key = jnp.log(jax.random.uniform(k_acc_k, shape=(n_chains,)))
            acc_key = jnp.isfinite(prop_lp_key) & (log_u_key < prop_lp_key - curr_lp)
            curr_pf_keys = jnp.where(acc_key[:, None], prop_pf_keys, curr_pf_keys)
            curr_lp = jnp.where(acc_key, prop_lp_key, curr_lp)

            return (curr_thetas, curr_pf_keys, curr_lp), curr_thetas

        total_steps = burn_in + num_samples_per_chain
        keys = jax.random.split(k_mcmc, total_steps)
        _, all_samples = jax.lax.scan(
            scan_step, (curr_thetas, curr_pf_keys, curr_lp), keys
        )
        post_burnin = all_samples[burn_in:]
        return post_burnin.transpose(1, 0, 2).reshape(-1, dim_parameters)

    return run_mcmc


def build_ode_ll(
    solve_single_fn,
    obs_dim: int,
    obs_sigma: float = 0.05,
):
    """Build a JIT-compiled exact Gaussian observation log-likelihood for deterministic ODEs.

    Parameters
    ----------
    solve_single_fn : callable
        Function ``(dummy_key, theta, x0)`` returning trajectory of shape ``(obs_dim, num_steps)``.
    obs_dim : int
        Dimensionality of state / observation vector.
    obs_sigma : float, default 0.05
        Standard deviation of additive Gaussian observation noise.
    """
    log_norm_const = obs_dim * jnp.log(obs_sigma) + (obs_dim / 2.0) * jnp.log(
        2.0 * jnp.pi
    )
    dummy_key = jax.random.PRNGKey(0)

    def ode_ll(theta, x_obs):
        u0 = x_obs[0]
        traj = solve_single_fn(dummy_key, theta, u0).T  # (num_steps, obs_dim)
        diff = (traj[1:] - x_obs[1:]) / obs_sigma
        log_p = -0.5 * jnp.sum(diff**2, axis=-1) - log_norm_const
        ll = jnp.sum(log_p)
        return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)

    return jax.jit(ode_ll)


def build_mh_mcmc_sampler(
    ll_fn,
    dim_parameters: int,
    prior_type: str = "gaussian",
    prior_low: float = -5.0,
    prior_high: float = 2.0,
    prior_loc: tuple[float, ...] | None = None,
    prior_scale: tuple[float, ...] | None = None,
    default_init_theta: tuple[float, ...] | None = None,
    init_strategy: str | None = None,
):
    """Build a unified JIT-compiled parallel Metropolis-Hastings MCMC sampler in JAX.

    Parameters
    ----------
    ll_fn : callable
        Deterministic scalar log-likelihood ``(theta, x_obs) -> float``.
    dim_parameters : int
        Dimensionality of ``theta``.
    prior_type : {"gaussian", "uniform", "lognormal"}, default "gaussian"
        Prior family over ``theta``.
    prior_low, prior_high : float
        Box bounds when ``prior_type == "uniform"``.
    prior_loc, prior_scale : tuple[float, ...], optional
        Mean and std of ``log(theta)`` when ``prior_type == "lognormal"``.
    default_init_theta : tuple[float, ...], optional
        Seed parameter vector used when ``init_strategy == "jitter"``.
    init_strategy : {"jitter", "sir_argmax", "sir_resample", "prior"}, optional
        Chain initialization strategy. Defaults to ``"jitter"`` when
        ``default_init_theta`` is provided, otherwise ``"sir_argmax"``.
    """
    if prior_type not in ("gaussian", "uniform", "lognormal"):
        raise ValueError(f"Unsupported prior_type: {prior_type}")

    if init_strategy is None:
        init_strategy = "jitter" if default_init_theta is not None else "sir_argmax"
    if init_strategy not in ("jitter", "sir_argmax", "sir_resample", "prior"):
        raise ValueError(f"Unsupported init_strategy: {init_strategy}")
    if init_strategy == "jitter" and default_init_theta is None:
        raise ValueError("init_strategy='jitter' requires default_init_theta.")

    log_prior_norm = (dim_parameters / 2.0) * jnp.log(2.0 * jnp.pi)
    loc_arr = (
        jnp.zeros(dim_parameters, dtype=jnp.float32)
        if prior_loc is None
        else jnp.asarray(prior_loc, dtype=jnp.float32)
    )
    scale_arr = (
        jnp.ones(dim_parameters, dtype=jnp.float32)
        if prior_scale is None
        else jnp.asarray(prior_scale, dtype=jnp.float32)
    )
    log_ln_norm = jnp.sum(jnp.log(scale_arr)) + log_prior_norm
    init_arr = (
        None
        if default_init_theta is None
        else jnp.asarray(default_init_theta, dtype=jnp.float32)
    )

    def log_joint(state_val, x_obs):
        if prior_type == "uniform":
            in_prior = jnp.all((state_val >= prior_low) & (state_val <= prior_high))
            ll = ll_fn(state_val, x_obs)
            return jnp.where(in_prior & jnp.isfinite(ll), ll, -jnp.inf)

        if prior_type == "lognormal":
            # state_val is z = log(theta) ~ Normal(loc_arr, scale_arr)
            lp_prior = (
                -0.5 * jnp.sum(((state_val - loc_arr) / scale_arr) ** 2) - log_ln_norm
            )
            ll = ll_fn(jnp.exp(state_val), x_obs)
        else:
            lp_prior = -0.5 * jnp.sum(state_val**2) - log_prior_norm
            ll = ll_fn(state_val, x_obs)

        total = lp_prior + ll
        return jnp.where(jnp.isfinite(total), total, -jnp.inf)

    v_log_joint = jax.vmap(log_joint, in_axes=(0, None))

    def sample_prior_batch(key, count: int):
        if prior_type == "uniform":
            return jax.random.uniform(
                key,
                shape=(count, dim_parameters),
                minval=prior_low,
                maxval=prior_high,
            )
        if prior_type == "lognormal":
            return (
                loc_arr + jax.random.normal(key, shape=(count, dim_parameters)) * scale_arr
            )
        return jax.random.normal(key, shape=(count, dim_parameters))

    @functools.partial(
        jax.jit,
        static_argnames=[
            "n_chains",
            "num_samples_per_chain",
            "burn_in",
            "inner_thin",
            "n_sir",
        ],
    )
    def run_mcmc(
        x_obs,
        rng_key,
        n_chains: int = 20,
        num_samples_per_chain: int = 500,
        burn_in: int = 500,
        inner_thin: int = 4,
        step_size: float = 0.05,
        n_sir: int = 100,
    ):
        k_init, k_mcmc = jax.random.split(rng_key)

        if init_strategy == "prior":
            curr_thetas = sample_prior_batch(k_init, n_chains)
            curr_lp = v_log_joint(curr_thetas, x_obs)
        else:

            def init_single_chain(k):
                if init_strategy == "jitter":
                    assert init_arr is not None
                    k1, _ = jax.random.split(k)
                    jitter = jax.random.normal(k1, shape=(dim_parameters,)) * 0.05
                    if prior_type == "lognormal":
                        cand_th = jnp.log(jnp.maximum(init_arr, 1e-12)) + jitter
                    else:
                        cand_th = init_arr + jitter
                        if prior_type == "uniform":
                            cand_th = jnp.clip(cand_th, prior_low, prior_high)
                    return cand_th, log_joint(cand_th, x_obs)

                if init_strategy == "sir_resample":
                    k1, k2 = jax.random.split(k)
                    cands = sample_prior_batch(k1, n_sir)
                    lps = v_log_joint(cands, x_obs)
                    safe_lps = jnp.where(jnp.isfinite(lps), lps, -jnp.inf)
                    max_lp = jnp.max(safe_lps)
                    has_finite = jnp.isfinite(max_lp)
                    shift = jnp.where(has_finite, max_lp, 0.0)
                    raw_w = jnp.where(has_finite, jnp.exp(safe_lps - shift), 1.0)
                    weights = raw_w / jnp.sum(raw_w)
                    idx = jax.random.choice(k2, n_sir, p=weights)
                    return cands[idx], lps[idx]

                # sir_argmax
                cands = sample_prior_batch(k, n_sir)
                lps = v_log_joint(cands, x_obs)
                safe_lps = jnp.where(jnp.isfinite(lps), lps, -jnp.inf)
                best_idx = jnp.argmax(safe_lps)
                return cands[best_idx], lps[best_idx]

            init_keys = jax.random.split(k_init, n_chains)
            curr_thetas, curr_lp = jax.vmap(init_single_chain)(init_keys)

        def scan_step(state, subkey):
            keys_th = jax.random.split(subkey, inner_thin)

            def theta_step(carry, subkey_th):
                th, lp = carry
                k_p, k_a = jax.random.split(subkey_th)
                prop_th = th + jax.random.normal(k_p, shape=th.shape) * step_size
                prop_lp = v_log_joint(prop_th, x_obs)
                log_u = jnp.log(jax.random.uniform(k_a, shape=(n_chains,)))
                acc = jnp.isfinite(prop_lp) & (log_u < prop_lp - lp)
                new_th = jnp.where(acc[:, None], prop_th, th)
                new_lp = jnp.where(acc, prop_lp, lp)
                return (new_th, new_lp), None

            (curr_thetas, curr_lp), _ = jax.lax.scan(theta_step, state, keys_th)
            return (curr_thetas, curr_lp), curr_thetas

        total_steps = burn_in + num_samples_per_chain
        keys = jax.random.split(k_mcmc, total_steps)
        _, all_samples = jax.lax.scan(scan_step, (curr_thetas, curr_lp), keys)
        post_burnin = all_samples[burn_in:]
        if prior_type == "lognormal":
            post_burnin = jnp.exp(post_burnin)
        return post_burnin.transpose(1, 0, 2).reshape(-1, dim_parameters)

    return run_mcmc
