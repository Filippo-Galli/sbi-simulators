"""Shared JAX sampling utilities for SDE models (Particle Filter and Gibbs PMMH)."""

import functools

import jax
import jax.numpy as jnp


def systematic_resample(weights: jax.Array, key: jax.Array) -> jax.Array:
    """Systematic resampling of particles in JAX.

    Args:
        weights: Array of particle weights.
        key: JAX random key.

    Returns:
        Array of resampled particle indices.
    """
    n = weights.shape[0]
    u = jax.random.uniform(key, shape=()) / n
    points = u + jnp.arange(n) / n
    cumsum = jnp.cumsum(weights)
    idx = jnp.searchsorted(cumsum, points)
    return jnp.clip(idx, 0, n - 1)


def build_particle_filter_ll(
    step_one_fn,
    obs_dim: int,
    obs_sigma: float = 0.05,
):
    """Build a JIT-compiled Particle Filter for estimating marginal log-likelihood.

    Parameters
    ----------
    step_one_fn : callable
        Function `(curr_x, subkey, theta)` advancing a single particle state by one
        observation time-step (e.g. across inner integration sub-steps).
    obs_dim : int
        Dimensionality of the state / observation vector (e.g. 2 for LV, 3 for SIR).
    obs_sigma : float
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

            max_w = jnp.max(log_w)
            log_z_t = max_w + jnp.log(jnp.sum(jnp.exp(log_w - max_w))) - log_n

            norm_w = jnp.exp(log_w - (max_w + jnp.log(jnp.sum(jnp.exp(log_w - max_w)))))
            resamp_idx = systematic_resample(norm_w, k_resamp)
            resampled_particles = new_particles[resamp_idx]
            return (resampled_particles, next_k), log_z_t

        ts = jnp.arange(1, x_obs.shape[0])
        _, log_zs = jax.lax.scan(scan_body, (init_particles, key), ts)
        ll = jnp.sum(log_zs)
        has_nan = jnp.isnan(ll)
        return jnp.where(has_nan, -1e9, ll)

    return jax.jit(particle_filter_ll, static_argnames=["n_particles"])


def build_gibbs_pmmh_sampler(
    pf_ll_fn,
    dim_parameters: int,
):
    """Build a JIT-compiled Gibbs Particle Marginal Metropolis-Hastings (PMMH) sampler in JAX.

    Features:
    - Per-chain independent SIR (Sequential Importance Resampling) mode discovery from prior
    - Gibbs updates alternating between inner_thin parameter steps (frozen particle key)
      and independent MH step on the particle filter key
    - Standard Normal prior N(0, I_D) on theta
    """
    log_prior_norm = (dim_parameters / 2.0) * jnp.log(2.0 * jnp.pi)

    def log_joint(theta, x_obs, key, n_particles: int = 150):
        lp_prior = -0.5 * jnp.sum(theta**2) - log_prior_norm
        ll = pf_ll_fn(theta, x_obs, key, n_particles=n_particles)
        return lp_prior + ll

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
            best_th = thetas_sir[jnp.argmax(lps)]
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
                alpha = jnp.minimum(1.0, jnp.exp(prop_lp - lp))
                acc = jax.random.uniform(k_a, shape=(n_chains,)) < alpha
                new_th = jnp.where(acc[:, None], prop_th, th)
                new_lp = jnp.where(acc, prop_lp, lp)
                return (new_th, k_pf, new_lp), None

            (curr_thetas, curr_pf_keys, curr_lp), _ = jax.lax.scan(
                theta_step, state, keys_th
            )

            # 1 step on key
            prop_pf_keys = jax.random.split(k_key, n_chains)
            prop_lp_key = v_log_joint(curr_thetas, x_obs, prop_pf_keys, n_particles)
            alpha_key = jnp.minimum(1.0, jnp.exp(prop_lp_key - curr_lp))
            acc_key = jax.random.uniform(k_acc_k, shape=(n_chains,)) < alpha_key
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
