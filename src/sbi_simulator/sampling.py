"""Shared sampling utilities for box-constrained benchmark simulators."""

from collections.abc import Callable

import numpy as np
import torch
from scipy import special as sp_special


def log_normal_box_prob(
    means: np.ndarray,
    sigma: float,
    low: float,
    high: float,
) -> np.ndarray:
    """Compute elementwise ``log P(X in [low, high])`` for ``X ~ N(means, sigma^2)``.

    Uses tail-symmetry of ``scipy.special.log_ndtr`` and ``log(-expm1(...))``
    to stay numerically stable deep in either tail.
    """
    a = (low - means) / sigma
    b = (high - means) / sigma

    # For a <= 0: Phi(b) - Phi(a) = Phi(b) * (1 - exp(log_Phi(a) - log_Phi(b)))
    # For a > 0:  Phi(b) - Phi(a) = Phi(-a) - Phi(-b) = Phi(-a) * (1 - exp(log_Phi(-b) - log_Phi(-a)))
    hi_arg = np.where(a <= 0.0, b, -a)
    lo_arg = np.where(a <= 0.0, a, -b)

    log_hi = sp_special.log_ndtr(hi_arg)
    log_lo = sp_special.log_ndtr(lo_arg)
    diff = np.minimum(log_lo - log_hi, -1e-300)
    return log_hi + np.log(-np.expm1(diff))


def rejection_sample_box(
    propose_fn: Callable[[int], torch.Tensor],
    num_samples: int,
    low: float,
    high: float,
    *,
    batch_multiplier: int = 2,
    min_batch: int = 1000,
    max_attempts: int = 200,
    fallback_fn: Callable[[int], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Draw ``num_samples`` points inside ``[low, high]^d`` via batched rejection sampling.

    Parameters
    ----------
    propose_fn : Callable[[int], torch.Tensor]
        Callable taking ``batch_size`` and returning candidate samples of shape ``(batch_size, d)``.
    num_samples : int
        Target number of accepted samples.
    low, high : float
        Lower and upper bounds of the hypercube ``[low, high]^d``.
    batch_multiplier : int, default 2
        Multiplier applied to remaining sample count for each proposal batch.
    min_batch : int, default 1000
        Minimum proposal batch size per attempt.
    max_attempts : int, default 200
        Maximum rejection rounds before invoking ``fallback_fn`` or raising ``RuntimeError``.
    fallback_fn : Callable[[int], torch.Tensor], optional
        Optional exact/fallback sampler invoked with remaining sample count if rejection
        does not finish within ``max_attempts``.
    """
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}.")
    if num_samples == 0:
        return propose_fn(1)[:0]

    out: list[torch.Tensor] = []
    collected = 0

    for _ in range(max_attempts):
        if collected >= num_samples:
            break
        needed = num_samples - collected
        batch_size = max(batch_multiplier * needed, min_batch)
        candidates = propose_fn(batch_size)
        keep = ((candidates >= low) & (candidates <= high)).all(dim=-1)
        valid = candidates[keep]
        if valid.shape[0] > 0:
            take = min(valid.shape[0], needed)
            out.append(valid[:take])
            collected += take

    if collected >= num_samples:
        return torch.cat(out, dim=0)[:num_samples]

    remaining = num_samples - collected
    if fallback_fn is not None:
        fallback = fallback_fn(remaining)[:remaining]
        if out:
            return torch.cat([torch.cat(out, dim=0), fallback], dim=0)
        return fallback

    raise RuntimeError(
        f"Rejection sampling failed to collect {num_samples} samples in "
        f"[{low}, {high}] after {max_attempts} attempts (collected {collected})."
    )
