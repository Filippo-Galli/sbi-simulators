"""Vectorized trajectory summary statistics shared across dynamical simulators."""

from collections.abc import Sequence

import torch


def _ensure_batch(us: torch.Tensor) -> tuple[torch.Tensor, bool]:
    is_2d = us.ndim == 2
    return (us.unsqueeze(0) if is_2d else us), is_2d


def per_channel_trajectory_stats(
    us: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-channel mean, log-variance, lag-1 and lag-2 autocorrelations.

    Parameters
    ----------
    us : torch.Tensor
        Trajectory batch of shape ``(N, D, T)``.
    eps : float, default 1e-6
        Lower clamp for variance denominator stability.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(means, log_vars, ac1, ac2)`` each of shape ``(N, D)``.
    """
    t_steps = us.shape[-1]
    means = us.mean(dim=-1)
    dus = us - means.unsqueeze(-1)
    vars_ = (dus**2).mean(dim=-1)
    denom = vars_.clamp_min(eps)
    log_vars = torch.log(denom)

    if t_steps >= 2:
        ac1 = (dus[:, :, 1:] * dus[:, :, :-1]).mean(dim=-1) / denom
    else:
        ac1 = torch.zeros_like(means)

    if t_steps >= 3:
        ac2 = (dus[:, :, 2:] * dus[:, :, :-2]).mean(dim=-1) / denom
    else:
        ac2 = torch.zeros_like(means)

    ac1 = torch.nan_to_num(ac1.clamp(-1.0, 1.0), nan=0.0)
    ac2 = torch.nan_to_num(ac2.clamp(-1.0, 1.0), nan=0.0)
    return means, log_vars, ac1, ac2


def pairwise_cross_correlations(
    us: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute zero-lag cross-correlations for specified channel pairs ``(i, j)``.

    Parameters
    ----------
    us : torch.Tensor
        Trajectory batch of shape ``(N, D, T)``.
    pairs : Sequence[tuple[int, int]]
        Channel index pairs to correlate.

    Returns
    -------
    torch.Tensor
        Cross-correlations of shape ``(N, len(pairs))``.
    """
    if not pairs:
        return torch.empty((us.shape[0], 0), dtype=us.dtype, device=us.device)

    dus = us - us.mean(dim=-1, keepdim=True)
    denom = (dus**2).mean(dim=-1).clamp_min(eps)
    idx_i = [p[0] for p in pairs]
    idx_j = [p[1] for p in pairs]
    cov = (dus[:, idx_i, :] * dus[:, idx_j, :]).mean(dim=-1)
    cc = cov / torch.sqrt(denom[:, idx_i] * denom[:, idx_j])
    return torch.nan_to_num(cc.clamp(-1.0, 1.0), nan=0.0)


def trajectory_stats_interleaved(us: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute ``4 * D`` stats per trajectory: ``[mean_d, log_var_d, ac1_d, ac2_d]`` per dim."""
    batch, is_2d = _ensure_batch(us)
    means, log_vars, ac1, ac2 = per_channel_trajectory_stats(batch, eps=eps)
    stats = torch.stack([means, log_vars, ac1, ac2], dim=-1).reshape(
        batch.shape[0], -1
    )
    return stats.squeeze(0) if is_2d else stats


def trajectory_stats_2d_lv(us: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute 9 Lotka-Volterra stats: ``[m0, m1, lv0, lv1, ac1_0, ac2_0, ac1_1, ac2_1, cc01]``."""
    batch, is_2d = _ensure_batch(us)
    means, log_vars, ac1, ac2 = per_channel_trajectory_stats(batch, eps=eps)
    cc = pairwise_cross_correlations(batch, [(0, 1)], eps=eps)
    stats = torch.cat(
        [
            means,
            log_vars,
            ac1[:, 0:1],
            ac2[:, 0:1],
            ac1[:, 1:2],
            ac2[:, 1:2],
            cc,
        ],
        dim=-1,
    )
    return stats.squeeze(0) if is_2d else stats


def trajectory_stats_2d_linear(us: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute 9 LinearSDE stats: ``[m0, m1, lv0, lv1, ac1_0, ac1_1, ac2_0, ac2_1, cc01]``."""
    batch, is_2d = _ensure_batch(us)
    means, log_vars, ac1, ac2 = per_channel_trajectory_stats(batch, eps=eps)
    cc = pairwise_cross_correlations(batch, [(0, 1)], eps=eps)
    stats = torch.cat([means, log_vars, ac1, ac2, cc], dim=-1)
    return stats.squeeze(0) if is_2d else stats


def trajectory_stats_3d_sir(us: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute 15 SIR stats: 3 means, 3 log-vars, interleaved (ac1, ac2) per channel, 3 cross-corrs."""
    batch, is_2d = _ensure_batch(us)
    means, log_vars, ac1, ac2 = per_channel_trajectory_stats(batch, eps=eps)
    cc = pairwise_cross_correlations(batch, [(0, 1), (1, 2), (0, 2)], eps=eps)
    ac_interleaved = torch.stack([ac1, ac2], dim=-1).reshape(batch.shape[0], -1)
    stats = torch.cat([means, log_vars, ac_interleaved, cc], dim=-1)
    return stats.squeeze(0) if is_2d else stats
