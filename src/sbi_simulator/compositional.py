"""Utilities for Compositional and Markov Simulation-Based Inference."""

import torch

from .Simulator import Simulator


def extract_two_step_pairs(
    trajectories: torch.Tensor,
    thetas: torch.Tensor | None = None,
    x0s: torch.Tensor | None = None,
    t: int | torch.Tensor | None = None,
    return_dict: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Extract consecutive transition pairs (x_t, x_{t+1}) from full trajectories.

    Parameters
    ----------
    trajectories : torch.Tensor
        Batch of trajectories. Can be formatted as either:
        - (batch_size, num_steps, state_dim) or
        - (batch_size, state_dim, num_steps)
    thetas : torch.Tensor, optional
        Parameter tensor of shape (batch_size, dim_parameters).
    x0s : torch.Tensor, optional
        Initial states of shape (batch_size, state_dim).
    t : int or torch.Tensor, optional
        Specific time index or tensor of time indices. If None, sampled
        uniformly at random from {0, ..., T - 2}.
    return_dict : bool, default False
        If True, returns a dictionary with keys:
        'thetas', 'xs', 'ts', 'x0s' (matching the markovsbi training interface).
        If False, returns tensor of shape (batch_size, 2, state_dim).
    """
    traj = torch.as_tensor(trajectories, dtype=torch.float32)
    is_single = traj.ndim == 2

    # Standardize to (batch_size, num_steps, state_dim)
    if is_single:
        traj = traj.unsqueeze(0)

    # If shape is (batch, state_dim, num_steps) with state_dim < num_steps, transpose
    if traj.shape[1] < traj.shape[2]:
        traj = traj.permute(0, 2, 1)

    batch_size, total_steps, _ = traj.shape
    if total_steps < 2:
        raise ValueError(
            f"Trajectory must have at least 2 time steps to extract pairs, got {total_steps}."
        )

    max_idx = total_steps - 2

    if t is None:
        ts = torch.randint(0, max_idx + 1, (batch_size,))
    elif isinstance(t, int):
        ts = torch.full((batch_size,), t, dtype=torch.long)
    else:
        ts = torch.as_tensor(t, dtype=torch.long)
        if ts.ndim == 0:
            ts = ts.expand(batch_size)

    safe_ts = ts.clamp(0, max_idx)
    batch_idx = torch.arange(batch_size)

    x_t = traj[batch_idx, safe_ts, :]
    x_next = traj[batch_idx, safe_ts + 1, :]
    xs = torch.stack([x_t, x_next], dim=1)  # (batch_size, 2, state_dim)

    if return_dict:
        out = {
            "xs": xs.squeeze(0) if is_single else xs,
            "ts": ts.squeeze(0) if is_single else ts,
        }
        if thetas is not None:
            out["thetas"] = thetas
        if x0s is not None:
            out["x0s"] = x0s.squeeze(0) if is_single else x0s
        else:
            out["x0s"] = traj[:, 0, :].squeeze(0) if is_single else traj[:, 0, :]
        return out

    return xs.squeeze(0) if is_single else xs


def sample_two_step(
    simulator: Simulator,
    parameters: torch.Tensor,
    x0: torch.Tensor | None = None,
    t: int | torch.Tensor | None = None,
    informed: bool = True,
    return_dict: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Two-step transition generator.

    Parameters
    ----------
    simulator : Simulator
        Simulator instance providing `get_prior_x0()`, `stepper()`, or trajectory simulation.
    parameters : torch.Tensor
        Parameters of shape (dim_parameters,) or (batch_size, dim_parameters).
    x0 : torch.Tensor, optional
        Initial state(s). If None, sampled from `simulator.get_prior_x0()`.
    t : int or torch.Tensor, optional
        Target time step(s). Sampled uniformly if None and informed=True.
    informed : bool, default True
        - True: Runs full trajectory under parameters and extracts (x_t, x_{t+1})
          at a random time t, capturing the true dynamical manifold.
        - False: Starts directly from prior x_0 (t=0) and takes one transition step.
    return_dict : bool, default False
        Whether to return the dict format or the (batch, 2, state_dim) tensor.
    """
    params = torch.as_tensor(parameters, dtype=torch.float32)
    is_1d = params.ndim == 1
    if is_1d:
        params = params.unsqueeze(0)

    batch_size = params.shape[0]

    # Resolve initial condition x0
    if x0 is None:
        if hasattr(simulator, "get_prior_x0"):
            init_x = simulator.get_prior_x0().sample((batch_size,))
        elif hasattr(simulator, "u0"):
            init_x = simulator.u0.unsqueeze(0).expand(batch_size, -1)
        else:
            raise ValueError(
                f"Simulator {simulator.__class__.__name__} has neither get_prior_x0() nor u0."
            )
        if init_x.ndim == 1:
            init_x = init_x.unsqueeze(0)
    else:
        init_x = torch.as_tensor(x0, dtype=torch.float32)
        if init_x.ndim == 1:
            init_x = init_x.unsqueeze(0)
        if init_x.shape[0] == 1 and batch_size > 1:
            init_x = init_x.expand(batch_size, -1)

    if not informed:
        ts = torch.zeros(batch_size, dtype=torch.long)
        if hasattr(simulator, "stepper"):
            x_t = init_x
            x_next = simulator.stepper(x_t, params)
            xs = torch.stack([x_t, x_next], dim=1)
        else:
            # Fallback: simulate short trajectory of length 2
            if hasattr(simulator, "_solve"):
                traj = simulator._solve(params, u0=init_x)
            else:
                traj = simulator.simulate(params)
            return extract_two_step_pairs(
                traj, thetas=params, x0s=init_x, t=0, return_dict=return_dict
            )

        if return_dict:
            return {
                "thetas": parameters if is_1d else params,
                "xs": xs.squeeze(0) if is_1d else xs,
                "ts": ts.squeeze(0) if is_1d else ts,
                "x0s": init_x.squeeze(0) if is_1d else init_x,
            }
        return xs.squeeze(0) if is_1d else xs

    # informed=True: simulate trajectory and slice
    if hasattr(simulator, "_solve"):
        traj = simulator._solve(params, u0=init_x)
    else:
        traj = simulator.simulate(params)

    return extract_two_step_pairs(
        traj,
        thetas=parameters if is_1d else params,
        x0s=init_x.squeeze(0) if is_1d else init_x,
        t=t,
        return_dict=return_dict,
    )
