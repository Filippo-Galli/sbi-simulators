from typing import override

import torch
from ..Simulator import Simulator
from pyro import distributions as pdist


class BimodalGaussians(Simulator):
    def __init__(
        self,
        dim_parameters: int = 2,
        seed: int = 42,
        sigma: float = 0.1,
        dim_data: int | None = None,
    ):
        self.sigma = float(sigma)
        self.shift = 10.0 * self.sigma
        if dim_data is None:
            dim_data = dim_parameters
        super().__init__(seed=seed, dim_parameters=dim_parameters, dim_data=dim_data)

    def get_prior(self) -> pdist.Distribution:
        return pdist.Uniform(
            -10.0 * torch.ones(self.dim_parameters),
            10.0 * torch.ones(self.dim_parameters),
        ).to_event(1)

    @override
    def simulate(self, parameters: torch.Tensor) -> torch.Tensor:
        params = torch.as_tensor(parameters, dtype=torch.float32)
        if params.numel() == 0:
            raise ValueError("Expected at least one parameter.")

        is_1d = params.ndim == 1
        if is_1d:
            params = params.unsqueeze(0)

        batch_size = params.shape[0]
        # Single sign per simulation sample (bimodal: mode 1 vs mode 2 across all dimensions)
        signs = torch.where(torch.rand((batch_size, 1), device=params.device) < 0.5, 1.0, -1.0)
        out = params + signs * self.shift + self.sigma * torch.randn_like(params)
        return out.squeeze(0) if is_1d else out

    def posterior_samples(
        self,
        x_obs: torch.Tensor | None = None,
        num_samples: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the true posterior for BimodalGaussians under Uniform([-10, 10]^d).
        For an observation x_obs, the posterior is a symmetric mixture of two Gaussians
        centered at x_obs - shift and x_obs + shift, truncated to [-10, 10]^d.
        """
        if x_obs is None and "num_samples" in kwargs and isinstance(num_samples, torch.Tensor):
            x_obs, num_samples = num_samples, kwargs.get("num_samples", 1000)
        if x_obs is None:
            x_obs = torch.zeros(self.dim_data, dtype=torch.float32)

        x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32)
        if x_obs_t.ndim == 1:
            x_obs_t = x_obs_t.unsqueeze(0)
        x_center = x_obs_t.mean(dim=0)
        d = x_center.shape[0]
        lo, hi = -10.0, 10.0

        out = []
        collected = 0
        while collected < num_samples:
            needed = num_samples - collected
            batch = max(2 * needed, 1000)
            signs = torch.where(torch.rand((batch, 1)) < 0.5, 1.0, -1.0)
            means = x_center.unsqueeze(0) - signs * self.shift
            candidates = means + self.sigma * torch.randn(batch, d)
            keep = torch.all((candidates >= lo) & (candidates <= hi), dim=-1)
            valid = candidates[keep]
            if valid.shape[0] > 0:
                out.append(valid)
                collected += valid.shape[0]

        samples = torch.cat(out, dim=0)[:num_samples]
        return samples
