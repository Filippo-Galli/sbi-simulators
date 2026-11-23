import math
import os

import corner
import matplotlib
import torch

from sbi_simulator import SIMULATORS, Simulator

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")


def print_corner(simulator: Simulator, simulator_name: str):
    if hasattr(simulator, "canonical_theta") and simulator.canonical_theta is not None:
        theta_0 = simulator.canonical_theta.unsqueeze(0)
    else:
        theta_0 = simulator.get_prior().sample((1,))

    x_0 = simulator.simulate(theta_0)
    samples = simulator.posterior_samples(x_0, num_samples=10000)
    plot = corner.corner(
        samples.detach().cpu().numpy(),
        truths=theta_0.squeeze().detach().cpu().numpy(),
        labels=[rf"$\theta_{i + 1}$" for i in range(simulator.dim_parameters)],
    )

    plot.savefig(f"corner_{simulator_name}_{simulator.seed}.png")


if __name__ == "__main__":
    seeds = [42, 1, 2, 3]
    sim = SIMULATORS["SIR"]
    for seed in seeds:
        print(f"Simulator: {sim.__name__}")
        simulator = sim(preset="sbibm", seed=seed)
        print_corner(simulator, sim.__name__)
