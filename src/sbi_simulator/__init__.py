"""sbi-simulator: Benchmark simulation models for Simulation-Based Inference."""

from .BimodalGaussians import BimodalGaussians
from .GaussianGaussian import GaussianGaussian
from .LotkaVolterra import LotkaVolterra
from .Simulator import Simulator
from .SIR import SIR
from .SpikeSlab import SpikeSlab
from .TwoMoons import TwoMoons
from .UniformGaussian import UniformGaussian

__version__ = "0.1.0"

SIMULATORS = {
    "BimodalGaussians": BimodalGaussians,
    "GaussianGaussian": GaussianGaussian,
    "SpikeSlab": SpikeSlab,
    "SIR": SIR,
    "TwoMoons": TwoMoons,
    "LotkaVolterra": LotkaVolterra,
    "UniformGaussian": UniformGaussian,
}

__all__ = [
    "SIMULATORS",
    "Simulator",
    "BimodalGaussians",
    "GaussianGaussian",
    "SpikeSlab",
    "SIR",
    "TwoMoons",
    "LotkaVolterra",
    "UniformGaussian",
]
