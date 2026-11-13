"""sbi-simulator: Benchmark simulation models for Simulation-Based Inference."""

from .AR1 import AR1
from .BimodalGaussians import BimodalGaussians
from .compositional import extract_two_step_pairs, sample_two_step
from .DoubleWell import DoubleWell
from .GaussianGaussian import GaussianGaussian
from .LinearSDE import LinearSDE
from .LotkaVolterra import LotkaVolterra
from .Simulator import Simulator
from .SIR import SIR
from .SLCP import SLCP
from .SpikeSlab import SpikeSlab
from .TwoMoons import TwoMoons
from .UniformGaussian import UniformGaussian

__version__ = "0.1.0"

SIMULATORS = {
    "BimodalGaussians": BimodalGaussians,
    "DoubleWell": DoubleWell,
    "GaussianGaussian": GaussianGaussian,
    "LinearSDE": LinearSDE,
    "SpikeSlab": SpikeSlab,
    "SIR": SIR,
    "SLCP": SLCP,
    "TwoMoons": TwoMoons,
    "LotkaVolterra": LotkaVolterra,
    "UniformGaussian": UniformGaussian,
    "AR1": AR1,
}

__all__ = [
    "AR1",
    "SIMULATORS",
    "SIR",
    "SLCP",
    "BimodalGaussians",
    "DoubleWell",
    "GaussianGaussian",
    "LinearSDE",
    "LotkaVolterra",
    "Simulator",
    "SpikeSlab",
    "TwoMoons",
    "UniformGaussian",
    "extract_two_step_pairs",
    "sample_two_step",
]

