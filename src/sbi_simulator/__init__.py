"""sbi-simulator: Benchmark simulation models for Simulation-Based Inference."""

from importlib.metadata import PackageNotFoundError, version

from .AR1 import AR1
from .base import Simulator
from .BimodalGaussians import BimodalGaussians
from .compositional import extract_two_step_pairs, sample_two_step
from .DoubleWell import DoubleWell
from .dynamical import DynamicalSimulator
from .GaussianGaussian import GaussianGaussian
from .LinearSDE import LinearSDE
from .LotkaVolterra import LotkaVolterra
from .SIR import SIR
from .SLCP import SLCP
from .SpikeSlab import SpikeSlab
from .TwoMoons import TwoMoons
from .UniformGaussian import UniformGaussian

try:
    __version__ = version("sbi-simulator")
except PackageNotFoundError:
    __version__ = "0.2.2"

SIMULATORS = {
    "AR1": AR1,
    "BimodalGaussians": BimodalGaussians,
    "DoubleWell": DoubleWell,
    "GaussianGaussian": GaussianGaussian,
    "LinearSDE": LinearSDE,
    "LotkaVolterra": LotkaVolterra,
    "SIR": SIR,
    "SLCP": SLCP,
    "SpikeSlab": SpikeSlab,
    "TwoMoons": TwoMoons,
    "UniformGaussian": UniformGaussian,
}

__all__ = [
    "AR1",
    "SIMULATORS",
    "SIR",
    "SLCP",
    "BimodalGaussians",
    "DoubleWell",
    "DynamicalSimulator",
    "GaussianGaussian",
    "LinearSDE",
    "LotkaVolterra",
    "Simulator",
    "SpikeSlab",
    "TwoMoons",
    "UniformGaussian",
    "__version__",
    "extract_two_step_pairs",
    "sample_two_step",
]
