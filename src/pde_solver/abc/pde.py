from collections.abc import Callable
from dataclasses import dataclass

from pde_solver.pde_types import Matrix, Vector


@dataclass
class PDE:
    """General diffusion-convection partial differential equation."""

    type ConvectionDirectionFunction = Callable[[Vector], Vector]
    type DiffusionCoefficientFunction = Callable[[Vector], Matrix]
    type InhomogeneousFunction = Callable[[Vector], Vector]

    dimension: int
    convection_direction_function: ConvectionDirectionFunction
    diffusion_coefficient_function: DiffusionCoefficientFunction
    inhomogeneous_function: InhomogeneousFunction


# TODO: Add subclasses for PDE types
