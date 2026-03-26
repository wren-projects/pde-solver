from __future__ import annotations
from typing import final

from pde_solver.abc.boundary import BoundaryCondition
from pde_solver.pde_types import Index


@final
class ConstantDirichletBoundaryCondition(BoundaryCondition):
    """Boundary condition with constant value in all directions."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, state: NDArray, position: Index, time: float) -> float:
        return self.value


@final
class VariableDirichletBoundaryCondition(BoundaryCondition):
    """Boundary condition with variable value in all directions."""

    func: Callable[[float], float]

    def __init__(self, func: Callable[[float], float]) -> None:
        self.func = func

    def __call__(self, state: NDArray, position: Index, time: float) -> float:
        return self.func(time)
