from __future__ import annotations

from typing import final, override

from pde_solver.abc.boundary import BoundaryCondition
from pde_solver.pde_types import DType, Index, NDArray


@final
class ConstantDirichletBoundaryCondition(BoundaryCondition):
    """Boundary condition with constant value in all directions."""

    def __init__(self, value: DType) -> None:
        """
        Create a constant Dirichlet boundary condition.

        Parameters
        ----------
        value : DType
            The value that should be present at all points of the boundary.

        """
        self.value = value

    @override
    def __call__(self, state: NDArray, position: Index, time: float) -> DType:
        return self.value
