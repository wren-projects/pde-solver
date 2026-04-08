from __future__ import annotations

from typing import final, override

from pde_solver.abc.boundary import BoundaryCondition
from pde_solver.pde_types import DType, Index, NDArray


@final
class ConstantDirichletBoundaryCondition(BoundaryCondition):
    """Boundary condition with constant value in all directions."""

    def __init__(self, value: float) -> None:
        """One single value is present at all points outside the grid."""
        self.value = value

    @override
    def __call__(self, state: NDArray, time: float) -> NDArray:
        """Set the boundary condition for the given state at a given time."""
        state = state.copy()
        slices: list[slice | int] = [slice(None)] * state.ndim
        for dim in range(state.ndim):
            slices[dim] = 0
            state[*slices] = self.value
            slices[dim] = state.shape[dim] - 1
            state[*slices] = self.value
            slices[dim] = slice(None)
        return state
