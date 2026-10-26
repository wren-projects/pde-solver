from __future__ import annotations

from typing import cast, final, override

import numpy as np

from wren_pde_solver.abc.boundary import BoundaryCondition
from wren_pde_solver.pde_types import NDArray


@final
class ConstantDirichletBoundaryCondition(BoundaryCondition):
    """Boundary condition with constant value in all directions."""

    def __init__(self, value: float) -> None:
        """
        Create a boundary condition where one single value is present everywhere.

        Parameters
        ----------
        value : float
            The value that is present at every point of the boundary
            and outside the grid.

        """
        self.value = value

    @override
    def apply_to_initial_condition[T: NDArray](self, state: T) -> T:
        return cast(T, np.pad(state, pad_width=1, constant_values=self.value))

    @override
    def __call__[T: NDArray](self, state_diff: T, time: float, delta_time: float) -> T:
        slices: list[slice | int] = [slice(None)] * state_diff.ndim
        for dim in range(state_diff.ndim):
            slices[dim] = 0
            state_diff[*slices] = 0
            slices[dim] = state_diff.shape[dim] - 1
            state_diff[*slices] = 0
            slices[dim] = slice(None)
        return state_diff
