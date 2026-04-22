from __future__ import annotations

from typing import final, override

import numpy as np

from pde_solver.abc.boundary import BoundaryCondition
from pde_solver.pde_types import NDArray


@final
class ConstantDirichletBoundaryCondition(BoundaryCondition):
    """Boundary condition with constant value in all directions."""

    def __init__(self, value: float) -> None:
        """One single value is present at all points outside the grid."""
        self.value = value

    @override
    def apply_to_initial_condition(self, state: NDArray) -> NDArray:
        padded = np.pad(state, pad_width=1)
        slices: list[slice | int] = [slice(None)] * padded.ndim
        for dim in range(padded.ndim):
            slices[dim] = 0
            padded[*slices] = self.value
            slices[dim] = padded.shape[dim] - 1
            padded[*slices] = self.value
            slices[dim] = slice(None)
        return padded

    @override
    def __call__(self, state_diff: NDArray, time: float, delta_time: float) -> NDArray:
        slices: list[slice | int] = [slice(None)] * state_diff.ndim
        for dim in range(state_diff.ndim):
            slices[dim] = 0
            state_diff[*slices] = 0
            slices[dim] = state_diff.shape[dim] - 1
            state_diff[*slices] = 0
            slices[dim] = slice(None)
        return state_diff
