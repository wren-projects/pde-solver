from abc import ABC, abstractmethod

import numpy as np

from pde_solver.abc.boundary import BoundaryCondition
from pde_solver.abc.pde import PDE
from pde_solver.pde_types import DType, NDArray, Vector


class Solver(ABC):
    """Interface for PDE solvers."""

    def _initial_apply_boundary_condition(
        self, initial_condition: NDArray, boundary_condition: BoundaryCondition
    ) -> NDArray:
        padded = np.pad(initial_condition, pad_width=1)
        return self._apply_boundary_condition(
            padded, boundary_condition=boundary_condition, time=DType(0)
        )

    def _apply_boundary_condition(
        self, step: NDArray, boundary_condition: BoundaryCondition, time: DType
    ) -> NDArray:
        """Apply the boundary condition to a state."""
        return boundary_condition(step, time=time)

    def __call__(
        self,
        pde: PDE,
        initial_condition: NDArray,
        spacial_discretization_step: Vector,
        boundary_condition: BoundaryCondition,
        time_discretization_step: DType,
        time: DType,
    ) -> NDArray:
        """Compute the state at the given time given the PDE and it's conditions."""
        state = self._initial_apply_boundary_condition(
            initial_condition, boundary_condition
        )
        prev_state = state

        current_time = 0
        while current_time < time:
            current_time += time_discretization_step
            prev_state = state
            step = self._get_time_step(
                pde,
                state,
                spacial_discretization_step=spacial_discretization_step,
                time_discretization_step=time_discretization_step,
            )
            step = self._apply_boundary_condition(
                step=step, boundary_condition=boundary_condition, time=current_time
            )
            state = prev_state + step

        # we have probably calculated too high of a time, so we
        # need to interpolate with previous_state
        overshot_by = current_time - time
        return prev_state * overshot_by + state * (1 - overshot_by)

    @abstractmethod
    def _get_time_step(
        self,
        pde: PDE,
        state: NDArray,
        spacial_discretization_step: Vector,
        time_discretization_step: DType,
    ) -> NDArray:
        raise NotImplementedError
