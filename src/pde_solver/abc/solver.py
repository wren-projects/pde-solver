from abc import ABC, abstractmethod

from pde_solver.abc.boundary import BoundaryCondition
from pde_solver.abc.pde import PDE
from pde_solver.pde_types import DType, NDArray, Vector


class Solver(ABC):
    """Interface for PDE evolutionary solvers."""

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
        state = boundary_condition.apply_to_initial_condition(initial_condition)
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
            step = boundary_condition(
                state_diff=step, time=time, delta_time=time_discretization_step
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
