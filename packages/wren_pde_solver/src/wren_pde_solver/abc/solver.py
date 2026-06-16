# ruff: noqa: PLR0913
from abc import ABC, abstractmethod
from types import get_original_bases
from typing import Any, get_args, get_origin

from wren_pde_solver.abc.boundary import BoundaryCondition
from wren_pde_solver.abc.pde import PDE
from wren_pde_solver.pde_types import DType, NDArray, Vector

registry: dict[type, type] = {}


class Solver[T: PDE](ABC):
    """Interface for PDE evolutionary solvers."""

    def __init_subclass__(cls, **kwargs: Any) -> None:  # pyright: ignore[reportAny]
        """Note down all subclasses created and their template type to registry."""
        super().__init_subclass__(**kwargs)

        for base in get_original_bases(cls):  # pyright: ignore[reportAny]
            if get_origin(base) is Solver:  # pyright: ignore[reportAny]
                args = get_args(base)
                if args:
                    registry[cls] = args[0]

    def __call__(
        self,
        pde: T,
        initial_condition: NDArray,
        spacial_step: Vector,
        boundary_condition: BoundaryCondition,
        time_step: DType,
        target_time: DType,
    ) -> NDArray:
        """
        Compute the state at the given time of the given partial differential equation.

        Here, a partial differential equation is a triple of the PDE itself, the initial
        condition, and the boundary condition.

        Parameters
        ----------
        pde : T
            The PDE to be solved.
        initial_condition : NDArray
            The initial condition of the PDE, already discretized.
        spacial_step : Vector
            Describes how was the initial condition discretized. Needs to have the same
            length as initial_condition has dimensions.
        boundary_condition : BoundaryCondition
            The boundary condition for the PDE
        time_step : DType
            In how big time increaments should the solver emulate the PDE.
        target_time : DType
            The time at which we want to find the PDE's solution.

        Returns
        -------
        NDArray
            The state of the PDE at t=time.

        """
        state = boundary_condition.apply_to_initial_condition(initial_condition)
        prev_state = state

        current_time = 0
        while current_time < target_time:
            current_time += time_step
            step = self._compute_time_step(
                pde=pde,
                state=state,
                spacial_step=spacial_step,
                time_step=time_step,
            )
            step = boundary_condition(
                state_diff=step, time=target_time, delta_time=time_step
            )
            prev_state = state
            state = prev_state + step

        # We probably overshot the target time due to the discretization, so
        # we fix it using a linear interpolation between current and previous state.
        overshot_by = (current_time - target_time) / time_step
        return boundary_condition.remove_boundary(
            prev_state * overshot_by + state * (1 - overshot_by)
        )

    @abstractmethod
    def _compute_time_step(
        self,
        pde: T,
        state: NDArray,
        spacial_step: Vector,
        time_step: DType,
    ) -> NDArray:
        """Compute the change to the state in one timestep."""
        raise NotImplementedError
