from abc import ABC, abstractmethod
from typing import Any, get_args, get_origin

from pde_solver.abc.boundary import BoundaryCondition
from pde_solver.pde_types import DType, NDArray, Vector

registry = {}


class Solver[T](ABC):
    """Interface for PDE evolutionary solvers."""

    def __init_subclass__(cls, **kwargs: Any) -> None:  # pyright: ignore[reportAny]
        """Note down all subclasses created and their template type to registry."""
        super().__init_subclass__(**kwargs)

        for base in getattr(cls, "__orig_bases__", []):  # pyright: ignore[reportAny]
            if get_origin(base) is Solver:  # pyright: ignore[reportAny]
                args = get_args(base)
                if args:
                    registry[cls] = args[0]

    def __call__(
        self,
        pde: T,
        initial_condition: NDArray,
        spacial_discretization_step: Vector,
        boundary_condition: BoundaryCondition,
        time_discretization_step: DType,
        time: DType,
    ) -> NDArray:
        """
        Compute the state at the given time given the PDE and it's conditions.

        Here, a PDE is a triple of the partial differential equation itself, the initial
        condition and the boundary condition.

        Parameters
        ----------
        pde : T
            The PDE to be solved.
        initial_condition : NDArray
            The initial condition of the PDE, already discretized.
        spacial_discretization_step : Vector
            Describes how was the initial condition discretized. Needs to have the same
            length as initial_condition has dimensions.
        boundary_condition : BoundaryCondition
            The boundary condition for the PDE
        time_discretization_step : DType
            In how big time increaments should the solver emulate the PDE.
        time : DType
            The time at which we want to find the PDE's solution.

        Returns
        -------
        NDArray
            The state of the PDE at t=time.

        """
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
        pde: T,
        state: NDArray,
        spacial_discretization_step: Vector,
        time_discretization_step: DType,
    ) -> NDArray:
        """Compute the change to the state in one timestep."""
        raise NotImplementedError
