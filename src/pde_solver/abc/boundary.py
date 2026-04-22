from abc import ABC, abstractmethod

from pde_solver.pde_types import NDArray


class BoundaryCondition(ABC):
    """Interface for boundary conditions."""

    @abstractmethod
    def apply_to_initial_condition(self, state: NDArray) -> NDArray:
        """
        Apply the boundary condition to the initial state.

        Increase the state size by one in each direction (two in each axis)
        by adding fields corresponding to the boundary condition.
        """

    @abstractmethod
    def __call__(self, state_diff: NDArray, time: float, delta_time: float) -> NDArray:
        """
        Apply the boundary condition for a specific state_diff.

        State_diff in this context is meant as the state delta within
        a single time-step.
        """
