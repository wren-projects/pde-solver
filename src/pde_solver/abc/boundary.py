from abc import ABC, abstractmethod

from pde_solver.pde_types import NDArray


class BoundaryCondition(ABC):
    """Interface for boundary conditions."""

    @abstractmethod
    def apply_to_initial_condition(self, state: NDArray) -> NDArray:
        """
        Apply the boundary condition to the initial state.

        Parameters
        ----------
        state : NDArray
            The state at which the boundary condition should be applied.

        Returns
        -------
        NDArray
            The state with the boundary condition added. The size is increased by one in
            each direction (two in each axis) by adding fields corresponding to the
            boundary condition.

        """

    @abstractmethod
    def __call__(self, state_diff: NDArray, time: float, delta_time: float) -> NDArray:
        """
        Apply the boundary condition for a specific state_diff.

        Parameters
        ----------
        state_diff : NDArray
            The difference between the new and previous state as computed by some
            iteration method.
        time : float
            the time of the new state.
        delta_time : float
            The time between the new and previous states.

        Returns
        -------
        NDArray
            The state_diff adjusted so that it preserves the boundary condition.

        """
