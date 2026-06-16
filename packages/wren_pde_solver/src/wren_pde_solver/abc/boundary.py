from abc import ABC, abstractmethod

from wren_pde_solver.pde_types import NDArray


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

    @staticmethod
    def remove_boundary(state: NDArray) -> NDArray:
        """
        Remove boundary that was added to the state via apply_to_initial_condition.

        Should be a an inverse to apply_to_initial_condition.

        Parameters
        ----------
        state : NDArray
            The state from which boundary is to be removed.

        Returns
        -------
        NDArray
            The state with the boundary removed.

        """
        slices = tuple(slice(1, -1) for _ in range(state.ndim))

        return state[slices]

    @abstractmethod
    def __call__(self, state_diff: NDArray, time: float, delta_time: float) -> NDArray:
        """
        Apply the boundary condition for a specific state_diff.

        Note this operation modifies the input state_diff!

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
