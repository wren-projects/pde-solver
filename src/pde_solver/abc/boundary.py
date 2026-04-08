from abc import ABC, abstractmethod

from pde_solver.pde_types import DType, Index, NDArray


class BoundaryCondition(ABC):
    """Interface for boundary conditions."""

    @abstractmethod
    def __call__(self, state: NDArray, time: float) -> NDArray:
        """Set the boundary condition for the given state at a given time."""
        raise NotImplementedError
