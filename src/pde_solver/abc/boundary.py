from abc import ABC, abstractmethod

from pde_solver.pde_types import DType, Index, NDArray


class BoundaryCondition(ABC):
    """Interface for boundary conditions."""

    @abstractmethod
    def __call__(self, state: NDArray, time: float) -> NDArray:
        """Return the boundary values in given point at given time."""
        raise NotImplementedError
