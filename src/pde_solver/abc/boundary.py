from abc import ABC, abstractmethod

from pde_solver.our_types import DType, Vector


class BoundaryCondition(ABC):
    """Interface for boundary conditions."""

    @abstractmethod
    def __call__(self, position: Vector, time: float) -> DType:
        """Return the boundary values in given point at given time."""
        raise NotImplementedError
