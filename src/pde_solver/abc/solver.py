from abc import ABC, abstractmethod

from pde_solver.abc.boundary import BoundaryCondition
from pde_solver.abc.pde import PDE
from pde_solver.pde_types import NDArray


class Solver(ABC):
    """Interface for PDE solvers."""

    @abstractmethod
    def __call__(
        self,
        pde: PDE,
        initial_condition: NDArray,
        discretization_step: DType,
        boundary_condition: BoundaryCondition,
        time: DType,
    ) -> NDArray:
        """Compute the state at the given time given the PDE and it's conditions."""
        raise NotImplementedError
