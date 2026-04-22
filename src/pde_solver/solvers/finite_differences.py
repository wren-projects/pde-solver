from typing import final, override

from pde_solver.abc.pde import PDE
from pde_solver.abc.solver import Solver
from pde_solver.pde_types import DType, NDArray, Vector


@final
class FiniteDifferences(Solver):
    """Solver using the Finite Differences method."""

    timestep: DType

    STABILITY_TRESHOLD = 0.5

    def __init__(self, timestep: DType) -> None:
        """
        Initialize the Finite Differences solver.

        Parameters
        ----------
        timestep: float
            The timestep to use for the solver.

        """
        super().__init__()

        self.timestep = timestep

    @override
    def _get_time_step(
        self,
        pde: PDE,
        state: NDArray,
        spacial_discretization_step: Vector,
        time_discretization_step: DType,
    ) -> NDArray:
        raise NotImplementedError
