from math import ceil
from typing import final, override

from pde_solver.abc.boundary import BoundaryCondition
from pde_solver.abc.pde import PDE
from pde_solver.abc.solver import Solver
from pde_solver.pde_types import DType, NDArray


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
    def __call__(
        self,
        pde: PDE,
        initial_condition: NDArray,
        discretization_step: DType,
        boundary_condition: BoundaryCondition,
        time: DType,
    ) -> NDArray:
        """Compute the state at the given time given the PDE and it's conditions."""
        r = self.timestep / discretization_step**2

        if r > self.STABILITY_TRESHOLD:
            raise ValueError("Timestep too big")

        dim = initial_condition.ndim

        state = initial_condition

        state = boundary_condition(state, 0.0)  # enforce boundary_condition at time = 0

        # look for the class of the PDE and set parameters True/False accordingly
        # So it knows if it needs time_derivative, second_order space_deriv, and first order space_deri
        for step in range(ceil(time / self.timestep)):
            current_time = step * self.timestep

            new_state = state.copy()
            for i in range(dim):
                # update state
                pass
            state = new_state

        return state
