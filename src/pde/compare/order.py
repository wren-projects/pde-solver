from typing import cast

import numpy as np
from numpy._typing import NDArray
from wren_pde_solver import PDE, CompleteSolutionBuilder
from wren_pde_solver.pde_types import DType
from wren_ttd import TTD


def compute_spacial_order[T: PDE, S: NDArray[DType] | TTD[DType]](
    solver_builder: CompleteSolutionBuilder[T, S],
) -> float:
    """
    Compute the order of a given solver in a given problem using step doubling.

    Parameters
    ----------
    solver_builder : SolverBuilderReady
        An InnerSolverBuilder (created via SolverBuilder) with all its arguments set.
        This specifies exactly how the solver is to be run. This function then takes
        the spacial_step and doubles/halfs it to find the order of the method in space.
        Note the initial condition needs to have each dimension size to be 1 mod 4.

    Returns
    -------
    float
        The spacial order of the method.

    Notes
    -----
    Let f(dx) be a method with spacial step dx. Let g be the true solution.
    e(dx) = g - f(dx) is called the error of the method (as a function of spacial step).
    For many methods, e behaves like a polynom of dx. We call the order of this polynom
    the spacial order of f.

    """
    dims = solver_builder.initial_condition.ndim

    assert all(n % 4 == 1 for n in solver_builder.initial_condition.shape)

    # We now need to gather the different computations at the same points
    # (i.e. at the grid points the least-fine solver uses)
    solutionA = cast(S, solver_builder.compute()[(np.s_[::4],) * dims])

    solver_builder = solver_builder.with_spatial_step(
        solver_builder.spatial_step * 2
    ).with_initial_condition(
        cast(S, solver_builder.initial_condition[(np.s_[::2],) * dims])
    )

    solutionB = cast(S, (solver_builder.compute())[(np.s_[::2],) * dims])

    solver_builder = solver_builder.with_spatial_step(
        solver_builder.spatial_step * 2
    ).with_initial_condition(
        cast(
            S,
            solver_builder.initial_condition[(np.s_[::2],) * dims],
        )
    )

    solutionC = solver_builder.compute()

    # don't forget to remove boundaries cause they can be very messy
    BORDER_SIZE = 3
    a = solutionA[(np.s_[BORDER_SIZE:-BORDER_SIZE],) * dims]
    b = solutionB[(np.s_[BORDER_SIZE:-BORDER_SIZE],) * dims]
    c = solutionC[(np.s_[BORDER_SIZE:-BORDER_SIZE],) * dims]

    return cast(float, -np.log2(np.linalg.norm(a - b) / np.linalg.norm(b - c)))
