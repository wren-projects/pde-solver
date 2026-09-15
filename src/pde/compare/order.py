import numpy as np
from wren_pde_solver import PDE, SolverBuilderReady


def compute_spacial_order[T: PDE](solver_builder: SolverBuilderReady[T]) -> float:
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
    dims = solver_builder.get_initial_condition().ndim

    assert all(n % 4 == 1 for n in solver_builder.get_initial_condition().shape)  # pyright: ignore[reportAny]

    # We now need to gather the different computations at the same points
    # (i.e. at the grid points the least-fine solver uses)
    solutionA = solver_builder.compute()[(np.s_[::4],) * dims]

    solver_builder = solver_builder.set_spatial_step(
        solver_builder.get_spatial_step() * 2
    ).set_initial_condition(
        solver_builder.get_initial_condition()[(np.s_[::2],) * dims]
    )

    solutionB = (solver_builder.compute())[(np.s_[::2],) * dims]

    solver_builder = solver_builder.set_spatial_step(
        solver_builder.get_spatial_step() * 2
    ).set_initial_condition(
        solver_builder.get_initial_condition()[(np.s_[::2],) * dims]
    )

    solutionC = solver_builder.compute()

    # don't forget to remove boundaries cause they can be very messy
    point_wise_orders = -np.log2(
        np.abs((solutionA - solutionB) / (solutionB - solutionC))
    )[(np.s_[3:-3],) * dims]
    # print(point_wise_orders)
    return np.average(point_wise_orders)  # pyright: ignore[reportAny]
