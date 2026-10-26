import numpy as np
from wren_pde_solver import boundary, partial_differential_equations
from wren_pde_solver.solution_builder import EMPTY_SOLUTION_BUILDER
from wren_pde_solver.solvers.finite_differences import FiniteDifferences
from wren_ttd import TTD

from pde.compare import compute_spacial_order


def test_spacial_order_returns_something() -> None:
    """Test spacial order returns something that remotly makes sense."""

    def fce(x, y):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]  # noqa: ANN001, ANN202
        return np.e ** (-(x**2 + y**2))  # pyright: ignore[reportUnknownVariableType]

    X_grid, Y_grid = np.meshgrid(np.linspace(-3, 3, 121), np.linspace(-3, 3, 121))
    IC = fce(X_grid, Y_grid)
    solver = (
        EMPTY_SOLUTION_BUILDER.with_spatial_step(np.array([0.1, 0.1]))
        .with_pde(
            partial_differential_equations.HomogeneousNoAdvectionScalarDiffusionPDE(
                2, np.float64(-2)
            )
        )
        .with_boundary_condition(boundary.ConstantDirichletBoundaryCondition(0))
        .with_initial_condition(IC)
        .with_solver(FiniteDifferences())
        .with_time_step(np.float64(0.001))
        .with_target_time(np.float64(1))
    )

    assert 1.5 < compute_spacial_order(solver) < 2.5


# not a test yet as TTD is broken
def spacial_order_with_ttd_returns_something() -> None:
    """Test spacial order returns something that remotly makes sense."""

    def fce(x, y):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]  # noqa: ANN001, ANN202
        return np.e ** (-(x**2 + y**2))  # pyright: ignore[reportUnknownVariableType]

    X_grid, Y_grid = np.meshgrid(np.linspace(-3, 3, 121), np.linspace(-3, 3, 121))
    IC = TTD(fce(X_grid, Y_grid))

    solver = (
        EMPTY_SOLUTION_BUILDER.with_spatial_step(np.array([0.1, 0.1]))
        .with_pde(
            partial_differential_equations.HomogeneousNoAdvectionScalarDiffusionPDE(
                2, np.float64(-2)
            )
        )
        .with_boundary_condition(boundary.ConstantDirichletBoundaryCondition(0))
        .with_initial_condition(IC)
        .with_solver(FiniteDifferences())
        .with_time_step(np.float64(0.001))
        .with_target_time(np.float64(1))
    )

    assert 1.5 < compute_spacial_order(solver) < 2.5
