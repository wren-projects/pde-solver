import numpy as np
from wren_pde_solver.boundary_conditions import ConstantDirichletBoundaryCondition
from wren_pde_solver.pde import HomogeneousNoAdvectionScalarDiffusionPDE
from wren_pde_solver.solution_builder import EMPTY_SOLUTION_BUILDER
from wren_pde_solver.solvers.finite_differences import FiniteDifferences

from pde.compare import compute_spacial_order


def test_spacial_order_returns_something() -> None:
    """Test spacial order returns something that remotly makes sense."""

    def fce(x, y):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]  # noqa: ANN001, ANN202
        return np.e ** (-(x**2 + y**2))  # pyright: ignore[reportUnknownVariableType]

    X_grid, Y_grid = np.meshgrid(np.linspace(-3, 3, 121), np.linspace(-3, 3, 121))
    IC = fce(X_grid, Y_grid)
    solver = (
        EMPTY_SOLUTION_BUILDER.with_spatial_step(np.array([0.1, 0.1]))
        .with_pde(HomogeneousNoAdvectionScalarDiffusionPDE(2, np.float64(-2)))
        .with_boundary_condition(ConstantDirichletBoundaryCondition(0))
        .with_initial_condition(IC)
        .with_solver(FiniteDifferences())
        .with_time_step(np.float64(0.001))
        .with_target_time(np.float64(1))
    )
    print(compute_spacial_order(solver))

    assert 1.5 < compute_spacial_order(solver) < 2.5


def test_spacial_order_with_TTD_returns_something() -> None:
    """Test spacial order returns something that remotly makes sense."""

    def fce(x, y):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]  # noqa: ANN001, ANN202
        return np.e ** (-(x**2 + y**2))  # pyright: ignore[reportUnknownVariableType]

    X_grid, Y_grid = np.meshgrid(np.linspace(-3, 3, 121), np.linspace(-3, 3, 121))
    IC = fce(X_grid, Y_grid)
    solver = (
        EMPTY_SOLUTION_BUILDER.with_spatial_step(np.array([0.1, 0.1]))
        .with_pde(HomogeneousNoAdvectionScalarDiffusionPDE(2, np.float64(-2)))
        .with_boundary_condition(ConstantDirichletBoundaryCondition(0))
        .with_initial_condition(IC)
        .with_solver(FiniteDifferences())
        .with_time_step(np.float64(0.001))
        .with_target_time(np.float64(1))
    )
    print(compute_spacial_order(solver))

    assert 1.5 < compute_spacial_order(solver) < 2.5


test_spacial_order_returns_something()
