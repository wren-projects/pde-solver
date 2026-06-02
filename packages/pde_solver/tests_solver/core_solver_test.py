import numpy as np
from pde_solver.boundary_conditions import ConstantDirichletBoundaryCondition
from pde_solver.pde import HomogeneousNoAdvectionNoDiffusionPDE
from pde_solver.solvers.finite_differences import FiniteDifferences


def test_empty_state_3_dims() -> None:
    """Test that in 3 dimensions with no diffusion nor advection."""
    solver = FiniteDifferences()
    PDE = HomogeneousNoAdvectionNoDiffusionPDE(3, None, None, None)
    initial_condition = np.random.default_rng(seed=1).random((10, 10, 10))
    solved = solver(
        pde=PDE,
        initial_condition=initial_condition,
        spacial_step=np.ones(3) * 0.01,
        time_step=np.float64(0.01),
        target_time=np.float64(1),
        boundary_condition=ConstantDirichletBoundaryCondition(0),
    )
    np.testing.assert_allclose(solved, initial_condition)
