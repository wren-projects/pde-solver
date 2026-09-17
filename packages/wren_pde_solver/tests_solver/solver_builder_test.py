import numpy as np
from wren_pde_solver.abc.boundary import BoundaryCondition
from wren_pde_solver.abc.pde import PDE
from wren_pde_solver.boundary_conditions import ConstantDirichletBoundaryCondition
from wren_pde_solver.pde import HomogeneousNoAdvectionScalarDiffusionPDE
from wren_pde_solver.pde_types import DType, NDArray, Vector
from wren_pde_solver.solver_builder import SolutionBuilder
from wren_pde_solver.solvers.finite_differences import FiniteDifferences


def test_builder_runs_and_returns_the_same_running_a_solver_directly_would() -> None:
    """Tests solver builder."""
    BC: BoundaryCondition = ConstantDirichletBoundaryCondition(0)
    IC: NDArray = np.random.default_rng().random(size=(20, 20, 18))
    PD: PDE = HomogeneousNoAdvectionScalarDiffusionPDE(
        3, None, None, scalar_diffusion=DType(10)
    )
    dS: Vector = np.array([1, 1, 2])
    dT: DType = DType(0.1)
    TT: DType = DType(1)

    np.testing.assert_array_equal(
        SolutionBuilder[HomogeneousNoAdvectionScalarDiffusionPDE]
        .create()
        .set_boundary_condition(BC)
        .set_initial_condition(IC)
        .set_pde(PD)
        .set_solver(FiniteDifferences())
        .set_spatial_step(dS)
        .set_time_step(dT)
        .set_target_time(TT)
        .compute(),
        FiniteDifferences()(PD, IC, dS, BC, dT, TT),
    )


def test_builder_getters_work() -> None:
    """Tests solver builder getters."""
    BC: BoundaryCondition = ConstantDirichletBoundaryCondition(0)
    IC: NDArray = np.random.default_rng().random(size=(20, 20, 18))
    PD: PDE = HomogeneousNoAdvectionScalarDiffusionPDE(
        3, None, None, scalar_diffusion=DType(10)
    )
    dS: Vector = np.array([1, 1, 2])
    dT: DType = DType(0.1)
    TT: DType = DType(1)
    solver = FiniteDifferences()
    assert (
        SolutionBuilder[HomogeneousNoAdvectionScalarDiffusionPDE]
        .create()
        .set_pde(PD)
        .get_pde()
        is PD
    )
    assert (
        SolutionBuilder[HomogeneousNoAdvectionScalarDiffusionPDE]
        .create()
        .set_solver(solver)
        .get_solver()
        is solver
    )
    assert (
        SolutionBuilder[HomogeneousNoAdvectionScalarDiffusionPDE]
        .create()
        .set_boundary_condition(BC)
        .get_boundary_condition()
        is BC
    )
    assert (
        SolutionBuilder[HomogeneousNoAdvectionScalarDiffusionPDE]
        .create()
        .set_initial_condition(IC)
        .get_initial_condition()
        is IC
    )
    assert (
        SolutionBuilder[HomogeneousNoAdvectionScalarDiffusionPDE]
        .create()
        .set_spatial_step(dS)
        .get_spatial_step()
        is dS
    )
    assert (
        SolutionBuilder[HomogeneousNoAdvectionScalarDiffusionPDE]
        .create()
        .set_time_step(dT)
        .get_time_step()
        is dT
    )
    assert (
        SolutionBuilder[HomogeneousNoAdvectionScalarDiffusionPDE]
        .create()
        .set_target_time(TT)
        .get_target_time()
        is TT
    )


def test_builder_setters_create_copies() -> None:
    """
    Test builder.

    Test if builder sets fields it returns a new copy which didn't
    modify the original.
    """
    A = (
        SolutionBuilder[HomogeneousNoAdvectionScalarDiffusionPDE]
        .create()
        .set_target_time(DType(10))
    )
    B = A.set_target_time(DType(20))
    assert A.get_target_time() == 10
    assert B.get_target_time() == 20
