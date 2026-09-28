import numpy as np
from wren_pde_solver.abc.boundary import BoundaryCondition
from wren_pde_solver.abc.pde import PDE
from wren_pde_solver.boundary_conditions import ConstantDirichletBoundaryCondition
from wren_pde_solver.pde import HomogeneousNoAdvectionScalarDiffusionPDE
from wren_pde_solver.pde_types import DType, NDArray, Vector
from wren_pde_solver.solution_builder import EMPTY_SOLUTION_BUILDER
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
        EMPTY_SOLUTION_BUILDER.with_boundary_condition(BC)
        .with_initial_condition(IC)
        .with_pde(PD)
        .with_solver(FiniteDifferences())
        .with_spatial_step(dS)
        .with_time_step(dT)
        .with_target_time(TT)
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
    assert EMPTY_SOLUTION_BUILDER.with_pde(PD).pde is PD
    assert EMPTY_SOLUTION_BUILDER.with_solver(solver).solver is solver
    assert EMPTY_SOLUTION_BUILDER.with_boundary_condition(BC).boundary_condition is BC
    assert EMPTY_SOLUTION_BUILDER.with_initial_condition(IC).initial_condition is IC
    assert EMPTY_SOLUTION_BUILDER.with_spatial_step(dS).spatial_step is dS
    assert EMPTY_SOLUTION_BUILDER.with_time_step(dT).time_step is dT
    assert EMPTY_SOLUTION_BUILDER.with_target_time(TT).target_time is TT


def test_builder_setters_create_copies() -> None:
    """
    Test builder.

    Test if builder withs fields it returns a new copy which didn't
    modify the original.
    """
    A = EMPTY_SOLUTION_BUILDER.with_target_time(DType(10))
    B = A.with_target_time(DType(20))
    assert A.target_time == 10
    assert B.target_time == 20


def type_test_dont_allow_build_unless_everything_is_set() -> None:
    """
    Test builder cannot be build.

    Note this is a pyright test, not a pytest one so it shouldn't start with 'test'.
    """
    BC: BoundaryCondition = ConstantDirichletBoundaryCondition(0)
    IC: NDArray = np.random.default_rng().random(size=(20, 20, 18))
    PD: PDE = HomogeneousNoAdvectionScalarDiffusionPDE(
        3, None, None, scalar_diffusion=DType(10)
    )
    dS: Vector = np.array([1, 1, 2])
    dT: DType = DType(0.1)
    TT: DType = DType(1)

    _ = (  # type: ignore[arg-type]
        EMPTY_SOLUTION_BUILDER.with_boundary_condition(BC)
        .with_initial_condition(IC)
        .with_pde(PD)
        .with_solver(FiniteDifferences())
        .with_spatial_step(dS)
        .with_time_step(dT)
        .with_target_time(TT)
        .compute(),
        FiniteDifferences()(PD, IC, dS, BC, dT, TT),
    )
