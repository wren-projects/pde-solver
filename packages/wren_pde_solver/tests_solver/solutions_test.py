from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from wren_common.types import NDArray, Vector
from wren_pde_solver.abc.boundary import BoundaryCondition
from wren_pde_solver.abc.pde import PDE
from wren_pde_solver.boundary_conditions import ConstantDirichletBoundaryCondition
from wren_pde_solver.pde import HomogeneousNoAdvectionScalarDiffusionPDE
from wren_pde_solver.pde_types import DType, Scalar
from wren_pde_solver.solvers.finite_differences import FiniteDifferences

DEFAULT_K = DType(1.0)
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3


@dataclass(frozen=True)
class PDETestCase:
    """Known analytical solution used to test PDE solvers."""

    name: str
    pde: PDE
    boundary_condition: BoundaryCondition
    initial_condition: NDArray
    expected_solution: Callable[[float], NDArray]
    delta_time: Scalar
    steps: int
    atol: float = DEFAULT_ATOL
    rtol: float = DEFAULT_RTOL


def _make_unit_cube_grid(
    shape: tuple[int, int, int],
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Create interior grid points on the unit cube.

    The boundary points 0 and 1 are excluded from the tensor because
    the solver represents boundary values through the boundary condition.
    """
    nx, ny, nz = shape

    x = np.linspace(0, 1, nx + 2)[1:-1]
    y = np.linspace(0, 1, ny + 2)[1:-1]
    z = np.linspace(0, 1, nz + 2)[1:-1]

    return np.meshgrid(x, y, z, indexing="ij")


def _make_spacial_step(shape: tuple[int, ...]) -> Vector:
    """Compute grid spacing for an interior grid on the unit cube."""
    return 1.0 / (np.array(shape) + 1)


def make_heat_3d_mode_111_case(
    shape: tuple[int, int, int],
    delta_time: float,
    steps: int,
) -> PDETestCase:
    """
    Create a benchmark for the three-dimensional heat equation.

    PDE:
        u_t = k Δu

    Domain:
        (x, y, z) ∈ (0, 1)^3

    Boundary conditions:
        u = 0 on all six faces of the unit cube.

    Initial condition:
        u(x,y,z,0) = sin(πx) sin(πy) sin(πz)

    Analytical solution:
        u(x,y,z,t)
            = sin(πx) sin(πy) sin(πz) exp(-3π²kt)

    The solver uses the convention

        u_t + div(B ∇u) = 0,

    therefore the heat equation is represented using

        B = -kI.
    """
    x, y, z = _make_unit_cube_grid(shape)
    decay_rate = 3 * np.pi**2 * DEFAULT_K

    def exact_solution(time: float) -> NDArray:
        """
        Analytical solution of the benchmark problem.

        u(x,y,z,t)
            = sin(πx) sin(πy) sin(πz) exp(-3π²kt)
        """
        return (
            np.sin(np.pi * x)
            * np.sin(np.pi * y)
            * np.sin(np.pi * z)
            * np.exp(-decay_rate * time)
        )

    pde = HomogeneousNoAdvectionScalarDiffusionPDE(
        dims=3,
        homogeneous=None,
        no_advection=None,
        scalar_diffusion=-DEFAULT_K,
    )

    return PDETestCase(
        name=f"3D heat equation mode (1,1,1), shape={shape}, dt={delta_time}",
        pde=pde,
        boundary_condition=ConstantDirichletBoundaryCondition(value=0.0),
        initial_condition=exact_solution(0.0),
        expected_solution=exact_solution,
        delta_time=DType(delta_time),
        steps=steps,
    )


PDE_TEST_CASES = [
    # Coarse grid: fast smoke test.
    make_heat_3d_mode_111_case((16, 16, 16), delta_time=1e-5, steps=100),
    # Medium grid: checks refinement against the same analytical solution.
    make_heat_3d_mode_111_case((32, 32, 32), delta_time=5e-6, steps=100),
    # Non-uniform grid: verifies that the solver handles different
    # resolutions in each spatial dimension correctly.
    make_heat_3d_mode_111_case((20, 30, 50), delta_time=2e-6, steps=100),
]


def _advance_case(case: PDETestCase) -> NDArray:
    """Run finite differences through the public solver interface."""
    solver = FiniteDifferences()

    return solver(
        pde=case.pde,
        initial_condition=case.initial_condition,
        spacial_step=_make_spacial_step(case.initial_condition.shape),
        boundary_condition=case.boundary_condition,
        time_step=case.delta_time,
        target_time=case.steps * case.delta_time,
    )


@pytest.mark.parametrize("case", PDE_TEST_CASES, ids=lambda case: case.name)
def test_finite_differences_heat_3d(case: PDETestCase) -> None:
    """Test finite differences against a known 3D heat equation solution."""
    actual = _advance_case(case)
    expected = case.expected_solution(case.steps * case.delta_time)

    np.testing.assert_allclose(actual, expected, atol=case.atol, rtol=case.rtol)
