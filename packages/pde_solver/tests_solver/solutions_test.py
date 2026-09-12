from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from pde_common.types import NDArray, Vector
from pde_solver.abc.boundary import BoundaryCondition
from pde_solver.abc.pde import PDE
from pde_solver.boundary_conditions import ConstantDirichletBoundaryCondition
from pde_solver.pde import HomogeneousNoAdvectionScalarDiffusionPDE
from pde_solver.pde_types import DType, Scalar
from pde_solver.solvers.finite_differences import FiniteDifferences

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


def make_heat_3d_mode_case(
    mode: tuple[int, int, int],
    shape: tuple[int, int, int],
    delta_time: float,
    steps: int,
    diffusion: DType = DEFAULT_K,
) -> PDETestCase:
    """
    Create a benchmark for a 3D heat equation eigenmode.

    PDE:
        u_t = k Δu

    Domain:
        (x, y, z) ∈ (0, 1)^3

    Boundary conditions:
        u = 0 on all six faces.

    Initial condition:
        u(x,y,z,0)
            = sin(lπx) sin(mπy) sin(nπz)

    Analytical solution:
        u(x,y,z,t)
            = sin(lπx) sin(mπy) sin(nπz)
              exp(-π²k(l²+m²+n²)t)
    """
    x, y, z = _make_unit_cube_grid(shape)

    l, m, n = mode
    decay_rate = np.pi**2 * diffusion * (
        l**2 + m**2 + n**2
    )

    spatial_mode = (
        np.sin(l * np.pi * x)
        * np.sin(m * np.pi * y)
        * np.sin(n * np.pi * z)
    )

    def exact_solution(time: float) -> NDArray:
        return spatial_mode * np.exp(-decay_rate * time)

    pde = HomogeneousNoAdvectionScalarDiffusionPDE(
        dims=3,
        homogeneous=None,
        no_advection=None,
        scalar_diffusion=-diffusion,
    )

    return PDETestCase(
        name=(
            f"3D heat equation mode {mode}, "
            f"shape={shape}, dt={delta_time}"
        ),
        pde=pde,
        boundary_condition=ConstantDirichletBoundaryCondition(
            value=0.0
        ),
        initial_condition=exact_solution(0.0),
        expected_solution=exact_solution,
        delta_time=DType(delta_time),
        steps=steps,
    )

def make_poisson_3d_mode_case(
    mode: tuple[int, int, int],
    shape: tuple[int, int, int],
):
    x, y, z = _make_unit_cube_grid(shape)

    l, m, n = mode

    exact = (
        np.sin(l*np.pi*x)
        * np.sin(m*np.pi*y)
        * np.sin(n*np.pi*z)
    )

    rhs = (
        np.pi**2
        * (l*l + m*m + n*n)
        * exact
    )

    PDE = 
    return PDETestCase(
        name=f"3D Poisson mode {mode}",
        pde=PoissonPDE(
            dims=3,
        ),
        boundary_condition=ConstantDirichletBoundaryCondition(
            value=0.0
        ),
        source_term=rhs,
        expected_solution=lambda _: exact,
    )

PDE_TEST_CASES = [
    # Baseline: fundamental eigenmode.
    make_heat_3d_mode_case(
        mode=(1, 1, 1),
        shape=(32, 32, 32),
        delta_time=5e-6,
        steps=100,
        diffusion=DEFAULT_K,
    ),

    # Higher spatial frequency mode.
    # Verifies Laplacian eigenvalue scaling.
    make_heat_3d_mode_case(
        mode=(2, 3, 1),
        shape=(32, 32, 32),
        delta_time=1e-6,
        steps=100,
        diffusion=DEFAULT_K,
    ),

    # Different diffusion coefficient.
    # Verifies decay rate depends on k.
    make_heat_3d_mode_case(
        mode=(1, 1, 1),
        shape=(32, 32, 32),
        delta_time=1e-6,
        steps=100,
        diffusion=0.1,
    ),

    # Non-uniform grid.
    # Verifies handling of different spatial resolutions.
    make_heat_3d_mode_case(
        mode=(1, 2, 1),
        shape=(20, 30, 50),
        delta_time=1e-6,
        steps=100,
        diffusion=DEFAULT_K,
    ),
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
