from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from pde_common.types import NDArray
from pde_solver.abc.pde import PDE
from pde_solver.boundary_conditions import ConstantDirichletBoundaryCondition
from pde_solver.pde import HomogeneousVectorAdvectionMatrixDiffusionPDE
from pde_solver.solvers.finite_differences import FiniteDifferences

DEFAULT_K = 1.0


@dataclass(frozen=True)
class PDETestCase:
    """Known analytical solution used to test PDE solvers."""

    name: str

    pde: PDE
    boundary_condition: ConstantDirichletBoundaryCondition

    initial_condition: NDArray
    expected_solution: Callable[[float], NDArray]

    start_time: float
    delta_time: float
    steps: int

    atol: float = 1e-8
    rtol: float = 1e-8


def _make_unit_cube_grid(
    shape: tuple[int, int, int],
) -> tuple[NDArray, NDArray, NDArray]:
    """Create an interior grid on the unit cube."""
    nx, ny, nz = shape

    x = np.linspace(0, 1, nx + 2)[1:-1]
    y = np.linspace(0, 1, ny + 2)[1:-1]
    z = np.linspace(0, 1, nz + 2)[1:-1]

    return np.meshgrid(x, y, z, indexing="ij")


def _make_spacial_step(shape: tuple[int, ...]) -> NDArray:
    """Compute grid spacing for an interior grid on the unit cube."""
    return np.array([1.0 / (n + 1) for n in shape])


def make_heat_3d_mode_111_case(
    shape: tuple[int, int, int] = (32, 32, 32),
    k: float = DEFAULT_K,
    delta_time: float = 1e-5,
    steps: int = 100,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> PDETestCase:
    """
    Create a 3D heat equation benchmark using the first Laplace eigenmode.

    The analytical solution is

        u(x,y,z,t) = sin(pi x) sin(pi y) sin(pi z) exp(-3 pi² k t).

    Since the solver represents diffusion as

        u_t + div(B grad u) = 0,

    the heat equation u_t = k Δu is represented by B = -kI.
    """
    x, y, z = _make_unit_cube_grid(shape)
    decay_rate = 3 * np.pi**2 * k

    def exact_solution(time: float) -> NDArray:
        return (
            np.sin(np.pi * x)
            * np.sin(np.pi * y)
            * np.sin(np.pi * z)
            * np.exp(-decay_rate * time)
        )

    pde = HomogeneousVectorAdvectionMatrixDiffusionPDE(
        dims=3,
        homogeneous=0.0,
        vector_advection=np.zeros(3),
        matrix_diffusion=-k * np.eye(3),
    )

    return PDETestCase(
        name=f"3D heat equation mode (1,1,1), shape={shape}, dt={delta_time}",
        pde=pde,
        boundary_condition=ConstantDirichletBoundaryCondition(value=0.0),
        initial_condition=exact_solution(0.0),
        expected_solution=exact_solution,
        start_time=0.0,
        delta_time=delta_time,
        steps=steps,
        atol=atol,
        rtol=rtol,
    )


PDE_TEST_CASES = [
    make_heat_3d_mode_111_case(
        shape=(16, 16, 16),
        delta_time=1e-5,
        steps=100,
        atol=2e-2,
        rtol=2e-2,
    ),
    make_heat_3d_mode_111_case(
        shape=(32, 32, 32),
        delta_time=5e-6,
        steps=200,
        atol=1e-2,
        rtol=1e-2,
    ),
    make_heat_3d_mode_111_case(
        shape=(48, 48, 48),
        delta_time=2e-6,
        steps=500,
        atol=5e-3,
        rtol=5e-3,
    ),
]


def _advance_case(case: PDETestCase) -> NDArray:
    """Run finite differences for one test case."""
    solver = FiniteDifferences()
    state = case.boundary_condition.apply_to_initial_condition(case.initial_condition)
    spacial_step = _make_spacial_step(case.initial_condition.shape)

    for _ in range(case.steps):
        state += solver._compute_time_step(
            pde=case.pde,
            state=state,
            spacial_step=spacial_step,
            time_step=case.delta_time,
        )
        state = case.boundary_condition(
            state,
            time=case.start_time,
            delta_time=case.delta_time,
        )

    return case.boundary_condition.remove_boundary(state)


@pytest.mark.parametrize("case", PDE_TEST_CASES, ids=lambda case: case.name)
def test_finite_differences_heat_3d(case: PDETestCase) -> None:
    """Test finite differences against a known 3D heat equation solution."""
    actual = _advance_case(case)
    expected = case.expected_solution(case.start_time + case.steps * case.delta_time)

    np.testing.assert_allclose(
        actual,
        expected,
        atol=case.atol,
        rtol=case.rtol,
    )
