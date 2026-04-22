from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from pde_solver.boundary_conditions import ConstantDirichletBoundaryCondition
from tests.data import TEST_TENSORS


@pytest.mark.parametrize("value", [-2, 0, 124567, 2.4567654345])
@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_dirichlet_boundary_setup_initial_condition(
    value: float, tensor: NDArray[Any]
) -> None:
    """
    Test Dirichlet boundary condition for setup initial.

    Should increase tensor size by 2 in each axis and adds the given value at boundary.
    """
    boundary = ConstantDirichletBoundaryCondition(value=value)
    new_tensor = boundary.apply_to_initial_condition(tensor)
    np.testing.assert_array_equal(new_tensor.shape, np.array(tensor.shape) + 2)

    mask = np.ones_like(new_tensor, dtype=bool)
    interior_slices = tuple(slice(1, -1) for _ in range(new_tensor.ndim))
    mask[interior_slices] = False
    expected = np.full_like(new_tensor[mask], value)
    np.testing.assert_equal(new_tensor[mask], expected)


@pytest.mark.parametrize("value", [-2, 0, 124567, 2.4567654345])
@pytest.mark.parametrize("time", [-2, 0, 124567, 2.4567654345])
@pytest.mark.parametrize("delta_time", [-2, 0, 124567, 2.4567654345])
@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_dirichlet_boundary_call(
    value: float, time: float, delta_time: float, tensor: NDArray[Any]
) -> None:
    """
    Test Dirichlet boundary condition for a time-step call.

    Should keep tensor size and set zero (not the given value) at boundary.
    """
    boundary = ConstantDirichletBoundaryCondition(value=value)
    new_tensor = boundary(tensor, time=time, delta_time=delta_time)
    np.testing.assert_array_equal(new_tensor.shape, tensor.shape)

    mask = np.ones_like(new_tensor, dtype=bool)
    interior_slices = tuple(slice(1, -1) for _ in range(new_tensor.ndim))
    mask[interior_slices] = False
    expected = np.full_like(new_tensor[mask], 0)
    np.testing.assert_equal(new_tensor[mask], expected)
