from copy import deepcopy

import numpy as np
import pytest

from pde_solver.boundary_conditions import ConstantDirichletBoundaryCondition
from pde_solver.pde_types import NDArray
from tests.common import TEST_SCALARS, TEST_TENSORS, tensor_boundary, tensor_interior


@pytest.mark.parametrize("value", TEST_SCALARS)
@pytest.mark.parametrize("tensor", deepcopy(TEST_TENSORS))
def test_dirichlet_boundary_setup_initial_condition(
    value: float, tensor: NDArray
) -> None:
    """
    Test Dirichlet boundary condition for setup initial.

    Should increase tensor size by 2 in each axis and adds the given value at boundary.
    """
    boundary = ConstantDirichletBoundaryCondition(value=value)
    new_tensor = boundary.apply_to_initial_condition(tensor)
    np.testing.assert_array_equal(new_tensor.shape, np.array(tensor.shape) + 2)
    np.testing.assert_equal(tensor_boundary(new_tensor), value)
    np.testing.assert_equal(
        tensor_interior(new_tensor), tensor.reshape(-1)
    )  # interior is returned as a 1D array


@pytest.mark.parametrize("value", TEST_SCALARS)
@pytest.mark.parametrize("time", TEST_SCALARS)
@pytest.mark.parametrize("delta_time", TEST_SCALARS)
@pytest.mark.parametrize("tensor", deepcopy(TEST_TENSORS))
def test_dirichlet_boundary_call(
    value: float, time: float, delta_time: float, tensor: NDArray
) -> None:
    """
    Test Dirichlet boundary condition for a time-step call.

    Should keep tensor size and set zero (not the given value) at boundary.
    """
    boundary = ConstantDirichletBoundaryCondition(value=value)
    new_tensor = boundary(tensor, time=time, delta_time=delta_time)
    np.testing.assert_array_equal(new_tensor.shape, tensor.shape)
    np.testing.assert_equal(tensor_boundary(new_tensor), 0)
    np.testing.assert_equal(
        tensor_interior(new_tensor), tensor_interior(tensor)
    )  # interior is returned as a 1D array
