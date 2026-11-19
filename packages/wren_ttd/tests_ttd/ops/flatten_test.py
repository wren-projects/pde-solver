# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor, arange_tensor
from wren_ttd import ops
from wren_ttd.core import TTD

from ..common import TEST_TTD, TestTTD, assert_default_epsilon

VECTOR = arange_tensor(6)
MATRIX = arange_tensor(2, 3)
TENSOR_3D = arange_tensor(2, 3, 4)


@pytest.mark.parametrize("tensor", [VECTOR, MATRIX, TENSOR_3D])
def test_flatten(tensor: TestTensor) -> None:
    """Test flatten against the dense ndarray method."""
    ttd = TTD(tensor)
    expected = tensor.flatten()

    assert_default_epsilon(ops.flatten(ttd), expected)
    assert_default_epsilon(ops.flatten(ttd, order="C"), expected)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_flatten_test_tensors(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test flatten on the shared test tensors."""
    assert_default_epsilon(ops.flatten(ttd), tensor.flatten())


@pytest.mark.parametrize("order", ["F", "A", "K"])
def test_flatten_rejects_order(order: str) -> None:
    """Test that non-C orders are rejected."""
    ttd = TTD(MATRIX)
    with pytest.raises(NotImplementedError):
        _ = ops.flatten(ttd, order=order)


def test_flatten_matches_ravel() -> None:
    """Test that flatten has the same semantics as ravel."""
    ttd = TTD(TENSOR_3D)
    assert_default_epsilon(ops.flatten(ttd), ops.ravel(ttd))


def test_flatten_not_a_numpy_function() -> None:
    """Test that there is no top-level NumPy function to register."""
    assert not hasattr(np, "flatten")


def test_flatten_preserves_dtype() -> None:
    """Test that the result keeps the input dtype."""
    ttd = TTD(TENSOR_3D.astype(np.float32))
    assert ops.flatten(ttd).dtype == np.float32
