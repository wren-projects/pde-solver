# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

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
def test_ravel(tensor: TestTensor) -> None:
    """Test ravel through both the ops and NumPy entry points."""
    ttd = TTD(tensor)
    expected = np.ravel(tensor)

    assert_default_epsilon(ops.ravel(ttd), expected)
    assert_default_epsilon(np.ravel(ttd), expected)
    assert_default_epsilon(ops.ravel(ttd, order="C"), expected)
    assert_default_epsilon(np.ravel(ttd, order="C"), expected)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_ravel_test_tensors(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test ravel on the shared test tensors."""
    assert_default_epsilon(np.ravel(ttd), np.ravel(tensor))


@pytest.mark.parametrize("order", ["F", "A", "K"])
def test_ravel_rejects_order(order: str) -> None:
    """Test that non-C orders are rejected."""
    ttd = TTD(MATRIX)
    with pytest.raises(NotImplementedError):
        _ = ops.ravel(ttd, order=order)
    with pytest.raises(NotImplementedError):
        _ = np.ravel(ttd, order=cast(Any, order))


@pytest.mark.parametrize("tensor", [VECTOR, MATRIX, TENSOR_3D])
def test_ravel_matches_reshape(tensor: TestTensor) -> None:
    """Test that ravel is equivalent to reshaping to one dimension."""
    ttd = TTD(tensor)
    assert_default_epsilon(ops.ravel(ttd), ops.reshape(ttd, (ttd.size,)))
    assert_default_epsilon(ops.ravel(ttd), ops.flatten(ttd))


def test_ravel_preserves_dtype() -> None:
    """Test that the result keeps the input dtype."""
    ttd = TTD(TENSOR_3D.astype(np.float32))
    assert ops.ravel(ttd).dtype == np.float32
    assert np.ravel(ttd).dtype == np.float32
