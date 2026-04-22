import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from numpy_ttd import DEFAULT_EPSILON, TTD
from tests.data import TEST_TENSORS


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_roundtrip_compression(tensor: NDArray[Any]) -> None:
    """Test the round-trip compression/decompression of a tensor."""
    ttd = TTD.from_ndarray(tensor.astype(np.float64), epsilon=DEFAULT_EPSILON)
    assert np.allclose(np.asarray(ttd), tensor), (
        "TTD round-trip does not equal original tensor"
    )


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_inner_product(tensor: NDArray[Any]) -> None:
    """Test inner norm."""
    tensor_float: NDArray[np.float64] = tensor.astype(np.float64)
    ttd = TTD.from_ndarray(tensor_float, epsilon=DEFAULT_EPSILON)
    assert np.allclose(np.vdot(ttd, ttd), np.vdot(tensor_float, tensor_float))


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_frobenius_norm(tensor: NDArray[Any]) -> None:
    """Test Frobenius norm."""
    tensor_float: NDArray[np.float64] = tensor.astype(np.float64)
    ttd = TTD.from_ndarray(tensor_float, epsilon=DEFAULT_EPSILON)
    assert np.allclose(np.linalg.norm(ttd), np.linalg.norm(tensor_float))


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_rounding(tensor: NDArray[Any]) -> None:
    """Test get value."""
    tensortrain = TTD.from_ndarray(tensor.astype(np.float64), epsilon=DEFAULT_EPSILON)
    assert np.allclose(np.asarray(tensortrain.rounded()), tensor)


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_get_element(tensor: NDArray[Any]) -> None:
    """Test get value."""
    tensortrain = TTD.from_ndarray(tensor.astype(np.float64), epsilon=DEFAULT_EPSILON)
    for index in np.ndindex(tensor.shape):
        assert np.allclose(tensortrain[index], tensor[index])


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_scalar_multiplication(tensor: NDArray[Any]) -> None:
    """Test that TTD scalar multiplication works."""
    A = tensor.astype(np.float64)

    ttd = TTD.from_ndarray(A)

    for scalar in [1, 0.5, -1, 0, math.pi, -math.e]:
        assert np.allclose(np.asarray(ttd * scalar), A * scalar)
        assert np.allclose(np.asarray(scalar * ttd), A * scalar)
        assert np.allclose(np.asarray(np.multiply(ttd, scalar)), A * scalar)
        assert np.allclose(np.asarray(np.multiply(scalar, ttd)), A * scalar)

        ttd_copy = ttd[...]
        ttd_copy *= scalar
        assert np.allclose(np.asarray(ttd_copy), A * scalar)


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_negation(tensor: NDArray[Any]) -> None:
    """Test that TTD negation works."""
    A = tensor.astype(np.float64)

    ttd = TTD.from_ndarray(A)

    assert np.allclose(np.asarray(-ttd), -A)
    assert np.allclose(np.asarray(np.negative(ttd)), -A)
