from itertools import product
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from numpy_ttd import DEFAULT_EPSILON, TTD
from numpy_ttd.laplace import laplace_1d, tt_laplace

TEST_TENSORS: tuple[NDArray, ...] = (
    # ---- 2D ----
    np.arange(12).reshape(3, 4),  # (3,4)
    np.random.default_rng(0).random((5, 2)),  # (5,2)
    np.array([x % 2 == 0 for x in range(16)]).reshape(4, 4),  # (4,4) bool
    # ---- 3D ----
    np.arange(48).reshape(3, 4, 4),  # (3,4,4)
    np.random.default_rng(1).normal(size=(2, 3, 5)),  # (2,3,5)
    # ---- 4D ----
    np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5),  # (2,3,4,5)
    np.ones((2, 3, 2, 4)),  # (2,3,2,4) float
    # ---- 5D ----
    np.arange(3 * 2 * 3 * 4 * 2).reshape(3, 2, 3, 4, 2),  # (3,2,3,4,2)
    # ---- 6D ----
    np.arange(2 * 3 * 2 * 3 * 2 * 4).reshape(2, 3, 2, 3, 2, 4),  # (2,3,2,3,2,4)
    # ---- 7D ----
    np.arange(2 * 2 * 3 * 2 * 4 * 3 * 2).reshape(
        2, 2, 3, 2, 4, 3, 2
    ),  # (2,2,3,2,4,3,2)
    # ---- 8D ----
    np.arange(2 * 3 * 2 * 4 * 2 * 3 * 2 * 3).reshape(
        2, 3, 2, 4, 2, 3, 2, 3
    ),  # (2,3,2,4,2,3,2,3)
    # ---- 9D ----
    np.arange(2 * 2 * 3 * 2 * 3 * 2 * 3 * 2 * 3).reshape(2, 2, 3, 2, 3, 2, 3, 2, 3),
    # ---- 10D ----
    np.arange(2 * 2 * 2 * 3 * 2 * 2 * 3 * 2 * 2 * 3).reshape(
        2, 2, 2, 3, 2, 2, 3, 2, 2, 3
    ),
)


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


def test_laplace() -> None:
    """Test."""
    laplacian = tt_laplace((4, 4), 1, np.dtype(np.float64))

    A = np.arange(4**2, dtype=np.float64).reshape(4, 4)

    laplaced = laplacian @ TTD.from_ndarray(A)
