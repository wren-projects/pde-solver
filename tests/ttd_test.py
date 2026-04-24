from itertools import product
from typing import Any
import math

import numpy as np
import pytest
from numpy.typing import NDArray

from numpy_ttd import DEFAULT_EPSILON, TTD
from numpy_ttd.laplace import laplace_1d, tt_laplace
from numpy_ttd.ops import matvec

rng = np.random.default_rng(0)


type TestTensor = NDArray[np.float64]
type TestTensorPair = tuple[TestTensor, TestTensor]

TEST_TENSORS: list[TestTensor] = [
    tensor.astype(np.float64)
    for tensor in [
        # ---- 2D ----
        np.arange(12).reshape(3, 4),
        rng.random((5, 2)),
        np.array([x % 2 == 0 for x in range(16)]).reshape(4, 4),
        # ---- 3D ----
        np.arange(48).reshape(3, 4, 4),
        rng.normal(size=(2, 3, 5)),
        # ---- 4D ----
        np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5),
        np.ones((2, 3, 2, 4)),
        # ---- 5D ----
        np.arange(3 * 2 * 3 * 4 * 2).reshape(3, 2, 3, 4, 2),
        # ---- 6D ----
        np.arange(2 * 3 * 2 * 3 * 2 * 4).reshape(2, 3, 2, 3, 2, 4),
        # ---- 7D ----
        np.arange(2 * 2 * 3 * 2 * 4 * 3 * 2).reshape(2, 2, 3, 2, 4, 3, 2),
        # ---- 8D ----
        np.arange(2 * 3 * 2 * 4 * 2 * 3 * 2 * 3).reshape(2, 3, 2, 4, 2, 3, 2, 3),
        # ---- 9D ----
        np.arange(2 * 2 * 3 * 2 * 3 * 2 * 3 * 2 * 3).reshape(2, 2, 3, 2, 3, 2, 3, 2, 3),
        # ---- 10D ----
        np.arange(2 * 2 * 2 * 3 * 2 * 2 * 3 * 2 * 2 * 3).reshape(
            2, 2, 2, 3, 2, 2, 3, 2, 2, 3
        ),
    ]
]


TEST_PAIR_TENSORS: list[TestTensorPair] = [
    (a.astype(np.float64), b.astype(np.float64))
    for a, b in [
        # ---- 1D ----
        (np.arange(8), np.ones(8)),
        # ---- 2D ----
        (
            np.arange(12).reshape(3, 4),
            rng.integers(0, 10, size=(3, 4)),
        ),
        # ---- 3D ----
        (np.arange(24).reshape(2, 3, 4), np.ones((2, 3, 4))),
        # ---- 4D ----
        (
            np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5),
            rng.normal(size=(2, 3, 4, 5)),
        ),
        # ---- 5D ----
        (
            np.zeros((2, 2, 3, 4, 3)),
            np.arange(2 * 2 * 3 * 4 * 3).reshape(2, 2, 3, 4, 3),
        ),
        # ---- 6D ----
        (
            np.arange(2 * 3 * 2 * 3 * 2 * 4).reshape(2, 3, 2, 3, 2, 4),
            rng.integers(0, 5, size=(2, 3, 2, 3, 2, 4)),
        ),
        # ---- 7D ----
        (
            np.arange(2 * 2 * 3 * 2 * 4 * 3 * 2).reshape(2, 2, 3, 2, 4, 3, 2),
            np.ones((2, 2, 3, 2, 4, 3, 2)),
        ),
        # ---- 8D ----
        (
            np.arange(2 * 3 * 2 * 4 * 2 * 3 * 2 * 3).reshape(2, 3, 2, 4, 2, 3, 2, 3),
            rng.random((2, 3, 2, 4, 2, 3, 2, 3)),
        ),
        # ---- 9D ----
        (
            np.arange(2 * 2 * 3 * 2 * 3 * 2 * 3 * 2 * 3).reshape(
                2, 2, 3, 2, 3, 2, 3, 2, 3
            ),
            np.ones((2, 2, 3, 2, 3, 2, 3, 2, 3)),
        ),
        # ---- 10D ----
        (
            np.arange(2 * 2 * 2 * 3 * 2 * 2 * 3 * 2 * 2 * 3).reshape(
                2, 2, 2, 3, 2, 2, 3, 2, 2, 3
            ),
            rng.integers(0, 10, size=(2, 2, 2, 3, 2, 2, 3, 2, 2, 3)),
        ),
    ]
]

type EpsilonComparable = NDArray[np.float64] | TTD[np.float64] | np.floating | float


def equals_default_epsilon(a: EpsilonComparable, b: EpsilonComparable) -> bool:
    """Compare two tensors for equality within the default epsilon."""
    return np.allclose(np.asarray(a), np.asarray(b), atol=DEFAULT_EPSILON)


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_roundtrip_compression(tensor: TestTensor) -> None:
    """Test the round-trip compression/decompression of a tensor."""
    ttd = TTD.from_ndarray(tensor)
    assert equals_default_epsilon(ttd, tensor)


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_inner_product(tensor: TestTensor) -> None:
    """Test inner norm."""
    ttd = TTD.from_ndarray(tensor)
    assert equals_default_epsilon(np.vdot(ttd, ttd), np.vdot(tensor, tensor))


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_frobenius_norm(tensor: TestTensor) -> None:
    """Test Frobenius norm."""
    ttd = TTD.from_ndarray(tensor)
    assert equals_default_epsilon(np.linalg.norm(ttd), np.linalg.norm(tensor))


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_rounding(tensor: TestTensor) -> None:
    """Test get value."""
    tensortrain = TTD.from_ndarray(tensor.astype(np.float64))
    assert equals_default_epsilon(tensortrain.rounded(), tensor)


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_get_element(tensor: TestTensor) -> None:
    """Test get value."""
    tensortrain = TTD.from_ndarray(tensor.astype(np.float64))
    for index in np.ndindex(tensor.shape):
        assert equals_default_epsilon(tensortrain[index], tensor[index])


@pytest.mark.parametrize(("a", "b"), TEST_PAIR_TENSORS)
def test_add(a: TestTensor, b: TestTensor) -> None:
    """Test that TTD addition works."""
    # TODO: abstract this for other operator tests
    assert a.shape == b.shape
    tensor_sum = a + b

    ttd_a = TTD.from_ndarray(a)
    ttd_b = TTD.from_ndarray(b)

    ttd_sum = ttd_a + ttd_b
    assert equals_default_epsilon(ttd_sum, tensor_sum)

    ttd_sum.round()
    assert equals_default_epsilon(ttd_sum, tensor_sum)

    assert equals_default_epsilon(ttd_a + ttd_b, tensor_sum)
    assert equals_default_epsilon(np.add(ttd_a, ttd_b), tensor_sum)
    assert equals_default_epsilon(np.add(ttd_b, ttd_a), tensor_sum)

    ttd_a += ttd_b
    assert equals_default_epsilon(ttd_a, tensor_sum)


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_scalar_multiplication(tensor: TestTensor) -> None:
    """Test that TTD scalar multiplication works."""
    ttd = TTD.from_ndarray(tensor)

    for scalar in [1, 0.5, -1, 0, math.pi, -math.e]:
        scaled_tensor = tensor * scalar
        assert equals_default_epsilon(ttd * scalar, scaled_tensor)
        assert equals_default_epsilon(scalar * ttd, scaled_tensor)
        assert equals_default_epsilon(np.multiply(ttd, scalar), scaled_tensor)
        assert equals_default_epsilon(np.multiply(scalar, ttd), scaled_tensor)

        ttd_copy = ttd[...]
        ttd_copy *= scalar
        assert equals_default_epsilon(ttd_copy, scaled_tensor)


@pytest.mark.parametrize("tensor", TEST_TENSORS)
def test_negation(tensor: TestTensor) -> None:
    """Test that TTD negation works."""
    ttd = TTD.from_ndarray(tensor)

    assert equals_default_epsilon(-ttd, -tensor)
    assert equals_default_epsilon(np.negative(ttd), np.negative(tensor))


def test_laplace() -> None:
    """Test."""
    laplacian = tt_laplace((3, 3), dtype=np.dtype(np.float64))
    print(np.asarray(laplacian))

    A = np.arange(3 * 3, dtype=np.float64).reshape(3, 3)

    laplaced = matvec(laplacian, TTD.from_ndarray(A))
    print(np.asarray(laplaced))
    raise AssertionError
