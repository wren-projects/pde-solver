from copy import deepcopy

import numpy as np
import pytest

from numpy_ttd import TTD
from tests.common import (
    TEST_PAIR_TTD,
    TEST_SCALARS,
    TEST_TTD,
    TestTensor,
    TestTensorPair,
    TestTTD,
    TestTTDPair,
    assert_default_epsilon,
    TEST_SHAPES,
)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_roundtrip_compression(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test the round-trip compression/decompression of a tensor."""
    # the compression is done in TEST_TTD and the decompression is done
    # implicitly inside assert_default_epsilon
    assert_default_epsilon(ttd, tensor)


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
def test_inner_product(tensors: TestTensorPair, ttds: TestTTDPair) -> None:
    """Test inner product."""
    ttd_a, ttd_b = ttds
    tensor_a, tensor_b = tensors

    assert_default_epsilon(np.vdot(ttd_a, ttd_b), np.vdot(tensor_a, tensor_b))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_frobenius_norm(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test Frobenius norm."""
    assert_default_epsilon(np.linalg.norm(ttd), np.linalg.norm(tensor))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_rounding(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test rounding."""
    assert_default_epsilon(ttd.rounded(), tensor)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_sum(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test sum."""
    assert_default_epsilon(np.sum(ttd), np.sum(tensor))

    for axis in range(tensor.ndim):
        assert_default_epsilon(ttd.sum(axis=axis), tensor.sum(axis=axis))


@pytest.mark.parametrize("shape", deepcopy(TEST_SHAPES))
def test_zeros(shape: tuple[int, ...]) -> None:
    """Test zeros."""
    ttd = TTD.zeros(shape, dtype=np.dtype(np.float64))
    assert ttd.shape == shape
    assert ttd.sum() == 0


@pytest.mark.parametrize("shape", deepcopy(TEST_SHAPES))
def test_ones(shape: tuple[int, ...]) -> None:
    """Test ones."""
    ttd = TTD.ones(shape, dtype=np.dtype(np.float64))
    assert ttd.shape == shape
    assert_default_epsilon(ttd.sum(), ttd.size)


@pytest.mark.parametrize("shape", deepcopy(TEST_SHAPES))
def test_eye(shape: tuple[int, ...]) -> None:
    """Test eye."""
    ttd = TTD.eye(shape, dtype=np.dtype(np.float64))
    assert ttd.size == np.prod(np.pow(shape, 2))


@pytest.mark.parametrize("shape", deepcopy(TEST_SHAPES))
def test_random(shape: tuple[int, ...]) -> None:
    """Test randomly generated TTD."""
    ttd = TTD.random(shape, dtype=np.dtype(np.float64))
    assert ttd.shape == shape
    np.testing.assert_allclose(ttd.sum() / ttd.size, 0.5, atol=0.5, rtol=0.5)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_get_element(ttd: TestTTD, tensor: TestTensor) -> None:
    """Test get value."""
    if tensor.ndim > 8:
        return

    for index in np.ndindex(tensor.shape):
        assert_default_epsilon(ttd[index], tensor[index])


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
def test_add(tensors: TestTensorPair, ttds: TestTTDPair) -> None:
    """Test that TTD addition works."""
    a, b = tensors
    ttd_a, ttd_b = ttds

    assert a.shape == b.shape
    tensor_sum = a + b

    assert_default_epsilon(ttd_a + ttd_b, tensor_sum)
    assert_default_epsilon(ttd_b + ttd_a, tensor_sum)
    assert_default_epsilon(np.add(ttd_a, ttd_b), tensor_sum)
    assert_default_epsilon(np.add(ttd_b, ttd_a), tensor_sum)

    ttd_copy = ttd_a.copy()
    ttd_copy += ttd_b
    assert_default_epsilon(ttd_copy, tensor_sum)


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
def test_sub(tensors: TestTensorPair, ttds: TestTTDPair) -> None:
    """Test that TTD addition works."""
    a, b = tensors
    ttd_a, ttd_b = ttds

    assert a.shape == b.shape
    tensor_diff = a - b

    assert_default_epsilon(ttd_a - ttd_b, tensor_diff)
    assert_default_epsilon(ttd_b - ttd_a, -tensor_diff)
    assert_default_epsilon(np.subtract(ttd_a, ttd_b), tensor_diff)
    assert_default_epsilon(np.subtract(ttd_b, ttd_a), -tensor_diff)

    ttd_copy = ttd_a.copy()
    ttd_copy -= ttd_b
    assert_default_epsilon(ttd_copy, tensor_diff)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize(("scalar"), deepcopy(TEST_SCALARS))
def test_scalar_multiplication(tensor: TestTensor, ttd: TestTTD, scalar: float) -> None:
    """Test that TTD scalar multiplication works."""
    scaled_tensor = tensor * scalar
    assert_default_epsilon(ttd * scalar, scaled_tensor)
    assert_default_epsilon(scalar * ttd, scaled_tensor)
    assert_default_epsilon(np.multiply(ttd, scalar), scaled_tensor)
    assert_default_epsilon(np.multiply(scalar, ttd), scaled_tensor)

    ttd_copy = ttd.copy()
    ttd_copy *= scalar
    assert_default_epsilon(ttd_copy, scaled_tensor)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_negation(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test that TTD negation works."""
    assert_default_epsilon(-ttd, -tensor)
    assert_default_epsilon(np.negative(ttd), np.negative(tensor))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_transpose(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test that TTD transpose works."""
    assert_default_epsilon(np.transpose(ttd), np.transpose(tensor))
    assert_default_epsilon(ttd.T, tensor.T)

    axes = (1, 0, *range(2, tensor.ndim))
    assert_default_epsilon(
        np.transpose(ttd, axes),
        np.transpose(tensor, axes),
    )

    axes = (*range(1, tensor.ndim), 0)
    assert_default_epsilon(
        np.transpose(ttd, axes),
        np.transpose(tensor, axes),
    )

    axes = tuple(reversed(range(tensor.ndim)))
    assert_default_epsilon(
        np.transpose(ttd, axes),
        np.transpose(tensor, axes),
    )


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_tensordot(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test that TTD tensordot works."""
    # the zero and single axis cases nearly double the dimensionality of the
    # tensor, so should be skipped for the large inputs to prevent OOMs
    if tensor.ndim <= 8:
        assert_default_epsilon(
            np.tensordot(ttd, ttd, axes=0),
            np.tensordot(tensor, tensor, axes=0),
        )

        assert_default_epsilon(
            np.tensordot(ttd, ttd.T, axes=1),
            np.tensordot(tensor, tensor.T, axes=1),
        )

        assert_default_epsilon(
            np.tensordot(ttd, ttd, axes=(0, 0)),
            np.tensordot(tensor, tensor, axes=(0, 0)),
        )

        assert_default_epsilon(
            np.tensordot(ttd, ttd, axes=(-1, -1)),
            np.tensordot(tensor, tensor, axes=(-1, -1)),
        )

        assert_default_epsilon(
            np.tensordot(ttd, ttd, axes=(1, 1)),
            np.tensordot(tensor, tensor, axes=(1, 1)),
        )

    axes = (1, 0, *range(2, ttd.ndim))
    assert_default_epsilon(
        np.tensordot(ttd, np.transpose(ttd.T, axes=axes)),
        np.tensordot(tensor, np.transpose(tensor.T, axes=axes)),
    )

    axes_single = tuple(range(tensor.ndim - 1))
    axes = (axes_single, axes_single)
    assert_default_epsilon(
        np.tensordot(ttd, ttd, axes=axes),
        np.tensordot(tensor, tensor, axes=axes),
    )

    axes_single = tuple(range(tensor.ndim))
    axes = (axes_single, axes_single)
    assert_default_epsilon(
        np.tensordot(ttd, ttd, axes=axes), np.tensordot(tensor, tensor, axes=axes)
    )
