from copy import deepcopy

import numpy as np
import pytest
from pde_common.tests import (
    TEST_SCALARS,
    TEST_SHAPES,
    TestTensor,
    TestTensorPair,
)
from pde_ttd import TTD

from .common import (
    TEST_PAIR_TTD,
    TEST_TTD,
    TestTTD,
    TestTTDPair,
    assert_default_epsilon,
)


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

    added = ttd + ttd
    rounded = added.rounded()
    assert_default_epsilon(rounded, 2 * tensor)


@pytest.mark.parametrize("shape", deepcopy(TEST_SHAPES))
def test_zeros(shape: tuple[int, ...]) -> None:
    """Test zeros."""
    ttd = TTD.zeros(shape, dtype=np.dtype(np.float64))
    tensor = np.zeros(shape, dtype=np.dtype(np.float64))
    assert_default_epsilon(ttd, tensor)


@pytest.mark.parametrize("shape", deepcopy(TEST_SHAPES))
def test_ones(shape: tuple[int, ...]) -> None:
    """Test ones."""
    ttd = TTD.ones(shape, dtype=np.dtype(np.float64))
    tensor = np.ones(shape, dtype=np.dtype(np.float64))
    assert_default_epsilon(ttd, tensor)


@pytest.mark.parametrize("shape", deepcopy(TEST_SHAPES))
@pytest.mark.parametrize("fill_value", deepcopy(TEST_SCALARS))
def test_full(shape: tuple[int, ...], fill_value: float) -> None:
    """Test full."""
    ttd = TTD.full(shape, fill_value, dtype=np.dtype(np.float64))
    tensor = np.full(shape, fill_value, dtype=np.dtype(np.float64))
    assert_default_epsilon(ttd, tensor)


def test_ranks() -> None:
    """Test ranks."""
    ttd = TTD([np.zeros((1, 2, 2)), np.zeros((2, 3, 2)), np.zeros((2, 2, 1))])
    assert ttd.ranks == (2, 2)


def test_compressed_size() -> None:
    """Test ranks."""
    ttd = TTD([np.zeros((1, 2, 2)), np.zeros((2, 3, 2)), np.zeros((2, 2, 1))])
    assert ttd.compressed_size == 2 * 2 + 2 * 3 * 2 + 2 * 2


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_indexing_full(ttd: TestTTD, tensor: TestTensor) -> None:
    """Test full indexing."""
    for index, value in np.ndenumerate(tensor):
        assert_default_epsilon(ttd[index], value)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_indexing_partial(ttd: TestTTD, tensor: TestTensor) -> None:
    """Test partial indexing."""
    for i in range(tensor.shape[0]):
        assert_default_epsilon(ttd[i], tensor[i])
        assert_default_epsilon(ttd[:i], tensor[:i])
        assert_default_epsilon(ttd[i:], tensor[i:])
        assert_default_epsilon(ttd[i::2], tensor[i::2])

    for j in range(tensor.shape[1]):
        assert_default_epsilon(ttd[:, j], tensor[:, j])
        n = tensor.shape[0] // 2
        assert_default_epsilon(ttd[:n, j], tensor[:n, j])
        assert_default_epsilon(ttd[:n:2, j], tensor[:n:2, j])

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            assert_default_epsilon(ttd[i, j], tensor[i, j])


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

    scale = np.linalg.norm(scaled_tensor)
    assert_default_epsilon(ttd * scalar, scaled_tensor, scale)
    assert_default_epsilon(scalar * ttd, scaled_tensor, scale)
    assert_default_epsilon(np.multiply(ttd, scalar), scaled_tensor, scale)
    assert_default_epsilon(np.multiply(scalar, ttd), scaled_tensor, scale)

    ttd_copy = ttd.copy()
    ttd_copy *= scalar
    assert_default_epsilon(ttd_copy, scaled_tensor, scale)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_negation(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test that TTD negation works."""
    assert_default_epsilon(-ttd, -tensor)
    assert_default_epsilon(np.negative(ttd), np.negative(tensor))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_transpose(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test that TTD transpose works."""
    assert_default_epsilon(ttd.transpose(), tensor.transpose())
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

    axes = (-1, *range(tensor.ndim - 1))
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
def test_swapaxes(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test that TTD swapaxes works."""
    assert_default_epsilon(np.swapaxes(ttd, 0, 1), np.swapaxes(tensor, 0, 1))
    assert_default_epsilon(np.swapaxes(ttd, 0, -1), np.swapaxes(tensor, 0, -1))
    assert_default_epsilon(ttd.swapaxes(-1, -2), tensor.swapaxes(-1, -2))
    assert_default_epsilon(ttd.swapaxes(0, 0), tensor.swapaxes(0, 0))


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
def test_tensordot(tensors: TestTensorPair, ttds: TestTTDPair) -> None:
    """Test that TTD tensordot works."""
    a, b = tensors
    ttd_a, ttd_b = ttds

    assert a.shape == b.shape
    assert ttd_a.shape == ttd_b.shape

    assert_default_epsilon(
        np.tensordot(ttd_a, ttd_b, axes=0),
        np.tensordot(a, b, axes=0),
    )

    assert_default_epsilon(
        np.tensordot(ttd_a, ttd_b.T, axes=1),
        np.tensordot(a, b.T, axes=1),
    )

    axes = (1, 0, *range(2, ttd_a.ndim))
    assert_default_epsilon(
        np.tensordot(ttd_a, np.transpose(ttd_b.T, axes=axes)),
        np.tensordot(a, np.transpose(b.T, axes=axes)),
    )

    assert_default_epsilon(
        np.tensordot(ttd_a, ttd_b, axes=(0, 0)),
        np.tensordot(a, b, axes=(0, 0)),
    )

    assert_default_epsilon(
        np.tensordot(ttd_a, ttd_b, axes=(-1, -1)),
        np.tensordot(a, b, axes=(-1, -1)),
    )

    assert_default_epsilon(
        np.tensordot(ttd_a, ttd_b, axes=(1, 1)),
        np.tensordot(a, b, axes=(1, 1)),
    )

    axes_single = tuple(range(a.ndim - 1))
    axes = (axes_single, axes_single)
    assert_default_epsilon(
        np.tensordot(ttd_a, ttd_b, axes=axes),
        np.tensordot(a, b, axes=axes),
    )

    axes_single = tuple(range(a.ndim))
    axes = (axes_single, axes_single)
    assert_default_epsilon(
        np.tensordot(ttd_a, ttd_b, axes=axes), np.tensordot(a, b, axes=axes)
    )


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_stack_single(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test that TTD stack works with single tensor."""
    assert_default_epsilon(
        np.stack(ttd),
        np.stack(tensor),
    )


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
def test_stack(tensors: TestTensorPair, ttds: TestTTDPair) -> None:
    """Test that TTD stack works."""
    assert_default_epsilon(
        np.stack(ttds),
        np.stack(tensors),
    )

    assert_default_epsilon(
        np.stack(ttds, axis=-1),
        np.stack(tensors, axis=-1),
    )

    assert_default_epsilon(
        np.stack(ttds, axis=1),
        np.stack(tensors, axis=1),
    )

    assert_default_epsilon(
        np.stack(ttds * 2, axis=0),
        np.stack(tensors * 2, axis=0),
    )

    assert_default_epsilon(
        np.stack(ttds * 2, axis=-1),
        np.stack(tensors * 2, axis=-1),
    )

    assert_default_epsilon(
        np.stack(ttds * 2, axis=1),
        np.stack(tensors * 2, axis=1),
    )


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_gradient(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test that TTD gradient works."""
    scale = np.linalg.norm(tensor)

    assert_default_epsilon(np.gradient(ttd), np.gradient(tensor), scale)

    assert_default_epsilon(np.gradient(ttd, axis=1), np.gradient(tensor, axis=1), scale)

    assert_default_epsilon(
        np.gradient(ttd, axis=range(1, ttd.ndim)),
        np.gradient(tensor, axis=range(1, tensor.ndim)),
        scale,
    )

    assert_default_epsilon(
        np.gradient(ttd, axis=range(-ttd.ndim + 1, 0)),
        np.gradient(tensor, axis=range(-tensor.ndim + 1, 0)),
        scale,
    )

    steps = tuple((n + 1) / 10 for n in range(ttd.ndim))
    assert_default_epsilon(np.gradient(ttd, *steps), np.gradient(tensor, *steps), scale)

    steps = tuple(tuple((x + 1) / 10 for x in range(n)) for n in ttd.shape)
    assert_default_epsilon(np.gradient(ttd, *steps), np.gradient(tensor, *steps), scale)

    assert_default_epsilon(
        np.gradient(ttd, edge_order=2), np.gradient(tensor, edge_order=2), scale
    )
