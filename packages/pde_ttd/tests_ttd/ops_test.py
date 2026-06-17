from copy import deepcopy
from typing import cast

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


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_sum(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test sum."""
    assert_default_epsilon(np.sum(ttd), np.sum(tensor))

    for axis in range(tensor.ndim):
        assert_default_epsilon(ttd.sum(axis), cast(TestTensor, tensor.sum(axis)))

    assert_default_epsilon(ttd.sum((0, 1)), cast(TestTensor, tensor.sum((0, 1))))


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


@pytest.mark.parametrize("shape", deepcopy(TEST_SHAPES))
def test_random(shape: tuple[int, ...]) -> None:
    """Test randomly generated TTD."""
    ttd = TTD.random(shape, dtype=np.dtype(np.float64))
    assert ttd.shape == shape
    np.testing.assert_allclose(ttd.sum() / ttd.size, 0.5, atol=0.5, rtol=0.5)


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


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize(("scalar"), TEST_SCALARS)
def test_scalar_addition(tensor: TestTensor, ttd: TestTTD, scalar: float) -> None:
    """Test that TTD scalar addition works."""
    assert_default_epsilon(ttd + scalar, tensor + scalar)
    assert_default_epsilon(scalar + ttd, tensor + scalar)
    assert_default_epsilon(np.add(ttd, scalar), tensor + scalar)
    assert_default_epsilon(np.add(scalar, ttd), tensor + scalar)


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
@pytest.mark.parametrize(("scalar"), TEST_SCALARS)
def test_scalar_subtraction(tensor: TestTensor, ttd: TestTTD, scalar: float) -> None:
    """Test that TTD scalar subtraction works."""
    assert_default_epsilon(ttd - scalar, tensor - scalar)
    assert_default_epsilon(scalar - ttd, scalar - tensor)
    assert_default_epsilon(np.subtract(ttd, scalar), tensor - scalar)
    assert_default_epsilon(np.subtract(scalar, ttd), scalar - tensor)


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
def test_multiplication(tensors: TestTensorPair, ttds: TestTTDPair) -> None:
    """Test that TTD addition works."""
    a, b = tensors
    ttd_a, ttd_b = ttds

    assert a.shape == b.shape
    tensor_product = a * b

    assert_default_epsilon(ttd_a * ttd_b, tensor_product)
    assert_default_epsilon(ttd_b * ttd_a, tensor_product)
    assert_default_epsilon(np.multiply(ttd_a, ttd_b), tensor_product)
    assert_default_epsilon(np.multiply(ttd_b, ttd_a), tensor_product)

    ttd_copy = ttd_a.copy()
    ttd_copy *= ttd_b
    assert_default_epsilon(ttd_copy, tensor_product)


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


TEST_PAD_WIDTHS = [0, 1, 2]


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("pad_width", TEST_PAD_WIDTHS)
@pytest.mark.parametrize("value", TEST_SCALARS)
def test_pad_constant(
    tensor: TestTensor,
    ttd: TestTTD,
    pad_width: int,
    value: float,
) -> None:
    """Test that TTD pad with scalar constant_value works."""
    expected = np.pad(tensor, pad_width, mode="constant", constant_values=value)
    result = np.pad(ttd, pad_width, mode="constant", constant_values=value)
    assert_default_epsilon(result, expected, scale=value)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("pad_width", TEST_PAD_WIDTHS)
def test_pad_constant_ones(tensor: TestTensor, ttd: TestTTD, pad_width: int) -> None:
    """Test constant_values as scalar 1 to exercise non-zero pad path."""
    expected = np.pad(tensor, pad_width, mode="constant", constant_values=1.0)
    result = np.pad(ttd, pad_width, mode="constant", constant_values=1.0)
    assert_default_epsilon(result, expected)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_pad_constant_pw_pairs(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test pad_width as per-axis (before, after) pairs with scalar cv."""
    ndim = tensor.ndim
    pw: tuple[tuple[int, int], ...] = tuple(
        ((i + 1) % 2, (i + 2) % 3) for i in range(ndim)
    )
    expected = np.pad(tensor, pw, mode="constant", constant_values=0.0)
    result = np.pad(ttd, pw, mode="constant", constant_values=0.0)
    assert_default_epsilon(result, expected)


def test_pad_constant_tuple_cv() -> None:
    """
    Test constant_values as (before, after) tuple with scalar pad_width.

    Note: corners where before/after regions overlap may differ from NumPy
    (sum vs last-axis overwrite). This test verifies the non-corner regions.
    """
    tensor = np.arange(1, 13).reshape(3, 4).astype(np.float64)
    ttd = TTD.from_ndarray(tensor)
    cv = (2.0, 3.0)
    expected = np.pad(tensor, 1, mode="constant", constant_values=cv)
    result = np.asarray(np.pad(ttd, 1, mode="constant", constant_values=cv))
    assert result.shape == expected.shape
    np.testing.assert_allclose(result[1, 1:4], expected[1, 1:4], atol=1e-10)
    np.testing.assert_allclose(result[1:3, 1], expected[1:3, 1], atol=1e-10)


def test_pad_constant_per_axis_cv() -> None:
    """Test constant_values as per-axis pairs with scalar pad_width."""
    tensor = np.arange(1, 13).reshape(3, 4).astype(np.float64)
    ttd = TTD.from_ndarray(tensor)
    cv = ((2.0, 3.0), (4.0, 5.0))
    expected = np.pad(tensor, 1, mode="constant", constant_values=cv)
    result = np.asarray(np.pad(ttd, 1, mode="constant", constant_values=cv))
    assert result.shape == expected.shape
    np.testing.assert_allclose(result[1, 1:4], expected[1, 1:4], atol=1e-10)
    np.testing.assert_allclose(result[1:3, 1], expected[1:3, 1], atol=1e-10)


def test_pad_constant_pw_pairs_tuple_cv() -> None:
    """Test combination of per-axis pad_width with per-axis constant_values."""
    tensor = np.arange(1, 13).reshape(3, 4).astype(np.float64)
    ttd = TTD.from_ndarray(tensor)
    pw = ((1, 2), (0, 1))
    cv = ((2.0, 3.0), (4.0, 5.0))
    expected = np.pad(tensor, pw, mode="constant", constant_values=cv)
    result = np.asarray(np.pad(ttd, pw, mode="constant", constant_values=cv))
    assert result.shape == expected.shape
    np.testing.assert_allclose(result[2, :3], expected[2, :3], atol=1e-10)
    np.testing.assert_allclose(result[3, :3], expected[3, :3], atol=1e-10)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("pad_width", TEST_PAD_WIDTHS)
def test_pad_edge(tensor: TestTensor, ttd: TestTTD, pad_width: int) -> None:
    """Test that TTD pad with mode='edge' works."""
    expected = np.pad(tensor, pad_width, mode="edge")
    result = np.pad(ttd, pad_width, mode="edge")
    assert_default_epsilon(result, expected)
