# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy
from typing import Any, Unpack, cast, override

import numpy as np
import pytest
from wren_common.tests import TEST_SCALARS, TEST_TENSORS, TestTensor
from wren_common.types import NDArray, Scalar
from wren_ttd import DEFAULT_EPSILON, TTD, CompressionParameters
from wren_ttd.types import Core

from .common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_roundtrip_compression(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test the round-trip compression/decompression of a tensor."""
    # the compression is done in TEST_TTD and the decompression is done
    # implicitly inside assert_default_epsilon
    assert_default_epsilon(ttd, tensor)


@pytest.mark.parametrize("tensor", deepcopy(TEST_TENSORS))
def test_construct_from_ndarray(tensor: TestTensor) -> None:
    """Test construction from an ndarray."""
    result = TTD(tensor)

    assert result.dtype == tensor.dtype
    assert_default_epsilon(result, tensor)


@pytest.mark.parametrize(
    "tensor", [np.arange(1, 6, dtype=np.float64), np.zeros(4), np.ones(3)]
)
def test_construct_from_1d_ndarray(tensor: TestTensor) -> None:
    """Test construction from a one-dimensional ndarray."""
    result = TTD(tensor)

    assert result.ndim == 1
    assert_default_epsilon(result, tensor)


def test_construct_from_ndarray_preserves_dtype() -> None:
    """Test construction from an ndarray preserves the dtype."""
    tensor = np.ones((2, 3, 4), dtype=np.float32)
    result = TTD(tensor)

    assert result.dtype == np.float32
    assert_default_epsilon(result, tensor, epsilon=1e-5)


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, 3.0],
        [[1.0, 2.0], [3.0, 4.0]],
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        ((1.0, 2.0), (3.0, 4.0)),
    ],
)
def test_construct_from_sequence(data: Any) -> None:
    """Test construction from raw Python sequences."""
    result: TTD[Any] = TTD(data)
    tensor = np.asarray(data, dtype=np.float64)

    assert result.dtype == np.float64
    assert_default_epsilon(result, tensor)


def test_construct_from_integer_ndarray() -> None:
    """Test construction from an integer ndarray promotes to floating."""
    tensor = np.arange(1, 7).reshape(2, 3)
    result: TTD[Any] = TTD(tensor)

    assert result.dtype == np.float64
    assert_default_epsilon(result, tensor.astype(np.float64))


def test_construct_from_integer_sequence() -> None:
    """Test construction from a sequence of integers promotes to floating."""
    data = [[1, 2], [3, 4]]
    result: TTD[Any] = TTD(data)

    assert result.dtype == np.float64
    assert_default_epsilon(result, np.asarray(data, dtype=np.float64))


def test_construct_from_sequence_preserves_dtype() -> None:
    """Test construction from a sequence preserves a floating dtype."""
    data = [[np.float32(1), np.float32(2)], [np.float32(3), np.float32(4)]]
    result = cast(TTD[np.float32], TTD(data))

    assert result.dtype == np.float32
    assert_default_epsilon(result, np.asarray(data), epsilon=1e-5)


def test_construct_from_sequence_with_dtype() -> None:
    """Test construction from a sequence honors the requested dtype."""
    result = TTD([[1, 2], [3, 4]], dtype=np.dtype(np.float32))

    assert result.dtype == np.float32
    assert_default_epsilon(
        result, np.array([[1, 2], [3, 4]], dtype=np.float32), epsilon=1e-5
    )


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_construct_from_ttd(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test construction from another TTD shallow-copies its data."""
    result = TTD(ttd)

    assert result.dtype == ttd.dtype
    assert result.data is not ttd.data
    assert all(a is b for a, b in zip(result.data, ttd.data, strict=True))
    assert_default_epsilon(result, tensor)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_construct_from_cores(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test construction from an iterable of cores."""
    for data in ([*ttd.data], iter(ttd.data)):
        result = TTD(data)

        assert result.dtype == ttd.dtype
        assert_default_epsilon(result, tensor)


@pytest.mark.parametrize("data", [42, 3.14, None, object()])
def test_construct_invalid_type(data: Any) -> None:
    """Test construction from an unsupported type raises an error."""
    with pytest.raises(TypeError):
        _ = TTD(data)


def test_construct_from_ndarray_with_epsilon() -> None:
    """Test that epsilon is passed through the constructor."""
    tensor = np.random.default_rng(0).random((4, 4))

    exact = TTD(tensor)
    approximate = TTD(tensor, epsilon=10.0)

    assert approximate.compressed_size < exact.compressed_size


def test_construct_from_ndarray_forwards_compression_parameters() -> None:
    """Test that subclasses can add their own compression parameters."""
    received: dict[str, Any] = {}

    class IteratedTTD(TTD[np.float64]):
        iterations: int

        @override
        def __init__(
            self,
            data: Any,
            *,
            dtype: np.dtype[np.float64] | None = None,
            epsilon: Scalar = DEFAULT_EPSILON,
            iterations: int = 1,
        ) -> None:
            self.iterations = iterations
            super().__init__(data, dtype=dtype, epsilon=epsilon)

        @override
        def _compress[DT: np.floating](
            self,
            array: NDArray[DT],
            **compression_params: Unpack[CompressionParameters],
        ) -> list[Core[DT]]:
            received.update(compression_params)
            received["iterations"] = self.iterations
            return super()._compress(array, **compression_params)

    tensor = np.ones((2, 3))
    result = IteratedTTD(tensor, epsilon=1e-5, iterations=7)

    assert received == {"epsilon": 1e-5, "iterations": 7}
    assert result.dtype == tensor.dtype
    assert_default_epsilon(result, tensor, epsilon=1e-5)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_shape(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test shape."""
    assert ttd.shape == tensor.shape
    assert np.shape(ttd) == np.shape(tensor)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_ndim(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test ndim."""
    assert ttd.ndim == tensor.ndim
    assert np.ndim(ttd) == np.ndim(tensor)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_size(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test size."""
    assert ttd.size == tensor.size
    assert np.size(ttd) == np.size(tensor)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_rounding(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test rounding."""
    assert_default_epsilon(ttd.rounded(), tensor)

    added = ttd + ttd
    rounded = added.rounded()
    assert_default_epsilon(rounded, 2 * tensor)


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


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize(
    "index",
    [
        0,
        2,
        -1,
        slice(1, -1),
        slice(None, None, 2),
        slice(1, None, 2),
        slice(None, None, -1),
    ],
)
@pytest.mark.parametrize("value", TEST_SCALARS)
def test_setitem(
    tensor: TestTensor, ttd: TestTTD, index: int | slice, value: Scalar
) -> None:
    """Test zeroing a single mode addressed by an integer or a slice."""
    for axis, _ in enumerate(tensor.shape):
        leading = (slice(None),) * axis
        trailing = (slice(None),) * (len(tensor.shape) - axis - 1)
        key = (*leading, index, *trailing)

        expected = tensor.copy()
        expected[key] = value

        result = ttd.copy()
        result[key] = value

        assert_default_epsilon(result, expected, scale=value)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_setitem_boundary(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test repeatedly zeroing boundary slices of a TTD."""
    expected = tensor.copy()
    result = ttd.copy()

    for axis, length in enumerate(tensor.shape):
        for index in (0, length - 1):
            key: list[int | slice] = [slice(None)] * tensor.ndim
            key[axis] = index
            expected[tuple(key)] = 0
            result[tuple(key)] = 0

    assert_default_epsilon(result, expected)
