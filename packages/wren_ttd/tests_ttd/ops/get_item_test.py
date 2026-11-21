# pyright: reportAny=false, reportUnknownVariableType=false
import warnings
from copy import deepcopy
from typing import Any, cast

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import TTD
from wren_ttd._numpy_api import DenseWarning

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestIndexing:
    """Tests for TTD indexing."""

    def test_full(self, ttd: TestTTD, tensor: TestTensor) -> None:
        """Test full indexing."""
        for index, value in np.ndenumerate(tensor):
            assert_default_epsilon(ttd[index], value)

    def test_single_axis(self, ttd: TestTTD, tensor: TestTensor) -> None:
        """Test indexing along a single axis."""
        for i in range(tensor.shape[0]):
            assert_default_epsilon(ttd[i], tensor[i])
            assert_default_epsilon(ttd[:i], tensor[:i])
            assert_default_epsilon(ttd[i:], tensor[i:])
            assert_default_epsilon(ttd[i::2], tensor[i::2])

    def test_double_axis(self, ttd: TestTTD, tensor: TestTensor) -> None:
        """Test indexing along two axes with integers."""
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                assert_default_epsilon(ttd[i, j], tensor[i, j])
                assert_default_epsilon(cast(TestTTD, ttd[i])[j], tensor[i][j])

    def test_double_axis_slice(self, ttd: TestTTD, tensor: TestTensor) -> None:
        """Test indexing along two axes with slices."""
        for j in range(tensor.shape[1]):
            assert_default_epsilon(ttd[:, j], tensor[:, j])
            n = tensor.shape[0] // 2
            assert_default_epsilon(ttd[:n, j], tensor[:n, j])
            assert_default_epsilon(ttd[:n:2, j], tensor[:n:2, j])


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_newaxis(tensor: TestTensor, ttd: TestTTD) -> None:
    """Insert singleton axes like NumPy."""
    assert_default_epsilon(ttd[None], tensor[None])
    assert_default_epsilon(ttd[..., None], tensor[..., None])
    assert_default_epsilon(ttd[:, None], tensor[:, None])
    assert_default_epsilon(ttd[None, ..., None], tensor[None, ..., None])


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_ellipsis(tensor: TestTensor, ttd: TestTTD) -> None:
    """Expand Ellipsis against the remaining modes."""
    assert_default_epsilon(ttd[...], tensor[...])
    assert_default_epsilon(ttd[0, ...], tensor[0, ...])
    assert_default_epsilon(ttd[..., 1], tensor[..., 1])
    assert_default_epsilon(ttd[0, ..., 1], tensor[0, ..., 1])


def test_newaxis_with_scalar_result() -> None:
    """Keep singleton axes when every original mode is indexed away."""
    dense = np.array([1.0, 2.0])
    ttd = TTD(dense)

    assert_default_epsilon(ttd[0, None], dense[0, None])
    assert_default_epsilon(ttd[None, 0], dense[None, 0])
    assert_default_epsilon(ttd[None, 0, None], dense[None, 0, None])
    assert ttd[0, None].shape == (1,)


def test_empty_tuple_returns_copy() -> None:
    """Treat an empty selection as a full slice."""
    ttd = TTD.ones((2, 3))

    assert_default_epsilon(cast(TestTTD, ttd[()]), np.ones((2, 3)))


def test_too_many_indices() -> None:
    """Reject selections that consume more modes than the tensor has."""
    ttd = TTD.ones((2, 3))

    with pytest.raises(IndexError, match="too many indices"):
        _ = ttd[0, 0, 0]


def test_multiple_ellipses_are_rejected() -> None:
    """Reject selections with more than one Ellipsis."""
    ttd = TTD.ones((2, 3))

    with pytest.raises(IndexError, match="single ellipsis"):
        _ = ttd[..., ...]


def test_basic_indexing_does_not_warn() -> None:
    """Keep basic indexing exact and warning-free."""
    ttd = TTD.ones((2, 3, 4))

    with warnings.catch_warnings():
        warnings.simplefilter("error", DenseWarning)
        _ = ttd[0]
        _ = ttd[None]
        _ = ttd[..., 1]
        _ = ttd[0, ..., None]


def test_integer_array_indexing() -> None:
    """Index with integer arrays through the dense fallback."""
    dense = np.arange(12, dtype=np.float64).reshape(3, 4)
    ttd = TTD(dense)
    keys = (
        np.array([0, 2]),
        (slice(None), np.array([1, 3])),
        (np.array([0, 2]), slice(None)),
    )

    for key in keys:
        with pytest.warns(DenseWarning):
            result = ttd[key]
        assert_default_epsilon(result, dense[key])


def test_boolean_indexing() -> None:
    """Index with boolean masks through the dense fallback."""
    dense = np.arange(12, dtype=np.float64).reshape(3, 4)
    ttd = TTD(dense)

    with pytest.warns(DenseWarning):
        masked = ttd[np.array([True, False, True])]
    assert_default_epsilon(masked, dense[[True, False, True]])

    with pytest.warns(DenseWarning):
        masked_2d = ttd[dense > 5]
    assert_default_epsilon(masked_2d, dense[dense > 5])


@pytest.mark.parametrize("key", [True, np.array([[0], [1]]), [[0, 2]]])
def test_advanced_indexing_variants(key: Any) -> None:
    """Support advanced indexing variants with a single warning."""
    dense = np.arange(12, dtype=np.float64).reshape(3, 4)
    ttd = TTD(dense)

    with pytest.warns(DenseWarning):
        result = ttd[key]
    assert_default_epsilon(result, dense[key])
