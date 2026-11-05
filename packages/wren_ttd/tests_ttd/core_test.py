# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TEST_SCALARS, TEST_SHAPES, TestTensor
from wren_ttd import TTD

from .common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_roundtrip_compression(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test the round-trip compression/decompression of a tensor."""
    # the compression is done in TEST_TTD and the decompression is done
    # implicitly inside assert_default_epsilon
    assert_default_epsilon(ttd, tensor)


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
class TestRounding:
    """Tests for TTD rounding."""

    def test_plain(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test rounding a plain TTD."""
        assert_default_epsilon(ttd.rounded(), tensor)

    def test_with_inflated_ranks(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test rounding a TTD with inflated ranks."""
        added = ttd + ttd
        rounded = added.rounded()
        assert_default_epsilon(rounded, 2 * tensor)


def test_ranks() -> None:
    """Test ranks."""
    ttd = TTD([np.zeros((1, 2, 2)), np.zeros((2, 3, 2)), np.zeros((2, 2, 1))])
    assert ttd.ranks == (2, 2)


def test_compressed_size() -> None:
    """Test compressed size."""
    ttd = TTD([np.zeros((1, 2, 2)), np.zeros((2, 3, 2)), np.zeros((2, 2, 1))])
    assert ttd.compressed_size == 2 * 2 + 2 * 3 * 2 + 2 * 2


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
