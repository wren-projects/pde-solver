from copy import deepcopy

import numpy as np
import pytest

from .common import TEST_TTD, TestTensor, TestTTD, assert_default_epsilon


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
