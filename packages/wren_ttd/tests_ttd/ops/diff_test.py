# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize(
    ("n", "axis"),
    [(0, 0), (1, 0), (2, 1), (5, -1)],
)
def test_diff(tensor: TestTensor, ttd: TestTTD, n: int, axis: int) -> None:
    """Test diff via ops and NumPy."""
    expected = np.diff(tensor, n=n, axis=axis)
    assert_default_epsilon(ops.diff(ttd, n=n, axis=axis), expected, scale=tensor)
    assert_default_epsilon(np.diff(ttd, n=n, axis=axis), expected, scale=tensor)


@pytest.mark.parametrize("ttd", [ttd for _, ttd in deepcopy(TEST_TTD)])
def test_diff_preserves_tt_ranks(ttd: TestTTD) -> None:
    """Differencing a mode changes its size without inflating TT ranks."""
    result = ops.diff(ttd, axis=0)
    assert result.ranks == ttd.ranks


def test_diff_rejects_negative_order() -> None:
    """Reject a negative difference order."""
    with pytest.raises(ValueError, match="non-negative"):
        _ = ops.diff(TTD(np.ones((2, 3))), n=-1)


def test_diff_rejects_prepend() -> None:
    """Report deferred prepend functionality explicitly."""
    with pytest.raises(NotImplementedError, match="prepend and append"):
        _ = ops.diff(TTD(np.ones((2, 3))), prepend=1.0)


def test_diff_rejects_append() -> None:
    """Report deferred append functionality explicitly."""
    with pytest.raises(NotImplementedError, match="prepend and append"):
        _ = ops.diff(TTD(np.ones((2, 3))), append=1.0)
