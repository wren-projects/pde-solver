# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("axis", [None, 0, -1, (0, -1)])
def test_squeeze(
    tensor: TestTensor, ttd: TestTTD, axis: int | tuple[int, ...] | None
) -> None:
    """Test squeeze via ops, method, and NumPy."""
    tensor = np.expand_dims(tensor, axis=(0, -1))
    ttd = ops.expand_dims(ttd, axis=(0, -1))
    expected = np.squeeze(tensor, axis=axis)
    assert_default_epsilon(ops.squeeze(ttd, axis), expected)
    assert_default_epsilon(ttd.squeeze(axis), expected)
    assert_default_epsilon(np.squeeze(ttd, axis=axis), expected)


def test_squeeze_all_dimensions_returns_scalar() -> None:
    """Squeezing every mode follows NumPy scalar-value semantics."""
    ttd = TTD(np.ones((1, 1)))
    result = ops.squeeze(ttd)

    assert np.isscalar(result)
    assert result == 1


def test_squeeze_rejects_non_singleton_axis() -> None:
    """Reject an explicitly selected mode whose size is not one."""
    with pytest.raises(ValueError, match="size not equal to one"):
        _ = ops.squeeze(TTD(np.ones((2, 1))), axis=0)
