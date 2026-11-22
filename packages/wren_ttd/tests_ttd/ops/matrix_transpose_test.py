# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_matrix_transpose(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test matrix_transpose via ops, NumPy, and the mT property."""
    expected = np.linalg.matrix_transpose(tensor)
    assert_default_epsilon(ops.matrix_transpose(ttd), expected)
    assert_default_epsilon(ttd.mT, expected)
    assert_default_epsilon(np.linalg.matrix_transpose(ttd), expected)


def test_matrix_transpose_rejects_one_dimension() -> None:
    """Match NumPy's minimum dimensionality requirement."""
    ttd = TTD(np.ones(3))
    with pytest.raises(ValueError, match="at least 2-d"):
        _ = ops.matrix_transpose(ttd)
    with pytest.raises(ValueError, match="at least 2-d"):
        _ = ttd.mT
