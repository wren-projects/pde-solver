# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("divisor", [2.0, -1.5])
class TestDivision:
    """Tests for TTD scalar division."""

    def test_operator(self, tensor: TestTensor, ttd: TestTTD, divisor: float) -> None:
        """Divide a TTD by a scalar with the true division operator."""
        assert_default_epsilon(ttd / divisor, tensor / divisor, scale=tensor)
        assert_default_epsilon(ops.divide(ttd, divisor), tensor / divisor, scale=tensor)

    def test_numpy_ufunc(
        self, tensor: TestTensor, ttd: TestTTD, divisor: float
    ) -> None:
        """Divide a TTD by a scalar through NumPy's divide ufunc."""
        assert_default_epsilon(np.divide(ttd, divisor), tensor / divisor, scale=tensor)


def test_division_rejects_ttd_divisor() -> None:
    """Do not silently densify elementwise TTD division."""
    ttd = TTD.ones((2, 3))
    with pytest.raises(NotImplementedError, match="scalar divisor"):
        _ = ttd / ttd
