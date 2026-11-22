# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import EDGE_TEST_SCALARS, SMALL_TEST_SCALARS, TestTensor
from wren_ttd import TTD

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("value", SMALL_TEST_SCALARS)
@pytest.mark.parametrize("width", [0, 1, 2])
class TestPad:
    """Tests for TTD padding."""

    def test_pad_single_constant(
        self, tensor: TestTensor, ttd: TestTTD, value: float, width: int
    ) -> None:
        """Test pad with the selected constant value in single width."""
        expected = np.pad(tensor, width, mode="constant", constant_values=value)
        result = np.pad(ttd, width, mode="constant", constant_values=value)
        assert_default_epsilon(result, expected, scale=value)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD[:2]))
@pytest.mark.parametrize("value", EDGE_TEST_SCALARS)
@pytest.mark.parametrize("width", [1, 2])
class TestPadEdgeCases:
    """Tests for TTD padding with large-magnitude values."""

    def test_pad_single_constant_edge(
        self, tensor: TestTensor, ttd: TestTTD, value: float, width: int
    ) -> None:
        """Test pad with large-magnitude constant values."""
        expected = np.pad(tensor, width, mode="constant", constant_values=value)
        result = np.pad(ttd, width, mode="constant", constant_values=value)
        assert_default_epsilon(result, expected, scale=value)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestPadWidthForms:
    """Tests for the accepted NumPy `pad_width` forms."""

    def test_scalar_width(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test a scalar width applied to every axis and side."""
        expected = np.pad(tensor, 1)
        result = np.pad(ttd, 1)
        assert_default_epsilon(result, expected)

    def test_numpy_integer_width(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that NumPy integers are accepted as scalar widths."""
        expected = np.pad(tensor, np.int64(2))
        result = np.pad(ttd, np.int64(2))
        assert_default_epsilon(result, expected)

    def test_uniform_pair_width(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test NumPy's broadcast `(before, after)` form."""
        expected = np.pad(tensor, (1, 1))
        result = np.pad(ttd, (1, 1))
        assert_default_epsilon(result, expected)

    def test_per_axis_uniform_pairs(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test one equal `(before, after)` pair per axis."""
        pad_width = ((1, 1),) * ttd.ndim
        expected = np.pad(tensor, pad_width)
        result = np.pad(ttd, pad_width)
        assert_default_epsilon(result, expected)


def test_pad_per_axis_uniform_pairs_literal() -> None:
    """Test the literal `((1, 1), (1, 1))` form on a matrix."""
    tensor, ttd = deepcopy(TEST_TTD[0])
    expected = np.pad(tensor, ((1, 1), (1, 1)))
    result = np.pad(ttd, ((1, 1), (1, 1)))
    assert_default_epsilon(result, expected)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD[:2]))
@pytest.mark.parametrize("value", [0, 2.5])
def test_pad_scalar_constant_values(
    tensor: TestTensor, ttd: TestTTD, value: float
) -> None:
    """Test zero and nonzero scalar constant values."""
    expected = np.pad(tensor, 1, mode="constant", constant_values=value)
    result = np.pad(ttd, 1, mode="constant", constant_values=value)
    assert_default_epsilon(result, expected)


@pytest.fixture
def matrix_ttd() -> TestTTD:
    """Return a 2-D TTD for testing rejected padding forms."""
    return TTD(np.arange(6, dtype=np.float64).reshape(2, 3))


class TestPadRejectedForms:
    """Tests for padding forms that cannot be represented exactly."""

    def test_rejects_uneven_per_axis_widths(self, matrix_ttd: TestTTD) -> None:
        """Test that different widths per axis are rejected."""
        with pytest.raises(NotImplementedError, match="uniform widths"):
            _ = np.pad(matrix_ttd, ((1, 1), (2, 2)))

    def test_rejects_missing_per_axis_pairs(self) -> None:
        """Test that per-axis pairs must match the tensor rank."""
        ttd = TTD(np.ones((2, 3, 4)))
        with pytest.raises(NotImplementedError, match="uniform widths"):
            _ = np.pad(ttd, ((1, 1), (1, 1)))

    def test_rejects_unequal_before_after(self, matrix_ttd: TestTTD) -> None:
        """Test that a `(before, after)` form with unequal sides is rejected."""
        with pytest.raises(NotImplementedError, match="both sides"):
            _ = np.pad(matrix_ttd, (1, 2))

    def test_rejects_unequal_before_after_per_axis(self, matrix_ttd: TestTTD) -> None:
        """Test that unequal per-axis sides are rejected."""
        with pytest.raises(NotImplementedError, match="both sides"):
            _ = np.pad(matrix_ttd, ((1, 2), (1, 2)))

    def test_rejects_negative_width(self, matrix_ttd: TestTTD) -> None:
        """Test that cropping via negative widths is rejected."""
        with pytest.raises(NotImplementedError, match="negative widths"):
            _ = np.pad(matrix_ttd, -1)

    def test_rejects_per_axis_constant_values(self, matrix_ttd: TestTTD) -> None:
        """Test that per-axis constant values are rejected."""
        with pytest.raises(NotImplementedError, match="Per-axis constant_values"):
            _ = np.pad(matrix_ttd, 1, constant_values=(1, 2))

    def test_rejects_unsupported_mode(self, matrix_ttd: TestTTD) -> None:
        """Test that non-constant padding modes are rejected."""
        with pytest.raises(NotImplementedError, match="only 'constant'"):
            _ = np.pad(matrix_ttd, 1, mode="edge")

    def test_rejects_unsupported_kwargs(self, matrix_ttd: TestTTD) -> None:
        """Test that keyword arguments of other modes are rejected."""
        with pytest.raises(NotImplementedError, match="Unsupported keyword arguments"):
            _ = np.pad(matrix_ttd, 1, stat_length=1)
