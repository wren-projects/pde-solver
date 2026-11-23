# pyright: reportAny=false, reportUnknownVariableType=false
import numpy as np
import pytest
from wren_common.tests import arange_tensor
from wren_ttd import TTD, ops

from ..common import assert_default_epsilon


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (arange_tensor(3), arange_tensor(3)),
        (arange_tensor(2, 3), arange_tensor(3)),
        (arange_tensor(3), arange_tensor(3, 2)),
        (arange_tensor(2, 3), arange_tensor(3, 4)),
        (
            arange_tensor(2, 3),
            arange_tensor(2, 3, 4),
        ),
        (
            arange_tensor(2, 3, 4),
            arange_tensor(4, 5),
        ),
        (
            arange_tensor(2, 3, 4),
            arange_tensor(4),
        ),
        (
            arange_tensor(4),
            arange_tensor(2, 4, 3),
        ),
    ],
)
def test_matmul(a: np.ndarray, b: np.ndarray) -> None:
    """Compare supported TTD operand ranks against NumPy."""
    result = ops.matmul(TTD(a), TTD(b))
    assert_default_epsilon(result, np.matmul(a, b))


def test_matrix_multiply_operator() -> None:
    """Support Python's matrix multiplication operator."""
    a, b = arrays = arange_tensor(2, 3), arange_tensor(3, 4)
    ttd_a, ttd_b = map(TTD, arrays)

    assert_default_epsilon(ttd_a @ ttd_b, a @ b)


def test_matrix_multiply_operator_inplace() -> None:
    """Support Python's matrix multiplication operator."""
    a, b = arrays = arange_tensor(3, 3), 5 * arange_tensor(3, 3)
    ttd_a, ttd_b = map(TTD, arrays)

    ttd_a @= ttd_b

    assert_default_epsilon(ttd_a, a @ b)


def test_matmul_rejects_out_for_scalar_result() -> None:
    """Reject an output TTD for a scalar-valued product."""
    a, b = TTD([1.0, 2.0]), TTD([3.0, 4.0])
    with pytest.raises(NotImplementedError, match="TTD-valued"):
        _ = ops.matmul(a, b, out=TTD([1.0]))
