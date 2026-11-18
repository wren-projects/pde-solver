# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy
from typing import Any, cast

import numpy as np
import pytest
from wren_common.tests import TestTensor, arange_tensor
from wren_ttd import TTD

from .common import TEST_TTD, TestTTD, assert_default_epsilon

# NumPy's ufunc stubs do not model TTD operands or one-element tuple outputs.
np_add = cast(Any, np.add)
np_subtract = cast(Any, np.subtract)
np_multiply = cast(Any, np.multiply)
np_negative = cast(Any, np.negative)
np_matmul = cast(Any, np.matmul)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_add_out_tuple(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test ``np.add`` with NumPy's one-element ``out`` tuple."""
    out = TTD.zeros(ttd.shape, dtype=ttd.dtype)

    result = np_add(ttd, ttd, out=(out,))

    assert result is out
    assert_default_epsilon(out, 2 * tensor, scale=tensor)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_subtract_out_tuple(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test ``np.subtract`` with NumPy's one-element ``out`` tuple."""
    out = TTD.zeros(ttd.shape, dtype=ttd.dtype)

    result = np_subtract(ttd, ttd, out=(out,))

    assert result is out
    assert_default_epsilon(out, np.zeros_like(tensor), scale=tensor)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_multiply_out_tuple(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test ``np.multiply`` with NumPy's one-element ``out`` tuple."""
    out = TTD.zeros(ttd.shape, dtype=ttd.dtype)

    result = np_multiply(ttd, ttd, out=(out,))

    assert result is out
    assert_default_epsilon(out, tensor * tensor, scale=tensor * tensor)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_negative_out_tuple(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test ``np.negative`` with NumPy's one-element ``out`` tuple."""
    out = TTD.zeros(ttd.shape, dtype=ttd.dtype)

    result = np_negative(ttd, out=(out,))

    assert result is out
    assert_default_epsilon(out, -tensor, scale=tensor)


def test_matmul_out_tuple() -> None:
    """Test ``np.matmul`` with NumPy's one-element ``out`` tuple."""
    a = arange_tensor(2, 3)
    b = arange_tensor(3, 4)
    out = TTD.zeros((2, 4), dtype=np.float64)

    result = np_matmul(TTD(a), TTD(b), out=(out,))

    assert result is out
    assert out.shape == (2, 4)
    assert_default_epsilon(out, np.matmul(a, b))


def test_add_dtype() -> None:
    """Test that ``dtype`` casts the result while preserving the TT form."""
    ttd = TTD(arange_tensor(2, 3))

    result = cast("TTD[np.float32]", np_add(ttd, ttd, dtype=np.float32))

    assert result.dtype == np.float32
    assert all(core.dtype == np.float32 for core in result.data)
    assert_default_epsilon(result, 2 * np.asarray(ttd), epsilon=1e-5)


def test_matmul_scalar_dtype() -> None:
    """Test that ``dtype`` casts a scalar ufunc result."""
    a = TTD(arange_tensor(3))
    b = TTD(arange_tensor(3))

    result = np_matmul(a, b, dtype=np.float32)

    assert np.ndim(result) == 0
    assert result.dtype == np.float32
    assert result == np.float32(np.matmul(np.asarray(a), np.asarray(b)))


def test_out_casts_dtype() -> None:
    """Test that a same-kind output dtype is cast instead of rejected."""
    ttd = TTD(arange_tensor(2, 3))
    out = TTD.zeros(ttd.shape, dtype=np.float32)

    result = np_add(ttd, ttd, out=(out,))

    assert result is out
    assert out.dtype == np.float32
    assert all(core.dtype == np.float32 for core in out.data)
    assert_default_epsilon(out, 2 * np.asarray(ttd), epsilon=1e-5)


def test_out_rejects_wrong_shape() -> None:
    """Test that a mismatched output shape is rejected without mutating it."""
    ttd = TTD(arange_tensor(2, 3))
    out: TTD[np.float64] = TTD.zeros((6,))

    with pytest.raises(ValueError, match="shape"):
        _ = np_add(ttd, ttd, out=(out,))

    np.testing.assert_array_equal(np.asarray(out), np.zeros(6))


def test_where_true_is_noop() -> None:
    """Test that ``where=True`` is accepted as a no-op."""
    ttd = TTD(arange_tensor(2, 3))

    result = np_add(ttd, ttd, where=True)

    assert_default_epsilon(result, 2 * np.asarray(ttd))


def test_where_false_rejected() -> None:
    """Test that a non-default ``where`` is rejected."""
    ttd = TTD(arange_tensor(2, 3))

    with pytest.raises(NotImplementedError, match="where"):
        _ = np_add(ttd, ttd, where=False)


@pytest.mark.parametrize(
    "kwargs",
    [{"casting": "unsafe"}, {"order": "C"}, {"subok": False}],
)
def test_unsupported_options_rejected(kwargs: dict[str, Any]) -> None:
    """Test that non-default ufunc options are rejected by name."""
    ttd = TTD(arange_tensor(2, 3))

    with pytest.raises(NotImplementedError, match=next(iter(kwargs))):
        _ = np_add(ttd, ttd, **kwargs)


def test_multiple_outputs_rejected() -> None:
    """Test that a multi-output ``out`` tuple is rejected."""
    a = TTD(arange_tensor(2, 3))
    b = TTD(arange_tensor(2, 3))
    out = TTD.zeros(a.shape)

    with pytest.raises(NotImplementedError, match="exactly one output"):
        _ = a.__array_ufunc__(np_add, "__call__", a, b, out=(out, out))


def test_signature_rejected() -> None:
    """Test that a non-default ``signature`` is rejected."""
    a = TTD(arange_tensor(2, 3))

    with pytest.raises(NotImplementedError, match="signature"):
        _ = a.__array_ufunc__(
            np_add,
            "__call__",
            a,
            a,
            signature=(np.float64, np.float64, np.float64),
        )
