# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from wren_common.tests import arange_tensor
from wren_ttd import TTD, ops

from ..common import assert_default_epsilon


@pytest.mark.parametrize("shape", [(2, 3), (4, 2, 3)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_matrix_norm(
    shape: tuple[int, ...],
    keepdims: bool,  # noqa: FBT001
) -> None:
    """Compare the default Frobenius norm against NumPy."""
    tensor = arange_tensor(*shape)
    ttd = TTD(tensor)
    expected = np.linalg.matrix_norm(tensor, keepdims=keepdims)

    assert_default_epsilon(
        np.linalg.matrix_norm(ttd, keepdims=keepdims), expected, scale=tensor
    )
    assert_default_epsilon(
        ops.matrix_norm(ttd, keepdims=keepdims), expected, scale=tensor
    )
    if not keepdims and len(shape) == 2:
        assert isinstance(np.linalg.matrix_norm(ttd), np.floating)
        assert isinstance(ops.matrix_norm(ttd), np.floating)


@pytest.mark.parametrize("ord", ["fro", "f"])
def test_matrix_norm_explicit_ord(ord: str) -> None:  # noqa: A002
    """Accept both spellings of the Frobenius order."""
    tensor = arange_tensor(3, 4)
    ttd = TTD(tensor)
    numpy_ord: Any = ord

    assert_default_epsilon(
        np.linalg.matrix_norm(ttd, ord=numpy_ord),
        np.linalg.matrix_norm(tensor, ord=numpy_ord),
        scale=tensor,
    )
    assert_default_epsilon(
        ops.matrix_norm(ttd, ord=ord),
        np.linalg.matrix_norm(tensor, ord=numpy_ord),
        scale=tensor,
    )


def test_matrix_norm_uses_last_two_axes() -> None:
    """Batch over leading axes and reduce the last two, like NumPy."""
    tensor = arange_tensor(2, 3, 4)
    ttd = TTD(tensor)

    result = ops.matrix_norm(ttd)

    assert isinstance(result, TTD)
    assert result.shape == (2,)
    assert_default_epsilon(result, np.linalg.matrix_norm(tensor), scale=tensor)


def test_matrix_norm_keepdims_shape() -> None:
    """Keep the last two axes as size-one dimensions."""
    ttd = TTD(arange_tensor(2, 3, 4))

    result = ops.matrix_norm(ttd, keepdims=True)

    assert isinstance(result, TTD)
    assert result.shape == (2, 1, 1)


def test_matrix_norm_two_dimensional_keepdims() -> None:
    """Return a size-one matrix for a 2-D input with keepdims."""
    tensor = arange_tensor(3, 4)
    ttd = TTD(tensor)

    result = np.linalg.matrix_norm(ttd, keepdims=True)

    assert isinstance(result, TTD)
    assert result.shape == (1, 1)
    assert_default_epsilon(result, np.linalg.matrix_norm(tensor, keepdims=True))


@pytest.mark.parametrize("ord", [2, -2, 1, np.inf, None])
def test_matrix_norm_rejects_other_orders(ord: float | None) -> None:  # noqa: A002
    """Reject norms other than the Frobenius norm."""
    ttd = TTD.ones((3, 4))

    with pytest.raises(NotImplementedError, match="Frobenius"):
        _ = np.linalg.matrix_norm(ttd, ord=ord)

    with pytest.raises(NotImplementedError, match="Frobenius"):
        _ = ops.matrix_norm(ttd, ord=ord)


def test_matrix_norm_rejects_vectors() -> None:
    """Require at least two dimensions."""
    ttd = TTD(arange_tensor(4))

    with pytest.raises(ValueError, match="at least two"):
        _ = np.linalg.matrix_norm(ttd)

    with pytest.raises(ValueError, match="at least two"):
        _ = ops.matrix_norm(ttd)


def test_matrix_norm_float32_dtype() -> None:
    """Preserve float32 inputs, matching NumPy."""
    tensor = arange_tensor(3, 4).astype(np.float32)
    ttd = TTD(tensor)

    scalar = ops.matrix_norm(ttd)
    batched = ops.matrix_norm(TTD(arange_tensor(2, 3, 4).astype(np.float32)))

    assert scalar.dtype == np.float32
    assert isinstance(batched, TTD)
    assert batched.dtype == np.float32
