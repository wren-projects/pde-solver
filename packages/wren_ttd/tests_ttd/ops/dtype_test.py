# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

from copy import deepcopy
from typing import cast

import numpy as np
import pytest
from wren_common.tests import TestTensor, arange_tensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon

FLOAT32 = np.dtype(np.float32)
FLOAT64 = np.dtype(np.float64)

# float32 TTD compression is only accurate to ~1e-7 relative to the dense tensor
FLOAT32_EPSILON = 1e-5


def test_add_promotes_mixed_dtypes() -> None:
    """Promote a float32/float64 sum to float64."""
    tensor = arange_tensor(3, 4)
    tensor32 = tensor.astype(np.float32)

    result = ops.add(TTD(tensor32), TTD(tensor))

    assert result.dtype == FLOAT64
    assert_default_epsilon(
        result, tensor32 + tensor, scale=tensor, epsilon=FLOAT32_EPSILON
    )


def test_subtract_promotes_mixed_dtypes() -> None:
    """Promote a float32/float64 difference to float64."""
    tensor = arange_tensor(3, 4)
    tensor32 = tensor.astype(np.float32)
    larger = 3 * tensor

    result = ops.subtract(TTD(tensor32), TTD(larger))

    assert result.dtype == FLOAT64
    assert_default_epsilon(
        result, tensor32 - larger, scale=larger, epsilon=FLOAT32_EPSILON
    )


def test_multiply_promotes_mixed_dtypes() -> None:
    """Promote a float32/float64 product to float64."""
    tensor = arange_tensor(3, 4)
    tensor32 = tensor.astype(np.float32)

    result = ops.multiply(TTD(tensor32), TTD(tensor))

    assert result.dtype == FLOAT64
    assert_default_epsilon(
        result, tensor32 * tensor, scale=tensor, epsilon=FLOAT32_EPSILON
    )


def test_matmul_promotes_mixed_dtypes() -> None:
    """Accept float32/float64 operands in a TTD matrix product."""
    tensor_a = arange_tensor(2, 3).astype(np.float32)
    tensor_b = arange_tensor(3, 2)

    result = ops.matmul(TTD(tensor_a), TTD(tensor_b))

    assert isinstance(result, TTD)
    result = cast("TTD[np.float64]", result)
    assert result.dtype == FLOAT64
    assert_default_epsilon(
        result,
        tensor_a.astype(np.float64) @ tensor_b,
        scale=tensor_b,
        epsilon=FLOAT32_EPSILON,
    )


def test_matmul_returns_promoted_scalar() -> None:
    """Cast a vector-vector product to the promoted dtype."""
    tensor_a = arange_tensor(3).astype(np.float32)
    tensor_b = arange_tensor(3)

    result = ops.matmul(TTD(tensor_a), TTD(tensor_b))

    assert isinstance(result, np.float64)
    assert_default_epsilon(
        result,
        tensor_a.astype(np.float64) @ tensor_b,
        scale=tensor_b,
        epsilon=FLOAT32_EPSILON,
    )


def test_tensordot_promotes_mixed_dtypes() -> None:
    """Accept float32/float64 operands in a TTD tensor contraction."""
    tensor_a = arange_tensor(2, 3).astype(np.float32)
    tensor_b = arange_tensor(3, 2)

    result = ops.tensordot(TTD(tensor_a), TTD(tensor_b), axes=1)

    assert isinstance(result, TTD)
    result = cast("TTD[np.float64]", result)
    assert result.dtype == FLOAT64
    assert_default_epsilon(
        result,
        np.tensordot(tensor_a.astype(np.float64), tensor_b, axes=1),
        scale=tensor_b,
        epsilon=FLOAT32_EPSILON,
    )


def test_add_scalar_promotes_dtype() -> None:
    """Distinguish strong NumPy scalars from weak Python scalars."""
    tensor = arange_tensor(3, 4).astype(np.float32)
    ttd = TTD(tensor)

    promoted = ops.add(ttd, np.float64(1))
    assert promoted.dtype == FLOAT64
    assert_default_epsilon(
        promoted, tensor + np.float64(1), scale=tensor, epsilon=FLOAT32_EPSILON
    )

    weak = ops.add(ttd, 1.0)
    assert weak.dtype == FLOAT32
    assert_default_epsilon(weak, tensor + 1.0, scale=tensor, epsilon=FLOAT32_EPSILON)


def test_subtract_scalar_promotes_dtype() -> None:
    """Inherit the scalar promotion semantics from addition."""
    tensor = arange_tensor(3, 4).astype(np.float32)
    ttd = TTD(tensor)

    promoted = ops.subtract(ttd, np.float64(2.5))
    assert promoted.dtype == FLOAT64
    assert_default_epsilon(
        promoted, tensor - np.float64(2.5), scale=tensor, epsilon=FLOAT32_EPSILON
    )

    weak = ops.subtract(ttd, 2.5)
    assert weak.dtype == FLOAT32
    assert_default_epsilon(weak, tensor - 2.5, scale=tensor, epsilon=FLOAT32_EPSILON)


def test_multiply_scalar_promotes_dtype() -> None:
    """Distinguish strong NumPy scalars from weak Python scalars."""
    tensor = arange_tensor(3, 4).astype(np.float32)
    ttd = TTD(tensor)

    promoted = ops.multiply(ttd, np.float64(2))
    assert promoted.dtype == FLOAT64
    assert_default_epsilon(
        promoted, tensor * np.float64(2), scale=tensor, epsilon=FLOAT32_EPSILON
    )

    weak = ops.multiply(ttd, 2.0)
    assert weak.dtype == FLOAT32
    assert_default_epsilon(weak, tensor * 2.0, scale=tensor, epsilon=FLOAT32_EPSILON)


def test_divide_scalar_promotes_dtype() -> None:
    """Promote the quotient to the dtype of the scalar divisor."""
    tensor = arange_tensor(3, 4).astype(np.float32)
    ttd = TTD(tensor)

    promoted = ops.divide(ttd, np.float64(2))
    assert promoted.dtype == FLOAT64
    assert_default_epsilon(
        promoted, tensor / np.float64(2), scale=tensor, epsilon=FLOAT32_EPSILON
    )

    weak = ops.divide(ttd, 2)
    assert weak.dtype == FLOAT32
    assert_default_epsilon(weak, tensor / 2, scale=tensor, epsilon=FLOAT32_EPSILON)


def test_add_explicit_dtype() -> None:
    """Cast the result of a TTD-TTD sum to an explicit dtype."""
    tensor = arange_tensor(3, 4)

    result = ops.add(TTD(tensor), TTD(tensor), dtype=np.float32)

    assert result.dtype == FLOAT32
    assert_default_epsilon(
        result, tensor + tensor, scale=tensor, epsilon=FLOAT32_EPSILON
    )


def test_multiply_explicit_dtype() -> None:
    """Cast the result of a scalar multiplication to an explicit dtype."""
    tensor = arange_tensor(3, 4)

    result = ops.multiply(TTD(tensor), 2, dtype=np.float32)

    assert result.dtype == FLOAT32
    assert_default_epsilon(result, tensor * 2, scale=tensor, epsilon=FLOAT32_EPSILON)


def test_matmul_explicit_dtype() -> None:
    """Cast the result of a TTD matrix product to an explicit dtype."""
    tensor_a = arange_tensor(2, 3)
    tensor_b = arange_tensor(3, 2)

    result = ops.matmul(TTD(tensor_a), TTD(tensor_b), dtype=np.float32)

    assert isinstance(result, TTD)
    assert result.dtype == FLOAT32
    assert_default_epsilon(
        result,
        tensor_a @ tensor_b,
        scale=tensor_b,
        epsilon=FLOAT32_EPSILON,
    )


def test_add_explicit_dtype_must_be_floating() -> None:
    """Reject non-floating result dtypes."""
    ttd = TTD.ones((2, 3))

    with pytest.raises(TypeError, match="not a valid dtype"):
        _ = ops.add(ttd, ttd, dtype=np.int32)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_add_out_matches_numpy(tensor: TestTensor, ttd: TestTTD) -> None:
    """Write the sum into a compatible output TTD."""
    out = TTD.zeros(tensor.shape)

    result = ops.add(ttd, ttd, out=out)

    assert result is out
    assert_default_epsilon(out, tensor + tensor, scale=tensor)


def test_add_out_casts_to_output_dtype() -> None:
    """Allow same-kind narrowing when writing to a float32 output."""
    tensor = arange_tensor(3, 4)

    out = TTD.zeros(tensor.shape, dtype=np.float32)
    result = ops.add(TTD(tensor), TTD(tensor), out=out)

    assert result is out
    assert out.dtype == FLOAT32
    assert_default_epsilon(out, tensor + tensor, scale=tensor, epsilon=FLOAT32_EPSILON)


def test_add_out_rejects_incorrect_shape() -> None:
    """Reject an output TTD with the wrong shape without mutating it."""
    tensor = arange_tensor(3, 4)
    out = TTD.zeros((4, 3))

    with pytest.raises(ValueError, match="shape"):
        _ = ops.add(TTD(tensor), TTD(tensor), out=out)

    assert_default_epsilon(out, np.zeros((4, 3)))


def test_add_out_rejects_incompatible_dtype() -> None:
    """Reject an output dtype that cannot hold the result."""
    tensor = arange_tensor(3, 4)
    out = TTD.zeros(tensor.shape, dtype=np.float64)
    cores = [core.copy() for core in out.data]
    # TTDs are always floating-point, so simulate an incompatible output dtype.
    out.dtype = np.dtype(np.int32)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="same_kind"):
        _ = ops.add(TTD(tensor), TTD(tensor), out=out)

    assert all(
        np.array_equal(core, previous)
        for core, previous in zip(out.data, cores, strict=True)
    )
