# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple
from wren_common.math import dot_product

from wren_ttd._helpers import contract_cores, reverse_cores
from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .transpose import transpose


@implements_function("tensordot")
def tensordot[DType: np.floating](
    a: TTD[DType],
    b: TTD[DType],
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> TTD[DType] | DType:
    """
    Compute a tensordot of two TTDs.

    Supports:
      * axes = int k: contracts cores a[-k:] with b[:k]
      * axes = (a_axis, b_axis): contracts cores a[a_axis] with b[b_axis]
      * axes = (a_axes, b_axes): contracts a_axes with b_axes

    Note: Axes may be specified by negative indices. In that case, they are
    counted from the end of the tensor.

    See :func:`numpy.tensordot` for more details about the axes argument.

    Returns
    -------
    TTD[DType] | DType
        TTD with uncontracted dimensions (first for a, than from b) or scalar.

    """
    from wren_ttd.core import TTD

    if a.dtype != b.dtype:
        raise ValueError("TTD objects must have the same dtype")

    dtype = a.dtype

    if isinstance(axes, int):
        k = axes

        if k == 0:
            return TTD([*a.data, *b.data], dtype=dtype)

        if k < 0:
            raise ValueError("axes must be non-negative")

        if k > a.ndim or k > b.ndim:
            raise ValueError("axes exceeds tensor dimensions")

        axes = (tuple(range(-k, 0)), tuple(range(k)))

    a_axes_raw, b_axes_raw = axes

    a_axes = normalize_axis_tuple(a_axes_raw, a.ndim, "a_axes")
    b_axes = normalize_axis_tuple(b_axes_raw, b.ndim, "b_axes")

    if len(a_axes) != len(b_axes):
        raise ValueError("a_axes and b_axes must have the same length")

    k = len(a_axes)

    for axis_a, axis_b in zip(a_axes, b_axes, strict=True):
        if a.shape[axis_a] != b.shape[axis_b]:
            raise ValueError("Shape mismatch on contracted axes")

    a_free = tuple(i for i in range(a.ndim) if i not in a_axes)
    b_free = tuple(i for i in range(b.ndim) if i not in b_axes)

    a_permutation = a_free + a_axes
    b_permutation = b_axes[::-1] + b_free

    a_t = transpose(a, a_permutation)
    b_t = transpose(b, b_permutation)

    return _tensordot_transposed(a_t, b_t, k, dtype=dtype)


def _tensordot_transposed[DType: np.floating](
    a: TTD[DType], b: TTD[DType], k: int, dtype: np.dtype[DType]
) -> TTD[DType] | DType:
    from wren_ttd.core import TTD

    assert a.ndim >= k, "k must be <= a.ndim"
    assert b.ndim >= k, "k must be <= b.ndim"

    a_free, a_contr = a.data[:-k], a.data[-k:]
    b_contr, b_free = b.data[:k], b.data[k:]

    message_matrix = contract_cores(reverse_cores(a_contr), b_contr, k)

    # complete contraction -> scalar
    if a.ndim == k and b.ndim == k:
        assert message_matrix.size == 1
        return dtype.type(message_matrix.squeeze())

    out_cores = a_free + b_free

    # multiply the message_matrix into either the first free b core or the last
    # free a core
    if b_free:
        out_cores[len(a_free)] = dot_product(message_matrix, b_free[0])
    else:
        out_cores[-1] = dot_product(a_free[-1], message_matrix)

    return TTD(out_cores, dtype=dtype)
