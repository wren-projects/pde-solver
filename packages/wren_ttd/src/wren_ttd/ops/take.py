# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from numpy.lib.array_utils import normalize_axis_index
from wren_common.types import Index1D

from wren_ttd._helpers import validate_out
from wren_ttd._numpy_api import implements_function
from wren_ttd.ops.ravel import ravel
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("take")
def take[DType: np.floating](
    a: TTD[DType],
    indices: int | npt.NDArray[np.integer[Any]] | Sequence[int],
    axis: int | None = None,
    out: TTD[DType] | None = None,
    mode: str = "raise",
) -> TTD[DType] | DType:
    """
    Take elements from a TTD along an axis.

    Equivalent to :func:`numpy.take` for dense arrays, extended to TTDs.

    For a core ``(l, n, r)`` on the selected axis, this selects
    ``core[:, indices, :]``. The selection is exact and preserves the
    TT-ranks of the input. A scalar index drops the selected axis, exactly
    like NumPy.

    Parameters
    ----------
    a : TTD[DType]
        The TTD to take elements from.
    indices : int | array of int | sequence of int
        The indices to take. Negative indices count from the end of the
        selected axis. Array-like indices may have any shape and are
        flattened by NumPy's indexing rules.
    axis : int | None, optional
        The axis over which to take the indices. May be negative. If
        ``None``, the input is flattened first, by default ``None``.
    out : TTD[DType], optional
        The output TTD object. If not provided, a new TTD object is created.
        If provided, it must have the same shape as the result and its cores
        are replaced with the cores of the result. The result is cast to the
        dtype of `out` if that is allowed by NumPy's ``"same_kind"`` casting.
    mode : str, optional
        Only ``"raise"`` is supported, by default ``"raise"``. The
        ``"wrap"`` and ``"clip"`` modes are rejected with
        :class:`NotImplementedError`.

    Returns
    -------
    TTD[DType] | DType
        The selected elements as a TTD, or a NumPy scalar if `indices` is a
        scalar and every axis is consumed.

    Raises
    ------
    IndexError
        If an index is out of bounds for the selected axis.
    ValueError
        If `axis` is out of range, or if `out` has an incompatible shape or
        dtype.
    NotImplementedError
        If `mode` is not ``"raise"``, or if `out` is requested for a scalar
        result, since TTD has no 0-D representation.

    See Also
    --------
    numpy.take : Equivalent function for dense arrays.
    TTD.__getitem__ : Index into a TTD with integers and slices.
    repeat : Repeat elements of a TTD.
    tile : Construct a TTD by repeating it.

    Notes
    -----
    With ``axis=None`` the input is flattened with :func:`ravel` first. Since
    :func:`ravel` effectively densifies the TTD, a :class:`DenseWarning` is
    emitted in that case.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.ones((2, 3))
    >>> ops.take(a, [2, 0], axis=1).shape
    (2, 2)
    >>> ops.take(a, 0, axis=0).shape
    (3,)
    >>> float(ops.take(TTD([1.0, 2.0]), 0))
    1.0
    >>> ops.take(a, [0, 4]).shape
    (2,)

    """
    from wren_ttd.core import TTD

    if mode != "raise":
        raise NotImplementedError(f"take only supports mode='raise', not {mode!r}")

    if axis is None:
        return take(ravel(a), indices, axis=0, out=out, mode=mode)

    axis = normalize_axis_index(axis, a.ndim)
    index_array = np.asarray(indices)
    if index_array.size == 0:
        index_array = index_array.astype(np.intp)
    elif not np.issubdtype(index_array.dtype, np.integer):
        raise TypeError(f"take indices must be integers, not dtype {index_array.dtype}")

    result = (
        _take_scalar(a, int(index_array), axis)
        if index_array.ndim == 0
        else _take_array(a, index_array, axis)
    )

    if out is None:
        return result

    if not isinstance(result, TTD):
        raise NotImplementedError(
            "out= is not supported for scalar take results; "
            "TTD has no 0-D representation"
        )

    out_dtype = validate_out(out, result.shape, result.dtype)
    out.data = [core.astype(out_dtype, copy=False) for core in result.data]
    return out


def _take_scalar[DType: np.floating](
    a: TTD[DType], index: int, axis: int
) -> TTD[DType] | DType:
    """Take a single index, dropping the selected axis."""
    from wren_ttd.ops.get_item import get_item

    size = a.shape[axis]
    normalized = index + size if index < 0 else index
    if not 0 <= normalized < size:
        raise IndexError(
            f"index {index} is out of bounds for axis {axis} with size {size}"
        )

    indexes: tuple[Index1D, ...] = (*((slice(None),) * axis), normalized)
    return get_item(a, indexes)


def _take_array[DType: np.floating](
    a: TTD[DType], indices: npt.NDArray[np.integer[Any]], axis: int
) -> TTD[DType]:
    """Take an integer array of indices from a single core."""
    from wren_ttd.core import TTD

    size = a.shape[axis]
    normalized = np.where(indices < 0, indices + size, indices)
    if np.any((normalized < 0) | (normalized >= size)):
        raise IndexError(
            f"take indices are out of bounds for axis {axis} with size {size}"
        )

    cores = a.data.copy()
    cores[axis] = cast(Core[DType], cores[axis][:, normalized, :])
    return TTD(cores, dtype=a.dtype)
