# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from numpy.lib.array_utils import normalize_axis_index

from wren_ttd._helpers import normalize_dtype, validate_out
from wren_ttd._numpy_api import implements_function
from wren_ttd.ops.ravel import ravel
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("cumsum")
def cumsum[DType: np.floating](
    a: TTD[DType],
    axis: int | None = None,
    dtype: npt.DTypeLike | None = None,
    out: TTD[Any] | None = None,
) -> TTD[Any]:
    """
    Cumulative sum of TTD elements along the given axis.

    Equivalent to :func:`numpy.cumsum` for dense arrays, extended to TTDs.
    For an integer `axis`, the cumulative sum is applied to the mode axis of
    each core. This is exact and preserves the TT ranks: for a core ``G`` of
    shape ``(l, n, r)``, the result satisfies
    ``G'[:, i, :] = sum_{j <= i} G[:, j, :]``.

    Parameters
    ----------
    a : TTD[DType]
        The TTD to accumulate.
    axis : int | None, optional
        The axis along which to accumulate. Negative axes count from the
        end. If None, the TTD is flattened in C order, accumulated, and
        returned as a 1-D TTD. By default None.
    dtype : np.dtype or type, optional
        The dtype of the result. Must be a floating-point dtype. If not
        provided, the dtype of `a` is used.
    out : TTD, optional
        The output TTD. If provided, it must have the result shape and its
        dtype must be able to represent the result under NumPy's
        ``"same_kind"`` casting.

    Returns
    -------
    TTD
        The cumulative sum. For an integer `axis`, it has the same shape as
        `a`.

    Raises
    ------
    TypeError
        If `dtype` is not a floating-point dtype.
    ValueError
        If `axis` is invalid or `out` is incompatible.

    See Also
    --------
    numpy.cumsum : Equivalent function for dense arrays.
    sum : Sum of a TTD over the given axes.
    mean : Arithmetic mean of a TTD.

    Notes
    -----
    The integer-axis path is exact and preserves the TT ranks. The
    ``axis=None`` path first calls :func:`ravel`, which flattens the TTD and
    therefore effectively densifies it; a :class:`DenseWarning` is emitted in
    that case.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd.core import TTD
    >>> t = TTD(np.arange(6.0).reshape(2, 3))
    >>> np.round(np.asarray(np.cumsum(t, axis=1)), 12)
    array([[ 0.,  1.,  3.],
           [ 3.,  7., 12.]])
    >>> np.round(np.asarray(np.cumsum(t)), 12)
    array([ 0.,  1.,  3.,  6., 10., 15.])

    """
    from wren_ttd.core import TTD

    result_dtype = normalize_dtype(dtype) if dtype is not None else a.dtype

    if axis is None:
        return cumsum(ravel(a), axis=0, dtype=result_dtype, out=out)

    normalized_axis = normalize_axis_index(axis, a.ndim, "axis")
    cores = a.data.copy()
    cores[normalized_axis] = cast(
        Core[Any],
        np.cumsum(cores[normalized_axis].astype(result_dtype, copy=False), axis=1),
    )
    result = TTD(cores, dtype=result_dtype)

    if out is not None:
        out_dtype = validate_out(out, result.shape, result_dtype)
        out.data = [core.astype(out_dtype, copy=False) for core in result.data]
        return out

    return result
