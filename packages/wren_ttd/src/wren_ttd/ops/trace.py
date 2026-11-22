# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from numpy.lib.array_utils import normalize_axis_index
from wren_common.types import Matrix

from wren_ttd._helpers import mode_3_product, normalize_dtype, validate_out
from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .moveaxis import moveaxis


@implements_function("trace")
def trace[DType: np.floating](  # noqa: PLR0913
    a: TTD[DType],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: npt.DTypeLike | None = None,
    out: TTD[Any] | None = None,
) -> DType | TTD[Any]:
    """
    Trace of a TTD along two axes.

    Equivalent to :func:`numpy.trace` with ``offset=0`` for dense arrays,
    extended to TTDs. The two selected axes are contracted with a diagonal
    tensor, which removes them from the result.

    If `axis1` and `axis2` are the last two axes (in either order), the two
    last cores are contracted directly with
    ``numpy.einsum("lim,mir->lr", ...)``, which is exact. Otherwise, the
    axes are moved to the end with :func:`moveaxis`; that permutation uses
    TTD-SVD truncation, so the result may be approximate.

    Parameters
    ----------
    a : TTD[DType]
        The TTD to trace.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Only the default
        offset of zero is supported. By default 0.
    axis1 : int, optional
        First axis of the 2-D sub-arrays whose diagonals are taken. Negative
        axes count from the end. By default 0.
    axis2 : int, optional
        Second axis of the 2-D sub-arrays. Must differ from `axis1` and have
        the same size. By default 1.
    dtype : np.dtype or type, optional
        The dtype of the result. Must be a floating-point dtype. If not
        provided, the dtype of `a` is used.
    out : TTD, optional
        The output TTD. Only TTD results can be written to `out`; a 2-D
        input produces a scalar. If provided, `out` must have the result
        shape and its dtype must be able to represent the result under
        NumPy's ``"same_kind"`` casting.

    Returns
    -------
    DType | TTD
        The trace. A NumPy scalar for a 2-D input, and a TTD with the two
        traced axes removed otherwise.

    Raises
    ------
    NotImplementedError
        If `offset` is nonzero, or if a scalar result is requested together
        with `out`.
    ValueError
        If `axis1` and `axis2` are equal or their sizes differ, or if `out`
        is incompatible.
    TypeError
        If `dtype` is not a floating-point dtype.
    numpy.exceptions.AxisError
        If an axis is out of range.

    See Also
    --------
    numpy.trace : Equivalent function for dense arrays.
    sum : Sum of a TTD over the given axes.
    moveaxis : Move TTD axes to new positions.

    Notes
    -----
    Moving non-final axes to the end applies :func:`moveaxis`, which
    permutes cores with TTD-SVD truncation at the default tolerance. The
    contraction of the two resulting final cores is exact.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd.core import TTD
    >>> t = TTD(np.arange(9.0).reshape(3, 3))
    >>> float(np.trace(t))
    12.0
    >>> t3 = TTD(np.arange(18.0).reshape(3, 3, 2))
    >>> np.trace(t3).shape
    (2,)

    """
    from wren_ttd.core import TTD

    if offset != 0:
        raise NotImplementedError(
            "The `offset` argument is not supported for TTD traces; "
            "only the main diagonal (offset=0) is implemented"
        )

    normalized_axis1 = normalize_axis_index(axis1, a.ndim, "axis1")
    normalized_axis2 = normalize_axis_index(axis2, a.ndim, "axis2")

    if normalized_axis1 == normalized_axis2:
        raise ValueError("axis1 and axis2 cannot be the same")

    if a.shape[normalized_axis1] != a.shape[normalized_axis2]:
        raise ValueError(
            "axis1 and axis2 must have the same size, got "
            f"{a.shape[normalized_axis1]} and {a.shape[normalized_axis2]}"
        )

    result_dtype = normalize_dtype(dtype) if dtype is not None else a.dtype

    moved = (
        a
        if {normalized_axis1, normalized_axis2} == {a.ndim - 2, a.ndim - 1}
        else moveaxis(a, (normalized_axis1, normalized_axis2), (-2, -1))
    )

    cores = list(moved.data)
    left = cores[-2].astype(result_dtype, copy=False)
    right = cores[-1].astype(result_dtype, copy=False)
    diagonal = cast(Matrix[DType], np.tensordot(left, right, axes=((1, 2), (1, 0))))

    remaining = cores[:-2]
    if not remaining:
        scalar = cast(DType, result_dtype.type(diagonal.squeeze()[()]))
        if out is not None:
            raise NotImplementedError(
                "`out` is not supported for a 2-D TTD trace; "
                "TTD has no 0-dimensional arrays"
            )
        return scalar

    remaining[-1] = mode_3_product(
        remaining[-1].astype(result_dtype, copy=False), diagonal
    )
    result = TTD(remaining, dtype=result_dtype)

    if out is not None:
        out_dtype = validate_out(out, result.shape, result_dtype)
        out.data = [core.astype(out_dtype, copy=False) for core in result.data]
        return out

    return result
