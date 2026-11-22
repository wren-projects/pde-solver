# ruff: noqa: PLC0415, FBT001, FBT002
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from numpy.lib.array_utils import normalize_axis_tuple
from wren_common.types import Matrix, Scalar

from wren_ttd._helpers import (
    mode_1_product,
    mode_3_product,
    normalize_dtype,
    validate_out,
)
from wren_ttd._numpy_api import implements_function
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .expand_dims import expand_dims


def _sum_cores(
    ttd: TTD[Any], axes: tuple[int, ...], dtype: np.dtype[Any]
) -> list[Core[Any]] | np.floating[Any]:
    """Contract the reduced cores into their neighbours, or compute the scalar."""
    cores: list[Core[Any]] = [core.astype(dtype, copy=not axes) for core in ttd.data]

    for axis in sorted(axes, reverse=True):
        if len(cores) == 1:
            return cast(np.floating[Any], np.sum(cores[0], axis=1)[0, 0])

        summed = cast(Matrix[Any], np.sum(cores[axis], axis=1))

        if axis == 0:
            cores[1] = mode_1_product(summed, cores[1])
        else:
            cores[axis - 1] = mode_3_product(cores[axis - 1], summed)

        _ = cores.pop(axis)

    return cores


def _write_out(out: TTD[Any], result: TTD[Any], dtype: np.dtype[Any]) -> TTD[Any]:
    """Validate `out` and replace its cores with those of `result`."""
    out_dtype = validate_out(out, result.shape, dtype)
    out.data = [core.astype(out_dtype, copy=False) for core in result.data]
    return out


@implements_function("sum")
def sum[DType: np.floating](  # noqa: A001, PLR0913
    a: TTD[DType],
    axis: int | Iterable[int] | None = None,
    dtype: npt.DTypeLike | None = None,
    out: TTD[Any] | None = None,
    keepdims: bool = False,
    *,
    initial: object = None,
    where: object = True,
) -> DType | TTD[Any]:
    """
    Sum of TTD elements over the given axes.

    Equivalent to :func:`numpy.sum` for dense arrays, extended to TTDs. The
    reduced cores are contracted into their neighbours with
    :func:`mode_1_product` and :func:`mode_3_product`, which is exact.

    A TTD has no zero-dimensional representation. A full reduction with
    ``keepdims=False`` therefore returns a NumPy scalar of the result dtype,
    and with ``keepdims=True`` it returns a TTD of shape
    ``(1,) * a.ndim``. Reductions that keep at least one axis return a TTD;
    with ``keepdims=True`` the reduced axes are retained with size one.

    Parameters
    ----------
    a : TTD[DType]
        The TTD to sum.
    axis : int | iterable of int | None, optional
        The axis or axes along which to sum. Negative axes count from the
        end. ``None`` sums over all axes, while ``()`` performs no
        reduction. By default None.
    dtype : np.dtype or type, optional
        The dtype of the result. Must be a floating-point dtype. If not
        provided, the dtype of `a` is used.
    out : TTD, optional
        The output TTD. Only TTD results can be written to `out`; a fully
        reduced sum without `keepdims` is scalar-valued. If provided, `out`
        must have the result shape and its dtype must be able to represent
        the result under NumPy's ``"same_kind"`` casting.
    keepdims : bool, optional
        Whether to keep the reduced axes as size-one axes, by default False.
    initial : object, optional
        Not supported; must be None.
    where : object, optional
        Not supported; only the default ``True`` is accepted.

    Returns
    -------
    DType | TTD
        The sum. A NumPy scalar for a full reduction without `keepdims`, and
        a TTD otherwise.

    Raises
    ------
    NotImplementedError
        If `initial` is given, if `where` is not the default, or if a scalar
        result is requested together with `out`.
    TypeError
        If `dtype` is not a floating-point dtype.
    ValueError
        If an axis is invalid or `out` is incompatible.

    See Also
    --------
    numpy.sum : Equivalent function for dense arrays.
    mean : Arithmetic mean of a TTD.
    cumsum : Cumulative sum of a TTD.
    trace : Trace along two axes of a TTD.

    Notes
    -----
    The dtype of the cores is ``dtype`` if provided, so the accumulation
    happens in that dtype, matching NumPy's ``dtype`` semantics.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd.core import TTD
    >>> t = TTD(np.arange(6.0).reshape(2, 3))
    >>> float(np.sum(t))
    15.0
    >>> np.asarray(np.sum(t, axis=0))
    array([3., 5., 7.])
    >>> np.asarray(np.sum(t, axis=0, keepdims=True))
    array([[3., 5., 7.]])

    """
    from wren_ttd.core import TTD

    if initial is not None:
        raise NotImplementedError(
            "The `initial` argument is not supported for TTD sums"
        )

    if not (isinstance(where, (bool, np.bool_)) and where):
        raise NotImplementedError("The `where` argument is not supported for TTD sums")

    axes = (
        tuple(range(a.ndim))
        if axis is None
        else normalize_axis_tuple(axis, a.ndim, "axis")
    )
    result_dtype = normalize_dtype(dtype) if dtype is not None else a.dtype

    result = _sum_cores(a, axes, result_dtype)

    if isinstance(result, list):
        reduced = TTD(result, dtype=result_dtype)
        if keepdims and axes:
            reduced = expand_dims(reduced, axes)
        if out is not None:
            return _write_out(out, reduced, result_dtype)
        return reduced

    assert result.size == 1
    scalar = cast(DType, result_dtype.type(result))

    if keepdims:
        shape = (1,) * a.ndim
        expanded = TTD.full(shape, cast(Scalar, scalar), dtype=result_dtype)
        if out is not None:
            return _write_out(out, expanded, result_dtype)
        return expanded

    if out is not None:
        raise NotImplementedError(
            "`out` is not supported for a fully reduced TTD sum; "
            "TTD has no 0-dimensional arrays"
        )

    return scalar
