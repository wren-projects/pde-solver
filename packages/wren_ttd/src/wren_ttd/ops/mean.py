# ruff: noqa: PLC0415, FBT001, FBT002
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from wren_ttd._helpers import normalize_axes, resolve_dtype, write_out
from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .divide import divide
from .sum import sum as reduce_sum


@implements_function("mean")
def mean[DType: np.floating](  # noqa: PLR0913
    a: TTD[DType],
    axis: int | Iterable[int] | None = None,
    dtype: npt.DTypeLike | None = None,
    out: TTD[Any] | None = None,
    keepdims: bool = False,
    *,
    where: object = True,
) -> DType | TTD[Any]:
    """
    Arithmetic mean of TTD elements over the given axes.

    Equivalent to :func:`numpy.mean` for dense arrays, extended to TTDs. The
    mean is computed as a :func:`sum` followed by division by the product of
    the reduced axis sizes. The sum contracts cores exactly; for TTD results
    the division scales a single core. NumPy's dtype behaviour is preserved,
    so the mean of a ``float32`` TTD is ``float32`` unless `dtype` says
    otherwise.

    A TTD has no zero-dimensional representation. A full reduction with
    ``keepdims=False`` therefore returns a NumPy scalar, and with
    ``keepdims=True`` it returns a TTD of shape ``(1,) * a.ndim``.
    Reductions that keep at least one axis return a TTD; with
    ``keepdims=True`` the reduced axes are retained with size one.

    Parameters
    ----------
    a : TTD[DType]
        The TTD to average.
    axis : int | iterable of int | None, optional
        The axis or axes along which to compute the mean. Negative axes
        count from the end. ``None`` uses all axes, while ``()`` performs no
        reduction. By default None.
    dtype : np.dtype or type, optional
        The dtype of the result. Must be a floating-point dtype. If not
        provided, the dtype of `a` is used.
    out : TTD, optional
        The output TTD. Only TTD results can be written to `out`; a fully
        reduced mean without `keepdims` is scalar-valued. If provided, `out`
        must have the result shape and its dtype must be able to represent
        the result under NumPy's ``"same_kind"`` casting.
    keepdims : bool, optional
        Whether to keep the reduced axes as size-one axes, by default False.
    where : object, optional
        Not supported; only the default ``True`` is accepted.

    Returns
    -------
    DType | TTD
        The mean. A NumPy scalar for a full reduction without `keepdims`,
        and a TTD otherwise.

    Raises
    ------
    NotImplementedError
        If `where` is not the default, or if a scalar result is requested
        together with `out`.
    TypeError
        If `dtype` is not a floating-point dtype.
    ValueError
        If an axis is invalid or `out` is incompatible.

    See Also
    --------
    numpy.mean : Equivalent function for dense arrays.
    sum : Sum of a TTD over the given axes.
    divide : Divide a TTD by a scalar.

    Notes
    -----
    The reduction is performed by :func:`sum`, which accumulates in the
    result dtype. The final division is exact up to floating-point rounding.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd.core import TTD
    >>> t = TTD(np.arange(6.0).reshape(2, 3))
    >>> float(np.mean(t))
    2.5
    >>> np.asarray(np.mean(t, axis=0))
    array([1.5, 2.5, 3.5])

    """
    from wren_ttd.core import TTD

    if not (isinstance(where, (bool, np.bool_)) and where):
        raise NotImplementedError("The `where` argument is not supported for TTD means")

    axes = normalize_axes(axis, a.ndim)
    result_dtype = resolve_dtype(dtype, a)
    count = math.prod(a.shape[axis] for axis in axes)

    summed = reduce_sum(a, axis=axes, dtype=result_dtype, keepdims=keepdims)

    if not axes:
        result = summed
    elif isinstance(summed, TTD):
        result = divide(summed, count, dtype=result_dtype)
    else:
        result = cast(DType, result_dtype.type(np.divide(summed, count)))

    if isinstance(result, TTD):
        return write_out(result, out)

    if out is not None:
        raise NotImplementedError(
            "`out` is not supported for a fully reduced TTD mean; "
            "TTD has no 0-dimensional arrays"
        )

    return result
