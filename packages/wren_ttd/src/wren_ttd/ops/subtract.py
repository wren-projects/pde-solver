from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from wren_common.types import Scalar

from wren_ttd._numpy_api import implements_ufunc

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .add import add


@implements_ufunc("subtract")
def subtract[DType: np.floating](
    a: TTD[DType] | Scalar,
    b: TTD[DType] | Scalar,
    out: TTD[Any] | None = None,
    *,
    dtype: npt.DTypeLike | None = None,
) -> TTD[DType]:
    """
    Subtract arguments, element-wise.

    Equivalent to :func:`numpy.subtract` for dense arrays, extended to
    TTDs. This is a shorthand for addition with the second operand
    negated. See `add` and `neg` for more details.

    If one of the operands is a scalar, it is broadcasted to the shape of
    the other operand before the subtraction. The dtypes of the operands
    are promoted with NumPy's rules, exactly as in `add`.

    Parameters
    ----------
    a : TTD[DType] | Scalar
        The TTD or scalar to subtract from.
    b : TTD[DType] | Scalar
        The TTD or scalar to subtract. A TTD must have the same shape as
        `a` if `a` is a TTD.
    out : TTD[DType], optional
        The output TTD object. If not provided, a new TTD object is created.
        If provided, it must have the same shape as the result and its cores
        are replaced with the cores of the difference. The result is cast to
        the dtype of `out` if that is allowed by NumPy's ``"same_kind"``
        casting.
    dtype : np.dtype or type, optional
        The dtype of the result. If not provided, it is promoted from the
        operand dtypes. Must be a floating-point dtype.

    Returns
    -------
    TTD[DType]
        The element-wise difference of `a` and `b`.

    Raises
    ------
    ValueError
        If the operand shapes differ, or if `out` has an incompatible shape
        or dtype.

    See Also
    --------
    numpy.subtract : Equivalent ufunc for dense arrays.
    add : Add TTDs element-wise.
    neg : Negate a TTD element-wise.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.full((2, 2), 3.0)
    >>> np.asarray(ops.subtract(a, 1))
    array([[2., 2.],
           [2., 2.]])
    >>> ops.subtract(TTD.ones((2, 2), dtype=np.float32), np.float64(1)).dtype
    dtype('float64')

    """
    negative = -b
    return add(a, negative, out=out, dtype=dtype)
