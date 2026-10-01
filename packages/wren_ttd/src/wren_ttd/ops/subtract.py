from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from wren_common.types import Scalar

from wren_ttd._numpy_api import implements_ufunc

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .add import add


@implements_ufunc("subtract")
def subtract[DType: np.floating](
    a: TTD[DType] | Scalar, b: TTD[DType] | Scalar, out: TTD[DType] | None = None
) -> TTD[DType]:
    """
    Subtract arguments, element-wise.

    Equivalent to :func:`numpy.subtract` for dense arrays, extended to
    TTDs. This is a shorthand for addition with the second operand
    negated. See `add` and `neg` for more details.

    If one of the operands is a scalar, it is broadcasted to the shape of
    the other operand before the subtraction.

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
        are replaced with the cores of the difference.

    Returns
    -------
    TTD[DType]
        The element-wise difference of `a` and `b`.

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

    """
    negative = cast("TTD[DType] | Scalar", cast(object, np.negative(b)))
    return add(a, negative, out=out)
