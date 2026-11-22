# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from wren_common.types import Scalar, ScalarTypes

from wren_ttd._helpers import normalize_dtype, promote_dtype
from wren_ttd._numpy_api import implements_ufunc

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .multiply import multiply


@implements_ufunc("divide")
def divide[DType: np.floating](
    a: TTD[DType] | Scalar,
    b: TTD[DType] | Scalar,
    out: TTD[Any] | None = None,
    *,
    dtype: npt.DTypeLike | None = None,
) -> TTD[DType]:
    """
    Divide a TTD by a scalar while preserving its TT representation.

    The result dtype is promoted from the TTD and the scalar with NumPy's
    rules, so a ``float32`` TTD divided by a ``numpy.float64`` scalar yields a
    ``float64`` TTD, while dividing by a Python float keeps it ``float32``.

    Parameters
    ----------
    a : TTD[DType] | Scalar
        The dividend. Must be a TTD.
    b : TTD[DType] | Scalar
        The divisor. Only scalars are supported.
    out : TTD[DType], optional
        The output TTD object. If not provided, a new TTD object is created.
        If provided, it must have the same shape as the result and its cores
        are replaced with the cores of the quotient. The result is cast to the
        dtype of `out` if that is allowed by NumPy's ``"same_kind"`` casting.
    dtype : np.dtype or type, optional
        The dtype of the result. If not provided, it is promoted from the
        operand dtypes. Must be a floating-point dtype.

    Returns
    -------
    TTD[DType]
        The element-wise quotient of `a` and `b`.

    Raises
    ------
    NotImplementedError
        If the divisor is not a scalar.
    ValueError
        If `out` has an incompatible shape or dtype.

    See Also
    --------
    numpy.divide : Equivalent ufunc for dense arrays.
    multiply : Multiply TTDs element-wise.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.full((2, 2), 2.0)
    >>> np.asarray(ops.divide(a, 2))
    array([[1., 1.],
           [1., 1.]])
    >>> ops.divide(TTD.ones((2, 2), dtype=np.float32), np.float64(2)).dtype
    dtype('float64')

    """
    from wren_ttd.core import TTD

    if not isinstance(a, TTD) or not isinstance(b, ScalarTypes):
        raise NotImplementedError("TTD division is only supported by a scalar divisor")

    result_dtype = (
        normalize_dtype(dtype) if dtype is not None else promote_dtype(a.dtype, b)
    )

    reciprocal = cast(Scalar, np.divide(cast(Scalar, result_dtype.type(1)), b))
    return multiply(a, reciprocal, out=out, dtype=result_dtype)
