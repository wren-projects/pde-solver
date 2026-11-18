# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from wren_common.types import Scalar

from wren_ttd._numpy_api import implements_ufunc

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def _integer_exponent(exponent: Scalar) -> int | None:
    """Return `exponent` as an integer, or None if it is not integer-like."""
    value = np.asarray(exponent)
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.number):
        return None
    if np.issubdtype(value.dtype, np.integer):
        return int(value)
    if np.issubdtype(value.dtype, np.floating) and float(value).is_integer():
        return int(value)
    return None


@implements_ufunc("power")
def power[DType: np.floating](a: TTD[DType], b: Scalar) -> TTD[DType]:
    """
    Raise a TTD to a scalar power, element-wise.

    Equivalent to :func:`numpy.power` for dense arrays, extended to TTDs.
    Only integer-like scalar exponents are supported. Non-negative powers are
    computed as repeated Hadamard products, which stay in TT format but may
    inflate the ranks. Negative powers fall back to a dense computation and
    recompress the result.

    Parameters
    ----------
    a : TTD[DType]
        The TTD base.
    b : scalar
        The exponent. Must be integer-like.

    Returns
    -------
    TTD[DType]
        The element-wise power.

    Raises
    ------
    NotImplementedError
        If `b` is a TTD or a non-integer scalar.

    See Also
    --------
    numpy.power : Equivalent ufunc for dense arrays.
    multiply : Multiply TTDs element-wise.

    Notes
    -----
    Negative exponents densify the tensor internally. Non-negative exponents
    do not densify, but the ranks grow with the exponent.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD([1.0, 2.0, 3.0])
    >>> np.asarray(ops.power(t, 2))
    array([1., 4., 9.])

    """
    from wren_ttd.core import TTD

    exponent = _integer_exponent(b)
    if exponent is None:
        raise NotImplementedError(
            "TTD exponentiation supports only integer-like scalar exponents"
        )

    dtype = np.dtype(np.result_type(a.dtype, b))

    if exponent == 0:
        return TTD.ones(a.shape, dtype=dtype)

    if exponent > 0:
        from .multiply import multiply

        result = a
        for _ in range(exponent - 1):
            result = multiply(result, a)
        return TTD(result.data, dtype=dtype)

    return TTD(np.power(np.asarray(a), exponent), dtype=dtype)
