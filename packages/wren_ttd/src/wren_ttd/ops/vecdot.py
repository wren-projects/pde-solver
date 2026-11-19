# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.lib.array_utils import normalize_axis_index

from wren_ttd._numpy_api import implements_function, implements_ufunc

if TYPE_CHECKING:
    from wren_ttd.core import TTD


# NumPy implements ``vecdot`` as a ufunc, so both protocols must be
# registered for ``np.vecdot`` and ``ops.vecdot`` to reach this handler.
@implements_function("vecdot")
@implements_ufunc("vecdot")
def vecdot[DType: np.floating](
    a: TTD[DType], b: TTD[DType], axis: int = -1
) -> DType | TTD[DType]:
    """
    Compute the (vector) dot product of two TTDs along an axis.

    Equivalent to :func:`numpy.vecdot` for dense arrays, extended to TTDs.
    The product is computed as

        vecdot(a, b) = sum(a * b, axis),

    which is exact: the element-wise product and the reduction both stay in
    the TT representation.

    Broadcasting is not supported. The operands must have exactly the same
    shape, so ``vecdot`` of two vectors is a scalar, and ``vecdot`` of two
    matrices is a vector of row-wise dot products.

    Parameters
    ----------
    a : TTD[DType]
        The first operand.
    b : TTD[DType]
        The second operand. Must have the same shape as `a`.
    axis : int, optional
        Axis along which to compute the dot product. Negative axes count from
        the end. The default is the last axis.

    Returns
    -------
    DType | TTD[DType]
        The dot product. Reducing a one-dimensional tensor returns a scalar;
        otherwise the reduced axis is removed and a TTD is returned.

    Raises
    ------
    NotImplementedError
        If the operand shapes differ, since broadcasting is not supported.

    See Also
    --------
    numpy.vecdot : Equivalent function for dense arrays.
    numpy.sum : Sum along an axis.
    numpy.multiply : Element-wise product.

    Notes
    -----
    The TT-ranks of the element-wise product are the products of the operand
    ranks, so the intermediate may need truncation for large ranks.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.ones((2, 3))
    >>> b = TTD.full((2, 3), 2.0)
    >>> np.asarray(ops.vecdot(a, b))
    array([6., 6.])
    >>> ops.vecdot(TTD.ones(3), TTD.ones(3))
    np.float64(3.0)

    """
    from wren_ttd.core import TTD
    from wren_ttd.ops.multiply import multiply
    from wren_ttd.ops.sum import sum as ttd_sum

    if a.shape != b.shape:
        raise NotImplementedError(
            "vecdot supports only operands with equal shapes; "
            f"broadcasting is not supported (got {a.shape} and {b.shape})"
        )

    result = ttd_sum(multiply(a, b), axis=normalize_axis_index(axis, a.ndim))

    if isinstance(result, TTD):
        return result

    # The current full reduction returns a size-one array; normalize it to a
    # NumPy scalar to match ``numpy.vecdot``.
    return cast(DType, cast(object, np.asarray(result).reshape(())[()]))
