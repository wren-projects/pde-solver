from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._numpy_api import implements_function, linalg

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .swapaxes import swapaxes

_MIN_MATRIX_NDIM = 2


@implements_function("matrix_transpose")
@linalg.function("matrix_transpose")
def matrix_transpose[DType: np.floating](ttd: TTD[DType]) -> TTD[DType]:
    """
    Transpose the final two dimensions of a TTD.

    Equivalent to :func:`numpy.matrix_transpose` and to
    :func:`numpy.linalg.matrix_transpose` for dense arrays, extended to TTDs.
    Both NumPy entry points are registered for this implementation.

    Parameters
    ----------
    ttd : TTD[DType]
        Input TTD tensor with at least two dimensions.

    Returns
    -------
    TTD[DType]
        TTD tensor with the final two axes swapped.

    Raises
    ------
    ValueError
        If `ttd` has fewer than two dimensions.

    See Also
    --------
    numpy.matrix_transpose : Equivalent function for dense arrays.
    numpy.linalg.matrix_transpose : Equivalent function in `numpy.linalg`.
    TTD.mT : Transpose the final two axes of a TTD.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 3, 4))
    >>> ops.matrix_transpose(t).shape
    (2, 4, 3)

    """
    if ttd.ndim < _MIN_MATRIX_NDIM:
        raise ValueError("input array must be at least 2-d")
    return swapaxes(ttd, -1, -2)
