from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("linalg.norm")
def frobenius_norm[DType: np.floating](ttd: TTD[DType]) -> DType:
    """
    Return the Frobenius norm of a TTD.

    Implements :func:`numpy.linalg.norm` for TTDs, restricted to the
    default Frobenius norm. The Frobenius norm of a TTD object is defined
    as the square root of its inner product with itself:

        ‖A‖ᶠ = √(⟨A, A⟩)

    Parameters
    ----------
    ttd : TTD[DType]
        The TTD object to compute the norm of.

    Returns
    -------
    DType
        The Frobenius norm of the TTD object.

    See Also
    --------
    numpy.linalg.norm : Equivalent function for dense arrays.
    inner_product : Inner product of two TTDs.

    Notes
    -----
    Only the Frobenius norm is supported; the ``ord`` and ``axis``
    arguments of :func:`numpy.linalg.norm` are not accepted.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 2))
    >>> ops.frobenius_norm(t)
    np.float64(2.0)

    """
    return cast(DType, np.sqrt(np.vdot(ttd, ttd)))
