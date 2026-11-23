# ruff: noqa: PLC0415
from __future__ import annotations

from operator import index
from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._numpy_api import implements_function, warn_dense

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("round")
@implements_function("around")
def round_[DType: np.floating](
    a: TTD[DType], decimals: int = 0, out: TTD[DType] | tuple[TTD[DType]] | None = None
) -> TTD[DType]:
    """
    Round a TTD element-wise to the given number of decimals.

    Equivalent to :func:`numpy.round` for dense arrays, extended to TTDs.
    Rounding is not multiplicative, so it cannot be applied independently to
    each core. This implementation expands the TTD, rounds the dense tensor,
    and compresses the result again.

    Parameters
    ----------
    a : TTD[DType]
        The TTD to round.
    decimals : int, optional
        Number of decimal places to round to, by default 0.
    out : TTD or tuple of TTD, optional
        Optional output TTD. A one-element tuple is accepted for NumPy
        compatibility.

    Returns
    -------
    TTD[DType]
        The rounded TTD.

    Raises
    ------
    TypeError
        If `decimals` is not an integer.
    ValueError
        If `out` has a different shape or dtype than the result.

    See Also
    --------
    numpy.round : Equivalent function for dense arrays.
    TTD.round : Truncate the TT ranks (legacy name, not element-wise rounding).
    TTD.truncate : Truncate the TT ranks.

    Notes
    -----
    This operation densifies the tensor internally and emits a
    :class:`DenseWarning`. It is intended for small tensors and for API
    compatibility, not for large TT computations.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd.core import TTD
    >>> t = TTD([1.234, 2.345])
    >>> np.asarray(np.round(t, 1))
    array([1.2, 2.3])

    """
    from wren_ttd.core import TTD

    decimals = index(decimals)
    warn_dense("Calling round densifies the TTD.")
    result = TTD(np.round(np.asarray(a), decimals), dtype=a.dtype)

    if out is None:
        return result

    target = out[0] if isinstance(out, tuple) else out
    if target.shape != result.shape:
        raise ValueError("Output tensor has an incorrect shape.")
    if target.dtype != result.dtype:
        raise ValueError("Output tensor has an incorrect dtype.")

    target.data = list(result.data)
    return target
