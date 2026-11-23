# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._numpy_api import implements_function, implements_ufunc, warn_dense

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_ufunc("absolute")
@implements_function("abs")
def abs[DType: np.floating](  # noqa: A001
    ttd: TTD[DType],
) -> TTD[DType]:
    """
    Compute the absolute value element-wise.

    Equivalent to :func:`numpy.abs` for dense arrays, extended to TTDs.
    The absolute value is not multiplicative, so it cannot be applied to a
    TTD core by core. The tensor is expanded to a dense array, transformed,
    and compressed again.

    Parameters
    ----------
    ttd : TTD[DType]
        The TTD to take the absolute value of.

    Returns
    -------
    TTD[DType]
        The element-wise absolute value.

    See Also
    --------
    numpy.abs : Equivalent ufunc for dense arrays.
    positive : Return the TTD unchanged.
    pow : Raise a TTD to a scalar power.

    Notes
    -----
    This operation densifies the tensor internally and emits a
    :class:`DenseWarning`. It is intended for small tensors and for API
    compatibility, not for large TT computations.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD([-2.0, 3.0])
    >>> np.asarray(ops.abs(t))
    array([2., 3.])

    """
    from wren_ttd.core import TTD

    warn_dense("Calling abs densifies the TTD.")

    return TTD(np.abs(np.asarray(ttd)), dtype=ttd.dtype)
