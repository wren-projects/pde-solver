# ruff: noqa: PLC0415
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._numpy_api import DenseWarning, implements_ufunc

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_ufunc("absolute")
def absolute[DType: np.floating](ttd: TTD[DType]) -> TTD[DType]:
    """
    Compute the absolute value element-wise.

    Equivalent to :func:`numpy.absolute` for dense arrays, extended to TTDs.
    The operation is exact and preserves the TT-ranks of the input.

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
    numpy.absolute : Equivalent ufunc for dense arrays.
    positive : Return the TTD unchanged.
    power : Raise a TTD to a scalar power.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD([-2.0, 3.0])
    >>> np.asarray(ops.absolute(t))
    array([2., 3.])

    """
    from wren_ttd.core import TTD

    warnings.warn("Calling absolute densifies the TTD.", DenseWarning, stacklevel=2)

    return TTD(np.abs(np.asarray(ttd)), dtype=ttd.dtype)
