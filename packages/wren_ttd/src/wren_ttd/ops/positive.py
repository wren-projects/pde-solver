from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._numpy_api import implements_ufunc

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_ufunc("positive")
def positive[DType: np.floating](a: TTD[DType]) -> TTD[DType]:
    """
    Return the TTD unchanged, element-wise.

    Equivalent to :func:`numpy.positive` for dense arrays, extended to TTDs.
    The result is a new TTD that shares no writable state with the input.

    Parameters
    ----------
    a : TTD[DType]
        The TTD to return.

    Returns
    -------
    TTD[DType]
        A copy of `a`.

    See Also
    --------
    numpy.positive : Equivalent ufunc for dense arrays.
    absolute : Compute the element-wise absolute value.
    negative : Negate a TTD element-wise.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD([1.0, 2.0])
    >>> np.asarray(ops.positive(t))
    array([1., 2.])

    """
    return a.copy()
