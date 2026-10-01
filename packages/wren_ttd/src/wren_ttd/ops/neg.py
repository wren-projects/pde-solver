from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._numpy_api import implements_ufunc

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .multiply import multiply


@implements_ufunc("negative")
def neg[DType: np.floating](a: TTD[DType]) -> TTD[DType]:
    """
    Numerical negative, element-wise.

    Equivalent to :func:`numpy.negative` for dense arrays, extended to
    TTDs. This is a shorthand for multiplication by -1. See `multiply`
    for more details.

    Parameters
    ----------
    a : TTD[DType]
        The TTD object to negate.

    Returns
    -------
    TTD[DType]
        The negated TTD object, i.e. ``-a`` element-wise.

    See Also
    --------
    numpy.negative : Equivalent ufunc for dense arrays.
    multiply : Multiply TTDs element-wise.
    subtract : Subtract TTDs element-wise.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.ones((2, 2))
    >>> np.asarray(ops.neg(a))
    array([[-1., -1.],
           [-1., -1.]])

    """
    return multiply(a, -1.0)
