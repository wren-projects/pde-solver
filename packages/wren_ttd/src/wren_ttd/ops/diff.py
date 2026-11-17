# ruff: noqa: PLC0415
from __future__ import annotations

from operator import index
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.array_utils import normalize_axis_index

from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("diff")
def diff[DType: np.floating](
    ttd: TTD[DType],
    n: int = 1,
    axis: int = -1,
    prepend: object | None = None,
    append: object | None = None,
) -> TTD[DType]:
    """
    Calculate the n-th discrete difference along an axis of a TTD.

    The `prepend` and `append` arguments are not supported.

    Parameters
    ----------
    ttd : TTD
        The TTD to take the difference of.
    n : int, default 1
        The number of times to take the difference.
    axis : int, default -1
        The axis to take the difference along.
    prepend : object, optional
        Not supported.
    append : object, optional
        Not supported.

    Returns
    -------
    TTD
        The differenced TTD.

    See Also
    --------
    numpy.diff: The equivalent function in NumPy.
    ops.gradient: The gradient of the differenced TTD.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> ttd = TTD(np.arange(10).reshape(2, 5), dtype=np.float64)
    >>> np.asarray(np.diff(ttd, n=1))
    array([[1., 1., 1., 1.],
           [1., 1., 1., 1.]])

    """
    from wren_ttd.core import TTD

    if prepend is not None or append is not None:
        raise NotImplementedError("prepend and append are not supported")

    n = index(n)
    if n < 0:
        raise ValueError("order must be non-negative")

    axis = normalize_axis_index(axis, ttd.ndim)
    cores = ttd.data.copy()
    cores[axis] = np.diff(cores[axis], n, axis=1)

    return TTD(cores, dtype=ttd.dtype)
