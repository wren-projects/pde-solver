from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .ravel import ravel

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def flatten[DType: np.floating](
    a: TTD[DType],
    order: str = "C",
) -> TTD[DType]:
    """
    Return a flattened one-dimensional TTD.

    Equivalent to :func:`numpy.ndarray.flatten` for dense arrays, extended to
    TTDs. There is no top-level :func:`numpy.flatten`, so this function is not
    registered with the NumPy function dispatch; it is exposed through
    :mod:`wren_ttd.ops` and the ``TTD.flatten`` method.


    Since there is no notion of contiguous memory and views in TTDs, this
    function is equivalent to both :func:`numpy.ravel` and :func:`numpy.reshape`
    with shape ``(-1)``.

    Parameters
    ----------
    a : TTD[DType]
        Input TTD tensor.
    order : str, optional
        The order in which to read the elements. Only ``"C"`` is supported;
        ``"F"``, ``"A"``, and ``"K"`` raise :exc:`NotImplementedError`.

    Returns
    -------
    TTD[DType]
        The flattened TTD of shape ``(a.size,)`` with the same dtype.

    Raises
    ------
    NotImplementedError
        If `order` is not ``"C"``.

    See Also
    --------
    numpy.ndarray.flatten : Equivalent method for dense arrays.
    ravel : Registered NumPy function with the same semantics.
    reshape : Reshape a TTD to an arbitrary shape.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 3))
    >>> ops.flatten(t).shape
    (6,)

    """
    return ravel(a, order=order)
