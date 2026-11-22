from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._numpy_api import implements_function, warn_dense

from .reshape import (
    _flatten_to_core,  # pyright: ignore[reportPrivateUsage]
    _validate_order,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("ravel")
def ravel[DType: np.floating](
    a: TTD[DType],
    order: str = "C",
) -> TTD[DType]:
    """
    Return a flattened one-dimensional TTD.

    Equivalent to :func:`numpy.ravel` for dense arrays, extended to TTDs. Since
    there is no notion of contiguous memory and views in TTDs, this function is
    equivalent to both :func:`numpy.ndarray.flatten` and :func:`numpy.reshape` with
    shape ``(-1)``.

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
    numpy.ravel : Equivalent function for dense arrays.
    reshape : Reshape a TTD to an arbitrary shape.
    flatten : Alias of :func:`ravel` for the TTD method.

    Notes
    -----
    While the tensor is not directly densified, the singular resulting core is
    by definition going to be equal in size to the dense tensor. A
    :class:`DenseWarning` is emitted to make this cost explicit. A
    one-dimensional input already has this representation, so it is returned
    unchanged and no warning is emitted.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 3))
    >>> ops.ravel(t).shape
    (6,)

    """
    _validate_order(order)
    if a.ndim == 1:
        return a.copy()

    warn_dense("Calling `ravel` effectively densifies the TTD. See Notes.")
    return _flatten_to_core(a)
