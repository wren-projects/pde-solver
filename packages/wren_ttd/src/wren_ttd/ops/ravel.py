from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._numpy_api import DenseWarning, implements_function

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
    :class:`DenseWarning` is emitted to make this cost explicit.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 3))
    >>> ops.ravel(t).shape
    (6,)

    """
    _validate_order(order)
    warnings.warn(
        "Calling `ravel` effectively densifies the TTD. See Notes.",
        DenseWarning,
        stacklevel=2,
    )
    return _flatten_to_core(a)
