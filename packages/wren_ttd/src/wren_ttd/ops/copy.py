# ruff: noqa: PLC0415, FBT001, FBT002
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("copy")
def copy[DType: np.floating](
    array: TTD[DType], order: str = "K", subok: bool = False
) -> TTD[DType]:
    """
    Return a copy of a TTD.

    Equivalent to :func:`numpy.copy` for dense arrays, extended to TTDs. The
    cores are copied, so the result shares no writable state with the input.

    Parameters
    ----------
    array : TTD[DType]
        The TTD to copy.
    order : {'C', 'F', 'A', 'K'}, optional
        Accepted for compatibility with :func:`numpy.copy`. TTDs do not
        expose a memory order, so this does not affect the result. By default
        ``'K'``.
    subok : bool, optional
        If True, preserve the class of `array`; otherwise return a plain
        :class:`~wren_ttd.core.TTD`, by default False.

    Returns
    -------
    TTD[DType]
        The copied TTD.

    Raises
    ------
    ValueError
        If `order` is not one of ``'C'``, ``'F'``, ``'A'``, or ``'K'``.

    See Also
    --------
    numpy.copy : Equivalent function for dense arrays.
    positive : Return a copy of a TTD.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 2))
    >>> ops.copy(t).shape
    (2, 2)

    """
    from wren_ttd.core import TTD

    if order not in ("C", "F", "A", "K"):
        raise ValueError(f"order must be one of 'C', 'F', 'A', or 'K', got {order!r}")

    cores = [core.copy() for core in array.data]
    if subok:
        return array.__class__(cores)
    return TTD(cores, dtype=array.dtype)
