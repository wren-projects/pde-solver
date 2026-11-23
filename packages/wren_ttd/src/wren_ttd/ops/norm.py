from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._numpy_api import linalg
from wren_ttd.ops.vector_norm import vector_norm

if TYPE_CHECKING:
    from wren_ttd.core import TTD


_SUPPORTED_ORDERS = (None, 2, "fro", "f")


@linalg.function("norm")
def norm[DType: np.floating](
    ttd: TTD[DType],
    ord: float | int | str | None = None,  # noqa: A002
    axis: int | Iterable[int] | None = None,
    *,
    keepdims: bool = False,
) -> DType | TTD[DType]:
    """
    Compute the Euclidean or Frobenius norm of a TTD.

    Implements :func:`numpy.linalg.norm` for TTDs, restricted to the
    Euclidean and Frobenius norms. For an integer `axis`, the result is the
    Euclidean norm along that axis. For a pair of axes, or for the default
    ``axis=None``, the result is the Frobenius norm over those axes.

    Parameters
    ----------
    ttd : TTD[DType]
        The input tensor.
    ord : float | int | str | None, optional
        The order of the norm. Only the Euclidean or Frobenius norm is
        supported, so `ord` must be ``None``, ``2``, ``"fro"``, or ``"f"``.
        By default ``None``.
    axis : int | Iterable[int] | None, optional
        Axis or axes along which to compute the norm. If ``None``, all axes
        are reduced. By default ``None``.
    keepdims : bool, optional
        If ``True``, the reduced axes are kept as size-one dimensions. By
        default ``False``.

    Returns
    -------
    DType | TTD[DType]
        The norm. A full reduction returns a NumPy scalar, while a partial
        reduction or ``keepdims=True`` returns a TTD.

    Raises
    ------
    NotImplementedError
        If `ord` is not ``None``, ``2``, ``"fro"``, or ``"f"``.

    See Also
    --------
    numpy.linalg.norm : Equivalent function for dense arrays.
    vector_norm : Euclidean norm along arbitrary axes.
    matrix_norm : Frobenius norm over the last two axes.
    vdot : Inner product of two TTDs.

    Notes
    -----
    This function is a NumPy compatibility helper. The current Array API
    standard does not define ``linalg.norm``; use :func:`vector_norm` or
    :func:`matrix_norm` instead.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 2))
    >>> ops.norm(t)
    np.float64(2.0)
    >>> ops.norm(t, axis=0).shape
    (2,)

    """
    if ord not in _SUPPORTED_ORDERS:
        raise NotImplementedError(
            "TTD norms support only the Euclidean and Frobenius orders"
        )

    return vector_norm(ttd, axis=axis, keepdims=keepdims, ord=2)
