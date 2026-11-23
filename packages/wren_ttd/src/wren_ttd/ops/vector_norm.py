# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._helpers import as_scalar, normalize_axes, sqrt_reduced
from wren_ttd._numpy_api import linalg

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@linalg.function("vector_norm")
def vector_norm[DType: np.floating](
    a: TTD[DType],
    axis: int | Iterable[int] | None = None,
    *,
    keepdims: bool = False,
    ord: float | str | None = 2,  # noqa: A002
) -> DType | TTD[DType]:
    """
    Compute the Euclidean vector norm of a TTD.

    Equivalent to :func:`numpy.linalg.vector_norm` for dense arrays, extended
    to TTDs and restricted to the Euclidean norm. The norm is computed as

        ‖a‖₂ = √(sum(a * a, axis)).

    Parameters
    ----------
    a : TTD[DType]
        The input tensor.
    axis : int | Iterable[int] | None, optional
        Axis or axes along which to compute the norm. Negative axes count
        from the end. The default is ``None``, which reduces all axes.
    keepdims : bool, optional
        If ``True``, the reduced axes are kept as size-one dimensions. The
        default is ``False``.
    ord : float | str | None, optional
        The order of the norm. Only the Euclidean norm, ``None`` or ``2``, is
        supported. The default is ``2``.

    Returns
    -------
    DType | TTD[DType]
        The vector norm. A full reduction returns a NumPy scalar, while a
        partial reduction returns a TTD. With ``keepdims=True``, a full
        reduction returns a TTD of shape ``(1,) * a.ndim``.

    Raises
    ------
    NotImplementedError
        If `ord` is anything other than ``None`` or ``2``.

    See Also
    --------
    numpy.linalg.vector_norm : Equivalent function for dense arrays.
    matrix_norm : Frobenius norm over the last two axes.
    norm : Frobenius norm of a whole TTD.

    Notes
    -----
    The sum of squares stays in the TT representation. Since the element-wise
    square root is not multiplicative, the final square root of a TTD result
    uses a documented dense fallback: the reduced tensor is expanded with
    :func:`numpy.asarray`, square-rooted, and recompressed into a TTD. A
    :class:`DenseWarning` is emitted in that case.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.ones((2, 3))
    >>> np.asarray(ops.vector_norm(a, axis=0))
    array([1.41421356, 1.41421356, 1.41421356])
    >>> ops.vector_norm(TTD.ones(3))
    np.float64(1.7320508075688772)

    """
    from wren_ttd.core import TTD
    from wren_ttd.ops.expand_dims import expand_dims
    from wren_ttd.ops.multiply import multiply
    from wren_ttd.ops.sum import sum as ttd_sum

    if ord not in (None, 2, 2.0):
        raise NotImplementedError(
            "vector_norm supports only the Euclidean norm "
            f"(ord=None or 2), got ord={ord!r}"
        )

    axes = normalize_axes(axis, a.ndim)

    squared = ttd_sum(multiply(a, a), axis=axis)

    if keepdims:
        if isinstance(squared, TTD):
            squared = expand_dims(squared, axes)
        else:
            squared = TTD.full(
                (1,) * a.ndim,
                as_scalar(squared),
                dtype=a.dtype,
            )

    return sqrt_reduced(squared, "vector_norm")
