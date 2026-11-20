# ruff: noqa: PLC0415
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np

from wren_ttd._numpy_api import DenseWarning, implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD


_MIN_MATRIX_DIMS = 2


def _as_scalar[DType: np.floating](value: np.ndarray | DType) -> DType:
    """Normalize a size-one array to a NumPy scalar."""
    return cast(DType, cast(object, np.asarray(value).reshape(())[()]))


def _sqrt[DType: np.floating](value: DType | TTD[DType]) -> DType | TTD[DType]:
    """Take the element-wise square root, recompressing TTD results."""
    from wren_ttd.core import TTD

    if isinstance(value, TTD):
        # ``sqrt`` is not multiplicative, so the square root is computed
        # densely and the result is recompressed.
        warnings.warn(
            "Calling matrix_norm with a TTD result densifies the reduced TTD.",
            DenseWarning,
            stacklevel=3,
        )
        return TTD(np.sqrt(np.asarray(value)), dtype=value.dtype)

    # The current full reduction returns a size-one array; normalize it to a
    # NumPy scalar to match ``numpy.linalg.matrix_norm``.
    return cast(DType, np.sqrt(_as_scalar(value)))


@implements_function("linalg.matrix_norm")
def matrix_norm[DType: np.floating](
    a: TTD[DType],
    *,
    keepdims: bool = False,
    ord: str | float | None = "fro",  # noqa: A002
) -> DType | TTD[DType]:
    """
    Compute the Frobenius norm of a TTD over its last two axes.

    Equivalent to :func:`numpy.linalg.matrix_norm` for dense arrays, extended
    to TTDs and restricted to the Frobenius norm. For a tensor with at least
    two dimensions, the norm is computed as

        ‖a‖ᶠ = √(sum(a * a, axis=(-2, -1))).

    Leading axes, if any, are treated as a batch: each matrix in the batch
    gets its own norm.

    Parameters
    ----------
    a : TTD[DType]
        The input tensor. Must have at least two dimensions.
    keepdims : bool, optional
        If ``True``, the last two axes are kept as size-one dimensions. The
        default is ``False``.
    ord : str | float | None, optional
        The order of the norm. Only the Frobenius norm, ``"fro"`` or ``"f"``,
        is supported. The default is ``"fro"``.

    Returns
    -------
    DType | TTD[DType]
        The Frobenius norm. A two-dimensional input reduces to a scalar,
        while higher-dimensional inputs return a TTD over the leading axes.
        With ``keepdims=True``, the last two axes are size one.

    Raises
    ------
    ValueError
        If `a` has fewer than two dimensions.
    NotImplementedError
        If `ord` is not ``"fro"`` or ``"f"``.

    See Also
    --------
    numpy.linalg.matrix_norm : Equivalent function for dense arrays.
    vector_norm : Euclidean norm along arbitrary axes.
    frobenius_norm : Frobenius norm of a whole TTD.

    Notes
    -----
    The sum of squares stays in the TT representation. Since the element-wise
    square root is not multiplicative, the final square root of a TTD result
    uses a documented dense fallback: the reduced tensor is expanded with
    :func:`numpy.asarray`, square-rooted, and recompressed into a TTD. A
    :class:`DenseWarning` is emitted in that case.

    Only the Frobenius norm is supported. Other orders, including nuclear and
    spectral norms, raise :exc:`NotImplementedError`.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.ones((2, 3))
    >>> ops.matrix_norm(a)
    np.float64(2.449489742783178)
    >>> np.asarray(ops.matrix_norm(TTD.ones((2, 2, 3)))).shape
    (2,)

    """
    from wren_ttd.core import TTD
    from wren_ttd.ops.expand_dims import expand_dims
    from wren_ttd.ops.multiply import multiply
    from wren_ttd.ops.sum import sum as ttd_sum

    if a.ndim < _MIN_MATRIX_DIMS:
        raise ValueError(
            f"matrix_norm requires an input with at least two dimensions, got {a.ndim}"
        )

    if ord not in ("fro", "f"):
        raise NotImplementedError(
            f"matrix_norm supports only the Frobenius norm (ord='fro' or 'f'), "
            f"got ord={ord!r}"
        )

    axes = (a.ndim - 2, a.ndim - 1)

    squared = ttd_sum(multiply(a, a), axis=axes)

    if keepdims:
        if isinstance(squared, TTD):
            squared = expand_dims(squared, axes)
        else:
            squared = TTD.full((1, 1), _as_scalar(squared), dtype=a.dtype)

    return _sqrt(squared)
