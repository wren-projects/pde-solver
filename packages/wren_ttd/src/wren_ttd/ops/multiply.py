# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt
from wren_common.types import Scalar, ScalarTypes

from wren_ttd._helpers import (
    normalize_dtype,
    promote_dtype,
    smallest_core,
    validate_out,
)
from wren_ttd._numpy_api import implements_ufunc
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def _scale_cores[DType: np.floating](
    ttd: TTD[DType], scalar: Scalar, dtype: np.dtype[Any]
) -> list[Core[DType]]:
    """Scale the smallest core of a TTD by a scalar, in `dtype`."""
    cores = [core.astype(dtype, copy=False) for core in ttd.data]

    core, index = smallest_core(cores)

    cores[index] = np.multiply(core, scalar)

    return cores


def _hadamard_cores[DType: np.floating](
    a: TTD[DType], b: TTD[DType], dtype: np.dtype[Any]
) -> list[Core[DType]]:
    """Compute the cores of the Hadamard product of two same-shaped TTDs."""
    if a.shape != b.shape:
        raise ValueError("Tensors must have the same shape.")

    N = np.newaxis

    new_cores: list[Core[DType]] = []
    for core_a, core_b in zip(a.data, b.data, strict=True):
        cast_a = core_a.astype(dtype, copy=False)
        cast_b = core_b.astype(dtype, copy=False)
        la, n, ra = cast_a.shape
        lb, _, rb = cast_b.shape

        # Expand dimensions to leverage standard NumPy broadcasting:
        # core_a expanded: (la,  1, n, ra,  1)
        # core_b expanded: ( 1, lb, n,  1, rb)
        # Resulting shape: (la, lb, n, ra, rb)
        # then multiply and flatten back
        core = np.multiply(
            cast_a[:, N, :, :, N],
            cast_b[N, :, :, N, :],
        ).reshape(la * lb, n, ra * rb)

        new_cores.append(core)

    return new_cores


@overload
def multiply[DType: np.floating](
    a: TTD[DType],
    b: Scalar,
    out: TTD[Any] | None = None,
    *,
    dtype: npt.DTypeLike | None = None,
) -> TTD[DType]: ...


@overload
def multiply[DType: np.floating](
    a: Scalar,
    b: TTD[DType],
    out: TTD[Any] | None = None,
    *,
    dtype: npt.DTypeLike | None = None,
) -> TTD[DType]: ...


@overload
def multiply[DType: np.floating](
    a: TTD[DType],
    b: TTD[DType],
    out: TTD[Any] | None = None,
    *,
    dtype: npt.DTypeLike | None = None,
) -> TTD[DType]: ...


@implements_ufunc("multiply")
def multiply[DType: np.floating](
    a: TTD[DType] | Scalar,
    b: TTD[DType] | Scalar,
    out: TTD[Any] | None = None,
    *,
    dtype: npt.DTypeLike | None = None,
) -> TTD[DType]:
    """
    Multiply arguments element-wise.

    Equivalent to :func:`numpy.multiply` for dense arrays, extended to TTDs.

    For a TTD object A = G₀ ⊗ G₁ ⊗ ... ⊗ Gₙ, the multiplication by a scalar k is defined
    as

        kA = G₀ ⊗ G₁ ⊗ … ⊗ kGᵢ ⊗ … ⊗ Gₙ,

    where the choice of i is arbitrary from 0 to n. For performance reasons, we choose
    the smallest core.

    Multiplication by another TTD object is implemented as the Hadamard (element-wise)
    product defined as

        A ⊙ B = (G₀ ⊙ H₀) ⊗ (G₁ ⊙ H₁) ⊗ … ⊗ (Gₙ ⊙ Hₙ),

    where Gᵣ ⊙ Hᵣ = (Gᵣ Hᵣ).

    The multiplication of two TTD objects requires that they have the same
    shape. Their dtypes are promoted with NumPy's rules, so multiplying a
    ``float32`` and a ``float64`` tensor yields a ``float64`` tensor. If one of
    the operands is a scalar, it scales the other operand as described above.
    Python scalars follow NumPy's weak promotion, while NumPy scalars do not.

    Parameters
    ----------
    a : TTD[DType] | Scalar
        The first factor. A scalar scales `b`.
    b : TTD[DType] | Scalar
        The second factor. A scalar scales `a`.
    out : TTD[DType], optional
        The output TTD object. If not provided, a new TTD object is created.
        If provided, it must have the same shape as the result and its cores
        are replaced with the cores of the product. The result is cast to the
        dtype of `out` if that is allowed by NumPy's ``"same_kind"`` casting.
    dtype : np.dtype or type, optional
        The dtype of the result. If not provided, it is promoted from the
        operand dtypes. Must be a floating-point dtype.

    Returns
    -------
    TTD[DType]
        The element-wise product of `a` and `b`.

    Raises
    ------
    ValueError
        If the operand shapes differ, or if `out` has an incompatible shape
        or dtype.

    See Also
    --------
    numpy.multiply : Equivalent ufunc for dense arrays.
    add : Add TTDs element-wise.
    neg : Negate a TTD element-wise.

    Notes
    -----
    The TT-ranks of a Hadamard product are the products of the operand
    ranks, so repeated multiplication inflates the ranks. Consider calling
    :meth:`TTD.round` on the result before performing further operations.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.ones((2, 2))
    >>> b = TTD.full((2, 2), 2.0)
    >>> np.asarray(ops.multiply(a, b))
    array([[2., 2.],
           [2., 2.]])
    >>> np.asarray(ops.multiply(a, 3))
    array([[3., 3.],
           [3., 3.]])
    >>> ops.multiply(TTD.ones((2, 2), dtype=np.float32), 2, dtype=np.float64).dtype
    dtype('float64')

    """
    from wren_ttd.core import TTD

    result_dtype = (
        normalize_dtype(dtype)
        if dtype is not None
        else promote_dtype(
            a.dtype if isinstance(a, TTD) else a,
            b.dtype if isinstance(b, TTD) else b,
        )
    )

    if isinstance(a, TTD) and isinstance(b, ScalarTypes):
        shape = a.shape
        cores = _scale_cores(a, b, result_dtype)
    elif isinstance(b, TTD) and isinstance(a, ScalarTypes):
        shape = b.shape
        cores = _scale_cores(b, a, result_dtype)
    elif isinstance(a, TTD) and isinstance(b, TTD):
        shape = a.shape
        cores = _hadamard_cores(a, b, result_dtype)
    else:
        return NotImplemented

    if out is None:
        return TTD(cores, dtype=result_dtype)

    out_dtype = validate_out(out, shape, result_dtype)
    out.data = [core.astype(out_dtype, copy=False) for core in cores]
    return out
