# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import numpy as np
from wren_common.types import Scalar, ScalarTypes

from wren_ttd._helpers import smallest_core
from wren_ttd._numpy_api import implements_ufunc
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def _hadamard_impl[DType: np.complexfloating](
    a: TTD[DType], b: TTD[DType], out: TTD[DType] | None = None
) -> TTD[DType]:
    """Compute the Hadamard (element-wise) product of two same-shaped TTDs."""
    from wren_ttd.core import TTD

    if a.shape != b.shape:
        raise ValueError(
            f"Cannot multiply TTDs with different shapes: {a.shape} and {b.shape}."
        )

    dtype = np.result_type(a.dtype, b.dtype)

    N = np.newaxis

    new_cores: list[Core[DType]] = []
    for core_a, core_b in zip(a.data, b.data, strict=True):
        la, n, ra = core_a.shape
        lb, _, rb = core_b.shape

        # Expand dimensions to leverage standard NumPy broadcasting:
        # core_a expanded: (la,  1, n, ra,  1)
        # core_b expanded: ( 1, lb, n,  1, rb)
        # Resulting shape: (la, lb, n, ra, rb)
        # then multiply and flatten back
        core = np.multiply(
            core_a[:, N, :, :, N],
            core_b[N, :, :, N, :],
        ).reshape(la * lb, n, ra * rb)

        new_cores.append(core)

    if out is not None:
        if out.shape != a.shape:
            raise ValueError(
                f"Output shape mismatch: got {out.shape}, expected {a.shape}."
            )
        if not np.can_cast(dtype, out.dtype, casting="same_kind"):
            raise TypeError(
                f"Cannot cast output from dtype '{dtype}' to dtype '{out.dtype}'"
                " with casting rule 'same_kind'."
            )
        out.data = (
            [cast(Core[DType], np.asarray(core, dtype=out.dtype)) for core in new_cores]
            if out.dtype != dtype
            else new_cores
        )
        return out

    return TTD(new_cores, dtype=dtype)


@overload
def multiply[DType: np.complexfloating](
    a: TTD[DType], b: Scalar, out: TTD[DType] | None = None
) -> TTD[DType]: ...


@overload
def multiply[DType: np.complexfloating](
    a: Scalar, b: TTD[DType], out: TTD[DType] | None = None
) -> TTD[DType]: ...


@overload
def multiply[DType: np.complexfloating](
    a: TTD[DType], b: TTD[DType], out: TTD[DType] | None = None
) -> TTD[DType]: ...


@implements_ufunc("multiply")
def multiply[DType: np.complexfloating](
    a: TTD[DType] | Scalar, b: TTD[DType] | Scalar, out: TTD[DType] | None = None
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
    shape. If one of the operands is a scalar, it scales the other operand
    as described above.

    Parameters
    ----------
    a : TTD[DType] | Scalar
        The first factor. A scalar scales `b`.
    b : TTD[DType] | Scalar
        The second factor. A scalar scales `a`.
    out : TTD[DType], optional
        The output TTD object. If not provided, a new TTD object is created.
        If provided, it must have the same shape as the result and a dtype
        to which the result can be cast with 'same_kind' casting (like the
        `out` argument of NumPy ufuncs); its cores are replaced with the
        cores of the product.

    Returns
    -------
    TTD[DType]
        The element-wise product of `a` and `b`.

    See Also
    --------
    numpy.multiply : Equivalent ufunc for dense arrays.
    add : Add TTDs element-wise.
    neg : Negate a TTD element-wise.

    Notes
    -----
    The result dtype follows :func:`numpy.result_type`, like NumPy.
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

    """
    from wren_ttd.core import TTD

    def scalar_impl(
        ttd: TTD[DType], scalar: Scalar, out: TTD[DType] | None = None
    ) -> TTD[DType]:
        cores = ttd.data.copy()

        core, index = smallest_core(cores)

        cores[index] = np.multiply(core, scalar)
        dtype = np.result_type(ttd.dtype, scalar)

        if out is None:
            return TTD(cores, dtype=dtype)

        if out.shape != ttd.shape:
            raise ValueError(
                f"Output shape mismatch: got {out.shape}, expected {ttd.shape}."
            )

        if not np.can_cast(dtype, out.dtype, casting="same_kind"):
            raise TypeError(
                f"Cannot cast output from dtype '{dtype}' to dtype '{out.dtype}'"
                " with casting rule 'same_kind'."
            )

        out.data = (
            [cast(Core[DType], np.asarray(core, dtype=out.dtype)) for core in cores]
            if out.dtype != dtype
            else cores
        )
        return out

    if isinstance(a, TTD) and isinstance(b, ScalarTypes):
        return scalar_impl(a, b, out=out)

    if isinstance(b, TTD) and isinstance(a, ScalarTypes):
        return scalar_impl(b, a, out=out)

    if isinstance(a, TTD) and isinstance(b, TTD):
        return _hadamard_impl(a, b, out=out)

    return NotImplemented
