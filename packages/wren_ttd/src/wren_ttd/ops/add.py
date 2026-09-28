# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from wren_common.types import Scalar, ScalarTypes

from wren_ttd._helpers import block_core
from wren_ttd._numpy_api import implements_ufunc
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_ufunc("add")
def add[DType: np.complexfloating](
    a: TTD[DType] | Scalar,
    b: TTD[DType] | Scalar,
    *,
    out: TTD[DType] | None = None,
) -> TTD[DType]:
    """
    Add arguments element-wise.

    Equivalent to :func:`numpy.add` for dense arrays, extended to TTDs.

    For two TTD objects A = G₀ ⊗ G₁ ⊗ … ⊗ Gₙ and B = H₀ ⊗ H₁ ⊗ … ⊗ Hₙ, the
    addition is defined as

        A + B = (G₀ H₀) ⊗ (G₁ 0 ; 0 H₁) ⊗ (G₂ 0 ; 0 H₂) ⊗ … ⊗ (Gₙ ; Hₙ).

    The addition requires that the TTD objects have the same shape. Operand
    dtypes are promoted following :func:`numpy.result_type`, like NumPy.
    If one of the operands is a scalar, it is broadcasted to the shape of
    the other operand and then added. That is equivalent to adding the scalar to
    each element of the other operand.

    Parameters
    ----------
    a : TTD[DType] | Scalar
        The first summand. A scalar is broadcast to the shape of `b`.
    b : TTD[DType] | Scalar
        The second summand. A scalar is broadcast to the shape of `a`.
    out : TTD[DType], optional
        The output TTD object. If not provided, a new TTD object is created.
        If provided, it must have the same shape as the result and a dtype
        to which the result can be cast with 'same_kind' casting (like the
        `out` argument of NumPy ufuncs); its cores are replaced with the
        cores of the sum.

    Returns
    -------
    TTD[DType]
        The element-wise sum of `a` and `b`.

    See Also
    --------
    numpy.add : Equivalent ufunc for dense arrays.
    subtract : Subtract TTDs element-wise.
    multiply : Multiply TTDs element-wise.

    Notes
    -----
    The TT-ranks of the sum are the sums of the operand ranks, so repeated
    addition inflates the ranks. Consider calling :meth:`TTD.round` on the
    result before performing further operations.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.ones((2, 2))
    >>> b = TTD.full((2, 2), 2.0)
    >>> np.asarray(ops.add(a, b))
    array([[3., 3.],
           [3., 3.]])
    >>> np.asarray(ops.add(a, 1))
    array([[2., 2.],
           [2., 2.]])

    """
    # the import has to be here to avoid circular imports
    from wren_ttd.core import TTD

    def normalize_operands(
        a: TTD[DType] | Scalar, b: TTD[DType] | Scalar
    ) -> tuple[TTD[DType], TTD[DType]]:

        if isinstance(a, TTD) and isinstance(b, TTD):
            return a, b

        if isinstance(a, TTD) and isinstance(b, ScalarTypes):
            return a, TTD.full(a.shape, b, dtype=np.result_type(a.dtype, b))

        if isinstance(b, TTD) and isinstance(a, ScalarTypes):
            return TTD.full(b.shape, a, dtype=np.result_type(b.dtype, a)), b

        raise TypeError(
            f"Unsupported operand types: {type(a).__name__} and {type(b).__name__}."
            " Operands must be either two TTDs, or a TTD and a scalar."
        )

    a, b = normalize_operands(a, b)

    if a.shape != b.shape:
        raise ValueError(
            f"Cannot add TTDs with different shapes: {a.shape} and {b.shape}."
        )

    dtype = np.result_type(a.dtype, b.dtype)

    cores = _add_cores(a.data, b.data)

    if out is None:
        return TTD(cores, dtype=dtype)

    if out.shape != a.shape:
        raise ValueError(f"Output shape mismatch: got {out.shape}, expected {a.shape}.")

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


def _add_cores[DType: np.complexfloating](
    a: list[Core[DType]], b: list[Core[DType]]
) -> list[Core[DType]]:
    """Build the cores of the sum of two same-shaped TTDs."""
    # Add vectors directly
    if len(a) == len(b) == 1:
        return [np.add(a[0], b[0])]

    # Upcast mixed-dtype inputs: block_core takes its dtype from the first block.
    if a[0].dtype != b[0].dtype:
        dtype = np.result_type(a[0].dtype, b[0].dtype)
        a = [cast(Core[DType], np.asarray(core, dtype=dtype)) for core in a]
        b = [cast(Core[DType], np.asarray(core, dtype=dtype)) for core in b]

    return [
        # stack first cores horizontally
        np.concatenate((a[0], b[0]), axis=2),
        # merge middle cores into blocks
        *map(block_core, zip(a[1:-1], b[1:-1], strict=True)),
        # stack last cores vertically
        np.concatenate((a[-1], b[-1]), axis=0),
    ]
