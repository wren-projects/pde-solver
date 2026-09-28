from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._helpers import contract_cores
from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("vdot")
def inner_product[DType: np.complexfloating](a: TTD[DType], b: TTD[DType]) -> DType:
    """
    Compute the inner product of two TTD objects.

    Equivalent to :func:`numpy.vdot` for dense arrays (which flattens its
    arguments), extended to TTDs.

    The inner product of two TTD objects is defined as the sum of inner
    products of corresponding cores. For two TTD objects A = G₀, G1, …, Gn
    and B = H₀, H₁, …, Hₙ, the inner product is defined as

        ⟨A, B⟩ = ∑ₖ₌₁ⁿ ⟨Gₖ, Hₖ⟩ = ∑ₖ₌₁ⁿ Gₖᴴ Hₖ,

    where ᴴ denotes the conjugate transpose (so the first operand is
    conjugated, like in :func:`numpy.vdot`).

    The inner product requires that the TTD objects have the same shape and the same
    dtype.

    Parameters
    ----------
    a : TTD[DType]
        The first TTD object.
    b : TTD[DType]
        The second TTD object. Must have the same shape and dtype as `a`.

    Returns
    -------
    DType
        The inner product of the two TTD objects.

    See Also
    --------
    numpy.vdot : Equivalent function for dense arrays.
    frobenius_norm : Frobenius norm of a TTD.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.ones((2, 2))
    >>> ops.inner_product(a, a)
    np.float64(4.0)

    """
    if a.shape != b.shape:
        raise ValueError("TTD objects must have the same shape")

    if a.dtype != b.dtype:
        raise ValueError("TTD objects must have the same dtype")

    n = len(a.data)

    contracted = contract_cores([core.conj() for core in a.data], b.data, n)

    assert contracted.size == 1

    return a.dtype.type(contracted.squeeze())
