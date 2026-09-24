from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._helpers import contract_cores
from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("vdot")
def inner_product[DType: np.floating](a: TTD[DType], b: TTD[DType]) -> DType:
    """
    Compute the inner product of two TTD objects.

    The inner product of two TTD objects is defined as the sum of inner
    products of corresponding cores. For two TTD objects A = G₀, G1, …, Gn
    and B = H₀, H₁, …, Hₙ, the inner product is defined as

        ⟨A, B⟩ = ∑ₖ₌₁ⁿ ⟨Gₖ, Hₖ⟩ = ∑ₖ₌₁ⁿ Gₖᵀ Hₖ.

    The inner product requires that the TTD objects have the same shape and the same
    dtype.

    Parameters
    ----------
    a : TTD[DType]
        The first TTD object.
    b : TTD[DType]
        The second TTD object.

    Returns
    -------
    DType
        The inner product of the two TTD objects.

    """
    if a.shape != b.shape:
        raise ValueError("TTD objects must have the same shape")

    if a.dtype != b.dtype:
        raise ValueError("TTD objects must have the same dtype")

    n = len(a.data)

    contracted = contract_cores(a.data, b.data, n)

    assert contracted.size == 1

    return a.dtype.type(contracted.squeeze())
