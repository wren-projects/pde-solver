from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("linalg.norm")
def frobenius_norm[DType: np.floating](ttd: TTD[DType]) -> DType:
    """
    Return the Frobenius norm of the TTD object.

    The Frobenius norm of a TTD object is defined as the square root of its
    inner product with itself:

        ‖A‖ᶠ = √(⟨A, A⟩)

    Returns
    -------
    DType
        The Frobenius norm of the TTD object.

    """
    return cast(DType, np.sqrt(np.vdot(ttd, ttd)))
