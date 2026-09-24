from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._numpy_api import implements_ufunc

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .multiply import multiply


@implements_ufunc("negative")
def neg[DType: np.floating](a: TTD[DType]) -> TTD[DType]:
    """
    Negate a TTD object.

    This is a shorthand for multiplication by -1.

    Parameters
    ----------
    a : TTD[DType]
        The TTD object to negate.

    Returns
    -------
    TTD[DType]
        The negated TTD object.

    """
    return multiply(a, -1.0)
