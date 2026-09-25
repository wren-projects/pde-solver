from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from wren_common.types import Scalar

from wren_ttd._numpy_api import implements_ufunc

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .add import add


@implements_ufunc("subtract")
def subtract[DType: np.floating](
    a: TTD[DType] | Scalar, b: TTD[DType] | Scalar, out: TTD[DType] | None = None
) -> TTD[DType]:
    """
    Subtract two TTD objects.

    This is a shorthand for addition with the second TTD object negated. See
    `add` and `neg` for more details.

    Parameters
    ----------
    a : TTD[DType]
        The first TTD object.
    b : TTD[DType]
        The second TTD object.
    out : TTD[DType], optional
        The output TTD object. If not provided, a new TTD object is created.

    Returns
    -------
    TTD[DType]
        The result of the subtraction.

    """
    negative = cast("TTD[DType] | Scalar", cast(object, np.negative(b)))
    return add(a, negative, out=out)
