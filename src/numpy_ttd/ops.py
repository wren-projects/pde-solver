from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy_ttd.ttd import TTD


def scalar_mul[DType: np.floating](
    a: TTD[DType],
    b: np.floating | float,
    out: TTD[DType] | None = None,
) -> TTD[DType]:
    """Multiply a TTD object by a scalar."""
    cores = a.data.copy()

    # find smallest core
    _, index = min((prod(core.shape), index) for index, core in enumerate(cores))

    cores[index] = np.multiply(cores[index], b)

    if out is not None:
        out.data = cores
        return out

    return a.__class__(cores, dtype=a.dtype)


def neg[DType: np.floating](
    a: TTD[DType],
) -> TTD[DType]:
    """Negate a TTD object."""
    return scalar_mul(a, -1.0)
