from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy_ttd.ttd import TTD


def scalar_mul[DType: np.floating](
    a: TTD[DType],
    b: np.floating | float,
    out: tuple[TTD[DType]] | None = None,
) -> TTD[DType]:
    """Multiply a TTD object by a scalar."""
    cores = a.data.copy()

    index, _ = min(enumerate(cores), key=lambda el: prod(el[1].shape))

    cores[index] = np.multiply(cores[index], b)

    if out is not None:
        out[0].data = cores
        return out[0]

    return a.__class__(cores, dtype=a.dtype)


def neg[DType: np.floating](
    a: TTD[DType],
) -> TTD[DType]:
    """Negate a TTD object."""
    return scalar_mul(a, -1.0)
