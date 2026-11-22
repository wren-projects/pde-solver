# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from wren_common.types import Scalar, ScalarTypes

from wren_ttd._numpy_api import implements_ufunc

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .multiply import multiply


@implements_ufunc("divide")
def divide[DType: np.floating](
    a: TTD[DType] | Scalar,
    b: TTD[DType] | Scalar,
    out: TTD[DType] | None = None,
) -> TTD[DType]:
    """Divide a TTD by a scalar while preserving its TT representation."""
    from wren_ttd.core import TTD

    if not isinstance(a, TTD) or not isinstance(b, ScalarTypes):
        raise NotImplementedError("TTD division is only supported by a scalar divisor")

    if out is not None and out.shape != a.shape:
        raise ValueError("Output tensor has an incorrect shape.")

    reciprocal = cast(Scalar, np.divide(a.dtype.type(1), b))
    return multiply(a, reciprocal, out=out)
