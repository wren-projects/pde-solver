from typing import TypedDict

import numpy as np
from wren_common.types import Scalar

# A single 3D core of a TTD: (left rank, mode size, right rank).
type Core[T: np.floating] = np.ndarray[tuple[int, int, int], np.dtype[T]]

CORE_DIMENSIONS = 3


class CompressionParameters(TypedDict, total=False):
    """Keyword parameters controlling TTD compression."""

    epsilon: Scalar


# A dtype accepted by TTD constructors: a NumPy dtype, scalar type, or name.
type TTDType[DT: np.floating] = np.dtype[DT] | type[DT] | type[float] | str
