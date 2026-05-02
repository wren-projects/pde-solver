from collections.abc import Iterable, Reversible

import numpy as np

from numpy_ttd.types import Core


def reverse_cores[DType: np.floating](
    cores: Reversible[Core[DType]],
) -> Iterable[Core[DType]]:
    return (core.T for core in reversed(cores))


def to_int_tuple(axes: int | Iterable[int]) -> tuple[int, ...]:
    return tuple(map(int, axes)) if isinstance(axes, Iterable) else (int(axes),)
