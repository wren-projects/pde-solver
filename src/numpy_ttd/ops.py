from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from numpy_ttd.math import DEFAULT_EPSILON
from numpy_ttd.types import Core, Matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from numpy_ttd.ttd import TTD


def add[DType: np.floating](
    a: TTD[DType],
    b: TTD[DType],
    *,
    out: tuple[TTD[DType]] | None = None,
) -> TTD[DType]:
    """Add two TTD objects."""
    # the import has to be here to avoid circular imports
    from numpy_ttd.ttd import TTD  # noqa: PLC0415

    if a.shape != b.shape:
        raise ValueError("Tensors with different shapes cannot be added.")

    if out is not None and out[0].shape != a.shape:
        raise ValueError("Output tensor has an incorrect shape.")

    cores = _add_cores(a.data, b.data, a.shape, a.dtype)

    if out is None:
        return TTD(cores, dtype=a.dtype)

    out[0].data = list(cores)
    return out[0]


def _add_cores[DType: np.floating](
    a: list[Core[DType]],
    b: list[Core[DType]],
    ranks: tuple[int, ...],
    dtype: np.dtype,
) -> Iterable[Core[DType]]:

    # Add vectors
    if len(a) == len(b) == 1:
        return [np.add(a[0], b[0])]

    def create_block(matrix_a: Matrix[DType], matrix_b: Matrix[DType]) -> Matrix[DType]:
        # first core
        if matrix_a.shape[0] == 1:
            return np.hstack((matrix_a, matrix_b))

        # last core
        if matrix_a.shape[1] == 1:
            return np.vstack((matrix_a, matrix_b))

        # middle cores
        am, an = matrix_a.shape
        bm, bn = matrix_b.shape

        result = np.zeros((am + bm, an + bn), dtype=dtype)
        result[:am, :an] = matrix_a
        result[am:, an:] = matrix_b
        return result

    def merge_cores(core_a: Core[DType], core_b: Core[DType], rank: int) -> Core[DType]:
        return np.stack(
            [create_block(core_a[:, i, :], core_b[:, i, :]) for i in range(rank)],
            axis=1,
        )

    return map(merge_cores, a, b, ranks)
