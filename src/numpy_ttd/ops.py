from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from numpy_ttd.types import Core

if TYPE_CHECKING:
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

    cores = _add_cores(a.data, b.data, a.dtype)

    if out is None:
        return TTD(cores, dtype=a.dtype)

    out[0].data = list(cores)
    return out[0]


def _add_cores[DType: np.floating](
    a: list[Core[DType]],
    b: list[Core[DType]],
    dtype: np.dtype,
) -> list[Core[DType]]:
    # Add vectors
    if len(a) == len(b) == 1:
        return [np.add(a[0], b[0])]

    def merge_cores(core_a: Core[DType], core_b: Core[DType]) -> Core[DType]:
        am, rank, an = core_a.shape
        bm, _, bn = core_b.shape

        result = np.zeros((am + bm, rank, an + bn), dtype=dtype)
        # upper left block
        result[:am, :, :an] = core_a
        # lower right block
        result[am:, :, an:] = core_b
        return result

    return [
        # stack first cores horizontally
        np.concatenate((a[0], b[0]), axis=2),
        # merge middle cores into blocks
        *map(merge_cores, a[1:-1], b[1:-1]),
        # stack last cores vertically
        np.concatenate((a[-1], b[-1]), axis=0),
    ]
