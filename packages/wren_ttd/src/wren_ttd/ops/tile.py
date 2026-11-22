# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

from wren_ttd._helpers import singleton_cores, to_int_tuple
from wren_ttd._numpy_api import implements_function
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("tile")
def tile[DType: np.floating](a: TTD[DType], reps: int | Sequence[int]) -> TTD[DType]:
    """
    Construct a TTD by repeating it.

    Equivalent to :func:`numpy.tile` for dense arrays, extended to TTDs.

    Each repetition of an axis repeats the mode slices of the corresponding
    core, so the operation is exact and preserves the TT-ranks of the input.
    If `reps` has more entries than ``a.ndim``, the extra leading axes are
    added as rank-one all-ones cores, as in :func:`expand_dims`. If `reps`
    has fewer entries, it is prepended with ones.

    Parameters
    ----------
    a : TTD[DType]
        The TTD to tile.
    reps : int | sequence of int
        The number of repetitions along each axis. A scalar applies to the
        last axis of `a`. A sequence is aligned with the last axes of the
        result, with missing leading entries treated as one. Negative
        repetitions are rejected.

    Returns
    -------
    TTD[DType]
        The tiled TTD.

    Raises
    ------
    ValueError
        If any entry of `reps` is negative.

    See Also
    --------
    numpy.tile : Equivalent function for dense arrays.
    repeat : Repeat elements of a TTD.
    expand_dims : Insert singleton axes into a TTD.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.ones((2, 3))
    >>> ops.tile(a, 2).shape
    (2, 6)
    >>> ops.tile(a, (2, 1, 2)).shape
    (2, 2, 6)
    >>> ops.tile(a, (1,)).shape
    (2, 3)

    """
    from wren_ttd.core import TTD

    reps_tuple = to_int_tuple(reps)
    if any(rep < 0 for rep in reps_tuple):
        raise ValueError("negative dimensions are not allowed")

    if len(reps_tuple) < a.ndim:
        reps_tuple = (1,) * (a.ndim - len(reps_tuple)) + reps_tuple

    extra = len(reps_tuple) - a.ndim
    cores = singleton_cores(reps_tuple[:extra], a.dtype)
    cores.extend(
        cast(Core[DType], np.tile(core, (1, rep, 1)))
        for core, rep in zip(a.data, reps_tuple[extra:], strict=True)
    )

    return TTD(cores, dtype=a.dtype)
