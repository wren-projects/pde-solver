from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, SupportsIndex, cast

import numpy as np
from wren_common.types import Index1D, Scalar

from .add import add
from .subtract import subtract

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def set_item[DType: np.floating](
    ttd: TTD[DType], indexes: Sequence[Index1D], value: Scalar
) -> None:
    """
    Set a subtensor of a TTD in place.

    Only supported indexing is targeting a single mode with either an integer
    or a slice. The rest of the indices have to be full slices (i.e. ``:``).

    Note that for nonzero values this performs TTD addition twice which may
    be slow and also inflates the ranks of the TTD by 2.

    Parameters
    ----------
    ttd : TTD[DType]
        The TTD to modify in place.
    indexes : Sequence[Index1D]
        One index per addressed mode, where each index is either an
        integer (or any :class:`SupportsIndex`) or a slice. Negative
        integer indices count from the end of the mode. Exactly one entry
        must target a mode with an integer or a non-trivial slice; all
        others must be full slices.
    value : Scalar
        Value to set.

    See Also
    --------
    get_item : Index into a TTD.
    TTD.__setitem__ : Set a subtensor of a TTD.

    """
    if len(indexes) > len(ttd.data):
        raise IndexError(f"Too many indices: {len(indexes)} > {len(ttd.data)}")

    only_single_concrete = (
        sum(isinstance(k, SupportsIndex) or k != slice(None) for k in indexes) == 1
    )

    if not only_single_concrete:
        raise IndexError(
            "TTD.__setitem__ requires exactly one mode indexed by an int or "
            "slice, with all other modes set to ':'"
        )

    only_zero = cast(bool, np.equal(value, 0))

    if not only_zero:
        _ = subtract(ttd, value, out=ttd)

    for axis, (i, core) in enumerate(zip(indexes, ttd.data, strict=False)):
        if i == slice(None):
            continue

        # copy before writing: the core may share memory with the caller's array
        updated = core.copy()
        updated[:, i, :] = 0
        ttd.data[axis] = updated

    if not only_zero:
        _ = add(ttd, value, out=ttd)
