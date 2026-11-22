# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt

from wren_ttd._helpers import normalize_dtype
from wren_ttd._numpy_api import implements_function
from wren_ttd.types import TTDType

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@overload
def astype[NewDType: np.floating](
    array: TTD[Any],
    dtype: TTDType[NewDType],
    /,
    *,
    copy: bool = True,
    device: str | None = None,
) -> TTD[NewDType]: ...


@overload
def astype(
    array: TTD[Any],
    dtype: npt.DTypeLike,
    /,
    *,
    copy: bool = True,
    device: str | None = None,
) -> TTD[Any]: ...


@implements_function("astype")
def astype[DType: np.floating](
    array: TTD[DType],
    dtype: npt.DTypeLike,
    /,
    *,
    copy: bool = True,
    device: str | None = None,
) -> TTD[Any]:
    """Cast a TTD while preserving its TT representation."""
    from wren_ttd.core import TTD

    if device not in (None, "cpu"):
        raise ValueError("TTD arrays are only supported on the CPU")

    target_dtype = normalize_dtype(dtype)
    if target_dtype == array.dtype and not copy:
        return array

    return TTD(
        (core.astype(target_dtype, copy=copy) for core in array.data),
        dtype=target_dtype,
    )
