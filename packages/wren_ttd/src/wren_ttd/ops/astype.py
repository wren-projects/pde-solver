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
    """
    Cast a TTD to the requested floating-point dtype.

    Equivalent to :meth:`numpy.ndarray.astype` for dense arrays, extended to
    TTDs. The cast is applied to every core, so the TT representation and the
    TT-ranks are preserved.

    Parameters
    ----------
    array : TTD[DType]
        The TTD to cast.
    dtype : np.dtype or type
        The target dtype. Must be a floating-point dtype.
    copy : bool, optional
        Whether to copy the cores even when the dtype does not change, by
        default True. If False and the dtype is already the target dtype, the
        input TTD is returned unchanged.
    device : str, optional
        The device where the result is placed. Only ``"cpu"`` is supported.

    Returns
    -------
    TTD
        The cast TTD, with the requested dtype.

    Raises
    ------
    TypeError
        If `dtype` is not a floating-point dtype.
    ValueError
        If `device` is not the CPU.

    See Also
    --------
    numpy.ndarray.astype : Equivalent method for dense arrays.
    TTD.astype : Cast a TTD.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 2), dtype=np.float64)
    >>> ops.astype(t, np.float32).dtype
    dtype('float32')

    """
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
