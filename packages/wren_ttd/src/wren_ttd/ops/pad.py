# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from wren_common.types import Scalar

from wren_ttd._numpy_api import implements_function
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def _padded_mask_core[DType: np.floating](
    n: int, pad_width: int, dtype: np.dtype[DType]
) -> Core[DType]:
    return np.ones((1, pad_width + n + pad_width, 1), dtype=dtype)


def _pad_constant[DType: np.floating](
    ttd: TTD[DType],
    pad_width: int,
    constant_value: Scalar,
) -> TTD[DType]:
    """Pad a TTD with constant values."""
    from wren_ttd.core import TTD

    # zero pad the TTD cores
    new_cores: list[Core[DType]] = []
    for core in ttd.data:
        r_left, n, r_right = core.shape
        new_core: Core[DType] = np.zeros(
            (r_left, pad_width + n + pad_width, r_right), dtype=ttd.dtype
        )
        new_core[:, pad_width : pad_width + n, :] = core
        new_cores.append(new_core)

    Z = TTD(new_cores, dtype=ttd.dtype)

    if constant_value == 0 or constant_value == ttd.dtype.type(0):
        return Z

    # Build an all-ones TTD of the padded shape (rank 1)
    ones_cores = (
        _padded_mask_core(orig_n, pad_width, dtype=ttd.dtype) for orig_n in ttd.shape
    )
    ones_ttd = TTD(ones_cores, dtype=ttd.dtype)

    # Build a rank-1 indicator TTD: 1 in the original domain, 0 in the padded region
    indicator_cores: list[Core[DType]] = []
    for orig_n in ttd.shape:
        ind_core: Core[DType] = _padded_mask_core(orig_n, pad_width, dtype=ttd.dtype)
        ind_core[0, :pad_width, 0] = 0
        ind_core[0, pad_width + orig_n :, 0] = 0
        indicator_cores.append(ind_core)
    indicator_ttd = TTD(indicator_cores, dtype=ttd.dtype)

    return Z + constant_value * (ones_ttd - indicator_ttd)


@implements_function("pad")
def pad[DType: np.floating](
    array: TTD[DType],
    pad_width: int,
    mode: str = "constant",
    **kwargs: Any,
) -> TTD[DType]:
    """
    Pad a TTD tensor.

    Parameters
    ----------
    array : TTD[DType]
        The TTD tensor to pad.
    pad_width : int
        The number of elements to pad on each side of each axis.
    mode : str, optional
        The padding mode. Only supported mode is the default 'constant' mode.
    **kwargs
        Additional keyword arguments for the padding mode.
        For 'constant' mode: constant_values (float, default 0.0).

    Returns
    -------
    TTD[DType]
        The padded TTD tensor.

    """
    if pad_width == 0:
        return array

    if mode == "constant":
        constant_values = kwargs.get("constant_values", 0.0)
        if isinstance(constant_values, (list, tuple)):
            raise NotImplementedError(
                "Per-axis constant_values are not supported, use a scalar"
            )
        return _pad_constant(array, pad_width, constant_values)

    return NotImplemented
