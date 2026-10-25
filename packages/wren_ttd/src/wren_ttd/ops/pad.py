# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from wren_common.types import Scalar, ScalarTypes

from wren_ttd._numpy_api import implements_function
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def _pad_constant[DType: np.floating](
    ttd: TTD[DType],
    pad_width: int,
    constant_value: Scalar,
) -> TTD[DType]:
    """Pad a TTD with constant values."""
    from wren_ttd.core import TTD

    inner = slice(pad_width, -pad_width)

    if ttd.ndim == 1:
        # pad the mode dimension of the single core
        core = np.pad(
            np.squeeze(ttd.data[0]),
            pad_width,
            constant_values=constant_value,
        )
        return TTD([np.expand_dims(core, axis=(0, 2))], dtype=ttd.dtype)

    last = ttd.ndim - 1
    result_cores: list[Core[DType]] = []

    for k, core in enumerate(ttd.data):
        left, n, right = core.shape

        # Else add 2 to all internal ranks if the constant_value is not 0.
        if constant_value == 0:
            extra_left = extra_right = 0
        elif k == 0:
            extra_left, extra_right = 0, 2
        elif k == last:
            extra_left, extra_right = 2, 0
        else:
            extra_left = extra_right = 2

        result = np.zeros(
            (left + extra_left, n + 2 * pad_width, right + extra_right),
            dtype=ttd.dtype,
        )
        result[:left, inner, :right] = core

        if constant_value == 0:
            result_cores.append(result)
            continue

        # First core handles the value for the padding. It essentially adds
        # constant_value to all elements and simultaneously subtracts
        # constant_value from just the interior elements.
        if k == 0:
            result[0, :, right] = constant_value
            result[0, inner, right + 1] = -constant_value
        # The remaining cores just allow the factor from the first core to propagate
        # to all remaining dimensions while preventing spillover from/into the
        # interior.
        elif k == last:
            result[left, :, 0] = 1
            result[left + 1, inner, 0] = 1
        else:
            result[left, :, right] = 1
            result[left + 1, inner, right + 1] = 1

        result_cores.append(result)

    return TTD(result_cores, dtype=ttd.dtype)


@implements_function("pad")
def pad[DType: np.floating](
    array: TTD[DType],
    pad_width: int,
    mode: str = "constant",
    **kwargs: Any,  # type: ignore[reportAny]
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
        constant_values: Any = kwargs.get("constant_values", 0.0)  # type: ignore[reportAny]
        if isinstance(constant_values, (list, tuple)):
            raise NotImplementedError(
                "Per-axis constant_values are not supported, use a scalar"
            )

        if isinstance(constant_values, ScalarTypes):
            constant_value = cast(Scalar, constant_values)
            return _pad_constant(array, pad_width, constant_value)

        raise ValueError(
            f"Unsupported type for constant_values: {type(constant_values)}"  # type: ignore[reportAny]
        )

    return NotImplemented
