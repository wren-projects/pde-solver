from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def to_device[DType: np.floating](
    array: TTD[DType], device: str, /, *, stream: object | None = None
) -> TTD[DType]:
    """
    Return a TTD on the requested device.

    TTDs are only supported on the CPU, so a CPU request returns `array`
    unchanged.

    Parameters
    ----------
    array : TTD[DType]
        The TTD to transfer.
    device : str
        The target device. Only ``"cpu"`` is supported.
    stream : object, optional
        Not supported; must be ``None``.

    Returns
    -------
    TTD[DType]
        The input TTD.

    Raises
    ------
    ValueError
        If `device` is not ``"cpu"``.
    NotImplementedError
        If `stream` is not None.

    See Also
    --------
    TTD.to_device : Transfer a TTD to another device.
    TTD.device : The device that stores a TTD.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 2))
    >>> ops.to_device(t, "cpu") is t
    True

    """
    if device != "cpu":
        raise ValueError("TTD arrays are only supported on the CPU")
    if stream is not None:
        raise NotImplementedError("TTD device transfers do not support streams")
    return array
