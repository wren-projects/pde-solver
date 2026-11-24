# ruff: noqa: PLC0415
# The API mirrors that of numpy.full_like, which fails those lints
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from wren_common.types import NDArray

from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD
    from wren_ttd.types import TTDType


@implements_function("asarray")
def asarray[DType: np.floating](
    data: TTD[DType] | NDArray[DType] | Sequence[Any],
    *,
    dtype: TTDType[DType] | None = None,
) -> TTD[DType]:
    """
    Convert the input to a Tensor Train Decomposition (TTD).

    Implements ``numpy.asarray`` dispatch via ``__array_function__`` when
    ``like=`` is passed with a ``TTD`` instance.

    Parameters
    ----------
    data : array_like
        Input data in any form that can be converted to an array, such as
        lists, tuples, ``numpy.ndarray``, or an existing ``TTD``.
    dtype : data-type, optional
        Target data-type. If ``None``, the data-type is inferred from `a`.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout forwarded to ``numpy.asarray`` prior to TTD
        decomposition.
    **kwargs : Any, optional
        Additional keyword arguments forwarded to ``numpy.asarray`` (such as
        ``copy`` or ``device`` in NumPy 2.0+).

    Returns
    -------
    out : TTD
        Tensor Train representation of `a`. Returns `a` without copying if it
        is already a ``TTD`` with a matching `dtype` and ``copy`` is not
        requested.

    See Also
    --------
    numpy.asarray : Standard NumPy array conversion function.
    TTD.from_dense : Construct a ``TTD`` from a dense ``numpy.ndarray``.

    """
    from wren_ttd.core import TTD

    return TTD(data, dtype=dtype)
