# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
from wren_common.types import Scalar, ScalarTypes

from wren_ttd._numpy_api import implements_ufunc
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def _hadamard_impl[DType: np.floating](
    a: TTD[DType], b: TTD[DType], out: TTD[DType] | None = None
) -> TTD[DType]:
    from wren_ttd.core import TTD

    if a.shape != b.shape:
        raise ValueError("Tensors must have the same shape.")

    N = np.newaxis

    new_cores: list[Core[DType]] = []
    for core_a, core_b in zip(a.data, b.data, strict=True):
        la, n, ra = core_a.shape
        lb, _, rb = core_b.shape

        # Expand dimensions to leverage standard NumPy broadcasting:
        # core_a expanded: (la,  1, n, ra,  1)
        # core_b expanded: ( 1, lb, n,  1, rb)
        # Resulting shape: (la, lb, n, ra, rb)
        # then multiply and flatten back
        core = np.multiply(
            core_a[:, N, :, :, N],
            core_b[N, :, :, N, :],
        ).reshape(la * lb, n, ra * rb)

        new_cores.append(core)

    if out is not None:
        if out.shape != a.shape:
            raise ValueError("Output tensor has an incorrect shape.")
        out.data = new_cores
        return out

    return TTD(new_cores)


@overload
def multiply[DType: np.floating](
    a: TTD[DType], b: Scalar, out: TTD[DType] | None = None
) -> TTD[DType]: ...


@overload
def multiply[DType: np.floating](
    a: Scalar, b: TTD[DType], out: TTD[DType] | None = None
) -> TTD[DType]: ...


@overload
def multiply[DType: np.floating](
    a: TTD[DType], b: TTD[DType], out: TTD[DType] | None = None
) -> TTD[DType]: ...


@implements_ufunc("multiply")
def multiply[DType: np.floating](
    a: TTD[DType] | Scalar, b: TTD[DType] | Scalar, out: TTD[DType] | None = None
) -> TTD[DType]:
    """
    Multiply a TTD object by a scalar or another TTD object.

    For a TTD object A = G₀ ⊗ G₁ ⊗ ... ⊗ Gₙ, the multiplication by a scalar k is defined
    as

        kA = G₀ ⊗ G₁ ⊗ … ⊗ kGᵢ ⊗ … ⊗ Gₙ,

    where the choice of i is arbitrary from 0 to n. For performance reasons, we choose
    the smallest core.

    Multiplication by another TTD object is implemented as the Hadamard (element-wise)
    product defined as

        A ⊙ B = (G₀ ⊙ H₀) ⊗ (G₁ ⊙ H₁) ⊗ … ⊗ (Gₙ ⊙ Hₙ),

    where Gᵣ ⊙ Hᵣ = (Gᵣ Hᵣ).

    Parameters
    ----------
    a : TTD[DType] | Scalar
        The TTD object to multiply.
    b : TTD[DType] | Scalar
        The scalar to multiply the TTD object by.
    out : TTD[DType], optional
        The output TTD object. If not provided, a new TTD object is created.

    Returns
    -------
    TTD[DType]
        The result of the multiplication.

    """
    from wren_ttd.core import TTD

    def scalar_impl(
        ttd: TTD[DType], scalar: np.floating | float, out: TTD[DType] | None = None
    ) -> TTD[DType]:
        cores = ttd.data.copy()

        # find smallest core
        _, index = min((core.size, index) for index, core in enumerate(cores))

        cores[index] = np.multiply(cores[index], scalar)

        if out is not None:
            out.data = cores
            return out

        return TTD(cores, dtype=ttd.dtype)

    if isinstance(a, TTD) and isinstance(b, ScalarTypes):
        return scalar_impl(a, b, out=out)

    if isinstance(b, TTD) and isinstance(a, ScalarTypes):
        return scalar_impl(b, a, out=out)

    if isinstance(a, TTD) and isinstance(b, TTD):
        return _hadamard_impl(a, b, out=out)

    return NotImplemented
