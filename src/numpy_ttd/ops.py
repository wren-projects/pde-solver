from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import prod
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np

from numpy_ttd._numpy_api import implements_function, implements_ufunc
from numpy_ttd.math import DEFAULT_EPSILON, delta_truncated_svd, truncation_parameter
from numpy_ttd.types import Core, Matrix, NDArray

if TYPE_CHECKING:
    from numpy_ttd.ttd import TTD


@implements_ufunc("add")
def add[DType: np.floating](
    a: TTD[DType],
    b: TTD[DType],
    *,
    out: TTD[DType] | None = None,
) -> TTD[DType]:
    """
    Add two tensors in the TTD representation.

    For two TTD objects A = G₀ ⊗ G₁ ⊗ … ⊗ Gₙ and B = H₀ ⊗ H₁ ⊗ … ⊗ Hₙ, the
    addition is defined as

        A + B = (G₀ H₀) ⊗ (G₁ 0 ; 0 H₁) ⊗ (G₂ 0 ; 0 H₂) ⊗ … ⊗ (Gₙ ; Hₙ).

    The addition requires that the TTD objects have the same shape and the same
    dtype.

    Parameters
    ----------
    a : TTD[DType]
        The first TTD object.
    b : TTD[DType]
        The second TTD object.
    out : TTD[DType], optional
        The output TTD object. If not provided, a new TTD object is created.

    Returns
    -------
    TTD[DType]
        The result of the addition.

    """
    # the import has to be here to avoid circular imports
    from numpy_ttd.ttd import TTD  # noqa: PLC0415

    if a.shape != b.shape:
        raise ValueError("Tensors with different shapes cannot be added.")

    if out is not None and out.shape != a.shape:
        raise ValueError("Output tensor has an incorrect shape.")

    cores = _add_cores(a.data, b.data, a.dtype)

    if out is None:
        return TTD(cores, dtype=a.dtype)

    out.data = list(cores)
    return out


def _add_cores[DType: np.floating](
    a: list[Core[DType]], b: list[Core[DType]], dtype: np.dtype
) -> list[Core[DType]]:
    # Add vectors directly
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


@overload
def multiply[DType: np.floating](
    a: TTD[DType], b: np.floating | float, out: TTD[DType] | None = None
) -> TTD[DType]: ...


@overload
def multiply[DType: np.floating](
    a: np.floating | float, b: TTD[DType], out: TTD[DType] | None = None
) -> TTD[DType]: ...


@implements_ufunc("multiply")
def multiply[DType: np.floating](
    a: TTD[DType] | np.floating | float,
    b: TTD[DType] | np.floating | float,
    out: TTD[DType] | None = None,
) -> TTD[DType]:
    """
    Multiply a TTD object by a scalar.

    For a TTD object A = G₀ ⊗ G₁ ⊗ ... ⊗ Gₙ, the multiplication by a scalar k is defined
    as

        kA = G₀ ⊗ G₁ ⊗ … ⊗ kGᵢ ⊗ … ⊗ Gₙ,

    where the choice of i is arbitrary from 0 to n. For performance reasons, we choose
    the smallest core.

    Parameters
    ----------
    a : TTD[DType]
        The TTD object to multiply.
    b : np.floating | float
        The scalar to multiply the TTD object by.
    out : TTD[DType], optional
        The output TTD object. If not provided, a new TTD object is created.

    Returns
    -------
    TTD[DType]
        The result of the multiplication.

    """
    from numpy_ttd.ttd import TTD  # noqa: PLC0415

    def impl(
        ttd: TTD[DType], scalar: np.floating | float, out: TTD[DType] | None = None
    ) -> TTD[DType]:
        cores = ttd.data.copy()

        # find smallest core
        _, index = min((prod(core.shape), index) for index, core in enumerate(cores))

        cores[index] = np.multiply(cores[index], scalar)

        if out is not None:
            out.data = cores
            return out

        return TTD(cores, dtype=ttd.dtype)

    if isinstance(a, TTD) and isinstance(b, (np.floating, float, int)):
        return impl(a, b, out=out)

    if isinstance(b, TTD) and isinstance(a, (np.floating, float, int)):
        return impl(b, a, out=out)

    return NotImplemented


@implements_ufunc("negative")
def neg[DType: np.floating](a: TTD[DType]) -> TTD[DType]:
    """
    Negate a TTD object.

    This is a shorthand for multiplication by -1.

    Parameters
    ----------
    a : TTD[DType]
        The TTD object to negate.

    Returns
    -------
    TTD[DType]
        The negated TTD object.

    """
    return multiply(a, -1.0)


@implements_ufunc("subtract")
def subtract[DType: np.floating](
    a: TTD[DType], b: TTD[DType], out: TTD[DType] | None = None
) -> TTD[DType]:
    """
    Subtract two TTD objects.

    This is a shorthand for addition with the second TTD object negated. See
    `add` and `neg` for more details.

    Parameters
    ----------
    a : TTD[DType]
        The first TTD object.
    b : TTD[DType]
        The second TTD object.
    out : TTD[DType], optional
        The output TTD object. If not provided, a new TTD object is created.

    Returns
    -------
    TTD[DType]
        The result of the subtraction.

    """
    return add(a, neg(b), out=out)


@implements_function("vdot")
def inner_product[DType: np.floating](a: TTD[DType], b: TTD[DType]) -> DType:
    """
    Compute the inner product of two TTD objects.

    The inner product of two TTD objects is defined as the sum of inner
    products of corresponding cores. For two TTD objects A = G₀, G1, …, Gn
    and B = H₀, H₁, …, Hₙ, the inner product is defined as

        ⟨A, B⟩ = ∑ₖ₌₁ⁿ ⟨Gₖ, Hₖ⟩ = ∑ₖ₌₁ⁿ Gₖᵀ Hₖ.

    The inner product requires that the TTD objects have the same shape and the same
    dtype.

    Parameters
    ----------
    a : TTD[DType]
        The first TTD object.
    b : TTD[DType]
        The second TTD object.

    Returns
    -------
    DType
        The inner product of the two TTD objects.

    """
    if a.shape != b.shape:
        raise ValueError("TTD objects must have the same shape")

    if a.dtype != b.dtype:
        raise ValueError("TTD objects must have the same dtype")

    n = len(a.data)

    # NOTE: See `TTD.__array__` for the general generated einsum index idea.

    # ‌The generated einsum expression is in the form ABC,EBG,CDE,GDI->AI.
    # ABC is the first core of A, EBG is the first core of B, CDE is the
    # second core of A, …. Consequently, it first sums the matching cores
    # along the second (rank) axis (ABC,EBG->ACEG / CDE,GDI->CEGI), then
    # sums the results along all axes except the first and the last
    # (ACEG,CEGI->AI). Since both A and I are always 1 long, the result is a
    # matrix of size 1 × 1, which can be squeezed to a scalar.

    summation_indices: list[Any] = [
        item
        for i, (a_core, b_core) in enumerate(zip(a.data, b.data, strict=True))
        for item in (
            a_core,
            (2 * i + 0, 2 * i + 1, 2 * i + 2),
            b_core,
            (2 * (n + i) + 0, 2 * i + 1, 2 * (n + i) + 2),
        )
    ]

    total = cast(
        DType,
        np.einsum(*summation_indices, optimize=True),  # pyright: ignore[reportAny]
    )

    return total.squeeze()


@implements_function("linalg.norm")
def frobenius_norm[DType: np.floating](ttd: TTD[DType]) -> DType:
    """
    Return the Frobenius norm of the TTD object.

    The Frobenius norm of a TTD object is defined as the square root of its
    inner product with itself:

        ‖A‖ᶠ = √(⟨A, A⟩)

    Returns
    -------
    DType
        The Frobenius norm of the TTD object.

    """
    return cast(DType, np.sqrt(np.vdot(ttd, ttd)))


def _normalize_axes(axes: Iterable[int], n: int) -> tuple[int, ...]:
    return tuple((i + n) % n for i in axes)


@implements_function("transpose")
def transpose[DType: np.floating](
    ttd: TTD[DType],
    axes: Sequence[int] | None = None,
    epsilon: float | DType = DEFAULT_EPSILON,
) -> TTD[DType]:
    """
    Permute the modes of a TT-compressed tensor.

    This is implemented via a sequence of adjacent swaps. Each adjacent swap
    is done by contracting two neighboring TT cores, permuting the two physical
    dimensions, then TT-SVD splitting back with truncation. This makes the operation
    very slow, so it should not be used often.

    Parameters
    ----------
    ttd : TTD
        Input TT tensor.
    axes : sequence[int] | None
        Permutation of axes. If None, reverse axes.
    epsilon : float
        Relative tolerance for truncation during swapping.

    Returns
    -------
    TTD
        Transposed TT tensor.

    """
    from numpy_ttd.ttd import TTD  # noqa: PLC0415

    d = ttd.ndim

    # if axes is None, reverse the order of cores
    if axes is None:
        return ttd.T

    perm = _normalize_axes(map(int, axes), d)

    if len(perm) != d:
        raise ValueError("axes must have length equal to a.ndim")

    if d <= 1 or perm == tuple(range(d)):
        return ttd.copy()

    if perm == tuple(reversed(range(d))):
        return ttd.T

    order = list(range(d))

    if sorted(perm) != order:
        raise ValueError("axes must be a permutation of range(a.ndim)")

    delta = truncation_parameter(ttd, epsilon)
    cores = ttd.data.copy()

    def swap_adjacent(k: int) -> None:
        """In-place swap cores at positions k and k + 1."""
        G1 = cores[k]
        G2 = cores[k + 1]
        r0, n1, r1 = G1.shape
        r1b, n2, r2 = G2.shape

        order[k], order[k + 1] = order[k + 1], order[k]

        assert r1 == r1b, "Core rank mismatch"

        # if the outside ranks are both equal to 1, we can just swap the cores
        if r0 == r2 == 1:
            cores[k] = G2
            cores[k + 1] = G1
            return

        residue = cast(
            NDArray[DType],
            np.einsum("ajb,biz", G1, G2),
        ).reshape((r0 * n2, n1 * r2))

        u, s, v_t = delta_truncated_svd(residue, delta)

        r1_new = len(s)
        assert r1_new > 0, "SVD truncated all singular values"

        cores[k] = u.reshape((r0, n2, r1_new))
        cores[k + 1] = cast(Matrix[DType], np.einsum("i,ij->ij", s, v_t)).reshape(
            (r1_new, n1, r2)
        )

    # Bubble-sort-like: for each desired position p, bring that axis there by swaps.
    for target_pos in range(d):
        cur_pos = order.index(perm[target_pos])
        while cur_pos > target_pos:
            swap_adjacent(cur_pos - 1)
            cur_pos -= 1

    return TTD(cores, dtype=ttd.dtype)
