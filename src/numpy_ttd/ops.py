from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import pairwise
from math import prod
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np

from numpy_ttd._helpers import reverse_cores
from numpy_ttd._numpy_api import implements_function, implements_ufunc
from numpy_ttd.math import (
    DEFAULT_EPSILON,
    delta_truncated_svd,
    qr_rows,
    truncation_parameter,
)
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

    contracted = _contract_cores(a.data, b.data, n)

    assert contracted.size == 1

    return a.dtype.type(contracted.squeeze())


def _to_int_tuple(axes: int | Iterable[int]) -> tuple[int, ...]:
    return tuple(map(int, axes)) if isinstance(axes, Iterable) else (int(axes),)


@implements_function("tensordot")
def tensordot[DType: np.floating](
    a: TTD[DType],
    b: TTD[DType],
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> TTD[DType] | DType:
    """
    Compute a tensordot of two TT tensors.

    Supports:
      * axes = int k: contracts cores a[-k:] with b[:k]
      * axes = (a_axis, b_axis): contracts cores a[a_axis] with b[b_axis]
      * axes = (a_axes, b_axes): contracts a_axes with b_axes

    Note: Axes may be specified by negative indices. In that case, they are
    counted from the end of the tensor.

    See :func:`numpy.tensordot` for more details about the axes argument.

    The performance of this function is heavily dependent on the target axes:
    contraction using the integer k or using all axes for both tensors is
    generally very fast. On the other hand, contraction using any other
    choice of axes is relatively slow due to the need for transposition.

    Returns
    -------
    TTD[DType] | DType
        TT tensor over uncontracted modes (a_free then b_free) or scalar.

    """
    from numpy_ttd.ttd import TTD  # noqa: PLC0415

    if a.dtype != b.dtype:
        raise ValueError("TTD objects must have the same dtype")

    dtype = a.dtype

    if isinstance(axes, int):
        k = axes

        if k == 0:
            return TTD([*a.data, *b.data], dtype=dtype)

        if k < 0:
            raise ValueError("axes must be non-negative")

        axes = (tuple(range(-k, 0)), tuple(range(k)))

    a_axes_raw, b_axes_raw = axes

    a_axes, b_axes = _to_int_tuple(a_axes_raw), _to_int_tuple(b_axes_raw)

    # normalize negative indices
    a_axes = _normalize_axes(a_axes, a.ndim)
    b_axes = _normalize_axes(b_axes, b.ndim)

    if len(a_axes) != len(b_axes):
        raise ValueError("a_axes and b_axes must have the same length")

    dedup_axes_a, dedup_axes_b = set(a_axes), set(b_axes)

    if len(dedup_axes_a) != len(a_axes) or len(dedup_axes_b) != len(b_axes):
        raise ValueError("axes entries must be unique")

    k = len(a_axes)

    for axis_a, axis_b in zip(a_axes, b_axes, strict=True):
        if a.shape[axis_a] != b.shape[axis_b]:
            raise ValueError("Shape mismatch on contracted axes")

    # k_prefix = tuple(range(k))
    # k_suffix = tuple(reversed(range(k)))

    # if a_axes == k_suffix and b_axes == k_prefix:
    #     return _tensordot_suffix_prefix(a, b, k, dtype=dtype)

    # set returns elements in the insertion order, so this preserves the order
    a_free = tuple(set(range(a.ndim)) - dedup_axes_a)
    b_free = tuple(set(range(b.ndim)) - dedup_axes_b)

    a_permutation = a_free + a_axes
    b_permutation = b_axes[::-1] + b_free

    a_t = transpose(a, a_permutation)
    b_t = transpose(b, b_permutation)

    return _tensordot_transposed(a_t, b_t, k, dtype=dtype)


def _contract_cores[DType: np.floating](
    a_cores: Iterable[Core[DType]],
    b_cores: Iterable[Core[DType]],
    n: int,
) -> Matrix[DType]:
    # ‌The generated einsum expression is in the form ABC,GBI,CDE,IDK->AEGK. ABC
    # is the first core of A, GBI is the first core of B, CDE is the second core
    # of A, …. Consequently, it first sums the matching cores along the second
    # (rank) axis (ABC,GBI->ACGI / CDE,IDK->CEIK), then sums the results along
    # all axes except the first and the last of a and b each (ACGI,CEIK->AEGK).

    # NOTE: See `TTD.__array__` for the general generated einsum index idea.

    summation_indices: list[Any] = [
        item
        for i, (a_core, b_core) in enumerate(zip(a_cores, b_cores, strict=True))
        for item in [
            a_core,
            (2 * i + 0, 2 * i + 1, 2 * i + 2),
            b_core,
            (2 * (n + i + 1) + 0, 2 * i + 1, 2 * (n + i + 1) + 2),
        ]
    ]

    result = cast(
        Matrix[DType],
        np.einsum(*summation_indices, optimize=True),  # pyright: ignore[reportAny]
    )

    # assumes, that both sequences started from the boundary core -> |A| = |G| = 1
    return result.squeeze((0, 2))


def _tensordot_transposed[DType: np.floating](
    a: TTD[DType], b: TTD[DType], k: int, dtype: np.dtype[DType]
) -> TTD[DType] | DType:
    from numpy_ttd.ttd import TTD  # noqa: PLC0415

    assert a.ndim >= k, "k must be <= a.ndim"
    assert b.ndim >= k, "k must be <= b.ndim"

    a_free, a_contr = a.data[:-k], a.data[-k:]
    b_contr, b_free = b.data[:k], b.data[k:]

    message_matrix = _contract_cores(reverse_cores(a_contr), b_contr, k)

    # complete contraction -> scalar
    if a.ndim == k and b.ndim == k:
        assert message_matrix.size == 1
        return dtype.type(message_matrix.squeeze())

    out_cores = a_free + b_free

    # multiply the message_matrix into either the first free b core or the last
    # free a core
    if b_free:
        out_cores[len(a_free)] = np.einsum("ab,bcd", message_matrix, b_free[0])
    else:
        out_cores[-1] = np.einsum("abc,cd", a_free[-1], message_matrix)

    return TTD(out_cores, dtype=dtype)


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

    # list of target positions of each core
    target = [perm.index(i) for i in range(d)]

    # TODO: consider, if the QR-orthogonalisation is actually needed. It is used
    # in some libraries (e.g. ttpy aka TT-toolbox) but non-othogonal cores still
    # produce correct results.

    # RL-orthogonalise
    for i in reversed(range(1, d)):
        core = cores[i]
        r0, n1, r1 = core.shape
        q, r = qr_rows(core.reshape((r0, n1 * r1)))
        cores[i] = q.reshape((-1, n1, r1))
        cores[i - 1] = np.einsum("ijk,kl", cores[i - 1], r)

    sorted_prefix = 0
    while True:
        # Find a pair of cores that's not yet transposed
        try:
            k = next(i for i, (a, b) in enumerate(pairwise(target)) if a > b)
        except StopIteration:
            # All cores are already in the correct order
            break

        # Move orthogonal center to k
        for i in range(sorted_prefix, k):
            core = cores[i]
            r0, n1, r1 = core.shape
            q, r = np.linalg.qr(core.reshape((r0 * n1, r1)))
            cores[i] = q.reshape((r0, n1, -1))
            cores[i + 1] = np.einsum("jkl,ji", cores[i + 1], r)

        # Swap cores
        core_0 = cores[k]
        core_1 = cores[k + 1]
        r0, n1, r1 = core_0.shape
        r1b, n2, r2 = core_1.shape

        assert r1 == r1b, f"Internal rank mismatch: {r1} != {r1b}"

        merged = cast(NDArray[Any], np.einsum("ajk,kiz", core_0, core_1)).reshape(
            (r0 * n2, n1 * r2)
        )
        u, s, v_t = delta_truncated_svd(merged, delta)

        r1_new = len(s)
        cores[k] = (u * s).reshape((r0, n2, r1_new))
        cores[k + 1] = v_t.reshape((r1_new, n1, r2))

        target[k], target[k + 1] = target[k + 1], target[k]

        sorted_prefix = max(k - 1, 0)

    return TTD(cores, dtype=ttd.dtype)
