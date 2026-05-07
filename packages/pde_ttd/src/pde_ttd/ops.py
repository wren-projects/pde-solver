# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import islice, pairwise
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple
from pde_common.types import Index1D, Matrix, Scalar, ScalarTypes

from pde_ttd._helpers import orthogonalize_right, reverse_cores
from pde_ttd._numpy_api import implements_function, implements_ufunc
from pde_ttd.math import (
    DEFAULT_EPSILON,
    delta_truncated_svd,
    dot_product,
    truncation_parameter,
)
from pde_ttd.types import Core

if TYPE_CHECKING:
    from pde_ttd.core import TTD


@implements_ufunc("add")
def add[DType: np.floating](
    a: TTD[DType] | Scalar,
    b: TTD[DType] | Scalar,
    *,
    out: TTD[DType] | None = None,
) -> TTD[DType]:
    """
    Add two tensors in the TTD representation.

    For two TTD objects A = G₀ ⊗ G₁ ⊗ … ⊗ Gₙ and B = H₀ ⊗ H₁ ⊗ … ⊗ Hₙ, the
    addition is defined as

        A + B = (G₀ H₀) ⊗ (G₁ 0 ; 0 H₁) ⊗ (G₂ 0 ; 0 H₂) ⊗ … ⊗ (Gₙ ; Hₙ).

    The addition requires that the TTD objects have the same shape and the same
    dtype. If one of the operands is a scalar, it is broadcasted to the shape of
    the other operand and then added. That is equivalent to adding the scalar to
    each element of the other operand.

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
    from .core import TTD

    def normalize_operands(
        a: TTD[DType] | Scalar, b: TTD[DType] | Scalar
    ) -> tuple[TTD[DType], TTD[DType]]:

        if isinstance(a, TTD) and isinstance(b, TTD):
            return a, b

        if isinstance(a, TTD) and isinstance(b, ScalarTypes):
            return a, TTD.full(a.shape, b, dtype=a.dtype)

        if isinstance(b, TTD) and isinstance(a, ScalarTypes):
            return TTD.full(b.shape, a, dtype=b.dtype), b

        raise TypeError("a and b must be either TTDs or a TTD and a scalar")

    a, b = normalize_operands(a, b)

    if a.shape != b.shape:
        raise ValueError("Tensors with different shapes cannot be added.")

    if out is not None and out.shape != a.shape:
        raise ValueError("Output tensor has an incorrect shape.")

    cores = _add_cores(a.data, b.data)

    if out is None:
        return TTD(cores, dtype=a.dtype)

    out.data = list(cores)
    return out


def _block_core[DType: np.floating](
    blocks: tuple[Core[DType], ...],
) -> Core[DType]:
    """
    Stack cores into a single block core.

    For cores C₀, C₁, ..., Cₙ with shapes (lᵢ, n, rᵢ), the block core is
    defined as

        ⌈ C₀ 0  … 0 ⌉
        | 0  C₁ … 0 |
        | ⋮  ⋮  ⋱ ⋮ |
        ⌊ 0  0  … Cₙ⌋

    with shape (l₀ + ⋯ + lₙ, n, r₀ + ⋯ + rₙ).

    Parameters
    ----------
    blocks : tuple[Core[DType], ...]
        The cores to stack.

    Returns
    -------
    Core[DType]
        The block core.

    """
    n = blocks[0].shape[1]
    dtype = blocks[0].dtype

    # compute offsets of the blocks
    l_offsets, r_offsets = (
        cast(
            list[int],  # not really, but it's to satisfy type checkers
            # cumsum of left/right ranks
            np.r_[0, np.cumsum([c.shape[axis] for c in blocks])],
        )
        for axis in (0, 2)
    )

    G = np.zeros((l_offsets[-1], n, r_offsets[-1]), dtype=dtype)

    for core, (l0, l1), (r0, r1) in zip(
        blocks, pairwise(l_offsets), pairwise(r_offsets), strict=True
    ):
        G[l0:l1, :, r0:r1] = core

    return G


def _add_cores[DType: np.floating](
    a: list[Core[DType]], b: list[Core[DType]]
) -> list[Core[DType]]:
    # Add vectors directly
    if len(a) == len(b) == 1:
        return [np.add(a[0], b[0])]

    return [
        # stack first cores horizontally
        np.concatenate((a[0], b[0]), axis=2),
        # merge middle cores into blocks
        *map(_block_core, zip(a[1:-1], b[1:-1], strict=True)),
        # stack last cores vertically
        np.concatenate((a[-1], b[-1]), axis=0),
    ]


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
    a : TTD[DType]
        The TTD object to multiply.
    b : Scalar
        The scalar to multiply the TTD object by.
    out : TTD[DType], optional
        The output TTD object. If not provided, a new TTD object is created.

    Returns
    -------
    TTD[DType]
        The result of the multiplication.

    """
    from .core import TTD

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

    def hadamard(
        a: TTD[DType], b: TTD[DType], out: TTD[DType] | None = None
    ) -> TTD[DType]:
        from .core import TTD

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
            out.data = new_cores
            return out

        return TTD(new_cores)

    if isinstance(a, TTD) and isinstance(b, ScalarTypes):
        return scalar_impl(a, b, out=out)

    if isinstance(b, TTD) and isinstance(a, ScalarTypes):
        return scalar_impl(b, a, out=out)

    if isinstance(a, TTD) and isinstance(b, TTD):
        return hadamard(a, b, out=out)

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
    a: TTD[DType] | Scalar, b: TTD[DType] | Scalar, out: TTD[DType] | None = None
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
    negative = cast("TTD[DType] | Scalar", cast(object, np.negative(b)))
    return add(a, negative, out=out)


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


@implements_function("tensordot")
def tensordot[DType: np.floating](
    a: TTD[DType],
    b: TTD[DType],
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> TTD[DType] | DType:
    """
    Compute a tensordot of two TTDs.

    Supports:
      * axes = int k: contracts cores a[-k:] with b[:k]
      * axes = (a_axis, b_axis): contracts cores a[a_axis] with b[b_axis]
      * axes = (a_axes, b_axes): contracts a_axes with b_axes

    Note: Axes may be specified by negative indices. In that case, they are
    counted from the end of the tensor.

    See :func:`numpy.tensordot` for more details about the axes argument.

    Returns
    -------
    TTD[DType] | DType
        TTD with uncontracted dimensions (first for a, than from b) or scalar.

    """
    from pde_ttd.core import TTD

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

    a_axes = normalize_axis_tuple(a_axes_raw, a.ndim, "a_axes")
    b_axes = normalize_axis_tuple(b_axes_raw, b.ndim, "b_axes")

    if len(a_axes) != len(b_axes):
        raise ValueError("a_axes and b_axes must have the same length")

    k = len(a_axes)

    for axis_a, axis_b in zip(a_axes, b_axes, strict=True):
        if a.shape[axis_a] != b.shape[axis_b]:
            raise ValueError("Shape mismatch on contracted axes")

    a_free = tuple(i for i in range(a.ndim) if i not in a_axes)
    b_free = tuple(i for i in range(b.ndim) if i not in b_axes)

    a_permutation = a_free + a_axes
    b_permutation = b_axes[::-1] + b_free

    a_t = transpose(a, a_permutation)
    b_t = transpose(b, b_permutation)

    return _tensordot_transposed(a_t, b_t, k, dtype=dtype)


def _tensordot_transposed[DType: np.floating](
    a: TTD[DType], b: TTD[DType], k: int, dtype: np.dtype[DType]
) -> TTD[DType] | DType:
    from pde_ttd.core import TTD

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
        out_cores[len(a_free)] = dot_product(message_matrix, b_free[0])
    else:
        out_cores[-1] = dot_product(a_free[-1], message_matrix)

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


@implements_function("swapaxes")
def swapaxes[DType: np.floating](ttd: TTD[DType], axis1: int, axis2: int) -> TTD[DType]:
    """
    Swap the axes of a TTD tensor.

    See ::func:`transpose` for more details.

    Parameters
    ----------
    ttd : TTD[DType]
        Input TTD tensor.
    axis1 : int
        First axis to swap.
    axis2 : int
        Second axis to swap.

    Returns
    -------
    TTD[DType]
        TTD tensor with `axis1` and `axis2` swapped.

    """
    axis1 = normalize_axis_index(axis1, ttd.ndim)
    axis2 = normalize_axis_index(axis2, ttd.ndim)

    if axis1 == axis2:
        return ttd

    axes = list(range(ttd.ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]

    return transpose(ttd, axes)


@implements_function("transpose")
def transpose[DType: np.floating](
    ttd: TTD[DType],
    axes: Sequence[int] | None = None,
    epsilon: float | DType = DEFAULT_EPSILON,
) -> TTD[DType]:
    """
    Permute the cores (dimensions) of a TTD.

    This is achieved by a sequence of adjacent swaps. Each adjacent swap is done
    by contracting two neighboring TTD cores, swapping the two physical
    dimensions, then splitting it back using TTD-SVD.

    Parameters
    ----------
    ttd : TTD
        Input TTD tensor.
    axes : sequence[int] | None
        Permutation of axes. If None, reverse axes.
    epsilon : float
        Relative tolerance for truncation during TTD-SVD.

    Returns
    -------
    TTD
        TTD with transposed axes.

    """
    from pde_ttd.core import TTD

    d = ttd.ndim

    # if axes is None, reverse the order of cores
    if axes is None:
        return ttd.T

    perm = normalize_axis_tuple(axes, d)

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

    orthogonalize_right(cores)

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
            cores[i + 1] = dot_product(r, cores[i + 1])

        # Swap cores
        core_0 = cores[k]
        core_1 = cores[k + 1]
        r0, n1, r1 = core_0.shape
        r1b, n2, r2 = core_1.shape

        assert r1 == r1b, f"Internal rank mismatch: {r1} != {r1b}"

        merged = dot_product(core_0, core_1).swapaxes(1, 2).reshape((r0 * n2, n1 * r2))
        u, s, v_t = delta_truncated_svd(merged, delta)

        r1_new = len(s)
        cores[k] = np.multiply(u, s).reshape((r0, n2, r1_new))
        cores[k + 1] = v_t.reshape((r1_new, n1, r2))

        target[k], target[k + 1] = target[k + 1], target[k]

        sorted_prefix = max(k - 1, 0)

    return TTD(cores, dtype=ttd.dtype)


@implements_function("stack")
def stack[DType: np.floating](ttds: Sequence[TTD[DType]], axis: int = 0) -> TTD[DType]:
    """
    Stack TTDs along a new axis.

    Create a new TTD by stacking the given sequence of TTDs along the new axis.
    All TTDs must have the same shape and dtype.

    If only a TTD is given, treat its first axis as the sequence and stack the
    remaining axes.

    The resulting TTD will have significantly inflated ranks, so it is
    recommended to round it before performing further operations.

    Parameters
    ----------
    ttds : Sequence[TTD[DType]]
        The TTDs to stack.
    axis : int, optional
        The index of the new axis along which to stack the TTDs, by default 0.

    Returns
    -------
    TTD[DType]
        The stacked TTD.

    """
    from pde_ttd.core import TTD

    if isinstance(ttds, TTD) and ttds.ndim == 1:
        return cast(TTD[DType], ttds)

    # typing is dumb
    ttds = list(cast(Sequence[TTD[DType]], ttds))

    ttd0 = ttds[0]
    dtype = ttd0.dtype
    shape = ttd0.shape
    d = ttd0.ndim

    for ttd in ttds:
        if ttd.shape != shape:
            raise ValueError(f"Shape mismatch: {ttd.shape} != {shape}")

        if ttd.dtype != dtype:
            raise ValueError(f"Dtype mismatch: {ttd.dtype} != {dtype}")

    # normalize w.r.t. the new tensor
    axis = normalize_axis_index(axis, d + 1)

    # implement axis != 0 in terms of axis == 0
    if axis == d:
        return stack([ttd.T for ttd in ttds]).T
    if axis != 0:
        return stack(ttds).swapaxes(0, axis)

    M = len(ttds)

    # create the selector core for the new axis
    G0 = np.eye(M, dtype=dtype).reshape(1, M, M)

    # iterator of tuples of matching i-th cores from all given TTDs
    zipped_cores: Iterable[tuple[Core[DType], ...]] = zip(
        *(ttd.data for ttd in ttds), strict=True
    )

    cores: list[Core[DType]] = [
        G0,
        # stack all but the last cores into block cores
        *map(_block_core, islice(zipped_cores, d - 1)),
        # stack last cores vertically
        np.vstack(next(zipped_cores)),
    ]

    return TTD(cores, dtype=dtype)


def get_item[DType: np.floating](
    ttd: TTD[DType], indexes: Sequence[Index1D]
) -> TTD[DType] | DType:
    """
    Index into a TTD.

    Supports basic NumPy-style indexing, e.g. `ttd[0, 1, 2]` or `ttd[0, :, 2]`.

    Parameters
    ----------
    ttd : TTD[DType]
        The TTD to index into.
    indexes : Sequence[Index1D]
        The indices to index into the TTD.

    Returns
    -------
    TTD[DType] | DType
        The indexed TTD or a scalar value.

    """
    from pde_ttd.core import TTD

    if len(indexes) == 0:
        raise IndexError("Cannot index with an empty tuple")

    if len(indexes) > len(ttd.data):
        raise IndexError("Too many indices")

    message_matrix: Matrix[DType] = np.ones((1, 1), dtype=ttd.dtype)

    cores: list[Core[DType]] = []

    for core, index in zip(ttd.data, indexes, strict=False):
        if isinstance(index, int):
            message_matrix = cast(Matrix[DType], message_matrix @ core[:, index, :])
            continue

        # merge the slice of the core and the message matrix
        core_slice: Core[DType] = core[:, index, :]
        cores.append(dot_product(message_matrix, core_slice))
        # continue with a clean new message matrix
        message_matrix = np.eye(core.shape[2], dtype=ttd.dtype)

    remaining = ttd.data[len(indexes) :]

    # if all cores have been consumed, the result is a single element
    if not cores and not remaining:
        return ttd.dtype.type(message_matrix.squeeze())

    # merge the message matrix into either the first remaining core or the last
    # preserved core
    if remaining:
        remaining[0] = dot_product(message_matrix, remaining[0])
        cores.extend(remaining)
    else:
        cores[-1] = dot_product(cores[-1], message_matrix)

    return TTD(cores, dtype=ttd.dtype)


@implements_function("gradient")
def gradient[DType: np.floating](
    ttd: TTD[DType],
    *varargs: float | Sequence[float],
    axis: int | Sequence[int] | None = None,
    edge_order: Literal[1, 2] = 1,
) -> TTD[DType] | tuple[TTD[DType], ...]:
    """
    Compute the gradient of a TTD.

    Parameters
    ----------
    ttd : TTD[DType]
        The TTD to compute the gradient of.
    varargs : float | Sequence[float]
        The step sizes to use for the gradient.
    axis : int | Sequence[SupportsIndex] | None, optional
        The axis or axes along which to compute the gradient, by default None,
        which is equivalent to all axes.
    edge_order : Literal[1, 2], optional
        The order of the finite differences used to compute the gradient, by
        default 1.

    """
    from pde_ttd.core import TTD

    if axis is None:
        axes = tuple(range(ttd.ndim))
    elif isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)

    axes = normalize_axis_tuple(axes, ttd.ndim)

    if len(set(axes)) != len(axes):
        raise ValueError("axes entries must be unique")

    if len(varargs) == len(axes):
        step_args = [(v,) for v in varargs]
    elif len(varargs) == 1:
        step_args = [(varargs[0],)] * len(axes)
    elif not varargs:
        step_args = [()] * len(axes)
    else:
        raise ValueError("invalid number of arguments")

    results: list[TTD[DType]] = []
    for axis_idx, spacing in zip(axes, step_args, strict=True):
        # differentiate along only a single axis at a time and copy the rest unchanged
        cores = ttd.data.copy()

        # gradient of single axis returns an NDArray despite what typing suggests
        grad = np.gradient(cores[axis_idx], *spacing, axis=1, edge_order=edge_order)
        cores[axis_idx] = cast(Core[DType], cast(object, grad))

        results.append(TTD(cores, dtype=ttd.dtype))

    if len(results) == 1:
        return results[0]

    return tuple(results)
