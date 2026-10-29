# ruff: noqa: PLC0415
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, overload

import numpy as np
from wren_common.types import Scalar, ScalarTypes

from wren_ttd._helpers import smallest_core
from wren_ttd._numpy_api import implements_ufunc
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


# Parallelizing only pays off once the result is large enough to amortize the
# cost of starting a thread pool.
_PARALLEL_MIN_BYTES = 1 << 21  # 2 MiB

# Environment variable to cap or disable the number of workers.
_NUM_THREADS_ENV = "WREN_TTD_NUM_THREADS"


def _num_workers(ntasks: int) -> int:
    """Return how many workers to use for ``ntasks`` independent cores."""
    override = os.environ.get(_NUM_THREADS_ENV)
    if override:
        try:
            requested = int(override)
        except ValueError:
            requested = 0
        if requested > 0:
            return min(ntasks, requested)

    return min(ntasks, os.process_cpu_count() or 1)


def _hadamard_core[DType: np.floating](
    spec: tuple[int, int, int, int, int],
    core_a: Core[DType],
    core_b: Core[DType],
) -> Core[DType]:
    """Compute one Hadamard core."""
    la, lb, n, ra, rb = spec

    # Expand dimensions to leverage standard NumPy broadcasting:
    # core_a expanded: (la,  1, n, ra,  1)
    # core_b expanded: ( 1, lb, n,  1, rb)
    # Resulting shape: (la, lb, n, ra, rb)
    # Allocate the flattened result up front and write the broadcast
    # product directly into it, avoiding a temporary and a copy.
    core: Core[DType] = np.empty(
        (la * lb, n, ra * rb), dtype=np.result_type(core_a.dtype, core_b.dtype)
    )
    _ = np.multiply(
        core_a[:, np.newaxis, :, :, np.newaxis],
        core_b[np.newaxis, :, :, np.newaxis, :],
        out=core.reshape(la, lb, n, ra, rb),
    )
    return core


def _hadamard_impl[DType: np.floating](
    a: TTD[DType], b: TTD[DType], out: TTD[DType] | None = None
) -> TTD[DType]:
    """Compute the Hadamard (element-wise) product of two same-shaped TTDs."""
    from wren_ttd.core import TTD

    if a.shape != b.shape:
        raise ValueError("Tensors must have the same shape.")

    if out is not None and out.shape != a.shape:
        raise ValueError("Output tensor has an incorrect shape.")

    specs: list[tuple[int, int, int, int, int]] = []
    itemsize = np.result_type(a.dtype, b.dtype).itemsize
    total_bytes = 0
    for core_a, core_b in zip(a.data, b.data, strict=True):
        la, n, ra = core_a.shape
        lb, _, rb = core_b.shape
        specs.append((la, lb, n, ra, rb))
        total_bytes += la * lb * n * ra * rb * itemsize

    # Each core is independent, and NumPy releases the GIL inside the
    # multiply, so the cores can be computed concurrently.
    workers = _num_workers(len(specs)) if total_bytes >= _PARALLEL_MIN_BYTES else 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            new_cores = list(executor.map(_hadamard_core, specs, a.data, b.data))
    else:
        new_cores = [
            _hadamard_core(spec, core_a, core_b)
            for spec, core_a, core_b in zip(specs, a.data, b.data, strict=True)
        ]

    if out is not None:
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
    Multiply arguments element-wise.

    Equivalent to :func:`numpy.multiply` for dense arrays, extended to TTDs.

    For a TTD object A = G₀ ⊗ G₁ ⊗ ... ⊗ Gₙ, the multiplication by a scalar k is defined
    as

        kA = G₀ ⊗ G₁ ⊗ … ⊗ kGᵢ ⊗ … ⊗ Gₙ,

    where the choice of i is arbitrary from 0 to n. For performance reasons, we choose
    the smallest core.

    Multiplication by another TTD object is implemented as the Hadamard (element-wise)
    product defined as

        A ⊙ B = (G₀ ⊙ H₀) ⊗ (G₁ ⊙ H₁) ⊗ … ⊗ (Gₙ ⊙ Hₙ),

    where Gᵣ ⊙ Hᵣ = (Gᵣ Hᵣ).

    The multiplication of two TTD objects requires that they have the same
    shape. If one of the operands is a scalar, it scales the other operand
    as described above.

    Parameters
    ----------
    a : TTD[DType] | Scalar
        The first factor. A scalar scales `b`.
    b : TTD[DType] | Scalar
        The second factor. A scalar scales `a`.
    out : TTD[DType], optional
        The output TTD object. If not provided, a new TTD object is created.
        If provided, it must have the same shape as the result and its cores
        are replaced with the cores of the product.

    Returns
    -------
    TTD[DType]
        The element-wise product of `a` and `b`.

    See Also
    --------
    numpy.multiply : Equivalent ufunc for dense arrays.
    add : Add TTDs element-wise.
    neg : Negate a TTD element-wise.

    Notes
    -----
    The TT-ranks of a Hadamard product are the products of the operand
    ranks, so repeated multiplication inflates the ranks. Consider calling
    :meth:`TTD.round` on the result before performing further operations.

    Large products are computed in parallel across cores using up to
    :func:`os.process_cpu_count` threads. Set the ``WREN_TTD_NUM_THREADS``
    environment variable to a positive integer to cap the worker count, or
    to ``1`` to disable parallelism.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.ones((2, 2))
    >>> b = TTD.full((2, 2), 2.0)
    >>> np.asarray(ops.multiply(a, b))
    array([[2., 2.],
           [2., 2.]])
    >>> np.asarray(ops.multiply(a, 3))
    array([[3., 3.],
           [3., 3.]])

    """
    from wren_ttd.core import TTD

    def scalar_impl(
        ttd: TTD[DType], scalar: Scalar, out: TTD[DType] | None = None
    ) -> TTD[DType]:
        cores = ttd.data.copy()

        core, index = smallest_core(cores)

        cores[index] = np.multiply(core, scalar)

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
