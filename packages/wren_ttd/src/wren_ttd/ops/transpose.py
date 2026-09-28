# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple
from wren_common.math import dot_product

from wren_ttd._helpers import orthogonalize_right, truncation_parameter
from wren_ttd._numpy_api import implements_function
from wren_ttd.math import (
    DEFAULT_EPSILON,
    delta_truncated_svd,
)

if TYPE_CHECKING:
    from wren_ttd.core import TTD


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
    from wren_ttd.core import TTD

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
