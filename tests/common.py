import math

import numpy as np
from numpy.typing import NDArray

from numpy_ttd import DEFAULT_EPSILON, TTD

rng = np.random.default_rng(0)


type TestTensor = NDArray[np.float64]
type TestTensorPair = tuple[TestTensor, TestTensor]
type TestTTD = TTD[np.float64]
type TestTTDPair = tuple[TTD[np.float64], TTD[np.float64]]

TEST_TENSORS: list[TestTensor] = [
    # add a small epsilon to all tensors to remove zero values as they break
    # relative errors
    tensor.astype(np.float64) + 1e-5
    for tensor in [
        # ---- 2D ----
        np.arange(12).reshape(3, 4),
        rng.random((5, 2)),
        np.array([x % 2 == 0 for x in range(16)]).reshape(4, 4),
        # ---- 3D ----
        np.arange(48).reshape(3, 4, 4),
        rng.normal(size=(2, 3, 5)),
        # ---- 4D ----
        np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5),
        10e50 * np.ones((2, 3, 2, 4)),
        # ---- 5D ----
        np.arange(3 * 2 * 3 * 4 * 2).reshape(3, 2, 3, 4, 2),
        -10e-5 * np.arange(3 * 2 * 3 * 4 * 2).reshape(3, 2, 3, 4, 2),
        # ---- 6D ----
        np.arange(2 * 3 * 2 * 3 * 2 * 4).reshape(2, 3, 2, 3, 2, 4),
        # ---- 7D ----
        np.arange(2 * 2 * 3 * 2 * 4 * 3 * 2).reshape(2, 2, 3, 2, 4, 3, 2),
        # ---- 8D ----
        np.arange(2 * 3 * 2 * 4 * 2 * 3 * 2 * 3).reshape(2, 3, 2, 4, 2, 3, 2, 3),
        # ---- 9D ----
        np.arange(2 * 2 * 3 * 2 * 3 * 2 * 3 * 2 * 3).reshape(2, 2, 3, 2, 3, 2, 3, 2, 3),
        # ---- 10D ----
        np.arange(2 * 2 * 2 * 3 * 2 * 2 * 3 * 2 * 2 * 3).reshape(
            2, 2, 2, 3, 2, 2, 3, 2, 2, 3
        ),
    ]
]

TEST_TTD: list[tuple[TestTensor, TestTTD]] = [
    (tensor, TTD.from_ndarray(tensor)) for tensor in TEST_TENSORS
]


# NOTE: each pair of tensors must have the same shape
TEST_PAIR_TENSORS: list[TestTensorPair] = [
    (a.astype(np.float64), b.astype(np.float64))
    for a, b in [
        # ---- 1D ----
        (np.arange(8), np.ones(8)),
        # ---- 2D ----
        (
            np.arange(12).reshape(3, 4),
            rng.integers(0, 10, size=(3, 4)),
        ),
        # ---- 3D ----
        (np.arange(24).reshape(2, 3, 4), np.ones((2, 3, 4))),
        # ---- 4D ----
        (
            np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5),
            rng.normal(size=(2, 3, 4, 5)),
        ),
        # ---- 5D ----
        (
            np.zeros((2, 2, 3, 4, 3)),
            np.arange(2 * 2 * 3 * 4 * 3).reshape(2, 2, 3, 4, 3),
        ),
        # ---- 6D ----
        (
            np.arange(2 * 3 * 2 * 3 * 2 * 4).reshape(2, 3, 2, 3, 2, 4),
            rng.integers(0, 5, size=(2, 3, 2, 3, 2, 4)),
        ),
        # ---- 7D ----
        (
            np.arange(2 * 2 * 3 * 2 * 4 * 3 * 2).reshape(2, 2, 3, 2, 4, 3, 2),
            np.ones((2, 2, 3, 2, 4, 3, 2)),
        ),
        # ---- 8D ----
        (
            np.arange(2 * 3 * 2 * 4 * 2 * 3 * 2 * 3).reshape(2, 3, 2, 4, 2, 3, 2, 3),
            rng.random((2, 3, 2, 4, 2, 3, 2, 3)),
        ),
        # ---- 9D ----
        (
            np.arange(2 * 2 * 3 * 2 * 3 * 2 * 3 * 2 * 3).reshape(
                2, 2, 3, 2, 3, 2, 3, 2, 3
            ),
            np.ones((2, 2, 3, 2, 3, 2, 3, 2, 3)),
        ),
        # ---- 10D ----
        (
            np.arange(2 * 2 * 2 * 3 * 2 * 2 * 3 * 2 * 2 * 3).reshape(
                2, 2, 2, 3, 2, 2, 3, 2, 2, 3
            ),
            rng.integers(0, 10, size=(2, 2, 2, 3, 2, 2, 3, 2, 2, 3)),
        ),
    ]
]

TEST_PAIR_TTD: list[tuple[TestTensorPair, TestTTDPair]] = [
    ((a, b), (TTD.from_ndarray(a), TTD.from_ndarray(b))) for a, b in TEST_PAIR_TENSORS
]


TEST_SCALARS: tuple[float, ...] = (1, 2, 0.5, -1, 0, math.pi, -math.e, 1e3)

type EpsilonComparable = NDArray[np.float64] | TTD[np.float64] | np.floating | float


def assert_default_epsilon(a: EpsilonComparable, b: EpsilonComparable) -> None:
    """
    Compare two tensors for equality within the default epsilon.

    Implicitly expands TTDs to ndarrays for comparison.

    Parameters
    ----------
    a
        The first tensor to compare.
    b
        The second tensor to compare.

    """
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=DEFAULT_EPSILON)
