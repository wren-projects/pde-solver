import numpy as np
from numpy.typing import NDArray

TEST_TENSORS: tuple[NDArray, ...] = (
    # ---- 2D ----
    np.arange(12).reshape(3, 4),  # (3,4)
    np.random.default_rng(0).random((5, 2)),  # (5,2)
    np.array([x % 2 == 0 for x in range(16)]).reshape(4, 4),  # (4,4) bool
    # ---- 3D ----
    np.arange(48).reshape(3, 4, 4),  # (3,4,4)
    np.random.default_rng(1).normal(size=(2, 3, 5)),  # (2,3,5)
    # ---- 4D ----
    np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5),  # (2,3,4,5)
    np.ones((2, 3, 2, 4)),  # (2,3,2,4) float
    # ---- 5D ----
    np.arange(3 * 2 * 3 * 4 * 2).reshape(3, 2, 3, 4, 2),  # (3,2,3,4,2)
    # ---- 6D ----
    np.arange(2 * 3 * 2 * 3 * 2 * 4).reshape(2, 3, 2, 3, 2, 4),  # (2,3,2,3,2,4)
    # ---- 7D ----
    np.arange(2 * 2 * 3 * 2 * 4 * 3 * 2).reshape(
        2, 2, 3, 2, 4, 3, 2
    ),  # (2,2,3,2,4,3,2)
    # ---- 8D ----
    np.arange(2 * 3 * 2 * 4 * 2 * 3 * 2 * 3).reshape(
        2, 3, 2, 4, 2, 3, 2, 3
    ),  # (2,3,2,4,2,3,2,3)
    # ---- 9D ----
    np.arange(2 * 2 * 3 * 2 * 3 * 2 * 3 * 2 * 3).reshape(2, 2, 3, 2, 3, 2, 3, 2, 3),
    # ---- 10D ----
    np.arange(2 * 2 * 2 * 3 * 2 * 2 * 3 * 2 * 2 * 3).reshape(
        2, 2, 2, 3, 2, 2, 3, 2, 2, 3
    ),
)
