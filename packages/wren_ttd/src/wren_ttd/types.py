import numpy as np

# A single 3D core of a TTD: (left rank, mode size, right rank).
type Core[T: np.floating] = np.ndarray[tuple[int, int, int], np.dtype[T]]
