import numpy as np

type Core[T: np.floating] = np.ndarray[tuple[int, int, int], np.dtype[T]]
