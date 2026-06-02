from typing import SupportsIndex

import numpy as np

type Vector[T: np.floating] = np.ndarray[tuple[int], np.dtype[T]]
type Matrix[T: np.floating] = np.ndarray[tuple[int, int], np.dtype[T]]
type NDArray[T: np.number] = np.ndarray[tuple[int, ...], np.dtype[T]]


type Index1D = SupportsIndex | slice[SupportsIndex | None]
