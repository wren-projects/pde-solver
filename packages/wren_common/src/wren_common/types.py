from typing import SupportsIndex

import numpy as np

type Vector[T: np.inexact] = np.ndarray[tuple[int], np.dtype[T]]
type Matrix[T: np.inexact] = np.ndarray[tuple[int, int], np.dtype[T]]
type NDArray[T: np.inexact] = np.ndarray[tuple[int, ...], np.dtype[T]]

type Real = np.floating | np.integer | float | int
RealTypes = (np.floating, np.integer, float, int)

type Scalar = Real | np.complexfloating | complex
ScalarTypes = (np.complexfloating, complex, *RealTypes)

type Index1D = SupportsIndex | slice[SupportsIndex | None]
