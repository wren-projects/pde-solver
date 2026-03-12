from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

DType = np.float64

type NDArray = npt.NDArray[DType]

type Scalar = DType
type Vector = np.ndarray[tuple[int], np.dtype[DType]]
type Matrix = np.ndarray[tuple[int, int], np.dtype[DType]]


type Function[T: Scalar | Vector | Matrix] = Callable[[Vector], T]
type TimeFunction[T: Scalar | Vector | Matrix] = Callable[[DType, Vector], T]
