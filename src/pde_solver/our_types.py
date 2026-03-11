from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

DType = np.float64

type NDArray = npt.NDArray[DType]

type Scalar = np.dtype[DType]
type Vector = np.ndarray[tuple[int], np.dtype[DType]]
type Matrix = np.ndarray[tuple[int, int], np.dtype[DType]]

type ScalarFunction = Callable[[Vector], Scalar]
type VectorFunction = Callable[[Vector], Vector]
type MatrixFunction = Callable[[Vector], Matrix]


def ScalarToVector(a: Scalar, dim: int) -> Vector: ...
def ScalarToMatrix(a: Scalar, dim: int) -> Matrix: ...
def ConstantFunction[T](a: T, dim: int) -> Callable[[Vector, int], T]:
    return lambda _, __: a
