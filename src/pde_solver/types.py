import numpy as np
import numpy.typing as npt

DType = np.float64

type NDArray = npt.NDArray[DType]

type Scalar = np.dtype[DType]
type Vector = np.ndarray[tuple[int], np.dtype[DType]]
type Matrix = np.ndarray[tuple[int, int], np.dtype[DType]]
