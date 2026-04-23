import numpy as np

type Vector[T: np.floating] = np.ndarray[tuple[int], np.dtype[T]]
type Matrix[T: np.floating] = np.ndarray[tuple[int, int], np.dtype[T]]
type Core[T: np.floating] = np.ndarray[tuple[int, int, int], np.dtype[T]]

type NDArray[T: np.floating] = np.ndarray[tuple[int, ...], np.dtype[T]]

type MatrixCore[T: np.floating] = np.ndarray[tuple[int, int, int, int], np.dtype[T]]
