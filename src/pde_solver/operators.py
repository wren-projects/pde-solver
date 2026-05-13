from typing import cast
import numpy as np

from pde_solver.pde_types import NDArray, Vector
from functools import reduce


class Gradient:
    """Gradient operator for NDArray."""

    def __call__(self, tensor: NDArray, spacial_step: Vector) -> NDArray:
        """
        Compute the gradient of the tensor with given spacial_steps.

        Spacial_step length must match the number of dimensions of tensor.

        The output is one dimension bigger with the first dimension corresponding
        to the dimension along which derivative is taken.
        """
        if tensor.ndim == 1:
            return np.stack((np.gradient(tensor, spacial_step),))
        return np.stack(cast(tuple[NDArray, ...], np.gradient(tensor, spacial_step)))


class Laplace:
    """Laplace operator for NDArray."""

    def __call__(self, tensor: NDArray, spacial_step: Vector) -> NDArray:
        """
        Compute the Laplace transform of the tensor with given spacial_steps.

        Spacial_step length must match the number of dimensions of tensor.

        The output is the same dimension as tensor.
        """
        return divergence(gradient(tensor, spacial_step), spacial_step)


class Divergence:
    """Divergence operator for NDArray."""

    def __call__(self, tensor: NDArray, spacial_step: Vector) -> NDArray:
        """
        Compute the divergence of the tensor with given spacial_steps.

        Spacial_step length must be one smaller than the number of dimensions of tensor.

        The output is one dimension smaller than tensor.
        First dimension of tensor is contracted with the nabla.
        """
        if tensor.ndim == 1:
            return np.gradient(tensor, spacial_step)
        grad = cast(tuple[NDArray, ...], np.gradient(tensor, spacial_step))

        return cast(
            NDArray,
            reduce(np.add, (value[i] for i, value in enumerate(grad))),
        )


gradient = Gradient()
laplace = Laplace()
divergence = Divergence()
