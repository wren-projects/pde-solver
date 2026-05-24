from functools import reduce
from typing import cast

import numpy as np

from pde_solver.pde_types import NDArray, Vector


class Gradient:
    """Gradient operator for NDArray."""

    def __call__(
        self, tensor: NDArray, spacial_step: Vector, axis: tuple[int, ...] | None = None
    ) -> NDArray:
        """
        Compute the gradient of the tensor with given spacial_steps.

        Spacial_step length must match the number of dimensions of tensor.

        The output is one dimension bigger with the first dimension corresponding
        to the dimension along which derivative is taken.
        """
        grads = np.gradient(tensor, *spacial_step, axis=axis)
        if tensor.ndim == 1:
            return np.expand_dims(grads, axis=0)
        if axis is not None and len(axis) == 1:
            return np.expand_dims(grads, axis=axis[0])

        return np.stack(cast(tuple[NDArray, ...], grads))


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
        if tensor.ndim <= 1:
            raise ValueError(
                "Divergence requires a vector field (at least 2 dimensions)."
            )
        grad = gradient(tensor, spacial_step, axis=tuple(range(1, tensor.ndim)))

        return cast(
            NDArray,
            reduce(np.add, (value[i] for i, value in enumerate(grad))),  # pyright: ignore[reportAny]
        )


gradient = Gradient()
laplace = Laplace()
divergence = Divergence()
