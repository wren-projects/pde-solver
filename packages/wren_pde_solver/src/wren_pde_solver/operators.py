from collections.abc import Sequence
from typing import SupportsIndex, cast

import numpy as np

from wren_pde_solver.pde_types import DType, NDArray, Vector


class Gradient:
    """Gradient operator for NDArray."""

    def __call__(
        self,
        tensor: NDArray,
        spacial_step: Vector,
        axis: Sequence[SupportsIndex] | None = None,
    ) -> NDArray:
        """
        Compute the gradient of the tensor with given spacial_steps.

        The output is one dimension bigger with the first dimension corresponding
        to the dimension along which derivative is taken.

        Parameters
        ----------
        tensor : NDArray
            The tensor of which gradient is computed
        spacial_step : Vector
            The spacial step in each direction. Must have length tensor.ndim.
        axis : tuple[int, …], optional
            The axis along which gradient is computed, by default computes along all
            axis.

        Returns
        -------
        NDArray
            The gradient

        """
        grads = np.gradient(tensor, *spacial_step, axis=axis)
        if tensor.ndim == 1:
            return np.expand_dims(grads, axis=0)
        if isinstance(axis, Sequence) and len(axis) == 1:
            return np.expand_dims(grads, axis=axis[0])

        return np.stack(cast(tuple[NDArray, ...], grads))


class Laplace:
    """Laplace operator for NDArray."""

    def __call__(self, tensor: NDArray, spacial_step: Vector) -> NDArray:
        """
        Compute the Laplace of the tensor with given spacial_steps.

        The output is the same dimension as tensor.

        Parameters
        ----------
        tensor : NDArray
            The tensor of which Laplace is computed
        spacial_step : Vector
            The spacial step in each direction. Must have length tensor.ndim.
        axis : tuple[int, …], optional
            The axis along which gradient is computed, by default computes along all
            axis.

        Returns
        -------
        NDArray
            The Laplace

        """
        return divergence(gradient(tensor, spacial_step), spacial_step)


class Divergence:
    """Divergence operator for NDArray."""

    ZERO: NDArray = np.zeros((1,), dtype=DType)

    def __call__(self, tensor: NDArray, spacial_step: Vector) -> NDArray:
        """
        Compute the divergence of the tensor with given spacial_steps.

        The output is one dimension smaller than the tensor.

        Parameters
        ----------
        tensor : NDArray
            The tensor of which divergence is computed
        spacial_step : Vector
            The spacial step in each direction. Must have length one smaller than
            tensor.ndim.
        axis : tuple[int, …], optional
            The axis along which gradient is computed, by default computes along all
            axis.

        Returns
        -------
        NDArray
            The divergence

        """
        if tensor.ndim <= 1:
            raise ValueError(
                "Divergence requires a vector field (at least 2 dimensions)."
            )

        # Compute dxᵢ/xᵢ for each component of the vector field
        gradients = (
            np.gradient(tensor[i], dx, axis=i) for i, dx in np.ndenumerate(spacial_step)
        )
        return sum(gradients, start=Divergence.ZERO)


gradient = Gradient()
laplace = Laplace()
divergence = Divergence()
