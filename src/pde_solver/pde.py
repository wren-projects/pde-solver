# pyright: reportUnsafeMultipleInheritance=false
# pyright: reportMissingSuperCall=false
# pyright: reportUnusedParameter=false
# pyright: reportAny=false
# ruff: noqa: ARG001, E501

from collections.abc import Callable
from types import NoneType
from typing import Any

import numpy as np

from pde_solver.abc.pde import PDE
from pde_solver.pde_types import (
    DType,
    Function,
    Matrix,
    MatrixFunction,
    Scalar,
    ScalarFunction,
    Vector,
    VectorFunction,
)


def identity[T](dim: int, value: T) -> T:
    """Transform value into itself."""
    return value


def constant_zero_vector(dim: int, value: None) -> Vector:
    """Transform None into zero vector."""
    return np.zeros(dim, dtype=DType)


def constant_zero(dim: int, value: None) -> Scalar:
    """Transform None into zero scalar."""
    return DType(0)


def constant_zero_function(dim: int, value: None) -> ScalarFunction:
    """Transform None into zero function."""
    return lambda _: DType(0)


def constant_to_function[T: Scalar | Vector | Matrix](
    dim: int, value: T
) -> Function[T]:
    """Transform scalar into a constant function."""
    return lambda _: value


def scalar_to_matrix(dim: int, value: Scalar) -> Matrix:
    """Transform scalar into a matrix."""
    return value * np.eye(dim, dtype=DType)


class VariableInhomogenityVariableVectorAdvectionVariableMatrixDiffusionPDE(PDE):
    """
    VariableInhomogenityVariableVectorAdvectionVariableMatrixDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some (matrix) function of positon..
    Advection is is some (vector) function of position..
    And right hand side is some (scalar) function of position..
    """

    def __init__(
        self,
        dims: int,
        variable_inhomogenity: ScalarFunction,
        variable_vector_advection: VectorFunction,
        variable_matrix_diffusion: MatrixFunction,
    ) -> None:
        """
        Create an implementation of the VariableInhomogenityVariableVectorAdvectionVariableMatrixDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        variable_inhomogenity: ScalarFunction
            The right side of the PDE.
            is some (scalar) function of position.

        variable_vector_advection: VectorFunction
            The advection part of the PDE.
            is some (vector) function of position.

        variable_matrix_diffusion: MatrixFunction
            The diffusion part of the PDE.
            is some (matrix) function of positon.

        """
        self._check_trait(dims, "variable_inhomogenity", variable_inhomogenity)
        self.variable_inhomogenity: ScalarFunction = variable_inhomogenity
        self._check_trait(dims, "variable_vector_advection", variable_vector_advection)
        self.variable_vector_advection: VectorFunction = variable_vector_advection
        self._check_trait(dims, "variable_matrix_diffusion", variable_matrix_diffusion)
        self.variable_matrix_diffusion: MatrixFunction = variable_matrix_diffusion

    def _check_function_equal(self, fce1: Callable, fce2: Callable, dims: int) -> bool:
        return np.array_equal(fce1(np.arange(dims)), fce2(np.arange(dims)))

    def _check_trait(self, dims: int, name: str, value: Any) -> None:
        if not hasattr(self, name):
            return
        old = getattr(self, name)
        if (old is not value) and (old is None or value is None):
            return
        if callable(value) and self._check_function_equal(old, value, dims):
            return
        if (not callable(value)) and ((old is value) or np.array_equal(old, value)):
            return
        raise TypeError(
            f"PDE structure latice is disrupted! Found value of attribute {name} to be {getattr(self, name)} when it should be {getattr(self, name)}"
        )


class VariableInhomogenityVariableVectorAdvectionMatrixDiffusionPDE(
    VariableInhomogenityVariableVectorAdvectionVariableMatrixDiffusionPDE
):
    """
    VariableInhomogenityVariableVectorAdvectionMatrixDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some constant matrix..
    Advection is is some (vector) function of position..
    And right hand side is some (scalar) function of position..
    """

    def __init__(
        self,
        dims: int,
        variable_inhomogenity: ScalarFunction,
        variable_vector_advection: VectorFunction,
        matrix_diffusion: Matrix,
    ) -> None:
        """
        Create an implementation of the VariableInhomogenityVariableVectorAdvectionMatrixDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        variable_inhomogenity: ScalarFunction
            The right side of the PDE.
            is some (scalar) function of position.

        variable_vector_advection: VectorFunction
            The advection part of the PDE.
            is some (vector) function of position.

        matrix_diffusion: Matrix
            The diffusion part of the PDE.
            is some constant matrix.

        """
        VariableInhomogenityVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            variable_vector_advection=identity(dims, variable_vector_advection),
            variable_matrix_diffusion=constant_to_function(dims, matrix_diffusion),
        )
        self._check_trait(dims, "variable_inhomogenity", variable_inhomogenity)
        self.variable_inhomogenity: ScalarFunction = variable_inhomogenity
        self._check_trait(dims, "variable_vector_advection", variable_vector_advection)
        self.variable_vector_advection: VectorFunction = variable_vector_advection
        self._check_trait(dims, "matrix_diffusion", matrix_diffusion)
        self.matrix_diffusion: Matrix = matrix_diffusion


class VariableInhomogenityVariableVectorAdvectionScalarDiffusionPDE(
    VariableInhomogenityVariableVectorAdvectionMatrixDiffusionPDE
):
    """
    VariableInhomogenityVariableVectorAdvectionScalarDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some constant scalar. The diffusion is then constant in all directions..
    Advection is is some (vector) function of position..
    And right hand side is some (scalar) function of position..
    """

    def __init__(
        self,
        dims: int,
        variable_inhomogenity: ScalarFunction,
        variable_vector_advection: VectorFunction,
        scalar_diffusion: Scalar,
    ) -> None:
        """
        Create an implementation of the VariableInhomogenityVariableVectorAdvectionScalarDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        variable_inhomogenity: ScalarFunction
            The right side of the PDE.
            is some (scalar) function of position.

        variable_vector_advection: VectorFunction
            The advection part of the PDE.
            is some (vector) function of position.

        scalar_diffusion: Scalar
            The diffusion part of the PDE.
            is some constant scalar. The diffusion is then constant in all directions.

        """
        VariableInhomogenityVariableVectorAdvectionMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            variable_vector_advection=identity(dims, variable_vector_advection),
            matrix_diffusion=scalar_to_matrix(dims, scalar_diffusion),
        )
        self._check_trait(dims, "variable_inhomogenity", variable_inhomogenity)
        self.variable_inhomogenity: ScalarFunction = variable_inhomogenity
        self._check_trait(dims, "variable_vector_advection", variable_vector_advection)
        self.variable_vector_advection: VectorFunction = variable_vector_advection
        self._check_trait(dims, "scalar_diffusion", scalar_diffusion)
        self.scalar_diffusion: Scalar = scalar_diffusion


class VariableInhomogenityVariableVectorAdvectionNoDiffusionPDE(
    VariableInhomogenityVariableVectorAdvectionScalarDiffusionPDE
):
    """
    VariableInhomogenityVariableVectorAdvectionNoDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is always zero (i.e. there is no diffusion). Note that the datatype is None..
    Advection is is some (vector) function of position..
    And right hand side is some (scalar) function of position..
    """

    def __init__(
        self,
        dims: int,
        variable_inhomogenity: ScalarFunction,
        variable_vector_advection: VectorFunction,
        no_diffusion: NoneType,
    ) -> None:
        """
        Create an implementation of the VariableInhomogenityVariableVectorAdvectionNoDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        variable_inhomogenity: ScalarFunction
            The right side of the PDE.
            is some (scalar) function of position.

        variable_vector_advection: VectorFunction
            The advection part of the PDE.
            is some (vector) function of position.

        no_diffusion: NoneType
            The diffusion part of the PDE.
            is always zero (i.e. there is no diffusion). Note that the datatype is None.

        """
        VariableInhomogenityVariableVectorAdvectionScalarDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            variable_vector_advection=identity(dims, variable_vector_advection),
            scalar_diffusion=constant_zero(dims, no_diffusion),
        )
        self._check_trait(dims, "variable_inhomogenity", variable_inhomogenity)
        self.variable_inhomogenity: ScalarFunction = variable_inhomogenity
        self._check_trait(dims, "variable_vector_advection", variable_vector_advection)
        self.variable_vector_advection: VectorFunction = variable_vector_advection
        self._check_trait(dims, "no_diffusion", no_diffusion)
        self.no_diffusion: NoneType = no_diffusion


class VariableInhomogenityVectorAdvectionVariableMatrixDiffusionPDE(
    VariableInhomogenityVariableVectorAdvectionVariableMatrixDiffusionPDE
):
    """
    VariableInhomogenityVectorAdvectionVariableMatrixDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some (matrix) function of positon..
    Advection is is some constant vector..
    And right hand side is some (scalar) function of position..
    """

    def __init__(
        self,
        dims: int,
        variable_inhomogenity: ScalarFunction,
        vector_advection: Vector,
        variable_matrix_diffusion: MatrixFunction,
    ) -> None:
        """
        Create an implementation of the VariableInhomogenityVectorAdvectionVariableMatrixDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        variable_inhomogenity: ScalarFunction
            The right side of the PDE.
            is some (scalar) function of position.

        vector_advection: Vector
            The advection part of the PDE.
            is some constant vector.

        variable_matrix_diffusion: MatrixFunction
            The diffusion part of the PDE.
            is some (matrix) function of positon.

        """
        VariableInhomogenityVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            variable_vector_advection=constant_to_function(dims, vector_advection),
            variable_matrix_diffusion=identity(dims, variable_matrix_diffusion),
        )
        self._check_trait(dims, "variable_inhomogenity", variable_inhomogenity)
        self.variable_inhomogenity: ScalarFunction = variable_inhomogenity
        self._check_trait(dims, "vector_advection", vector_advection)
        self.vector_advection: Vector = vector_advection
        self._check_trait(dims, "variable_matrix_diffusion", variable_matrix_diffusion)
        self.variable_matrix_diffusion: MatrixFunction = variable_matrix_diffusion


class VariableInhomogenityVectorAdvectionMatrixDiffusionPDE(
    VariableInhomogenityVariableVectorAdvectionMatrixDiffusionPDE,
    VariableInhomogenityVectorAdvectionVariableMatrixDiffusionPDE,
):
    """
    VariableInhomogenityVectorAdvectionMatrixDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some constant matrix..
    Advection is is some constant vector..
    And right hand side is some (scalar) function of position..
    """

    def __init__(
        self,
        dims: int,
        variable_inhomogenity: ScalarFunction,
        vector_advection: Vector,
        matrix_diffusion: Matrix,
    ) -> None:
        """
        Create an implementation of the VariableInhomogenityVectorAdvectionMatrixDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        variable_inhomogenity: ScalarFunction
            The right side of the PDE.
            is some (scalar) function of position.

        vector_advection: Vector
            The advection part of the PDE.
            is some constant vector.

        matrix_diffusion: Matrix
            The diffusion part of the PDE.
            is some constant matrix.

        """
        VariableInhomogenityVariableVectorAdvectionMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            variable_vector_advection=constant_to_function(dims, vector_advection),
            matrix_diffusion=identity(dims, matrix_diffusion),
        )
        VariableInhomogenityVectorAdvectionVariableMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            vector_advection=identity(dims, vector_advection),
            variable_matrix_diffusion=constant_to_function(dims, matrix_diffusion),
        )
        self._check_trait(dims, "variable_inhomogenity", variable_inhomogenity)
        self.variable_inhomogenity: ScalarFunction = variable_inhomogenity
        self._check_trait(dims, "vector_advection", vector_advection)
        self.vector_advection: Vector = vector_advection
        self._check_trait(dims, "matrix_diffusion", matrix_diffusion)
        self.matrix_diffusion: Matrix = matrix_diffusion


class VariableInhomogenityVectorAdvectionScalarDiffusionPDE(
    VariableInhomogenityVariableVectorAdvectionScalarDiffusionPDE,
    VariableInhomogenityVectorAdvectionMatrixDiffusionPDE,
):
    """
    VariableInhomogenityVectorAdvectionScalarDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some constant scalar. The diffusion is then constant in all directions..
    Advection is is some constant vector..
    And right hand side is some (scalar) function of position..
    """

    def __init__(
        self,
        dims: int,
        variable_inhomogenity: ScalarFunction,
        vector_advection: Vector,
        scalar_diffusion: Scalar,
    ) -> None:
        """
        Create an implementation of the VariableInhomogenityVectorAdvectionScalarDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        variable_inhomogenity: ScalarFunction
            The right side of the PDE.
            is some (scalar) function of position.

        vector_advection: Vector
            The advection part of the PDE.
            is some constant vector.

        scalar_diffusion: Scalar
            The diffusion part of the PDE.
            is some constant scalar. The diffusion is then constant in all directions.

        """
        VariableInhomogenityVariableVectorAdvectionScalarDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            variable_vector_advection=constant_to_function(dims, vector_advection),
            scalar_diffusion=identity(dims, scalar_diffusion),
        )
        VariableInhomogenityVectorAdvectionMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            vector_advection=identity(dims, vector_advection),
            matrix_diffusion=scalar_to_matrix(dims, scalar_diffusion),
        )
        self._check_trait(dims, "variable_inhomogenity", variable_inhomogenity)
        self.variable_inhomogenity: ScalarFunction = variable_inhomogenity
        self._check_trait(dims, "vector_advection", vector_advection)
        self.vector_advection: Vector = vector_advection
        self._check_trait(dims, "scalar_diffusion", scalar_diffusion)
        self.scalar_diffusion: Scalar = scalar_diffusion


class VariableInhomogenityVectorAdvectionNoDiffusionPDE(
    VariableInhomogenityVariableVectorAdvectionNoDiffusionPDE,
    VariableInhomogenityVectorAdvectionScalarDiffusionPDE,
):
    """
    VariableInhomogenityVectorAdvectionNoDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is always zero (i.e. there is no diffusion). Note that the datatype is None..
    Advection is is some constant vector..
    And right hand side is some (scalar) function of position..
    """

    def __init__(
        self,
        dims: int,
        variable_inhomogenity: ScalarFunction,
        vector_advection: Vector,
        no_diffusion: NoneType,
    ) -> None:
        """
        Create an implementation of the VariableInhomogenityVectorAdvectionNoDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        variable_inhomogenity: ScalarFunction
            The right side of the PDE.
            is some (scalar) function of position.

        vector_advection: Vector
            The advection part of the PDE.
            is some constant vector.

        no_diffusion: NoneType
            The diffusion part of the PDE.
            is always zero (i.e. there is no diffusion). Note that the datatype is None.

        """
        VariableInhomogenityVariableVectorAdvectionNoDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            variable_vector_advection=constant_to_function(dims, vector_advection),
            no_diffusion=identity(dims, no_diffusion),
        )
        VariableInhomogenityVectorAdvectionScalarDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            vector_advection=identity(dims, vector_advection),
            scalar_diffusion=constant_zero(dims, no_diffusion),
        )
        self._check_trait(dims, "variable_inhomogenity", variable_inhomogenity)
        self.variable_inhomogenity: ScalarFunction = variable_inhomogenity
        self._check_trait(dims, "vector_advection", vector_advection)
        self.vector_advection: Vector = vector_advection
        self._check_trait(dims, "no_diffusion", no_diffusion)
        self.no_diffusion: NoneType = no_diffusion


class VariableInhomogenityNoAdvectionVariableMatrixDiffusionPDE(
    VariableInhomogenityVectorAdvectionVariableMatrixDiffusionPDE
):
    """
    VariableInhomogenityNoAdvectionVariableMatrixDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some (matrix) function of positon..
    Advection is is always zero (i.e. there is no advection). Note that the datatype is None..
    And right hand side is some (scalar) function of position..
    """

    def __init__(
        self,
        dims: int,
        variable_inhomogenity: ScalarFunction,
        no_advection: NoneType,
        variable_matrix_diffusion: MatrixFunction,
    ) -> None:
        """
        Create an implementation of the VariableInhomogenityNoAdvectionVariableMatrixDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        variable_inhomogenity: ScalarFunction
            The right side of the PDE.
            is some (scalar) function of position.

        no_advection: NoneType
            The advection part of the PDE.
            is always zero (i.e. there is no advection). Note that the datatype is None.

        variable_matrix_diffusion: MatrixFunction
            The diffusion part of the PDE.
            is some (matrix) function of positon.

        """
        VariableInhomogenityVectorAdvectionVariableMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            vector_advection=constant_zero_vector(dims, no_advection),
            variable_matrix_diffusion=identity(dims, variable_matrix_diffusion),
        )
        self._check_trait(dims, "variable_inhomogenity", variable_inhomogenity)
        self.variable_inhomogenity: ScalarFunction = variable_inhomogenity
        self._check_trait(dims, "no_advection", no_advection)
        self.no_advection: NoneType = no_advection
        self._check_trait(dims, "variable_matrix_diffusion", variable_matrix_diffusion)
        self.variable_matrix_diffusion: MatrixFunction = variable_matrix_diffusion


class VariableInhomogenityNoAdvectionMatrixDiffusionPDE(
    VariableInhomogenityVectorAdvectionMatrixDiffusionPDE,
    VariableInhomogenityNoAdvectionVariableMatrixDiffusionPDE,
):
    """
    VariableInhomogenityNoAdvectionMatrixDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some constant matrix..
    Advection is is always zero (i.e. there is no advection). Note that the datatype is None..
    And right hand side is some (scalar) function of position..
    """

    def __init__(
        self,
        dims: int,
        variable_inhomogenity: ScalarFunction,
        no_advection: NoneType,
        matrix_diffusion: Matrix,
    ) -> None:
        """
        Create an implementation of the VariableInhomogenityNoAdvectionMatrixDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        variable_inhomogenity: ScalarFunction
            The right side of the PDE.
            is some (scalar) function of position.

        no_advection: NoneType
            The advection part of the PDE.
            is always zero (i.e. there is no advection). Note that the datatype is None.

        matrix_diffusion: Matrix
            The diffusion part of the PDE.
            is some constant matrix.

        """
        VariableInhomogenityVectorAdvectionMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            vector_advection=constant_zero_vector(dims, no_advection),
            matrix_diffusion=identity(dims, matrix_diffusion),
        )
        VariableInhomogenityNoAdvectionVariableMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            no_advection=identity(dims, no_advection),
            variable_matrix_diffusion=constant_to_function(dims, matrix_diffusion),
        )
        self._check_trait(dims, "variable_inhomogenity", variable_inhomogenity)
        self.variable_inhomogenity: ScalarFunction = variable_inhomogenity
        self._check_trait(dims, "no_advection", no_advection)
        self.no_advection: NoneType = no_advection
        self._check_trait(dims, "matrix_diffusion", matrix_diffusion)
        self.matrix_diffusion: Matrix = matrix_diffusion


class VariableInhomogenityNoAdvectionScalarDiffusionPDE(
    VariableInhomogenityVectorAdvectionScalarDiffusionPDE,
    VariableInhomogenityNoAdvectionMatrixDiffusionPDE,
):
    """
    VariableInhomogenityNoAdvectionScalarDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some constant scalar. The diffusion is then constant in all directions..
    Advection is is always zero (i.e. there is no advection). Note that the datatype is None..
    And right hand side is some (scalar) function of position..
    """

    def __init__(
        self,
        dims: int,
        variable_inhomogenity: ScalarFunction,
        no_advection: NoneType,
        scalar_diffusion: Scalar,
    ) -> None:
        """
        Create an implementation of the VariableInhomogenityNoAdvectionScalarDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        variable_inhomogenity: ScalarFunction
            The right side of the PDE.
            is some (scalar) function of position.

        no_advection: NoneType
            The advection part of the PDE.
            is always zero (i.e. there is no advection). Note that the datatype is None.

        scalar_diffusion: Scalar
            The diffusion part of the PDE.
            is some constant scalar. The diffusion is then constant in all directions.

        """
        VariableInhomogenityVectorAdvectionScalarDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            vector_advection=constant_zero_vector(dims, no_advection),
            scalar_diffusion=identity(dims, scalar_diffusion),
        )
        VariableInhomogenityNoAdvectionMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            no_advection=identity(dims, no_advection),
            matrix_diffusion=scalar_to_matrix(dims, scalar_diffusion),
        )
        self._check_trait(dims, "variable_inhomogenity", variable_inhomogenity)
        self.variable_inhomogenity: ScalarFunction = variable_inhomogenity
        self._check_trait(dims, "no_advection", no_advection)
        self.no_advection: NoneType = no_advection
        self._check_trait(dims, "scalar_diffusion", scalar_diffusion)
        self.scalar_diffusion: Scalar = scalar_diffusion


class VariableInhomogenityNoAdvectionNoDiffusionPDE(
    VariableInhomogenityVectorAdvectionNoDiffusionPDE,
    VariableInhomogenityNoAdvectionScalarDiffusionPDE,
):
    """
    VariableInhomogenityNoAdvectionNoDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is always zero (i.e. there is no diffusion). Note that the datatype is None..
    Advection is is always zero (i.e. there is no advection). Note that the datatype is None..
    And right hand side is some (scalar) function of position..
    """

    def __init__(
        self,
        dims: int,
        variable_inhomogenity: ScalarFunction,
        no_advection: NoneType,
        no_diffusion: NoneType,
    ) -> None:
        """
        Create an implementation of the VariableInhomogenityNoAdvectionNoDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        variable_inhomogenity: ScalarFunction
            The right side of the PDE.
            is some (scalar) function of position.

        no_advection: NoneType
            The advection part of the PDE.
            is always zero (i.e. there is no advection). Note that the datatype is None.

        no_diffusion: NoneType
            The diffusion part of the PDE.
            is always zero (i.e. there is no diffusion). Note that the datatype is None.

        """
        VariableInhomogenityVectorAdvectionNoDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            vector_advection=constant_zero_vector(dims, no_advection),
            no_diffusion=identity(dims, no_diffusion),
        )
        VariableInhomogenityNoAdvectionScalarDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=identity(dims, variable_inhomogenity),
            no_advection=identity(dims, no_advection),
            scalar_diffusion=constant_zero(dims, no_diffusion),
        )
        self._check_trait(dims, "variable_inhomogenity", variable_inhomogenity)
        self.variable_inhomogenity: ScalarFunction = variable_inhomogenity
        self._check_trait(dims, "no_advection", no_advection)
        self.no_advection: NoneType = no_advection
        self._check_trait(dims, "no_diffusion", no_diffusion)
        self.no_diffusion: NoneType = no_diffusion


class HomogeneousVariableVectorAdvectionVariableMatrixDiffusionPDE(
    VariableInhomogenityVariableVectorAdvectionVariableMatrixDiffusionPDE
):
    """
    HomogeneousVariableVectorAdvectionVariableMatrixDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some (matrix) function of positon..
    Advection is is some (vector) function of position..
    And right hand side is always zero (i.e. homogenoues). Note that the datatype is None..
    """

    def __init__(
        self,
        dims: int,
        homogeneous: NoneType,
        variable_vector_advection: VectorFunction,
        variable_matrix_diffusion: MatrixFunction,
    ) -> None:
        """
        Create an implementation of the HomogeneousVariableVectorAdvectionVariableMatrixDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        homogeneous: NoneType
            The right side of the PDE.
            is always zero (i.e. homogenoues). Note that the datatype is None.

        variable_vector_advection: VectorFunction
            The advection part of the PDE.
            is some (vector) function of position.

        variable_matrix_diffusion: MatrixFunction
            The diffusion part of the PDE.
            is some (matrix) function of positon.

        """
        VariableInhomogenityVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=constant_zero_function(dims, homogeneous),
            variable_vector_advection=identity(dims, variable_vector_advection),
            variable_matrix_diffusion=identity(dims, variable_matrix_diffusion),
        )
        self._check_trait(dims, "homogeneous", homogeneous)
        self.homogeneous: NoneType = homogeneous
        self._check_trait(dims, "variable_vector_advection", variable_vector_advection)
        self.variable_vector_advection: VectorFunction = variable_vector_advection
        self._check_trait(dims, "variable_matrix_diffusion", variable_matrix_diffusion)
        self.variable_matrix_diffusion: MatrixFunction = variable_matrix_diffusion


class HomogeneousVariableVectorAdvectionMatrixDiffusionPDE(
    VariableInhomogenityVariableVectorAdvectionMatrixDiffusionPDE,
    HomogeneousVariableVectorAdvectionVariableMatrixDiffusionPDE,
):
    """
    HomogeneousVariableVectorAdvectionMatrixDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some constant matrix..
    Advection is is some (vector) function of position..
    And right hand side is always zero (i.e. homogenoues). Note that the datatype is None..
    """

    def __init__(
        self,
        dims: int,
        homogeneous: NoneType,
        variable_vector_advection: VectorFunction,
        matrix_diffusion: Matrix,
    ) -> None:
        """
        Create an implementation of the HomogeneousVariableVectorAdvectionMatrixDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        homogeneous: NoneType
            The right side of the PDE.
            is always zero (i.e. homogenoues). Note that the datatype is None.

        variable_vector_advection: VectorFunction
            The advection part of the PDE.
            is some (vector) function of position.

        matrix_diffusion: Matrix
            The diffusion part of the PDE.
            is some constant matrix.

        """
        VariableInhomogenityVariableVectorAdvectionMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=constant_zero_function(dims, homogeneous),
            variable_vector_advection=identity(dims, variable_vector_advection),
            matrix_diffusion=identity(dims, matrix_diffusion),
        )
        HomogeneousVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            variable_vector_advection=identity(dims, variable_vector_advection),
            variable_matrix_diffusion=constant_to_function(dims, matrix_diffusion),
        )
        self._check_trait(dims, "homogeneous", homogeneous)
        self.homogeneous: NoneType = homogeneous
        self._check_trait(dims, "variable_vector_advection", variable_vector_advection)
        self.variable_vector_advection: VectorFunction = variable_vector_advection
        self._check_trait(dims, "matrix_diffusion", matrix_diffusion)
        self.matrix_diffusion: Matrix = matrix_diffusion


class HomogeneousVariableVectorAdvectionScalarDiffusionPDE(
    VariableInhomogenityVariableVectorAdvectionScalarDiffusionPDE,
    HomogeneousVariableVectorAdvectionMatrixDiffusionPDE,
):
    """
    HomogeneousVariableVectorAdvectionScalarDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some constant scalar. The diffusion is then constant in all directions..
    Advection is is some (vector) function of position..
    And right hand side is always zero (i.e. homogenoues). Note that the datatype is None..
    """

    def __init__(
        self,
        dims: int,
        homogeneous: NoneType,
        variable_vector_advection: VectorFunction,
        scalar_diffusion: Scalar,
    ) -> None:
        """
        Create an implementation of the HomogeneousVariableVectorAdvectionScalarDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        homogeneous: NoneType
            The right side of the PDE.
            is always zero (i.e. homogenoues). Note that the datatype is None.

        variable_vector_advection: VectorFunction
            The advection part of the PDE.
            is some (vector) function of position.

        scalar_diffusion: Scalar
            The diffusion part of the PDE.
            is some constant scalar. The diffusion is then constant in all directions.

        """
        VariableInhomogenityVariableVectorAdvectionScalarDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=constant_zero_function(dims, homogeneous),
            variable_vector_advection=identity(dims, variable_vector_advection),
            scalar_diffusion=identity(dims, scalar_diffusion),
        )
        HomogeneousVariableVectorAdvectionMatrixDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            variable_vector_advection=identity(dims, variable_vector_advection),
            matrix_diffusion=scalar_to_matrix(dims, scalar_diffusion),
        )
        self._check_trait(dims, "homogeneous", homogeneous)
        self.homogeneous: NoneType = homogeneous
        self._check_trait(dims, "variable_vector_advection", variable_vector_advection)
        self.variable_vector_advection: VectorFunction = variable_vector_advection
        self._check_trait(dims, "scalar_diffusion", scalar_diffusion)
        self.scalar_diffusion: Scalar = scalar_diffusion


class HomogeneousVariableVectorAdvectionNoDiffusionPDE(
    VariableInhomogenityVariableVectorAdvectionNoDiffusionPDE,
    HomogeneousVariableVectorAdvectionScalarDiffusionPDE,
):
    """
    HomogeneousVariableVectorAdvectionNoDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is always zero (i.e. there is no diffusion). Note that the datatype is None..
    Advection is is some (vector) function of position..
    And right hand side is always zero (i.e. homogenoues). Note that the datatype is None..
    """

    def __init__(
        self,
        dims: int,
        homogeneous: NoneType,
        variable_vector_advection: VectorFunction,
        no_diffusion: NoneType,
    ) -> None:
        """
        Create an implementation of the HomogeneousVariableVectorAdvectionNoDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        homogeneous: NoneType
            The right side of the PDE.
            is always zero (i.e. homogenoues). Note that the datatype is None.

        variable_vector_advection: VectorFunction
            The advection part of the PDE.
            is some (vector) function of position.

        no_diffusion: NoneType
            The diffusion part of the PDE.
            is always zero (i.e. there is no diffusion). Note that the datatype is None.

        """
        VariableInhomogenityVariableVectorAdvectionNoDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=constant_zero_function(dims, homogeneous),
            variable_vector_advection=identity(dims, variable_vector_advection),
            no_diffusion=identity(dims, no_diffusion),
        )
        HomogeneousVariableVectorAdvectionScalarDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            variable_vector_advection=identity(dims, variable_vector_advection),
            scalar_diffusion=constant_zero(dims, no_diffusion),
        )
        self._check_trait(dims, "homogeneous", homogeneous)
        self.homogeneous: NoneType = homogeneous
        self._check_trait(dims, "variable_vector_advection", variable_vector_advection)
        self.variable_vector_advection: VectorFunction = variable_vector_advection
        self._check_trait(dims, "no_diffusion", no_diffusion)
        self.no_diffusion: NoneType = no_diffusion


class HomogeneousVectorAdvectionVariableMatrixDiffusionPDE(
    VariableInhomogenityVectorAdvectionVariableMatrixDiffusionPDE,
    HomogeneousVariableVectorAdvectionVariableMatrixDiffusionPDE,
):
    """
    HomogeneousVectorAdvectionVariableMatrixDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some (matrix) function of positon..
    Advection is is some constant vector..
    And right hand side is always zero (i.e. homogenoues). Note that the datatype is None..
    """

    def __init__(
        self,
        dims: int,
        homogeneous: NoneType,
        vector_advection: Vector,
        variable_matrix_diffusion: MatrixFunction,
    ) -> None:
        """
        Create an implementation of the HomogeneousVectorAdvectionVariableMatrixDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        homogeneous: NoneType
            The right side of the PDE.
            is always zero (i.e. homogenoues). Note that the datatype is None.

        vector_advection: Vector
            The advection part of the PDE.
            is some constant vector.

        variable_matrix_diffusion: MatrixFunction
            The diffusion part of the PDE.
            is some (matrix) function of positon.

        """
        VariableInhomogenityVectorAdvectionVariableMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=constant_zero_function(dims, homogeneous),
            vector_advection=identity(dims, vector_advection),
            variable_matrix_diffusion=identity(dims, variable_matrix_diffusion),
        )
        HomogeneousVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            variable_vector_advection=constant_to_function(dims, vector_advection),
            variable_matrix_diffusion=identity(dims, variable_matrix_diffusion),
        )
        self._check_trait(dims, "homogeneous", homogeneous)
        self.homogeneous: NoneType = homogeneous
        self._check_trait(dims, "vector_advection", vector_advection)
        self.vector_advection: Vector = vector_advection
        self._check_trait(dims, "variable_matrix_diffusion", variable_matrix_diffusion)
        self.variable_matrix_diffusion: MatrixFunction = variable_matrix_diffusion


class HomogeneousVectorAdvectionMatrixDiffusionPDE(
    VariableInhomogenityVectorAdvectionMatrixDiffusionPDE,
    HomogeneousVariableVectorAdvectionMatrixDiffusionPDE,
    HomogeneousVectorAdvectionVariableMatrixDiffusionPDE,
):
    """
    HomogeneousVectorAdvectionMatrixDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some constant matrix..
    Advection is is some constant vector..
    And right hand side is always zero (i.e. homogenoues). Note that the datatype is None..
    """

    def __init__(
        self,
        dims: int,
        homogeneous: NoneType,
        vector_advection: Vector,
        matrix_diffusion: Matrix,
    ) -> None:
        """
        Create an implementation of the HomogeneousVectorAdvectionMatrixDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        homogeneous: NoneType
            The right side of the PDE.
            is always zero (i.e. homogenoues). Note that the datatype is None.

        vector_advection: Vector
            The advection part of the PDE.
            is some constant vector.

        matrix_diffusion: Matrix
            The diffusion part of the PDE.
            is some constant matrix.

        """
        VariableInhomogenityVectorAdvectionMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=constant_zero_function(dims, homogeneous),
            vector_advection=identity(dims, vector_advection),
            matrix_diffusion=identity(dims, matrix_diffusion),
        )
        HomogeneousVariableVectorAdvectionMatrixDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            variable_vector_advection=constant_to_function(dims, vector_advection),
            matrix_diffusion=identity(dims, matrix_diffusion),
        )
        HomogeneousVectorAdvectionVariableMatrixDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            vector_advection=identity(dims, vector_advection),
            variable_matrix_diffusion=constant_to_function(dims, matrix_diffusion),
        )
        self._check_trait(dims, "homogeneous", homogeneous)
        self.homogeneous: NoneType = homogeneous
        self._check_trait(dims, "vector_advection", vector_advection)
        self.vector_advection: Vector = vector_advection
        self._check_trait(dims, "matrix_diffusion", matrix_diffusion)
        self.matrix_diffusion: Matrix = matrix_diffusion


class HomogeneousVectorAdvectionScalarDiffusionPDE(
    VariableInhomogenityVectorAdvectionScalarDiffusionPDE,
    HomogeneousVariableVectorAdvectionScalarDiffusionPDE,
    HomogeneousVectorAdvectionMatrixDiffusionPDE,
):
    """
    HomogeneousVectorAdvectionScalarDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some constant scalar. The diffusion is then constant in all directions..
    Advection is is some constant vector..
    And right hand side is always zero (i.e. homogenoues). Note that the datatype is None..
    """

    def __init__(
        self,
        dims: int,
        homogeneous: NoneType,
        vector_advection: Vector,
        scalar_diffusion: Scalar,
    ) -> None:
        """
        Create an implementation of the HomogeneousVectorAdvectionScalarDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        homogeneous: NoneType
            The right side of the PDE.
            is always zero (i.e. homogenoues). Note that the datatype is None.

        vector_advection: Vector
            The advection part of the PDE.
            is some constant vector.

        scalar_diffusion: Scalar
            The diffusion part of the PDE.
            is some constant scalar. The diffusion is then constant in all directions.

        """
        VariableInhomogenityVectorAdvectionScalarDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=constant_zero_function(dims, homogeneous),
            vector_advection=identity(dims, vector_advection),
            scalar_diffusion=identity(dims, scalar_diffusion),
        )
        HomogeneousVariableVectorAdvectionScalarDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            variable_vector_advection=constant_to_function(dims, vector_advection),
            scalar_diffusion=identity(dims, scalar_diffusion),
        )
        HomogeneousVectorAdvectionMatrixDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            vector_advection=identity(dims, vector_advection),
            matrix_diffusion=scalar_to_matrix(dims, scalar_diffusion),
        )
        self._check_trait(dims, "homogeneous", homogeneous)
        self.homogeneous: NoneType = homogeneous
        self._check_trait(dims, "vector_advection", vector_advection)
        self.vector_advection: Vector = vector_advection
        self._check_trait(dims, "scalar_diffusion", scalar_diffusion)
        self.scalar_diffusion: Scalar = scalar_diffusion


class HomogeneousVectorAdvectionNoDiffusionPDE(
    VariableInhomogenityVectorAdvectionNoDiffusionPDE,
    HomogeneousVariableVectorAdvectionNoDiffusionPDE,
    HomogeneousVectorAdvectionScalarDiffusionPDE,
):
    """
    HomogeneousVectorAdvectionNoDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is always zero (i.e. there is no diffusion). Note that the datatype is None..
    Advection is is some constant vector..
    And right hand side is always zero (i.e. homogenoues). Note that the datatype is None..
    """

    def __init__(
        self,
        dims: int,
        homogeneous: NoneType,
        vector_advection: Vector,
        no_diffusion: NoneType,
    ) -> None:
        """
        Create an implementation of the HomogeneousVectorAdvectionNoDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        homogeneous: NoneType
            The right side of the PDE.
            is always zero (i.e. homogenoues). Note that the datatype is None.

        vector_advection: Vector
            The advection part of the PDE.
            is some constant vector.

        no_diffusion: NoneType
            The diffusion part of the PDE.
            is always zero (i.e. there is no diffusion). Note that the datatype is None.

        """
        VariableInhomogenityVectorAdvectionNoDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=constant_zero_function(dims, homogeneous),
            vector_advection=identity(dims, vector_advection),
            no_diffusion=identity(dims, no_diffusion),
        )
        HomogeneousVariableVectorAdvectionNoDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            variable_vector_advection=constant_to_function(dims, vector_advection),
            no_diffusion=identity(dims, no_diffusion),
        )
        HomogeneousVectorAdvectionScalarDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            vector_advection=identity(dims, vector_advection),
            scalar_diffusion=constant_zero(dims, no_diffusion),
        )
        self._check_trait(dims, "homogeneous", homogeneous)
        self.homogeneous: NoneType = homogeneous
        self._check_trait(dims, "vector_advection", vector_advection)
        self.vector_advection: Vector = vector_advection
        self._check_trait(dims, "no_diffusion", no_diffusion)
        self.no_diffusion: NoneType = no_diffusion


class HomogeneousNoAdvectionVariableMatrixDiffusionPDE(
    VariableInhomogenityNoAdvectionVariableMatrixDiffusionPDE,
    HomogeneousVectorAdvectionVariableMatrixDiffusionPDE,
):
    """
    HomogeneousNoAdvectionVariableMatrixDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some (matrix) function of positon..
    Advection is is always zero (i.e. there is no advection). Note that the datatype is None..
    And right hand side is always zero (i.e. homogenoues). Note that the datatype is None..
    """

    def __init__(
        self,
        dims: int,
        homogeneous: NoneType,
        no_advection: NoneType,
        variable_matrix_diffusion: MatrixFunction,
    ) -> None:
        """
        Create an implementation of the HomogeneousNoAdvectionVariableMatrixDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        homogeneous: NoneType
            The right side of the PDE.
            is always zero (i.e. homogenoues). Note that the datatype is None.

        no_advection: NoneType
            The advection part of the PDE.
            is always zero (i.e. there is no advection). Note that the datatype is None.

        variable_matrix_diffusion: MatrixFunction
            The diffusion part of the PDE.
            is some (matrix) function of positon.

        """
        VariableInhomogenityNoAdvectionVariableMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=constant_zero_function(dims, homogeneous),
            no_advection=identity(dims, no_advection),
            variable_matrix_diffusion=identity(dims, variable_matrix_diffusion),
        )
        HomogeneousVectorAdvectionVariableMatrixDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            vector_advection=constant_zero_vector(dims, no_advection),
            variable_matrix_diffusion=identity(dims, variable_matrix_diffusion),
        )
        self._check_trait(dims, "homogeneous", homogeneous)
        self.homogeneous: NoneType = homogeneous
        self._check_trait(dims, "no_advection", no_advection)
        self.no_advection: NoneType = no_advection
        self._check_trait(dims, "variable_matrix_diffusion", variable_matrix_diffusion)
        self.variable_matrix_diffusion: MatrixFunction = variable_matrix_diffusion


class HomogeneousNoAdvectionMatrixDiffusionPDE(
    VariableInhomogenityNoAdvectionMatrixDiffusionPDE,
    HomogeneousVectorAdvectionMatrixDiffusionPDE,
    HomogeneousNoAdvectionVariableMatrixDiffusionPDE,
):
    """
    HomogeneousNoAdvectionMatrixDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some constant matrix..
    Advection is is always zero (i.e. there is no advection). Note that the datatype is None..
    And right hand side is always zero (i.e. homogenoues). Note that the datatype is None..
    """

    def __init__(
        self,
        dims: int,
        homogeneous: NoneType,
        no_advection: NoneType,
        matrix_diffusion: Matrix,
    ) -> None:
        """
        Create an implementation of the HomogeneousNoAdvectionMatrixDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        homogeneous: NoneType
            The right side of the PDE.
            is always zero (i.e. homogenoues). Note that the datatype is None.

        no_advection: NoneType
            The advection part of the PDE.
            is always zero (i.e. there is no advection). Note that the datatype is None.

        matrix_diffusion: Matrix
            The diffusion part of the PDE.
            is some constant matrix.

        """
        VariableInhomogenityNoAdvectionMatrixDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=constant_zero_function(dims, homogeneous),
            no_advection=identity(dims, no_advection),
            matrix_diffusion=identity(dims, matrix_diffusion),
        )
        HomogeneousVectorAdvectionMatrixDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            vector_advection=constant_zero_vector(dims, no_advection),
            matrix_diffusion=identity(dims, matrix_diffusion),
        )
        HomogeneousNoAdvectionVariableMatrixDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            no_advection=identity(dims, no_advection),
            variable_matrix_diffusion=constant_to_function(dims, matrix_diffusion),
        )
        self._check_trait(dims, "homogeneous", homogeneous)
        self.homogeneous: NoneType = homogeneous
        self._check_trait(dims, "no_advection", no_advection)
        self.no_advection: NoneType = no_advection
        self._check_trait(dims, "matrix_diffusion", matrix_diffusion)
        self.matrix_diffusion: Matrix = matrix_diffusion


class HomogeneousNoAdvectionScalarDiffusionPDE(
    VariableInhomogenityNoAdvectionScalarDiffusionPDE,
    HomogeneousVectorAdvectionScalarDiffusionPDE,
    HomogeneousNoAdvectionMatrixDiffusionPDE,
):
    """
    HomogeneousNoAdvectionScalarDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is some constant scalar. The diffusion is then constant in all directions..
    Advection is is always zero (i.e. there is no advection). Note that the datatype is None..
    And right hand side is always zero (i.e. homogenoues). Note that the datatype is None..
    """

    def __init__(
        self,
        dims: int,
        homogeneous: NoneType,
        no_advection: NoneType,
        scalar_diffusion: Scalar,
    ) -> None:
        """
        Create an implementation of the HomogeneousNoAdvectionScalarDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        homogeneous: NoneType
            The right side of the PDE.
            is always zero (i.e. homogenoues). Note that the datatype is None.

        no_advection: NoneType
            The advection part of the PDE.
            is always zero (i.e. there is no advection). Note that the datatype is None.

        scalar_diffusion: Scalar
            The diffusion part of the PDE.
            is some constant scalar. The diffusion is then constant in all directions.

        """
        VariableInhomogenityNoAdvectionScalarDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=constant_zero_function(dims, homogeneous),
            no_advection=identity(dims, no_advection),
            scalar_diffusion=identity(dims, scalar_diffusion),
        )
        HomogeneousVectorAdvectionScalarDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            vector_advection=constant_zero_vector(dims, no_advection),
            scalar_diffusion=identity(dims, scalar_diffusion),
        )
        HomogeneousNoAdvectionMatrixDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            no_advection=identity(dims, no_advection),
            matrix_diffusion=scalar_to_matrix(dims, scalar_diffusion),
        )
        self._check_trait(dims, "homogeneous", homogeneous)
        self.homogeneous: NoneType = homogeneous
        self._check_trait(dims, "no_advection", no_advection)
        self.no_advection: NoneType = no_advection
        self._check_trait(dims, "scalar_diffusion", scalar_diffusion)
        self.scalar_diffusion: Scalar = scalar_diffusion


class HomogeneousNoAdvectionNoDiffusionPDE(
    VariableInhomogenityNoAdvectionNoDiffusionPDE,
    HomogeneousVectorAdvectionNoDiffusionPDE,
    HomogeneousNoAdvectionScalarDiffusionPDE,
):
    """
    HomogeneousNoAdvectionNoDiffusionPDE is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is is always zero (i.e. there is no diffusion). Note that the datatype is None..
    Advection is is always zero (i.e. there is no advection). Note that the datatype is None..
    And right hand side is always zero (i.e. homogenoues). Note that the datatype is None..
    """

    def __init__(
        self,
        dims: int,
        homogeneous: NoneType,
        no_advection: NoneType,
        no_diffusion: NoneType,
    ) -> None:
        """
        Create an implementation of the HomogeneousNoAdvectionNoDiffusionPDE class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        homogeneous: NoneType
            The right side of the PDE.
            is always zero (i.e. homogenoues). Note that the datatype is None.

        no_advection: NoneType
            The advection part of the PDE.
            is always zero (i.e. there is no advection). Note that the datatype is None.

        no_diffusion: NoneType
            The diffusion part of the PDE.
            is always zero (i.e. there is no diffusion). Note that the datatype is None.

        """
        VariableInhomogenityNoAdvectionNoDiffusionPDE.__init__(
            self,
            dims,
            variable_inhomogenity=constant_zero_function(dims, homogeneous),
            no_advection=identity(dims, no_advection),
            no_diffusion=identity(dims, no_diffusion),
        )
        HomogeneousVectorAdvectionNoDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            vector_advection=constant_zero_vector(dims, no_advection),
            no_diffusion=identity(dims, no_diffusion),
        )
        HomogeneousNoAdvectionScalarDiffusionPDE.__init__(
            self,
            dims,
            homogeneous=identity(dims, homogeneous),
            no_advection=identity(dims, no_advection),
            scalar_diffusion=constant_zero(dims, no_diffusion),
        )
        self._check_trait(dims, "homogeneous", homogeneous)
        self.homogeneous: NoneType = homogeneous
        self._check_trait(dims, "no_advection", no_advection)
        self.no_advection: NoneType = no_advection
        self._check_trait(dims, "no_diffusion", no_diffusion)
        self.no_diffusion: NoneType = no_diffusion
