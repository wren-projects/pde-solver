# pyright: reportUnsafeMultipleInheritance=false
# pyright reportMissingSuperCall=false
import inspect
import itertools
from collections.abc import Callable, Iterable
from multiprocessing import parent_process
from typing import Any, TypeAliasType, cast
from types import NoneType
import numpy as np
from pde_solver.our_types import DType, Function, Matrix, Scalar, TimeFunction, Vector

def constant_zero_vector(dim: int, value: None) -> Vector:
    """Cast None into Vector."""
    return np.zeros(dim, dtype=DType)

def constant_zero(dim: int, value: None) -> Scalar:
    """Cast None into Scalar."""
    return DType(0)

def indentity[T](dim: int, value: T) -> T:
    """Cast Any value into itself."""
    return value

def constant_zero_function(dim: int, value: None) -> Function[Scalar]:
    """Cast None into Scalar."""
    return lambda _: DType(0)

def constant_to_function[T: Scalar | Vector | Matrix](
    dim: int, value: T
) -> Function[T]:
    """Transform scalar into a vector."""
    return lambda _: value

def scalar_to_matrix(dim: int, value: Scalar) -> Matrix:
    """Transform scalar into a vector."""
    return value * np.eye(dim, dtype=DType)


class VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE ():
    def __init__(self, dims: int, VariableInhomoginity: Function, VariableVectorAdvection: Function, VariableMatrixDiffusion: Function):
        
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('VariableMatrixDiffusion', VariableMatrixDiffusion)

    def _set_trait(self, name: str, value: str):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError(
                r"PDE structure latice is disrupted! Found value of attribute"
                f"{name} to be {getattr(self, name)} when it should be"
                f"{getattr(self, value)}"
            )

        setattr(self,name, value)
        

class VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE (VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VariableVectorAdvection: Function, MatrixDiffusion: Matrix):
        VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VariableVectorAdvection = indentity(dims, VariableVectorAdvection), VariableMatrixDiffusion = constant_to_function(dims, MatrixDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('MatrixDiffusion', MatrixDiffusion)

class VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE (VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VariableVectorAdvection: Function, ScalarDiffusion: Scalar):
        VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VariableVectorAdvection = indentity(dims, VariableVectorAdvection), MatrixDiffusion = scalar_to_matrix(dims, ScalarDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('ScalarDiffusion', ScalarDiffusion)

class VariableInhomoginityVariableVectorAdvectionNoDiffusionPDE (VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VariableVectorAdvection: Function, NoDiffusion: NoneType):
        VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VariableVectorAdvection = indentity(dims, VariableVectorAdvection), ScalarDiffusion = constant_zero(dims, NoDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('NoDiffusion', NoDiffusion)

class VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE (VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VectorAdvection: Vector, VariableMatrixDiffusion: Function):
        VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), VariableMatrixDiffusion = indentity(dims, VariableMatrixDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('VariableMatrixDiffusion', VariableMatrixDiffusion)

class VariableInhomoginityVectorAdvectionMatrixDiffusionPDE (VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE, VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VectorAdvection: Vector, MatrixDiffusion: Matrix):
        VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), MatrixDiffusion = indentity(dims, MatrixDiffusion))
        VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VectorAdvection = indentity(dims, VectorAdvection), VariableMatrixDiffusion = constant_to_function(dims, MatrixDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('MatrixDiffusion', MatrixDiffusion)

class VariableInhomoginityVectorAdvectionScalarDiffusionPDE (VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE, VariableInhomoginityVectorAdvectionMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VectorAdvection: Vector, ScalarDiffusion: Scalar):
        VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), ScalarDiffusion = indentity(dims, ScalarDiffusion))
        VariableInhomoginityVectorAdvectionMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VectorAdvection = indentity(dims, VectorAdvection), MatrixDiffusion = scalar_to_matrix(dims, ScalarDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('ScalarDiffusion', ScalarDiffusion)

class VariableInhomoginityVectorAdvectionNoDiffusionPDE (VariableInhomoginityVariableVectorAdvectionNoDiffusionPDE, VariableInhomoginityVectorAdvectionScalarDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VectorAdvection: Vector, NoDiffusion: NoneType):
        VariableInhomoginityVariableVectorAdvectionNoDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), NoDiffusion = indentity(dims, NoDiffusion))
        VariableInhomoginityVectorAdvectionScalarDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VectorAdvection = indentity(dims, VectorAdvection), ScalarDiffusion = constant_zero(dims, NoDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('NoDiffusion', NoDiffusion)

class VariableInhomoginityNoAdvectionVariableMatrixDiffusionPDE (VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, NoAdvection: NoneType, VariableMatrixDiffusion: Function):
        VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), VariableMatrixDiffusion = indentity(dims, VariableMatrixDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('VariableMatrixDiffusion', VariableMatrixDiffusion)

class VariableInhomoginityNoAdvectionMatrixDiffusionPDE (VariableInhomoginityVectorAdvectionMatrixDiffusionPDE, VariableInhomoginityNoAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, NoAdvection: NoneType, MatrixDiffusion: Matrix):
        VariableInhomoginityVectorAdvectionMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), MatrixDiffusion = indentity(dims, MatrixDiffusion))
        VariableInhomoginityNoAdvectionVariableMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), NoAdvection = indentity(dims, NoAdvection), VariableMatrixDiffusion = constant_to_function(dims, MatrixDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('MatrixDiffusion', MatrixDiffusion)

class VariableInhomoginityNoAdvectionScalarDiffusionPDE (VariableInhomoginityVectorAdvectionScalarDiffusionPDE, VariableInhomoginityNoAdvectionMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, NoAdvection: NoneType, ScalarDiffusion: Scalar):
        VariableInhomoginityVectorAdvectionScalarDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), ScalarDiffusion = indentity(dims, ScalarDiffusion))
        VariableInhomoginityNoAdvectionMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), NoAdvection = indentity(dims, NoAdvection), MatrixDiffusion = scalar_to_matrix(dims, ScalarDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('ScalarDiffusion', ScalarDiffusion)

class VariableInhomoginityNoAdvectionNoDiffusionPDE (VariableInhomoginityVectorAdvectionNoDiffusionPDE, VariableInhomoginityNoAdvectionScalarDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, NoAdvection: NoneType, NoDiffusion: NoneType):
        VariableInhomoginityVectorAdvectionNoDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), NoDiffusion = indentity(dims, NoDiffusion))
        VariableInhomoginityNoAdvectionScalarDiffusionPDE.__init__(self, dims, VariableInhomoginity = indentity(dims, VariableInhomoginity), NoAdvection = indentity(dims, NoAdvection), ScalarDiffusion = constant_zero(dims, NoDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('NoDiffusion', NoDiffusion)

class HomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE (VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: NoneType, VariableVectorAdvection: Function, VariableMatrixDiffusion: Function):
        VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = constant_zero_function(dims, Homoginity), VariableVectorAdvection = indentity(dims, VariableVectorAdvection), VariableMatrixDiffusion = indentity(dims, VariableMatrixDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('VariableMatrixDiffusion', VariableMatrixDiffusion)

class HomoginityVariableVectorAdvectionMatrixDiffusionPDE (VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE, HomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: NoneType, VariableVectorAdvection: Function, MatrixDiffusion: Matrix):
        VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = constant_zero_function(dims, Homoginity), VariableVectorAdvection = indentity(dims, VariableVectorAdvection), MatrixDiffusion = indentity(dims, MatrixDiffusion))
        HomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VariableVectorAdvection = indentity(dims, VariableVectorAdvection), VariableMatrixDiffusion = constant_to_function(dims, MatrixDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('MatrixDiffusion', MatrixDiffusion)

class HomoginityVariableVectorAdvectionScalarDiffusionPDE (VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE, HomoginityVariableVectorAdvectionMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: NoneType, VariableVectorAdvection: Function, ScalarDiffusion: Scalar):
        VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE.__init__(self, dims, VariableInhomoginity = constant_zero_function(dims, Homoginity), VariableVectorAdvection = indentity(dims, VariableVectorAdvection), ScalarDiffusion = indentity(dims, ScalarDiffusion))
        HomoginityVariableVectorAdvectionMatrixDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VariableVectorAdvection = indentity(dims, VariableVectorAdvection), MatrixDiffusion = scalar_to_matrix(dims, ScalarDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('ScalarDiffusion', ScalarDiffusion)

class HomoginityVariableVectorAdvectionNoDiffusionPDE (VariableInhomoginityVariableVectorAdvectionNoDiffusionPDE, HomoginityVariableVectorAdvectionScalarDiffusionPDE):
    def __init__(self, dims: int, Homoginity: NoneType, VariableVectorAdvection: Function, NoDiffusion: NoneType):
        VariableInhomoginityVariableVectorAdvectionNoDiffusionPDE.__init__(self, dims, VariableInhomoginity = constant_zero_function(dims, Homoginity), VariableVectorAdvection = indentity(dims, VariableVectorAdvection), NoDiffusion = indentity(dims, NoDiffusion))
        HomoginityVariableVectorAdvectionScalarDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VariableVectorAdvection = indentity(dims, VariableVectorAdvection), ScalarDiffusion = constant_zero(dims, NoDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('NoDiffusion', NoDiffusion)

class HomoginityVectorAdvectionVariableMatrixDiffusionPDE (VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE, HomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: NoneType, VectorAdvection: Vector, VariableMatrixDiffusion: Function):
        VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = constant_zero_function(dims, Homoginity), VectorAdvection = indentity(dims, VectorAdvection), VariableMatrixDiffusion = indentity(dims, VariableMatrixDiffusion))
        HomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), VariableMatrixDiffusion = indentity(dims, VariableMatrixDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('VariableMatrixDiffusion', VariableMatrixDiffusion)

class HomoginityVectorAdvectionMatrixDiffusionPDE (VariableInhomoginityVectorAdvectionMatrixDiffusionPDE, HomoginityVariableVectorAdvectionMatrixDiffusionPDE, HomoginityVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: NoneType, VectorAdvection: Vector, MatrixDiffusion: Matrix):
        VariableInhomoginityVectorAdvectionMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = constant_zero_function(dims, Homoginity), VectorAdvection = indentity(dims, VectorAdvection), MatrixDiffusion = indentity(dims, MatrixDiffusion))
        HomoginityVariableVectorAdvectionMatrixDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), MatrixDiffusion = indentity(dims, MatrixDiffusion))
        HomoginityVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VectorAdvection = indentity(dims, VectorAdvection), VariableMatrixDiffusion = constant_to_function(dims, MatrixDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('MatrixDiffusion', MatrixDiffusion)

class HomoginityVectorAdvectionScalarDiffusionPDE (VariableInhomoginityVectorAdvectionScalarDiffusionPDE, HomoginityVariableVectorAdvectionScalarDiffusionPDE, HomoginityVectorAdvectionMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: NoneType, VectorAdvection: Vector, ScalarDiffusion: Scalar):
        VariableInhomoginityVectorAdvectionScalarDiffusionPDE.__init__(self, dims, VariableInhomoginity = constant_zero_function(dims, Homoginity), VectorAdvection = indentity(dims, VectorAdvection), ScalarDiffusion = indentity(dims, ScalarDiffusion))
        HomoginityVariableVectorAdvectionScalarDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), ScalarDiffusion = indentity(dims, ScalarDiffusion))
        HomoginityVectorAdvectionMatrixDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VectorAdvection = indentity(dims, VectorAdvection), MatrixDiffusion = scalar_to_matrix(dims, ScalarDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('ScalarDiffusion', ScalarDiffusion)

class HomoginityVectorAdvectionNoDiffusionPDE (VariableInhomoginityVectorAdvectionNoDiffusionPDE, HomoginityVariableVectorAdvectionNoDiffusionPDE, HomoginityVectorAdvectionScalarDiffusionPDE):
    def __init__(self, dims: int, Homoginity: NoneType, VectorAdvection: Vector, NoDiffusion: NoneType):
        VariableInhomoginityVectorAdvectionNoDiffusionPDE.__init__(self, dims, VariableInhomoginity = constant_zero_function(dims, Homoginity), VectorAdvection = indentity(dims, VectorAdvection), NoDiffusion = indentity(dims, NoDiffusion))
        HomoginityVariableVectorAdvectionNoDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), NoDiffusion = indentity(dims, NoDiffusion))
        HomoginityVectorAdvectionScalarDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VectorAdvection = indentity(dims, VectorAdvection), ScalarDiffusion = constant_zero(dims, NoDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('NoDiffusion', NoDiffusion)

class HomoginityNoAdvectionVariableMatrixDiffusionPDE (VariableInhomoginityNoAdvectionVariableMatrixDiffusionPDE, HomoginityVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: NoneType, NoAdvection: NoneType, VariableMatrixDiffusion: Function):
        VariableInhomoginityNoAdvectionVariableMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = constant_zero_function(dims, Homoginity), NoAdvection = indentity(dims, NoAdvection), VariableMatrixDiffusion = indentity(dims, VariableMatrixDiffusion))
        HomoginityVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), VariableMatrixDiffusion = indentity(dims, VariableMatrixDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('VariableMatrixDiffusion', VariableMatrixDiffusion)

class HomoginityNoAdvectionMatrixDiffusionPDE (VariableInhomoginityNoAdvectionMatrixDiffusionPDE, HomoginityVectorAdvectionMatrixDiffusionPDE, HomoginityNoAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: NoneType, NoAdvection: NoneType, MatrixDiffusion: Matrix):
        VariableInhomoginityNoAdvectionMatrixDiffusionPDE.__init__(self, dims, VariableInhomoginity = constant_zero_function(dims, Homoginity), NoAdvection = indentity(dims, NoAdvection), MatrixDiffusion = indentity(dims, MatrixDiffusion))
        HomoginityVectorAdvectionMatrixDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), MatrixDiffusion = indentity(dims, MatrixDiffusion))
        HomoginityNoAdvectionVariableMatrixDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), NoAdvection = indentity(dims, NoAdvection), VariableMatrixDiffusion = constant_to_function(dims, MatrixDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('MatrixDiffusion', MatrixDiffusion)

class HomoginityNoAdvectionScalarDiffusionPDE (VariableInhomoginityNoAdvectionScalarDiffusionPDE, HomoginityVectorAdvectionScalarDiffusionPDE, HomoginityNoAdvectionMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: NoneType, NoAdvection: NoneType, ScalarDiffusion: Scalar):
        VariableInhomoginityNoAdvectionScalarDiffusionPDE.__init__(self, dims, VariableInhomoginity = constant_zero_function(dims, Homoginity), NoAdvection = indentity(dims, NoAdvection), ScalarDiffusion = indentity(dims, ScalarDiffusion))
        HomoginityVectorAdvectionScalarDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), ScalarDiffusion = indentity(dims, ScalarDiffusion))
        HomoginityNoAdvectionMatrixDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), NoAdvection = indentity(dims, NoAdvection), MatrixDiffusion = scalar_to_matrix(dims, ScalarDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('ScalarDiffusion', ScalarDiffusion)

class HomoginityNoAdvectionNoDiffusionPDE (VariableInhomoginityNoAdvectionNoDiffusionPDE, HomoginityVectorAdvectionNoDiffusionPDE, HomoginityNoAdvectionScalarDiffusionPDE):
    def __init__(self, dims: int, Homoginity: NoneType, NoAdvection: NoneType, NoDiffusion: NoneType):
        VariableInhomoginityNoAdvectionNoDiffusionPDE.__init__(self, dims, VariableInhomoginity = constant_zero_function(dims, Homoginity), NoAdvection = indentity(dims, NoAdvection), NoDiffusion = indentity(dims, NoDiffusion))
        HomoginityVectorAdvectionNoDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), NoDiffusion = indentity(dims, NoDiffusion))
        HomoginityNoAdvectionScalarDiffusionPDE.__init__(self, dims, Homoginity = indentity(dims, Homoginity), NoAdvection = indentity(dims, NoAdvection), ScalarDiffusion = constant_zero(dims, NoDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('NoDiffusion', NoDiffusion)
