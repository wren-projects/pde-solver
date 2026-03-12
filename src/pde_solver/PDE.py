from typing import Any
from pde_solver.our_types import TimeFunction, Vector, NoneType, Scalar, Function, Matrix
def constant_to_function[T: Scalar | Vector | Matrix](
    dim: int, value: T
) -> Function[T]:
    """Transform scalar into a vector."""
    return lambda _: value

def scalar_to_matrix(dim: int, value: Scalar) -> Matrix:
    """Transform scalar into a vector."""
    return value * np.eye(dim, dtype=DType)

def keep_same[T](dim: int, value: T) -> T:
    return value

def constant_zero_vector(dim: int, value: None) -> Vector:
    return np.array([0] * dim, dtype=DType)

def constant_zero(dim: int, value: None) -> Scalar:
    return DType(0)

def constant_zero_function(dim: int, value: None) -> Function[Scalar]:
    return lambda _: DType(0)


class VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE ():
    def __init__(self, dims: int, VariableInhomoginity: Function, VariableVectorAdvection: Function, VariableMatrixDiffusion: Function):
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('VariableMatrixDiffusion', VariableMatrixDiffusion)

    def _set_trait(self, name:Any, value:Any):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+value)
        setattr(self,name, value)
        

class VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE (VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VariableVectorAdvection: Function, MatrixDiffusion: Matrix):
        VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VariableVectorAdvection = keep_same(dims, VariableVectorAdvection), VariableMatrixDiffusion = constant_to_function(dims, MatrixDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('MatrixDiffusion', MatrixDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE (VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VariableVectorAdvection: Function, ScalarDiffusion: Scalar):
        VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VariableVectorAdvection = keep_same(dims, VariableVectorAdvection), MatrixDiffusion = scalar_to_matrix(dims, ScalarDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('ScalarDiffusion', ScalarDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class VariableInhomoginityVariableVectorAdvectionNoDiffusionPDE (VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VariableVectorAdvection: Function, NoDiffusion: <class 'NoneType'>):
        VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VariableVectorAdvection = keep_same(dims, VariableVectorAdvection), ScalarDiffusion = constant_zero(dims, NoDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('NoDiffusion', NoDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE (VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VectorAdvection: Vector, VariableMatrixDiffusion: Function):
        VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), VariableMatrixDiffusion = keep_same(dims, VariableMatrixDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('VariableMatrixDiffusion', VariableMatrixDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class VariableInhomoginityVectorAdvectionMatrixDiffusionPDE (VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE, VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VectorAdvection: Vector, MatrixDiffusion: Matrix):
        VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), MatrixDiffusion = keep_same(dims, MatrixDiffusion))
        VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VectorAdvection = keep_same(dims, VectorAdvection), VariableMatrixDiffusion = constant_to_function(dims, MatrixDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('MatrixDiffusion', MatrixDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class VariableInhomoginityVectorAdvectionScalarDiffusionPDE (VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE, VariableInhomoginityVectorAdvectionMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VectorAdvection: Vector, ScalarDiffusion: Scalar):
        VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), ScalarDiffusion = keep_same(dims, ScalarDiffusion))
        VariableInhomoginityVectorAdvectionMatrixDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VectorAdvection = keep_same(dims, VectorAdvection), MatrixDiffusion = scalar_to_matrix(dims, ScalarDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('ScalarDiffusion', ScalarDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class VariableInhomoginityVectorAdvectionNoDiffusionPDE (VariableInhomoginityVariableVectorAdvectionNoDiffusionPDE, VariableInhomoginityVectorAdvectionScalarDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, VectorAdvection: Vector, NoDiffusion: <class 'NoneType'>):
        VariableInhomoginityVariableVectorAdvectionNoDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), NoDiffusion = keep_same(dims, NoDiffusion))
        VariableInhomoginityVectorAdvectionScalarDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VectorAdvection = keep_same(dims, VectorAdvection), ScalarDiffusion = constant_zero(dims, NoDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('NoDiffusion', NoDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class VariableInhomoginityNoAdvectionVariableMatrixDiffusionPDE (VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, NoAdvection: <class 'NoneType'>, VariableMatrixDiffusion: Function):
        VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), VariableMatrixDiffusion = keep_same(dims, VariableMatrixDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('VariableMatrixDiffusion', VariableMatrixDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class VariableInhomoginityNoAdvectionMatrixDiffusionPDE (VariableInhomoginityVectorAdvectionMatrixDiffusionPDE, VariableInhomoginityNoAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, NoAdvection: <class 'NoneType'>, MatrixDiffusion: Matrix):
        VariableInhomoginityVectorAdvectionMatrixDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), MatrixDiffusion = keep_same(dims, MatrixDiffusion))
        VariableInhomoginityNoAdvectionVariableMatrixDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), NoAdvection = keep_same(dims, NoAdvection), VariableMatrixDiffusion = constant_to_function(dims, MatrixDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('MatrixDiffusion', MatrixDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class VariableInhomoginityNoAdvectionScalarDiffusionPDE (VariableInhomoginityVectorAdvectionScalarDiffusionPDE, VariableInhomoginityNoAdvectionMatrixDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, NoAdvection: <class 'NoneType'>, ScalarDiffusion: Scalar):
        VariableInhomoginityVectorAdvectionScalarDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), ScalarDiffusion = keep_same(dims, ScalarDiffusion))
        VariableInhomoginityNoAdvectionMatrixDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), NoAdvection = keep_same(dims, NoAdvection), MatrixDiffusion = scalar_to_matrix(dims, ScalarDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('ScalarDiffusion', ScalarDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class VariableInhomoginityNoAdvectionNoDiffusionPDE (VariableInhomoginityVectorAdvectionNoDiffusionPDE, VariableInhomoginityNoAdvectionScalarDiffusionPDE):
    def __init__(self, dims: int, VariableInhomoginity: Function, NoAdvection: <class 'NoneType'>, NoDiffusion: <class 'NoneType'>):
        VariableInhomoginityVectorAdvectionNoDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), NoDiffusion = keep_same(dims, NoDiffusion))
        VariableInhomoginityNoAdvectionScalarDiffusionPDE.__init__(self, VariableInhomoginity = keep_same(dims, VariableInhomoginity), NoAdvection = keep_same(dims, NoAdvection), ScalarDiffusion = constant_zero(dims, NoDiffusion))
        self._set_trait('VariableInhomoginity', VariableInhomoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('NoDiffusion', NoDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class HomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE (VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: <class 'NoneType'>, VariableVectorAdvection: Function, VariableMatrixDiffusion: Function):
        VariableInhomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, VariableInhomoginity = constant_zero_function(dims, Homoginity), VariableVectorAdvection = keep_same(dims, VariableVectorAdvection), VariableMatrixDiffusion = keep_same(dims, VariableMatrixDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('VariableMatrixDiffusion', VariableMatrixDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class HomoginityVariableVectorAdvectionMatrixDiffusionPDE (VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE, HomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: <class 'NoneType'>, VariableVectorAdvection: Function, MatrixDiffusion: Matrix):
        VariableInhomoginityVariableVectorAdvectionMatrixDiffusionPDE.__init__(self, VariableInhomoginity = constant_zero_function(dims, Homoginity), VariableVectorAdvection = keep_same(dims, VariableVectorAdvection), MatrixDiffusion = keep_same(dims, MatrixDiffusion))
        HomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VariableVectorAdvection = keep_same(dims, VariableVectorAdvection), VariableMatrixDiffusion = constant_to_function(dims, MatrixDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('MatrixDiffusion', MatrixDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class HomoginityVariableVectorAdvectionScalarDiffusionPDE (VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE, HomoginityVariableVectorAdvectionMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: <class 'NoneType'>, VariableVectorAdvection: Function, ScalarDiffusion: Scalar):
        VariableInhomoginityVariableVectorAdvectionScalarDiffusionPDE.__init__(self, VariableInhomoginity = constant_zero_function(dims, Homoginity), VariableVectorAdvection = keep_same(dims, VariableVectorAdvection), ScalarDiffusion = keep_same(dims, ScalarDiffusion))
        HomoginityVariableVectorAdvectionMatrixDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VariableVectorAdvection = keep_same(dims, VariableVectorAdvection), MatrixDiffusion = scalar_to_matrix(dims, ScalarDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('ScalarDiffusion', ScalarDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class HomoginityVariableVectorAdvectionNoDiffusionPDE (VariableInhomoginityVariableVectorAdvectionNoDiffusionPDE, HomoginityVariableVectorAdvectionScalarDiffusionPDE):
    def __init__(self, dims: int, Homoginity: <class 'NoneType'>, VariableVectorAdvection: Function, NoDiffusion: <class 'NoneType'>):
        VariableInhomoginityVariableVectorAdvectionNoDiffusionPDE.__init__(self, VariableInhomoginity = constant_zero_function(dims, Homoginity), VariableVectorAdvection = keep_same(dims, VariableVectorAdvection), NoDiffusion = keep_same(dims, NoDiffusion))
        HomoginityVariableVectorAdvectionScalarDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VariableVectorAdvection = keep_same(dims, VariableVectorAdvection), ScalarDiffusion = constant_zero(dims, NoDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VariableVectorAdvection', VariableVectorAdvection)
        self._set_trait('NoDiffusion', NoDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class HomoginityVectorAdvectionVariableMatrixDiffusionPDE (VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE, HomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: <class 'NoneType'>, VectorAdvection: Vector, VariableMatrixDiffusion: Function):
        VariableInhomoginityVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, VariableInhomoginity = constant_zero_function(dims, Homoginity), VectorAdvection = keep_same(dims, VectorAdvection), VariableMatrixDiffusion = keep_same(dims, VariableMatrixDiffusion))
        HomoginityVariableVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), VariableMatrixDiffusion = keep_same(dims, VariableMatrixDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('VariableMatrixDiffusion', VariableMatrixDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class HomoginityVectorAdvectionMatrixDiffusionPDE (VariableInhomoginityVectorAdvectionMatrixDiffusionPDE, HomoginityVariableVectorAdvectionMatrixDiffusionPDE, HomoginityVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: <class 'NoneType'>, VectorAdvection: Vector, MatrixDiffusion: Matrix):
        VariableInhomoginityVectorAdvectionMatrixDiffusionPDE.__init__(self, VariableInhomoginity = constant_zero_function(dims, Homoginity), VectorAdvection = keep_same(dims, VectorAdvection), MatrixDiffusion = keep_same(dims, MatrixDiffusion))
        HomoginityVariableVectorAdvectionMatrixDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), MatrixDiffusion = keep_same(dims, MatrixDiffusion))
        HomoginityVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VectorAdvection = keep_same(dims, VectorAdvection), VariableMatrixDiffusion = constant_to_function(dims, MatrixDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('MatrixDiffusion', MatrixDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class HomoginityVectorAdvectionScalarDiffusionPDE (VariableInhomoginityVectorAdvectionScalarDiffusionPDE, HomoginityVariableVectorAdvectionScalarDiffusionPDE, HomoginityVectorAdvectionMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: <class 'NoneType'>, VectorAdvection: Vector, ScalarDiffusion: Scalar):
        VariableInhomoginityVectorAdvectionScalarDiffusionPDE.__init__(self, VariableInhomoginity = constant_zero_function(dims, Homoginity), VectorAdvection = keep_same(dims, VectorAdvection), ScalarDiffusion = keep_same(dims, ScalarDiffusion))
        HomoginityVariableVectorAdvectionScalarDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), ScalarDiffusion = keep_same(dims, ScalarDiffusion))
        HomoginityVectorAdvectionMatrixDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VectorAdvection = keep_same(dims, VectorAdvection), MatrixDiffusion = scalar_to_matrix(dims, ScalarDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('ScalarDiffusion', ScalarDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class HomoginityVectorAdvectionNoDiffusionPDE (VariableInhomoginityVectorAdvectionNoDiffusionPDE, HomoginityVariableVectorAdvectionNoDiffusionPDE, HomoginityVectorAdvectionScalarDiffusionPDE):
    def __init__(self, dims: int, Homoginity: <class 'NoneType'>, VectorAdvection: Vector, NoDiffusion: <class 'NoneType'>):
        VariableInhomoginityVectorAdvectionNoDiffusionPDE.__init__(self, VariableInhomoginity = constant_zero_function(dims, Homoginity), VectorAdvection = keep_same(dims, VectorAdvection), NoDiffusion = keep_same(dims, NoDiffusion))
        HomoginityVariableVectorAdvectionNoDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VariableVectorAdvection = constant_to_function(dims, VectorAdvection), NoDiffusion = keep_same(dims, NoDiffusion))
        HomoginityVectorAdvectionScalarDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VectorAdvection = keep_same(dims, VectorAdvection), ScalarDiffusion = constant_zero(dims, NoDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('VectorAdvection', VectorAdvection)
        self._set_trait('NoDiffusion', NoDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class HomoginityNoAdvectionVariableMatrixDiffusionPDE (VariableInhomoginityNoAdvectionVariableMatrixDiffusionPDE, HomoginityVectorAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: <class 'NoneType'>, NoAdvection: <class 'NoneType'>, VariableMatrixDiffusion: Function):
        VariableInhomoginityNoAdvectionVariableMatrixDiffusionPDE.__init__(self, VariableInhomoginity = constant_zero_function(dims, Homoginity), NoAdvection = keep_same(dims, NoAdvection), VariableMatrixDiffusion = keep_same(dims, VariableMatrixDiffusion))
        HomoginityVectorAdvectionVariableMatrixDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), VariableMatrixDiffusion = keep_same(dims, VariableMatrixDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('VariableMatrixDiffusion', VariableMatrixDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class HomoginityNoAdvectionMatrixDiffusionPDE (VariableInhomoginityNoAdvectionMatrixDiffusionPDE, HomoginityVectorAdvectionMatrixDiffusionPDE, HomoginityNoAdvectionVariableMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: <class 'NoneType'>, NoAdvection: <class 'NoneType'>, MatrixDiffusion: Matrix):
        VariableInhomoginityNoAdvectionMatrixDiffusionPDE.__init__(self, VariableInhomoginity = constant_zero_function(dims, Homoginity), NoAdvection = keep_same(dims, NoAdvection), MatrixDiffusion = keep_same(dims, MatrixDiffusion))
        HomoginityVectorAdvectionMatrixDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), MatrixDiffusion = keep_same(dims, MatrixDiffusion))
        HomoginityNoAdvectionVariableMatrixDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), NoAdvection = keep_same(dims, NoAdvection), VariableMatrixDiffusion = constant_to_function(dims, MatrixDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('MatrixDiffusion', MatrixDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class HomoginityNoAdvectionScalarDiffusionPDE (VariableInhomoginityNoAdvectionScalarDiffusionPDE, HomoginityVectorAdvectionScalarDiffusionPDE, HomoginityNoAdvectionMatrixDiffusionPDE):
    def __init__(self, dims: int, Homoginity: <class 'NoneType'>, NoAdvection: <class 'NoneType'>, ScalarDiffusion: Scalar):
        VariableInhomoginityNoAdvectionScalarDiffusionPDE.__init__(self, VariableInhomoginity = constant_zero_function(dims, Homoginity), NoAdvection = keep_same(dims, NoAdvection), ScalarDiffusion = keep_same(dims, ScalarDiffusion))
        HomoginityVectorAdvectionScalarDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), ScalarDiffusion = keep_same(dims, ScalarDiffusion))
        HomoginityNoAdvectionMatrixDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), NoAdvection = keep_same(dims, NoAdvection), MatrixDiffusion = scalar_to_matrix(dims, ScalarDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('ScalarDiffusion', ScalarDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        

class HomoginityNoAdvectionNoDiffusionPDE (VariableInhomoginityNoAdvectionNoDiffusionPDE, HomoginityVectorAdvectionNoDiffusionPDE, HomoginityNoAdvectionScalarDiffusionPDE):
    def __init__(self, dims: int, Homoginity: <class 'NoneType'>, NoAdvection: <class 'NoneType'>, NoDiffusion: <class 'NoneType'>):
        VariableInhomoginityNoAdvectionNoDiffusionPDE.__init__(self, VariableInhomoginity = constant_zero_function(dims, Homoginity), NoAdvection = keep_same(dims, NoAdvection), NoDiffusion = keep_same(dims, NoDiffusion))
        HomoginityVectorAdvectionNoDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), VectorAdvection = constant_zero_vector(dims, NoAdvection), NoDiffusion = keep_same(dims, NoDiffusion))
        HomoginityNoAdvectionScalarDiffusionPDE.__init__(self, Homoginity = keep_same(dims, Homoginity), NoAdvection = keep_same(dims, NoAdvection), ScalarDiffusion = constant_zero(dims, NoDiffusion))
        self._set_trait('Homoginity', Homoginity)
        self._set_trait('NoAdvection', NoAdvection)
        self._set_trait('NoDiffusion', NoDiffusion)

    def _set_trait(self, name, value):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError("PDE structure latice is disrupted! Found value of attribute "+name+" to be "+self.name+" when it should be "+self.value)
        setattr(self,name, value)
        
