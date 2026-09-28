import inspect
import types
from typing import Any, TypeAliasType, cast

import numpy as np
from wren_pde_solver import pde
from wren_pde_solver.pde_types import (
    DType,
    Matrix,
    MatrixFunction,
    Scalar,
    ScalarFunction,
    Vector,
    VectorFunction,
)


def get_defined_classes(module: types.ModuleType) -> list[type]:
    """Return a list of (name, class_object) tuples defined in the module."""
    all_classes = inspect.getmembers(module, inspect.isclass)

    # Filter-out classes imported from other modules
    return [cls for _, cls in all_classes if cls.__module__ == module.__name__]


all_pde_classes = get_defined_classes(pde)


def test_pde_has_smallest_element() -> None:
    """Check there is a smallest PDE - one that inherits from every other."""
    # find the smallest element
    current_smallest = all_pde_classes[0]
    for element in all_pde_classes:
        if issubclass(current_smallest, element):
            current_smallest = element

    # check it is truly the smallest element
    for element in all_pde_classes:
        if element == current_smallest:
            continue
        assert issubclass(element, current_smallest)


def test_pde_has_largerst_element() -> None:
    """Check there is a largest PDE - one that every other inherits from."""
    # find the smallest element
    current_largest = all_pde_classes[0]
    for element in all_pde_classes:
        if issubclass(element, current_largest):
            current_largest = element

    # check it is truly the smallest element
    for element in all_pde_classes:
        if element == current_largest:
            continue
        assert issubclass(current_largest, element)


def test_all_pdes_can_be_constructed() -> None:
    """Test that all PDE's constructors work."""

    def dummy_scalar_function(_: int) -> DType:
        return DType(0)

    def dummy_vector_function(_: int) -> Vector:
        return np.arange(3, dtype=DType)

    def dummy_matrix_function(_: int) -> Matrix:
        return np.arange(9, dtype=DType).reshape((3, 3))

    dummy_value_by_type: dict[type | TypeAliasType, Any] = {
        int: 3,
        Scalar: 3,
        Vector: np.arange(3),
        Matrix: np.arange(9).reshape((3, 3)),
        ScalarFunction: dummy_scalar_function,
        VectorFunction: dummy_vector_function,
        MatrixFunction: dummy_matrix_function,
        types.NoneType: None,
    }
    for element in all_pde_classes:
        arg_names = [
            (name, cast(type | TypeAliasType, param.annotation))
            for name, param in inspect.signature(element).parameters.items()
        ]
        args = {name: dummy_value_by_type[annotation] for name, annotation in arg_names}
        element(**args)


test_all_pdes_can_be_constructed()
