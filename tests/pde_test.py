import inspect
import types
from typing import Any, TypeAlias, TypeAliasType

import numpy

from pde_solver import PDE
from pde_solver.pde_types import (
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
    return [cls for name, cls in all_classes if cls.__module__ == module.__name__]


all_pde_classes = get_defined_classes(PDE)


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
    dummy_value_by_type: dict[type | TypeAliasType, Any] = {
        int: 3,
        Scalar: 3,
        Vector: numpy.arange(3),
        Matrix: numpy.arange(9).reshape((3, 3)),
        ScalarFunction: lambda _: 0,
        VectorFunction: lambda _: numpy.arange(3),
        MatrixFunction: lambda _: numpy.arange(9).reshape((3, 3)),
        types.NoneType: None,
    }
    for element in all_pde_classes:
        arg_names = [
            (name, param.annotation)
            for name, param in inspect.signature(element.__init__).parameters.items()
            if name != "self"
        ]
        args = {name: dummy_value_by_type[annotation] for name, annotation in arg_names}
        element(**args)


test_all_pdes_can_be_constructed()
