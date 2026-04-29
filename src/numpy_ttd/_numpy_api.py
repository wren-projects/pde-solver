from __future__ import annotations

from collections.abc import Callable
from typing import Any

type AnyCallable = Callable[..., Any]

HANDLED_UFUNCS: dict[str, AnyCallable] = {}
HANDLED_FUNCTIONS: dict[str, AnyCallable] = {}


def implements_ufunc[F: AnyCallable](name: str) -> Callable[[F], F]:
    """Register an `__array_ufunc__` implementation for TTD objects."""

    def decorator(func: F) -> F:
        HANDLED_UFUNCS[name] = func
        return func

    return decorator


def implements_function[F: AnyCallable](name: str) -> Callable[[F], F]:
    """Register an `__array_function__` implementation for TTD objects."""

    def decorator(func: F) -> F:
        HANDLED_FUNCTIONS[name] = func
        return func

    return decorator
