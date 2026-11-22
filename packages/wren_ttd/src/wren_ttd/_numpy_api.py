from __future__ import annotations

from collections.abc import Callable
from typing import Any

type AnyCallable = Callable[..., Any]

HANDLED_UFUNCS: dict[str, AnyCallable] = {}
HANDLED_FUNCTIONS: dict[str, AnyCallable] = {}


def implements_ufunc[F: AnyCallable](name: str) -> Callable[[F], F]:
    """Register an `__array_ufunc__` implementation for TTD objects."""

    def decorator(func: F) -> F:
        if name in HANDLED_UFUNCS:
            raise ValueError(f"Ufunc {name} is already registered")
        HANDLED_UFUNCS[name] = func
        return func

    return decorator


def implements_function[F: AnyCallable](name: str) -> Callable[[F], F]:
    """Register an `__array_function__` implementation for TTD objects."""

    def decorator(func: F) -> F:
        if name in HANDLED_FUNCTIONS:
            raise ValueError(f"Function {name} is already registered")

        HANDLED_FUNCTIONS[name] = func
        return func

    return decorator


def get_operation(name: str) -> AnyCallable | None:
    """Return the registered TTD implementation for `name`, if any."""
    return HANDLED_UFUNCS.get(name) or HANDLED_FUNCTIONS.get(name)


def get_operations(prefix: str) -> dict[str, AnyCallable]:
    """
    Return the registered operations below a dotted namespace prefix.

    The result is keyed by the attribute name below `prefix`, for example
    ``get_operations("linalg")["vector_norm"]``.

    Parameters
    ----------
    prefix : str
        The dotted namespace prefix, e.g. ``"linalg"``.

    Returns
    -------
    dict[str, collections.abc.Callable]
        The registered implementations keyed by their local name.

    """
    marker = f"{prefix}."
    return {
        name.removeprefix(marker): func
        for registry in (HANDLED_UFUNCS, HANDLED_FUNCTIONS)
        for name, func in registry.items()
        if name.startswith(marker)
    }


class DenseWarning(UserWarning):
    """Warning raised by operations that fall back on dense arrays."""
