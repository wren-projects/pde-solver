from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

type AnyCallable = Callable[..., Any]

_PACKAGE_ROOT = str(Path(__file__).resolve().parent)

HANDLED_UFUNCS: dict[str, AnyCallable] = {}
HANDLED_FUNCTIONS: dict[str, AnyCallable] = {}


def implements_ufunc[F: AnyCallable](name: str) -> Callable[[F], F]:
    """
    Register an `__array_ufunc__` implementation for TTD objects.

    Parameters
    ----------
    name : str
        The NumPy ufunc name to dispatch on, e.g. ``"add"``.

    Returns
    -------
    collections.abc.Callable
        A decorator that registers the decorated function and returns it
        unchanged.

    Raises
    ------
    ValueError
        If a handler for `name` is already registered.

    """

    def decorator(func: F) -> F:
        if name in HANDLED_UFUNCS:
            raise ValueError(f"Ufunc {name} is already registered")
        HANDLED_UFUNCS[name] = func
        return func

    return decorator


def implements_function[F: AnyCallable](name: str) -> Callable[[F], F]:
    """
    Register an `__array_function__` implementation for TTD objects.

    Parameters
    ----------
    name : str
        The NumPy function name to dispatch on, e.g. ``"sum"``.

    Returns
    -------
    collections.abc.Callable
        A decorator that registers the decorated function and returns it
        unchanged.

    Raises
    ------
    ValueError
        If a handler for `name` is already registered.

    """

    def decorator(func: F) -> F:
        if name in HANDLED_FUNCTIONS:
            raise ValueError(f"Function {name} is already registered")

        HANDLED_FUNCTIONS[name] = func
        return func

    return decorator


class DenseWarning(UserWarning):
    """Warning raised by operations that fall back on dense arrays."""


def warn_dense(message: str) -> None:
    """
    Emit a :class:`DenseWarning` for the first frame outside `wren_ttd`.

    The warning is attributed to the first stack frame outside the package, so
    internal delegation (for example ``concat`` calling ``ravel``) does not
    affect the reported location.

    Parameters
    ----------
    message : str
        The warning message.

    """
    warnings.warn(
        message,
        DenseWarning,
        skip_file_prefixes=(_PACKAGE_ROOT,),
    )
