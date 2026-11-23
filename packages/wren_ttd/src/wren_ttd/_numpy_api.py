from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

type AnyCallable = Callable[..., Any]

_PACKAGE_ROOT = str(Path(__file__).resolve().parent)


class OperationNamespace:
    """
    Registry of TTD implementations for NumPy-compatible operations.

    A namespace groups implementations by the NumPy module they extend.
    ``linalg.function("norm")`` therefore registers ``numpy.linalg.norm``
    without a dotted string key.

    Parameters
    ----------
    name : str, optional
        The namespace name, empty for the root namespace.

    """

    name: str
    functions: dict[str, AnyCallable]
    ufuncs: dict[str, AnyCallable]
    namespaces: dict[str, OperationNamespace]

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.functions = {}
        self.ufuncs = {}
        self.namespaces = {}

    def namespace(self, name: str) -> OperationNamespace:
        """
        Return the nested namespace `name`, creating it if necessary.

        Parameters
        ----------
        name : str
            The local namespace name, e.g. ``"linalg"``.

        Returns
        -------
        OperationNamespace
            The nested namespace.

        Raises
        ------
        ValueError
            If `name` is empty or dotted.

        """
        if not name or "." in name:
            raise ValueError(f"Invalid namespace name: {name!r}")
        namespace = self.namespaces.get(name)
        if namespace is None:
            namespace = OperationNamespace(name)
            self.namespaces[name] = namespace
        return namespace

    def function[F: AnyCallable](self, name: str) -> Callable[[F], F]:
        """
        Register an `__array_function__` implementation.

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
        return self._register(self.functions, name, "Function")

    def ufunc[F: AnyCallable](self, name: str) -> Callable[[F], F]:
        """
        Register an `__array_ufunc__` implementation.

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
        return self._register(self.ufuncs, name, "Ufunc")

    def get_namespace(self, name: str) -> OperationNamespace | None:
        """Return the nested namespace `name`, if it exists."""
        return self.namespaces.get(name)

    def get_function(self, name: str) -> AnyCallable | None:
        """Return the registered `__array_function__` handler for `name`."""
        return self.functions.get(name)

    def get_ufunc(self, name: str) -> AnyCallable | None:
        """Return the registered `__array_ufunc__` handler for `name`."""
        return self.ufuncs.get(name)

    def get_operation(self, name: str) -> AnyCallable | None:
        """
        Resolve an operation by its dotted name.

        Parameters
        ----------
        name : str
            The operation name, e.g. ``"sum"`` or ``"linalg.norm"``.

        Returns
        -------
        collections.abc.Callable | None
            The registered implementation, or None if `name` is unknown.

        """
        *path, operation = name.split(".")
        namespace = self
        for part in path:
            nested = namespace.get_namespace(part)
            if nested is None:
                return None
            namespace = nested
        return namespace.get_ufunc(operation) or namespace.get_function(operation)

    def get_operations(self, prefix: str = "") -> dict[str, AnyCallable]:
        """
        Return the registered operations below a dotted prefix.

        The result is keyed by the name relative to `prefix`, for example
        ``ROOT.get_operations("linalg")["vector_norm"]``.

        Parameters
        ----------
        prefix : str, optional
            The dotted namespace prefix, empty for the whole registry.

        Returns
        -------
        dict[str, collections.abc.Callable]
            The registered implementations keyed by their local name.

        """
        namespace = self
        for part in filter(None, prefix.split(".")):
            nested = namespace.get_namespace(part)
            if nested is None:
                return {}
            namespace = nested
        return namespace._flatten()

    def _flatten(self) -> dict[str, AnyCallable]:
        operations = {**self.ufuncs, **self.functions}
        for name, namespace in self.namespaces.items():
            operations.update(
                {
                    f"{name}.{operation}": func
                    for operation, func in namespace._flatten().items()
                }
            )
        return operations

    @staticmethod
    def _register[F: AnyCallable](
        registry: dict[str, AnyCallable], name: str, kind: str
    ) -> Callable[[F], F]:
        if not name or "." in name:
            raise ValueError(f"Invalid {kind.lower()} name: {name!r}")

        def decorator(func: F) -> F:
            if name in registry:
                raise ValueError(f"{kind} {name!r} is already registered")
            registry[name] = func
            return func

        return decorator


ROOT = OperationNamespace()
linalg = ROOT.namespace("linalg")


def implements_ufunc[F: AnyCallable](name: str) -> Callable[[F], F]:
    """
    Register an `__array_ufunc__` implementation in the root namespace.

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
    return ROOT.ufunc(name)


def implements_function[F: AnyCallable](name: str) -> Callable[[F], F]:
    """
    Register an `__array_function__` implementation in the root namespace.

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
    return ROOT.function(name)


def get_operation(name: str) -> AnyCallable | None:
    """Return the registered TTD implementation for `name`, if any."""
    return ROOT.get_operation(name)


def get_operations(prefix: str = "") -> dict[str, AnyCallable]:
    """Return the registered operations below a dotted namespace prefix."""
    return ROOT.get_operations(prefix)


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
