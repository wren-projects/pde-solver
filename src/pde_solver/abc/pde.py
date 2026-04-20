from abc import ABC, abstractmethod
from typing import override


class PDE(ABC):
    """
    Abstract base class for PDEs.

    On its own, this class is empty and pointless, as all PDEs behave vastly differently
    and have completely different interfaces. The only reason this ABC exists is to
    allow us to type hint solvers.
    """
    @abstractmethod
    def __init__(self) -> None:
        ...

    @override
    def __str__(self) -> str:
        return f"This is an implementation of {self.__class__.__name__} PDE."
