import numpy as np
import pytest
from wren_common.tests import TEST_SEED


@pytest.fixture
def seeded_default_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the default_rng inside the marked test case to use fixed seed."""
    # NOTE: we have to create a new RNG for each test case so they are independent,
    # since they run in non-deterministic order
    original = np.random.default_rng

    def seeded_rng(seed: int | None = None) -> np.random.Generator:
        return original(TEST_SEED if seed is None else seed)

    monkeypatch.setattr(np.random, "default_rng", seeded_rng)


__all__ = ["seeded_default_rng"]
