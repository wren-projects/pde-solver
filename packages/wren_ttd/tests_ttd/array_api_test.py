# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import TTD, ops
from wren_ttd import array_api as xp
from wren_ttd._numpy_api import ROOT

from .common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_array_namespace(tensor: TestTensor, ttd: TestTTD) -> None:
    """Return the TTD Array API-compatible namespace."""
    namespace = ttd.__array_namespace__()

    assert namespace is xp
    assert namespace.__array_api_version__ == np.__array_api_version__
    assert_default_epsilon(namespace.add(ttd, ttd), tensor * 2, scale=tensor)


def test_array_namespace_versions() -> None:
    """Accept versions supported by NumPy's Array API namespace."""
    ttd = TTD.ones((2, 3))

    assert ttd.__array_namespace__(api_version="2023.12") is xp
    assert ttd.__array_namespace__(api_version=np.__array_api_version__) is xp


def test_array_namespace_rejects_unsupported_version() -> None:
    """Reject Array API versions unsupported by NumPy."""
    with pytest.raises(ValueError, match="not supported"):
        _ = TTD.ones((2, 3)).__array_namespace__(api_version="unsupported")


def test_namespace_delegates_to_numpy() -> None:
    """Support operations directly and fall back to NumPy for the rest."""
    ttd = TTD([[1.0, 2.0], [3.0, 4.0]])

    assert xp.asarray(ttd).shape == (2, 2)
    assert_default_epsilon(xp.sum(ttd, axis=0), np.array([4.0, 6.0]))
    assert_default_epsilon(xp.mean(ttd, axis=1), np.array([1.5, 3.5]))
    assert xp.expand_dims(ttd, 0).shape == (1, 2, 2)
    assert xp.matrix_transpose(ttd).shape == (2, 2)
    assert xp.permute_dims(ttd, (1, 0)).shape == (2, 2)
    np.testing.assert_allclose(
        xp.linalg.vector_norm(ttd), np.linalg.norm(np.asarray(ttd))
    )


def test_namespace_binds_operations_directly() -> None:
    """Bind registered operations directly instead of dispatching through NumPy."""
    assert xp.sum is ops.sum
    assert xp.add is ops.add
    assert xp.reshape is ops.reshape


def test_namespace_binds_linalg_operations_directly() -> None:
    """Bind registered linalg operations and fall back to NumPy for the rest."""
    assert xp.linalg.vector_norm is ops.vector_norm
    assert xp.linalg.matrix_norm is ops.matrix_norm
    assert xp.linalg.matrix_transpose is ops.matrix_transpose
    assert xp.linalg.norm is ops.frobenius_norm
    assert xp.linalg.det is np.linalg.det
    assert "vector_norm" in dir(xp.linalg)
    assert "det" in dir(xp.linalg)


def test_registry_resolves_namespaced_operations() -> None:
    """Walk the operation registry tree by namespace."""
    namespace = ROOT.get_namespace("linalg")

    assert namespace is not None
    assert namespace.get_function("vector_norm") is ops.vector_norm
    assert ROOT.get_operation("linalg.matrix_norm") is ops.matrix_norm
    assert ROOT.get_operation("linalg.norm") is ops.frobenius_norm
    assert ROOT.get_operations("linalg")["matrix_transpose"] is ops.matrix_transpose


def test_implements_registers_both_protocols() -> None:
    """Register one implementation for both dispatch protocols."""
    assert ROOT.get_function("vecdot") is ops.vecdot
    assert ROOT.get_ufunc("vecdot") is ops.vecdot

    with pytest.raises(ValueError, match="at least one"):
        _ = ROOT.implements("invalid")


def test_namespace_vecdot() -> None:
    """Compute vector dot products through the namespace."""
    a = TTD([1.0, 2.0, 3.0])
    b = TTD([4.0, 5.0, 6.0])

    np.testing.assert_allclose(xp.vecdot(a, b), 32.0)


def test_constructors_return_ttds() -> None:
    """Return TTDs from the shape-based constructors."""
    assert isinstance(xp.zeros((2, 3)), TTD)
    assert isinstance(xp.ones(3), TTD)
    assert isinstance(xp.empty((2,)), TTD)
    assert isinstance(xp.full((2, 2), 3.5), TTD)
    assert isinstance(xp.arange(3), TTD)
    assert isinstance(xp.eye(2), TTD)
    assert isinstance(xp.linspace(0, 1, 3), TTD)


def test_constructor_values() -> None:
    """Match NumPy values for the shape-based constructors."""
    assert_default_epsilon(xp.zeros((2, 3)), np.zeros((2, 3)))
    assert_default_epsilon(xp.ones((2, 3)), np.ones((2, 3)))
    assert_default_epsilon(xp.empty((2, 3)), np.zeros((2, 3)))
    assert_default_epsilon(xp.full((2, 2), 3.5), np.full((2, 2), 3.5))
    assert_default_epsilon(xp.arange(4), np.arange(4, dtype=np.float64))
    assert_default_epsilon(xp.arange(1, 6, 2), np.arange(1, 6, 2, dtype=np.float64))
    assert_default_epsilon(xp.eye(3), np.eye(3))
    assert_default_epsilon(xp.eye(3, 4, k=1), np.eye(3, 4, k=1))
    assert_default_epsilon(xp.linspace(0, 1, 5), np.linspace(0, 1, 5))


def test_constructor_dtypes() -> None:
    """Honour the requested dtype in the constructors."""
    assert xp.zeros((2, 2), dtype=np.float32).dtype == np.float32
    assert xp.ones(3, dtype=np.float32).dtype == np.float32
    assert xp.full((2,), 1, dtype=np.float32).dtype == np.float32
    assert xp.arange(3, dtype=np.float32).dtype == np.float32
    assert xp.eye(2, dtype=np.float32).dtype == np.float32
    assert xp.linspace(0, 1, 3, dtype=np.float32).dtype == np.float32


def test_constructors_reject_rank_zero() -> None:
    """Reject rank-zero shapes because TTD cannot represent scalars."""
    for constructor in (xp.zeros, xp.ones, xp.empty):
        with pytest.raises(NotImplementedError, match="at least one dimension"):
            _ = constructor(())

    with pytest.raises(NotImplementedError, match="at least one dimension"):
        _ = xp.full((), 1.0)


def test_asarray_from_ttd() -> None:
    """Return the input TTD when no conversion is needed."""
    ttd = TTD([[1.0, 2.0], [3.0, 4.0]])

    assert xp.asarray(ttd) is ttd
    assert xp.asarray(ttd, copy=False) is ttd


def test_asarray_copy_semantics() -> None:
    """Copy only when requested."""
    ttd = TTD([[1.0, 2.0], [3.0, 4.0]])

    copied = xp.asarray(ttd, copy=True)
    assert copied is not ttd
    assert_default_epsilon(copied, ttd)

    with pytest.raises(ValueError, match="avoid copy"):
        _ = xp.asarray(np.ones((2, 2)), copy=False)


def test_asarray_dtype_and_arrays() -> None:
    """Convert array-like input and cast to the requested dtype."""
    array = np.array([[1.0, 2.0], [3.0, 4.0]])

    converted = xp.asarray(array)
    assert isinstance(converted, TTD)
    assert_default_epsilon(converted, array)

    cast = xp.asarray(array, dtype=np.float32)
    assert cast.dtype == np.float32
    np.testing.assert_allclose(np.asarray(cast), array, rtol=1e-6)

    cast_ttd = xp.asarray(converted, dtype=np.float32)
    assert cast_ttd.dtype == np.float32
    np.testing.assert_allclose(np.asarray(cast_ttd), array, rtol=1e-6)


def test_asarray_rejects_rank_zero() -> None:
    """Reject scalar input because TTD cannot represent scalars."""
    with pytest.raises(NotImplementedError, match="at least one dimension"):
        _ = xp.asarray(3.0)

    with pytest.raises(NotImplementedError, match="at least one dimension"):
        _ = xp.asarray(np.array(3.0))


def test_constructors_reject_unsupported_device() -> None:
    """Reject devices other than the CPU."""
    for constructor in (xp.zeros, xp.ones, xp.empty):
        with pytest.raises(ValueError, match="CPU"):
            _ = constructor((2,), device="gpu")

    with pytest.raises(ValueError, match="CPU"):
        _ = xp.asarray([1.0, 2.0], device="gpu")


def test_result_type() -> None:
    """Promote TTD dtypes like NumPy dtypes."""
    ttd = xp.asarray([1.0, 2.0], dtype=np.float32)

    assert xp.result_type(ttd, np.float64) == np.dtype(np.float64)
    assert xp.result_type(ttd, ttd) == np.dtype(np.float32)
    assert xp.result_type(np.int32, np.float32) == np.result_type(np.int32, np.float32)


def test_can_cast() -> None:
    """Report casting rules for TTD dtypes."""
    ttd = xp.asarray([1.0, 2.0], dtype=np.float32)

    assert xp.can_cast(ttd, np.float64)
    assert not xp.can_cast(np.float64, ttd)


def test_finfo_and_iinfo() -> None:
    """Expose NumPy machine limits."""
    ttd = xp.asarray([1.0, 2.0], dtype=np.float32)

    assert xp.finfo(ttd).eps == np.finfo(np.float32).eps
    assert xp.finfo(np.float64).bits == 64
    assert xp.iinfo(np.int8).max == 127


def test_isdtype() -> None:
    """Classify dtypes using Array API kind names."""
    ttd = xp.asarray([1.0, 2.0], dtype=np.float32)

    assert xp.isdtype(ttd, "real floating")
    assert not xp.isdtype(ttd, "integral")
    assert xp.isdtype(np.int32, "integral")
