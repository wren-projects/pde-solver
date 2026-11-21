# TTD capability matrix

This matrix describes the current `matmul` branch after Phase 2 of the
[parity plan](ttd-numpy-parity-plan.md). It is a snapshot, not a promise that
every listed function is exact: see the notes on approximation and dense
fallbacks below.

## NumPy dispatch

Functions below work when called as `np.<name>(ttd, ...)` unless noted
otherwise.

| Area | Supported | Notes |
|---|---|---|
| Elementwise | `add`, `subtract`, `multiply`, `divide`, `negative`, `positive`, `absolute`, `power`, `matmul` | `divide` supports a scalar divisor only. `power` supports integer-like scalar exponents. `absolute` densifies; negative exponents densify. |
| Rounding | `round`, `around` | Element-wise. Densifies internally. `TTD.round`/`rounded` are legacy names for rank truncation; prefer `TTD.truncate`/`truncated`. |
| Reductions | `sum`, `mean`, `cumsum`, `trace` | Full reductions without `keepdims` return a NumPy scalar; `keepdims=True` returns a TTD of shape `(1,) * ndim`. `cumsum(axis=None)` flattens first. `trace` supports `offset=0` only. |
| Manipulation | `expand_dims`, `squeeze`, `diff`, `flip`, `roll`, `moveaxis`, `transpose`/`permute_dims`, `swapaxes`, `matrix_transpose`, `reshape`, `ravel`, `broadcast_to`, `stack`, `pad` | `roll` requires an explicit `axis`. `pad` supports only uniform widths and a scalar `constant_values`. `ravel` and full-size `reshape` effectively densify. |
| Selection and joining | `take`, `repeat`, `tile`, `concatenate`/`concat` | `axis=None` flattens the inputs first. `take` supports only `mode="raise"`. `concat` rejects `out` and `casting` other than `"same_kind"`. |
| Linear algebra | `tensordot`, `vdot`, `vecdot`, `linalg.norm`, `linalg.vector_norm`, `linalg.matrix_norm`, `matmul` | `linalg.norm` is Frobenius only (`ord`/`axis` unsupported). `vector_norm` supports the Euclidean norm; `matrix_norm` supports the Frobenius norm. Mixed dtypes are promoted. |
| Metadata | `shape`, `ndim`, `size`, `copy`, `astype` | `np.size(ttd, axis=...)` accepts an int or tuple of axes. |
| Creation | `full_like`, `ones_like`, `zeros_like` | `np.zeros`/`ones`/`full` take no TTD argument, so they cannot be intercepted and return dense arrays. |
| Indexing | integers, slices, `...`, `None`/`newaxis`, integer arrays, boolean masks | Basic indexing (integers, slices, `...`, `None`) is exact. All-integer indexing returns a NumPy scalar. Advanced indexing (integer arrays, boolean masks, and other array-like indices) falls back on a dense array, emits a `DenseWarning`, and recompresses the result. |

## Array API namespace

`ttd.__array_namespace__()` returns :mod:`wren_ttd.array_api` for the API
versions supported by the installed NumPy. The namespace overrides the
shape-based constructors so they return TTDs; every other attribute delegates
to NumPy and dispatches through the registered handlers.

| Area | Namespace surface | Notes |
|---|---|---|
| Creation | `asarray`, `zeros`, `ones`, `empty`, `full`, `arange`, `eye`, `linspace` | Return TTDs. `empty` returns zeros because compressed tensors have no uninitialised memory. Rank-0 shapes and scalar input raise `NotImplementedError`. |
| Dtype utilities | `result_type`, `can_cast`, `finfo`, `iinfo`, `isdtype` | Accept dtypes or TTDs; TTDs contribute their dtype without densifying. |
| Delegated functions | `add`, `astype`, `ones_like`, `zeros_like`, `full_like`, `expand_dims`, `permute_dims`, `matrix_transpose`, `tensordot`, `vecdot`, `reshape`, `sum`, `mean`, `cumsum`, `take`, `concat`, `round`, `linalg.norm`, `linalg.vector_norm`, `linalg.matrix_norm` | Delegates to the NumPy handlers above. |
| Constants and dtypes | everything NumPy exposes (`newaxis`, `inf`, `nan`, `float32`, ...) | Attribute access is delegated to NumPy. |

Notable gaps:

- Rank-0 arrays are not supported: TTD has no scalar storage, so reductions
  without `keepdims` return NumPy scalars and constructors reject empty
  shapes.
- TTD supports floating-point dtypes only. Integer and boolean dtypes are
  rejected by constructors and casts.
- A full `array-api-tests` run is not wired into CI; the local smoke tests in
  `tests_ttd/array_api_test.py` and `tests_ttd/ops/get_item_test.py` cover the
  implemented subset.

## Approximation and fallbacks

| Class | Operations |
|---|---|
| Exact core-level | `add`, `subtract`, `multiply`, scalar `divide`, `negative`, `positive`, `power` with a non-negative exponent, `expand_dims`, `squeeze`, `diff`, `flip`, `roll`, `stack`, `sum`, `mean` (up to the final scalar division), `cumsum`, `vecdot`, `trace` when the traced axes are last, `take`, `repeat`, `tile`, `concat`, `broadcast_to`, `tensordot`/`matmul` with identity axis layouts, `vdot`. |
| SVD-truncated | `transpose`, `swapaxes`, `moveaxis`, `matrix_transpose`, `trace` when the traced axes must be moved, and `tensordot` when a nontrivial axis permutation is required. The default tolerance is `DEFAULT_EPSILON`. |
| Dense fallback | `absolute`, `round`/`around`, `power` with a negative exponent, `ravel`/`flatten`, `reshape` to the full tensor size, `cumsum`/`take`/`repeat`/`concat` with `axis=None`, advanced (array-like or boolean) indexing, and `linalg.vector_norm`/`linalg.matrix_norm` when the result is a TTD. These densify the tensor and recompress the result; every path emits a `DenseWarning`. |

`reshape` merges adjacent modes exactly and splits them with an untruncated
SVD: the result is exact up to floating-point error, but the ranks may grow.
`concat` and `stack` inflate the ranks by summing the input ranks.

## Verification

The support claims above are exercised by the test suite:

```sh
just test
just lint
just typecheck
```

When adding a new operation, add at least one test through `np.*` (or the
namespace) in addition to any direct `ops.*` test, and update this matrix.
