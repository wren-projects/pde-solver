# TTD capability matrix

This matrix describes the current `matmul` branch after Phase 0 of the
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
| Manipulation | `expand_dims`, `squeeze`, `diff`, `flip`, `roll`, `moveaxis`, `transpose`/`permute_dims`, `swapaxes`, `matrix_transpose`, `stack`, `pad` | `roll` requires an explicit `axis`. `pad` supports only uniform widths and a scalar `constant_values`. |
| Linear algebra | `tensordot`, `vdot`, `linalg.norm`, `matmul` | `linalg.norm` is Frobenius only (`ord`/`axis` unsupported). Mixed dtypes are promoted. |
| Metadata | `shape`, `ndim`, `size`, `copy`, `astype` | `np.size(ttd, axis=...)` accepts an int or tuple of axes. |
| Creation | `full_like`, `ones_like`, `zeros_like` | `np.zeros`/`ones`/`full` take no TTD argument, so they cannot be intercepted and return dense arrays. |
| Indexing | integers, slices, partial `...` | All-integer indexing returns a NumPy scalar. Fancy/boolean indexing and `newaxis` are unsupported. |

## Array API namespace

`ttd.__array_namespace__()` returns NumPy for API versions supported by the
installed NumPy. This is a partial bridge, not a conformant namespace.

| Works through `xp.*` | Notes |
|---|---|
| `add`, `astype`, `ones_like`, `zeros_like`, `full_like`, `expand_dims`, `permute_dims`, `matrix_transpose`, `tensordot`, `round`, `linalg.norm` | Delegates to the NumPy handlers above. |

Notable gaps:

- `xp.zeros`, `xp.ones`, `xp.full`, `xp.arange`, `xp.eye`, and `xp.asarray`
  return dense ndarrays, not TTDs.
- `xp.reshape`, `xp.sum`, `xp.mean`, `xp.vecdot`, `xp.linalg.vector_norm`, and
  `xp.linalg.matrix_norm` fail.
- 0-D arrays are not supported.
- Indexing does not follow the Array API rules (fully specified selection
  tuples, 0-D results, `None`, boolean/integer arrays).
- Dtype promotion is now consistent for binary operations, but the full Array
  API casting rules are not implemented.

## Approximation and fallbacks

| Class | Operations |
|---|---|
| Exact core-level | `add`, `subtract`, `multiply`, scalar `divide`, `negative`, `positive`, `power` with non-negative exponent, `expand_dims`, `squeeze`, `diff`, `flip`, `roll`, `stack`, `tensordot`/`matmul` with identity axis layouts, `vdot`. |
| SVD-truncated | `transpose`, `swapaxes`, `moveaxis`, `matrix_transpose`, and `tensordot` when a nontrivial axis permutation is required. The default tolerance is `DEFAULT_EPSILON`. |
| Dense fallback | `absolute`, `round`/`around`, and `power` with a negative exponent. These densify the tensor and recompress the result. |

## Verification

The support claims above are exercised by the test suite:

```sh
just test
just lint
just typecheck
```

When adding a new operation, add at least one test through `np.*` (or the
namespace) in addition to any direct `ops.*` test, and update this matrix.
