# TTD NumPy / Array API parity plan

## Context

`wren_ttd` is a NumPy-protocol array container (NEP 13/18), not a full
Array API implementation. At the start of the plan, the branch had 23 commits
on top of `origin/main` and exposed 29 registered NumPy entry points:

- **6 ufuncs:** `add`, `subtract`, `multiply`, `divide`, `negative`, `matmul`
- **23 functions:** `pad`, `stack`, `transpose`, `swapaxes`, `tensordot`,
  `vdot`, `linalg.norm`, `gradient`, `full_like`, `ones_like`, `zeros_like`,
  `shape`, `ndim`, `size`, `copy`, `astype`, `moveaxis`, `diff`,
  `expand_dims`, `flip`, `squeeze`, `roll`, `linalg.matrix_transpose`

`__array_namespace__()` currently returns the `numpy` module. This is useful
for simple interop, but it is not a conformant Array API namespace: namespace
constructors such as `xp.zeros` return dense arrays, and required functions
such as `xp.reshape`, `xp.sum`, and `xp.vecdot` fail on TTDs.

The solver is also still blocked on TTD integration: mixed ndarray/TTD
`tensordot` fails, `Divergence.ZERO` is an ndarray, and ranks must be truncated
after each timestep.

## Status

- **Phase 0: complete.** Elementwise scalar operations, binary dtype
  promotion, ufunc `out`/`dtype` handling, `np.size(axis=...)`, truncation
  aliases, top-level `matrix_transpose`, and validated `np.pad` are
  implemented and covered by tests.
- **Phase 1: complete.** Reductions, norms, reshape/broadcasting, and
  selection/joining are implemented, tested, and exposed through `ops.*`,
  `np.*`, and the matching `TTD` methods. `diagonal`, `prod`, `max`, and
  `min` remain deferred.
- **Phase 2: complete.** A dedicated `wren_ttd.array_api` namespace is
  returned by `TTD.__array_namespace__()`. Constructors and `asarray` return
  TTDs, indexing follows NumPy basic and advanced rules, and dtype utilities
  are exposed. Rank-0 arrays remain explicitly unsupported.
- **Phase 3: pending.** See the phase below.

Current supported surface: [capability matrix](ttd-capability-matrix.md).

## Goals

1. Make the existing NumPy-protocol surface internally consistent and safe.
2. Add the high-value reductions and manipulation operations needed by both
   library users and the PDE solver.
3. Keep the Array API namespace claim honest: either implement the required
   subset properly or document TTD as a NumPy-compatible container with a
   partial Array API bridge.
4. Unblock TTD use in the finite-difference solver and add tests that exercise
   that path.

## Non-goals

- Implementing every NumPy function.
- General, exact TT algorithms for `prod`, `max`, `min`, or arbitrary
  elementwise nonlinearities.
- Silent densification by default. Any dense fallback must be explicit and
  documented.
- Complex dtypes or GPU devices.

## Principles

- Prefer exact core-level algorithms. A dense fallback is acceptable only when
  an exact TT representation is unavailable, and it must be documented.
- Keep dtype semantics close to NumPy 2 (NEP 50). Centralise promotion rather
  than duplicating ad-hoc casts.
- Every registered entry point needs a test through `np.*` (or the namespace),
  not only through `ops.*`.
- Keep public claims proportional to the test suite. Add a capability matrix
  before advertising broader compatibility.

## Phase 0: correctness and consistency

Phase 0 fixes wrong results and sharp edges before adding surface. It is split
into non-overlapping work packages so several branches can be developed in
parallel.

### WP 0.1: Elementwise scalar operations (shared scaffold)

Add and register:

- `np.absolute` / `abs(ttd)`
- `np.positive` / `+ttd`
- `np.power` / `ttd ** n` for integer-like scalar exponents
- `np.round` elementwise, using a documented dense fallback

Acceptance criteria:

- `abs(ttd)`, `+ttd`, `ttd ** 2`, and `np.round(ttd, decimals)` agree with
  NumPy on all existing test tensors.
- Non-integer exponents and `np.power(scalar, ttd)` raise a clear
  `NotImplementedError`.
- `TTD.round`/`rounded` keep their current truncation semantics; add
  `truncate`/`truncated` as the preferred names and document the collision.
- Tests live in a dedicated file and do not modify shared test helpers.

### WP 0.2: Binary dispatch and dtype promotion

Files: `ops/add.py`, `ops/subtract.py`, `ops/multiply.py`, `ops/divide.py`,
`ops/matmul.py`, `ops/tensordot.py`, `_helpers.py`, plus dedicated tests.

Changes:

- Add a single dtype-promotion helper using `np.result_type`. Apply it to all
  binary operations, including scalar operands and `np.float64` scalars.
- Make mixed `float32`/`float64` operations return the NumPy-promoted dtype.
- Remove the "same dtype required" errors from `matmul`/`tensordot`; promote
  both operands instead.
- Accept a NumPy-style `dtype=` keyword in the ufunc handlers and cast the
  result to it.
- Validate `out.shape` and `out.dtype` consistently. Do not silently write a
  result with a different dtype.
- Add tests for every binary operation: same dtype, mixed dtype, scalar
  promotion, `out=`, and `dtype=`.

Acceptance criteria:

- `float32 + float64` returns `float64`; `float32 * float64` returns
  `float64`; `float32 + np.float64(1)` returns `float64`.
- `TTD`/`TTD` `matmul` and `tensordot` accept mixed floating dtypes.
- `np.add(a, b, dtype=np.float32)` returns a `float32` TTD.
- `out` with an incompatible shape or dtype raises a descriptive error.

### WP 0.3: Core dispatch and metadata

Files: `core.py`, new `tests_ttd/ufunc_test.py`, `tests_ttd/core_test.py`.

Changes:

- Normalise NumPy's `out=(value,)` tuple to a single TTD in
  `__array_ufunc__`; support one output and reject multi-output ufuncs.
- Accept `dtype=` and pass it to the handler. Reject `where`, `casting`,
  `order`, `subok`, and `signature` with a clear `NotImplementedError` unless
  they are at their defaults.
- Implement `np.size(ttd, axis=...)` as a real function instead of reusing the
  property getter.
- Add `TTD.truncate`/`TTD.truncated` aliases for rank truncation and document
  that `TTD.round` is the legacy truncation name, while `np.round` is
  elementwise.
- Add tests for `out=`, `dtype=`, unsupported kwargs, multi-output ufuncs,
  `np.size(axis=)`, and the truncation aliases.

Acceptance criteria:

- `np.add(a, b, out=(c,))` works for every registered binary ufunc.
- Unsupported ufunc options fail with a message naming the option.
- `np.size(a, axis=0)` and `np.size(a, axis=-1)` match NumPy.
- Existing truncation tests still pass through the new aliases.

### WP 0.4: Small NumPy parity fixes

Files: `ops/matrix_transpose.py`, `ops/pad.py`, their tests.

Changes:

- Register both `matrix_transpose` and `linalg.matrix_transpose`, so
  `np.matrix_transpose` and `xp.matrix_transpose` work.
- Validate `np.pad` inputs:
  - accept a scalar `pad_width`;
  - accept a uniform tuple form such as `((1, 1), (1, 1))`;
  - raise `NotImplementedError` (not an internal `TypeError`) for uneven
    per-axis padding;
  - raise a clear error for per-axis `constant_values` and unsupported modes.
- Add tests for all supported and rejected forms.

Acceptance criteria:

- `np.matrix_transpose(a)` and `np.linalg.matrix_transpose(a)` both return a
  TTD.
- `np.pad(a, 1)` and `np.pad(a, ((1, 1), (1, 1)))` work.
- `np.pad(a, ((1, 1), (2, 2)))` raises `NotImplementedError` with an
  actionable message.

### WP 0.5: Verification and documentation

- Run `just lint`, `just typecheck`, and `just test`.
- Add a short capability matrix to the docs listing what is supported through
  `np.*`, what is not, and which functions use dense fallbacks.
- Update `docs/design_document.md` and `docs/roadmap.md` so they no longer
  claim a full NumPy interface or mention nonexistent `np.mul`.

## Phase 1: reductions and manipulation

Phase 1 adds the reductions and manipulation operations that the solver and
ordinary NumPy callers need. It is split into four non-overlapping work
packages. Each package owns its operation modules and its test files; the
integration step owns `core.py` methods, `ops/__init__.py` conflicts, the
capability matrix, and documentation.

Every function accepts the same core arguments as its NumPy counterpart for
the documented subset and rejects unsupported options with an actionable
`NotImplementedError`.

### Reduction result shape

TTD has no 0-D representation yet (see Phase 2). A reduction that removes
every axis therefore returns a NumPy scalar when `keepdims=False`. With
`keepdims=True`, it returns a TTD of shape ``(1,) * ndim`` built with
:meth:`TTD.full`. Reductions that leave at least one axis return a TTD.
`out=` is supported only when the result is a TTD; requesting a dense 0-D
output raises `NotImplementedError`.

### WP 1.1: Reductions

Files: `ops/sum.py` (extend the existing implementation), new
`ops/mean.py`, `ops/cumsum.py`, `ops/trace.py`, and their test files.

- `sum(a, axis=None, dtype=None, out=None, keepdims=False)`:
  - accept `int`, tuple of ints, negative axes, `None`, and `()`;
  - contract each reduced core into its neighbour with
    :func:`mode_1_product` / :func:`mode_3_product`;
  - return a scalar for a full reduction with `keepdims=False`, and an
    all-ones TTD for `keepdims=True`;
  - apply `dtype=` as the result dtype and validate `out=`;
  - reject `initial` and non-default `where` with `NotImplementedError`.
- `mean(...)`: `sum` followed by division by the product of the reduced
  sizes. Same `axis`, `dtype`, `out`, and `keepdims` semantics as `sum`.
- `cumsum(a, axis=None, dtype=None, out=None)`: apply a lower-triangular
  matrix of ones to the selected core mode. This is exact and preserves
  ranks. `axis=None` flattens the input with `ravel` and then accumulates
  along axis 0.
- `trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None)`:
  - support `offset=0` only and reject nonzero offsets;
  - when the two axes are the last two, contract the two cores directly
    with `einsum`, which is exact;
  - otherwise move the axes with the existing `transpose`/`moveaxis`
    (SVD-truncated) and contract the resulting last two cores;
  - validate `dtype` and `out`.

### WP 1.2: Vector and matrix norms

Files: new `ops/vecdot.py`, `ops/vector_norm.py`, `ops/matrix_norm.py`, and
their test files.

- `vecdot(a, b, axis=-1)`: implement as `sum(a * b, axis)` for equal shapes.
  Broadcasting incompatible shapes raises `NotImplementedError`. This is
  exact and keeps the TT representation.
- `linalg.vector_norm(a, axis=None, keepdims=False, ord=None)`: support the
  Euclidean norm (`ord` in `None, 2`) as `sqrt(sum(a * a, axis))`. Other
  orders raise `NotImplementedError`.
- `linalg.matrix_norm(a, keepdims=False, ord="fro")`: support the Frobenius
  norm over the last two axes (`ord` in `"fro", "f"`). Other orders raise
  `NotImplementedError`.
- The final element-wise square root is not multiplicative. For scalar
  results it is a NumPy scalar operation; for TTD results it uses a
  documented dense fallback and recompresses the result. The sum of squares
  itself stays in TT format.

### WP 1.3: Reshape and broadcasting

Files: new `ops/reshape.py`, `ops/ravel.py`, `ops/flatten.py`,
`ops/broadcast_to.py`, their test files, and new helpers in `_helpers.py` if
needed.

- `reshape(a, shape, order="C")`:
  - accept an int or tuple shape and one `-1` entry;
  - merge adjacent cores exactly by contracting their shared rank;
  - split a core by reshaping it to ``(left * n_left, n_right * right)``
    and factoring with an SVD, which is exact but may inflate ranks;
  - walk the source and target shapes one mode at a time, merging or
    splitting as required, so interleaved cases such as ``(2, 6) -> (3, 4)``
    work;
  - support `order="C"` only; reject `"F"`/`"A"`/`"K"`.
- `ravel(a, order="C")` and `flatten(a, order="C")`: reshape to
  ``(a.size,)``. `flatten` is also exposed as a `TTD` method.
- `broadcast_to(a, shape, subok=False)`:
  - prepend identity cores for new leading axes;
  - expand size-1 axes by repeating the mode slices;
  - reject broadcast shapes that are not compatible with the input;
  - reject `subok=True`.
- The reshape path must not densify the whole tensor. Document the rank
  behaviour in the docstring.

### WP 1.4: Selection and joining

Files: new `ops/take.py`, `ops/repeat.py`, `ops/tile.py`, `ops/concat.py`,
and their test files.

- `take(a, indices, axis=None, out=None, mode="raise")`: select mode slices
  from one core, which is exact and rank-preserving. Support scalar and
  integer-array indices, negative indices, and `axis=None` via ravel.
  Reject `mode` other than `"raise"`.
- `repeat(a, repeats, axis=None)`: repeat mode slices, with a scalar or
  per-index `repeats`; `axis=None` uses ravel first. Exact and
  rank-preserving.
- `tile(a, reps)`: tile mode slices and prepend identity cores when `reps`
  has more entries than `a.ndim`; normalise a scalar `reps`. Exact and
  rank-preserving.
- `concatenate(ttds, axis=0, dtype=None)` and its `np.concat` alias:
  - build block-diagonal cores for the joined axis and block-structured
    ranks for every other axis, as in :func:`stack`;
  - promote dtypes with :func:`promote_dtype`;
  - support an integer `axis` and `axis=None` (flatten inputs first);
  - reject `casting` other than `"same_kind"` and `out`.
- `diagonal` is deferred until the solver needs it. It would follow the
  `trace` core-contraction pattern while keeping the diagonal mode.

### WP 1.5: Integration and documentation

- Add the corresponding `TTD` methods (`mean`, `cumsum`, `trace`, `reshape`,
  `ravel`, `flatten`, `take`, `repeat`, `tile`, etc.) that delegate to
  `ops.*`.
- Resolve `ops/__init__.py` imports and exports.
- Run `just check`.
- Update `docs/ttd-capability-matrix.md` and the status section above.

`prod`, `max`, and `min` are deferred: exact TT algorithms are not
low-hanging and should not be approximated silently.

Acceptance criteria: every function is tested against NumPy on a vector, a
matrix, and at least one 3-D tensor, through both `ops.*` and `np.*` (or the
documented namespace path). Reduction tests cover `axis`, `keepdims`,
`dtype`, and `out`. Manipulation tests cover round trips
(``reshape(reshape(a, s), a.shape)``), negative axes, and rejected forms.

## Phase 2: Array API contract

Decision: provide a dedicated `wren_ttd.array_api` namespace for the
documented subset instead of returning NumPy directly. The namespace is
conformant for arrays of rank at least one. Rank-0 arrays stay explicitly
unsupported: a TTD has no place to store a scalar without a dedicated scalar
slot, and adding one would touch every operation. Constructors and `asarray`
reject rank-0 input with `NotImplementedError`, and the capability matrix
states this limitation instead of claiming full Array API conformance.

### WP 2.1: Namespace module and constructors

Files: new `packages/wren_ttd/src/wren_ttd/array_api.py`, `core.py`, and
`tests_ttd/array_api_test.py`.

- Add an `array_api` module with `__array_api_version__` and a small public
  surface. Unknown attributes delegate to NumPy, so dispatched functions such
  as `sum`, `mean`, `reshape`, `vecdot`, and `linalg.*` keep working.
- Override the shape-based constructors so they return TTDs:
  `zeros`, `ones`, `empty`, `full`, `arange`, `eye`, `linspace`, and
  `asarray`.
- `asarray` accepts TTDs and array-like input, honours `dtype` with
  `astype`, respects `copy=True/False/None`, and rejects non-CPU `device`.
- `TTD.__array_namespace__` returns the new module and keeps validating the
  requested `api_version`.
- Constructors and `asarray` reject rank-0 shapes/input with a clear
  `NotImplementedError`.

Acceptance criteria:

- `xp = ttd.__array_namespace__()` is the TTD namespace, not NumPy.
- `xp.zeros/ones/full/arange/eye/linspace/asarray` return TTDs with values
  matching NumPy.
- `xp.sum(ttd)`, `xp.mean(ttd, axis=0)`, `xp.reshape(ttd, ...)`, and
  `xp.vecdot(a, b)` work through the namespace.
- `xp.asarray(ttd)` returns the same object when no copy is needed and a new
  TTD when `copy=True`.

### WP 2.2: Indexing normalisation

Files: `core.py`, `ops/get_item.py`, and dedicated indexing tests.

- Normalise selection tuples before calling `get_item`: wrap bare keys in a
  tuple, expand a single `...` against the number of consumed axes, and allow
  implicit trailing slices.
- Support `None`/`np.newaxis` by removing it from the basic selection and
  inserting the corresponding singleton axes with `expand_dims`. Preserve
  scalar results when every original axis is consumed and no new axis remains.
- Keep integer and slice indexing exact.
- Route boolean and integer-array indexing through a documented dense
  fallback with a `DenseWarning`, then recompress the result. This covers
  masks, multi-dimensional index arrays, and mixed advanced indices without
  pretending they are exact.
- Keep returning a NumPy scalar for fully-consumed selections, and document
  that Array API 0-D results are not represented.

Acceptance criteria:

- `a[None]`, `a[..., None]`, `a[0, ...]`, `a[..., 1]`, and
  `a[None, ..., 1]` agree with NumPy on vectors, matrices, and 3-D tensors.
- Boolean and integer-array indexing agree with NumPy and emit exactly one
  `DenseWarning`.
- Basic integer/slice indexing does not emit a warning.

### WP 2.3: Dtype utilities

Files: `array_api.py`, their tests.

- Expose `result_type`, `can_cast`, `finfo`, `iinfo`, and `isdtype`.
- Translate TTD operands to their dtypes before delegating to NumPy, so no
  tensor is densified.
- Keep the documented restriction to floating-point dtypes; unsupported
  dtypes raise the existing `TypeError` from `normalize_dtype`.

Acceptance criteria:

- `xp.result_type(ttd_float32, np.float64)` is `float64`.
- `xp.can_cast(ttd_float32, np.float64)` is true and the reverse is false
  under NumPy's safe-casting rules.
- `xp.finfo(xp.asarray(...).dtype).eps` and `xp.isdtype(..., "real floating")`
  match NumPy.

### WP 2.4: Conformance smoke tests

Files: `tests_ttd/array_api_test.py`, `tests_ttd/indexing_test.py`.

- Exercise the namespace directly rather than through `np.*`.
- Cover constructors, `asarray` copy/device semantics, reductions,
  manipulation, dtype utilities, and the indexing forms above.
- Do not add external `array-api-tests` yet. A full conformance run needs a
  dependency and a documented subset configuration; keep the local smoke
  tests dependency-free for now. Ask before adding `array-api-tests` or
  `array-api-compat` to the project.

### WP 2.5: Documentation

- Update `docs/ttd-capability-matrix.md` to describe the namespace, the
  supported functions, the dense indexing fallback, and the rank-0 gap.
- Update the status section above to mark Phase 2 complete.
- Keep `docs/design_document.md` aligned with the namespace story.

## Phase 3: Solver integration

1. Support mixed ndarray/TTD `tensordot`, `dot`, and `matmul`. `axes=0` should
   concatenate cores exactly; small matrices can be converted or contracted
   directly.
2. Replace `Divergence.ZERO` (`np.zeros((1,))`) with a scalar zero so
   `sum(gradients, start=...)` works for both ndarray and TTD.
3. Add TTD solver tests for 1-D, 2-D, and 3-D cases against the dense solution.
4. Truncate ranks after each timestep. Without this, ranks grow from `(4, 4)`
   to `(50, 50)` in a single step and the solver exhausts memory on modest
   grids.
5. Relax `pde_types` and `dot_product` typing so `NDArray` can be a TTD.
6. Wire the truncation tolerance into solver configuration and benchmark the
   resulting cost.

## Verification commands

```sh
just lint
just typecheck
just test
```

All Phase 0 packages must leave the full suite green before being
cherry-picked onto the integration branch.
