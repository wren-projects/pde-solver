# TTD NumPy / Array API parity plan

## Context

`wren_ttd` is a NumPy-protocol array container (NEP 13/18), not a full
Array API implementation. The current branch (`matmul`) has 23 commits on top
of `origin/main` and exposes 29 registered NumPy entry points:

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
- **Phase 1-3: pending.** See the phases below.

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

Implement, in order:

1. `sum` (axis, keepdims, dtype, out), `mean`, `cumsum`, `trace`.
2. `vecdot`, `linalg.vector_norm`, `linalg.matrix_norm`.
3. `reshape` for contiguous merge/split groups and `-1`, plus `ravel`,
   `flatten`, and `broadcast_to` for leading and size-1 axes.
4. `take`, `repeat`, `tile`, and `concat` along existing axes.
5. `diagonal` if the solver needs it.

`prod`, `max`, and `min` are deferred: exact TT algorithms are not
low-hanging and should not be approximated silently.

Acceptance criteria: every function is tested against NumPy on vectors,
matrices, and at least one 3-D tensor, through both `ops.*` and `np.*`.

## Phase 2: Array API contract

Decision required before starting: commit to a conformant namespace or keep
the current partial bridge.

If conformant:

1. Replace the `numpy` return value of `__array_namespace__` with a small
   namespace module that delegates to NumPy but overrides constructors
   (`zeros`, `ones`, `full`, `arange`, `eye`, `asarray`, and friends) to
   return TTDs.
2. Implement 0-D arrays or explicitly document that TTD is not Array API
   conformant.
3. Normalise indexing: newaxis, full selection tuples, ellipsis expansion,
   boolean and integer-array indexing, and 0-D scalar results.
4. Add dtype utilities and consistent promotion/casting rules.
5. Add an `array-api-tests` subset or at least `array_api_compat` smoke tests
   to CI.
6. Publish a capability matrix and stop claiming "full NumPy".

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
