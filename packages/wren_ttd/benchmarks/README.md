# `wren_ttd` benchmarks

Benchmarks use [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
and live outside the regular test suite, so `just test` does not run them.

```sh
# all benchmarks
just bench

# only benchmarks that stay compressed
just bench-ttd

# only benchmarks that need a dense array
just bench-dense

# pass any pytest/pytest-benchmark option through the recipe
just bench -k frobenius_norm
just bench --benchmark-json=benchmark.json
```

`bench_ttd.py` parametrizes tensors over dimensions `(3, 6, 10)`, modes
`(16, 128, 1024, 4096)`, and TT-ranks `(2, 8, 24, 64)`. The tensors are
generated directly from random cores, so even the largest tensor
(`4096 ** 10` with rank 64) only stores about 1 GiB of compressed data.

`bench_dense.py` covers compression and expansion, which need a dense array.
Dense shapes are restricted to tensors of at most 2 GiB
(`DENSE_MAX_TENSOR_BYTES` in `common.py`). Compression and expansion need a few
copies of the data for cores, SVD workspace, and results, which keeps the peak
memory of the dense benchmarks within roughly 8 GiB. Because `mode ** dims`
grows quickly, the resulting grid is smaller than the compressed one, e.g.
`(512, 512, 512)` takes 1 GiB while `(1024, 1024, 1024)` would already take
8 GiB. Dense benchmark tensors are built from a TT with the case's rank and
then expanded with `np.asarray`.

The rounding benchmark uses a truncation tolerance of zero to keep the random
full-rank cores at their original ranks and measure the QR/SVD sweeps.

Dense compression is much slower than the compressed benchmarks: the largest
case (`(512, 512, 512)`) can take tens of seconds per sample, so the dense suite
can run for several minutes.
