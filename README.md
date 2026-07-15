# Partial Differential Equation Solver

Package for solving PDEs using Tensor-Train decomposition.

> This is a work in progress. Currently, only a proof of concept version is implemented.

## Development

The project uses [uv](https://docs.astral.sh/uv/) package manager and build
system and the [Just](https://just.systems/) command runner. To fetch the
dependencies and install [pre-commit](https://pre-commit.com/) hooks, run

```sh
just setup
```

There are Just recipes for most often repeated tasks:

```sh
just lint
just typecheck
just format
just test
just coverage
```

The project also makes use of script-generated code for the structure of PDEs.
It is automatically generated during setup and can be updated by running

```sh
just generate-pdes
```
