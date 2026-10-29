uv := require('uv')

code_folders := "src tests packages"

[private]
default:
    @just --list --justfile {{ justfile() }}

format +files=code_folders:
    {{ uv }} run ruff format {{ files }}

lint +files=code_folders:
    {{ uv }} run ruff check {{ files }}
    {{ uv }} run ruff format --check {{ files }}

typecheck +files=code_folders:
    {{ uv }} run basedpyright {{ files }}

test:
    {{ uv }} run pytest -n 4 --dist=load

coverage:
    {{ uv }} run pytest --cov=src --cov=packages

setup: && generate-pdes
    {{ uv }} sync
    {{ uv }} run pre-commit install --hook-type pre-commit --hook-type pre-push

pde_file := "packages/wren_pde_solver/src/wren_pde_solver/pde.py"

generate-pdes:
    {{ uv }} run scripts/generate_pdes.py > {{ pde_file }}
    {{ uv }} run ruff check {{ pde_file }} --fix
    {{ uv }} run ruff format {{ pde_file }}

check +files=code_folders: (lint files) test (typecheck files)

bench *args:
    {{ uv }} run pytest packages/wren_ttd/benchmarks {{ args }}

bench-ttd *args:
    {{ uv }} run pytest packages/wren_ttd/benchmarks/bench_ttd.py {{ args }}

bench-dense *args:
    {{ uv }} run pytest packages/wren_ttd/benchmarks/bench_dense.py {{ args }}

generate-math-doc:
    tectonic -Z shell-escape docs/main.tex
