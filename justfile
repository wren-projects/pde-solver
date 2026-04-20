uv := require('uv')

[private]
default:
    @just --list --justfile {{ justfile() }}

format:
    {{ uv }} run ruff format src tests

lint:
    {{ uv }} run ruff check src tests
    {{ uv }} run basedpyright src tests

test:
    {{ uv }} run pytest

setup:
    {{ uv }} run pre-commit install --hook-type pre-commit --hook-type pre-push

generate-pdes:
    {{ uv }} run scripts/generate_pdes.py > src/pde_solver/pde.py
    {{ uv }} run ruff check src/pde_solver/pde.py --fix

check: lint test
