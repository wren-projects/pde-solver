uv := require('uv')

code_folders := "src tests packages"

[private]
default:
    @just --list --justfile {{ justfile() }}

format:
    {{ uv }} run ruff format {{ code_folders }}

lint:
    {{ uv }} run ruff check {{ code_folders }}
    {{ uv }} run basedpyright

test:
    {{ uv }} run pytest

coverage:
    {{ uv }} run pytest --cov=src --cov=packages

setup:
    {{ uv }} run pre-commit install --hook-type pre-commit --hook-type pre-push

pde_file := "packages/pde_solver/src/pde_solver/pde.py"

generate-pdes:
    {{ uv }} run scripts/generate_pdes.py > {{ pde_file }}
    {{ uv }} run ruff check {{ pde_file }} --fix
    {{ uv }} run ruff format {{ pde_file }}

check: lint test
