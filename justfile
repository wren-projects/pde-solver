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

check: lint test
