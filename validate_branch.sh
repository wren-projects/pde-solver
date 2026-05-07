#!/usr/bin/env bash
set -euo pipefail

main_branch="${1:-main}"

# Find divergence point with main, then list commits reachable from HEAD but not from main.
base="$(git merge-base HEAD "$main_branch")"
mapfile -t commits < <(git rev-list --reverse --ancestry-path "${base}..HEAD")

if ((${#commits[@]} == 0)); then
    echo "No commits to test (branch contains no commits beyond $main_branch)."
    exit 0
fi

start_ref="$(git rev-parse --abbrev-ref HEAD)"
start_sha="$(git rev-parse HEAD)"

cleanup() {
    git checkout -q "$start_ref" 2>/dev/null || git checkout -q "$start_sha"
}
trap cleanup EXIT

echo "Main branch: $main_branch"
echo "Base (merge-base): $base"
echo "Testing ${#commits[@]} commits from first unique commit to HEAD..."

for sha in "${commits[@]}"; do
    subject="$(git log -1 --pretty=%s "$sha")"
    echo
    echo "==> Testing $sha  $subject"
    git checkout -q "$sha"
    uv run pytest
    uv run ruff check src packages tests
    uv run ruff format src packages tests --check
done

echo
echo "All commits passed"
