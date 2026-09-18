#!/usr/bin/env bash
# Fast, complete Python suite. Keep test modules grouped by file: this gets
# xdist's CPU parallelism without interleaving stateful module fixtures.
set -euo pipefail

workers=${PYTEST_WORKERS:-auto}
python_bin=${PYTHON_BIN:-python3}
if "$python_bin" -m pytest --help | grep -q -- '--numprocesses'; then
  exec "$python_bin" -m pytest -q hawking -n "$workers" --dist loadfile "$@"
fi

printf '%s\n' 'pytest-xdist is unavailable; running the complete suite serially.' >&2
exec "$python_bin" -m pytest -q hawking "$@"
