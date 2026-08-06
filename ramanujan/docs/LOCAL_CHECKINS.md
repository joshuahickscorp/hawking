# Local check-ins

Use this loop while Ramanujan remains blocked on Hawking completion:

```bash
export PYTHONDONTWRITEBYTECODE=1
python3.12 -m ramanujan.status
python3.12 -m pytest -q ramanujan/scaffold/tests ramanujan/scaffold/data/tests
python3.12 -m ramanujan.gen_data_matrix --check
```

`ramanujan.status` verifies that the dependency boundary still refuses
research and production authority.  The test suite exercises fixture-only
scaffold behavior.  The matrix check confirms generated intake records still
match their offline manifest and current content digests.
The bytecode setting keeps routine check-ins from adding cache folders to this
deliberately shallow layout.

Do not use the parent-restream launcher for a local check-in.  It is a
fail-closed future-launch guard and remains blocked until Hawking completion
and explicit owner evidence are present.
