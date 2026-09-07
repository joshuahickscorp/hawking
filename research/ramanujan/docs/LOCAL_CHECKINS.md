# Local check-ins

The retained Ramanujan records are evidence-only; live checks run from the
canonical verification homes:

```bash
export PYTHONDONTWRITEBYTECODE=1
python3.12 -m tools.verify.ramanujan_boundary
python3.12 -m pytest -q tools/verify/proof tools/condense/tests/test_restream_guard.py
```

`tools.verify.ramanujan_boundary` verifies that the dependency boundary still
refuses research and production authority. The proof tests exercise exact,
symbolic, and fail-closed Lean verifier behavior against retained fixtures.
The retained data records remain available for offline audit and replay.
The bytecode setting keeps routine check-ins from adding cache folders to this
deliberately shallow layout.

Do not use the parent-restream launcher for a local check-in.  It is a
fail-closed future-launch guard and remains blocked until Hawking completion
and explicit owner evidence are present.
