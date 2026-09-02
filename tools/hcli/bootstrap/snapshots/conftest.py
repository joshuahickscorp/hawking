"""These are archives, not tests.

Every file here is a timestamped frozen copy taken during the haider bootstrap
(`*.stale.<ts>.py`, `*.pre-fast-selfhost.<ts>.py`). Two of them are named
`test_*.py`, so pytest collected them, tried to import them under their original
module names, and failed -- `test_work_unit.stale.*.py` imports
`tools.haider.haider`, which was retired. Six collection errors abort the whole
`pytest tools` run, so these archives were costing every other test in the tree.

Ignoring them adds coverage rather than removing it: nothing here has ever run,
and nothing here can.
"""
collect_ignore_glob = ["*.py"]
