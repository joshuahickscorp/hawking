#!/usr/bin/env python3
# eval-ppl.py — COMPATIBILITY SHIM over tools/strand_eval (the canon eval module).
# The eval math no longer lives here; this preserves the pod-side argv contract
# exactly while delegating to the single canonical implementation:
#   argv: <load_dir> <ctx> <limit_chunks> <device: cuda|cpu|mps|offload|auto> <dtype> <tag> <out_json>
#   env:  EVAL_GPU_GB (offload mode GPU budget, default 21)
# Behavior preserved: RESULT_JSON line on stdout; <out_json> written at the exact
# requested path. Added by the module: a canonically-named ppl_<model>_<tag>.json
# next to it (collision-impossible), a harness_key in every record, and a
# results-ledger append (skipped silently when no repo/ledger is reachable).
# NOTE: not shipped to the pod during the live campaign — pod migration is
# POST-campaign (run-frozen rule).
import json
import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
for _cand in (os.path.join(os.path.dirname(_HERE), "tools"),  # repo: ops/../tools
              _HERE,                                          # shipped: package alongside
              os.path.join(os.environ.get("STRAND_ROOT", "/nonexistent"), "tools")):
    if os.path.isdir(os.path.join(_cand, "strand_eval")):
        sys.path.insert(0, _cand)
        break
else:
    raise SystemExit("[ppl] FATAL: cannot find the strand_eval package "
                     f"(looked next to {_HERE} and under STRAND_ROOT). "
                     "Ship tools/strand_eval alongside or set STRAND_ROOT.")

from strand_eval.core import run_eval  # noqa: E402

load_dir, ctx_s, limit_s, device_s, dtype_s, tag, out_json = sys.argv[1:8]

out_dir = os.path.dirname(os.path.abspath(out_json)) or "."
try:
    from strand_eval import default_ledger_path
    ledger = default_ledger_path()
    no_ledger = False
except Exception:  # shipped off-repo with no STRAND_ROOT: eval still runs
    ledger, no_ledger = None, True

rec, canon_path = run_eval(load_dir, tag, ctx=int(ctx_s), limit_chunks=int(limit_s),
                           device=device_s, dtype=dtype_s, out_dir=out_dir,
                           ledger_path=ledger, no_ledger=no_ledger)

# argv contract: the caller's exact out_json path must exist with the result
if os.path.abspath(out_json) != os.path.abspath(canon_path):
    with open(out_json, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[ppl] legacy-path copy: {out_json} (canonical: {canon_path})", flush=True)
