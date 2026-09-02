#!/usr/bin/env python3
"""Replay a real bootstrap-director epoch through HCLI, end to end.

This is the proof that the G003 fix is worth something. The director recorded the
identical no-progress fingerprint `fa1e64e89916fb4265c6` at epochs 7, 17, 24, 31
and 38, and every one of those epochs left a single line in its log:

    Model did not return a valid structured JSON object

Each of those epochs burned ~314 s producing nothing. This replays one of them
against the live model through the real `hcli` binary and reports what the
receipt actually says — not what the model claims.

It is deliberately NOT a mock. The whole class of bug being fixed here is
"something reported success while checking nothing", so the check has to run the
real thing against the real mission text and read the real receipt off disk.

Usage:
    python3 tools/headless/director_epoch_replay.py                # newest epoch with a mission
    python3 tools/headless/director_epoch_replay.py --epoch 7
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(os.path.expanduser("~/Downloads/hawking-copy"))
RUNS = REPO / ".hcli-legacy/bootstrap-director-v6/runs"


def find_epoch(epoch: int | None) -> Path:
    dirs = sorted(p for p in RUNS.glob("*-epoch-*") if (p / "mission.md").is_file())
    if not dirs:
        raise SystemExit(f"no epoch directory with a mission.md under {RUNS}")
    if epoch is None:
        return dirs[-1]
    for d in dirs:
        if d.name.endswith(f"epoch-{epoch:05d}"):
            return d
    raise SystemExit(f"epoch {epoch} not found; have "
                     f"{[d.name.split('epoch-')[-1] for d in dirs][:10]}")


def historical_outcome(d: Path) -> dict:
    logs = {}
    for f in d.glob("*.log"):
        txt = f.read_text(errors="replace").strip()
        logs[f.name] = txt.splitlines()[-1][:300] if txt else "(empty)"
    return logs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--runtimes", default="1")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--keep", action="store_true", help="keep the scratch workspace")
    args = ap.parse_args()

    epoch_dir = find_epoch(args.epoch)
    mission = (epoch_dir / "mission.md").read_text()
    print(f"epoch      {epoch_dir.name}")
    print(f"mission    {len(mission)} bytes")
    print(f"historical {json.dumps(historical_outcome(epoch_dir))[:300]}")

    hcli = shutil.which("hcli") or os.path.expanduser("~/.local/bin/hcli")
    if not os.path.exists(hcli):
        print("FAIL: hcli is not installed", file=sys.stderr)
        return 2

    # A scratch git workspace, so the replay cannot mutate the real repo. The
    # mission text references repo paths that will not exist here; that is fine
    # and expected — what is under test is whether the model returns a parseable
    # structured result at this prompt size, which is where it used to die.
    ws = tempfile.mkdtemp(prefix="hcli-epoch-replay-")
    subprocess.run(["git", "init", "-q", ws], check=False)
    (Path(ws) / "README.md").write_text("scratch workspace for a director epoch replay\n")
    subprocess.run(["git", "-C", ws, "add", "-A"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "-C", ws, "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t0 = time.time()
    proc = subprocess.run([hcli, args.runtimes, mission], cwd=ws,
                          capture_output=True, text=True, timeout=args.timeout)
    wall = time.time() - t0

    receipts = sorted(Path(ws).glob(".hcli/receipts/*.json"), key=os.path.getmtime)
    receipt = None
    if receipts:
        try:
            receipt = json.loads(receipts[-1].read_text())
        except Exception as e:
            receipt = {"_parse_error": str(e)}

    out = {
        "schema": "hawking.headless.director_epoch_replay.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epoch_dir": str(epoch_dir),
        "mission_bytes": len(mission),
        "historical_outcome": historical_outcome(epoch_dir),
        "historical_note": ("every recorded epoch failed with 'Model did not return a valid "
                            "structured JSON object' after roughly 314 s, and the director's "
                            "fingerprint never moved"),
        "replay": {
            "hcli": hcli,
            "runtimes": args.runtimes,
            "workspace": ws,
            "exit_code": proc.returncode,
            "wall_s": round(wall, 2),
            "stdout_tail": proc.stdout[-1500:],
            "stderr_tail": proc.stderr[-1500:],
        },
        "receipt": receipt,
    }

    ok = False
    if receipt and not receipt.get("_parse_error"):
        status = receipt.get("status")
        calls = receipt.get("model_calls") or []
        finish = [c.get("finish_reason") for c in calls] if calls else None
        out["verdict"] = {
            "status": status,
            "kind": receipt.get("kind"),
            "operations": len(receipt.get("operations") or []),
            "finish_reasons": finish,
            "error": receipt.get("error"),
            "error_type": receipt.get("error_type"),
        }
        # The bar: the model produced a parseable structured result. Whether it
        # then proposed a good mutation against a scratch tree is a different
        # question and not what this replay is testing.
        ok = status in ("completed",) or (
            status == "failed" and receipt.get("error")
            and "structured JSON" not in str(receipt.get("error")))
        if finish and "length" in finish:
            ok = False
            out["verdict"]["note"] = ("a model call still hit finish_reason=length — the "
                                      "reasoning budget is not actually constrained")

    dest = REPO / "receipts/headless/DIRECTOR_EPOCH_REPLAY.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))

    print(f"\nreplay     exit={proc.returncode} wall={wall:.1f}s")
    if receipt:
        v = out.get("verdict", {})
        print(f"receipt    status={v.get('status')} kind={v.get('kind')} "
              f"ops={v.get('operations')} finish={v.get('finish_reasons')}")
        if v.get("error"):
            print(f"error      {v.get('error_type')}: {str(v.get('error'))[:200]}")
    else:
        print("receipt    NONE WRITTEN — the engine did not get far enough to record anything")
    print(f"\nverdict    {'STRUCTURED RESULT PRODUCED' if ok else 'STILL FAILING'}")
    print(f"-> {dest}")
    if not args.keep:
        shutil.rmtree(ws, ignore_errors=True)
    else:
        print(f"workspace kept at {ws}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
