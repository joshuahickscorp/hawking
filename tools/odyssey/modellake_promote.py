"""Move verified-complete acquisitions out of partial/ and into specimens/.

The ModelLake pipeline could recognise a finished download and could not finish
it. ``modellake_watch.complete()`` checks every manifest file for exact size,
and when it returns True the watcher emits ``already_complete`` and ``continue``s
-- ``SPECIMEN_ROOT`` is only ever READ in that module, never written. So a model
that finished downloading stayed in ``partial/`` indefinitely, invisible to the
specimen registry: 29 GB across two models on 2026-09-01, one of them idle there
for two days.

This is the missing final step, and nothing more. It does not download, does not
resolve manifests, and does not decide what is worth acquiring.

Safety, because this moves real acquisitions:

* completeness is re-verified here against the manifest, file by file, exact
  size. A directory that merely looks finished is not promoted.
* ``partial/`` and ``specimens/`` are on one filesystem, so promotion is an
  ``os.rename`` -- atomic, instant, and it cannot half-copy a 135 GB model.
* an existing destination is never overwritten. That is a refusal, not a merge.
* dry-run is the default. ``--go`` is required to move anything.

    python3 tools/odyssey/modellake_promote.py            # plan only
    python3 tools/odyssey/modellake_promote.py --go
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path("/Volumes/corpdrive/hawking-modellake")
PARTIAL_ROOT = MODEL_ROOT / "partial"
SPECIMEN_ROOT = MODEL_ROOT / "specimens"
MANIFEST_DIR = REPO / "workspace" / "campaign" / "odyssey" / "watch-manifests"


def _manifest(tag: str) -> Optional[Dict[str, Any]]:
    path = MANIFEST_DIR / f"{tag}.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or "files" not in doc or "sizes" not in doc:
        return None
    return doc


def _verify_dir(root: Path, tag: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Exact per-file size check against an arbitrary directory.

    Shared by verify() (checks the partial source) and promote()'s replay
    path (checks an already-promoted destination), so both apply the exact
    same rule modellake_watch.complete() does: a second opinion about
    completeness is how a half-downloaded model becomes a sealed specimen.
    """
    if not root.is_dir():
        return False, "NO_PARTIAL_DIR", {}
    doc = _manifest(tag)
    if doc is None:
        return False, "NO_MANIFEST", {}
    files: List[str] = list(doc["files"])
    sizes: Dict[str, int] = dict(doc["sizes"])
    missing, wrong = [], []
    total = 0
    for name in files:
        path = root / name
        try:
            observed = path.stat().st_size
        except (FileNotFoundError, OSError):
            missing.append(name)
            continue
        if observed != sizes.get(name):
            wrong.append({"file": name, "on_disk": observed, "expected": sizes.get(name)})
        total += observed
    detail = {
        "files": len(files), "missing": len(missing), "wrong_size": len(wrong),
        "bytes": total, "first_missing": missing[:3], "first_wrong": wrong[:3],
    }
    if missing or wrong:
        return False, "INCOMPLETE", detail
    return True, "", detail


def verify(tag: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Exact per-file size check of the partial source. Returns
    (complete, reason, detail)."""
    return _verify_dir(PARTIAL_ROOT / tag, tag)


def promote(tag: str, *, go: bool = False) -> Dict[str, Any]:
    source = PARTIAL_ROOT / tag
    destination = SPECIMEN_ROOT / tag
    result: Dict[str, Any] = {"tag": tag, "source": str(source), "destination": str(destination)}

    if not source.is_dir():
        if not destination.is_dir():
            result.update(complete=False, detail={}, action="REFUSED", reason="NO_PARTIAL_DIR")
            return result
        # A replayed event over an already-correct specimen -- the partial
        # is gone because a prior run already moved it. Must be a NOOP, never
        # a REFUSED that hides a promotion that already succeeded.
        complete, reason, detail = _verify_dir(destination, tag)
        result.update(complete=complete, detail=detail)
        if complete:
            result["action"] = "ALREADY_PROMOTED"
        else:
            result["action"] = "REFUSED"
            result["reason"] = f"DESTINATION_INCOMPLETE:{reason}"
        return result

    complete, reason, detail = verify(tag)
    result.update(complete=complete, detail=detail)
    if not complete:
        result["action"] = "REFUSED"
        result["reason"] = reason
        return result
    if destination.exists():
        # Never merge into an existing specimen. Two directories claiming one
        # identity is worse than a stalled promotion.
        result["action"] = "REFUSED"
        result["reason"] = "DESTINATION_EXISTS"
        return result
    if not go:
        result["action"] = "WOULD_PROMOTE"
        return result
    SPECIMEN_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(source, destination)
    except OSError as exc:
        result["action"] = "FAILED"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result
    # Re-verify AT THE DESTINATION. A rename that reported success and left an
    # unreadable tree is exactly the failure this step exists to prevent.
    after_missing = [n for n in _manifest(tag)["files"] if not (destination / n).is_file()]
    result["action"] = "PROMOTED" if not after_missing else "PROMOTED_BUT_INCOMPLETE"
    result["verified_at_destination"] = not after_missing
    if after_missing:
        result["missing_after_move"] = after_missing[:5]
    return result


def survey() -> List[Dict[str, Any]]:
    if not PARTIAL_ROOT.is_dir():
        return []
    out = []
    for entry in sorted(PARTIAL_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        complete, reason, detail = verify(entry.name)
        out.append({
            "tag": entry.name, "complete": complete,
            "reason": reason or "COMPLETE", "detail": detail,
        })
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--go", action="store_true", help="actually move (default: plan only)")
    ap.add_argument("--tag", default=None, help="promote one tag instead of every complete one")
    args = ap.parse_args(argv)

    rows = survey()
    if args.tag:
        rows = [r for r in rows if r["tag"] == args.tag]
        if not rows:
            print(f"no partial named {args.tag}", file=sys.stderr)
            return 2

    promoted, refused = [], []
    for row in rows:
        if not row["complete"]:
            refused.append(row)
            print(f"{'skip':<8} {row['tag'][:56]:<57} {row['reason']}"
                  f" (missing {row['detail'].get('missing', '?')})")
            continue
        outcome = promote(row["tag"], go=args.go)
        promoted.append(outcome)
        gib = row["detail"].get("bytes", 0) / 1024 ** 3
        print(f"{outcome['action']:<8} {row['tag'][:56]:<57} {gib:.1f} GiB"
              f"{'' if outcome.get('reason') is None else ' ' + str(outcome.get('reason'))}")

    moved = sum(1 for p in promoted if p["action"] == "PROMOTED")
    print(f"\n{len(rows)} partial(s): {len(promoted)} complete, {len(refused)} still downloading"
          f"{f', {moved} promoted' if args.go else ' (dry run, pass --go to move)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
