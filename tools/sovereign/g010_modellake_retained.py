#!/usr/bin/env python3
"""G010 producer: ModelLake retained verified bytes per wall second.

RETAINED means bytes that survive a downloader restart AND match the pinned
upstream manifest. That is exactly the set of files sitting at their final
manifest path at their exact manifest size, in either the active `partial/`
destination or the promoted `specimens/` directory. Bytes still parked in
`.cache/huggingface/download/*.incomplete` staging are deliberately NOT
counted: the metric is retention, not transfer rate, and staging is the part a
restart can throw away. `modellake_watch.durable_bytes()` counts staging on
purpose (it is a progress probe); this producer must not.

Read-only against the lake. It stats files, lists directories, reads the
watcher's JSONL log and runs `ps`. It never writes, deletes, promotes,
signals a download, or touches .hcli/.

    python3 tools/sovereign/g010_modellake_retained.py [--window-s 660]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.odyssey import modellake_watch as mw  # noqa: E402

RECEIPT = REPO_ROOT / "receipts" / "sovereign" / "G010_modellake_retained.json"
PARTIAL_ROOT = mw.MODEL_ROOT / "partial"
SPECIMEN_ROOT = mw.SPECIMEN_ROOT

# A partial directory nothing has written to for this long, with no live
# writer process, is stale. Mirrors the watcher's own STALL_SECONDS.
STALE_SECONDS = mw.STALL_SECONDS


def manifests() -> dict[str, dict]:
    """Pinned exact-revision inventories the watcher already resolved."""
    out: dict[str, dict] = {}
    if not mw.MANIFEST_DIR.is_dir():
        return out
    for path in sorted(mw.MANIFEST_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sizes = {str(k): int(v) for k, v in data["sizes"].items()}
            files = [str(n) for n in data["files"]]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not files:
            continue
        out[path.stem] = {
            "repo": str(data.get("repo", "")),
            "revision": str(data.get("revision", "")),
            "resolved_sha": str(data.get("resolved_sha", "")),
            "expected_recorded": data.get("expected"),
            "files": files,
            "sizes": sizes,
        }
    return out


def _retained_in(root: Path, man: dict) -> tuple[int, int]:
    """(bytes at exact manifest size, count of size-mismatched present files)."""
    if not root.is_dir():
        return 0, 0
    good = 0
    mismatched = 0
    sizes = man["sizes"]
    for name in man["files"]:
        path = root / name
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == sizes[name]:
            good += sizes[name]
        else:
            mismatched += 1
    return good, mismatched


def retained(tag: str, man: dict) -> int:
    """Best of the two canonical roots -- never the sum, so a mid-window
    promotion (partial/ -> specimens/) reads as continuity, not a doubling."""
    return max(_retained_in(PARTIAL_ROOT / tag, man)[0],
               _retained_in(SPECIMEN_ROOT / tag, man)[0])


def active_tags(mans: dict[str, dict]) -> list[str]:
    """Tags whose retained bytes can change: anything with a partial/ dir."""
    try:
        present = {p.name for p in PARTIAL_ROOT.iterdir() if p.is_dir()}
    except OSError:
        present = set()
    return sorted(t for t in mans if t in present)


def measure(tags: list[str], mans: dict[str, dict]) -> tuple[dict[str, int], float]:
    """Per-tag retained bytes plus the monotonic midpoint of the pass.

    The midpoint is the honest timestamp for a walk that takes non-zero time;
    both passes walk the same tags in the same order, so the midpoints differ
    by the true elapsed wall time between equivalent points of the two scans.
    """
    start = time.monotonic()
    out = {tag: retained(tag, mans[tag]) for tag in tags}
    return out, (start + time.monotonic()) / 2.0


def log_events_since(offset: int) -> tuple[list[dict], int]:
    """Watcher events appended after `offset`, and the new end offset."""
    events: list[dict] = []
    try:
        with mw.LOG.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            for line in handle:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
            return events, handle.tell()
    except OSError:
        return events, offset


def log_size() -> int:
    try:
        return mw.LOG.stat().st_size
    except OSError:
        return 0


# ---------------------------------------------------------------- lifecycle


def check_complete_but_unpromoted(mans: dict[str, dict], proc_rows) -> dict:
    """Finished in partial/ but never moved to specimens/ -- the seven-day bug."""
    defects = []
    for tag, man in sorted(mans.items()):
        root = PARTIAL_ROOT / tag
        if not root.is_dir() or (SPECIMEN_ROOT / tag).is_dir():
            continue
        good, _ = _retained_in(root, man)
        if good == sum(man["sizes"].values()):
            defects.append({
                "tag": tag,
                "bytes": good,
                "live_writer": mw._has_live_writer(root, proc_rows),
            })
    return {
        "checked": True,
        "method": "every manifest file present at exact manifest size in "
                  "partial/<tag> while specimens/<tag> does not exist",
        "scanned": len(mans),
        "defects": defects,
        "count": len(defects),
    }


def check_stale_partial(mans: dict[str, dict], proc_rows) -> dict:
    """A partial directory no process is writing and nothing has touched."""
    defects = []
    scanned = 0
    now = time.time()
    try:
        listing = sorted(PARTIAL_ROOT.iterdir())
    except OSError:
        listing = []
    entries = [p for p in listing if p.is_dir()]
    for stray in listing:
        # Downloader scratch left behind in the acquisition area. Not a payload
        # directory, so the walk below never sees it; still stale debris.
        if stray.is_dir() or not stray.name.startswith("tmp_"):
            continue
        scanned += 1
        try:
            stat = stray.stat()
        except OSError:
            continue
        if now - stat.st_mtime >= STALE_SECONDS:
            defects.append({"tag": stray.name, "kind": "stray_temp_file",
                            "idle_s": round(now - stat.st_mtime, 1),
                            "bytes": stat.st_size})
    for root in entries:
        scanned += 1
        newest = 0.0
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in [dirpath] + [os.path.join(dirpath, f) for f in filenames]:
                try:
                    newest = max(newest, os.stat(name).st_mtime)
                except OSError:
                    continue
        idle = now - newest if newest else None
        if mw._has_live_writer(root, proc_rows):
            continue
        man = mans.get(root.name)
        if man is not None:
            good, _ = _retained_in(root, man)
            if good == sum(man["sizes"].values()):
                continue  # finished, not stale; that is the unpromoted check
        if idle is not None and idle >= STALE_SECONDS:
            defects.append({
                "tag": root.name,
                "idle_s": round(idle, 1),
                "has_manifest": man is not None,
                "retained_bytes": (_retained_in(root, man)[0] if man else 0),
            })
    return {
        "checked": True,
        "method": f"directories: no process command line references the "
                  f"directory AND newest mtime anywhere under it is >= "
                  f"{STALE_SECONDS}s old AND it is not manifest-complete. "
                  f"stray tmp_* files in partial/ older than {STALE_SECONDS}s "
                  f"are reported as leftover downloader scratch.",
        "scanned": scanned,
        "defects": defects,
        "count": len(defects),
    }


def check_restart_data_loss(before: dict[str, int], after: dict[str, int],
                            events: list[dict]) -> dict:
    """Did any restart in this window cost retained bytes?

    Restarts are not hypothetical here: the watcher refreshes transfers on
    purpose. The test is whether retained (manifest-verified) bytes ever went
    DOWN for a job that was relaunched inside the window.
    """
    restarted: dict[str, int] = {}
    for row in events:
        if row.get("event") == "download_started":
            job = str(row.get("job", ""))
            restarted[job] = restarted.get(job, 0) + 1
    defects = []
    for tag in sorted(set(before) | set(after)):
        delta = after.get(tag, 0) - before.get(tag, 0)
        if delta < 0:
            defects.append({
                "tag": tag,
                "lost_bytes": -delta,
                "restarts_in_window": restarted.get(tag, 0),
            })
    return {
        "checked": True,
        "method": "per-job retained bytes compared across the measurement "
                  "window against download_started events observed in the "
                  "watcher JSONL over the same window",
        "restarts_in_window": restarted,
        "jobs_restarted": len(restarted),
        "defects": defects,
        "count": len(defects),
    }


def check_manifest_mismatch(mans: dict[str, dict]) -> dict:
    """Cached manifest self-consistency, and on-disk files that contradict it."""
    defects = []
    for tag, man in sorted(mans.items()):
        expected = sum(man["sizes"].values())
        if man["resolved_sha"] and man["resolved_sha"] != man["revision"]:
            defects.append({"tag": tag, "kind": "revision_mismatch",
                            "revision": man["revision"],
                            "resolved_sha": man["resolved_sha"]})
        recorded = man["expected_recorded"]
        if isinstance(recorded, int) and recorded != expected:
            defects.append({"tag": tag, "kind": "expected_total_mismatch",
                            "recorded": recorded, "summed": expected})
        for root in (PARTIAL_ROOT / tag, SPECIMEN_ROOT / tag):
            _, bad = _retained_in(root, man)
            if bad:
                defects.append({"tag": tag, "kind": "file_size_mismatch",
                                "root": str(root), "files": bad})
    return {
        "checked": True,
        "method": "resolved_sha == pinned revision, recorded expected == summed "
                  "sizes, and every present final-path file matches its manifest "
                  "size exactly",
        "scanned": len(mans),
        "defects": defects,
        "count": len(defects),
    }


def check_duplicate_acquisition(mans: dict[str, dict], proc_rows) -> dict:
    """Two things claiming one payload: two directories, two revisions of one
    repo on disk, or two live downloaders for the same repo."""
    defects = []
    for tag in sorted(mans):
        if (PARTIAL_ROOT / tag).is_dir() and (SPECIMEN_ROOT / tag).is_dir():
            defects.append({"kind": "partial_and_specimen", "tag": tag})
    by_repo: dict[str, list[str]] = {}
    for tag, man in mans.items():
        if (PARTIAL_ROOT / tag).is_dir() or (SPECIMEN_ROOT / tag).is_dir():
            by_repo.setdefault(man["repo"], []).append(tag)
    for repo, tags in sorted(by_repo.items()):
        if repo and len(tags) > 1:
            defects.append({"kind": "two_revisions_on_disk", "repo": repo,
                            "tags": sorted(tags)})
    live: dict[str, list[int]] = {}
    for pid, command in proc_rows:
        if "hf download " not in command:
            continue
        repo = command.split("hf download ", 1)[1].split()[0]
        live.setdefault(repo, []).append(pid)
    for repo, pids in sorted(live.items()):
        if len(pids) > 1:
            defects.append({"kind": "concurrent_downloaders", "repo": repo,
                            "pids": sorted(pids)})
    return {
        "checked": True,
        "method": "tag present in both partial/ and specimens/; two pinned "
                  "revisions of one repo with directories on disk; more than one "
                  "live `hf download` process for one repo",
        "scanned": len(mans),
        "defects": defects,
        "count": len(defects),
    }


# --------------------------------------------------------------------- main


def self_check() -> None:
    """The whole metric rests on _retained_in counting ONLY exact matches."""
    import tempfile

    man = {"files": ["a.bin", "b.bin", "c.bin"],
           "sizes": {"a.bin": 100, "b.bin": 200, "c.bin": 300}}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.bin").write_bytes(b"x" * 100)          # exact -> counted
        (root / "b.bin").write_bytes(b"x" * 50)           # short -> mismatch
        # c.bin only exists as staging, which must never count
        stage = root / ".cache" / "huggingface" / "download"
        stage.mkdir(parents=True)
        (stage / "hashed.incomplete").write_bytes(b"x" * 300)
        good, bad = _retained_in(root, man)
        assert good == 100, good
        assert bad == 1, bad
    print("self-check ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-s", type=float, default=660.0,
                        help="minimum wall seconds between the two passes")
    parser.add_argument("--self-check", action="store_true",
                        help="verify the retained predicate and exit")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0

    command = "python3 tools/sovereign/g010_modellake_retained.py " \
              f"--window-s {args.window_s:g}"
    started_at = datetime.now(timezone.utc)

    mans = manifests()
    if not mans:
        print("no cached manifests; nothing can be verified", file=sys.stderr)
        return 2
    tags = active_tags(mans)
    if not tags:
        print("no partial/ directory has a manifest; no live acquisition",
              file=sys.stderr)
        return 2

    log_offset = log_size()
    before, t0 = measure(tags, mans)
    print(f"T0 retained {sum(before.values())/1e9:.3f} GB over {len(tags)} jobs",
          flush=True)

    # The lifecycle audit runs inside the window; it is a survey, not a timed
    # measurement, and it is the slow walk (whole lake).
    proc_rows = mw.process_rows()
    lifecycle = {
        "complete_but_unpromoted": check_complete_but_unpromoted(mans, proc_rows),
        "stale_partial": check_stale_partial(mans, proc_rows),
        "manifest_mismatch": check_manifest_mismatch(mans),
        "duplicate_acquisition": check_duplicate_acquisition(mans, proc_rows),
    }

    remaining = args.window_s - (time.monotonic() - t0)
    while remaining > 0:
        time.sleep(min(remaining, 30.0))
        remaining = args.window_s - (time.monotonic() - t0)

    after, t1 = measure(tags, mans)
    window_s = t1 - t0
    events, _ = log_events_since(log_offset)
    lifecycle["restart_data_loss"] = check_restart_data_loss(before, after, events)

    delta = sum(after.values()) - sum(before.values())
    rate = delta / window_s
    restarts = sum(1 for row in events if row.get("event") == "download_started")
    exits = sum(1 for row in events if row.get("event") == "download_exit")
    print(f"T1 retained {sum(after.values())/1e9:.3f} GB  "
          f"delta {delta/1e9:.3f} GB over {window_s:.1f}s = {rate/1e6:.1f} MB/s",
          flush=True)

    receipt = {
        "schema": "hawking.sovereign.g010.modellake_retained.v1",
        "gate": "G010",
        "produced_by": "tools/sovereign/g010_modellake_retained.py",
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at.isoformat(),
        "command": command,
        "status": "measured",
        "metric": "retained_verified_bytes_per_wall_second",
        "retained_bytes_per_s": rate,
        "window_s": window_s,
        "restarts": restarts,
        "download_exits_in_window": exits,
        "retained_bytes_t0": sum(before.values()),
        "retained_bytes_t1": sum(after.values()),
        "retained_delta_bytes": delta,
        "measurement": {
            "estimated": False,
            "note": "Both endpoints are direct stat() readings of files on the "
                    "live lake, divided by the true monotonic elapsed time "
                    "between the midpoints of the two identical scans. No "
                    "extrapolation from a shorter window.",
            "verification": "every counted byte belongs to a file sitting at its "
                            "final manifest path at exactly its pinned manifest "
                            "size, in partial/<tag> or specimens/<tag>",
            "content_hash_verified": False,
            "content_hash_note": "sizes are checked against the pinned upstream "
                                 "manifest; file contents are NOT re-hashed "
                                 "(multi-TB read on a volume with four live "
                                 "downloads). This is the same completeness "
                                 "predicate modellake_watch.complete() uses.",
            "excludes": ".cache/huggingface/download/*.incomplete staging bytes, "
                        "which a restart may discard and which no manifest entry "
                        "verifies",
            "scope_tags": tags,
            "per_tag_t0": before,
            "per_tag_t1": after,
            "lake_root": str(mw.MODEL_ROOT),
            "watcher_log": str(mw.LOG),
        },
        "lifecycle_checks": lifecycle,
        "evidence": {
            "live_downloaders": sorted(
                pid for pid, command in proc_rows if "hf download " in command),
            "manifests_scanned": len(mans),
            "lifecycle_defect_counts": {
                name: check["count"] for name, check in lifecycle.items()},
        },
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    print(f"wrote {RECEIPT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
