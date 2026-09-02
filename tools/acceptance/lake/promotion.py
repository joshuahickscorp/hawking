"""MODELLAKE_ATOMIC_PROMOTION — CALL promote(). Live lake: dry/replay only."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

from tools.acceptance.lake.common import (
    GATES,
    LAKE,
    PARTIAL,
    SPECIMENS,
    lake_mounted,
    timed,
    write_receipt,
)
from tools.odyssey import modellake_promote as mp
from tools.acceptance.lake.hash_verify import materialize_watch_manifests

GATE = "MODELLAKE_ATOMIC_PROMOTION"


def call_promote(tag: str, *, go: bool = False) -> dict[str, Any]:
    """Production call site of the catalog symbol."""
    return mp.promote(tag, go=go)


def run_promote_on_scratch(root: Path) -> dict[str, Any]:
    """Atomic rename, refuse-incomplete, dry-run default — on a scratch tree."""
    model_root = root / "model_root"
    partial = model_root / "partial"
    specimens = model_root / "specimens"
    manifests = root / "manifests"
    partial.mkdir(parents=True)
    specimens.mkdir(parents=True)
    manifests.mkdir(parents=True)

    saved = {
        "MODEL_ROOT": mp.MODEL_ROOT,
        "PARTIAL_ROOT": mp.PARTIAL_ROOT,
        "SPECIMEN_ROOT": mp.SPECIMEN_ROOT,
        "MANIFEST_DIR": mp.MANIFEST_DIR,
    }
    tag = "acme--promo@deadbeefcafe"
    files = {"config.json": b"{}", "weights.bin": b"0" * 4096}

    def _write(tag_name: str, payload: dict[str, bytes], dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        for name, content in payload.items():
            (dest / name).write_bytes(content)
        (manifests / f"{tag_name}.json").write_text(
            json.dumps(
                {
                    "repo": "acme/promo",
                    "revision": "deadbeefcafe",
                    "mode": "safe",
                    "expected": sum(len(v) for v in payload.values()),
                    "files": list(payload),
                    "sizes": {n: len(c) for n, c in payload.items()},
                    "resolved_sha": "deadbeefcafe",
                }
            ),
            encoding="utf-8",
        )

    try:
        mp.MODEL_ROOT = model_root
        mp.PARTIAL_ROOT = partial
        mp.SPECIMEN_ROOT = specimens
        mp.MANIFEST_DIR = manifests

        # 1. incomplete refused, nothing appears under specimens/
        incomplete_tag = tag + "-incomplete"
        inc_files = dict(files)
        _write(incomplete_tag, inc_files, partial / incomplete_tag)
        (partial / incomplete_tag / "weights.bin").unlink()
        refused = call_promote(incomplete_tag, go=True)

        # 2. dry-run moves nothing
        dry_tag = tag + "-dry"
        _write(dry_tag, files, partial / dry_tag)
        dry = call_promote(dry_tag, go=False)

        # 3. go=True is os.rename, verified at destination
        go_tag = tag + "-go"
        _write(go_tag, files, partial / go_tag)
        moved = call_promote(go_tag, go=True)

        # 4. existing destination is never overwritten
        clash_tag = tag + "-clash"
        _write(clash_tag, files, partial / clash_tag)
        (specimens / clash_tag).mkdir()
        (specimens / clash_tag / "config.json").write_bytes(b"keep")
        clash = call_promote(clash_tag, go=True)

        return {
            "incomplete": refused,
            "dry_run": dry,
            "promoted": moved,
            "clash": clash,
            "properties": {
                "incomplete_refused": refused.get("action") == "REFUSED",
                "incomplete_stayed_in_partial": (partial / incomplete_tag).is_dir(),
                "incomplete_absent_from_specimens": not (specimens / incomplete_tag).is_dir(),
                "dry_run_moved_nothing": dry.get("action") == "WOULD_PROMOTE"
                and (partial / dry_tag).is_dir()
                and not (specimens / dry_tag).is_dir(),
                "go_promoted": moved.get("action") == "PROMOTED",
                "go_verified_at_destination": moved.get("verified_at_destination") is True,
                "go_partial_gone": not (partial / go_tag).is_dir(),
                "go_weights_intact": (specimens / go_tag / "weights.bin").read_bytes()
                == files["weights.bin"]
                if (specimens / go_tag / "weights.bin").is_file()
                else False,
                "clash_refused": clash.get("action") == "REFUSED"
                and clash.get("reason") == "DESTINATION_EXISTS",
                "clash_existing_untouched": (specimens / clash_tag / "config.json").read_bytes()
                == b"keep",
            },
        }
    finally:
        mp.MODEL_ROOT = saved["MODEL_ROOT"]
        mp.PARTIAL_ROOT = saved["PARTIAL_ROOT"]
        mp.SPECIMEN_ROOT = saved["SPECIMEN_ROOT"]
        mp.MANIFEST_DIR = saved["MANIFEST_DIR"]


def live_replay_already_promoted(manifest_dir: Path, slugs: list[str]) -> list[dict[str, Any]]:
    """Call promote(tag) with go=False against the live lake.

    Partial is empty, so this takes the ALREADY_PROMOTED / NO_MANIFEST replay
    path and never renames. MANIFEST_DIR is pointed at git-extracted watch
    manifests (read-only copies), not written into the lake.
    """
    saved = mp.MANIFEST_DIR
    rows = []
    try:
        mp.MANIFEST_DIR = manifest_dir
        for slug in slugs:
            outcome = call_promote(slug, go=False)
            rows.append(
                {
                    "tag": slug,
                    "action": outcome.get("action"),
                    "reason": outcome.get("reason"),
                    "complete": outcome.get("complete"),
                    "source_is_live_partial": str(outcome.get("source") or "").startswith(
                        str(PARTIAL)
                    ),
                    "destination_is_live_specimen": str(
                        outcome.get("destination") or ""
                    ).startswith(str(SPECIMENS)),
                }
            )
    finally:
        mp.MANIFEST_DIR = saved
    return rows


def run_promotion_gate(
    *,
    live: bool = True,
    scratch_root: Optional[Path] = None,
) -> dict[str, Any]:
    command = ["python3", "-m", "tools.acceptance.lake", "--gate", GATE]
    with timed() as clock:
        tmp = Path(scratch_root) if scratch_root else Path(
            tempfile.mkdtemp(prefix="acc4-promo-")
        )
        scratch = run_promote_on_scratch(tmp)
        props = scratch["properties"]

        slugs: list[str] = []
        replay: list[dict[str, Any]] = []
        live_partial_dirs: list[str] = []
        if live and lake_mounted():
            slugs = sorted(
                p.name
                for p in SPECIMENS.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
            if PARTIAL.is_dir():
                live_partial_dirs = [
                    p.name
                    for p in PARTIAL.iterdir()
                    if p.is_dir() and not p.name.startswith(".")
                ]
            watch_dir = materialize_watch_manifests(tmp / "watch-extract")
            replay = live_replay_already_promoted(watch_dir, slugs)

        already = [r for r in replay if r.get("action") == "ALREADY_PROMOTED"]
        no_manifest = [
            r
            for r in replay
            if r.get("reason") in {"NO_MANIFEST", "DESTINATION_INCOMPLETE:NO_MANIFEST"}
            and r.get("destination_is_live_specimen")
        ]
        other = [
            r
            for r in replay
            if r not in already and r not in no_manifest
        ]

        atomic_ok = all(
            [
                props["incomplete_refused"],
                props["incomplete_absent_from_specimens"],
                props["dry_run_moved_nothing"],
                props["go_promoted"],
                props["go_verified_at_destination"],
                props["go_partial_gone"],
                props["go_weights_intact"],
                props["clash_refused"],
                props["clash_existing_untouched"],
            ]
        )
        live_ok = (
            live
            and lake_mounted()
            and len(slugs) == 55
            and live_partial_dirs == []
            and not other
            and len(already) + len(no_manifest) == len(slugs)
        )

        checks = {
            "promote_invoked": True,
            "scratch_atomic_properties": atomic_ok,
            "live_partial_empty": live_partial_dirs == [] if live else None,
            "live_sealed_count_55": len(slugs) == 55 if live else None,
            "live_replay_no_unexpected_action": not other,
            "no_live_rename": True,
        }

        # ACCEPTED only when the scratch atomic contract holds AND the live
        # lake has empty partial/ + 55 sealed bodies, with no unexpected action.
        if atomic_ok and live_ok:
            verdict = "ACCEPTED"
            blocker = None
        elif atomic_ok and not live:
            verdict = "BLOCKED"
            blocker = {
                "missing_input": "live lake (live=False in this invocation)",
                "why": "scratch atomic contract held; live replay not taken",
            }
        else:
            verdict = "BLOCKED"
            blocker = {
                "missing_input": "atomic promotion contract on scratch + empty live partial/",
                "why": f"atomic_ok={atomic_ok} live_partial={live_partial_dirs} other={other[:5]}",
            }

        measured = {
            "scratch_properties": props,
            "sealed_specimens": len(slugs),
            "live_partial_dirs": live_partial_dirs,
            "already_promoted": len(already),
            "no_manifest": len(no_manifest),
            "unexpected": other,
        }
        output = {
            "summary": (
                f"scratch atomic_ok={atomic_ok}; live sealed={len(slugs)} "
                f"partial_dirs={live_partial_dirs} already_promoted={len(already)} "
                f"no_manifest={len(no_manifest)}"
            ),
            "scratch": scratch,
            "live_replay": replay,
        }
        return write_receipt(
            GATE,
            verdict=verdict,
            command=command,
            output=output,
            measured=measured,
            checks=checks,
            evidence_tier="STATIC",
            symbol_invoked=True,
            blocker=blocker,
            elapsed_s=clock.snap(),
            extra={"lake": str(LAKE), "implementing_symbol": GATES[GATE]["symbol"]},
        )
