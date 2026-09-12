#!/usr/bin/env python3
"""Deterministic discriminator for the initial Claude -> Codex Exodus pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "CLAUDE_CODEX_EXODUS.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-integrity",
        action="store_true",
        help="also compare the source hashes recorded in the manifest",
    )
    args = parser.parse_args()

    required = [
        ROOT / "AGENTS.md",
        ROOT / "CLAUDE-CODEX-EXODUS.md",
        ROOT / "CLAUDE_ONLY_REMAINING",
        ROOT / "docs" / "CLAUDE_CODEX_EXODUS_HCLI_HANDOFF.md",
        MANIFEST,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        fail("missing migration artifact(s): " + ", ".join(missing))

    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"manifest is not valid JSON: {exc}")

    if data.get("schema_version") != "1.0":
        fail("unsupported manifest schema")
    if data.get("source_mutated") is not False:
        fail("manifest does not assert source preservation")
    if data.get("source_recoverable") is not True:
        fail("manifest does not assert source recoverability")

    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for required_text in ("H-MANIFESTO.md", "CLAUDE-CODEX-EXODUS.md", "HCLI", "receipts"):
        if required_text not in agents:
            fail(f"AGENTS.md does not mention {required_text}")

    remaining = (ROOT / "CLAUDE_ONLY_REMAINING").read_text(encoding="utf-8")
    if "no high-value Hawking capability" not in remaining:
        fail("Claude-only boundary is not explicit")

    command_source = ROOT / "hcli" / "commands.py"
    command_text = command_source.read_text(encoding="utf-8")
    for command in ("/ultragoal", "/mission", "/steer", "/grok", "/receipts"):
        if f'"{command}"' not in command_text:
            fail(f"HCLI command registry is missing {command}")

    for entry in data.get("source_integrity", []):
        path = Path(entry["path"])
        if not path.is_file():
            fail(f"recorded source is missing: {path}")
        if args.source_integrity and sha256(path) != entry["sha256"]:
            fail(f"source hash changed: {path}")

    # The manifest must not contain obvious credential-shaped material. This is
    # a cheap guard, not a claim that a regex can prove a filesystem is secret-free.
    manifest_text = MANIFEST.read_text(encoding="utf-8")
    if re.search(r"(?i)(sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]{20,})", manifest_text):
        fail("credential-shaped value found in manifest")

    help_result = subprocess.run(
        [sys.executable, "-m", "hcli", "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if help_result.returncode != 0:
        fail(f"python -m hcli --help exited {help_result.returncode}")

    print("PASS: Exodus artifacts, HCLI command surface, handoff, and manifest invariants are present")
    print(f"PASS: {len(data.get('inventory', []))} semantic inventory records")
    print(f"PASS: {len(data.get('source_integrity', []))} source hashes recorded")
    if args.source_integrity:
        print("PASS: recorded source hashes match")
    print("PASS: python -m hcli --help")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
