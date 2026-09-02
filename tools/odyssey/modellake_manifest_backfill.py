#!/usr/bin/env python3
"""Backfill lake manifests offline, from evidence the bodies already carry.

44 of 55 specimens reached tier 2 without a manifest. That is not cosmetic:
modellake.retire() refuses to relegate a specimen it has no manifest for, so a
lake that is already over its tier-2 budget could not retire 80% of itself.

The missing recipe does not need the network. `hf download` leaves
<body>/.cache/huggingface/download/<rel>.metadata whose first line is the full
commit sha and whose second line is the hub etag (the sha256 oid for an LFS
file). The slug's 12-char revision must prefix that sha or this refuses to
write -- a manifest whose revision disagrees with the body is worse than none.

What this does NOT claim: it does not recompute sha256. `hf` verified those at
download time; re-reading 4 TB off an HDD to restate a check that already
passed is not evidence, it is a long wait. Backfilled manifests say so in
`backfill.verified_now` rather than inheriting acquire()'s verified counts.

Byte convention: `bytes` is the sum of st_size over regular files outside
.cache -- the same number tools.odyssey.modellake_index puts in the catalog, so
a manifest and the catalog agree instead of producing a stale_manifest_bytes
anomaly per specimen. acquire()'s du() allocated figure is kept beside it as
`bytes_allocated` where it is known.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.odyssey.modellake import MANIFESTS, TIER2  # noqa: E402

CACHE = ".cache"
HUB_META = "huggingface/download"


def body_files(body: Path) -> list[Path]:
    """Regular files that are the specimen, excluding the downloader's cache."""
    return sorted(p for p in body.rglob("*")
                  if p.is_file() and not p.is_symlink()
                  and CACHE not in p.relative_to(body).parts)


def hub_metadata(body: Path) -> tuple[str | None, set[str], list[str]]:
    """(commit sha, body-relative paths with an etag, disagreeing shas)."""
    root = body / CACHE / HUB_META
    if not root.is_dir():
        return None, set(), []
    shas: dict[str, int] = {}
    with_etag: set[str] = set()
    for meta in root.rglob("*.metadata"):
        try:
            lines = meta.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if not lines or len(lines[0].strip()) != 40:
            continue
        shas[lines[0].strip()] = shas.get(lines[0].strip(), 0) + 1
        if len(lines) > 1 and lines[1].strip():
            rel = str(meta.relative_to(root))[: -len(".metadata")]
            with_etag.add(rel)
    if not shas:
        return None, set(), []
    winner = max(shas, key=lambda s: shas[s])
    return winner, with_etag, sorted(s for s in shas if s != winner)


def split_slug(slug: str) -> tuple[str, str]:
    """`org--name@rev12` -> (`org/name`, rev12). Split on the FIRST `--` only:
    model names contain hyphens, org names do not contain `--`."""
    name, _, rev = slug.rpartition("@")
    org, sep, model = name.partition("--")
    if not sep or not rev:
        raise ValueError(f"not a lake slug: {slug}")
    return f"{org}/{model}", rev


def build(body: Path) -> dict:
    slug = body.name
    repo, rev12 = split_slug(slug)
    sha, with_etag, disagreeing = hub_metadata(body)
    if sha is None:
        return {"slug": slug, "written": False,
                "why": "no .cache/huggingface/download metadata: revision is not "
                       "recoverable offline for this body"}
    if not sha.startswith(rev12):
        return {"slug": slug, "written": False,
                "why": f"revision disagrees: slug says {rev12}, body metadata says {sha}"}
    files = body_files(body)
    return {
        "slug": slug, "written": True, "repo": repo, "revision": sha,
        "resolved_sha": sha, "path": str(body),
        "bytes": sum(f.stat().st_size for f in files), "n_files": len(files),
        "n_files_with_hub_etag": len(with_etag),
        "reacquisition": f"hf download {repo} --revision {sha} --local-dir <dest>",
        "disagreeing_shas": disagreeing,
    }


def manifest_doc(rec: dict, *, prior: dict | None = None) -> dict:
    """Backfilled manifest. A prior acquire()-written doc keeps its own fields;
    only `bytes` is restated in the catalog's convention, with acquire()'s du()
    figure preserved as bytes_allocated rather than overwritten."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if prior is not None:
        out = dict(prior)
        if out.get("bytes") is not None and out["bytes"] != rec["bytes"]:
            out.setdefault("bytes_allocated", out["bytes"])
        out["bytes"] = rec["bytes"]
        out["bytes_are"] = "sum of st_size over regular files outside .cache"
        out["restated_at"] = now
        return out
    doc = {
        "repo": rec["repo"], "revision": rec["revision"],
        "resolved_sha": rec["resolved_sha"], "path": rec["path"],
        "bytes": rec["bytes"],
        "bytes_are": "sum of st_size over regular files outside .cache",
        "n_files": rec["n_files"],
        "acquired_at": None,
        "reacquisition": rec["reacquisition"],
        "provenance": "backfill",
        "backfill": {
            "source": "<body>/.cache/huggingface/download/**/*.metadata, written by "
                      "`hf download` at acquisition",
            "revision_agrees_with_slug": True,
            "n_files_with_hub_etag": rec["n_files_with_hub_etag"],
            "n_files_without_hub_etag": rec["n_files"] - rec["n_files_with_hub_etag"],
            "verified_now": "revision and byte count only. Per-file sha256 was checked "
                            "by `hf` at download time and is NOT re-checked here.",
            "backfilled_at": now,
        },
    }
    if rec["disagreeing_shas"]:
        doc["backfill"]["disagreeing_shas_in_body"] = rec["disagreeing_shas"]
    return doc


def run(*, apply: bool, restate: bool) -> dict:
    bodies = sorted(p for p in TIER2.iterdir()
                    if p.is_dir() and not p.name.startswith("."))
    written, restated, skipped, refused = [], [], [], []
    for body in bodies:
        man = MANIFESTS / f"{body.name}.json"
        prior = None
        if man.is_file():
            if not restate:
                skipped.append(body.name)
                continue
            prior = json.loads(man.read_text(encoding="utf-8"))
        rec = build(body)
        if not rec["written"]:
            refused.append(rec)
            continue
        if apply:
            MANIFESTS.mkdir(parents=True, exist_ok=True)
            man.write_text(json.dumps(manifest_doc(rec, prior=prior), indent=1) + "\n")
        (restated if prior is not None else written).append(rec["slug"])
    return {
        "applied": apply, "n_bodies": len(bodies),
        "n_written": len(written), "written": written,
        "n_restated": len(restated), "restated": restated,
        "n_skipped_existing": len(skipped),
        "n_refused": len(refused), "refused": refused,
        "manifests_dir": str(MANIFESTS),
    }


def demo() -> None:
    """One runnable check: the two things that can silently corrupt a manifest are
    slug parsing and a revision that does not match the body."""
    assert split_slug("Qwen--Qwen3-VL-8B-Instruct@0c351dd01ed8") == (
        "Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed8")
    assert split_slug("depth-anything--Depth-Anything-V2-Large-hf@7581137eff8d") == (
        "depth-anything/Depth-Anything-V2-Large-hf", "7581137eff8d")
    for bad in ("no-at-sign", "nodashes@abc"):
        try:
            split_slug(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should not parse as a slug")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        body = Path(td) / "org--m@aaaaaaaaaaaa"
        meta = body / CACHE / HUB_META
        meta.mkdir(parents=True)
        (body / "w.safetensors").write_bytes(b"x" * 100)
        (meta / "w.safetensors.metadata").write_text("a" * 40 + "\netag\n1.0\n")
        rec = build(body)
        assert rec["written"] and rec["bytes"] == 100 and rec["n_files"] == 1, rec
        assert rec["n_files_with_hub_etag"] == 1, rec
        # a body whose metadata names a different commit must be refused, not written
        (meta / "w.safetensors.metadata").write_text("b" * 40 + "\netag\n1.0\n")
        assert build(body)["written"] is False, "revision mismatch must refuse"
    print("demo ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write manifests (default: dry run)")
    ap.add_argument("--restate", action="store_true",
                    help="also restate `bytes` on manifests that already exist")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
    else:
        print(json.dumps(run(apply=a.apply, restate=a.restate), indent=1))
