#!/usr/bin/env python3
"""dismantle headbank pull — fetch a model's Eagle5 head + runtime profile.

Reads a ``headbank_manifest.json`` (either a local copy or one downloaded
from Drive / GitHub release) and stages the artifacts for a given model
slug into ``$DISMANTLE_HOME/headbank/<slug>/``. Emits a shell snippet you
can ``source`` to set the runtime env vars.

Usage
-----
    # List every model in the bank
    python3 tools/headbank/pull.py --manifest path/to/headbank_manifest.json --list

    # Stage a specific model
    python3 tools/headbank/pull.py --manifest path/to/headbank_manifest.json --slug q3b

    # Stage a specific model and emit an env file to source
    python3 tools/headbank/pull.py --manifest ... --slug q3b --env-file ~/.dismantle/env

After staging, the runtime profile JSON path is printed. Source the env
file then run ``dismantle bench`` / ``dismantle serve`` and the Eagle5
head + AWQ scales + locked-config env are all in place.

Manifest schema is ``dismantle-headbank-manifest-v1`` produced by
``colab/maximal_spec_headbank_500u.ipynb``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def _dismantle_home() -> Path:
    return Path(os.environ.get("DISMANTLE_HOME") or (Path.home() / ".dismantle")).resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_local(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and src.samefile(dst):
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _stage_entry(entry: dict, headbank_root: Path, manifest_dir: Path, *, verify_sha: bool) -> dict:
    slug = entry["slug"]
    dst_dir = headbank_root / slug
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Resolve every path. Manifests written by the Colab notebook contain
    # absolute Drive paths; when running locally those paths won't exist
    # so we also try ``<manifest_dir>/<slug>/<filename>`` which matches the
    # Drive export layout (``dismantle_export/headbank_500u/<slug>/...``).
    def _resolve(maybe_abs: str | None, *fallback_relative: str) -> Path | None:
        if not maybe_abs:
            return None
        p = Path(maybe_abs)
        if p.is_file():
            return p
        for rel in fallback_relative:
            cand = manifest_dir / rel
            if cand.is_file():
                return cand
        return None

    staged: dict[str, str | None] = {"slug": slug}

    head_src = _resolve(entry.get("head_path"), f"{slug}/heads", f"heads/{slug}")
    if head_src is None:
        # Fall back to scanning the slug heads folder for a *.safetensors.
        for cand in sorted((manifest_dir / slug / "heads").glob("*.safetensors")) if (manifest_dir / slug / "heads").exists() else []:
            head_src = cand
            break
    if head_src is None:
        raise FileNotFoundError(f"can't locate head safetensors for slug={slug}; manifest said {entry.get('head_path')}")
    head_dst = dst_dir / "head.safetensors"
    _copy_local(head_src, head_dst)
    staged["head"] = str(head_dst)
    if verify_sha and entry.get("head_sha256"):
        got = _sha256_file(head_dst)
        if got != entry["head_sha256"]:
            raise RuntimeError(f"head sha mismatch for {slug}: expected {entry['head_sha256']}, got {got}")
        staged["head_sha256"] = got

    awq_src = _resolve(entry.get("awq_scales"), f"{slug}/awq/awq_smoothing.json")
    if awq_src is not None:
        awq_dst = dst_dir / "awq_smoothing.json"
        _copy_local(awq_src, awq_dst)
        staged["awq_scales"] = str(awq_dst)
    else:
        staged["awq_scales"] = None

    profile_src = _resolve(entry.get("runtime_profile"),
                           f"{slug}/runtime_profiles/{slug}_winner.runtime.json")
    if profile_src is None:
        raise FileNotFoundError(f"can't locate runtime profile for slug={slug}")
    profile_dst = dst_dir / "runtime_profile.json"
    _copy_local(profile_src, profile_dst)
    staged["runtime_profile"] = str(profile_dst)

    # Patch the staged runtime profile so EAGLE5_HEAD / DISMANTLE_AWQ_SCALES
    # point at the local copies, not the Drive original.
    payload = json.loads(profile_dst.read_text())
    env = dict(payload.get("runtime_env") or {})
    env["EAGLE5_HEAD"] = str(head_dst)
    if staged["awq_scales"]:
        env["DISMANTLE_AWQ_SCALES"] = staged["awq_scales"]
    payload["runtime_env"] = env
    payload["head"] = str(head_dst)
    payload["awq_scales"] = staged["awq_scales"]
    profile_dst.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    staged["metrics"] = entry.get("metrics") or {}
    return staged


def main() -> int:
    p = argparse.ArgumentParser(prog="dismantle-headbank-pull")
    p.add_argument("--manifest", required=True, type=Path,
                   help="path to headbank_manifest.json (Drive export or local copy)")
    p.add_argument("--slug", help="model slug to stage (e.g. q3b, q7b, q05b, dsv2)")
    p.add_argument("--list", action="store_true", help="list available models and exit")
    p.add_argument("--env-file", type=Path,
                   help="if set, write a sh-compatible env file you can `source`")
    p.add_argument("--home", type=Path, default=None,
                   help="override DISMANTLE_HOME (defaults to $DISMANTLE_HOME or ~/.dismantle)")
    p.add_argument("--no-verify-sha", action="store_true",
                   help="skip sha256 check after copy")
    args = p.parse_args()

    if not args.manifest.exists():
        sys.stderr.write(f"manifest not found: {args.manifest}\n")
        return 2

    manifest = json.loads(args.manifest.read_text())
    entries = manifest.get("entries") or []
    if not entries:
        sys.stderr.write("manifest has no entries\n")
        return 2

    if args.list or not args.slug:
        print(f"manifest: {args.manifest} ({manifest.get('repo_sha')})")
        print(f"models in bank: {len(entries)}")
        for e in entries:
            m = e.get("metrics") or {}
            print(
                f"  {e['slug']:6s} hf={e.get('hf_id'):48s} "
                f"tps={m.get('offline_projected_tps') or 0:>7.1f} "
                f"tau={m.get('tau') or 0:>5.2f} "
                f"policy={(m.get('policy_kind') or '-'):>24s}"
            )
        if args.list:
            return 0
        sys.stderr.write("--slug required when not --list\n")
        return 2

    by_slug = {e["slug"]: e for e in entries}
    if args.slug not in by_slug:
        sys.stderr.write(f"unknown slug {args.slug}; available: {sorted(by_slug)}\n")
        return 2

    home = args.home or _dismantle_home()
    headbank_root = home / "headbank"
    headbank_root.mkdir(parents=True, exist_ok=True)

    manifest_dir = args.manifest.parent
    staged = _stage_entry(by_slug[args.slug], headbank_root, manifest_dir,
                          verify_sha=not args.no_verify_sha)
    print(f"[headbank] staged {args.slug} → {headbank_root / args.slug}")
    print(f"[headbank] runtime profile: {staged['runtime_profile']}")

    profile_payload = json.loads(Path(staged["runtime_profile"]).read_text())
    env = profile_payload.get("runtime_env") or {}
    if args.env_file:
        args.env_file.parent.mkdir(parents=True, exist_ok=True)
        with args.env_file.open("w") as f:
            f.write(f"# dismantle headbank env for slug={args.slug}\n")
            f.write(f"# generated from {args.manifest}\n")
            for k, v in sorted(env.items()):
                # quote conservatively; values are simple strings.
                f.write(f'export {k}="{v}"\n')
        print(f"[headbank] wrote env file: {args.env_file}")
        print(f"[headbank] then: source {args.env_file} && dismantle bench --prompt 'hello'")
    else:
        print("[headbank] runtime env (export these or pass --env-file):")
        for k, v in sorted(env.items()):
            print(f'  export {k}="{v}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
