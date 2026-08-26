#!/usr/bin/env python3
"""Odyssey Model Lake — HDD stores, SSD stages, RAM/GPU executes.

Tier 2 is the external volume; Tier 1 is a bounded SSD hot cache. Acquisition goes
straight to Tier 2 (never network -> SSD -> HDD), resumable, hash-verified, and
atomic: a specimen is only visible as complete once every declared file's sha256
matches the upstream LFS oid, and the whole tree is renamed into place in one step.

Resume and integrity come from `hf download`, which already does range-resume and
etag checks; this module owns placement, the byte budget, the manifest, the atomic
publish, and the capacity event. Reimplementing HTTP range resume would be strictly
worse code for the same behaviour.
"""
import argparse, json, os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAKE = Path("/Volumes/corpdrive/hawking-modellake")
TIER2 = LAKE / "specimens"
PARTIAL = LAKE / "partial"
MANIFESTS = LAKE / "manifests"
SSD_STAGE = Path.home() / "noetic/stage"

# Tier 2 allocation: the directive's ~3.5 TB rolling window. Everything above it on
# the volume is protected headroom the lake may never consume.
TIER2_BUDGET = 3_500 * 10**9
# Tier 1: the SSD is the hot bench, not a second archive. Two specimens' worth.
TIER1_BUDGET = 140 * 2**30
# Never touched by the lake, at any tier. These predate it and are not ours.
PROTECTED = ["legal-scans-2026-08-23.tar.zst", "substrate",
             "legal-scans-2026-08-23.README.txt"]
# substrate-git-backup-20260824-190835.tar was on this list and is no longer on the
# volume. It was moved to /Volumes/corpdrive/.Trashes, which is Finder-owned and
# permission-denied to this process; the lake has no trash path and every delete it makes
# is constructed under LAKE. Recorded rather than quietly dropped from the list.
EXTERNALLY_REMOVED = [{"name": "substrate-git-backup-20260824-190835.tar",
                       "observed": "absent from /Volumes/corpdrive; .Trashes created the "
                                   "same minute the volume mtime changed",
                       "by": "an external GUI action, not this tool",
                       "lake_delete_surface": "retire() guarded by guard_protected; the "
                                              "demo cycle only removes paths built from "
                                              "PARTIAL/TIER2/SSD_STAGE"}]

API = "https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"


def _api(repo, rev):
    req = urllib.request.Request(API.format(repo=repo, rev=rev),
                                 headers={"User-Agent": "hawking-odyssey/1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def du(p):
    p = Path(p)
    if not p.exists():
        return 0
    out = subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True).stdout
    return int(out.split()[0]) * 1024 if out.strip() else 0


def free(p):
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize


def guard_protected(path):
    """Refuse any operation that would reach outside the lake root."""
    path = Path(path).resolve()
    if not str(path).startswith(str(LAKE.resolve())):
        raise PermissionError(f"outside the lake root: {path}")
    for name in PROTECTED:
        if str(path) == f"/Volumes/corpdrive/{name}" or str(path).startswith(f"/Volumes/corpdrive/{name}/"):
            raise PermissionError(f"protected path: {path}")
    return path


def tier2_used():
    return du(TIER2) + du(PARTIAL)


def admit(nbytes, tier):
    """Enforced, not advisory: an over-budget request is refused with a reason."""
    if tier == 2:
        used, budget, avail = tier2_used(), TIER2_BUDGET, free("/Volumes/corpdrive")
    elif tier == 1:
        used, budget, avail = du(SSD_STAGE), TIER1_BUDGET, free("/")
    else:
        raise ValueError(tier)
    if used + nbytes > budget:
        return False, (f"tier{tier} budget: {used + nbytes} would exceed {budget} "
                       f"(used {used}, request {nbytes})")
    if nbytes > avail * 0.95:
        return False, f"tier{tier} physical: request {nbytes} against {avail} free"
    return True, f"tier{tier} ok: used {used} + {nbytes} <= {budget}"


def sha256(path, buf=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def acquire(repo, rev, emit_progress=True):
    """Direct to tier 2. Resumable, hash-verified, atomic publish."""
    for d in (TIER2, PARTIAL, MANIFESTS):
        d.mkdir(parents=True, exist_ok=True)
    meta = _api(repo, rev)
    if meta.get("sha") != rev:
        raise ValueError(f"revision mismatch: asked {rev}, hub resolved {meta.get('sha')}")
    sibs = meta.get("siblings") or []
    want = sum(s.get("size") or 0 for s in sibs)
    ok, why = admit(want, tier=2)
    if not ok:
        return {"acquired": False, "refused": why, "repo": repo, "revision": rev}

    slug = repo.replace("/", "--") + "@" + rev[:12]
    part = guard_protected(PARTIAL / slug)
    final = guard_protected(TIER2 / slug)
    if final.exists():
        return {"acquired": True, "already_present": True, "path": str(final),
                "repo": repo, "revision": rev, "bytes": du(final)}

    part.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # hf_transfer is faster in isolation but buffers chunks in RAM: eight concurrent
    # workers held ~24 GB RSS and drove the box to 10.9 GB of 12.3 GB swap with 158 MB of
    # free pages. Throughput collapsed 160 -> 10 MiB/s from thrashing, not from CPU. On a
    # 96 GB box doing a 3 TB fill, the memory-cheap path wins.
    # hf_transfer is ~1.5x faster in isolation and is the WRONG choice for a long unattended
    # fill on this box, for two measured reasons. It buffers chunks in RAM: eight workers
    # held ~47 GB and drove free memory to 255 MB, collapsing throughput to 6.5 MiB/s. And
    # it does not resume in-flight chunks -- three restart cycles took the partials from
    # 7-9.5 GB each back down to 1.1-2.6 GB. The plain path's .incomplete files DO resume,
    # which matters more than peak rate when the run spans hours.
    env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="0")
    cmd = ["hf", "download", repo, "--revision", rev, "--local-dir", str(part)]
    r = subprocess.run(cmd, env=env, capture_output=not emit_progress, text=True)
    if r.returncode != 0:
        return {"acquired": False, "error": f"hf download exit {r.returncode}",
                "stderr": (r.stderr or "")[-2000:], "partial": str(part)}

    # Verify against the hub's own per-file oids. A file the hub reports without an
    # lfs oid is verified by size only, and that weaker check is recorded as such.
    verified, weak, bad = [], [], []
    for s in sibs:
        rel = s.get("rfilename")
        f = part / rel
        if not f.exists():
            bad.append({"file": rel, "why": "missing after download"})
            continue
        oid = (s.get("lfs") or {}).get("sha256") or (s.get("lfs") or {}).get("oid")
        if oid:
            got = sha256(f)
            (verified if got == oid else bad).append(
                {"file": rel, "expected": oid, "got": got} if got != oid else rel)
        else:
            if s.get("size") is not None and f.stat().st_size != s["size"]:
                bad.append({"file": rel, "why": "size mismatch",
                            "expected": s["size"], "got": f.stat().st_size})
            else:
                weak.append(rel)
    if bad:
        return {"acquired": False, "hash_rejected": bad[:20], "n_rejected": len(bad),
                "partial": str(part)}

    os.rename(part, final)                      # atomic: never half-visible in TIER2
    manifest = {
        "repo": repo, "revision": rev, "resolved_sha": meta.get("sha"),
        "path": str(final), "bytes": du(final), "n_files": len(sibs),
        "n_sha256_verified": len(verified), "n_size_only_verified": len(weak),
        "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(time.time() - t0, 1),
        "reacquisition": f"hf download {repo} --revision {rev} --local-dir <dest>",
    }
    (MANIFESTS / f"{slug}.json").write_text(json.dumps(manifest, indent=1))
    return {"acquired": True, **manifest}


def stage(slug, dest=None):
    """Tier 2 -> tier 1. Bounded; refuses rather than filling the SSD."""
    src = guard_protected(TIER2 / slug)
    if not src.exists():
        return {"staged": False, "why": f"not in tier 2: {src}"}
    n = du(src)
    ok, why = admit(n, tier=1)
    if not ok:
        return {"staged": False, "refused": why}
    SSD_STAGE.mkdir(parents=True, exist_ok=True)
    d = Path(dest) if dest else SSD_STAGE / slug
    if not d.exists():
        subprocess.run(["cp", "-c", "-R", str(src), str(d)], check=True)  # APFS clone
    return {"staged": True, "path": str(d), "bytes": du(d), "reason": why}


def retire(slug):
    """Relegate tier-2 bytes. Reversible: the manifest keeps the reacquisition recipe."""
    p = guard_protected(TIER2 / slug)
    man = MANIFESTS / f"{slug}.json"
    if not man.exists():
        return {"retired": False, "why": "no manifest — refusing to delete an unrecorded specimen"}
    freed = du(p)
    if p.exists():
        shutil.rmtree(p)
    return {"retired": True, "slug": slug, "bytes_freed": freed,
            "reacquisition": json.loads(man.read_text())["reacquisition"]}


def resident_slugs():
    # dotfiles are not specimens; Finder leaves .DS_Store in any directory it displays
    return ({p.name for p in TIER2.iterdir() if p.is_dir() and not p.name.startswith(".")}
            if TIER2.exists() else set())


def capacity_event(reason, freed_bytes, start=False):
    """A capacity release advances the queue with no human step.

    Presence is re-measured against the lake's own tier 2, never read from the selection
    receipt's `disk` snapshot -- that snapshot was taken before the current specimen was
    acquired and selected an already-resident model on the first run.
    """
    sel = REPO / "receipts/headless/MODEL_2_SELECTION.json"
    here, nxt, skipped = resident_slugs(), None, []
    if sel.exists():
        d = json.load(open(sel))
        pool = []
        for r in d.get("candidates", []):
            if not (r.get("score") and r["score"].get("acquirable")):
                continue
            slug = r["canonical_source"].replace("/", "--") + "@" + r["canonical_revision"][:12]
            if slug in here:
                skipped.append({"oxx": r["oxx"], "why": "already resident in tier 2"})
                continue
            pool.append((r, slug))
        pool.sort(key=lambda t: -t[0]["score"]["information_gain_per_gib"])
        if pool:
            r, slug = pool[0]
            nxt = {"oxx": r["oxx"], "repo": r["canonical_source"],
                   "revision": r["canonical_revision"],
                   "download_gib": r["score"]["download_gib"], "slug": slug}
    ev = {"event": "CAPACITY_RELEASED", "reason": reason, "bytes_freed": freed_bytes,
          "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "resident_at_event": sorted(here), "skipped_already_resident": skipped,
          "scheduler_selected_next": nxt, "human_step_required": False}
    if start and nxt:
        ok, why = admit(int(nxt["download_gib"] * 2**30), 2)
        if ok:
            log = LAKE / f"autoadvance-{nxt['slug']}.log"
            LAKE.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "acquire",
                 "--repo", nxt["repo"], "--revision", nxt["revision"]],
                stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=str(REPO))
            ev["acquisition_started"] = {"pid": proc.pid, "log": str(log),
                                         "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                     time.gmtime())}
        else:
            ev["acquisition_refused"] = why
    return ev


def verify_only(repo, rev, root):
    """Re-verify an existing tree against the hub's oids. Used to prove that a corrupted
    file is REJECTED rather than accepted."""
    meta = _api(repo, rev)
    bad, ok = [], 0
    for s in meta.get("siblings") or []:
        f = Path(root) / s["rfilename"]
        oid = (s.get("lfs") or {}).get("sha256") or (s.get("lfs") or {}).get("oid")
        if not f.exists():
            bad.append({"file": s["rfilename"], "why": "missing"})
            continue
        if oid:
            got = sha256(f)
            if got != oid:
                bad.append({"file": s["rfilename"], "expected": oid, "got": got})
            else:
                ok += 1
        elif s.get("size") is not None and f.stat().st_size != s["size"]:
            bad.append({"file": s["rfilename"], "why": "size mismatch"})
        else:
            ok += 1
    return {"verified": not bad, "n_ok": ok, "rejected": bad[:10], "n_rejected": len(bad)}


def demo_cycle(repo, rev, emit):
    """One full turn of the circular buffer, on a real small specimen, with the two
    integrity properties proven by breaking them rather than by asserting them."""
    import shutil as _sh
    slug = repo.replace("/", "--") + "@" + rev[:12]
    log = []

    def step(name, **kw):
        log.append({"step": name, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw})
        print(f"[{len(log)}] {name}: {json.dumps(kw)[:160]}")

    # 0. budget refusals, enforced not advisory
    ok4t, why4t = admit(4_000 * 10**9, 2)
    ok1t, why1t = admit(200 * 2**30, 1)
    step("budget_refusals", tier2_4TB_admitted=ok4t, tier2_reason=why4t,
         tier1_200GiB_admitted=ok1t, tier1_reason=why1t)

    # 1. interrupted acquisition, then resume
    part = guard_protected(PARTIAL / slug)
    if part.exists():
        _sh.rmtree(part)
    if (TIER2 / slug).exists():
        _sh.rmtree(guard_protected(TIER2 / slug))
    part.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["hf", "download", repo, "--revision", rev,
                             "--local-dir", str(part)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    waited = 0.0
    while waited < 25 and proc.poll() is None and du(part) < 200 * 2**20:
        time.sleep(1.0)
        waited += 1.0
    proc.kill(); proc.wait()
    step("interrupted", partial_bytes=du(part), killed_after_s=waited)

    r = acquire(repo, rev, emit_progress=False)
    step("resumed_and_verified", **{k: r.get(k) for k in
                                    ("acquired", "bytes", "n_files", "n_sha256_verified",
                                     "n_size_only_verified", "wall_s")})
    if not r.get("acquired"):
        step("FAILED", detail=r)
        Path(emit).write_text(json.dumps({"pass": False, "cycle": log}, indent=1))
        return 1

    # 2. corruption must be REJECTED, not accepted
    tree = Path(r["path"])
    victim = max((f for f in tree.rglob("*.safetensors")), key=lambda f: f.stat().st_size)
    backup = victim.read_bytes()[:1 << 20]
    with open(victim, "r+b") as f:
        f.seek(0)
        f.write(b"\x00" * 4096)
    v = verify_only(repo, rev, tree)
    with open(victim, "r+b") as f:
        f.seek(0)
        f.write(backup[:4096])
    step("corruption_rejected", verified=v["verified"], n_rejected=v["n_rejected"],
         first=v["rejected"][:1])
    v2 = verify_only(repo, rev, tree)
    step("restored_and_reverified", verified=v2["verified"], n_ok=v2["n_ok"])

    # 3. stage to the bounded SSD hot cache
    st = stage(slug)
    step("staged_to_tier1", **{k: st.get(k) for k in ("staged", "path", "bytes", "refused")})

    # 4. study window (out of scope here; the buffer only needs the slot to exist)
    step("study_window", note="representation search happens here; not part of this demo")

    # 5. retire, and the capacity event that must advance the queue with no human step
    if st.get("staged") and Path(st["path"]).exists():
        staged = Path(st["path"]).resolve()
        # the SSD stage is outside the lake root, so guard_protected does not apply; assert
        # containment explicitly rather than trusting that stage() built the path
        if not str(staged).startswith(str(SSD_STAGE.resolve())):
            raise PermissionError(f"refusing to remove outside the SSD stage: {staged}")
        _sh.rmtree(staged)
    ret = retire(slug)
    step("retired", **{k: ret.get(k) for k in ("retired", "bytes_freed", "reacquisition")})
    ev = capacity_event(f"retired {slug}", ret.get("bytes_freed", 0), start=True)
    step("capacity_event", **ev)

    out = {
        "schema": "hawking.headless.model_lake.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/modellake.py",
        "obligation": "G012 — MODEL_LAKE_ROLLING_PIPELINE (directive §19-§25, §18)",
        "lake_root": str(LAKE), "tier2_budget": TIER2_BUDGET, "tier1_budget": TIER1_BUDGET,
        "protected_paths_never_touched": PROTECTED,
        "externally_removed_not_by_this_tool": EXTERNALLY_REMOVED,
        "demo_specimen": {"repo": repo, "revision": rev},
        "resident_specimens": sorted(resident_slugs()),
        "manifests": sorted(p.name for p in MANIFESTS.glob("*.json")),
        "cycle": log,
        "pass": bool(r.get("acquired") and not v["verified"] and v2["verified"]
                     and ret.get("retired") and ev["scheduler_selected_next"]
                     and ev.get("acquisition_started")
                     and not ok4t and not ok1t),
    }
    Path(emit).write_text(json.dumps(out, indent=1))
    print("pass:", out["pass"])
    return 0 if out["pass"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["acquire", "stage", "retire", "status", "admit",
                                    "demo-cycle", "verify"])
    ap.add_argument("--repo"); ap.add_argument("--revision"); ap.add_argument("--slug")
    ap.add_argument("--bytes", type=int); ap.add_argument("--tier", type=int, default=2)
    ap.add_argument("--emit")
    a = ap.parse_args()

    if a.cmd == "demo-cycle":
        return demo_cycle(a.repo, a.revision, a.emit or "/dev/stdout")
    if a.cmd == "verify":
        out = verify_only(a.repo, a.revision, TIER2 / a.slug)
    elif a.cmd == "acquire":
        out = acquire(a.repo, a.revision)
    elif a.cmd == "stage":
        out = stage(a.slug)
    elif a.cmd == "retire":
        out = retire(a.slug)
    elif a.cmd == "admit":
        ok, why = admit(a.bytes, a.tier)
        out = {"admitted": ok, "reason": why}
    else:
        out = {"lake_root": str(LAKE), "tier2_used": tier2_used(),
               "tier2_budget": TIER2_BUDGET, "tier1_used": du(SSD_STAGE),
               "tier1_budget": TIER1_BUDGET,
               "free_hdd": free("/Volumes/corpdrive"), "free_ssd": free("/"),
               "specimens": sorted(resident_slugs()),
               "protected_untouched": [n for n in PROTECTED
                                       if Path(f"/Volumes/corpdrive/{n}").exists()],
               "externally_removed": EXTERNALLY_REMOVED}
    print(json.dumps(out, indent=1))
    if a.emit:
        Path(a.emit).write_text(json.dumps(out, indent=1))
    return 0 if out.get("acquired", out.get("staged", out.get("retired", out.get("admitted", True)))) is not False else 1


if __name__ == "__main__":
    sys.exit(main())
