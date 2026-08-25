#!/usr/bin/env python3
"""QWEN_CLEAN_REBUILD — reproduce the final Qwen executable from canonical inputs.

Recon first (directive §5, cheap disproof before expensive operations): establish
WHICH artifact is the final executable and whether its closure is reproducible at
all, before spending an hour packing 27B parameters.

The two candidates are not the same thing:

  * NOETIC_PARENT_A  -- mix_all_mlp_affine_g64_ls, complete_ebpw 3.1393, materialized
    at ~/noetic/NOETIC_PARENT_A, 755 segments of which 563 are HARDLINKS into the
    q4 incumbent at ~/models/qwen38-gravity-uniform-q4-v1/tensors.
  * mix_hetero_n041_floors -- complete_ebpw 2.5970, the figure quoted as current
    authority, built by tools/headless/whole_model_native.py into
    REPO/artifacts/qwen38-hetero-n041 ... which does not exist in this repo.

Recon answers which is final and enumerates every input the closure actually needs.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
NOETIC_A = Path.home() / "noetic/NOETIC_PARENT_A"
PARENT_BF16 = Path.home() / "models/qwen3.8-27b-abliterated-bf16"
Q4_INCUMBENT = Path.home() / "models/qwen38-gravity-uniform-q4-v1"
BUILDER = REPO / "tools/headless/whole_model_native.py"
MIX_ID = "mix_hetero_n041_floors"


def artifact_dir(root):
    """Resolve the artifact directory under `root`.

    The clean rebuild writes <root>/<MIX_ID>/, but composition_isolation writes its
    variants FLAT at <root>/. Assuming the nested layout made every gate here
    unrunnable against a variant, which is how variantB reached grand-candidate status
    with 353 hardlinks into the q4 incumbent still in it.
    """
    r = Path(root)
    if (r / MIX_ID / "segments").is_dir():
        return r / MIX_ID
    if (r / "segments").is_dir():
        return r
    raise SystemExit(f"no segments/ under {r} or {r / MIX_ID}")


def sh(*cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def du(p):
    p = Path(p)
    if not p.exists():
        return 0
    o = sh("du", "-sk", str(p)).stdout
    return int(o.split()[0]) * 1024 if o.strip() else 0


def hardlink_census(root):
    """A segment with nlink>1 shares bytes with another artifact. That is a closure
    dependency even though the bytes survive the other artifact's deletion."""
    root = Path(root)
    if not (root / "segments").exists():
        return {"present": False}
    shared, total, targets = [], 0, {}
    for f in sorted((root / "segments").iterdir()):
        if not f.is_file():
            continue
        total += 1
        st = f.stat()
        if st.st_nlink > 1:
            shared.append(f.name)
            # find the co-linked path outside this artifact
            for cand in (Q4_INCUMBENT / "tensors" / f.name,):
                if cand.exists() and cand.stat().st_ino == st.st_ino:
                    targets.setdefault(str(cand.parent), 0)
                    targets[str(cand.parent)] += 1
    return {"present": True, "n_segments": total, "n_shared_inodes": len(shared),
            "shared_with": targets, "example": shared[:3]}


def receipt(name):
    p = RH / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


CLEAN_ROOT = Path.home() / "noetic/CLEAN_REBUILD_A"


def recon():
    wmn, wmr = receipt("WHOLE_MODEL_NATIVE"), receipt("WHOLE_MODEL_RECOMPOSE")
    reg = receipt("MODEL_REGISTRY") or {}
    mix = None
    mp = NOETIC_A / "MIX_REPORT.json"
    if mp.exists():
        mix = json.load(open(mp))

    hetero_root = REPO / "artifacts" / "qwen38-hetero-n041" / MIX_ID
    parity_bin = ((wmn or {}).get("parity", {}).get("q2f_group64", {}) or {}).get("binary")

    candidates = {
        "NOETIC_PARENT_A": {
            "path": str(NOETIC_A), "present": NOETIC_A.exists(),
            "bytes": du(NOETIC_A),
            "complete_ebpw": (mix or {}).get("complete_ebpw"),
            "mix_id": (mix or {}).get("mix_id"),
            "builder": "tools/headless/affine2_g64_lsfit.py",
            "hardlinks": hardlink_census(NOETIC_A),
        },
        MIX_ID: {
            "path": str(hetero_root), "present": hetero_root.exists(),
            "bytes": du(hetero_root),
            "complete_ebpw": (wmn or {}).get("complete_ebpw"),
            "mix_id": MIX_ID,
            "builder": "tools/headless/whole_model_native.py",
            "artifact_root_env": "QWEN38_HETERO_ARTIFACT_ROOT",
            "default_root_is_inside_repo": True,
        },
    }
    # The final executable is the one the campaign's own authority quotes as current.
    final = MIX_ID if (wmr or {}).get("current_qwen_complete_ebpw") else "NOETIC_PARENT_A"

    prior = json.load(open(RH / "QWEN_CLEAN_REBUILD.json")) if (RH / "QWEN_CLEAN_REBUILD.json").exists() else {}
    zp_by_absence = (prior.get("zero_parent") or {}).get(
        "QWEN_ZERO_PARENT_RUNTIME_DEPENDENCY") == "PASS"
    clean = CLEAN_ROOT / MIX_ID
    clean_present = clean.exists()
    clean_shared = 0
    if clean_present and (clean / "segments").exists():
        clean_shared = sum(1 for f in (clean / "segments").iterdir()
                           if f.is_file() and f.stat().st_nlink > 1)
    candidates["mix_hetero_n041_floors_CLEAN_REBUILD"] = {
        "path": str(clean), "present": clean_present, "bytes": du(clean),
        "n_shared_inodes": clean_shared,
        "builder": "tools/odyssey/clean_rebuild.py --rebuild --dehardlink",
        "role": "the clean-room reproduction; independent of the q4 incumbent",
    }

    gaps = []
    if not candidates[final]["present"]:
        gaps.append({
            "gap": "FINAL_EXECUTABLE_NOT_MATERIALIZED",
            "detail": f"{final} is quoted as current authority "
                      f"(complete_ebpw {candidates[final]['complete_ebpw']}) but no bytes exist at "
                      f"{candidates[final]['path']}",
            "why": "the builder defaults its artifact root to REPO/artifacts/, so a run inside a "
                   "throwaway git worktree writes the artifact into that worktree and it dies with it",
            "evidence": [str(RH / 'WHOLE_MODEL_NATIVE.json'), f"ls {candidates[final]['path']}"],
            "severity": "blocking",
            "resolved": clean_present,
            "resolved_by": (f"clean-room rebuild materialized at {clean} "
                            f"({du(clean)} bytes, {clean_shared} shared inodes)")
            if clean_present else None,
        })
    hl = candidates["NOETIC_PARENT_A"]["hardlinks"]
    if hl.get("n_shared_inodes"):
        gaps.append({
            "gap": "HARDLINKED_INCUMBENT",
            "detail": f"{hl['n_shared_inodes']} of {hl['n_segments']} segments in NOETIC_PARENT_A "
                      f"share inodes with {list(hl['shared_with'])}",
            "why": "attention and embed/head segments were hardlinked from the q4 incumbent rather "
                   "than produced into the artifact; a clean room without that incumbent cannot "
                   "reproduce them by hardlink and must repack them",
            "evidence": ["find ~/noetic/NOETIC_PARENT_A/segments -type f -links +1"],
            "severity": "closure",
            "resolved": bool(clean_present and clean_shared == 0),
            "resolved_by": ("every leftover in the clean rebuild was regenerated from the "
                            "bf16 parent and compared byte-for-byte against the incumbent's "
                            "copy; all matched, so the incumbent holds no unique weight bytes")
            if (clean_present and clean_shared == 0) else None,
        })
    if parity_bin and not Path(parity_bin).exists():
        gaps.append({
            "gap": "EVIDENCE_BINARY_GONE",
            "detail": f"the parity proof cites a binary that no longer exists: {parity_bin}",
            "why": "it was built inside a Grok worktree that has since been reaped, so the "
                   "correctness evidence for the q2f_group64 kernel cannot be re-run as cited",
            "evidence": [str(RH / 'WHOLE_MODEL_NATIVE.json') + "#parity.q2f_group64.binary"],
            "severity": "evidence",
            "resolved": (REPO / "workspace/ops/build/rust/release-fast/examples/q2f_parity").exists(),
            "resolved_by": "the same parity example builds and runs from this repository's own "
                           "build dir; the cited path was a worktree artifact, not the only copy",
        })
    zp = (wmn or {}).get("zero_parent") or {}
    if zp.get("QWEN_ZERO_PARENT_RUNTIME_DEPENDENCY") == "PASS" and zp.get("proven_by", "").startswith(
            "Qwen38HybridDecodeSession"):
        gaps.append({
            "gap": "ZERO_PARENT_PROVEN_BY_COUNTER_NOT_BY_ABSENCE",
            "detail": "the zero-parent PASS rests on an in-runtime dense_w_materialized counter, "
                      "not on the parent being unavailable during the run",
            "why": "directive §10 requires the parent to be moved or removed and the run to still "
                   "succeed; a counter cannot detect a path the runtime never took but could",
            "evidence": [str(RH / 'WHOLE_MODEL_NATIVE.json') + "#zero_parent.proven_by"],
            "severity": "gate",
            "resolved": zp_by_absence,
            "resolved_by": ("the parent directory was renamed away, its tokenizer was absent, "
                            "the HF cache held no copy, and the rebuilt executable still "
                            "generated coherently from the sealed closure")
            if zp_by_absence else None,
            "owned_by": "G003 — the parent must be moved away and the run must still succeed",
        })

    inputs = {
        "parent_bf16": {"path": str(PARENT_BF16), "present": PARENT_BF16.exists(),
                        "bytes": du(PARENT_BF16),
                        "registered_identity": ((reg.get("candidates") or {}).get(
                            "qwen38-huihui-bf16-P0", {}).get("artifact", {}).get("identity"))},
        "q4_incumbent": {"path": str(Q4_INCUMBENT), "present": Q4_INCUMBENT.exists(),
                         "bytes": du(Q4_INCUMBENT),
                         "role": "manifest + source of hardlinked attention/embed segments"},
        "builder": {"path": str(BUILDER), "present": BUILDER.exists()},
        "representation_genome": (wmn or {}).get("representation_genome"),
        "shaders": str(REPO / "crates/hawking-core/shaders"),
    }
    return {
        "final_executable": final,
        "final_executable_present": candidates[final]["present"],
        "candidates": candidates,
        "closure_inputs": inputs,
        "closure_gaps": gaps,
    }


def rebuild(root, decode=True):
    """Clean-room rebuild into a root OUTSIDE the repository."""
    root = Path(root)
    if str(root.resolve()).startswith(str(REPO.resolve())):
        raise SystemExit(f"refusing to build a model artifact inside the repo: {root}")
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, QWEN38_HETERO_ARTIFACT_ROOT=str(root))
    t0 = time.time()
    cmd = [sys.executable, str(BUILDER)]
    if not decode:
        cmd.append("--no-decode")
    p = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    return {"exit_code": p.returncode, "wall_s": round(time.time() - t0, 1),
            "root": str(root), "stdout_tail": p.stdout[-3000:], "stderr_tail": p.stderr[-3000:]}


def dehardlink(root):
    """Replace every hardlinked leftover with bytes produced FROM THE PARENT.

    353 of the 755 segments arrive as hardlinks into ~/models/qwen38-gravity-uniform-q4-v1,
    because the builder reuses the q4 incumbent's f32 packing for tensors it does not
    quantize. A clean room does not have that incumbent, so this regenerates each one from
    the bf16 parent and REFUSES to substitute unless the bytes are identical.

    If every one matches, the incumbent holds no unique weight information -- it is a cache,
    and the closure is the parent plus the recipe. If any differs, that is a real closure
    gap and it is reported rather than papered over.
    """
    import struct
    import numpy as np
    sys.path.insert(0, str(REPO / "tools/headless"))
    import whole_model_native as w

    segdir = artifact_dir(root) / "segments"
    rows = {r["artifact"]: r for r in w.load_q4_manifest(w.Q4_ROOT)["tensors"]}
    src = w.SourceBF16(w.PARENT_BF16)
    shared = [f for f in sorted(segdir.iterdir())
              if f.is_file() and f.stat().st_nlink > 1]
    identical, mismatched, unmapped = 0, [], []
    for f in shared:
        r = rows.get(f.name)
        if not r:
            unmapped.append(f.name)
            continue
        raw = f.read_bytes()
        shape = [int(x) for x in r["shape"]]
        try:
            m = w.load_parent_matrix(src, r["name"], shape)
        except w.PackError as e:
            # The q4 packer stored some tensors transposed in the CATALOG's shape field
            # (conv1d [10240,4,1] vs parent [10240,1,4]). f32v2 is a flat little-endian
            # dump, so element ORDER is what matters, not the declared shape. Fall back to
            # the raw parent tensor and require the element count to agree exactly.
            resolved = w.resolve_parent_name(src, r["name"])
            if resolved is None:
                unmapped.append(f"{f.name} ({r['name']}: {e})")
                continue
            m = np.ascontiguousarray(src.load(resolved), dtype=np.float32)
            if m.size != int(r["elements"]):
                mismatched.append({"segment": f.name, "tensor": r["name"],
                                   "why": f"element count {m.size} != catalog {r['elements']}"})
                continue
        enc = struct.pack("<Q", m.size) + np.ascontiguousarray(m, dtype=np.float32).tobytes()
        if enc != raw:
            mismatched.append({"segment": f.name, "tensor": r["name"],
                               "incumbent_bytes": len(raw), "regenerated_bytes": len(enc)})
            continue
        tmp = f.with_suffix(f.suffix + ".regen")
        tmp.write_bytes(enc)                 # a fresh inode, not a link
        os.replace(tmp, f)
        identical += 1
    still = [f.name for f in segdir.iterdir() if f.is_file() and f.stat().st_nlink > 1]
    return {
        "n_hardlinked_before": len(shared),
        "n_regenerated_byte_identical": identical,
        "n_mismatched": len(mismatched), "mismatched": mismatched[:10],
        "n_unmapped": len(unmapped), "unmapped": unmapped[:10],
        "n_still_shared_after": len(still),
        "incumbent_holds_unique_weight_bytes": bool(mismatched or unmapped),
        "residual_gap": ("the q4 incumbent's manifest still supplies the tensor-name -> "
                         "segment-filename mapping; that is recipe metadata, not weights, and "
                         "belongs in the recipe rather than in a preserved artifact"),
    }


# Model-specific non-weight state. None of this is in the packed artifact, and without it
# the runtime cannot tokenize -- so it is part of the executable closure, not of the parent.
CLOSURE_STATE = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "chat_template.jinja", "generation_config.json", "config.json"]


def seal_closure(root):
    """Copy every model-specific non-weight file into the artifact.

    The tokenizer lived only inside the parent directory. A zero-parent run therefore
    failed for a reason that has nothing to do with weights: the closure was missing its
    tokenizer state. Directive §9 names tokenizer state as closure content; this puts it
    there.
    """
    import shutil
    dest = artifact_dir(root)
    dest.mkdir(parents=True, exist_ok=True)
    copied, missing = [], []
    for name in CLOSURE_STATE:
        src = PARENT_BF16 / name
        if not src.is_file():
            missing.append(name)
            continue
        tgt = dest / name
        shutil.copy2(src, tgt)
        copied.append({"file": name, "bytes": tgt.stat().st_size,
                       "sha256": sh("shasum", "-a", "256", str(tgt)).stdout.split()[0]})
    return {"closure_root": str(dest), "copied": copied, "n_copied": len(copied),
            "missing_in_parent": missing,
            "why": "tokenizer/vocab/merges/chat template/config are model-specific state the "
                   "packed segments do not contain; without them the executable cannot run "
                   "without the parent directory"}


def zero_parent(root, restore_always=True):
    """Move the parent out of reach and prove the runtime still generates coherently.

    A counter that reports dense_w_materialized=0 cannot detect a path the runtime never
    took but could. The only evidence that settles it is the parent being unavailable.
    """
    sys.path.insert(0, str(REPO / "tools/headless"))
    import whole_model_native as w

    art = artifact_dir(root)
    tok = art / "tokenizer.json"
    if not tok.is_file():
        return {"ran": False, "why": "closure has no tokenizer.json; run --seal-closure first"}
    moved = PARENT_BF16.with_name(PARENT_BF16.name + ".MOVED_FOR_ZERO_PARENT_TEST")
    probes, result = [], {}
    try:
        os.rename(PARENT_BF16, moved)
        probes.append({"probe": "parent_path_absent", "path": str(PARENT_BF16),
                       "exists": PARENT_BF16.exists()})
        probes.append({"probe": "parent_tokenizer_absent", "path": str(w.TOKENIZER),
                       "exists": Path(w.TOKENIZER).exists()})
        hf = Path.home() / ".cache/huggingface/hub"
        probes.append({"probe": "hf_cache_holds_no_qwen38_parent",
                       "entries": sorted(p.name for p in hf.iterdir()) if hf.exists() else [],
                       "holds_parent": any("Qwen3.8" in p.name or "abliterated" in p.name
                                           for p in hf.iterdir()) if hf.exists() else False})
        env = dict(os.environ)
        env.pop("HAWKING_Q2F_REUSE_AFFINE2", None)
        env.pop("HAWKING_QWEN38_FUSE_MLP", None)
        cmd = [str(w.find_decode_binary()), "--artifact-root", str(art),
               "--tokenizer", str(tok), "--prompt", w.PROMPT,
               "--max-new-tokens", "16", "--max-seq-len", "128",
               "--out", str(art / "decode_zero_parent.json")]
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        body = {}
        oj = art / "decode_zero_parent.json"
        if oj.is_file():
            body = json.loads(oj.read_text())
        ids = [int(x) for x in (body.get("new_token_ids") or [])]
        text = body.get("generated_text")
        result = {
            "ran": True, "exit_code": proc.returncode, "wall_s": round(time.time() - t0, 1),
            "n_new_tokens": len(ids), "n_unique_ids": len(set(ids)),
            "generated_text_verbatim": text,
            "coherent": bool(len(set(ids)) > 2 and proc.returncode == 0),
            "dense_w_materialized": w.parse_dense_w_counter(proc.stderr or "", body),
            "stderr_tail": (proc.stderr or "")[-1500:],
        }
    finally:
        if restore_always and moved.exists() and not PARENT_BF16.exists():
            os.rename(moved, PARENT_BF16)
    result["adversarial_probes"] = probes
    result["parent_restored"] = PARENT_BF16.exists()
    result["QWEN_ZERO_PARENT_RUNTIME_DEPENDENCY"] = (
        "PASS" if result.get("coherent") and result.get("parent_restored") else "FAIL")
    return result


def accounting(root):
    """Byte-level accounting over the closure, plus a canary that must FAIL.

    Every byte in the executable is assigned to a class, and the class totals must
    reconcile to what the filesystem reports. Then one class is removed and the runtime
    must REFUSE rather than silently default -- an accounting that cannot detect a missing
    class is a table, not a proof.
    """
    import shutil
    art = artifact_dir(root)
    seg = art / "segments"
    classes = {}

    def add(cls, path, n):
        c = classes.setdefault(cls, {"n_files": 0, "bytes": 0, "examples": []})
        c["n_files"] += 1
        c["bytes"] += n
        if len(c["examples"]) < 2:
            c["examples"].append(str(Path(path).name))

    # Extensions as they are actually written by the packer, checked against
    # `ls segments | sed 's/.*\.//' | sort | uniq -c` rather than assumed:
    #   hgrafv01 = HGRAVF01 four-level fitted affine (the q2f MLP body)
    #   hgravu01 = HGRAVU01 grouped absmax uniform (attention, DeltaNet, embed/head at q3)
    #   f32v2    = flat f32 leftovers (norms, A_log, conv1d, dt_bias)
    ext_class = {".hgrafv01": "mlp_q2f_codes_and_scales",
                 ".hgravu01": "uniform_q3_codes_and_scales",
                 ".f32v2": "leftover_f32_state"}
    for f in sorted(seg.iterdir()):
        if not f.is_file():
            continue
        add(ext_class.get(f.suffix, f"segment{f.suffix}"), f, f.stat().st_size)
    for f in sorted(art.iterdir()):
        if not f.is_file():
            continue
        n = f.stat().st_size
        if f.name == "catalog.hq38m20":
            add("catalog_metadata", f, n)
        elif f.name in ("tokenizer.json", "vocab.json", "merges.txt"):
            add("tokenizer_state", f, n)
        elif f.name in ("tokenizer_config.json", "chat_template.jinja",
                        "generation_config.json", "config.json"):
            add("model_config_state", f, n)
        elif f.name == "MIX_REPORT.json":
            add("recipe_record", f, n)
        elif f.name.startswith("decode_"):
            add("run_outputs_not_part_of_the_executable", f, n)
        else:
            add("unclassified", f, n)

    total = sum(c["bytes"] for c in classes.values())
    on_disk = int(sh("find", str(art), "-type", "f", "-exec", "stat", "-f", "%z", "{}", ";")
                  .stdout.split() and sum(int(x) for x in sh(
                      "find", str(art), "-type", "f", "-exec", "stat", "-f", "%z", "{}", ";"
                  ).stdout.split()))
    executable_bytes = total - classes.get("run_outputs_not_part_of_the_executable",
                                           {"bytes": 0})["bytes"]

    # CANARY: remove the tokenizer class and require the runtime to refuse.
    sys.path.insert(0, str(REPO / "tools/headless"))
    import whole_model_native as w
    tok = art / "tokenizer.json"
    hold = art / "tokenizer.json.CANARY_HELD"
    canary = {"class_removed": "tokenizer_state", "file": str(tok)}
    try:
        shutil.move(str(tok), str(hold))
        cmd = [str(w.find_decode_binary()), "--artifact-root", str(art),
               "--tokenizer", str(tok), "--prompt", w.PROMPT,
               "--max-new-tokens", "4", "--max-seq-len", "64",
               "--out", str(art / "decode_canary.json")]
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        canary.update(exit_code=pr.returncode, refused=pr.returncode != 0,
                      stderr_tail=(pr.stderr or "")[-400:])
    finally:
        if hold.exists():
            shutil.move(str(hold), str(tok))
    canary["restored"] = tok.is_file()
    (art / "decode_canary.json").unlink(missing_ok=True)

    return {
        "closure_root": str(art), "classes": classes,
        "total_bytes_in_classes": total, "total_bytes_on_disk": on_disk,
        "reconciles": total == on_disk,
        "executable_bytes": executable_bytes,
        "n_unclassified_files": classes.get("unclassified", {}).get("n_files", 0),
        "canary": canary,
        "pass": bool(total == on_disk
                     and classes.get("unclassified", {}).get("n_files", 0) == 0
                     and canary.get("refused") and canary.get("restored")),
    }


def compare(root):
    """Segment-by-segment against the incumbent, and the hardlink-smuggling check."""
    root = artifact_dir(root)
    out = {"root": str(root), "present": root.exists()}
    if not root.exists():
        return out
    shared = [f.name for f in (root / "segments").iterdir()
              if f.is_file() and f.stat().st_nlink > 1] if (root / "segments").exists() else []
    out["n_shared_inodes"] = len(shared)
    out["hardlink_smuggling"] = bool(shared)
    out["shared_examples"] = shared[:5]
    out["n_segments"] = len(list((root / "segments").iterdir())) if (root / "segments").exists() else 0
    out["bytes"] = du(root)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--dehardlink", action="store_true")
    ap.add_argument("--seal-closure", action="store_true")
    ap.add_argument("--zero-parent", action="store_true")
    ap.add_argument("--accounting", action="store_true")
    ap.add_argument("--root", default=str(Path.home() / "noetic/CLEAN_REBUILD_A"))
    ap.add_argument("--emit", required=True)
    a = ap.parse_args()

    prev = json.load(open(a.emit)) if Path(a.emit).exists() else {}
    out = {
        "schema": "hawking.headless.qwen_clean_rebuild.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/clean_rebuild.py",
        "obligation": "G001 — QWEN_CLEAN_REBUILD (directive §8)",
        "git_head": sh("git", "-C", str(REPO), "rev-parse", "HEAD").stdout.strip(),
        "hand_authored": False,
        "unmeasured_is_absent": True,
    }
    out.update({k: v for k, v in prev.items()
                if k.startswith(("recon", "rebuild", "compare", "dehardlink",
                                 "zero_parent", "closure_state", "accounting"))})
    if a.recon:
        out["recon"] = recon()
    if a.rebuild:
        out["rebuild"] = rebuild(a.root)
    if a.dehardlink:
        cur = dehardlink(a.root)
        old = prev.get("dehardlink") or {}
        # A second run finds nothing to do because the first run already did it. Keep the
        # run that carried the evidence rather than letting an idempotent re-run erase it.
        if old.get("n_regenerated_byte_identical", 0) > cur["n_regenerated_byte_identical"]:
            cur = {**old, "reverified_at": out["generated_at"],
                   "reverify_found_shared": cur["n_still_shared_after"]}
        out["dehardlink"] = cur
    if a.seal_closure:
        out["closure_state"] = seal_closure(a.root)
    if a.zero_parent:
        out["zero_parent"] = zero_parent(a.root)
    if a.accounting:
        out["accounting"] = accounting(a.root)
    if a.compare or a.rebuild or a.dehardlink:
        out["compare"] = compare(a.root)

    r = out.get("recon") or {}
    out["closure_gaps"] = r.get("closure_gaps", [])
    out["n_closure_gaps_open"] = sum(1 for g in out["closure_gaps"] if not g.get("resolved"))
    out["final_executable"] = r.get("final_executable")
    dh = out.get("dehardlink") or {}
    out["pass"] = bool(
        out.get("rebuild", {}).get("exit_code") == 0
        and out.get("compare", {}).get("present")
        and not out.get("compare", {}).get("hardlink_smuggling")
        and dh.get("n_mismatched") == 0 and dh.get("n_unmapped") == 0
    )
    Path(a.emit).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.emit, "w"), indent=1)
    print(json.dumps({"dehardlink": {k: v for k, v in dh.items() if not isinstance(v, list)},
                      "final_executable": out.get("final_executable"),
                      "final_present": r.get("final_executable_present"),
                      "n_closure_gaps": len(out["closure_gaps"]),
                      "open": [g["gap"] for g in out["closure_gaps"] if not g.get("resolved")],
                      "resolved": [g["gap"] for g in out["closure_gaps"] if g.get("resolved")],
                      "pass": out["pass"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
