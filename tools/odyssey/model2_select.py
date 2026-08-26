#!/usr/bin/env python3
"""Select Odyssey specimen #2 from measured disk truth and verified upstream identity.

The recovered queue's `on_disk` flags are stale (O003/O006 were deleted from the HF
cache after it was written), so presence is re-measured by stat here and never read
from the flag. Repository ids and revisions are only ever copied from the queue and
confirmed upstream -- never constructed, never guessed.
"""
import argparse, json, os, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QUEUE = REPO / "receipts/headless/ODYSSEY_QUEUE_RECOVERED.json"
ORGANS = REPO / "receipts/headless/ORGAN_LIBRARY.json"
CANON = REPO / "receipts/headless/ARCHITECTURE_CANON.json"
HF_HUB = Path.home() / ".cache/huggingface/hub"
SEARCH_ROOTS = [HF_HUB, Path.home() / "models", Path("/Volumes/corpdrive")]
API = "https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"

# Organs the Qwen textbook already covers. Read from the canonical library when it
# exposes a name list; the literal fallback is recorded in the receipt when used.
QWEN_ORGAN_FALLBACK = [
    "embedding", "rmsnorm", "mlp_gate_up", "mlp_down", "gqa_attention",
    "output_head", "kv_state",
]
# Config keys that betray an organ Qwen does not have.
NOVEL_MARKERS = {
    "num_experts": "moe_experts", "n_routed_experts": "moe_experts",
    "num_local_experts": "moe_experts", "moe_intermediate_size": "moe_experts",
    "shared_expert_intermediate_size": "shared_expert",
    "num_experts_per_tok": "moe_router",
    "vision_config": "vision_encoder", "mm_projector_type": "mm_projector",
    "linear_attn_config": "recurrent_state", "mamba_d_state": "recurrent_state",
    "ssm_cfg": "recurrent_state", "linear_num_key_heads": "recurrent_state",
    "kv_lora_rank": "latent_attention", "q_lora_rank": "latent_attention",
}


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "hawking-odyssey/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.load(r)


def upstream(repo, rev):
    """Confirm the exact repo+revision resolves. Never substitutes a near match."""
    out = {"repo": repo, "revision": rev, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        status, d = _get(API.format(repo=repo, rev=rev))
        sibs = d.get("siblings") or []
        out.update(
            http_status=status, resolved_sha=d.get("sha"), gated=bool(d.get("gated")),
            n_files=len(sibs),
            download_bytes=sum(s.get("size") or 0 for s in sibs),
            n_files_without_size=sum(1 for s in sibs if s.get("size") is None),
            config=d.get("config") or {},
        )
        out["revision_matches"] = (out["resolved_sha"] == rev)
    except urllib.error.HTTPError as e:
        out.update(http_status=e.code, error=e.reason, resolved_sha=None, gated=(e.code in (401, 403)))
    except Exception as e:                                   # network down, DNS, timeout
        out.update(http_status=None, error=f"{type(e).__name__}: {e}", resolved_sha=None)
    return out


def config_json(repo, rev):
    """config.json only -- small, and the only thing organ scoring needs."""
    url = f"https://huggingface.co/{repo}/resolve/{rev}/config.json"
    try:
        _, d = _get(url, timeout=25)
        return d
    except Exception:
        return None


def _hf_cache_dirname(repo):
    return "models--" + repo.replace("/", "--")


def measure_disk(repo, model_name):
    """Stat the candidate locations. Presence is bytes on disk, never a flag."""
    hits = []
    cand = HF_HUB / _hf_cache_dirname(repo)
    if cand.exists():
        hits.append(cand)
    for root in SEARCH_ROOTS[1:]:
        if not root.exists():
            continue
        try:
            for child in root.iterdir():
                n = child.name.lower()
                if model_name.lower() in n or repo.split("/")[-1].lower() in n:
                    hits.append(child)
        except PermissionError:
            pass
    total = 0
    for h in hits:
        try:
            total += int(subprocess.run(["du", "-sk", str(h)], capture_output=True, text=True,
                                        timeout=120).stdout.split()[0]) * 1024
        except Exception:
            pass
    return {"paths": [str(h) for h in hits], "bytes_present": total, "present": total > 0}


def free_bytes(path):
    try:
        s = os.statvfs(path)
        return s.f_bavail * s.f_frsize
    except Exception:
        return None


def known_organs():
    for p, key in ((ORGANS, "organs"), (CANON, "organs")):
        if not p.exists():
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        v = d.get(key)
        if isinstance(v, dict) and v:
            return sorted(v), str(p)
        if isinstance(v, list) and v:
            names = [o.get("name") or o.get("organ") for o in v if isinstance(o, dict)]
            names = [n for n in names if n]
            if names:
                return sorted(set(names)), str(p)
    return sorted(QWEN_ORGAN_FALLBACK), "FALLBACK_LITERAL"


def novelty(cfg):
    """Organ families present that the Qwen textbook does not cover."""
    if not cfg:
        return [], None
    found = set()

    def walk(d, depth=0):
        if depth > 3 or not isinstance(d, dict):
            return
        for k, v in d.items():
            if k in NOVEL_MARKERS:
                found.add(NOVEL_MARKERS[k])
            if isinstance(v, dict):
                walk(v, depth + 1)
    walk(cfg)
    return sorted(found), cfg.get("model_type")


# Independent novelty AXES. Two novel organs on the same axis (moe_experts and
# moe_router always arrive together) are one new thing to attribute, not two.
# Attribution matters because directive §89 requires each compounding demonstration
# to be attributable; a specimen novel on four axes at once confounds the measurement.
NOVELTY_AXES = {
    "moe_experts": "routing", "moe_router": "routing", "shared_expert": "routing",
    "recurrent_state": "state",
    "vision_encoder": "modality", "mm_projector": "modality",
    "latent_attention": "attention",
}
# Directive §91 requires classifying every Qwen law as QWEN-SPECIFIC / FAMILY-TRANSFERRED
# / ARCHITECTURE-GENERAL / MACHINE-GENERAL. FAMILY-TRANSFERRED cannot be separated from
# ARCHITECTURE-GENERAL without a same-family specimen, so a same-family candidate carries
# mission value a stranger cannot supply at any price. This is the one weighting in the
# scorer that comes from the directive rather than from the artifact.
FAMILY_TRANSFER_BONUS = 1.0
PARENT_FAMILY = "qwen"


def score(entry, up, cfg, ssd_free, hdd_free):
    novel, mtype = novelty(cfg)
    dl = up.get("download_bytes") or 0
    acquirable = bool(
        up.get("resolved_sha") and not up.get("gated")
        and up.get("http_status") == 200 and cfg
        and dl and hdd_free and dl < hdd_free * 0.9
    )
    overlap = 0.0
    if cfg:
        overlap = 1.0
        if "vision_encoder" in novel:
            overlap -= 0.25
        if "recurrent_state" in novel:
            overlap -= 0.25
        if "latent_attention" in novel:
            overlap -= 0.15
        overlap = max(overlap, 0.0)

    axes = sorted({NOVELTY_AXES[n] for n in novel if n in NOVELTY_AXES})
    attribution = 1.0 / len(axes) if axes else 0.0
    family = bool(
        entry["canonical_source"].split("/")[0].lower() == PARENT_FAMILY
        or (mtype or "").lower().startswith(PARENT_FAMILY)
    )
    family_mult = 1.0 + (FAMILY_TRANSFER_BONUS if family else 0.0)

    cost_gib = dl / 2**30 if dl else None
    mission_value = overlap * len(novel) * attribution * family_mult
    ratio = (mission_value / cost_gib) if (cost_gib and mission_value) else 0.0
    return {
        "model_type": mtype, "novel_organs": novel, "n_novel": len(novel),
        "novelty_axes": axes, "attribution": round(attribution, 4),
        "same_family_as_parent": family, "family_multiplier": family_mult,
        "transfer_overlap": round(overlap, 3),
        "download_gib": round(cost_gib, 2) if cost_gib else None,
        "acquirable": acquirable,
        "mission_value": round(mission_value, 4),
        "information_gain_per_gib": round(ratio, 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", required=True)
    ap.add_argument("--offline", action="store_true", help="skip network; disk truth only")
    a = ap.parse_args()

    q = json.load(open(QUEUE))
    organs, organ_src = known_organs()
    ssd_free, hdd_free = free_bytes("/"), free_bytes("/Volumes/corpdrive")

    rows, stale = [], []
    for e in q["queue"]:
        repo, rev = e["canonical_source"], e["canonical_revision"]
        disk = measure_disk(repo, e["model"])
        if bool(e.get("on_disk")) != disk["present"]:
            stale.append({"oxx": e["oxx"], "recovered_on_disk": bool(e.get("on_disk")),
                          "measured_present": disk["present"], "bytes_present": disk["bytes_present"]})
        up = {"skipped": "offline"} if a.offline else upstream(repo, rev)
        cfg = None if a.offline or not up.get("resolved_sha") else config_json(repo, rev)
        rows.append({**{k: e[k] for k in ("oxx", "model", "class", "canonical_source",
                                          "canonical_revision", "identity_status", "state")},
                     "recovered_on_disk": bool(e.get("on_disk")), "disk": disk, "upstream": up,
                     "score": score(e, up, cfg, ssd_free, hdd_free) if not a.offline else None})

    ranked = sorted([r for r in rows if r["score"] and r["score"]["acquirable"]],
                    key=lambda r: -r["score"]["information_gain_per_gib"])
    rec, runners = (ranked[0] if ranked else None), ranked[1:4]

    out = {
        "schema": "hawking.headless.model2_selection.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/model2_select.py",
        "obligation": "G024 — MODEL_2_SELECTION (directive §87, §44, §31, §46)",
        "hand_authored": False,
        "did_not_invent_hf_repo_ids": True,
        "did_not_download_weights": True,
        "unmeasured_is_absent": True,
        "scoring_criteria": {
            "transfer_overlap": "directive §87 — how much of the Qwen dense-backbone science still applies",
            "novelty_axes": "directive §89 — independent new things to learn; attribution = 1/len(axes)",
            "family_multiplier": "directive §91 — only a same-family specimen separates FAMILY-TRANSFERRED from ARCHITECTURE-GENERAL",
            "cost": "directive §46 — information gain per resource cost, cost measured as upstream download bytes",
            "formula": "mission_value = overlap * n_novel * attribution * family_multiplier;  rank = mission_value / download_gib",
        },
        "sources_read": [str(QUEUE), organ_src],
        "known_organs_from": organ_src,
        "known_organs": organs,
        "free_bytes": {"ssd": ssd_free, "hdd_corpdrive": hdd_free},
        "stale_flags": stale,
        "candidates": rows,
        "recommendation": None if not rec else {
            "oxx": rec["oxx"], "model": rec["model"],
            "canonical_source": rec["canonical_source"],
            "canonical_revision": rec["canonical_revision"],
            "resolved_sha": rec["upstream"].get("resolved_sha"),
            "download_gib": rec["score"]["download_gib"],
            "novel_organs": rec["score"]["novel_organs"],
            "transfer_overlap": rec["score"]["transfer_overlap"],
            "why": "highest mission value per download GiB among acquirable candidates",
            "mission_value": rec["score"]["mission_value"],
            "novelty_axes": rec["score"]["novelty_axes"],
            "same_family_as_parent": rec["score"]["same_family_as_parent"],
        },
        "runners_up": [{"oxx": r["oxx"], "model": r["model"],
                        "information_gain_per_gib": r["score"]["information_gain_per_gib"],
                        "lost_because": "lower information gain per GiB than the recommendation"}
                       for r in runners],
        "not_acquirable": [{"oxx": r["oxx"], "model": r["model"],
                            "http_status": r["upstream"].get("http_status"),
                            "gated": r["upstream"].get("gated"),
                            "error": r["upstream"].get("error")}
                           for r in rows if r["score"] and not r["score"]["acquirable"]],
        "pass": bool(rec),
    }
    Path(a.emit).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.emit, "w"), indent=1)
    print(f"stale_flags={len(stale)} acquirable={len(ranked)} "
          f"recommendation={out['recommendation'] and out['recommendation']['model']}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
