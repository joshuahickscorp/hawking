#!/usr/bin/env python3
"""G011 — QWEN_RETIREMENT_GATE (§17).

Qwen may become RETIREMENT_READY only after the transfer substrate has succeeded, and
only once everything §17 names is provably preserved. This gate is written to FAIL: each
category is checked against disk, and a missing one blocks retirement rather than being
noted in passing.

RELEGATION IS NOT DELETION. Bulky parents are classified and the reclaimable figure is
reported; nothing is removed here. Deleting model weights is the operator's call, and a
relegation candidate stays a candidate until its recipe metadata is preserved somewhere
that survives it -- which is why the q4 manifest was copied into research/recipes/ first.
"""
import hashlib, json, os, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
NOETIC = Path("/Users/scammermike/noetic")
MODELS = Path("/Users/scammermike/models")
PARENT_BF16 = MODELS / "qwen3.8-27b-abliterated-bf16"
CLOSURE = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
           "chat_template.jinja", "generation_config.json", "config.json"]


def sha(p, cap=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(cap)):
            h.update(b)
    return h.hexdigest()


def du(p):
    r = subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True)
    try:
        return int(r.stdout.split()[0]) * 1024
    except Exception:
        return 0


def check_executable(name, root):
    root = Path(root)
    if not root.is_dir():
        return {"name": name, "present": False, "why": "artifact root missing"}
    mixp = root / "MIX_REPORT.json"
    payload = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()
                  and f.suffix in (".hgrafv01", ".hgravu01", ".f32v2", ".hq30uq4"))
    closure = [c for c in CLOSURE if (root / c).is_file()]
    hard = sum(1 for f in root.rglob("*") if f.is_file() and f.stat().st_nlink > 1)
    mix = json.load(open(mixp)) if mixp.is_file() else {}
    return {
        "name": name, "present": True, "root": str(root),
        "payload_bytes": payload,
        "complete_ebpw_physical": (round(8.0 * payload / mix["parent_params"], 6)
                                   if mix.get("parent_params") else None),
        "mix_report_sha256": sha(mixp) if mixp.is_file() else None,
        "closure_files": len(closure), "closure_required": len(CLOSURE),
        "closure_complete": len(closure) == len(CLOSURE),
        "hardlinked_files": hard,
        "self_contained": len(closure) == len(CLOSURE) and hard == 0,
        "genome": mix.get("genome"),
        "legacy_genome": (sorted(mix["codecs"]) if isinstance(mix.get("codecs"), dict)
                          else mix.get("codecs")) if not mix.get("genome") else None,
    }


def main():
    # ---- prerequisite: the transfer substrate must have succeeded ---------------
    substrate = {}
    for f in ("ODYSSEY_TRANSFER_PROVEN", "CROSS_MODEL_LAWS", "MODEL_2_COMPOUNDING",
              "ODYSSEY_LEARNING_CURVE"):
        p = RH / f"{f}.json"
        substrate[f] = {"present": p.is_file(),
                        "pass": json.load(open(p)).get("pass") if p.is_file() else None}
    laws = json.load(open(RH / "CROSS_MODEL_LAWS.json"))["laws"]
    levels = {}
    for l in laws:
        levels.setdefault(l["level"], []).append(l["id"])
    substrate_ok = (all(v["pass"] for v in substrate.values())
                    and bool(levels.get("ARCHITECTURE_GENERAL")))

    # ---- §17 preservation ------------------------------------------------------
    execs = [
        check_executable("resident (sealed-3.14)", NOETIC / "NOETIC_PARENT_A"),
        check_executable("grand candidate (variantB-2.76)",
                         NOETIC / "VARIANT_B_MLP_BIAS_Q3"),
    ]
    idx = PARENT_BF16 / "model.safetensors.index.json"
    cfg = PARENT_BF16 / "config.json"
    kern = json.load(open(RH / "KERNEL_LIBRARY.json"))
    neg = json.load(open(RH / "NOETIC_NEGATIVE_SCIENCE.json"))
    q4man = REPO / "research/recipes/qwen38-gravity-uniform-q4-v1/manifest.json"

    preserved = {
        "final_executable": {
            "ok": any(e.get("self_contained") for e in execs),
            "detail": execs,
            "why": "at least one executable must be self-contained: complete closure and "
                   "no hardlinks into another artifact",
        },
        "exact_source_identity": {
            "ok": idx.is_file() and cfg.is_file(),
            "detail": {"parent": str(PARENT_BF16),
                       "config_sha256": sha(cfg) if cfg.is_file() else None,
                       "tensor_index_sha256": sha(idx) if idx.is_file() else None,
                       "architectures": (json.load(open(cfg)).get("architectures")
                                         if cfg.is_file() else None)},
            "why": "identity is the config and tensor-index hashes, which pin WHICH "
                   "model this was without preserving its 52 GB",
        },
        "genomes": {
            # the sealed body predates the `genome` key and records the same information
            # under a legacy schema (`codecs`, `affine_*`). Requiring the new key alone
            # reported a preserved genome as missing.
            "ok": all(e.get("genome") or e.get("legacy_genome")
                      for e in execs if e["present"]),
            "detail": {e["name"]: {"schema": ("genome" if e.get("genome")
                                              else "legacy" if e.get("legacy_genome")
                                              else None),
                                   "organs": (sorted(e["genome"]) if e.get("genome")
                                              else e.get("legacy_genome"))}
                       for e in execs},
            "note": "two schemas are in play: the current per-organ `genome` block, and "
                    "the older `codecs`/`affine_*` fields on the sealed body. Both are "
                    "accepted; the schema each uses is recorded.",
        },
        "kernels": {
            "ok": kern.get("n_kernels", 0) > 0,
            "detail": {"n_kernels": kern["n_kernels"],
                       "receipt": "receipts/headless/KERNEL_LIBRARY.json",
                       "n_without_runnable_contract":
                           kern.get("n_kernels_without_a_runnable_contract")},
        },
        "recipes": {
            "ok": q4man.is_file(),
            "detail": {
                "builders": ["tools/headless/whole_model_native.py",
                             "tools/odyssey/composition_isolation.py",
                             "tools/odyssey/clean_rebuild.py"],
                "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                           capture_output=True, text=True).stdout.strip(),
                "q4_manifest_preserved": str(q4man.relative_to(REPO)),
                "q4_manifest_sha256": sha(q4man) if q4man.is_file() else None,
            },
            "why": "the q4 manifest is the tensor-name to segment-filename mapping and "
                   "lived ONLY inside the 13 GB incumbent. Preserving it is the "
                   "precondition for relegating those bytes.",
        },
        "receipts": {
            "ok": len(list(RH.glob("*.json"))) > 100,
            "detail": {"n_receipts": len(list(RH.glob("*.json"))),
                       "dir": "receipts/headless/"},
        },
        "negative_science": {
            "ok": len(neg.get("entries", [])) > 0,
            "detail": {"n_entries": len(neg["entries"]),
                       "receipt": "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
                       "campaign_refutations_also_preserved": [
                           "TOKENIZER_GRAVITY.json — vocabulary reduction refuted on "
                           "held-out language",
                           "DECODING_GRAVITY.json — full-size drafts refuted on cost "
                           "ratio",
                           "VARIANT_LOCALIZATION.json — its own axis claim refuted and "
                           "retained",
                           "GRAND_CANDIDATE.json — three landed attacks retained",
                       ]},
        },
    }
    all_preserved = all(v["ok"] for v in preserved.values())

    # ---- relegation: classify, never delete ------------------------------------
    def classify(path, kind, why, reproducible):
        return {"path": str(path), "bytes": du(path), "gib": round(du(path) / 2**30, 2),
                "class": kind, "why": why, "reproducible": reproducible}

    items = [
        classify(PARENT_BF16, "INDISPENSABLE",
                 "the bf16 source. Every clean-room regeneration in this campaign reads "
                 "it, and it is the only thing that can rebuild an artifact from scratch.",
                 "in principle re-downloadable, but it is an abliterated community "
                 "checkpoint and re-acquisition is not guaranteed"),
        classify(MODELS / "qwen38-gravity-uniform-q4-v1", "INDISPENSABLE_FOR_NOW",
                 "NOT relegatable while sealed is the resident. Its 353 f32 passthrough "
                 "segments regenerate byte-identically from the parent, but the resident "
                 "still depends on 210 .hq30uq4 segments that no current tool can "
                 "re-encode -- the de-hardlink path writes f32, so every quantized "
                 "segment came back at exactly 7.53x its packed size. The information is "
                 "derivable by re-running the q4 quantizer; until that exists, these "
                 "bytes are load-bearing. The grand candidate needs none of them.",
                 "in principle yes, by re-running the q4 packer; no such path exists "
                 "today"),
        classify(MODELS / "qwen3.8-27b-abliterated-mlx", "RELEGATE",
                 "an MLX conversion of the same parent, used only as a capability "
                 "baseline. Regenerable by re-converting.",
                 "yes, from the bf16 parent"),
        classify(NOETIC / "NOETIC_PARENT_A", "INDISPENSABLE",
                 "the selected resident and the best-capability body.", "no"),
        classify(NOETIC / "VARIANT_B_MLP_BIAS_Q3", "INDISPENSABLE",
                 "the grand candidate: self-contained, 2.756 EBPW, the only sub-3.0 body "
                 "that works.", "no"),
        classify(NOETIC / "VARIANT_A_MLP_ONLY", "PRESERVE_AS_NEGATIVE",
                 "scores 0/43. It is the evidence that the MLP per-group bias is "
                 "load-bearing, and it is also the best-agreeing draft at 75%.", "no"),
        classify(NOETIC / "CLEAN_REBUILD_A", "PRESERVE_AS_NEGATIVE",
                 "the 2.5970 body, capability-dead. It anchors the composition ladder "
                 "and the floor-effect argument.", "no"),
        classify(NOETIC / "VARIANT_B_BIAS_ONLY", "SUPERSEDED",
                 "a partial earlier variant-B build, superseded by "
                 "VARIANT_B_MLP_BIAS_Q3. Not referenced by any receipt.",
                 "yes, by rebuilding"),
    ]
    relegatable = [i for i in items if i["class"] in ("RELEGATE", "SUPERSEDED")]
    reclaimable = sum(i["bytes"] for i in relegatable)

    out = {
        "schema": "hawking.odyssey.retirement_gate.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/retirement_gate.py",
        "obligation": "G011 — QWEN_RETIREMENT_GATE (§17)",
        "hand_authored": False,
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "prerequisite_transfer_substrate": {
            "receipts": substrate,
            "law_levels": {k: len(v) for k, v in levels.items()},
            "architecture_general_laws": levels.get("ARCHITECTURE_GENERAL", []),
            "ok": substrate_ok,
            "why": "§17 forbids retirement until the transfer substrate SUCCEEDS. A law "
                   "proven on two distinct architecture families is the evidence that "
                   "the science outlives this model.",
        },
        "preservation": preserved,
        "all_preserved": all_preserved,
        "relegation": {
            "policy": "CLASSIFY AND REPORT. Nothing is deleted by this gate. A "
                      "relegation candidate stays a candidate until its recipe metadata "
                      "is preserved somewhere that survives it.",
            "items": items,
            "relegatable_now": [i["path"] for i in relegatable],
            "reclaimable_bytes": reclaimable,
            "reclaimable_gib": round(reclaimable / 2**30, 2),
            "operator_action_required": "deleting model weights is the operator's call; "
                                        "this gate never removes them",
        },
        "shippability": {
            "finding": "the GRAND CANDIDATE is self-contained and the SELECTED RESIDENT "
                       "is not. variantB carries its own closure and zero hardlinks; "
                       "sealed carries 210 .hq30uq4 segments hardlinked into the q4 "
                       "incumbent that no current tool can regenerate.",
            "consequence": "retiring Qwen by relegating the q4 incumbent would break the "
                           "resident while leaving the candidate intact. Either the q4 "
                           "incumbent is preserved, or a q4 re-encode path is built, or "
                           "the resident becomes the body that is already shippable.",
            "not_a_reopening_of_G040": "the selection stands on the composite. This is a "
                                       "packaging property that selection never scored.",
        },
        "RETIREMENT_READY": bool(substrate_ok and all_preserved),
    }
    out["pass"] = out["RETIREMENT_READY"]
    p = RH / "QWEN_RETIREMENT_GATE.json"
    p.write_text(json.dumps(out, indent=1))

    print(f"transfer substrate: {substrate_ok}  "
          f"(ARCHITECTURE_GENERAL laws: {levels.get('ARCHITECTURE_GENERAL')})")
    print()
    for k, v in preserved.items():
        print(f"  {'OK  ' if v['ok'] else 'GAP '} {k}")
    print()
    for e in execs:
        print(f"  executable {e['name']:34s} self_contained={e.get('self_contained')} "
              f"closure={e.get('closure_files')}/7 hardlinks={e.get('hardlinked_files')}")
    print()
    for i in items:
        print(f"  {i['class']:20s} {i['gib']:>7.2f} GiB  {i['path'].split('/')[-1]}")
    print(f"\nreclaimable by relegation: {out['relegation']['reclaimable_gib']} GiB "
          f"(NOT deleted)")
    print(f"RETIREMENT_READY = {out['RETIREMENT_READY']}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
