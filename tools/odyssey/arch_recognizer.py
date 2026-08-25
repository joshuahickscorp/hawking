#!/usr/bin/env python3
"""ARCHITECTURE RECOGNIZER — first stage of the Noetic compiler.

Hand it a foreign model's config and tensor map; it returns which organs Hawking
already knows, which are variants, which are novel, and how confident it is.

The load-bearing property is NOT accuracy on models we already studied. It is
CALIBRATION: a wrong answer must not carry high confidence, and an unfamiliar block
must come back NOVEL rather than force-fit into the nearest known family. A
recognizer that confidently mislabels a linear-attention state block as an MLP is
worse than no recognizer, because the compiler downstream would then prescribe an
MLP treatment for it.

Weights are never loaded. config.json and model.safetensors.index.json are enough,
and both are small.
"""
import argparse, json, re, sys, time, urllib.error, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
ORGAN_LIB = RH / "ORGAN_LIBRARY.json"
FIXTURES = RH / "ARCHITECTURE_RECOGNIZER_FIXTURES.json"
RAW = "https://huggingface.co/{repo}/resolve/{rev}/{f}"

# A fingerprint is a ROLE signature: what the tensor does in the block, expressed as
# name-token patterns plus a shape law. Name matching alone is how `feed_forward.w1`
# and `mlp.gate_proj` get called different organs when they are the same thing.
FINGERPRINTS = [
    ("mlp_gate_up",   r"(mlp|feed_forward|ffn)\.(gate_proj|up_proj|w1|w3|gate_up_proj)\b", "2d"),
    ("mlp_down",      r"(mlp|feed_forward|ffn)\.(down_proj|w2)\b",                          "2d"),
    ("moe_expert",    r"(experts?)\.\d+\.(gate_proj|up_proj|down_proj|w1|w2|w3)\b",         "2d"),
    ("moe_expert",    r"experts?\.(gate_up_proj|down_proj)\b",                              "3d"),
    ("moe_router",    r"(mlp\.)?(gate|router)\.(weight|wg)\b|\brouter\b",                   "2d"),
    ("shared_expert", r"shared_expert(s)?\.",                                               "2d"),
    # latent attention before GQA: an MLA block still carries q_proj/o_proj, so testing
    # GQA first labels a latent-attention model as plain GQA. Measured on Kimi-VL, which
    # scored a spurious gqa_attention until this order was fixed.
    ("latent_attention", r"(kv_a_proj|kv_b_proj|q_a_proj|q_b_proj|kv_a_layernorm)",         "*"),
    ("gqa_attention", r"self_attn\.(q_proj|k_proj|v_proj|o_proj)\b|attention\.w[qkvo]\b",   "2d"),
    ("recurrent_state", r"(mamba|ssm|conv1d|A_log|dt_bias|linear_attn|\bba_proj\b|"
                        r"\bdt_proj\b|in_proj|out_proj)\b",                                 "*"),
    ("rmsnorm",       r"(input_layernorm|post_attention_layernorm|norm|ln_\d|"
                      r"[qk]_norm)\.weight$",                                               "1d"),
    ("embed",         r"(embed_tokens|wte|word_embeddings)\.weight$",                       "2d"),
    ("lm_head",       r"(^|\.)lm_head\.weight$|(^|\.)output\.weight$",                       "2d"),
    ("mm_projector",  r"(multi_modal_projector|mm_projector|merger|vision_proj)",           "*"),
    # (^|\.) not ^: Qwen3-VL nests the tower as `model.visual.blocks.N...`, so an
    # anchored pattern missed 327 of its 882 tensors -- 37% of the model, silently
    # reported as UNRECOGNIZED. Same bug class as the lm_head `language_model.` prefix.
    ("vision_encoder", r"(^|\.)(vision_tower|visual|vision_model)\.",                       "*"),
]
LAYER_RE = re.compile(r"\.(?:layers?|h|blocks)\.(\d+)\.")


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "hawking-odyssey/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fetch(repo, rev):
    cfg = _get(RAW.format(repo=repo, rev=rev, f="config.json"))
    try:
        idx = _get(RAW.format(repo=repo, rev=rev, f="model.safetensors.index.json"))
        wmap = idx.get("weight_map") or {}
    except urllib.error.HTTPError:
        wmap = {}                      # single-shard repos have no index; recorded as such
    return cfg, sorted(wmap)


MATRIX = RH / "ORGAN_FRONTIER_MATRIX.json"


def known_organs():
    """Read the canonical library THROUGH its API. The frontier matrix is the single
    authority for which organ families Hawking knows; ORGAN_LIBRARY.json is the older
    per-organ genome file and is used only as a fallback when the matrix is absent."""
    if MATRIX.exists():
        d = json.load(open(MATRIX))
        measured = {e["organ"] for e in d.get("organs", []) if e.get("status") == "MEASURED"}
        declared = {e["organ"] for e in d.get("organs", [])} - measured
        return measured, declared
    if not ORGAN_LIB.exists():
        return set(), set()
    d = json.load(open(ORGAN_LIB))
    return {o.get("organ") for o in d.get("organs", []) if o.get("organ")}, set()


# The recognizer's organ spellings that the canonical library carries under another name.
ALIAS_TO_FAMILY = {"rmsnorm": "normalization"}


def shape_kind(name, cfg):
    """Rank from the name alone where the index gives no shapes: norms and biases are
    1d, expert stacks are 3d, everything else 2d. Recorded as INFERRED, never CITED."""
    if name.endswith(".bias") or "norm" in name or name.endswith("A_log") or "dt_bias" in name:
        return "1d"
    if re.search(r"experts?\.(gate_up_proj|down_proj|w[123])$", name):
        return "3d"
    return "2d"


def classify(names, cfg, known, declared=frozenset()):
    hits, unmatched = defaultdict(list), []
    for n in names:
        matched = False
        for organ, pat, want in FINGERPRINTS:
            if re.search(pat, n):
                k = shape_kind(n, cfg)
                if want in ("*", k):
                    hits[organ].append(n)
                    matched = True
                    break
        if not matched:
            unmatched.append(n)

    # An MLA model's residual q/o projections belong to its latent-attention organ, not
    # to a second, independent GQA organ. Fold them in and record the fold.
    folded = None
    if "latent_attention" in hits and "gqa_attention" in hits:
        folded = {"from": "gqa_attention", "into": "latent_attention",
                  "n_tensors": len(hits["gqa_attention"]),
                  "why": "latent-attention markers present; the remaining q/o projections are "
                         "part of that organ, not a separate GQA organ"}
        hits["latent_attention"] += hits.pop("gqa_attention")

    organs = []
    for organ, tensors in sorted(hits.items()):
        layers = {int(m.group(1)) for t in tensors if (m := LAYER_RE.search(t))}
        n = len(tensors)
        # Confidence is evidence-weighted, not a constant. One stray tensor matching a
        # pattern is weak; a pattern repeating across many layers is strong.
        rep = len(layers)
        conf = min(0.5 + 0.05 * rep + 0.002 * n, 0.98) if rep else min(0.3 + 0.02 * n, 0.6)
        fam = ALIAS_TO_FAMILY.get(organ, organ)
        # Three states, not two. A family the library DECLARES but has never measured is
        # not novel to Odyssey and it is not known science either -- calling it either
        # would mislead the planner downstream.
        status = ("KNOWN" if fam in known else
                  "DECLARED_UNMEASURED" if fam in declared else "NOVEL")
        organs.append({"organ": organ, "status": status, "n_tensors": n,
                       "n_layers": rep, "confidence": round(conf, 3),
                       "example": tensors[:2]})

    # Unmatched blocks are reported as unknown structure, never absorbed into a
    # neighbouring family. Cluster them by their name skeleton so the report is short.
    skel = Counter(re.sub(r"\.\d+\.", ".N.", u) for u in unmatched)
    unknown = [{"organ": "UNRECOGNIZED", "status": "NOVEL", "skeleton": s, "n_tensors": c,
                "confidence": round(min(0.25 + 0.01 * c, 0.5), 3)}
               for s, c in skel.most_common(12)]
    return organs, unknown, len(unmatched), folded


def novelty_score(cfg, organs, unknown):
    novel = [o["organ"] for o in organs if o["status"] in ("NOVEL", "DECLARED_UNMEASURED")]
    axes = set()
    for o in novel:
        if o.startswith("moe") or o == "shared_expert":
            axes.add("routing")
        elif o == "recurrent_state":
            axes.add("state")
        elif o in ("vision_encoder", "mm_projector"):
            axes.add("modality")
        elif o == "latent_attention":
            axes.add("attention")
    return {"novel_organs": sorted(set(novel)), "novel_axes": sorted(axes),
            "n_unrecognized_clusters": len(unknown),
            "new_scale_regime": cfg.get("num_hidden_layers"),
            "transfer_test_value": round(1.0 - 0.2 * len(axes), 3)}


def recognize(repo, rev, cfg=None, names=None):
    t0 = time.time()
    if cfg is None:
        cfg, names = fetch(repo, rev)
    known, declared = known_organs()
    organs, unknown, n_un, folded = classify(names, cfg, known, declared)
    return {
        "repo": repo, "revision": rev,
        "architectures": cfg.get("architectures"), "model_type": cfg.get("model_type"),
        "n_tensors": len(names), "n_unmatched": n_un,
        "organs": organs, "unrecognized": unknown, "folded_organ": folded,
        "novelty": novelty_score(cfg, organs, unknown),
        "analysis_wall_s": round(time.time() - t0, 3),
        "loaded_weights": False,
    }


def calibration(specimens):
    """Accuracy per confidence bucket. A high-confidence wrong answer is the failure
    this table exists to expose, so it is reported even when it is embarrassing."""
    buckets = defaultdict(lambda: [0, 0])
    tp = fp = fn = 0
    for s in specimens:
        truth = set(s["ground_truth"])
        got = {o["organ"] for o in s["result"]["organs"]}
        tp += len(truth & got); fp += len(got - truth); fn += len(truth - got)
        for o in s["result"]["organs"]:
            b = f"{int(o['confidence'] * 10) / 10:.1f}"
            buckets[b][1] += 1
            if o["organ"] in truth:
                buckets[b][0] += 1
    table = {b: {"n": n, "correct": c, "accuracy": round(c / n, 3)}
             for b, (c, n) in sorted(buckets.items())}
    miscal = [b for b, v in table.items() if float(b) >= 0.8 and v["accuracy"] < 0.8]
    return {"precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "per_confidence_bucket": table,
            "high_confidence_buckets_below_80pct_accuracy": miscal,
            "calibrated": not miscal}


# Blind specimens: none of the fingerprints were written by looking at these files.
# Ground truth is hand-checked from each config and recorded here so the score is
# auditable rather than self-graded.
BLIND = [
    # Ground truth is read off the fetched tensor index, one entry per organ actually
    # present. The first hand-check of O005 wrongly listed a dense mlp_gate_up/mlp_down:
    # Qwen3-30B-A3B has NO dense MLP, every MLP tensor lives under mlp.experts.N. That
    # error cost 2 false negatives in the first calibration run and is corrected here
    # rather than quietly dropped -- a wrong ground truth scores the recognizer wrong in
    # both directions.
    ("O005", "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
     ["moe_expert", "moe_router", "gqa_attention", "rmsnorm", "embed", "lm_head"]),
    ("O001", "tiiuae/Falcon-H1-7B-Instruct", "41e72f27effbab80cd45b6e884688452253a3686",
     ["mlp_gate_up", "mlp_down", "gqa_attention", "recurrent_state", "rmsnorm", "embed",
      "lm_head"]),
    ("O003", "moonshotai/Kimi-VL-A3B-Instruct", None,
     ["mlp_gate_up", "mlp_down", "moe_expert", "moe_router", "shared_expert",
      "latent_attention", "rmsnorm", "embed", "lm_head", "vision_encoder", "mm_projector"]),
]

# HELD OUT. The three specimens above stopped being blind the moment two fingerprint
# bugs they exposed (lm_head under a `language_model.` prefix; latent attention tested
# after GQA) were fixed, so their 1.0/1.0 is IN-SAMPLE. These two were never looked at
# while the fingerprints were being written or fixed, and their score is the honest
# out-of-sample number.
# EXPOSED A BUG, THEREFORE IN-SAMPLE. Qwen3-VL-30B-A3B was resident but uncensused, and
# censusing it is what revealed that the vision fingerprint was anchored with ^ while the
# model nests its tower as `model.visual.blocks.N...` -- 327 of 882 tensors, 37% of the
# model, silently UNRECOGNIZED. Fixing that then exposed a second bug: the broad tower
# pattern swallowed `model.visual.merger.*`, so mm_projector had to be tested first.
# Both were found using this specimen, so its 8/8 is in-sample and is recorded as such
# rather than being passed off as a held-out score.
EXPOSED_BUGS = [
    ("O011", "Qwen/Qwen3-VL-30B-A3B-Instruct", "9c4b90e1e4ba",
     ["embed", "gqa_attention", "lm_head", "moe_expert", "moe_router", "rmsnorm",
      "vision_encoder", "mm_projector"]),
]

HELDOUT = [
    ("O010", "zai-org/GLM-4.5-Air", None,
     ["mlp_gate_up", "mlp_down", "moe_expert", "moe_router", "shared_expert",
      "gqa_attention", "rmsnorm", "embed", "lm_head"]),
    ("O004", "mistralai/Mistral-Small-3.1-24B-Instruct-2503", None,
     ["mlp_gate_up", "mlp_down", "gqa_attention", "rmsnorm", "embed", "lm_head",
      "vision_encoder", "mm_projector"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", required=True)
    ap.add_argument("--repo"); ap.add_argument("--revision")
    a = ap.parse_args()

    if a.repo:
        print(json.dumps(recognize(a.repo, a.revision or "main"), indent=1))
        return 0

    q = json.load(open(RH / "ODYSSEY_QUEUE_RECOVERED.json"))
    revs = {e["oxx"]: e["canonical_revision"] for e in q["queue"]}
    fixtures, specimens, heldout, errors = {}, [], [], []
    for oxx, repo, rev, truth in BLIND + EXPOSED_BUGS + HELDOUT:
        # EXPOSED_BUGS belongs with BLIND, not with HELDOUT. A specimen used to FIND a
        # fingerprint bug is in-sample by definition; routing it to the held-out bucket
        # would inflate the only out-of-sample number this receipt has.
        in_sample = (oxx, repo, rev, truth) in BLIND + EXPOSED_BUGS
        into = specimens if in_sample else heldout
        rev = rev or revs.get(oxx) or "main"
        try:
            cfg, names = fetch(repo, rev)
        except Exception as e:
            errors.append({"oxx": oxx, "repo": repo, "error": f"{type(e).__name__}: {e}"})
            continue
        fixtures[repo] = {"revision": rev, "config": cfg, "n_tensors": len(names),
                          "tensor_names": names}
        into.append({"oxx": oxx, "model": repo, "ground_truth": sorted(truth),
                          "ground_truth_source": "hand-checked from the fetched config.json "
                                                 "and tensor index, recorded here for audit",
                          "result": recognize(repo, rev, cfg, names)})

    # Negative control: the same tensor map with every organ name replaced by nonsense.
    # A force-fitting recognizer would still report KNOWN organs here.
    control = None
    if specimens:
        src = fixtures[specimens[0]["model"]]
        scrambled = [re.sub(r"[A-Za-z_]+", lambda m: "zzz" + str(len(m.group(0))), n)
                     for n in src["tensor_names"]]
        control = {"what": "every tensor name replaced by a nonsense token of the same length",
                   "result": recognize("CONTROL/scrambled", "n/a", src["config"], scrambled)}

    cal = calibration(specimens)
    cal_out = calibration(heldout) if heldout else None
    out = {
        "schema": "hawking.headless.architecture_recognizer.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/arch_recognizer.py",
        "obligation": "G016 — ARCHITECTURE_RECOGNIZER (directive §47, §46)",
        "hand_authored": False, "did_not_load_weights": True, "unmeasured_is_absent": True,
        "known_organs_from": str(RH / "ORGAN_FRONTIER_MATRIX.json"),
        "known_organs": sorted(known_organs()[0]),
        "declared_but_unmeasured": sorted(known_organs()[1]),
        "consumer_trace": "known_organs() reads receipts/headless/ORGAN_FRONTIER_MATRIX.json "
                          "through the canonical library; MEASURED families are KNOWN, declared "
                          "families with no measurement are DECLARED_UNMEASURED, anything else "
                          "is NOVEL",
        "organ_library_gaps": sorted({o["organ"] for s in specimens + heldout
                                      for o in s["result"]["organs"]
                                      if o["status"] == "NOVEL"}),
        "organ_library_gaps_note": "organs recognized here that the canonical OrganLibrary "
                                   "does not yet carry and which are NOT genuinely novel to "
                                   "Odyssey -- normalization is listed as a family in the "
                                   "directive but absent from the library, so it reports NOVEL",
        "specimens": specimens, "heldout_specimens": heldout, "fetch_errors": errors,
        "in_sample_warning": "the three `specimens` exposed two fingerprint bugs that were then "
                             "fixed, so their calibration is IN-SAMPLE. `calibration_heldout` is "
                             "the out-of-sample number and is the one that counts.",
        "negative_control": control,
        "calibration_in_sample": cal, "calibration_heldout": cal_out,
        "pass": bool(len(specimens) >= 3 and cal["calibrated"]
                     and cal_out and cal_out["calibrated"]
                     and control and not [o for o in control["result"]["organs"]
                                          if o["status"] == "KNOWN"]),
    }
    FIXTURES.write_text(json.dumps(
        {"note": "config + tensor-name metadata fetched for the blind run; no weights",
         "fixtures": fixtures}, indent=1))
    Path(a.emit).write_text(json.dumps(out, indent=1))
    for s in specimens:
        r = s["result"]
        print(f"{s['oxx']:5} {s['model'][:38]:40} organs={len(r['organs']):2} "
              f"novel={r['novelty']['novel_organs']} unmatched={r['n_unmatched']}")
    for s2 in heldout:
        r = s2["result"]
        print(f"HELD {s2['oxx']:4} {s2['model'][:38]:40} organs={len(r['organs']):2} "
              f"unmatched={r['n_unmatched']}")
    print("in-sample :", json.dumps({k: cal[k] for k in ("precision", "recall", "calibrated")}))
    print("held-out  :", json.dumps({k: cal_out[k] for k in ("precision", "recall", "calibrated")})
          if cal_out else "none")
    if control:
        print("control KNOWN organs:", [o["organ"] for o in control["result"]["organs"]
                                        if o["status"] == "KNOWN"])
    print("pass:", out["pass"])
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
