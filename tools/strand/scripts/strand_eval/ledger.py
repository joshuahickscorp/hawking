# strand_eval.ledger — the results ledger + the tells as code (audit 3.2).
#
# Append-only jsonl (research/results-ledger.jsonl). `check` automates the two
# §5.4 habits that previously lived in a human eyeball:
#   ERROR  the 15-digit tell — two records with BIT-IDENTICAL ppl but different
#          configs (model/tag/harness_key) = contamination/bug, every time
#          (this exact tell caught the MP_FALLBACK=4 lie).
#   WARN   harness_key mismatch — same model evaluated under different keys
#          (the 64w-vs-146w "~comparable" prose, made mechanical).
#   WARN   un-provenanced records (no harness_key — e.g. ingested legacy jsons)
#          cannot silently enter a canon comparison.
#
# Torch-free by design: conductor calls `ingest` every pod poll.

import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def append_record(ledger_path, rec):
    """One jsonl line, append-only, dirs created. Caller-side timestamp is already
    in the record (build_record)."""
    os.makedirs(os.path.dirname(os.path.abspath(ledger_path)), exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def read_ledger(ledger_path):
    recs = []
    if not os.path.exists(ledger_path):
        return recs
    with open(ledger_path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append((ln, json.loads(line)))
            except ValueError:
                recs.append((ln, {"_corrupt_line": line[:120]}))
    return recs


def _config_id(rec):
    return (rec.get("model"), rec.get("tag"), rec.get("harness_key8"))


def check(ledger_path):
    """Returns (errors, warnings) — lists of human-readable strings.

    Advisory by design: append-only data, the checker never mutates."""
    errors, warnings = [], []
    recs = read_ledger(ledger_path)

    by_ppl = {}
    for ln, r in recs:
        if "_corrupt_line" in r:
            warnings.append(f"line {ln}: corrupt jsonl: {r['_corrupt_line']}")
            continue
        ppl = r.get("ppl")
        if ppl is None:
            warnings.append(f"line {ln}: record without ppl "
                            f"(model={r.get('model')} tag={r.get('tag')})")
            continue
        # bit-identity: compare the full repr of the float64, not a rounding
        by_ppl.setdefault(repr(float(ppl)), []).append((ln, r))
        if not r.get("harness_key8"):
            warnings.append(
                f"line {ln}: UN-PROVENANCED record (no harness_key) — "
                f"model={r.get('model')} tag={r.get('tag')} ppl={ppl} "
                f"[provenance={r.get('provenance', '?')}] — must not enter a canon comparison")
        elif not r.get("harness_version"):
            warnings.append(f"line {ln}: record without harness_version "
                            f"(model={r.get('model')} tag={r.get('tag')})")

    # THE 15-DIGIT TELL
    for ppl_repr, group in by_ppl.items():
        if len(group) < 2:
            continue
        configs = {_config_id(r) for _, r in group}
        if len(configs) > 1:
            desc = "; ".join(
                f"line {ln}: model={r.get('model')} tag={r.get('tag')} "
                f"key={r.get('harness_key8')}" for ln, r in group)
            errors.append(
                f"15-DIGIT TELL: ppl={ppl_repr} bit-identical across DIFFERENT configs "
                f"→ contamination/bug ({desc})")
        else:
            warnings.append(
                f"duplicate result: ppl={ppl_repr} appears {len(group)}× for the same "
                f"config {next(iter(configs))} (re-run/resume echo — harmless, noted)")

    # HARNESS-KEY MISMATCH (cross-key comparison hazard)
    by_model = {}
    for ln, r in recs:
        if r.get("ppl") is None or "_corrupt_line" in r:
            continue
        by_model.setdefault(r.get("model"), []).append((ln, r))
    for model, group in by_model.items():
        keys = {}
        for ln, r in group:
            k = r.get("harness_key8") or "<none>"
            keys.setdefault(k, []).append(r.get("tag"))
        if len(keys) > 1:
            kdesc = "; ".join(f"{k}: tags={sorted(set(t for t in tags if t))}"
                              for k, tags in sorted(keys.items()))
            warnings.append(
                f"HARNESS MISMATCH: model={model} has results under {len(keys)} "
                f"harness_keys — do NOT compare across keys ({kdesc})")
        # same (model, tag) under different keys = someone WILL compare them
        by_tag = {}
        for ln, r in group:
            by_tag.setdefault(r.get("tag"), set()).add(r.get("harness_key8") or "<none>")
        for tag, tkeys in by_tag.items():
            if len(tkeys) > 1:
                warnings.append(
                    f"HARNESS MISMATCH (same model+tag): model={model} tag={tag} "
                    f"spans keys {sorted(tkeys)} — these are NOT the same measurement")

    return errors, warnings


def _legacy_model_id(src_path, j):
    """Model id for a legacy json: the ppl_<model>_<tag>.json filename prefix when
    present (pod-chain v3.6 names), else the generic-leaf walk over the recorded
    model path ('scratch/qwen-7b/.../q2_l12_out1/recon' must NOT become 'recon' —
    that string was the llama2-overwrote-qwen incident)."""
    base = os.path.basename(src_path)
    stem = base[4:-5] if base.startswith("ppl_") and base.endswith(".json") else ""
    tag = str(j.get("tag") or "")
    if tag and stem.endswith("_" + tag) and stem != "_" + tag:
        return stem[: -len(tag) - 1]
    mp = str(j.get("model") or "").rstrip("/")
    if mp:
        from strand_eval.core import model_id_from_dir
        return model_id_from_dir(mp)
    return None


def _ingest_key(rec):
    return (os.path.basename(rec.get("source_path", "") or rec.get("out_json", "") or ""),
            repr(float(rec.get("ppl"))) if rec.get("ppl") is not None else None)


def ingest(ledger_path, src_dir, quiet=False):
    """Scan src_dir for ppl_*.json (legacy or canonical shape) and append any not
    already in the ledger. Idempotent (dedupe key = source basename + bit-exact ppl).
    Legacy records get provenance='ingested' and NO harness_key — `check` keeps
    them visibly un-provenanced. Returns the number of records appended."""
    import glob
    existing = {_ingest_key(r) for _, r in read_ledger(ledger_path)}
    added = 0
    for p in sorted(glob.glob(os.path.join(src_dir, "ppl_*.json"))):
        try:
            with open(p) as f:
                j = json.load(f)
        except (OSError, ValueError):
            if not quiet:
                print(f"[ledger] skip unreadable {p}", flush=True)
            continue
        if j.get("ppl") is None:
            continue
        if j.get("schema"):  # canonical record: take it whole
            rec = dict(j)
            rec.setdefault("source_path", p)
        else:  # legacy eval json: wrap, no fabricated harness_key
            rec = {
                "provenance": "ingested",
                "source_path": p,
                "model": _legacy_model_id(p, j),
                "model_path": j.get("model"),
                "tag": j.get("tag"),
                "ppl": j.get("ppl"),
                "ctx": j.get("ctx"),
                "chunks": j.get("chunks"),
                "tokens": j.get("tokens"),
                "device": j.get("device"),
                "dtype": j.get("dtype"),
                "harness_key": None,
                "harness_key8": None,
            }
        k = _ingest_key(rec)
        if k in existing:
            continue
        import time
        rec.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        rec.setdefault("epoch", int(time.time()))
        append_record(ledger_path, rec)
        existing.add(k)
        added += 1
        if not quiet:
            print(f"[ledger] ingested {os.path.basename(p)} "
                  f"(model={rec.get('model')} tag={rec.get('tag')} ppl={rec.get('ppl')})",
                  flush=True)
    if added and not quiet:
        print(f"[ledger] +{added} record(s) from {src_dir}", flush=True)
    return added
