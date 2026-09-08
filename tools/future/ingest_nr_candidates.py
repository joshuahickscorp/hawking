"""Attribute already-measured NR receipts onto the Odyssey ledger.

G009 wants every axis MEASURED or REFUSED-with-mechanism. This does not invent
GPU rates. It records NR work that already sits in receipts/future, including
the G004 finding that capability was correctly NOT substituted by reconstruction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from odyssey_ledger import measured, refused, progress  # noqa: E402

LEDGER = "receipts/future/G034_ODYSSEY_LEDGER.json"

# slug, axis, kind, payload
ROWS = (
    {
        "slug": "moonshotai--Kimi-VL-A3B-Instruct@398eede0903c",
        "nr_receipt": "receipts/future/G004_LOWRANK_O003.json",
        "nr_value": {
            "form": "within-tensor low-rank NR vs affine/PQ at matched EBPW",
            "structure_exists": "partly, organ- and depth-specific; gate_proj REAL at depth, down/up ABSENT",
            "factoring_pays": "barely; best cell saves 6.3% of bytes",
            "capability": "NOT_MEASURED (Metal required for gravity_outlier_eval.evaluate; reconstruction is not a substitute)",
            "wait": "receipts/future/G004_CAPABILITY_RESOURCE_WAIT.json",
        },
        "nx_reason": (
            "G004 low-rank is an NR that rematerializes dense W_hat; the capability "
            "conjunction on Kimi-VL-A3B has not run because mlx_lm cannot load Metal "
            "in this sandbox (G004_CAPABILITY_RESOURCE_WAIT.json). Reopen when "
            "gravity_outlier_eval.evaluate runs D vs E vs broken on unsandboxed Metal."
        ),
    },
    {
        "slug": "Qwen--Qwen3-30B-A3B@ad44e777bcd1",
        "nr_receipt": None,
        "nr_refuse": (
            "G003 fourth-moment energy discriminator vs a permutation null is DEAD "
            "on the real organ (energy deficit 0.71% below the 2% LIVE bar, "
            "receipts/future/G003_NONLINEAR_DISCRIMINATOR.json). That does not license "
            "an NR. Reopen if a different statistic on this organ goes LIVE against "
            "the same permutation null."
        ),
        "nx_reason": (
            "No NX: G003 is a discriminator, not a stored representation, and it "
            "returned NEGATIVE. Reopen when a representation that exploits a LIVE "
            "energy statistic is built and gated."
        ),
    },
)


def ingest(ledger_path: str = LEDGER, dry_run: bool = False) -> dict:
    led = json.loads(Path(ledger_path).read_text())
    by = {r["slug"]: r for r in led["specimens"]}
    actions = []
    for spec in ROWS:
        rec = by.get(spec["slug"])
        if rec is None:
            raise RuntimeError(f"not in ledger: {spec['slug']}")
        if rec["axes"]["nr_candidate"]["state"] == "OWED":
            if spec.get("nr_refuse"):
                refused(rec, "nr_candidate", spec["nr_refuse"])
                actions.append(f"{spec['slug']}: nr_candidate REFUSED")
            else:
                measured(rec, "nr_candidate", spec["nr_value"], spec["nr_receipt"])
                actions.append(f"{spec['slug']}: nr_candidate MEASURED")
        else:
            actions.append(
                f"{spec['slug']}: nr_candidate already {rec['axes']['nr_candidate']['state']}"
            )
        if rec["axes"]["nx_disposition"]["state"] == "OWED":
            refused(rec, "nx_disposition", spec["nx_reason"])
            actions.append(f"{spec['slug']}: nx_disposition REFUSED")
        else:
            actions.append(
                f"{spec['slug']}: nx_disposition already {rec['axes']['nx_disposition']['state']}"
            )
    if not dry_run:
        Path(ledger_path).write_text(json.dumps(led, indent=1) + "\n")
    return {"actions": actions, "progress": progress(led)}


# Mechanism-backed nr/nx refusals. Not "mlx_lm has no module" for a causal LM
# (G017 gated Qwen3-0.6B on transformers CPU). These axes cannot exist until
# the named reopen fires.
REFUSALS = (
    {
        "match": lambda r: r.get("class") == "NO-SAFETENSORS",
        "nr": (
            "no safetensors payload exists under the snapshot, so there is no "
            "parent weight an NR could represent. REOPEN CONDITION: a loadable "
            "float or quantized weight file is present and readable."
        ),
        "nx": (
            "no parent weights on disk, so no NX can be packed. REOPEN CONDITION: "
            "a loadable weight file exists."
        ),
    },
    {
        "match": lambda r: r.get("class") == "MOE-PREQUANTIZED",
        "nr": (
            "expert payload is pre-quantized on disk (see anatomy refusal). An NR "
            "built on those codes would measure the codebook, not the organism. "
            "REOPEN CONDITION: a float parent of the experts is available, or an "
            "NR is defined over the packed codes with the codebook billed."
        ),
        "nx": (
            "no float expert parent to freeze into an NX; the on-disk codes are a "
            "foreign quantiser. REOPEN CONDITION: same as nr_candidate."
        ),
    },
    {
        "slugs": (
            "openai--whisper-large-v3-turbo@41f01f3fe87f",
            "answerdotai--ModernBERT-large@45bb4654a4d5",
            "Qwen--Qwen3-Embedding-0.6B@97b0c614be4d",
            "lerobot--pi0_base@25c379b52ba2",
        ),
        "nr": (
            "NR/NX are defined against this campaign's capability gate, a "
            "CONJUNCTION of perplexity AND n-gram generation diversity on TEXT. "
            "This body cannot generate text under that gate (ASR/encoder/embedding/"
            "robotics). G017 skipped Qwen3-Embedding-0.6B for this reason. "
            "REOPEN CONDITION: a role-specific capability gate is defined for this "
            "modality (directive 115), or the body is used only as a component."
        ),
        "nx": (
            "the campaign NX disposition requires the text conjunction gate, which "
            "is undefined for this modality. REOPEN CONDITION: same as nr_candidate."
        ),
    },
)


def ingest_refusals(ledger_path: str = LEDGER, dry_run: bool = False) -> dict:
    led = json.loads(Path(ledger_path).read_text())
    actions = []
    for rec in led["specimens"]:
        for rule in REFUSALS:
            slugs = rule.get("slugs")
            if slugs and rec["slug"] not in slugs:
                continue
            if "match" in rule and not rule["match"](rec):
                continue
            if rec["axes"]["nr_candidate"]["state"] == "OWED":
                refused(rec, "nr_candidate", rule["nr"])
                actions.append(f"{rec['slug']}: nr_candidate REFUSED")
            if rec["axes"]["nx_disposition"]["state"] == "OWED":
                refused(rec, "nx_disposition", rule["nx"])
                actions.append(f"{rec['slug']}: nx_disposition REFUSED")
    if not dry_run:
        Path(ledger_path).write_text(json.dumps(led, indent=1) + "\n")
    return {"actions": actions, "progress": progress(led)}


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    out = ingest(dry_run=dry)
    out2 = ingest_refusals(dry_run=dry)
    print(json.dumps({"known_nr": out, "refusals": out2}, indent=2, default=str))
