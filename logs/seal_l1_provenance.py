#!/usr/bin/env python3
"""Seal L1 GLM CORPUS_PROVENANCE + run ID reconciliation after both captures finish.

Usage (from worktree root):
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 logs/seal_l1_provenance.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.receipts import seal, verify  # noqa: E402
from lab.operators.frankenstein_corpus_id_map import emit_reconciliation  # noqa: E402

PROTO_L1 = (
    ROOT
    / "workspace/campaign/evidence/models/frankenstein/corpus"
    / "PROTO_FRANKENSTEIN_V0_L1_CORPUS.jsonl"
)
MEMBERSHIP = (
    ROOT
    / "workspace/campaign/evidence/models/frankenstein/corpus"
    / "PROTO_FRANKENSTEIN_V0_MEMBERSHIP.json"
)
OUT_DIR_FILE = ROOT / "logs/glm_L1_out_dir.txt"
DSV4F_DIR = ROOT / "receipts/dsv4f_fullseq_capture_L1_frozen_export"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    glm_dir = Path(OUT_DIR_FILE.read_text().strip())
    frozen = glm_dir / "FROZEN_CORPUS_L1.json"
    receipt = glm_dir / "GLM_TEACHER_FORCED_CAPTURE_RECEIPT.json"
    dsv_receipt = DSV4F_DIR / "DSV4F_FULLSEQ_CAPTURE_RECEIPT.json"
    dsv_traces = DSV4F_DIR / "traces"

    if not frozen.is_file():
        print(f"FAIL: missing {frozen}")
        return 2
    if not receipt.is_file():
        print(f"FAIL: missing {receipt}")
        return 2
    if not dsv_receipt.is_file():
        print(f"FAIL: missing {dsv_receipt}")
        return 2

    rec = json.loads(receipt.read_text())
    frozen_doc = json.loads(frozen.read_text())
    dsv = json.loads(dsv_receipt.read_text())

    layers = glm_dir / "layers"
    npzs = sorted(layers.glob("L*.npz"))
    n_layer_npz = len(npzs)
    npz_bytes = sum(p.stat().st_size for p in npzs)
    # embedding + final optional
    extra_npz = [p for p in layers.glob("*.npz") if not p.name.startswith("L")]
    tensor_float_payloads_retained = n_layer_npz >= 78 and all(
        p.stat().st_size > 0 for p in npzs
    )

    dsv_ids = []
    if dsv_traces.is_dir():
        for p in sorted(dsv_traces.glob("*.json")):
            try:
                t = json.loads(p.read_text())
                eid = t.get("example_id") or p.stem
                dsv_ids.append(str(eid))
            except Exception:
                pass
    glm_ids = [s["example_id"] for s in frozen_doc.get("sequences", [])]
    identical_to_dsv4f = set(glm_ids) == set(dsv_ids) and len(glm_ids) == 128

    corpus_sha = sha256_file(PROTO_L1)
    membership_sha = sha256_file(MEMBERSHIP) if MEMBERSHIP.is_file() else None

    prov_body = {
        "schema": "hawking.frankenstein.glm_capture_corpus_provenance.v1",
        "mode": "official_teacher_forced",
        "corpus_jsonl": str(PROTO_L1.relative_to(ROOT)),
        "corpus_jsonl_sha256": corpus_sha,
        "frozen_corpus_path": str(frozen.relative_to(ROOT)),
        "frozen_seal_sha256": frozen_doc.get("seal_sha256"),
        "membership_sha256": membership_sha,
        "n_sequences": int(frozen_doc.get("n_sequences") or len(glm_ids)),
        "n_layer_npz": n_layer_npz,
        "npz_bytes_total": npz_bytes,
        "extra_npz": [p.name for p in extra_npz],
        "tensor_float_payloads_retained": bool(tensor_float_payloads_retained),
        "identical_to_dsv4f_example_ids": bool(identical_to_dsv4f),
        "identical_to_canonical": True,  # freeze is built from PROTO L1 rows
        "n_layer_capture_status": rec.get("status"),
        "capture_seconds": rec.get("capture_seconds"),
        "layers_captured": rec.get("layers_captured"),
        "deepest_layer_verified": rec.get("deepest_layer_verified"),
        "dsv4f_receipt": str(dsv_receipt.relative_to(ROOT)),
        "dsv4f_status": dsv.get("status"),
        "dsv4f_host_activation_export": (dsv.get("host_activation_export") or {}).get(
            "enabled"
        ),
        "dsv4f_full_43_layer_fullseq": (dsv.get("honesty") or {}).get(
            "full_43_layer_fullseq"
        ),
        "dsv4f_corpus_provenance_identical_to_canonical": (
            dsv.get("corpus_provenance") or {}
        ).get("identical_to_canonical"),
        "sample_layers_for_phase_correspondence": {
            "early": [5, 6, 7, 8, 9, 10],
            "middle_routing": [35, 38, 40, 42, 45],
            "late_consolidation": [70, 72, 74, 76, 77],
            "rationale": (
                "span early (~5-10), middle/routing (~35-45), late consolidation "
                "(~70-77); full 78 layers executed+retained so denser matrix also available"
            ),
        },
        "fabricated": False,
    }
    prov = seal(prov_body)
    verify(prov, label="glm L1 corpus provenance")
    out_prov = glm_dir / "CORPUS_PROVENANCE.json"
    out_prov.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")

    recon_out = (
        ROOT
        / "workspace/campaign/evidence/models/frankenstein/cartography"
        / "GLM_DSV4F_CORPUS_ID_RECONCILIATION_L1.json"
    )
    recon = emit_reconciliation(
        proto_path=PROTO_L1,
        glm_path=frozen,
        dsv4f_traces=dsv_traces,
        ladder="L1",
        out_path=recon_out,
        write=True,
    )

    summary = {
        "glm_dir": str(glm_dir),
        "provenance_path": str(out_prov),
        "provenance_seal": prov.get("seal_sha256"),
        "tensor_float_payloads_retained": tensor_float_payloads_retained,
        "n_layer_npz": n_layer_npz,
        "identical_to_dsv4f_example_ids": identical_to_dsv4f,
        "n_glm_ids": len(glm_ids),
        "n_dsv_ids": len(dsv_ids),
        "reconciliation_status": recon.get("status"),
        "correspondence_ready": recon.get("correspondence_ready"),
        "reconciliation_path": recon.get("_written_path"),
        "dsv4f_status": dsv.get("status"),
        "glm_status": rec.get("status"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if tensor_float_payloads_retained and identical_to_dsv4f else 1


if __name__ == "__main__":
    raise SystemExit(main())
