import json

from tools.future import mutation_surface as ms
from tools.future._common import RECEIPTS


def test_sidecar_carves_out_of_codex_glob():
    # tools/* is Codex-owned, but tools/future/* must win.
    assert ms.owner("tools/future/hwir.py") == "SIDECAR"
    assert ms.owner("receipts/future/X.json") == "SIDECAR"
    assert ms.owner("tools/accelerator/scoreboard.py") == "CODEX"
    assert ms.owner("crates/hawking-core/shaders/mha.metal") == "CODEX"
    assert ms.owner("receipts/headless/ACCELERATOR_SCOREBOARD.json") == "CODEX"


def test_checker_passes_on_sidecar_namespace():
    assert ms.check_disjoint(["tools/future", "receipts/future"]) == 0


def test_checker_fails_on_codex_path():
    # The negative control: the checker must actually refuse.
    assert ms.check_disjoint(["crates"]) == 1
    assert ms.check_disjoint(["tools/accelerator"]) == 1


def test_build_emits_sealed_receipt_with_evidence():
    out = ms.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert doc["schema"] == "hawking.future.codex_mutation_surface.v1"
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["surface_evidence"], "ownership map must carry mtime evidence"
    for row in doc["surface_evidence"]:
        assert row["newest_mtime_epoch"] > 0
        assert row["newest_file"]
