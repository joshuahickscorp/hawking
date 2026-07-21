from __future__ import annotations

import json
from pathlib import Path
import sys


TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_doctor_prime as doctor  # noqa: E402


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def test_causal_harness_recovers_known_injected_failure(tmp_path: Path) -> None:
    corpus, parent, output = (tmp_path / name for name in
                              ("corpus.json", "parent.json", "causal.json"))
    write(corpus, {"status": "PASS", "seal_sha256": "corpus"})
    write(parent, {"status": "PASS", "seal_sha256": "parent"})
    result = doctor.causal_harness(corpus, parent, output)
    assert result["status"] == "PASS"
    assert result["teacher_replay_exact"] is True
    assert result["classified_ground_truth"] == "ROUTED_EXPERT_OUTPUT_BOUND"
    assert result["real_parent_candidate_classification"].startswith("PENDING")


def test_exact_byte_auction_never_crosses_candidate_rate(tmp_path: Path) -> None:
    ledger, causal, output = (tmp_path / name for name in
                              ("ledger.json", "causal.json", "auction.json"))
    denominator = 1_000_000_003
    write(ledger, {
        "all_logical_original_weights": denominator,
        "text_core_logical_weights": 990_000_000,
        "compressible_logical_weights": 900_000_000,
        "active_text_core_logical_weights_per_token": 120_000_000,
        "seal_sha256": "ledger",
    })
    write(causal, {"status": "PASS", "seal_sha256": "causal"})
    result = doctor.doctor_auction(ledger, causal, output)
    assert result["status"] == "PASS"
    for row in result["rows"]:
        numerator, target_denominator = map(int, row["target_complete_bpw"].split("/"))
        installed_bits = row["installed_byte_ceiling"] * 8
        assert installed_bits * target_denominator <= denominator * numerator
        assert sum(row["allocation_bytes"].values()) == row["installed_byte_ceiling"]
        assert isinstance(row["resident_fit_with_32_gib_floor"], bool)
