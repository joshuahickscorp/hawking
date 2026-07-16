from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
MODULE_PATH = CONDENSE / "appendix_handoff.py"
SPEC = importlib.util.spec_from_file_location("appendix_handoff", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_handoff_is_deterministic_complete_and_self_verifying() -> None:
    packet = MODULE.build_packet()
    assert packet == MODULE.build_packet()
    assert MODULE.verify_packet(packet) == []
    assert packet["coverage"]["capability_sectors"] == 25
    assert packet["coverage"]["static_tq_cells"] > 0
    assert packet["fingerprints"]["counter_authority_registry_sha256"]
    assert packet["fingerprints"]["counter_executor_capability_sha256"]
    assert packet["fingerprints"]["counter_request_builder_config_sha256"]
    assert all(row["exists"] for row in packet["source_manifest"])


def test_handoff_detects_source_manifest_tampering_after_restamp() -> None:
    packet = MODULE.build_packet()
    packet["source_manifest"][0]["sha256"] = "0" * 64
    packet = MODULE._stamp_packet(packet)
    errors = MODULE.verify_packet(packet)
    assert any("source file fingerprint mismatch" in error for error in errors)
