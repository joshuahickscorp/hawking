"""N023 information-accounting canary: hidden model-specific bytes must be counted.

If the existing accountant is fooled on any real hiding channel, these tests
FAIL. They plant, walk the closure, and write the receipt — they do not trust
a pre-written PASS.

    python3 -m pytest tools/headless/test_information_accounting_canary.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import noetic_information_accounting as accounting  # noqa: E402
from noetic_information_accounting_canary import (  # noqa: E402
    CONTROL_CHANNELS,
    REAL_CHANNELS,
    RECEIPT,
    SCHEMA,
    measure_plant,
    npy_f32_blob,
    non_weightlike_entropy,
    plant_cache_sidecar,
    plant_generated_source,
    plant_header18_alignment,
    plant_runtime_mlx_blobs,
    plant_shader_constants,
    write_receipt,
)

_DOC: dict | None = None


def doc() -> dict:
    """Run the adversary every collection. A stale PASS on disk is not evidence."""
    global _DOC
    if _DOC is None:
        _DOC = write_receipt()
    return _DOC


def test_receipt_written_with_schema_and_no_model_load():
    d = doc()
    assert RECEIPT.is_file(), f"missing {RECEIPT}"
    disk = json.loads(RECEIPT.read_text())
    assert disk["schema"] == SCHEMA
    assert d["schema"] == SCHEMA
    assert d["did_not_load_model"] is True
    assert d["did_not_mutate_parent_a"] is True
    assert d["did_not_touch_crates"] is True
    assert d["accountant"]["invented_second_system"] is False
    assert "account_tree" in d["accountant"]["closure_walk"]
    assert "embedded_weightlike_bytes" in d["accountant"]["detector"]


def test_every_real_channel_is_caught():
    """The finding if this fails is a HOLE: the accountant was fooled."""
    d = doc()
    holes = []
    for name in REAL_CHANNELS:
        row = d["channels"][name]
        if not row["caught"] or row["counted_bytes"] < row["planted_bytes"]:
            holes.append(
                f"{name}: planted={row['planted_bytes']} counted={row['counted_bytes']} "
                f"where={row['where']}"
            )
        assert row["planted_bytes"] > 0, name
        assert row["counted_bytes"] >= row["planted_bytes"], (
            f"HOLE: {name} fooled the accountant "
            f"(planted {row['planted_bytes']} B, counted {row['counted_bytes']} B) "
            f"at {row['where']}: {row['hide']}"
        )
        assert row["caught"] is True, f"HOLE: {name}"
    assert holes == [], f"hiding channels uncaught: {holes}"
    assert d["holes"] == []
    assert d["NOETIC_INFORMATION_ACCOUNTING"] == "PASS"


def test_shader_constants_channel():
    row = doc()["channels"]["shader_constants"]
    assert row["where"].endswith(".metal")
    assert row["counted_bytes"] >= row["planted_bytes"] >= 256 * 4
    live = measure_plant(plant_shader_constants)
    assert live["caught"] is True
    assert live["counted_bytes"] >= live["planted_bytes"]


def test_generated_source_python_and_rust():
    row = doc()["channels"]["generated_source"]
    assert "baked_bias.py" in row["where"] and "baked_bias.rs" in row["where"]
    parts = row["parts"]
    assert parts["python"] > 0 and parts["rust"] > 0
    assert row["planted_bytes"] == parts["python"] + parts["rust"]
    assert row["counted_bytes"] >= row["planted_bytes"]
    live = measure_plant(plant_generated_source)
    assert live["caught"] is True


def test_cache_sidecar_bin_next_to_executable():
    row = doc()["channels"]["cache_sidecar"]
    assert row["where"] == "runtime/sidecar.bin"
    assert row["counted_bytes"] >= row["planted_bytes"] >= 512 * 4
    live = measure_plant(plant_cache_sidecar)
    assert live["caught"] is True
    ew = live["embedded_weightlike_bytes"]
    assert ew["kind"] == "MEASURED"
    assert ew["value"] >= live["planted_bytes"]


def test_route_tables_oversized_metadata():
    row = doc()["channels"]["route_tables"]
    assert "machine_genome.json" in row["where"]
    assert row["planted_bytes"] >= 128 * 4
    assert row["counted_bytes"] >= row["planted_bytes"]


def test_runtime_mlx_npy_is_opened_not_trusted_by_suffix():
    row = doc()["channels"]["runtime_mlx_blobs"]
    assert row["where"].endswith(".npy")
    assert row["counted_bytes"] >= row["planted_bytes"]
    live = measure_plant(plant_runtime_mlx_blobs)
    assert live["caught"] is True
    ew = live["embedded_weightlike_bytes"]
    assert ew["kind"] == "MEASURED"
    assert ew["value"] >= live["planted_bytes"]


def test_eighteen_byte_header_alignment_trick():
    """v1 scanned floats only from offset 0; an 18-byte header defeated it."""
    payload = accounting.canary_f32_payload(128, b"n023-header18")
    blob = b"\x00" * 18 + payload
    got = accounting.embedded_weightlike_bytes(blob)
    assert got >= len(payload) - 4, (
        f"header of 18 bytes defeated the detector: {got} of {len(payload)}"
    )
    row = doc()["channels"]["header18_alignment"]
    assert row["counted_bytes"] >= row["planted_bytes"]
    live = measure_plant(plant_header18_alignment)
    assert live["caught"] is True
    ew = live["embedded_weightlike_bytes"]
    assert ew["kind"] == "MEASURED"
    assert ew["value"] >= live["planted_bytes"] - 4


def test_negative_control_is_not_flagged_as_weights():
    """Causal law: the canary must not just flag everything."""
    row = doc()["controls"]["negative_random_non_weightlike"]
    assert row["planted_bytes"] > 0
    assert row["counted_bytes"] == 0, (
        f"negative control was booked as MODEL_SPECIFIC ({row['counted_bytes']} B); "
        "the accountant is flagging non-weight-like entropy"
    )
    assert row["ok"] is True
    blob = non_weightlike_entropy(4096, b"n023-negative-entropy")
    assert accounting.embedded_weightlike_bytes(blob) == 0


def test_noop_zero_plant_zero_extra_count():
    row = doc()["controls"]["noop_zero_plant"]
    assert row["planted_bytes"] == 0
    assert row["counted_bytes"] == 0
    assert row["ok"] is True
    assert row["delta"]["MODEL_SPECIFIC_BYTES"] == 0
    for k, v in row["delta"].items():
        assert v == 0, f"noop changed {k} by {v}"


def test_both_controls_and_overall_pass():
    d = doc()
    for name in CONTROL_CHANNELS:
        assert d["controls"][name]["ok"] is True, name
    assert d["NOETIC_INFORMATION_ACCOUNTING"] == "PASS"
    assert d["tally"]["real_caught"] == len(REAL_CHANNELS)
    assert d["tally"]["real_missed"] == 0
    assert d["tally"]["controls_ok"] is True


def test_unmeasured_items_are_absent_with_reason():
    d = doc()
    for key in (
        "integer_coded_weights",
        "live_artifact_ebpw",
        "geometry_shaped_u8_codebook_n_le_64",
    ):
        item = d["unmeasured"][key]
        assert item["kind"] == "ABSENT", key
        assert item["reason"], key


def test_canary_reuses_existing_accountant_not_a_second_system():
    src = (HERE / "noetic_information_accounting_canary.py").read_text()
    assert "import noetic_information_accounting as accounting" in src
    assert "accounting.account_tree" in src
    assert "accounting.embedded_weightlike_bytes" in src
    assert "def embedded_weightlike_bytes" not in src
    assert "def account_tree" not in src
    assert "def scan_metal_constants" not in src


def test_npy_wrapper_is_a_real_numpy_v1_header():
    payload = accounting.canary_f32_payload(64, b"npy-wrap")
    blob = npy_f32_blob(payload)
    assert blob.startswith(b"\x93NUMPY\x01\x00")
    hlen = int.from_bytes(blob[8:10], "little")
    header = blob[10:10 + hlen]
    assert header.endswith(b"\n")
    assert b"<f4" in header
    assert blob[10 + hlen:] == payload
    assert accounting.embedded_weightlike_bytes(blob) >= len(payload) - 4


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
            print(f"ok  {n}")
    print("passed")
