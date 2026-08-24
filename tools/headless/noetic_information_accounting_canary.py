#!/usr/bin/env python3
"""N023 — information-accounting canary (S020 §40).

A "1 EBPW" model must not secretly carry 3 EBPW in another bucket. This
harness plants known model-specific bytes in every hiding channel the
canon names, then runs the EXISTING accountant
(`noetic_information_accounting.account_tree` + `embedded_weightlike_bytes`).
It does not invent a second detector.

Real channels (PASS requires counted_bytes >= planted_bytes on each):

  shader_constants     large `constant float[]` baked into a .metal
  generated_source     weights as a Python list AND a Rust `[f32; N]`
  cache_sidecar        a .bin sitting next to the host executable
  route_tables         an oversized routing table masquerading as NX metadata
  runtime_mlx_blobs    an MLX-style .npy parked under runtime/
  header18_alignment   18-byte header + f32 payload (the v1 alignment trick)

Causal-law controls (S020 §36):

  negative_random_non_weightlike  high-entropy non-f32 bytes must NOT be
                                  flagged as weights (canary is not "flags everything")
  noop_zero_plant                 zero planted bytes -> zero extra MODEL_SPECIFIC

Integer-coded weights (the N014 named gap) are ABSENT here: this canary does
not paper over that hole by counting every byte.

  python3 tools/headless/noetic_information_accounting_canary.py
  python3 -m pytest tools/headless/test_information_accounting_canary.py -q

Never loads a model. Never writes under ~/noetic/NOETIC_PARENT_A or crates/.
The canary lives in tempfile; the receipt is the only durable write.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import noetic_information_accounting as accounting  # noqa: E402

SCHEMA = "hawking.headless.noetic_information_accounting_canary.v1"
RECEIPT = REPO / "receipts" / "headless" / "NOETIC_INFORMATION_ACCOUNTING_CANARY.json"

REAL_CHANNELS = (
    "shader_constants",
    "generated_source",
    "cache_sidecar",
    "route_tables",
    "runtime_mlx_blobs",
    "header18_alignment",
)
CONTROL_CHANNELS = (
    "negative_random_non_weightlike",
    "noop_zero_plant",
)


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, timeout=20
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "UNKNOWN"


def npy_f32_blob(payload: bytes) -> bytes:
    """NumPy v1 .npy wrapping a little-endian f32 vector — MLX mx.save format."""
    n = len(payload) // 4
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d,), }" % n
    raw = header.encode("ascii")
    # magic(6)+ver(2)+hlen(2) = 10; (10 + header_len) must be 16-aligned.
    hlen_unpadded = len(raw) + 1
    pad = (16 - ((10 + hlen_unpadded) % 16)) % 16
    padded = raw + b" " * pad + b"\n"
    assert (10 + len(padded)) % 16 == 0
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(padded)) + padded + payload


def non_weightlike_entropy(n: int, salt: bytes) -> bytes:
    """High-entropy bytes that are not a plausible f32 weight run.

    canary_f32_payload maps xorshift into [-1, 1] and WILL flag. Raw xorshift
    bytes will not: IEEE-754 interpretation is almost all NaN/Inf/huge.
    """
    return accounting.xorshift64_bytes(accounting.canary_seed(salt), n)


# ---------------------------------------------------------------------------
# plants — each hides parent-derived f32 bytes somewhere the path classifier
# would park as SHARED / MACHINE / GENERATED.
# ---------------------------------------------------------------------------


def plant_shader_constants(root: Path) -> dict[str, Any]:
    n = 256
    payload = accounting.canary_f32_payload(n, b"n023-shader-constants")
    vals = accounting.floats_from_payload(payload)
    body = (
        "// canary: learned constant table baked into a dispatched shader\n"
        f"constant float learned_scale[{n}] = {{\n"
        + accounting.fmt_list(vals)
        + "\n};\n"
        "kernel void canary_scaled(device float *x [[buffer(0)]]) {\n"
        "    x[0] *= learned_scale[0];\n"
        "}\n"
    )
    dest = root / "runtime" / "learned_scale.metal"
    dest.write_text(body)
    return {
        "channel": "shader_constants",
        "payload_bytes": len(payload),
        "n": n,
        "where": "runtime/learned_scale.metal",
        "hide": "large constant float[] baked into a .metal that path_role books SHARED_RUNTIME",
        "blob": None,
    }


def plant_generated_source(root: Path) -> dict[str, Any]:
    n = 64
    py_payload = accounting.canary_f32_payload(n, b"n023-generated-py")
    rs_payload = accounting.canary_f32_payload(n, b"n023-generated-rs")
    py_vals = accounting.floats_from_payload(py_payload)
    rs_vals = accounting.floats_from_payload(rs_payload)
    (root / "runtime" / "baked_bias.py").write_text(
        "LEARNED_BIAS = [\n" + accounting.fmt_list(py_vals) + "\n]\n"
    )
    (root / "runtime" / "baked_bias.rs").write_text(
        f"const LEARNED_BIAS: [f32; {n}] = [\n"
        + accounting.fmt_list(rs_vals)
        + "\n];\n"
    )
    return {
        "channel": "generated_source",
        "payload_bytes": len(py_payload) + len(rs_payload),
        "n": n * 2,
        "where": "runtime/baked_bias.py + runtime/baked_bias.rs",
        "hide": "learned weights emitted as a Python list and a Rust [f32; N] literal",
        "blob": None,
        "parts": {
            "python": len(py_payload),
            "rust": len(rs_payload),
        },
    }


def plant_cache_sidecar(root: Path) -> dict[str, Any]:
    n = 512
    payload = accounting.canary_f32_payload(n, b"n023-cache-sidecar")
    dest = root / "runtime" / "sidecar.bin"
    dest.write_bytes(payload)
    return {
        "channel": "cache_sidecar",
        "payload_bytes": len(payload),
        "n": n,
        "where": "runtime/sidecar.bin",
        "hide": ".bin sitting next to the host executable (runtime/host.py); path_role books SHARED_RUNTIME",
        "blob": payload,
    }


def plant_route_tables(root: Path) -> dict[str, Any]:
    n = 128
    payload = accounting.canary_f32_payload(n, b"n023-route-table")
    vals = accounting.floats_from_payload(payload)
    nx_path = root / "nx" / "machine_genome.json"
    nx = json.loads(nx_path.read_text())
    nx["route_table"] = {
        "kind": "metadata.routing",
        "description": "per-expert mix; definitely not weights",
        "routes": vals,
    }
    nx_path.write_text(json.dumps(nx, indent=2) + "\n")
    return {
        "channel": "route_tables",
        "payload_bytes": len(payload),
        "n": n,
        "where": "nx/machine_genome.json#route_table.routes",
        "hide": "oversized routing table masquerading as NX metadata (MACHINE_SPECIFIC)",
        "blob": None,
    }


def plant_runtime_mlx_blobs(root: Path) -> dict[str, Any]:
    n = 256
    payload = accounting.canary_f32_payload(n, b"n023-mlx-npy")
    blob = npy_f32_blob(payload)
    dest = root / "runtime" / "hidden_weights.npy"
    dest.write_bytes(blob)
    return {
        "channel": "runtime_mlx_blobs",
        "payload_bytes": len(payload),
        "n": n,
        "where": "runtime/hidden_weights.npy",
        "hide": "MLX-style .npy (numpy v1 f32) parked under runtime/ as a 'runtime blob'",
        "blob": blob,
        "file_bytes": len(blob),
    }


def plant_header18_alignment(root: Path) -> dict[str, Any]:
    n = 128
    payload = accounting.canary_f32_payload(n, b"n023-header18")
    header = b"MTLB" + b"\x00\x00\x00\x01" + b"CANARY\x00\x00" + b"\x00\x00"
    assert len(header) == 18, len(header)
    blob = header + payload
    dest = root / "nx" / "qwen_uniform_q4_group64.metallib"
    dest.write_bytes(blob)
    return {
        "channel": "header18_alignment",
        "payload_bytes": len(payload),
        "n": n,
        "where": "nx/qwen_uniform_q4_group64.metallib",
        "hide": "18-byte MTLB header + f32 payload; v1 of the accountant scanned only offset 0",
        "blob": blob,
        "header_bytes": 18,
        "file_bytes": len(blob),
    }


def plant_negative_random(root: Path) -> dict[str, Any]:
    payload = non_weightlike_entropy(4096, b"n023-negative-entropy")
    dest = root / "runtime" / "rand.bin"
    dest.write_bytes(payload)
    return {
        "channel": "negative_random_non_weightlike",
        "payload_bytes": len(payload),
        "n": len(payload),
        "where": "runtime/rand.bin",
        "hide": "high-entropy non-weight-like bytes next to the host; must NOT flag as weights",
        "blob": payload,
    }


def plant_noop(_root: Path) -> dict[str, Any]:
    return {
        "channel": "noop_zero_plant",
        "payload_bytes": 0,
        "n": 0,
        "where": None,
        "hide": "no-op: zero planted bytes must produce zero extra MODEL_SPECIFIC",
        "blob": None,
    }


PLANT_FNS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "shader_constants": plant_shader_constants,
    "generated_source": plant_generated_source,
    "cache_sidecar": plant_cache_sidecar,
    "route_tables": plant_route_tables,
    "runtime_mlx_blobs": plant_runtime_mlx_blobs,
    "header18_alignment": plant_header18_alignment,
    "negative_random_non_weightlike": plant_negative_random,
    "noop_zero_plant": plant_noop,
}


def _delta(before: dict, after: dict) -> dict[str, int]:
    b, a = before["buckets"], after["buckets"]
    return {k: a[k] - b[k] for k in accounting.BUCKETS_7}


def measure_plant(plant_fn: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
    """Isolated plant against write_honest. Closure walk = account_tree."""
    with tempfile.TemporaryDirectory(prefix="n023-canary-") as td:
        root = Path(td)
        accounting.write_honest(root)
        before = accounting.account_tree(root, "content")
        plant = plant_fn(root)
        after = accounting.account_tree(root, "content")
        delta = _delta(before, after)
        planted = int(plant["payload_bytes"])
        counted = int(delta["MODEL_SPECIFIC_BYTES"])
        blob = plant.get("blob")
        if isinstance(blob, (bytes, bytearray)):
            ew = accounting.embedded_weightlike_bytes(bytes(blob))
            ew_field: dict[str, Any] = {
                "kind": "MEASURED",
                "value": ew,
                "command": "noetic_information_accounting.embedded_weightlike_bytes",
            }
        elif planted == 0:
            ew_field = {
                "kind": "ABSENT",
                "value": None,
                "reason": "no blob planted (noop control)",
            }
        else:
            ew_field = {
                "kind": "ABSENT",
                "value": None,
                "reason": (
                    "payload is decimal text (source/JSON), not a binary f32 run; "
                    "counted via account_tree text scanners (scan_metal_constants / "
                    "scan_python_literals / scan_bracket_numeric_arrays / scan_json_payloads)"
                ),
            }
        events = []
        for e in after.get("evidence") or []:
            events.append(e.get("event") or e.get("why") or e.get("file"))
        row = {
            "channel": plant["channel"],
            "hide": plant["hide"],
            "where": plant["where"],
            "planted_bytes": planted,
            "counted_bytes": counted,
            "caught": counted >= planted and planted > 0,
            "delta": delta,
            "before_model_specific": before["buckets"]["MODEL_SPECIFIC_BYTES"],
            "after_model_specific": after["buckets"]["MODEL_SPECIFIC_BYTES"],
            "embedded_weightlike_bytes": ew_field,
            "evidence_events": events,
            "n": plant.get("n"),
            "accountant": {
                "closure_walk": "noetic_information_accounting.account_tree(mode='content')",
                "detector": "noetic_information_accounting.embedded_weightlike_bytes",
            },
        }
        if plant.get("parts"):
            row["parts"] = plant["parts"]
        if plant.get("file_bytes") is not None:
            row["file_bytes"] = plant["file_bytes"]
        return row


def _control_ok(row: dict[str, Any]) -> bool:
    name = row["channel"]
    if name == "noop_zero_plant":
        return row["planted_bytes"] == 0 and row["counted_bytes"] == 0
    if name == "negative_random_non_weightlike":
        # Must not flag as weights. File bytes may sit in SHARED_RUNTIME.
        ew = (row.get("embedded_weightlike_bytes") or {}).get("value") or 0
        return row["counted_bytes"] == 0 and ew == 0
    return False


def run() -> dict[str, Any]:
    channels: dict[str, Any] = {}
    for name in REAL_CHANNELS:
        channels[name] = measure_plant(PLANT_FNS[name])

    controls: dict[str, Any] = {}
    for name in CONTROL_CHANNELS:
        row = measure_plant(PLANT_FNS[name])
        row["caught"] = bool(row["caught"])  # real-channel sense; for controls see ok
        row["ok"] = _control_ok(row)
        row["expected"] = (
            "zero extra MODEL_SPECIFIC"
            if name == "noop_zero_plant"
            else "NOT flagged as weights (canary is not 'flags everything')"
        )
        controls[name] = row

    holes = []
    for name, row in channels.items():
        if not row["caught"]:
            holes.append({
                "channel": name,
                "planted_bytes": row["planted_bytes"],
                "counted_bytes": row["counted_bytes"],
                "where": row["where"],
                "reason": (
                    f"MODEL_SPECIFIC grew by {row['counted_bytes']} B, "
                    f"not the planted {row['planted_bytes']} B"
                ),
            })

    controls_ok = all(c["ok"] for c in controls.values())
    channels_ok = all(c["caught"] for c in channels.values())
    verdict = "PASS" if channels_ok and controls_ok and not holes else "FAIL"

    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git_head(),
        "why": (
            "S020 §40 / canon law 2: EBPW is measured from the full executable "
            "closure. A representation that reports 1 EBPW must not secretly "
            "carry 3 EBPW in shader constants, generated source, caches, route "
            "tables, runtime/MLX blobs, or an alignment-prefixed compiled blob. "
            "This canary plants known bytes in each channel and asserts the "
            "existing accountant counts them. It does not invent a second system."
        ),
        "accountant": {
            "module": "tools/headless/noetic_information_accounting.py",
            "closure_walk": "account_tree(mode='content')",
            "detector": "embedded_weightlike_bytes",
            "invented_second_system": False,
        },
        "did_not_load_model": True,
        "did_not_mutate_parent_a": True,
        "did_not_touch_crates": True,
        "channels": channels,
        "controls": controls,
        "holes": holes,
        "unmeasured": {
            "integer_coded_weights": {
                "kind": "ABSENT",
                "reason": (
                    "named remaining gap of the f32-run detector (N014 "
                    "integer-coded .metallib / packed 2-bit affine codes). "
                    "This canary plants f32 hiding channels, not integer codes. "
                    "Papering over that hole by counting every byte would make "
                    "the negative control fail."
                ),
            },
            "live_artifact_ebpw": {
                "kind": "ABSENT",
                "reason": (
                    "this canary does not load a model. Live seven-bucket "
                    "accounting of uniform-q4-v1 lives in "
                    "NOETIC_INFORMATION_ACCOUNTING.json."
                ),
            },
            "geometry_shaped_u8_codebook_n_le_64": {
                "kind": "ABSENT",
                "reason": (
                    "JSON walker deliberately allows short power-of-two integer "
                    "lists as threadgroup geometry (cap 64). The route_tables "
                    "channel plants an OVERSIZED non-geometry table, which is "
                    "the hide S020 §40 names. The short geometry codebook remains "
                    "the completeness gap recorded by the parent accountant."
                ),
            },
        },
        "NOETIC_INFORMATION_ACCOUNTING": verdict,
        "tally": {
            "real_channels": len(REAL_CHANNELS),
            "real_caught": sum(1 for c in channels.values() if c["caught"]),
            "real_missed": sum(1 for c in channels.values() if not c["caught"]),
            "controls_ok": controls_ok,
        },
    }


def write_receipt(doc: dict[str, Any] | None = None) -> dict[str, Any]:
    if doc is None:
        doc = run()
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def print_report(doc: dict[str, Any]) -> None:
    print("=== NOETIC INFORMATION ACCOUNTING CANARY (N023) ===")
    print(f"verdict  NOETIC_INFORMATION_ACCOUNTING = {doc['NOETIC_INFORMATION_ACCOUNTING']}")
    print()
    print("real channels (caught iff counted_bytes >= planted_bytes):")
    for name in REAL_CHANNELS:
        row = doc["channels"][name]
        flag = "CAUGHT" if row["caught"] else "HOLE  "
        print(
            f"  {flag}  {name:<24} planted={row['planted_bytes']:<8} "
            f"counted={row['counted_bytes']:<8}  {row['where']}"
        )
        if not row["caught"]:
            print(f"         {row['hide']}")
    print()
    print("controls:")
    for name in CONTROL_CHANNELS:
        row = doc["controls"][name]
        flag = "OK    " if row["ok"] else "FAIL  "
        print(
            f"  {flag}  {name:<32} planted={row['planted_bytes']:<8} "
            f"counted={row['counted_bytes']:<8}  {row['expected']}"
        )
    print()
    if doc["holes"]:
        print("HOLES:")
        for h in doc["holes"]:
            print(f"  {h['channel']}: {h['reason']}")
        print()
    print("unmeasured:")
    for k, v in doc["unmeasured"].items():
        print(f"  {k}: ABSENT — {v['reason'][:90]}...")
    print()
    print(f"-> {RECEIPT}")


def main() -> int:
    doc = write_receipt()
    print_report(doc)
    return 0 if doc["NOETIC_INFORMATION_ACCOUNTING"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
