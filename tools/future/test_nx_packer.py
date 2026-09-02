"""Negative controls for the generic NX packer.

A renamed source pointer, a placeholder organ, a missing metallib, or a
manifest whose byte total disagrees with the parts must raise. Packing then
making the source unreadable must still leave a complete runtime description.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import struct
from pathlib import Path

import pytest

from tools.future import nx_packer as nxp
from tools.future import nr_nx_generic as nng
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, write_receipt


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_safetensors(path: Path, tensors: dict[str, bytes], shapes: dict[str, list[int]]) -> None:
    header: dict[str, object] = {}
    body = bytearray()
    off = 0
    for name, raw in tensors.items():
        header[name] = {
            "dtype": "F32",
            "shape": shapes[name],
            "data_offsets": [off, off + len(raw)],
        }
        body.extend(raw)
        off += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(body))


def _tiny_specimen(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    weights = (b"\x00\x00\x80\x3f" * 8)  # 8 f32 ones
    _write_safetensors(
        root / "model.safetensors",
        {"model.embed_tokens.weight": weights},
        {"model.embed_tokens.weight": [2, 4]},
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "vocab_size": 2,
            },
            indent=2,
        )
        + "\n"
    )
    (root / "tokenizer.json").write_text('{"model":{"vocab":{"a":0,"b":1}}}\n')
    (root / "tokenizer_config.json").write_text('{"tokenizer_class":"X"}\n')
    (root / "generation_config.json").write_text('{"max_new_tokens":1}\n')
    return root


def _archive_bytes(tag: str) -> bytes:
    # Must not look like Metal source.
    return b"MTLARCH\0" + tag.encode("utf-8") + bytes(range(64))


def _compiled_slot(organ: str, raw: bytes, *, entry: str = "organ_embed_gather") -> dict:
    digest = _sha(raw)
    src_hash = _sha(b"not-the-archive")
    return {
        "organ": organ,
        "status": nxp.COMPILED,
        "occupying": {"kind": nxp.COMPILED, "compiled_kernel": entry},
        "compiled_identity": {
            "kind": nxp.COMPILED_IDENTITY_KIND,
            "shader_hash": digest,
            "value": digest,
            "source_sha256": src_hash,
            "entry_point": entry,
            "archive_bytes": len(raw),
            "unit": "mtl_binary_archive_sha256",
            "pipeline": {
                "object": nxp.PIPELINE_OBJECT,
                "created": True,
                "function_found": True,
                "created_command_queue": False,
                "dispatched": False,
            },
        },
        "entry_point": entry,
        "shader_hash": digest,
    }


def _fragment(slots: list[dict]) -> dict:
    compiled = [s for s in slots if s.get("status") == nxp.COMPILED]
    return {
        "status": "COMPILED_KERNELS_NOT_PACKED",
        "source_independent": False,
        "serialized_artifact": None,
        "physical_program": {
            "kernels": slots,
            "n_compiled": len(compiled),
            "n_planned": len(slots) - len(compiled),
        },
        "native_kernel": {
            "status": "BOUND" if compiled else "ABSENT",
            "n_compiled": len(compiled),
            "organs": [s.get("organ") for s in compiled],
        },
        "compiled_organs": [s.get("organ") for s in compiled],
        "planned_organs": [s.get("organ") for s in slots if s.get("status") != nxp.COMPILED],
    }


def test_pack_then_source_unreadable_still_describes_runtime(tmp_path):
    src = _tiny_specimen(tmp_path / "src")
    dest = tmp_path / "nx"
    raw = _archive_bytes("embed")
    slot = _compiled_slot("embed", raw)
    packed = nxp.pack(
        nx_fragment=_fragment([slot]),
        specimen_path=src,
        dest=dest,
        specimen_id="tiny",
        family="dense_swiglu_transformer",
        archives={"embed": raw},
        dcomp=None,
    )
    assert packed["ok"] is True
    nx = packed["nx"]
    assert nx["source_independent"] is True
    assert nx["serialized_artifact"]["self_contained"] is True
    assert nx["byte_ledger"]["reconciles"] is True
    assert nx["byte_ledger"]["total_bytes"] == nx["byte_ledger"]["parts_sum_bytes"]
    needs = {r["need"]: r for r in nx["runtime_needs"]}
    assert needs["representation"]["present"] is True
    assert needs["model_specific_code"]["present"] is True
    assert needs["metadata"]["present"] is True
    assert needs["tables"]["present"] is True

    judged = nng.source_independence(nx, source_trees=[str(src)])
    assert judged["ok"] is True

    for p in src.rglob("*"):
        if p.is_file():
            p.chmod(0)
    src.chmod(0)
    try:
        with pytest.raises(PermissionError):
            (src / "model.safetensors").read_bytes()
        man = json.loads((dest / "MANIFEST.json").read_text())
        assert man["billing"]["reconciles"] is True
        assert man["billing"]["total_bytes"] == sum(p["bytes"] for p in man["parts"])
        for part in man["parts"]:
            if part["bytes"] <= 0:
                continue
            body = (dest / part["path"]).read_bytes()
            assert len(body) == part["bytes"]
            assert _sha(body) == part["sha256"]
        nx2 = json.loads((dest / "packed.nx.json").read_text())
        assert nx2["source_independent"] is True
        for need in nx2["runtime_needs"]:
            if need["need"] in {"representation", "model_specific_code", "metadata", "tables"}:
                assert need["present"] is True
                assert need["bytes"] > 0
        judged2 = nng.source_independence(nx2, source_trees=[str(src)])
        assert judged2["ok"] is True
    finally:
        src.chmod(0o755)
        for p in src.rglob("*"):
            if p.is_file():
                p.chmod(0o644)


def test_renamed_source_pointer_is_refused(tmp_path):
    src = _tiny_specimen(tmp_path / "src")
    raw = _archive_bytes("embed")
    frag = _fragment([_compiled_slot("embed", raw)])
    with pytest.raises(nxp.RenamedSourcePointer):
        nxp.pack(
            nx_fragment=frag,
            specimen_path=src,
            dest=src,
            archives={"embed": raw},
            dcomp=None,
        )
    with pytest.raises(nxp.RenamedSourcePointer):
        nxp.pack(
            nx_fragment=frag,
            specimen_path=src,
            dest=src / "inside",
            archives={"embed": raw},
            dcomp=None,
        )
    with pytest.raises(nxp.RenamedSourcePointer):
        nxp.pack(
            nx_fragment=frag,
            specimen_path=src,
            dest=tmp_path / "model.safetensors",
            archives={"embed": raw},
            dcomp=None,
        )
    with pytest.raises(nxp.RenamedSourcePointer):
        nxp.refuse_renamed_source_pointer(
            dest=tmp_path / "packed.nx.json",
            source_tree=src,
            serialized_path=str(src / "model.safetensors"),
        )


def test_placeholder_organ_raises(tmp_path):
    src = _tiny_specimen(tmp_path / "src")
    raw = _archive_bytes("embed")
    slot = _compiled_slot("embed", raw)
    slot["compiled_identity"]["kind"] = "PLACEHOLDER"
    with pytest.raises(nxp.PlaceholderOrgan):
        nxp.pack(
            nx_fragment=_fragment([slot]),
            specimen_path=src,
            dest=tmp_path / "nx",
            archives={"embed": raw},
            dcomp=None,
        )
    slot2 = _compiled_slot("embed", raw)
    slot2["compiled_identity"]["shader_hash"] = slot2["compiled_identity"]["source_sha256"]
    slot2["compiled_identity"]["value"] = slot2["compiled_identity"]["source_sha256"]
    with pytest.raises(nxp.PlaceholderOrgan):
        nxp.pack(
            nx_fragment=_fragment([slot2]),
            specimen_path=src,
            dest=tmp_path / "nx2",
            archives={"embed": raw},
            dcomp=None,
        )
    src_as_archive = b"kernel void organ_embed_gather() { }"
    slot3 = _compiled_slot("embed", _archive_bytes("x"))
    with pytest.raises(nxp.PlaceholderOrgan):
        nxp.pack(
            nx_fragment=_fragment([slot3]),
            specimen_path=src,
            dest=tmp_path / "nx3",
            archives={"embed": src_as_archive},
            dcomp=None,
        )


def test_missing_metallib_raises(tmp_path):
    src = _tiny_specimen(tmp_path / "src")
    raw = _archive_bytes("embed")
    slot = _compiled_slot("embed", raw)
    slot["compiled_identity"].pop("archive_path", None)
    with pytest.raises(nxp.MissingMetallib):
        nxp.pack(
            nx_fragment=_fragment([slot]),
            specimen_path=src,
            dest=tmp_path / "nx",
            archives=None,
            dcomp=None,
        )
    with pytest.raises(nxp.MissingMetallib):
        nxp.pack(
            nx_fragment=_fragment(
                [
                    {
                        "organ": "embed",
                        "status": nxp.NATIVE_UNMEASURED,
                        "compiled_identity": None,
                    }
                ]
            ),
            specimen_path=src,
            dest=tmp_path / "nx2",
            dcomp=None,
        )


def test_billing_mismatch_raises():
    parts = [
        {
            "id": "parts/representation/model.safetensors",
            "path": "parts/representation/model.safetensors",
            "category": "representation",
            "bytes": 10,
            "sha256": "ab" * 32,
            "runtime_required": True,
        }
    ]
    with pytest.raises(nxp.BillingMismatch):
        nxp.reconcile_manifest(parts, claimed_total=9)
    with pytest.raises(nxp.BillingMismatch):
        nxp.reconcile_manifest(
            [
                {
                    "id": "x",
                    "category": "representation",
                    "bytes": 4,
                    "sha256": None,
                    "runtime_required": True,
                }
            ]
        )
    ok = nxp.reconcile_manifest(parts, claimed_total=10)
    assert ok["billing"]["reconciles"] is True
    assert ok["billing"]["total_bytes"] == 10
    assert ok["billing"]["parts_sum_bytes"] == 10


def test_manifest_total_reconciles_with_parts(tmp_path):
    src = _tiny_specimen(tmp_path / "src")
    dest = tmp_path / "nx"
    raw = _archive_bytes("embed")
    packed = nxp.pack(
        nx_fragment=_fragment([_compiled_slot("embed", raw)]),
        specimen_path=src,
        dest=dest,
        specimen_id="tiny",
        archives={"embed": raw},
        dcomp=None,
    )
    billing = packed["nx"]["byte_ledger"]
    assert billing["total_bytes"] == billing["parts_sum_bytes"]
    assert billing["reconciles"] is True
    summed = sum(p["bytes"] for p in packed["manifest"]["parts"])
    assert summed == billing["total_bytes"]
    for cat in nxp.BILLING_CATEGORIES:
        assert cat in billing["categories"]
    assert billing["categories"]["representation"]["bytes"] > 0
    assert billing["categories"]["model_specific_code"]["bytes"] > 0
    assert packed["identity"]["did_not_hardlink"] is True


def test_nr_is_billed_when_referenced(tmp_path):
    src = _tiny_specimen(tmp_path / "src")
    dest = tmp_path / "nx"
    raw = _archive_bytes("embed")
    nr_path = tmp_path / "nr.bin"
    nr_path.write_bytes(b"NRPAYLOAD")
    packed = nxp.pack(
        nx_fragment=_fragment([_compiled_slot("embed", raw)]),
        specimen_path=src,
        dest=dest,
        archives={"embed": raw},
        nr=nr_path,
        dcomp=None,
    )
    assert packed["identity"]["nr_runtime_referenced"] is True
    assert packed["nx"]["byte_ledger"]["categories"]["nr"]["bytes"] == 9
    assert (dest / "parts" / "nr" / "nr.bin").read_bytes() == b"NRPAYLOAD"


def test_build_emits_sealed_static_receipt():
    out = nxp.build()
    assert out.name == "NX_PACKER.json"
    assert out.parent == RECEIPTS
    doc = json.loads(out.read_text())
    assert doc["schema"] == nxp.SCHEMA
    assert doc["packer_callable"] is True
    assert doc["gpu_authority"] is False
    assert doc["did_not_fork_qwen38_special_case"] is True
    assert "RenamedSourcePointer" in "".join(doc["refusals"])
    assert "PlaceholderOrgan" in "".join(doc["refusals"])
    assert "MissingMetallib" in "".join(doc["refusals"])
    assert "BillingMismatch" in "".join(doc["refusals"])
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]


def test_receipt_refuses_hardware_fields():
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "_nx_packer_hw_probe.json",
            {"schema": nxp.SCHEMA, "tps": 12.0},
            nxp.RECORDED_BY,
        )


def test_no_pytest_skip_in_this_file():
    src = Path(__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            assert name != "skip", "pytest.skip that actually fires is a P0"


def test_concurrent_writers_do_not_share_one_staging_name(tmp_path):
    """Two processes packing the same content-addressed part must not collide.

    The staging sibling used to be a fixed `<dest>.tmp`. Two workers writing
    the same artifact both created that one path, and whichever `os.replace`d
    it first left the other raising FileNotFoundError on a name that no longer
    existed. The destination is a content hash, so a concurrent write is
    legitimate; only the staging name has to be private to the writer.
    """
    dest = tmp_path / "embed.deadbeef.mtlarchive"

    # The real race, driven for real: fork a second writer that stages and
    # replaces the same destination while this one does. Both must succeed.
    payload = b"\xcb\xfe\xba\xbe" + b"nx" * 4096
    child = os.fork()
    if child == 0:  # pragma: no cover - the child exits, never reports
        try:
            for _ in range(200):
                nxp._write_and_hash(dest, payload)
        except BaseException:
            os._exit(1)
        os._exit(0)
    try:
        for _ in range(200):
            size, digest = nxp._write_and_hash(dest, payload)
            assert size == len(payload)
            assert digest == hashlib.sha256(payload).hexdigest()
    finally:
        _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0, "the concurrent writer failed"
    assert dest.read_bytes() == payload
    assert not list(tmp_path.glob("*.tmp")), "a staging file was left behind"
    assert str(os.getpid()) in nxp._tmp_sibling(dest).name
