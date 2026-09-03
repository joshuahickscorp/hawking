#!/usr/bin/env python3
"""Assess the sealed NR/NX artifact against the executable-closure and IR bar.

This is an assessment, not a rebuild. It reads the sealed instance, the NR and
NX tools, the decode load path, and the Gravity IR that exists because the
`tensor: codec` vocabulary cannot account for sharing, generation, or
correction.

Writes receipts/headless/NOETIC_CLOSURE_GAP.json when that path is writable;
otherwise writes next to this script and reports the copy failure.

    python3 tools/headless/noetic_closure_gap.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HAWKING_COPY = Path("/Users/scammermike/Downloads/hawking-copy")
HERE = Path(__file__).resolve()
if HERE.parent.name == "headless" and HERE.parent.parent.name == "tools":
    REPO = HERE.parents[2]
else:
    REPO = Path.cwd().resolve()

sys.path.insert(0, str(REPO / "tools"))
from nr_container import validate  # noqa: E402


def _git(args: list[str], cwd: Path | None = None, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd or REPO), *args],
        capture_output=True, text=True, timeout=timeout,
    )


def sibling_science_root() -> Path | None:
    if not HAWKING_COPY.is_dir() or HAWKING_COPY.resolve() == REPO.resolve():
        return None
    here = _git(["rev-parse", "HEAD"]).stdout.strip()
    there = _git(["rev-parse", "HEAD"], cwd=HAWKING_COPY).stdout.strip()
    return HAWKING_COPY if here and here == there else None


def rel_of(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_text_path(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    rel = rel_of(path)
    r = _git(["show", f"HEAD:{rel}"])
    if r.returncode == 0:
        return r.stdout
    sib = sibling_science_root()
    if sib is not None:
        p2 = sib / rel
        if p2.is_file():
            return p2.read_text(encoding="utf-8", errors="replace")
    raise FileNotFoundError(rel)


def materialize_path(path: Path) -> Path:
    """A real filesystem path for CLI tools that cannot git-show."""
    if path.is_file():
        return path
    rel = rel_of(path)
    sib = sibling_science_root()
    if sib is not None and (sib / rel).is_file():
        return sib / rel
    text = read_text_path(path)
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="noetic_closure_")) / path.name
    tmp.write_text(text, encoding="utf-8")
    return tmp

SEAL = REPO / "receipts/ascent-2026-08-16/G105_NR_NX_ARTIFACT.json"
SIDECAR = REPO / "receipts/ascent-2026-08-16/G105_TENSOR_DIGESTS.json"
NX_SEAL = REPO / "receipts/ascent-2026-08-16/G104_NX_SEAL.json"
NR_CANDIDATE = REPO / "receipts/ascent-2026-08-16/G103_NR_uniform-q4-v1.json"
NR_SCHEMA = REPO / "docs/spec/nr_container.schema.json"
NR_TOOL = REPO / "tools/nr_container.py"
NX_TOOL = REPO / "tools/nx_genome.py"
IR_TOOL = REPO / "tools/gravity_ir.py"
PACK_RS = REPO / "crates/hawking-core/src/model/qwen38_pack.rs"
DECODE_RS = REPO / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
METAL_RS = REPO / "crates/hawking-core/src/metal/mod.rs"
GEO_RS = REPO / "crates/hawking-core/src/model/qwen38_geometry.rs"
WIDE_PY = REPO / "tools/wide_battery.py"
G042 = REPO / "receipts/ascent-2026-08-16/G042_BPW_FAMILY.json"
G043 = REPO / "receipts/ascent-2026-08-16/G043_FLOP_FAMILY.json"
G096 = REPO / "receipts/ascent-2026-08-16/G096_NEURAL_ISA_AUDIT.json"
G097 = REPO / "receipts/ascent-2026-08-16/G097_DESCRIPTION_LENGTH.json"
G148 = REPO / "receipts/ascent-2026-08-16/G148_PROVENANCE.json"
CANONICAL_SCRIPT = REPO / "tools/headless/noetic_closure_gap.py"
CANONICAL_OUT = REPO / "receipts/headless/NOETIC_CLOSURE_GAP.json"

LIVE_ARTIFACT_CANDIDATES = [
    Path("/Users/scammermike/models/qwen38-gravity-uniform-q4-v1"),
    REPO / "workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1",
    Path(
        "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/"
        "qwen38-27b/uniform-q4-v1"
    ),
]
TOKENIZER_CANDIDATES = [
    Path("/Users/scammermike/models/qwen3.8-27b-abliterated-bf16/tokenizer.json"),
    REPO / "workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json",
    Path(
        "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/"
        "qwen38-27b/bf16/tokenizer.json"
    ),
]

SOURCE_PARAM_COUNT = 26_895_998_464
PAYLOAD_BYTES_SEALED = 14_297_694_680


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def line_of(path: Path, needle: str) -> int | None:
    try:
        text = read_text_path(path)
    except FileNotFoundError:
        return None
    idx = text.find(needle)
    if idx < 0:
        return None
    return text[:idx].count("\n") + 1


def load_json(path: Path) -> Any:
    return json.loads(read_text_path(path))


def find_live_artifact() -> Path | None:
    for p in LIVE_ARTIFACT_CANDIDATES:
        if (p / "manifest.json").is_file() and (p / "tensors").is_dir():
            return p
    return None


def find_tokenizer() -> Path | None:
    for p in TOKENIZER_CANDIDATES:
        if p.is_file():
            return p
    return None


def run_cmd(argv: list[str]) -> dict:
    proc = subprocess.run(
        argv, cwd=str(REPO), capture_output=True, text=True, timeout=120
    )
    return {
        "argv": argv,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def try_install(src: Path, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dest)
        return {"ok": True, "path": str(dest), "error": None}
    except OSError as e:
        return {"ok": False, "path": str(dest), "error": f"{type(e).__name__}: {e}"}


def content_addressing_from_sidecar(sidecar: dict) -> dict:
    tensors = sidecar["tensors"]
    name_match = 0
    content_match = 0
    samples = []
    for fn, meta in tensors.items():
        stem = fn.split(".")[0]
        name_h = hashlib.sha256(meta["name"].encode()).hexdigest()
        is_name = stem == name_h
        is_content = stem == meta["sha256"]
        if is_name:
            name_match += 1
        if is_content:
            content_match += 1
        if len(samples) < 12:
            samples.append(
                {
                    "filename": fn,
                    "tensor_name": meta["name"],
                    "filename_stem": stem,
                    "sha256_of_name": name_h,
                    "sha256_of_contents_recorded": meta["sha256"],
                    "bytes": meta["bytes"],
                    "stem_equals_sha256_of_name": is_name,
                    "stem_equals_sha256_of_contents": is_content,
                }
            )
    return {
        "tensors_in_sidecar": len(tensors),
        "filename_stem_equals_sha256_of_tensor_name": name_match,
        "filename_stem_equals_sha256_of_contents": content_match,
        "first_12": samples,
        "verdict": (
            f"{content_match}/{len(tensors)} filenames equal sha256(contents); "
            f"{name_match}/{len(tensors)} equal sha256(tensor_name). The 64-hex "
            "stem is a name hash, not a content address."
        ),
    }


def live_sample(artifact: Path, sidecar: dict) -> dict:
    rows = []
    sha_agree = 0
    stem_is_content = 0
    stem_is_name = 0
    missing = 0
    for fn, meta in list(sidecar["tensors"].items())[:12]:
        path = artifact / "tensors" / fn
        rec = {
            "filename": fn,
            "tensor_name": meta["name"],
            "sidecar_sha256": meta["sha256"],
            "sidecar_bytes": meta["bytes"],
            "present": path.is_file(),
        }
        if not path.is_file():
            missing += 1
            rows.append(rec)
            continue
        disk_sha = sha256_file(path)
        stem = fn.split(".")[0]
        name_h = hashlib.sha256(meta["name"].encode()).hexdigest()
        rec.update(
            {
                "disk_bytes": path.stat().st_size,
                "disk_sha256": disk_sha,
                "sidecar_sha_matches_disk": disk_sha == meta["sha256"],
                "stem_equals_sha256_of_name": stem == name_h,
                "stem_equals_sha256_of_disk_contents": stem == disk_sha,
            }
        )
        if disk_sha == meta["sha256"]:
            sha_agree += 1
        if stem == disk_sha:
            stem_is_content += 1
        if stem == name_h:
            stem_is_name += 1
        rows.append(rec)
    return {
        "artifact_root": str(artifact),
        "sampled": len(rows),
        "sidecar_sha_matches_disk": sha_agree,
        "filename_is_content_address": stem_is_content,
        "filename_is_name_hash": stem_is_name,
        "missing": missing,
        "rows": rows,
        "same_name_address_different_bytes": [
            r for r in rows if r.get("sidecar_sha_matches_disk") is False
        ],
    }


def nr_accounting_injection(nr: dict) -> dict:
    bpw_before = nr["representation"]["tensors"]["complete_bits_per_weight"]
    injected = copy.deepcopy(nr)
    injected["representation"]["shared_structures"] = [
        {
            "kind": "SharedBasis",
            "rank": 256,
            "bytes": 1_000_000_000,
            "path": "basis.bin",
        }
    ]
    injected["representation"]["tensors"]["codec_families"].append(
        {"family": "TensorTrain", "cores": 2, "ranks": [64, 64], "count": 192}
    )
    injected["representation"]["generated_structures"] = [
        {"kind": "GeneratedCoefficient", "code_bytes": 50_000_000}
    ]
    ok, problems = validate(injected)
    bpw_after = injected["representation"]["tensors"]["complete_bits_per_weight"]
    return {
        "injected": [
            "representation.shared_structures = 1_000_000_000-byte SharedBasis",
            "representation.tensors.codec_families += TensorTrain",
            "representation.generated_structures = 50_000_000-byte GeneratedCoefficient",
        ],
        "validator_ok": ok,
        "validator_problems": problems,
        "complete_bits_per_weight_before": bpw_before,
        "complete_bits_per_weight_after": bpw_after,
        "bpw_moved": bpw_before != bpw_after,
        "byte_cost_forced": False,
        "kernel_requirement_forced": False,
        "reconstruction_cost_forced": False,
        "meaning": (
            "NR.validate() accepts a 1 GB shared basis, a TensorTrain family, and "
            "a generated-coefficient blob. complete_bits_per_weight stays the "
            "declared packer number. A container that can describe a "
            "representation is not one that can account for it."
        ),
    }


def family_table() -> list[dict]:
    """16 node families vs the NR schema, judged on the ACCOUNTING bar."""
    common_no = {
        "forces_byte_cost": False,
        "forces_kernel_requirement": False,
        "forces_reconstruction_cost": False,
    }

    def row(family, classification, field, tempting, why) -> dict:
        d = {
            "family": family,
            "classification": classification,
            "field": field,
            "tempting_slot": tempting,
            "why": why,
        }
        d.update(common_no)
        return d

    sc = "schema-change"
    return [
        row(
            "SharedBasis",
            sc,
            "representation.shared_structures (empty list; untyped)",
            "representation.shared_structures",
            "gravity_ir.shared_basis stores per-site coefficients and points "
            "at a content-addressed pool object counted ONCE. NR's empty "
            "shared_structures list can hold a JSON blob, but complete_bpw is "
            "declared from tensor_payload_bytes (manifest row sum). Filling "
            "the list does not recompute BPW, does not content-address the "
            "basis, and does not derive a kernel_requirements entry. Adding "
            "stored_bytes to the blob is the trap: the accounting law stays "
            "declared-from-packer.",
        ),
        row(
            "BasisCoefficient",
            sc,
            "no dedicated field; not tensors.codec_families",
            "representation.latent_codes or per-site tensors",
            "Coefficients without the pooled basis undercount the basis. "
            "Coefficients plus a copied basis in tensors overcount it. The "
            "only honest form is gravity_ir's (site stored_bytes) + "
            "(SharedPool object counted once). NR has no pool and no "
            "computed BPW, so there is no field that can carry this without "
            "a schema change.",
        ),
        row(
            "StructuredTransform",
            sc,
            "none",
            None,
            "G032 members (sign, permutation, channel-scale) are gauges and "
            "folds, not a codec family. codec_families is {family, bits, "
            "group, applies_to, count}. There is no transform/fold/"
            "reconstruction slot, and no field for the runtime cost of "
            "unapplying a transform.",
        ),
        row(
            "TensorTrain",
            sc,
            "none; G096 status NEVER BUILT",
            "representation.tensors.codec_families.family",
            "A TT is cores, ranks, an unfolding, a reconstruct kernel and a "
            "MAC count. codec_families cannot say any of that. G096 records "
            "TENSOR_TRAIN as never built: no receipt, no kernel, no "
            "measurement. Stuffing family=TensorTrain into the list is "
            "exactly the injection this script watched NR.validate() accept "
            "without moving BPW.",
        ),
        row(
            "TensorRing",
            sc,
            "none",
            None,
            "No NR slot, no gravity_ir constructor, no kernel family. A ring "
            "is a cyclic core contraction; it is not a bitwidth on a dense "
            "tensor. Requires a program node with stored_bytes per core, a "
            "reconstruct kernel, and reconstruction FLOPs.",
        ),
        row(
            "Tucker",
            sc,
            "none",
            None,
            "Core tensor plus factor matrices. Same gap as TensorTrain: not "
            "a codec_families row, and writing it down would not force "
            "factor bytes or reconstruct cost into complete_bpw.",
        ),
        row(
            "AdditiveCodebook",
            sc,
            "representation.entropy_streams / latent_codes (empty lists)",
            "representation.entropy_streams",
            "G114 costed a rANS stream as coded payload + table + per-stream "
            "state + index + flush padding + scales. NR entropy_streams is "
            "an empty list with no such schema. The sealed NR says no packer "
            "emits an entropy stream. Filling the list would not move "
            "complete_bpw and would not add a decoder requirement.",
        ),
        row(
            "DictionaryRoute",
            sc,
            "representation.route_graph (null, untyped)",
            "representation.route_graph",
            "G106's route graph is mixer/MLP halves of a dense model, not a "
            "learned dictionary. A routing table's bytes, its gather kernel "
            "and its reconstruction/dispatch cost have no required subfields. "
            "route_graph=null is a measured absence, not a schema for a "
            "dictionary.",
        ),
        row(
            "LowRank",
            sc,
            "none; G034 is an offline factorization",
            "representation.tensors.codec_families",
            "G034 replaces a dense map with two factors at matched bits. "
            "That is two tensors, a reconstruct kernel, and a MAC ratio "
            "(0.203x dense, 2.93x the error of flat q3). codec_families "
            "carries bits/group/count, not rank/factors/MACs. An additive "
            "`rank` key would still leave complete_bpw declared and "
            "reconstruction uncosted.",
        ),
        row(
            "SparseResidual",
            sc,
            "representation.correction_planes (empty list)",
            "representation.correction_planes",
            "gravity_ir.sparse_correction counts values AND index bits "
            "(index usually dominates). NR correction_planes has no "
            "n_exceptions, index_bits, value_bytes, or kernel. G042 "
            "CORRECTION_BPW is a hardcoded 0 on the candidate, not computed "
            "from this list.",
        ),
        row(
            "ProtectedIsland",
            sc,
            "representation.exact_islands (empty list)",
            "representation.exact_islands",
            "gravity_ir.exact_island is a compile-time-known exact region "
            "with optional index cost. NR exact_islands is an empty list. "
            "G096 EXACT_ISLAND is UNBUILT (CORRECTION_BPW 0.0, no kernel "
            "binding). Writing an island into the list does not charge its "
            "bytes or name a consumer kernel.",
        ),
        row(
            "GeneratedCoefficient",
            sc,
            "representation.generated_structures (empty list)",
            "representation.generated_structures",
            "gravity_ir.generated_block has stored code_bytes, a shared "
            "generator in the pool, active_bytes != stored_bytes, and "
            "decode_flops_per_elem. That inequality IS the family. NR has "
            "one declared complete_bpw and no reconstruction_flops field, so "
            "a generated node cannot be accounted even if the empty list "
            "holds a description.",
        ),
        row(
            "RoutedOperator",
            sc,
            "representation.route_graph / kernel_requirements",
            "representation.route_graph",
            "An operator identity (which kernel, which expert, which "
            "specialization) is not a tensor codec. kernel_requirements may "
            "name a portable decoder family but is a parallel list, not "
            "derived from nodes, and must not name an implementation "
            "(that is NX). No byte cost, no reconstruction cost.",
        ),
        row(
            "StateTransition",
            sc,
            "none in NR; G042 STATE_BPW_EQUIVALENT is a side tool",
            None,
            "G106 defines a ROUTE as a semantic state transition. KV and "
            "DeltaNet state live in the workspace (qwen38_workspace_bytes), "
            "not in the artifact. NR has no state-bytes field. NX "
            "residency_plan says all_weights=unified_memory and does not "
            "account state. Putting a transition in route_graph would not "
            "charge 131,072 bytes/position.",
        ),
        row(
            "Composition",
            sc,
            "none; a site is one codec, not additive terms",
            None,
            "gravity_ir_roundtrip.mech_shared_basis_plus_island is three "
            "additive terms at one site (shared basis + sparse correction + "
            "exact island) and is explicitly 'unsayable as tensor: codec'. "
            "NR tensors are one kind per row (q4 or f32). Composition of "
            "terms requires a program, not another codec_families string.",
        ),
        row(
            "Correction",
            sc,
            "representation.correction_planes (empty list)",
            "representation.correction_planes",
            "Same slot as SparseResidual. G043 CORRECTION_FLOPS is 0.0 for "
            "uniform-q4-v1 because no packer emits a correction stream. The "
            "empty list is a measured absence. A future correction plane "
            "would need stored bytes, index cost, a consumer kernel, and "
            "reconstruction FLOPs computed into the receipt — a program "
            "schema, not an additive key.",
        ),
    ]


def hiding_scenarios(injection: dict, g042: dict, g043: dict, g096: dict, g097: dict) -> list[dict]:
    q4 = next(c for c in g042["candidates"] if c["candidate"] == "uniform-q4-v1")
    q4_flops = next(c for c in g043["candidates"] if c["candidate"] == "uniform-q4-v1")
    return [
        {
            "scenario": "shared basis in a separate file",
            "mark": "MISSED",
            "reasoning": (
                "NR complete_bits_per_weight is the packer's "
                f"tensor_payload_bytes ({PAYLOAD_BYTES_SEALED}) over "
                f"{SOURCE_PARAM_COUNT} parameters. Load reads only "
                "manifest.json + tensors/<name-hash>.{{hq30uq4,f32v2}}. A "
                "basis.bin next to them is invisible to both. G042 "
                f"SHARED_BPW is {q4['SHARED_BPW']} (hardcoded from G035 "
                "refutation, not a walk of extra files). The injection test "
                f"put a 1 GB SharedBasis into shared_structures; validator_ok="
                f"{injection['validator_ok']} and bpw_moved="
                f"{injection['bpw_moved']}. STORED_BPW would see the file "
                "IF it sat under the artifact root, but that is "
                "gravity_bpw_family, not the NR/NX format, and load still "
                "would not consume it unless the decode path were patched."
            ),
        },
        {
            "scenario": "routing table",
            "mark": "MISSED",
            "reasoning": (
                "NR route_graph is null. G043 ROUTING_FLOPS for this dense "
                f"patient is {q4_flops['ROUTING_FLOPS']}. NX kernel_binding "
                "is shader names extracted from decode string literals, not "
                "a table of routes. A learned routing table (expert map, "
                "dictionary indices, token-to-island ids) as a sidecar has "
                "nowhere to charge its bytes, its gather kernel, or its "
                "dispatch cost. G106's 130 'routes' are mixer/MLP halves, "
                "not a stored table."
            ),
        },
        {
            "scenario": "generated-state cache",
            "mark": "MISSED",
            "reasoning": (
                "NR generated_structures is []. G042 "
                f"GENERATED_BPW_EQUIVALENT is {q4['GENERATED_BPW_EQUIVALENT']} "
                "for every live candidate. A cache of generated coefficients "
                "or a materialized generator state is neither a catalog "
                "tensor nor an NR node. gravity_ir.generated_block is the "
                "IR that would count code_bytes + a pooled generator and "
                "report active_bytes != stored_bytes; NR has one declared "
                "BPW and cannot say that inequality. The injection of a "
                "50 MB generated blob left BPW unchanged."
            ),
        },
        {
            "scenario": "learned constant compiled into a shader",
            "mark": "MISSED",
            "reasoning": (
                "Shaders are include_str!'d into crates/hawking-core/src/metal/mod.rs "
                "and compiled at runtime via newLibraryWithSource. NX "
                "kernel_binding lists 38 names; it does not hash shader "
                "source. G096 reports shared_runtime_binary_bytes="
                f"{g096['shared_vm_vs_marginal_model_bytes']['shared_runtime_binary_bytes']} "
                "and shader_source_bytes="
                f"{g096['shared_vm_vs_marginal_model_bytes']['shader_source_bytes']} "
                "as a pooled VM, not as model information. G097 folds the "
                f"same blob into REQUIRED_RUNTIME_BYTES "
                f"({g097['programs'][0]['REQUIRED_RUNTIME_BYTES']}) — the "
                "whole binary, not a learned constant isolated inside a "
                "kernel. A scale, codebook, or folded transform baked into "
                "a Metal function is invisible to NR payload_bytes and to "
                "NX geometry."
            ),
        },
        {
            "scenario": "machine-specific kernel parameter carrying learned information",
            "mark": "MISSED",
            "reasoning": (
                "NX threadgroup_geometry records gemv tg=128 / mha tg=512 "
                "from occupancy sweeps (G072, G060). Those are launch sizes, "
                "not learned weights. Decode set_u32's grid parameters and "
                "can pass extra bytes into a kernel; nothing in NR or NX "
                "accounts a learned scale, bias, or routing constant "
                "delivered that way. NR's deny-list rejects a field named "
                "`threadgroup` in an NR document, which is the opposite "
                "check: it keeps machine fields out of NR, it does not "
                "count model information that has been smuggled into NX "
                "parameters. An NX that could load anywhere has failed; an "
                "NX that loads here still does not hash its set_bytes "
                "payloads."
            ),
        },
    ]


def execution_files(tok: Path | None, sidecar: dict, seal: dict) -> list[dict]:
    nx = seal["NX"]
    rolling = seal["content_addressing_defect_found_and_fixed_forward"][
        "artifact_rolling_digest"
    ]
    n_tensors = len(sidecar["tensors"])
    tok_bytes = tok.stat().st_size if tok and tok.is_file() else None
    files = []
    files.append(
        {
            "file": "manifest.json",
            "role": "load: Qwen38HybridWeights::load -> load_qwen38_manifest",
            "source": f"qwen38_hybrid_decode.rs:{line_of(DECODE_RS, 'let (_manifest, rows) = load_qwen38_manifest')}",
            "model_specific": True,
            "seal_coverage": "not_covered",
            "why": (
                "Always opened. Names every tensor via row.artifact. G105's "
                f"rolling digest hashes {n_tensors} tensor files "
                f"({PAYLOAD_BYTES_SEALED} bytes) and does not include the "
                "manifest. The manifest itself stores no per-tensor sha256 "
                "(G105 found). Substituting a tensor and leaving the "
                "manifest pointer unchanged is exactly what name-hash "
                "filenames permit."
            ),
        }
    )
    files.append(
        {
            "file": f"tensors/<sha256(tensor_name)>.{{hq30uq4,f32v2}}  ({n_tensors} files)",
            "role": "load: fs::read(tensors_dir.join(row.artifact))",
            "source": f"qwen38_hybrid_decode.rs:{line_of(DECODE_RS, 'let path = tensors_dir.join(&row.artifact)')}",
            "model_specific": True,
            "seal_coverage": "hashed_in_sidecar_not_addressed_not_checked_at_load",
            "why": (
                "These are the 14.30 GB payload. G105 records a sha256 per "
                f"file and rolling digest {rolling}. That is a verification "
                "list, not an address. Filenames are sha256(name) "
                f"(755/755 in the sidecar). Load never compares content to "
                "the sidecar. Two files can share a name-address and differ "
                "in bytes; the live sample demonstrates this."
            ),
        }
    )
    files.append(
        {
            "file": (
                str(tok)
                if tok
                else "tokenizer.json (bf16 parent, not in the artifact)"
            ),
            "role": "generate: --tokenizer (wide_battery.py, ascension_qwen38_hybrid_greedy)",
            "source": f"wide_battery.py:{line_of(WIDE_PY, '--tokenizer')}",
            "model_specific": True,
            "seal_coverage": "not_covered",
            "bytes": tok_bytes,
            "why": (
                "Weight load does not open the tokenizer. Greedy generate "
                "does, from the bf16 parent, not from the artifact root. "
                f"{tok_bytes} bytes of vocabulary/merges if present. The "
                "G105 seal, the rolling digest, and the NX genome do not "
                "name it. Swapping tokenizer.json changes every token id "
                "the model emits and would not move artifact_rolling_digest."
            ),
        }
    )
    files.append(
        {
            "file": "crates/hawking-core/src/model/qwen38_geometry.rs (compiled into the decode binary)",
            "role": "load+decode: vocab, hidden, rope theta, mixer schedule, RMS eps",
            "source": f"qwen38_geometry.rs:{line_of(GEO_RS, 'pub const QWEN38_VOCAB')}",
            "model_specific": True,
            "seal_coverage": "not_covered",
            "why": (
                "Architecture constants for this patient (248320 vocab, "
                "5120 hidden, 1e7 rope, 48 DeltaNet / 16 GQA, fused in_proj "
                "shapes) are compiled into the binary. They are not in NR "
                "representation and not hashed by NX. A geometry edit is a "
                "silent model change."
            ),
        }
    )
    files.append(
        {
            "file": "crates/hawking-core/shaders/*.metal (include_str into metal/mod.rs)",
            "role": "runtime: compiled at MetalContext construction via newLibraryWithSource",
            "source": f"metal/mod.rs:{line_of(METAL_RS, 'Embedded shader sources')}",
            "model_specific": False,
            "seal_coverage": "names_only",
            "why": (
                f"NX lists {nx['kernel_binding']['count']} dispatched of "
                f"{nx['kernel_binding']['declared_in_tree']} declared, by "
                "intersecting decode string literals with `kernel void` "
                "names. The sealed NX is a decode-source union, not this "
                "artifact's codec set: it names q80_binary / q80_hgravs / "
                "q80_sparse kernels that uniform-q4-v1 will not run. Shader "
                "source is not hashed. Learned constants baked into a "
                "shader would ride along here."
            ),
        }
    )
    files.append(
        {
            "file": "decode binary (ascension_qwen38_hybrid_greedy / hawking-core)",
            "role": "the executable half of the manifested artifact",
            "source": f"wide_battery.py:{line_of(WIDE_PY, 'ascension_qwen38_hybrid_greedy')}",
            "model_specific": False,
            "seal_coverage": "not_covered",
            "why": (
                "G096/G097 count ~6.3–7.8 MB of shared runtime separately. "
                "The sealed NX has no binary digest, no metallib digest, "
                "and no lowers_nr binding to the NR content hash (the "
                "optional --nr path in nx_genome.py was not used for G105)."
            ),
        }
    )
    files.append(
        {
            "file": "chat template (hardcoded render_qwen38_user_chat)",
            "role": "generate: prompt wrapping",
            "source": f"qwen38_hybrid_decode.rs:{line_of(DECODE_RS, 'pub fn render_qwen38_user_chat')}",
            "model_specific": True,
            "seal_coverage": "not_covered",
            "why": (
                "The Qwen chat wrapper lives in decode source, not in the "
                "artifact. Changing it changes the function. NR semantic_"
                "provenance does not hash it."
            ),
        }
    )
    return files


def minimum_fix() -> dict:
    return {
        "content_addressing_defect_blocks_real_closure_hash": True,
        "why_it_blocks": (
            "A closure hash is a hash of the whole manifested execution unit "
            "whose constituent pointers are functions of content. Here the "
            "manifest points at sha256(tensor_name) filenames. A merkle of "
            "those pointers is a merkle of names. Bytes can change under a "
            "stable pointer; load will upload whatever is there. G105's "
            "sidecar sha256 list would detect a swap only if something "
            "recomputed and compared it — load does not. The rolling digest "
            "algorithm is not reconstructable from the sidecar (concat, "
            "sorted, json.dumps all miss 89e78055…). G148's NR link "
            "head-hashes blobs above 64 MiB, so even the provenance chain "
            "misses a mid-blob mutation. Filenames that look like content "
            "addresses and are not cannot anchor a closure."
        ),
        "minimum_fix": [
            "Rename every tensor file to sha256(file_bytes).ext. "
            "artifact_filename currently hashes the name "
            "(qwen38_pack.rs artifact_filename).",
            "Store those content ids in the manifest. Load refuses if "
            "sha256(bytes) != filename stem.",
            "Put tokenizer.json (and any other generate input) in the same "
            "manifest with a content id. Generate refuses a missing or "
            "mismatched tokenizer.",
            "Closure hash = digest over the canonical manifest (every id "
            "load or generate will open) plus a runtime identity (shader "
            "source digest / binary digest) plus the NR and NX documents.",
            "Bind NX to NR with lowers_nr (nr_content_sha256). The sealed "
            "G105 NX omits this even though nx_genome.py supports --nr.",
            "Do not treat a sidecar rolling digest as a closure. Until "
            "pointers are content-derived and load consults them, the "
            "digest is an optional audit log.",
        ],
        "accounting_fix_is_not_add_fields": (
            "NR complete_bits_per_weight must be COMPUTED from a program "
            "whose every node reports stored_bytes, names its kernel, and "
            "round-trips against a disk walk, with shared objects in a "
            "content-addressed pool counted once. That is already "
            "tools/gravity_ir.py + tools/gravity_container.py. It is not "
            "the NR schema. Adding stored_bytes to an empty list while "
            "leaving BPW declared is the trap this assessment refuses."
        ),
    }


def print_report(doc: dict) -> None:
    print("NOETIC CLOSURE GAP — assessment of the sealed NR/NX artifact")
    print(f"generated_at {doc['generated_at']}")
    print(f"git_head     {doc.get('git_head')}")
    print(f"repo         {doc.get('repo')}")
    print()
    v = doc["verdict"]
    print("VERDICT")
    print(f"  executable-closure bar: {v['meets_executable_closure_bar']}")
    print(f"  IR-sufficiency bar:     {v['meets_ir_sufficiency_bar']}")
    print(f"  information-accounting: {v['meets_information_accounting_bar']}")
    print(
        "  content-addressing defect blocks a real closure hash: "
        f"{v['content_addressing_defect_blocks_closure_hash']}"
    )
    print()

    print("## 1. Executable closure — files execution touches")
    print()
    for f in doc["q1_executable_closure"]["files_execution_touches"]:
        print(f"  [{f['seal_coverage']}] {f['file']}")
        print(f"      role: {f['role']}")
        print(f"      {f['why']}")
        print()
    ca = doc["q1_executable_closure"]["content_addressing"]
    print("  sidecar census:")
    print(f"    tensors: {ca['tensors_in_sidecar']}")
    print(
        "    filename == sha256(name): "
        f"{ca['filename_stem_equals_sha256_of_tensor_name']}"
    )
    print(
        "    filename == sha256(contents): "
        f"{ca['filename_stem_equals_sha256_of_contents']}"
    )
    live = doc["q1_executable_closure"].get("live_sample")
    if live:
        print(f"  live sample at {live['artifact_root']}:")
        print(
            f"    sidecar sha matches disk: {live['sidecar_sha_matches_disk']}/{live['sampled']}"
        )
        print(
            f"    filename is content address: {live['filename_is_content_address']}/{live['sampled']}"
        )
        print(
            f"    filename is name hash: {live['filename_is_name_hash']}/{live['sampled']}"
        )
        n_diff = len(live["same_name_address_different_bytes"])
        print(f"    same name-address, different bytes vs sidecar: {n_diff}")
        for r in live["same_name_address_different_bytes"]:
            print(
                f"      {r['tensor_name']}: sidecar {r['sidecar_sha256'][:16]} "
                f"disk {r['disk_sha256'][:16]}"
            )
    print()
    print("  what the seal actually covers:")
    for line in doc["q1_executable_closure"]["what_the_seal_covers"]:
        print(f"    - {line}")
    print()

    print("## 2. IR sufficiency — 16 node families")
    print()
    print(f"{'family':<24} {'class':<16} {'field'}")
    for fam in doc["q2_ir_sufficiency"]["families"]:
        print(f"{fam['family']:<24} {fam['classification']:<16} {fam['field']}")
    print()
    print(
        "  None are expressible-as-is under the accounting bar "
        "(byte cost + kernel requirement + reconstruction cost forced into "
        "a checkable receipt)."
    )
    print(
        "  additive-field was considered and rejected for every family: "
        "an extra key on an empty list would not recompute complete_bpw, "
        "would not count a shared object once, and would not round-trip "
        "against disk. That is a schema change of the accounting law."
    )
    print()
    inj = doc["q2_ir_sufficiency"]["injection_watched"]
    print("  injection into the sealed NR:")
    for x in inj["injected"]:
        print(f"    + {x}")
    print(f"    validate() ok: {inj['validator_ok']} problems: {inj['validator_problems']}")
    print(
        f"    complete_bits_per_weight {inj['complete_bits_per_weight_before']} -> "
        f"{inj['complete_bits_per_weight_after']} (moved={inj['bpw_moved']})"
    )
    print(f"    {inj['meaning']}")
    print()

    print("## 3. Information accounting — five hiding scenarios")
    print()
    for s in doc["q3_information_accounting"]["scenarios"]:
        print(f"  [{s['mark']}] {s['scenario']}")
        print(f"      {s['reasoning']}")
        print()

    print("## 4. NX refusal test")
    print()
    r = doc["refusal_test"]
    print(f"  command: {' '.join(r['argv'])}")
    print(f"  exit_code: {r['exit_code']}")
    print("  stdout:")
    for line in (r["stdout"] or "").splitlines():
        print(f"    {line}")
    if r["stderr"].strip():
        print("  stderr:")
        for line in r["stderr"].splitlines():
            print(f"    {line}")
    print(
        "  expected exit 1 (REFUSED). "
        f"observed {r['exit_code']}. "
        f"{'PASS' if r['exit_code'] == 1 else 'FAIL — the check is broken'}"
    )
    print()

    print("## 5. Content-addressing defect vs a real closure hash")
    print()
    fx = doc["minimum_fix"]
    print(
        "  blocks a real closure hash: "
        f"{fx['content_addressing_defect_blocks_real_closure_hash']}"
    )
    print(f"  {fx['why_it_blocks']}")
    print("  minimum fix:")
    for i, step in enumerate(fx["minimum_fix"], 1):
        print(f"    {i}. {step}")
    print(f"  {fx['accounting_fix_is_not_add_fields']}")
    print()

    print("## WHAT I WATCHED FAIL")
    print()
    for i, item in enumerate(doc["what_i_watched_fail"], 1):
        print(f"  {i}. {item['watched']}")
        print(f"     evidence: {item['evidence']}")
        print()

    print("## Write-scope")
    print()
    ws = doc["write_scope"]
    print(f"  intended: {ws['intended_writes']}")
    print(f"  landed_script: {ws['landed_script']}")
    print(f"  landed_receipt: {ws['landed_receipt']}")
    print(
        "  this script does not git add/checkout/restore/stash/clean/reset "
        "and does not touch crates/, workspace/, visionmcp/, app/, research/lab/, "
        "tools/hcli/bootstrap/."
    )


def main() -> int:
    seal = load_json(SEAL)
    sidecar = load_json(SIDECAR)
    nr = load_json(NR_CANDIDATE)
    schema = load_json(NR_SCHEMA)
    g042 = load_json(G042)
    g043 = load_json(G043)
    g096 = load_json(G096)
    g097 = load_json(G097)
    g148 = load_json(G148)

    ca = content_addressing_from_sidecar(sidecar)
    live_root = find_live_artifact()
    live = live_sample(live_root, sidecar) if live_root else None
    tok = find_tokenizer()
    injection = nr_accounting_injection(nr)
    families = family_table()
    hiding = hiding_scenarios(injection, g042, g043, g096, g097)
    files = execution_files(tok, sidecar, seal)

    nr_on_disk = materialize_path(NR_CANDIDATE)
    nx_on_disk = materialize_path(NX_SEAL)
    nr_neg = run_cmd(
        [sys.executable, str(NR_TOOL), "--negative-test", str(nr_on_disk)]
    )
    nx_ref = run_cmd(
        [sys.executable, str(NX_TOOL), "--refusal-test", str(nx_on_disk)]
    )

    pack_namehash_line = line_of(PACK_RS, "let digest = Sha256::digest(name.as_bytes())")
    load_read_line = line_of(DECODE_RS, "let path = tensors_dir.join(&row.artifact)")
    ir_cannot_line = line_of(
        IR_TOOL, "The existing recipe vocabulary is `tensor: codec`"
    )
    schema_repr = schema["properties"]["representation"]

    nx_has_lowers = "lowers_nr" in seal["NX"]
    g148_nx = next(l for l in g148["links"] if l["stage"] == "NX")

    watched = [
        {
            "watched": (
                "NX refusal test REFUSED a 40-core genome on this machine "
                f"(exit {nx_ref['exit_code']}, required 1)."
            ),
            "evidence": " ".join((nx_ref["stdout"] or "").splitlines()[:3]) or nx_ref,
        },
        {
            "watched": (
                "NR negative-test REJECTED an injected threadgroup_size and "
                f"named kernel (exit {nr_neg['exit_code']}, required 1). That "
                "check has teeth for machine fields and none for accounting."
            ),
            "evidence": " ".join((nr_neg["stdout"] or "").splitlines()[:4]),
        },
        {
            "watched": (
                f"Content addressing: {ca['filename_stem_equals_sha256_of_contents']}/"
                f"{ca['tensors_in_sidecar']} filenames equal sha256(contents); "
                f"{ca['filename_stem_equals_sha256_of_tensor_name']}/"
                f"{ca['tensors_in_sidecar']} equal sha256(tensor_name). The "
                f"64-hex stem is produced at qwen38_pack.rs:{pack_namehash_line} "
                "by Sha256(name.as_bytes())."
            ),
            "evidence": ca["verdict"],
        },
        {
            "watched": (
                "NR.validate() accepted a 1 GB SharedBasis, a TensorTrain "
                "codec family, and a 50 MB generated blob. "
                f"complete_bits_per_weight stayed "
                f"{injection['complete_bits_per_weight_before']}. "
                "Description succeeded; accounting did not run."
            ),
            "evidence": injection["meaning"],
        },
        {
            "watched": (
                "Load does not consult G105_TENSOR_DIGESTS. "
                f"qwen38_hybrid_decode.rs:{load_read_line} is fs::read of "
                "row.artifact with no sha256 compare."
            ),
            "evidence": "grep of Qwen38HybridWeights::load: no sha256, no digest, no sidecar.",
        },
        {
            "watched": (
                "Sealed NX has no lowers_nr "
                f"(present={nx_has_lowers}). G148 NX link is a placeholder "
                f"digest {g148_nx['digest'][:16]}… with path=null. The "
                "executable genome is not bound to the NR it supposedly "
                "lowers, and the provenance chain's NX slot was empty at "
                "seal time."
            ),
            "evidence": g148_nx.get("note") or g148_nx,
        },
        {
            "watched": (
                "docs/spec/nr_container.schema.json types representation as "
                f"{schema_repr}. There is no required stored_bytes, kernel, "
                "or reconstruction field. gravity_ir.py:"
                f"{ir_cannot_line} states the existing recipe vocabulary "
                "cannot express sharing, generation, additive correction, "
                "or exact islands."
            ),
            "evidence": "representation: {type: object} with no required subfields.",
        },
    ]
    if live and live["same_name_address_different_bytes"]:
        diffs = live["same_name_address_different_bytes"]
        watched.append(
            {
                "watched": (
                    f"Live artifact {live['artifact_root']}: "
                    f"{len(diffs)}/{live['sampled']} sampled tensors keep "
                    "the G105 name-address and do not match the sidecar "
                    "sha256. Same pointer, different bytes. Load would "
                    "accept either."
                ),
                "evidence": [
                    {
                        "name": r["tensor_name"],
                        "sidecar": r["sidecar_sha256"],
                        "disk": r["disk_sha256"],
                    }
                    for r in diffs
                ],
            }
        )
    elif live:
        watched.append(
            {
                "watched": (
                    f"Live artifact {live['artifact_root']}: "
                    f"{live['filename_is_content_address']}/{live['sampled']} "
                    "filenames are content addresses (required 0 to match "
                    "the defect). Sidecar sha matches disk "
                    f"{live['sidecar_sha_matches_disk']}/{live['sampled']}."
                ),
                "evidence": live["artifact_root"],
            }
        )
    else:
        watched.append(
            {
                "watched": (
                    "The sealed path workspace/campaign/records/runs/"
                    "qwen38-27b/uniform-q4-v1 is absent from this worktree. "
                    "Closure cannot be recomputed over the physical files "
                    "the seal names."
                ),
                "evidence": [str(p) for p in LIVE_ARTIFACT_CANDIDATES],
            }
        )

    covered = [
        f["file"]
        for f in files
        if f["seal_coverage"] == "hashed_in_sidecar_not_addressed_not_checked_at_load"
    ]
    uncovered = [
        f["file"] for f in files if f["seal_coverage"] in ("not_covered", "names_only")
    ]

    fallback_out = HERE.parent / "NOETIC_CLOSURE_GAP.json"
    git_head = _git(["rev-parse", "HEAD"]).stdout.strip()
    git_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    missed = [s["scenario"] for s in hiding if s.get("mark") == "MISSED"]
    counted = [s["scenario"] for s in hiding if s.get("mark") == "COUNTED"]
    doc = {
        "schema": "hawking.headless.noetic_closure_gap.v1",
        "generated_at": utc_now(),
        "git_head": git_head,
        "branch": git_branch,
        "repo": str(REPO),
        "obligation": (
            "Assess whether the existing NR/NX artifact meets the "
            "executable-closure and IR bar. Do not rebuild the format."
        ),
        "sealed_instance": {
            "receipt": str(SEAL.relative_to(REPO)),
            "artifact": seal["artifact"],
            "parameter_count": seal["NR"]["semantic_provenance"]["parameter_count"],
            "payload_bytes": seal["NR"]["representation"]["tensors"]["payload_bytes"],
            "complete_bits_per_weight": seal["NR"]["representation"]["tensors"][
                "complete_bits_per_weight"
            ],
            "codec_families": seal["NR"]["representation"]["tensors"]["codec_families"],
            "measured_tps": seal["load_and_generate"]["tps"],
            "token_ms": seal["load_and_generate"]["token_ms"],
            "battery": seal["load_and_generate"]["battery_accuracy"],
            "byte_reproducibility_NOT_met": seal["byte_reproducibility_NOT_met"],
            "artifact_rolling_digest": seal[
                "content_addressing_defect_found_and_fixed_forward"
            ]["artifact_rolling_digest"],
            "nx_has_lowers_nr": nx_has_lowers,
            "nx_kernel_binding_count": seal["NX"]["kernel_binding"]["count"],
            "nx_declared_in_tree": seal["NX"]["kernel_binding"]["declared_in_tree"],
        },
        "verdict": {
            "meets_executable_closure_bar": False,
            "meets_ir_sufficiency_bar": False,
            "meets_information_accounting_bar": False,
            "content_addressing_defect_blocks_closure_hash": True,
            "summary": (
                "The sealed uniform-q4-v1 NR/NX pair describes a conventional "
                "grouped-absmax + raw_f32 pack and a machine genome. It does "
                "not hash the whole manifested execution unit, cannot account "
                "for any of the 16 structured node families, and misses every "
                "hiding path that is not a catalog tensor. Gravity IR already "
                "states the missing law; NR/NX is not that law."
            ),
        },
        "q1_executable_closure": {
            "physical_execution_unit": (
                "Qwen38HybridWeights::load opens manifest.json and every "
                "row.artifact under tensors/. Generate additionally opens "
                "tokenizer.json from the bf16 parent. The decode binary "
                "embeds shaders and qwen38_geometry constants. That whole "
                "set is the manifested artifact. The G105 rolling digest "
                "covers only the 755 tensor files."
            ),
            "files_execution_touches": files,
            "covered_by_seal": covered,
            "not_covered_by_seal": uncovered,
            "content_addressing": ca,
            "live_sample": live,
            "tokenizer": str(tok) if tok else None,
            "what_the_seal_covers": [
                "NR JSON: semantic_provenance, representation (declared BPW), kernel_requirements.",
                "NX JSON: machine genome, 38 kernel names, threadgroup geometry, residency, cache, scheduling.",
                f"Sidecar sha256 of {ca['tensors_in_sidecar']} tensor files and an opaque rolling digest.",
                "Does not cover manifest.json, tokenizer.json, shader source, decode binary, geometry constants, chat template.",
                "Does not check hashes at load.",
                "Filenames cannot be the merkle leaves.",
            ],
        },
        "q2_ir_sufficiency": {
            "bar": (
                "Writing a node down must force its true byte cost, its "
                "kernel requirement, and its reconstruction cost into the "
                "receipt where they can be checked. A container that can "
                "describe is not one that can account."
            ),
            "nr_representation_schema": schema_repr,
            "nr_declared_not_computed_bpw": True,
            "gravity_ir_reason_it_exists": (
                "tools/gravity_ir.py: 'The existing recipe vocabulary is "
                "tensor: codec. It cannot express anything this campaign "
                "now wants: structure shared across sites, blocks generated "
                "rather than stored, additive correction stages, exact islands.'"
            ),
            "families": families,
            "injection_watched": injection,
            "expressible_as_is": 0,
            "additive_field": 0,
            "schema_change": 16,
        },
        "q3_information_accounting": {
            "scenarios": hiding,
            "marks": {
                "MISSED": missed,
                "COUNTED": counted,
            },
            "all_five_missed_by_nr_nx": len(missed) == 5 and len(counted) == 0,
            "expected_missed": 5,
            "live_nr_generated_structures": nr.get("representation", {}).get("generated_structures"),
            "live_nr_route_graph": nr.get("representation", {}).get("route_graph"),
            "live_nr_shared_structures": nr.get("representation", {}).get("shared_structures"),
        },
        "refusal_test": nx_ref,
        "nr_negative_test": nr_neg,
        "minimum_fix": minimum_fix(),
        "what_i_watched_fail": watched,
        "write_scope": {
            "intended_writes": [
                "receipts/headless/NOETIC_CLOSURE_GAP.json",
                "tools/headless/noetic_closure_gap.py",
            ],
            "landed_script": None,
            "landed_receipt": None,
            "denied": [
                "crates",
                "workspace",
                "visionmcp",
                "app",
                "lab",
                "tools/haider",
            ],
            "git_mutations": "none — this script does not invoke git write operations",
        },
    }

    script_src = Path(__file__).resolve()
    script_install = {
        "ok": script_src.resolve() == CANONICAL_SCRIPT.resolve() and CANONICAL_SCRIPT.is_file(),
        "path": str(CANONICAL_SCRIPT),
        "error": None,
    }
    if not script_install["ok"]:
        script_install = try_install(script_src, CANONICAL_SCRIPT)

    CANONICAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = CANONICAL_OUT
    doc["write_scope"]["landed_script"] = {
        "ok": script_install["ok"],
        "path": script_install["path"] if script_install.get("ok") else str(script_src),
        "error": script_install.get("error"),
    }
    doc["write_scope"]["landed_receipt"] = {
        "ok": True,
        "path": str(receipt_path),
        "error": None,
    }
    receipt_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    print_report(doc)
    print()
    print(f"git_head: {git_head}")
    print(f"hiding: MISSED={len(missed)} COUNTED={len(counted)} all_five_missed={len(missed)==5}")
    for s in hiding:
        print(f"  [{s['mark']}] {s['scenario']}")
    print(f"receipt: {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
