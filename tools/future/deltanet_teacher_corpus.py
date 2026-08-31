#!/usr/bin/env python3
"""DELTANET TEACHER CORPUS — real (pre-S, x, post-S, h) for a generator fit.

generated_transition_coefficients is a program that emits qkvz activations
from a skinny map, then runs the incumbent rearrange + gated-delta. That
question is only well-posed against real input and a real recurrent-state
trajectory. This module captures that corpus the same way the MLP teacher
corpus was captured: prompt-split, refusal on synthetic rows and held-out
leakage, real activations, no Gaussian proxy (NNS-001).

Guards are imported from tools.future.mlp_teacher_corpus rather than
rewritten weaker. emit_manifest is that module's emit_manifest.

    python3 tools/future/deltanet_teacher_corpus.py --build
    python3 tools/future/deltanet_teacher_corpus.py --capture
    python3 -m pytest tools/future/test_deltanet_generated_transition.py -q

X is capture_diverse2 post_attn_norm (the named Qwen3.8 teacher under
NNS-006). It is a real activation, not a mixer-input identity: the JSONL
resident does not dump hidden states, and this module refuses to substitute
Gaussian X. Y = W_qkvz x is reconstructed from the sealed-3.14 HQ30UQ4
tensor the resident fused in-proj consumes. S is rolled from zero along
each prompt with the incumbent rearrange + gated-delta. evidence_class
DIAGNOSTIC_RELATIVE.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.future._common import REPO, git, load_json, sha256_file
from tools.future.mlp_byte_census import CatalogAbsent, resolve_artifact_root
from tools.future.mlp_teacher_corpus import (
    CAPABILITY_DOMAINS,
    CAPTURE_PROMPTS,
    DUP_RATE_MAX,
    FAMILY_TO_DOMAIN,
    FUSION_ENV,
    HOLD_FRAC,
    SYNTHETIC_KINDS,
    CaptureUnavailable,
    CorpusInadequate,
    CorpusRefused,
    annotate_row,
    assign_split,
    emit_manifest as mlp_emit_manifest,
    expand_capture_rows,
    is_synthetic_row,
    load_x_f16,
    load_x_manifest,
    make_fixture_row as mlp_make_fixture_row,
    make_gaussian_row as mlp_make_gaussian_row,
    position_band,
    resolve_resident_binary,
    resolve_x_capture_dir,
    specimen_identity as mlp_specimen_identity,
    split_by_prompt,
    vector_sha256,
    write_f32,
)


RECEIPT = "DELTANET_TEACHER_CORPUS.json"
SCHEMA = "hawking.future.deltanet_teacher_corpus.v1"
VERSION = 1
RECORDED_BY = "tools/future/deltanet_teacher_corpus.py"

HIDDEN = 5120
QKVZ_ROWS = 16384
N_DN_LAYERS = 48
N_LAYERS = 64
# First DN, a mid-depth DN used as the QKVZ-precision typical, a late DN.
# Layer 0 is included as first, never as "typical".
DN_REPRESENTATIVE_LAYERS: tuple[int, ...] = (0, 21, 42)
DN_LAYER_ROLES: dict[int, str] = {
    0: "first_deltanet",
    21: "typical",
    42: "late_deltanet",
}

PAYLOAD_DIR = REPO / "workspace" / "ops" / "local" / "scratch" / "deltanet_teacher_corpus"

CLAIM_BOUNDARY = (
    "DIAGNOSTIC_RELATIVE sidecar. X is real post_attn_norm from teacher-forced "
    "prefill (capture_diverse2; named Qwen3.8 teacher under NNS-006). It is not "
    "claimed to be the sealed mixer input (post_input_norm): the JSONL resident "
    "does not dump hidden states, and this module refuses to substitute Gaussian "
    "X (NNS-001). Y = W_qkvz x is reconstructed from the sealed-3.14 HQ30UQ4 "
    "linear_qkvz the resident fused in-proj consumes. pre-state / post-state / h "
    "are the incumbent rearrange + gated-delta rolled from S=0 along each prompt. "
    "capture_elapsed_s is process elapsed, not a GPU-lease measurement. "
    "bench.gpu_authority is false."
)


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def is_linear_attention_layer(layer: int) -> bool:
    """Sealed-3.14 interval-4 pattern: full attention on layers 3,7,...,63."""
    return int(layer) % 4 != 3


def pick_representatives() -> dict[str, Any]:
    linear = [i for i in range(N_LAYERS) if is_linear_attention_layer(i)]
    if len(linear) != N_DN_LAYERS:
        raise CaptureUnavailable(
            f"REFUSED: linear_attention count {len(linear)} != {N_DN_LAYERS}"
        )
    chosen = []
    for layer in DN_REPRESENTATIVE_LAYERS:
        if layer not in linear:
            raise CaptureUnavailable(
                f"REFUSED: representative layer {layer} is not linear_attention"
            )
        role = DN_LAYER_ROLES[layer]
        chosen.append(
            {
                "layer": int(layer),
                "role": role,
                "mixer": "linear_attention",
                "typical": role == "typical",
                "why": {
                    "first_deltanet": "First DeltaNet layer. Included as the depth-axis start, not as typical.",
                    "typical": "Mid-depth DeltaNet (also a QKVZ-precision sensitivity layer). Layer 0 is not typical.",
                    "late_deltanet": "Late DeltaNet, QKVZ-precision sensitivity layer 42.",
                }[role],
            }
        )
    return {
        "n_linear_attention": len(linear),
        "n_full_attention": N_LAYERS - len(linear),
        "linear_layers": linear,
        "layer0_typical": False,
        "chosen": chosen,
        "chosen_layers": [p["layer"] for p in chosen],
    }


def specimen_identity() -> dict[str, Any]:
    spec = mlp_specimen_identity()
    spec["organ"] = "attention.linear_qkvz"
    spec["fusion_env"] = dict(FUSION_ENV)
    spec["geometry"] = {
        "hidden_size": HIDDEN,
        "qkvz_rows": QKVZ_ROWS,
        "n_deltanet_layers": N_DN_LAYERS,
        "num_hidden_layers": N_LAYERS,
    }
    return spec


def dn_content_sha256_of(row: Mapping[str, Any]) -> str:
    """Identity of captured DN content. row_id / provenance excluded so copies collide."""
    identity = {
        "layer": int(row.get("layer", -1)),
        "prompt_id": str(row.get("prompt_id") or ""),
        "token_position": int(row.get("token_position", -1)),
        "x_sha256": row.get("x_sha256") or row.get("input_sha256"),
        "y_sha256": row.get("y_sha256"),
        "pre_state_sha256": row.get("pre_state_sha256"),
        "post_state_sha256": row.get("post_state_sha256"),
        "output_sha256": row.get("output_sha256"),
    }
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(blob.encode("utf-8"))


def annotate_dn_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "x_sha256" not in out and out.get("input_sha256"):
        out["x_sha256"] = out["input_sha256"]
    if "y_sha256" not in out and out.get("output_sha256"):
        out["y_sha256"] = out["output_sha256"]
    if "input_sha256" not in out and out.get("x_sha256"):
        out["input_sha256"] = out["x_sha256"]
    out["synthetic"] = bool(is_synthetic_row(out))
    out["content_sha256"] = dn_content_sha256_of(out)
    body = {k: v for k, v in out.items() if k not in {"content_sha256", "envelope_sha256"}}
    out["envelope_sha256"] = _sha256_bytes(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    return out


def _require_dn_fields(rows: Sequence[Mapping[str, Any]]) -> None:
    missing: list[str] = []
    for i, row in enumerate(rows):
        for key in (
            "pre_state_sha256",
            "post_state_sha256",
            "output_sha256",
            "prompt_id",
            "token_position",
            "capability_domain",
            "layer",
        ):
            if row.get(key) in (None, ""):
                missing.append(f"row[{i}].{key}")
                if len(missing) >= 8:
                    break
        if len(missing) >= 8:
            break
    if missing:
        raise CorpusRefused(
            f"REFUSED: DeltaNet row missing pre-state/input/post-state/output fields ({missing[:8]})",
            {
                "accepted": False,
                "refusals": ["MISSING_DN_TRAJECTORY_FIELDS"],
                "missing": missing[:12],
            },
        )


def emit_manifest(
    rows: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    *,
    min_train_rows: int | None = None,
    allow_fixture: bool = False,
    require_sizing: bool = False,
) -> dict[str, Any]:
    """Same leak / synthetic / dup refusals as the MLP corpus. Loud exceptions."""
    _require_dn_fields(rows)
    mapped: list[dict[str, Any]] = []
    for row in rows:
        item = annotate_dn_row(row)
        mapped.append(item)
    # The MLP emit is the guard. Do not replace it with a flag.
    result = mlp_emit_manifest(
        mapped,
        split,
        min_train_rows=min_train_rows,
        allow_fixture=allow_fixture,
        require_sizing=require_sizing,
    )
    result["schema"] = "hawking.future.deltanet_teacher_corpus.manifest.v1"
    result["organ"] = "deltanet"
    result["trajectory_fields"] = (
        "pre_state_sha256",
        "input_sha256",
        "post_state_sha256",
        "output_sha256",
        "token_position",
        "prompt_id",
        "capability_domain",
    )
    return result


def provenance_captured(*, source_path: str, source_sha256: str | None, layer: int) -> dict[str, Any]:
    return {
        "kind": "captured",
        "authority": "DIAGNOSTIC_RELATIVE",
        "source_path": source_path,
        "source_sha256": source_sha256,
        "capture_tool": RECORDED_BY,
        "layer": int(layer),
        "x_kind": "post_attn_norm",
        "y_kind": "sealed_hq30uq4_qkvz_then_incumbent_gated_delta",
        "not": ["gaussian", "gaussian_proxy", "synthetic", "position_0_only"],
        "scars": ["NNS-001", "NNS-007", "DELTANET_QKVZ_PRECISION"],
    }


def _fixture_provenance() -> dict[str, Any]:
    return {
        "kind": "fixture",
        "authority": "STATIC_ONLY",
        "source_path": "tools/future/deltanet_teacher_corpus.py::fixture",
        "source_sha256": _sha256_bytes(b"deltanet-teacher-corpus-fixture-v1"),
        "capture_tool": RECORDED_BY,
        "note": "deterministic fixture; not a promotion and not a Gaussian proxy",
    }


def make_fixture_row(
    *,
    row_id: str,
    layer: int,
    prompt_id: str,
    prompt_text: str,
    token_position: int,
    seq_len: int,
    capability_domain: str,
    payload_seed: bytes,
    provenance: Mapping[str, Any] | None = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    base = mlp_make_fixture_row(
        row_id=row_id,
        layer=layer,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        token_position=token_position,
        seq_len=seq_len,
        capability_domain=capability_domain,
        payload_seed=payload_seed,
        provenance=provenance or _fixture_provenance(),
        synthetic=synthetic,
    )
    seed = payload_seed
    pre = hashlib.sha256(seed + b"|pre_state").hexdigest()
    post = hashlib.sha256(seed + b"|post_state").hexdigest()
    base["input_sha256"] = base["x_sha256"]
    base["pre_state_sha256"] = pre
    base["post_state_sha256"] = post
    base["output_sha256"] = base["y_sha256"]
    base["pre_state_frob"] = float((int(pre[:8], 16) % 10000) + 1)
    base["post_state_frob"] = float((int(post[:8], 16) % 10000) + 1)
    return annotate_dn_row(base)


def make_diverse_fixture_corpus(
    n_prompts_per_domain: int = 4, positions: int = 3
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0
    layers = DN_REPRESENTATIVE_LAYERS
    for domain in CAPABILITY_DOMAINS:
        domain_prompts = [p for p in CAPTURE_PROMPTS if p[0] == domain]
        for p_i in range(n_prompts_per_domain):
            text = domain_prompts[p_i % len(domain_prompts)][1]
            prompt_id = f"{domain}:p{p_i:02d}"
            seq_len = max(positions, 6)
            pos_list = [0, seq_len // 2, seq_len - 1][:positions]
            for pos in pos_list:
                layer = layers[idx % len(layers)]
                rows.append(
                    make_fixture_row(
                        row_id=f"dn-fx-{idx:04d}",
                        layer=layer,
                        prompt_id=prompt_id,
                        prompt_text=text,
                        token_position=pos,
                        seq_len=seq_len,
                        capability_domain=domain,
                        payload_seed=f"{prompt_id}|{pos}|{layer}|dn".encode(),
                    )
                )
                idx += 1
    return rows


def make_gaussian_row(template: Mapping[str, Any]) -> dict[str, Any]:
    g = mlp_make_gaussian_row(template)
    g["input_sha256"] = g["x_sha256"]
    g["output_sha256"] = g.get("y_sha256") or g["x_sha256"]
    g["pre_state_sha256"] = hashlib.sha256(b"gaussian-pre").hexdigest()
    g["post_state_sha256"] = hashlib.sha256(b"gaussian-post").hexdigest()
    g["x_generator"] = "gaussian"
    return annotate_dn_row(g)


def capture_dir_complete(payload_dir: Path | None = None) -> bool:
    root = payload_dir if payload_dir is not None else PAYLOAD_DIR
    return (root / "CAPTURE.json").is_file()


def load_existing_capture(payload_dir: Path | None = None) -> dict[str, Any] | None:
    root = payload_dir if payload_dir is not None else PAYLOAD_DIR
    marker = root / "CAPTURE.json"
    if not marker.is_file():
        return None
    return load_json(marker)


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        import torch

        if torch.backends.mps.is_available():
            ta = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to("mps")
            tb = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)).to("mps")
            out = ta.matmul(tb).cpu().numpy()
            del ta, tb
            return out
    except Exception:
        pass
    return np.matmul(a, b)


def _load_layer_w_and_aux(layer: int) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    from tools.future import deltanet_qkvz_precision as dqp
    from tools.future import deltanet_representation as dnr

    rows, geo = dnr.census_rows()
    rec = next(
        r
        for r in rows
        if int(r["layer"]) == int(layer) and str(r["organ"]) == "attention.linear_qkvz"
    )
    W = dqp._load_q4_matrix(rec)
    aux = dqp._layer_aux(rows, geo, int(layer))
    return W, aux, geo


def _roll_prompt(
    *,
    W: np.ndarray,
    X: np.ndarray,
    aux: Mapping[str, Any],
    geo: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Incumbent consume path along one prompt. S starts at 0."""
    from tools.future.deltanet_qkvz_precision import (
        _ba_decay_beta,
        _gated_delta,
        _rearrange_conv,
    )

    vh = int(geo["value_heads"])
    kd = int(geo["key_head_dim"])
    vd = int(geo["value_head_dim"])
    C = int(geo["conv_channels"])
    S = np.zeros((vh, kd, vd), dtype=np.float32)
    cs = np.zeros((C, int(geo["conv_kernel"]) - 1), dtype=np.float32)
    Wba = aux["ba"]
    conv = aux["conv"]
    out: list[dict[str, Any]] = []
    n = int(X.shape[0])
    # Project the whole prompt, then consume sequentially (conv/S are recurrent).
    Y = _matmul(X, W.T)
    BA = _matmul(X, Wba.T)
    for t in range(n):
        pre = vector_sha256(S)
        pre_frob = float(np.sqrt(np.square(S).sum()))
        y = Y[t]
        ba = BA[t]
        q, k, v, z = _rearrange_conv(y, conv, cs, geo)
        decay, beta = _ba_decay_beta(ba, aux["a_log"], aux["dt_bias"], geo)
        S, h = _gated_delta(S, q, k, v, decay, beta)
        out.append(
            {
                "token_position": t,
                "x_sha256": vector_sha256(X[t]),
                "y_sha256": vector_sha256(y),
                "pre_state_sha256": pre,
                "post_state_sha256": vector_sha256(S),
                "output_sha256": vector_sha256(h),
                "pre_state_frob": pre_frob,
                "post_state_frob": float(np.sqrt(np.square(S).sum())),
                "output_frob": float(np.sqrt(np.square(h).sum())),
            }
        )
    return out


def run_capture(
    *,
    payload_dir: Path | None = None,
    layers: Sequence[int] | None = None,
    x_dir: Path | None = None,
) -> dict[str, Any]:
    """Roll incumbent DeltaNet on real X for the representative layers."""
    started = time.perf_counter()
    root = payload_dir if payload_dir is not None else PAYLOAD_DIR
    root.mkdir(parents=True, exist_ok=True)
    x_capture = x_dir if x_dir is not None else resolve_x_capture_dir()
    x_man = load_x_manifest(x_capture)
    if int(x_man.get("hidden") or 0) != HIDDEN:
        raise CaptureUnavailable(f"REFUSED: X hidden {x_man.get('hidden')} != {HIDDEN}")
    if str(x_man.get("input") or "") != "post_attn_norm":
        raise CaptureUnavailable(
            f"REFUSED: X input is {x_man.get('input')!r}, want post_attn_norm"
        )
    reps = pick_representatives()
    chosen = list(layers) if layers is not None else list(reps["chosen_layers"])
    for layer in chosen:
        if not is_linear_attention_layer(int(layer)):
            raise CaptureUnavailable(
                f"REFUSED: layer {layer} is not a DeltaNet (linear_attention) layer"
            )
    spec = specimen_identity()
    all_rows: list[dict[str, Any]] = []
    layer_files: list[dict[str, Any]] = []
    x_man_sha = sha256_file(x_capture / "manifest.json")

    for layer in chosen:
        x_f16 = load_x_f16(x_capture, int(layer))
        n_tokens = int(x_f16.shape[0])
        meta = expand_capture_rows(layer=int(layer), x_manifest=x_man, n_tokens=n_tokens)
        x = np.ascontiguousarray(x_f16, dtype=np.float32)
        del x_f16
        W, aux, geo = _load_layer_w_and_aux(int(layer))
        x_path = root / f"L{int(layer):02d}_x.f32"
        x_file_sha = write_f32(x_path, x)
        x_src = x_capture / f"L{int(layer):02d}.f16"
        prov = provenance_captured(
            source_path=str(x_src),
            source_sha256=sha256_file(x_src),
            layer=int(layer),
        )
        # Group meta by prompt so S is rolled from zero per prompt, not across prompts.
        by_prompt: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for rec in meta:
            pid = str(rec["prompt_id"])
            if pid not in by_prompt:
                by_prompt[pid] = []
                order.append(pid)
            by_prompt[pid].append(rec)
        rolled_by_index: dict[int, dict[str, Any]] = {}
        for pid in order:
            recs = sorted(by_prompt[pid], key=lambda r: int(r["token_position"]))
            idx = [int(r["x_row_index"]) for r in recs]
            xp = x[idx]
            traj = _roll_prompt(W=W, X=xp, aux=aux, geo=geo)
            if len(traj) != len(recs):
                raise CaptureUnavailable(
                    f"REFUSED: rollout length {len(traj)} != prompt tokens {len(recs)}"
                )
            for rec, step in zip(recs, traj):
                rolled_by_index[int(rec["x_row_index"])] = step
        del W

        for rec in meta:
            i = int(rec["x_row_index"])
            step = rolled_by_index[i]
            row = {
                "row_id": f"DN-L{int(layer):02d}-{i:06d}",
                "layer": int(layer),
                "prompt_id": rec["prompt_id"],
                "prompt_family": rec["prompt_family"],
                "prompt_idx": rec["prompt_idx"],
                "token_position": rec["token_position"],
                "seq_len": rec["seq_len"],
                "position_band": rec["position_band"],
                "capability_domain": rec["capability_domain"],
                "x_row_index": i,
                "x_path": str(x_path.relative_to(REPO)),
                "x_sha256": step["x_sha256"],
                "input_sha256": step["x_sha256"],
                "y_sha256": step["y_sha256"],
                "pre_state_sha256": step["pre_state_sha256"],
                "post_state_sha256": step["post_state_sha256"],
                "output_sha256": step["output_sha256"],
                "pre_state_frob": step["pre_state_frob"],
                "post_state_frob": step["post_state_frob"],
                "output_frob": step["output_frob"],
                "synthetic": False,
                "provenance": prov,
            }
            all_rows.append(annotate_dn_row(row))
        layer_files.append(
            {
                "layer": int(layer),
                "role": DN_LAYER_ROLES.get(int(layer), "other"),
                "mixer": "linear_attention",
                "n_rows": n_tokens,
                "x_path": str(x_path.relative_to(REPO)),
                "x_sha256": x_file_sha,
                "x_source_f16": str(x_src),
                "x_source_f16_sha256": sha256_file(x_src),
            }
        )
        del x

    split = split_by_prompt(all_rows, hold_frac=HOLD_FRAC)
    manifest = emit_manifest(
        all_rows,
        split,
        allow_fixture=False,
        require_sizing=False,
    )
    row_table = [{k: v for k, v in r.items() if k != "prompt_text"} for r in manifest["rows"]]
    rows_path = root / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for rec in row_table:
            handle.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
    elapsed = time.perf_counter() - started
    capture_doc = {
        "schema": "hawking.future.deltanet_teacher_corpus.capture.v1",
        "status": "captured",
        "payload_dir": str(root.relative_to(REPO)),
        "x_capture_dir": str(x_capture),
        "x_manifest_sha256": x_man_sha,
        "x_input": "post_attn_norm",
        "y_function": "HQ30UQ4 W_qkvz @ x, then incumbent rearrange+gated-delta from S=0",
        "specimen": spec,
        "fusion_env": dict(FUSION_ENV),
        "layers": layer_files,
        "n_rows": manifest["n_rows"],
        "n_train_rows": manifest["n_train_rows"],
        "n_hold_rows": manifest["n_hold_rows"],
        "n_prompts_per_domain": manifest["n_prompts_per_domain"],
        "n_rows_per_domain": manifest["n_rows_per_domain"],
        "position_bands": manifest["position_bands"],
        "split": manifest["split"],
        "rows_jsonl": str(rows_path.relative_to(REPO)),
        "rows_jsonl_sha256": sha256_file(rows_path),
        "representatives": reps,
        "capture_elapsed_s": elapsed,
        "claim_boundary": CLAIM_BOUNDARY,
        "resident_binary": spec.get("resident_binary"),
        "resident_used_for": (
            "specimen identity and sealed fusion env; the JSONL resident does "
            "not dump hidden states, so X is capture_diverse2 and Y/S are the "
            "incumbent operator reconstructed from catalog W"
        ),
    }
    (root / "CAPTURE.json").write_text(json.dumps(capture_doc, indent=1, sort_keys=True) + "\n")
    slim = dict(manifest)
    slim["rows"] = row_table
    slim["payload"] = capture_doc
    return slim


def selftest() -> dict[str, Any]:
    diverse = make_diverse_fixture_corpus(4, 3)
    split = split_by_prompt(diverse)
    ok = emit_manifest(diverse, split, allow_fixture=True, require_sizing=False)
    if not ok["accepted"]:
        raise SystemExit(f"selftest: diverse fixture must emit, got {ok}")
    leak_refused = False
    leak_codes: list[str] = []
    leaked = {
        "train_prompt_ids": list(split["train_prompt_ids"]) + list(split["hold_prompt_ids"][:1]),
        "hold_prompt_ids": list(split["hold_prompt_ids"]),
        "hold_frac": split["hold_frac"],
    }
    try:
        emit_manifest(diverse, leaked, allow_fixture=True)
    except CorpusRefused as exc:
        leak_refused = True
        leak_codes = list(exc.codes)
    else:
        raise SystemExit("selftest: leaked split was NOT refused — the guard is dead")
    if "HELD_OUT_PROMPT_LEAK" not in leak_codes:
        raise SystemExit(f"selftest: expected HELD_OUT_PROMPT_LEAK, got {leak_codes}")

    gauss = list(diverse)
    gauss[0] = make_gaussian_row(diverse[0])
    syn_refused = False
    syn_codes: list[str] = []
    try:
        emit_manifest(gauss, split, allow_fixture=True)
    except CorpusRefused as exc:
        syn_refused = True
        syn_codes = list(exc.codes)
    else:
        raise SystemExit("selftest: gaussian row was NOT refused — NNS-001 guard is dead")
    if "SYNTHETIC_ROW" not in syn_codes:
        raise SystemExit(f"selftest: expected SYNTHETIC_ROW, got {syn_codes}")

    reps = pick_representatives()
    typical = next(p for p in reps["chosen"] if p["role"] == "typical")
    if typical["layer"] == 0:
        raise SystemExit("selftest: layer 0 was picked as typical")
    return {
        "diverse_accepted": True,
        "diverse_n": ok["n_rows"],
        "held_out_leak_refused": leak_refused,
        "held_out_leak_codes": leak_codes,
        "synthetic_refused": syn_refused,
        "synthetic_codes": syn_codes,
        "chosen_layers": reps["chosen_layers"],
        "layer0_typical": False,
        "guards_module": "tools.future.mlp_teacher_corpus",
    }


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else _sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args(argv_list)
    if args.selftest or args.build:
        json.dump(selftest(), _sys.stdout, indent=2, sort_keys=True)
        _sys.stdout.write("\n")
        return 0
    if args.capture:
        out = run_capture()
        json.dump({k: v for k, v in out["payload"].items() if k != "rows"}, _sys.stdout, indent=2)
        _sys.stdout.write("\n")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
