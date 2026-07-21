#!/usr/bin/env python3.12
"""Functionally tune the installed P1 PQ codebooks on the cached Kimi F1 seam.

Indices, protected rows, codebook shapes, dtypes, and physical ceilings remain
unchanged. Only FP16 codebook values are trained, using fit tokens exclusively:
gate/up codebooks target the nonlinear hidden state, down codebooks target expert
output, then a short joint phase targets route-weighted output fidelity.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import ml_dtypes
import numpy as np


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_f1_bracket as f1  # noqa: E402


def torch_runtime() -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as functional
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch, (functional, device)


def decode_array(component: dict[str, Any]) -> np.ndarray:
    dtype = (ml_dtypes.bfloat16 if component["encoding"] == "bfloat16" else np.float16)
    return np.frombuffer(component["data"], dtype=dtype).astype(np.float32).reshape(
        tuple(int(value) for value in component["shape"])
    )


class MatrixCodec:
    def __init__(self, matrix: str, components: list[dict[str, Any]], torch: Any, device: Any):
        self.matrix = matrix
        by_name = {component["name"]: component for component in components}
        matches = [name for name in by_name if name.endswith(f".{matrix}.base.indices")]
        if len(matches) != 1:
            raise f1.F1Error(f"cannot identify physical codec for {matrix}")
        self.stem = matches[0].removesuffix(".base.indices")
        self.by_name = by_name
        scale = decode_array(by_name[f"{self.stem}.base.salience_scale"])
        self.scale = torch.tensor(scale, device=device)
        self.cols = scale.size
        self.codebooks = {}
        self.initial = {}
        self.indices = {}
        for role in ("base", "doctor"):
            codebook_component = by_name[f"{self.stem}.{role}.codebooks"]
            codebook = decode_array(codebook_component)
            parameter = torch.nn.Parameter(torch.tensor(codebook, device=device))
            self.codebooks[role] = parameter
            self.initial[role] = torch.tensor(codebook, device=device)
            index_component = by_name[f"{self.stem}.{role}.indices"]
            shape = tuple(int(value) for value in index_component["shape"])
            index = f1.unpack_unsigned(
                index_component["data"], math.prod(shape), int(index_component["bits"]),
            ).reshape(shape).astype(np.int64)
            self.indices[role] = torch.tensor(index, device=device)
        n_vectors = self.indices["base"].shape[0]
        dimension = self.codebooks["base"].shape[0] * self.codebooks["base"].shape[-1]
        self.rows = int(n_vectors * dimension // self.cols)
        self.protected = {}
        for role in ("base", "doctor"):
            row_component = by_name.get(f"{self.stem}.{role}.protected_row_indices")
            value_component = by_name.get(f"{self.stem}.{role}.protected_row_values")
            if row_component is None:
                self.protected[role] = None
                continue
            count = int(row_component["shape"][0])
            rows = f1.unpack_unsigned(
                row_component["data"], count, int(row_component["bits"]),
            ).astype(np.int64)
            values = decode_array(value_component)
            self.protected[role] = (
                rows.tolist(),
                torch.tensor(values, device=device),
            )

    def parameters(self) -> list[Any]:
        return [self.codebooks["base"], self.codebooks["doctor"]]

    @staticmethod
    def quantized_straight_through(parameter: Any) -> Any:
        quantized = parameter.to(dtype=parameter.new_zeros(()).half().dtype).float()
        return parameter + (quantized - parameter).detach()

    def reconstruct(self) -> Any:
        outputs = {}
        for role in ("base", "doctor"):
            codebook = self.quantized_straight_through(self.codebooks[role])
            index = self.indices[role]
            vectors = [codebook[subspace][index[:, subspace]]
                       for subspace in range(index.shape[1])]
            outputs[role] = __import__("torch").cat(vectors, dim=1).reshape(
                self.rows, self.cols,
            ) / self.scale[None, :]
        base = outputs["base"]
        if self.protected["base"] is not None:
            rows, values = self.protected["base"]
            base = self.replace_rows(base, rows, values, additive=False)
        result = base + outputs["doctor"]
        if self.protected["doctor"] is not None:
            rows, values = self.protected["doctor"]
            result = self.replace_rows(result, rows, values, additive=True)
        return result

    @staticmethod
    def replace_rows(matrix: Any, rows: list[int], values: Any, *, additive: bool) -> Any:
        """Differentiable sparse rows without MPS-unsupported index_copy/index_add."""
        import torch
        pieces = []
        start = 0
        for value_index, row in enumerate(rows):
            pieces.append(matrix[start:row])
            replacement = values[value_index:value_index + 1]
            if additive:
                replacement = matrix[row:row + 1] + replacement
            pieces.append(replacement)
            start = row + 1
        pieces.append(matrix[start:])
        return torch.cat(pieces, dim=0)

    def regularization(self) -> Any:
        return sum((self.codebooks[role] - self.initial[role]).square().mean()
                   for role in ("base", "doctor"))

    def install(self, components: list[dict[str, Any]]) -> None:
        by_name = {component["name"]: component for component in components}
        for role in ("base", "doctor"):
            value = self.codebooks[role].detach().cpu().numpy().astype(np.float16)
            target = by_name[f"{self.stem}.{role}.codebooks"]
            target["data"] = value.tobytes(order="C")


def objective(prediction: Any, target: Any, weight: Any | None = None) -> Any:
    import torch
    if weight is not None:
        prediction = prediction * weight[:, None]
        target = target * weight[:, None]
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    normalized_mse = (prediction - target).square().mean() / (target.square().mean() + 1e-8)
    norm_ratio = prediction.norm(dim=-1) / (target.norm(dim=-1) + 1e-8)
    return ((1 - cosine).mean() + 0.35 * (1 - cosine).amax() +
            0.20 * normalized_mse + 0.05 * torch.log(norm_ratio + 1e-8).square().mean())


def train_phase(
    label: str,
    parameters: list[Any],
    closure: Any,
    torch: Any,
    steps: int,
    learning_rate: float,
) -> list[float]:
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = closure()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % 10 == 0:
            print(json.dumps({"phase": label, "step": step + 1, "loss": losses[-1]}),
                  flush=True)
    return losses


def run(f1_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = f1_dir / "teacher_capture.npz"
    receipt = f1.read_json(f1_dir / "teacher_capture.json")
    if f1.sha256_file(capture_path) != receipt["capture_sha256"]:
        raise f1.F1Error("teacher capture hash mismatch")
    with np.load(capture_path, allow_pickle=False) as loaded:
        capture = {key: loaded[key] for key in loaded.files}
    source_payload = f1_dir / "P1_sentinel_expert.k26f1"
    source_result = f1.read_json(f1_dir / "P1_F1_RESULT.json")
    if f1.sha256_file(source_payload) != source_result["payload"]["sha256"]:
        raise f1.F1Error("source P1 payload hash mismatch")
    header, components = f1.read_payload(source_payload)
    torch, (functional, device) = torch_runtime()
    codecs = {matrix: MatrixCodec(matrix, components, torch, device) for matrix in f1.MATRICES}
    fit_x = torch.tensor(capture["fit_x"].astype(np.float32), device=device)
    score_x_np = capture["score_x"].astype(np.float32)
    teacher_hidden = torch.tensor(
        capture["fit_teacher_hidden"].astype(np.float32), device=device,
    )
    teacher_output = torch.tensor(
        capture["fit_teacher_output"].astype(np.float32), device=device,
    )
    sentinel = int(capture["sentinel_expert"][0])
    fit_routes = capture["fit_routes"].astype(np.int32)
    fit_route_weights = capture["fit_route_weights"].astype(np.float32)
    route_weight_np = np.sum(np.where(
        fit_routes == sentinel, fit_route_weights, 0.0,
    ), axis=1).astype(np.float32)
    route_weight = torch.tensor(route_weight_np, device=device)

    gate_up_parameters = codecs["gate_proj"].parameters() + codecs["up_proj"].parameters()
    down_parameters = codecs["down_proj"].parameters()
    all_parameters = gate_up_parameters + down_parameters

    def gate_up_closure() -> Any:
        gate = functional.silu(fit_x @ codecs["gate_proj"].reconstruct().T)
        hidden = gate * (fit_x @ codecs["up_proj"].reconstruct().T)
        return objective(hidden, teacher_hidden) + 1e-4 * (
            codecs["gate_proj"].regularization() + codecs["up_proj"].regularization()
        )

    phase_gate_up = train_phase(
        "NONLINEAR_HIDDEN", gate_up_parameters, gate_up_closure, torch, 40, 0.01,
    )

    def down_closure() -> Any:
        output = teacher_hidden @ codecs["down_proj"].reconstruct().T
        return objective(output, teacher_output, route_weight) + 1e-4 * (
            codecs["down_proj"].regularization()
        )

    phase_down = train_phase("EXPERT_OUTPUT", down_parameters, down_closure, torch, 40, 0.01)

    def joint_closure() -> Any:
        gate = functional.silu(fit_x @ codecs["gate_proj"].reconstruct().T)
        hidden = gate * (fit_x @ codecs["up_proj"].reconstruct().T)
        output = hidden @ codecs["down_proj"].reconstruct().T
        return objective(output, teacher_output, route_weight) + 5e-5 * sum(
            codec.regularization() for codec in codecs.values()
        )

    phase_joint = train_phase("JOINT_ROUTE_WEIGHTED_OUTPUT", all_parameters,
                              joint_closure, torch, 20, 0.003)
    for codec in codecs.values():
        codec.install(components)
    payload_path = output_dir / "P1_FUNCTIONAL_CODEBOOKS.k26f1"
    payload = f1.write_payload(payload_path, {
        **{key: value for key, value in header.items() if key != "components"},
        "schema": "hawking.kimi_k26.f1_functional_codebooks_payload.v1",
        "family": "FUNCTIONAL_CODEBOOKS", "fit_uses_score_data": False,
        "source_payload_sha256": source_result["payload"]["sha256"],
    }, components)
    budget = source_result["physical_budget"]
    if payload["base_component_bytes"] > int(budget["base_ceiling_bytes"]):
        raise f1.F1Error("functional base exceeds original ceiling")
    if payload["doctor_component_bytes"] > int(budget["doctor_ceiling_bytes"]):
        raise f1.F1Error("functional Doctor exceeds original ceiling")
    if (payload["header_overhead_bytes"] > int(budget["overhead_ceiling_bytes"]) or
            payload["bytes"] > int(budget["complete_ceiling_bytes"])):
        raise f1.F1Error("functional payload exceeds complete ceiling")

    final_weights = {
        matrix: codec.reconstruct().detach().cpu().numpy().astype(np.float32)
        for matrix, codec in codecs.items()
    }
    fit_output, fit_internal = f1.expert_forward(capture["fit_x"], final_weights)
    score_output, score_internal = f1.expert_forward(score_x_np, final_weights)
    fit_metric = f1.quality(capture["fit_teacher_output"], fit_output)
    score_metric = f1.quality(capture["score_teacher_output"], score_output)
    hidden_fit_metric = f1.quality(capture["fit_teacher_hidden"], fit_internal["hidden"])
    hidden_score_metric = f1.quality(capture["score_teacher_hidden"], score_internal["hidden"])
    verdict = f1.fidelity_verdict(score_metric)
    result = f1.seal({
        "schema": "hawking.kimi_k26.f1_functional_codebooks_result.v1", "status": "PASS",
        "sealed_at": f1.now(), "runtime_seconds": time.time() - started,
        "source": {"repo": f1.REPO, "revision": f1.REVISION},
        "layer": f1.LAYER, "sentinel_expert": sentinel,
        "claim_boundary": "F1 ONE REAL SENTINEL EXPERT; not full shard capability",
        "reuse": {"parent_forwards": 0, "teacher_captures": 0, "base_refits": 0,
                  "physical_indices_reused": True, "protected_rows_reused": True},
        "fit_uses_score_data": False,
        "physical_budget": {
            **budget,
            "actual_complete_bpw": payload["bytes"] * 8 /
                                   int(budget["logical_weights_represented"]),
            "unused_ceiling_bytes": int(budget["complete_ceiling_bytes"]) - payload["bytes"],
        },
        "payload": payload,
        "training": {
            "optimizer": "ADAM_QUANTIZATION_AWARE_FIXED_INDICES",
            "phase_losses": {
                "nonlinear_hidden": {"first": phase_gate_up[0], "last": phase_gate_up[-1]},
                "expert_output": {"first": phase_down[0], "last": phase_down[-1]},
                "joint_route_weighted_output": {
                    "first": phase_joint[0], "last": phase_joint[-1],
                },
            },
        },
        "metrics": {
            "fit_expert_output": fit_metric, "score_expert_output": score_metric,
            "fit_nonlinear_hidden": hidden_fit_metric,
            "score_nonlinear_hidden": hidden_score_metric,
            "score_cosine_gain_vs_weight_residual_pq": (
                score_metric["cosine_mean"] -
                source_result["metrics"]["score"]["doctored"]["cosine_mean"]
            ),
        },
        "candidate_verdict": verdict,
        "decision": ("PROMOTE_P1_FUNCTIONAL_CODEBOOKS_TO_F2" if verdict == "SURVIVES_F1" else
                     "RETIRE_P1_FUNCTIONAL_CODEBOOK_INSTANCE"),
        "current_next_experiment": (
            "P1_FUNCTIONAL_CODEBOOKS_F2_ROUTE_STABILITY" if verdict == "SURVIVES_F1" else
            "ESTABLISH_SUB_1BPW_KIMI_EXPERT_REPRESENTATION_BOUNDARY"
        ),
    })
    f1.atomic_json(output_dir / "KIMI_K26_P1_F1_FUNCTIONAL_CODEBOOKS.json", result)
    del final_weights
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.f1_dir.resolve(strict=True), args.output_dir.resolve())
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
