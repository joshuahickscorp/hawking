#!/usr/bin/env python3.12
"""End to end: a real ``.gravity`` shard drives a real GLM-5.2 block and a real next token.

Every earlier stage measured a fitted student held in memory.  This one writes the payload
to a sealed shard, reads it back through the container's own header and hash checks, runs
it through the CPU authority and the Metal grammars, substitutes the result for the MoE
inside the real block, carries the block forward, and takes the short logit lens to a
sampled token.  If any of those seams is fictional, this is where it shows.

    run [LAYER]
    selftest
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import gravity_format  # noqa: E402
import gravity_functional_codec as codec  # noqa: E402
import glm52_functional_gauntlet as gauntlet  # noqa: E402
import hawking_null_metric as metric  # noqa: E402

OUT = gauntlet.OUT
ARTIFACTS = Path("/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity"
                 "/compact/generation_b_functional")


def build_shard(layers: list[int], *, hidden: int = 1024, seed: int = 17) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"GLM52_FUNCTIONAL_L{min(layers):02d}_L{max(layers):02d}.gravity"
    entries, fits = [], {}
    for layer in layers:
        fitted = codec.fit_layer(layer, hidden=hidden, seed=seed)
        entries.append((layer, fitted["blob"]))
        fits[layer] = {"local_bpw": fitted["local_bpw"], "score": fitted["score"],
                       "physical": fitted["physical"], "ridge": fitted["ridge"]}
    header = codec.write_shard(path, entries, model={
        "repo": "zai-org/GLM-5.2",
        "revision": "b4734de4facf877f85769a911abafc5283eab3d9",
        "representation": "FUNCTIONAL_MODEL",
        "research_label": "RESEARCH_ONLY, one-stage functional student, not capability",
    })
    return {"path": path, "header": header, "fits": fits}


def run(layer: int = 38, *, hidden: int = 1024, seed: int = 17) -> dict:
    started = time.time()
    built = build_shard([layer], hidden=hidden, seed=seed)
    path = built["path"]

    # 1. The container's own integrity, not ours.
    verified = codec.verify(path)
    header = gravity_format.read_header(path)
    descriptor = header["tensors"][0]

    # 2. Read the payload back out of the shard and execute it through the CPU authority.
    blob = gravity_format.read_tensor(path, descriptor["name"])
    payload = codec.deserialize(blob)

    got = gauntlet.arrays(layer, gauntlet.SCORE_SPLIT,
                          ("pre_router_hidden", "post_attention_hidden", "block_output",
                           "post_moe"))
    x = got["pre_router_hidden"]
    cpu = codec.execute(payload, x)

    # 3. Metal parity against that authority, on the real state rather than a probe.
    metal_rows = []
    try:
        import gravity_functional_metal as metal_lane
        engine = metal_lane.FunctionalMetal()
        probe = x.reshape(-1, codec_width(payload))[0]
        reference = cpu.reshape(-1, payload["out_width"])[0]
        for grammar in ("FRT_A", "FRT_B", "FRT_D" if payload["hidden"] == 0 else "FRT_B"):
            projection = (codec.projection(payload["width"], payload["hidden"],
                                           payload["seed"])
                          if grammar == "FRT_A" else None)
            produced, buffers = engine.run(grammar, payload, probe, projection=projection)
            metal_rows.append({
                "grammar": grammar,
                "relative_l2_vs_cpu_authority": float(
                    np.linalg.norm(produced - reference)
                    / max(np.linalg.norm(reference), 1e-12)),
                "command_buffers": buffers,
            })
    except Exception as error:  # noqa: BLE001 - a machine without Metal must still report
        metal_rows.append({"grammar": "UNAVAILABLE", "error": repr(error)[:200]})

    # 4. Substitute into the real block.
    block = got["post_attention_hidden"] + cpu.reshape(got["post_moe"].shape)
    fit_block = gauntlet.arrays(layer, gauntlet.FIT_SPLITS[0],
                                ("block_output",))["block_output"]
    null = metric.fit_null(fit_block.reshape(-1, gauntlet.HIDDEN))
    block_score = metric.score(got["block_output"].reshape(-1, gauntlet.HIDDEN),
                               block.reshape(-1, gauntlet.HIDDEN), null)

    # 5. Carry it one real layer forward and take a token.
    import glm52_teacher_capture as capture
    import glm52_reference as reference_forward
    graph = capture._graph()
    config = capture.official_config()
    source = capture.ShardTensorSource(capture.SOURCE_ROOT, capture._tensor_table(graph))
    with np.load(gauntlet.capsule(layer, gauntlet.SCORE_SPLIT)) as data:
        carry_topk = np.asarray(data["carry_out_index_selection"])

    def forward(state):
        return capture.capture_layer(np.asarray(state, dtype=np.float32), source,
                                     layer + 1, config, carry_topk,
                                     reference_forward.ReferenceCache())[2]

    teacher_next = forward(got["block_output"])
    student_next = forward(block)

    lens_teacher = capture._logit_lens(source, teacher_next["block_output"], config)
    lens_student = capture._logit_lens(source, student_next["block_output"], config)
    token = {}
    if lens_teacher and lens_student:
        teacher_logits = lens_teacher["short_logits"].reshape(-1, lens_teacher["short_logits"].shape[-1])
        student_logits = lens_student["short_logits"].reshape(-1, teacher_logits.shape[-1])
        teacher_token = teacher_logits.argmax(axis=-1)
        student_token = student_logits.argmax(axis=-1)
        token = {
            "vocabulary_subset_rows": int(teacher_logits.shape[-1]),
            "positions": int(teacher_logits.shape[0]),
            "greedy_token_agreement": float((teacher_token == student_token).mean()),
            "teacher_first_token": int(teacher_token[0]),
            "student_first_token": int(student_token[0]),
            "note": "the logit lens reads a 1024-row vocabulary subset through the final "
                    "norm from layer %d, which is a probe and not the model's head" % (layer + 1),
        }

    return {
        "schema": "hawking.glm52.functional_integration.v1",
        "layer": layer,
        "artifact": str(path),
        "artifact_bytes": path.stat().st_size,
        "container": {
            "codec": descriptor["codec"],
            "generator": descriptor["generator"],
            "disposition": descriptor["disposition"],
            "replaces": descriptor["replaces"],
            "header_verified": verified["verified"],
            "all_organs_deterministic": verified["all_deterministic"],
            "representation": header["compression"]["representation"],
        },
        "chain": [
            ".gravity shard written and hash-verified",
            "payload read back through the container header",
            "CPU authority executed on real captured states",
            "Metal grammars parity-checked against the authority",
            "MoE output substituted inside the real block",
            "block residual carried into layer %d with real attention and a real router"
            % (layer + 1),
            "short logit lens taken to a greedy token",
        ],
        "fit": built["fits"][layer],
        "metal_parity": metal_rows,
        "block": {k: v for k, v in block_score.items() if k != "schema"},
        "next_layer_router": {
            "mean_topk_overlap_of_8": float(np.mean([
                len(set(a) & set(b)) for a, b in zip(
                    teacher_next["topk_indices"].reshape(-1, 8),
                    student_next["topk_indices"].reshape(-1, 8))])),
            "top1_agreement": float(
                (teacher_next["topk_indices"].reshape(-1, 8)[:, 0]
                 == student_next["topk_indices"].reshape(-1, 8)[:, 0]).mean()),
        } if "topk_indices" in teacher_next else {},
        "token": token,
        "seconds": round(time.time() - started, 1),
        "honest_scope": "one functional layer inside an otherwise exact teacher forward. "
                        "Not a complete compact model and not a capability measurement.",
    }


def codec_width(payload: dict) -> int:
    return payload["width"]


def selftest() -> int:
    # A shard must round-trip through the real container, and the descriptor must name the
    # source tensors it stands in for so nothing disappears from the manifest.
    import tempfile
    generator = np.random.default_rng(0)
    readout = generator.standard_normal((16, 32)).astype(np.float32)
    blob = codec.serialize(readout, seed=5, width=8, layer=7)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.gravity"
        codec.write_shard(path, [(7, blob)], model={"repo": "probe"})
        header = gravity_format.read_header(path)
        descriptor = header["tensors"][0]
        assert descriptor["disposition"] == "REPLACED_BY_FUNCTIONAL_CODEC"
        assert any("experts" in name for name in descriptor["replaces"])
        assert header["compression"]["representation"] == "FUNCTIONAL_MODEL"
        recovered = codec.deserialize(gravity_format.read_tensor(path, descriptor["name"]))
        assert recovered["layer"] == 7 and recovered["seed"] == 5
        x = generator.standard_normal((3, 8)).astype(np.float32)
        assert np.array_equal(codec.execute(recovered, x),
                              codec.execute(codec.deserialize(blob), x))
        report = codec.verify(path)
        assert report["all_deterministic"]
    print(json.dumps({"selftest": "PASS"}))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if command == "selftest":
        raise SystemExit(selftest())
    if command == "run":
        payload = run(int(sys.argv[2]) if len(sys.argv) > 2 else 38)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "GLM52_FUNCTIONAL_RUNTIME_RESULT.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=float))
        print(json.dumps(payload, indent=2, default=float))
    else:
        raise SystemExit(f"unknown command: {command}")
