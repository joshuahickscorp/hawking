#!/usr/bin/env python3
"""Dependency windows for GLM-5.2, derived from the official tensor index.

A shard is a storage boundary.  A dependency window is an architectural one: the smallest
complete set of tensors needed for one sequential teacher-capture and packing step.  The
two do not line up in this model, and assuming they do is how a run ends up quantizing
half a layer because one source shard happened to finish first.

The live index says so plainly: layer 2 lives in shard 38, layer 3 spans shards 75 to 79,
layer 38 spans 108 to 112.  Literal shard 0 is not the model, and neither is any other
single shard.

IndexShare is the second reason windows are not shards.  `config.indexer_types` marks each
layer `full` or `shared`; a `full` layer owns an indexer and the `shared` layers after it
reuse it.  A window over a `shared` layer therefore depends on its group owner's indexer
tensors, and packing it without them is packing a layer that cannot run.

    plan      write GLM52_GENERATION_B_WINDOW_PLAN.json
    pilot     write GLM52_GENERATION_B_PILOT_PROGRAM.json
    selftest
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_contract as contract  # noqa: E402

REPO = HERE.parent.parent
STATE = Path.home() / "Library/Application Support/Hawking/GLM52Gravity"
REPORTS = REPO / "reports/condense/glm52_generation_b"

REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--zai-org--GLM-5.2/snapshots"
            / REVISION)
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")

# The pilot may cover at most six windows.  These are the architectural roles the
# directive requires, in order; a role that no window can fill is dropped rather than
# faked, and a window that fills two roles is reused rather than duplicated.
PILOT_ROLES = (
    ("P0", "global tensors, embeddings, first dense layer"),
    ("P1", "final dense layer and dense-to-MoE transition"),
    ("P2", "first sparse MoE layer and first complete IndexShare group"),
    ("P3", "middle sparse/IndexShare group"),
    ("P4", "late sparse/IndexShare group"),
    ("P5", "final layer, final norm, lm_head, and the MTP boundary"),
)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_index() -> dict:
    path = SNAPSHOT / "model.safetensors.index.json"
    if not path.exists():
        raise SystemExit(f"official tensor index not cached at {path}")
    return json.loads(path.read_text())


def load_config() -> dict:
    return json.loads((STATE / "control/config.json").read_text())


def indexshare_groups(config: dict) -> tuple[dict[int, int], dict[int, list[int]]]:
    """Map each layer to the layer that owns its indexer, and each owner to its members.

    `indexer_types[i] == "full"` means layer i carries its own indexer tensors.  A run of
    `shared` layers after a `full` one reuses that owner's indexer.
    """
    types = config["indexer_types"]
    owner_of: dict[int, int] = {}
    members: dict[int, list[int]] = defaultdict(list)
    current: int | None = None
    for layer, kind in enumerate(types):
        if kind == "full":
            current = layer
        if current is None:
            raise ValueError(f"layer {layer} is {kind} with no preceding full layer")
        owner_of[layer] = current
        members[current].append(layer)
    return owner_of, dict(members)


def build_windows(index: dict, config: dict) -> list[dict]:
    """One window per layer, plus one for the global organs.

    A layer is the smallest set that can be teacher-captured and packed as a unit: its
    attention, its router, its experts and its norms all participate in one forward step.
    Going finer would mean packing a partial layer; going coarser would hold more BF16
    resident than the step needs.
    """
    weight_map = index["weight_map"]
    dense_layers = int(config["first_k_dense_replace"])
    declared_layers = int(config["num_hidden_layers"]) if "num_hidden_layers" in config \
        else len(config["indexer_types"])
    owner_of, members = indexshare_groups(config)

    by_layer: dict[int, list[str]] = defaultdict(list)
    globals_: list[str] = []
    for name in weight_map:
        match = LAYER_RE.match(name)
        if match:
            by_layer[int(match.group(1))].append(name)
        else:
            globals_.append(name)

    windows: list[dict] = [{
        "window": "W_GLOBAL",
        "kind": "GLOBAL_ORGANS",
        "layers": [],
        "indexshare_owner": None,
        "indexshare_role": None,
        "tensors": sorted(globals_),
        "shards": sorted({weight_map[name] for name in globals_}),
        "roles": ["embedding", "final_norm", "lm_head"],
    }]

    for layer in sorted(by_layer):
        names = sorted(by_layer[layer])
        is_mtp = layer >= declared_layers
        if is_mtp:
            kind, owner, role = "MTP_BOUNDARY", None, None
        elif layer < dense_layers:
            kind = "DENSE"
            owner, role = owner_of[layer], config["indexer_types"][layer]
        else:
            kind = "SPARSE_MOE"
            owner, role = owner_of[layer], config["indexer_types"][layer]

        experts = sum(1 for name in names if ".experts." in name)
        window = {
            "window": f"W_L{layer:02d}",
            "kind": kind,
            "layers": [layer],
            "indexshare_owner": owner,
            "indexshare_role": role,
            "indexshare_group_members": members.get(layer) if role == "full" else None,
            "tensors": names,
            "tensor_count": len(names),
            "expert_tensor_count": experts,
            "organ_tensor_count": len(names) - experts,
            "shards": sorted({weight_map[name] for name in names}),
        }
        # A shared layer cannot run without its owner's indexer.  Record the dependency so
        # the controller never seals a window whose indexer was never captured.
        if role == "shared" and owner is not None:
            window["depends_on_window"] = f"W_L{owner:02d}"
            window["dependency_reason"] = "INDEXSHARE_OWNER_SUPPLIES_INDEXER"
        windows.append(window)

    return windows


def attach_sizes(windows: list[dict], headers: list) -> None:
    """Fold exact per-tensor byte counts in from the official safetensors headers."""
    sizes: dict[str, int] = {}
    elements: dict[str, int] = {}
    for header in headers:
        for tensor in header.tensors:
            sizes[tensor.name] = int(tensor.payload_bytes)
            count = 1
            for dim in tensor.shape:
                count *= int(dim)
            elements[tensor.name] = count
    for window in windows:
        known = [name for name in window["tensors"] if name in sizes]
        window["bytes"] = sum(sizes[name] for name in known)
        window["elements"] = sum(elements[name] for name in known)
        window["sized_tensors"] = len(known)


def select_pilot(windows: list[dict], config: dict) -> list[dict]:
    """Pick at most six architecture-representative windows, one per required role."""
    by_id = {window["window"]: window for window in windows}
    dense_layers = int(config["first_k_dense_replace"])
    declared = len(config["indexer_types"])
    owner_of, members = indexshare_groups(config)

    moe_layers = sorted(layer for layer in range(dense_layers, declared))
    full_owners = sorted(owner for owner in members if owner >= dense_layers - 1)

    picks: list[tuple[str, str, list[str], str]] = []

    picks.append(("P0", PILOT_ROLES[0][1], ["W_GLOBAL", "W_L00"],
                  "the only window carrying embeddings, and the first dense block"))

    last_dense = dense_layers - 1
    first_moe = moe_layers[0]
    picks.append(("P1", PILOT_ROLES[1][1], [f"W_L{last_dense:02d}", f"W_L{first_moe:02d}"],
                  "the dense-to-MoE transition, and layer "
                  f"{last_dense} owns the indexer the first MoE group reuses"))

    first_group_owner = owner_of[first_moe]
    first_group = members[first_group_owner]
    picks.append(("P2", PILOT_ROLES[2][1],
                  [f"W_L{layer:02d}" for layer in first_group],
                  f"the first complete IndexShare group, owner L{first_group_owner}"))

    mid_owner = full_owners[len(full_owners) // 2]
    picks.append(("P3", PILOT_ROLES[3][1],
                  [f"W_L{layer:02d}" for layer in members[mid_owner]],
                  f"a middle IndexShare group, owner L{mid_owner}"))

    late_owner = full_owners[-1]
    picks.append(("P4", PILOT_ROLES[4][1],
                  [f"W_L{layer:02d}" for layer in members[late_owner]],
                  f"the last complete IndexShare group, owner L{late_owner}"))

    tail = [f"W_L{declared - 1:02d}", "W_GLOBAL"]
    mtp = [w["window"] for w in windows if w["kind"] == "MTP_BOUNDARY"]
    picks.append(("P5", PILOT_ROLES[5][1], tail + mtp,
                  "the final layer, the final norm and lm_head, and the MTP boundary"))

    pilot: list[dict] = []
    for name, role, window_ids, why in picks:
        present = [wid for wid in window_ids if wid in by_id]
        if not present:
            continue
        members_ = [by_id[wid] for wid in present]
        pilot.append({
            "pilot": name, "architectural_role": role, "why_these_windows": why,
            "windows": present,
            "layers": sorted({layer for w in members_ for layer in w["layers"]}),
            "shards": sorted({shard for w in members_ for shard in w["shards"]}),
            "tensor_count": sum(w["tensor_count"] if "tensor_count" in w
                                else len(w["tensors"]) for w in members_),
            "bytes": sum(w.get("bytes", 0) for w in members_),
            "elements": sum(w.get("elements", 0) for w in members_),
            "categories_covered": sorted({
                contract.classify_tensor(name_, config).category
                for w in members_ for name_ in w["tensors"]}),
        })
    return pilot


def _headers_cache() -> Path:
    return REPORTS / "GLM52_SOURCE_SHARD_HEADERS.json"


def load_or_fetch_headers(index: dict, config: dict, *, workers: int = 16):
    """Exact per-tensor extents.  Cached, because 282 range requests is not free."""
    cache = _headers_cache()
    if cache.exists():
        raw = json.loads(cache.read_text())

        class _Tensor:
            __slots__ = ("name", "shard", "shape", "payload_bytes", "dtype")

            def __init__(self, row):
                self.name = row["name"]
                self.shard = row["shard"]
                self.shape = row["shape"]
                self.payload_bytes = row["payload_bytes"]
                self.dtype = row["dtype"]

        class _Header:
            __slots__ = ("path", "tensors")

            def __init__(self, row):
                self.path = row["path"]
                self.tensors = [_Tensor(t) for t in row["tensors"]]

        return [_Header(row) for row in raw["headers"]]

    # Range-GET the safetensors header of all 282 shards.  The contract module validates
    # the manifest against the index first, and fetch_all_headers refuses to return
    # unless every tensor, shard assignment and the total payload size reconcile.
    _info, rows = contract._manifest_info()
    contract.validate_remote_manifest(rows, index)  # returns the shard set, raises on drift
    headers = contract.fetch_all_headers(rows, index, config, workers=workers)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "schema": "hawking.glm52.source_shard_headers.v1",
        "revision": REVISION, "generated_at": now(),
        "headers": [{"path": h.path,
                     "tensors": [{"name": t.name, "shard": t.shard, "shape": list(t.shape),
                                  "dtype": t.dtype, "payload_bytes": int(t.payload_bytes)}
                                 for t in h.tensors]} for h in headers],
    }))
    return headers


def plan() -> int:
    index, config = load_index(), load_config()
    windows = build_windows(index, config)
    try:
        attach_sizes(windows, load_or_fetch_headers(index, config))
        sized = True
    except SystemExit as exc:
        sys.stderr.write(f"sizes unavailable ({exc}); planning shape only\n")
        sized = False

    owner_of, members = indexshare_groups(config)
    payload = {
        "schema": "hawking.glm52.generation_b_window_plan.v1",
        "generated_at": now(),
        "revision": REVISION,
        "source": {"repo": "zai-org/GLM-5.2",
                   "total_source_shards": len({s for w in windows for s in w["shards"]}),
                   "total_tensors": len(index["weight_map"]),
                   "total_source_bytes": int(index["metadata"]["total_size"])},
        "architecture": {
            "type": config["architectures"][0],
            "hidden_layers": len(config["indexer_types"]),
            "first_k_dense_replace": int(config["first_k_dense_replace"]),
            "indexshare_groups": {str(k): v for k, v in members.items()},
            "index_topk": config.get("index_topk"),
        },
        "why_windows_are_not_shards": (
            "layer 2 lives in shard 38, layer 3 spans shards 75-79 and layer 38 spans "
            "108-112; shard order is not layer order, so a shard-shaped schedule packs "
            "partial layers"),
        "sizes_attached": sized,
        "window_count": len(windows),
        "windows": windows,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    target = REPORTS / "GLM52_GENERATION_B_WINDOW_PLAN.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"wrote": str(target.relative_to(REPO)),
                      "windows": len(windows), "sizes_attached": sized,
                      "shards_covered": payload["source"]["total_source_shards"]}, indent=2))
    return 0


def pilot() -> int:
    index, config = load_index(), load_config()
    windows = build_windows(index, config)
    try:
        attach_sizes(windows, load_or_fetch_headers(index, config))
    except SystemExit:
        pass
    chosen = select_pilot(windows, config)

    payload = {
        "schema": "hawking.glm52.generation_b_pilot_program.v1",
        "generated_at": now(), "revision": REVISION,
        "policy": {
            "max_windows": 6,
            "literal_shard_zero_is_not_a_proxy": True,
            "rule": ("pilot windows are chosen by architectural role, and a role with no "
                     "distinct window is dropped rather than filled with a stand-in"),
        },
        "pilot_windows": chosen,
        "totals": {
            "windows": len(chosen),
            "distinct_shards": len({s for p in chosen for s in p["shards"]}),
            "bytes": sum(p["bytes"] for p in chosen),
            "elements": sum(p["elements"] for p in chosen),
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    target = REPORTS / "GLM52_GENERATION_B_PILOT_PROGRAM.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"wrote": str(target.relative_to(REPO)),
                      "pilot_windows": [{"pilot": p["pilot"], "layers": p["layers"],
                                         "shards": len(p["shards"]),
                                         "gib": round(p["bytes"] / (1 << 30), 2)}
                                        for p in chosen],
                      "totals": payload["totals"]}, indent=2))
    return 0


def selftest() -> int:
    config = load_config()
    index = load_index()

    owner_of, members = indexshare_groups(config)
    # Every layer has an owner, an owner owns itself, and the groups partition the layers.
    assert owner_of[0] == 0 and owner_of[3] == 2, (owner_of[0], owner_of[3])
    assert sorted(layer for group in members.values() for layer in group) == \
        list(range(len(config["indexer_types"])))
    for owner, group in members.items():
        assert group[0] == owner, (owner, group)
        assert config["indexer_types"][owner] == "full"
        assert all(config["indexer_types"][m] == "shared" for m in group[1:]), (owner, group)

    windows = build_windows(index, config)
    # Every declared tensor lands in exactly one window: a plan that loses a tensor would
    # silently exclude it from coverage forever.
    seen: dict[str, str] = {}
    for window in windows:
        for name in window["tensors"]:
            assert name not in seen, f"{name} appears in {seen[name]} and {window['window']}"
            seen[name] = window["window"]
    assert set(seen) == set(index["weight_map"]), "window plan does not cover the index"

    # The claim the whole module rests on: shard order is not layer order.
    weight_map = index["weight_map"]
    layer2 = {weight_map[n] for n in weight_map if n.startswith("model.layers.2.")}
    layer3 = {weight_map[n] for n in weight_map if n.startswith("model.layers.3.")}
    assert layer2 and layer3 and min(layer3) > max(layer2), \
        "fixture assumption broken: layers happen to be in shard order"

    # Shared layers must declare their indexer dependency.
    shared = [w for w in windows if w.get("indexshare_role") == "shared"]
    assert shared, "no shared layers found"
    assert all(w.get("depends_on_window") for w in shared)

    chosen = select_pilot(windows, config)
    assert len(chosen) <= 6, len(chosen)
    assert [p["pilot"] for p in chosen] == ["P0", "P1", "P2", "P3", "P4", "P5"]
    covered = {layer for p in chosen for layer in p["layers"]}
    assert 0 in covered, "pilot must cover the first dense layer"
    assert len(config["indexer_types"]) - 1 in covered, "pilot must cover the final layer"

    print("glm52_window_plan selftest OK")
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "plan"
    raise SystemExit({"plan": plan, "pilot": pilot, "selftest": selftest}[command]())
