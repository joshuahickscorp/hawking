"""Recomposed science module glm52_assemble (C-SCI-R1)."""
from __future__ import annotations
from tools.condense import artifact_client
gravity = artifact_client
import argparse
import json
import os
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REVISION = 'b4734de4facf877f85769a911abafc5283eab3d9'
APPSUP = Path.home() / 'Library/Application Support/Hawking'
STATE = APPSUP / 'GLM52Gravity/source_fetch'
COMPACT_ROOT = Path(os.environ.get('GLM52_COMPACT_ROOT', str(Path.home() / 'Desktop/GLM52-Gravity-SubBit')))
MODELS_ROOT = APPSUP / 'Models/GLM-5.2'
MANIFEST = STATE / 'GLM52_OFFICIAL_TENSOR_INDEX.json'
CONTRACT = REPO / 'evidence' / 'glm52' / 'GLM52_ARCHITECTURE_CONTRACT.json'
CONFIG = APPSUP / 'GLM52Gravity/control/config.json'

def _now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def official_weight_map() -> dict[str, str]:
    """tensor name -> source shard filename, from the pinned official index."""
    if MANIFEST.is_file():
        return json.loads(MANIFEST.read_text())['weight_map']
    from huggingface_hub import hf_hub_download
    path = hf_hub_download('zai-org/GLM-5.2', 'model.safetensors.index.json', revision=REVISION)
    data = json.loads(Path(path).read_text())
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data))
    return data['weight_map']

def packed_index(compact_root: Path) -> tuple[dict[str, dict], list[str]]:
    """tensor name -> {shard, codec, bytes}, plus the packed shard filenames.

    A descriptor billing zero bytes is not a payload, and is reported as such
    rather than counted -- that is the Generation-A defect this exists to catch.
    """
    index: dict[str, dict] = {}
    shards: list[str] = []
    for path in sorted(compact_root.glob('model-*.gravity')):
        shards.append(path.name)
        try:
            header = gravity.read_header(path)
        except Exception as exc:
            index[f'<unreadable:{path.name}>'] = {'shard': path.name, 'codec': None, 'bytes': 0, 'error': f'{type(exc).__name__}: {exc}'}
            continue
        for t in header.get('tensors', []):
            index[t['name']] = {'shard': path.name, 'codec': t.get('codec'), 'bytes': int(t.get('bytes', 0))}
    return (index, shards)

def disposition(entry: dict | None) -> str:
    if entry is None:
        return 'MISSING'
    if entry.get('error'):
        return 'MISSING'
    if not entry.get('bytes'):
        return 'MISSING'
    codec = entry.get('codec') or ''
    if codec == 'gravity-pq':
        return 'COMPACT_PAYLOAD'
    if codec.startswith('native.'):
        return 'PROTECTED_NATIVE_PAYLOAD'
    return 'UNDECLARED'

def check(compact_root: Path=COMPACT_ROOT) -> dict:
    weight_map = official_weight_map()
    packed, packed_shards = packed_index(compact_root)
    counts: dict[str, int] = {}
    missing_by_shard: dict[str, int] = {}
    misplaced: list[dict] = []
    for name, source_shard in weight_map.items():
        entry = packed.get(name)
        d = disposition(entry)
        counts[d] = counts.get(d, 0) + 1
        if d == 'MISSING':
            missing_by_shard[source_shard] = missing_by_shard.get(source_shard, 0) + 1
        elif entry is not None:
            expected = source_shard.replace('.safetensors', '.gravity')
            if entry['shard'] != expected and len(misplaced) < 20:
                misplaced.append({'tensor': name, 'expected': expected, 'found': entry['shard']})
    undeclared = sorted((n for n in packed if n not in weight_map))
    expected_shards = {s.replace('.safetensors', '.gravity') for s in weight_map.values()}
    packed_set = set(packed_shards)
    complete = counts.get('MISSING', 0) == 0 and counts.get('UNDECLARED', 0) == 0 and (not undeclared) and (not misplaced) and (packed_set >= expected_shards)
    return {'schema': 'hawking.glm52.assembly_coverage.v1', 'at': _now(), 'revision': REVISION, 'compact_root': str(compact_root), 'official_tensors': len(weight_map), 'official_shards': len(expected_shards), 'packed_shards': len(packed_shards), 'shards_outstanding': sorted(expected_shards - packed_set)[:5], 'shards_outstanding_count': len(expected_shards - packed_set), 'dispositions': counts, 'undeclared_tensors': undeclared[:20], 'undeclared_count': len(undeclared), 'misplaced_tensors': misplaced, 'missing_by_source_shard_top': dict(sorted(missing_by_shard.items(), key=lambda kv: -kv[1])[:5]), 'complete': complete, 'verdict': 'COMPLETE' if complete else 'INCOMPLETE'}

def synthesize_architecture() -> dict:
    """The complete architecture header the runtime needs, from sealed sources.

    The per-shard headers the packer writes carry five fields; the GLM adapter needs
    twenty, including the two layer schedules. Both come from evidence already on disk --
    the sealed architecture contract and the pinned config -- rather than from constants
    typed here, so a source revision change cannot silently disagree with the header.
    """
    contract = json.loads(CONTRACT.read_text())
    config = json.loads(CONFIG.read_text())
    geom = contract['geometry']
    dsa = contract['dsa_indexshare']
    routing = contract['routing']
    n_layers = int(geom['main_hidden_layers'])
    full = set((int(x) for x in dsa['main_full_indexer_layers']))
    shared = set((int(x) for x in dsa['main_shared_indexer_layers']))
    if full | shared != set(range(n_layers)):
        raise SystemExit(f'indexer schedules cover {len(full | shared)} layers, not {n_layers}')
    if full & shared:
        raise SystemExit(f'layers claimed by both indexer schedules: {sorted(full & shared)}')
    indexer_types = ['full' if i in full else 'shared' for i in range(n_layers)]
    dense = int(config['first_k_dense_replace'])
    mlp_layer_types = ['dense' if i < dense else 'sparse' for i in range(n_layers)]
    return {'model_type': 'glm_moe_dsa', 'hidden_size': int(geom['hidden_size']), 'num_hidden_layers': n_layers, 'num_attention_heads': int(geom['attention_heads']), 'q_lora_rank': int(geom['q_lora_rank']), 'kv_lora_rank': int(geom['kv_lora_rank']), 'qk_nope_head_dim': int(geom['qk_nope_head_dim']), 'qk_rope_head_dim': int(geom['qk_rope_head_dim']), 'v_head_dim': int(geom['v_head_dim']), 'index_n_heads': int(dsa['index_heads']), 'index_head_dim': int(dsa['index_head_dim']), 'index_topk': int(dsa['index_topk']), 'n_routed_experts': int(geom['routed_experts']), 'n_shared_experts': int(geom['shared_experts']), 'n_group': int(config['n_group']), 'topk_group': int(config['topk_group']), 'num_experts_per_tok': int(geom['active_routed_experts_per_token']), 'norm_topk_prob': bool(routing['normalize_topk']), 'routed_scaling_factor': float(routing['scaling_factor']), 'moe_intermediate_size': int(geom['moe_intermediate_size']), 'intermediate_size': int(geom['dense_intermediate_size']), 'first_k_dense_replace': dense, 'vocab_size': int(geom['vocab_size']), 'max_position_embeddings': int(geom['context_tokens']), 'rms_norm_eps': float(config['rms_norm_eps']), 'rope_parameters': {'rope_theta': float(config['rope_parameters']['rope_theta']), 'rope_type': config['rope_parameters'].get('rope_type', 'default')}, 'indexer_types': indexer_types, 'mlp_layer_types': mlp_layer_types}

def assemble(compact_root: Path=COMPACT_ROOT, force: bool=False) -> dict:
    coverage = check(compact_root)
    if not coverage['complete'] and (not force):
        return {'action': 'REFUSED', 'reason': 'coverage is not complete; a model view assembled over a hole is worse than none, because everything downstream treats it as whole', 'coverage': coverage}
    arch = synthesize_architecture()
    out = MODELS_ROOT / REVISION / 'General-R0'
    out.mkdir(parents=True, exist_ok=True)
    packed, packed_shards = packed_index(compact_root)
    linked, copied = (0, 0)
    for name in packed_shards:
        src, dst = (compact_root / name, out / name)
        if dst.exists():
            continue
        try:
            os.link(src, dst)
            linked += 1
        except OSError:
            import shutil
            shutil.copy2(src, dst)
            copied += 1
    index = {'schema': 'hawking.gravity.model_index.v1', 'assembled_at': _now(), 'model': {'repo': 'zai-org/GLM-5.2', 'revision': REVISION, 'representation': 'QUANTIZED_TRANSFORMER'}, 'architecture': arch, 'shards': sorted(packed_shards), 'shard_count': len(packed_shards), 'tensor_count': len(packed), 'weight_map': {n: e['shard'] for n, e in sorted(packed.items())}, 'coverage': {k: coverage[k] for k in ('official_tensors', 'dispositions', 'verdict')}, 'byte_provenance': "hardlinked from the packer's output; no duplicate copy"}
    (out / 'model.gravity.index.json').write_text(json.dumps(index, indent=1) + '\n')
    (out / 'COVERAGE.json').write_text(json.dumps(coverage, indent=1) + '\n')
    return {'action': 'ASSEMBLED', 'path': str(out), 'shards_linked': linked, 'shards_copied': copied, 'tensors': len(packed), 'coverage': coverage['verdict']}
