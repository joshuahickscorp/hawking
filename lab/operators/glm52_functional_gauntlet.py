"""Recomposed science module glm52_functional_gauntlet (C-SCI-R1)."""
from __future__ import annotations
from lab.operators import hawking_null_metric as metric
from lab.operators import glm52_moe_student as student
from lab.operators import glm52_teacher_capture as capture
from lab.layout import REPORTS_ROOT
import sys as _sys_a1
from pathlib import Path as _Path_a1
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
_A1_HERE = _Path_a1(__file__).resolve().parent
_A1_CONDENSE = _A1_HERE.parent if _A1_HERE.name == 'archive' else _A1_HERE
_A1_REPO = _A1_CONDENSE.parents[1]
if str(_A1_CONDENSE) not in _sys_a1.path:
    _sys_a1.path.insert(0, str(_A1_CONDENSE))
CONDENSE = _A1_CONDENSE
ROOT = _A1_REPO
SUPPORT = Path(os.environ.get('GLM52_SUPPORT_ROOT', '/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity'))
CAPSULES = SUPPORT / 'source_fetch' / 'teacher' / 'capsules_generation_b'
OUT = REPORTS_ROOT / 'condense' / 'glm52_generation_b'
LEDGER = OUT / 'GLM52_FUNCTIONAL_EXPERIMENT_LEDGER.jsonl'
FIT_SPLITS = ('teacher_fit', 'teacher_router', 'teacher_doctor')
SCORE_SPLIT = 'teacher_score'
REPLICATION_SPLITS = ('teacher_cv', 'teacher_holdout', 'teacher_replication', 'teacher_protected', 'teacher_longctx')
STRATA = {5: 'early', 38: 'middle', 74: 'late', 77: 'final'}
HIDDEN = 6144
MOE_INTERMEDIATE = 2048
ROUTED_EXPERTS = 256
ROUTED_WEIGHTS = ROUTED_EXPERTS * MOE_INTERMEDIATE * HIDDEN * 3
SHARED_WEIGHTS = MOE_INTERMEDIATE * HIDDEN * 3
ROUTER_WEIGHTS = HIDDEN * ROUTED_EXPERTS + ROUTED_EXPERTS
REPLACED_WEIGHTS = ROUTED_WEIGHTS + SHARED_WEIGHTS + ROUTER_WEIGHTS
PRIMARY_HIDDEN = 1024
SEEDS = (17, 101, 202, 303)

class GauntletError(RuntimeError):
    pass

def capsule(layer: int, split: str) -> Path:
    name = f'L{layer:02d}_L{layer:02d}.npz'
    return CAPSULES / name if split == 'teacher_fit' else CAPSULES / split / name

def available_splits(layer: int) -> list[str]:
    return [split for split in FIT_SPLITS + (SCORE_SPLIT,) + REPLICATION_SPLITS if capsule(layer, split).exists()]

def arrays(layer: int, split: str, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    path = capsule(layer, split)
    if not path.exists():
        raise GauntletError(f'no capsule for layer {layer} split {split}')
    with np.load(path) as data:
        out = {}
        for key in keys:
            name = key if key.startswith('carry_') else f'layer_{layer:02d}/{key}'
            out[key] = np.asarray(data[name], dtype=np.float32 if 'index' not in key else data[name].dtype)
    return out

def pairs(layer: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    got = arrays(layer, split, ('pre_router_hidden', 'post_moe'))
    return (got['pre_router_hidden'].reshape(-1, HIDDEN), got['post_moe'].reshape(-1, HIDDEN))

def fit_evidence(layer: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = ([], [])
    for split in FIT_SPLITS:
        if capsule(layer, split).exists():
            x, y = pairs(layer, split)
            xs.append(x)
            ys.append(y)
    if not xs:
        raise GauntletError(f'layer {layer} has no fit evidence')
    return (np.concatenate(xs), np.concatenate(ys))

def candidate_student(x: np.ndarray, y: np.ndarray, *, hidden: int, seed: int) -> dict:
    fitted = student.fit(x, y, hidden=hidden, seed=seed, replaced_weights=REPLACED_WEIGHTS)
    blob = fitted['blob']
    return {'family': 'random_feature_student', 'label': f'student_h{hidden}_s{seed}', 'bytes': len(blob), 'predict': lambda z, blob=blob: student.apply_student(blob, z), 'blob': blob, 'hidden': hidden, 'seed': seed, 'ridge': fitted['ridge'], 'stored_parameters': fitted['parameters_stored'], 'expanded_bytes': len(blob) + HIDDEN * hidden * 4}

def _normal_equations(a: np.ndarray, b: np.ndarray):
    return (a.T @ a, a.T @ b)

def _solve(gram: np.ndarray, cross: np.ndarray, ridge: float) -> np.ndarray:
    regularized = gram.copy()
    regularized[np.diag_indices_from(regularized)] += np.float32(ridge)
    return np.linalg.solve(regularized, cross)

def _ridge_solve(a: np.ndarray, b: np.ndarray, ridge: float) -> np.ndarray:
    return _solve(*_normal_equations(a, b), ridge)

def _pick_ridge(a: np.ndarray, b: np.ndarray, holdout: float=0.2) -> float:
    """Chosen on a split of the fit data, never on the score split.

    The Gram matrix is formed once: it does not depend on the ridge, and reforming it per
    grid point is the dominant cost at 6144 columns.
    """
    cut = int(a.shape[0] * (1.0 - holdout))
    gram, cross = _normal_equations(a[:cut], b[:cut])
    best, best_error = (student.RIDGE_GRID[0], np.inf)
    for ridge in student.RIDGE_GRID:
        error = float(np.linalg.norm(a[cut:] @ _solve(gram, cross, ridge) - b[cut:]))
        if error < best_error:
            best, best_error = (ridge, error)
    return best

def candidate_linear(x: np.ndarray, y: np.ndarray, *, affine: bool) -> dict:
    """The upper control.  If a dense map does no better, the student is the whole story."""
    a = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1) if affine else x
    ridge = _pick_ridge(a, y)
    weight = _ridge_solve(a, y, ridge).astype(np.float16)
    payload = weight.nbytes + 64

    def predict(z, weight=weight, affine=affine):
        flat = z.reshape(-1, z.shape[-1]).astype(np.float32)
        if affine:
            flat = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1)
        return (flat @ weight.astype(np.float32)).reshape(*z.shape[:-1], HIDDEN)
    return {'family': 'affine_upper_control' if affine else 'linear_upper_control', 'label': 'affine_full' if affine else 'linear_full', 'bytes': payload, 'predict': predict, 'ridge': ridge, 'stored_parameters': int(weight.size), 'expanded_bytes': payload}

def candidate_lowrank(x: np.ndarray, y: np.ndarray, *, rank: int, seed: int=0) -> dict:
    """Structured student: a seeded random projection down to ``rank``, then a readout.

    The projection costs a seed, so the payload is the [rank, 6144] readout alone.  This is
    the same trick as the random-feature student without the nonlinearity, and it is the
    cheapest thing in the auction that still fits a map.
    """
    generator = np.random.default_rng(seed ^ 24301)
    projection = generator.standard_normal((HIDDEN, rank), dtype=np.float32) / np.float32(np.sqrt(HIDDEN))
    reduced = x @ projection
    ridge = _pick_ridge(reduced, y)
    readout = _ridge_solve(reduced, y, ridge).astype(np.float16)
    payload = readout.nbytes + 64

    def predict(z, projection=projection, readout=readout):
        flat = z.reshape(-1, z.shape[-1]).astype(np.float32)
        return (flat @ projection @ readout.astype(np.float32)).reshape(*z.shape[:-1], HIDDEN)
    return {'family': 'structured_linear_student', 'label': f'linear_rank{rank}_s{seed}', 'bytes': payload, 'predict': predict, 'rank': rank, 'seed': seed, 'ridge': ridge, 'stored_parameters': int(readout.size), 'expanded_bytes': payload + HIDDEN * rank * 4}

def candidate_structured_student(x: np.ndarray, y: np.ndarray, *, hidden: int, rank: int, seed: int) -> dict:
    """The small structured student: random features, then a rank-limited readout.

    The readout is the only stored tensor in the random-feature student and it is
    [hidden, 6144].  Factoring it as [hidden, rank] @ [rank, 6144] is the one structural
    knob that lowers the rate without changing the feature map, which keeps the comparison
    against the unfactored student honest.
    """
    phi = student.features(x, seed, hidden)
    ridge = _pick_ridge(phi, y)
    full = _ridge_solve(phi, y, ridge)
    u, s, vt = np.linalg.svd(full, full_matrices=False)
    left = (u[:, :rank] * s[:rank]).astype(np.float16)
    right = vt[:rank].astype(np.float16)
    payload = left.nbytes + right.nbytes + 64

    def predict(z, seed=seed, hidden=hidden, left=left, right=right):
        flat = z.reshape(-1, z.shape[-1]).astype(np.float32)
        out = student.features(flat, seed, hidden) @ left.astype(np.float32)
        return (out @ right.astype(np.float32)).reshape(*z.shape[:-1], HIDDEN)
    return {'family': 'structured_random_feature_student', 'label': f'structured_h{hidden}_r{rank}_s{seed}', 'bytes': payload, 'predict': predict, 'hidden': hidden, 'rank': rank, 'seed': seed, 'ridge': ridge, 'stored_parameters': int(left.size + right.size), 'expanded_bytes': payload + HIDDEN * hidden * 4}

def bpw(payload_bytes: int) -> float:
    return payload_bytes * 8 / REPLACED_WEIGHTS

def evaluate(candidate: dict, x_score: np.ndarray, y_score: np.ndarray, null: dict) -> dict:
    prediction = candidate['predict'](x_score)
    scored = metric.score(y_score, prediction, null)
    return {'label': candidate['label'], 'family': candidate['family'], 'bytes': candidate['bytes'], 'local_bpw': bpw(candidate['bytes']), 'expanded_bytes': candidate.get('expanded_bytes', candidate['bytes']), 'stored_parameters': candidate.get('stored_parameters'), **{k: v for k, v in scored.items() if k != 'schema'}}

def controls(x_fit, y_fit, x_score, y_score, null, candidate) -> dict:
    """The controls the contract makes mandatory, run against the same null."""
    constant = np.broadcast_to(null['mean'].astype(np.float32), y_score.shape)
    rows = np.random.default_rng(9).permutation(x_score.shape[0])
    return {'mean_null': {'bytes': HIDDEN * 2, 'local_bpw': bpw(HIDDEN * 2), **{k: v for k, v in metric.score(y_score, constant, null).items() if k != 'schema'}}, 'identity_passthrough': {'bytes': 0, 'local_bpw': 0.0, **{k: v for k, v in metric.score(y_score, x_score, null).items() if k != 'schema'}}, 'shuffled_input': {'bytes': candidate['bytes'], 'local_bpw': bpw(candidate['bytes']), **{k: v for k, v in metric.score(y_score, candidate['predict'](x_score[rows]), null).items() if k != 'schema'}}, 'constant_null_raw_cosine': metric.constant_null_raw_cosine(y_score, null)}

def record(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with LEDGER.open('a') as handle:
        handle.write(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **row}, sort_keys=True) + '\n')

def fs0(layer: int, *, hidden: int=PRIMARY_HIDDEN, seeds=SEEDS) -> dict:
    """Reproduce the positive row in a fresh process with fresh seeds and every control."""
    x_fit, y_fit = fit_evidence(layer)
    x_score, y_score = pairs(layer, SCORE_SPLIT)
    null = metric.fit_null(y_fit)
    rows, primary = ([], None)
    for seed in seeds:
        made = candidate_student(x_fit, y_fit, hidden=hidden, seed=seed)
        rows.append(evaluate(made, x_score, y_score, null))
        primary = primary or made
    for control in (candidate_linear(x_fit, y_fit, affine=False), candidate_linear(x_fit, y_fit, affine=True), candidate_lowrank(x_fit, y_fit, rank=64), candidate_lowrank(x_fit, y_fit, rank=256)):
        rows.append(evaluate(control, x_score, y_score, null))
    skills = [row['skill'] for row in rows if row['label'].startswith(f'student_h{hidden}')]
    result = {'stage': 'FS0', 'layer': layer, 'target': 'pre_router_hidden -> post_moe', 'fit_positions': int(x_fit.shape[0]), 'score_positions': int(x_score.shape[0]), 'fit_splits': [s for s in FIT_SPLITS if capsule(layer, s).exists()], 'score_split': SCORE_SPLIT, 'candidates': rows, 'controls': controls(x_fit, y_fit, x_score, y_score, null, primary), 'seed_spread': {'min': min(skills), 'max': max(skills), 'range': max(skills) - min(skills), 'seeds': list(seeds)}, 'replaced_weights': REPLACED_WEIGHTS}
    record({'stage': 'FS0', 'layer': layer, 'primary_skill': rows[0]['skill'], 'primary_centered': rows[0]['centered_cosine'], 'primary_skill_lower': rows[0]['skill_lower'], 'seed_range': result['seed_spread']['range']})
    return result

def fs1(layer: int, *, hidden: int=PRIMARY_HIDDEN, seed: int=SEEDS[0]) -> dict:
    """Put the student inside the real block.

    ``block_output = post_attention_hidden + post_moe`` exactly in the sealed reference, so
    the block-level score needs no re-forward: it needs the block-level null, which is a
    different and much harder null than the MoE-output null because the residual carries
    most of the norm.
    """
    x_fit, y_fit = fit_evidence(layer)
    fitted = candidate_student(x_fit, y_fit, hidden=hidden, seed=seed)
    linear = candidate_linear(x_fit, y_fit, affine=False)
    got = arrays(layer, SCORE_SPLIT, ('pre_router_hidden', 'post_moe', 'post_attention_hidden', 'block_output'))
    residual = got['post_attention_hidden'].reshape(-1, HIDDEN)
    teacher_block = got['block_output'].reshape(-1, HIDDEN)
    x_score = got['pre_router_hidden'].reshape(-1, HIDDEN)
    fit_block = arrays(layer, FIT_SPLITS[0], ('block_output',))['block_output'].reshape(-1, HIDDEN)
    block_null = metric.fit_null(fit_block)

    def block_of(prediction):
        return residual + prediction.reshape(-1, HIDDEN)
    rows = []
    for candidate in (fitted, linear):
        prediction = candidate['predict'](x_score)
        scored = metric.score(teacher_block, block_of(prediction), block_null)
        rows.append({'label': candidate['label'], 'bytes': candidate['bytes'], 'local_bpw': bpw(candidate['bytes']), **{k: v for k, v in scored.items() if k != 'schema'}})
    moe_fit_mean = metric.fit_null(fit_evidence(layer)[1])['mean'].astype(np.float32)
    weight_space_like = metric.score(teacher_block, block_of(np.broadcast_to(moe_fit_mean, x_score.shape)), block_null)
    residual_only = metric.score(teacher_block, residual, block_null)
    result = {'stage': 'FS1', 'layer': layer, 'target': 'block_output', 'block_identity': 'block_output = post_attention_hidden + post_moe, exact in the sealed reference', 'score_positions': int(teacher_block.shape[0]), 'candidates': rows, 'controls': {'block_mean_null': {'skill': 0.0, 'note': 'the reference null'}, 'residual_only_no_moe': {k: v for k, v in residual_only.items() if k != 'schema'}, 'moe_constant_weight_space_equivalent': {k: v for k, v in weight_space_like.items() if k != 'schema'}, 'constant_null_raw_cosine': metric.constant_null_raw_cosine(teacher_block, block_null)}, 'gate': 'student block output beats the block mean-null and the weight-space control with a positive held-out confidence lower bound', 'passes': bool(rows[0]['passes'] and rows[0]['skill_lower'] > weight_space_like['skill'] and (rows[0]['skill'] > residual_only['skill']))}
    record({'stage': 'FS1', 'layer': layer, 'skill': rows[0]['skill'], 'centered': rows[0]['centered_cosine'], 'passes': result['passes'], 'residual_only_skill': residual_only['skill']})
    return result

def _forward_next_layer(layer: int, hidden_in: np.ndarray, previous_topk: np.ndarray):
    """One real block, driven from an arbitrary input state."""
    graph = capture._graph()
    config = capture.official_config()
    source = capture.ShardTensorSource(capture.SOURCE_ROOT, capture._tensor_table(graph))
    cache = __import__('glm52_reference').ReferenceCache()
    _, _, produced = capture.capture_layer(np.asarray(hidden_in, dtype=np.float32), source, layer, config, np.asarray(previous_topk), cache)
    return produced

def fs2(layer: int, *, hidden: int=PRIMARY_HIDDEN, seed: int=SEEDS[0]) -> dict:
    """Carry the perturbation one complete layer forward, through real attention and a
    real router.

    Both forwards are run here rather than read from capsules, because the captured
    layer-(n+1) capsule was seeded from the embedding table and is not the continuation of
    layer n.  Teacher and student differ only in the MoE output of layer n.
    """
    following = layer + 1
    x_fit, y_fit = fit_evidence(layer)
    fitted = candidate_student(x_fit, y_fit, hidden=hidden, seed=seed)
    got = arrays(layer, SCORE_SPLIT, ('pre_router_hidden', 'post_attention_hidden', 'block_output'))
    with np.load(capsule(layer, SCORE_SPLIT)) as data:
        carry_topk = np.asarray(data['carry_out_index_selection'])
    shape = got['block_output'].shape
    teacher_in = got['block_output']
    student_in = got['post_attention_hidden'] + fitted['predict'](got['pre_router_hidden']).reshape(shape)
    started = time.time()
    teacher = _forward_next_layer(following, teacher_in, carry_topk)
    student_side = _forward_next_layer(following, student_in, carry_topk)
    fit_next = _forward_next_layer(following, arrays(layer, FIT_SPLITS[0], ('block_output',))['block_output'], np.load(capsule(layer, FIT_SPLITS[0]))['carry_out_index_selection'])

    def compare(key):
        null = metric.fit_null(fit_next[key].reshape(-1, fit_next[key].shape[-1]))
        scored = metric.score(teacher[key].reshape(-1, teacher[key].shape[-1]), student_side[key].reshape(-1, student_side[key].shape[-1]), null)
        return {k: v for k, v in scored.items() if k != 'schema'}
    routing = {}
    if 'topk_indices' in teacher:
        t_idx = teacher['topk_indices'].reshape(-1, 8)
        s_idx = student_side['topk_indices'].reshape(-1, 8)
        overlap = np.array([len(set(a) & set(b)) for a, b in zip(t_idx, s_idx)])
        t_margin = teacher['topk_margin_8th_vs_9th'].reshape(-1)
        low = t_margin <= np.quantile(t_margin, 0.1)
        routing = {'mean_topk_overlap_of_8': float(overlap.mean()), 'fraction_identical_topk_set': float((overlap == 8).mean()), 'top1_agreement': float((t_idx[:, 0] == s_idx[:, 0]).mean()), 'mean_overlap_on_lowest_decile_margin': float(overlap[low].mean()), 'teacher_margin_median': float(np.median(t_margin))}
    drift = {key: float(np.linalg.norm(student_side[key] - teacher[key]) / max(np.linalg.norm(teacher[key]), 1e-12)) for key in ('attention_output', 'pre_router_hidden', 'post_moe', 'block_output') if key in teacher}
    entering = float(np.linalg.norm(student_in - teacher_in) / max(np.linalg.norm(teacher_in), 1e-12))
    result = {'stage': 'FS2', 'layer': layer, 'propagated_through': following, 'entering_relative_l2': entering, 'leaving_relative_l2': drift.get('block_output'), 'amplification': drift.get('block_output', 0.0) / entering if entering else None, 'relative_l2_by_stage': drift, 'next_layer_scores': {key: compare(key) for key in ('post_moe', 'block_output') if key in teacher}, 'routing': routing, 'seconds': round(time.time() - started, 1), 'note': 'teacher and student forwards differ only in the MoE output of the perturbed layer; both use the same captured indexer carry'}
    result['passes'] = bool(result['next_layer_scores']['block_output']['skill_lower'] > 0.0 and (result['amplification'] is None or result['amplification'] <= 1.0))
    record({'stage': 'FS2', 'layer': layer, 'through': following, 'entering': entering, 'leaving': result['leaving_relative_l2'], 'amplification': result['amplification'], 'passes': result['passes']})
    return result

def fs3(layers=tuple(STRATA)) -> dict:
    """Fit and score each stratum independently.  Layer 38 is one row, not the law."""
    rows = []
    for layer in layers:
        if not capsule(layer, SCORE_SPLIT).exists():
            rows.append({'layer': layer, 'stratum': STRATA.get(layer), 'status': 'NO_EVIDENCE'})
            continue
        x_fit, y_fit = fit_evidence(layer)
        x_score, y_score = pairs(layer, SCORE_SPLIT)
        null = metric.fit_null(y_fit)
        made = [candidate_student(x_fit, y_fit, hidden=PRIMARY_HIDDEN, seed=SEEDS[0]), candidate_linear(x_fit, y_fit, affine=False), candidate_structured_student(x_fit, y_fit, hidden=PRIMARY_HIDDEN, rank=256, seed=SEEDS[0]), candidate_lowrank(x_fit, y_fit, rank=256)]
        rows.append({'layer': layer, 'stratum': STRATA.get(layer), 'status': 'SCORED', 'fit_positions': int(x_fit.shape[0]), 'constant_null_raw_cosine': metric.constant_null_raw_cosine(y_score, null), 'candidates': [evaluate(c, x_score, y_score, null) for c in made], 'controls': controls(x_fit, y_fit, x_score, y_score, null, made[0])})
        record({'stage': 'FS3', 'layer': layer, 'stratum': STRATA.get(layer), 'student_skill': rows[-1]['candidates'][0]['skill'], 'linear_skill': rows[-1]['candidates'][1]['skill']})
    scored = [row for row in rows if row['status'] == 'SCORED']
    return {'stage': 'FS3', 'strata': rows, 'layers_scored': [row['layer'] for row in scored], 'all_strata_pass': bool(scored) and all((row['candidates'][0]['passes'] for row in scored)), 'skill_range': [min((row['candidates'][0]['skill'] for row in scored)), max((row['candidates'][0]['skill'] for row in scored))] if scored else None}

def fs5(layer: int, *, hidden: int=PRIMARY_HIDDEN) -> dict:
    """Replication with no refit: other documents, other domains, other student seeds."""
    x_fit, y_fit = fit_evidence(layer)
    null = metric.fit_null(y_fit)
    made = {seed: candidate_student(x_fit, y_fit, hidden=hidden, seed=seed) for seed in SEEDS}
    rows = []
    for split in (SCORE_SPLIT,) + REPLICATION_SPLITS:
        if not capsule(layer, split).exists():
            continue
        x_split, y_split = pairs(layer, split)
        for seed, candidate in made.items():
            scored = evaluate(candidate, x_split, y_split, null)
            rows.append({'split': split, 'seed': seed, 'refit': False, 'constant_null_raw_cosine': metric.constant_null_raw_cosine(y_split, null), **scored})
    passing = [row for row in rows if row['passes']]
    result = {'stage': 'FS5', 'layer': layer, 'rows': rows, 'splits_tested': sorted({row['split'] for row in rows}), 'seeds_tested': list(SEEDS), 'fraction_passing': len(passing) / max(len(rows), 1), 'skill_range': [min((r['skill'] for r in rows)), max((r['skill'] for r in rows))] if rows else None, 'all_pass': bool(rows) and len(passing) == len(rows)}
    record({'stage': 'FS5', 'layer': layer, 'splits': len(result['splits_tested']), 'all_pass': result['all_pass'], 'skill_range': result['skill_range']})
    return result

def fs4(layers=tuple(STRATA)) -> dict:
    """Does anything transfer between layers, or is each MoE its own function?"""
    usable = [layer for layer in layers if capsule(layer, SCORE_SPLIT).exists()]
    if len(usable) < 2:
        return {'stage': 'FS4', 'status': 'INSUFFICIENT_LAYERS', 'layers': usable}
    evidence = {layer: (fit_evidence(layer), pairs(layer, SCORE_SPLIT)) for layer in usable}
    per_layer = {}
    for layer, ((x_fit, y_fit), (x_score, y_score)) in evidence.items():
        null = metric.fit_null(y_fit)
        made = candidate_student(x_fit, y_fit, hidden=PRIMARY_HIDDEN, seed=SEEDS[0])
        per_layer[layer] = {'candidate': made, 'null': null, 'own': evaluate(made, x_score, y_score, null)}
    cross = []
    for source_layer in usable:
        for target_layer in usable:
            if source_layer == target_layer:
                continue
            (_, _), (x_score, y_score) = evidence[target_layer]
            scored = evaluate(per_layer[source_layer]['candidate'], x_score, y_score, per_layer[target_layer]['null'])
            cross.append({'fitted_on': source_layer, 'scored_on': target_layer, 'skill': scored['skill'], 'centered_cosine': scored['centered_cosine'], 'own_layer_skill': per_layer[target_layer]['own']['skill'], 'retained_fraction': scored['skill'] / max(per_layer[target_layer]['own']['skill'], 1e-09)})
    shared_seed = SEEDS[0]
    shared_rows = []
    for layer in usable:
        (x_fit, y_fit), (x_score, y_score) = evidence[layer]
        made = candidate_student(x_fit, y_fit, hidden=PRIMARY_HIDDEN, seed=shared_seed)
        shared_rows.append({'layer': layer, 'bytes': made['bytes'], **{k: v for k, v in evaluate(made, x_score, y_score, per_layer[layer]['null']).items() if k in ('skill', 'centered_cosine', 'local_bpw')}})
    retained = [row['retained_fraction'] for row in cross]
    result = {'stage': 'FS4', 'layers': usable, 'shared_feature_map_seed': shared_seed, 'per_layer_readout': shared_rows, 'cross_layer_application': cross, 'mean_retained_fraction': float(np.mean(retained)) if retained else None, 'max_retained_fraction': float(np.max(retained)) if retained else None, 'verdict': 'READOUT_IS_LAYER_SPECIFIC' if retained and max(retained) < 0.5 else 'SOME_CROSS_LAYER_TRANSFER', 'bytes_saved_by_sharing_readout': 0, 'note': 'the feature map is a shared seed and costs 4 bytes per layer either way; only the readout is a real byte, so sharing it is the only saving on offer'}
    record({'stage': 'FS4', 'layers': usable, 'verdict': result['verdict'], 'max_retained': result['max_retained_fraction']})
    return result

def _emit(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    print(f'wrote {path}', file=sys.stderr)
