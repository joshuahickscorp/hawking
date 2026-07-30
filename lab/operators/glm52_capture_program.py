"""Recomposed science module glm52_capture_program (C-SCI-R1)."""
from __future__ import annotations
from lab.operators import glm52_corpus as corpus
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REPORTS = REPO / 'reports/condense/glm52_generation_b'
SPLIT_PARTITIONS = {'teacher_fit': 'representation fit', 'teacher_router': 'router/indexer calibration', 'teacher_doctor': 'Doctor training', 'teacher_cv': 'cross-validation', 'teacher_score': 'score', 'teacher_holdout': 'held-out', 'teacher_replication': 'replication', 'teacher_protected': 'protected-domain holdout', 'teacher_longctx': 'long-context holdout'}
MAX_SEQUENCE = int(os.environ.get('GLM52_CAPTURE_SEQUENCE', '256'))
MAX_RECORDS = int(os.environ.get('GLM52_CAPTURE_RECORDS', '16'))
PAD_ID = 0

def capsule_bytes_per_layer(records: int=MAX_RECORDS, sequence: int=MAX_SEQUENCE, *, hidden: int=6144, trajectories: int=8) -> int:
    """What one layer's capsule will cost at this profile, before it is captured."""
    return trajectories * records * sequence * hidden * 4
_BUNDLE = None
_RECORDS = None

def _corpus():
    global _BUNDLE, _RECORDS
    if _RECORDS is None:
        _BUNDLE = corpus.load_pinned_tokenizer()
        _RECORDS = corpus.build_records(_BUNDLE)
    return (_BUNDLE, _RECORDS)

def records_for(split: str) -> list:
    if split not in SPLIT_PARTITIONS:
        raise KeyError(f'unknown capture split: {split!r}')
    _, records = _corpus()
    partition = SPLIT_PARTITIONS[split]
    chosen = [record for record in records if record.partition == partition]
    chosen.sort(key=lambda record: record.record_id)
    return chosen[:MAX_RECORDS]

def batch_for(split: str) -> tuple[np.ndarray, list[dict]]:
    """One right-padded [records, sequence] batch of real token ids, plus its provenance."""
    bundle, _ = _corpus()
    rows, meta = ([], [])
    for record in records_for(split):
        text = record.context_window if record.context_rung_tokens else record.prompt
        ids = list(corpus._encode(bundle, text))[:MAX_SEQUENCE]
        meta.append({'record_id': record.record_id, 'domain': record.domain, 'partition': record.partition, 'kind': record.kind, 'source': 'context_window' if record.context_rung_tokens else 'prompt', 'tokens': len(ids), 'context_rung_tokens': record.context_rung_tokens, 'token_ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()})
        rows.append(ids)
    if not rows:
        raise ValueError(f'no records for split {split!r}')
    width = max((len(row) for row in rows))
    batch = np.full((len(rows), width), PAD_ID, dtype=np.int64)
    for index, row in enumerate(rows):
        batch[index, :len(row)] = row
    return (batch, meta)

def membership_sha256(split: str) -> str:
    batch, _ = batch_for(split)
    return hashlib.sha256(json.dumps({'split': split, 'token_ids': batch.tolist()}, separators=(',', ':')).encode()).hexdigest()
MATH_DOMAIN = 'mathematics'

def math_calibration_records() -> list:
    _, records = _corpus()
    chosen = [record for record in records if record.domain == MATH_DOMAIN and record.partition in corpus.TRAIN_PARTITIONS]
    chosen.sort(key=lambda record: record.record_id)
    return chosen

def math_calibration_batch() -> tuple[np.ndarray, list[dict]]:
    """Same shape/provenance contract as `batch_for`, pooled instead of single-partition."""
    bundle, _ = _corpus()
    rows, meta = ([], [])
    for record in math_calibration_records():
        text = record.context_window if record.context_rung_tokens else record.prompt
        ids = list(corpus._encode(bundle, text))[:MAX_SEQUENCE]
        meta.append({'record_id': record.record_id, 'domain': record.domain, 'partition': record.partition, 'kind': record.kind, 'source': 'context_window' if record.context_rung_tokens else 'prompt', 'tokens': len(ids), 'context_rung_tokens': record.context_rung_tokens, 'token_ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()})
        rows.append(ids)
    if not rows:
        raise ValueError('no mathematics records in TRAIN_PARTITIONS')
    width = max((len(row) for row in rows))
    batch = np.full((len(rows), width), PAD_ID, dtype=np.int64)
    for index, row in enumerate(rows):
        batch[index, :len(row)] = row
    return (batch, meta)

def math_calibration_membership_sha256() -> str:
    batch, _ = math_calibration_batch()
    return hashlib.sha256(json.dumps({'split': 'math_calibration_v1', 'token_ids': batch.tolist()}, separators=(',', ':')).encode()).hexdigest()

def build() -> int:
    _, records = _corpus()
    splits = {}
    for split in SPLIT_PARTITIONS:
        batch, meta = batch_for(split)
        domains = Counter((row['domain'] for row in meta))
        splits[split] = {'corpus_partition': SPLIT_PARTITIONS[split], 'records': len(meta), 'batch_shape': list(batch.shape), 'real_tokens': int(sum((row['tokens'] for row in meta))), 'padded_positions': int(batch.size - sum((row['tokens'] for row in meta))), 'domains': dict(sorted(domains.items())), 'membership_sha256': membership_sha256(split), 'members': meta}
    overlaps = {}
    id_sets = {split: {row['token_ids_sha256'] for row in splits[split]['members']} for split in splits}
    for left in id_sets:
        for right in id_sets:
            if left < right and id_sets[left] & id_sets[right]:
                overlaps[f'{left}|{right}'] = sorted(id_sets[left] & id_sets[right])
    payload = {'schema': 'hawking.glm52.generation_b_capture_program.v1', 'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'revision': corpus.REVISION, 'supersedes': {'what': 'synthetic SHA-256 token ids, 8 positions, 2 splits', 'why': 'an eight-token random probe exercises no domain, no language, no router margin and no context length; a representation fitted against it has been fitted against noise'}, 'tokenizer': {'path': str(corpus.DEFAULT_TOKENIZER_PATH), 'vocab_size': corpus.TOKENIZER_VOCAB_SIZE, 'bytes': corpus.TOKENIZER_BYTES}, 'bounds': {'max_sequence': MAX_SEQUENCE, 'max_records': MAX_RECORDS, 'pad_id': PAD_ID, 'capsule_bytes_per_layer': capsule_bytes_per_layer(), 'env_overrides': ['GLM52_CAPTURE_RECORDS', 'GLM52_CAPTURE_SEQUENCE'], 'note': 'a capsule holds eight [records, sequence, 6144] float32 trajectories, so its size is linear in records*sequence; the frozen program pins the production profile'}, 'corpus_records_total': len(records), 'domains_available': list(corpus.DOMAINS), 'splits': splits, 'split_token_overlaps': overlaps, 'splits_disjoint': not overlaps, 'train_splits': ['teacher_fit', 'teacher_router', 'teacher_doctor'], 'evaluation_splits': ['teacher_cv', 'teacher_score', 'teacher_holdout', 'teacher_replication', 'teacher_protected', 'teacher_longctx'], 'promotion_rule': 'a candidate is fitted only on train splits and promoted only on evaluation splits; replication must hold without refit'}
    REPORTS.mkdir(parents=True, exist_ok=True)
    target = REPORTS / 'GLM52_GENERATION_B_CAPTURE_PROGRAM.json'
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({'wrote': str(target.relative_to(REPO)), 'splits': {split: {'records': splits[split]['records'], 'shape': splits[split]['batch_shape'], 'real_tokens': splits[split]['real_tokens'], 'domains': len(splits[split]['domains'])} for split in splits}, 'splits_disjoint': payload['splits_disjoint'], 'domains_covered': len({d for s in splits.values() for d in s['domains']})}, indent=2))
    return 0
