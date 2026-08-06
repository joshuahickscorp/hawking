"""Shared path constants for the Odyssey T0 and data tools."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.layout import (
    ODYSSEY_DOMAINS_ROOT,
    ODYSSEY_PROGRAM_ROOT,
    ODYSSEY_RECORDS_ROOT,
    ODYSSEY_RESOURCES_ROOT,
    ODYSSEY_ROOT,
    ODYSSEY_STATE_ROOT,
    REPO_ROOT,
    odyssey_path,
)

ROOT = REPO_ROOT
ODYSSEY = ODYSSEY_ROOT
DOMAINS_DIR = ODYSSEY_DOMAINS_ROOT
PROGRAM_DIR = ODYSSEY_PROGRAM_ROOT
RESOURCES_DIR = ODYSSEY_RESOURCES_ROOT
STATE_DIR = ODYSSEY_STATE_ROOT
RECORDS_DIR = ODYSSEY_RECORDS_ROOT

# --- T0 reproduction ------------------------------------------------------
T0_DIR = odyssey_path("t0")
T0_STATE = T0_DIR / "state"
CHECKPOINTS = odyssey_path("checkpoints")

# Sealed substrate facts (verified live; do not re-derive).
MATH_ARTIFACT = Path.home() / (
    "Library/Application Support/Hawking/Models/GLM-5.2/"
    "b4734de4facf877f85769a911abafc5283eab3d9/"
    "GLM-5.2-H0.98-Math-Preserve.gravity"
)
EXPECTED_INDEX_SHA256 = "33d40c254eb982d4a495f5f0792a116e9d9810d937f5f3969f4f84742b2364d9"
EXPECTED_MANIFEST_SHA256 = "b34596f5d4df0b09903845302648736ee2345d7662688176c851a4d749211a83"
EXPECTED_SHARD_COUNT = 282
EXPECTED_DECISION_COUNT = 59585
EXPECTED_BPW = 0.9774017488417455
EXPECTED_BYTES = 92_038_250_160

LAUNCH_DIR = odyssey_path("launch")
TRAINING_DIR = odyssey_path("training")
FENCE = LAUNCH_DIR / "ODYSSEY_LAUNCH_AUTHORIZED"
STOP = LAUNCH_DIR / "STOP"

# --- data membership, inventory, contamination barrier --------------------
DATA_DIR = odyssey_path("data")
EVAL_DIR = odyssey_path("evaluation")
HIDDEN_DIR = EVAL_DIR / "hidden"
PUBLIC_EVAL_DIR = T0_DIR / "public_eval"
TEACHER_DIR = odyssey_path("teacher_traces")
FIXTURE_DIR = DATA_DIR / "fixtures" / "ingestion_fixture_v0"
MEMBERSHIP_DIR = DATA_DIR / "membership"

DATA_MANIFEST = DATA_DIR / "ODYSSEY_DATA_MANIFEST.json"
TEACHER_MANIFEST = TEACHER_DIR / "ODYSSEY_TEACHER_TRACE_MANIFEST.json"
SUPPORT_HALO_CORPUS = EVAL_DIR / "support_halo_corpus_v0.jsonl"
SUPPORT_HALO_SEAL = EVAL_DIR / "SUPPORT_HALO_SEAL.json"
HIDDEN_ITEMS = HIDDEN_DIR / "hidden_items.jsonl"
HIDDEN_COMMITMENT = HIDDEN_DIR / "HIDDEN_MEMBERSHIP_COMMITMENT.json"
PUBLIC_SELECTION = PUBLIC_EVAL_DIR / "selection_items.jsonl"

# Sealed support-halo corpus hash (load-bearing; do not recompute as truth).
EXPECTED_SUPPORT_HALO_CORPUS_SHA256 = (
    "b3ebda04ce48aa84b51faf47bff6284083029e0517b33e8c7c3f55b5fb54ec67"
)

# Character-shingle Jaccard thresholds (aligned with GLM52 train-vs-eval policy).
JACCARD_TRAIN_VS_EVAL = 0.25
JACCARD_WITHIN_CORPUS_NEAR_DUP = 0.80
SHINGLE_SIZE = 5
