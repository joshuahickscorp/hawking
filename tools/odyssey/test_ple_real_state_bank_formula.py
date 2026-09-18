from __future__ import annotations

import hashlib
import json

import numpy as np

from tools.odyssey.ple_context_response_screen import _compact_utf8_canonical_sha256
from tools.odyssey.ple_real_state_bank_formula import (
    SESSION_SCHEMA,
    SESSION_STATUS,
    STATE_BANK_SCHEMA,
    STATE_BANK_STATUS,
    load_native_pre_ple_state_bank,
)


def test_native_state_bank_reader_binds_states_to_the_exact_session(tmp_path) -> None:
    session = {
        "schema": SESSION_SCHEMA,
        "status": SESSION_STATUS,
        "token_ids": [11, 12],
        "accepted_generation_tokens": 1,
        "execution": {
            "source_reset_or_reprefill": False,
            "state_memory": {"total_persistent_bytes": 64},
        },
    }
    session["seal_sha256"] = _compact_utf8_canonical_sha256(session)
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")
    bank = tmp_path / "bank"
    bank.mkdir()
    states = []
    for step, token_id in enumerate((11, 12)):
        values = np.full(10_240, step + 1, dtype="<f4")
        raw = values.tobytes()
        state_path = bank / f"state-{step}.f32"
        state_path.write_bytes(raw)
        states.append({
            "step": step,
            "token_id": token_id,
            "layer": 0,
            "path": str(state_path),
            "dtype": "F32_LE",
            "elements": 10_240,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "finite": True,
        })
    manifest = {
        "schema": STATE_BANK_SCHEMA,
        "status": STATE_BANK_STATUS,
        "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
        "session_receipt": {
            "path": str(session_path),
            "sha256": hashlib.sha256(session_path.read_bytes()).hexdigest(),
            "seal_sha256": session["seal_sha256"],
        },
        "token_ids": [11, 12],
        "prompt_length": 1,
        "capture": {
            "layer": 0,
            "state_width": 10_240,
            "upstream_PLE_inclusive_source_trajectory": False,
        },
        "states": states,
    }
    manifest["seal_sha256"] = _compact_utf8_canonical_sha256(manifest)
    manifest_path = bank / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    observed, binding, vectors = load_native_pre_ple_state_bank(manifest_path)
    assert observed["token_ids"] == [11, 12]
    assert binding["upstream_PLE_inclusive_source_trajectory"] is False
    assert vectors.shape == (2, 10_240)
    np.testing.assert_array_equal(vectors[1], np.full(10_240, 2.0, dtype=np.float32))
