"""Pure guard tests for the current pre-execution HCLI compiler trace."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from lab.operators.ascension_qwen30_quality_repack_current_hcli_trace_capture import (
    TRACE_MODE,
    annotate_compiler_trace,
    _render_source_one_user_template,
)


def _tiny_tokenizer(tmp_path: Path) -> Tokenizer:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": {
                        "<unk>": 0,
                        "alpha": 1,
                        "beta": 2,
                        "user": 3,
                    },
                    "unk_token": "<unk>",
                },
            }
        ),
        encoding="utf-8",
    )
    return Tokenizer.from_file(str(tokenizer_path))


def test_current_trace_annotation_preserves_pre_execution_boundary_and_all_span_ids(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.json"
    raw = {
        "schema": "hawking.ascension.hcli_compiler_pre_execution_trace.v1",
        "status": TRACE_MODE,
        "capture_timing": "AFTER_CONTEXT_COMPILATION_BEFORE_PROVIDER_OR_MODEL_EXECUTION",
        "model_execution_started": False,
        "selected_context_spans": [
            {"content_id": "one", "text": "alpha beta", "token_count": 2},
            {"content_id": "two", "text": "beta", "token_count": 1},
        ],
        "folded_native_prompt_utf8": "alpha",
    }
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    annotated = annotate_compiler_trace(
        raw_trace=raw,
        tokenizer=_tiny_tokenizer(tmp_path),
        source_template_binding={"renderer": "test-renderer"},
        raw_trace_path=raw_path,
    )
    spans = annotated["source_tokenizer_annotations"]["selected_context_spans"]
    assert [span["content_id"] for span in spans] == ["one", "two"]
    assert all(span["hcli_compiler_token_ids"]["token_count"] > 0 for span in spans)
    assert annotated["compiler_trace"]["model_execution_started"] is False
    assert annotated["claim_boundary"]["new_diagnostic_not_historical"] is True


def test_source_user_template_render_is_byte_stable() -> None:
    folded = "alpha\nbeta"
    assert _render_source_one_user_template(folded) == (
        "<|im_start|>user\nalpha\nbeta<|im_end|>\n<|im_start|>assistant\n"
    )
