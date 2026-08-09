#!/usr/bin/env python3
"""Prepare a broad, source-bound L0 route-capture input for activation pricing.

Diagnostic only. Not a capability, HCLI, TPS, or coherence claim.

Builds many one-user native chat prompts (diverse length/domain/structure),
tokenizes them with the source Qwen3-Coder tokenizer, and writes the sealed
input JSON consumed by `ascension_qwen30_current_hcli_layer0_route_capture`
under the broad-activation schema.

The three-probe HCLI quality path is left untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TOKENIZER = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/"
    "qwen-30b/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/"
    "quality-diagnostics/broad-activation-v1"
)

BROAD_INPUT_SCHEMA = "hawking.ascension.qwen30_broad_activation_layer0_route_capture_input.v1"
STATUS = "NEW_DIAGNOSTIC_NOT_HISTORICAL"

# ---------------------------------------------------------------------------
# Prompt corpus — intentionally diverse so constant-mean null can fall.
# Domains: code, prose, structured/JSON, multi-turn, long-context, math,
# instructions, dialogue, mixed-language-ish technical, lists.
# ---------------------------------------------------------------------------


def build_corpus() -> list[dict[str, str]]:
    """Return list of {probe_id, domain, text} (user message body only)."""
    long_prose = (
        "Write a careful essay about why measurement instruments dominate "
        "conclusions in scientific work. Discuss selection bias, sample size, "
        "and the difference between operator recovery and distribution-local "
        "fit. Use three short sections with headings, then a one-paragraph "
        "conclusion that refuses to overclaim."
    )
    long_code = '''\
Implement a pure-Python priority queue that supports:
1) push(item, priority)
2) pop() -> item with lowest priority
3) decrease_key(item, new_priority)
Use a binary heap of (priority, counter, item) triples so equal priorities are
FIFO-stable. Include a minimal unittest block under if __name__ == "__main__".
Do not use heapq's nlargest. Comment each public method in one sentence.
'''
    long_json_spec = (
        "Design a JSON schema for a lab receipt with fields: schema (string), "
        "status (enum of EARNED/REFUSED/INCOMPLETE), claim_boundary (object of "
        "booleans), binding (object with absolute paths and sha256 hex strings), "
        "and measurements (array of objects with name, value, unit). "
        "Return ONLY valid JSON for one example instance populated with "
        "plausible diagnostic values. No markdown fences."
    )
    multi_turn_body = (
        "Conversation so far:\n"
        "User: What is a MoE expert?\n"
        "Assistant: A specialist sub-network selected by a router for a token.\n"
        "User: Why would weight cosine mislead packing quality?\n"
        "Assistant: Because output error on real activations can disagree with "
        "weight-space distance.\n"
        "User: Given that, list three concrete checks you would run on a new "
        "activation capture before trusting a family ranking. Be terse."
    )
    long_context = (
        "Context dump (use only what you need):\n"
        + "\n".join(
            f"- note[{i}]: activation diversity requires domain mix; null falls when "
            f"expert outputs are not near-constant across prompts; sample-{i}."
            for i in range(40)
        )
        + "\n\nTask: In at most five bullets, state how you would detect a "
        "constant-mean null trap from held-out expert outputs alone."
    )
    rows: list[tuple[str, str, str]] = [
        ("code_fib_iter", "code", "Write an iterative Fibonacci function in Rust that returns u64 and panics on overflow."),
        ("code_py_bisect", "code", "Write a Python binary search that returns the insertion index for a sorted list of ints."),
        ("code_sql_window", "code", "Write a SQL query using ROW_NUMBER() to pick the latest event per user_id from events(user_id, ts, payload)."),
        ("code_go_http", "code", "Show a minimal Go net/http handler that returns {\"ok\":true} as JSON with status 200."),
        ("code_ts_reduce", "code", "TypeScript: write a typed reduce that sums numbers and rejects non-finite values at runtime."),
        ("code_long_pq", "code", long_code),
        ("prose_measure", "prose", long_prose),
        ("prose_short_haiku", "prose", "Compose one English haiku about fog over a harbor. No title."),
        ("prose_argument", "prose", "Argue in two paragraphs that negative experimental results are first-class scientific deliverables."),
        ("prose_narrative", "prose", "Narrate 120-150 words of a technician discovering a high constant-mean null in a capture log, without inventing false numbers."),
        ("json_status_strict", "structured", 'Return exactly this JSON and nothing else: {"status":"ok","phase":"diagnostic"}'),
        ("json_schema_example", "structured", long_json_spec),
        ("json_array_table", "structured", "Return a JSON array of 5 objects with keys expert_id (int), hits (int), null_cosine (float). Invent realistic diagnostic numbers only."),
        ("json_nested_error", "structured", 'Return JSON: {"error":{"code":"INSUFFICIENT_DIVERSITY","detail":"null>=0.9","retry":{"min_prompts":12}}} with no prose.'),
        ("multi_turn_moe", "multi_turn", multi_turn_body),
        ("multi_turn_debug", "multi_turn", "User earlier said the server on :18430 must not be restarted. Assistant acknowledged. Now answer: list safe steps to capture L0 activations without disturbing that server."),
        ("multi_turn_review", "multi_turn", "Prior message claimed surplus +0.04 with weight cosine 0.46. Reply as a skeptical reviewer: what is still unproven?"),
        ("long_ctx_notes", "long_context", long_context),
        ("long_ctx_log", "long_context", "Log excerpt:\n" + "\n".join(f"t={i:04d} expert={i%128} cos={0.9+((i*17)%10)*0.001:.3f}" for i in range(80)) + "\nSummarize whether these cosines alone prove operator recovery. Answer in 3 sentences."),
        ("math_bayes", "math", "A test has sensitivity 0.9 and specificity 0.95. Prevalence is 1%. What is PPV? Show the arithmetic."),
        ("math_rank", "math", "For a matrix W in R^{m x n} and rank r, how many free parameters does a factored LR form have? Give the formula and a short explanation."),
        ("math_entropy", "math", "Compute Shannon entropy in bits for the distribution [0.5, 0.25, 0.125, 0.125]. Show steps."),
        ("instr_checklist", "instruction", "Produce a numbered checklist (8 items) for source-bound activation capture provenance. Imperative verbs only."),
        ("instr_refuse", "instruction", "If asked to claim full-model coherence from a component probe, refuse in one sentence and name the missing evidence."),
        ("instr_compare", "instruction", "Compare in a table with 4 rows: weight cosine vs output cosine vs null-subtracted surplus vs BPW. Columns: metric, what it measures, common failure mode."),
        ("dialogue_lab", "dialogue", "A: Is 0.95 output cosine good?\nB: Only relative to the constant-mean null.\nA: Ours is 0.94 null.\nB: Then surplus is thin.\nContinue one more A/B exchange that ends with a concrete next measurement."),
        ("list_domains", "list", "List 12 prompt domain tags useful for lowering activation nulls, one per line, no numbers."),
        ("mixed_api", "mixed", "Given this partial OpenAPI path /v1/capture with POST body {prompts: string[]}, write a curl example and a one-line claim_boundary comment."),
        ("mixed_regex", "mixed", "Write a regex that matches sha256 hex digests and a Python snippet that validates a path is absolute."),
        ("code_rust_parse", "code", "Rust: parse CLI flags --input-json and --output-dir that must be absolute Paths; return a struct or error string."),
        ("prose_policy", "prose", "Explain in plain language why absolute output cosine is inadmissible when the constant-mean null is enormous."),
        ("structured_yamlish", "structured", "Emit a YAML-like plain text config (not JSON) with keys: capture_layer, max_seq_len, probe_count, claim: diagnostic_only."),
    ]
    return [
        {"probe_id": pid, "domain": domain, "text": text}
        for pid, domain, text in rows
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def token_ids_u32le_sha256(ids: list[int]) -> str:
    return sha256_bytes(b"".join(struct.pack("<I", int(x)) for x in ids))


def one_user_native_prompt(tokenizer: Any, user_text: str) -> list[int]:
    """Match Qwen chat shape used by the sealed three-probe capture tails.

    <|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n
    """
    rendered = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    # tokenizers encode with special tokens already present as literal text
    ids = list(tokenizer.encode(rendered).ids)
    if not ids:
        raise RuntimeError("empty tokenization")
    if ids[0] != 151644:  # <|im_start|>
        raise RuntimeError(f"expected chat start token, got {ids[0]}")
    return ids


# Repo-source corpus. The hand-written corpus above is 32 probes / ~3.9k tokens,
# which routes only 5.1 of the 32 hits an expert needs to be fittable: 42.8% of
# the 6144 (layer, expert) pairs were never routed at all. Coverage of experts is
# what the fit is starved of, so this draws real source text off disk to widen
# routing. Instruction framings vary so the router does not see one flat shape.
REPO_TASK_FRAMES = (
    "Explain what this code does, then name one edge case it mishandles:\n\n",
    "Review this code for correctness. Be specific about failure modes:\n\n",
    "Write a concise docstring or module comment for this code:\n\n",
    "Summarise the control flow here in three sentences:\n\n",
    "What would break if this ran concurrently? Reason about the shared state:\n\n",
)
REPO_SOURCE_GLOBS = (("*.rs", "repo_rust"), ("*.py", "repo_python"), ("*.md", "repo_docs"))


def approx_raw_token_count(text: str) -> int:
    """Budget heuristic when the HF `tokenizers` package is unavailable.

    Used only to decide how many repo files to draw; the sealed document always
    uses exact token counts from either HF encode or `--tokenized-json`.
    Qwen BPE on mixed code/prose is closer to ~4 chars/token than 3; using 4
    tends to slightly overshoot the raw budget so the chat-framed total still
    lands near the requested target rather than starving it.
    """
    return max(1, (len(text) + 3) // 4)


def build_repo_corpus(
    root: Path,
    target_tokens: int,
    count_tokens: Any,
    *,
    chunk_chars: int = 2800,
) -> list[dict[str, str]]:
    """Draw probes from real repo source until ~target_tokens is reached.

    Deterministic: files sorted, round-robin across extensions so one language
    cannot dominate the routing distribution. Skips generated/vendored trees and
    anything too small to carry structure.

    `count_tokens` is a callable(text) -> int used only for the budget loop
    (HF encode when available, approx when using the Rust tokenizer bridge).
    """
    skip = ("/.git/", "/target/", "/workspace/ops/build/", "/node_modules/",
            "/.worktrees/", "/vendor/", "/workspace/campaign/evidence/")
    buckets: list[list[Path]] = []
    for pattern, _domain in REPO_SOURCE_GLOBS:
        found = sorted(
            p for p in root.rglob(pattern)
            if p.is_file()
            and not any(s in f"/{p.relative_to(root).as_posix()}/" for s in skip)
            and 1500 < p.stat().st_size < 400_000
        )
        buckets.append(found)

    rows: list[dict[str, str]] = []
    total = 0
    idx = [0] * len(buckets)
    exhausted = [False] * len(buckets)
    while total < target_tokens and not all(exhausted):
        for b, (_, domain) in enumerate(REPO_SOURCE_GLOBS):
            if total >= target_tokens:
                break
            if idx[b] >= len(buckets[b]):
                exhausted[b] = True
                continue
            path = buckets[b][idx[b]]
            idx[b] += 1
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            chunk = text[:chunk_chars].strip()
            if len(chunk) < 400:
                continue
            frame = REPO_TASK_FRAMES[len(rows) % len(REPO_TASK_FRAMES)]
            body = f"{frame}```\n{chunk}\n```"
            n = int(count_tokens(body))
            rows.append(
                {
                    "probe_id": f"{domain}_{idx[b]:05d}",
                    "domain": domain,
                    "text": body,
                }
            )
            total += n
    return rows


def load_tokenized_json(path: Path) -> dict[str, dict[str, Any]]:
    """Load Rust bridge output: array of {probe_id, domain, token_ids, ...}."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise RuntimeError(f"--tokenized-json must be a JSON array: {path}")
    by_id: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise RuntimeError(f"--tokenized-json row {i} is not an object")
        probe_id = row.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id:
            raise RuntimeError(f"--tokenized-json row {i} lacks non-empty probe_id")
        if probe_id in by_id:
            raise RuntimeError(f"--tokenized-json repeats probe_id {probe_id!r}")
        ids = row.get("token_ids")
        if not isinstance(ids, list) or not ids:
            raise RuntimeError(f"--tokenized-json {probe_id}: missing token_ids")
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in ids):
            raise RuntimeError(f"--tokenized-json {probe_id}: token_ids must be ints")
        if int(ids[0]) != 151644:
            raise RuntimeError(
                f"--tokenized-json {probe_id}: expected chat start token 151644, got {ids[0]}"
            )
        by_id[probe_id] = row
    return by_id


def probe_from_tokenized(
    row: dict[str, str], tok_row: dict[str, Any]
) -> dict[str, Any]:
    """Build one sealed probe from corpus text + Rust bridge receipt."""
    ids = [int(x) for x in tok_row["token_ids"]]
    text_sha = sha256_bytes(row["text"].encode("utf-8"))
    ids_sha = token_ids_u32le_sha256(ids)
    got_text_sha = tok_row.get("user_text_sha256")
    if got_text_sha != text_sha:
        raise RuntimeError(
            f"probe {row['probe_id']}: user_text_sha256 mismatch "
            f"(tokenized={got_text_sha}, corpus={text_sha})"
        )
    got_ids_sha = tok_row.get("token_ids_u32le_sha256")
    if got_ids_sha is not None and got_ids_sha != ids_sha:
        raise RuntimeError(
            f"probe {row['probe_id']}: token_ids_u32le_sha256 mismatch "
            f"(tokenized={got_ids_sha}, recomputed={ids_sha})"
        )
    token_count = tok_row.get("token_count")
    if token_count is not None and int(token_count) != len(ids):
        raise RuntimeError(
            f"probe {row['probe_id']}: token_count {token_count} != len(token_ids) {len(ids)}"
        )
    return {
        "probe_id": row["probe_id"],
        "domain": row["domain"],
        "source_one_user_native_prompt": {
            "token_ids": ids,
            "token_count": len(ids),
            "token_ids_u32le_sha256": ids_sha,
            "add_special_tokens": True,
            "chat_shape": "one_user_message_no_system_no_tools",
            "user_text_sha256": text_sha,
        },
        "user_text": row["text"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="optional cap for smoke tests (0 = full corpus)",
    )
    parser.add_argument(
        "--repo-corpus-root",
        type=Path,
        default=None,
        help="draw additional probes from real source under this root "
        "(default: hand-written corpus only, unchanged)",
    )
    parser.add_argument(
        "--repo-corpus-target-tokens",
        type=int,
        default=0,
        help="approximate token budget for repo-drawn probes (0 = none)",
    )
    parser.add_argument(
        "--tokenized-json",
        type=Path,
        default=None,
        help="use pre-tokenized rows from qwen30_corpus_tokenize (absolute path). "
        "When set, the Python tokenizers package is not imported.",
    )
    parser.add_argument(
        "--emit-in-json",
        type=Path,
        default=None,
        help="write the untokenized corpus array {probe_id,domain,text} and exit "
        "(for the Rust tokenizer bridge; does not require tokenizers)",
    )
    args = parser.parse_args()

    tokenized_path = (
        args.tokenized_json.expanduser().resolve() if args.tokenized_json is not None else None
    )
    tokenized_by_id: dict[str, dict[str, Any]] | None = None
    tokenizer: Any = None

    if tokenized_path is not None:
        if not tokenized_path.is_file():
            print(f"tokenized-json missing: {tokenized_path}", file=sys.stderr)
            return 2
        try:
            tokenized_by_id = load_tokenized_json(tokenized_path)
        except (OSError, json.JSONDecodeError, RuntimeError) as exc:
            print(f"tokenized-json load failed: {exc}", file=sys.stderr)
            return 2
        count_tokens = approx_raw_token_count
    elif args.emit_in_json is not None:
        # Corpus dump only: approximate budget for repo draws.
        count_tokens = approx_raw_token_count
    else:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            print(
                "tokenizers package required (pip install tokenizers). "
                f"Import failed: {exc}",
                file=sys.stderr,
            )
            return 2
        tok_path_early = args.tokenizer_json.expanduser().resolve()
        if not tok_path_early.is_file():
            print(f"tokenizer missing: {tok_path_early}", file=sys.stderr)
            return 2
        tokenizer = Tokenizer.from_file(str(tok_path_early))
        count_tokens = lambda text: len(  # noqa: E731
            tokenizer.encode(text, add_special_tokens=False).ids
        )

    tok_path = args.tokenizer_json.expanduser().resolve()
    if not tok_path.is_file():
        print(f"tokenizer missing: {tok_path}", file=sys.stderr)
        return 2

    corpus = build_corpus()
    if args.max_prompts > 0:
        corpus = corpus[: args.max_prompts]
    if args.repo_corpus_root is not None and args.repo_corpus_target_tokens > 0:
        root = args.repo_corpus_root.expanduser().resolve()
        if not root.is_dir():
            print(f"repo corpus root missing: {root}", file=sys.stderr)
            return 2
        extra = build_repo_corpus(root, args.repo_corpus_target_tokens, count_tokens)
        if not extra:
            print("repo corpus root yielded no usable probes", file=sys.stderr)
            return 2
        corpus = corpus + extra
    if len(corpus) < 12:
        print("need at least 12 prompts for broad schema", file=sys.stderr)
        return 2

    if args.emit_in_json is not None:
        emit_path = args.emit_in_json.expanduser().resolve()
        if not emit_path.is_absolute():
            print(f"--emit-in-json must be absolute: {args.emit_in_json}", file=sys.stderr)
            return 2
        emit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"probe_id": r["probe_id"], "domain": r["domain"], "text": r["text"]}
            for r in corpus
        ]
        emit_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "emit_in_json": str(emit_path),
                    "probe_count": len(payload),
                    "note": "untokenized corpus only; run qwen30_corpus_tokenize next",
                },
                indent=2,
            )
        )
        return 0

    probes: list[dict[str, Any]] = []
    domain_counts: dict[str, int] = {}
    for row in corpus:
        domain_counts[row["domain"]] = domain_counts.get(row["domain"], 0) + 1
        if tokenized_by_id is not None:
            tok_row = tokenized_by_id.get(row["probe_id"])
            if tok_row is None:
                print(
                    f"tokenized-json missing probe_id {row['probe_id']!r}",
                    file=sys.stderr,
                )
                return 2
            try:
                probes.append(probe_from_tokenized(row, tok_row))
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 2
        else:
            ids = one_user_native_prompt(tokenizer, row["text"])
            probes.append(
                {
                    "probe_id": row["probe_id"],
                    "domain": row["domain"],
                    "source_one_user_native_prompt": {
                        "token_ids": ids,
                        "token_count": len(ids),
                        "token_ids_u32le_sha256": token_ids_u32le_sha256(ids),
                        "add_special_tokens": True,
                        "chat_shape": "one_user_message_no_system_no_tools",
                        "user_text_sha256": sha256_bytes(row["text"].encode("utf-8")),
                    },
                    "user_text": row["text"],
                }
            )

    out_dir = args.out_dir.expanduser().resolve()
    req_dir = out_dir / "requests"
    req_dir.mkdir(parents=True, exist_ok=True)

    document: dict[str, Any] = {
        "schema": BROAD_INPUT_SCHEMA,
        "status": STATUS,
        "prepared_at": utc_now(),
        "purpose": (
            "broaden L0 router-input activations so constant-mean null can be "
            "re-priced honestly for activation-aware family ranking"
        ),
        "claim_boundary": {
            "model_execution_started": False,
            "new_diagnostic_not_historical": True,
            "diagnostic_activation_pricing_only": True,
            "does_not_claim_coherence_hcli_tps_or_capability": True,
            "not_hcli_compiler_trace_bound": True,
            "source_tokenizer_one_user_native_prompts": True,
            "production_server_not_used": True,
        },
        "tokenizer_binding": {
            "path": str(tok_path),
            "sha256": sha256_file(tok_path),
        },
        "corpus_summary": {
            "probe_count": len(probes),
            "domain_counts": domain_counts,
            "total_tokens": sum(p["source_one_user_native_prompt"]["token_count"] for p in probes),
            "min_tokens": min(p["source_one_user_native_prompt"]["token_count"] for p in probes),
            "max_tokens": max(p["source_one_user_native_prompt"]["token_count"] for p in probes),
            "mean_tokens": round(
                sum(p["source_one_user_native_prompt"]["token_count"] for p in probes) / len(probes),
                2,
            ),
        },
        "probes": probes,
    }

    # Content-addressed filename
    body_for_hash = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = sha256_bytes(body_for_hash.encode("utf-8"))
    out_path = req_dir / f"QWEN30_BROAD_ACTIVATION_L0_ROUTE_CAPTURE_INPUT_{digest[:16]}.json"
    out_path.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    file_sha = sha256_file(out_path)

    provenance = {
        "schema": "hawking.ascension.qwen30_broad_activation_capture_prepare.v1",
        "prepared_at": utc_now(),
        "input_path": str(out_path),
        "input_sha256": file_sha,
        "corpus_summary": document["corpus_summary"],
        "claim_boundary": document["claim_boundary"],
        "tokenizer_binding": document["tokenizer_binding"],
        "note": (
            "Input only. Metal L0 capture not started. Live production server "
            "must not be disturbed; run capture serialized if GPU residency conflicts."
        ),
    }
    prov_path = out_dir / "PREPARE_PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"input_path": str(out_path), "input_sha256": file_sha, **document["corpus_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
