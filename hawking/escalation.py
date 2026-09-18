"""Resident-callable escalation: one frontier question, one worker packet.

Two capabilities, both previously ABSENT from the typed-tool surface
(SUPERWAVE_STATE.md, probe ``cloud-escalation``):

1. ``escalate_to_frontier`` sends one scoped question plus a small, curated
   packet (mission kernel + named artifacts + an output schema) to a cloud
   frontier model over Anthropic's native wire protocol
   (``x-api-key``/``anthropic-version`` at ``/v1/messages`` -- the OpenAI
   shape in ``hawking/backends.py`` does not speak this). It never invents a
   result: with no API key configured it raises ``EscalationCredentialsError``
   before any network attempt (fail closed, not a silent no-op), and a
   successful call is wrapped through ``hawking.result_envelope.ResultEnvelope``
   as an ``UNVERIFIED`` hypothesis -- a cloud answer is a proposal, structurally,
   never promoted to a verified fact by this module.

2. ``propose_swarm`` / ``launch_swarm`` stage Hawking WorkUnit contracts only.
   They never invoke an external bot or create a second scheduler; Mission and
   the central verifier own execution and acceptance. There is no fixed lane
   count ceiling; real process, provider, budget, dependency, and effect-scope
   guards remain authoritative.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .result_envelope import ResultEnvelope

# Packet size ceilings protect transport/context memory. Lane count is not a
# Hawking slot policy; it is left unbounded and governed by the real scheduler.
MAX_ARTIFACTS = 8
MAX_ARTIFACT_CHARS = 4000
MAX_PACKET_CHARS = 8000
MAX_SWARM_LANES: Optional[int] = None

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"

_LANE_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")


class EscalationCredentialsError(PermissionError):
    """No cloud credential is configured. Fail closed; never invented."""


class EscalationError(RuntimeError):
    """The frontier call was attempted and failed, or its response was unusable."""


class SwarmBoundsError(ValueError):
    """More lanes were requested than the bounded ceiling allows."""


def curate_packet(
    mission_kernel: Any,
    artifacts: Any,
    output_schema: Any,
) -> Dict[str, Any]:
    """Build the curated packet: mission kernel + named artifacts + schema.

    Raises rather than truncates: a caller that hands this an archive-sized
    blob gets told to curate it, not a silently clipped copy of it.
    """
    kernel = str(mission_kernel or "").strip()
    if not kernel:
        raise ValueError("mission_kernel is required")
    if not isinstance(artifacts, list):
        raise ValueError("artifacts must be a list of {name, content} objects")
    if len(artifacts) > MAX_ARTIFACTS:
        raise ValueError(
            f"{len(artifacts)} artifacts requested, exceeds the curated ceiling of "
            f"{MAX_ARTIFACTS}; name the ones that actually matter"
        )
    if not isinstance(output_schema, dict) or not output_schema:
        raise ValueError("output_schema is required so a frontier answer has a checkable shape")
    named: List[Dict[str, str]] = []
    total = len(kernel)
    seen = set()
    for item in artifacts:
        if not isinstance(item, dict) or not item.get("name") or "content" not in item:
            raise ValueError("each artifact needs a non-empty 'name' and a 'content' string")
        name = str(item["name"])
        if name in seen:
            raise ValueError(f"duplicate artifact name: {name}")
        seen.add(name)
        content = str(item["content"])
        if len(content) > MAX_ARTIFACT_CHARS:
            raise ValueError(
                f"artifact {name!r} is {len(content)} chars, exceeds the per-artifact "
                f"ceiling of {MAX_ARTIFACT_CHARS}; curate a slice, not the whole file"
            )
        total += len(content)
        named.append({"name": name, "content": content})
    if total > MAX_PACKET_CHARS:
        raise ValueError(
            f"curated packet is {total} chars, exceeds {MAX_PACKET_CHARS}; "
            "this is a scoped question, not an archive dump -- trim artifacts"
        )
    return {"mission_kernel": kernel, "artifacts": named, "output_schema": output_schema}


def _render_prompt(question: str, packet: Dict[str, Any]) -> str:
    lines = [
        "You are being consulted as a scoped external reviewer.",
        "Your answer is a PROPOSAL. It is not accepted as fact on receipt; a local",
        "deterministic process must verify it before it counts as anything more.",
        "",
        f"QUESTION: {question}",
        "",
        "MISSION_KERNEL:",
        packet["mission_kernel"],
    ]
    if packet["artifacts"]:
        lines.append("")
        lines.append("NAMED ARTIFACTS:")
        for item in packet["artifacts"]:
            lines.append(f"--- {item['name']} ---")
            lines.append(item["content"])
    lines.append("")
    lines.append("Respond so the answer can be parsed against this OUTPUT_SCHEMA:")
    lines.append(json.dumps(packet["output_schema"]))
    return "\n".join(lines)


def _extract_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise EscalationError("frontier response was not a JSON object")
    if "error" in payload:
        err = payload["error"]
        message = err.get("message") if isinstance(err, dict) else str(err)
        raise EscalationError(f"frontier API returned an error: {message}")
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        raise EscalationError("frontier response is missing its content blocks")
    texts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    text = "\n".join(t for t in texts if t)
    if not text:
        raise EscalationError("frontier response contained no text content")
    return text


def escalate_to_frontier(
    question: Any,
    mission_kernel: Any,
    artifacts: Any,
    output_schema: Any,
    *,
    model: Optional[str] = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """Escalate one scoped question. Returns a proposal envelope, never a fact.

    Fails closed with ``EscalationCredentialsError`` before any network call
    if no key is configured; fails with ``EscalationError`` if the call is
    attempted and does not come back usable. Never fabricates an answer.
    """
    text_question = str(question or "").strip()
    if not text_question:
        raise ValueError("question is required")
    packet = curate_packet(mission_kernel, artifacts, output_schema)
    key = api_key or os.environ.get(api_key_env)
    if not key:
        raise EscalationCredentialsError(
            f"{api_key_env} is not set; refusing to escalate to a cloud frontier "
            "model without a credential (fail closed, not a silent no-op)"
        )
    resolved_model = model or DEFAULT_ANTHROPIC_MODEL
    prompt = _render_prompt(text_question, packet)
    body = json.dumps(
        {
            "model": resolved_model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
    )
    retrieved_at = time.time()
    try:
        with urllib.request.urlopen(request, timeout=max(0.1, float(timeout_s))) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace") if exc.fp else str(exc)
        raise EscalationError(f"frontier call failed: HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise EscalationError(f"frontier call failed: {exc.reason}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EscalationError("frontier response was not valid JSON") from exc
    text = _extract_text(payload)
    envelope = ResultEnvelope(
        verdict="UNVERIFIED",
        claim=text_question,
        hypotheses=[{"claim": text, "source": f"cloud_frontier:anthropic:{resolved_model}"}],
        uncertainty=[
            "model output is not a verified fact",
            "cloud response requires local deterministic verification before acceptance",
        ],
        next_action="verify this proposal locally before treating it as fact",
    ).to_dict()
    return {
        "packet": packet,
        "envelope": envelope,
        "provenance": {
            "provider": "anthropic",
            "model": resolved_model,
            "endpoint": ANTHROPIC_API_URL,
            "retrieved_at": retrieved_at,
            "credential_values_recorded": False,
        },
    }


def render_lane_contract(
    objective: Any,
    write_scope: Any,
    verify_command: Any,
    acceptance: Any = None,
) -> str:
    """Deterministically format caller-supplied fields into a WRITE/VERIFY contract.

    Does not choose the objective, write scope, or verify command -- those are
    the calling model's judgment, already made by the time this runs. This
    only renders them into Hawking's bounded worker packet shape: each path line is
    prefixed with its section keyword (``WRITE:``/``VERIFY:``) because the
    linter classifies a line by matching a keyword/verb *on that same line*,
    not by which heading it falls under, and the verify command is
    backtick-quoted because the linter's only path-free verification signal
    is a backtick-quoted command containing a known test/build tool name. An
    ACCEPTANCE section is always emitted -- its heading alone satisfies the
    linter's "no acceptance criterion" check -- even with no caller-supplied
    acceptance items.
    """
    text_objective = str(objective or "").strip()
    text_verify = str(verify_command or "").strip()
    if not text_objective:
        raise ValueError("objective is required")
    if not text_verify:
        raise ValueError("verify_command is required")
    if not isinstance(write_scope, list) or not write_scope:
        raise ValueError("write_scope must be a non-empty list of paths")
    lines = [f"# {text_objective}", "", "## WRITE", ""]
    lines.extend(f"WRITE: {path}" for path in write_scope)
    lines.extend(["", "## VERIFY", "", f"VERIFY: `{text_verify}`", "", "## ACCEPTANCE", ""])
    if acceptance:
        if not isinstance(acceptance, list):
            raise ValueError("acceptance must be a list of strings")
        lines.extend(f"- {item}" for item in acceptance)
    else:
        lines.append(f"- `{text_verify}` exits 0")
    return "\n".join(lines) + "\n"


def propose_swarm(
    problem_statement: Any,
    lanes: Any,
    *,
    max_lanes: Optional[int] = MAX_SWARM_LANES,
) -> Dict[str, Any]:
    """Validate and render lane contracts without launching anything.

    An explicit finite max_lanes remains available as an operator policy.
    The default is unbounded by Hawking count; no request is silently
    truncated, and the scheduler remains the admission authority.
    """
    text_problem = str(problem_statement or "").strip()
    if not text_problem:
        raise ValueError("problem_statement is required")
    if not isinstance(lanes, list) or not lanes:
        raise ValueError("lanes must be a non-empty list of lane specs")
    if max_lanes is not None and len(lanes) > max_lanes:
        raise SwarmBoundsError(
            f"{len(lanes)} lanes requested, exceeds the bounded ceiling of {max_lanes}; "
            "use a larger explicit operator policy or split the swarm"
        )
    built: List[Dict[str, str]] = []
    seen_names = set()
    for lane in lanes:
        if not isinstance(lane, dict) or not lane.get("name"):
            raise ValueError("each lane needs a unique 'name'")
        name = str(lane["name"]).strip()
        if not _LANE_NAME_RE.fullmatch(name):
            raise ValueError(f"lane name {name!r} must match [A-Za-z0-9_-]+")
        if name in seen_names:
            raise ValueError(f"duplicate lane name: {name}")
        seen_names.add(name)
        contract_text = lane.get("contract_text")
        if not contract_text:
            contract_text = render_lane_contract(
                lane.get("objective"),
                lane.get("write_scope"),
                lane.get("verify_command"),
                lane.get("acceptance"),
            )
        contract_text = str(contract_text)
        if "WRITE:" not in contract_text or "VERIFY:" not in contract_text:
            raise ValueError("lane contract must contain WRITE and VERIFY sections")
        built.append({"name": name, "contract_text": str(contract_text)})
    return {"problem_statement": text_problem, "lane_count": len(built), "lanes": built}


def launch_swarm(
    workspace: Any,
    problem_statement: Any,
    lanes: Any,
    *,
    mode: str = "audit",
    max_lanes: Optional[int] = MAX_SWARM_LANES,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Stage Hawking worker packets without launching external work."""
    if mode not in ("audit", "consult"):
        raise ValueError("mode must be 'audit' or 'consult'")
    del workspace, dry_run
    proposal = propose_swarm(problem_statement, lanes, max_lanes=max_lanes)
    staged = [
        {
            "lane": lane["name"],
            "workunit_id": f"hawking-worker-{lane['name']}",
            "contract_text": lane["contract_text"],
            "status": "staged",
            "promotion_allowed": False,
        }
        for lane in proposal["lanes"]
    ]
    return {
        "provider": "hawking",
        "problem_statement": proposal["problem_statement"],
        "mode": mode,
        "status": "staged",
        "lanes": staged,
    }


__all__ = [
    "MAX_ARTIFACTS",
    "MAX_ARTIFACT_CHARS",
    "MAX_PACKET_CHARS",
    "MAX_SWARM_LANES",
    "EscalationCredentialsError",
    "EscalationError",
    "SwarmBoundsError",
    "curate_packet",
    "escalate_to_frontier",
    "render_lane_contract",
    "propose_swarm",
    "launch_swarm",
]
#
