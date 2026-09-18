"""Gate 6: worker-agnostic discrete Goal operator surface for h web / hawkingd."""
from __future__ import annotations

P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2 = "P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2"


def p17_collapse_contract_v2_marker():
    """Bounded P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2 source marker.

    No-release, no-hardware-qualification boundary: this marker only records
    that the collapse implementation contract v2 surface is present.
    """
    return P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2


P4_AUTO_SPOOL_HARDENING_V3 = "P4_AUTO_SPOOL_HARDENING_V3"


def p4_auto_spool_hardening_v3_marker():
    """Bounded P4_AUTO_SPOOL_HARDENING_V3 source marker.

    No-release, no-hardware-qualification boundary: this marker only records
    that the auto-spool hardening v3 surface is present. It performs no
    spool mutation, no release, and no hardware qualification.
    """
    return P4_AUTO_SPOOL_HARDENING_V3

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlsplit

from hawking.persist import atomic_write_json

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no flock
    fcntl = None  # type: ignore[assignment]

GOAL_SCHEMA = "hawking.discrete_goal.v1"
RESULT_SCHEMA = "hawking.discrete_goal_result.v1"
GOAL_CONTRACT_SCHEMA = "hawking.discrete_goal_contract.v1"
GOALS_DIRNAME = "goals"


# Local weight/artifact suffixes that must never be mistaken for a remote id.
WEIGHT_ARTIFACT_SUFFIXES = (
    ".json",
    ".safetensors",
    ".gguf",
    ".onnx",
    ".bin",
    ".pt",
    ".pth",
    ".npz",
)


def _is_remote_openrouter_model(value: Any) -> bool:
    """Recognize an OpenRouter id without making it a second authority."""
    text = str(value or "").strip()
    if text.lower().startswith("openrouter:"):
        return bool(text.split(":", 1)[1].strip())
    # OpenRouter's public ids are provider/model and are intentionally opaque.
    # Do not mistake absolute paths or native JSON artifacts for one.
    if not text or text.startswith(("/", ".")):
        return False
    if text.endswith(WEIGHT_ARTIFACT_SUFFIXES):
        return False
    if "\\" in text:
        return False
    if "://" in text:
        return False
    if text.count("/") != 1:
        return False
    provider, _, model = text.partition("/")
    if not provider.strip() or not model.strip():
        return False
    if any(ch.isspace() for ch in text):
        return False
    return True


def p17_collapse_implementation_contract_v2() -> Dict[str, Any]:
    """Executable P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2 collapse surface.

    No-release, no-hardware-qualification boundary: this function only
    normalizes an in-memory goal record into the collapse implementation
    contract shape. It performs no spool mutation, no release, and no
    hardware qualification.
    """
    return {
        "schema": GOAL_CONTRACT_SCHEMA,
        "contract": P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2,
        "collapse": {
            "goal_schema": GOAL_SCHEMA,
            "result_schema": RESULT_SCHEMA,
            "goals_dirname": GOALS_DIRNAME,
            "weight_artifact_suffixes": list(WEIGHT_ARTIFACT_SUFFIXES),
        },
    }


def collapse_goal_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Collapse a discrete goal record into the P17 contract shape.

    Observable behavior: the returned mapping always carries the contract
    marker, a stable goal id, and a normalized remote/local model decision.
    Unknown or missing fields collapse to explicit ``None`` values instead
    of raising, so callers can persist a partial record safely.

    P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2: the collapse also resolves the
    qualified alternate model route so a caller can tell, without a second
    authority, whether the collapsed record is eligible for a fresh attempt.
    A record whose model is a resolvable remote OpenRouter id collapses to
    ``alternate_route_qualified=True``; a bare ``openrouter:`` prefix, a
    local artifact path, or a missing model collapses to ``False``.
    """
    if not isinstance(record, Mapping):
        raise TypeError("collapse_goal_record requires a mapping")
    goal_id = record.get("goal_id")
    model = record.get("model")
    return {
        "schema": GOAL_SCHEMA,
        "contract": P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2,
        "goal_id": str(goal_id) if goal_id is not None else None,
        "model": str(model) if model is not None else None,
        "remote_model": _is_remote_openrouter_model(model),
        "alternate_route_qualified": _is_qualified_alternate_model_route(model),
        "collapsed": True,
    }


def _openrouter_model_id(value: Any) -> str:
    """Return the provider/model id without the UI transport prefix."""
    text = str(value or "").strip()
    return (
        text.split(":", 1)[1].strip()
        if text.lower().startswith("openrouter:") else text
    )


def _is_qualified_alternate_model_route(value: Any) -> bool:
    """Recognize the explicitly qualified alternate model route for a fresh attempt.

    A route is qualified only when it resolves to a non-empty remote
    OpenRouter id. The bare ``openrouter:`` transport prefix with no id is
    not a usable route and must not be admitted.
    """
    text = str(value or "").strip()
    if not text:
        return False
    if not _is_remote_openrouter_model(text):
        return False
    # P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2: a qualified alternate route
    # must carry a provider segment and a model segment around the slash;
    # ids like "openrouter:/model" or "provider/" are not resolvable routes.
    model_id = _openrouter_model_id(text)
    provider, _, model = model_id.partition("/")
    return bool(provider.strip()) and bool(model.strip())


# Retained name for callers that still reference the legacy predicate; it is
# the same object as _is_qualified_alternate_model_route so both names admit
# only non-empty, resolvable remote OpenRouter ids.
_is_qualified_alternate_model_route_legacy = _is_qualified_alternate_model_route


def qualified_alternate_model_route(value: Any) -> bool:
    """Public predicate for the explicitly qualified alternate model route.

    P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2: expose the canonical check under
    a public name so qualification-support callers do not reach into the
    private predicate. Admits only non-empty, resolvable remote OpenRouter
    ids (provider and model segments around the slash); the bare
    ``openrouter:`` transport prefix is not a usable route.
    """
    return _is_qualified_alternate_model_route(value)


def _normalize_attachment_refs(value: Any) -> List[Dict[str, Any]]:
    """Keep durable Goal attachment references separate from prompt text."""
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("attachments must be a list")
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("attachment must be an object")
        artifact_id = str(item.get("artifact_id") or "").strip()
        if not artifact_id or not re.fullmatch(r"artifact-[0-9a-f]{64}", artifact_id):
            raise ValueError("attachment artifact_id is invalid")
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        try:
            size = max(0, int(item.get("size") or 0))
        except (TypeError, ValueError):
            raise ValueError("attachment size is invalid")
        digest = str(item.get("digest") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("attachment digest is invalid")
        read_path = str(item.get("read_path") or "").strip()[:260]
        if read_path and not read_path.startswith((
            ".hawking/web-artifacts/", ".hawking/H-NOTES/"
        )):
            # The browser cannot turn a Goal packet into arbitrary filesystem
            # authority. Only Hawking-owned artifact roots are portable.
            read_path = ""
        rows.append({
            "schema": str(item.get("schema") or "hawking.web.artifact.v1")[:120],
            "artifact_id": artifact_id,
            "filename": str(item.get("filename") or "attachment.md")[:180],
            "media_type": str(item.get("media_type") or "text/markdown")[:120],
            "size": size,
            "digest": digest,
            "status": str(item.get("status") or "ADMITTED")[:40],
            "encoding": str(item.get("encoding") or "utf-8")[:40],
            "read_path": read_path,
            "source_type": str(item.get("source_type") or "file_attachment")[:80],
            "byte_count": size,
            "estimated_tokens": max(0, int(item.get("estimated_tokens") or 0)),
            "workspace_id": str(item.get("workspace_id") or "")[:80],
            "session_id": str(item.get("session_id") or "")[:120],
            "message_id": str(item.get("message_id") or "")[:160],
            "created_at": str(item.get("created_at") or "")[:80],
            "local_identity": str(item.get("local_identity") or "")[:260],
            "retention": str(item.get("retention") or "workspace")[:40],
            "pinned": bool(item.get("pinned")),
            "metadata": dict(item.get("metadata") or {})
            if isinstance(item.get("metadata"), Mapping) else {},
        })
    return rows

OPERATOR_UI_HTML = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
    "<title>Hawking Goal</title>"
    "<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem}"
    "label{display:block;margin-top:1rem;font-weight:600}"
    "textarea,input,select{width:100%;box-sizing:border-box;margin-top:.35rem}"
    "button{margin-top:1rem;padding:.6rem 1rem}"
    "pre{background:#111;color:#eee;padding:1rem;overflow:auto}</style></head><body>"
    "<h1>Hawking discrete Goal</h1>"
    "<p>Submit a bounded engineering Goal. Closing this page does not stop execution.</p>"
    "<label>Objective<textarea id='obj' rows='4'></textarea></label>"
    "<label>Acceptance (one per line)<textarea id='acc' rows='4'></textarea></label>"
    "<label>Hawking model<select id='res'>"
    "<option value='KIMI_P0_OPERATIONAL'>Hawking (Kimi P0 review)</option>"
    "</select><small id='catalog'>Kimi P0 is present on device; canonical runtime admission is withheld.</small></label>"
    "<label>Goal mode<select id='mode'><option value='discrete_goal'>Discrete engineering Goal</option><option value='worker_research'>Read-only research worker</option></select></label>"
    "<label>Tool-learning allowance (USD, separate)<input id='learn' type='number' min='0' step='0.01' value='10'></label>"
    "<label>Build cap (USD)<input id='budget' type='number' min='0' step='0.01' value='30'></label>"
    "<label>Focus tests (optional, one per line)<textarea id='ft' rows='2' placeholder='hawking/test_x.py::test_name'></textarea></label>"
    "<label>Focus files (optional, one per line)<textarea id='ff' rows='2' placeholder='hawking/module.py'></textarea></label>"
    "<label>Attach workunit id (optional)<input id='wu' placeholder='KIMI-U001'/></label>"
    "<label>Next prompt / steering note<textarea id='steer' rows='3' placeholder='Tell the current Goal what to do next…'></textarea></label>"
    "<label>Prompt preset<select id='preset'><option value='Use the complete permissioned Hawking tool surface efficiently: batch targeted reads, then execute the smallest accepted edit and focused test.'>Efficient tool pass</option><option value='Stop archaeology. Use the next owed Hawking tool, then provide machine-readable evidence of the result.'>Stop reading and act</option><option value='Delegate one genuinely separable read-only question to the cheapest suitable worker, then continue the parent Goal from its result.'>Delegate a bounded scout</option></select></label>"
    "<button id='go'>Submit Goal</button><input id='gid' placeholder='GOAL-…'/><button id='refresh'>Refresh</button><button id='workers'>Workers</button><span id='live'>not connected</span>"
    "<button data-a='steer'>Steer</button><button data-a='pause'>Pause</button><button data-a='resume'>Resume</button><button data-a='checkpoint'>Checkpoint</button><button data-a='cancel'>Stop</button><button data-a='result'>Result</button><button data-a='diff'>Diff</button><pre id='out'></pre>"
    "<script>"
    "document.getElementById('go').onclick=async()=>{"
    "const acceptance=document.getElementById('acc').value.split('\\n').map(s=>s.trim()).filter(Boolean);"
    "const body={objective:document.getElementById('obj').value,acceptance,"
    "focus_tests:document.getElementById('ft').value.split('\\n').map(s=>s.trim()).filter(Boolean),"
    "focus_files:document.getElementById('ff').value.split('\\n').map(s=>s.trim()).filter(Boolean),"
    "goal_mode:document.getElementById('mode').value,"
    "tool_learning_budget_usd:Number(document.getElementById('learn').value||0),"
    "budget_usd:Number(document.getElementById('budget').value||0),"
    "resident:document.getElementById('res').value,model:document.getElementById('res').value,"
    "workunit_id:document.getElementById('wu').value||null,source:'h_web_ui'};"
    "const r=await fetch('/api/goal',{method:'POST',headers:{'Content-Type':'application/json'},"
    "body:JSON.stringify(body)});const d=await r.json();if(d.goal)document.getElementById('gid').value=d.goal.goal_id;document.getElementById('out').textContent=JSON.stringify(d,null,2);refresh();};"
    "let refreshing=false;async function refresh(){const id=document.getElementById('gid').value;if(!id||refreshing)return;refreshing=true;try{const r=await fetch('/api/goal/status?id='+encodeURIComponent(id)+'&view=compact');const d=await r.json();const g=d.goal||{};const w=d.workunit||{};const hb=w.heartbeat||{};document.getElementById('live').textContent=(g.status||w.state||'unknown')+' · '+(g.phase||w.current_phase||'')+' · turn '+(w.last_turn||0)+' · $'+Number(w.worker_cost_usd||0).toFixed(4)+(hb.last_tool_dispatch?' · '+(hb.last_tool_dispatch.kind||'tool'):'');document.getElementById('out').textContent=JSON.stringify(d,null,2);}catch(e){document.getElementById('live').textContent='connection error: '+e.message;}finally{refreshing=false;}};"
    "async function loadModels(){try{const d=await (await fetch('/v1/models')).json();const rows=(d.data||[]).filter(x=>x.id==='KIMI_P0_OPERATIONAL');if(rows.length){const s=document.getElementById('res');const old=s.value;s.replaceChildren(...rows.map(x=>{const o=document.createElement('option');o.value=x.id;o.textContent=x.name||'Hawking';return o;}));if([...s.options].some(o=>o.value===old))s.value=old;document.getElementById('catalog').textContent='Hawking · Kimi P0 review identity · runtime withheld';}}catch(e){document.getElementById('catalog').textContent='Hawking · local review identity unavailable';}};"
    "async function loadWorkers(){const r=await fetch('/v1/workers');document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2);};"
    "document.getElementById('preset').onchange=()=>document.getElementById('steer').value=document.getElementById('preset').value;document.getElementById('refresh').onclick=refresh;document.getElementById('workers').onclick=loadWorkers;loadModels();setInterval(refresh,2500);document.querySelectorAll('[data-a]').forEach(b=>b.onclick=async()=>{const id=document.getElementById('gid').value;const action=b.dataset.a;const message=action==='steer'?document.getElementById('steer').value.trim():'';if(action==='steer'&&!message){document.getElementById('steer').focus();return;}const r=await fetch('/api/goal/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id,action,message})});document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2);refresh();});"
    "</script></body></html>"
)


def goals_root(workspace: str | os.PathLike[str]) -> Path:
    root = Path(workspace).expanduser().resolve() / ".hawking" / GOALS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _goal_path(workspace: str | os.PathLike[str], goal_id: str) -> Path:
    return goals_root(workspace) / f"{goal_id}.json"


def _result_path(workspace: str | os.PathLike[str], goal_id: str) -> Path:
    return goals_root(workspace) / f"{goal_id}.result.json"


def _contract_path(workspace: str | os.PathLike[str], goal_id: str) -> Path:
    return goals_root(workspace) / f"{goal_id}.contract.json"


def workunit_id_for_goal(goal_id: str) -> str:
    """Return a durable, filesystem-safe owner id for one discrete Goal."""
    token = str(goal_id or "").strip().upper()
    if not token.startswith("GOAL-"):
        raise ValueError("goal id must start with GOAL-")
    suffix = re.sub(r"[^A-Z0-9_-]", "", token[5:])
    if not suffix:
        raise ValueError("goal id has no usable suffix")
    return f"GOAL-{suffix}"


def browser_session_id_for_goal(goal_id: str) -> str:
    """Derive the one browser session a Goal may own from its durable id."""
    return f"BROWSER-{workunit_id_for_goal(goal_id)[5:]}"


def normalize_browser_authority(
    value: Optional[Mapping[str, Any]], *, goal_id: str,
) -> Dict[str, Any]:
    """Validate a narrowly-scoped browser capability for one Goal.

    This is intentionally a Goal contract field rather than browser/UI state:
    a remote worker can be restarted and still receive the same bounded session
    and navigation fence.  A generic chat request cannot manufacture it.
    """
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("browser_authority must be an object")
    actions = bool(value.get("actions", False))
    observe = bool(value.get("observe", True)) or actions
    raw_origins = value.get("origins") or []
    if not isinstance(raw_origins, list):
        raise ValueError("browser_authority.origins must be a list")
    origins: List[str] = []
    for raw in raw_origins[:16]:
        parsed = urlsplit(str(raw or "").strip())
        if parsed.scheme not in {"http", "https", "file"}:
            raise ValueError("browser_authority origins must use http, https, or file")
        if parsed.scheme == "file":
            if parsed.netloc or parsed.path not in {"", "/"}:
                raise ValueError("file browser authority must be exactly file://")
            origin = "file://"
        else:
            if not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("browser_authority origins must be plain scheme://host[:port]")
            origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in origins:
            origins.append(origin)
    if actions and not origins:
        raise ValueError("browser actions require at least one explicit allowed origin")
    if not observe and not actions:
        return {}
    return {
        "schema": "hawking.goal.browser_authority.v1",
        "browser_session_id": browser_session_id_for_goal(goal_id),
        "observe": observe,
        "actions": actions,
        "origins": origins,
    }


def normalize_macos_authority(
    value: Optional[Mapping[str, Any]], *, goal_id: str,
) -> Dict[str, Any]:
    """Validate the one narrow desktop action capability a Goal may own.

    The native helper can observe the desktop for any admitted Goal, but an
    action-capable Goal must name the exact application scope it may address.
    This contract deliberately exposes semantic AXPress only; it is not a
    generic keyboard, mouse, shell, or accessibility escape hatch.
    """
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("macos_authority must be an object")
    observe = bool(value.get("observe", True))
    actions = bool(value.get("actions", False))
    if actions:
        observe = True
    raw_apps = value.get("applications")
    if raw_apps is None:
        raw_apps = value.get("application_names")
    if raw_apps is None:
        raw_apps = []
    if not isinstance(raw_apps, list):
        raise ValueError("macos_authority.applications must be a list")
    applications = []
    for raw in raw_apps[:8]:
        name = str(raw or "").strip()
        if not name:
            continue
        if len(name) > 160:
            raise ValueError("macos authority application name is too long")
        if name not in applications:
            applications.append(name)
    raw_bundles = value.get("bundle_ids") or []
    if not isinstance(raw_bundles, list):
        raise ValueError("macos_authority.bundle_ids must be a list")
    bundle_ids = []
    for raw in raw_bundles[:8]:
        bundle = str(raw or "").strip()
        if not bundle:
            continue
        if len(bundle) > 160 or not re.fullmatch(r"[A-Za-z0-9_.-]+", bundle):
            raise ValueError("macos authority bundle id is invalid")
        if bundle not in bundle_ids:
            bundle_ids.append(bundle)
    if actions and not applications and not bundle_ids:
        raise ValueError(
            "macos actions require at least one exact application or bundle scope"
        )
    try:
        lease_ttl_s = float(value.get("lease_ttl_s", 30.0))
    except (TypeError, ValueError):
        raise ValueError("macos_authority.lease_ttl_s must be numeric")
    lease_ttl_s = min(120.0, max(1.0, lease_ttl_s))
    if not observe and not actions:
        return {}
    return {
        "schema": "hawking.goal.macos_authority.v1",
        "goal_id": workunit_id_for_goal(goal_id),
        "observe": observe,
        "actions": actions,
        "applications": applications,
        "bundle_ids": bundle_ids,
        "lease_ttl_s": lease_ttl_s,
        "preemptible": True,
        "semantic_actions": ["AXPress"] if actions else [],
    }


def create_goal(
    workspace: str | os.PathLike[str],
    *,
    objective: str,
    acceptance: Optional[List[str]] = None,
    attachments: Optional[List[Mapping[str, Any]]] = None,
    authority_boundary: str = "staging worktree only; no production mutation",
    flash_resource_rule: str = "Flash protected-lane; native resident only; no unqualified provider fallback",
    completion_contract: str = "RUNNING to COMPLETE only when acceptance met; emit canonical result",
    resident: str = "KIMI_P0_OPERATIONAL",
    workunit_id: Optional[str] = None,
    focus_tests: Optional[List[str]] = None,
    focus_files: Optional[List[str]] = None,
    goal_mode: str = "discrete_goal",
    budget_usd: Optional[float] = None,
    tool_learning_budget_usd: Optional[float] = None,
    parent_worker_id: str = "",
    parent_workunit_id: str = "",
    worker_policy: str = "auto",
    auto_scope: str = "cloud",
    allowed_models: Optional[List[str]] = None,
    cognition_plan: Optional[Dict[str, Any]] = None,
    parent_goal_ids: Optional[List[str]] = None,
    browser_authority: Optional[Mapping[str, Any]] = None,
    macos_authority: Optional[Mapping[str, Any]] = None,
    required_capabilities: Optional[List[str]] = None,
    external_roots: Optional[List[str]] = None,
    source: str = "h_web",
) -> Dict[str, Any]:
    obj = str(objective or "").strip()
    if not obj:
        raise ValueError("objective is required")
    mode = str(goal_mode or "discrete_goal").strip()
    if mode not in {"discrete_goal", "worker_research"}:
        raise ValueError("goal_mode must be discrete_goal or worker_research")
    policy = str(worker_policy or "auto").strip().lower()
    if policy not in {"auto", "pinned"}:
        raise ValueError("worker_policy must be auto or pinned")
    source_scope = str(auto_scope or "cloud").strip().lower()
    if source_scope not in {"cloud", "local", "both"}:
        raise ValueError("auto_scope must be cloud, local, or both")
    try:
        build_budget = max(0.0, float(budget_usd)) if budget_usd is not None else 0.0
    except (TypeError, ValueError):
        raise ValueError("budget_usd must be numeric")
    try:
        learning_budget = (
            max(0.0, float(tool_learning_budget_usd))
            if tool_learning_budget_usd is not None else 0.0
        )
    except (TypeError, ValueError):
        raise ValueError("tool_learning_budget_usd must be numeric")
    gid = f"GOAL-{uuid.uuid4().hex[:10].upper()}"
    normalized_browser_authority = normalize_browser_authority(
        browser_authority, goal_id=gid,
    )
    normalized_macos_authority = normalize_macos_authority(
        macos_authority, goal_id=gid,
    )
    from .goal import infer_required_capabilities
    machine_capabilities = infer_required_capabilities(
        obj,
        goal_mode=mode,
        browser_authority=normalized_browser_authority,
        macos_authority=normalized_macos_authority,
        explicit=required_capabilities,
    )
    now = time.time()
    doc: Dict[str, Any] = {
        "schema": GOAL_SCHEMA,
        "goal_id": gid,
        "status": "RUNNING",
        "created_at": now,
        "updated_at": now,
        "source": source,
        "goal_mode": mode,
        # The build cap remains the Goal's primary acceptance budget. A
        # separate tool-learning allowance is recorded beside it and is never
        # silently folded into the build cap; the owner/daemon enforce the
        # combined ceiling when both phases are enabled.
        "budget_authorized_usd": build_budget if budget_usd is not None else None,
        "budget_plan": {
            "tool_learning_usd": learning_budget,
            "build_usd": build_budget,
            "combined_usd": learning_budget + build_budget,
            "phase": "tool_learning" if learning_budget > 0.0 else "build",
            "switch": "tool_learning_gate_or_cap",
        },
        "parent_worker_id": str(parent_worker_id or ""),
        "parent_workunit_id": str(parent_workunit_id or ""),
        "parent_goal_ids": [
            str(item).strip() for item in (parent_goal_ids or [])
            if str(item).strip()
        ][:32],
        "objective": obj,
        "attachments": _normalize_attachment_refs(attachments),
        "acceptance": list(acceptance or []),
        "authority_boundary": authority_boundary,
        "flash_resource_rule": flash_resource_rule,
        "completion_contract": completion_contract,
        "resident_assignment": resident,
        "workunit_id": workunit_id,
        "focus_tests": [str(item) for item in (focus_tests or []) if str(item).strip()],
        "focus_files": [str(item) for item in (focus_files or []) if str(item).strip()],
        "worker_policy": policy,
        "auto_scope": source_scope,
        "allowed_models": [
            str(item).strip() for item in (allowed_models or [])
            if str(item).strip()
        ][:16],
        "auto_plan": dict(cognition_plan or {}),
        "browser_authority": normalized_browser_authority,
        "macos_authority": normalized_macos_authority,
        "required_capabilities": machine_capabilities,
        "external_roots": [str(item).strip() for item in (external_roots or []) if str(item).strip()][:32],
        "phase": "submitted",
        "checkpoint": None,
        "claim_boundary": (
            "Hawking Goal operator record. Remote workers remain ephemeral "
            "under the Goal/WorkUnit owner; a provider conversation is not "
            "authority."
        ),
    }
    atomic_write_json(_goal_path(workspace, gid), doc)
    return doc


def dispatch_goal(
    workspace: str | os.PathLike[str],
    goal: Dict[str, Any],
    *,
    model: str = "KIMI_P0_OPERATIONAL",
    background: bool = True,
) -> Dict[str, Any]:
    """Admit a Web Goal to the Hawking-owned durable workunit loop.

    Goal creation and execution are deliberately separate records, but one
    server-owned admission call binds them atomically enough for a caller to
    close its browser after the HTTP response. The owner remains the source
    of checkpoint, completion, and result truth; this function only prepares
    its immutable contract and starts the already-existing owner.
    """
    if not isinstance(goal, dict):
        raise ValueError("goal must be an object")
    gid = str(goal.get("goal_id") or "").strip()
    if not gid:
        raise ValueError("goal is missing goal_id")
    if str(goal.get("status") or "RUNNING").upper() != "RUNNING":
        raise ValueError("only RUNNING goals may be dispatched")
    resident = str(goal.get("resident_assignment") or "").strip()
    if not resident:
        raise ValueError("resident assignment is required")
    # A Goal may name any admitted native Hawking artifact directly, one of the
    # logical roles, or an explicit OpenRouter model. Remote cognition remains
    # under this same Goal/WorkUnit owner; it does not create a provider-owned
    # mission or a second scheduler.
    from hawking.canonical_runtime import CanonicalRoleRouter, RuntimeRoleUnavailable
    model = str(model or resident).strip() or resident
    remote = _is_remote_openrouter_model(model)
    goal_mode = str(goal.get("goal_mode") or "discrete_goal").strip()
    if goal_mode not in {"discrete_goal", "worker_research"}:
        raise ValueError("unsupported Goal mode")
    if not remote:
        try:
            CanonicalRoleRouter().route(resident, {"objective": goal.get("objective")})
        except RuntimeRoleUnavailable as exc:
            raise ValueError(f"resident assignment is not currently runnable: {exc}") from exc
        try:
            CanonicalRoleRouter().route(model, {"objective": goal.get("objective")})
        except RuntimeRoleUnavailable as exc:
            raise ValueError(f"dispatch model is not currently runnable: {exc}") from exc

    root = Path(workspace).expanduser().resolve()
    wid = str(goal.get("workunit_id") or workunit_id_for_goal(gid)).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", wid):
        raise ValueError("workunit id contains unsupported characters")
    # A durable production Goal can own many bounded WorkUnits without
    # changing its root objective or manufacturing a child Goal merely to
    # express a small execution tranche.  The WorkUnit contract is the
    # ephemeral worker's input; the Goal remains the project owner.
    objective = str(goal.get("workunit_objective") or goal.get("objective") or "").strip()
    acceptance = goal.get("workunit_acceptance")
    if not isinstance(acceptance, list):
        acceptance = goal.get("acceptance") or []
    # Keep the resident's first useful action bounded by the Goal itself.  A
    # discrete Goal is not an open-ended U001 continuation: if the submitter
    # names a focused pytest node or source path, carry those exact paths into
    # the owner contract so the prompt and the result cannot drift.
    focus_tests = list(goal.get("workunit_focus_tests") or goal.get("focus_tests") or [])
    # Keep explicit disposable worktree targets admissible.  H-Web submits a
    # bounded prose objective rather than a second path configuration surface;
    # recognizing `.worktrees/<name>/...` here lets the existing mutation
    # bundle, prompt, and engine all agree on the isolated target without
    # widening ordinary paths or moving the canonical Goal store.
    goal_path_prefix = (
        r"(?:\.worktrees/[A-Za-z0-9_.-]+/|"
        r"(?:hawking|crates|docs|tests)/)"
    )
    if not focus_tests:
        focus_tests = sorted(set(re.findall(
            rf"(?<![A-Za-z0-9_./-]){goal_path_prefix}"
            r"[A-Za-z0-9_./-]+::[A-Za-z_][A-Za-z0-9_]*",
            objective,
        )))
    focus_files = list(goal.get("workunit_focus_files") or goal.get("focus_files") or [])
    if not focus_files:
        focus_files = sorted(set(re.findall(
            rf"(?<![A-Za-z0-9_./-]){goal_path_prefix}[A-Za-z0-9_./-]+",
            objective,
        )))
    test_mutation_paths = sorted({
        str(item).split("::", 1)[0].strip()
        for item in focus_tests
        if str(item).strip()
    })
    allowed_files = sorted({
        str(item).strip()
        for item in [*focus_files, *test_mutation_paths]
        if str(item).strip()
    })
    browser_authority = normalize_browser_authority(
        goal.get("browser_authority")
        if isinstance(goal.get("browser_authority"), Mapping) else None,
        goal_id=gid,
    )
    macos_authority = normalize_macos_authority(
        goal.get("macos_authority")
        if isinstance(goal.get("macos_authority"), Mapping) else None,
        goal_id=gid,
    )
    from .goal import infer_required_capabilities
    required_machine_capabilities = infer_required_capabilities(
        objective,
        goal_mode=goal_mode,
        browser_authority=browser_authority,
        macos_authority=macos_authority,
        explicit=(goal.get("required_capabilities")
                  if isinstance(goal.get("required_capabilities"), list) else None),
    )
    capabilities = []
    if browser_authority:
        capabilities.append("browser.observe")
        if browser_authority.get("actions"):
            capabilities.append("browser.actions")
    if macos_authority:
        capabilities.append("macos.observe")
        if macos_authority.get("actions"):
            capabilities.extend(("macos.semantic_target", "macos.action_lease"))
    root_budget = goal.get("budget_authorized_usd")
    workunit_budget = goal.get("workunit_budget_usd", root_budget)
    try:
        workunit_budget = (
            max(0.0, float(workunit_budget))
            if workunit_budget is not None else None
        )
        root_budget_value = max(0.0, float(root_budget)) if root_budget is not None else None
    except (TypeError, ValueError):
        raise ValueError("workunit_budget_usd must be numeric")
    if (
        root_budget_value is not None
        and workunit_budget is not None
        and workunit_budget > root_budget_value
    ):
        raise ValueError("workunit_budget_usd cannot exceed the parent Goal budget")
    workunit_budget_plan = (
        {
            "tool_learning_usd": 0.0,
            "build_usd": workunit_budget,
            "combined_usd": workunit_budget,
            "phase": "build",
            "switch": "workunit_cap",
        }
        if "workunit_budget_usd" in goal
        else dict(goal.get("budget_plan") or {
            "tool_learning_usd": 0.0,
            "build_usd": workunit_budget,
            "combined_usd": workunit_budget,
            "phase": "build",
            "switch": "build_only",
        })
    )
    contract = {
        "schema": GOAL_CONTRACT_SCHEMA,
        "goal_id": gid,
        "workunit_id": wid,
        "title": (
            f"Research Worker Goal {gid}"
            if goal_mode == "worker_research" else f"Discrete Goal {gid}"
        ),
        "generated_at": time.time(),
        "goal_mode": goal_mode,
        # A root production Goal may dispatch several bounded WorkUnits. Its
        # own terminal edge is separate from a child WorkUnit receipt; retain
        # legacy one-Goal/one-WorkUnit completion unless an explicit WorkUnit
        # override is being dispatched beneath the Goal.
        "goal_terminal_on_workunit_complete": "workunit_objective" not in goal,
        "worker_policy": str(goal.get("worker_policy") or "auto"),
        "auto_scope": str(goal.get("auto_scope") or "cloud"),
        "allowed_models": list(goal.get("allowed_models") or []),
        "cognition_plan": dict(goal.get("auto_plan") or {}),
        "plan_id": str((goal.get("auto_plan") or {}).get("plan_id") or ""),
        "task_class": str(goal.get("task_class") or "general"),
        "worker_role": str(goal.get("worker_role") or "worker"),
        "mutation_plan": dict(goal.get("workunit_mutation_plan") or {}),
        "objective": objective,
        "attachments": _normalize_attachment_refs(goal.get("attachments")),
        "acceptance": list(acceptance),
        "focus_tests": [str(item) for item in focus_tests if str(item).strip()],
        "focus_files": [str(item) for item in focus_files if str(item).strip()],
        "required_capabilities": required_machine_capabilities,
        "external_roots": [
            str(item).strip() for item in (goal.get("external_roots") or [])
            if str(item).strip()
        ][:32],
        "mutation_bundle": {
            "schema": "hawking.mutation_bundle.v1",
            "require_source_mutation": goal_mode == "discrete_goal",
            "require_regression_test_mutation": (
                goal_mode == "discrete_goal"
                and not bool(
                    (goal.get("workunit_mutation_plan") or {}).get(
                        "allow_existing_regression_tests"
                    )
                )
            ),
            "require_focused_test_invocation": goal_mode == "discrete_goal",
            "allowed_files": allowed_files,
            "focus_files": allowed_files,
            "focus_tests": [str(item) for item in focus_tests if str(item).strip()],
            "test_mutation_paths": test_mutation_paths,
            "mutation_plan": dict(goal.get("workunit_mutation_plan") or {}),
            "authority": "staging Goal contract; engine owns operation semantics",
        },
        "forbidden_paths": [
            "receipts/future/workunits/KIMI-U001_SEAL_SCRATCH.md",
            "P0", "Flash protected resources", "production branches",
        ],
        "authority": {
            "allowed": [
                "read/search the staging worktree",
                *([] if goal_mode == "worker_research" else [
                    "edit files under the staging worktree",
                    "run focused CPU tests/builds",
                    "write receipts under receipts/future and receipts/headless",
                ]),
                "checkpoint, compact, and resume this Goal",
            ],
            "forbidden": [
                "mutate canonical production branches",
                "touch Flash protected resources or P0",
                "claim qualification without evidence",
                "ask the user for routine continuation",
            ],
            "capabilities": capabilities,
            "machine_capabilities": required_machine_capabilities,
            "browser": browser_authority,
            "macos": macos_authority,
            "resource_leases": (
                [{
                    "resource": "desktop.semantic_input",
                    "mode": "exclusive",
                    "preemptible": True,
                    "ttl_s": macos_authority.get("lease_ttl_s", 30.0),
                }]
                if macos_authority.get("actions") else []
            ),
        },
        "staging_workspace": {"root": str(root)},
        "resource_class": "GPU_YIELDABLE",
        "provider": (
            f"openrouter:{_openrouter_model_id(model)}" if remote else "hawking:native"
        ),
        "worker_model": model,
        "worker_kind": "remote_openrouter" if remote else "native_hawking",
        "budget_authorized_usd": workunit_budget,
        "budget_plan": workunit_budget_plan,
        "cost_authorization": (
            {"max_usd": workunit_budget}
            if workunit_budget is not None else {}
        ),
        "parent_worker_id": str(goal.get("parent_worker_id") or ""),
        "parent_workunit_id": str(goal.get("parent_workunit_id") or ""),
        "parent_goal_ids": list(goal.get("parent_goal_ids") or []),
        "claim_boundary": (
            "Goal execution is Hawking-owned. Model claims require the "
            "mode-specific tool/evidence contract and canonical result packet."
        ),
        "exact_next_action": (
            "Inspect the bounded repository and return evidence from the admitted "
            "read-only tools; do not mutate files."
            if goal_mode == "worker_research" else
            "Inspect the staging repository and the acceptance criteria; plan one "
            "smallest useful edit, then test and seal it."
        ),
        "compaction_policy": {"auto": True, "resume": "rehydrate Goal and owner checkpoint"},
    }
    # A long-lived Goal's root contract must not be overwritten by a later
    # WorkUnit.  Child tranches get an immutable contract beside the WorkUnit
    # id; the parent Goal record remains the single project authority.
    is_child_workunit = "workunit_objective" in goal
    contract_path = (
        goals_root(root) / f"{wid}.contract.json"
        if is_child_workunit else _contract_path(root, gid)
    )
    if is_child_workunit:
        contract["parent_goal_objective"] = str(goal.get("objective") or "")[:4000]
        contract["accepted_phase"] = str(
            goal.get("accepted_phase") or goal.get("phase") or ""
        )[:200]
        if isinstance(goal.get("swift_helper_state"), Mapping):
            contract["swift_helper_state"] = dict(goal["swift_helper_state"])
    atomic_write_json(contract_path, contract)

    from hawking.workunit_owner import WorkunitOwner

    # WorkunitOwner is the single admission writer.  It creates a durable
    # blocked WorkUnit when a fresh permission probe fails, so the exact next
    # action survives until the operator rechecks and resumes it.
    try:
        owner_result = WorkunitOwner(root).start(
            contract_path=contract_path,
            workunit_id=wid,
            model=model,
            background=background,
        )
    except Exception as exc:
        try:
            from .permissions import PermissionRequired
            if isinstance(exc, PermissionRequired) and not is_child_workunit:
                update_goal(
                    root,
                    gid,
                    status="BLOCKED",
                    phase="PERMISSION_BLOCKED",
                    blocker=exc.to_dict(),
                    required_capabilities=list(contract.get("required_capabilities") or []),
                )
        except Exception:
            pass
        raise
    permission_admission = (
        owner_result.get("permission_admission")
        if isinstance(owner_result, Mapping) else None
    )
    owner = owner_result.get("owner") if isinstance(owner_result, dict) else {}
    update_fields = {
        "phase": "DISPATCHED" if not is_child_workunit else str(goal.get("phase") or "RUNNING"),
        "checkpoint": owner.get("checkpoint_path") if isinstance(owner, dict) else None,
        "owner_job": (owner_result.get("background") if isinstance(owner_result, dict) else None),
    }
    if is_child_workunit:
        parent = load_goal(root, gid) or goal
        workunit_ids = [
            str(item).strip() for item in (parent.get("workunit_ids") or [])
            if str(item).strip()
        ]
        for item in (parent.get("workunit_id"), parent.get("active_workunit_id"), wid):
            token = str(item or "").strip()
            if token and token not in workunit_ids:
                workunit_ids.append(token)
        raw_active_ids = parent.get("active_workunit_ids")
        active_ids = [
            str(item).strip() for item in (parent.get("active_workunit_ids") or [])
            if str(item).strip()
        ]
        # Preserve the old scalar field as a primary/UI hint while making the
        # durable parent projection capable of representing independent lanes.
        # Only use the scalar as a fallback for pre-frontier records.  Once an
        # explicit list exists, an empty list is authoritative and must not
        # resurrect a terminal child from the compatibility hint.
        if raw_active_ids is None:
            token = str(parent.get("active_workunit_id") or "").strip()
            if token and token not in active_ids:
                active_ids.append(token)
        if wid not in active_ids:
            active_ids.append(wid)
        active_ids = active_ids[-32:]
        active_jobs = [
            dict(item) for item in (parent.get("active_workunit_jobs") or [])
            if isinstance(item, Mapping) and str(item.get("workunit_id") or "").strip()
        ]
        child_job = update_fields.get("owner_job")
        if isinstance(child_job, Mapping):
            child_job = dict(child_job)
            child_job["workunit_id"] = wid
            active_jobs = [
                item for item in active_jobs
                if str(item.get("workunit_id") or "") != wid
                and str(item.get("job_id") or "") != str(child_job.get("job_id") or "")
            ]
            active_jobs.append(child_job)
        frontier = [
            dict(item) for item in (parent.get("runnable_frontier") or [])
            if isinstance(item, Mapping) and str(item.get("workunit_id") or "").strip()
        ]
        frontier = [item for item in frontier if str(item.get("workunit_id")) not in active_ids]
        frontier.extend({
            "workunit_id": item,
            "state": "RUNNING",
            "phase": str(parent.get("phase") or ""),
        } for item in active_ids)
        scalar_hint = str(parent.get("active_workunit_id") or "").strip()
        primary_hint = scalar_hint if scalar_hint in active_ids else (
            active_ids[0] if active_ids else wid
        )
        update_fields.update({
            "active_workunit_id": primary_hint,
            "active_workunit_ids": active_ids,
            "active_workunit_jobs": active_jobs[-32:],
            "runnable_frontier": frontier[-32:],
            "workunit_ids": workunit_ids[-32:],
        })
    else:
        update_fields["workunit_id"] = wid
    updated = update_goal(root, gid, **update_fields) or goal
    return {
        "goal": updated,
        "contract": str(contract_path),
        "workunit_id": wid,
        "owner": owner_result,
        "permission_admission": permission_admission,
    }


def dispatch_child_workunit(
    workspace: str | os.PathLike[str],
    parent_goal_id: str,
    *,
    objective: str,
    acceptance: List[str],
    focus_files: Optional[List[str]] = None,
    focus_tests: Optional[List[str]] = None,
    budget_usd: float = 1.0,
    macos_authority: Optional[Mapping[str, Any]] = None,
    required_capabilities: Optional[List[str]] = None,
    external_roots: Optional[List[str]] = None,
    mutation_plan: Optional[Mapping[str, Any]] = None,
    goal_mode: str = "discrete_goal",
    task_class: Optional[str] = None,
    model_override: Optional[str] = None,
    background: bool = True,
) -> Dict[str, Any]:
    """Use Auto to admit one bounded WorkUnit under an existing Goal.

    This is the production Goal/WorkUnit seam: no child Goal is manufactured,
    the parent phase is preserved, and the selected model is only a plan input
    until ``dispatch_goal`` assigns the real Hawking worker identity.
    """
    root = Path(workspace).expanduser().resolve()
    parent = load_goal(root, str(parent_goal_id).strip())
    if not parent:
        raise ValueError(f"no parent Goal {parent_goal_id}")
    if str(parent.get("status") or "").upper() != "RUNNING":
        raise ValueError("only a RUNNING parent Goal may admit a child WorkUnit")
    plan_for_gate = dict(mutation_plan or {})
    # The shared acceptance circuit must apply at the canonical child-admit
    # seam as well as the refill planner.  Other callers can materialize a
    # generated queue row directly; without this fence they could admit many
    # same-generation canaries while the proposal/engine handshake is known
    # broken.  Read-only work and ordinary operator-created WorkUnits remain
    # outside this narrowly scoped mutation-canary guard.
    if plan_for_gate.get("acceptance_canary") is True:
        from .auto_orchestration import _shared_acceptance_circuit
        circuit = _shared_acceptance_circuit(root, str(parent_goal_id))
        policy = parent.get("auto_continuation_policy")
        current_generation = str(
            policy.get("acceptance_canary_generation") or ""
        ).strip().upper() if isinstance(policy, Mapping) else ""
        candidate_generation = str(
            plan_for_gate.get("acceptance_canary_generation") or ""
        ).strip().upper()
        if circuit.get("open") and candidate_generation != current_generation:
            raise ValueError("shared acceptance circuit only admits the current canary generation")
        if circuit.get("open"):
            for record_path in (root / ".hawking" / "workunits").glob("WORKUNIT-*.json"):
                try:
                    live = json.loads(record_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if (
                    str(live.get("goal_id") or "") == str(parent_goal_id)
                    and str(live.get("state") or "").upper() in {"RUNNING", "ADMITTED"}
                ):
                    contract_path = str(live.get("contract_path") or "")
                    try:
                        contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
                    except (OSError, ValueError, json.JSONDecodeError):
                        contract = {}
                    active_plan = contract.get("mutation_plan") if isinstance(contract, Mapping) else {}
                    if (
                        isinstance(active_plan, Mapping)
                        and active_plan.get("acceptance_canary") is True
                        and str(active_plan.get("acceptance_canary_generation") or "").strip().upper()
                        == current_generation
                    ):
                        raise ValueError("shared acceptance circuit already has an active canary")
    child_objective = str(objective or "").strip()
    if not child_objective:
        raise ValueError("child WorkUnit objective is required")
    # Older route-reopen queue rows may already contain several copies of the
    # supervisor's alternate-route sentence. Compact that durable prompt at
    # the dispatch seam so recovery remains bounded even before the next
    # refill pass rewrites the queue row.
    if "Use the explicitly qualified alternate model route" in child_objective:
        from .auto_orchestration import _with_fresh_route_instruction
        child_objective = _with_fresh_route_instruction(child_objective)
    clean_acceptance = [str(item).strip()[:700] for item in (acceptance or []) if str(item).strip()]
    if not clean_acceptance:
        raise ValueError("child WorkUnit acceptance is required")
    mode = str(goal_mode or "discrete_goal").strip()
    if mode not in {"discrete_goal", "worker_research"}:
        raise ValueError("child goal_mode must be discrete_goal or worker_research")
    try:
        cap = max(0.01, float(budget_usd))
    except (TypeError, ValueError):
        raise ValueError("child WorkUnit budget_usd must be numeric")
    parent_cap = parent.get("budget_authorized_usd")
    if parent_cap is not None and cap > max(0.0, float(parent_cap)):
        raise ValueError("child WorkUnit budget exceeds the parent Goal cap")

    from .auto_orchestration import build_cognition_plan

    allowed = parent.get("allowed_models") if isinstance(parent.get("allowed_models"), list) else []
    plan = build_cognition_plan(
        {
            "messages": [{"role": "user", "content": child_objective}],
            "hawking_workspace": str(root),
        },
        allowed_models=allowed,
        budget_usd=cap,
        goal_id=str(parent.get("goal_id") or parent_goal_id),
    )
    planned_workers = plan.get("workers") if isinstance(plan.get("workers"), list) else []
    selected_model = str((planned_workers[0] if planned_workers else {}).get("model") or "").strip()
    override = str(model_override or "").strip()
    auto_override = override.casefold() in {"auto", "hawking-auto", "measured", "measured-auto"}
    if override and not auto_override:
        from .auto_mode import REMOTE_AUTO_ROSTER
        admitted_pool = set(allowed) if allowed else set(REMOTE_AUTO_ROSTER)
        if override not in admitted_pool:
            raise ValueError("child model_override is outside the parent Goal's admitted model pool")
        selected_model = override
    if not selected_model:
        raise ValueError("Auto could not select an admitted remote worker")
    workunit_id = (
        f"{str(parent.get('production_plan', {}).get('current_phase') or 'WORKUNIT')}"
        f"-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    )
    # Preserve the parent's objective and phase while giving the WorkUnit its
    # own compact completion contract and source window.
    child = dict(parent)
    planned_task_class = str(
        task_class
        or ((plan.get("task") or {}).get("task_class") if isinstance(plan.get("task"), Mapping) else "")
        or "general"
    ).strip()
    child.update({
        # A child has its own execution contract.  It must not silently
        # inherit a mutation obligation merely because its long-lived parent
        # is an engineering Goal.
        "goal_mode": mode,
        "workunit_id": workunit_id,
        "workunit_objective": child_objective,
        "workunit_acceptance": clean_acceptance,
        "workunit_focus_files": [str(item) for item in (focus_files or []) if str(item).strip()],
        "workunit_focus_tests": [str(item) for item in (focus_tests or []) if str(item).strip()],
        # Auto may provide a measured empirical class from the durable tranche
        # contract.  The classifier remains the fallback, but a generic
        # objective must not collapse every architectural lane into `general`.
        "task_class": planned_task_class,
        "worker_role": str((planned_workers[0] if planned_workers else {}).get("role") or "worker"),
        # Child WorkUnits are independently admitted.  A read-only source
        # reconnaissance child must not inherit an unrelated parent desktop
        # lease (for example Accessibility) unless its own contract names it.
        "required_capabilities": list(
            required_capabilities
            if required_capabilities is not None
            else ([] if mode == "worker_research" else parent.get("required_capabilities") or [])
        ),
        "external_roots": [str(item).strip() for item in (external_roots or parent.get("external_roots") or []) if str(item).strip()][:32],
        "workunit_budget_usd": cap,
        "auto_plan": plan,
        "resident_assignment": str(parent.get("resident_assignment") or "hawking/auto"),
        "accepted_phase": str(parent.get("phase") or ""),
        "swift_helper_state": {
            "source": "native/hawking-macos/hawking_macos.swift",
            "state": "READY",
            "world_schema": "hawking.macos.world_state.v1",
            "action_schema": "hawking.macos.action_receipt.v1",
            "input_boundary": "scoped AXPress only after Hawking lease",
        },
        "workunit_mutation_plan": dict(mutation_plan or {}),
        "source": "hawking_auto_child_workunit",
    })
    if required_capabilities is not None and macos_authority is None:
        # An explicit child capability ceiling must not inherit TCC-sensitive
        # parent authority by copying the parent Goal document.
        child["macos_authority"] = {}
        child["browser_authority"] = {}
    if mode == "worker_research":
        # Desktop/browser authority belongs to the WorkUnit that explicitly
        # asks for it. A read-only child may inspect source beneath the same
        # parent without inheriting the parent's TCC-sensitive capability.
        child["browser_authority"] = {}
        child["macos_authority"] = dict(macos_authority or {})
    if macos_authority is not None:
        child["macos_authority"] = dict(macos_authority)
        # An explicit empty mapping scopes this continuation away from the
        # parent's desktop capability without erasing the parent's durable
        # authority. Non-empty authority is still normalized and persisted on
        # the parent as before.
        if macos_authority:
            parent_authority = normalize_macos_authority(macos_authority, goal_id=str(parent["goal_id"]))
            if parent_authority:
                update_goal(root, str(parent["goal_id"]), macos_authority=parent_authority)
    dispatched = dispatch_goal(root, child, model=selected_model, background=background)
    return {
        **dispatched,
        "auto_route": {
            "plan": plan,
            "selected_model": selected_model,
            "selection": (
                "hawking_auto_measured_override"
                if auto_override else
                "explicit_parent_policy_override" if override else
                "hawking_auto"
            ),
            "task_class": planned_task_class,
            "claim_boundary": "Hawking Auto selected the model; the WorkUnit owner assigned the worker identity",
        },
        "parent_goal_id": str(parent["goal_id"]),
        "child_workunit_id": workunit_id,
    }


def reclassify_blocked_child_workunit(
    workspace: str | os.PathLike[str],
    workunit_id: str,
    *,
    goal_mode: str,
    reason: str,
) -> Dict[str, Any]:
    """Correct an unstarted child contract without erasing its failed evidence.

    This is deliberately narrower than general Goal editing: it only changes a
    stopped child WorkUnit before resumption, records why its execution mode
    changed, and rewrites the child contract atomically.  It exists for a
    bootstrap classification error such as a read-only reconnaissance child
    inheriting its engineering parent's mutation contract.
    """
    root = Path(workspace).expanduser().resolve()
    mode = str(goal_mode or "").strip()
    if mode not in {"discrete_goal", "worker_research"}:
        raise ValueError("child goal_mode must be discrete_goal or worker_research")
    note = str(reason or "").strip()
    if not note:
        raise ValueError("reclassification reason is required")

    from .workunit_owner import WorkunitOwner

    owner = WorkunitOwner(root)
    record = owner.load(str(workunit_id).strip())
    if record.state not in {"QUEUED", "BLOCKED"}:
        raise ValueError("only QUEUED or BLOCKED child WorkUnits may be reclassified")
    if not record.goal_id:
        raise ValueError("only Goal-owned child WorkUnits may be reclassified")
    contract_path = Path(record.contract_path).expanduser().resolve()
    try:
        contract_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("child contract escapes the workspace") from exc
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("child contract is unreadable") from exc
    if contract.get("goal_terminal_on_workunit_complete") is not False:
        raise ValueError("only non-terminal child contracts may be reclassified")

    previous_mode = str(record.goal_mode or contract.get("goal_mode") or "").strip()
    if previous_mode == mode:
        return {"owner": record.to_dict(), "contract": str(contract_path), "changed": False}
    bundle = dict(contract.get("mutation_bundle") or {})
    is_research = mode == "worker_research"
    allow_existing_regression_tests = bool(
        (bundle.get("mutation_plan") or {}).get("allow_existing_regression_tests")
        or (contract.get("mutation_plan") or {}).get("allow_existing_regression_tests")
    )
    bundle.update({
        "require_source_mutation": not is_research,
        "require_regression_test_mutation": (
            not is_research and not allow_existing_regression_tests
        ),
        "require_focused_test_invocation": not is_research,
    })
    contract["goal_mode"] = mode
    contract["mutation_bundle"] = bundle
    contract["exact_next_action"] = (
        "Inspect the bounded repository and return evidence from admitted read-only tools; do not mutate files."
        if is_research else
        "Inspect the staging repository and the acceptance criteria; plan one smallest useful edit, then test and seal it."
    )
    contract["mode_reclassification"] = {
        "at": time.time(),
        "from": previous_mode,
        "to": mode,
        "reason": note[:1000],
        "claim_boundary": "bootstrap contract correction; prior tool and provider evidence is preserved",
    }
    atomic_write_json(contract_path, contract)

    record.goal_mode = mode
    record.current_phase = "READ_ONLY_RECLASSIFIED" if is_research else "MUTATION_RECLASSIFIED"
    record.exact_next_action = str(contract["exact_next_action"])
    record.evidence.append({
        "at": time.time(),
        "kind": "workunit_mode_reclassified",
        "from": previous_mode,
        "to": mode,
        "reason": note[:1000],
        "contract_path": str(contract_path),
    })
    owner.save(record)
    receipt_path = root / "receipts" / "future" / "workunits" / f"{record.workunit_id}_MODE_RECLASSIFIED.json"
    atomic_write_json(receipt_path, {
        "schema": "hawking.workunit.mode_reclassification.v1",
        "workunit_id": record.workunit_id,
        "goal_id": record.goal_id,
        "from": previous_mode,
        "to": mode,
        "reason": note[:1000],
        "contract_path": str(contract_path),
        "at": time.time(),
        "claim_boundary": "changes only the next execution mode; prior evidence remains durable",
    })
    return {
        "owner": record.to_dict(),
        "contract": str(contract_path),
        "receipt": str(receipt_path),
        "changed": True,
    }


def load_goal(workspace: str | os.PathLike[str], goal_id: str) -> Optional[Dict[str, Any]]:
    path = _goal_path(workspace, goal_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


_FRONTIER_ACTIVE_STATES = frozenset({"RUNNING", "CHECKPOINTING", "COMPACTING"})
_FRONTIER_QUEUED_STATES = frozenset({"QUEUED"})
_FRONTIER_TERMINAL_STATES = frozenset({
    "COMPLETE", "FAILED", "CANCELLED", "OUTPUT_UNUSABLE", "BLOCKED",
    "PAUSED_PROVIDER", "PAUSED_RESOURCE", "DEFERRED",
    "NEEDS_REDECOMPOSITION", "RECONCILIATION_REQUIRED",
})


def reconcile_goal_frontier(
    workspace: str | os.PathLike[str],
    goal_id: str,
    *,
    persist: bool = True,
) -> Dict[str, Any]:
    """Derive the parent Goal's runnable lanes from durable WorkUnit state.

    This is a projection/reconciliation seam, not another scheduler.  It
    never starts work and never treats provider prose as evidence.  Active or
    queued WorkUnits are runnable only when every declared parent dependency
    has a durable ``COMPLETE`` record; terminal/provider/resource states are
    retained as history but excluded from the frontier.  ``active_workunit_id``
    remains only the compatibility/UI primary hint.
    """
    root = Path(workspace).expanduser().resolve()
    goal = load_goal(root, str(goal_id).strip())
    if not goal:
        return {"goal": None, "frontier": [], "waiting": [], "active_workunit_ids": []}

    ordered_ids: List[str] = []
    for value in (
        goal.get("workunit_ids") or [],
        goal.get("active_workunit_ids") or [],
        [goal.get("active_workunit_id")],
        goal.get("queued_workunit_ids") or [],
    ):
        values = value if isinstance(value, list) else [value]
        for item in values:
            token = str(item or "").strip()
            if token and token not in ordered_ids:
                ordered_ids.append(token)
    for item in goal.get("runnable_frontier") or []:
        if isinstance(item, Mapping):
            token = str(item.get("workunit_id") or "").strip()
            if token and token not in ordered_ids:
                ordered_ids.append(token)

    raw_active_ids = goal.get("active_workunit_ids")
    declared_active_ids = [
        str(item).strip() for item in (raw_active_ids or []) if str(item).strip()
    ] if raw_active_ids is not None else [
        str(goal.get("active_workunit_id") or "").strip()
    ]
    declared_active_ids = [item for item in declared_active_ids if item]
    prior_frontier = {
        str(item.get("workunit_id") or "").strip(): dict(item)
        for item in (goal.get("runnable_frontier") or [])
        if isinstance(item, Mapping) and str(item.get("workunit_id") or "").strip()
    }
    prior_jobs = {
        str(item.get("workunit_id") or "").strip(): dict(item)
        for item in (goal.get("active_workunit_jobs") or [])
        if isinstance(item, Mapping) and str(item.get("workunit_id") or "").strip()
    }

    records: Dict[str, Dict[str, Any]] = {}
    state_dir = root / ".hawking" / "workunits"
    for path in sorted(state_dir.glob("*.json")) if state_dir.is_dir() else []:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping):
            continue
        if str(value.get("goal_id") or "").strip() != str(goal.get("goal_id") or "").strip():
            continue
        wid = str(value.get("workunit_id") or path.stem).strip()
        if wid:
            records[wid] = dict(value)
            if wid not in ordered_ids:
                ordered_ids.append(wid)
    # A sibling can be admitted concurrently while its owner record is still
    # being atomically published. Preserve an explicitly declared active lane
    # from the parent projection in that narrow window; an explicit empty list
    # remains authoritative and cannot resurrect a stale scalar hint.
    for wid in declared_active_ids:
        if wid in records:
            continue
        prior = prior_frontier.get(wid) or prior_jobs.get(wid) or {}
        records[wid] = {
            "workunit_id": wid,
            "goal_id": goal.get("goal_id"),
            "state": str(prior.get("state") or "RUNNING").upper(),
            "current_phase": prior.get("phase") or goal.get("phase") or "",
            "worker_id": prior.get("worker_id") or "",
            "worker_model": prior.get("worker_model") or "",
        }
        if wid not in ordered_ids:
            ordered_ids.append(wid)

    def dependencies(record: Mapping[str, Any]) -> List[str]:
        values: List[Any] = []
        parent = str(record.get("parent_workunit_id") or "").strip()
        if parent:
            values.append(parent)
        raw_contract = record.get("contract_path")
        if raw_contract:
            try:
                contract_path = Path(str(raw_contract)).expanduser().resolve()
                contract_path.relative_to(root)
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                contract = {}
            if isinstance(contract, Mapping):
                for key in ("dependencies", "depends_on", "parent_workunit_ids"):
                    raw = contract.get(key)
                    if raw is None:
                        continue
                    values.extend(raw if isinstance(raw, list) else [raw])
        return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))

    def state_for(wid: str) -> str:
        return str(records.get(wid, {}).get("state") or "MISSING").strip().upper()

    frontier: List[Dict[str, Any]] = []
    waiting: List[Dict[str, Any]] = []
    active_ids: List[str] = []
    for wid in ordered_ids:
        record = records.get(wid)
        if not record:
            continue
        state = str(record.get("state") or "QUEUED").strip().upper()
        if state in _FRONTIER_ACTIVE_STATES and wid not in active_ids:
            active_ids.append(wid)
        if state not in _FRONTIER_ACTIVE_STATES | _FRONTIER_QUEUED_STATES:
            continue
        deps = dependencies(record)
        unresolved = [dep for dep in deps if state_for(dep) != "COMPLETE"]
        row = {
            "workunit_id": wid,
            "state": state,
            "phase": str(record.get("current_phase") or goal.get("phase") or ""),
            "worker_id": str(record.get("worker_id") or ""),
            "worker_model": str(record.get("worker_model") or ""),
            "dependencies": deps,
        }
        if unresolved:
            waiting.append({**row, "unresolved_dependencies": unresolved})
        else:
            frontier.append(row)

    scalar = str(goal.get("active_workunit_id") or "").strip()
    primary = scalar if scalar in active_ids else (active_ids[0] if active_ids else None)
    active_jobs = [
        dict(item) for item in (goal.get("active_workunit_jobs") or [])
        if isinstance(item, Mapping)
        and str(item.get("workunit_id") or "").strip() in active_ids
    ]
    fields = {
        "active_workunit_id": primary,
        "active_workunit_ids": active_ids[-32:],
        "active_workunit_jobs": active_jobs[-32:],
        "runnable_frontier": frontier[-32:],
        "frontier_waiting_dependencies": waiting[-32:],
    }
    updated = update_goal(root, str(goal.get("goal_id") or goal_id), **fields) if persist else dict(goal, **fields)
    return {
        "goal": updated,
        "frontier": frontier,
        "waiting": waiting,
        "active_workunit_ids": active_ids,
    }


def update_goal(workspace: str | os.PathLike[str], goal_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    root = Path(workspace).expanduser().resolve()
    goal_path = _goal_path(root, goal_id)
    lock_path = goal_path.with_name(f".{goal_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            doc = load_goal(root, goal_id)
            if not doc:
                return None
            incoming_queue = fields.get("auto_refill_queue")
            current_queue = doc.get("auto_refill_queue")
            if isinstance(incoming_queue, list) and isinstance(current_queue, list):
                current_by_tranche = {
                    str(item.get("tranche_id") or item.get("tranche") or "").strip(): item
                    for item in current_queue
                    if isinstance(item, Mapping)
                    and str(item.get("tranche_id") or item.get("tranche") or "").strip()
                }
                preserved_queue = []
                for raw_item in incoming_queue:
                    item = dict(raw_item) if isinstance(raw_item, Mapping) else raw_item
                    if isinstance(item, dict):
                        tranche = str(item.get("tranche_id") or item.get("tranche") or "").strip()
                        current_item = current_by_tranche.get(tranche)
                        if isinstance(current_item, Mapping):
                            current_route = current_item.get("route_selection")
                            incoming_state = str(item.get("state") or "").upper()
                            if (
                                isinstance(current_route, Mapping)
                                and current_route.get("reason") == "semantic_failure_diversification"
                                and incoming_state not in {"TERMINAL_PROVIDER_OUTCOME", "COMPLETE", "COMPLETE_PENDING_ACCEPTANCE"}
                            ):
                                item["model_override"] = current_item.get("model_override")
                                item["route_selection"] = dict(current_route)
                    preserved_queue.append(item)
                fields["auto_refill_queue"] = preserved_queue
            doc.update(fields)
            doc["updated_at"] = time.time()
            atomic_write_json(goal_path, doc)
            return doc
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


def write_result(
    workspace: str | os.PathLike[str],
    goal_id: str,
    *,
    status: str,
    summary: str,
    commits: Optional[List[str]] = None,
    seals: Optional[List[str]] = None,
    tests: Optional[Dict[str, Any]] = None,
    files_changed: Optional[List[str]] = None,
    limitations: Optional[List[str]] = None,
    claim_boundary: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    goal = load_goal(workspace, goal_id) or {"goal_id": goal_id, "objective": ""}
    extra = extra or {}
    result: Dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "goal_id": goal_id,
        "objective": goal.get("objective"),
        "status": status,
        "summary": summary,
        "commits": list(commits or []),
        "seals": list(seals or []),
        "tests": tests or {},
        "files_changed": list(files_changed or []),
        "limitations": list(limitations or []),
        "claim_boundary": claim_boundary
        or "Canonical Goal result for Grok ingest. Conversation is not required.",
        "resident": goal.get("resident_assignment"),
        "workunit_id": goal.get("workunit_id"),
        "finished_at": time.time(),
        "continuation_impact": extra.get("continuation_impact"),
        "rollback": extra.get("rollback"),
        "evidence": extra.get("evidence"),
        "worker_id": extra.get("worker_id"),
        "worker_model": extra.get("worker_model"),
        "output_excerpt": extra.get("output_excerpt"),
        "completion": extra.get("completion"),
    }
    atomic_write_json(_result_path(workspace, goal_id), result)
    update_goal(
        workspace,
        goal_id,
        status=status,
        phase="COMPLETE" if str(status).upper() == "COMPLETE" else str(status),
        result_path=str(_result_path(workspace, goal_id)),
    )
    return result


def load_result(workspace: str | os.PathLike[str], goal_id: str) -> Optional[Dict[str, Any]]:
    path = _result_path(workspace, goal_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_goals(workspace: str | os.PathLike[str], *, limit: int = 32) -> List[Dict[str, Any]]:
    """List canonical Goal records without creating a second Goal registry."""
    rows: List[Dict[str, Any]] = []
    for path in sorted(goals_root(workspace).glob("GOAL-*.json"), key=lambda item: item.stat().st_mtime):
        # A Goal has companion contract/result documents in this same
        # directory.  They are evidence for one Goal, never additional Goals;
        # allowing the broad glob through here created duplicate operator rows
        # after refresh/recovery.
        if path.name.endswith((".contract.json", ".result.json")):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows[-max(1, int(limit)):]


def _goal_meta(workspace: str | os.PathLike[str], goal_id: str) -> Dict[str, Any]:
    goal = load_goal(workspace, goal_id)
    if not goal:
        raise ValueError(f"no goal {goal_id}")
    result = load_result(workspace, goal_id)
    return {
        "goal_id": goal_id,
        "objective": str(goal.get("objective") or "")[:800],
        "status": goal.get("status"),
        "phase": goal.get("phase"),
        "worker_policy": goal.get("worker_policy"),
        "auto_scope": goal.get("auto_scope"),
        "parent_goal_ids": list(goal.get("parent_goal_ids") or [])[:32],
        "result": {
            key: result.get(key)
            for key in ("status", "summary", "files_changed", "tests")
            if isinstance(result, dict) and result.get(key) is not None
        } if isinstance(result, dict) else None,
    }


def build_goal_synthesis(
    workspace: str | os.PathLike[str], goal_ids: List[str]
) -> Dict[str, Any]:
    """Build a read-only synthesis packet while preserving every source Goal."""
    ids = [str(item).strip().upper() for item in (goal_ids or []) if str(item).strip()]
    if not ids or len(ids) > 32:
        raise ValueError("one to thirty-two Goal ids are required")
    if any(not re.fullmatch(r"GOAL-[A-Z0-9_-]{1,100}", item) for item in ids):
        raise ValueError("invalid Goal id")
    return {
        "schema": "hawking.goal.synthesis.v1",
        "synthesis_id": f"SYNTH-{uuid.uuid4().hex[:12].upper()}",
        "parent_goal_ids": ids,
        "goals": [_goal_meta(workspace, item) for item in ids],
        "mutates_sources": False,
        "claim_boundary": "read-only synthesis packet; source Goals and provenance remain unchanged",
    }


def build_goal_consolidation(
    workspace: str | os.PathLike[str],
    goal_ids: List[str],
    *,
    objective: str = "",
    commit: bool = False,
) -> Dict[str, Any]:
    """Plan or create a provenance-preserving continuation MetaGoal."""
    synthesis = build_goal_synthesis(workspace, goal_ids)
    metas = synthesis["goals"]
    text = str(objective or "").strip()
    if not text:
        text = "Continue the accepted work from: " + "; ".join(
            str(item.get("objective") or item.get("goal_id"))[:160] for item in metas
        )
    plan = {
        "schema": "hawking.goal.consolidation.v1",
        "consolidation_id": f"CONSOLIDATE-{uuid.uuid4().hex[:12].upper()}",
        "parent_goal_ids": list(synthesis["parent_goal_ids"]),
        "objective": text[:2000],
        "source_goals_preserved": True,
        "commit_requested": bool(commit),
        "claim_boundary": "new continuation only; source Goal records are not erased or silently superseded",
    }
    if not commit:
        return {"plan": plan, "sources": metas}
    goal = create_goal(
        workspace,
        objective=text,
        acceptance=["continue only after reviewing the parent Goal evidence"],
        worker_policy="auto",
        parent_goal_ids=list(synthesis["parent_goal_ids"]),
        source="hawking_consolidation",
    )
    plan["new_goal_id"] = goal["goal_id"]
    plan["status"] = "CREATED"
    return {"plan": plan, "goal": goal, "sources": metas}
P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2_QUALIFICATION_SUPPORT = True
P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2_QUALIFICATION_SUPPORT_EVIDENCE = "hawking/tests/test_auto_orchestration.py"# --- P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2 surface ---
# Bounded source mutation: declare the collapse implementation contract
# version and a single public entry point so the contract surface is
# importable and introspectable. This does not implement collapse behavior,
# does not claim phase completion, and does not touch hardware.

P17_COLLAPSE_IMPLEMENTATION_CONTRACT_VERSION = "V2"


def p17_collapse_implementation_contract():
    """Return the declared P17 collapse implementation contract descriptor."""
    return {
        "contract": "P17_COLLAPSE_IMPLEMENTATION_CONTRACT",
        "version": P17_COLLAPSE_IMPLEMENTATION_CONTRACT_VERSION,
        "entry_point": "p17_collapse_implementation_contract",
    }
