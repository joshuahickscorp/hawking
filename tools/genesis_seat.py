#!/usr/bin/env python3
"""Seat the current best verified ancestor as GENESIS, and report the lineage.

The lineage machinery (lab/lineage/) was built and tested but never turned on, so
nothing was actually seated - Genesis existed as a tournament result and a set of
receipts rather than as a running lineage. This seats it.

Generation 0 is Qwen3.8 uniform-q4-v1 at 4.2527 BPW / 35,227,918 ns. Nothing has
beaten it: the fusion child regressed 10.68 ms, the attention codec produced a
cosine-only candidate with the complete token unmoved, cross-token reuse was
refuted, and shared sessions save memory without amortizing DRAM. That is what
makes it CURRENT - not that it is good, but that it is the best VERIFIED ancestor.

    genesis_seat.py seat     # record generation 0 as CURRENT + LAST_KNOWN_GOOD
    genesis_seat.py status   # print the three slots
    genesis_seat.py bind-live --verification PATH  # bind an observed resident

Seating records artifact authority. It does not launch a model process and must
not claim that CURRENT is launched or live merely because it occupies that slot.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lab.lineage.identity import (  # noqa: E402
    GENESIS_ARTIFACT_MANIFEST_PATH,
    GENESIS_MODEL,
    GenesisInstance,
    file_sha256,
    make_qwen38_genesis,
)
from lab.lineage.canon import digest, require_sha256, utc_now  # noqa: E402
from lab.lineage.state import LineageState  # noqa: E402
from lab.qwen38_protected_run_verifier import VERIFICATION_SCHEMA  # noqa: E402
from tools.agentos.genesis_resident import (  # noqa: E402
    default_socket,
    health as resident_health,
    process_alive,
)

STATE_FILE = REPO / "receipts" / "ascent-2026-08-16" / "GENESIS_LINEAGE_CURRENT.json"
ARTIFACT_MANIFEST = REPO / GENESIS_ARTIFACT_MANIFEST_PATH
ARTIFACT_ROOT = ARTIFACT_MANIFEST.parent
WALL_RECEIPT = (
    REPO / "receipts" / "ascent-2026-08-16" / "QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL.json"
)
KERNEL_SOURCE = REPO / "crates" / "hawking-core" / "shaders" / "qwen_uniform_q4.metal"
EXPECTED_SESSION_ROLES = ("parent", "child_a", "child_b", "protected_test")
EXPECTED_SESSION_KINDS = {
    "parent": "lineage_parent",
    "child_a": "worker_session",
    "child_b": "worker_session",
    "protected_test": "protected_test_slot",
}
RESIDENT_PROTOCOL = "hawking.genesis.resident.v1"
LIVE_BINDING_SCHEMA = "hawking.lineage.live_binding.v1"


class BindLiveError(ValueError):
    """Observed resident evidence is insufficient or inconsistent."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BindLiveError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BindLiveError(f"cannot read {label} {path}: {exc}") from exc
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BindLiveError(f"invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BindLiveError(f"{label} must be a JSON object")
    return value, raw


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BindLiveError(f"{label} must be an object")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise BindLiveError(f"{label} must be a positive integer")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BindLiveError(f"{label} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise BindLiveError(f"{label} must be a positive finite number")
    return number


def _required_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BindLiveError(f"{label} must be a non-empty path string")
    return Path(value)


def _regular_file(path: Path, label: str, *, executable: bool = False) -> Path:
    try:
        resolved = path.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise BindLiveError(f"cannot resolve {label} {path}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise BindLiveError(f"{label} is not a regular file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise BindLiveError(f"{label} is not executable: {resolved}")
    return resolved


def process_executable_path(pid: int) -> Path:
    """Resolve the executable owned by an already-live pid without launching it."""
    pid_i = _positive_int(pid, "resident pid")
    proc_exe = Path(f"/proc/{pid_i}/exe")
    if proc_exe.exists():
        return _regular_file(proc_exe, "resident process executable", executable=True)

    if sys.platform == "darwin":
        try:
            import ctypes

            libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            buffer = ctypes.create_string_buffer(4096)
            written = libproc.proc_pidpath(pid_i, buffer, len(buffer))
            if written > 0:
                return _regular_file(
                    Path(os.fsdecode(buffer.value)),
                    "resident process executable",
                    executable=True,
                )
        except (OSError, ValueError):
            pass

    try:
        result = subprocess.run(
            ["ps", "-p", str(pid_i), "-o", "comm="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BindLiveError(f"cannot resolve executable for resident pid {pid_i}: {exc}") from exc
    candidate = result.stdout.strip()
    if result.returncode != 0 or not candidate:
        raise BindLiveError(f"cannot resolve executable for resident pid {pid_i}")
    return _regular_file(Path(candidate), "resident process executable", executable=True)


def _require_matching_sha(binding: Mapping[str, Any], name: str, actual: str) -> None:
    try:
        claimed = require_sha256(binding.get(name), f"binding.{name}")
    except ValueError as exc:
        raise BindLiveError(str(exc)) from exc
    if not hmac.compare_digest(claimed, actual):
        raise BindLiveError(f"binding.{name} does not match independently hashed bytes")


def _atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    expected_previous_sha256: str | None,
) -> None:
    """Fsync a sibling temporary file and replace only expected prior state."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        previous_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        if expected_previous_sha256 is not None:
            raise BindLiveError("lineage state disappeared before atomic write")
        previous_mode = 0o644
    except OSError as exc:
        raise BindLiveError(f"cannot stat lineage state before atomic write: {exc}") from exc
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.bind-live-", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, previous_mode)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if expected_previous_sha256 is None:
            if path.exists():
                raise BindLiveError("lineage state appeared during atomic write")
        else:
            try:
                observed_previous = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise BindLiveError(
                    f"cannot re-read lineage state before replace: {exc}"
                ) from exc
            if not hmac.compare_digest(observed_previous, expected_previous_sha256):
                raise BindLiveError("lineage state changed during bind-live; refusing overwrite")
        try:
            os.replace(tmp, path)
        except OSError as exc:
            raise BindLiveError(f"atomic lineage replace failed: {exc}") from exc
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        if fd >= 0:
            os.close(fd)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def build_seated_lineage(
    *, artifact_manifest: Path = ARTIFACT_MANIFEST
) -> tuple[LineageState, GenesisInstance]:
    """Build a generation-0 lineage without claiming process liveness."""
    artifact_sha = file_sha256(artifact_manifest)
    g0 = make_qwen38_genesis(artifact_sha=artifact_sha)
    lineage = LineageState()
    lineage.install(g0)
    lineage.snapshot_current_as_lkg()
    return lineage, g0


def bind_live_state(
    *,
    verification_path: Path,
    state_file: Path = STATE_FILE,
    capture_path: Path = WALL_RECEIPT,
    artifact_root: Path = ARTIFACT_ROOT,
    kernel_source: Path = KERNEL_SOURCE,
    sock_path: Path | None = None,
    health_query: Callable[[Path], Mapping[str, Any] | None] = resident_health,
    alive_query: Callable[[int], bool] = process_alive,
    executable_query: Callable[[int], Path] = process_executable_path,
) -> dict[str, Any]:
    """Bind CURRENT to an observed resident without starting or stopping it.

    The immutable timing capture is accepted only through a separately generated
    protected-verifier result. All local file identities and resident process
    facts are derived again here before the lineage state is atomically replaced.
    """
    state_path = Path(state_file)
    capture_file = _regular_file(Path(capture_path), "complete-token capture")
    verification_file = _regular_file(Path(verification_path), "protected verification")
    manifest_file = _regular_file(Path(artifact_root) / "manifest.json", "artifact manifest")
    kernel_file = _regular_file(Path(kernel_source), "kernel source")

    state, state_raw = _load_json_object(state_path, "lineage state")
    capture, capture_raw = _load_json_object(capture_file, "complete-token capture")
    verification, verification_raw = _load_json_object(
        verification_file, "protected verification"
    )
    verification_sha = hashlib.sha256(verification_raw).hexdigest()
    manifest, _manifest_raw = _load_json_object(manifest_file, "artifact manifest")

    if state.get("schema") != "hawking.lineage.state.v1" or state.get("armed") is not True:
        raise BindLiveError("lineage state must be an armed hawking.lineage.state.v1")
    slots = _mapping(state.get("slots"), "lineage slots")
    if set(slots) != {"CURRENT", "CANDIDATE", "LAST_KNOWN_GOOD"}:
        raise BindLiveError("lineage must contain exactly CURRENT, CANDIDATE, LAST_KNOWN_GOOD")
    current_raw = _mapping(slots.get("CURRENT"), "CURRENT")
    lkg_raw = _mapping(slots.get("LAST_KNOWN_GOOD"), "LAST_KNOWN_GOOD")
    try:
        current = GenesisInstance.from_mapping(current_raw)
        lkg = GenesisInstance.from_mapping(lkg_raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise BindLiveError(f"invalid lineage occupant: {exc}") from exc
    if not current.valid or current.terminated:
        raise BindLiveError("CURRENT must be valid and non-terminated before live binding")
    if current.identity.get("model") != GENESIS_MODEL:
        raise BindLiveError("CURRENT is not the protected Qwen3.8 model")

    artifact_dir = Path(artifact_root).resolve(strict=True)
    manifest_sha = file_sha256(manifest_file)
    if current.artifact_sha != manifest_sha:
        raise BindLiveError("CURRENT artifact_sha does not match actual manifest bytes")
    current_artifact = current.identity.get("artifact")
    if not isinstance(current_artifact, str) or not current_artifact:
        raise BindLiveError("CURRENT identity.artifact is missing")
    claimed_current_artifact = Path(current_artifact)
    if not claimed_current_artifact.is_absolute():
        claimed_current_artifact = REPO / claimed_current_artifact
    try:
        claimed_current_artifact = claimed_current_artifact.resolve(strict=True)
    except OSError as exc:
        raise BindLiveError(f"cannot resolve CURRENT artifact: {exc}") from exc
    if claimed_current_artifact != artifact_dir:
        raise BindLiveError("CURRENT artifact path does not match protected artifact root")

    if verification.get("schema") != VERIFICATION_SCHEMA:
        raise BindLiveError(f"verification schema must be {VERIFICATION_SCHEMA}")
    if verification.get("status") != "PASS":
        raise BindLiveError("protected verification status must be PASS")
    capture_sha = hashlib.sha256(capture_raw).hexdigest()
    try:
        claimed_capture_sha = require_sha256(
            verification.get("capture_sha256"), "verification.capture_sha256"
        )
    except ValueError as exc:
        raise BindLiveError(str(exc)) from exc
    if not hmac.compare_digest(capture_sha, claimed_capture_sha):
        raise BindLiveError("protected verification does not bind the immutable timing capture")

    protected_binding = _mapping(
        verification.get("protected_binding"), "verification.protected_binding"
    )
    try:
        verified_artifact_root = _required_path(
            protected_binding.get("artifact_root"),
            "verification.protected_binding.artifact_root",
        ).resolve(strict=True)
        verified_manifest_path = _regular_file(
            _required_path(
                protected_binding.get("artifact_manifest_path"),
                "verification.protected_binding.artifact_manifest_path",
            ),
            "verified artifact manifest",
        )
        verified_kernel_path = _regular_file(
            _required_path(
                protected_binding.get("kernel_source_path"),
                "verification.protected_binding.kernel_source_path",
            ),
            "verified kernel source",
        )
    except OSError as exc:
        raise BindLiveError(f"cannot resolve protected binding path: {exc}") from exc
    if verified_artifact_root != artifact_dir or not verified_artifact_root.is_dir():
        raise BindLiveError("verification artifact root differs from protected artifact root")
    if verified_manifest_path != manifest_file:
        raise BindLiveError("verification manifest path differs from protected manifest")
    if verified_kernel_path != kernel_file:
        raise BindLiveError("verification kernel path differs from protected kernel source")
    _require_matching_sha(
        protected_binding, "artifact_manifest_sha256", manifest_sha
    )
    kernel_sha = file_sha256(kernel_file)
    _require_matching_sha(protected_binding, "kernel_source_sha256", kernel_sha)
    try:
        measurement_runtime_sha = require_sha256(
            protected_binding.get("runtime_executable_sha256"),
            "verification.protected_binding.runtime_executable_sha256",
        )
    except ValueError as exc:
        raise BindLiveError(str(exc)) from exc
    measurement_runtime = _regular_file(
        _required_path(
            protected_binding.get("runtime_executable_path"),
            "verification.protected_binding.runtime_executable_path",
        ),
        "verified measurement runtime",
        executable=True,
    )
    if not hmac.compare_digest(file_sha256(measurement_runtime), measurement_runtime_sha):
        raise BindLiveError(
            "verification runtime SHA does not match independently hashed runtime bytes"
        )
    if protected_binding.get("candidate_hashes_were_not_authority") is not True:
        raise BindLiveError("verification must derive protected hashes independently")

    measurement = _mapping(verification.get("measurement"), "verification.measurement")
    if measurement.get("model") != GENESIS_MODEL:
        raise BindLiveError("verified measurement is not the protected Qwen3.8 model")
    complete_token_ns = _positive_int(
        measurement.get("derived_headline_complete_wall_ns_per_token"),
        "verification.measurement.derived_headline_complete_wall_ns_per_token",
    )
    complete_tps = _positive_number(
        measurement.get("derived_complete_tps"),
        "verification.measurement.derived_complete_tps",
    )
    if not math.isclose(
        complete_tps,
        1_000_000_000.0 / complete_token_ns,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise BindLiveError("verified TPS is inconsistent with verified complete-token wall")
    if measurement.get("fallbacks") != 0:
        raise BindLiveError("verified measurement must have zero silent fallbacks")

    if capture.get("schema") != "hawking.ascent.qwen38_complete_token_wall.v1":
        raise BindLiveError("capture schema is not the current-main complete-token wall")
    claim_boundary = _mapping(
        verification.get("claim_boundary"), "verification.claim_boundary"
    )
    if claim_boundary.get("wall_rederived_from_raw_capture") is not True:
        raise BindLiveError("verification did not rederive the wall from raw capture samples")
    if claim_boundary.get("capture_origin_attested") is not False:
        raise BindLiveError("current timing capture must remain origin-unattested")
    if capture.get("timing_label") != "DIRTY_ENGINEERING":
        raise BindLiveError("current timing capture must retain DIRTY_ENGINEERING label")
    capture_authority = _mapping(capture.get("authority"), "capture.authority")
    if (
        _positive_int(
            capture_authority.get("headline_complete_wall_ns_per_token"),
            "capture.authority.headline_complete_wall_ns_per_token",
        )
        != complete_token_ns
    ):
        raise BindLiveError("capture wall differs from protected verification")
    capture_tps = _positive_number(
        capture_authority.get("headline_complete_tps"),
        "capture.authority.headline_complete_tps",
    )
    if not math.isclose(capture_tps, complete_tps, rel_tol=1e-12, abs_tol=1e-12):
        raise BindLiveError("capture TPS differs from protected verification")
    capture_identity = _mapping(capture.get("identity"), "capture.identity")
    if capture_identity.get("model") != GENESIS_MODEL or capture_identity.get("fallbacks") != 0:
        raise BindLiveError("capture identity is not fallback-free protected Qwen3.8")
    if capture_identity.get("kernel") != "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128":
        raise BindLiveError("capture kernel is not the protected Qwen3.8 Q4 kernel")

    vehicle = _mapping(capture.get("vehicle"), "capture.vehicle")
    physical_bpw = _positive_number(
        vehicle.get("complete_physical_bpw"), "capture.vehicle.complete_physical_bpw"
    )
    manifest_bpw = _positive_number(
        manifest.get("complete_physical_bpw"), "manifest.complete_physical_bpw"
    )
    if not math.isclose(physical_bpw, manifest_bpw, rel_tol=1e-12, abs_tol=1e-12):
        raise BindLiveError("capture physical BPW differs from artifact manifest")
    claimed_vehicle_root = vehicle.get("artifact_root")
    if not isinstance(claimed_vehicle_root, str):
        raise BindLiveError("capture.vehicle.artifact_root must be a path string")
    claimed_vehicle_path = Path(claimed_vehicle_root)
    if not claimed_vehicle_path.is_absolute():
        claimed_vehicle_path = REPO / claimed_vehicle_path
    try:
        claimed_vehicle_path = claimed_vehicle_path.resolve(strict=True)
    except OSError as exc:
        raise BindLiveError(f"cannot resolve capture artifact root: {exc}") from exc
    if claimed_vehicle_path != artifact_dir:
        raise BindLiveError("capture artifact root differs from protected artifact root")

    socket_path = Path(sock_path) if sock_path is not None else default_socket(REPO)
    observed = health_query(socket_path)
    if not isinstance(observed, Mapping):
        raise BindLiveError("resident health socket did not return a live observation")
    if observed.get("ok") is not True or observed.get("protocol") != RESIDENT_PROTOCOL:
        raise BindLiveError("resident health protocol is invalid")
    if observed.get("body_resident") is not True:
        raise BindLiveError("resident health did not prove body_resident=true")
    if observed.get("stub") is True:
        raise BindLiveError("stub resident cannot be bound live")
    if observed.get("load_count") != 1 or type(observed.get("load_count")) is not int:
        raise BindLiveError("resident must report exactly one body load")
    roles = observed.get("session_roles")
    if not isinstance(roles, list) or tuple(roles) != EXPECTED_SESSION_ROLES:
        raise BindLiveError(f"resident roles must be exactly {EXPECTED_SESSION_ROLES!r}")
    if observed.get("session_count") != len(EXPECTED_SESSION_ROLES):
        raise BindLiveError("resident session_count must be exactly four")
    session_workspaces = _mapping(
        observed.get("session_workspace_bytes"), "resident session_workspace_bytes"
    )
    if set(session_workspaces) != set(EXPECTED_SESSION_ROLES):
        raise BindLiveError("resident workspace roles do not match the four protected roles")
    semantics = _mapping(observed.get("session_semantics"), "resident session_semantics")
    if set(semantics) != set(EXPECTED_SESSION_ROLES):
        raise BindLiveError("resident session semantics do not cover exactly four roles")
    for role, expected_kind in EXPECTED_SESSION_KINDS.items():
        role_semantics = _mapping(
            semantics.get(role), f"resident session_semantics.{role}"
        )
        if role_semantics.get("kind") != expected_kind:
            raise BindLiveError(
                f"resident role {role!r} must have kind {expected_kind!r}"
            )
    if observed.get("lineage_children") != 0 or type(observed.get("lineage_children")) is not int:
        raise BindLiveError("resident must report zero lineage children")
    _positive_int(observed.get("resident_weight_bytes"), "resident_weight_bytes")
    if observed.get("reload_error") not in (None, ""):
        raise BindLiveError("resident reports a reload error")
    pid = _positive_int(observed.get("pid"), "resident pid")
    if not alive_query(pid):
        raise BindLiveError("resident pid is not live")
    if observed.get("generation") != current.generation:
        raise BindLiveError("resident generation differs from CURRENT")
    observed_artifact = observed.get("artifact")
    if not isinstance(observed_artifact, str):
        raise BindLiveError("resident artifact path is missing")
    try:
        observed_artifact_path = Path(observed_artifact).resolve(strict=True)
    except OSError as exc:
        raise BindLiveError(f"cannot resolve resident artifact path: {exc}") from exc
    if observed_artifact_path != artifact_dir:
        raise BindLiveError("resident artifact path differs from CURRENT")
    try:
        resident_manifest_sha = require_sha256(
            observed.get("artifact_sha"), "resident artifact_sha"
        )
    except ValueError as exc:
        raise BindLiveError(str(exc)) from exc
    if not hmac.compare_digest(resident_manifest_sha, manifest_sha):
        raise BindLiveError("resident artifact SHA differs from actual manifest bytes")

    resident_executable = _regular_file(
        Path(executable_query(pid)), "resident process executable", executable=True
    )
    if resident_executable.name != "genesis-resident":
        raise BindLiveError("live pid does not own the genesis-resident executable")
    resident_executable_sha = file_sha256(resident_executable)
    if not alive_query(pid):
        raise BindLiveError("resident pid exited during live binding")

    current.complete_token_ns = complete_token_ns
    current.physical_bpw = physical_bpw
    current.artifact_sha = manifest_sha
    current.runtime_sha = resident_executable_sha
    current.kernel_genome_sha = kernel_sha
    current.live = True
    current.launched = True
    current.terminated = False
    current.identity.update(
        {
            "artifact_manifest": str(manifest_file),
            "artifact_sha_authority": "sha256(manifest.json bytes)",
            "resident_pid": str(pid),
            "resident_socket": str(socket_path),
            "resident_executable": str(resident_executable),
            "resident_executable_sha256": resident_executable_sha,
            "kernel_source": str(kernel_file),
            "kernel_source_sha256": kernel_sha,
            "measurement_runtime_sha256": measurement_runtime_sha,
            "complete_token_receipt": str(capture_file),
            "protected_verification": str(verification_file),
            "protected_verification_sha256": verification_sha,
            "complete_physical_bpw": repr(physical_bpw),
        }
    )
    lkg.live = False
    lkg.launched = False

    updated = copy.deepcopy(state)
    updated_slots = dict(_mapping(updated.get("slots"), "lineage slots"))
    updated_slots["CURRENT"] = current.to_dict()
    updated_slots["LAST_KNOWN_GOOD"] = lkg.to_dict()
    updated["slots"] = updated_slots
    observed_at = utc_now()
    observation = {
        "schema": LIVE_BINDING_SCHEMA,
        "observed_at": observed_at,
        "pid": pid,
        "socket": str(socket_path),
        "body_resident": True,
        "load_count": 1,
        "session_roles": list(EXPECTED_SESSION_ROLES),
        "session_kinds": dict(EXPECTED_SESSION_KINDS),
        "lineage_children": 0,
        "artifact_manifest_sha256": manifest_sha,
        "resident_executable_sha256": resident_executable_sha,
        "kernel_source_sha256": kernel_sha,
        "measurement_runtime_sha256": measurement_runtime_sha,
        "capture_sha256": capture_sha,
        "verification_sha256": verification_sha,
        "verification_path": str(verification_file),
        "complete_token_ns": complete_token_ns,
        "tps": complete_tps,
        "complete_physical_bpw": physical_bpw,
        "timing_label": capture.get("timing_label"),
        "capture_origin_attested": bool(
            claim_boundary.get("capture_origin_attested", False)
        ),
    }
    observation["observation_sha256"] = digest(observation)
    events = updated.get("events")
    if not isinstance(events, list):
        raise BindLiveError("lineage events must be a list")
    events.append({"ts": observed_at, "kind": "bind_live_observed", "payload": observation})

    previous_state_sha = hashlib.sha256(state_raw).hexdigest()
    _atomic_write_json(
        state_path,
        updated,
        expected_previous_sha256=previous_state_sha,
    )
    return updated


def seat() -> int:
    lineage, g0 = build_seated_lineage()

    previous_sha: str | None = None
    if STATE_FILE.is_file():
        previous_sha = file_sha256(STATE_FILE)
    _atomic_write_json(
        STATE_FILE,
        lineage.to_dict(),
        expected_previous_sha256=previous_sha,
    )
    print(f"SEATED generation {g0.generation}: {g0.instance_id}")
    print(f"  BPW              {g0.representation_bpw}")
    print(f"  complete token   {g0.complete_token_ns:,} ns")
    print(f"  artifact sha     {g0.artifact_sha[:16]}")
    print(f"  valid instances  {lineage.valid_count()}")
    print(f"  state            {STATE_FILE.relative_to(REPO)}")
    return 0


def bind_live(*, verification_path: Path, sock_path: Path | None = None) -> int:
    """Bind CURRENT to an already-live, fully verified resident observation."""
    try:
        updated = bind_live_state(
            verification_path=verification_path,
            sock_path=sock_path,
        )
    except BindLiveError as exc:
        print(f"bind-live refused: {exc}", file=sys.stderr)
        return 2
    current = updated["slots"]["CURRENT"]
    print(f"BOUND LIVE generation {current['generation']}: {current['instance_id']}")
    print(f"  pid              {current['identity']['resident_pid']}")
    print(f"  complete token   {current['complete_token_ns']:,} ns")
    print(f"  TPS              {current['tps']:.6f}")
    print(f"  physical BPW     {current['physical_bpw']}")
    print(f"  artifact sha     {current['artifact_sha'][:16]}")
    print(f"  runtime sha      {current['runtime_sha'][:16]}")
    print(f"  kernel sha       {current['kernel_genome_sha'][:16]}")
    return 0


def status() -> int:
    if not STATE_FILE.is_file():
        print("no lineage seated - run `genesis_seat.py seat`", file=sys.stderr)
        return 2
    d = json.loads(STATE_FILE.read_text())
    slots = d.get("slots", {})
    for slot in ("CURRENT", "CANDIDATE", "LAST_KNOWN_GOOD"):
        v = slots.get(slot)
        if not v:
            print(f"{slot:16} (empty)")
            continue
        print(f"{slot:16} gen {v.get('generation')} {v.get('instance_id')} "
              f"{v.get('representation_bpw')} BPW  {v.get('complete_token_ns'):,} ns  "
              f"{v.get('tps'):.2f} TPS  live={bool(v.get('live'))} "
              f"launched={bool(v.get('launched'))}")
    print(f"{'valid':16} {d.get('valid_count')}   zero_valid={d.get('zero_valid_genesis')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        args = ["status"]
    if args[0] == "seat" and len(args) == 1:
        return seat()
    if args[0] == "status" and len(args) == 1:
        return status()
    if args[0] == "bind-live":
        verification: Path | None = None
        socket_path: Path | None = None
        index = 1
        while index < len(args):
            flag = args[index]
            if flag not in {"--verification", "--socket"} or index + 1 >= len(args):
                print(
                    "usage: genesis_seat.py bind-live --verification PATH [--socket PATH]",
                    file=sys.stderr,
                )
                return 2
            value = Path(args[index + 1])
            if flag == "--verification":
                verification = value
            else:
                socket_path = value
            index += 2
        if verification is None:
            print("bind-live requires --verification PATH", file=sys.stderr)
            return 2
        return bind_live(verification_path=verification, sock_path=socket_path)
    print("usage: genesis_seat.py {seat|status|bind-live}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
