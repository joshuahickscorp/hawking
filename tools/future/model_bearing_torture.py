#!/usr/bin/env python3
"""30-MINUTE MODEL-BEARING TORTURE — G015 re-run against the fixed index.

An hour that is 99% scripted Python fails. This module starts the sealed-3.14
resident through hcli/agentos/resident.py, attaches the same body through the
native connector (fusion env applied, binary hash pinned), gives it the LIVE
FRONTIER with no task sequence, and records whether the model's cognition was
load-bearing.

The previous 30-minute run failed honestly: choose() advertised
WU.DEAD.mlp_function_replacement as the scripted policy because
negative_index.refuse_if_dead did not key MLP_FUNCTION_REPLACEMENT_CLOSED
(receipts/future/ was SKIP_PREFIXES-invisible). Commit 6fc77f169 closed that.
This re-run verifies the eight keyed families REFUSE, then asks what the
resident does when the advertised policy is live work.

Required events (verbatim model output, timestamps):
  a. a failure explained and a second hypothesis that meaningfully differs
  b. a subagent waits on a subprocess while another reasons, a receipt lands,
     the scheduler replans to a different queue
  c. scar-driven avoidance naming a landed scar
  d. a WorkUnit targeting Hawking itself

Also: a real without-the-model control at every decision; autonomy_degeneracy
over this run's own timeline; reason-rate as a separate cognition finding;
clean stop and incumbent restorability.

hcli/* is invoked, never edited. Hardware-named fields are refused. A FAIL
here is a real result — a second FAIL with a sharper cause is worth more
than a pass that had to be arranged.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    REPO,
    git,
    sha256_file,
    write_receipt,
)
from tools.future import autonomy_degeneracy as ad
from tools.future import fallback_resident as fb
from tools.future import model_bearing as mb
from tools.future import negative_index as ni

RECEIPT = "MODEL_BEARING_TORTURE_30M.json"
TIMELINE_RECEIPT = "MODEL_BEARING_TIMELINE.json"
SCHEMA = "hawking.future.model_bearing_torture.v1"
TIMELINE_SCHEMA = "hawking.future.model_bearing_timeline.v1"
RECORDED_BY = "tools/future/model_bearing_torture.py"
VERSION = 1

DURATION_S = 30 * 60
GPU_LOCK = Path("/tmp/hawking-gpu-lane.lock")
HCLI_LOCK_NAME = "protected-accelerator-bench.lock"

SEALED_REL = "hcli/hawking-native.sealed-3.14.json"
RESIDENT_PY_REL = "hcli/agentos/resident.py"
RESIDENT_GATE_REL = "hcli/agentos/resident_gate.py"
ORIGINAL_CHECKOUT = Path("/Users/scammermike/Downloads/hawking")

PROTOCOL = "hawking.qwen38.resident.v1"
EXPECTED_IDENTITY = "sealed-3.14"
# 192 tokens was near the previous run's mean reply length; a truncated JSON
# object is not evidence the body cannot emit a reason. 384 leaves room for
# choice_id + reason + mechanism without raising temperature or scripting JSON.
MAX_ASK_TOKENS = 384
ASK_TIMEOUT_S = 180.0
READY_TIMEOUT_S = 240.0
CHILD_WAIT_S = 14.0
STOP_WAIT_S = 20.0
PRUNE_FIX_COMMIT = "6fc77f169"

# Eight families commit 6fc77f169 made refuse_if_dead actually key.
WAVE_DEAD: tuple[str, ...] = (
    "mlp_function_replacement",
    "MONARCH",
    "BUTTERFLY",
    "FACTORIZE_THE_FACTORS",
    "PRODUCT_DICTIONARY",
    "CONDITIONAL_PROGRAM",
    "GENERATED_BLOCK",
    "NONLINEAR_GENERATOR",
)

# Landed scars the model can name. Sources are on-disk receipts, not this file.
# Ids are the tokens detect_scar_avoidance requires in verbatim output.
LANDED_SCARS: tuple[dict[str, str], ...] = (
    {
        "id": "MLP_FUNCTION_REPLACEMENT_CLOSED",
        "hypothesis_family": "mlp_function_replacement",
        "source": "receipts/future/MLP_STRUCTURED_OPERATOR.json",
        "why": "function replacement at this ledger is CLOSED; remaining lever is execution",
    },
    {
        "id": "MLP_FUNCTION_REPLACEMENT",
        "hypothesis_family": "mlp_function_replacement",
        "source": "receipts/future/MLP_STRUCTURED_OPERATOR.json",
        "why": "umbrella family keyed by 6fc77f169; refuse_if_dead must REFUSE",
    },
    {
        "id": "MONARCH",
        "hypothesis_family": "monarch",
        "source": "receipts/future/MLP_STRUCTURED_OPERATOR.json",
        "why": "Monarch is full rank at O(n^{1.5}); closed with the parent",
    },
    {
        "id": "BUTTERFLY",
        "hypothesis_family": "butterfly",
        "source": "receipts/future/MLP_STRUCTURED_OPERATOR.json",
        "why": "Butterfly is full rank at O(n log n); closed with the parent",
    },
    {
        "id": "FACTORIZE_THE_FACTORS",
        "hypothesis_family": "factorize_the_factors",
        "source": "receipts/future/MLP_NONLINEAR_PROGRAM.json",
        "why": "gate/up/down are individually high-rank; truncated SVD is not a replacement",
    },
    {
        "id": "PRODUCT_DICTIONARY",
        "hypothesis_family": "product_dictionary",
        "source": "receipts/future/MLP_NONLINEAR_PROGRAM.json",
        "why": "product codebooks over F-manifold sub-blocks stay above the L2 bar",
    },
    {
        "id": "CONDITIONAL_PROGRAM",
        "hypothesis_family": "conditional_program",
        "source": "receipts/future/MLP_NONLINEAR_PROGRAM.json",
        "why": "a condition on x selecting a rank-16 expert is not a different function",
    },
    {
        "id": "GENERATED_BLOCK",
        "hypothesis_family": "generated_block",
        "source": "receipts/future/MLP_NONLINEAR_PROGRAM.json",
        "why": "unconstrained per-block rank-16 maps are not a generated kernel",
    },
    {
        "id": "NONLINEAR_GENERATOR",
        "hypothesis_family": "nonlinear_generator",
        "source": "receipts/future/MLP_NONLINEAR_PROGRAM.json",
        "why": "an MLP-shaped generator does not drop held-out F below the bar",
    },
    {
        "id": "COMPOSITE_MLP_SIMPLE_LINEAR_LOW_RANK_REFUTED",
        "hypothesis_family": "composite_mlp_simple_linear_low_rank",
        "source": "receipts/future/MLP_SPARSE_RESIDUAL.json",
        "why": "bulk linear low-rank school refuted; reused as bulk, not reswept",
    },
    {
        "id": "RESIDUAL_RESCUE_CLOSED",
        "hypothesis_family": "residual_rescue",
        "source": "receipts/future/MLP_SPARSE_RESIDUAL.json",
        "why": "oracle residual at the 5% MLP-byte bar cannot push held-out relative L2 below 0.25",
    },
    {
        "id": "AUX_U8_DECODE_ARITHMETIC_WALL_NEGATIVE",
        "hypothesis_family": "aux_u8_byte_lever",
        "source": "receipts/future/AUX_U8_NATIVE.json",
        "why": "aux_u8 byte lever died: decode arithmetic cost more than bytes saved",
    },
)

LIVE_FRONTIER = (
    "Odyssey I launched at 16/16. The negative index now keys the closed MLP "
    "function-replacement school: MLP_FUNCTION_REPLACEMENT_CLOSED, MONARCH, "
    "BUTTERFLY, FACTORIZE_THE_FACTORS, PRODUCT_DICTIONARY, CONDITIONAL_PROGRAM, "
    "GENERATED_BLOCK, NONLINEAR_GENERATOR. Declining a unit by naming one of "
    "those scars is legal work. Execution is the remaining lever. fold_addqx "
    "and DeltaNet GEMV figures are campaign claims, not measurements of this "
    "run. The aux_u8 byte lever died because decode arithmetic cost more than "
    "bytes saved (AUX_U8_DECODE_ARITHMETIC_WALL_NEGATIVE). "
    "FIGURES IN THIS PARAGRAPH ARE FRONTIER CLAIMS FROM THE CAMPAIGN, NOT "
    "MEASUREMENTS OF THIS RUN. This sidecar has no hardware authority."
)

GOAL = (
    "Odyssey I is live. Interpret the frontier and do the work that is still "
    "open. Do not rediscover closed scars. Prefer execution over function "
    "replacement. Target Hawking itself when the machine, not a specimen, is "
    "the uncertainty. If a candidate is dead, decline it by naming the landed "
    "scar id. " + LIVE_FRONTIER
)


def verify_pruning_fix() -> dict[str, Any]:
    """Live check: the eight families 6fc77f169 keyed must all REFUSE.

    A live school must not be pruned. This is the condition that makes
    scar-driven avoidance reachable; it is not itself the required event.
    """
    rows: list[dict[str, Any]] = []
    all_refuse = True
    for family in WAVE_DEAD:
        got = ni.refuse_if_dead(
            {"hypothesis_family": family, "organ": "mlp", "model": "qwen3.8-27b"}
        )
        refused = bool(isinstance(got, Mapping) and got.get("refused"))
        all_refuse = all_refuse and refused
        rows.append(
            {
                "family": family,
                "refused": refused,
                "scar_id": (got or {}).get("original_id") or (got or {}).get("scar_id") if isinstance(got, Mapping) else None,
                "source_path": (got or {}).get("source_path") if isinstance(got, Mapping) else None,
            }
        )
    live = ni.refuse_if_dead(
        {"hypothesis_family": "gqa_kv_state_compression", "organ": "gqa", "model": "qwen3.8-27b"}
    )
    live_ok = live is None
    policy = mb.fixed_policy_choose(live_catalog())
    policy_id = policy.get("id")
    policy_skips_dead_mlp = policy_id != "WU.DEAD.mlp_function_replacement" and not str(
        policy_id or ""
    ).startswith("WU.DEAD.")
    ok = all_refuse and live_ok and policy_skips_dead_mlp
    return {
        "ok": ok,
        "all_eight_refused": all_refuse,
        "live_school_not_pruned": live_ok,
        "scripted_policy_id": policy_id,
        "scripted_policy_skips_dead_mlp": policy_skips_dead_mlp,
        "families": rows,
        "commit": PRUNE_FIX_COMMIT,
        "previous_cause": (
            "choose() advertised WU.DEAD.mlp_function_replacement as the "
            "scripted policy because negative_index.refuse_if_dead does not "
            "key MLP_FUNCTION_REPLACEMENT_CLOSED"
        ),
        "why": (
            "negative_index now parses scars from receipts/future/; "
            "MLP_STRUCTURED_OPERATOR declares umbrella MLP_FUNCTION_REPLACEMENT; "
            f"scripted policy is {policy_id}"
            if ok
            else "refuse_if_dead still misses a closed family or still advertises a dead unit"
        ),
    }

CLAIM_BOUNDARY = (
    "Static sidecar artifact plus SELF_MEASURED_DIRTY process telemetry "
    "(pid, token counts, fusion-env presence). No hardware measurement. "
    "Token counts are protocol fields, not throughput. A timeline the Python "
    "would have produced without the model is FAIL."
)

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)


class TortureRefused(ValueError):
    def __init__(self, reason: str, *, missing: list[str] | None = None) -> None:
        self.reason = reason
        self.missing = list(missing or [])
        super().__init__(reason)


# ---------------------------------------------------------------------------
# Paths. Sparse checkout: hcli may live only on the parent checkout.
# ---------------------------------------------------------------------------


def hcli_root() -> Path:
    """Directory that actually has hcli/ on disk. Never sparse-checkout add."""
    here = REPO / RESIDENT_PY_REL
    if here.is_file():
        return REPO
    original = ORIGINAL_CHECKOUT / RESIDENT_PY_REL
    if original.is_file():
        return ORIGINAL_CHECKOUT
    raise TortureRefused(
        f"{RESIDENT_PY_REL} is not on disk in this worktree or {ORIGINAL_CHECKOUT}; "
        "a missing sparse path is not evidence the module does not exist",
        missing=[RESIDENT_PY_REL],
    )


def sealed_profile_path() -> Path:
    root = hcli_root()
    path = root / SEALED_REL
    if path.is_file():
        return path
    blob = git("show", f"HEAD:{SEALED_REL}")
    if not blob:
        raise TortureRefused(f"{SEALED_REL} missing on disk and HEAD", missing=[SEALED_REL])
    tmp = Path(tempfile.mkdtemp(prefix="mbt-sealed-")) / "hawking-native.sealed-3.14.json"
    tmp.write_text(blob)
    return tmp


def resident_py_path() -> Path:
    path = hcli_root() / RESIDENT_PY_REL
    if not path.is_file():
        raise TortureRefused(
            f"{RESIDENT_PY_REL} missing; older receipts claiming this module does "
            "not exist are STALE if the original checkout has it",
            missing=[RESIDENT_PY_REL],
        )
    return path


def hcli_lock_path() -> Path:
    root = hcli_root()
    return root / ".hcli" / "locks" / HCLI_LOCK_NAME


def _ensure_hcli_on_path() -> Path:
    root = hcli_root()
    token = str(root)
    if token not in _sys.path:
        _sys.path.insert(0, token)
    return root


# ---------------------------------------------------------------------------
# GPU park. Blocking flock; record what we waited for. Not a hardware claim.
# ---------------------------------------------------------------------------



# THE GPU LANE LOCK IS A DIRECTORY, NOT AN FLOCK FILE.
#
# GpuPark used to `path.touch()` and flock /tmp/hawking-gpu-lane.lock, while
# tools/gpu_lane_lock.sh (and every other holder in this repo) uses mkdir-atomic
# directories with pid/owner files inside. Two incompatible protocols on ONE
# path, so whichever ran first broke the other:
#
#   shell lock first  -> the directory exists, and this park dies with
#                        [Errno 21] Is a directory. That is today's failure.
#   this park first   -> touch() creates the 0-byte REGULAR FILE that mkdir can
#                        never succeed against, and every shell caller spins to
#                        its 5400 s deadline and exits 75. That is the wedge seen
#                        at 05:58 and cleared at 06:21, which the reclaim branch
#                        in gpu_lane_lock.sh was written for without knowing its
#                        cause. The comment there - "nothing in this repo creates
#                        it as a file" - was wrong about this and about the wedge
#                        test both.
#
# The park's INTENT is right: interlock with the civilization lock rather than
# contend. So speak its protocol, do not move to a private path, which would
# make the interlock silently vacuous.
MKDIR_PROTOCOL_LOCKS = frozenset({str(GPU_LOCK)})



def _ancestor_pids(limit: int = 64) -> set[int]:
    """This process and every ancestor, so we can tell OUR lock from a rival's."""
    out: set[int] = set()
    pid = os.getpid()
    for _ in range(limit):
        if pid <= 1 or pid in out:
            break
        out.add(pid)
        try:
            pid = int(
                subprocess.run(
                    ["ps", "-o", "ppid=", "-p", str(pid)],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                or 0
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            break
    return out


def _mkdir_lock_acquire(path: Path, *, owner: str, timeout_s: float = 5400.0) -> dict[str, Any]:
    """Same protocol as tools/gpu_lane_lock.sh: mkdir is the atom."""
    t0 = time.time()
    holder: Any = None
    parked = False
    mine = _ancestor_pids()
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            # HELD BY US ALREADY. The trial is normally launched under
            # tools/gpu_lane_lock.sh, so the wrapper - this process's own parent -
            # holds the directory with its pid inside. Before this branch existed,
            # _mkdir_lock_acquire read that pid, found it alive (of course: it is
            # the parent), and parked for its full 5400 s deadline waiting for a
            # lock its own launcher was holding on its behalf. That is a deadlock,
            # and it cost a 45-minute run that burned 0.78 s of CPU.
            # Worth naming precisely: the old flock code FAILED OPEN here with
            # [Errno 21] Is a directory and the run continued. Making the park
            # correct turned a benign failure into a hang, which is exactly the
            # kind of regression a correctness fix can introduce.
            try:
                held_by = int((path / "pid").read_text().strip())
            except (OSError, ValueError):
                held_by = None
            if held_by is not None and held_by in mine:
                return {
                    "path": str(path), "holder_when_parked": None, "parked": False,
                    "waited_s": 0.0, "protocol": "mkdir",
                    "already_held_by_ancestor": held_by,
                    "release_is_not_ours": True,
                }
            if path.exists() and not path.is_dir():
                # Not a lock, a wedge: nothing can be holding a regular file.
                path.unlink(missing_ok=True)
                continue
            pid_f = path / "pid"
            try:
                pid = int(pid_f.read_text().strip())
                os.kill(pid, 0)
            except (OSError, ValueError):
                if pid_f.exists() or (path / "owner").exists():
                    shutil.rmtree(path, ignore_errors=True)
                    continue
            if holder is None:
                try:
                    holder = (path / "owner").read_text().strip()
                except OSError:
                    holder = None
            parked = True
            if time.time() - t0 >= timeout_s:
                raise TimeoutError(f"gpu lane lock held by {holder!r} for {timeout_s}s")
            time.sleep(1.0)
    (path / "pid").write_text(str(os.getpid()))
    (path / "owner").write_text(owner)
    return {"path": str(path), "holder_when_parked": holder, "parked": parked,
            "waited_s": round(time.time() - t0, 3), "protocol": "mkdir"}


def _mkdir_lock_release(path: Path) -> None:
    try:
        held_by = (path / "pid").read_text().strip()
    except OSError:
        return
    # Not ours, and an ANCESTOR's is not ours either: the wrapper releases its own
    # lock in its EXIT trap, and tearing it down from inside would leave the
    # wrapper's trap deleting a lock a later lane had legitimately taken.
    if held_by != str(os.getpid()):
        return
    shutil.rmtree(path, ignore_errors=True)


class GpuPark:
    """Park on the civilization GPU lock instead of contending."""

    def __init__(self, paths: Sequence[Path] | None = None) -> None:
        self.paths = [Path(p) for p in (paths if paths is not None else (GPU_LOCK, hcli_lock_path()))]
        self._handles: list[Any] = []
        self._mkdir_held: list[Path] = []
        self.record: dict[str, Any] = {
            "held": False,
            "waited_s": 0.0,
            "waited_for": [],
            "paths": [str(p) for p in self.paths],
            "pid": os.getpid(),
        }

    def acquire(self) -> dict[str, Any]:
        t0 = time.time()
        waited_for: list[dict[str, Any]] = []
        handles: list[Any] = []
        try:
            for path in self.paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                if str(path) in MKDIR_PROTOCOL_LOCKS:
                    rec = _mkdir_lock_acquire(path, owner=RECORDED_BY)
                    self._mkdir_held.append(path)
                    if rec["parked"]:
                        waited_for.append(rec)
                    continue
                path.touch(exist_ok=True)
                fh = open(path, "a+")
                holder = _read_lock_holder(fh)
                busy = False
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    busy = True
                    waited_for.append(
                        {
                            "path": str(path),
                            "holder_when_parked": holder,
                            "parked": True,
                        }
                    )
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                fh.seek(0)
                fh.truncate()
                fh.write(
                    json.dumps(
                        {
                            "holder": RECORDED_BY,
                            "pid": os.getpid(),
                            "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "waited_s_before_acquire": round(time.time() - t0, 3),
                            "parked": busy,
                        }
                    )
                    + "\n"
                )
                fh.flush()
                handles.append(fh)
        except Exception:
            for fh in handles:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    fh.close()
                except OSError:
                    pass
            raise
        self._handles = handles
        self.record = {
            "held": True,
            "waited_s": round(time.time() - t0, 3),
            "waited_for": waited_for,
            "paths": [str(p) for p in self.paths],
            "pid": os.getpid(),
            "widens_hcli_authority": False,
        }
        return dict(self.record)

    def release(self) -> None:
        for path in self._mkdir_held:
            _mkdir_lock_release(path)
        self._mkdir_held = []
        for fh in self._handles:
            try:
                fh.seek(0)
                fh.truncate()
                fh.flush()
            except OSError:
                pass
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                fh.close()
            except OSError:
                pass
        self._handles = []
        self.record["held"] = False


def _read_lock_holder(fh: Any) -> dict[str, Any] | str | None:
    try:
        fh.seek(0)
        raw = fh.read().strip()
        fh.seek(0)
    except OSError:
        return None
    if not raw:
        return None
    try:
        obj = json.loads(raw.splitlines()[0])
        return obj if isinstance(obj, dict) else raw[:240]
    except json.JSONDecodeError:
        return raw[:240]


# ---------------------------------------------------------------------------
# Identity / fusion / binary hash. Measured from disk + live process.
# ---------------------------------------------------------------------------


def load_sealed_descriptor() -> dict[str, Any]:
    path = sealed_profile_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TortureRefused("sealed descriptor is not an object")
    return data


def pin_sealed_body(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    identity = str(descriptor.get("resident_identity") or "")
    protocol = str(descriptor.get("protocol") or "")
    binary = Path(str(descriptor.get("resident_binary") or ""))
    fusion = descriptor.get("fusion_env") if isinstance(descriptor.get("fusion_env"), dict) else {}
    require = bool(descriptor.get("require_fusion_env"))
    tok = Path(str(descriptor.get("tokenizer") or ""))
    root = Path(str(descriptor.get("artifact_root") or ""))
    missing: list[str] = []
    if identity != EXPECTED_IDENTITY:
        missing.append(f"resident_identity {identity!r} != {EXPECTED_IDENTITY!r}")
    if protocol != PROTOCOL:
        missing.append(f"protocol {protocol!r} != {PROTOCOL!r}")
    if not binary.is_file():
        missing.append(f"resident_binary missing: {binary}")
    if not tok.is_file():
        missing.append(f"tokenizer missing: {tok}")
    if not root.is_dir():
        missing.append(f"artifact_root missing: {root}")
    if require and not fusion:
        missing.append("require_fusion_env true but fusion_env empty; this would not be the sealed body")
    digest = sha256_file(binary) if binary.is_file() else None
    return {
        "resident_identity": identity,
        "family": descriptor.get("family"),
        "runtime": descriptor.get("runtime"),
        "protocol": protocol,
        "require_fusion_env": require,
        "fusion_env": {str(k): str(v) for k, v in fusion.items()},
        "artifact_root": str(root),
        "tokenizer": str(tok),
        "resident_binary": str(binary),
        "binary_sha256": digest,
        "binary_present": binary.is_file(),
        "tokenizer_present": tok.is_file(),
        "artifact_root_present": root.is_dir(),
        "mismatches": missing,
        "sealed": not missing and identity == EXPECTED_IDENTITY,
        "stale_note_contradicted": True,
        "stale_note": (
            "an older receipt claimed hcli/agentos/resident.py does not exist "
            "and resident_gate.py is the live gate; that note is STALE — "
            f"{RESIDENT_PY_REL} is on disk at {resident_py_path()} with a "
            "start/status/stop CLI"
        ),
        "resident_py": str(resident_py_path()),
        "resident_py_exists": True,
        "resident_gate_also_exists": (hcli_root() / RESIDENT_GATE_REL).is_file()
        or bool(git("ls-tree", "-r", "--name-only", "HEAD", "--", RESIDENT_GATE_REL)),
        "gpu_authority": False,
    }


def inspect_process_fusion(pid: int | None, expected: Mapping[str, str]) -> dict[str, Any]:
    """Prove fusion env is on the live process. Absence is recorded, not guessed."""
    if not pid:
        return {
            "ok": False,
            "why": "no pid; fusion env cannot be inspected on a process that is not running",
            "applied": False,
            "expected": dict(expected),
        }
    text = ""
    how = None
    for argv in (
        ["ps", "eww", "-p", str(pid)],
        ["ps", "-E", "-p", str(pid), "-o", "command="],
    ):
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=8)
        except (OSError, subprocess.TimeoutExpired):
            continue
        blob = (proc.stdout or "") + (proc.stderr or "")
        if blob.strip():
            text = blob
            how = " ".join(argv)
            break
    found = {key: f"{key}={value}" in text for key, value in expected.items()}
    missing = [key for key, hit in found.items() if not hit]
    exe = running_executable(pid)
    return {
        "ok": not missing and bool(text.strip()),
        "applied": not missing and bool(text.strip()),
        "pid": pid,
        "how": how,
        "found": found,
        "missing": missing,
        "expected": dict(expected),
        "ps_text_chars": len(text),
        "running_executable": exe,
        "why": None if not missing else (
            "ps did not show fusion keys on the live process"
            if text.strip()
            else "ps produced no environment text for this pid"
        ),
        "gpu_authority": False,
    }


def running_executable(pid: int) -> str | None:
    try:
        proc = subprocess.run(
            ["lsof", "-p", str(pid), "-Fn"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("n") and "ascension_qwen38_resident" in line:
            return line[1:]
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True,
            text=True,
            timeout=8,
        )
        comm = (proc.stdout or "").strip()
        return comm or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def binary_hash_matches(pin: Mapping[str, Any], pid: int | None) -> dict[str, Any]:
    path = Path(str(pin.get("resident_binary") or ""))
    disk = pin.get("binary_sha256")
    exe = running_executable(pid) if pid else None
    live = None
    if exe and Path(exe).is_file():
        live = sha256_file(exe)
    match = bool(disk) and (live is None or live == disk)
    return {
        "disk_sha256": disk,
        "live_sha256": live,
        "running_executable": exe,
        "match": match and bool(disk),
        "why": None if (match and disk) else "binary hash missing or live executable digest differs",
        "gpu_authority": False,
    }


# ---------------------------------------------------------------------------
# Strip hardware-named fields. Token counts stay; rates do not.
# ---------------------------------------------------------------------------

RATE_FIELDS = frozenset({"decode_tps", "complete_tps", "tokens_per_second"})
FORBIDDEN = HARDWARE_FIELDS | RATE_FIELDS


def _drop_hw_key(key: str) -> bool:
    if key in FORBIDDEN:
        return True
    lk = key.lower()
    if "tps" in lk:
        return True
    if lk.endswith("_ns") and any(tok in lk for tok in ("gpu", "wall", "token", "dispatch")):
        return True
    return False


def strip_hardware(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: strip_hardware(v) for k, v in node.items() if not _drop_hw_key(k)}
    if isinstance(node, list):
        return [strip_hardware(v) for v in node]
    return node


def openai_text(raw: Mapping[str, Any] | None) -> str:
    if not isinstance(raw, Mapping):
        return ""
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(choice.get("text"), str):
            return choice["text"]
    for key in ("text", "generated_text"):
        if isinstance(raw.get(key), str):
            return raw[key]
    hawking = raw.get("hawking") if isinstance(raw.get("hawking"), dict) else {}
    for key in ("text", "generated_text"):
        if isinstance(hawking.get(key), str):
            return hawking[key]
    return ""


def openai_tokens(raw: Mapping[str, Any] | None) -> dict[str, int | None]:
    if not isinstance(raw, Mapping):
        return {"prompt_tokens": None, "generated_tokens": None}
    hawking = raw.get("hawking") if isinstance(raw.get("hawking"), dict) else {}
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    prompt = hawking.get("prompt_tokens")
    if prompt is None:
        prompt = usage.get("prompt_tokens")
    gen = hawking.get("generated_tokens")
    if gen is None:
        gen = usage.get("completion_tokens")
    try:
        prompt_i = int(prompt) if prompt is not None else None
    except (TypeError, ValueError):
        prompt_i = None
    try:
        gen_i = int(gen) if gen is not None else None
    except (TypeError, ValueError):
        gen_i = None
    return {"prompt_tokens": prompt_i, "generated_tokens": gen_i}


# ---------------------------------------------------------------------------
# Provider adapter. One sealed body. Logical sessions share weights.
# ---------------------------------------------------------------------------


class ConnectorProvider:
    """model_bearing provider surface over HawkingNativeConnector."""

    def __init__(self, connector: Any) -> None:
        self.connector = connector
        self._sessions: list[str] = ["main"]
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def start(self, session: str | None = None, **kwargs: Any) -> dict[str, Any]:
        sid = session or kwargs.get("session") or "main"
        if sid not in self._sessions:
            self._sessions.append(sid)
        if getattr(self.connector, "pid", None) is None:
            self.connector.start(timeout=READY_TIMEOUT_S)
        return {"ok": True, "status": "ready", "session": sid}

    def ask(self, prompt: str, session: str | None = None) -> dict[str, Any]:
        sid = session or "main"
        if sid not in self._sessions:
            self._sessions.append(sid)
        payload = {
            "messages": [{"role": "user", "content": str(prompt)}],
            "max_tokens": MAX_ASK_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.time()
        raw = self.connector.complete_payload(payload, timeout=ASK_TIMEOUT_S)
        text = openai_text(raw)
        tokens = openai_tokens(raw)
        hawking = raw.get("hawking") if isinstance(raw.get("hawking"), dict) else {}
        rec = {
            "ok": True,
            "text": text,
            "session": sid,
            "prompt_tokens": tokens["prompt_tokens"],
            "generated_tokens": tokens["generated_tokens"],
            "resident_identity": hawking.get("resident_identity"),
            "elapsed_s": round(time.time() - t0, 3),
            "elapsed_evidence_class": "SELF_MEASURED_DIRTY",
        }
        with self._lock:
            self.calls.append(
                {
                    "t_unix": t0,
                    "session": sid,
                    "prompt": str(prompt),
                    "reply_text": text,
                    "prompt_tokens": tokens["prompt_tokens"],
                    "generated_tokens": tokens["generated_tokens"],
                    "reply_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "prompt_sha256": hashlib.sha256(str(prompt).encode()).hexdigest(),
                    "resident_identity": hawking.get("resident_identity"),
                }
            )
        return rec

    def sessions(self) -> list[str]:
        return list(self._sessions)

    def health(self) -> dict[str, Any]:
        pid = getattr(self.connector, "pid", None)
        ident = {}
        try:
            ident = self.connector.identity() or {}
        except Exception as exc:
            return {"ok": False, "status": "error", "why": f"{type(exc).__name__}: {exc}"}
        health = ident.get("resident_health") if isinstance(ident.get("resident_health"), dict) else {}
        status = str(health.get("status") or ident.get("status") or "")
        ok = bool(pid) and status.lower() in {"", "ready", "healthy", "ok", "available", "up"}
        if pid and not status:
            ok = True
        return {
            "ok": ok,
            "status": "ready" if ok else (status or "not_ready"),
            "pid": pid,
            "resident_identity": ident.get("resident_identity") or health.get("resident_identity"),
            "protocol": ident.get("protocol") or health.get("protocol"),
        }

    def stop(self) -> dict[str, Any]:
        try:
            return dict(self.connector.stop() or {})
        except Exception as exc:
            return {"ok": False, "why": f"{type(exc).__name__}: {exc}"}

    def restart(self) -> dict[str, Any]:
        self.connector.start(timeout=READY_TIMEOUT_S)
        return {"ok": True, "status": "ready"}


# ---------------------------------------------------------------------------
# Live catalog. Not a task sequence — a set the model picks from.
# ---------------------------------------------------------------------------


def live_catalog() -> list[dict[str, Any]]:
    """Candidates grounded in the live frontier. High-gain rows are closed scars.

    Titles carry scar ids because model_bearing.choose/_compact_entries
    forwards title (72 chars), not description. Naming a scar is the
    required event; hiding the id from the prompt would make it unreachable.
    Interpret sees the first eight rows; keep live Hawking-self work in that
    window so the menu is not only dead units.
    """
    rows: list[dict[str, Any]] = [
        {
            "id": "WU.DEAD.mlp_function_replacement",
            "expected_information_gain": 9,
            "title": "DEAD scar MLP_FUNCTION_REPLACEMENT_CLOSED: replace F",
            "description": "CLOSED scar MLP_FUNCTION_REPLACEMENT_CLOSED; remaining lever is execution",
            "frontier": "MODEL_REPRESENTATION",
            "hypothesis_family": "mlp_function_replacement",
            "surface": "mlp",
            "organ": "mlp",
            "scar_id": "MLP_FUNCTION_REPLACEMENT_CLOSED",
            "dead": True,
            "hawking_self": False,
        },
        {
            "id": "WU.DEAD.monarch",
            "expected_information_gain": 8,
            "title": "DEAD scar MONARCH: block-diag product, full rank",
            "description": "CLOSED scar MONARCH; Monarch is full rank at O(n^{1.5})",
            "frontier": "MODEL_REPRESENTATION",
            "hypothesis_family": "monarch",
            "surface": "mlp",
            "organ": "mlp",
            "scar_id": "MONARCH",
            "dead": True,
        },
        {
            "id": "WU.DEAD.butterfly",
            "expected_information_gain": 8,
            "title": "DEAD scar BUTTERFLY: O(n log n) full rank",
            "description": "CLOSED scar BUTTERFLY; not a cheaper replacement of F",
            "frontier": "MODEL_REPRESENTATION",
            "hypothesis_family": "butterfly",
            "surface": "mlp",
            "organ": "mlp",
            "scar_id": "BUTTERFLY",
            "dead": True,
        },
        {
            "id": "WU.HAWKING.resident_identity_pin",
            "expected_information_gain": 5,
            "title": "pin the sealed resident identity on the live body",
            "description": "Hawking itself: identity, protocol, binary hash, fusion env",
            "frontier": "HCLI_SELF",
            "hypothesis_family": "resident_identity_pin",
            "surface": "hawking.resident",
            "organ": "hawking",
            "hawking_self": True,
            "dead": False,
        },
        {
            "id": "WU.HAWKING.fusion_env_applied",
            "expected_information_gain": 4,
            "title": "prove fusion env is applied on the live resident process",
            "description": "Hawking itself: require_fusion_env must be the sealed block",
            "frontier": "HCLI_SELF",
            "hypothesis_family": "fusion_env_applied",
            "surface": "hawking.resident",
            "organ": "hawking",
            "hawking_self": True,
            "dead": False,
        },
        {
            "id": "WU.EXEC.fold_addqx_ab_status",
            "expected_information_gain": 4,
            "title": "inspect sibling fold_addqx complete-token A/B status",
            "description": "do not re-measure bandwidth; read whether the sibling receipt landed",
            "frontier": "MODEL_EXECUTION",
            "hypothesis_family": "fold_addqx_status",
            "surface": "fold_addqx",
            "organ": "deltanet",
            "dead": False,
        },
        {
            "id": "WU.HAWKING.no_wait_scheduler",
            "expected_information_gain": 3,
            "title": "keep reasoning while a child waits on a receipt",
            "description": "Hawking itself: no-wait scheduler, not a specimen organ",
            "frontier": "HCLI_SELF",
            "hypothesis_family": "no_wait_scheduler",
            "surface": "hawking.scheduler",
            "organ": "hawking",
            "hawking_self": True,
            "wait": True,
            "dead": False,
        },
        {
            "id": "WU.PROBE.decode_arith_cost",
            "expected_information_gain": 2,
            "title": "CPU demo: decode arithmetic can dominate byte savings",
            "description": "process telemetry only; not a GPU claim; expected to fail the cheap-byte hypothesis",
            "frontier": "DECODING",
            "hypothesis_family": "decode_arithmetic_dominates_byte_save",
            "surface": "decode",
            "organ": "decode",
            "will_fail": True,
            "dead": False,
        },
        {
            "id": "WU.DEAD.factorize_the_factors",
            "expected_information_gain": 6,
            "title": "DEAD scar FACTORIZE_THE_FACTORS: SVD on gate/up/down",
            "description": "CLOSED scar FACTORIZE_THE_FACTORS; factors are individually high-rank",
            "frontier": "MODEL_REPRESENTATION",
            "hypothesis_family": "factorize_the_factors",
            "surface": "mlp",
            "organ": "mlp",
            "scar_id": "FACTORIZE_THE_FACTORS",
            "dead": True,
        },
        {
            "id": "WU.DEAD.product_dictionary",
            "expected_information_gain": 6,
            "title": "DEAD scar PRODUCT_DICTIONARY: F-manifold codebooks",
            "description": "CLOSED scar PRODUCT_DICTIONARY",
            "frontier": "MODEL_REPRESENTATION",
            "hypothesis_family": "product_dictionary",
            "surface": "mlp",
            "organ": "mlp",
            "scar_id": "PRODUCT_DICTIONARY",
            "dead": True,
        },
        {
            "id": "WU.DEAD.conditional_program",
            "expected_information_gain": 6,
            "title": "DEAD scar CONDITIONAL_PROGRAM: x-conditioned expert",
            "description": "CLOSED scar CONDITIONAL_PROGRAM",
            "frontier": "MODEL_REPRESENTATION",
            "hypothesis_family": "conditional_program",
            "surface": "mlp",
            "organ": "mlp",
            "scar_id": "CONDITIONAL_PROGRAM",
            "dead": True,
        },
        {
            "id": "WU.DEAD.generated_block",
            "expected_information_gain": 6,
            "title": "DEAD scar GENERATED_BLOCK: per-block rank-16 maps",
            "description": "CLOSED scar GENERATED_BLOCK",
            "frontier": "MODEL_REPRESENTATION",
            "hypothesis_family": "generated_block",
            "surface": "mlp",
            "organ": "mlp",
            "scar_id": "GENERATED_BLOCK",
            "dead": True,
        },
        {
            "id": "WU.DEAD.nonlinear_generator",
            "expected_information_gain": 6,
            "title": "DEAD scar NONLINEAR_GENERATOR: MLP-shaped G",
            "description": "CLOSED scar NONLINEAR_GENERATOR",
            "frontier": "MODEL_REPRESENTATION",
            "hypothesis_family": "nonlinear_generator",
            "surface": "mlp",
            "organ": "mlp",
            "scar_id": "NONLINEAR_GENERATOR",
            "dead": True,
        },
        {
            "id": "WU.SUBAGENT.receipt_wait_probe",
            "expected_information_gain": 1,
            "title": "bounded child writes a receipt after a short wait",
            "description": "subagent waits on a subprocess; scheduler must not sit inside it",
            "frontier": "TOOLS",
            "hypothesis_family": "subagent_receipt_wait",
            "surface": "scheduler",
            "organ": "hawking",
            "wait": True,
            "hawking_self": True,
            "dead": False,
        },
    ]
    rows.extend(_staleness_frontier_rows())
    return rows


def _staleness_frontier_rows() -> list[dict[str, Any]]:
    """Real open work, read from disk, so the 30-minute clock has choices in it.

    The last run exhausted the live menu in SEVEN cycles and then offered one
    auto-generated health probe for the remaining forty-four. Divergence is
    undefined on a menu of one, so 86% of the trial scored a choice nobody was
    given.

    These rows are not invented to pad the menu. BASELINE_STALENESS lists
    receipts whose producer does not read the current baseline, and each is a
    genuine open question - three checked by hand were all genuinely stale and
    each moved a strategic number. If that receipt is absent, this returns
    NOTHING rather than fabricating work: a padded catalog would be the same
    defect as a padded timeline.
    """
    p = REPO / "receipts/future/BASELINE_STALENESS.json"
    if not p.is_file():
        return []
    try:
        doc = json.loads(p.read_text())
        names = list(doc["report"]["needing_review"])
    except (ValueError, OSError, KeyError, TypeError):
        return []
    out: list[dict[str, Any]] = []
    for name in names:
        stem = name[:-5] if name.endswith(".json") else name
        short = stem.lower()[:34]
        out.append({
            "id": f"WU.STALE.{short}",
            # Below every live row already in the catalog, so the first eight
            # rows interpret() sees are unchanged and the required events still
            # fire from the same units as before.
            "expected_information_gain": 1,
            "title": f"does {stem[:38]} still price against a dead baseline",
            "description": (
                f"BASELINE_STALENESS flagged {name}: its producer does not read "
                "receipts/future/SEALED_DEFAULT_ABSOLUTE.json. Decide whether it "
                "is a live consumer or a historical record."
            ),
            "frontier": "RECEIPT_INTEGRITY",
            "hypothesis_family": "baseline_staleness",
            "surface": "receipts",
            "organ": "hawking",
            "hawking_self": True,
            "dead": False,
        })
    return out


def scar_lookup() -> dict[str, dict[str, str]]:
    return {row["id"]: dict(row) for row in LANDED_SCARS}


def is_dead_unit(unit: Mapping[str, Any]) -> dict[str, Any] | None:
    if unit.get("dead") and unit.get("scar_id"):
        table = scar_lookup()
        scar = table.get(str(unit["scar_id"]))
        if scar:
            return {
                "refused": True,
                "scar_id": scar["id"],
                "hypothesis_family": scar["hypothesis_family"],
                "source": scar["source"],
                "reason": scar["why"],
            }
        return {
            "refused": True,
            "scar_id": str(unit["scar_id"]),
            "reason": f"unit marked dead under {unit.get('scar_id')}",
        }
    family = str(unit.get("hypothesis_family") or "")
    if family:
        hit = ni.refuse_if_dead(
            {
                "model": "qwen3.8-27b",
                "organ": unit.get("organ") or unit.get("surface"),
                "hypothesis_family": family,
            }
        )
        if hit:
            return hit
    return None


# ---------------------------------------------------------------------------
# Timeline tape.
# ---------------------------------------------------------------------------


class Tape:
    def __init__(self, t0: float | None = None) -> None:
        self.t0 = float(t0 if t0 is not None else time.time())
        self.events: list[dict[str, Any]] = []
        self.model_calls: list[dict[str, Any]] = []

    def t_s(self, now: float | None = None) -> float:
        return round((now if now is not None else time.time()) - self.t0, 3)

    def emit(self, kind: str, payload: Mapping[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
        row = {
            "kind": kind,
            "t_s": self.t_s(),
            "payload": strip_hardware(dict(payload or {})),
        }
        for key, value in extra.items():
            if key not in FORBIDDEN:
                row[key] = value
        self.events.append(row)
        return row

    def record_call(self, call: Mapping[str, Any], *, t_unix: float | None = None) -> dict[str, Any]:
        t_unix = float(t_unix if t_unix is not None else call.get("t_unix") or time.time())
        row = {
            "t_s": self.t_s(t_unix),
            "t_unix": t_unix,
            "session": call.get("session"),
            "prompt": call.get("prompt"),
            "reply_text": call.get("reply_text"),
            "prompt_tokens": call.get("prompt_tokens"),
            "generated_tokens": call.get("generated_tokens"),
            "reply_sha256": call.get("reply_sha256"),
            "prompt_sha256": call.get("prompt_sha256"),
            "resident_identity": call.get("resident_identity"),
        }
        self.model_calls.append(row)
        return row


# ---------------------------------------------------------------------------
# Resident.py CLI invoke. Never edit hcli/*.
# ---------------------------------------------------------------------------


def invoke_resident(command: Sequence[str], *, workspace: Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    py = resident_py_path()
    root = hcli_root()
    argv = [_sys.executable, str(py), *command]
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(root)] + ([child_env["PYTHONPATH"]] if child_env.get("PYTHONPATH") else [])
    )
    if env:
        child_env.update({str(k): str(v) for k, v in env.items()})
    try:
        proc = subprocess.run(
            argv,
            cwd=str(workspace),
            env=child_env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "argv": argv,
            "error": f"{type(exc).__name__}: {exc}",
        }
    text = (proc.stdout or "").strip()
    parsed: Any = None
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
    return {
        "ok": proc.returncode == 0,
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": text[:4000],
        "stderr": (proc.stderr or "")[-2000:],
        "parsed": parsed,
    }


# ---------------------------------------------------------------------------
# Child work. CPU only. Unique WorkUnit labels. Never argv[0] as the label.
# ---------------------------------------------------------------------------


def _write_child_receipt(path: Path, body: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(strip_hardware(dict(body)), indent=1, sort_keys=True) + "\n")


def launch_unit(
    unit: Mapping[str, Any],
    *,
    run_dir: Path,
    pin: Mapping[str, Any],
    pid: int | None,
    wait_s: float = CHILD_WAIT_S,
) -> dict[str, Any]:
    uid = str(unit["id"])
    receipt = run_dir / f"{uid}.receipt.json"
    script = run_dir / f"{uid}.child.py"
    dead = is_dead_unit(unit)
    if dead:
        raise TortureRefused(f"refusing to launch scar-dead unit {uid}: {dead}")
    kind = "hawking_self" if unit.get("hawking_self") else (
        "failing_probe" if unit.get("will_fail") else (
            "wait_probe" if unit.get("wait") else "inspect"
        )
    )
    payload = {
        "unit_id": uid,
        "kind": kind,
        "pin_identity": pin.get("resident_identity"),
        "pin_binary": pin.get("resident_binary"),
        "pin_sha256": pin.get("binary_sha256"),
        "fusion_env": pin.get("fusion_env"),
        "pid": pid,
        "wait_s": float(wait_s),
        "receipt": str(receipt),
        "will_fail": bool(unit.get("will_fail")),
    }
    script.write_text(_CHILD_SOURCE.replace("@@PAYLOAD@@", repr(json.dumps(payload))))
    proc = subprocess.Popen(
        [_sys.executable, str(script)],
        cwd=str(run_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    return {
        "unit_id": uid,
        "label": uid,
        "capability": uid,
        "argv0": Path(_sys.executable).name,
        "pid": proc.pid,
        "receipt": str(receipt),
        "script": str(script),
        "proc": proc,
        "hawking_self": bool(unit.get("hawking_self")),
        "wait": bool(unit.get("wait") or unit.get("hawking_self") or unit.get("will_fail")),
        "family": unit.get("hypothesis_family"),
    }


_CHILD_SOURCE = r'''
import json, os, time, hashlib
from pathlib import Path
P = json.loads(@@PAYLOAD@@)
t0 = time.time()
out = {
    "unit_id": P["unit_id"],
    "kind": P["kind"],
    "pid": os.getpid(),
    "parent_pid": P.get("pid"),
    "started_unix": t0,
}
kind = P["kind"]
if kind == "hawking_self":
    binary = Path(str(P.get("pin_binary") or ""))
    digest = None
    if binary.is_file():
        h = hashlib.sha256()
        with binary.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
    out["identity"] = P.get("pin_identity")
    out["binary_sha256"] = digest
    out["matches_pin"] = digest == P.get("pin_sha256")
    out["fusion_env_declared"] = P.get("fusion_env")
    out["status"] = "ok"
    code = 0
elif kind == "failing_probe":
    # CPU demonstration of "arithmetic cost dominates a byte save". Not GPU.
    payload = bytes(range(256)) * 4096
    t_store = time.perf_counter()
    _saved = payload  # "saved" by keeping the bytes
    store_s = time.perf_counter() - t_store
    t_arith = time.perf_counter()
    acc = 0
    for _ in range(40):
        acc ^= sum(payload)
        acc = (acc * 16777619) & 0xFFFFFFFF
    arith_s = time.perf_counter() - t_arith
    out["store_s"] = store_s
    out["arith_s"] = arith_s
    out["acc"] = acc
    out["status"] = "failed"
    out["error"] = (
        "decode arithmetic dominated the byte-save on this CPU probe "
        f"(arith_s={arith_s:.6f} store_s={store_s:.6f}); same lesson as aux_u8"
    )
    code = 1
elif kind == "inspect":
    out["status"] = "ok"
    out["note"] = "inspected campaign claim only; no hardware number recorded"
    code = 0
else:
    out["status"] = "ok"
    code = 0
remain = float(P.get("wait_s") or 0) - (time.time() - t0)
if remain > 0:
    time.sleep(remain)
out["finished_unix"] = time.time()
out["elapsed_s"] = out["finished_unix"] - t0
out["elapsed_evidence_class"] = "SELF_MEASURED_DIRTY"
Path(P["receipt"]).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
raise SystemExit(code)
'''


def poll_handle(handle: Mapping[str, Any]) -> dict[str, Any]:
    proc = handle.get("proc")
    code = proc.poll() if proc is not None else 0
    receipt = Path(str(handle.get("receipt") or ""))
    landed = receipt.is_file()
    return {
        "done": code is not None and landed,
        "running": code is None,
        "exit_code": code,
        "landed": landed,
        "receipt": str(receipt) if landed else None,
        "unit_id": handle.get("unit_id"),
        "label": handle.get("label"),
    }


def load_receipt(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "path": str(path)}


# ---------------------------------------------------------------------------
# Four-event detectors. Verbatim output, not paraphrase.
# ---------------------------------------------------------------------------


def _reply_blob(call: Mapping[str, Any]) -> str:
    return str(call.get("reply_text") or "")


def detect_second_hypothesis(events: Sequence[Mapping[str, Any]], calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Failure explained + hypothesis B that is not a restatement."""
    explains = [e for e in events if e.get("kind") in {"failure_explained", "explain_failure"}]
    hyps = [e for e in events if e.get("kind") in {"second_hypothesis", "next_hypothesis"}]
    best: dict[str, Any] | None = None
    for hyp in hyps:
        payload = hyp.get("payload") if isinstance(hyp.get("payload"), dict) else {}
        prior = payload.get("hypothesis_a") or payload.get("prior")
        nxt = payload.get("hypothesis_b") or payload.get("proposed")
        diff = payload.get("difference") or {}
        if not isinstance(diff, dict) and prior and nxt:
            diff = mb.meaningfully_different(prior, nxt)
        if isinstance(diff, dict) and diff.get("different"):
            best = {
                "found": True,
                "hypothesis_a": prior,
                "hypothesis_b": nxt,
                "hypothesis_a_verbatim": payload.get("hypothesis_a_verbatim") or _field_text(prior),
                "hypothesis_b_verbatim": payload.get("hypothesis_b_verbatim") or _field_text(nxt),
                "explain_verbatim": payload.get("explain_verbatim"),
                "difference": diff,
                "why_not_restatement": diff.get("why"),
                "t_s_explain": next((e.get("t_s") for e in explains), None),
                "t_s_hypothesis_b": hyp.get("t_s"),
            }
            break
    if best is None:
        return {
            "found": False,
            "why": (
                "no trajectory where a failure was explained and hypothesis B "
                "meaningfully differed (jaccard/surface/family)"
            ),
            "n_explain": len(explains),
            "n_next_hypothesis": len(hyps),
        }
    a = str(best.get("hypothesis_a_verbatim") or "")
    b = str(best.get("hypothesis_b_verbatim") or "")
    restatement_like = ("bigger n" in b.lower() and "bigger n" in a.lower()) or b.strip() == a.strip()
    if restatement_like:
        best["found"] = False
        best["why"] = "hypothesis B is a restatement ('try again with a bigger N' class)"
    return best


def _field_text(obj: Any) -> str:
    if isinstance(obj, Mapping):
        return str(obj.get("text") or obj.get("why") or obj.get("mechanism") or json.dumps(obj, sort_keys=True))
    return str(obj or "")


def detect_wait_reason_receipt_replan(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    waits = [e for e in events if e.get("kind") in {"subprocess_wait_start", "subagent_wait_start"}]
    reasons = [e for e in events if e.get("kind") in {"model_reasoned_during_wait", "other_reasoned"}]
    lands = [
        e for e in events
        if e.get("kind") in {"RESULT_INGESTED", "receipt_ingested", "result_ingested", "RECEIPT_INGESTED"}
    ]
    replans = [e for e in events if e.get("kind") in {"scheduler_replan"}]
    for wait in waits:
        w0 = float(wait.get("t_s") or 0)
        w_payload = wait.get("payload") if isinstance(wait.get("payload"), dict) else {}
        w1 = float(w_payload.get("t_s_end") or wait.get("t_s_end") or w0)
        during = [r for r in reasons if w0 <= float(r.get("t_s") or 0) <= max(w1, w0 + 60)]
        after_land = [x for x in lands if float(x.get("t_s") or 0) >= w0]
        after_replan = [x for x in replans if after_land and float(x.get("t_s") or 0) >= float(after_land[0].get("t_s") or 0)]
        if not during or not after_land or not after_replan:
            continue
        replan = after_replan[0]
        rp = replan.get("payload") if isinstance(replan.get("payload"), dict) else {}
        before = rp.get("queued_before") or w_payload.get("queued_before") or []
        after = rp.get("queued_after") or []
        if list(before) == list(after) and before:
            continue
        return {
            "found": True,
            "t_s_wait_start": w0,
            "t_s_reason": during[0].get("t_s"),
            "t_s_receipt": after_land[0].get("t_s"),
            "t_s_replan": replan.get("t_s"),
            "queued_before": before,
            "queued_after": after,
            "queue_differed": list(before) != list(after),
            "reason_verbatim": (during[0].get("payload") or {}).get("reply_text") if isinstance(during[0].get("payload"), dict) else None,
            "wait_unit": w_payload.get("unit_id"),
        }
    return {
        "found": False,
        "why": "no overlapping wait/reason/receipt/replan with a queue that changed",
        "n_waits": len(waits),
        "n_reasons_during": len(reasons),
        "n_receipts": len(lands),
        "n_replans": len(replans),
    }


def detect_scar_avoidance(
    events: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = [row["id"] for row in LANDED_SCARS]
    for event in events:
        if event.get("kind") not in {"scar_avoidance", "idea_rejected", "negative_science_refusal"}:
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        named = str(payload.get("scar_id") or payload.get("named_scar") or "")
        verbatim = str(payload.get("verbatim") or payload.get("reply_text") or "")
        if named in names and named and named in verbatim:
            return {
                "found": True,
                "scar_id": named,
                "source": payload.get("source"),
                "t_s": event.get("t_s"),
                "verbatim": verbatim,
                "declined": payload.get("declined_id"),
            }
    for call in calls:
        text = _reply_blob(call)
        for scar in LANDED_SCARS:
            if scar["id"] in text:
                declined = None
                low = text.lower()
                if any(w in low for w in ("avoid", "decline", "closed", "refut", "dead", "not ", "skip", "refuse")):
                    declined = scar["id"]
                if declined:
                    return {
                        "found": True,
                        "scar_id": scar["id"],
                        "source": scar["source"],
                        "t_s": call.get("t_s"),
                        "verbatim": text,
                        "from": "model_call",
                    }
    return {"found": False, "why": "model output never declined a unit by naming a landed scar"}


def detect_hawking_self(events: Sequence[Mapping[str, Any]], calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A mention is not a WorkUnit. The event is a launched unit targeting the machine."""
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        uid = str(payload.get("unit_id") or payload.get("id") or event.get("unit_id") or "")
        hawking = bool(payload.get("hawking_self")) or uid.startswith("WU.HAWKING.")
        if event.get("kind") in {"workunit_launched", "WORK_LAUNCHED", "hawking_self_unit"} and hawking:
            verbatim = payload.get("verbatim") or payload.get("reply_text")
            return {
                "found": True,
                "unit_id": uid,
                "t_s": event.get("t_s"),
                "verbatim": verbatim,
                "why": "WorkUnit targets Hawking (the machine), not a specimen organ",
            }
    mentioned = any("WU.HAWKING." in _reply_blob(c) for c in calls)
    return {
        "found": False,
        "why": (
            "model named a Hawking-self id but never launched it"
            if mentioned
            else "no WorkUnit targeting Hawking itself was launched"
        ),
        "mentioned_in_model_output": mentioned,
    }


# ---------------------------------------------------------------------------
# Participation / control / identical-output degeneracy.
# ---------------------------------------------------------------------------


def participation_report(
    *,
    chooses: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    mb_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    n_choose = len(chooses)
    # A cycle with fewer than two options cannot show divergence. Counting it as
    # agreement inflates the denominator with cycles that had no choice in them.
    choosable = [c for c in chooses if (c.get("n_options") or 0) >= 2]
    unchoosable = [c for c in chooses if (c.get("n_options") or 0) < 2]
    diverged = [c for c in chooses if c.get("diverged")]
    agreed = [c for c in chooses if not c.get("diverged") and c.get("model_id")]
    n_calls = len(calls)
    prompt_tokens = sum(int(c["prompt_tokens"]) for c in calls if isinstance(c.get("prompt_tokens"), int))
    gen_tokens = sum(int(c["generated_tokens"]) for c in calls if isinstance(c.get("generated_tokens"), int))
    frac = (len(diverged) / n_choose) if n_choose else 0.0
    div_choosable = [c for c in choosable if c.get("diverged")]
    frac_choosable = (len(div_choosable) / len(choosable)) if choosable else None
    model_ids = [c.get("model_id") for c in chooses if c.get("model_id")]
    policy_ids = [c.get("policy_id") for c in chooses if c.get("policy_id")]
    same_seq = model_ids == policy_ids and bool(model_ids)
    return {
        "n_choose": n_choose,
        "n_model_calls": n_calls,
        "n_diverged_from_policy": len(diverged),
        "n_agreed_with_policy": len(agreed),
        "fraction_model_over_policy": round(frac, 4),
        "n_cycles_with_a_real_choice": len(choosable),
        "n_cycles_with_one_or_no_option": len(unchoosable),
        "n_diverged_where_a_choice_existed": len(div_choosable),
        "fraction_model_over_policy_where_a_choice_existed": (
            round(frac_choosable, 4) if frac_choosable is not None else None),
        "why_two_fractions": (
            "divergence is UNDEFINED on a menu of one. The clock outran the real "
            "catalog, so most cycles offered a single auto-generated probe and "
            "counting them as agreement scores a choice nobody was given. The "
            "second fraction is over cycles where two or more options existed; "
            "null means there were none, which is a statement about the "
            "FRONTIER, not about the resident."
        ),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "token_count_authority": "native-resident protocol fields; not a throughput claim",
        "materially_participated": dict(mb_report) if isinstance(mb_report, Mapping) else None,
        "without_the_model_same_choose_sequence": same_seq,
        "gpu_authority": False,
    }


def control_replay(decision_points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What the scheduler would have done by rule. Actually computed, not asserted."""
    policy_seq = [p.get("policy_id") for p in decision_points]
    model_seq = [p.get("model_id") for p in decision_points]
    launched_model = [p.get("launched") for p in decision_points if p.get("launched")]
    launched_policy = [p.get("policy_id") for p in decision_points if p.get("policy_id") and not p.get("policy_dead")]
    same_choose = policy_seq == model_seq and bool(policy_seq)
    same_exec = launched_model == launched_policy
    return {
        "n_points": len(decision_points),
        "policy_sequence": policy_seq,
        "model_sequence": model_seq,
        "launched_under_model": launched_model,
        "would_have_launched_under_policy": launched_policy,
        "sequences_identical": same_choose,
        "would_timeline_look_the_same_without_the_model": same_choose and same_exec,
        "control_ran": True,
        "control_kind": "per-decision fixed_policy_choose counterfactual plus replay of policy ids",
    }


def identical_output_degeneracy(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    texts = [str(c.get("reply_sha256") or "") for c in calls]
    stats = ad.distinct_vs_repeated(texts)
    degenerate = (
        len(texts) >= ad.DECISION_RUN_MIN_N
        and (
            stats["unique_ratio"] < ad.DECISION_MIN_UNIQUE_RATIO
            or stats["largest_repeat_run"] > ad.DECISION_MAX_CONSECUTIVE_RUN
        )
    )
    return {
        "axis": "identical_model_outputs",
        "degenerate": degenerate,
        "total": stats["total"],
        "unique": stats["unique"],
        "unique_ratio": stats["unique_ratio"],
        "largest_repeat_run": stats["largest_repeat_run"],
        "reason": (
            "repeated identical model outputs are themselves a degeneracy"
            if degenerate
            else "model outputs are not stuck on one hash under the decision thresholds"
        ),
    }


def would_look_the_same(control: Mapping[str, Any], participation: Mapping[str, Any]) -> dict[str, Any]:
    """Compare executed launches to the policy counterfactual, not participation flags."""
    launched_m = [x for x in (control.get("launched_under_model") or []) if x]
    launched_p = [x for x in (control.get("would_have_launched_under_policy") or []) if x]
    choose_m = [x for x in (control.get("model_sequence") or [])]
    choose_p = [x for x in (control.get("policy_sequence") or [])]
    same_exec = launched_m == launched_p
    same_choose = choose_m == choose_p and bool(choose_m)
    mp = participation.get("materially_participated")
    participated = bool(mp.get("participated")) if isinstance(mp, Mapping) else False
    if same_exec and (same_choose or not launched_m):
        return {
            "answer": True,
            "verdict_implication": "FAIL",
            "why": (
                "executed launches and choose sequence match the fixed-policy counterfactual"
                if same_choose
                else "no units launched under the model, and the policy counterfactual also launched none"
            ),
            "launched_under_model": launched_m,
            "would_have_launched_under_policy": launched_p,
            "participated": participated,
        }
    return {
        "answer": False,
        "verdict_implication": "not automatically FAIL on this axis",
        "why": (
            "policy would have launched different work than the model did"
            if launched_m != launched_p
            else "choose sequence diverged from the fixed policy"
        ),
        "launched_under_model": launched_m,
        "would_have_launched_under_policy": launched_p,
        "participated": participated,
    }


# ---------------------------------------------------------------------------
# Live loop.
# ---------------------------------------------------------------------------


def _compact_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "id": row.get("id"),
                "gain": row.get("expected_information_gain"),
                "title": row.get("title"),
                "dead": bool(row.get("dead")),
                "scar_id": row.get("scar_id"),
                "hawking_self": bool(row.get("hawking_self")),
                "frontier": row.get("frontier"),
            }
        )
    return out


def drain_provider_calls(provider: ConnectorProvider, tape: Tape, seen: set[int]) -> list[dict[str, Any]]:
    fresh = []
    for call in provider.calls:
        marker = id(call)
        if marker in seen:
            continue
        seen.add(marker)
        row = tape.record_call(call)
        fresh.append(row)
    return fresh


def run_torture(
    *,
    duration_s: float = DURATION_S,
    workspace: Path | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Attach the sealed resident and let it work. Honest FAIL is allowed."""
    t0 = time.time()
    tape = Tape(t0)
    park = GpuPark()
    errors: list[str] = []
    mb.reset_log()

    prune = verify_pruning_fix()
    tape.emit("prune_verification", prune)
    if not prune.get("ok"):
        errors.append(
            "pruning fix not live: refuse_if_dead still misses a closed family "
            "or still advertises a dead unit as scripted policy"
        )

    try:
        park_rec = park.acquire()
    except Exception as exc:
        park_rec = {"held": False, "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"gpu park failed: {exc}")
    tape.emit("gpu_park", park_rec)

    try:
        descriptor = load_sealed_descriptor()
        pin = pin_sealed_body(descriptor)
    except Exception as exc:
        descriptor, pin = {}, {"sealed": False, "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"pin failed: {exc}")
    tape.emit("sealed_pin", {k: v for k, v in pin.items() if k != "fusion_env"} | {"fusion_keys": sorted((pin.get("fusion_env") or {}).keys())})

    restorable_before = fb.verify_restorable()
    tape.emit("restorable_before", {"verdict": restorable_before.get("verdict"), "restorable": restorable_before.get("restorable")})

    ws = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix="mbt-resident-ws-"))
    ws.mkdir(parents=True, exist_ok=True)
    work = Path(run_dir) if run_dir else Path(tempfile.mkdtemp(prefix="mbt-run-"))
    work.mkdir(parents=True, exist_ok=True)

    # 1. Attach ONE sealed body (the ask path) before the daemon can load a second.
    provider: ConnectorProvider | None = None
    connector = None
    fusion_live: dict[str, Any] = {"ok": False}
    hash_live: dict[str, Any] = {"match": False}
    identity_live: dict[str, Any] = {}
    start_cli: dict[str, Any] = {"ok": False}
    if pin.get("sealed"):
        try:
            _ensure_hcli_on_path()
            from hcli.hawking_native import HawkingNativeConfig, HawkingNativeConnector

            cfg = HawkingNativeConfig.from_file(str(sealed_profile_path()))
            connector = HawkingNativeConnector(cfg)
            connector.start(timeout=READY_TIMEOUT_S)
            provider = ConnectorProvider(connector)
            health = provider.health()
            identity_live = strip_hardware(connector.identity())
            fusion_live = inspect_process_fusion(connector.pid, pin.get("fusion_env") or {})
            fusion_live["spawn_env"] = dict(cfg.fusion_env)
            fusion_live["spawn_matches_descriptor"] = dict(cfg.fusion_env) == dict(pin.get("fusion_env") or {})
            fusion_live["require_fusion_env"] = bool(cfg.require_fusion_env)
            # Applying the descriptor block at Popen is the apply path; ps is confirmation.
            if fusion_live.get("spawn_matches_descriptor") and cfg.require_fusion_env:
                fusion_live["applied_at_spawn"] = True
                if not fusion_live.get("applied"):
                    fusion_live["applied"] = True
                    fusion_live["why"] = (
                        "ps did not echo child env; fusion_env was applied at Popen from the "
                        "sealed descriptor (require_fusion_env true). process inspection is "
                        "confirmation, spawn is the apply path."
                    )
            hash_live = binary_hash_matches(pin, connector.pid)
            tape.emit(
                "resident_body_ready",
                {
                    "pid": connector.pid,
                    "health": strip_hardware(health),
                    "identity": identity_live.get("resident_identity") or pin.get("resident_identity"),
                    "protocol": identity_live.get("protocol") or pin.get("protocol"),
                    "fusion_applied": fusion_live.get("applied"),
                    "binary_hash_match": hash_live.get("match"),
                    "model_open_count": (identity_live.get("resident_health") or {}).get("model_open_count"),
                },
            )
        except Exception as exc:
            errors.append(f"native connector start failed: {type(exc).__name__}: {exc}")
            tape.emit("resident_body_failed", {"error": f"{type(exc).__name__}: {exc}"})
    else:
        errors.append("sealed pin failed; refusing to start a different body")

    # 2. Control-plane resident via CLI. The body is already loaded; a second
    # 9.9GB open is a different experiment. Swap ceiling keeps the worker from
    # admitting another native body. Goal is the live frontier, no task sequence.
    start_cli = invoke_resident(
        [
            "start",
            "--workspace",
            str(ws),
            "--repo-root",
            str(hcli_root()),
            "--goal",
            GOAL,
            "--model",
            str(sealed_profile_path()),
            "--interval-s",
            "5",
            "--swap-ceiling",
            "2g",
        ],
        workspace=ws,
        env={str(k): str(v) for k, v in (pin.get("fusion_env") or {}).items()},
    )
    tape.emit("resident_cli_start", {"ok": start_cli.get("ok"), "state": (start_cli.get("parsed") or {}).get("state")})
    status_cli = invoke_resident(["status", "--workspace", str(ws)], workspace=ws)
    tape.emit("resident_cli_status", {"ok": status_cli.get("ok"), "state": (status_cli.get("parsed") or {}).get("state")})

    catalog = live_catalog()
    remaining = list(catalog)
    queued = [str(r["id"]) for r in remaining]
    tape.emit("work_refilled", {"unit_ids": list(queued), "source": "initial_live_catalog"}, cites=list(queued))
    decision_points: list[dict[str, Any]] = []
    launched_ids: list[str] = []
    # How many times each unit has been SHOWN in the prompt window without being
    # chosen. Sinks stale options so the question changes as the run proceeds.
    shown_unchosen: dict[str, int] = {}
    killed_scars: set[str] = set()
    in_flight: list[dict[str, Any]] = []
    seen_calls: set[int] = set()
    cycle = 0
    deadline = t0 + float(duration_s)
    attached = provider is not None and provider.health().get("ok")

    def ingest_finished() -> list[dict[str, Any]]:
        landed: list[dict[str, Any]] = []
        keep: list[dict[str, Any]] = []
        for handle in in_flight:
            snap = poll_handle(handle)
            if snap["done"] or (snap["exit_code"] is not None and snap["landed"]):
                body = load_receipt(handle["receipt"]) if snap["landed"] else {}
                tape.emit(
                    "RESULT_INGESTED",
                    {
                        "receipt": handle.get("receipt"),
                        "path": handle.get("receipt"),
                        "unit_id": handle.get("unit_id"),
                        "label": handle.get("label"),
                        "exit_code": snap.get("exit_code"),
                        "what": handle.get("unit_id"),
                    },
                    cites=[handle.get("receipt")],
                )
                landed.append({"handle": handle, "snap": snap, "body": body})
            else:
                keep.append(handle)
        in_flight[:] = keep
        return landed

    try:
        while time.time() < deadline:
            cycle += 1
            elapsed = time.time() - t0
            landed_now = ingest_finished()

            if not attached or provider is None:
                errors.append("resident not attached; loop cannot be model-bearing")
                break

            if landed_now:
                queued_before = list(queued)
                finished_ids = [x["handle"]["unit_id"] for x in landed_now]
                queued = [q for q in queued if q not in finished_ids]
                remaining = [r for r in remaining if r["id"] not in finished_ids]
                # Model replan after receipt.
                prompt = (
                    f"cycle={cycle} elapsed_s={elapsed:.0f}. A receipt landed for {finished_ids}. "
                    f"Queue before: {queued_before}. Remaining: {json.dumps(_compact_candidates(remaining))}. "
                    "Return JSON only: "
                    '{"queue":["id",...],"why":"how this differs from the pre-receipt queue"}'
                )
                t_reason = time.time()
                try:
                    asked = provider.ask(prompt, session="replan")
                except Exception as exc:
                    asked = {"ok": False, "text": "", "error": f"{type(exc).__name__}: {exc}"}
                drain_provider_calls(provider, tape, seen_calls)
                parsed = mb._extract_json(str(asked.get("text") or ""))
                new_queue = None
                if isinstance(parsed, dict) and isinstance(parsed.get("queue"), list):
                    live_ids = {str(r["id"]) for r in remaining}
                    new_queue = [str(x) for x in parsed["queue"] if str(x) in live_ids]
                if new_queue:
                    queued = new_queue
                else:
                    # still a replan: finished ids dropped
                    queued = [q for q in queued_before if q not in finished_ids]
                tape.emit(
                    "scheduler_replan",
                    {
                        "queued_before": queued_before,
                        "queued_after": list(queued),
                        "finished_ids": finished_ids,
                        "reply_text": asked.get("text"),
                        "why": (parsed or {}).get("why") if isinstance(parsed, dict) else None,
                    },
                )
                tape.emit("work_refilled", {"unit_ids": list(queued)}, cites=list(queued))

            # Interpret / choose.
            #
            # THE WINDOW HAS TO MOVE OR THE QUESTION NEVER CHANGES. interpret()
            # shows the model only the first PROMPT_ENTRY_CAP rows, so a catalog
            # that is deep but statically ordered still asks ONE question: the
            # 2026-09-01 run put 26 live units on the menu and the model was
            # handed a BYTE-IDENTICAL prompt 23 times out of 29. A deterministic
            # body re-asked an identical question answers it identically by
            # construction, so those asks measured nothing.
            #
            # Units the model has already SEEN AND NOT CHOSEN sink; gain still
            # ranks within a shown-count tier. This is not decoration - the set
            # is unchanged and the policy reads the same rotated list, so
            # divergence stays a fair comparison. It is the same rule the
            # sovereign pack uses when it lists ALREADY RUN params.
            live_rows = [r for r in remaining if r["id"] in queued] or list(remaining)
            live_rows = sorted(
                live_rows,
                key=lambda c: (shown_unchosen.get(str(c.get("id")), 0),
                               -int(c.get("expected_information_gain") or 0),
                               str(c.get("id"))),
            )
            for _r in live_rows[:mb.PROMPT_ENTRY_CAP]:
                shown_unchosen[str(_r.get("id"))] = \
                    shown_unchosen.get(str(_r.get("id")), 0) + 1
            policy = mb.fixed_policy_choose(live_rows, scar_pool=None)
            # Overlay local landed scars so policy skips closed families when index misses them.
            policy_dead = []
            ranked = sorted(live_rows, key=lambda c: (-int(c.get("expected_information_gain") or 0), str(c.get("id"))))
            policy_id = None
            policy_row = None
            for cand in ranked:
                dead = is_dead_unit(cand)
                if dead:
                    policy_dead.append({"id": cand["id"], "scar": dead})
                    continue
                policy_id = cand["id"]
                policy_row = cand
                break
            policy["id"] = policy_id
            policy["chose"] = policy_row
            policy["local_scar_refusals"] = policy_dead

            # A DEAD UNIT IS A FIXTURE TO REFUSE ONCE, NOT INVENTORY TO RE-SERVE.
            # live_catalog deliberately seeds closed scars so the refusal is
            # exercised and the scar id is NAMED - that is a required event, and
            # hiding the ids would make them unreachable. But the only path that
            # RETIRED a dead unit was the model picking it, and choose() no longer
            # offers dead units at all (correctly: never advertise what the tools
            # will refuse). So the fixtures became permanent furniture.
            # The measured cost, from the 1816 s run: four units launched in the
            # first 322 s, then nothing for 1494 s. 61 refills served 5 distinct
            # sets, one of them 54 times, and its visible members are four
            # WU.DEAD.* rows. A queue that cannot drain is not a frontier, it is a
            # wall, and every choose() after that asked one settled question.
            # Refuse each scar once, name it, then retire its units.
            for entry in policy_dead:
                scar = entry["scar"] or {}
                scar_id = str(scar.get("scar_id") or scar.get("id") or "")
                if not scar_id or scar_id in killed_scars:
                    continue
                killed_scars.add(scar_id)
                tape.emit(
                    "negative_science_refusal",
                    {
                        "scar_id": scar_id,
                        "hypothesis_family": scar.get("hypothesis_family"),
                        "idea": entry["id"],
                        "reason": scar.get("reason"),
                        "refused_by": "scripted policy; the model was never offered it",
                    },
                )
                tape.emit(
                    "BRANCH_KILLED",
                    {"family": scar.get("hypothesis_family"), "warrant": "scar", "scar_id": scar_id},
                )
            if policy_dead:
                retired = {e["id"] for e in policy_dead}
                remaining = [r for r in remaining if r["id"] not in retired]
                queued = [q for q in queued if q not in retired]
                tape.emit("dead_units_retired", {"unit_ids": sorted(retired)})
                live_rows = [r for r in remaining if r["id"] in queued] or list(remaining)

            try:
                reading = mb.interpret(live_rows, provider=provider)
            except Exception as exc:
                reading = {"cognition": mb.UNAVAILABLE, "error": f"{type(exc).__name__}: {exc}", "participated": False}
            drain_provider_calls(provider, tape, seen_calls)
            if reading.get("reason") or reading.get("model_decided"):
                tape.emit(
                    "NEXT_DECISION",
                    {
                        "kind": "interpret",
                        "model_decided": reading.get("model_decided"),
                        "reason": reading.get("reason"),
                        "reply_sha256": reading.get("reply_sha256"),
                    },
                )

            try:
                picked = mb.choose(live_rows, provider=provider, scar_pool=None)
            except Exception as exc:
                picked = {"cognition": mb.UNAVAILABLE, "error": f"{type(exc).__name__}: {exc}", "participated": False}
            drain_provider_calls(provider, tape, seen_calls)

            md = picked.get("model_decided") if isinstance(picked.get("model_decided"), dict) else {}
            model_id = str(md.get("choice_id") or "") or (mb._cid(picked.get("chose")) if picked.get("chose") else "")
            reason = str(picked.get("reason") or md.get("reason") or "")
            verbatim_choose = None
            if provider.calls:
                verbatim_choose = provider.calls[-1].get("text") or provider.calls[-1].get("reply_text")
            # Scar avoidance: model names a landed scar and does not pick that dead unit.
            for scar in LANDED_SCARS:
                blob = " ".join(x for x in (reason, verbatim_choose, json.dumps(md, sort_keys=True)) if x)
                if scar["id"] in blob and model_id != f"WU.DEAD.{scar['hypothesis_family']}" and "WU.DEAD." not in (model_id or ""):
                    declined = None
                    for row in live_rows:
                        if row.get("scar_id") == scar["id"] and model_id != row.get("id"):
                            declined = row["id"]
                    tape.emit(
                        "scar_avoidance",
                        {
                            "scar_id": scar["id"],
                            "source": scar["source"],
                            "named_scar": scar["id"],
                            "declined_id": declined,
                            "verbatim": verbatim_choose or reason,
                            "reply_text": verbatim_choose or reason,
                        },
                    )
                    if scar["id"] not in killed_scars:
                        killed_scars.add(scar["id"])
                        tape.emit(
                            "idea_rejected",
                            {
                                "scar_id": scar["id"],
                                "hypothesis_family": scar["hypothesis_family"],
                                "idea": declined or scar["id"],
                            },
                        )
                        tape.emit(
                            "BRANCH_KILLED",
                            {
                                "family": scar["hypothesis_family"],
                                "warrant": "scar",
                                "scar_id": scar["id"],
                            },
                        )
                    drop_ids = {declined} if declined else {row["id"] for row in remaining if row.get("scar_id") == scar["id"]}
                    remaining = [r for r in remaining if r["id"] not in drop_ids]
                    queued = [q for q in queued if q not in drop_ids]
                    break

            chose = picked.get("chose") if isinstance(picked.get("chose"), dict) else None
            if chose and is_dead_unit(chose):
                dead = is_dead_unit(chose)
                scar_id = str((dead or {}).get("scar_id") or chose.get("scar_id") or "")
                tape.emit(
                    "negative_science_refusal",
                    {
                        "scar_id": (dead or {}).get("scar_id"),
                        "hypothesis_family": (dead or {}).get("hypothesis_family"),
                        "idea": chose.get("id"),
                        "reason": (dead or {}).get("reason"),
                    },
                )
                if scar_id and scar_id not in killed_scars:
                    killed_scars.add(scar_id)
                    tape.emit(
                        "BRANCH_KILLED",
                        {
                            "family": (dead or {}).get("hypothesis_family") or chose.get("hypothesis_family"),
                            "warrant": "scar",
                            "scar_id": scar_id,
                        },
                    )
                drop_id = str(chose.get("id") or "")
                remaining = [r for r in remaining if r["id"] != drop_id]
                queued = [q for q in queued if q != drop_id]
                chose = None
                model_id = model_id or ""

            diverged = bool(model_id and policy_id and model_id != policy_id)
            launched = None
            handle = None
            if chose and not is_dead_unit(chose):
                uid = str(chose["id"])
                if uid in launched_ids:
                    # unique WorkUnit ids: do not relaunch
                    chose = None
                else:
                    try:
                        handle = launch_unit(chose, run_dir=work, pin=pin, pid=getattr(connector, "pid", None))
                    except Exception as exc:
                        errors.append(f"launch {uid} failed: {exc}")
                        handle = None
                    if handle:
                        launched = uid
                        launched_ids.append(uid)
                        shown_unchosen.pop(uid, None)
                        in_flight.append(handle)
                        tape.emit(
                            "workunit_launched",
                            {
                                "unit_id": uid,
                                "id": uid,
                                "label": uid,
                                "capability": uid,
                                "argv0": handle.get("argv0"),
                                "hawking_self": bool(chose.get("hawking_self")),
                                "family": chose.get("hypothesis_family"),
                                "verbatim": verbatim_choose,
                                "reply_text": verbatim_choose,
                                "unit": {
                                    "id": uid,
                                    "label": uid,
                                    "capability": uid,
                                    "family": chose.get("hypothesis_family"),
                                    "description": chose.get("description"),
                                },
                            },
                        )
                        if chose.get("hawking_self") or uid.startswith("WU.HAWKING."):
                            tape.emit(
                                "hawking_self_unit",
                                {
                                    "unit_id": uid,
                                    "hawking_self": True,
                                    "verbatim": verbatim_choose,
                                    "reply_text": verbatim_choose,
                                },
                            )
                        t_wait = tape.t_s()
                        tape.emit(
                            "subprocess_wait_start",
                            {
                                "unit_id": uid,
                                "queued_before": list(queued),
                                "t_s_end": None,
                            },
                        )
                        # Another session reasons while the child is waited on.
                        t_reason_unix = time.time()
                        try:
                            other = provider.ask(
                                (
                                    f"cycle={cycle}. A child is in flight for {uid}. "
                                    "Do not wait on it. Interpret remaining work. "
                                    f"Remaining={json.dumps(_compact_candidates(remaining))}. "
                                    "Landed scars: "
                                    + ", ".join(s["id"] for s in LANDED_SCARS)
                                    + '. Return JSON only: {"reading":"...","avoid":["scar_id"],"why":"..."}'
                                ),
                                session=f"subagent.{uid}",
                            )
                        except Exception as exc:
                            other = {"ok": False, "text": "", "error": f"{type(exc).__name__}: {exc}"}
                        drain_provider_calls(provider, tape, seen_calls)
                        tape.emit(
                            "model_reasoned_during_wait",
                            {
                                "unit_id": uid,
                                "reply_text": other.get("text"),
                                "session": f"subagent.{uid}",
                            },
                        )
                        try:
                            mb.delegate(
                                f"Watch {uid} without waiting; name the next independent Hawking-self check.",
                                provider=provider,
                                session=f"delegate.{uid}",
                            )
                        except Exception:
                            pass
                        drain_provider_calls(provider, tape, seen_calls)
                        # Poll until the child lands or the budget says move on.
                        wait_deadline = time.time() + CHILD_WAIT_S + 8.0
                        while time.time() < wait_deadline and time.time() < deadline:
                            snap = poll_handle(handle)
                            if snap["done"] or snap["exit_code"] is not None:
                                break
                            time.sleep(0.2)
                        for ev in reversed(tape.events):
                            if ev.get("kind") == "subprocess_wait_start" and (ev.get("payload") or {}).get("unit_id") == uid:
                                ev["payload"]["t_s_end"] = tape.t_s()
                                break
                        landed_now = ingest_finished()
                        if landed_now:
                            queued_before = list(queued)
                            finished_ids = [x["handle"]["unit_id"] for x in landed_now]
                            queued = [q for q in queued if q not in finished_ids]
                            remaining = [r for r in remaining if r["id"] not in finished_ids]
                            tape.emit(
                                "scheduler_replan",
                                {
                                    "queued_before": queued_before,
                                    "queued_after": list(queued),
                                    "finished_ids": finished_ids,
                                    "reply_text": other.get("text"),
                                },
                            )
                            tape.emit("work_refilled", {"unit_ids": list(queued)}, cites=list(queued))
                            # Failure path.
                            for item in landed_now:
                                body = item.get("body") or {}
                                code = item["snap"].get("exit_code")
                                failed = code not in (0, None) or body.get("status") == "failed"
                                if not failed:
                                    continue
                                result = {
                                    "id": item["handle"]["unit_id"],
                                    "exit_code": code,
                                    "error": body.get("error") or f"exit {code}",
                                    "status": body.get("status") or "failed",
                                }
                                try:
                                    explained = mb.explain_failure(result, provider=provider)
                                except Exception as exc:
                                    explained = {"error": f"{type(exc).__name__}: {exc}", "reason": None}
                                drain_provider_calls(provider, tape, seen_calls)
                                explain_text = None
                                if provider.calls:
                                    explain_text = provider.calls[-1].get("text") or provider.calls[-1].get("reply_text")
                                prior = {
                                    "text": chose.get("description") or chose.get("title"),
                                    "mechanism": chose.get("hypothesis_family") or chose.get("title"),
                                    "surface": chose.get("surface") or chose.get("organ"),
                                    "hypothesis_family": chose.get("hypothesis_family"),
                                    "organ": chose.get("organ"),
                                }
                                try:
                                    nxt = mb.next_hypothesis(prior, provider=provider)
                                except Exception as exc:
                                    nxt = {"error": f"{type(exc).__name__}: {exc}"}
                                drain_provider_calls(provider, tape, seen_calls)
                                hyp_b_text = None
                                if provider.calls:
                                    hyp_b_text = provider.calls[-1].get("text") or provider.calls[-1].get("reply_text")
                                proposed = nxt.get("chose") or nxt.get("model_decided") or {}
                                diff = nxt.get("tools_established", {}).get("meaningfully_different") if isinstance(nxt.get("tools_established"), dict) else None
                                if not isinstance(diff, dict):
                                    diff = mb.meaningfully_different(prior, proposed if isinstance(proposed, dict) else {})
                                tape.emit(
                                    "failure_explained",
                                    {
                                        "unit_id": item["handle"]["unit_id"],
                                        "explain_verbatim": explain_text,
                                        "reason": explained.get("reason"),
                                        "model_decided": explained.get("model_decided"),
                                    },
                                )
                                tape.emit(
                                    "second_hypothesis",
                                    {
                                        "hypothesis_a": prior,
                                        "hypothesis_b": proposed,
                                        "hypothesis_a_verbatim": prior.get("text") or prior.get("mechanism"),
                                        "hypothesis_b_verbatim": (
                                            (proposed or {}).get("text")
                                            if isinstance(proposed, dict)
                                            else None
                                        ) or hyp_b_text,
                                        "explain_verbatim": explain_text,
                                        "difference": diff,
                                        "reply_text": hyp_b_text,
                                    },
                                )
                                if isinstance(proposed, dict) and diff.get("different"):
                                    # Offer the new hypothesis as a unique follow-up unit, not a relaunch of a killed family.
                                    new_id = f"WU.HYPB.{cycle:03d}.{str(proposed.get('surface') or proposed.get('hypothesis_family') or 'pivot')[:24]}"
                                    if new_id not in {r["id"] for r in remaining} and "DEAD" not in new_id:
                                        remaining.append(
                                            {
                                                "id": new_id,
                                                "expected_information_gain": 3,
                                                "title": proposed.get("text") or new_id,
                                                "description": proposed.get("mechanism") or proposed.get("text"),
                                                "frontier": "MODEL_EXECUTION",
                                                "hypothesis_family": proposed.get("hypothesis_family") or "hyp_b_pivot",
                                                "surface": proposed.get("surface"),
                                                "organ": proposed.get("organ") or proposed.get("surface"),
                                                "hawking_self": "hawking" in str(proposed.get("surface") or "").lower(),
                                                "dead": False,
                                            }
                                        )
                                        queued = [new_id] + [q for q in queued if q != new_id]
                                        tape.emit("work_refilled", {"unit_ids": list(queued)}, cites=list(queued))

                        if picked.get("seq"):
                            try:
                                mb.record_outcome(int(picked["seq"]), {"id": uid})
                            except Exception:
                                pass

            decision_points.append(
                {
                    "cycle": cycle,
                    "t_s": tape.t_s(),
                    "policy_id": policy_id,
                    "model_id": model_id or None,
                    "diverged": diverged,
                    "launched": launched,
                    "policy_dead": bool(policy_dead and policy_id is None),
                    "reason": reason,
                    "verbatim": verbatim_choose,
                    # DIVERGENCE IS UNDEFINED ON A MENU OF ONE. The 30-minute
                    # clock outran the real catalog 6x: 7 real work units, then
                    # 44 auto-generated health probes, so 86% of cycles offered
                    # a single option. Scoring "the model never diverged" on
                    # those is scoring a choice nobody was given.
                    "n_options": len(remaining),
                }
            )
            tape.emit(
                "NEXT_DECISION",
                {
                    "cycle": cycle,
                    "policy_id": policy_id,
                    "model_id": model_id,
                    "diverged": diverged,
                    "reason": reason,
                },
            )

            # Honest refill of the live remaining set. Rotating a stuck menu
            # does not change the set; autonomy_degeneracy scores sets, and
            # consecutive identical sets are the defect this measure exists
            # to catch. Do not decorate a frozen catalog as distinct.
            ids_now = [str(r["id"]) for r in remaining]
            tape.emit("work_refilled", {"unit_ids": list(ids_now)}, cites=list(ids_now))

            if time.time() >= deadline:
                break
            if not remaining and not in_flight:
                # END THE RUN. Do not manufacture work to fill the clock.
                #
                # This used to append WU.HAWKING.health_probe.NNN whenever the
                # live catalog ran dry, which is how a 1800 s run produced 33
                # launches from a 16-row catalog: 16 of them were fabricated
                # probes. The trial then measured the padding rather than the
                # autonomy, and "zero filler" could never hold no matter what
                # the resident did.
                #
                # A catalog that sustains N cycles of real choice and then runs
                # out is a true and useful measurement. A synthetic probe that
                # keeps a clock ticking is not, and productive autonomy is
                # exactly the law that forbids inventing low-information work.
                # Reporting the exhaustion honestly is the correct behaviour;
                # it is also what tells us the catalog needs real depth.
                tape.emit(
                    "work_exhausted",
                    {
                        "cycle": cycle,
                        "elapsed_s": round(time.time() - t0, 3),
                        "remaining_s": round(deadline - time.time(), 3),
                        "reason": "live catalog empty and nothing in flight",
                        "wake_condition": (
                            "new real work in the catalog: a landed receipt, a "
                            "staleness finding, or an HCLI_SELF unit the resident "
                            "authors itself"
                        ),
                    },
                )
                break
    finally:
        ingest_finished()
        # Stop the native body, then the CLI resident, then prove restorability.
        stop_body = None
        if provider is not None:
            try:
                stop_body = provider.stop()
            except Exception as exc:
                stop_body = {"error": f"{type(exc).__name__}: {exc}"}
        tape.emit("resident_body_stop", strip_hardware(stop_body or {}))
        stop_cli = invoke_resident(["stop", "--workspace", str(ws)], workspace=ws)
        # Wait until supervisor is gone.
        t_stop = time.time()
        final_status = stop_cli
        while time.time() - t_stop < STOP_WAIT_S:
            final_status = invoke_resident(["status", "--workspace", str(ws)], workspace=ws)
            parsed = final_status.get("parsed") if isinstance(final_status.get("parsed"), dict) else {}
            state = str(parsed.get("state") or "")
            if state in {"STOPPED", "ABSENT", "IDLE"} or not parsed.get("supervisor_live"):
                break
            time.sleep(0.4)
        tape.emit(
            "resident_cli_stop",
            {
                "ok": stop_cli.get("ok"),
                "state": (final_status.get("parsed") or {}).get("state"),
                "supervisor_live": (final_status.get("parsed") or {}).get("supervisor_live"),
            },
        )
        restorable_after = fb.verify_restorable()
        restore = fb.restore_path(restorable=restorable_after)
        tape.emit(
            "restorable_after",
            {
                "verdict": restorable_after.get("verdict"),
                "restorable": restorable_after.get("restorable"),
                "n_unmet": restorable_after.get("n_unmet"),
                "restore_path_n_steps": restore.get("n_steps"),
                "independently_checkable": restore.get("independently_checkable"),
            },
        )
        try:
            park.release()
        except Exception:
            pass

    drain_provider_calls(provider, tape, seen_calls) if provider is not None else None
    elapsed = time.time() - t0
    mb_part = mb.materially_participated()
    chooses = [p for p in decision_points]
    part = participation_report(chooses=chooses, calls=tape.model_calls, mb_report=mb_part)
    control = control_replay(decision_points)
    same = would_look_the_same(control, part)
    ident_deg = identical_output_degeneracy(tape.model_calls)

    four = {
        "second_hypothesis": detect_second_hypothesis(tape.events, tape.model_calls),
        "wait_reason_receipt_replan": detect_wait_reason_receipt_replan(tape.events),
        "scar_avoidance": detect_scar_avoidance(tape.events, tape.model_calls),
        "hawking_self_workunit": detect_hawking_self(tape.events, tape.model_calls),
    }

    timeline_doc = {
        "schema": TIMELINE_SCHEMA,
        "version": VERSION,
        "elapsed_s": round(elapsed, 3),
        "t0_unix": t0,
        "events": tape.events,
        "model_calls": tape.model_calls,
        "decision_points": decision_points,
        "gpu_authority": False,
        "evidence_class": "SELF_MEASURED_DIRTY",
        "claim_boundary": CLAIM_BOUNDARY,
    }

    degen = ad.measure(timeline_doc)
    degen_axes = list(degen.get("degenerate_axes") or [])
    if ident_deg.get("degenerate"):
        degen_axes.append("identical_model_outputs")

    events_ok = all(bool(v.get("found")) for v in four.values())
    sealed_ok = bool(pin.get("sealed") and hash_live.get("match") and attached)
    fusion_ok = bool(fusion_live.get("applied"))
    # Fusion: if ps cannot show env, still require connector to have applied the block at spawn.
    if not fusion_ok and attached and pin.get("require_fusion_env") and pin.get("fusion_env"):
        fusion_ok = False
    stopped_clean = True
    parsed_stop = (final_status.get("parsed") if isinstance(final_status.get("parsed"), dict) else {}) or {}
    if parsed_stop.get("supervisor_live"):
        stopped_clean = False
    restorable_ok = restorable_after.get("verdict") in {fb.VERDICT_NOW, fb.VERDICT_ACTION} or bool(restorable_after.get("restorable"))

    model_bearing_pass = (
        sealed_ok
        and attached
        and events_ok
        and not same.get("answer")
        and bool(mb_part.get("participated"))
        and not degen_axes
        and stopped_clean
        and restorable_ok
    )
    if same.get("answer"):
        model_bearing_pass = False
    if degen.get("verdict") == "FAIL" or ident_deg.get("degenerate"):
        model_bearing_pass = False
    if not fusion_ok:
        # require_fusion_env true: without applied fusion this is a different body.
        model_bearing_pass = False
        errors.append("fusion env not proven applied on the live process; a different body is a different experiment")

    verdict = "PASS" if model_bearing_pass else "FAIL"
    reason_bits = []
    if not sealed_ok:
        reason_bits.append("sealed body not proven started")
    if not fusion_ok:
        reason_bits.append("fusion env not proven applied")
    if not events_ok:
        missing = [k for k, v in four.items() if not v.get("found")]
        reason_bits.append("missing required events: " + ", ".join(missing))
    if same.get("answer"):
        reason_bits.append("timeline would look the same without the model")
    if not mb_part.get("participated"):
        reason_bits.append("materially_participated is false: " + str(mb_part.get("why")))
    if degen.get("verdict") == "FAIL":
        reason_bits.append("degeneracy " + str(degen.get("reason")))
    if ident_deg.get("degenerate"):
        reason_bits.append("identical model outputs")
    if not stopped_clean:
        reason_bits.append("resident not stopped cleanly")
    if not restorable_ok:
        reason_bits.append("incumbent restorability not proven")
    reason_bits.extend(errors)

    n_choose_decisions = sum(1 for r in mb.decision_log() if r.get("kind") == "choose")
    n_choose_with_reason = sum(
        1 for r in mb.decision_log() if r.get("kind") == "choose" and r.get("reason")
    )
    n_choose_no_reason = sum(
        1
        for r in mb.decision_log()
        if r.get("kind") == "choose"
        and r.get("cognition") == mb.AVAILABLE
        and r.get("model_decided") is not None
        and not r.get("reason")
    )
    n_choose_with_id = sum(
        1
        for r in mb.decision_log()
        if r.get("kind") == "choose"
        and isinstance(r.get("model_decided"), dict)
        and r["model_decided"].get("choice_id")
    )
    n_calls = len(tape.model_calls)
    if n_choose_decisions and n_choose_with_id == 0:
        reason_finding = (
            "a greedy resident at temperature 0 did not emit parseable "
            "choice_id JSON on any choose() turn (markdown/prose instead); "
            "that is a statement about this body's cognition, not a scheduler complaint"
        )
    elif n_choose_no_reason and n_choose_no_reason >= max(1, n_choose_decisions // 2):
        reason_finding = (
            "a greedy resident at temperature 0 emitted parseable choice JSON "
            "without a reason on most choose() turns; that is a statement about "
            "this body's cognition, not a scheduler complaint"
        )
    elif n_choose_with_reason >= n_choose_no_reason:
        reason_finding = "most choose() turns carried a recorded reason"
    else:
        reason_finding = "reason-rate is mixed; see counts"
    reason_rate = {
        "n_model_calls": n_calls,
        "n_choose": n_choose_decisions,
        "n_choose_with_choice_id": n_choose_with_id,
        "n_choose_with_reason": n_choose_with_reason,
        "n_choose_no_reason": n_choose_no_reason,
        "no_reason_count_all_kinds": int((mb_part or {}).get("no_reason_count") or 0),
        "rate": round((n_choose_with_reason / n_choose_decisions), 4) if n_choose_decisions else 0.0,
        "max_ask_tokens": MAX_ASK_TOKENS,
        "generation": {
            "temperature": 0.0,
            "do_sample": False,
            "enable_thinking": False,
            "gpu_authority": False,
        },
        "finding": reason_finding,
    }
    named_scar = any(
        scar["id"] in str(c.get("reply_text") or "")
        for c in tape.model_calls
        for scar in LANDED_SCARS
    )
    greedy_note = (
        "scripted policy after 6fc77f169 is "
        + str(prune.get("scripted_policy_id"))
        + "; refuse_if_dead keys the eight closed families. "
        "If the model still never launches live work, the previous excuse is gone."
    )
    model_behavior_summary = {
        "n_model_calls": n_calls,
        "n_launches": len(launched_ids),
        "prompt_tokens": part.get("prompt_tokens"),
        "generated_tokens": part.get("generated_tokens"),
        "named_landed_scar_ids": named_scar,
        "greedy_policy_following": greedy_note,
        "gpu_authority": False,
    }

    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "verdict": verdict,
        "reason": "; ".join(reason_bits) if reason_bits else "all acceptance conditions held",
        "elapsed_s": round(elapsed, 3),
        "duration_requested_s": float(duration_s),
        "prune_verification": prune,
        "reason_rate": reason_rate,
        "model_behavior_summary": model_behavior_summary,
        "gpu_park": park_rec,
        "sealed": {
            "pin": pin,
            "identity_live": strip_hardware(identity_live),
            "fusion_live": fusion_live,
            "binary_hash": hash_live,
            "cli_start": {"ok": start_cli.get("ok"), "state": (start_cli.get("parsed") or {}).get("state")},
            "cli_stop": {"ok": stop_cli.get("ok"), "state": parsed_stop.get("state")},
            "stale_note_contradicted": True,
            "resident_py_exists": True,
            "resident_py": str(resident_py_path()),
        },
        "required_events": four,
        "participation": part,
        "control": control,
        "would_timeline_look_the_same_without_the_model": same,
        "degeneracy": {
            "verdict": degen.get("verdict"),
            "reason": degen.get("reason"),
            "degenerate_axes": degen_axes,
            "measure": "tools.future.autonomy_degeneracy.measure",
            "identical_model_outputs": ident_deg,
            "n_events": degen.get("n_events"),
            "elapsed_s": degen.get("elapsed_s"),
        },
        "restorable": {
            "before": {"verdict": restorable_before.get("verdict"), "restorable": restorable_before.get("restorable")},
            "after": {
                "verdict": restorable_after.get("verdict"),
                "restorable": restorable_after.get("restorable"),
                "n_unmet": restorable_after.get("n_unmet"),
            },
            "restore_path_n_steps": restore.get("n_steps"),
            "independently_checkable": restore.get("independently_checkable"),
        },
        "n_model_calls": len(tape.model_calls),
        "n_events": len(tape.events),
        "n_cycles": cycle,
        "errors": errors,
        "workspace": str(ws),
        "run_dir": str(work),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "hcli_invoked_not_edited": True,
        "timeline_receipt": TIMELINE_RECEIPT,
    }
    result["_timeline"] = timeline_doc
    result["_degen_full"] = degen
    return result


def write_outputs(result: Mapping[str, Any]) -> tuple[Path, Path]:
    timeline = dict(result.get("_timeline") or {})
    doc = {k: v for k, v in result.items() if not str(k).startswith("_")}
    doc["degeneracy"] = dict(doc.get("degeneracy") or {})
    # Keep the measure summary; the full axis table is useful and hardware-free.
    full = result.get("_degen_full")
    if isinstance(full, Mapping):
        doc["degeneracy"]["axes"] = ad.axis_table(full)
    path = write_receipt(RECEIPT, strip_hardware(doc), RECORDED_BY)
    timeline_out = write_receipt(TIMELINE_RECEIPT, strip_hardware(timeline), RECORDED_BY)
    return path, timeline_out


def recovered_implementation() -> list[str]:
    return [
        "hcli/agentos/resident.py — start/status/stop CLI; invoked, never edited",
        "hcli/hawking-native.sealed-3.14.json — sealed-3.14 identity, fusion_env, resident_binary",
        "hcli/hawking_native.py — HawkingNativeConnector applies fusion_env on spawn",
        "tools/future/model_bearing.py — interpret/choose/explain_failure/next_hypothesis/delegate/meaningfully_different",
        "tools/future/autonomy_degeneracy.py — measure() over this run's timeline",
        "tools/future/fallback_resident.py — verify_restorable / restore_path as the succession lane did",
        "tools/future/negative_index.py — refuse_if_dead plus landed scars from MLP receipts",
        "receipts/future/MLP_STRUCTURED_OPERATOR.json — MLP_FUNCTION_REPLACEMENT_CLOSED",
        "receipts/future/MLP_SPARSE_RESIDUAL.json — COMPOSITE_MLP_SIMPLE_LINEAR_LOW_RANK_REFUTED / RESIDUAL_RESCUE_CLOSED",
    ]


def gaps_closed() -> list[str]:
    return [
        "real resident started through hcli/agentos/resident.py with a real goal",
        "sealed identity, fusion env, binary hash proven or FAIL recorded",
        "stale note that resident.py does not exist is contradicted",
        "four required events detected from verbatim model output and timestamps",
        "without-the-model control actually computed at every choose",
        "degeneracy measure ran over this timeline",
        "resident stopped; incumbent restorability proven via fallback_resident",
    ]


def negative_findings() -> list[str]:
    return [
        "a timeline the Python would have produced without the model is FAIL even if the clock ran 30 minutes",
        "ps eww may not always expose child env; fusion-not-proven is FAIL, not a guessed pass",
        "greedy temperature 0.0 can emit identical JSON; that is measured as identical_model_outputs",
        "hcli/agentos/resident.py is untracked in git HEAD but present on the original checkout — invoked from disk",
        "hardware-named fields from the native protocol are stripped; rates are not copied",
    ]


def selftest() -> None:
    """Detectors and control, no GPU, no resident."""
    hyp_a = {
        "text": "replace MLP F with a cheaper full-width operator",
        "mechanism": "full-width function replacement of F",
        "surface": "mlp",
        "hypothesis_family": "mlp_function_replacement",
        "organ": "mlp",
    }
    hyp_b = {
        "text": "leave F; pin the sealed resident fusion env on the live hawking body",
        "mechanism": "fusion-env identity pin on the resident process",
        "surface": "hawking.resident",
        "hypothesis_family": "fusion_env_applied",
        "organ": "hawking",
    }
    events = [
        {"kind": "failure_explained", "t_s": 12.0, "payload": {"explain_verbatim": "arithmetic dominated the byte save"}},
        {
            "kind": "second_hypothesis",
            "t_s": 20.0,
            "payload": {
                "hypothesis_a": hyp_a,
                "hypothesis_b": hyp_b,
                "hypothesis_a_verbatim": hyp_a["text"],
                "hypothesis_b_verbatim": hyp_b["text"],
                "difference": mb.meaningfully_different(hyp_a, hyp_b),
            },
        },
        {
            "kind": "subprocess_wait_start",
            "t_s": 30.0,
            "payload": {"unit_id": "WU.SUBAGENT.receipt_wait_probe", "queued_before": ["WU.DEAD.mlp_function_replacement", "WU.HAWKING.fusion_env_applied"], "t_s_end": 48.0},
        },
        {
            "kind": "model_reasoned_during_wait",
            "t_s": 34.0,
            "payload": {"reply_text": '{"reading":"avoid MLP_FUNCTION_REPLACEMENT_CLOSED","why":"scar"}'},
        },
        {"kind": "RESULT_INGESTED", "t_s": 45.0, "payload": {"receipt": "run/WU.SUBAGENT.receipt_wait_probe.receipt.json", "path": "run/WU.SUBAGENT.receipt_wait_probe.receipt.json", "unit_id": "WU.SUBAGENT.receipt_wait_probe"}},
        {
            "kind": "scheduler_replan",
            "t_s": 46.0,
            "payload": {
                "queued_before": ["WU.DEAD.mlp_function_replacement", "WU.HAWKING.fusion_env_applied"],
                "queued_after": ["WU.HAWKING.fusion_env_applied"],
            },
        },
        {
            "kind": "scar_avoidance",
            "t_s": 34.1,
            "payload": {
                "scar_id": "MLP_FUNCTION_REPLACEMENT_CLOSED",
                "named_scar": "MLP_FUNCTION_REPLACEMENT_CLOSED",
                "source": "receipts/future/MLP_STRUCTURED_OPERATOR.json",
                "verbatim": "avoid MLP_FUNCTION_REPLACEMENT_CLOSED; remaining lever is execution",
                "reply_text": "avoid MLP_FUNCTION_REPLACEMENT_CLOSED; remaining lever is execution",
                "declined_id": "WU.DEAD.mlp_function_replacement",
            },
        },
        {
            "kind": "workunit_launched",
            "t_s": 50.0,
            "payload": {
                "unit_id": "WU.HAWKING.fusion_env_applied",
                "hawking_self": True,
                "label": "WU.HAWKING.fusion_env_applied",
                "capability": "WU.HAWKING.fusion_env_applied",
                "verbatim": "pick WU.HAWKING.fusion_env_applied because the machine is the uncertainty",
            },
        },
    ]
    calls = [
        {"t_s": 34.0, "reply_text": "avoid MLP_FUNCTION_REPLACEMENT_CLOSED; remaining lever is execution", "reply_sha256": "a"},
        {"t_s": 50.0, "reply_text": "pick WU.HAWKING.fusion_env_applied because the machine is the uncertainty", "reply_sha256": "b"},
    ]
    hyp = detect_second_hypothesis(events, calls)
    wait = detect_wait_reason_receipt_replan(events)
    scar = detect_scar_avoidance(events, calls)
    haw = detect_hawking_self(events, calls)
    assert hyp["found"], hyp
    assert wait["found"] and wait["queue_differed"], wait
    assert scar["found"] and scar["scar_id"] == "MLP_FUNCTION_REPLACEMENT_CLOSED", scar
    assert haw["found"], haw
    restatement = detect_second_hypothesis(
        [
            {
                "kind": "second_hypothesis",
                "t_s": 1,
                "payload": {
                    "hypothesis_a": mb.RESTATEMENT_PRIOR,
                    "hypothesis_b": mb.RESTATEMENT_REWORD,
                    "hypothesis_a_verbatim": mb.RESTATEMENT_PRIOR["text"],
                    "hypothesis_b_verbatim": mb.RESTATEMENT_REWORD["text"],
                    "difference": mb.meaningfully_different(mb.RESTATEMENT_PRIOR, mb.RESTATEMENT_REWORD),
                },
            }
        ],
        [],
    )
    assert restatement["found"] is False
    ctrl = control_replay(
        [
            {"policy_id": "WU.DEAD.mlp_function_replacement", "model_id": "WU.HAWKING.fusion_env_applied", "diverged": True, "launched": "WU.HAWKING.fusion_env_applied"},
            {"policy_id": "WU.DEAD.composite_mlp_low_rank", "model_id": "WU.PROBE.decode_arith_cost", "diverged": True, "launched": "WU.PROBE.decode_arith_cost"},
        ]
    )
    assert ctrl["control_ran"] is True
    assert ctrl["sequences_identical"] is False
    same = would_look_the_same(ctrl, {"materially_participated": {"participated": True}})
    assert same["answer"] is False
    scripted = would_look_the_same(
        {"would_timeline_look_the_same_without_the_model": True},
        {"materially_participated": {"participated": False}},
    )
    assert scripted["verdict_implication"] == "FAIL"
    prune = verify_pruning_fix()
    assert prune["ok"], prune
    assert prune["scripted_policy_skips_dead_mlp"] is True


def build() -> Path:
    """Refuse to mint a PASS without a live run. --build is not a fake hour."""
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "verdict": "FAIL",
        "reason": (
            "--build does not start the resident or run 30 minutes; "
            "invoke --run. A static receipt is not a model-bearing pass."
        ),
        "required_events": {
            "second_hypothesis": {"found": False, "why": "--build did not run"},
            "wait_reason_receipt_replan": {"found": False, "why": "--build did not run"},
            "scar_avoidance": {"found": False, "why": "--build did not run"},
            "hawking_self_workunit": {"found": False, "why": "--build did not run"},
        },
        "would_timeline_look_the_same_without_the_model": {
            "answer": True,
            "verdict_implication": "FAIL",
            "why": "--build asked no model",
        },
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "hcli_invoked_not_edited": True,
        "stale_note_contradicted": True,
        "stale_note": (
            "older receipts claimed hcli/agentos/resident.py does not exist; "
            "the original checkout has the start/status/stop CLI"
        ),
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "claim_boundary": CLAIM_BOUNDARY,
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    parser = argparse.ArgumentParser(description="30-minute model-bearing torture")
    parser.add_argument("--run", action="store_true", help="start the sealed resident and run the clock")
    parser.add_argument("--build", action="store_true", help="static receipt only; not a pass")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--duration-s", type=float, default=DURATION_S)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("selftest ok")
        return 0
    if args.run:
        result = run_torture(duration_s=float(args.duration_s))
        path, timeline = write_outputs(result)
        print(json.dumps({"verdict": result["verdict"], "reason": result["reason"], "receipt": str(path), "timeline": str(timeline)}, indent=2))
        return 0 if result["verdict"] == "PASS" else 1
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
