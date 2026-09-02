"""Shared plumbing for the tools/future sidecar.

Every module in this package writes a sealed JSON receipt under receipts/future/
and never asserts a hardware number. The bench block below is the single place
that enforces the second half of that.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable, Sequence,  Any

REPO = Path(__file__).resolve().parents[2]
RECEIPTS = REPO / "receipts" / "future"

# Same override the lock script honors. Tests point this at a private path
# so they do not wait 25 minutes on a live GPU-lane holder (the daemon).
_DEFAULT_GPU_LANE_LOCK = "/tmp/hawking-gpu-lane.lock"


def gpu_lane_lock_path() -> Path:
    return Path(os.environ.get("HAWKING_GPU_LANE_LOCK", _DEFAULT_GPU_LANE_LOCK))

# Anything the sidecar could accidentally claim without hardware authority.
HARDWARE_FIELDS = frozenset(
    {
        "tps",
        "accepted_tps",
        "token_ns",
        "complete_token_ns",
        "gpu_ns",
        "joules_per_token",
        "bandwidth_gbps",
        "wall_ns",
        "dispatch_ns",
    }
)


def bench_block(recorded_by: str) -> dict[str, Any]:
    """The only bench state this campaign is allowed to record.

    Claude/Grok have no protected GPU lease, so every receipt produced here is
    STATIC_ONLY with state UNKNOWN. S032-style rule: a budget or a plan is not
    a physical measurement.
    """
    return {
        "state": "UNKNOWN",
        "measurement_state": "STATIC_ONLY",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recorded_by": recorded_by,
        "machine": "Apple host CPU; receipt/header metadata only",
        "gpu_authority": False,
        "rule": "no hardware measurement claim without hardware",
    }


class HardwareClaimError(ValueError):
    """Raised when a sidecar receipt tries to assert a measured hardware value."""


def _assert_no_hardware_claims(node: Any, path: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                raise HardwareClaimError(
                    f"{here} = {value!r}: sidecar has no GPU authority, "
                    f"hardware fields must be null/UNKNOWN"
                )
            _assert_no_hardware_claims(value, here)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _assert_no_hardware_claims(value, f"{path}[{i}]")


def seal(doc: dict[str, Any]) -> dict[str, Any]:
    """Attach a content hash over everything except the hash itself."""
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    doc["seal_sha256"] = hashlib.sha256(blob).hexdigest()
    return doc


# Fields every producer is free to restamp on every run without any new
# evidence existing: a fresh write clock (bench.recorded_at, set by
# bench_block() above) and whatever commit/branch happened to be checked out
# (several producers embed "head"/"branch" via _common.git()). On a repo that
# lands commits constantly, HEAD has moved by the next run even when nothing
# the producer measured did. Diffing on these fields alone turns an unchanged
# receipt into git noise every time its producer is re-run -- G151 hand-fixed
# a batch of these once; this makes the fix permanent at the one place every
# producer writes through.
_BOOKKEEPING_KEYS = frozenset({"bench", "head", "branch", "seal_sha256"})


def _same_but_for_bookkeeping(out: Path, doc: dict[str, Any]) -> bool:
    if not out.is_file():
        return False
    try:
        prior = json.loads(out.read_text())
    except (ValueError, OSError):
        return False
    if not isinstance(prior, dict):
        return False
    strip = lambda d: {k: v for k, v in d.items() if k not in _BOOKKEEPING_KEYS}
    return strip(prior) == strip(doc)


class ReceiptPathCollision(ValueError):
    """Two producers, one path. The later writer would destroy the earlier one."""


def _refuse_foreign_overwrite(out: Path, doc: dict[str, Any], recorded_by: str) -> None:
    """A receipt path belongs to ONE producer.

    tps_budget.py and causal_budget_71.py both wrote
    RESIDENT_71TPS_CAUSAL_BUDGET.json with different schemas. The later writer
    won and silently destroyed every citation resolving against `ladder[]` and
    `measured_now` - four rows of the roof-anchor audit stopped resolving, and
    the audit honestly reported "field is not a resolvable path in this receipt"
    about a field that HAD been resolvable the day it was written. Nothing
    raised. The overwrite is a WRITE, and writes succeed.

    An overwrite by the same producer, or a schema-compatible one, is normal
    regeneration and is allowed. A DIFFERENT producer writing a DIFFERENT schema
    over an existing receipt is the collision, and it raises.
    """
    if not out.is_file():
        return
    try:
        prior = json.loads(out.read_text())
    except (ValueError, OSError):
        return  # unreadable prior is not evidence of ownership
    if not isinstance(prior, dict):
        return
    prior_by = prior.get("recorded_by") or (prior.get("bench") or {}).get("recorded_by")
    if not prior_by or prior_by == recorded_by:
        return
    prior_schema = prior.get("schema")
    new_schema = doc.get("schema")
    if prior_schema is None or new_schema is None or prior_schema == new_schema:
        return
    raise ReceiptPathCollision(
        f"{out.name} was written by {prior_by} with schema {prior_schema!r}; "
        f"{recorded_by} would overwrite it with schema {new_schema!r}. A receipt "
        "path belongs to one producer - give this one its own name rather than "
        "destroying the other's citations."
    )


def write_receipt(name: str, doc: dict[str, Any], recorded_by: str) -> Path:
    """Validate, seal and write a sidecar receipt. Returns its path."""
    doc.setdefault("bench", bench_block(recorded_by))
    doc.setdefault("claim_boundary", "Static sidecar artifact. No hardware measurement.")
    _assert_no_hardware_claims(doc)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    out = RECEIPTS / name
    _refuse_foreign_overwrite(out, doc, recorded_by)
    if _same_but_for_bookkeeping(out, doc):
        return out
    seal(doc)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return out


class MeasurementProvenanceError(ValueError):
    """Raised when a receipt asserts a hardware number without saying when, under
    what load, and whether the GPU lane lock was held."""


def measurement_provenance(
    *,
    lock_held: bool,
    loadavg: str | None = None,
    lane: str | None = None,
    measured_at: str | None = None,
    retrofit: bool = False,
) -> dict[str, Any]:
    """Provenance every hardware number needs and none of them carried.

    write_receipt REFUSES hardware fields, so measurement receipts hand-rolled
    their own json.dumps and inherited no bench block at all. The consequence
    showed up the day /tmp/hawking-gpu-lane.lock was found wedged as a stale file:
    placing the four headline measurements against that contention window had to
    be done from GIT LANDING TIMES, because none of them recorded when it ran.
    Landing time is a proxy - a receipt can be produced long before it lands - and
    that margin happened to be hours. Contention windows are discovered after the
    fact, so the stamp has to exist before anyone knows they need it.
    """
    if loadavg is None:
        try:
            one, five, fifteen = os.getloadavg()
            loadavg = f"{{ {one:.2f} {five:.2f} {fifteen:.2f} }}"
        except (OSError, AttributeError):
            loadavg = None
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if retrofit and measured_at is None:
        # A receipt REGENERATED from a stored raw measurement must not stamp the
        # regeneration time as the measurement time. That would be a fabricated
        # provenance - precisely the thing this block exists to prevent - and it
        # is an easy mistake because the writer happens to be running now.
        return {
            "measured_at": None,
            "measured_at_source": "RETROFIT_UNKNOWN",
            "receipt_regenerated_at": now,
            "gpu_lane_lock_held": None,
            "loadavg": loadavg,
            "lane": lane,
            "absolutes_are_measured_under_load": True,
            "why": (
                "this receipt predates measurement provenance; the raw capture "
                "carries no timestamp, so the measurement time is genuinely "
                "unknown and is recorded as unknown rather than invented"
            ),
        }
    return {
        "measured_at": measured_at or now,
        "measured_at_source": "CAPTURED" if measured_at else "WRITE_TIME",
        "gpu_lane_lock_held": bool(lock_held),
        "loadavg": loadavg,
        "lane": lane,
        "absolutes_are_measured_under_load": True,
        "why": (
            "a hardware number that cannot be placed in time cannot be audited "
            "against a contention window"
        ),
    }


REQUIRED_PROVENANCE = ("measured_at", "gpu_lane_lock_held", "loadavg")


def _has_hardware_number(node: Any) -> bool:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                return True
            if _has_hardware_number(value):
                return True
    elif isinstance(node, list):
        return any(_has_hardware_number(v) for v in node)
    return False


def write_measured_receipt(
    path: str | Path,
    doc: dict[str, Any],
    recorded_by: str,
    *,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write a receipt that IS allowed to carry hardware numbers - and therefore
    must say when it measured them, under what load, and whether it held the lane
    lock. The sidecar writer refuses hardware; this one requires provenance for it.
    """
    doc.setdefault("recorded_by", recorded_by)
    if provenance is not None:
        doc["measurement_provenance"] = provenance
    prov = doc.get("measurement_provenance")
    if _has_hardware_number(doc):
        if not isinstance(prov, dict):
            raise MeasurementProvenanceError(
                f"{recorded_by}: receipt carries a hardware number and no "
                f"measurement_provenance block; use measurement_provenance()"
            )
        missing = [k for k in REQUIRED_PROVENANCE if k not in prov]
        if missing:
            raise MeasurementProvenanceError(
                f"{recorded_by}: measurement_provenance is missing {missing}; "
                f"a hardware number that cannot be placed in time cannot be "
                f"audited against a contention window"
            )
        if not prov.get("measured_at") and prov.get("measured_at_source") != "RETROFIT_UNKNOWN":
            raise MeasurementProvenanceError(
                f"{recorded_by}: measured_at is empty; landing time is a proxy, "
                f"not a measurement time"
            )
    seal(doc)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return out


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


# Seconds before a read-only git query is abandoned. The tree is ~43GB and dirty,
# so `git status` can run for minutes; a query with no timeout is a query that
# can hang a caller forever.
GIT_TIMEOUT_S = 120

# Per-(REPO, argv) cache of read-only queries. Keyed by REPO so a test that
# points REPO at a tmpdir cannot inherit the live tree's ls-tree. Status/diff
# are not cached: those are the queries whose answer moves when a test writes.
_GIT_CACHE: dict[tuple[str, tuple[str, ...]], str] = {}
_HEAD_PATHS: dict[str, tuple[str, ...]] = {}
_HEAD_PATH_SET: dict[str, frozenset[str]] = {}
# (REPO, "commit:rel") -> blob text. Missing blobs are "".
_BLOB_CACHE: dict[tuple[str, str], str] = {}

_UNCACHED_GIT = frozenset({
    "status", "diff", "add", "commit", "apply", "update-index",
    "stash", "reset", "checkout", "merge", "rebase", "worktree",
})


def clear_git_cache() -> None:
    """Drop every memo. Tests that re-point REPO at a tmpdir do not need this
    (the key includes REPO); tests that rewrite HEAD in place do."""
    _GIT_CACHE.clear()
    _HEAD_PATHS.clear()
    _HEAD_PATH_SET.clear()
    _BLOB_CACHE.clear()


def _git_uncached(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=REPO, capture_output=True, text=True, check=False,
            timeout=GIT_TIMEOUT_S,
        ).stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return ""


def _parse_cat_file_batch(data: bytes, specs: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    idx = 0
    for spec in specs:
        if idx >= len(data):
            out[spec] = ""
            continue
        nl = data.find(b"\n", idx)
        if nl < 0:
            out[spec] = ""
            break
        header = data[idx:nl].decode("utf-8", errors="replace")
        idx = nl + 1
        if " missing" in header:
            out[spec] = ""
            continue
        parts = header.split()
        if len(parts) < 3 or parts[1] != "blob":
            out[spec] = ""
            continue
        try:
            size = int(parts[2])
        except ValueError:
            out[spec] = ""
            continue
        blob = data[idx : idx + size]
        idx = idx + size
        if idx < len(data) and data[idx : idx + 1] == b"\n":
            idx += 1
        out[spec] = blob.decode("utf-8", errors="replace")
    return out


def prefetch_blobs(specs: Sequence[str]) -> None:
    """One `git cat-file --batch` for many `commit:rel` specs. Missing -> ''."""
    repo = str(REPO)
    pending: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if not spec or spec in seen:
            continue
        seen.add(spec)
        if (repo, spec) not in _BLOB_CACHE:
            pending.append(spec)
    if not pending:
        return
    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "cat-file", "--batch"],
            cwd=str(REPO),
            input=("\n".join(pending) + "\n").encode("utf-8"),
            capture_output=True,
            timeout=GIT_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError):
        for spec in pending:
            _BLOB_CACHE.setdefault((repo, spec), "")
        return
    parsed = _parse_cat_file_batch(proc.stdout or b"", pending)
    for spec, text in parsed.items():
        _BLOB_CACHE[(repo, spec)] = text


def blob_text(spec: str) -> str:
    """`git show <spec>` via the blob cache. Empty string if missing."""
    if not spec:
        return ""
    key = (str(REPO), spec)
    cached = _BLOB_CACHE.get(key)
    if cached is not None:
        return cached
    prefetch_blobs([spec])
    return _BLOB_CACHE.get(key, "")


def head_paths_ordered() -> tuple[str, ...]:
    """Every path in HEAD, git-ls-tree order. One subprocess per REPO."""
    repo = str(REPO)
    cached = _HEAD_PATHS.get(repo)
    if cached is not None:
        return cached
    # Uncached: git() would call _ls_tree_from_head which calls us.
    raw = _git_uncached("ls-tree", "-r", "--name-only", "HEAD")
    ordered = tuple(p for p in raw.splitlines() if p)
    _HEAD_PATHS[repo] = ordered
    _HEAD_PATH_SET[repo] = frozenset(ordered)
    _GIT_CACHE[(repo, ("ls-tree", "-r", "--name-only", "HEAD"))] = raw
    return ordered


def head_path_set() -> frozenset[str]:
    repo = str(REPO)
    cached = _HEAD_PATH_SET.get(repo)
    if cached is not None:
        return cached
    head_paths_ordered()
    return _HEAD_PATH_SET.get(repo, frozenset())


def _ls_tree_from_head(args: tuple[str, ...]) -> str | None:
    """Answer a HEAD `--name-only` ls-tree from the cached path list.

    Returns None when the argv is not a HEAD name-only listing we can
    reconstruct (other revisions, unknown flags).
    """
    if "ls-tree" not in args or "--name-only" not in args or "HEAD" not in args:
        return None
    recursive = False
    pathspecs: list[str] = []
    for a in args:
        if a in {"ls-tree", "--name-only", "--", "HEAD"}:
            continue
        if a == "-r":
            recursive = True
            continue
        if a.startswith("-"):
            return None
        pathspecs.append(a)
    ordered = head_paths_ordered()
    if not pathspecs:
        return "\n".join(ordered)
    hp = head_path_set()
    out: list[str] = []
    seen: set[str] = set()
    for spec in pathspecs:
        if spec in hp:
            if spec not in seen:
                out.append(spec)
                seen.add(spec)
            continue
        prefix = spec.rstrip("/") + "/"
        if recursive:
            for p in ordered:
                if p.startswith(prefix) and p not in seen:
                    out.append(p)
                    seen.add(p)
        else:
            # Immediate children only, matching `git ls-tree --name-only HEAD dir`.
            for p in ordered:
                if not p.startswith(prefix):
                    continue
                rest = p[len(prefix):]
                child = prefix + rest.split("/", 1)[0]
                if child not in seen:
                    out.append(child)
                    seen.add(child)
    return "\n".join(out)


def prefetch_session_artifacts() -> None:
    """One ls-tree + one cat-file batch of receipts/future. Call from conftest.

    This worktree is sparse: receipts/future is in HEAD and not on disk, so
    every producer that `git show`s a receipt would otherwise spawn a
    subprocess per file. Building the blob cache once is the same evidence
    (HEAD bytes), not a fixture standing in for a run.
    """
    ordered = head_paths_ordered()
    specs = [f"HEAD:{p}" for p in ordered if p.startswith("receipts/future/")]
    prefetch_blobs(specs)


def git(*args: str) -> str:
    """A READ-ONLY git query that cannot take, or strand, the index lock.

    Every caller in this package reads: show, ls-tree, rev-parse, status,
    worktree list. `git status` refreshes and therefore WRITES .git/index.lock
    on a tree this size, and a git killed while holding it leaves a stale lock
    that blocks every later commit in the repo -- which has happened repeatedly
    here, each time with a lock several minutes old and no process holding it.

    --no-optional-locks tells git not to take that lock for a query that does
    not need it. The timeout stops a slow query becoming a hung caller. A
    timeout returns empty, which every caller already treats as "not found".

    Read-only answers are memoized per (REPO, argv) for the process. The
    suite was spawning one `git show` / `git ls-tree` per producer per test
    against the same HEAD; the bytes do not change mid-session.
    """
    if not args:
        return ""
    repo = str(REPO)
    key = (repo, args)
    cached = _GIT_CACHE.get(key)
    if cached is not None:
        return cached

    if args[0] == "show" and len(args) == 2 and ":" in args[1]:
        result = blob_text(args[1])
        _GIT_CACHE[key] = result
        return result

    if args[0] == "ls-tree":
        rebuilt = _ls_tree_from_head(args)
        if rebuilt is not None:
            _GIT_CACHE[key] = rebuilt
            return rebuilt

    result = _git_uncached(*args)
    if args[0] not in _UNCACHED_GIT:
        _GIT_CACHE[key] = result
    return result


_SWIFT_BIN: dict[tuple[str, tuple[str, ...]], Path] = {}


def compile_swift(
    source: str,
    extra_args: Sequence[str] = ("-O", "-framework", "CoreML"),
) -> tuple[Path | None, str]:
    """Compile a Swift snippet once per (source, flags) this process.

    ANE probe lab and the preboard inspector were each `xcrun swiftc`ing the
    same enumerator on every build() (~0.9s). The binary is a function of the
    source text; a mutated .swift string is a new key.
    """
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    key = (digest, tuple(extra_args))
    cached = _SWIFT_BIN.get(key)
    if cached is not None and cached.is_file():
        return cached, ""
    tmp = Path(tempfile.mkdtemp(prefix="hawking-swiftc-"))
    src = tmp / "a.swift"
    binary = tmp / "a"
    src.write_text(source)
    try:
        proc = subprocess.run(
            ["xcrun", "swiftc", *extra_args, "-o", str(binary), str(src)],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0 or not binary.is_file():
        err = (proc.stderr or proc.stdout or "swiftc produced no binary").strip()
        return None, err
    _SWIFT_BIN[key] = binary
    return binary, ""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def newest_mtime(root: Path, skip: tuple[str, ...] = ()) -> tuple[float, str | None]:
    """Newest mtime under root, and which file it was. (0.0, None) if empty."""
    best, who = 0.0, None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", "target"}]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if any(s in p for s in skip):
                continue
            try:
                m = os.path.getmtime(p)
            except OSError:
                continue
            if m > best:
                best, who = m, os.path.relpath(p, REPO)
    return best, who


class UnknownFlag(SystemExit):
    """A CLI was handed a flag it does not implement."""


def require_known_flags(known: "Iterable[str]", argv: "Sequence[str] | None" = None) -> None:
    """Refuse an unrecognised flag instead of ignoring it.

    Modules that dispatch with `if "--record" in sys.argv` treat every other
    argument as absent. So `--build` - the verb most of tools/future uses -
    printed a freshly computed table, exited 0, and WROTE NOTHING. The terminal
    showed current numbers while the receipt on disk stayed stale, and that cost
    two silently-stale receipts before it was noticed (path_to_71,
    causal_budget_71).

    A tool that reports success without doing the work is the failure this
    campaign keeps finding in its own checks. Call this first in __main__.
    """
    import sys as _sys
    args = list(argv if argv is not None else _sys.argv[1:])
    ok = set(known)
    bad = [a for a in args if a.startswith("-") and a.split("=", 1)[0] not in ok]
    if bad:
        raise UnknownFlag(
            f"unknown flag(s) {bad}; known flags are {sorted(ok)}. Refusing "
            "rather than running with the argument silently ignored."
        )
