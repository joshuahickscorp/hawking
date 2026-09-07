"""G002: dense anatomy across every reachable DENSE body in the ModelLake.

Walks klass==DENSE bodies smallest-first. For each one: estimate the float32
working set FROM HEADERS (largest 2D organ tensor plus a same-shape null),
ask campaign_memory_guard.sample(expected_gb=...), and if the guard says STOP
record a named GUARD_STOP and move on. Anatomy is bracketed in resource_cost.
The receipt is rewritten after every body. --resume skips slugs already present.

A deficit is a comparison against a matched null; organ ranking goes through
dense_anatomy.resolved_ordering so pairs the instrument cannot separate are
tied. Absence is a named refusal, never an empty anatomy.

One body at a time. Each body runs in a child process so ru_maxrss is a
per-body reading (the kernel counter is a latch and cannot go back down).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np

import campaign_memory_guard as cmg
import dense_anatomy as da
from lake_scheme_census import FLOAT_DTYPES

GIB = 1 << 30
PEAK_RSS_CAP_GIB = 20.0
LAKE_DEFAULT = "/Volumes/corpdrive/hawking-modellake/specimens"
CENSUS_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "receipts", "future", "G034_LAKE_CENSUS_FULL.json",
)
RECEIPT_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "receipts", "future", "G002_DENSE_ANATOMY_SWEEP.json",
)
ROW_KEYS = (
    "slug", "gib", "family", "status", "wall_s", "peak_rss_gib",
    "swapfile_delta", "organ_ordering", "cross_layer", "refusal",
)
STATUSES = ("ANATOMY", "REFUSED", "GUARD_STOP")

_RUNNING = False  # in-process mutex: two anatomies in one process is a defect


def _refuse_volumes_write(path: str) -> None:
    """The lake is read-only for this driver. A write under /Volumes is a bug."""
    abs_path = os.path.abspath(path)
    if abs_path == "/Volumes" or abs_path.startswith("/Volumes/"):
        raise RuntimeError(f"refusing to write under /Volumes: {abs_path}")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if x is None or isinstance(x, (str, int)):
        return x
    return str(x)


def _atomic_write(path: str, doc: dict) -> None:
    _refuse_volumes_write(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = json.dumps(_jsonable(doc), indent=1) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".g002-", suffix=".tmp",
                               dir=os.path.dirname(os.path.abspath(path)) or ".")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _blank_row(slug: str, gib: float, family: str) -> dict:
    return {
        "slug": slug,
        "gib": gib,
        "family": family,
        "status": None,
        "wall_s": 0.0,
        "peak_rss_gib": 0.0,
        "swapfile_delta": 0,
        "organ_ordering": None,
        "cross_layer": None,
        "refusal": None,
    }


def _validate_row(row: dict) -> None:
    missing = [k for k in ROW_KEYS if k not in row]
    if missing:
        raise RuntimeError(f"row missing keys {missing}: {row.get('slug')}")
    if row["status"] not in STATUSES:
        raise RuntimeError(f"{row.get('slug')}: status {row['status']!r} is not a named result")
    if row["status"] == "ANATOMY":
        if not row.get("organ_ordering"):
            raise RuntimeError(
                f"{row['slug']}: ANATOMY with empty organ_ordering -- absence must be a refusal"
            )
        if row.get("refusal"):
            raise RuntimeError(f"{row['slug']}: ANATOMY row also carries a refusal")
    else:
        if not row.get("refusal") or not str(row["refusal"]).strip():
            raise RuntimeError(
                f"{row['slug']}: {row['status']} without a named mechanism"
            )


# ---------------------------------------------------------------------------
# header-only working-set estimate
# ---------------------------------------------------------------------------

def estimate_from_headers(snapshot: str) -> dict:
    """Largest 2D float organ as float32, plus a same-shape null. Headers only.

    On-disk BF16/F16 is decoded to float32 by the anatomy path, so the estimate
    uses 4 bytes/elem regardless of the stored dtype. Non-float organs are
    skipped: the anatomy refuses those before any payload is loaded.
    """
    out = {
        "expected_gb": 0.0,
        "largest": None,
        "unavailable": None,
        "n_float_organs": 0,
    }
    try:
        _shards, index, _offsets = da._index_snapshot(snapshot)
    except da.DenseAnatomyUnavailable as exc:
        out["unavailable"] = str(exc)
        return out
    max_elems = 0
    max_shape = None
    max_key = None
    n_float = 0
    for store_k, (path, entry) in index.items():
        if entry.get("dtype") not in FLOAT_DTYPES:
            continue
        raw_key = store_k.split("::", 1)[-1]
        parsed = da._parse_organ(raw_key, da._namespace(snapshot, path))
        shape = entry.get("shape") or []
        if not parsed or len(shape) != 2:
            continue
        n_float += 1
        m, n = int(shape[0]), int(shape[1])
        elems = m * n
        if elems > max_elems:
            max_elems, max_shape, max_key = elems, [m, n], raw_key
    out["n_float_organs"] = n_float
    if max_elems == 0:
        return out
    tensor_bytes = max_elems * 4
    expected_gb = (2.0 * tensor_bytes) / GIB  # tensor + null, as specified
    out["expected_gb"] = expected_gb
    out["largest"] = {
        "key": max_key,
        "shape": max_shape,
        "tensor_f32_gib": tensor_bytes / GIB,
        "null_f32_gib": tensor_bytes / GIB,
    }
    return out


# ---------------------------------------------------------------------------
# row construction -- resolved_ordering is the ranking, not raw sort
# ---------------------------------------------------------------------------

def _slim_cross(cross: list) -> list:
    keep = ("organ", "deficit_pct", "verdict", "n", "d", "participation",
            "null_participation", "ratio_n", "null_ratio_n", "subsampled_features")
    return [{k: r[k] for k in keep if k in r} for r in (cross or [])]


def organ_ordering_from_anatomy(anatomy: dict) -> list:
    """Mid-depth ranking with ties the null cannot resolve.

    Using sorted(deficit) here is the Wan2.2 error: 0.04 pp and 0.05 pp gaps
    were published as ranks. resolved_ordering is the only legal ranking.
    """
    within = anatomy.get("within_tensor") or []
    if not within:
        raise da.DenseAnatomyUnavailable(
            f"{anatomy.get('snapshot', '<unknown>')}: anatomy returned no within_tensor "
            "organs -- refusing to record an empty ranking"
        )
    mids = da._mid_rows(within)
    if not mids:
        raise da.DenseAnatomyUnavailable(
            f"{anatomy.get('snapshot', '<unknown>')}: no mid-depth organ survived"
        )
    triples = []
    for r in mids:
        se = r.get("deficit_stderr_pct")
        se = float(se) if se is not None else float("nan")
        triples.append((r["organ"], float(r["deficit_pct"]), se))
    ordering = da.resolved_ordering(triples)
    if not ordering:
        raise da.DenseAnatomyUnavailable(
            f"{anatomy.get('snapshot', '<unknown>')}: resolved_ordering produced no rows"
        )
    return ordering


def row_from_anatomy(anatomy: dict, cost: dict, peak_rss_gib: float) -> dict:
    hyps = anatomy.get("hypotheses") or []
    if not hyps:
        raise da.DenseAnatomyUnavailable(
            f"{anatomy.get('snapshot', '<unknown>')}: spectra computed but yielded no "
            "hypothesis -- refusing to record an anatomy that says nothing"
        )
    ordering = organ_ordering_from_anatomy(anatomy)
    return {
        "status": "ANATOMY",
        "wall_s": float(cost.get("wall_s") or 0.0),
        "peak_rss_gib": round(float(peak_rss_gib or 0.0), 3),
        "swapfile_delta": int(cost.get("swapfiles_delta") or 0),
        "organ_ordering": ordering,
        "cross_layer": _slim_cross(anatomy.get("cross_layer") or []),
        "refusal": None,
        "hypotheses": list(hyps),
        "n_layers": anatomy.get("n_layers"),
        "organs": anatomy.get("organs"),
        "n_shards": anatomy.get("n_shards"),
        "resource": {k: cost[k] for k in cost if k != "label"} if cost else None,
    }


def _named_refusal(msg: str) -> str:
    text = str(msg or "").strip()
    if not text:
        return ("unnamed refusal: an empty message is how an unmeasured body "
                "disguises itself as a finding")
    return text


def row_refused(msg: str, cost: dict, peak_rss_gib: float, status: str = "REFUSED") -> dict:
    if status not in ("REFUSED", "GUARD_STOP"):
        raise RuntimeError(f"internal: {status} is not a refusal status")
    return {
        "status": status,
        "wall_s": float(cost.get("wall_s") or 0.0),
        "peak_rss_gib": round(float(peak_rss_gib or 0.0), 3),
        "swapfile_delta": int(cost.get("swapfiles_delta") or 0),
        "organ_ordering": None,
        "cross_layer": None,
        "refusal": _named_refusal(msg),
        "resource": {k: cost[k] for k in cost if k != "label"} if cost else None,
    }


def _guard_stop_message(est: dict, snap: cmg.Snapshot) -> str:
    largest = est.get("largest") or {}
    shape = largest.get("shape")
    reasons = "; ".join(snap.reasons) if snap.reasons else "no reason string"
    return (
        f"campaign guard STOP: expected {est.get('expected_gb', 0.0):.3f} GiB "
        f"(largest organ {shape} float32 + same-shape null"
        + (f", key {largest.get('key')}" if largest.get("key") else "")
        + f"); free {snap.free_gb:.2f} GB, headroom {snap.headroom_gb:.2f} GB, "
        f"compressor {snap.compressor_gb:.2f} GB, swapfiles {snap.swapfiles}, "
        f"wired {snap.wired_gb:.2f} GB. reasons: {reasons}"
    )


# ---------------------------------------------------------------------------
# one body
# ---------------------------------------------------------------------------

def _measure_anatomy(snapshot: str) -> dict:
    """Run anatomy_from_safetensors inside resource_cost. Never returns empty."""
    global _RUNNING
    if _RUNNING:
        raise RuntimeError(
            "refusing to run two dense anatomies concurrently -- one body at a time"
        )
    _RUNNING = True
    anatomy = None
    err = None
    try:
        with cmg.resource_cost(interval_s=2.0, label=os.path.basename(snapshot)) as cost:
            try:
                anatomy = da.anatomy_from_safetensors(snapshot)
            except da.DenseAnatomyUnavailable as exc:
                err = str(exc)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        peak = da.peak_rss_bytes() / GIB
        if anatomy is not None:
            try:
                return row_from_anatomy(anatomy, cost, peak)
            except da.DenseAnatomyUnavailable as exc:
                return row_refused(str(exc), cost, peak)
        return row_refused(err, cost, peak)
    finally:
        _RUNNING = False
        del anatomy


def _run_one_body_child(snapshot: str, row_out: str) -> None:
    _refuse_volumes_write(row_out)
    row = _measure_anatomy(snapshot)
    _atomic_write(row_out, row)
    status = row["status"]
    extra = ""
    if status == "ANATOMY" and row.get("organ_ordering"):
        top = row["organ_ordering"][0]
        extra = f"  top={top['organ']} {top['deficit_pct']:.2f}%"
    elif row.get("refusal"):
        extra = "  " + str(row["refusal"])[:160]
    sys.stderr.write(
        f"one-body {os.path.basename(snapshot)}  {status}  "
        f"wall={row['wall_s']:.1f}s  peak_rss={row['peak_rss_gib']:.2f}GiB{extra}\n"
    )


def _run_isolated(snapshot: str, executable: str | None = None) -> dict:
    """Child process so ru_maxrss is this body's peak, not a process-lifetime latch."""
    exe = executable or sys.executable
    with tempfile.TemporaryDirectory(prefix="g002-body-") as td:
        row_out = os.path.join(td, "row.json")
        cmd = [exe, os.path.abspath(__file__), "--one-body", snapshot, "--row-out", row_out]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        here = os.path.dirname(os.path.abspath(__file__))
        env["PYTHONPATH"] = here + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        t0 = time.time()
        proc = subprocess.Popen(cmd, env=env)
        last_hb = t0
        while True:
            try:
                rc = proc.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                now = time.time()
                if now - last_hb >= 60.0:
                    sys.stderr.write(
                        f"  ... still running {os.path.basename(snapshot)}  "
                        f"elapsed={now - t0:.0f}s\n"
                    )
                    sys.stderr.flush()
                    last_hb = now
        if os.path.isfile(row_out):
            with open(row_out) as fh:
                row = json.load(fh)
            if rc != 0 and row.get("status") == "ANATOMY":
                # child wrote a result then died; keep the result, note the exit
                row.setdefault("child_exit", rc)
            return row
        if rc < 0:
            msg = (f"child killed by signal {-rc} while anatomising {snapshot} "
                   f"(possible OOM); no row was written")
        else:
            msg = (f"child exited {rc} while anatomising {snapshot} "
                   f"without writing a row")
        return row_refused(msg, {"wall_s": round(time.time() - t0, 2),
                                 "swapfiles_delta": 0}, 0.0)


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------

def _tally(rows: list) -> dict:
    t = {s: 0 for s in STATUSES}
    for r in rows:
        st = r.get("status")
        if st in t:
            t[st] += 1
        else:
            t.setdefault("OTHER", 0)
            t["OTHER"] += 1
    t["n_rows"] = len(rows)
    return t


def _new_receipt(*, lake: str, census_path: str, n_dense: int,
                 dense_slugs: list, extra: dict | None = None) -> dict:
    doc = {
        "obligation": "G002",
        "what": "dense anatomy of every reachable DENSE body, ranked by resolved_ordering",
        "method": (
            "For each klass==DENSE body, smallest first: estimate float32(largest 2D organ) "
            "+ same-shape null from safetensors headers; campaign_memory_guard.sample("
            "expected_gb=that); GUARD_STOP is a named row and does not abort. Anatomy is "
            "bracketed in resource_cost. Organ ranking is dense_anatomy.resolved_ordering "
            "on mid-depth (name, deficit_pct, deficit_stderr_pct) -- pairs the null cannot "
            "separate share a rank. One body at a time, one child process per body so "
            "ru_maxrss is not a latch. Receipt rewritten after every body."
        ),
        "tool": "tools/future/dense_sweep.py",
        "census": census_path,
        "lake": lake,
        "n_dense": n_dense,
        "dense_slugs": list(dense_slugs),
        "tally": _tally([]),
        "rows": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at": None,
        "peak_rss_cap_gib": PEAK_RSS_CAP_GIB,
    }
    if extra:
        doc.update(extra)
    return doc


def _sync_pending(doc: dict, targets: list[dict]) -> None:
    have = {r["slug"] for r in doc.get("rows") or []}
    doc["pending_slugs"] = [t["slug"] for t in targets if t["slug"] not in have]


def _commit(receipt_path: str, doc: dict) -> None:
    doc["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc["tally"] = _tally(doc.get("rows") or [])
    _atomic_write(receipt_path, doc)


class ReceiptLock:
    def __init__(self, receipt_path: str):
        self.receipt_path = receipt_path
        self.lock_path = receipt_path + ".lock"
        self.fd = None

    def __enter__(self):
        _refuse_volumes_write(self.lock_path)
        os.makedirs(os.path.dirname(os.path.abspath(self.lock_path)) or ".", exist_ok=True)
        self.fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError(
                f"another dense_sweep holds {self.lock_path}; two sweeps would run "
                "bodies concurrently"
            ) from exc
        os.write(self.fd, f"{os.getpid()}\n".encode())
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None
        try:
            os.unlink(self.lock_path)
        except OSError:
            pass


def load_dense_targets(census_path: str, lake: str) -> tuple[list[dict], dict]:
    with open(census_path) as fh:
        census = json.load(fh)
    rows = census.get("rows") or []
    dense = [r for r in rows if r.get("klass") == "DENSE"]
    dense.sort(key=lambda r: (float(r.get("gib") or 0.0), r.get("slug") or ""))
    tally = census.get("tally") or {}
    meta = {
        "census_n": census.get("n"),
        "census_tally": tally,
        "dense_count": len(dense),
        "census_dense_tally": tally.get("DENSE"),
    }
    targets = []
    for r in dense:
        slug = r["slug"]
        path = r.get("path") or os.path.join(lake, slug)
        targets.append({
            "slug": slug,
            "gib": float(r.get("gib") or 0.0),
            "family": r.get("family") or "UNKNOWN",
            "path": path,
            "klass": r.get("klass"),
        })
    return targets, meta


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def sweep(
    *,
    lake: str = LAKE_DEFAULT,
    census_path: str = CENSUS_DEFAULT,
    receipt_path: str = RECEIPT_DEFAULT,
    resume: bool = True,
    isolate: bool = True,
    only: list[str] | None = None,
    guard_sample: Callable[..., cmg.Snapshot] | None = None,
    executable: str | None = None,
) -> dict:
    sample_fn = guard_sample or cmg.sample
    targets, meta = load_dense_targets(census_path, lake)
    if only:
        want = set(only)
        targets = [t for t in targets if t["slug"] in want]
        missing = want - {t["slug"] for t in targets}
        if missing:
            raise RuntimeError(f"--only slugs not in DENSE census: {sorted(missing)}")

    with ReceiptLock(receipt_path):
        done: dict[str, dict] = {}
        if resume and os.path.isfile(receipt_path):
            with open(receipt_path) as fh:
                doc = json.load(fh)
            if not isinstance(doc, dict) or "rows" not in doc:
                raise RuntimeError(f"{receipt_path}: not a sweep receipt (missing rows)")
            for row in doc.get("rows") or []:
                if row.get("slug") and row.get("status") in STATUSES:
                    try:
                        _validate_row(row)
                    except RuntimeError:
                        continue  # incomplete row: redo
                    done[row["slug"]] = row
            doc["rows"] = [done[t["slug"]] for t in targets if t["slug"] in done]
            doc["n_dense"] = meta["dense_count"]
            doc["dense_slugs"] = [t["slug"] for t in load_dense_targets(census_path, lake)[0]]
        else:
            doc = _new_receipt(
                lake=lake, census_path=census_path,
                n_dense=meta["dense_count"],
                dense_slugs=[t["slug"] for t in load_dense_targets(census_path, lake)[0]],
                extra={"census_meta": meta},
            )

        doc["census_meta"] = meta
        if meta["census_dense_tally"] not in (None, meta["dense_count"]):
            doc["dense_count_mismatch"] = (
                f"census tally DENSE={meta['census_dense_tally']} but "
                f"len(klass==DENSE rows)={meta['dense_count']}"
            )

        pending = [t for t in targets if t["slug"] not in done]
        _sync_pending(doc, targets)
        if pending:
            doc.pop("finished_at", None)
        _commit(receipt_path, doc)

        for t in pending:
            slug, path = t["slug"], t["path"]
            row = _blank_row(slug, t["gib"], t["family"])
            sys.stderr.write(
                f"\n== {slug}  {t['gib']:.1f} GiB  {t['family']} ==\n"
            )
            sys.stderr.flush()

            t_est = time.time()
            if not os.path.isdir(path):
                est = {"expected_gb": 0.0, "largest": None,
                       "unavailable": f"{path}: snapshot directory absent"}
            else:
                try:
                    est = estimate_from_headers(path)
                except Exception as exc:
                    est = {"expected_gb": 0.0, "largest": None,
                           "unavailable": f"{type(exc).__name__}: {exc}"}
            expected_gb = float(est.get("expected_gb") or 0.0)
            row["expected_gb"] = expected_gb
            row["estimate"] = est

            # RSS cap is a hard rule, independent of host free memory.
            if expected_gb > PEAK_RSS_CAP_GIB:
                snap = sample_fn(expected_gb=expected_gb)
                row.update(row_refused(
                    f"header estimate {expected_gb:.3f} GiB exceeds the "
                    f"{PEAK_RSS_CAP_GIB:.0f} GiB peak-RSS cap "
                    f"(largest organ { (est.get('largest') or {}).get('shape') } "
                    f"float32 + null); guard state was {snap.state} "
                    f"free={snap.free_gb:.2f} GB headroom={snap.headroom_gb:.2f} GB",
                    {"wall_s": round(time.time() - t_est, 2), "swapfiles_delta": 0},
                    0.0,
                    status="GUARD_STOP",
                ))
                row["guard"] = snap.as_dict()
                _validate_row(row)
                doc["rows"].append(row)
                _sync_pending(doc, targets)
                _commit(receipt_path, doc)
                sys.stderr.write(f"  GUARD_STOP  {row['refusal'][:200]}\n")
                continue

            snap = sample_fn(expected_gb=expected_gb)
            row["guard"] = snap.as_dict()
            if snap.state == "STOP":
                row.update(row_refused(
                    _guard_stop_message(est, snap),
                    {"wall_s": round(time.time() - t_est, 2), "swapfiles_delta": 0},
                    0.0,
                    status="GUARD_STOP",
                ))
                _validate_row(row)
                doc["rows"].append(row)
                _sync_pending(doc, targets)
                _commit(receipt_path, doc)
                sys.stderr.write(f"  GUARD_STOP  {row['refusal'][:200]}\n")
                continue

            try:
                if isolate:
                    measured = _run_isolated(path, executable=executable)
                else:
                    measured = _measure_anatomy(path)
            except Exception as exc:
                measured = row_refused(
                    f"{type(exc).__name__}: {exc}",
                    {"wall_s": round(time.time() - t_est, 2), "swapfiles_delta": 0},
                    0.0,
                )

            for k in ("status", "wall_s", "peak_rss_gib", "swapfile_delta",
                      "organ_ordering", "cross_layer", "refusal"):
                if k in measured:
                    row[k] = measured[k]
            for extra_k in ("hypotheses", "n_layers", "organs", "n_shards",
                            "resource", "child_exit"):
                if extra_k in measured:
                    row[extra_k] = measured[extra_k]

            # A child that somehow returned an empty ANATOMY is rewritten as a refusal.
            try:
                _validate_row(row)
            except RuntimeError as exc:
                row.update(row_refused(str(exc), {
                    "wall_s": row.get("wall_s") or 0.0,
                    "swapfiles_delta": row.get("swapfile_delta") or 0,
                }, row.get("peak_rss_gib") or 0.0))
                _validate_row(row)

            if row["status"] == "ANATOMY" and (row.get("peak_rss_gib") or 0) > PEAK_RSS_CAP_GIB:
                row["rss_cap_breach"] = (
                    f"peak RSS {row['peak_rss_gib']:.2f} GiB exceeded "
                    f"{PEAK_RSS_CAP_GIB:.0f} GiB -- the streaming path buffered a payload"
                )

            doc["rows"].append(row)
            _sync_pending(doc, targets)
            _commit(receipt_path, doc)

            top = ""
            if row["status"] == "ANATOMY" and row.get("organ_ordering"):
                o = row["organ_ordering"][0]
                top = f"  top={o['organ']} {o['deficit_pct']:.2f}% rank={o['rank']}"
            sys.stderr.write(
                f"  {row['status']:11s}  wall={row['wall_s']:.1f}s  "
                f"peak_rss={row['peak_rss_gib']:.2f}GiB  "
                f"swapΔ={row['swapfile_delta']}{top}\n"
            )
            if row.get("refusal"):
                sys.stderr.write(f"    {row['refusal'][:240]}\n")
            sys.stderr.flush()

        doc["pending_slugs"] = []
        doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _commit(receipt_path, doc)
        return doc


# ---------------------------------------------------------------------------
# selfcheck
# ---------------------------------------------------------------------------

def _selfcheck() -> None:
    rng = np.random.default_rng(0)

    # Volumes writes are refused -- the lake is read-only.
    try:
        _refuse_volumes_write("/Volumes/corpdrive/hawking-modellake/specimens/x.json")
        raise AssertionError("wrote under /Volumes")
    except RuntimeError as exc:
        assert "/Volumes" in str(exc)

    # Header estimate uses float32 bytes, not the on-disk dtype.
    with tempfile.TemporaryDirectory() as d:
        tensors = {
            f"model.layers.{i}.self_attn.q_proj.weight":
                rng.standard_normal((32, 16), dtype=np.float32).astype(np.float16)
            for i in range(4)
        }
        da._write_safetensors(os.path.join(d, "m.safetensors"), tensors)
        est = estimate_from_headers(d)
        want = 2 * 32 * 16 * 4 / GIB
        assert abs(est["expected_gb"] - want) < 1e-12, (est, want)
        assert est["largest"]["shape"] == [32, 16], est

    # resolved_ordering is what the row builder uses -- the Wan2.2 0.04/0.05 pp case.
    fake = {
        "snapshot": "fake",
        "hypotheses": ["structure is ALIVE"],
        "within_tensor": [
            {"organ": "a", "depth": "middle", "deficit_pct": 25.31, "deficit_stderr_pct": 0.045},
            {"organ": "b", "depth": "middle", "deficit_pct": 25.27, "deficit_stderr_pct": 0.045},
            {"organ": "c", "depth": "middle", "deficit_pct": 20.14, "deficit_stderr_pct": 0.045},
            {"organ": "d", "depth": "middle", "deficit_pct": 9.70, "deficit_stderr_pct": 0.045},
            {"organ": "e", "depth": "middle", "deficit_pct": 9.65, "deficit_stderr_pct": 0.045},
        ],
        "cross_layer": [{"organ": "a", "deficit_pct": 0.1, "verdict": "DEAD", "n": 4, "d": 8}],
    }
    ordering = organ_ordering_from_anatomy(fake)
    ranks = {x["organ"]: x["rank"] for x in ordering}
    assert ranks["a"] == ranks["b"], "0.04 pp apart was reported as ordered"
    assert ranks["d"] == ranks["e"], "0.05 pp apart was reported as ordered"
    assert ranks["c"] not in (ranks["a"], ranks["d"])
    built = row_from_anatomy(fake, {"wall_s": 1.2, "swapfiles_delta": 0}, 0.4)
    assert built["status"] == "ANATOMY"
    assert built["organ_ordering"] == ordering
    assert built["wall_s"] == 1.2
    assert built["peak_rss_gib"] == 0.4

    # Empty anatomy is a named refusal, not a well-formed empty row.
    try:
        organ_ordering_from_anatomy({"snapshot": "x", "within_tensor": [], "hypotheses": ["h"]})
        raise AssertionError("empty within_tensor produced a ranking")
    except da.DenseAnatomyUnavailable as exc:
        assert "no within_tensor" in str(exc)

    try:
        row_from_anatomy({"snapshot": "x", "within_tensor": fake["within_tensor"],
                          "hypotheses": [], "cross_layer": []},
                         {"wall_s": 0, "swapfiles_delta": 0}, 0.0)
        raise AssertionError("empty hypotheses produced ANATOMY")
    except da.DenseAnatomyUnavailable as exc:
        assert "no hypothesis" in str(exc) or "says nothing" in str(exc)

    # Concurrent in-process runs are refused.
    global _RUNNING
    _RUNNING = True
    try:
        try:
            _measure_anatomy("/nonexistent")
            raise AssertionError("concurrent anatomy was allowed")
        except RuntimeError as exc:
            assert "concurrently" in str(exc)
    finally:
        _RUNNING = False

    # End-to-end: three synthetic DENSE bodies, guard STOP on the first, resume.
    with tempfile.TemporaryDirectory() as root:
        lake = os.path.join(root, "specimens")
        rec = os.path.join(root, "receipt.json")
        census_p = os.path.join(root, "census.json")
        os.makedirs(lake)

        tiny = os.path.join(lake, "synth--tiny@aaa")
        os.makedirs(tiny)
        B = rng.standard_normal((3, 24), dtype=np.float32)
        tensors = {}
        for i in range(4):
            C = rng.standard_normal((32, 3), dtype=np.float32)
            tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = C @ B
        da._write_safetensors(os.path.join(tiny, "m.safetensors"), tensors)

        empty = os.path.join(lake, "synth--empty@bbb")
        os.makedirs(empty)
        open(os.path.join(empty, "pytorch_model.bin"), "wb").write(b"\0")

        tiny2 = os.path.join(lake, "synth--tiny2@ccc")
        os.makedirs(tiny2)
        da._write_safetensors(os.path.join(tiny2, "m.safetensors"), tensors)

        census = {
            "n": 3,
            "tally": {"DENSE": 3},
            "rows": [
                {"slug": "synth--tiny@aaa", "gib": 0.01, "family": "qwen3", "klass": "DENSE"},
                {"slug": "synth--empty@bbb", "gib": 0.02, "family": "bin", "klass": "DENSE"},
                {"slug": "synth--tiny2@ccc", "gib": 0.03, "family": "qwen3", "klass": "DENSE"},
                {"slug": "moe--skip@ddd", "gib": 9.0, "family": "qwen3_moe", "klass": "MOE-FLOAT"},
            ],
        }
        json.dump(census, open(census_p, "w"))

        n_calls = {"n": 0}

        def fake_sample(expected_gb: float = 0.0) -> cmg.Snapshot:
            n_calls["n"] += 1
            if n_calls["n"] == 1:
                return cmg.Snapshot(
                    3.0, 0.4, 2, 6.0, "STOP",
                    (f"expected {expected_gb:.1f} GB leaves 0.4 GB headroom, below 1.5 GB",),
                    expected_gb=expected_gb, headroom_gb=0.4,
                )
            return cmg.Snapshot(
                40.0, 0.4, 2, 6.0, "OK", (),
                expected_gb=expected_gb, headroom_gb=40.0 - expected_gb,
            )

        # Incremental write: after the first body the receipt must already exist.
        # The first body is GUARD_STOP (fake_sample), which commits before body 2.
        doc = sweep(lake=lake, census_path=census_p, receipt_path=rec,
                    resume=False, isolate=False, guard_sample=fake_sample)
        assert os.path.isfile(rec), "receipt was not written"
        rows = doc["rows"]
        assert len(rows) == 3, [r["slug"] for r in rows]
        assert {r["slug"] for r in rows} == {
            "synth--tiny@aaa", "synth--empty@bbb", "synth--tiny2@ccc"
        }, "MOE-FLOAT leaked into the sweep"
        assert rows[0]["status"] == "GUARD_STOP", rows[0]
        assert "headroom" in rows[0]["refusal"] and "expected" in rows[0]["refusal"], rows[0]["refusal"]
        assert rows[1]["status"] == "REFUSED", rows[1]
        assert "no .safetensors" in rows[1]["refusal"], rows[1]["refusal"]
        assert rows[2]["status"] == "ANATOMY", rows[2]
        assert rows[2]["organ_ordering"], rows[2]
        assert rows[2]["refusal"] is None
        for r in rows:
            _validate_row(r)
        t = _tally(rows)
        assert t["ANATOMY"] + t["REFUSED"] + t["GUARD_STOP"] == 3, t
        assert t["ANATOMY"] == 1 and t["REFUSED"] == 1 and t["GUARD_STOP"] == 1, t

        # Resume skips completed slugs -- fake_sample must not be called again
        # for already-recorded bodies. A fresh sample function that STOPs
        # everything would change outcomes if resume were broken.
        def stop_everything(expected_gb: float = 0.0) -> cmg.Snapshot:
            return cmg.Snapshot(0.1, 60.0, 100, 20.0, "STOP", ("injected",),
                                expected_gb=expected_gb, headroom_gb=-1.0)

        doc2 = sweep(lake=lake, census_path=census_p, receipt_path=rec,
                     resume=True, isolate=False, guard_sample=stop_everything)
        assert len(doc2["rows"]) == 3
        assert [r["status"] for r in doc2["rows"]] == ["GUARD_STOP", "REFUSED", "ANATOMY"]
        # The ANATOMY row must still be an anatomy, not rewritten as GUARD_STOP.
        assert doc2["rows"][2]["organ_ordering"]

        # Isolated child protocol: one tiny body, real subprocess.
        rec3 = os.path.join(root, "receipt-iso.json")
        census_iso = {
            "n": 1, "tally": {"DENSE": 1},
            "rows": [{"slug": "synth--tiny@aaa", "gib": 0.01, "family": "qwen3",
                      "klass": "DENSE"}],
        }
        census_iso_p = os.path.join(root, "census-iso.json")
        json.dump(census_iso, open(census_iso_p, "w"))

        def ok_sample(expected_gb: float = 0.0) -> cmg.Snapshot:
            return cmg.Snapshot(40.0, 0.4, 2, 6.0, "OK", (),
                                expected_gb=expected_gb, headroom_gb=40.0)

        doc3 = sweep(lake=lake, census_path=census_iso_p, receipt_path=rec3,
                     resume=False, isolate=True, guard_sample=ok_sample,
                     executable=sys.executable)
        assert len(doc3["rows"]) == 1, doc3
        assert doc3["rows"][0]["status"] == "ANATOMY", doc3["rows"][0]
        assert doc3["rows"][0]["organ_ordering"]
        assert doc3["rows"][0]["peak_rss_gib"] > 0.0, "child ru_maxrss was zero"

    print("selfcheck OK -- header estimate is float32+null; resolved_ordering ties "
          "the 0.04/0.05 pp pairs; empty anatomy refused; GUARD_STOP does not abort; "
          "receipt is incremental; --resume skips; isolate child returns ANATOMY")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--one-body", metavar="SNAPSHOT",
                   help="child entry: anatomise one snapshot, write --row-out")
    p.add_argument("--row-out", metavar="PATH")
    p.add_argument("--lake", default=LAKE_DEFAULT)
    p.add_argument("--census", default=CENSUS_DEFAULT)
    p.add_argument("--receipt", default=RECEIPT_DEFAULT)
    p.add_argument("--resume", action="store_true", default=True,
                   help="skip slugs already in the receipt (default)")
    p.add_argument("--fresh", action="store_true",
                   help="ignore an existing receipt and start over")
    p.add_argument("--in-process", action="store_true",
                   help="do not isolate bodies in child processes (selfcheck/debug)")
    p.add_argument("--only", action="append", default=None,
                   help="restrict to this slug (repeatable)")
    args = p.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        return 0
    if args.one_body:
        if not args.row_out:
            raise SystemExit("--one-body requires --row-out")
        _run_one_body_child(args.one_body, args.row_out)
        return 0

    if not os.path.isdir(args.lake):
        raise SystemExit(
            f"LAKE ABSENT -- {args.lake} is not a directory. Sweep DID NOT RUN."
        )
    if not os.path.isfile(args.census):
        raise SystemExit(f"census missing: {args.census}")

    doc = sweep(
        lake=args.lake,
        census_path=args.census,
        receipt_path=args.receipt,
        resume=not args.fresh,
        isolate=not args.in_process,
        only=args.only,
    )
    t = doc.get("tally") or {}
    n_dense = doc.get("n_dense")
    print(json.dumps({
        "receipt": args.receipt,
        "n_dense": n_dense,
        "tally": t,
        "sum_matches_dense": (
            (t.get("ANATOMY", 0) + t.get("REFUSED", 0) + t.get("GUARD_STOP", 0)) == n_dense
        ),
        "census_meta": doc.get("census_meta"),
        "dense_count_mismatch": doc.get("dense_count_mismatch"),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
