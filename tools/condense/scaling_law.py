#!/usr/bin/env python3.12
"""scaling_law.py — find each model's bit-floor, then fit the floor-vs-scale curve (plan §4 / T3.1).

Two modes:
  --floor <label> <model_ladder.jsonl> <floors_out.jsonl>
      Read a model's measured results (PTQ + recovered), pick its FLOOR = the lowest EFFECTIVE bpw
      whose degradation is <= the +2% (~1:1) gate, append one datapoint to the floors file.
  --fit <floors.jsonl>
      Regress floor vs log10(params) across the ladder, decide H1 (monotone descent) vs H0 (flat),
      and extrapolate the 70B/405B floor as a PRE-REGISTERED prediction (a result only once an
      off-box run confirms it). Writes a markdown curve report next to the jsonl.

Proof discipline: effective bpw only; 0.5B/1.5B are lab points (printed, but the verdict is read
off 7B+); report honestly whether the floor descends or is flat.
"""
import sys, json, math, os, hashlib, subprocess, tempfile
from floor_integrity import FLOOR_POINT_SCHEMA, canonical_row_sha256, locked_upsert_floor_row
from tripwire_gate import compare as compare_tripwire
from tripwire_gate import policy as tripwire_policy
from tripwire_gate import validate_baseline as validate_tripwire_baseline

GATE = float(os.environ.get("FLOOR_GATE_PCT", "2.0"))      # the ~1:1 quality gate
Q4K_BPW = 4.5                                              # the llama Q4_K reference
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _sha256_bundle(paths):
    digest = hashlib.sha256()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        digest.update(os.path.abspath(path).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _atomic_text(path, text):
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _capture_f16_tripwire(label, jsonl, model_dir):
    """Late baseline fallback for ladders already live when the relative gate was introduced.

    The sidecar avoids mutating the audit JSONL after Studio bound its coverage hash. New ladders
    capture the same evidence inside audit_ladder; this path only keeps an old detached run from
    becoming an orchestration failure after otherwise completing.
    """
    sidecar = f"{jsonl}.f16_tripwire.json"
    expected_label = f"{label}-f16"
    expected_model = os.path.realpath(model_dir)
    try:
        cached = json.load(open(sidecar))
    except Exception:
        cached = None
    if (isinstance(cached, dict) and cached.get("schema") == "hawking.tripwire_baseline.v1"
            and cached.get("model") == label
            and os.path.realpath(str(cached.get("model_dir", ""))) == expected_model
            and isinstance(cached.get("result"), dict)
            and os.path.realpath(str(cached["result"].get("model", ""))) == expected_model
            and validate_tripwire_baseline(cached.get("result"), expected_label)["ok"]):
        return cached
    timeout = int(os.environ.get("TRIPWIRE_TIMEOUT", "14400"))
    cmd = [sys.executable, os.path.join(TOOL_DIR, "multi_eval.py"),
           expected_model, "-", expected_label]
    print(f"[floor] {label}: capturing missing f16 capability baseline", file=sys.stderr)
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=os.environ)
    last = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
    try:
        result = json.loads(last)
    except Exception:
        result = None
    check = validate_tripwire_baseline(result, expected_label)
    if isinstance(result, dict) and result.get("model") != expected_model:
        check["ok"] = False
        check["problems"].append(
            f"baseline model={result.get('model')!r}, expected {expected_model!r}"
        )
    if run.returncode != 0 or not check["ok"]:
        raise RuntimeError(
            f"f16 capability baseline failed rc={run.returncode}: {check['problems']} "
            f"stderr={run.stderr[-200:]}"
        )
    receipt = {"schema": "hawking.tripwire_baseline.v1", "status": "pass",
               "model": label, "model_dir": expected_model, "command": cmd, "result": result}
    _atomic_json(sidecar, receipt)
    return receipt


def _src_hash(model_dir):
    """Stable 64-hex id of a parent: the shard index for sharded models (fast), else model.safetensors."""
    for cand in ("model.safetensors.index.json", "model.safetensors"):
        p = os.path.join(model_dir, cand)
        if os.path.exists(p):
            return _sha256(p)
    return "0" * 64


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _machine():
    try:
        gb = round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                      text=True).stdout) / 1e9)
    except Exception:
        gb = 0
    cls = f"Studio-{gb}" if gb >= 64 else f"M-{gb}"
    return f"Apple Silicon, {gb}GB unified", cls


def _suite_hash():
    for p in ("receipts/prompt_suite_v1.sha256", "prompts/frozen/suite_v1.sha256"):
        if os.path.exists(p):
            return open(p).read().strip().split()[0]
    return None


def emit_receipt(label, rec, model_dir, jsonl):
    """Write a schema-valid floor receipt (receipts/official/<label>-floor.json). 7B+ floor points
    are scale-points (R2); labs are baselines (R1, never set the verdict)."""
    params = PARAMS.get(label, 0)
    machine, mclass = _machine()
    is_lab = params < 7.0
    floor_bpw = rec.get("floor_bpw")
    bpw = floor_bpw or rec.get("best_measured_bpw")
    degr = rec.get("degr_pct") if floor_bpw else rec.get("best_measured_degr_pct")
    config = rec.get("winning_config") if floor_bpw else rec.get("best_measured_config")
    capability_gate = rec.get("tripwire_gate", {})
    gate = "pass" if (floor_bpw and degr is not None and degr <= GATE
                      and capability_gate.get("status") == "pass") else "fail"
    claim_type = "baseline" if is_lab else ("scale-point" if floor_bpw else "negative")
    baseline_evidence = rec.get("tripwire_baseline_evidence") or {}
    evidence_paths = [jsonl, baseline_evidence.get("path")]
    evidence_note = (f"; f16_baseline={baseline_evidence.get('path')} "
                     f"sha256={baseline_evidence.get('sha256')}"
                     if baseline_evidence.get("path") else "")
    r = {
        "project": "hawking", "receipt_version": "0.2",
        "repro_level": "R1" if is_lab else "R2",
        "claim_type": claim_type,
        "machine": machine, "machine_class": mclass,
        "model_family": "qwen", "source_model": f"{label} ({model_dir})",
        "source_sha256": _src_hash(model_dir), "source_precision": "bf16",
        "condensed_artifact": f"{config} @ {bpw} eff-bpw ({jsonl}{evidence_note})",
        "artifact_sha256": _sha256_bundle(evidence_paths),
        "floor_point_sha256": canonical_row_sha256(rec),
        "effective_bpw": float(bpw),
        "nominal_bpw": float(bpw),
        "peak_rss_gb": _peak_rss_gb(label),
        "multiwindow_n": int(os.environ.get("MULTIWINDOW", "4")),
        "quality_gate": gate,
        "hawking_commit": _commit(),
        "commands": [f"python3.12 tools/condense/studio_run.py --model {label}",
                     f"python3.12 tools/condense/scaling_law.py --floor {label} {jsonl} {os.path.basename(jsonl)}"],
        "notes": (f"Bit-floor experiment for the §4 scale curve. hypothesis_status="
                  f"{rec.get('hypothesis_status')}. floor = lowest effective bpw at "
                  f"<= +{GATE}% ppl (multiwindow) and capability loss <= one of 22 total items / "
                  f"one item per task family. When no floor passes, this schema-valid negative "
                  f"receipt names the best measured deployable candidate ({config}, {bpw} bpw, "
                  f"+{degr}% ppl) without claiming success. "
                  f"{'LAB rung - never sets the verdict (§0.5).' if is_lab else ''} "
                  f"beats-Q4_K={'yes' if (floor_bpw and floor_bpw < Q4K_BPW) else 'no'}."),
    }
    sh = _suite_hash()
    if sh:
        r["prompt_suite_hash"] = sh; r["prompt_suite_version"] = "v1"
    out = f"receipts/official/{label}-floor.json"
    os.makedirs("receipts/official", exist_ok=True)
    _atomic_json(out, r)
    print(f"[receipt] wrote {out} ({r['claim_type']}, {r['quality_gate']})", file=sys.stderr)


def _peak_rss_gb(label):
    """Best-effort: read the scheduler's measured peak for this job if available, else 0.0."""
    p = "reports/cron/ram_actuals.jsonl"
    if os.path.exists(p):
        best = 0.0
        for ln in open(p):
            try:
                r = json.loads(ln)
                if r.get("name") == label:
                    best = max(best, r.get("peak_gb", 0.0))
            except Exception:
                pass
        return round(best, 2)
    return 0.0
PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "7B": 7.0, "14B": 14.0, "32B": 32.0,
          "70B": 70.0, "72B": 72.0, "405B": 405.0}


def find_floor(label, jsonl, baseline_override=None):
    """Lowest bpw that passes both PPL and an independently recomputed f16-relative task gate."""
    best = None
    baseline = baseline_override if validate_tripwire_baseline(
        baseline_override, f"{label}-f16"
    )["ok"] else None
    candidates = []
    measured_deployable = []
    excluded_non_deployable = []
    for ln in open(jsonl):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("model") not in (None, label):
            continue
        if r.get("config") == "f16" \
                and validate_tripwire_baseline(r.get("tripwire"), f"{label}-f16")["ok"]:
            baseline = r["tripwire"]
            continue
        if "ppl" not in r or "eff_bpw" not in r:
            continue
        bpw, degr = r.get("eff_bpw"), r.get("degr_pct")
        if degr is None or bpw is None:
            continue
        if r.get("deployable") is False:
            excluded_non_deployable.append({"config": r.get("config"), "eff_bpw": bpw,
                                            "reason": "deployable=false"})
            continue
        if (isinstance(bpw, (int, float)) and float(bpw) > 0
                and isinstance(degr, (int, float)) and math.isfinite(float(degr))):
            measured_deployable.append(r)
        if degr <= GATE:
            candidates.append(r)
    audit = {
        "schema": "hawking.floor_capability_gate.v1",
        "status": "fail",
        "policy": tripwire_policy(),
        "baseline_present": baseline is not None,
        "candidates": [],
        "excluded_non_deployable": excluded_non_deployable,
    }
    best_measured = min(
        measured_deployable,
        key=lambda row: (float(row["degr_pct"]), float(row["eff_bpw"])),
        default=None,
    )
    audit["best_measured"] = ({
        "config": best_measured.get("config"),
        "eff_bpw": best_measured.get("eff_bpw"),
        "degr_pct": best_measured.get("degr_pct"),
        "tripwire_gate": compare_tripwire(baseline, best_measured.get("tripwire"))
                           if baseline is not None else None,
    } if best_measured else None)
    if baseline is None:
        audit["problems"] = ["missing/invalid f16 multi_eval baseline"]
        return None, audit
    for row in candidates:
        gate = compare_tripwire(baseline, row.get("tripwire"))
        audit["candidates"].append({"config": row.get("config"), "eff_bpw": row.get("eff_bpw"),
                                    "degr_pct": row.get("degr_pct"), "gate": gate})
        if gate["status"] == "pass" and (best is None or row["eff_bpw"] < best[0]):
            best = (row["eff_bpw"], row.get("config"), row["degr_pct"], gate)
    audit["status"] = "pass" if best is not None else "no-passing-candidate"
    audit["problems"] = [] if best is not None else [
        "no PPL-eligible candidate passed the f16-relative capability gate"
    ]
    return best, audit


def cmd_floor(label, jsonl, out, model_dir=None):
    if not os.path.exists(jsonl):
        print(f"[floor] {label}: no results at {jsonl} yet", file=sys.stderr); return 4
    fl, capability_audit = find_floor(label, jsonl)
    baseline_evidence = None
    if not capability_audit["baseline_present"] and model_dir:
        try:
            baseline_receipt = _capture_f16_tripwire(label, jsonl, model_dir)
        except Exception as exc:
            print(f"[floor] {label}: BLOCKED — f16 capability baseline failed: {exc}",
                  file=sys.stderr)
            return 4
        baseline_path = f"{jsonl}.f16_tripwire.json"
        baseline_evidence = {"path": baseline_path, "sha256": _sha256(baseline_path)}
        fl, capability_audit = find_floor(label, jsonl, baseline_receipt["result"])
        capability_audit["baseline_evidence"] = baseline_evidence
    if not capability_audit["baseline_present"]:
        print(f"[floor] {label}: BLOCKED — missing/invalid f16 capability baseline", file=sys.stderr)
        return 4
    if not fl:
        print(f"[floor] {label}: no config passed both +{GATE}% PPL and capability gates",
              file=sys.stderr)
        best_measured = capability_audit.get("best_measured")
        if not best_measured:
            print(f"[floor] {label}: BLOCKED — no measured deployable candidate evidence",
                  file=sys.stderr)
            return 4
        rec = {"model": label, "params_b": PARAMS.get(label), "floor_bpw": None,
               "hypothesis_status": "not_supported",
               "best_measured_config": best_measured["config"],
               "best_measured_bpw": best_measured["eff_bpw"],
               "best_measured_degr_pct": best_measured["degr_pct"],
               "gate_pct": GATE, "tripwire_gate": capability_audit,
               "note": "no config within PPL + capability gates"}
    else:
        bpw, cfg, degr, gate = fl
        rec = {"model": label, "params_b": PARAMS.get(label), "floor_bpw": bpw,
               "hypothesis_status": "supported",
               "winning_config": cfg, "degr_pct": degr, "gate_pct": GATE,
               "tripwire_gate": gate}
        print(f"[floor] {label}: floor = {bpw:.3f} eff-bpw via {cfg} (+{degr}%, "
              f"task drop={gate.get('aggregate_drop', 0):.4f})", file=sys.stderr)
    if baseline_evidence:
        rec["tripwire_baseline_evidence"] = baseline_evidence
    # Bind the selected point to the exact audit bytes. The shared curve is a concurrent RMW: early
    # model tiers finish in parallel, so a stable sidecar flock must cover read, de-dup, and replace.
    rec.update({
        "schema": FLOOR_POINT_SCHEMA,
        "audit_jsonl": os.path.realpath(jsonl),
        "audit_sha256": _sha256(jsonl),
    })
    try:
        locked_upsert_floor_row(out, label, rec)
    except Exception as exc:
        print(f"[floor] {label}: floor JSONL commit failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 4
    if model_dir:
        try:
            emit_receipt(label, rec, model_dir, jsonl)
        except Exception as e:
            print(f"[receipt] {label}: emit failed ({e})", file=sys.stderr)
    # A negative bit-floor result is still a completed experiment. Only missing proof inputs or
    # missing measured artifacts are execution failures.
    return 0


def selftest():
    base = {"schema": "hawking.multi_eval.v1", "suite": "hawking.multi_eval.v1",
            "label": "7B-f16", "override": None, "adapter": None,
            "n": 22, "task_n": {"qa": 6, "cloze": 5, "math": 6, "code": 5},
            "per_task": {"qa": 1.0, "cloze": 1.0, "math": 1.0, "code": 1.0},
            "aggregate": 1.0}
    one_loss = {"n": 22, "task_n": base["task_n"],
                "per_task": {"qa": 0.8333, "cloze": 1.0, "math": 1.0, "code": 1.0},
                "aggregate": 0.9545}
    two_loss = {"n": 22, "task_n": base["task_n"],
                "per_task": {"qa": 0.6667, "cloze": 1.0, "math": 1.0, "code": 1.0},
                "aggregate": 0.9091}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ladder.jsonl")
        rows = [
            {"model": "7B", "config": "f16", "eff_bpw": 16, "ppl": 10,
             "degr_pct": 0, "tripwire": base},
            {"model": "7B", "config": "one-loss", "eff_bpw": 3.0, "ppl": 10.1,
             "degr_pct": 1.0, "tripwire": one_loss},
            {"model": "7B", "config": "two-loss", "eff_bpw": 2.0, "ppl": 10.1,
             "degr_pct": 1.0, "tripwire": two_loss},
            {"model": "7B", "config": "vtq-oracle", "eff_bpw": 0.5, "ppl": 10.0,
             "degr_pct": 0.0, "tripwire": base, "deployable": False},
        ]
        open(path, "w").write("".join(json.dumps(row) + "\n" for row in rows))
        floor, audit = find_floor("7B", path)
        assert audit["baseline_present"] and floor[:3] == (3.0, "one-loss", 1.0)
        assert audit["excluded_non_deployable"] == [
            {"config": "vtq-oracle", "eff_bpw": 0.5, "reason": "deployable=false"}
        ]
        open(path, "w").write(json.dumps(rows[1]) + "\n")
        floor, audit = find_floor("7B", path)
        assert floor is None and not audit["baseline_present"]

        # A complete negative experiment is success at the orchestration layer: it records no
        # floor, names the best measured deployable artifact, and returns zero.
        negative_path = os.path.join(td, "negative.jsonl")
        negative_rows = [rows[0], rows[2]]
        open(negative_path, "w").write(
            "".join(json.dumps(row) + "\n" for row in negative_rows)
        )
        negative_out = os.path.join(td, "negative_floor.jsonl")
        assert cmd_floor("7B", negative_path, negative_out) == 0
        negative = json.loads(open(negative_out).read())
        assert negative["schema"] == FLOOR_POINT_SCHEMA
        assert negative["audit_sha256"] == _sha256(negative_path)
        assert negative["floor_bpw"] is None
        assert negative["hypothesis_status"] == "not_supported"
        assert negative["best_measured_config"] == "two-loss"
        assert negative["best_measured_bpw"] == 2.0

        # The negative/baseline receipt itself must pass the production schema/rule verifier.
        import receipt_verify
        old_cwd = os.getcwd()
        receipt_root = os.path.join(td, "receipt-root")
        os.makedirs(receipt_root)
        try:
            os.chdir(receipt_root)
            model_dir = os.path.join(receipt_root, "model")
            os.makedirs(model_dir)
            open(os.path.join(model_dir, "model.safetensors"), "wb").write(b"source")
            base_05 = {**base, "label": "0.5B-f16", "model": model_dir}
            rows_05 = [
                {**negative_rows[1], "model": "0.5B"},
            ]
            jsonl_05 = os.path.join(receipt_root, "negative_05.jsonl")
            open(jsonl_05, "w").write(
                "".join(json.dumps(row) + "\n" for row in rows_05)
            )
            _atomic_json(f"{jsonl_05}.f16_tripwire.json", {
                "schema": "hawking.tripwire_baseline.v1", "status": "pass",
                "model": "0.5B", "model_dir": model_dir, "result": base_05,
            })
            floor_05 = os.path.join(receipt_root, "floor_05.jsonl")
            assert cmd_floor("0.5B", jsonl_05, floor_05, model_dir) == 0
            receipt = json.load(open("receipts/official/0.5B-floor.json"))
            ok, reasons = receipt_verify.verify_receipt(receipt, receipt_verify.load_schema())
            assert ok, reasons
            assert receipt["claim_type"] == "baseline"
            assert receipt["quality_gate"] == "fail"
            assert receipt["effective_bpw"] == 2.0
            floor_row = json.loads(open(floor_05).read())
            assert receipt["floor_point_sha256"] == canonical_row_sha256(floor_row)
        finally:
            os.chdir(old_cwd)

        # Shared-floor updates are canonical, de-duplicated, and preserve other model rows.
        locked = os.path.join(td, "locked_floor.jsonl")
        row_a = {"schema": FLOOR_POINT_SCHEMA, "model": "A", "value": 1}
        row_b = {"schema": FLOOR_POINT_SCHEMA, "model": "B", "value": 2}
        locked_upsert_floor_row(locked, "A", row_a)
        locked_upsert_floor_row(locked, "B", row_b)
        row_a2 = {**row_a, "value": 3}
        locked_upsert_floor_row(locked, "A", row_a2)
        locked_rows = [json.loads(line) for line in open(locked)]
        assert locked_rows == [row_b, row_a2]
        assert os.path.exists(locked + ".lock")
    print("scaling_law.py selftest OK")
    return 0


def _safe_model(ln):
    try:
        return json.loads(ln).get("model")
    except Exception:
        return None


def _linfit(xs, ys):
    """Least-squares y = m x + b (pure python; no numpy dependency). Returns (m, b, r2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    m = sxy / sxx if sxx else 0.0
    b = my - m * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (m * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return m, b, r2


def cmd_fit(floors):
    pts = []
    for ln in open(floors):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("floor_bpw") and r.get("params_b"):
            pts.append((r["params_b"], r["floor_bpw"], r["model"]))
    pts.sort()
    if len(pts) < 2:
        print(f"[fit] need >=2 floor points, have {len(pts)} — run more rungs first", file=sys.stderr)
        return
    # verdict is read off 7B+; labs (<7B) shown but excluded from the law (they floor ~3-bit, §0 rule 5)
    big = [(p, f) for (p, f, m) in pts if p >= 7.0]
    xs = [math.log10(p) for p, f in big]
    ys = [f for p, f in big]
    m, b, r2 = _linfit(xs, ys)
    descends = m < -0.05          # meaningful negative slope
    pred = {N: m * math.log10(N) + b for N in (70, 405)}
    md = floors.replace(".jsonl", "_curve.md")
    with open(md, "w") as o:
        o.write("# Bit-floor vs scale (plan §4 / T3.1)\n\n")
        o.write(f"Gate: <= +{GATE}% ppl vs f16 parent, effective bpw, multiwindow.\n\n")
        o.write("| model | params (B) | floor eff-bpw | role |\n|---|--:|--:|---|\n")
        for p, f, mlabel in pts:
            role = "lab (not in fit)" if p < 7.0 else "verdict"
            o.write(f"| {mlabel} | {p} | {f:.3f} | {role} |\n")
        o.write(f"\n**Law (7B+):** floor ~= {m:.3f}*log10(N) + {b:.3f}  (R^2={r2:.3f})\n\n")
        o.write(f"**Verdict:** {'H1 CONFIRMED - floor DESCENDS with scale' if descends else 'H0 - floor ~FLAT, redundancy buys little'} "
                f"(slope {m:.3f} bpw/decade)\n\n")
        o.write("**Pre-registered extrapolation (a PREDICTION until an off-box run confirms it):**\n")
        o.write(f"- 70B  -> ~{pred[70]:.2f} eff-bpw\n")
        o.write(f"- 405B -> ~{pred[405]:.2f} eff-bpw {'(< 1-bit territory!)' if pred[405] < 1 else ''}\n")
    print(f"[fit] {len(big)} verdict points, slope {m:.3f}/decade, R^2 {r2:.3f} -> {md}", file=sys.stderr)
    print(f"[fit] {'H1 (descends)' if descends else 'H0 (flat)'}; 70B~{pred[70]:.2f}bpw 405B~{pred[405]:.2f}bpw",
          file=sys.stderr)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else ""
    if a == "--floor":
        raise SystemExit(cmd_floor(sys.argv[2], sys.argv[3], sys.argv[4],
                                   sys.argv[5] if len(sys.argv) > 5 else None))
    elif a == "--fit":
        cmd_fit(sys.argv[2])
    elif a == "--selftest":
        raise SystemExit(selftest())
    else:
        print(__doc__)
