#!/usr/bin/env python3.12
"""Hawking rebuild performance gate — contamination-aware, paired/relative.

Every metric is measured | skipped | unavailable with a reason. Never fabricates
TPS. Prefer --paired (ABAB) for A/B decisions on a shared machine.

  python3.12 tools/verify/perfgate.py --list
  python3.12 tools/verify/perfgate.py --capture --out REBUILD_PERFORMANCE_BASELINE_MEASURED.json
  python3.12 tools/verify/perfgate.py --compare A.json B.json --gate 2.0
  python3.12 tools/verify/perfgate.py --paired --a-cmd '…' --b-cmd '…' --n 9
"""
from __future__ import annotations

import argparse, json, math, os, platform, re, shutil, statistics, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA = "hawking.rebuild.perfgate.v1"
DEFAULT_N = 8  # 1 warm-up discarded + 7 kept
HEAVY_CPU_PCT = 400.0
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lab.layout import OPS_ROOT, evidence_dir

HOME = Path.home()
APP_HAWK = HOME / "Library/Application Support/Hawking"
MAIN_TARGET = Path("/Users/scammermike/Downloads/hawking/target")

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def sample_system() -> dict[str, Any]:
    out: dict[str, Any] = {
        "ts": utc_now(), "loadavg_1_5_15": list(os.getloadavg()), "ncpu": os.cpu_count(),
        "free_bytes": None, "active_bytes": None,
        "heavy_process_gt_4_cores": None, "heavy_process_detail": None, "ps_status": "unknown",
    }
    try:
        page = int(subprocess.check_output(["pagesize"], text=True).strip())
    except Exception:
        page = 16384
    try:
        pages: dict[str, int] = {}
        for line in subprocess.check_output(["vm_stat"], text=True).splitlines():
            m = re.match(r"([^:]+):\s+(\d+)", line)
            if m:
                pages[m.group(1).strip()] = int(m.group(2).rstrip("."))
        out["free_bytes"] = pages.get("Pages free", 0) * page
        out["active_bytes"] = pages.get("Pages active", 0) * page
    except Exception as e:
        out["vm_stat_error"] = str(e)
    try:
        r = subprocess.run(["ps", "-Ao", "%cpu=", "-o", "comm="],
                           capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            out["ps_status"] = f"unavailable: rc={r.returncode} {(r.stderr or '')[:100]}"
        else:
            heavy = []
            for line in r.stdout.splitlines():
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                try:
                    cpu = float(parts[0])
                except ValueError:
                    continue
                if cpu >= HEAVY_CPU_PCT:
                    heavy.append({"cpu_pct": cpu, "comm": parts[1][:80]})
            heavy.sort(key=lambda x: -x["cpu_pct"])
            out["heavy_process_gt_4_cores"] = bool(heavy)
            out["heavy_process_detail"] = heavy[:8]
            out["ps_status"] = "ok"
    except Exception as e:
        out["ps_status"] = f"unavailable: {e}"
    return out

def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
        timeout: float | None = None) -> tuple[int, str, str, float]:
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=str(cwd or REPO), env=env, capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr, time.perf_counter() - t0
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", f"timeout {timeout}s", time.perf_counter() - t0
    except FileNotFoundError as e:
        return 127, "", str(e), time.perf_counter() - t0

def find_hawking_bin() -> Path | None:
    if os.environ.get("HAWKING_BIN") and Path(os.environ["HAWKING_BIN"]).is_file():
        return Path(os.environ["HAWKING_BIN"])
    for p in (REPO / "target/release/hawking", MAIN_TARGET / "release/hawking",
              REPO / "target/debug/hawking", Path(shutil.which("hawking") or "")):
        if p and p.is_file() and os.access(p, os.X_OK):
            return p
    return None

def find_example(name: str) -> Path | None:
    for root in (REPO / "target/release/examples", MAIN_TARGET / "release/examples"):
        p = root / name
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None

def find_gguf() -> Path | None:
    if os.environ.get("HAWKING_GGUF") and Path(os.environ["HAWKING_GGUF"]).is_file():
        return Path(os.environ["HAWKING_GGUF"])
    for root in (OPS_ROOT / "local" / "models", REPO / "models"):
        if root.is_dir():
            for p in sorted(root.rglob("*.gguf")):
                if p.is_file() and p.stat().st_size > 1_000_000:
                    return p
    return None

def llama_gravity() -> Path | None:
    for p in (
        APP_HAWK / "CampaignS08/llama32-1b-R0.v2.gravity",
        APP_HAWK / "CampaignS08/llama32-1b-R0.gravity",
    ):
        if p.is_file():
            return p
    return None

def glm_math_preserve() -> Path | None:
    p = (APP_HAWK / "Models/GLM-5.2/b4734de4facf877f85769a911abafc5283eab3d9"
         "/GLM-5.2-H0.98-Math-Preserve.gravity")
    return p if p.is_dir() else None

def cargo_env() -> dict[str, str]:
    """Prefer an explicitly set or writable target. Shared main target may EPERM under sandbox."""
    env = os.environ.copy()
    if env.get("CARGO_TARGET_DIR"):
        Path(env["CARGO_TARGET_DIR"]).mkdir(parents=True, exist_ok=True)
        return env
    scratch = Path("/tmp/hawking-perfgate-target")
    scratch.mkdir(parents=True, exist_ok=True)
    env["CARGO_TARGET_DIR"] = str(scratch)
    return env

def metal_ok(bin_path: Path | None) -> tuple[bool, str]:
    if not bin_path:
        return False, "no hawking binary"
    rc, out, err, _ = run([str(bin_path), "bench-q4k-shapes", "--iters", "1"], timeout=30)
    text = (out + err).lower()
    if rc == 0:
        return True, "bench-q4k-shapes ok"
    return False, f"Metal unavailable: {(err or out).strip()[:240]}"

def metric(name: str, family: str, status: str, reason: str = "", **kw: Any) -> dict[str, Any]:
    m = {"name": name, "family": family, "status": status, "reason": reason,
         "label": kw.pop("label", "measured" if status == "measured" else status),
         "unit": kw.pop("unit", ""), "higher_is_better": kw.pop("higher_is_better", False),
         "warm_up_discarded": kw.pop("warm_up_discarded", status == "measured"),
         "n_requested": kw.pop("n_requested", 0), "n_kept": kw.pop("n_kept", 0),
         "values": kw.pop("values", []), "median": kw.pop("median", None),
         "min": kw.pop("min", None), "max": kw.pop("max", None),
         "samples": kw.pop("samples", [])}
    if kw:
        m["extras"] = kw
    return m

def measure(name: str, family: str, n: int, unit: str, higher_is_better: bool,
            once: Callable[[], tuple[float, dict[str, Any]]]) -> dict[str, Any]:
    samples, values = [], []
    for i in range(n):
        before = sample_system()
        try:
            val, extra = once()
        except Exception as e:
            return metric(name, family, "unavailable", f"sample {i} raised: {e}", n_requested=n)
        rec = {"i": i, "value": val, "warm_up": i == 0, "system_before": before,
               "system_after": sample_system(), **extra}
        samples.append(rec)
        if i > 0:
            values.append(val)
    if not values:
        return metric(name, family, "unavailable", "no kept samples", n_requested=n, samples=samples)
    return metric(name, family, "measured", "ok", unit=unit, higher_is_better=higher_is_better,
                  label="measured", n_requested=n, n_kept=len(values), warm_up_discarded=True,
                  values=values, median=statistics.median(values), min=min(values), max=max(values),
                  samples=samples)

def inventory(n: int, include_cold: bool) -> list[dict[str, Any]]:
    cargo, bin_path, gguf = shutil.which("cargo"), find_hawking_bin(), find_gguf()
    mok, mwhy = metal_ok(bin_path) if bin_path else (False, "no binary")
    llama, glm = llama_gravity(), glm_math_preserve()
    gtps, gglm = find_example("gravity_tps"), find_example("gravity_glm_tps")
    fmt, pack = REPO / "tools/condense/artifact_client.py", REPO / "lab/operators/glm52_pack.py"
    rows: list[dict[str, Any]] = []

    def add(name: str, fam: str, st: str, reason: str, **kw: Any) -> None:
        rows.append({"name": name, "family": fam, "status": st, "reason": reason, **kw})

    if cargo:
        add("build.cargo_check_s", "build", "measurable", f"cargo check -p hawking (n={n})", unit="s")
        add("build.cargo_build_warm_s", "build", "measurable",
            f"cargo build -p hawking --release after touch (n={n})", unit="s")
        add("build.cargo_build_cold_s", "build",
            "measurable" if include_cold else "skipped",
            "cargo clean -p hawking && release build" if include_cold
            else "opt-in via --include-cold", unit="s")
    else:
        for nm in ("build.cargo_check_s", "build.cargo_build_warm_s", "build.cargo_build_cold_s"):
            add(nm, "build", "unavailable", "cargo not on PATH")
    if bin_path:
        add("build.binary_size_bytes", "build", "measurable", f"stat {bin_path}", unit="bytes")
        add("startup.help_s", "startup", "measurable", f"{bin_path} --help", unit="s")
        add("startup.version_s", "startup", "measurable", f"{bin_path} version", unit="s")
        add("startup.doctor_s", "startup",
            "measurable" if gguf else "unavailable",
            f"doctor --weights {gguf}" if gguf else
            "no GGUF (doctor needs GGUF magic); looked workspace/ops/local/models/**/*.gguf "
            "and $HAWKING_GGUF", unit="s")
    else:
        for nm in ("build.binary_size_bytes", "startup.help_s", "startup.version_s", "startup.doctor_s"):
            add(nm, "build" if nm.startswith("build") else "startup", "unavailable", "no hawking binary")

    if not mok:
        add("base_tps.llama1b_decode_tps", "base_tps", "unavailable",
            f"Metal required for BASE_TRUE_TPS; {mwhy}. No synthetic proxy.")
        add("base_tps.glm52_math_preserve_tps", "base_tps", "unavailable", f"Metal required; {mwhy}")
        add("accelerated_tps.profile_fast_decode_tps", "accelerated_tps", "unavailable",
            f"Metal required; {mwhy}")
        add("kernel.bench_q4k_shapes_median_us", "kernel", "unavailable", f"Metal required; {mwhy}")
    else:
        if llama and gtps:
            add("base_tps.llama1b_decode_tps", "base_tps", "measurable",
                f"gravity_tps --artifact {llama} --context 128 --decode 16", unit="tps",
                higher_is_better=True)
        else:
            add("base_tps.llama1b_decode_tps", "base_tps", "unavailable",
                f"need llama .gravity ({llama}) and gravity_tps binary ({gtps})")
        if glm and gglm:
            add("base_tps.glm52_math_preserve_tps", "base_tps", "skipped",
                f"artifact at {glm} (~86GiB); default capture skips (use --include-glm-tps)")
        else:
            add("base_tps.glm52_math_preserve_tps", "base_tps", "unavailable",
                f"need GLM Math-Preserve dir and gravity_glm_tps; looked {APP_HAWK}/Models/GLM-5.2/…")
        if gguf:
            add("accelerated_tps.profile_fast_decode_tps", "accelerated_tps", "measurable",
                f"hawking bench --suite decode --profile fast --weights {gguf}", unit="tps",
                higher_is_better=True)
        else:
            add("accelerated_tps.profile_fast_decode_tps", "accelerated_tps", "unavailable",
                "no GGUF for accelerated path; workspace/ops/local/models/**/*.gguf or $HAWKING_GGUF")
        add("kernel.bench_q4k_shapes_median_us", "kernel", "measurable",
            "bench-q4k-shapes --iters 50 (no model)", unit="us")

    add("kernel.bench_kernel_dispatch", "kernel", "unavailable",
        "hawking bench-kernel not on CLI (extracted to hawking-bench pack); use bench-q4k-shapes")
    add("transform.gravity_format_selftest_s", "transform",
        "measurable" if fmt.is_file() else "unavailable",
        "artifact_client performance-smoke (equal-work vs old gravity_format selftest)"
        if fmt.is_file() else f"missing {fmt}", unit="s")
    add("transform.pack_indices_bytes_per_s", "transform",
        "measurable" if pack.is_file() else "unavailable",
        "fixture-scale lab.operators.glm52_pack pack/unpack indices (no 1.4TB source)" if pack.is_file()
        else f"missing {pack}", unit="bytes/s", higher_is_better=True)
    add("transform.shard_write_verify_bytes_per_s", "transform",
        "measurable" if fmt.is_file() else "unavailable",
        "fixture-scale artifact write_shard+verify (32x4KiB tensors, no torch/MPS)"
        if fmt.is_file() else f"missing {fmt}", unit="bytes/s", higher_is_better=True)
    add("numeric_parity.gravity_verify_roundtrip_s", "numeric_parity",
        "measurable" if fmt.is_file() else "unavailable",
        "artifact performance-smoke write/verify/tamper (CPU container oracle)", unit="s")
    return rows

def cap_build(n: int, include_cold: bool) -> list[dict[str, Any]]:
    cargo, env = shutil.which("cargo"), cargo_env()
    if not cargo:
        return [metric(nm, "build", "unavailable", "cargo not on PATH")
                for nm in ("build.cargo_check_s", "build.cargo_build_warm_s",
                           "build.cargo_build_cold_s", "build.binary_size_bytes")]
    touch = REPO / "crates/hawking/src/main.rs"
    if not touch.is_file():
        touch = REPO / "crates/hawking/Cargo.toml"

    def check() -> tuple[float, dict[str, Any]]:
        if touch.is_file():
            touch.touch()
        rc, out, err, wall = run([cargo, "check", "-p", "hawking", "--quiet"], env=env, timeout=1800)
        if rc:
            raise RuntimeError(f"cargo check rc={rc}: {(err or out)[-300:]}")
        return wall, {"cmd": "cargo check -p hawking"}

    def warm() -> tuple[float, dict[str, Any]]:
        if touch.is_file():
            touch.touch()
        rc, out, err, wall = run([cargo, "build", "-p", "hawking", "--release", "--quiet"],
                                 env=env, timeout=3600)
        if rc:
            raise RuntimeError(f"cargo build rc={rc}: {(err or out)[-300:]}")
        return wall, {"cmd": "cargo build -p hawking --release"}

    out = [
        measure("build.cargo_check_s", "build", n, "s", False, check),
        measure("build.cargo_build_warm_s", "build", n, "s", False, warm),
    ]
    if include_cold:
        def cold() -> tuple[float, dict[str, Any]]:
            run([cargo, "clean", "-p", "hawking"], env=env, timeout=600)
            rc, o, e, wall = run([cargo, "build", "-p", "hawking", "--release", "--quiet"],
                                 env=env, timeout=7200)
            if rc:
                raise RuntimeError(f"cold build rc={rc}: {(e or o)[-300:]}")
            return wall, {"cmd": "clean -p hawking && release build"}
        out.append(measure("build.cargo_build_cold_s", "build", n, "s", False, cold))
    else:
        out.append(metric("build.cargo_build_cold_s", "build", "skipped",
                          "opt-in via --include-cold", unit="s", n_requested=n))
    b = find_hawking_bin()
    if b:
        sz = float(b.stat().st_size)
        out.append(metric(
            "build.binary_size_bytes", "build", "measured", f"stat {b}",
            unit="bytes", label="measured", n_requested=1, n_kept=1,
            warm_up_discarded=False, values=[sz], median=sz, min=sz, max=sz,
            samples=[{"i": 0, "value": sz, "path": str(b),
                      "system_before": sample_system()}], path=str(b)))
    else:
        out.append(metric("build.binary_size_bytes", "build", "unavailable", "no binary after build"))
    return out

def cap_startup(n: int) -> list[dict[str, Any]]:
    b = find_hawking_bin()
    if not b:
        return [metric(nm, "startup", "unavailable", "no hawking binary")
                for nm in ("startup.help_s", "startup.version_s", "startup.doctor_s")]

    def help_once() -> tuple[float, dict[str, Any]]:
        rc, o, e, wall = run([str(b), "--help"], timeout=30)
        if rc:
            raise RuntimeError(f"--help rc={rc}: {e[:200]}")
        return wall, {"stdout_bytes": len(o)}

    def ver_once() -> tuple[float, dict[str, Any]]:
        rc, o, e, wall = run([str(b), "version"], timeout=30)
        if rc:
            raise RuntimeError(f"version rc={rc}: {e[:200]}")
        return wall, {"stdout": o.strip()[:80]}

    out = [
        measure("startup.help_s", "startup", n, "s", False, help_once),
        measure("startup.version_s", "startup", n, "s", False, ver_once),
    ]
    gguf = find_gguf()
    if not gguf:
        out.append(metric("startup.doctor_s", "startup", "unavailable",
                          "no GGUF for doctor; workspace/ops/local/models/**/*.gguf "
                          "or $HAWKING_GGUF"))
    else:
        def doc() -> tuple[float, dict[str, Any]]:
            rc, o, e, wall = run([str(b), "doctor", "--weights", str(gguf), "--json"], timeout=120)
            if rc:
                raise RuntimeError(f"doctor rc={rc}: {(e or o)[:240]}")
            return wall, {"weights": str(gguf)}
        out.append(measure("startup.doctor_s", "startup", n, "s", False, doc))
    return out

def cap_base_tps(n: int, include_glm: bool) -> list[dict[str, Any]]:
    b = find_hawking_bin()
    mok, mwhy = metal_ok(b)
    if not mok:
        why = f"Metal required for true TPS; {mwhy}. Refusing synthetic proxy."
        return [metric("base_tps.llama1b_decode_tps", "base_tps", "unavailable", why),
                metric("base_tps.glm52_math_preserve_tps", "base_tps", "unavailable", why)]
    out: list[dict[str, Any]] = []
    llama, gtps = llama_gravity(), find_example("gravity_tps")
    if not llama:
        out.append(metric("base_tps.llama1b_decode_tps", "base_tps", "unavailable",
                          f"missing {APP_HAWK}/CampaignS08/llama32-1b-R0.v2.gravity"))
    elif not gtps:
        out.append(metric("base_tps.llama1b_decode_tps", "base_tps", "unavailable",
                          "gravity_tps example not built"))
    else:
        def once() -> tuple[float, dict[str, Any]]:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                op = Path(tf.name)
            try:
                rc, o, e, wall = run(
                    [str(gtps), "--artifact", str(llama), "--context", "128",
                     "--decode", "16", "--out", str(op)], timeout=600)
                if rc:
                    raise RuntimeError(f"gravity_tps rc={rc}: {(e or o)[:300]}")
                data = json.loads(op.read_text())
                rows = data.get("measurements") or []
                row = next((r for r in rows if r.get("context_tokens") == 128), rows[0])
                return float(row["base_true_decode_tps"]), {
                    "artifact": str(llama), "wall_s": wall,
                    "dispatches_per_token": row.get("dispatches_per_token"),
                }
            finally:
                op.unlink(missing_ok=True)
        out.append(measure("base_tps.llama1b_decode_tps", "base_tps", n, "tps", True, once))

    glm, gbin = glm_math_preserve(), find_example("gravity_glm_tps")
    if not glm:
        out.append(metric("base_tps.glm52_math_preserve_tps", "base_tps", "unavailable",
                          "missing GLM Math-Preserve .gravity directory"))
    elif not include_glm:
        out.append(metric("base_tps.glm52_math_preserve_tps", "base_tps", "skipped",
                          f"artifact present ({glm}); pass --include-glm-tps (multi-minute samples)",
                          path=str(glm)))
    elif not gbin:
        out.append(metric("base_tps.glm52_math_preserve_tps", "base_tps", "unavailable",
                          "gravity_glm_tps example not built"))
    else:
        def gonce() -> tuple[float, dict[str, Any]]:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                op = Path(tf.name)
            try:
                rc, o, e, wall = run(
                    [str(gbin), "--dir", str(glm), "--context", "4", "--decode", "12",
                     "--out", str(op)], timeout=7200)
                if rc:
                    raise RuntimeError(f"gravity_glm_tps rc={rc}: {(e or o)[:300]}")
                rows = json.loads(op.read_text()).get("measurements") or []
                return float(rows[0]["base_true_decode_tps"]), {"wall_s": wall, "dir": str(glm)}
            finally:
                op.unlink(missing_ok=True)
        out.append(measure("base_tps.glm52_math_preserve_tps", "base_tps", n, "tps", True, gonce))
    return out

def cap_accelerated(n: int) -> list[dict[str, Any]]:
    b, gguf = find_hawking_bin(), find_gguf()
    mok, mwhy = metal_ok(b)
    if not mok:
        return [metric("accelerated_tps.profile_fast_decode_tps", "accelerated_tps", "unavailable",
                       f"Metal required; {mwhy}. Refusing synthetic proxy.")]
    if not gguf:
        return [metric("accelerated_tps.profile_fast_decode_tps", "accelerated_tps", "unavailable",
                       "no GGUF; workspace/ops/local/models/**/*.gguf or $HAWKING_GGUF")]
    if not b:
        return [metric("accelerated_tps.profile_fast_decode_tps", "accelerated_tps", "unavailable",
                       "no hawking binary")]

    def once() -> tuple[float, dict[str, Any]]:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            op = Path(tf.name)
        try:
            rc, o, e, wall = run(
                [str(b), "bench", "--weights", str(gguf), "--suite", "decode",
                 "--profile", "fast", "--trials", "1", "--max-new-tokens", "32",
                 "--json", str(op)], timeout=1800)
            if rc:
                raise RuntimeError(f"bench rc={rc}: {(e or o)[:300]}")
            data = json.loads(op.read_text()) if op.stat().st_size else {}
            tps = None
            if isinstance(data.get("results"), dict):
                tps = data["results"].get("decode_tps") or data["results"].get("median_tps")
            if tps is None:
                m = re.search(r"decode[_\s-]?tps[\"\s:=]+([0-9.]+)", o + e, re.I)
                if m:
                    tps = float(m.group(1))
            if tps is None:
                raise RuntimeError(f"could not parse decode tps: {(o or e)[:200]}")
            return float(tps), {"weights": str(gguf), "wall_s": wall}
        finally:
            op.unlink(missing_ok=True)
    return [measure("accelerated_tps.profile_fast_decode_tps", "accelerated_tps", n, "tps", True, once)]

_IDX_SCRIPT = r"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
from lab.operators.glm52_pack import pack_indices, unpack_indices
rng = np.random.default_rng(0)
vals = rng.integers(0, 16, size=2_000_000, dtype=np.uint64)
t0 = time.perf_counter()
for _ in range(8):
    b = pack_indices(vals, 4)
    _ = unpack_indices(b, vals.size, 4)
elapsed = time.perf_counter() - t0
print(json.dumps({"bytes_per_s": len(b) * 8 / elapsed, "wall_s": elapsed, "bytes": len(b)}))
"""

_SHARD_SCRIPT = r"""
import json, sys, time, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools.condense import artifact_client as gf
payloads = []
for i in range(32):
    body = bytes((i * 17 + j) % 256 for j in range(4096))
    payloads.append(({"name": f"t{i}", "elements": 4096, "shape": [64, 64],
                      "codec": "fixture.raw", "bpw": 8.0}, body))
bits = sum(len(b) for _, b in payloads) * 8
els = sum(d["elements"] for d, _ in payloads)
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "fixture.gravity"
    t0 = time.perf_counter()
    for _ in range(20):
        gf.write_shard(path, payloads, model={"repo": "fixture", "revision": "0"},
                       compression={"codec": "fixture", "packed_bpw": bits / els})
        rep = gf.verify(path)
        assert rep["ok"], rep
    elapsed = time.perf_counter() - t0
total = sum(len(b) for _, b in payloads) * 20
print(json.dumps({"bytes_per_s": total / elapsed, "tensors_per_s": 32 * 20 / elapsed,
                  "wall_s": elapsed, "bytes": total}))
"""

def cap_transform(n: int) -> list[dict[str, Any]]:
    py, fmt, pack = sys.executable, REPO / "tools/condense/artifact_client.py", REPO / "lab/operators/glm52_pack.py"
    condense = str(REPO / "tools/condense")
    out: list[dict[str, Any]] = []
    if fmt.is_file():
        def fmt_once() -> tuple[float, dict[str, Any]]:
            rc, o, e, wall = run([py, str(fmt), "performance-smoke"], timeout=120)
            if rc:
                raise RuntimeError(f"performance-smoke rc={rc}: {(e or o)[:200]}")
            return wall, {"tool": "artifact_client.py performance-smoke"}
        out.append(measure("transform.gravity_format_selftest_s", "transform", n, "s", False, fmt_once))

        def shard() -> tuple[float, dict[str, Any]]:
            rc, o, e, wall = run([py, "-c", _SHARD_SCRIPT, condense], timeout=120)
            if rc:
                raise RuntimeError(f"shard bench rc={rc}: {(e or o)[:300]}")
            data = json.loads(o.strip().splitlines()[-1])
            return float(data["bytes_per_s"]), data
        out.append(measure("transform.shard_write_verify_bytes_per_s", "transform", n, "bytes/s", True, shard))
    else:
        out.append(metric("transform.gravity_format_selftest_s", "transform", "unavailable", f"missing {fmt}"))
        out.append(metric("transform.shard_write_verify_bytes_per_s", "transform", "unavailable", f"missing {fmt}"))
    if pack.is_file():
        def idx() -> tuple[float, dict[str, Any]]:
            rc, o, e, wall = run([py, "-c", _IDX_SCRIPT, condense], timeout=120)
            if rc:
                raise RuntimeError(f"idx bench rc={rc}: {(e or o)[:240]}")
            data = json.loads(o.strip().splitlines()[-1])
            return float(data["bytes_per_s"]), data
        out.append(measure("transform.pack_indices_bytes_per_s", "transform", n, "bytes/s", True, idx))
    else:
        out.append(metric("transform.pack_indices_bytes_per_s", "transform", "unavailable", f"missing {pack}"))
    return out

def cap_kernel(n: int) -> list[dict[str, Any]]:
    out = [metric("kernel.bench_kernel_dispatch", "kernel", "unavailable",
                  "hawking bench-kernel not on CLI (extracted to hawking-bench pack)")]
    b = find_hawking_bin()
    mok, mwhy = metal_ok(b)
    if not mok:
        out.append(metric("kernel.bench_q4k_shapes_median_us", "kernel", "unavailable",
                          f"Metal required; {mwhy}"))
        return out

    def once() -> tuple[float, dict[str, Any]]:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            op = Path(tf.name)
        try:
            rc, o, e, wall = run([str(b), "bench-q4k-shapes", "--iters", "50", "--out", str(op)],
                                 timeout=300)
            if rc:
                raise RuntimeError(f"bench-q4k-shapes rc={rc}: {(e or o)[:240]}")
            raw = op.read_text() if op.stat().st_size else o
            data = json.loads(raw) if raw.strip()[:1] in "{[" else {}
            med = None
            if isinstance(data, list) and data:
                med = statistics.median(float(x.get("p50_us") or x.get("mean_us") or 0) for x in data)
            elif isinstance(data, dict):
                for key in ("p50_us", "median_us", "mean_us"):
                    if key in data:
                        med = float(data[key]); break
                if med is None and isinstance(data.get("results"), list) and data["results"]:
                    med = statistics.median(
                        float(x.get("p50_us") or x.get("mean_us") or 0) for x in data["results"])
            if med is None:
                return wall * 1e6, {"label": "estimated",
                                    "note": "JSON shape unexpected; used wall_s*1e6", "wall_s": wall}
            return med, {"wall_s": wall}
        finally:
            op.unlink(missing_ok=True)

    m = measure("kernel.bench_q4k_shapes_median_us", "kernel", n, "us", False, once)
    # promote estimated label if any sample used wall fallback
    if any(s.get("label") == "estimated" for s in m.get("samples", [])):
        m["label"] = "estimated"
        m["reason"] = "some samples used wall-time fallback; see sample notes"
    out.append(m)
    return out

def cap_numeric(n: int) -> list[dict[str, Any]]:
    fmt = REPO / "tools/condense/artifact_client.py"
    if not fmt.is_file():
        return [metric("numeric_parity.gravity_verify_roundtrip_s", "numeric_parity", "unavailable",
                       f"missing {fmt}")]

    def once() -> tuple[float, dict[str, Any]]:
        rc, o, e, wall = run([sys.executable, str(fmt), "performance-smoke"], timeout=120)
        if rc:
            raise RuntimeError(f"performance-smoke rc={rc}: {(e or o)[:200]}")
        return wall, {"path": "artifact performance-smoke write/verify/tamper/rate-claim"}
    return [measure("numeric_parity.gravity_verify_roundtrip_s", "numeric_parity", n, "s", False, once)]

def do_capture(n: int, include_cold: bool, include_glm: bool) -> dict[str, Any]:
    started, sys0 = utc_now(), sample_system()
    b = find_hawking_bin()
    mok, mwhy = metal_ok(b)
    metrics: list[dict[str, Any]] = []
    metrics += cap_build(n, include_cold)
    metrics += cap_startup(n)
    metrics += cap_base_tps(n, include_glm)
    metrics += cap_accelerated(n)
    metrics += cap_transform(n)
    metrics += cap_kernel(n)
    metrics += cap_numeric(n)
    by = {s: [m["name"] for m in metrics if m["status"] == s]
          for s in ("measured", "skipped", "unavailable")}
    enforce = {
        "base_tps_enforceable": any(m["name"].startswith("base_tps") and m["status"] == "measured"
                                    for m in metrics),
        "accelerated_tps_enforceable": any(
            m["name"].startswith("accelerated_tps") and m["status"] == "measured" for m in metrics),
        "transform_enforceable": any(m["family"] == "transform" and m["status"] == "measured"
                                     for m in metrics),
        "build_startup_enforceable": any(m["family"] in ("build", "startup") and m["status"] == "measured"
                                         for m in metrics),
        "kernel_enforceable": any(m["family"] == "kernel" and m["status"] == "measured" for m in metrics),
    }
    return {
        "schema": SCHEMA, "mode": "capture", "at": started, "finished_at": utc_now(),
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python": sys.version.split()[0], "ncpu": os.cpu_count(),
                 "repo": str(REPO), "hawking_bin": str(b) if b else None},
        "protocol": {
            "n_runs_per_metric": n, "warm_up_discarded": True,
            "summary": "median + min/max; never mean alone",
            "contamination_note": (
                "This machine is not a clean room. Absolute numbers are contaminated. "
                "Prefer --paired for A/B. Every sample records loadavg/memory; ps heavy-process "
                "probe when permitted."
            ),
            "include_cold_build": include_cold, "include_glm_tps": include_glm,
            "metal": {"ok": mok, "reason": mwhy},
        },
        "system_at_start": sys0, "system_at_end": sample_system(),
        "metrics": metrics,
        "summary": {
            "measured": len(by["measured"]), "skipped": len(by["skipped"]),
            "unavailable": len(by["unavailable"]),
            "gate_enforceability": enforce,
            "measured_names": by["measured"], "skipped_names": by["skipped"],
            "unavailable_names": by["unavailable"],
        },
    }

def do_compare(a_path: Path, b_path: Path, gate_pct: float) -> dict[str, Any]:
    a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    am = {m["name"]: m for m in a.get("metrics", [])}
    bm = {m["name"]: m for m in b.get("metrics", [])}
    rows, failures = [], []
    for name in sorted(set(am) | set(bm)):
        left, right = am.get(name), bm.get(name)
        row: dict[str, Any] = {"name": name}
        if left is None or right is None:
            row.update(status="fail", reason=f"missing in {'A' if left is None else 'B'}")
            failures.append(name); rows.append(row); continue
        row["a_status"], row["b_status"] = left.get("status"), right.get("status")
        if left.get("status") == "measured" and right.get("status") != "measured":
            row.update(status="fail",
                       reason=f"was measured in A, now {right.get('status')}: {right.get('reason')}")
            failures.append(name); rows.append(row); continue
        if left.get("status") != "measured" or right.get("status") != "measured":
            row.update(status="skip", reason="not measured on both sides"); rows.append(row); continue
        a_med, b_med = left.get("median"), right.get("median")
        hib = bool(left.get("higher_is_better"))
        row.update(a_median=a_med, b_median=b_med, higher_is_better=hib,
                   a_min_max=[left.get("min"), left.get("max")],
                   b_min_max=[right.get("min"), right.get("max")])
        if a_med is None or b_med is None or a_med == 0:
            row.update(status="fail", reason="null/zero median"); failures.append(name); rows.append(row)
            continue
        # positive delta_pct_improvement = B better
        delta = ((b_med - a_med) / abs(a_med) * 100.0) if hib else ((a_med - b_med) / abs(a_med) * 100.0)
        row["delta_pct_improvement"] = delta
        if delta < -gate_pct:
            row.update(status="fail", reason=f">{gate_pct}% regression (delta_improvement={delta:.3f}%)")
            failures.append(name)
        else:
            row.update(status="pass", reason=f"within {gate_pct}% (delta_improvement={delta:.3f}%)")
        rows.append(row)
    return {"schema": SCHEMA, "mode": "compare", "a": str(a_path), "b": str(b_path),
            "gate_pct": gate_pct, "pass": not failures, "failures": failures, "rows": rows, "at": utc_now()}

def do_paired(a_cmd: str, b_cmd: str, n: int) -> dict[str, Any]:
    if n < 7:
        raise SystemExit("--paired requires n >= 7 kept pairs (plus 1 warm-up)")
    pairs = []
    for i in range(n + 1):
        sb = sample_system()
        r = subprocess.run(a_cmd, shell=True, capture_output=True, text=True, cwd=str(REPO))
        if r.returncode:
            raise RuntimeError(f"a-cmd failed: {r.stderr[:300] or r.stdout[:300]}")
        va = float([ln for ln in r.stdout.splitlines() if ln.strip()][-1])
        sm = sample_system()
        r = subprocess.run(b_cmd, shell=True, capture_output=True, text=True, cwd=str(REPO))
        if r.returncode:
            raise RuntimeError(f"b-cmd failed: {r.stderr[:300] or r.stdout[:300]}")
        vb = float([ln for ln in r.stdout.splitlines() if ln.strip()][-1])
        pairs.append({"i": i, "warm_up": i == 0, "a": va, "b": vb, "delta_b_minus_a": vb - va,
                      "system_before": sb, "system_mid": sm, "system_after": sample_system()})
    kept = [p for p in pairs if not p["warm_up"]]
    deltas = [p["delta_b_minus_a"] for p in kept]
    a_vals, b_vals = [p["a"] for p in kept], [p["b"] for p in kept]
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    zero = sum(1 for d in deltas if d == 0)
    n_eff, p_value = pos + neg, None
    if n_eff:
        k = min(pos, neg)
        p_value = min(1.0, 2.0 * sum(math.comb(n_eff, i) for i in range(k + 1)) / (2 ** n_eff))
    a_med, b_med = statistics.median(a_vals), statistics.median(b_vals)
    return {
        "schema": SCHEMA, "mode": "paired", "a_cmd": a_cmd, "b_cmd": b_cmd,
        "n_kept_pairs": len(kept), "warm_up_discarded": True, "pairs": pairs,
        "a_median": a_med, "b_median": b_med,
        "paired_median_delta_b_minus_a": statistics.median(deltas),
        "paired_delta_min_max": [min(deltas), max(deltas)],
        "ratio_of_medians_b_over_a": (b_med / a_med) if a_med else None,
        "sign_test": {"positive_deltas": pos, "negative_deltas": neg, "zero_deltas": zero,
                      "n_effective": n_eff, "two_sided_p": p_value,
                      "note": "Sign test on paired (B-A); contamination-resistant vs ratio-of-medians alone."},
        "system_at_end": sample_system(), "at": utc_now(),
    }

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hawking rebuild performance gate")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--a-cmd"), ap.add_argument("--b-cmd")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--gate", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--include-cold", action="store_true")
    ap.add_argument("--include-glm-tps", action="store_true")
    args = ap.parse_args(argv)
    if args.n < 2:
        print("--n must be >= 2 (1 warm-up + kept)", file=sys.stderr); return 2
    modes = sum(bool(x) for x in (args.list, args.capture, args.compare, args.paired))
    if modes != 1:
        ap.error("exactly one of --list / --capture / --compare / --paired")

    if args.list:
        inv = inventory(args.n, args.include_cold)
        print(f"perfgate inventory @ {REPO}")
        print(f"loadavg={os.getloadavg()} ncpu={os.cpu_count()} hawking_bin={find_hawking_bin()}")
        print(f"{'STATUS':<14}{'FAMILY':<16}{'NAME':<44}REASON")
        print("-" * 110)
        for r in inv:
            print(f"{r['status']:<14}{r['family']:<16}{r['name']:<44}{r['reason'][:70]}")
        counts: dict[str, int] = {}
        for r in inv:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        print("-" * 110); print("counts:", counts)
        return 0

    if args.capture:
        if args.n < 8:
            print(f"warning: protocol prefers n>=8 (1 warm-up + 7 kept); got {args.n}", file=sys.stderr)
        doc = do_capture(args.n, args.include_cold, args.include_glm_tps)
        out = args.out or (evidence_dir("rebuild") / "REBUILD_PERFORMANCE_BASELINE_MEASURED.json")
        out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {out}")
        print(json.dumps(doc["summary"], indent=2))
        return 0

    if args.compare:
        doc = do_compare(Path(args.compare[0]), Path(args.compare[1]), args.gate)
        text = json.dumps(doc, indent=2)
        if args.out:
            args.out.write_text(text + "\n")
        print(text)
        return 0 if doc["pass"] else 1

    if not args.a_cmd or not args.b_cmd:
        ap.error("--paired requires --a-cmd and --b-cmd")
    doc = do_paired(args.a_cmd, args.b_cmd, args.n)
    text = json.dumps(doc, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
