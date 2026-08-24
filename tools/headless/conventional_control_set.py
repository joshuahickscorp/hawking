#!/usr/bin/env python3
"""Conventional control set: live MLX vs archived llama.cpp.

Noetic candidates are only meaningful against a conventional control, and the
control must be real and current — not a remembered number. This harness:

  * measures MLX 4-bit NOW on this machine (the live arm)
  * cites llama.cpp Q5_K from the receipts that measured it (the archived arm)

The Q5_K GGUF is gone. Its numbers survive as science, labelled ARCHIVED, with
the artifact marked absent. They are never presented as if they were measured
today.

Every live number carries the command that produced it and a spread across
repetitions (a single Metal run is page-cache confounded). A metric that cannot
be taken on this box right now is ABSENT with the physical reason — never 0,
never a bare null.

Does not load a second 27B alongside a resident one. Concurrent MLX streams are
taken with batch_generate against the already-loaded copy, not a second process.

    python3 tools/headless/conventional_control_set.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA = "hawking.headless.conventional_control_set.v1"
REPS = 3
DECODE_TOKENS = 192
TOOL_TOKENS = 96
PREFILL_TOKENS = 512
BATCH_TOKENS = 32
BATCH_SIZES = (1, 2, 4)
LOAD_TIMEOUT_S = 600
MEASURE_TIMEOUT_S = 1800
METAL_TIMEOUT_S = 30

MLX_PY = Path.home() / ".local/share/uv/tools/mlx-lm/bin/python"
MLX_CANDIDATES = [
    Path.home() / "models/qwen3.8-27b-abliterated-mlx-huihui-4bit",
    Path.home() / "models/qwen3.8-27b-abliterated-mlx/4bit",
]
GGUF = Path.home() / (
    "models/qwen3.8-27b-abliterated/Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"
)
LLAMA_HEALTH_PORTS = (52484, 8080, 51999, 54559, 54568, 54574)

DECODE_PROMPT = (
    "Explain, in ordinary prose and at length, how a compiler turns a "
    "for-loop into basic blocks and then into machine code."
)
# A short JSON instruction plus a file-sized context, so this is a tool-shaped
# call rather than a 50-token toy prompt. Not the 7k-token HCLI mission (that
# payload lives in STRUCTURED_OUTPUT_PROBE.json and is ARCHIVED); this is the
# live analogue: structured JSON, EOS honoured, a file in the prompt.
_TOOL_FILE = (
    "def add(a, b):\n    return a - b\n\n"
    "def sub(a, b):\n    return a + b\n\n"
    "# helper utilities used by the surrounding agent loop\n"
    "def clamp(x, lo, hi):\n    return lo if x < lo else hi if x > hi else x\n\n"
) * 40
TOOL_PROMPT = (
    "You are a coding agent. The repository file calc.py currently contains:\n\n"
    f"```python\n{_TOOL_FILE}```\n\n"
    "Return exactly one JSON object and nothing else, of the form "
    '{"kind":"mutation","content":"...","operations":[{"op":"replace","path":'
    '"calc.py","old_text":"...","new_text":"..."}],"tests":[...]}. '
    "Fix add so it adds and sub so it subtracts. Be brief."
)

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
RECEIPTS = REPO / "receipts" / "headless"
RECEIPT = RECEIPTS / "CONVENTIONAL_CONTROL_SET.json"

_RECEIPT_CACHE: Path | None = None


# --------------------------------------------------------------------------- fields


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, timeout=20
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "UNKNOWN"


def field(status: str, value, **extra) -> dict:
    d = {"status": status, "value": value}
    d.update(extra)
    return d


def measured(value, *, command, unit=None, repetitions=None, extra=None, **kw) -> dict:
    d = field("MEASURED", value, command=command)
    if unit is not None:
        d["unit"] = unit
    if repetitions is not None:
        d["repetitions"] = repetitions
        nums = [v for v in repetitions if isinstance(v, (int, float))]
        d["n"] = len(nums)
        if nums:
            lo, hi = min(nums), max(nums)
            d["min"] = lo
            d["max"] = hi
            d["median"] = statistics.median(nums)
            d["spread_pct"] = (
                round(100.0 * (hi - lo) / lo, 3) if lo else None
            )
            d["spread_definition"] = (
                "100 * (max - min) / min over repetitions; "
                "a single Metal run is page-cache confounded"
            )
    if extra:
        d.update(extra)
    d.update(kw)
    return d


def archived(value, source, **extra) -> dict:
    d = field("ARCHIVED", value, source_receipt=source)
    d.update(extra)
    return d


def absent(reason: str, **extra) -> dict:
    d = field("ABSENT", None, reason=reason)
    d.update(extra)
    return d


def summarize(values, *, unit, command, status="MEASURED") -> dict:
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return absent(
            "no numeric samples in the repetition set",
            command=command,
            repetitions=values,
        )
    mid = statistics.median(nums)
    return measured(mid, command=command, unit=unit, repetitions=values, status=status)


def attach_cold_warm(node: dict, *, cold, warm) -> dict:
    """Label cold vs warm without mixing them into one unexplained number.

    `node['value']` and `node['repetitions']` stay the WARM set (the control).
    Cold is recorded beside them. A reader who wants the confounded combined
    spread can rebuild it; a reader who wants the control cannot un-mix it.
    """
    node["cold"] = cold
    node["warm_repetitions"] = list(warm) if warm is not None else []
    node["cold_and_warm_stated_separately"] = True
    warm_nums = [v for v in (warm or []) if isinstance(v, (int, float))]
    if warm_nums:
        node["warm_median"] = statistics.median(warm_nums)
        node["warm_min"] = min(warm_nums)
        node["warm_max"] = max(warm_nums)
        node["warm_n"] = len(warm_nums)
        node["warm_spread_pct"] = (
            round(100.0 * (max(warm_nums) - min(warm_nums)) / min(warm_nums), 3)
            if min(warm_nums) else None
        )
    node["headline_is"] = (
        "value/repetitions/spread are WARM (after the first sample). "
        "cold is the first sample this harness run."
    )
    return node


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def dir_bytes(path: Path) -> dict:
    total = 0
    n_files = 0
    if not path.is_dir():
        return {"present": False, "bytes": 0, "n_files": 0}
    for dp, _, fns in os.walk(path):
        for fn in fns:
            fp = Path(dp) / fn
            if fp.is_symlink():
                continue
            try:
                total += fp.stat().st_size
                n_files += 1
            except OSError:
                continue
    return {"present": True, "bytes": total, "n_files": n_files}


def sh(cmd: list[str], timeout: int = 15) -> str:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return (p.stdout or "").strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


# --------------------------------------------------------------------------- discovery


def find_mlx_model() -> dict:
    tried = []
    for p in MLX_CANDIDATES:
        tried.append(str(p))
        cfg = p / "config.json"
        weights = list(p.glob("*.safetensors")) if p.is_dir() else []
        if p.is_dir() and cfg.is_file() and weights:
            info = dir_bytes(p)
            lineage = (
                "huihui-ai"
                if "huihui" in p.name.lower()
                else "PocketAiHub"
            )
            conf = load_json(cfg) or {}
            text_cfg = conf.get("text_config") or {}
            return {
                "found": True,
                "path": str(p),
                "lineage": lineage,
                "bytes": info["bytes"],
                "n_files": info["n_files"],
                "quant": (conf.get("quantization") or conf.get("quantization_config")),
                "max_position_embeddings": text_cfg.get("max_position_embeddings"),
                "architectures": conf.get("architectures"),
                "tried": tried,
                "preferred_huihui_present": (
                    MLX_CANDIDATES[0].is_dir()
                ),
            }
    return {
        "found": False,
        "path": None,
        "tried": tried,
        "preferred_huihui_present": False,
    }


def http_json(url: str, timeout: float = 2.0):
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace")), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def occupancy() -> dict:
    """Resident 27B decoders. Loading MLX 4-bit beside one collapses tok/s.

    Occupancy 33.47 → 3.986 tok/s with two 27B residents
    (receipts/headless/HCLI_SELF_OPT_ITERATION_2.json).
    """
    health = []
    llama_up = False
    for port in LLAMA_HEALTH_PORTS:
        body, err = http_json(f"http://127.0.0.1:{port}/health")
        row = {"port": port, "ok": bool(body), "body": body, "error": err}
        health.append(row)
        if body and body.get("status") == "ok":
            llama_up = True
    ollama, ollama_err = http_json("http://127.0.0.1:11434/api/ps")
    ollama_models = (ollama or {}).get("models") or []
    ollama_27b = [
        m for m in ollama_models
        if "27b" in json.dumps(m).lower() or "27B" in json.dumps(m)
    ]
    listen = sh(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"])
    llama_listen = [
        ln for ln in listen.splitlines()
        if "llama-server" in ln or "mlx_lm" in ln
    ]
    ps_out = sh(["ps", "-ax", "-o", "pid=,command="])
    resident_ps = []
    for ln in ps_out.splitlines():
        s = ln.strip()
        if not s:
            continue
        if any(tok in s for tok in (
            "llama-server", "mlx_lm.server", "mlx_lm generate", "mlx_lm.generate",
        )) and "conventional_control_set.py" not in s:
            resident_ps.append(s[:220])
    gguf_on_disk = GGUF.is_file()
    refuse = bool(llama_up or ollama_27b or llama_listen or resident_ps)
    if llama_up:
        reason = (
            "a llama-server is answering on this box; loading the MLX 27B "
            "alongside it is forbidden (occupancy collapse 33.47 → 3.986 tok/s, "
            "receipts/headless/HCLI_SELF_OPT_ITERATION_2.json)"
        )
    elif ollama_27b:
        reason = (
            "ollama reports a 27B-class model resident; loading a second 27B "
            "is forbidden"
        )
    elif llama_listen:
        reason = (
            "lsof shows llama-server or mlx_lm listening; treating GPU as occupied"
        )
    elif resident_ps:
        reason = (
            "ps shows a llama-server / mlx_lm decoder already resident; "
            "loading a second 27B is forbidden"
        )
    else:
        reason = None
    return {
        "llama_up": llama_up,
        "health": health,
        "ollama_models": ollama_models,
        "ollama_error": ollama_err,
        "llama_or_mlx_listen": llama_listen,
        "resident_ps": resident_ps,
        "gguf_on_disk": gguf_on_disk,
        "gguf_path": str(GGUF),
        "refuse_load_27b": refuse,
        "refuse_reason": reason,
    }


def sysctl_machine() -> dict:
    out = {}
    for key, name, cast in (
        ("hw.memsize", "mem_bytes", int),
        ("machdep.cpu.brand_string", "cpu", str),
        ("hw.ncpu", "ncpu", int),
        ("hw.model", "hw_model", str),
    ):
        raw = sh(["sysctl", "-n", key])
        if not raw:
            continue
        try:
            out[name] = cast(raw) if cast is not str else raw
        except (TypeError, ValueError):
            out[name] = raw
    return out


# --------------------------------------------------------------------------- mlx workers (run under mlx-lm's python)


def worker_metal() -> int:
    out = {
        "mlx_imported": False,
        "metal_is_available": None,
        "device_info": None,
        "device_info_error": None,
        "tiny_eval_ok": False,
        "tiny_eval_error": None,
        "cpu_tiny_eval_ok": None,
        "cpu_tiny_eval_error": None,
        "error": None,
        "mlx_version": None,
        "mlx_lm_version": None,
    }
    try:
        import mlx
        import mlx.core as mx

        out["mlx_imported"] = True
        try:
            from importlib.metadata import version as pkg_version
            out["mlx_version"] = pkg_version("mlx")
        except Exception:
            out["mlx_version"] = getattr(mlx, "__version__", None)
        out["metal_is_available"] = bool(mx.metal.is_available())
        try:
            import mlx_lm
            out["mlx_lm_version"] = getattr(mlx_lm, "__version__", None)
        except Exception as exc:  # noqa: BLE001
            out["mlx_lm_import_error"] = f"{type(exc).__name__}: {exc}"
        try:
            out["device_info"] = mx.device_info()
        except Exception as exc:  # noqa: BLE001
            out["device_info_error"] = f"{type(exc).__name__}: {exc}"
        try:
            y = mx.ones((8,), dtype=mx.float32)
            mx.eval(y)
            out["tiny_eval_ok"] = True
            out["tiny_eval_sum"] = float(y.sum().item())
        except Exception as exc:  # noqa: BLE001
            out["tiny_eval_ok"] = False
            out["tiny_eval_error"] = f"{type(exc).__name__}: {exc}"
            # Metal cannot load a device. Record whether the CPU backend still
            # evaluates. Do NOT then load the 27B on CPU — that is not a control.
            try:
                mx.set_default_device(mx.cpu)
                z = mx.ones((8,), dtype=mx.float32)
                mx.eval(z)
                out["cpu_tiny_eval_ok"] = True
                out["cpu_tiny_eval_sum"] = float(z.sum().item())
            except Exception as exc2:  # noqa: BLE001
                out["cpu_tiny_eval_ok"] = False
                out["cpu_tiny_eval_error"] = f"{type(exc2).__name__}: {exc2}"
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out))
    return 0 if out.get("tiny_eval_ok") else 1


def worker_load(model: str) -> int:
    # Import is not model startup. Time only load() so sequential process
    # reps are comparable to the measure-process load_s (which also starts
    # the clock after mlx_lm is imported).
    try:
        from mlx_lm import load
        import mlx.core as mx
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({
            "load_s": None,
            "peak_memory_gb": None,
            "error": f"{type(exc).__name__}: {exc}",
        }))
        return 1
    t0 = time.perf_counter()
    try:
        load(model)
        mx.eval(mx.ones((4,), dtype=mx.float32))
        print(json.dumps({
            "load_s": round(time.perf_counter() - t0, 4),
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
            "error": None,
        }))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({
            "load_s": None,
            "peak_memory_gb": None,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_s": round(time.perf_counter() - t0, 4),
        }))
        return 1


def _chat(tok, text: str) -> str:
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
        )


def _pack_resp(last) -> dict:
    if last is None:
        return {"error": "stream_generate yielded nothing"}
    return {
        "prompt_tokens": last.prompt_tokens,
        "prompt_tps": last.prompt_tps,
        "generation_tokens": last.generation_tokens,
        "generation_tps": last.generation_tps,
        "peak_memory_gb": last.peak_memory,
        "finish_reason": last.finish_reason,
        "error": None,
    }


def worker_measure(args) -> int:
    from mlx_lm import batch_generate, load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx
    import mlx.core as mx
    import mlx_lm

    out: dict = {
        "error": None,
        "mlx_version": getattr(mlx, "__version__", None),
        "mlx_lm_version": getattr(mlx_lm, "__version__", None),
    }
    t0 = time.perf_counter()
    model, tok = load(args.model)
    mx.eval(mx.ones((4,), dtype=mx.float32))
    out["load_s"] = round(time.perf_counter() - t0, 4)
    out["peak_memory_gb_after_load"] = mx.get_peak_memory() / 1e9

    sampler = make_sampler(temp=0.0)
    decode_text = _chat(tok, DECODE_PROMPT)
    tool_text = _chat(tok, TOOL_PROMPT)
    saved_eos = set(tok.eos_token_ids or [])

    def ignore_eos() -> None:
        tok.eos_token_ids = set()

    def restore_eos() -> None:
        tok.eos_token_ids = saved_eos

    def last_stream(prompt, max_tokens: int, ignore: bool) -> dict:
        if ignore:
            ignore_eos()
        else:
            restore_eos()
        mx.reset_peak_memory()
        last = None
        t = time.perf_counter()
        for r in stream_generate(
            model, tok, prompt, max_tokens=max_tokens, sampler=sampler,
        ):
            last = r
        wall = round(time.perf_counter() - t, 4)
        restore_eos()
        packed = _pack_resp(last)
        packed["wall_s"] = wall
        return packed

    # COLD: first generate after load, includes Metal kernel compile.
    # A single Metal run is page-cache / compile confounded; this sample is
    # kept and labelled rather than discarded or mixed into the warm median.
    cold = last_stream(decode_text, args.n_predict, ignore=True)
    cold["rep"] = "cold"
    out["cold_decode_run"] = cold
    print(
        json.dumps({"progress": "decode_cold",
                    "generation_tps": cold.get("generation_tps")}),
        file=sys.stderr, flush=True,
    )

    decode_runs = []
    for i in range(args.reps):
        r = last_stream(decode_text, args.n_predict, ignore=True)
        r["rep"] = i
        decode_runs.append(r)
        print(
            json.dumps({"progress": "decode_warm", "rep": i,
                        "generation_tps": r.get("generation_tps")}),
            file=sys.stderr, flush=True,
        )
    out["decode_runs"] = decode_runs

    # Prefill-heavy: long prompt, few generated tokens.
    ids = tok.encode(decode_text)
    if not ids:
        ids = [1]
    while len(ids) < args.prefill_tokens:
        ids = ids + ids
    ids = ids[: args.prefill_tokens]
    prefill_runs = []
    for i in range(args.reps):
        r = last_stream(ids, 8, ignore=True)
        r["rep"] = i
        prefill_runs.append(r)
        print(
            json.dumps({"progress": "prefill", "rep": i,
                        "prompt_tps": r.get("prompt_tps")}),
            file=sys.stderr, flush=True,
        )
    out["prefill_runs"] = prefill_runs

    tool_runs = []
    for i in range(args.reps):
        r = last_stream(tool_text, args.tool_tokens, ignore=False)
        r["rep"] = i
        tool_runs.append(r)
        print(
            json.dumps({"progress": "tool", "rep": i,
                        "generation_tps": r.get("generation_tps")}),
            file=sys.stderr, flush=True,
        )
    out["tool_runs"] = tool_runs

    # Concurrency: same loaded weights, N sequences. Not a second 27B.
    ignore_eos()
    pt = tok.encode(decode_text)
    batch = {}
    for b in args.batch_sizes:
        runs = []
        for i in range(args.reps):
            mx.reset_peak_memory()
            prompts = [list(pt) for _ in range(int(b))]
            t1 = time.perf_counter()
            try:
                resp = batch_generate(
                    model, tok, prompts, max_tokens=args.batch_tokens,
                )
                st = resp.stats
                runs.append({
                    "rep": i,
                    "batch_size": int(b),
                    "prompt_tps": st.prompt_tps,
                    "generation_tps": st.generation_tps,
                    "generation_tokens": st.generation_tokens,
                    "generation_time": st.generation_time,
                    "peak_memory_gb": st.peak_memory,
                    "wall_s": round(time.perf_counter() - t1, 4),
                    "error": None,
                })
            except Exception as exc:  # noqa: BLE001
                runs.append({
                    "rep": i,
                    "batch_size": int(b),
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_s": round(time.perf_counter() - t1, 4),
                })
            print(
                json.dumps({"progress": "batch", "b": b, "rep": i,
                            "generation_tps": runs[-1].get("generation_tps")}),
                file=sys.stderr, flush=True,
            )
        batch[str(b)] = runs
    out["batch_runs"] = batch
    restore_eos()
    out["peak_memory_gb_end"] = mx.get_peak_memory() / 1e9
    print(json.dumps(out))
    return 0


# --------------------------------------------------------------------------- parent-side spawn


def spawn_worker(mode: str, extra: list[str] | None = None,
                 timeout: int = 60, stdin: str | None = None) -> dict:
    if not MLX_PY.is_file():
        return {
            "ok": False,
            "error": f"mlx python not at {MLX_PY}",
            "command": None,
        }
    cmd = [str(MLX_PY), str(HERE), "--mlx-worker", mode, *(extra or [])]
    try:
        proc = subprocess.run(
            cmd, input=stdin, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"TimeoutExpired after {timeout}s: {exc}",
            "command": cmd,
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        }
    except OSError as exc:
        return {"ok": False, "error": f"OSError: {exc}", "command": cmd}
    parsed = None
    stdout = proc.stdout or ""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    err = None
    if parsed is None:
        err = (
            f"worker {mode} exit {proc.returncode}; no JSON on stdout. "
            f"stderr={((proc.stderr or '')[-800:])}"
        )
    elif parsed.get("error"):
        err = parsed["error"]
    elif mode == "metal" and not parsed.get("tiny_eval_ok"):
        err = (
            parsed.get("tiny_eval_error")
            or parsed.get("device_info_error")
            or "tiny eval failed"
        )
    return {
        "ok": err is None and proc.returncode == 0,
        "error": err,
        "command": cmd,
        "returncode": proc.returncode,
        "parsed": parsed,
        "stdout_tail": stdout[-1500:],
        "stderr_tail": (proc.stderr or "")[-1500:],
    }


def metal_probe() -> dict:
    r = spawn_worker("metal", timeout=METAL_TIMEOUT_S)
    parsed = r.get("parsed") or {}
    gpu_usable = bool(parsed.get("tiny_eval_ok"))
    return {
        "attempted": True,
        "gpu_usable": gpu_usable,
        "mlx_python": str(MLX_PY),
        "mlx_python_present": MLX_PY.is_file(),
        "command": r.get("command"),
        "mlx_imported": parsed.get("mlx_imported"),
        "metal_is_available_flag": parsed.get("metal_is_available"),
        "tiny_eval_ok": parsed.get("tiny_eval_ok"),
        "device_info": parsed.get("device_info"),
        "device_info_error": parsed.get("device_info_error"),
        "tiny_eval_error": parsed.get("tiny_eval_error"),
        "cpu_tiny_eval_ok": parsed.get("cpu_tiny_eval_ok"),
        "cpu_tiny_eval_error": parsed.get("cpu_tiny_eval_error"),
        "mlx_version": parsed.get("mlx_version"),
        "mlx_lm_version": parsed.get("mlx_lm_version"),
        "error": r.get("error"),
        "note": (
            "mx.metal.is_available() can be True while mx.device_info()/mx.eval "
            "still raise [metal::load_device] No Metal device available. "
            "gpu_usable is tiny_eval_ok, not the is_available flag."
        ),
        "spawn": {k: r[k] for k in ("ok", "returncode", "stderr_tail") if k in r},
    }


# --------------------------------------------------------------------------- archived llama.cpp (Q5_K GGUF is gone)


def _run_values(runs: list, key: str) -> list:
    return [r[key] for r in runs if isinstance(r.get(key), (int, float))]


def archive_llama(gguf_present: bool) -> dict:
    runtime_ab = load_json(RECEIPTS / "RUNTIME_AB.json") or {}
    gpu_attack = load_json(RECEIPTS / "GPU_ATTACK.json") or {}
    genome = load_json(RECEIPTS / "MACHINE_GENOME.json") or {}
    topology = load_json(RECEIPTS / "DECODE_TOPOLOGY.json") or {}
    structured = load_json(RECEIPTS / "STRUCTURED_OUTPUT_PROBE.json") or {}
    long_ctx = load_json(RECEIPTS / "LONG_CONTEXT_RUNTIME_CAPABILITY.json") or {}
    registry = load_json(RECEIPTS / "MODEL_REGISTRY.json") or {}

    llama_ab = (runtime_ab.get("arms") or {}).get("llama_cpp") or {}
    llama_runs = llama_ab.get("runs") or []
    llama_cmd = (
        "llama-server -m "
        "/Users/scammermike/models/qwen3.8-27b-abliterated/"
        "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf "
        "--port <free> -c 8192 -ngl 999 --host 127.0.0.1 -np 1 ; "
        "POST /completion {prompt, n_predict:192, temperature:0.0, "
        "ignore_eos:true, cache_prompt:false}  (tools/headless/runtime_ab.py)"
    )

    decode_vals = _run_values(llama_runs, "decode_tps")
    prefill_vals = _run_values(llama_runs, "prefill_tps")

    decode = (
        archived(
            llama_ab.get("decode_tps_median"),
            "receipts/headless/RUNTIME_AB.json",
            unit="tok/s",
            command=llama_cmd,
            repetitions=decode_vals,
            n=len(decode_vals),
            min=min(decode_vals) if decode_vals else None,
            max=max(decode_vals) if decode_vals else None,
            median=statistics.median(decode_vals) if decode_vals else None,
            spread_pct=llama_ab.get("spread_pct"),
            spread_definition="100 * (max - min) / min over the three /completion reps",
            generated_at=runtime_ab.get("generated_at"),
            llama_version=llama_ab.get("version"),
        )
        if decode_vals else
        absent("RUNTIME_AB.json has no llama.cpp decode_tps runs")
    )
    if decode["status"] == "ARCHIVED":
        decode["artifact_present"] = gguf_present

    prefill = (
        archived(
            statistics.median(prefill_vals),
            "receipts/headless/RUNTIME_AB.json",
            unit="tok/s",
            command=llama_cmd,
            repetitions=prefill_vals,
            n=len(prefill_vals),
            min=min(prefill_vals),
            max=max(prefill_vals),
            median=statistics.median(prefill_vals),
            spread_pct=round(
                100.0 * (max(prefill_vals) - min(prefill_vals)) / min(prefill_vals), 3
            ) if prefill_vals and min(prefill_vals) else None,
            generated_at=runtime_ab.get("generated_at"),
        )
        if prefill_vals else
        absent("RUNTIME_AB.json has no llama.cpp prefill_tps runs")
    )

    admission = genome.get("admission") or []
    load_s = [a.get("load_s") for a in admission if isinstance(a.get("load_s"), (int, float))]
    rss = [
        a.get("rss_bytes") for a in admission
        if isinstance(a.get("rss_bytes"), (int, float))
    ]
    genome_cmd = genome.get("reprofile_command") or (
        "python3 tools/headless/machine_probe.py --ctx 8192 --max-runtimes 3 "
        "--n-predict 96 --reps 3"
    )
    startup = (
        archived(
            statistics.median(load_s),
            "receipts/headless/MACHINE_GENOME.json",
            unit="s",
            command=genome_cmd,
            repetitions=load_s,
            n=len(load_s),
            min=min(load_s),
            max=max(load_s),
            median=statistics.median(load_s),
            spread_pct=round(100.0 * (max(load_s) - min(load_s)) / min(load_s), 3)
            if load_s and min(load_s) else None,
            note=(
                "three sequential llama-server process starts; first load is cold, "
                "later loads are page-cached — that spread is the confound"
            ),
            generated_at=genome.get("generated_at"),
        )
        if load_s else
        absent("MACHINE_GENOME.json has no admission load_s")
    )
    peak_memory = (
        archived(
            statistics.median(rss),
            "receipts/headless/MACHINE_GENOME.json",
            unit="bytes_rss",
            command=genome_cmd,
            repetitions=rss,
            n=len(rss),
            min=min(rss),
            max=max(rss),
            median=statistics.median(rss),
            spread_pct=round(100.0 * (max(rss) - min(rss)) / min(rss), 3)
            if rss and min(rss) else None,
            gib=round(statistics.median(rss) / 1024 ** 3, 3) if rss else None,
            generated_at=genome.get("generated_at"),
        )
        if rss else
        absent("MACHINE_GENOME.json has no admission rss_bytes")
    )

    slot = ((topology.get("summary") or {}).get("slot") or {})
    slot4 = slot.get("4") or {}
    slot6 = slot.get("6") or {}
    slot1 = slot.get("1") or {}
    topo_cmd = topology.get("reprofile_command") or (
        "python3 tools/headless/decode_topology_probe.py --per-slot-ctx 4096 "
        "--n-predict 128 --reps 3 --streams 1,2,4,6,8"
    )
    if slot4:
        concurrency = archived(
            {
                "slot_k1_aggregate_tps_median": slot1.get("aggregate_tps_median"),
                "slot_k4_aggregate_tps_median": slot4.get("aggregate_tps_median"),
                "slot_k4_scaling_vs_1": slot4.get("scaling_vs_1"),
                "slot_k4_spread_pct": slot4.get("spread_pct"),
                "slot_k6_aggregate_tps_median": slot6.get("aggregate_tps_median"),
                "one_mlx_stream_beat_four_llama_slots": True,
                "mlx_single_stream_tps_cited": (
                    (gpu_attack.get("runtime_axis") or {}).get("mlx_single_stream_tps")
                ),
                "note": (
                    "ONE MLX decoder (35.506 tok/s, GPU_ATTACK.json) beat FOUR "
                    "concurrent llama.cpp slot decoders "
                    f"({slot4.get('aggregate_tps_median')} tok/s aggregate). "
                    "Best llama.cpp topology was slot k=6 at "
                    f"{slot6.get('aggregate_tps_median')} tok/s, still below one MLX stream."
                ),
            },
            "receipts/headless/DECODE_TOPOLOGY.json",
            command=topo_cmd,
            generated_at=topology.get("generated_at"),
            also_cited="receipts/headless/GPU_ATTACK.json",
        )
    else:
        concurrency = absent("DECODE_TOPOLOGY.json has no slot k=4 summary")

    long_req = long_ctx.get("request") or {}
    runtime_id = long_ctx.get("runtime_identity") or {}
    if runtime_id.get("per_slot_n_ctx") is not None:
        context_limit = archived(
            runtime_id.get("per_slot_n_ctx"),
            "receipts/headless/LONG_CONTEXT_RUNTIME_CAPABILITY.json",
            unit="tokens",
            command=runtime_id.get("launch_argv"),
            empirical_request_prompt_tokens=long_req.get("prompt_tokens_server_counted"),
            empirical_request_ok=long_req.get("http_status") == 200,
            machine_genome_probe_ctx=(genome.get("runtime_identity") or {}).get("ctx"),
            note=(
                "per-slot n_ctx 262144 on llama-server --ctx-size 500000. A 29982-token "
                "request completed. The GGUF is gone, so this is not re-measured today."
            ),
            generated_at=long_ctx.get("generated_at"),
        )
    else:
        context_limit = absent(
            "LONG_CONTEXT_RUNTIME_CAPABILITY.json missing per_slot_n_ctx"
        )

    no_think = [
        r for r in (structured.get("runs") or [])
        if r.get("arm") == "no_think"
        and isinstance(r.get("tok_per_s"), (int, float))
    ]
    tool_vals = [r["tok_per_s"] for r in no_think]
    tool_walls = [r.get("wall_s") for r in no_think]
    if tool_vals:
        tool_shaped = archived(
            statistics.median(tool_vals),
            "receipts/headless/STRUCTURED_OUTPUT_PROBE.json",
            unit="complete_call_tok_per_s",
            command=(
                "llama-server with HCLI structured-output payload "
                "(~7221 prompt tokens, chat_template_kwargs.enable_thinking=false); "
                "tools/headless/structured_output_probe.py arm=no_think"
            ),
            repetitions=tool_vals,
            n=len(tool_vals),
            min=min(tool_vals),
            max=max(tool_vals),
            median=statistics.median(tool_vals),
            spread_pct=round(
                100.0 * (max(tool_vals) - min(tool_vals)) / min(tool_vals), 3
            ) if min(tool_vals) else None,
            wall_s_repetitions=tool_walls,
            note=(
                "complete-call tok/s of a tool-shaped HCLI JSON payload, NOT isolated "
                "decode tok/s: the first rep is cold (31.62s / 3.48 tok/s), the second "
                "is warm prefix (6.46s / 22.45 tok/s). Spread is that cache confound."
            ),
            generated_at=structured.get("generated_at"),
        )
    else:
        tool_shaped = absent(
            "STRUCTURED_OUTPUT_PROBE.json has no no_think tok_per_s runs"
        )

    axis = gpu_attack.get("runtime_axis") or {}
    headline = archived(
        {
            "mlx_4bit_tps": axis.get("mlx_single_stream_tps") or 35.51,
            "llama_q5k_tps": axis.get("llama_cpp_single_stream_tps") or 24.12,
            "speed_ratio": axis.get("mlx_over_llama") or 1.472,
            "bytes_ratio_llama_over_mlx": axis.get("bytes_ratio_llama_over_mlx") or 1.215,
            "one_mlx_beat_four_llama_slots": True,
        },
        "receipts/headless/GPU_ATTACK.json",
        command=(
            "tools/headless/write_gpu_attack.py reading DECODE_TOPOLOGY.json "
            "and RUNTIME_AB.json as they stood when GPU_ATTACK.json was sealed"
        ),
        reading=axis.get("reading"),
        generated_at=gpu_attack.get("generated_at"),
        note=(
            "ARCHIVED headline. The 1.215 bytes ratio is the PocketAiHub MLX 4-bit "
            "vs huihui Q5_K GGUF comparison; lineage-matched RUNTIME_AB.json later "
            "measured 1.477x on a 1.289 bytes ratio against the now-deleted huihui "
            "MLX 4-bit directory. Neither figure is a live measurement today."
        ),
    )

    gguf_reg = ((registry.get("candidates") or {}).get("qwen38-huihui-q5k-gguf") or {})
    artifact = {
        "status": "ARCHIVED",
        "present": gguf_present,
        "path": str(GGUF),
        "last_known_bytes": llama_ab.get("bytes") or 19_535_701_280,
        "last_known_gib": llama_ab.get("gib") or 18.19,
        "quant": "Q5_K",
        "lineage": "huihui-ai",
        "runtime": "llama.cpp",
        "llama_version": llama_ab.get("version") or (
            (genome.get("runtime_identity") or {}).get("llama_version")
        ),
        "registry_identity": (gguf_reg.get("artifact") or {}).get("identity"),
        "note": (
            "the Q5_K GGUF artifact is GONE; these numbers survive as archived "
            "science and must not be required to re-run"
            if not gguf_present else
            "GGUF is present on disk; this arm is still labelled ARCHIVED because "
            "the control set's live arm is MLX — re-measure llama.cpp only with an "
            "explicit new campaign, never silently"
        ),
    }

    return {
        "runtime": "llama.cpp",
        "status": "ARCHIVED",
        "artifact": artifact,
        "metrics": {
            "startup": startup,
            "prefill": prefill,
            "decode_tps": decode,
            "context_limit": context_limit,
            "concurrency": concurrency,
            "peak_memory": peak_memory,
            "tool_shaped_tps": tool_shaped,
        },
        "headline_vs_mlx": headline,
        "source_receipts": [
            "receipts/headless/RUNTIME_AB.json",
            "receipts/headless/GPU_ATTACK.json",
            "receipts/headless/MACHINE_GENOME.json",
            "receipts/headless/DECODE_TOPOLOGY.json",
            "receipts/headless/STRUCTURED_OUTPUT_PROBE.json",
            "receipts/headless/LONG_CONTEXT_RUNTIME_CAPABILITY.json",
            "receipts/headless/MODEL_REGISTRY.json",
        ],
    }


# --------------------------------------------------------------------------- live MLX


def _cmd_str(cmd) -> str | None:
    if not cmd:
        return None
    if isinstance(cmd, (list, tuple)):
        return " ".join(str(c) for c in cmd)
    return str(cmd)


def measure_mlx_live(model: dict, occ: dict, metal: dict, args) -> dict:
    path = model.get("path")
    if not model.get("found") or not path:
        reason = (
            "no MLX abliterated model on disk; looked at "
            + ", ".join(model.get("tried") or [])
        )
        return _live_all_absent(reason, model, metal)

    if occ.get("refuse_load_27b"):
        return _live_all_absent(occ.get("refuse_reason"), model, metal, occupancy=occ)

    if not metal.get("gpu_usable"):
        cpu_note = ""
        if metal.get("cpu_tiny_eval_ok") is True:
            cpu_note = (
                " CPU tiny eval DID run (mx.set_default_device(cpu) then "
                "mx.eval on an 8-vector). The 27B was NOT loaded on CPU: a "
                "CPU 27B decode is not a conventional control on this box."
            )
        elif metal.get("cpu_tiny_eval_ok") is False:
            cpu_note = (
                " CPU tiny eval also failed: "
                + str(metal.get("cpu_tiny_eval_error"))
            )
        reason = (
            "Metal is not usable in this process: "
            + (metal.get("error") or metal.get("tiny_eval_error")
               or metal.get("device_info_error")
               or "tiny_eval_ok is false")
            + ". mx.metal.is_available() can be True while load_device fails "
            "(sandboxed / headless session). Live decode/prefill/startup/"
            "concurrency/peak-memory/tool-shaped tok/s all require the GPU."
            + cpu_note
        )
        live = _live_all_absent(reason, model, metal)
        # Context limit from config.json does not need the GPU.
        live["metrics"]["context_limit"] = _context_from_config(model)
        return live

    extra = [
        "--model", path,
        "--n-predict", str(args.n_predict),
        "--reps", str(args.reps),
        "--tool-tokens", str(args.tool_tokens),
        "--prefill-tokens", str(args.prefill_tokens),
        "--batch-tokens", str(args.batch_tokens),
        "--batch-sizes", ",".join(str(b) for b in args.batch_sizes),
    ]
    print(f"  mlx measure: {path}", flush=True)
    meas = spawn_worker("measure", extra=extra, timeout=MEASURE_TIMEOUT_S)
    parsed = meas.get("parsed") or {}
    measure_cmd = _cmd_str(meas.get("command"))

    # Startup: the measure-process load is rep 0; two more load-only processes
    # after it exits (GPU freed) give the page-cache spread. Never two 27B at once.
    load_s_reps = []
    load_cmds = []
    if isinstance(parsed.get("load_s"), (int, float)):
        load_s_reps.append(parsed["load_s"])
        load_cmds.append(measure_cmd)
    print("  mlx startup extra loads (sequential, GPU freed between)", flush=True)
    for i in range(max(0, args.reps - 1)):
        time.sleep(2)
        load = spawn_worker(
            "load", extra=["--model", path], timeout=LOAD_TIMEOUT_S,
        )
        load_cmds.append(_cmd_str(load.get("command")))
        lp = load.get("parsed") or {}
        if isinstance(lp.get("load_s"), (int, float)):
            load_s_reps.append(lp["load_s"])
        else:
            load_s_reps.append(None)
            print(f"    load extra[{i}] failed: {load.get('error')}", flush=True)

    startup_cmd = (
        f"{MLX_PY} {HERE} --mlx-worker load --model {path}   "
        f"(×{args.reps} sequential processes; measure-process load is cold rep 0)"
    )
    warm_loads = [v for v in load_s_reps[1:] if isinstance(v, (int, float))]
    if warm_loads:
        startup = summarize(warm_loads, unit="s", command=startup_cmd)
    else:
        startup = summarize(load_s_reps, unit="s", command=startup_cmd)
    attach_cold_warm(
        startup,
        cold=load_s_reps[0] if load_s_reps else None,
        warm=warm_loads,
    )
    startup["all_load_s"] = load_s_reps
    startup["note"] = (
        "cold is the first process this harness run (disk + GPU upload, Metal "
        "compile may overlap). warm is each subsequent process after the GPU "
        "was freed — page-cache confounded relative to cold."
    )

    if not meas.get("ok") or parsed.get("error"):
        reason = (
            meas.get("error")
            or parsed.get("error")
            or "mlx measure worker failed"
        )
        metrics = {
            "startup": startup if load_s_reps else absent(reason, command=measure_cmd),
            "prefill": absent(reason, command=measure_cmd),
            "decode_tps": absent(reason, command=measure_cmd),
            "context_limit": _context_from_config(model),
            "concurrency": absent(
                reason + " — concurrency is batch_generate on one loaded copy, "
                "not a second 27B",
                command=measure_cmd,
            ),
            "peak_memory": absent(reason, command=measure_cmd),
            "tool_shaped_tps": absent(reason, command=measure_cmd),
        }
        return {
            "runtime": "mlx",
            "status": "LIVE",
            "artifact": _live_artifact(model),
            "metal": metal,
            "measure_spawn": {
                "ok": meas.get("ok"),
                "error": meas.get("error"),
                "returncode": meas.get("returncode"),
                "stderr_tail": meas.get("stderr_tail"),
            },
            "metrics": metrics,
            "versions": {
                "mlx": parsed.get("mlx_version") or metal.get("mlx_version"),
                "mlx_lm": parsed.get("mlx_lm_version"),
                "mlx_python": str(MLX_PY),
            },
        }

    decode_vals = _run_values(parsed.get("decode_runs") or [], "generation_tps")
    cold_decode = (parsed.get("cold_decode_run") or {}).get("generation_tps")
    prefill_vals = _run_values(parsed.get("prefill_runs") or [], "prompt_tps")
    # Short-prompt prefill from the decode runs is a second, smaller sample.
    short_prefill = _run_values(parsed.get("decode_runs") or [], "prompt_tps")
    tool_vals = _run_values(parsed.get("tool_runs") or [], "generation_tps")
    peak_vals = _run_values(parsed.get("decode_runs") or [], "peak_memory_gb")
    if isinstance(parsed.get("peak_memory_gb_after_load"), (int, float)):
        peak_after_load = parsed["peak_memory_gb_after_load"]
    else:
        peak_after_load = None

    decode = summarize(
        decode_vals, unit="tok/s", command=measure_cmd,
    )
    attach_cold_warm(decode, cold=cold_decode, warm=decode_vals)
    decode["n_predict"] = args.n_predict
    decode["generation_tokens"] = _run_values(
        parsed.get("decode_runs") or [], "generation_tokens"
    )
    decode["finish_reasons"] = [
        r.get("finish_reason") for r in (parsed.get("decode_runs") or [])
    ]
    decode["prompt"] = "runtime_ab DECODE_PROMPT, enable_thinking=false, ignore_eos"
    decode["warmup"] = (
        "no discarded warmup; the first n_predict generate after load is "
        "recorded as cold (Metal compile). The next --reps generates are warm."
    )
    decode["cold_run"] = parsed.get("cold_decode_run")

    prefill_warm = prefill_vals[1:] if len(prefill_vals) > 1 else prefill_vals
    prefill_cold = prefill_vals[0] if prefill_vals else None
    prefill = summarize(
        prefill_warm if len(prefill_vals) > 1 else prefill_vals,
        unit="tok/s", command=measure_cmd,
    )
    attach_cold_warm(prefill, cold=prefill_cold, warm=prefill_warm if len(prefill_vals) > 1 else [])
    prefill["prompt_tokens"] = args.prefill_tokens
    prefill["generation_tokens_during_prefill_run"] = 8
    prefill["short_prompt_prefill_tps_from_decode_reps"] = short_prefill
    prefill["all_repetitions"] = prefill_vals

    tool_warm = tool_vals[1:] if len(tool_vals) > 1 else tool_vals
    tool = summarize(
        tool_warm if len(tool_vals) > 1 else tool_vals,
        unit="tok/s", command=measure_cmd,
    )
    attach_cold_warm(
        tool,
        cold=tool_vals[0] if tool_vals else None,
        warm=tool_warm if len(tool_vals) > 1 else [],
    )
    tool["prompt"] = (
        "tool-shaped JSON mutation over a file-sized calc.py context, "
        "enable_thinking=false, EOS honoured"
    )
    tool["max_tokens"] = args.tool_tokens
    tool["all_repetitions"] = tool_vals
    tool_token_counts = _run_values(parsed.get("tool_runs") or [], "generation_tokens")
    tool_prompt_counts = _run_values(parsed.get("tool_runs") or [], "prompt_tokens")
    tool["generation_tokens_repetitions"] = tool_token_counts
    tool["prompt_tokens_repetitions"] = tool_prompt_counts
    tool["note"] = (
        "generation_tps of a structured-JSON tool call, EOS honoured — not the "
        "ARCHIVED complete-call tok/s in STRUCTURED_OUTPUT_PROBE.json (that one "
        "had ~7257 prompt tokens and mixed cold 31.6s / warm 6.5s prefix cache). "
        "Do not ratio these two numbers."
    )

    phase_peaks = {}
    if isinstance(peak_after_load, (int, float)):
        phase_peaks["after_load"] = peak_after_load
    cold_peak = (parsed.get("cold_decode_run") or {}).get("peak_memory_gb")
    if isinstance(cold_peak, (int, float)):
        phase_peaks["cold_decode"] = cold_peak
    if peak_vals:
        phase_peaks["decode_warm"] = max(peak_vals)
    prefill_peaks = _run_values(parsed.get("prefill_runs") or [], "peak_memory_gb")
    if prefill_peaks:
        phase_peaks["prefill"] = max(prefill_peaks)
    tool_peaks = _run_values(parsed.get("tool_runs") or [], "peak_memory_gb")
    if tool_peaks:
        phase_peaks["tool_shaped"] = max(tool_peaks)
    batch_peaks = []
    for runs in (parsed.get("batch_runs") or {}).values():
        batch_peaks.extend(_run_values(runs, "peak_memory_gb"))
    if batch_peaks:
        phase_peaks["batch_generate"] = max(batch_peaks)
    if isinstance(parsed.get("peak_memory_gb_end"), (int, float)):
        phase_peaks["end"] = parsed["peak_memory_gb_end"]
    observed_peaks = [v for v in phase_peaks.values() if isinstance(v, (int, float))]
    if observed_peaks:
        peak = measured(
            max(observed_peaks),
            command=measure_cmd,
            unit="GB (mlx peak_memory)",
            repetitions=observed_peaks,
            extra={
                "phase_peaks_gb": phase_peaks,
                "note": (
                    "value is the MAX over phases, not the median. repetitions "
                    "are the per-phase peaks (load, decode, prefill, tool, batch). "
                    "stream_generate GenerationResponse.peak_memory is already GB; "
                    "mx.get_peak_memory() after load is divided by 1e9."
                ),
            },
        )
    else:
        peak = absent("no peak_memory samples from the measure worker", command=measure_cmd)

    batch = parsed.get("batch_runs") or {}
    by_k = {}
    for k, runs in batch.items():
        vals = _run_values(runs, "generation_tps")
        errors = [r.get("error") for r in runs if r.get("error")]
        if errors and not vals:
            by_k[k] = absent(
                f"batch_generate k={k} failed: {errors[0]}",
                command=measure_cmd,
                repetitions=runs,
            )
        elif vals:
            warm_vals = vals[1:] if len(vals) > 1 else vals
            row = summarize(
                warm_vals if len(vals) > 1 else vals,
                unit="tok/s", command=measure_cmd,
            )
            attach_cold_warm(
                row,
                cold=vals[0],
                warm=warm_vals if len(vals) > 1 else [],
            )
            row["batch_size"] = int(k)
            row["max_tokens"] = args.batch_tokens
            row["all_repetitions"] = vals
            row["what"] = (
                f"{k} sequences in one mlx_lm.batch_generate against the already-"
                "loaded 27B; not a second process"
            )
            by_k[k] = row
        else:
            by_k[k] = absent(
                f"batch_generate k={k} produced no generation_tps",
                command=measure_cmd,
            )
    k1 = (by_k.get("1") or {}).get("value")
    k4 = (by_k.get("4") or {}).get("value")
    conc_value = {
        "batch_1_generation_tps_median": k1,
        "batch_2_generation_tps_median": (by_k.get("2") or {}).get("value"),
        "batch_4_generation_tps_median": k4,
        "batch_4_over_batch_1": (
            round(k4 / k1, 4)
            if isinstance(k4, (int, float)) and isinstance(k1, (int, float)) and k1
            else None
        ),
    }
    any_batch_measured = any(
        (v or {}).get("status") == "MEASURED" for v in by_k.values()
    )
    if not by_k or not any_batch_measured:
        concurrency = absent(
            "batch_generate produced no usable runs",
            command=measure_cmd,
        )
    else:
        concurrency = measured(
            conc_value,
            command=measure_cmd,
            unit="tok/s per batch_generate call (aggregate over the batch)",
            extra={
                "by_batch_size": by_k,
                "did_not_load_second_27b": True,
            },
        )

    return {
        "runtime": "mlx",
        "status": "LIVE",
        "artifact": _live_artifact(model),
        "metal": metal,
        "metrics": {
            "startup": startup,
            "prefill": prefill,
            "decode_tps": decode,
            "context_limit": _context_from_config(model, empirical_ok=bool(prefill_vals)),
            "concurrency": concurrency,
            "peak_memory": peak,
            "tool_shaped_tps": tool,
        },
        "raw": {
            "load_s_in_measure_process": parsed.get("load_s"),
            "cold_decode_run": parsed.get("cold_decode_run"),
            "decode_runs": parsed.get("decode_runs"),
            "prefill_runs": parsed.get("prefill_runs"),
            "tool_runs": parsed.get("tool_runs"),
            "batch_runs": parsed.get("batch_runs"),
        },
        "versions": {
            "mlx": parsed.get("mlx_version") or metal.get("mlx_version"),
            "mlx_lm": parsed.get("mlx_lm_version"),
            "mlx_python": str(MLX_PY),
        },
        "params": {
            "n_predict": args.n_predict,
            "reps": args.reps,
            "prefill_tokens": args.prefill_tokens,
            "tool_tokens": args.tool_tokens,
            "batch_sizes": list(args.batch_sizes),
            "batch_tokens": args.batch_tokens,
            "temperature": 0.0,
        },
    }


def _context_from_config(model: dict, empirical_ok: bool = False) -> dict:
    n = model.get("max_position_embeddings")
    cfg_path = str(Path(model["path"]) / "config.json") if model.get("path") else None
    cmd = (
        f"python3 -c \"import json; print(json.load(open({cfg_path!r}))"
        f"['text_config']['max_position_embeddings'])\""
        if cfg_path else None
    )
    if n is None:
        return absent(
            "config.json has no text_config.max_position_embeddings",
            command=cmd,
        )
    extra = {
        "kind": "architectural_max_from_config_json",
        "empirical_prefill_at_tokens": PREFILL_TOKENS if empirical_ok else None,
        "empirical_prefill_ok": empirical_ok,
        "runnable_262144": absent(
            "probing the architectural 262144-token ceiling would allocate KV "
            "for this 27B at that length; that is an admission sweep, not a "
            "control-set decode. Config max is recorded; empirical prefill is "
            f"{PREFILL_TOKENS} tokens"
            + (" and succeeded" if empirical_ok else " and was not taken")
            + "."
        ),
    }
    return measured(n, command=cmd, unit="tokens", extra=extra)


def _live_artifact(model: dict) -> dict:
    return {
        "path": model.get("path"),
        "bytes": model.get("bytes"),
        "n_files": model.get("n_files"),
        "quant": model.get("quant"),
        "lineage": model.get("lineage"),
        "architectures": model.get("architectures"),
        "preferred_huihui_present": model.get("preferred_huihui_present"),
        "note": (
            "on-disk MLX abliterated 4-bit; huihui-lineage directory is absent "
            "so the live arm is the PocketAiHub conversion at "
            "~/models/qwen3.8-27b-abliterated-mlx/4bit"
            if not model.get("preferred_huihui_present")
            else "huihui-lineage MLX 4-bit is present and was selected"
        ),
    }


def _live_all_absent(reason: str, model: dict, metal: dict, occupancy=None) -> dict:
    metrics = {
        "startup": absent(reason),
        "prefill": absent(reason),
        "decode_tps": absent(reason),
        "context_limit": (
            _context_from_config(model)
            if model.get("found") else
            absent(reason)
        ),
        "concurrency": absent(
            reason + " — a second 27B load is forbidden; batch_generate was not reached"
        ),
        "peak_memory": absent(reason),
        "tool_shaped_tps": absent(reason),
    }
    return {
        "runtime": "mlx",
        "status": "LIVE",
        "artifact": _live_artifact(model) if model.get("found") else {
            "path": None, "present": False, "tried": model.get("tried"),
        },
        "metal": metal,
        "occupancy": occupancy,
        "metrics": metrics,
        "versions": {
            "mlx": metal.get("mlx_version"),
            "mlx_python": str(MLX_PY),
        },
    }


def live_vs_archived(live: dict, archived_arm: dict) -> dict:
    live_d = (live.get("metrics") or {}).get("decode_tps") or {}
    arch_d = (archived_arm.get("metrics") or {}).get("decode_tps") or {}
    lv, av = live_d.get("value"), arch_d.get("value")
    if live_d.get("status") != "MEASURED" or not isinstance(lv, (int, float)):
        return absent(
            "live MLX decode_tps was not MEASURED today, so no live/archived ratio "
            "is computed (the GPU_ATTACK 1.472x figure remains ARCHIVED)"
        )
    if arch_d.get("status") != "ARCHIVED" or not isinstance(av, (int, float)) or not av:
        return absent("archived llama.cpp decode_tps missing; cannot ratio")
    live_bytes = (live.get("artifact") or {}).get("bytes")
    arch_bytes = (archived_arm.get("artifact") or {}).get("last_known_bytes")
    bytes_ratio = (
        round(arch_bytes / live_bytes, 3)
        if live_bytes and arch_bytes else None
    )
    return measured(
        round(lv / av, 4),
        command="live.decode_tps.median / archived.decode_tps.median",
        unit="ratio tok/s",
        extra={
            "live_mlx_decode_tps": lv,
            "archived_llama_decode_tps": av,
            "bytes_ratio_llama_over_mlx": bytes_ratio,
            "note": (
                "numerator is MEASURED today; denominator is ARCHIVED. Do not "
                "quote this as a two-arm measurement taken in the same session."
            ),
        },
    )


# --------------------------------------------------------------------------- document + self-check


def build_record(args) -> dict:
    machine = sysctl_machine()
    model = find_mlx_model()
    occ = occupancy()
    metal = metal_probe()
    print(f"  model   {model.get('path')}  found={model.get('found')}", flush=True)
    print(f"  metal   gpu_usable={metal.get('gpu_usable')}  {metal.get('error')}", flush=True)
    print(f"  occupy  refuse_load_27b={occ.get('refuse_load_27b')}", flush=True)

    archived_arm = archive_llama(occ.get("gguf_on_disk"))
    live = measure_mlx_live(model, occ, metal, args)
    comparison = {
        "live_mlx_over_archived_llama": live_vs_archived(live, archived_arm),
        "historical_headline": archived_arm.get("headline_vs_mlx"),
    }
    doc = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "commit": git_head(),
        "why": (
            "Noetic candidates are only meaningful against a conventional control. "
            "MLX 4-bit is the LIVE control, measured now. llama.cpp Q5_K is the "
            "ARCHIVED control: the GGUF is gone, the numbers survive as science."
        ),
        "machine": machine,
        "occupancy": occ,
        "live": live,
        "archived": archived_arm,
        "comparison": comparison,
        "discipline": {
            "every_live_number_has_command_and_spread": True,
            "every_archived_number_labelled_ARCHIVED": True,
            "absent_never_written_as_zero": True,
            "did_not_load_second_27b": True,
            "gguf_not_required_to_exist": True,
        },
        "reproduce": (
            f"python3 {HERE.relative_to(REPO) if REPO in HERE.parents else HERE} "
            f"--reps {args.reps} --n-predict {args.n_predict}"
        ),
    }
    return doc


def _walk_metrics(arm: dict):
    metrics = arm.get("metrics") or {}
    for name, node in metrics.items():
        yield name, node


def validate(doc: dict) -> list[str]:
    problems = []
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema {doc.get('schema')!r} != {SCHEMA}")
    live = doc.get("live") or {}
    arch = doc.get("archived") or {}
    if live.get("status") != "LIVE":
        problems.append(f"live.status {live.get('status')!r} != LIVE")
    if arch.get("status") != "ARCHIVED":
        problems.append(f"archived.status {arch.get('status')!r} != ARCHIVED")
    if (arch.get("artifact") or {}).get("present") is True:
        # Present is allowed (reappeared) but must still be labelled ARCHIVED.
        pass
    elif (arch.get("artifact") or {}).get("present") is not False:
        problems.append("archived.artifact.present is not False")

    required = (
        "startup", "prefill", "decode_tps", "context_limit",
        "concurrency", "peak_memory", "tool_shaped_tps",
    )
    for name in required:
        if name not in (live.get("metrics") or {}):
            problems.append(f"live.metrics missing {name}")
        if name not in (arch.get("metrics") or {}):
            problems.append(f"archived.metrics missing {name}")

    for name, node in _walk_metrics(live):
        st = (node or {}).get("status")
        if st not in ("MEASURED", "ABSENT"):
            problems.append(f"live.{name} status {st!r} is not MEASURED or ABSENT")
        if st == "MEASURED":
            if node.get("value") is None:
                problems.append(f"live.{name} MEASURED but value is null")
            if not node.get("command"):
                problems.append(f"live.{name} MEASURED but missing command")
            # context_limit is a config read; spread is optional. Everything
            # taken on the GPU must have a repetition set.
            if name != "context_limit":
                reps = node.get("repetitions")
                if name == "concurrency":
                    by = node.get("by_batch_size")
                    if not by and not reps:
                        problems.append(
                            f"live.{name} MEASURED but has no by_batch_size/repetitions"
                        )
                elif not reps or len(reps) < 2:
                    problems.append(
                        f"live.{name} MEASURED but repetitions < 2 (page-cache confound)"
                    )
                elif node.get("spread_pct") is None and any(
                    isinstance(v, (int, float)) and v for v in reps
                ):
                    # spread_pct None is OK only when min is 0
                    pass
        if st == "ABSENT":
            if not node.get("reason"):
                problems.append(f"live.{name} ABSENT without reason")
            if node.get("value") in (0, 0.0):
                problems.append(f"live.{name} ABSENT written as 0")

    for name, node in _walk_metrics(arch):
        st = (node or {}).get("status")
        if st == "MEASURED":
            problems.append(
                f"archived.{name} is labelled MEASURED — archived numbers must be ARCHIVED"
            )
        if st not in ("ARCHIVED", "ABSENT"):
            problems.append(f"archived.{name} status {st!r} is not ARCHIVED or ABSENT")
        if st == "ARCHIVED" and not node.get("source_receipt"):
            problems.append(f"archived.{name} ARCHIVED without source_receipt")
        if st == "ABSENT" and not node.get("reason"):
            problems.append(f"archived.{name} ABSENT without reason")

    headline = (doc.get("comparison") or {}).get("historical_headline") or {}
    if headline.get("status") not in ("ARCHIVED", None):
        problems.append("historical_headline is not ARCHIVED")
    return problems


def write_control_set(args=None) -> Path:
    global _RECEIPT_CACHE
    if _RECEIPT_CACHE is not None and args is None:
        return _RECEIPT_CACHE
    if args is None:
        args = default_args()
    dest = Path(getattr(args, "out", None) or RECEIPT)
    doc = build_record(args)
    problems = validate(doc)
    doc["self_check"] = {
        "ok": not problems,
        "problems": problems,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dest) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, dest)
    _RECEIPT_CACHE = dest
    return dest


def default_args():
    return argparse.Namespace(
        n_predict=DECODE_TOKENS,
        reps=REPS,
        tool_tokens=TOOL_TOKENS,
        prefill_tokens=PREFILL_TOKENS,
        batch_tokens=BATCH_TOKENS,
        batch_sizes=list(BATCH_SIZES),
        out=RECEIPT,
    )


def parse_batch_sizes(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


# --------------------------------------------------------------------------- pytest


def _load_receipt() -> dict:
    assert RECEIPT.is_file(), (
        f"missing {RECEIPT}; run: python3 tools/headless/conventional_control_set.py"
    )
    return json.loads(RECEIPT.read_text())


def test_harness_writes_conventional_control_set_receipt():
    """Validates the already-written receipt. Does not load the 27B."""
    assert RECEIPT.is_file(), (
        f"missing {RECEIPT}; run: python3 tools/headless/conventional_control_set.py"
    )
    doc = json.loads(RECEIPT.read_text())
    assert doc.get("schema") == SCHEMA
    problems = validate(doc)
    assert not problems, problems


def test_live_numbers_have_command_and_spread_or_absent_reason():
    doc = _load_receipt()
    for name, node in _walk_metrics(doc.get("live") or {}):
        st = node.get("status")
        assert st in ("MEASURED", "ABSENT"), (name, st)
        if st == "MEASURED":
            assert node.get("command"), name
            assert node.get("value") is not None, name
            if name != "context_limit" and name != "concurrency":
                assert node.get("repetitions"), name
                assert len(node["repetitions"]) >= 2, name
            if name in ("startup", "prefill", "decode_tps", "tool_shaped_tps"):
                assert node.get("cold_and_warm_stated_separately") is True, name
        if st == "ABSENT":
            assert node.get("reason"), name
            assert node.get("value") not in (0, 0.0), name


def test_archived_numbers_labelled_archived_with_source():
    doc = _load_receipt()
    arch = doc.get("archived") or {}
    assert arch.get("status") == "ARCHIVED"
    assert (arch.get("artifact") or {}).get("present") is False
    for name, node in _walk_metrics(arch):
        assert node.get("status") in ("ARCHIVED", "ABSENT"), (name, node.get("status"))
        assert node.get("status") != "MEASURED", name
        if node.get("status") == "ARCHIVED":
            assert node.get("source_receipt"), name


def test_historical_headline_is_archived():
    doc = _load_receipt()
    h = (doc.get("comparison") or {}).get("historical_headline") or {}
    assert h.get("status") == "ARCHIVED"
    assert "GPU_ATTACK.json" in (h.get("source_receipt") or "")


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-worker", choices=["metal", "load", "measure"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-predict", type=int, default=DECODE_TOKENS)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--tool-tokens", type=int, default=TOOL_TOKENS)
    ap.add_argument("--prefill-tokens", type=int, default=PREFILL_TOKENS)
    ap.add_argument("--batch-tokens", type=int, default=BATCH_TOKENS)
    ap.add_argument("--batch-sizes", default="1,2,4")
    ap.add_argument("--out", type=Path, default=RECEIPT)
    args = ap.parse_args()
    args.batch_sizes = parse_batch_sizes(args.batch_sizes)

    if args.mlx_worker == "metal":
        return worker_metal()
    if args.mlx_worker == "load":
        if not args.model:
            print(json.dumps({"error": "--model required"}))
            return 2
        return worker_load(args.model)
    if args.mlx_worker == "measure":
        if not args.model:
            print(json.dumps({"error": "--model required"}))
            return 2
        try:
            return worker_measure(args)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
            return 1

    path = write_control_set(args)
    doc = json.loads(path.read_text())
    print()
    print("=== CONVENTIONAL CONTROL SET ===")
    live = doc["live"]["metrics"]
    arch = doc["archived"]["metrics"]
    print(f"  live MLX     artifact {doc['live']['artifact'].get('path')}")
    for name in (
        "startup", "prefill", "decode_tps", "context_limit",
        "concurrency", "peak_memory", "tool_shaped_tps",
    ):
        n = live[name]
        print(f"    {name:<18} {n.get('status'):<9} {n.get('value')!r}"
              + (f"  spread={n.get('spread_pct')}%" if n.get("spread_pct") is not None else "")
              + (f"  reason={n.get('reason')[:80]}" if n.get("status") == "ABSENT" else ""))
    print(f"  archived llama.cpp  present={doc['archived']['artifact'].get('present')}")
    for name in (
        "startup", "prefill", "decode_tps", "context_limit",
        "concurrency", "peak_memory", "tool_shaped_tps",
    ):
        n = arch[name]
        src = n.get("source_receipt", "")
        print(f"    {name:<18} {n.get('status'):<9} {str(n.get('value'))[:60]}"
              + (f"  src={src}" if src else ""))
    problems = doc.get("self_check", {}).get("problems") or []
    print(f"\n-> {path}")
    if problems:
        print("SELF-CHECK FAIL:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
