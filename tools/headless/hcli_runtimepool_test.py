#!/usr/bin/env python3
"""Protected RuntimePool checks. Plain python3 + assert. Must also pass under pytest.

Run:
    python3 tools/headless/hcli_runtimepool_test.py
    pytest tools/headless/hcli_runtimepool_test.py -q

Every check below was watched FAILING against the pre-change RuntimePool
(a clamp on HCLI_MAX_RUNTIMES, no MemGate, no ownership record, llama-server
hard-coded, no prefix affinity). Failure text is recorded in the lane report.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

KNOWN_MODEL = (
    Path.home()
    / "models"
    / "qwen3.8-27b-abliterated"
    / "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"
)

FAILS: list[str] = []
SKIPS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    if cond:
        print(f"ok   {name}")
        return True
    print(f"FAIL {name}: {detail}")
    FAILS.append(f"{name}: {detail}")
    return False


def skip(name: str, why: str) -> None:
    print(f"SKIP {name}: {why}")
    SKIPS.append(f"{name}: {why}")


def _have_model() -> bool:
    return KNOWN_MODEL.is_file()


def _have_llama() -> bool:
    import shutil

    return shutil.which("llama-server") is not None or bool(
        os.environ.get("HCLI_LLAMA_SERVER")
    )


def _env_snapshot(keys):
    return {k: os.environ.get(k) for k in keys}


def _restore_env(snap):
    for k, v in snap.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


LIMIT_KEYS = (
    "HCLI_RESIDENT_RUNTIME_LIMIT",
    "HCLI_ACTIVE_DECODE_LIMIT",
    "RESIDENT_RUNTIME_LIMIT",
    "ACTIVE_DECODE_LIMIT",
    "HCLI_MAX_RUNTIMES",
    "HCLI_CTX_SIZE",
    "HCLI_READY_TIMEOUT",
    "HCLI_MEM_RESERVE_BYTES",
    "HCLI_RESERVE_GIB",
    "HCLI_DECODE_TOPOLOGY",
    "HCLI_WORKSPACE",
    "HCLI_LLAMA_DEVICE",
    "HCLI_N_GPU_LAYERS",
    "HCLI_LLAMA_FIT",
)


def _arm_cpu_llama() -> None:
    """llama-server in this sandbox cannot create a Metal command queue.

    Production still defaults to GPU (-ngl 999). The protected checks need a
    live server, so they pin CPU via --device none.
    """
    os.environ["HCLI_LLAMA_DEVICE"] = "none"
    os.environ["HCLI_N_GPU_LAYERS"] = "0"
    os.environ["HCLI_LLAMA_FIT"] = "off"


def _tiny_complete_payload():
    return {
        "prompt": "Reply with the single word ok.",
        "n_predict": 1,
        "temperature": 0.0,
        "cache_prompt": True,
    }


def _allocate_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _unique_pids(runtimes) -> list[int]:
    seen = []
    for rt in runtimes:
        pid = getattr(rt, "pid", None)
        if isinstance(pid, int) and pid > 0 and pid not in seen:
            seen.append(pid)
    return seen


# ---------------------------------------------------------------------------
# 1. Two limits are independent
# ---------------------------------------------------------------------------

def check_two_limits_independent():
    name = "two limits are independent"
    snap = _env_snapshot(LIMIT_KEYS)
    try:
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "3"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        os.environ["HCLI_CTX_SIZE"] = "2048"
        os.environ["HCLI_READY_TIMEOUT"] = "180"
        _arm_cpu_llama()
        from hcli.runtime import RuntimePool

        with tempfile.TemporaryDirectory() as tmp:
            pool = RuntimePool(
                str(KNOWN_MODEL if _have_model() else Path(tmp) / "missing.gguf"),
                requested_n=8,
                workspace=tmp,
            )
            resident = int(getattr(pool, "resident_limit"))
            active = int(getattr(pool, "active_decode_limit"))
            if not check(
                name + " (attributes)",
                resident == 3 and active == 1 and resident != active,
                f"resident_limit={resident} active_decode_limit={active}",
            ):
                return
            if not _have_model():
                skip(name + " (in-flight)", f"GGUF absent: {KNOWN_MODEL}")
                return
            if not _have_llama():
                skip(name + " (in-flight)", "llama-server not on PATH")
                return
            try:
                pool.start()
                admitted = int(getattr(pool, "admitted_n", 0))
                nrt = len(getattr(pool, "runtimes", []) or [])
                if not check(
                    name + " (admitted 3)",
                    admitted == 3 and nrt == 3,
                    f"admitted_n={admitted} n_runtimes={nrt}",
                ):
                    return
                if not hasattr(pool, "complete"):
                    check(
                        name + " (in-flight)",
                        False,
                        "RuntimePool.complete is missing; cannot instrument decode concurrency",
                    )
                    return
                errors = []

                def worker():
                    try:
                        pool.complete(_tiny_complete_payload(), timeout=180, prefix_key="t1")
                    except Exception as exc:  # noqa: BLE001
                        errors.append(exc)

                threads = [threading.Thread(target=worker) for _ in range(3)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=240)
                observed = int(getattr(pool, "max_in_flight_observed", -1))
                check(
                    name + " (max in-flight == 1)",
                    observed == 1 and not errors and all(not t.is_alive() for t in threads),
                    f"max_in_flight_observed={observed} errors={errors!r}",
                )
            finally:
                try:
                    pool.stop()
                except Exception:
                    pass
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        _restore_env(snap)


# ---------------------------------------------------------------------------
# 2. MemGate refuses (both directions)
# ---------------------------------------------------------------------------

def check_memgate_refuses():
    name = "MemGate refuses"
    snap = _env_snapshot(LIMIT_KEYS)
    try:
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "4"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        os.environ["HCLI_CTX_SIZE"] = "2048"
        os.environ["HCLI_READY_TIMEOUT"] = "180"
        _arm_cpu_llama()
        from hcli.runtime import RuntimePool

        physical = 0
        try:
            from hcli.machine import MachineProbe

            physical = int(MachineProbe().probe().physical_memory_bytes or 0)
        except Exception:
            physical = 0
        absurd = max(physical * 4, 10**18)

        with tempfile.TemporaryDirectory() as tmp:
            model = str(KNOWN_MODEL if _have_model() else Path(tmp) / "dummy.gguf")
            if not _have_model():
                Path(model).write_bytes(b"not-a-gguf")
            refused = RuntimePool(
                model,
                requested_n=4,
                workspace=tmp,
                reserve_bytes=absurd,
            )
            try:
                try:
                    refused.start()
                except FileNotFoundError:
                    # still a refusal-shaped outcome if we never admitted
                    pass
                admitted_refuse = int(getattr(refused, "admitted_n", -1))
                reason = getattr(refused, "refusal_reason", None)
                ok_refuse = admitted_refuse in (0, 1) and bool(reason)
                if not check(
                    name + " (absurd reserve)",
                    ok_refuse,
                    f"admitted_n={admitted_refuse} refusal_reason={reason!r}",
                ):
                    return
            finally:
                try:
                    refused.stop()
                except Exception:
                    pass

            if not _have_model() or not _have_llama():
                skip(
                    name + " (sane reserve)",
                    "GGUF or llama-server absent; cannot admit more under a sane reserve",
                )
                return

            sane = RuntimePool(
                str(KNOWN_MODEL),
                requested_n=4,
                workspace=tmp,
                reserve_bytes=8 * 1024**3,
            )
            try:
                sane.start()
                admitted_sane = int(getattr(sane, "admitted_n", 0))
                check(
                    name + " (sane reserve admits more)",
                    admitted_sane > admitted_refuse,
                    f"sane admitted_n={admitted_sane} absurd admitted_n={admitted_refuse}",
                )
            finally:
                try:
                    sane.stop()
                except Exception:
                    pass
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        _restore_env(snap)


# ---------------------------------------------------------------------------
# 3. Marginal cost, not RSS
# ---------------------------------------------------------------------------

def check_marginal_cost_not_rss():
    name = "marginal cost, not RSS"
    if not _have_model():
        skip(name, f"GGUF absent: {KNOWN_MODEL}")
        return
    if not _have_llama():
        skip(name, "llama-server not on PATH")
        return
    snap = _env_snapshot(LIMIT_KEYS)
    try:
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "2"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        os.environ["HCLI_CTX_SIZE"] = "2048"
        os.environ["HCLI_READY_TIMEOUT"] = "180"
        _arm_cpu_llama()
        from hcli.runtime import RuntimePool

        model_bytes = KNOWN_MODEL.stat().st_size
        with tempfile.TemporaryDirectory() as tmp:
            pool = RuntimePool(str(KNOWN_MODEL), requested_n=2, workspace=tmp)
            try:
                pool.start()
                records = list(getattr(pool, "admission_records", None) or [])
                if len(records) < 2:
                    check(
                        name,
                        False,
                        f"need 2 admission records, got {len(records)}: {records!r}",
                    )
                    return
                cost = records[1].get("marginal_free_ram_cost_bytes")
                if cost is None:
                    check(
                        name,
                        False,
                        f"second record missing marginal_free_ram_cost_bytes: {records[1]!r}",
                    )
                    return
                # mmap sharing / slot-weight sharing: second runtime must not
                # be charged a full extra copy of the weights.
                check(
                    name,
                    int(cost) < int(model_bytes) * 0.5,
                    f"runtime-2 marginal={cost} model_bytes={model_bytes} "
                    f"ratio={int(cost)/model_bytes:.3f} record={records[1]!r}",
                )
            finally:
                try:
                    pool.stop()
                except Exception:
                    pass
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        _restore_env(snap)


# ---------------------------------------------------------------------------
# 4. No orphans on clean stop
# ---------------------------------------------------------------------------

def check_no_orphans_clean_stop():
    name = "no orphans on clean stop"
    if not _have_model():
        skip(name, f"GGUF absent: {KNOWN_MODEL}")
        return
    if not _have_llama():
        skip(name, "llama-server not on PATH")
        return
    snap = _env_snapshot(LIMIT_KEYS)
    try:
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "1"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        os.environ["HCLI_CTX_SIZE"] = "2048"
        os.environ["HCLI_READY_TIMEOUT"] = "180"
        _arm_cpu_llama()
        from hcli.runtime import RuntimePool

        with tempfile.TemporaryDirectory() as tmp:
            pool = RuntimePool(str(KNOWN_MODEL), requested_n=1, workspace=tmp)
            try:
                pool.start()
                pids = _unique_pids(getattr(pool, "runtimes", []) or [])
                if not pids:
                    check(name, False, "start() left no child pids to track")
                    return
                report = pool.stop()
                alive = [p for p in pids if _pid_alive(p)]
                # stop must be idempotent
                pool.stop()
                check(
                    name,
                    not alive,
                    f"still alive after stop(): {alive} report={report!r}",
                )
            finally:
                try:
                    pool.stop()
                except Exception:
                    pass
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        _restore_env(snap)


# ---------------------------------------------------------------------------
# 5. No orphans on abnormal death (SIGKILL parent; new pool reaps)
# ---------------------------------------------------------------------------

def check_no_orphans_abnormal_death():
    name = "no orphans on abnormal death"
    if not _have_model():
        skip(name, f"GGUF absent: {KNOWN_MODEL}")
        return
    if not _have_llama():
        skip(name, "llama-server not on PATH")
        return
    snap = _env_snapshot(LIMIT_KEYS)
    try:
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "1"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        os.environ["HCLI_CTX_SIZE"] = "2048"
        os.environ["HCLI_READY_TIMEOUT"] = "180"
        _arm_cpu_llama()
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "ready.json"
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json, os, sys, time\n"
                                                "from hcli.runtime import RuntimePool\n"
                        f"os.environ['HCLI_RESIDENT_RUNTIME_LIMIT']='1'\n"
                        f"os.environ['HCLI_ACTIVE_DECODE_LIMIT']='1'\n"
                        f"os.environ['HCLI_CTX_SIZE']='2048'\n"
                        f"os.environ['HCLI_READY_TIMEOUT']='180'\n"
                        "os.environ['HCLI_LLAMA_DEVICE']='none'\n"
                        "os.environ['HCLI_N_GPU_LAYERS']='0'\n"
                        "os.environ['HCLI_LLAMA_FIT']='off'\n"
                        f"pool = RuntimePool({str(KNOWN_MODEL)!r}, requested_n=1, workspace={tmp!r})\n"
                        "pool.start()\n"
                        "pids = []\n"
                        "seen = set()\n"
                        "for rt in pool.runtimes:\n"
                        "    if rt.pid and rt.pid not in seen:\n"
                        "        seen.add(rt.pid)\n"
                        "        pids.append(rt.pid)\n"
                        f"open({str(marker)!r}, 'w').write(json.dumps(pids))\n"
                        "time.sleep(3600)\n"
                    ),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            deadline = time.time() + 180
            pids = None
            while time.time() < deadline:
                if marker.is_file() and marker.stat().st_size > 2:
                    try:
                        pids = json.loads(marker.read_text())
                        if pids:
                            break
                    except json.JSONDecodeError:
                        pass
                if child.poll() is not None:
                    err = (child.stderr.read() if child.stderr else b"")[:2000]
                    check(
                        name,
                        False,
                        f"helper exited before READY code={child.returncode} err={err!r}",
                    )
                    return
                time.sleep(0.2)
            if not pids:
                child.kill()
                check(name, False, "timed out waiting for helper READY pids")
                return
            os.kill(child.pid, signal.SIGKILL)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            still = [p for p in pids if _pid_alive(p)]
            if not still:
                check(
                    name,
                    False,
                    f"children died with the SIGKILL parent, so reaper was not exercised: pids={pids}",
                )
                return
            from hcli.runtime import RuntimePool

            pool2 = RuntimePool(str(KNOWN_MODEL), requested_n=1, workspace=tmp)
            try:
                if hasattr(pool2, "reap_orphans"):
                    pool2.reap_orphans()
                else:
                    pool2.start()
                alive = [p for p in pids if _pid_alive(p)]
                check(
                    name,
                    not alive,
                    f"orphans still alive after new pool: {alive} original={pids}",
                )
            finally:
                try:
                    pool2.stop()
                except Exception:
                    pass
                for p in pids:
                    if _pid_alive(p):
                        try:
                            os.kill(p, signal.SIGKILL)
                        except OSError:
                            pass
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        _restore_env(snap)


# ---------------------------------------------------------------------------
# 6. Never kills a foreign process
# ---------------------------------------------------------------------------

def check_never_kills_foreign():
    name = "never kills a foreign process"
    snap = _env_snapshot(LIMIT_KEYS)
    foreign = None
    try:
        # Appear in `ps` as llama-server so a forbidden `pkill llama-server`
        # would murder us. Ownership-record reaping must leave this alone.
        foreign = subprocess.Popen(
            ["bash", "-lc", "exec -a llama-server sleep 180"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(0.2)
        if not _pid_alive(foreign.pid):
            check(name, False, "foreign stand-in exited immediately")
            return
        from hcli.runtime import RuntimePool

        with tempfile.TemporaryDirectory() as tmp:
            # Seed an ownership record that does NOT mention the foreign pid,
            # then construct a pool (startup reaper) and call reap_orphans.
            hcli = Path(tmp) / ".hcli"
            hcli.mkdir(parents=True)
            (hcli / "runtime_pool.json").write_text(
                json.dumps(
                    {
                        "schema": "hcli.runtime_pool.v1",
                        "pool_pid": 99999999,
                        "pool_start_time": "dead-owner",
                        "children": [
                            {
                                "pid": 88888888,
                                "start_time": "not-a-real-process",
                                "port": 1,
                                "model": "/no/such.gguf",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            model = str(KNOWN_MODEL if _have_model() else Path(tmp) / "missing.gguf")
            pool = RuntimePool(model, requested_n=1, workspace=tmp)
            if hasattr(pool, "reap_orphans"):
                pool.reap_orphans()
            else:
                try:
                    pool.start()
                except Exception:
                    pass
            alive = _pid_alive(foreign.pid)
            check(
                name,
                alive,
                f"foreign pid {foreign.pid} was killed by the reaper",
            )
            try:
                pool.stop()
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        _restore_env(snap)
        if foreign is not None and _pid_alive(foreign.pid):
            try:
                os.kill(foreign.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                foreign.wait(timeout=5)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 7. Backend identity + honest supports()
# ---------------------------------------------------------------------------

def check_backend_identity():
    name = "backend identity"
    if not _have_llama():
        skip(name, "llama-server not on PATH")
        return
    try:
        from hcli.backends import LlamaServerBackend

        model = str(KNOWN_MODEL if _have_model() else "/nonexistent/model.gguf")
        backend = LlamaServerBackend(model_path=model, port=_allocate_port())
        ident = backend.identity()
        required = (
            "backend",
            "binary",
            "version",
            "model_path",
            "model_bytes",
            "context",
            "quantisation",
        )
        missing = [k for k in required if k not in ident]
        if missing:
            check(name, False, f"identity() missing {missing}: {ident!r}")
            return
        rf = backend.supports("response_format")
        ctk = backend.supports("chat_template_kwargs")
        # llama.cpp on this box supports both; a False here is a lie.
        check(
            name,
            rf is True and ctk is True and ident.get("backend") in {"llama_server", "llama.cpp"},
            f"supports(response_format)={rf} supports(chat_template_kwargs)={ctk} ident={ident!r}",
        )
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 8. Prefix affinity
# ---------------------------------------------------------------------------

def check_prefix_affinity():
    name = "prefix affinity"
    if not _have_model():
        skip(name, f"GGUF absent: {KNOWN_MODEL}")
        return
    if not _have_llama():
        skip(name, "llama-server not on PATH")
        return
    snap = _env_snapshot(LIMIT_KEYS)
    try:
        os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "2"
        os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "1"
        os.environ["HCLI_CTX_SIZE"] = "2048"
        os.environ["HCLI_READY_TIMEOUT"] = "180"
        _arm_cpu_llama()
        from hcli.runtime import RuntimePool

        with tempfile.TemporaryDirectory() as tmp:
            pool = RuntimePool(str(KNOWN_MODEL), requested_n=2, workspace=tmp)
            try:
                pool.start()
                if int(getattr(pool, "admitted_n", 0)) < 1:
                    check(name, False, f"pool admitted nothing: {getattr(pool, 'refusal_reason', None)!r}")
                    return
                if not hasattr(pool, "complete"):
                    check(name, False, "RuntimePool.complete is missing")
                    return
                r1 = pool.complete(
                    _tiny_complete_payload(), timeout=180, prefix_key="mission-stable"
                )
                r2 = pool.complete(
                    _tiny_complete_payload(), timeout=180, prefix_key="mission-stable"
                )
                idx1 = getattr(r1, "runtime_index", None)
                if idx1 is None and isinstance(r1, dict):
                    idx1 = r1.get("runtime_index")
                idx2 = getattr(r2, "runtime_index", None)
                if idx2 is None and isinstance(r2, dict):
                    idx2 = r2.get("runtime_index")
                hits = int(getattr(pool, "prefix_hits", 0))
                check(
                    name,
                    idx1 is not None and idx1 == idx2 and hits >= 1,
                    f"idx1={idx1} idx2={idx2} prefix_hits={hits} misses={getattr(pool, 'prefix_misses', None)}",
                )
            finally:
                try:
                    pool.stop()
                except Exception:
                    pass
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        _restore_env(snap)


def main() -> int:
    checks = [
        check_two_limits_independent,
        check_memgate_refuses,
        check_marginal_cost_not_rss,
        check_no_orphans_clean_stop,
        check_no_orphans_abnormal_death,
        check_never_kills_foreign,
        check_backend_identity,
        check_prefix_affinity,
    ]
    for fn in checks:
        fn()
    if SKIPS:
        print(f"{len(SKIPS)} skipped")
    if FAILS:
        print(f"{len(FAILS)} failed")
        return 1
    print("all runtimepool checks passed")
    return 0


def test_hcli_runtimepool():
    """pytest entry: the same checks as running this file directly."""
    rc = main()
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
