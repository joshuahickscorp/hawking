#!/usr/bin/env python3.12
"""Capture and verify signed Studio environment evidence.

This is intentionally lightweight: no model downloads, no benchmark runs, and no sudo-only probes.
It records whether the host has the expected RAM/disk/network envelope and whether the power,
thermal, and powermetrics surfaces needed by later receipts are visible.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = "reports/condense/studio_environment.json"
SCHEMA = "hawking.studio_environment.v1"
SIGNATURE_ALGORITHM = "sha256-json-v1"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def run(cmd, timeout=10):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
            "ok": p.returncode == 0,
        }
    except Exception as e:
        return {
            "cmd": cmd,
            "returncode": 127,
            "stdout": "",
            "stderr": f"{type(e).__name__}: {e}",
            "ok": False,
        }


def short_text(value, limit=1000):
    text = (value or "").strip()
    return text[:limit]


def command_text(result, limit=1000):
    text = result.get("stdout") or result.get("stderr") or ""
    return short_text(text, limit)


def git_commit(root):
    result = run(["git", "rev-parse", "--short", "HEAD"], timeout=10)
    if result["ok"] and result["stdout"]:
        return result["stdout"].strip()
    return "unknown"


def sysctl_number(name):
    result = run(["sysctl", "-n", name], timeout=10)
    if not result["ok"]:
        return None
    try:
        return int(result["stdout"].strip())
    except ValueError:
        return None


def sysctl_text(name):
    result = run(["sysctl", "-n", name], timeout=10)
    return result["stdout"].strip() if result["ok"] and result["stdout"].strip() else None


def route_summary(host):
    result = run(["route", "-n", "get", host], timeout=10)
    out = {"ok": result["ok"]}
    if not result["ok"]:
        out["error"] = command_text(result, 500)
        return out
    for line in result["stdout"].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in ("interface", "gateway", "source", "ifscope"):
            out[key] = value
    return out


def network_summary(skip):
    host = "huggingface.co"
    api_url = "https://huggingface.co/api/models?limit=1"
    out = {
        "schema": "hawking.studio_environment.network.v1",
        "host": host,
        "api_url": api_url,
        "probe_is_download": False,
        "skipped": bool(skip),
    }
    if skip:
        out["ok"] = True
        out["warning"] = "network probe skipped by operator"
        return out

    out["route"] = route_summary(host)
    out["route_interface"] = out["route"].get("interface")
    out["route_gateway"] = out["route"].get("gateway")
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        addresses = sorted({info[4][0] for info in infos})
        out["dns_ok"] = True
        out["addresses"] = addresses[:8]
        out["address_count"] = len(addresses)
    except Exception as e:
        out["dns_ok"] = False
        out["dns_error"] = f"{type(e).__name__}: {e}"[:500]

    try:
        req = urllib.request.Request(api_url, method="HEAD", headers={"User-Agent": "hawking-studio-env/1"})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=10) as response:
            out["hf_api_status"] = getattr(response, "status", None)
            out["hf_api_elapsed_ms"] = round((time.monotonic() - start) * 1000, 1)
            out["hf_api_server"] = response.headers.get("server")
            out["hf_api_cache"] = response.headers.get("x-cache")
        out["hf_api_ok"] = 200 <= int(out.get("hf_api_status") or 0) < 500
    except Exception as e:
        out["hf_api_ok"] = False
        out["hf_api_error"] = f"{type(e).__name__}: {e}"[:500]

    out["ok"] = bool(out.get("route", {}).get("ok")) and bool(out.get("dns_ok")) and bool(out.get("hf_api_ok"))
    return out


def hardware_summary(root, min_ram_gb, min_free_disk_gb):
    ram_bytes = sysctl_number("hw.memsize")
    usage = shutil.disk_usage(root)
    ram_gb = ram_bytes / 1e9 if ram_bytes else None
    disk_free_gb = usage.free / 1e9
    return {
        "target_ram_gb": min_ram_gb,
        "target_free_disk_gb": min_free_disk_gb,
        "actual_ram_gb": round(ram_gb, 3) if ram_gb else None,
        "actual_cpu_brand": sysctl_text("machdep.cpu.brand_string"),
        "actual_cpu_count": sysctl_number("hw.ncpu"),
        "actual_hw_model": sysctl_text("hw.model"),
        "disk_total_gb": round(usage.total / 1e9, 3),
        "disk_free_gb": round(disk_free_gb, 3),
        "ram_ok": bool(ram_gb is not None and ram_gb >= min_ram_gb),
        "disk_ok": bool(disk_free_gb >= min_free_disk_gb),
    }


def power_thermal_summary():
    batt = run(["pmset", "-g", "batt"], timeout=10)
    therm = run(["pmset", "-g", "therm"], timeout=10)
    powermetrics_path = shutil.which("powermetrics")
    return {
        "power_source": command_text(batt, 1000) or None,
        "power_source_ok": bool((batt["ok"] or batt["stdout"] or batt["stderr"]) and command_text(batt, 1000)),
        "thermal_status": command_text(therm, 1000) or None,
        "thermal_status_ok": bool((therm["ok"] or therm["stdout"] or therm["stderr"]) and command_text(therm, 1000)),
        "powermetrics_path": powermetrics_path,
        "powermetrics_available": bool(powermetrics_path),
        "powermetrics_requires_sudo": os.geteuid() != 0,
        "suggested_powermetrics_probe": "sudo powermetrics --samplers smc -n 1 -i 1000",
    }


def platform_summary():
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def build_checks(doc):
    hardware = doc["hardware"]
    network = doc["network"]
    power = doc["power_thermal"]
    network_skipped = bool(network.get("skipped"))
    checks = [
        {
            "name": "ram",
            "ok": bool(hardware.get("ram_ok")),
            "severity": "fail",
            "detail": f"RAM {hardware.get('actual_ram_gb')}GB, target {hardware.get('target_ram_gb')}GB",
        },
        {
            "name": "disk",
            "ok": bool(hardware.get("disk_ok")),
            "severity": "fail",
            "detail": f"disk free {hardware.get('disk_free_gb')}GB, target {hardware.get('target_free_disk_gb')}GB",
        },
        {
            "name": "network",
            "ok": bool(network.get("ok")) and not network_skipped,
            "severity": "warn" if network_skipped else "fail",
            "detail": "network probe skipped by operator" if network_skipped else (
                f"dns={network.get('dns_ok')} hf_api={network.get('hf_api_ok')} "
                f"route={network.get('route', {}).get('ok')}"
            ),
        },
        {
            "name": "power_source",
            "ok": bool(power.get("power_source_ok")),
            "severity": "warn",
            "detail": short_text(power.get("power_source") or "not captured", 200),
        },
        {
            "name": "thermal_status",
            "ok": bool(power.get("thermal_status_ok")),
            "severity": "warn",
            "detail": short_text(power.get("thermal_status") or "not captured", 200),
        },
        {
            "name": "powermetrics_available",
            "ok": bool(power.get("powermetrics_available")),
            "severity": "fail",
            "detail": power.get("powermetrics_path") or "powermetrics not found",
        },
    ]
    return checks


def sign(doc):
    unsigned = {k: v for k, v in doc.items() if k != "signature"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    doc["signature"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "digest": hashlib.sha256(canonical).hexdigest(),
    }
    return doc


def signature_valid(doc):
    signature = doc.get("signature") or {}
    if signature.get("algorithm") != SIGNATURE_ALGORITHM:
        return False
    expected = signature.get("digest")
    unsigned = {k: v for k, v in doc.items() if k != "signature"}
    actual = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return bool(expected) and expected == actual


def capture(args):
    root = os.path.abspath(args.root)
    os.chdir(root)
    doc = {
        "schema": SCHEMA,
        "generated_at": now(),
        "root": root,
        "git_commit": git_commit(root),
        "machine_class": args.machine_class,
        "operator_targets": {
            "expected_link_mbs": args.link_mbs,
            "min_ram_gb": args.min_ram_gb,
            "min_free_disk_gb": args.min_free_disk_gb,
        },
        "platform": platform_summary(),
        "hardware": hardware_summary(root, args.min_ram_gb, args.min_free_disk_gb),
        "network": network_summary(args.skip_network),
        "power_thermal": power_thermal_summary(),
    }
    doc["checks"] = build_checks(doc)
    doc["failure_count"] = sum(1 for c in doc["checks"] if c["severity"] == "fail" and not c["ok"])
    doc["warning_count"] = sum(1 for c in doc["checks"] if c["severity"] == "warn" and not c["ok"])
    doc["ok"] = doc["failure_count"] == 0
    sign(doc)

    out = args.out
    if not os.path.isabs(out):
        out = os.path.join(root, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")

    if args.json:
        print(json.dumps({"ok": doc["ok"], "path": out, "failure_count": doc["failure_count"], "warning_count": doc["warning_count"]}, indent=2, sort_keys=True))
    else:
        status = "GREEN" if doc["ok"] else "RED"
        print(f"studio environment {status}: {out}", file=sys.stderr)
        for check in doc["checks"]:
            if not check["ok"]:
                print(f"- {check['severity']}: {check['name']}: {check['detail']}", file=sys.stderr)
    return 0 if doc["ok"] else 1


def verify(args):
    path = args.path
    with open(path) as f:
        doc = json.load(f)
    valid = signature_valid(doc)
    ok = valid and doc.get("schema") == SCHEMA
    if args.json:
        print(json.dumps({
            "signature_ok": valid,
            "schema_ok": doc.get("schema") == SCHEMA,
            "environment_ok": bool(doc.get("ok")),
            "path": path,
        }, indent=2, sort_keys=True))
    else:
        verdict = "valid" if ok else "INVALID"
        print(f"studio environment signature {verdict}: {path}", file=sys.stderr)
        if ok and not doc.get("ok"):
            print("studio environment readiness is RED; signature still verifies", file=sys.stderr)
    return 0 if ok else 1


def selftest():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "env.json")
        doc = {
            "schema": SCHEMA,
            "generated_at": "2026-07-08T00:00:00+00:00",
            "root": td,
            "git_commit": "test",
            "machine_class": "Studio-M1Ultra-128",
            "operator_targets": {"expected_link_mbs": 300.0, "min_ram_gb": 120.0, "min_free_disk_gb": 500.0},
            "platform": {"system": "Darwin", "machine": "arm64", "python": "3.12.0"},
            "hardware": {"ram_ok": True, "disk_ok": True, "actual_ram_gb": 128.0, "disk_free_gb": 8000.0},
            "network": {"ok": True, "dns_ok": True, "hf_api_ok": True, "route": {"ok": True}},
            "power_thermal": {
                "power_source_ok": True,
                "thermal_status_ok": True,
                "powermetrics_available": True,
                "powermetrics_path": "/usr/bin/powermetrics",
            },
            "checks": [],
            "failure_count": 0,
            "warning_count": 0,
            "ok": True,
        }
        sign(doc)
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
            f.write("\n")
        with open(path) as f:
            loaded = json.load(f)
        assert signature_valid(loaded)
        loaded["ok"] = False
        assert not signature_valid(loaded)
    print("studio_environment.py selftest OK")
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="write a signed Studio environment receipt")
    cap.add_argument("--root", default=ROOT)
    cap.add_argument("--out", default=DEFAULT_OUT)
    cap.add_argument("--machine-class", default="Studio-M1Ultra-128")
    cap.add_argument("--link-mbs", type=float, default=300.0)
    cap.add_argument("--min-ram-gb", type=float, default=120.0)
    cap.add_argument("--min-free-disk-gb", type=float, default=500.0)
    cap.add_argument("--skip-network", action="store_true")
    cap.add_argument("--json", action="store_true")
    cap.set_defaults(func=capture)

    ver = sub.add_parser("verify", help="verify a signed Studio environment receipt")
    ver.add_argument("--path", default=DEFAULT_OUT)
    ver.add_argument("--json", action="store_true")
    ver.set_defaults(func=verify)

    st = sub.add_parser("selftest")
    st.set_defaults(func=lambda _args: selftest())
    return p


def main():
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
