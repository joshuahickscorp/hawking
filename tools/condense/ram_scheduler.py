#!/usr/bin/env python3.12
"""Memory-pressure-aware scheduler for the 96 GiB M3 Ultra Studio.

Jobs are greedily packed under the shared 78 GB interactive process budget. The
scheduler samples macOS memory pressure and swap before every launch, refuses
over-budget jobs (including ``solo`` jobs), and can drain at a durable file boundary.
Checkpoint-safe jobs receive SIGTERM on critical pressure or operator drain so they
can persist their current rung before exiting.

``resource_probe`` is injectable for deterministic tests. The production probe is
stdlib-only and reads ``kern.memorystatus_vm_pressure_level`` plus ``vm.swapusage``.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)
from studio_manifest import DEFAULT_HARDWARE


YELLOW_SWAP_MB = 2048.0
RED_SWAP_MB = 6144.0
OVER_BUDGET_RC = 78
DRAINED_RC = 75
HEAVY_LEASE_FD_ENV = "HAWKING_HEAVY_LEASE_FD"


def inherited_lease_fds(env=None):
    """Return the validated admission FD that heavy descendants must retain."""
    source = os.environ if env is None else env
    try:
        fd = int(source.get(HEAVY_LEASE_FD_ENV, ""))
        os.fstat(fd)
        return (fd,)
    except (TypeError, ValueError, OSError):
        return ()


def total_gb():
    """Physical memory in GiB (the unit used by Apple's configured-memory label)."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
        ).stdout
        return int(out.strip()) / (1024 ** 3)
    except Exception:
        return DEFAULT_HARDWARE.ram_gb


def swap_mb():
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True
        ).stdout
        match = re.search(r"used\s*=\s*([0-9.]+)([MGT])", out)
        if not match:
            return None
        value, unit = float(match.group(1)), match.group(2)
        return value * {"M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}[unit]
    except Exception:
        return None


def pressure_level():
    try:
        out = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return int(out.strip())
    except Exception:
        return None


def system_resource_probe():
    level = pressure_level()
    swap = swap_mb()
    return {
        "sampled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "pressure_level": level,
        "pressure_name": {1: "normal", 2: "warning", 4: "critical"}.get(level, "unknown"),
        "swap_used_mb": round(swap, 3) if isinstance(swap, (int, float)) else None,
        "pressure_probe_ok": level in {1, 2, 4},
        "swap_probe_ok": isinstance(swap, (int, float)),
    }


def resource_snapshot(path="."):
    """Cheap, read-only resource envelope for Studio status and E0 receipts."""
    memory = system_resource_probe()
    usage = shutil.disk_usage(path)
    disk_total_gb = usage.total / 1e9
    disk_free_gb = usage.free / 1e9
    try:
        power = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5
        )
        power_source = next(
            (line.strip() for line in (power.stdout or power.stderr).splitlines() if line.strip()),
            None,
        )
    except Exception:
        power_source = None
    return {
        "schema": "hawking.studio_resource_snapshot.v1",
        "ok": bool(
            memory.get("pressure_probe_ok") is True
            and memory.get("swap_probe_ok") is True
        ),
        "sampled_at": memory.get("sampled_at"),
        "profile": DEFAULT_HARDWARE.name,
        "physical_ram_gib": round(total_gb(), 3),
        "process_budget_gb": DEFAULT_HARDWARE.process_budget_gb,
        "resident_weight_budget_gb": DEFAULT_HARDWARE.weight_budget_gb,
        "memory_bandwidth_gbps": DEFAULT_HARDWARE.ram_gbps,
        "pressure_level": memory.get("pressure_level"),
        "pressure_name": memory.get("pressure_name"),
        "swap_used_mb": memory.get("swap_used_mb"),
        "memory": memory,
        "disk_total_gb": round(disk_total_gb, 3),
        "disk_free_gb": round(disk_free_gb, 3),
        "disk_reserve_gb": DEFAULT_HARDWARE.disk_reserve_gb,
        "disk_usable_now_gb": round(
            max(0.0, disk_free_gb - DEFAULT_HARDWARE.disk_reserve_gb), 3
        ),
        "scratch_reserve_gb": DEFAULT_HARDWARE.scratch_reserve_gb,
        "hf_cache_reserve_gb": DEFAULT_HARDWARE.cache_reserve_gb,
        "power_source": power_source,
    }


def classify_resource_state(sample, yellow_swap_mb=YELLOW_SWAP_MB, red_swap_mb=RED_SWAP_MB):
    level = sample.get("pressure_level")
    raw_swap = sample.get("swap_used_mb")
    if level not in {1, 2, 4} or isinstance(raw_swap, bool) \
            or not isinstance(raw_swap, (int, float)) or not math.isfinite(float(raw_swap)):
        return "red"
    swap = float(raw_swap)
    if level == 4 or (level is not None and level > 4) or swap >= red_swap_mb:
        return "red"
    if level == 2 or (level is not None and level > 1) or swap >= yellow_swap_mb:
        return "yellow"
    return "green"


def thermal_output_ok(returncode, text):
    """Accept only an explicit English green receipt or a complete recognized numeric schema."""
    if int(returncode) != 0 or not isinstance(text, str) or not text.strip():
        return False
    low = text.lower()
    explicit_green = (
        "no thermal warning level has been recorded" in low
        and "no performance warning level has been recorded" in low
    )
    numeric = {
        key.lower(): int(value)
        for key, value in re.findall(r"([A-Za-z_]+)\s*[:=]\s*(\d+)", text)
    }
    numeric_green = bool(
        {"cpu_speed_limit", "scheduler_limit", "available_cpus"}.issubset(numeric)
        and numeric["cpu_speed_limit"] >= 100
        and numeric["scheduler_limit"] >= 100
        and numeric["available_cpus"] > 0
    )
    return explicit_green or numeric_green


class Job:
    def __init__(self, name, argv, est_gb, solo=False, done_when=None, log=None, env=None,
                 checkpoint_safe=False):
        self.name = name
        self.argv = list(argv)
        self.est_gb = float(est_gb)
        self.solo = bool(solo)
        self.done_when = done_when
        self.log = log
        self.env = env
        self.checkpoint_safe = bool(checkpoint_safe)
        self.proc = None
        self.t0 = None
        self.peak_gb = 0.0
        self.stop_requested = False
        self.stop_reason = None
        self.stop_pids = set()
        self._log_fh = None

    def is_done(self):
        return bool(self.done_when and os.path.exists(self.done_when))


class Scheduler:
    """Greedy Studio runner with pressure gates and an operator drain boundary."""

    def __init__(self, budget_gb=None, swap_ceil_mb=None, poll=15, statusf=None, log=print,
                 resource_probe=None, yellow_swap_mb=YELLOW_SWAP_MB,
                 red_swap_mb=RED_SWAP_MB, drain_file=None, sleep_fn=time.sleep):
        self.tot = total_gb()
        self.budget = float(
            DEFAULT_HARDWARE.process_budget_gb if budget_gb is None else budget_gb
        )
        # ``swap_ceil_mb`` remains a compatibility alias for the old launch-only ceiling.
        if swap_ceil_mb is not None:
            yellow_swap_mb = float(swap_ceil_mb)
            red_swap_mb = max(float(red_swap_mb), float(swap_ceil_mb))
        self.yellow_swap_mb = float(yellow_swap_mb)
        self.red_swap_mb = float(red_swap_mb)
        self.poll = float(poll)
        self.statusf = statusf
        self.log = log
        self.resource_probe = resource_probe or system_resource_probe
        self.drain_file = drain_file
        self.sleep_fn = sleep_fn

    def _used(self, running):
        return sum(j.est_gb for j in running)

    @staticmethod
    def _rss_gb(pgid):
        """Sum RSS for the root process group. Returns GiB; zero on probe failure."""
        try:
            out = subprocess.run(
                ["ps", "-o", "rss=", "-g", str(pgid)], capture_output=True, text=True
            ).stdout
            return sum(int(x) for x in out.split()) / (1024 * 1024)
        except Exception:
            return 0.0

    @staticmethod
    def _descendant_pids(root_pid):
        """Return descendants, including children that created a new process group/session."""
        try:
            out = subprocess.run(
                ["ps", "-axo", "pid=,ppid="], capture_output=True, text=True, check=True
            ).stdout
            children = {}
            for line in out.splitlines():
                pid, ppid = (int(x) for x in line.split())
                children.setdefault(ppid, []).append(pid)
            found, stack = [], [root_pid]
            while stack:
                parent = stack.pop()
                for pid in children.get(parent, []):
                    found.append(pid)
                    stack.append(pid)
            return found
        except Exception:
            return []

    @staticmethod
    def _pid_alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except ProcessLookupError:
            return False

    def _probe(self):
        try:
            sample = dict(self.resource_probe() or {})
        except Exception as exc:
            sample = {"probe_error": f"{type(exc).__name__}: {exc}"}
        sample.setdefault(
            "sampled_at", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        )
        sample.setdefault("pressure_level", None)
        sample.setdefault("pressure_name", "unknown")
        sample.setdefault("swap_used_mb", None)
        sample["state"] = classify_resource_state(
            sample, self.yellow_swap_mb, self.red_swap_mb
        )
        return sample

    def _launch(self, job):
        if job.est_gb > self.budget:
            raise ValueError(
                f"refusing over-budget job {job.name}: {job.est_gb:.1f}GB > {self.budget:.1f}GB"
            )
        job._log_fh = open(job.log, "a") if job.log else None
        sink = job._log_fh if job._log_fh else subprocess.DEVNULL
        env = {**os.environ, **(job.env or {})}
        job.proc = subprocess.Popen(
            job.argv, stdout=sink, stderr=subprocess.STDOUT, env=env, start_new_session=True,
            pass_fds=inherited_lease_fds(env),
        )
        job.t0 = time.time()
        self.log(
            f"[sched] launch {job.name} est={job.est_gb:.0f}GB "
            f"solo={job.solo} checkpoint_safe={job.checkpoint_safe} pid={job.proc.pid}"
        )

    def _request_checkpoint_stop(self, job, reason):
        if job.stop_requested or not job.checkpoint_safe or not job.proc:
            return False
        job.stop_requested = True
        job.stop_reason = reason
        descendants = self._descendant_pids(job.proc.pid)
        job.stop_pids.update(descendants)
        # Signal escaped descendants first (doctor workers may create their own session), then
        # the scheduler-owned process group. SIGTERM is the checkpoint contract.
        for pid in reversed(descendants):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            os.killpg(job.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                job.proc.terminate()
            except ProcessLookupError:
                pass
        self.log(f"[sched] checkpoint-stop {job.name}: {reason}")
        return True

    def _job_finished(self, job):
        if not job.proc or job.proc.poll() is None:
            return False
        if job.stop_requested:
            job.stop_pids = {pid for pid in job.stop_pids if self._pid_alive(pid)}
            if job.stop_pids:
                return False
        return True

    def _partition_jobs(self, jobs):
        pending, results = [], {}
        for job in jobs:
            if job.is_done():
                self.log(f"[sched] skip {job.name} (artifact exists)")
            elif job.est_gb > self.budget:
                results[job.name] = OVER_BUDGET_RC
                self.log(
                    f"[sched] REFUSE {job.name}: est={job.est_gb:.1f}GB exceeds "
                    f"interactive budget={self.budget:.1f}GB (solo does not bypass the cap)"
                )
            else:
                pending.append(job)
        return pending, results

    def _drain_requested(self):
        return bool(self.drain_file and os.path.exists(self.drain_file))

    def run(self, jobs):
        jobs = list(jobs)
        pending, results = self._partition_jobs(jobs)
        running = []
        draining = self._drain_requested()
        self.log(
            f"[sched] RAM {self.tot:.0f}GiB, interactive budget {self.budget:.0f}GB, "
            f"{len(pending)} runnable, {len(results)} refused"
        )
        if not pending:
            resource = self._probe()
            state = "drained" if draining else "complete"
            self._status(running, pending, results, resource, state)
            return results

        while pending or running:
            for job in running:
                job.peak_gb = max(job.peak_gb, self._rss_gb(job.proc.pid))
            for job in list(running):
                if not self._job_finished(job):
                    continue
                results[job.name] = job.proc.returncode
                self.log(
                    f"[sched] done {job.name} rc={job.proc.returncode} "
                    f"{time.time()-job.t0:.0f}s peak={job.peak_gb:.1f}GiB "
                    f"(est {job.est_gb:.0f}GB)"
                )
                try:
                    os.makedirs("reports/cron", exist_ok=True)
                    with open("reports/cron/ram_actuals.jsonl", "a") as handle:
                        handle.write(json.dumps({
                            "name": job.name,
                            "est_gb": job.est_gb,
                            "peak_gb": round(job.peak_gb, 2),
                            "wall_s": round(time.time() - job.t0),
                            "stop_reason": job.stop_reason,
                        }) + "\n")
                except Exception:
                    pass
                if job._log_fh:
                    job._log_fh.close()
                    job._log_fh = None
                running.remove(job)

            resource = self._probe()
            draining = draining or self._drain_requested()
            if resource["state"] == "red":
                for job in running:
                    self._request_checkpoint_stop(
                        job,
                        f"red memory state: pressure={resource.get('pressure_level')} "
                        f"swap={resource.get('swap_used_mb')}MB",
                    )
            if draining:
                for job in running:
                    self._request_checkpoint_stop(job, f"operator drain file: {self.drain_file}")

            if not draining and resource["state"] == "green":
                solo_running = any(job.solo for job in running)
                for job in list(pending):
                    if job.solo:
                        if not running:
                            self._launch(job)
                            running.append(job)
                            pending.remove(job)
                            solo_running = True
                        break
                    if solo_running:
                        break
                    if self._used(running) + job.est_gb <= self.budget:
                        self._launch(job)
                        running.append(job)
                        pending.remove(job)

            state = "draining" if draining else resource["state"]
            self._status(running, pending, results, resource, state)
            if draining and not running:
                for job in pending:
                    results[job.name] = DRAINED_RC
                pending.clear()
                self._status(running, pending, results, resource, "drained")
                break
            if pending or running:
                self.sleep_fn(self.poll)
        return results

    def _status(self, running, pending, results, resource, state):
        if not self.statusf:
            return
        doc = {
            "schema": "hawking.ram_scheduler_status.v2",
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "state": state,
            "profile": DEFAULT_HARDWARE.name,
            "physical_ram_gib": round(self.tot, 3),
            "process_budget_gb": self.budget,
            "estimated_running_gb": round(self._used(running), 3),
            "resource": resource,
            "drain_file": self.drain_file,
            "drain_requested": self._drain_requested(),
            "running": [
                {
                    "name": job.name,
                    "pid": job.proc.pid,
                    "est_gb": job.est_gb,
                    "peak_rss_gib": round(job.peak_gb, 3),
                    "solo": job.solo,
                    "checkpoint_safe": job.checkpoint_safe,
                    "stop_requested": job.stop_requested,
                    "stop_reason": job.stop_reason,
                }
                for job in running
            ],
            "pending": [job.name for job in pending],
            "results": results,
        }
        parent = os.path.dirname(self.statusf)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{self.statusf}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as handle:
                json.dump(doc, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.statusf)
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def plan(self, jobs):
        """Dry-run the packing schedule without starting a process."""
        q = list(jobs)
        wave = 1
        print(f"# RAM-pack plan: {self.tot:.0f}GiB physical, {self.budget:.0f}GB budget")
        while q:
            if q[0].solo:
                job = q.pop(0)
                if job.est_gb > self.budget:
                    print(
                        f"  wave {wave} [REFUSED OVER-BUDGET]: {job.est_gb:.0f}GB :: "
                        f"{job.name}({job.est_gb:.0f})"
                    )
                else:
                    print(
                        f"  wave {wave} [SOLO]: {job.est_gb:.0f}GB :: "
                        f"{job.name}({job.est_gb:.0f})"
                    )
                wave += 1
                continue
            batch, used, i = [], 0.0, 0
            while i < len(q):
                job = q[i]
                if job.solo:
                    break
                if used + job.est_gb <= self.budget:
                    used += job.est_gb
                    batch.append(q.pop(i))
                else:
                    i += 1
            if not batch:
                job = min(q, key=lambda item: item.est_gb)
                q.remove(job)
                print(
                    f"  wave {wave} [REFUSED OVER-BUDGET]: {job.est_gb:.0f}GB :: "
                    f"{job.name}({job.est_gb:.0f})"
                )
            else:
                print(
                    f"  wave {wave}: {used:.0f}GB :: "
                    + ", ".join(f"{job.name}({job.est_gb:.0f})" for job in batch)
                )
            wave += 1
        print(f"# {wave-1} waves; REFUSED jobs require a lower-memory/blockwise recipe")


def selftest():
    assert classify_resource_state({"pressure_level": 1, "swap_used_mb": 0}) == "green"
    assert classify_resource_state({"pressure_level": 2, "swap_used_mb": 0}) == "yellow"
    assert classify_resource_state({"pressure_level": 4, "swap_used_mb": 0}) == "red"
    assert classify_resource_state({"pressure_level": 1, "swap_used_mb": 7000}) == "red"
    assert classify_resource_state({"pressure_level": None, "swap_used_mb": 0}) == "red"
    assert classify_resource_state({"pressure_level": 1, "swap_used_mb": None}) == "red"
    assert inherited_lease_fds({}) == ()
    assert thermal_output_ok(0, "Note: No thermal warning level has been recorded\n"
                             "Note: No performance warning level has been recorded")
    assert thermal_output_ok(0, "CPU_Speed_Limit=100 Scheduler_Limit=100 Available_CPUs=16")
    assert not thermal_output_ok(0, "")
    assert not thermal_output_ok(0, "localized or unknown output")
    assert not thermal_output_ok(0, "CPU_Speed_Limit=100")

    samples = iter([
        {"pressure_level": 1, "pressure_name": "normal", "swap_used_mb": 0},
        {"pressure_level": 2, "pressure_name": "warning", "swap_used_mb": 0},
    ])
    scheduler = Scheduler(budget_gb=78, resource_probe=lambda: next(samples), log=lambda _msg: None)
    assert scheduler._probe()["state"] == "green"
    assert scheduler._probe()["state"] == "yellow"
    pending, results = scheduler._partition_jobs([
        Job("safe", ["true"], 40, checkpoint_safe=True),
        Job("solo-over", ["true"], 85, solo=True, checkpoint_safe=True),
    ])
    assert [job.name for job in pending] == ["safe"]
    assert results == {"solo-over": OVER_BUDGET_RC}

    with tempfile.TemporaryDirectory() as directory:
        status = os.path.join(directory, "status.json")
        drain = os.path.join(directory, "drain")
        scheduler = Scheduler(
            budget_gb=78,
            statusf=status,
            drain_file=drain,
            resource_probe=lambda: {"pressure_level": 1, "swap_used_mb": 0},
            log=lambda _msg: None,
        )
        resource = scheduler._probe()
        scheduler._status([], [], {"done": 0}, resource, "green")
        with open(status) as handle:
            doc = json.load(handle)
        assert doc["schema"] == "hawking.ram_scheduler_status.v2"
        assert doc["process_budget_gb"] == 78
        assert not scheduler._drain_requested()
        open(drain, "w").close()
        assert scheduler._drain_requested()
        result = scheduler.run([
            Job("deferred", ["must-not-run"], 1, checkpoint_safe=True),
        ])
        assert result == {"deferred": DRAINED_RC}
    print("ram_scheduler.py selftest OK")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else None
    demo = [
        *[Job(f"bake-{model}", ["true"], 14) for model in ("05b", "15b", "7b", "14b", "32b")],
        Job("0.5B-lab", ["true"], 10),
        Job("1.5B-lab", ["true"], 10),
        Job("7B-doctor", ["true"], 40, checkpoint_safe=True),
        Job("14B-doctor", ["true"], 65, checkpoint_safe=True),
        Job("32B-doctor", ["true"], 85, solo=True, checkpoint_safe=True),
    ]
    Scheduler(budget_gb=budget).plan(demo)
