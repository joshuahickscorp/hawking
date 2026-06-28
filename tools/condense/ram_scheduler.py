#!/usr/bin/env python3.12
"""RAM-aware parallel job scheduler — pack the Studio's unified memory, run many jobs at once.

The 18 GB box forced one-job-at-a-time. On the 96 GB Studio the binding constraint is RAM, not
cores, so the win is PACKING: run as many jobs concurrently as fit under a memory budget — heavy
jobs (32B doctor ~85 GB) run SOLO (whole box), light jobs (0.5B/1.5B labs, PTQ bakes ~8-16 GB)
run many-at-once. Deterministic launch order, a live swap watchdog (stop launching if swap climbs
past a ceiling), and resumable (a job whose `done_when` artifact exists is skipped).

Generic: studio_run.py builds the Job list; this just runs it inside the RAM envelope. Pure stdlib.

A Job = (name, argv, est_gb, solo, done_when, log, env).
"""
import os, sys, time, json, subprocess


def total_gb():
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 1e9
    except Exception:
        return 18.0


def swap_mb():
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        return float(out.split()[5].rstrip("M,"))
    except Exception:
        return 0.0


class Job:
    def __init__(self, name, argv, est_gb, solo=False, done_when=None, log=None, env=None):
        self.name = name; self.argv = list(argv); self.est_gb = float(est_gb)
        self.solo = bool(solo); self.done_when = done_when; self.log = log; self.env = env
        self.proc = None; self.t0 = None; self.peak_gb = 0.0

    def is_done(self):
        return bool(self.done_when and os.path.exists(self.done_when))


class Scheduler:
    """Greedy RAM-packing runner. budget_gb defaults to 85% of physical RAM (headroom for OS+KV)."""
    def __init__(self, budget_gb=None, swap_ceil_mb=20000, poll=15, statusf=None, log=print):
        self.tot = total_gb()
        self.budget = budget_gb if budget_gb else round(0.85 * self.tot, 1)
        self.swap_ceil = swap_ceil_mb; self.poll = poll; self.statusf = statusf; self.log = log

    def _used(self, running):
        return sum(j.est_gb for j in running)

    @staticmethod
    def _rss_gb(pgid):
        """Sum RSS of a job's whole process group (it spawns bakers/doctors). start_new_session
        makes pgid == the job pid. Returns GB; 0 on any error."""
        try:
            out = subprocess.run(["ps", "-o", "rss=", "-g", str(pgid)], capture_output=True, text=True).stdout
            return sum(int(x) for x in out.split()) / (1024 * 1024)
        except Exception:
            return 0.0

    def _launch(self, j):
        fh = open(j.log, "a") if j.log else subprocess.DEVNULL
        env = {**os.environ, **(j.env or {})}
        j.proc = subprocess.Popen(j.argv, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                  start_new_session=True)
        j.t0 = time.time()
        self.log(f"[sched] launch {j.name} est={j.est_gb:.0f}GB solo={j.solo} pid={j.proc.pid}")

    def run(self, jobs):
        pending = [j for j in jobs if not j.is_done()]
        for j in (set(jobs) - set(pending)):
            self.log(f"[sched] skip {j.name} (artifact exists)")
        running, results = [], {}
        self.log(f"[sched] RAM {self.tot:.0f}GB, budget {self.budget:.0f}GB, "
                 f"{len(pending)} jobs to run, {len(jobs)-len(pending)} already done")
        while pending or running:
            for j in running:                             # sample real peak RSS (adaptive estimates)
                j.peak_gb = max(j.peak_gb, self._rss_gb(j.proc.pid))
            for j in list(running):                       # reap finished
                if j.proc.poll() is not None:
                    results[j.name] = j.proc.returncode
                    self.log(f"[sched] done {j.name} rc={j.proc.returncode} "
                             f"{time.time()-j.t0:.0f}s peak={j.peak_gb:.1f}GB (est {j.est_gb:.0f})")
                    try:
                        with open("reports/cron/ram_actuals.jsonl", "a") as f:
                            f.write(json.dumps({"name": j.name, "est_gb": j.est_gb,
                                                "peak_gb": round(j.peak_gb, 2),
                                                "wall_s": round(time.time() - j.t0)}) + "\n")
                    except Exception:
                        pass
                    running.remove(j)
            sw = swap_mb()
            if sw < self.swap_ceil:                        # only launch when swap is healthy
                solo_running = any(r.solo for r in running)
                for j in list(pending):
                    if j.solo:
                        if not running:                    # solo gets the whole empty box
                            self._launch(j); running.append(j); pending.remove(j); solo_running = True
                        break                              # nothing else while a solo is queued next
                    if solo_running:
                        break
                    if self._used(running) + j.est_gb <= self.budget:
                        self._launch(j); running.append(j); pending.remove(j)
            self._status(running, pending, results, sw)
            if pending or running:
                time.sleep(self.poll)
        return results

    def _status(self, running, pending, results, sw):
        line = (f"running={len(running)} ({self._used(running):.0f}/{self.budget:.0f}GB) "
                f"pending={len(pending)} done={len(results)} swap={sw:.0f}MB :: "
                + ", ".join(j.name for j in running))
        if self.statusf:
            try:
                open(self.statusf, "w").write(line + "\n")
            except Exception:
                pass

    def plan(self, jobs):
        """Dry-run: simulate the packing schedule and print the wave plan. Lets the whole chain be
        verified on the small box without running any compute."""
        q = list(jobs); wave = 1
        print(f"# RAM-pack plan: {self.tot:.0f}GB physical, {self.budget:.0f}GB budget")
        while q:
            if q[0].solo:                                  # solo job: its own wave, whole box
                j = q.pop(0)
                print(f"  wave {wave} [SOLO]: {j.est_gb:.0f}GB :: {j.name}({j.est_gb:.0f})")
                wave += 1; continue
            batch, used, i = [], 0.0, 0
            while i < len(q):
                j = q[i]
                if j.solo:
                    break                                  # stop packing at the next solo boundary
                if used + j.est_gb <= self.budget:
                    used += j.est_gb; batch.append(q.pop(i))
                else:
                    i += 1
            if not batch:                                  # nothing fit: biggest job > budget on this box
                j = min(q, key=lambda x: x.est_gb); q.remove(j)
                print(f"  wave {wave} [OVER-BUDGET: needs a bigger box]: "
                      f"{j.est_gb:.0f}GB :: {j.name}({j.est_gb:.0f})")
            else:
                print(f"  wave {wave}: {used:.0f}GB :: "
                      + ", ".join(f"{j.name}({j.est_gb:.0f})" for j in batch))
            wave += 1
        print(f"# {wave-1} waves (wall-clock ~= sum over waves of the slowest job in each wave)")


if __name__ == "__main__":
    # self-test: pack the real model ladder. Pass a budget GB to simulate a target box
    # (e.g. `ram_scheduler.py 82` for the 96 GB Studio); default = this machine.
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else None
    demo = [
        # bakes (cheap, chunk-bounded ~14GB) — parallelize 5-wide on the Studio
        *[Job(f"bake-{m}", ["true"], 14) for m in ("05b", "15b", "7b", "14b", "32b")],
        # labs (small) pack together
        Job("0.5B-lab", ["true"], 10), Job("1.5B-lab", ["true"], 10),
        # doctors (heavy) sized per plan §3
        Job("7B-doctor", ["true"], 40), Job("14B-doctor", ["true"], 65),
        Job("32B-doctor", ["true"], 85, solo=True),
    ]
    Scheduler(budget_gb=budget).plan(demo)
