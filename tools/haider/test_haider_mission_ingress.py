import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HAIDER = ROOT / "tools" / "haider" / "haider.py"


def run(*args):
    return subprocess.run(
        [sys.executable, str(HAIDER), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


help_run = run("--help")

assert help_run.returncode == 0
assert "--task" in help_run.stdout
assert "--task-file" in help_run.stdout

missing = run("1", "--task-file", "/definitely/not/a/real/haider/task.md")

assert missing.returncode != 0
assert "task file not found" in missing.stderr.lower()

both = run(
    "1",
    "--task",
    "inline",
    "--task-file",
    "/tmp/not-used-haider-task.md",
)

assert both.returncode != 0
assert "mutually exclusive" in both.stderr.lower()

print("PASS: mission ingress CLI tests")
