"""A scratch git repository that is COPIED, not rebuilt.

Sixty-six tests across four files each ran `git init` + two `git config` +
`add` + `commit`: five process spawns and 95 ms per test, for repositories
that were byte-identical within each file. Building one per interpreter and
copying it costs 8 ms, and removes ~330 process spawns from a suite whose
wall clock is dominated by process contention.

The copy is a real repository, not a snapshot of one: `git init` writes no
absolute paths into `.git/config`, so a copied tree commits, diffs and
rev-parses exactly like a freshly built one. `test_a_copied_repo_is_a_real_repo`
holds that.
"""
from __future__ import annotations

import atexit
import functools
import shutil
import subprocess
import tempfile
from pathlib import Path

_TEMPLATES: list[str] = []


@functools.lru_cache(maxsize=None)
def _template(email: str, name: str, filename: str, body: str, branch: str) -> str:
    root = tempfile.mkdtemp(prefix="hcli-git-template-")
    _TEMPLATES.append(root)
    repo = Path(root) / "repo"
    repo.mkdir()
    # An empty template skips git's sample hooks: 27 files in the repo become
    # 11, and every test copies the smaller tree (5.9 ms -> 3.3 ms). The hooks
    # are inert samples no test runs.
    empty_template = Path(root) / "empty-template"
    empty_template.mkdir()
    init = ["git", "init", "-q", f"--template={empty_template}"]
    init += ["-b", branch] if branch else []
    subprocess.run(init, cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", email], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", name], cwd=repo, check=True)
    (repo / filename).write_text(body, encoding="utf-8")
    subprocess.run(["git", "add", filename], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return str(repo)


@atexit.register
def _cleanup() -> None:
    for root in _TEMPLATES:
        shutil.rmtree(root, ignore_errors=True)


def scratch_repo(
    dest: Path,
    *,
    email: str,
    name: str,
    filename: str,
    body: str,
    branch: str = "main",
) -> Path:
    """Copy the template for these parameters to `dest` and return it."""
    shutil.copytree(_template(email, name, filename, body, branch), dest)
    return Path(dest)
