from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

IGNORED_DIRS: Set[str] = {
    ".git", ".haider", "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", "target", "coverage", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".idea", ".vscode", ".eggs",
}

MANIFEST_PATTERNS: Set[str] = {
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "package.json", "tsconfig.json", "Cargo.toml", "go.mod", "go.sum",
    "Makefile", "CMakeLists.txt", "Dockerfile", "docker-compose.yml",
    "README.md", "README.rst", "LICENSE", "LICENSE.md",
}

TEST_PATTERNS: Set[str] = {
    "test_", "_test", "tests", "spec", "conftest",
}


def _is_test_path(rel: str) -> bool:
    parts = rel.replace("\\", "/").split("/")
    for p in parts:
        if p in TEST_PATTERNS or p.startswith("test_") or p.endswith("_test.py"):
            return True
    return False


def _infer_language(path: str) -> Optional[str]:
    ext = Path(path).suffix.lower()
    mapping = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".rs": "rust", ".go": "go", ".c": "c", ".cpp": "cpp",
        ".h": "c", ".hpp": "cpp", ".java": "java", ".rb": "ruby",
        ".md": "markdown", ".json": "json", ".toml": "toml",
        ".yml": "yaml", ".yaml": "yaml", ".html": "html", ".css": "css",
    }
    return mapping.get(ext)


@dataclass
class WorkspaceIndex:
    root: str
    files: List[str] = field(default_factory=list)
    languages: Dict[str, int] = field(default_factory=dict)
    manifests: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    entrypoints: List[str] = field(default_factory=list)
    git_root: Optional[str] = None
    git_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "file_count": len(self.files),
            "languages": self.languages,
            "manifests": self.manifests,
            "test_files": self.test_files,
            "entrypoints": self.entrypoints,
            "git_root": self.git_root,
            "git_available": self.git_available,
        }


def build_workspace_index(root: str) -> WorkspaceIndex:
    root = os.path.realpath(root)
    idx = WorkspaceIndex(root=root)

    # Detect git
    try:
        import subprocess
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            idx.git_available = True
            idx.git_root = os.path.realpath(proc.stdout.strip())
    except Exception:
        pass

    # Walk files
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if rel.startswith(".haider"):
                continue
            idx.files.append(rel)
            lang = _infer_language(fn)
            if lang:
                idx.languages[lang] = idx.languages.get(lang, 0) + 1
            if fn in MANIFEST_PATTERNS:
                idx.manifests.append(rel)
            if _is_test_path(rel):
                idx.test_files.append(rel)
            if fn in ("main.py", "__main__.py", "index.js", "index.ts", "main.rs", "main.go"):
                idx.entrypoints.append(rel)

    return idx


def resolve_file_reference(text: str, root: str) -> Optional[str]:
    """Resolve explicit file references in natural language.

    Returns the relative path if found, else None.
    """
    # Match common file extensions in the text
    pattern = re.compile(r'\b([\w./-]+\.(?:md|py|js|ts|rs|go|c|cpp|h|json|toml|yml|yaml|txt|html|css|sh))\b')
    for match in pattern.finditer(text):
        candidate = match.group(1)
        full = os.path.join(root, candidate)
        if os.path.isfile(full):
            return candidate
    return None


def read_file_reference(text: str, root: str) -> Optional[Dict[str, Any]]:
    """Resolve and read a file reference from natural language text."""
    rel = resolve_file_reference(text, root)
    if rel is None:
        return None
    full = os.path.join(root, rel)
    try:
        content = Path(full).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    return {"path": rel, "content": content, "size": len(content)}
