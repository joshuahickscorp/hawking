"""Cargo workspace extraction: crates, targets, features, dep graph."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from graph_model import Graph, make_node, subsystem_for


def _run_cargo_metadata(repo: Path) -> dict[str, Any]:
    r = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        # Fallback: parse workspace Cargo.toml members
        return {"packages": [], "workspace_members": [], "error": r.stderr}
    return json.loads(r.stdout)


def _parse_toml_features(cargo_toml: Path) -> dict[str, list[str]]:
    """Minimal [features] table reader without a TOML dependency."""
    if not cargo_toml.exists():
        return {}
    text = cargo_toml.read_text(errors="ignore")
    features: dict[str, list[str]] = {}
    in_features = False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("["):
            in_features = s == "[features]"
            continue
        if not in_features:
            continue
        if "=" not in s:
            continue
        key, _, rest = s.partition("=")
        key = key.strip().strip('"')
        rest = rest.strip()
        deps: list[str] = []
        if rest.startswith("["):
            for m in re.finditer(r'"([^"]+)"', rest):
                deps.append(m.group(1))
        features[key] = deps
    return features


def _parse_toml_deps(cargo_toml: Path) -> list[dict[str, Any]]:
    """Parse [dependencies] / [dev-dependencies] / [build-dependencies] for path/workspace deps."""
    if not cargo_toml.exists():
        return []
    text = cargo_toml.read_text(errors="ignore")
    deps: list[dict[str, Any]] = []
    section = None
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("["):
            if s in (
                "[dependencies]",
                "[dev-dependencies]",
                "[build-dependencies]",
                "[workspace.dependencies]",
            ):
                section = s.strip("[]")
            elif s.startswith("[target.") and s.endswith("dependencies]"):
                section = "target-dependencies"
            elif s.startswith("["):
                section = None
            continue
        if section is None or section == "workspace.dependencies":
            # still collect workspace.dependencies names for optional resolution
            if section != "workspace.dependencies":
                continue
        if "=" not in s:
            continue
        name, _, rest = s.partition("=")
        name = name.strip().strip('"')
        rest = rest.strip()
        optional = "optional" in rest and "true" in rest
        # feature-gated: optional = true
        package = name
        pm = re.search(r'package\s*=\s*"([^"]+)"', rest)
        if pm:
            package = pm.group(1)
        # only path or workspace members interest us for imports edges later
        is_path = "path" in rest
        is_workspace = "workspace" in rest and "true" in rest
        deps.append({
            "name": name,
            "package": package.replace("_", "-") if False else package,
            "optional": optional,
            "section": section,
            "is_path": is_path,
            "is_workspace": is_workspace,
            "raw": rest,
        })
    return deps


def extract_cargo(repo: Path, g: Graph) -> dict[str, Any]:
    """Populate crate nodes and crate-level imports / feature_gates edges.

    Returns context: {
      package_names: set[str],
      name_to_id: dict[str, str],  # both hyphen and underscore forms
      package_root: dict[str, str],  # package name -> repo-relative dir
      members: list[dict],
    }
    """
    meta = _run_cargo_metadata(repo)
    packages = meta.get("packages") or []
    ctx: dict[str, Any] = {
        "package_names": set(),
        "name_to_id": {},
        "package_root": {},
        "members": [],
        "manifest_deps": {},  # package -> list deps
    }

    # Also parse workspace root for members if cargo failed
    if not packages:
        root_toml = repo / "Cargo.toml"
        text = root_toml.read_text(errors="ignore") if root_toml.exists() else ""
        members = re.findall(r'"([^"]+)"', text.split("[workspace]")[-1].split("[")[0] if "[workspace]" in text else "")
        for m in members:
            if m.endswith("*"):
                continue
            pdir = repo / m
            ct = pdir / "Cargo.toml"
            if not ct.exists():
                continue
            name_m = re.search(r'name\s*=\s*"([^"]+)"', ct.read_text(errors="ignore"))
            if not name_m:
                continue
            packages.append({
                "name": name_m.group(1),
                "manifest_path": str(ct),
                "targets": [],
                "features": {},
            })

    for pkg in packages:
        name = pkg["name"]
        manifest = Path(pkg["manifest_path"])
        try:
            rel_manifest = str(manifest.relative_to(repo))
        except ValueError:
            rel_manifest = str(manifest)
        root_dir = str(Path(rel_manifest).parent)
        if root_dir == ".":
            root_dir = ""
        nid = f"crate:{name}"
        g.add_node(make_node(
            "crate",
            nid,
            name,
            path=rel_manifest,
            lang="toml",
            subsystem=subsystem_for(root_dir + "/" if root_dir else "crates/"),
            public=True,
            loc=0,
        ))
        ctx["package_names"].add(name)
        ctx["name_to_id"][name] = nid
        ctx["name_to_id"][name.replace("-", "_")] = nid
        ctx["name_to_id"][name.replace("_", "-")] = nid
        ctx["package_root"][name] = root_dir
        ctx["members"].append({
            "name": name,
            "manifest": rel_manifest,
            "root": root_dir,
            "targets": pkg.get("targets") or [],
            "features": pkg.get("features") or {},
        })

        # features from cargo metadata + toml
        feats = pkg.get("features") or _parse_toml_features(manifest)
        for fname, enables in (feats.items() if isinstance(feats, dict) else []):
            # feature_flag node
            flag_id = f"flag:{name}/{fname}" if False else f"flag:{fname}"
            # Prefer HAWKING-style later; for cargo features use package-qualified only if collision
            # Schema: flag:<name> — use feature name; disambiguate with package prefix when needed
            flag_id = f"flag:cargo:{name}:{fname}"
            g.add_node(make_node(
                "feature_flag",
                flag_id,
                f"{name}/{fname}",
                path=rel_manifest,
                lang="toml",
                public=True,
            ))
            # feature_gates edges from feature -> enabled deps that look like crates
            for en in enables or []:
                # "dep:foo" or "foo/bar" or just "foo"
                dep_name = en
                if dep_name.startswith("dep:"):
                    dep_name = dep_name[4:]
                dep_name = dep_name.split("/")[0]
                target = ctx["name_to_id"].get(dep_name) or ctx["name_to_id"].get(
                    dep_name.replace("_", "-")
                )
                if target:
                    g.add_edge(
                        nid, "feature_gates", target,
                        evidence="cargo", confidence=1.0,
                    )

        deps = _parse_toml_deps(manifest)
        ctx["manifest_deps"][name] = deps

    # Second pass: crate imports edges from deps to known workspace members
    for name, deps in ctx["manifest_deps"].items():
        src = ctx["name_to_id"][name]
        for d in deps:
            pkg_name = d["package"]
            # normalise
            dst = (
                ctx["name_to_id"].get(pkg_name)
                or ctx["name_to_id"].get(pkg_name.replace("_", "-"))
                or ctx["name_to_id"].get(pkg_name.replace("-", "_"))
            )
            if not dst:
                continue
            if d.get("optional"):
                g.add_edge(
                    src, "feature_gates", dst,
                    evidence="cargo", confidence=1.0,
                )
            g.add_edge(
                src, "imports", dst,
                evidence="cargo", confidence=1.0,
            )

    # contains: repository -> crates
    for name in sorted(ctx["package_names"]):
        g.ensure_contains("repo", ctx["name_to_id"][name], evidence="cargo")

    return ctx


def crate_for_path(path: str, ctx: dict[str, Any]) -> str | None:
    """Map a repo-relative source path to its crate package name."""
    best = None
    best_len = -1
    for name, root in ctx["package_root"].items():
        if not root:
            continue
        prefix = root if root.endswith("/") else root + "/"
        if path.startswith(prefix) or path.startswith(root + "/"):
            if len(root) > best_len:
                best = name
                best_len = len(root)
        elif path == root or path == root + "/Cargo.toml":
            if len(root) > best_len:
                best = name
                best_len = len(root)
    # tools/* cargo projects
    if best is None and path.startswith("tools/"):
        parts = path.split("/")
        if len(parts) >= 2:
            candidate = "/".join(parts[:2])
            for name, root in ctx["package_root"].items():
                if root == candidate:
                    return name
    return best
