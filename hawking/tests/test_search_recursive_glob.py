"""Recursive fs.search globs must find root and nested files."""

from pathlib import Path

from hawking.tool_registry import READ_ONLY, default_tool_registry


def test_fs_search_recursive_glob_finds_root_file(tmp_path):
    (tmp_path / "root.json").write_text("candidate marker\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.json").write_text("candidate marker\n")

    registry = default_tool_registry(
        tmp_path,
        repo_root=tmp_path,
        permissions={READ_ONLY},
    )
    result = registry.invoke("fs.search", {
        "path": ".",
        "pattern": "candidate marker",
        "glob": "**/*",
        "max_results": 10,
    })
    assert result.ok, result.error
    paths = {Path(row["path"]).name for row in result.value["matches"]}
    assert paths == {"root.json", "child.json"}
