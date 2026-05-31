#!/usr/bin/env python3
"""Dependency-free percent-format .py -> Colab .ipynb converter.

Cells split on lines starting with `# %%`. `# %% [markdown]` starts a markdown
cell (leading "# " stripped from each line); `# %%` starts a code cell. Keeps
the notebooks authorable as real, py_compile-able Python files.

Usage: python3 colab/py_to_ipynb.py colab/01_awq_bytecut.py [...]
"""
import json
import sys
from pathlib import Path


def parse_cells(text):
    cells, cur, kind = [], [], None
    def flush():
        if kind is not None:
            src = cur[:]
            while src and src[0].strip() == "":
                src.pop(0)
            while src and src[-1].strip() == "":
                src.pop()
            if src or kind == "code":
                cells.append((kind, src))
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# %%"):
            flush()
            cur, kind = [], ("markdown" if "[markdown]" in s else "code")
            continue
        if kind is None:
            continue
        if kind == "markdown":
            cur.append(line[2:] if line.startswith("# ") else
                       (line[1:] if line.startswith("#") else line))
        else:
            cur.append(line)
    flush()
    return cells


def to_ipynb(cells):
    out = []
    for kind, src in cells:
        lines = [l + "\n" for l in src[:-1]] + ([src[-1]] if src else [])
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    for arg in sys.argv[1:]:
        p = Path(arg)
        cells = parse_cells(p.read_text())
        nb = to_ipynb(cells)
        out = p.with_suffix(".ipynb")
        out.write_text(json.dumps(nb, indent=1))
        n_code = sum(1 for k, _ in cells if k == "code")
        n_md = sum(1 for k, _ in cells if k == "markdown")
        print(f"{out}  ({n_code} code + {n_md} md cells)")


if __name__ == "__main__":
    main()
