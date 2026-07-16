import { describe, it, expect } from "vitest";
import { cycleIndex, flattenVisible, treeKeyTarget } from "./policies";
import type { FileNode } from "../surfaces/types";

describe("cycleIndex", () => {
  it("wraps forward off the end", () => {
    expect(cycleIndex(3, 2, false)).toBe(0);
    expect(cycleIndex(3, 0, false)).toBe(1);
  });
  it("wraps backward off the start", () => {
    expect(cycleIndex(3, 0, true)).toBe(2);
    expect(cycleIndex(3, 2, true)).toBe(1);
  });
  it("is safe for an empty list", () => {
    expect(cycleIndex(0, 0, false)).toBe(0);
    expect(cycleIndex(0, -1, true)).toBe(0);
  });
});

const TREE: FileNode[] = [
  {
    path: "src",
    name: "src",
    dir: true,
    children: [
      { path: "src/a.ts", name: "a.ts", dir: false },
      {
        path: "src/sub",
        name: "sub",
        dir: true,
        children: [{ path: "src/sub/b.ts", name: "b.ts", dir: false }],
      },
    ],
  },
  { path: "README.md", name: "README.md", dir: false },
];

describe("flattenVisible", () => {
  it("includes children of expanded dirs with correct levels", () => {
    const rows = flattenVisible(TREE, {});
    expect(rows.map((r) => r.path)).toEqual(["src", "src/a.ts", "src/sub", "src/sub/b.ts", "README.md"]);
    expect(rows.find((r) => r.path === "src")?.level).toBe(0);
    expect(rows.find((r) => r.path === "src/sub/b.ts")?.level).toBe(2);
  });
  it("hides children under a collapsed dir", () => {
    const rows = flattenVisible(TREE, { src: true });
    expect(rows.map((r) => r.path)).toEqual(["src", "README.md"]);
  });
  it("collapses only the marked subtree", () => {
    const rows = flattenVisible(TREE, { "src/sub": true });
    expect(rows.map((r) => r.path)).toEqual(["src", "src/a.ts", "src/sub", "README.md"]);
  });
});

describe("treeKeyTarget", () => {
  const rows = flattenVisible(TREE, {});

  it("ArrowDown / ArrowUp move between visible rows", () => {
    expect(treeKeyTarget(rows, "src", "ArrowDown")).toEqual({ kind: "focus", path: "src/a.ts" });
    expect(treeKeyTarget(rows, "src/a.ts", "ArrowUp")).toEqual({ kind: "focus", path: "src" });
  });
  it("stops at the ends", () => {
    expect(treeKeyTarget(rows, "README.md", "ArrowDown")).toBeNull();
    expect(treeKeyTarget(rows, "src", "ArrowUp")).toBeNull();
  });
  it("Home / End jump to the extremes", () => {
    expect(treeKeyTarget(rows, "src/a.ts", "Home")).toEqual({ kind: "focus", path: "src" });
    expect(treeKeyTarget(rows, "src/a.ts", "End")).toEqual({ kind: "focus", path: "README.md" });
  });
  it("ArrowRight expands a collapsed dir", () => {
    const collapsedRows = flattenVisible(TREE, { src: true });
    expect(treeKeyTarget(collapsedRows, "src", "ArrowRight")).toEqual({ kind: "toggle", path: "src", expand: true });
  });
  it("ArrowRight on an expanded dir steps into the first child", () => {
    expect(treeKeyTarget(rows, "src", "ArrowRight")).toEqual({ kind: "focus", path: "src/a.ts" });
  });
  it("ArrowLeft collapses an expanded dir", () => {
    expect(treeKeyTarget(rows, "src", "ArrowLeft")).toEqual({ kind: "toggle", path: "src", expand: false });
  });
  it("ArrowLeft on a leaf jumps to the parent", () => {
    expect(treeKeyTarget(rows, "src/sub/b.ts", "ArrowLeft")).toEqual({ kind: "focus", path: "src/sub" });
  });
  it("ArrowRight on a file is a no-op", () => {
    expect(treeKeyTarget(rows, "README.md", "ArrowRight")).toBeNull();
  });
});
