/*
  voice.test.ts — the house voice as a CI gate, not a memory. Banned typographic characters
  (em dash, en dash, middot, ellipsis, bullet) must never reach rendered UI copy. Code comments
  are exempt; this strips them and asserts the remaining source (strings + JSX text) is clean.
*/
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, it, expect } from "vitest";

const BANNED = /[—–·…•]/; // — – · … •
const SRC = join(__dirname);

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const s = statSync(p);
    if (s.isDirectory()) out.push(...walk(p));
    else if (/\.(ts|tsx)$/.test(name) && !/\.test\.ts$/.test(name)) out.push(p);
  }
  return out;
}

// strip block comments then line comments so only strings + JSX text remain
function stripComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");
}

describe("house voice: no banned typographic characters in UI copy", () => {
  const files = walk(SRC);
  it("scans the whole source tree", () => {
    expect(files.length).toBeGreaterThan(20);
  });
  for (const file of files) {
    it(`clean: ${file.replace(SRC, "src")}`, () => {
      const body = stripComments(readFileSync(file, "utf8"));
      const lines = body.split("\n");
      const hits: string[] = [];
      lines.forEach((line, i) => {
        if (BANNED.test(line)) hits.push(`${i + 1}: ${line.trim().slice(0, 80)}`);
      });
      expect(hits, `banned chars in ${file}:\n${hits.join("\n")}`).toEqual([]);
    });
  }
});
