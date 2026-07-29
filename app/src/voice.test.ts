import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, it, expect } from "vitest";
import { stripComments, walk } from "./test_fixtures";

const BANNED = /[—–·…•]/; // — – · … •
const SRC = join(__dirname);

describe("house voice: no banned typographic characters in UI copy", () => {
  const files = walk(SRC, { excludeTests: true });
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
