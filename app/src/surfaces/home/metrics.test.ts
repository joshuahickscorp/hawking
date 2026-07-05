import { describe, it, expect } from "vitest";
import { fmtCount, fmtHour, fmtAge, fmtDays, heatLevels, heatColumns } from "./metrics";

describe("digest formatting", () => {
  it("formats counts with commas and M suffix", () => {
    expect(fmtCount(1182)).toBe("1,182");
    expect(fmtCount(119695)).toBe("119,695");
    expect(fmtCount(222_900_000)).toBe("222.9M");
    expect(fmtCount(4_000_000)).toBe("4.0M");
    expect(fmtCount(0)).toBe("0");
  });

  it("formats an hour as a 12h meridiem label", () => {
    expect(fmtHour(7)).toBe("7 AM");
    expect(fmtHour(0)).toBe("12 AM");
    expect(fmtHour(12)).toBe("12 PM");
    expect(fmtHour(17)).toBe("5 PM");
    expect(fmtHour(23)).toBe("11 PM");
  });

  it("formats relative age in the terse ladder", () => {
    const now = 10_000_000_000;
    expect(fmtAge(now, now)).toBe("now");
    expect(fmtAge(now - 5 * 60_000, now)).toBe("5m");
    expect(fmtAge(now - 3 * 3_600_000, now)).toBe("3h");
    expect(fmtAge(now - 2 * 86_400_000, now)).toBe("2d");
    expect(fmtAge(now - 21 * 86_400_000, now)).toBe("3w");
  });

  it("formats day counts", () => {
    expect(fmtDays(4)).toBe("4d");
    expect(fmtDays(0)).toBe("0d");
  });
});

describe("heatmap layout", () => {
  it("quantizes into 0..4 levels relative to the busiest cell", () => {
    // max=4: 1/4=0.25 -> 2, 2/4=0.5 -> 3, 4/4=1 -> 4; a zero cell stays dark.
    const cells = heatLevels([0, 1, 2, 4]);
    expect(cells.map((c) => c.level)).toEqual([0, 2, 3, 4]);
    // a lone busiest cell reads level 4; small fractions read level 1.
    expect(heatLevels([1, 100]).map((c) => c.level)).toEqual([1, 4]);
  });

  it("returns all-dark for an empty or zero series", () => {
    expect(heatLevels([0, 0, 0]).every((c) => c.level === 0)).toBe(true);
    expect(heatLevels([]).length).toBe(0);
  });

  it("lays out columns of exactly 7 cells, padding short tails", () => {
    const counts = Array.from({ length: 10 }, (_, i) => i);
    const cols = heatColumns(counts, 2);
    expect(cols.length).toBe(2);
    expect(cols.every((c) => c.length === 7)).toBe(true);
    // index 13 is missing from a 10-length series -> padded to level 0
    expect(cols[1][6].level).toBe(0);
  });
});
