/*
  digest.ts — pure formatting + layout helpers for the courtyard digest (the "what's up next" read).
  Kept dependency-free so the number/heatmap logic is unit-tested without a DOM.

  Doctrine note: this is a RETROSPECTIVE activity read (what happened), never a budget cap. Totals are
  fine; a "remaining" percentage or a context-window meter is not, and none is computed here.
*/

// Compact magnitude for a stat value: 1_182 -> "1,182", 119_695 -> "119,695", 222_900_000 -> "222.9M".
export function fmtCount(n: number): string {
  if (!Number.isFinite(n)) return "0";
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  return Math.round(n).toLocaleString("en-US");
}

// 0..23 -> "7 AM" / "5 PM" / "12 AM". No leading zero, uppercase meridiem.
export function fmtHour(h: number): string {
  const hh = ((Math.round(h) % 24) + 24) % 24;
  const mer = hh < 12 ? "AM" : "PM";
  const twelve = hh % 12 === 0 ? 12 : hh % 12;
  return `${twelve} ${mer}`;
}

// Relative age from a timestamp (ms) to now (ms): "now", "4m", "2h", "3d", "5w".
export function fmtAge(tsMs: number, nowMs: number): string {
  const s = Math.max(0, Math.floor((nowMs - tsMs) / 1000));
  if (s < 45) return "now";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  const d = Math.floor(h / 24);
  if (d < 7) return `${d}d`;
  return `${Math.floor(d / 7)}w`;
}

// A day count worth of "d streak" / "d longest" etc. Keeps the unit terse.
export function fmtDays(n: number): string {
  return `${Math.max(0, Math.round(n))}d`;
}

export interface HeatCell {
  count: number;
  level: 0 | 1 | 2 | 3 | 4; // quantized intensity for the light ramp (0 = unlit)
}

// Quantize a flat activity series into cells with a 0..4 light level. Levels are relative to the busiest
// cell so a quiet week still reads, and an empty series is all level 0 (dark). Column-major is the
// caller's concern; this preserves order.
export function heatLevels(counts: readonly number[]): HeatCell[] {
  const max = counts.reduce((a, b) => Math.max(a, b), 0);
  return counts.map((count) => {
    if (count <= 0 || max <= 0) return { count, level: 0 as const };
    const frac = count / max;
    const level = frac >= 0.75 ? 4 : frac >= 0.5 ? 3 : frac >= 0.25 ? 2 : 1;
    return { count, level: level as HeatCell["level"] };
  });
}

// Grid geometry for the heatmap: 7 rows (days), N columns (weeks). Given a flat, row-major series of
// (cols x 7) and the column count, return columns of 7 cells each so the SVG/DOM can lay them out.
export function heatColumns(counts: readonly number[], cols: number): HeatCell[][] {
  const cells = heatLevels(counts);
  const safeCols = Math.max(1, Math.floor(cols));
  const out: HeatCell[][] = [];
  for (let c = 0; c < safeCols; c++) {
    const col: HeatCell[] = [];
    for (let r = 0; r < 7; r++) {
      const idx = c * 7 + r;
      col.push(cells[idx] ?? { count: 0, level: 0 });
    }
    out.push(col);
  }
  return out;
}
