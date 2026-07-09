/*
  Digest.tsx — the courtyard's retrospective read: "what happened". A calm grid of totals and an
  activity heatmap rendered in the light ramp (dark cell = quiet day, near-white = the busiest).

  Doctrine: totals only, never a budget cap or a remaining percentage. The range control slices the
  heatmap window (7d / 30d / all), it is not a spend meter. One volume, generous air, no color.
*/
import { useMemo, useState } from "react";
import type { HomeDigest } from "../../store";
import { fmtCount, fmtDays, fmtHour, heatColumns } from "./metrics";

type Range = "all" | "30d" | "7d";
// Column counts for the week-grid view. 7d is rendered as a horizontal day strip instead (a single
// column read as a lonely sliver in a mostly-empty card), so it is not in this map.
const RANGE_COLS: Record<Exclude<Range, "7d">, number> = { all: Infinity, "30d": 5 };

export function Digest({ digest }: { digest: HomeDigest | null }) {
  const [range, setRange] = useState<Range>("all");
  const d = digest;

  const cols = d?.heatmap_cols ?? (d?.heatmap ? Math.ceil(d.heatmap.length / 7) : 0);
  const allColumns = useMemo(
    () => (d?.heatmap && cols > 0 ? heatColumns(d.heatmap, cols) : []),
    [d?.heatmap, cols],
  );
  const columns = useMemo(() => {
    if (range === "7d") return [];
    const want = RANGE_COLS[range];
    return want === Infinity ? allColumns : allColumns.slice(Math.max(0, allColumns.length - want));
  }, [allColumns, range]);
  // 7d: the last seven days as a full-width horizontal strip, so the card reads as a week, not a sliver.
  const days7 = useMemo(() => allColumns.flat().slice(-7), [allColumns]);

  const hasTokens = typeof d?.tokens === "number";
  // Two tiers so the card reads: a wall of primary totals, then quieter context.
  // Without this every stat is one 24px numeral and a model name reads like a metric.
  const primary: { label: string; value: string }[] = d
    ? [
        { label: "Sessions", value: fmtCount(d.sessions) },
        { label: "Messages", value: fmtCount(d.messages) },
        // Tokens only when the engine actually reports them (the local host omits, never faking a total).
        ...(hasTokens ? [{ label: "Tokens", value: fmtCount(d.tokens as number) }] : []),
        { label: "Active days", value: fmtCount(d.active_days) },
      ]
    : [];
  const secondary: { label: string; value: string }[] = d
    ? [
        { label: "Current streak", value: fmtDays(d.streak_current) },
        { label: "Longest streak", value: fmtDays(d.streak_longest) },
        { label: "Peak hour", value: fmtHour(d.peak_hour) },
        { label: "Favorite model", value: d.favorite_model || "local" },
      ]
    : [];

  return (
    <section className="volume digest" aria-label="Activity digest">
      <div className="digest__head">
        <span className="t-label">Digest</span>
        <div className="digest__range" role="tablist" aria-label="Digest range">
          {(["all", "30d", "7d"] as const).map((r) => (
            <button
              key={r}
              role="tab"
              aria-selected={range === r}
              className={"digest__rangebtn" + (range === r ? " digest__rangebtn--on" : "")}
              onClick={() => setRange(r)}
            >
              {r === "all" ? "All" : r}
            </button>
          ))}
        </div>
      </div>

      {d ? (
        <>
          <div className="metric-grid digest__metrics">
            {primary.map((m) => (
              <div key={m.label} className="metric">
                <div className="metric__value">{m.value}</div>
                <div className="metric__label t-label">{m.label}</div>
              </div>
            ))}
          </div>

          {range === "7d" ? (
            days7.length ? (
              <div className="heatweek" aria-hidden>
                {days7.map((cell, i) => (
                  <span key={i} className="heatweek__cell" data-level={cell.level} />
                ))}
              </div>
            ) : null
          ) : columns.length ? (
            <div className="heatmap" aria-hidden>
              {columns.map((col, ci) => (
                <div key={ci} className="heatmap__col">
                  {col.map((cell, ri) => (
                    <span key={ri} className="heatmap__cell" data-level={cell.level} />
                  ))}
                </div>
              ))}
            </div>
          ) : null}

          <div className="digest__sub">
            {secondary.map((m) => (
              <div key={m.label} className="substat">
                <span className="substat__value">{m.value}</span>
                <span className="substat__label t-label">{m.label}</span>
              </div>
            ))}
          </div>

          <div className="digest__note t-micro">
            {hasTokens ? `${fmtCount(d.tokens as number)} tokens, all local. ` : ""}Nothing left your machine.
          </div>
        </>
      ) : (
        <div className="digest__empty t-body">No activity yet. Describe a task to start a session.</div>
      )}
    </section>
  );
}
