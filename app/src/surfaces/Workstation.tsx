import { useMemo, useState } from "react";
import { useStore, type FleetRun } from "../store";
import { SectionLabel, SurfaceHeader, Volume } from "../ui";
import { FleetBoard } from "./workstation/board";
import { HunkReview, type ReviewBranch } from "./workstation/hunkreview";
import { MOCK_BRANCHES } from "./workstation/mockdiffs";

export function Workstation() {
  const fleet = useStore((s) => s.fleet);
  const counts = useMemo(() => tally(fleet), [fleet]);
  const ran = fleet.length;
  const title =
    ran === 0
      ? "No agents running"
      : `${ran} agent${ran === 1 ? "" : "s"} ran, ${counts.waiting} need${counts.waiting === 1 ? "s" : ""} you`;

  return (
    <div className="surface surface--workstation" style={{ gap: "var(--ma-14)" }}>
      <SurfaceHeader
        label="Workstation"
        title={title}
        meta={<span>{counts.active} active</span>}
      >
        Review live branches, finished diffs, and the work that needs a decision
      </SurfaceHeader>

      <Digest counts={counts} />
      <FleetBoard />
      <MergeQueue branches={MOCK_BRANCHES} />
    </div>
  );
}

function tally(fleet: FleetRun[]) {
  const c = { active: 0, waiting: 0, done: 0, failed: 0 };
  for (const r of fleet) c[r.state] += 1;
  return c;
}

function Digest({ counts }: { counts: ReturnType<typeof tally> }) {
  return (
    <section className="metric-grid" aria-label="Run summary">
      <Metric value={counts.waiting} label="Need you" glyph="◆" tone="var(--light)" lit={counts.waiting > 0} />
      <Metric value={counts.active} label="Active" glyph="●" tone="var(--light)" />
      <Metric value={counts.done} label="Ready" glyph="✓" tone="var(--ok)" />
      <Metric value={counts.failed} label="Failed" glyph="✕" tone="var(--bad)" />
    </section>
  );
}

function Metric({
  value,
  label,
  glyph,
  tone,
  lit,
}: {
  value: number;
  label: string;
  glyph: string;
  tone: string;
  lit?: boolean;
}) {
  return (
    <Volume className="metric" style={lit ? { boxShadow: "var(--hairline-strong), var(--light-bloom), var(--inner-glow)" } : undefined}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span className="t-label">{label}</span>
        <span aria-hidden style={{ color: value > 0 ? tone : "var(--text-3)" }}>{glyph}</span>
      </div>
      <div className="metric__value" style={{ color: value > 0 ? "var(--text-1)" : "var(--text-3)" }}>
        {value}
      </div>
    </Volume>
  );
}

function MergeQueue({ branches }: { branches: ReviewBranch[] }) {
  const [openId, setOpenId] = useState<string>(branches[0]?.diff_id ?? "");
  const open = branches.find((b) => b.diff_id === openId) ?? branches[0];

  if (branches.length === 0) {
    return (
      <section className="section-block">
        <SectionLabel>Merge review</SectionLabel>
        <Volume style={{ color: "var(--text-3)" }}>No diffs waiting</Volume>
      </section>
    );
  }

  return (
    <section className="section-block">
      <div className="section-head">
        <SectionLabel count={branches.length}>Merge review</SectionLabel>
      </div>
      <div className="merge-layout">
        <ul className="queue-list">
          {branches.map((b) => {
            const selected = b.diff_id === open?.diff_id;
            return (
              <li key={b.diff_id}>
                <button
                  className="ghost-button queue-item"
                  aria-selected={selected}
                  onClick={() => setOpenId(b.diff_id)}
                >
                  <div>
                    <div className="t-body" style={{ color: selected ? "var(--text-1)" : "var(--text-2)" }}>{b.label}</div>
                    <div className="t-micro" style={{ marginTop: "var(--ma-1)" }}>{b.path}</div>
                  </div>
                </button>
              </li>
            );
          })}
        </ul>

        <Volume style={{ minHeight: 0 }}>
          {open ? <HunkReview branch={open} /> : null}
        </Volume>
      </div>
    </section>
  );
}
