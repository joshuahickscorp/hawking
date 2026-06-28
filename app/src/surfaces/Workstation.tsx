/*
  Workstation.tsx: the AI Workstation surface (HIDE_PLAN §D, §111-114). THE FRONT DOOR.
  An observatory, not a cockpit: a calm board of parallel agents, the one-control WEDGE that
  fans out N branches with the gold edge travelling parent->child (the state memcpy, rendered),
  a morning-digest big-number (032c editorial moment), and a merge-review queue whose hunk gesture
  (j/k/a/r) is identical to the IDE's. Per-run timeline strips scrub/fork over the event log.

  Fed by projection_patch(fleet) via the store (the source of truth); the fork/timeline/merge
  interaction layers local view-state the mock transport does not script, so the surface is ALIVE.

  Sends: ForkSession, ScrubToEvent, Custom:fleet_run, PauseRun/CancelRun, AcceptDiff/RejectDiff,
  Custom:revert_diff. Consumes: projection_patch(fleet) (store.fleet) + (in live mode) projection_patch(diff).

  Touches no shared foundation file. Helpers live under surfaces/workstation/.
*/
import { useMemo, useState } from "react";
import { useStore, type FleetRun } from "../store";
import { Display, Panel, SectionLabel } from "../ui";
import { FleetBoard } from "./workstation/board";
import { HunkReview, type ReviewBranch } from "./workstation/hunkreview";
import { MOCK_BRANCHES } from "./workstation/mockdiffs";

export function Workstation() {
  const fleet = useStore((s) => s.fleet);
  return (
    <div style={{ height: "100%", overflowY: "auto", padding: "var(--s6)" }}>
      <div style={{ maxWidth: 1040, margin: "0 auto", display: "flex", flexDirection: "column", gap: "var(--s6)" }}>
        <Digest fleet={fleet} />
        <FleetBoard />
        <MergeQueue branches={MOCK_BRANCHES} />
      </div>
    </div>
  );
}

// ---- The morning digest: one editorial Cormorant number, the observatory at dawn (C §411). ----
// Ranked the way the doctrine ranks it: what needs you FIRST, then ready-to-merge, then failed.
function Digest({ fleet }: { fleet: FleetRun[] }) {
  const counts = useMemo(() => tally(fleet), [fleet]);
  const ran = fleet.length;
  // the hero line resolves to the calm state of the night's work.
  const hero =
    ran === 0
      ? "Nothing running. A quiet observatory."
      : `${ran} agent${ran === 1 ? "" : "s"} ran. ${counts.waiting} need${counts.waiting === 1 ? "s" : ""} you.`;

  return (
    <header>
      <Display size={44}>{hero}</Display>
      <p style={{ color: "var(--text-low)", marginTop: "var(--s3)" }}>
        Walk in to the night composed into one view, not fifty notifications. Spend lavishly, locally.
      </p>
      <div style={{ display: "flex", gap: "var(--s2)", flexWrap: "wrap", marginTop: "var(--s4)" }}>
        {/* needs-you ranked first, in amber; then ready-to-merge jade; then failed red. */}
        <DigestStat n={counts.waiting} label="need you" tone="var(--warning)" lead />
        <DigestStat n={counts.done} label="ready to merge" tone="var(--success)" />
        <DigestStat n={counts.active} label="still running" tone="var(--radiation)" />
        <DigestStat n={counts.failed} label="failed" tone="var(--danger)" />
      </div>
    </header>
  );
}

function tally(fleet: FleetRun[]) {
  const c = { active: 0, waiting: 0, done: 0, failed: 0 };
  for (const r of fleet) c[r.state] += 1;
  return c;
}

function DigestStat({ n, label, tone, lead }: { n: number; label: string; tone: string; lead?: boolean }) {
  const lit = n > 0;
  return (
    <Panel
      pad="var(--s3) var(--s4)"
      active={lead && lit}
      style={{ display: "flex", alignItems: "baseline", gap: "var(--s2)", minWidth: 132 }}
    >
      <Display size={28} style={{ color: lit ? tone : "var(--text-low)" }}>
        {n}
      </Display>
      <span style={{ color: lit ? "var(--text-mid)" : "var(--text-low)", fontSize: "var(--text-xs)" }}>{label}</span>
    </Panel>
  );
}

// ---- Merge review: a calm queue of finished branches, each reviewed with the IDE's hunk gesture. ----
function MergeQueue({ branches }: { branches: ReviewBranch[] }) {
  const [openId, setOpenId] = useState<string>(branches[0]?.diff_id ?? "");
  const open = branches.find((b) => b.diff_id === openId) ?? branches[0];

  if (branches.length === 0) {
    return (
      <section>
        <SectionLabel>Merge review</SectionLabel>
        <Panel pad="var(--s5)" style={{ color: "var(--text-low)" }}>
          Nothing to merge yet. Finished branches queue here for hunk-by-hunk review.
        </Panel>
      </section>
    );
  }

  return (
    <section>
      <SectionLabel count={branches.length}>Merge review</SectionLabel>
      <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: "var(--s3)", minHeight: 0 }}>
        {/* the queue: pick a branch's diff to review. */}
        <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: "var(--s2)" }}>
          {branches.map((b) => {
            const on = b.diff_id === open?.diff_id;
            return (
              <li key={b.diff_id}>
                <button
                  onClick={() => setOpenId(b.diff_id)}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    padding: "var(--s2) var(--s3)",
                    borderRadius: "var(--radius)",
                    color: on ? "var(--text-hi)" : "var(--text-mid)",
                    background: on ? "var(--surface-1)" : "transparent",
                    boxShadow: on ? "inset 0 0 0 1px var(--radiation)" : "inset 0 0 0 1px var(--rim)",
                  }}
                >
                  <div>{b.label}</div>
                  <div style={{ color: "var(--text-low)", fontSize: "var(--text-xs)" }}>{b.path}</div>
                </button>
              </li>
            );
          })}
        </ul>

        {/* the reviewer: same j/k/a/r gesture as the IDE diff focus. */}
        <Panel pad="var(--s4)" style={{ minHeight: 0 }}>
          {open ? <HunkReview branch={open} /> : null}
        </Panel>
      </div>
    </section>
  );
}
