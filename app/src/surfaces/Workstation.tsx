/*
  Workstation.tsx: the AI Workstation surface. THE FRONT DOOR, the courtyard.
  An observatory, not a cockpit: a calm board of parallel agents, the one-control WEDGE that
  fans out N branches with the light edge travelling parent->child (the state memcpy, rendered),
  a morning-digest headline alone in .t-display (the editorial moment), and a merge-review queue
  whose hunk gesture (j/k/a/r) is identical to the IDE's. Per-run timeline strips scrub/fork.

  Doctrine v3 (Tadao Ando grayscale concrete): the largest, calmest voids in the building.
  The digest headline stands alone in generous void (--ma-18 above, --ma-14 below); the fleet
  cards are .volume slabs floating with >= --ma-8 gaps; state is read by LIGHT, never a colored
  badge; there is no third color and no amber, only the two pigments --ok and --bad. Nothing
  touches an edge; the void is the subject.

  Fed by projection_patch(fleet) via the store (the source of truth); the fork/timeline/merge
  interaction layers local view-state the mock transport does not script, so the surface is ALIVE.

  Sends: ForkSession, ScrubToEvent, Custom:fleet_run, PauseRun/CancelRun, AcceptDiff/RejectDiff,
  Custom:revert_diff. Consumes: projection_patch(fleet) (store.fleet) + (in live mode) projection_patch(diff).

  Touches no shared foundation file. Helpers live under surfaces/workstation/.
*/
import { useMemo, useState } from "react";
import { useStore, type FleetRun } from "../store";
import { Display, SectionLabel, Volume } from "../ui";
import { FleetBoard } from "./workstation/board";
import { HunkReview, type ReviewBranch } from "./workstation/hunkreview";
import { MOCK_BRANCHES } from "./workstation/mockdiffs";

export function Workstation() {
  const fleet = useStore((s) => s.fleet);
  return (
    <div style={{ height: "100%", overflowY: "auto", padding: "var(--ma-10)" }}>
      <div style={{ maxWidth: 1040, margin: "0 auto", display: "flex", flexDirection: "column", gap: "var(--ma-14)" }}>
        <Digest fleet={fleet} />
        <FleetBoard />
        <MergeQueue branches={MOCK_BRANCHES} />
      </div>
    </div>
  );
}

// ---- The morning digest: the headline alone in .t-display, the observatory at dawn. ----
// Ranked the way the doctrine ranks it: what needs you FIRST, then ready-to-merge, then failed.
function Digest({ fleet }: { fleet: FleetRun[] }) {
  const counts = useMemo(() => tally(fleet), [fleet]);
  const ran = fleet.length;
  // the hero line resolves the night's work to one calm sentence (no em/en dash, drop trailing period).
  const hero =
    ran === 0
      ? "Nothing running. A quiet observatory"
      : `${ran} agent${ran === 1 ? "" : "s"} ran. ${counts.waiting} need${counts.waiting === 1 ? "s" : ""} you`;

  return (
    <header style={{ paddingTop: "var(--ma-18)" }}>
      <Display>{hero}</Display>
      <p className="t-body" style={{ color: "var(--text-2)", marginTop: "var(--ma-6)", marginBottom: 0, maxWidth: 640 }}>
        Walk in to the night composed into one view, not fifty notifications. Spend lavishly, locally.
      </p>
      <div style={{ display: "flex", gap: "var(--ma-4)", flexWrap: "wrap", marginTop: "var(--ma-14)" }}>
        {/* needs-you ranked first (a lit volume, the agent asking); then ready-to-merge (--ok);
            then still running (light); then failed (--bad). No orange, no third color. */}
        <DigestStat n={counts.waiting} label="need you" glyph="◆" tone="var(--light)" lead />
        <DigestStat n={counts.done} label="ready to merge" glyph="●" tone="var(--ok)" />
        <DigestStat n={counts.active} label="still running" glyph="›" tone="var(--light)" />
        <DigestStat n={counts.failed} label="failed" glyph="✕" tone="var(--bad)" />
      </div>
    </header>
  );
}

function tally(fleet: FleetRun[]) {
  const c = { active: 0, waiting: 0, done: 0, failed: 0 };
  for (const r of fleet) c[r.state] += 1;
  return c;
}

function DigestStat({
  n,
  label,
  glyph,
  tone,
  lead,
}: {
  n: number;
  label: string;
  glyph: string;
  tone: string;
  lead?: boolean;
}) {
  const lit = n > 0;
  // a "needs you" volume that has anything pending holds the steady light (the agent asking for you).
  const litSteady = lead && lit
    ? { boxShadow: "var(--hairline-strong), var(--light-bloom), var(--inner-glow)" }
    : {};
  return (
    <Volume
      pad="var(--ma-4) var(--ma-6)"
      style={{ display: "flex", alignItems: "baseline", gap: "var(--ma-3)", minWidth: 152, ...litSteady }}
    >
      <span style={{ color: lit ? tone : "var(--text-3)", fontSize: "11px", lineHeight: 1, alignSelf: "center" }}>
        {glyph}
      </span>
      <Display style={{ fontSize: "28px", letterSpacing: "-0.02em", color: lit ? "var(--text-1)" : "var(--text-3)" }}>
        {n}
      </Display>
      <span className="t-micro" style={{ color: lit ? "var(--text-2)" : "var(--text-3)" }}>{label}</span>
    </Volume>
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
        <Volume pad="var(--ma-8)" style={{ color: "var(--text-3)", marginTop: "var(--ma-6)" }}>
          Nothing to merge yet. Finished branches queue here for hunk-by-hunk review.
        </Volume>
      </section>
    );
  }

  return (
    <section>
      <SectionLabel count={branches.length}>Merge review</SectionLabel>
      <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: "var(--ma-6)", minHeight: 0, marginTop: "var(--ma-6)" }}>
        {/* the queue: pick a branch's diff to review. */}
        <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: "var(--ma-3)" }}>
          {branches.map((b) => {
            const on = b.diff_id === open?.diff_id;
            return (
              <li key={b.diff_id}>
                <button
                  onClick={() => setOpenId(b.diff_id)}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    padding: "var(--ma-3) var(--ma-4)",
                    borderRadius: "var(--radius)",
                    color: on ? "var(--text-1)" : "var(--text-2)",
                    background: on ? "var(--concrete-3)" : "transparent",
                    boxShadow: on ? "var(--hairline-strong), var(--light-bloom)" : "var(--hairline)",
                    transition: "color var(--dur) var(--ease), box-shadow var(--dur) var(--ease)",
                  }}
                >
                  <div className="t-body">{b.label}</div>
                  <div className="t-micro" style={{ marginTop: "var(--ma-1)" }}>{b.path}</div>
                </button>
              </li>
            );
          })}
        </ul>

        {/* the reviewer: same j/k/a/r gesture as the IDE diff focus. */}
        <Volume pad="var(--ma-6)" style={{ minHeight: 0 }}>
          {open ? <HunkReview branch={open} /> : null}
        </Volume>
      </div>
    </section>
  );
}
