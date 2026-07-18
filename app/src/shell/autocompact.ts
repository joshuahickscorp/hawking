/*
  autocompact.ts — secret, proactive context compaction. The engine (hawking-context::compiler) already
  fires its own compaction at the CRITICAL watermark (>= 90% occupancy) with a recall gate that reverts
  if a needle drops. This hook gets AHEAD of that cliff: during live work (generation or code changes)
  it silently asks the host to compact once the window is climbing, so a turn never stalls at the edge.

  Doctrine: this is invisible. No cap, no meter, no percentage reaches the UI. The only visible trace is
  ambient light in the Context Stack (the "compacting" watermark word), which already exists. The policy
  is pure and unit-tested; the hook is the thin wiring to the store and the intent seam.
*/
import { useEffect, useRef } from "react";
import { sendIntent } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";
import type { RunPhase } from "../store";

export type Watermark = "normal" | "soft" | "warn" | "critical";

export interface CompactSignal {
  watermark?: Watermark;
  occupancy?: number; // 0..1 against the effective ceiling
  active: boolean; // the agent is generating or changing code right now
}

export interface CompactPolicyState {
  lastFiredOccupancy: number; // occupancy the last time we fired (-1 = never)
  armed: boolean; // re-armed once occupancy fell back a clear margin (a compaction took effect)
}

export const INITIAL_POLICY: CompactPolicyState = { lastFiredOccupancy: -1, armed: true };

// The band at which we glide into a soft compaction instead of thrashing on every soft tick.
const SOFT_GLIDE = 0.68;
// Re-arm once occupancy has fallen this far below the last fire (proof the compaction freed room).
const REARM_MARGIN = 0.15;

// Decide whether to proactively compact. Only while active (at rest the engine's own critical gate is
// enough and a background pass would be wasted work), only while armed (one fire per climb), and only
// once the window is genuinely climbing. Returns true to fire a silent compact_context intent.
export function shouldCompact(sig: CompactSignal, st: CompactPolicyState): boolean {
  if (!sig.active || !st.armed) return false;
  const wm = sig.watermark ?? "normal";
  const occ = sig.occupancy ?? 0;
  if (wm === "critical" || wm === "warn") return true;
  if (wm === "soft" && occ >= SOFT_GLIDE) return true;
  return false;
}

// Advance the policy after a tick. Firing disarms until occupancy drops REARM_MARGIN below the fire
// point; that fall is the signal the host actually compacted, so we may glide again on the next climb.
export function nextPolicyState(fired: boolean, occ: number, st: CompactPolicyState): CompactPolicyState {
  if (fired) return { lastFiredOccupancy: occ, armed: false };
  if (!st.armed && occ <= Math.max(0, st.lastFiredOccupancy - REARM_MARGIN)) {
    return { ...st, armed: true };
  }
  return st;
}

const ACTIVE_PHASES: ReadonlySet<RunPhase> = new Set<RunPhase>(["planning", "executing", "awaiting"]);

// Wire the policy to the live manifest. Runs whenever the watermark, occupancy, or phase changes; fires
// at most once per climb and never surfaces anything to the user.
export function useAutoCompact(): void {
  const watermark = useStore((s) => s.manifest?.live?.watermark) as Watermark | undefined;
  const occupancy = useStore((s) => s.manifest?.live?.occupancy);
  const runPhase = useStore((s) => s.runPhase);
  const sessionId = useStore((s) => s.sessionId);
  const st = useRef<CompactPolicyState>({ ...INITIAL_POLICY });

  useEffect(() => {
    const sig: CompactSignal = { watermark, occupancy, active: ACTIVE_PHASES.has(runPhase) };
    const fire = shouldCompact(sig, st.current);
    if (fire) {
      void sendIntent(
        intent.custom("compact_context", {
          session_id: sessionId,
          reason: "watermark",
          occupancy: sig.occupancy ?? null,
          watermark: sig.watermark ?? null,
        }),
      );
    }
    st.current = nextPolicyState(fire, sig.occupancy ?? 0, st.current);
  }, [watermark, occupancy, runPhase, sessionId]);
}
