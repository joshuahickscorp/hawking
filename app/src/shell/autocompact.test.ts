import { describe, it, expect } from "vitest";
import { shouldCompact, nextPolicyState, INITIAL_POLICY, type CompactPolicyState } from "./policies";

const armed: CompactPolicyState = { lastFiredOccupancy: -1, armed: true };

describe("shouldCompact", () => {
  it("never fires at rest, however full the window", () => {
    expect(shouldCompact({ watermark: "critical", occupancy: 0.95, active: false }, armed)).toBe(false);
  });

  it("never fires while disarmed (one compaction per climb)", () => {
    const disarmed: CompactPolicyState = { lastFiredOccupancy: 0.8, armed: false };
    expect(shouldCompact({ watermark: "warn", occupancy: 0.8, active: true }, disarmed)).toBe(false);
  });

  it("fires on warn and critical during active work", () => {
    expect(shouldCompact({ watermark: "warn", occupancy: 0.78, active: true }, armed)).toBe(true);
    expect(shouldCompact({ watermark: "critical", occupancy: 0.92, active: true }, armed)).toBe(true);
  });

  it("glides on soft only once the window is climbing past the soft band", () => {
    expect(shouldCompact({ watermark: "soft", occupancy: 0.62, active: true }, armed)).toBe(false);
    expect(shouldCompact({ watermark: "soft", occupancy: 0.7, active: true }, armed)).toBe(true);
  });

  it("does not fire in the normal band", () => {
    expect(shouldCompact({ watermark: "normal", occupancy: 0.4, active: true }, armed)).toBe(false);
    expect(shouldCompact({ occupancy: 0.4, active: true }, armed)).toBe(false);
  });
});

describe("nextPolicyState", () => {
  it("disarms and records the fire point after firing", () => {
    const st = nextPolicyState(true, 0.8, armed);
    expect(st.armed).toBe(false);
    expect(st.lastFiredOccupancy).toBe(0.8);
  });

  it("re-arms only after occupancy falls a clear margin below the fire point", () => {
    const fired = nextPolicyState(true, 0.8, armed); // disarmed at 0.8
    // still high -> stays disarmed
    expect(nextPolicyState(false, 0.72, fired).armed).toBe(false);
    // dropped past the 0.15 margin (compaction freed room) -> re-armed
    expect(nextPolicyState(false, 0.6, fired).armed).toBe(true);
  });

  it("models a full climb, fire, hold, fall, re-arm cycle", () => {
    let st = { ...INITIAL_POLICY };
    // climb into warn -> fire
    let fire = shouldCompact({ watermark: "warn", occupancy: 0.78, active: true }, st);
    expect(fire).toBe(true);
    st = nextPolicyState(fire, 0.78, st);
    // next tick still elevated -> no double fire
    fire = shouldCompact({ watermark: "warn", occupancy: 0.79, active: true }, st);
    expect(fire).toBe(false);
    st = nextPolicyState(fire, 0.79, st);
    // compaction lands, window falls -> re-arm
    st = nextPolicyState(false, 0.55, st);
    expect(st.armed).toBe(true);
    // it can fire again on the next climb
    expect(shouldCompact({ watermark: "warn", occupancy: 0.77, active: true }, st)).toBe(true);
  });
});
