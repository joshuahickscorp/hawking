/*
  Radiate.tsx — HIDE's one progress signature, the event-horizon ring: a dark disc with a thin arc of
  LIGHT on the rim (the agent radiating, the cross of light entering the dark). This is the only loading
  indicator in the app. Two modes:
    - indeterminate: the arc sweeps (a run is in flight, no laddered progress yet).
    - laddered: pass `stage` (0..stages) and the arc length encodes the oracle ladder (build, typecheck,
      test, lint). It sharpens as work lands and does not spin.
  Honors prefers-reduced-motion (static lit arc).
*/
export function Radiate({
  size = 16,
  active = true,
  stage,
  stages = 4,
  title,
}: {
  size?: number;
  active?: boolean;
  stage?: number;
  stages?: number;
  title?: string;
}) {
  const sw = Math.max(1.25, size / 12);
  const r = size / 2 - sw;
  const c = 2 * Math.PI * r;
  const laddered = typeof stage === "number";
  const clamped = laddered ? Math.min(Math.max(stage as number, 0), stages) : 0;
  // indeterminate arc is a fixed wedge that sweeps; laddered arc grows from a stub toward full.
  const frac = laddered ? 0.12 + 0.8 * (clamped / stages) : 0.26;
  const label = title ?? (laddered ? `verifying ${clamped} of ${stages}` : active ? "working" : "idle");
  return (
    <span
      className={["radiate", active && !laddered && "radiate--active", laddered && "radiate--laddered"].filter(Boolean).join(" ")}
      style={{ width: size, height: size }}
      role="img"
      aria-label={label}
      title={title}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} fill="none">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--line-strong)" strokeWidth={sw} />
        <circle
          className="radiate__arc"
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="var(--light)"
          strokeWidth={sw}
          strokeLinecap="round"
          strokeDasharray={`${c * frac} ${c}`}
        />
      </svg>
    </span>
  );
}
