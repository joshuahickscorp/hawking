/*
  Radiate.tsx — HIDE's own progress signature: the event-horizon ring. A dark disc with a thin arc of
  LIGHT that sweeps the rim (the agent radiating, the cross of light entering the dark). Used wherever
  the agent works: the chat stream cusp, an active run, the toolbar phase, the Context Stack live row.
  Not a stock spinner. Honors prefers-reduced-motion (static lit arc).
*/
export function Radiate({
  size = 16,
  active = true,
  title,
}: {
  size?: number;
  active?: boolean;
  title?: string;
}) {
  const sw = Math.max(1.25, size / 12);
  const r = size / 2 - sw;
  const c = 2 * Math.PI * r;
  return (
    <span
      className={["radiate", active && "radiate--active"].filter(Boolean).join(" ")}
      style={{ width: size, height: size }}
      role="img"
      aria-label={title ?? (active ? "working" : "idle")}
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
          strokeDasharray={`${c * 0.26} ${c}`}
        />
      </svg>
    </span>
  );
}
