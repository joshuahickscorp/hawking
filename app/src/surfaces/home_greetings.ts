/*
  greetings.ts — the courtyard opening line. Instead of one static prompt, rotate through a set so the
  room greets you differently each time you arrive. Terse flight-log voice; some lines carry {name}, some
  do not (name fills reactively once the host reports it). No em or en dashes, no middot, no emoji.
*/
export const GREETINGS: readonly string[] = [
  "What is up next, {name}?",
  "Where to, {name}?",
  "What are we building?",
  "Pick up where you left off.",
  "The room is quiet. What first?",
  "Point me at something.",
  "What needs doing, {name}?",
  "Name the goal.",
  "One task. Let us start.",
  "What is on your mind, {name}?",
  "Ready when you are.",
  "What should we ship?",
  "Give me a thread to pull.",
  "What are we hunting today?",
  "Where does the light go next?",
  "What is worth doing, {name}?",
  "What is broken, {name}?",
  "Describe the work.",
];

// Fill the template at a stable index with the current name (kept reactive so a late-arriving name
// still lands). Index wraps and tolerates negatives.
export function fillGreeting(ix: number, name: string): string {
  const n = GREETINGS.length;
  const t = GREETINGS[((ix % n) + n) % n];
  return t.replace(/\{name\}/g, name);
}

// Read the persisted rotation counter, advance it, and return the index to use this open. Wrapped in
// try/catch so a disabled localStorage never breaks the greeting (falls back to the first line).
export function nextGreetingIndex(): number {
  try {
    const raw = localStorage.getItem("hide.greetIx");
    const ix = raw == null ? 0 : Number.parseInt(raw, 10) || 0;
    localStorage.setItem("hide.greetIx", String(ix + 1));
    return ix;
  } catch {
    return 0;
  }
}
