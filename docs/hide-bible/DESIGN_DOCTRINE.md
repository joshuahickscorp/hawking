# HIDE Design Doctrine v3
### Hawking IDE: the vision, the architecture, and the material system

> Canonical. Supersedes v1 and v2 (the v2 gold rim-light doctrine in `HIDE_PLAN.md` Part C and `archive/04-design-doctrine.md` is retired). Read Part I before writing a single line of code. The design tokens are implemented in `app/src/theme.css`.

## Part I. The Vision

Build a Tadao Ando building, and put a coding agent inside it. That is the entire brief in one sentence. If a decision is ever unclear, return here and ask: what would this be, if it were a room by Ando?

**The room.** Stand inside the Church of the Light, or the buried galleries at Naoshima. You are in a chamber of board-formed concrete: grayscale, smooth, honest, a hair warm like ash, marked only by the faint regular grid the wooden forms left behind. The room is dark, still, large, and mostly empty. The space between things is not wasted, it is the subject. Then, somewhere, light enters: a narrow slot cut in the wall, daylight landing on the concrete, and the room is dark and full of light at once. Nothing is decorated. The only events are the concrete, the void, and the light moving across it.

That is HIDE. The chamber is the application. The concrete is every surface. The void is the air we leave around everything. The light entering the dark is the agent: when it works, light comes into the room; when it rests, the room is still, like the surface of a reflecting pool. This is why the name works: a black hole is the darkest object there is, and Hawking proved it still radiates. Ando is how we build the box. The cross of light is how it radiates.

**What it feels like.** Serene, powerful, exact. You move through it the way you move through a good building: compression and release, calm thresholds, no clutter, your attention pulled only by light. Never a cockpit (alarms, gauges, anxiety). An observatory, a chapel, a quiet workshop with one perfect instrument on a concrete plinth. The single adjective the whole product optimizes for is **legible**, now inseparable from **airy**: you can read everything at a glance, and everything has the room to be read. Under those sit **restraint** (color is functional, never decorative) and **light** (the dark material is made alive by light entering it, never by a colored accent).

**The references, in order.**
- **Tadao Ando** is the soul: board-formed concrete, grayscale, the void as a positive element (ma), light as a building material, water and stillness, geometric purity (line, square, circle), honest raw material, no ornament, monastic calm. Everything spatial and material descends from Ando.
- **Aesop** is the same DNA at hand-scale: concrete, terrazzo, muted pigment, tactile restraint, a contemplative room you do not want to leave.
- **Teenage Engineering** is the one object in the room: the precise, labeled, brushed-metal instrument (the OP-1 on a plinth). HIDE's controls are TE instruments inside an Ando space.
- **Vercel / Geist** is the execution grammar that keeps it software: dark as the canonical surface, monochrome neutrals, shadow-as-border, the inner-light ring that makes a dark plane glow from within, the Geist Mono type discipline, terse copy. Use the grammar; do not clone it.
- **Zed** is the discipline: the editor disappears, content over chrome, a hard refusal of translucency.
- **opencode, Cline, Aider, Linear** are the working organs (action-log, steering and approval, the git safety net, the craft floor). Mapped in Part VI.
- The **CX Launch page** is the proof it works in your hands: a single metal volume floating in 56px of air (44 to 72px so nothing touches an edge), a felt-not-noticed lift behind the hero, line-glyph rings that brighten when alive and dim when idle. Reuse its spacing and material vocabulary directly.

One non-software reference, pinned: the Event Horizon Telescope image of M87*. Black core, warm light ring. That image is the brand.

**Things this is not.** Not a cockpit. Not dense or packed. Not VS Code, not Cursor, not a chat box bolted onto a black box. And specifically not the vibecoded look: a single bright accent on near-black (especially yellow, gold, or acid), packed edge-to-edge panels, glassmorphism or translucency, soft-focus gradient heroes, a serif-plus-sans display pairing, neon syntax, rounded everything with no material. Any one of those is a failure of the vision.

## Part II. The Architecture

Ando designs in volumes, voids, thresholds, and light. HIDE is laid out the same way.

**Spatial: volumes in a void.** Every surface is a poured volume resting in a void. The void (`--void`, the deepest grayscale) is the background of the whole structure, the dark air of the chamber. Volumes (panels, cards, the editor, the stack) are slabs of slightly lighter concrete that sit in that air with generous space around them. Objects in a room, not regions tiling a grid. This is the single most important spatial rule, and it is what v2 violated by packing everything edge to edge.

Three principles govern the space:
1. **Ma, the void as subject.** Gaps are large (40 to 56px between regions, up to 96px for the largest courtyard voids). Nothing touches an edge (44 to 72px of air at the top and bottom of a hero surface). When in doubt, add space.
2. **Procession and threshold.** Moving between the three surfaces is a procession through a building, not a tab switch. Transitions carry weight (`--dur-door`). The command palette is the threshold you pass through to go anywhere, the one fast path, keyboard-summoned.
3. **Light enters from one place.** The light source is the agent; the place light enters is the Context Stack, the light well along the east wall, always present.

**Information: three chambers and one light well.** One unified structure, three chambers (modes), one light well (the Context Stack) present in all of them. You never leave the building.
- **The Workstation** is the courtyard and the front door. You arrive here: the overnight digest, the fleet of running agents, the merge review. The most novel, most only-local surface, so it is what you see first. Open, calm, the largest voids.
- **The IDE** is the workshop chamber: editor, diff review, file tree, terminal. The most information-rich chamber, but it still breathes; density is achieved by ranking and light, never by packing.
- **The Chat** is the conversation chamber: you and the agent. The calmest and most spacious, content held to a readable column down the center.
- **The Context Stack** is the light well: a narrow vertical shaft along the east wall, present in every chamber, where the agent's work enters as light. Always there, calm and spacious by default, openable into a full inspector.

Center of gravity: observation-first. Three lenses on one activity, watching the agent work. Movement: a quiet mode rail on the west wall plus the command palette. No tab-hell.

```tsx
// The shell is one concrete structure: a wall, three chambers, one light well.
<Shell>                                   {/* full viewport, background: var(--void) */}
  <ModeRail />                             {/* west wall: three named modes, quiet */}
  <Stage>                                  {/* the active chamber */}
    {mode === "workstation" && <Workstation />}  {/* the courtyard, the front door */}
    {mode === "ide"         && <IDE />}          {/* the workshop chamber */}
    {mode === "chat"        && <Chat />}         {/* the conversation chamber */}
  </Stage>
  <ContextStack />                         {/* the light well, always on the east wall */}
</Shell>
```

```css
.shell {
  display: grid;
  /* west wall | chamber | light well */
  grid-template-columns: 56px 1fr clamp(320px, 26vw, 380px);
  height: 100vh;
  background: var(--void);
  color: var(--text-1);
  font-family: var(--font);
}
```

**Re-housing the harvested modules.** We assemble HIDE from the cleanest open-source design languages (opencode, Cline, Aider, others). The rule that prevents a patchwork: every borrowed stone is recast in the same concrete. No component is dropped in as-is. Each is re-poured with HIDE's tokens, padding, type, inner-glow, and motion before it enters the building. If you can see a seam, the module was not recast.

## Part III. The Material System

These tokens supersede all earlier sets. (Full copy-paste block in the Appendix; recipes in Part IV.)

**Concrete (surface).** Grayscale, board-formed, never pure black, a hair warm like ash. `--void #070707` (the unlit chamber, the app background), `--concrete-1..4` (planes set at different depths), `--formwork` (the faint grid, used as PROPORTION first; as literal texture only a whisper, <= 3%, never a photo). `--line` / `--line-strong` and `--hairline` (shadow-as-border, applied as box-shadow not border, replacing ALL CSS borders).

**Light (the only accent there is).** There is no brand hue. A dark plane is made alive by light entering it: a top-edge highlight (`--inner-glow`), a soft halo (`--light-bloom`), and a soft natural gradient (`--grade-wall`, light falling across a wall). When an element is alive, it brightens toward `--light #F4F2EE` and glows. This is the cross of light, the entire emotional range.

**Text (chalk on concrete).** Warm ash-grays, every level verified WCAG AA against `--void`: `--text-1 #ECEAE6`, `--text-2 #9B9A95`, `--text-3 #6E6D68`, `--mute #5C5B57`.

**Pigment (semantic punctuation only).** The only two colors: `--ok #7E9E86` (lichen) and `--bad #C0807A` (oxide). Desaturated mineral pigment, used like punctuation, always paired with a glyph so color is never the sole signal. They appear in diffs and state, nowhere as identity.

**Type (one instrument, Geist Mono).** One typeface for the whole product: Geist Mono, UI, headings, code, all of it. Geist Sans stays logo-only. Hierarchy comes from weight, size, letter-spacing, and color, never a second family (400 read, 500 interact, 600 announce). `.t-display` is Geist Mono 600 (the confident large-type moment, replacing the serif). Enable ligatures globally. If prose in Chat ever feels tiring in mono, the only sanctioned change is adding Geist Sans for body copy alone (which relaxes the logo-only rule); default is pure mono.

**Void (the ma scale).** Space is the subject. `--ma-1..24` (4px to 96px). Hard airiness rules: panel padding never below 16px (default 24px); section gaps 40 to 56px, courtyard voids to 96px; nothing touches an edge (44 to 72px hero air); conversational surfaces cap content near 680 to 720px and center it; volumes float in the void, never packed edge to edge; if a panel needs more than it can show airily, it summarizes and expands on demand, never gets denser.

**Form and motion.** Soft-cornered volumes (`--radius 8px`); one fully-round tactile control (`--radius-pill`). Motion has the weight of concrete (`--ease` cubic-bezier(0.2,0,0,1)); nothing snaps or bounces. `--breathe 2400ms` is the slow pulse of light, the agent breathing. `prefers-reduced-motion` resolves all breathing and transitions to static states.

## Part IV. The Surfaces (recipes)

**The volume** (the base component every surface is made of):
```css
.volume { background: var(--concrete-2); border-radius: var(--radius); box-shadow: var(--hairline), var(--inner-glow); padding: var(--ma-6); }
.volume--raised { background: var(--concrete-3); }
.volume:hover { background: var(--concrete-4); transition: background var(--dur) var(--ease); }
```

**The alive element** (the cross of light), any element the agent is currently animating:
```css
.alive { box-shadow: var(--inner-glow); animation: breathe var(--breathe) var(--ease) infinite; }
@keyframes breathe { 0%,100% { box-shadow: var(--inner-glow); } 50% { box-shadow: var(--inner-glow), var(--light-bloom); } }
@media (prefers-reduced-motion: reduce) { .alive { animation: none; box-shadow: var(--inner-glow), var(--light-bloom); } }
```

**The Workstation** (courtyard, front door): the digest headline in `.t-display`, alone, with `--ma-18` air above and `--ma-14` below ("7 agents ran. 312 files changed. 4 need you."). Agent cards are `.volume` slabs in a responsive grid (`--ma-8` gaps minimum), each with task title (`.t-title`), one line of live feed (`.t-code`, `--text-2`), and state read by light (breathing / steady / lit), never a colored badge. The budget meter (memory and context, not money) is a calm monochrome bar, framed as abundance, never an alarm. Merge review reuses the diff-hunk component, identical to the IDE.

**The IDE** (workshop): three volumes (file tree, editor, terminal) with `--ma-4` to `--ma-6` between them, the editor largest. The file tree is a quiet list (14px, `--text-2`, line-height 1.8+, the active file at `--text-1` with a faint `--inner-glow` row), not a dense tree. Diff review is inline by default, per-hunk accept/reject (Cline's model), keyboard-driven (j/k move, a/r accept/reject), side-by-side on toggle. Every change rests on a git-checkpoint (Aider), instantly reversible.
```css
.hunk-add { color: var(--ok);  background: var(--ok-bg);  } /* line prefixed with + */
.hunk-del { color: var(--bad); background: var(--bad-bg); } /* line prefixed with - */
```

**The Chat** (conversation): message column capped near 700px, centered, `--ma-18` top, `--ma-14` between turns. Agent prose is `.t-body` line-height 1.6; the streaming leading edge carries a faint `--light-soft` cusp, no spinner. The steering field is a `.volume--raised` input pinned at the bottom with `--ma-8` air around it, plus @-references to point the agent at exact files.

**The Context Stack** (light well): the strata top to bottom, each a `.volume` with `--ma-6` interior and `--ma-4` between them: retrieved files and symbols, tools called, memory in play, tests and state, current action (the live feed). The current-action stratum is `.alive` and scrolls the agent's real moves as `.t-code`, `--text-2` ("Reading guard.rs", "Running 12 tests"). Touch affordances are quiet TE line-glyph controls on each stratum (pin, evict, mute, @-add), brighter on hover, dim at rest. Not a graph, not a swarm.

**The approval capsule** (the lit instrument): the one tactile control, a capsule that lights fully and holds steady (not breathing) when the agent needs you.
```css
.gate { border-radius: var(--radius-pill); background: var(--concrete-3); color: var(--light); box-shadow: var(--hairline-strong), var(--light-bloom), var(--inner-glow); padding: var(--ma-3) var(--ma-6); font-weight: 500; letter-spacing: 0.02em; }
/* states plainly what will happen, e.g. "Run migration. Approve." */
```

## Part V. Aliveness, Interaction, and Voice

**Aliveness.** Light is the heartbeat, not color. When the agent is alive, the relevant element glows from within and breathes. Token streaming has a faint light cusp at the leading edge. Progress is the agent's real work, never theater: no spinner, no percentage bar. You convey "working" with the breathing glow on the active element and the live-feed stratum scrolling the agent's actual moves.

**Interaction.** Keyboard-first, command-palette-centric, mouse-rich where physical (diffs, the Context Stack touch, the agent board). Vim is an option in the editor, never forced. Plan and Build are explicit modes with a visible indicator. Three verbs: **see** (the Context Stack), **steer** (a persistent steering input plus @-references; the agent is interruptible, never fire-and-forget), **gate** (per-step approval for consequential actions via the lit capsule, with a plain statement of what will happen; toggle-able toward more autonomy, default human-in-the-loop). Diff review: inline by default, per-hunk, keyboard-driven, side-by-side on toggle, every change on a git-checkpoint; identical in the IDE and Workstation merge review.

**Voice.** Terse telemetry, the voice of a flight log: specific, plain, undramatic, never cute, no emoji. Name the specific thing that changed, drop the trailing period, never "successfully" ("Diff accepted"). Empty states point to the first action. In-progress uses the present participle and an ellipsis ("Reading guard.rs..."). Numerals; skip "please" and superlatives. Errors are direct, specific, blame-free, never apologize ("Couldn't reach the local engine. It may not be running."). Labels name what the user controls, and keep the same word through a flow. Standing rule: no em dashes, no en dashes, no middot separators, anywhere.

## Part VI. Constraints, Amalgamation, and Self-Check

**Standing rules (non-negotiable).** No em/en dashes or middot separators in any copy. Geist (Sans) is logo only; Geist Mono is the entire type system. Dark, material, airy: brutalist concrete that breathes, never flat vibecoded gradients, never packed density. Dual Fahrenheit and Celsius if temperature appears.

**Absolute nevers (design).** No yellow, gold, acid, or neon. No brand hue. The only colors are desaturated lichen (`--ok`) and oxide (`--bad`), semantic and glyph-paired. No blue, no purple. No true `#000`. No translucency or glassmorphism. No gradient or soft-focus color hero. No serif-plus-sans display pairing. No packed, edge-to-edge, dense panels. No spinner standing in for real work. No gratuitous bounce or snap. No literal concrete-photo texture or skeuomorphic ornament. Not a VS Code or Cursor clone; recast every harvested module in the same concrete.

**Accessibility (hard, enforced).** All body and label text meets WCAG AA (4.5:1) against `--void`; verify `--text-3` and `--mute` specifically. Color is never the only signal (glyph or +/- marker). A visible focus ring on every interactive element at `:focus-visible`. `prefers-reduced-motion` honored. Geist Mono set at comfortable size and line-height (1.6 body).

**The amalgamation map.** Ando gives the soul and the space; Aesop the hand-scale materiality; Teenage Engineering the instrument; Vercel/Geist the execution grammar; Zed the discipline; opencode the transparency (the action-log becomes the Context Stack's live feed; keyboard plan/build, parallel sessions, privacy-first); Cline the human-in-the-loop (plan/act, per-step approve and per-hunk reject, @-references, checkpoints, the multi-agent board that becomes the Workstation; HIDE inverts Cline's spend anxiety into local abundance); Aider the safety net (git checkpoints); Linear the craft floor (command palette, speed, polish); the CX Launch page the proven airy material. HIDE is the synthesis, unified by one concept (the box that radiates light) and one device (light entering dark concrete).

**Self-check (run before anything ships; if any is true, it failed).** A bright accent color anywhere (especially yellow/gold/acid). Blue or purple. Any panel packed edge to edge, anything touching an edge, sections tighter than 40px. Translucency, frosted glass, or a soft-focus gradient. Two type families (serif and sans) instead of Geist Mono alone. A spinner or percentage bar standing in for real work. The Context Stack a packed data-wall or busy graph instead of a calm light well. A harvested module dropped in as-is with a visible seam. Motion that bounces or snaps. A literal concrete-texture photo or skeuomorphic ornament. Geist Sans outside the logo. An em/en dash or middot in copy, or an error that apologized, or a toast that said "successfully". Body or label text failing AA contrast on `--void`. If none are true: the chamber is dark, grayscale, and still, the volumes float in generous void, and light enters where the agent works. That is HIDE.

## Appendix. Consolidated tokens (copy-paste; mirror of `app/src/theme.css`)

```css
:root {
  /* type */
  --font: "Geist Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, "Courier New", monospace;
  /* concrete (surface) */
  --void: #070707; --concrete-1: #0E0E0F; --concrete-2: #141416; --concrete-3: #1B1B1E; --concrete-4: #222226;
  --formwork: rgba(255,255,255,0.02);
  /* shadow-lines (replace all CSS borders) */
  --line: rgba(255,255,255,0.06); --line-strong: rgba(255,255,255,0.11);
  --hairline: 0 0 0 1px rgba(255,255,255,0.06); --hairline-strong: 0 0 0 1px rgba(255,255,255,0.11);
  --depth: 0 0 0 1px rgba(255,255,255,0.06), 0 12px 32px -16px rgba(0,0,0,0.7);
  /* light (the only accent) */
  --light: #F4F2EE; --light-soft: rgba(244,242,238,0.06); --light-bloom: 0 0 28px rgba(244,242,238,0.10);
  --inner-glow: inset 0 1px 0 rgba(255,255,255,0.06);
  --grade-wall: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0) 42%);
  /* text (chalk on concrete) */
  --text-1: #ECEAE6; --text-2: #9B9A95; --text-3: #6E6D68; --mute: #5C5B57;
  /* pigment (semantic punctuation only, glyph-paired) */
  --ok: #7E9E86; --ok-bg: rgba(126,158,134,0.08); --bad: #C0807A; --bad-bg: rgba(192,128,122,0.08);
  /* void (ma scale) */
  --ma-1: 4px; --ma-2: 8px; --ma-3: 12px; --ma-4: 16px; --ma-6: 24px; --ma-8: 32px; --ma-10: 40px; --ma-14: 56px; --ma-18: 72px; --ma-24: 96px;
  /* form and motion */
  --radius: 8px; --radius-pill: 9999px; --ease: cubic-bezier(0.2, 0, 0, 1);
  --dur-fast: 120ms; --dur: 220ms; --dur-slow: 360ms; --dur-door: 480ms; --breathe: 2400ms;
}
```

One sentence to hold the whole doctrine: build a Tadao Ando building in grayscale concrete, leave generous void around everything, let no color in, and make it come alive only where light enters the dark, which is where the agent works.
