# HIDE

### Hawking IDE: vision, architecture, material system, and the build order, in one document

This is the single canonical document. It supersedes every earlier draft. Parts I to VI are the design authority; Part VII is the actionable work order; the Appendix is the copy-paste token set. Read Part I before writing a single line of code.

The one sentence that governs everything: build a Tadao Ando building in grayscale concrete, leave generous void around everything, let no color in, make it come alive only where light enters the dark (which is where the agent works), and wire that to the real engine.

The dual face: HIDE hides you from the cloud (offline, local, nothing leaves your machine) and hides nothing from you (the Context Stack). Privacy out, transparency in.

The mission right now: the current build runs and is a competent VS Code plus Cursor clone. Keep the skeleton, destroy the skin, replace the wiring. By the end it should look like Ando, Aesop, and Teenage Engineering shipped it, not Microsoft and Anysphere.

## Part I. The Vision

Build a Tadao Ando building, and put a coding agent inside it. That is the entire brief in one sentence, and everything below is what it means. If a decision is ever unclear, return here and ask: what would this be, if it were a room by Ando?

### The room

Stand inside the Church of the Light, or in the long buried galleries at Naoshima. You are in a chamber of board-formed concrete. The concrete is grayscale, smooth, and honest, a hair warm like ash, marked only by the faint regular grid the wooden forms left behind. The room is dark, still, and large, and it is mostly empty. The space between things is not wasted, it is the subject. Then, somewhere, light enters: a narrow slot cut in the wall, daylight pouring through and landing on the concrete, and the room is no longer dark. It is dark and full of light at once. Nothing is decorated. Nothing is loud. The only events are the concrete, the void, and the light moving across it.

That is HIDE. The chamber is the application. The concrete is every surface. The void is the air we leave around everything. The light entering the dark is the agent: when it works, light comes into the room. When it rests, the room is still, like the surface of a reflecting pool.

This is also why the name works. A black hole is the darkest object there is, and Hawking proved it still radiates. Ando is how we build the box. The cross of light is how it radiates. The architecture and the brand concept are the same idea: a dark, silent, grayscale volume from which light escapes to the one person watching.

### What it feels like to use

Serene, and powerful, and exact. You move through it the way you move through a good building: a sequence of compression and release, calm thresholds, no clutter, your attention pulled only by light. It never feels like a cockpit, with its alarms and gauges and anxiety. It feels like an observatory, or a chapel, or a quiet workshop with one perfect instrument on a concrete plinth. The agent can be doing enormous work, dozens of runs all night, and the room stays calm, because the work arrives as light and as plain words, never as noise.

The single adjective the whole product optimizes for is legible, and it is now inseparable from airy: you can read everything at a glance, and everything has the room to be read. Under those sit two qualities that govern every detail: restraint (color is functional, never decorative; strip every surface to structure and light) and light (the dark material is made alive by light entering it, never by a colored accent).

### The references, in order

- Tadao Ando is the soul. Board-formed concrete, grayscale, the void as a positive element (ma), light as a building material, water and stillness, geometric purity (the line, the square, the circle), honest raw material, no ornament, monastic calm. Everything spatial and material in HIDE descends from Ando.
- Aesop is the same DNA at hand-scale: concrete, terrazzo, muted pigment, tactile restraint, a contemplative room you do not want to leave. HIDE should feel as considered as an Aesop interior.
- Teenage Engineering is the one object in the room: the precise, labeled, brushed-metal instrument (the OP-1 on a plinth). HIDE's controls are TE instruments sitting inside an Ando space. Tactile, exact, quietly mechanical.
- Vercel / Geist is the execution grammar that keeps it software: dark as the canonical surface, monochrome neutrals, shadow-as-border, the inner-light ring that makes a dark plane glow from within, the Geist Mono type discipline, and terse copy. Use the grammar; do not clone it, or it reads as "made with Geist" instead of HIDE. Our soul is the building, not the palette.
- Zed is the discipline: the editor should disappear, content over chrome, and a hard refusal of translucency as plastic and cheap.
- opencode, Cline, Aider, Linear are the working organs (the action-log, the steering and approval model, the git safety net, the craft floor). They are mapped one by one in Part VI.
- Your own CX Launch page is the proof that this already works in your hands: a single metal volume floating in 56px of air, 44 to 72px so nothing touches an edge, a felt-not-noticed lift behind the hero, line-glyph rings that brighten when alive and dim when idle. Reuse its spacing and its material vocabulary directly.

### Things this is not

Not a cockpit. Not dense or packed. Not VS Code, not Cursor, not a chat box bolted onto a black box. And specifically not the vibecoded look, which we now define so the build can detect and reject it: a single bright accent on near-black (especially yellow, gold, or acid), packed edge-to-edge panels, glassmorphism or translucency, soft-focus gradient heroes, a serif-plus-sans display pairing, neon syntax, and rounded everything with no material. Any one of those is a failure of the vision.

## Part II. The Architecture

Ando designs in volumes, voids, thresholds, and light. HIDE is laid out the same way. This covers both the spatial architecture (how the interface sits in space) and the information architecture (how the product is organized).

### Spatial architecture: volumes in a void

Every surface in HIDE is a poured volume resting in a void. The void (`--void`, the deepest grayscale) is the background of the whole structure, the dark air of the chamber. Volumes (panels, cards, the editor, the stack) are slabs of slightly lighter concrete that sit in that air with generous space around them. They are objects in a room, not regions tiling a grid. This is the single most important spatial rule, and it is the one the current clone violates by packing everything edge to edge.

Three principles govern the space:

1. Ma, the void as subject. The space between volumes is designed, not leftover. Gaps are large (40 to 56px between regions, up to 96px for the largest courtyard voids). Nothing touches an edge (44 to 72px of air at the top and bottom of a hero surface). When in doubt, add space.
2. Procession and threshold. Moving between the three surfaces is a procession through a building, not a tab switch. Transitions carry weight, like a heavy concrete door swinging slowly (`--dur-door`). The command palette is the threshold you pass through to go anywhere; it is the one fast path, and it is keyboard-summoned.
3. Light enters from one place. In an Ando chamber, light has a source. In HIDE, the light source is the agent, and the place light enters is the Context Stack, the light well along the east wall, always present. The agent's life appears there as light, and it spills subtly into the rest of the room.

### Information architecture: three chambers and one light well

One unified structure, three chambers (modes), and one light well (the Context Stack) present in all of them. You never leave the building; you walk from one chamber to another while the light well stays at your side.

- The Workstation is the courtyard, and it is the front door. You arrive here. It holds the overnight digest, the fleet of running agents, and the merge review. It is the most novel, most only-local surface, so it is what you see first. Open, calm, the largest voids in the whole product.
- The IDE is the workshop chamber: the editor, the diff review, the file tree, the terminal. Where focused, single-agent work happens. The most information-rich chamber, but it still breathes; density is achieved by ranking and light, never by packing.
- The Chat is the conversation chamber: you and the agent, talking. The calmest and most spacious of the three, content held to a readable column down the center.
- The Context Stack is the light well: a narrow vertical shaft along the east wall, present in every chamber, where the agent's work enters as light. Always there, calm and spacious by default, openable into a full inspector.

Center of gravity: observation-first. The three chambers are three lenses on one activity, watching the agent work. Not editor-first (that is Cursor), not purely chat-first. You are always observing; the chambers change what you observe.

Movement: a quiet mode rail on the west wall (three named modes, the spatial anchor) plus the command palette (the keyboard threshold, the reference pattern from opencode, Linear, and Geist). No tab-hell.

### The shell, in code

The whole structure is one grid: a west wall, the active chamber, and the east light well.

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

### Re-housing the harvested modules

We are assembling HIDE from the cleanest open-source design languages (opencode, Cline, Aider, and others). The architectural rule that prevents a patchwork: every borrowed stone is recast in the same concrete. No component is dropped in as-is. Each is re-poured with HIDE's tokens, padding, type, inner-glow, and motion before it enters the building. If you can see a seam, the module was not recast. A harvested file tree, a harvested diff view, a harvested chat panel: all arrive raw, all leave as the same grayscale concrete.

## Part III. The Material System

The concrete, the light, the type, and the void, as tokens. These are the canonical values; the full set is copy-paste in the Appendix.

### Concrete (surface)

Grayscale, board-formed, never pure black, a hair warm like ash. The ramp is a set of planes set at different depths in the chamber, each catching slightly more or less light.

```css
:root {
  --void:        #070707; /* the unlit chamber, the deepest field, the app background */
  --concrete-1:  #0E0E0F; /* a wall set back in shadow */
  --concrete-2:  #141416; /* a poured volume resting in the room (the default panel) */
  --concrete-3:  #1B1B1E; /* a raised module, catching a little more light */
  --concrete-4:  #222226; /* the surface a hand is on: hover and active */

  /* FORMWORK: the faint regular grid board-formed concrete leaves behind.
     Use as PROPORTION first (align modules to a regular rhythm). As literal texture,
     only ever a whisper, <= 3% opacity, never a photograph, never skeuomorphic. */
  --formwork:    rgba(255,255,255,0.02);

  /* SHADOW-LINE: where two concrete planes meet. Applied as box-shadow, not border,
     so it never doubles a layout edge. This replaces all CSS borders. */
  --line:        rgba(255,255,255,0.06);
  --line-strong: rgba(255,255,255,0.11);
  --hairline:    0 0 0 1px var(--line);
  --hairline-strong: 0 0 0 1px var(--line-strong);
  --depth:       0 0 0 1px var(--line), 0 12px 32px -16px rgba(0,0,0,0.7); /* overlays only */
}
```

### Light (the only accent there is)

There is no brand hue. Light is the accent. A dark plane is made alive by light entering it: a top-edge highlight (the inner-glow), a soft halo (the bloom), and a soft natural gradient (light falling across a wall, top brighter, never harsh). When an element is alive, it brightens toward near-white and glows. This is the cross of light, and it is the entire emotional range of the product.

```css
:root {
  --light:        #F4F2EE; /* the daylight that enters: warm near-white */
  --light-soft:   rgba(244,242,238,0.06); /* light grazing a surface */
  --light-bloom:  0 0 28px rgba(244,242,238,0.10); /* a soft halo, natural, no hue */
  --inner-glow:   inset 0 1px 0 rgba(255,255,255,0.06); /* the top edge catching light */

  /* natural-light gradient: light falling down a wall, never a designed gradient */
  --grade-wall:   linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0) 42%);
}
```

### Text (chalk on concrete)

Warm ash-grays, every level verified at WCAG AA against `--void`.

```css
:root {
  --text-1:  #ECEAE6; /* primary, like chalk on concrete */
  --text-2:  #9B9A95; /* secondary */
  --text-3:  #6E6D68; /* metadata, AA-verified */
  --mute:    #5C5B57; /* the quietest label, for 12px uppercase at relaxed line-height */
}
```

### Pigment (semantic punctuation only)

The only two colors in the entire product. Desaturated, like natural mineral pigment, used like punctuation, always paired with a glyph so color is never the sole signal. They appear in diffs and in state, and nowhere as identity.

```css
:root {
  --ok:   #7E9E86; --ok-bg:  rgba(126,158,134,0.08); /* lichen: success, tests passed, diff added */
  --bad:  #C0807A; --bad-bg: rgba(192,128,122,0.08); /* oxide: error, diff removed */
}
```

### Type (one instrument, Geist Mono)

One typeface for the whole product: Geist Mono, UI, headings, code, all of it. This respects the standing rule (Geist Sans stays logo-only; Geist Mono is the UI and code font), it is a true unification, and Vercel proves mono headings read as engineered and clean rather than techy. Hierarchy comes from weight, size, letter-spacing, and color, never a second family. Geist Mono is the labeled engraving on a Teenage Engineering instrument.

```css
:root {
  --font: "Geist Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, "Courier New", monospace;
}
* { font-feature-settings: "liga" 1, "calt" 1; } /* ligatures on, for the engineered feel */
```

Roles (weight carries meaning: 400 read, 500 interact, 600 announce):

```css
/* DISPLAY: one per surface, in air. Replaces the serif as the confident large-type moment. */
.t-display { font-weight: 600; font-size: clamp(32px, 4vw, 52px); letter-spacing: -0.035em; line-height: 1.05; }
.t-title   { font-weight: 500; font-size: 20px; letter-spacing: -0.02em; line-height: 1.2; }
.t-body    { font-weight: 400; font-size: 14px; line-height: 1.6; }
.t-code    { font-weight: 400; font-size: 13.5px; line-height: 1.55; }
.t-label   { font-weight: 500; font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--mute); line-height: 1.4; }
.t-micro   { font-weight: 400; font-size: 11px; letter-spacing: 0.04em; color: var(--mute); }
```

If prose in Chat ever feels tiring in mono, the only sanctioned change is adding Geist Sans for body copy alone, which would require relaxing the logo-only rule. Default is pure mono; the terse voice keeps prose short enough that mono stays comfortable.

### Void (the ma scale)

Space is the subject. Generous by default. These are the intervals between and inside volumes.

```css
:root {
  --ma-1: 4px;  --ma-2: 8px;   --ma-3: 12px;  --ma-4: 16px;
  --ma-6: 24px; --ma-8: 32px;  --ma-10: 40px; --ma-14: 56px; --ma-18: 72px; --ma-24: 96px;
}
```

Hard airiness rules (derived from your CX page so the feel is guaranteed):

- Panel padding is never below 16px, default 24px (`--ma-6`).
- Section gaps are 40 to 56px (`--ma-10` to `--ma-14`); the largest courtyard voids go to 96px (`--ma-24`).
- Nothing touches an edge: 44 to 72px of air at the top and bottom of hero surfaces.
- Conversational surfaces cap content width near 680 to 720px and center it.
- Volumes float in the void; never pack edge to edge.
- If a panel needs more than it can show airily, it summarizes and expands on demand. It never gets denser.

### Form and motion

Soft-cornered volumes (the slight chamfer of a poured edge); one fully-round tactile control. Motion has the weight of concrete: nothing snaps or bounces.

```css
:root {
  --radius:      8px;      /* poured volumes */
  --radius-pill: 9999px;   /* the single tactile control and the approval capsule */

  --ease:     cubic-bezier(0.2, 0, 0, 1); /* the weight of a heavy concrete door */
  --dur-fast: 120ms; --dur: 220ms; --dur-slow: 360ms; --dur-door: 480ms;
  --breathe:  2400ms;      /* the slow pulse of light: the agent breathing */
}
@media (prefers-reduced-motion: reduce) {
  /* all breathing and transitions resolve to static states (see component recipes) */
}
```

## Part IV. The Surfaces, in Detail

For each surface: first what it is as a room, in words, so the vision is unambiguous, then how it is built, in code.

### The volume (the base component every surface is made of)

A poured slab of concrete resting in the void: shadow-as-border, a top edge catching light, generous interior space.

```css
.volume {
  background: var(--concrete-2);
  border-radius: var(--radius);
  box-shadow: var(--hairline), var(--inner-glow); /* shadow-line + light catching the top edge */
  padding: var(--ma-6);                           /* 24px, never below 16 */
}
.volume--raised { background: var(--concrete-3); }
.volume:hover   { background: var(--concrete-4); transition: background var(--dur) var(--ease); }
```

### The alive element (the cross of light)

Any element the agent is currently animating: it breathes, light entering and easing back, slow.

```css
.alive { box-shadow: var(--inner-glow); animation: breathe var(--breathe) var(--ease) infinite; }
@keyframes breathe {
  0%, 100% { box-shadow: var(--inner-glow); }
  50%      { box-shadow: var(--inner-glow), var(--light-bloom); }
}
@media (prefers-reduced-motion: reduce) {
  .alive { animation: none; box-shadow: var(--inner-glow), var(--light-bloom); }
}
```

### The Workstation (the courtyard, the front door)

As a room: the largest, calmest, most open space, the courtyard you step into when you arrive. At the top, in air, a single confident line in display type tells you the state of the night: "7 agents ran. 312 files changed. 4 need you." Below it, the fleet: a small board of agent cards, each breathing if alive, steady if done, brightest-and-steady if it needs you, with the largest voids in the product between them. Below that, the merge review, ready when you are. No fifty notifications. One composed view, the chapel at dawn.

Built:

- The digest headline is `.t-display`, alone, with `--ma-18` of air above and `--ma-14` below.
- Agent cards are `.volume` slabs in a responsive grid with `--ma-8` gaps minimum, each carrying: task title (`.t-title`), one line of its live feed (`.t-code`, `--text-2`), and a state read by light (breathing / steady / lit). State is light, never a colored badge.
- There is no budget meter and no context-window cap, anywhere. Cloud tools meter context because it is scarce and expensive; HIDE is local and our format carries effectively unbounded context, so we differentiate by simply not metering. Do not show a token budget, a context percentage, or a budget bar on any surface. Abundance is expressed by the absence of the meter, not by a prettier one.
- Merge review reuses the diff-hunk component below, identical to the IDE.

### The IDE (the workshop chamber)

As a room: the workshop, the one chamber with real tools laid out. A file tree along one wall, the editor as the central work surface, a terminal below. The most information-rich room, and it still breathes: panels are slabs with air between them, the editor sits in space, the file tree is a quiet list with room, not a dense outline. When the agent proposes a change, it arrives in the editor as a diff you accept or reject by the hunk, the changed lines catching a faint lichen or oxide light, the rest of the code calm and unlit.

Built:

- Three volumes (file tree, editor, terminal) in the chamber grid, each `.volume`, with `--ma-4` to `--ma-6` between them. The editor is the largest.
- The file tree is a quiet list: 14px, `--text-2`, generous row height (line-height 1.8 or more), the active file at `--text-1` with a faint `--inner-glow` on its row. Not a dense tree.
- Diff review is inline by default, per-hunk accept and reject (Cline's model), keyboard-driven (j and k move, a accept, r reject), side-by-side on toggle for large diffs. Every accepted or rejected change rests on a git-checkpoint (Aider), instantly reversible.

```css
.hunk-add { color: var(--ok);  background: var(--ok-bg);  } /* line is prefixed with + */
.hunk-del { color: var(--bad); background: var(--bad-bg); } /* line is prefixed with - */
/* desaturated; the +/- glyph carries the meaning so color is never alone */
.hunk-actions { display: flex; gap: var(--ma-2); } /* a accept, r reject, keyboard-first */
```

### The Chat (the conversation chamber)

As a room: the calmest, most spacious chamber, just you and the agent. The conversation runs down a single readable column in the center of a wide, empty room, plenty of concrete on either side. The agent's words stream in plainly, with a faint light cusp at the leading edge. A persistent steering field waits at the bottom. There is no clutter, no avatars, no chat-app chrome; it is a quiet exchange in a large still space.

Built:

- Message column capped near 700px, centered, `--ma-18` top, `--ma-14` between turns.
- Agent prose is `.t-body` at line-height 1.6; the streaming leading edge carries a faint `--light-soft` cusp, no spinner.
- The steering field is a `.volume--raised` input pinned at the bottom with `--ma-8` of air around it, the one place you redirect the agent mid-flight, plus @-references to point it at exact files (Cline).

### The Context Stack (the light well)

As a room: a narrow vertical shaft of light along the east wall, present in every chamber. This is where the agent's work enters the building as light. By default it is calm and spacious, a quiet column of ledges, each a stratum of what the agent holds now, with real air between them. Glance at it and you know the state of the room. The bottom stratum, the current action, is a live feed of the agent's actual moves (opencode's action-log) catching the most light because it is the now. Reach into it and you can touch any stratum: pin a file, evict a memory, mute a tool, or @-pull a file, folder, problem, or git-diff into context. It is the signature of the whole product, and it must breathe, not pack.

Built:

- The strata, top to bottom, each a `.volume` with `--ma-6` interior and `--ma-4` between them: retrieved files and symbols, tools called, memory in play, tests and state, current action (the live feed). No budget stratum.
- The current-action stratum is `.alive` (it breathes) and scrolls the agent's real moves as `.t-code`, `--text-2`: "Reading guard.rs", "Running 12 tests".
- Touch affordances are quiet TE line-glyph controls on each stratum (pin, evict, mute, @-add), brighter on hover, dim at rest. Touch is how transparency becomes control.
- Not a graph, not a swarm. The cosmos lives in the mark and the light, never in a busy diagram (that would be the cockpit).

### The approval capsule (the lit instrument)

The one tactile control: a brushed-metal capsule that lights fully and holds steady when the agent needs you. Steadiness (not breathing) says it waits for you rather than works.

```css
.gate {
  border-radius: var(--radius-pill);
  background: var(--concrete-3);
  color: var(--light);
  box-shadow: var(--hairline-strong), var(--light-bloom), var(--inner-glow); /* lit, steady */
  padding: var(--ma-3) var(--ma-6);
  font-weight: 500; letter-spacing: 0.02em;
}
/* it states plainly what will happen, e.g. "Run migration. Approve." */
```

## Part V. Aliveness, Interaction, and Voice

### Aliveness, and HIDE's own progress signature

Light is the heartbeat, not color. When the agent is alive, the relevant element glows from within and breathes (the slow `--breathe` pulse). This is light entering the dark chamber, the Ando cross, the Hawking radiation.

HIDE has its own progress signature, not a stock spinner. Build it from the vision, light entering the dark: the event-horizon ring slowly filling or breathing with light, or a soft pulse of light rising into a dark volume. Grayscale and light only, weighted-calm motion, no color, no fast spin. The same signature appears wherever the agent is working (the chat leading edge, an active Context Stack stratum, a running fleet card, the status phase), so it reads as HIDE the way a spinning ring reads as one other tool. It is the ambient "alive" glyph layered on top of the truth, never a substitute for it: the live feed still shows the agent's actual moves. No percentage bar, no mystery spinner. Static lit state when `prefers-reduced-motion` is set.

### Interaction

Keyboard-first, command-palette-centric (opencode, Linear, Geist), mouse-rich where the interaction is physical (diffs, the Context Stack touch, the agent board). Vim is an option in the editor, never forced.

Plan and Build are explicit modes with a visible indicator (opencode and Cline both prove this). Plan reads and reasons without touching the project; Build acts.

Steering and approving, three verbs: see, steer, gate.

- See: the Context Stack.
- Steer: a persistent steering input plus @-references to point the agent at exactly what to read (Cline). The agent is interruptible, never fire-and-forget.
- Gate: per-step approval for consequential actions (Cline's per-step model), via the lit approval capsule, with a plain statement of what will happen. Toggle-able toward more autonomy for trusted work, but the default is human-in-the-loop.

Diff review: inline by default, accept or reject per hunk (Cline), keyboard-driven, side-by-side on toggle, every change on a git-checkpoint (Aider). Identical in the IDE and in Workstation merge review, so the harvested seams never show.

### Voice

Terse telemetry, the voice of a flight log: specific, plain, undramatic, in the interface's own voice, never cute, no emoji. Concrete patterns:

- Name the specific thing that changed, drop the trailing period, never say "successfully." "Diff accepted", not "Successfully accepted the change."
- Empty states point to the first action. "No agents yet. Describe a task to start one."
- In-progress uses the present participle and an ellipsis. "Reading guard.rs...", "Running tests...".
- Use numerals; skip "please" and any superlative. "3 agents ran".
- Errors are direct, specific, blame-free, and never apologize. "Couldn't reach the local engine. It may not be running."
- Labels name what the user controls, never how the engine is built, and keep the same word through a flow (Approve produces "Approved").

Standing copy rule: no em dashes, no en dashes, no middot separators, anywhere. Commas, colons, parentheses, or restructure. The agent's personality is competence and transparency, expressed by being specific and undramatic, like a senior engineer thinking out loud in short phrases.

## Part VI. Constraints, Amalgamation, and Self-Check

### Hard constraints

Standing rules (non-negotiable):

- No em dashes, en dashes, or middot separators in any copy or content.
- Geist (Sans) is logo only. Geist Mono is the entire type system (UI and code).
- Dark, material, airy: brutalist concrete that breathes, never flat vibecoded gradients and never packed density.
- Dual Fahrenheit and Celsius if temperature ever appears in copy.

Absolute nevers (design):

- No yellow, gold, acid, or neon. No brand hue. The only colors are desaturated lichen (`--ok`) and oxide (`--bad`), semantic and glyph-paired.
- No blue, no purple.
- No true `#000`.
- No translucency or glassmorphism.
- No gradient or soft-focus color hero.
- No serif-plus-sans display pairing. One font, Geist Mono.
- No packed, edge-to-edge, dense panels. Everything breathes.
- No budget meter, no context-window cap, no context percentage, anywhere.
- No spinner standing in for the agent's real work.
- No gratuitous bounce or snap.
- No literal concrete-photo texture or skeuomorphic ornament. The formwork is a whisper or a proportion, never a picture.
- Not a VS Code or Cursor clone; recast every harvested module in the same concrete.

Accessibility (hard requirements; you have shipped a low-contrast bug before, so this is enforced):

- All body and label text meets WCAG AA (4.5:1) against `--void`. Verify `--text-3` and `--mute` specifically; restraint must not cost contrast.
- Color is never the only signal; success, error, and diff add and remove each carry a glyph or a +/- marker.
- A visible focus ring on every interactive element at `:focus-visible`; never remove an outline without a visible replacement.
- `prefers-reduced-motion` honored: static fallbacks for the breathe and all transitions.
- Geist Mono set at comfortable size and line-height (1.6 body); the instrument look must not cramp legibility.

### The amalgamation map

Each design language contributes one layer, recast into one HIDE building. Not copying, a synthesis.

- Tadao Ando gives the soul and the space: board-formed grayscale concrete, the void as subject (ma), light as a building material (the cross of light), stillness, geometric purity, honest material, no ornament. The entire spatial and material language is his.
- Aesop gives the hand-scale materiality: muted pigment, tactile concrete-and-terrazzo restraint, a contemplative room.
- Teenage Engineering gives the instrument: the precise, labeled, brushed-metal control on a plinth; the engraving voice in the type.
- Vercel / Geist gives the execution grammar: dark as canonical, monochrome with accent as punctuation, shadow-as-border, the inner-light ring, the Geist Mono discipline, terse copy.
- Zed gives the discipline: the interface should disappear, content over chrome, no translucency.
- opencode gives the transparency, made real: the action-log of the agent's actual moves becomes the Context Stack's live feed; plus keyboard-driven plan and build modes, parallel sessions, and a privacy-first stance.
- Cline gives the human-in-the-loop interaction: plan and act, per-step approve and per-hunk reject, @-references to pull files and problems and git-diffs into context, checkpoints, and the multi-agent board that becomes the Workstation. HIDE inverts Cline's spend-limit anxiety into local abundance (no metering at all).
- Aider gives the safety net: git-native checkpoints, every change reversible.
- Linear gives the craft floor: command palette, speed, micro-interaction polish, dark that feels light through space.
- Your CX Launch page gives the proven airy material: the floating-in-air spacing, the felt-not-noticed lift, the line-glyph rings that brighten and dim, the brushed-metal inner-glow.

HIDE is the synthesis: Ando's space, Aesop's materiality, TE's instrument, Geist's grammar, Zed's discipline, opencode's transparency, Cline's steering, Aider's safety, Linear's craft, and your CX material, unified by one concept (the box that radiates light) and one device (light entering dark concrete).

### Self-check: tells that the build drifted to vibecoded

Run before anything ships. If any are true, it failed the vision.

- There is a bright accent color, especially yellow, gold, or acid, anywhere.
- There is blue or purple.
- Any panel is packed edge to edge or feels dense; anything touches an edge; sections are tighter than 40px.
- There is translucency, frosted glass, or a soft-focus gradient.
- Two type families appear (a serif and a sans) instead of Geist Mono alone.
- There is a budget meter, a context cap, or a context percentage anywhere.
- A spinner or percentage bar stands in for the agent's real work.
- The Context Stack is a packed data-wall or a busy graph instead of a calm, spacious light well.
- A harvested module was dropped in as-is and a seam shows.
- Motion bounces or snaps.
- There is a literal concrete-texture photo or any skeuomorphic ornament.
- Geist Sans appears outside the logo.
- An em dash, en dash, or middot reached the copy; or an error apologized; or a toast said "successfully".
- Body or label text fails AA contrast on `--void`.

If none are true: the chamber is dark, grayscale, and still, the volumes float in generous void, and light enters where the agent works. The box is radiating light. That is HIDE.

## Part VII. The Work Order

The current build runs and is a competent VS Code plus Cursor clone: file tree, tabbed editor, inline diff with per-hunk accept and reject, a terminal, a chat panel, a context panel with a budget bar, a status bar reading "mock transport." Keep the skeleton. Destroy the skin and replace the wiring. Parts I to VI above are the authority for every visual and material decision. `BLACKHOLE.md` still governs engineering discipline.

### Phase 0. Explore before you change a single line

Do not edit anything in this phase. Read the codebase and write a short map.

1. Surfaces and components. Inventory every component (activity rail, file tree, editor tabs, editor, inline and side-by-side diff, terminal, chat, context panel, status bar, command palette). Note which are harvested or VS-Code-derived.
2. The theme layer. Find where colors, fonts, spacing, and component styles live (CSS variables, Tailwind config, theme files). This is what you replace wholesale.
3. The transport layer. Find where "mock transport" is defined and how each surface talks to it.
4. The real engine contract. Find the actual backend API for HIDE (the agent and IDE engine) and for Hawking (the model engine running our local compressed models): the HTTP, WebSocket, or IPC surface, the message and event schema, and how streaming works. Do not guess endpoints; read them.
5. The context and format handling. Find the code handling context and our model format, specifically the function or capability for effectively unbounded or very long context. Locate it; Phase 2 depends on it.
6. The agent loop. Find plan and build handling, the diff apply and reject logic, file read and write, terminal execution, and how agent actions are surfaced.

Deliver a short written map, then proceed.

### Phase 1. Wire every surface to the real engine (HIDE and Hawking)

Replace mock transport with the real connection, using the contract from Phase 0. Do not invent endpoints; if something needed is missing, list it instead of faking it.

- Hawking (model engine): connect inference to the real local model serving; the model indicator reflects the actual loaded model and real connection state.
- HIDE (agent and IDE engine): connect the agent loop, plan and build modes, file read and write, diff apply and reject, terminal execution, and context retrieval.
- Streaming: chat and the agent action feed stream from the real engine.
- Status bar: "mock transport" becomes the true connection state (Ready, Connecting, Offline). Keep the phase indicator (idle, plan, build), wired to real state.

### Phase 2. Remove the budget and the context cap entirely

A core differentiator, not cosmetic.

- Our format carries effectively unbounded or very long context (use the capability found in Phase 0).
- Remove the BUDGET section from the context panel completely: the "14,210 / 16,384" figure and the multi-color segmented bar (system, code, tools, memory, history). The segmented color bar also violates the no-hue rule, so it is doubly out.
- Remove the context-percent indicator next to the model (the "43%").
- Do not replace these with a prettier meter. Show no token budget, no context-window cap, and no budget bar anywhere. The context panel keeps its real strata (retrieved, tools, memory, current action) and loses only the budget (see Part IV, the Context Stack).

### Phase 3. Restore the theme, hard (the main work)

Recast the entire surface into the grayscale concrete and light system in Parts III and IV. Pull tokens from the Appendix verbatim. Non-negotiables, restated so they cannot be missed:

- Grayscale concrete only. `--void` through `--concrete-4`, never pure black. Every surface is a poured volume in a dark void.
- Light is the only accent. Light entering dark concrete: inner-glow, bloom, breathe. Nothing glows in a color.
- No hue. Only the two desaturated pigments (`--ok` lichen, `--bad` oxide), glyph-paired, in diffs and state only. The current saturated red and green diff blocks recast to these; kill every other color, including the budget segments and any blue or purple chrome.
- One font: Geist Mono, everywhere. Geist Sans logo-only. Drop any other family.
- Airy. The ma scale: 24px panel padding min, 40 to 56px section gaps, nothing touches an edge, volumes float. The packed VS Code density is out; the editor and context panel must breathe.
- Material rules. Shadow-as-border, no CSS borders, no translucency, no gradient hero, no neon, no concrete-photo texture.
- Recast every harvested component (activity rail, file tree, tabs, diff, terminal, chat, context, status bar) out of its VS Code skin into the concrete, re-housed into the chamber model (mode rail is the west wall, editor and panels are the workshop chamber, context panel is the east light well, chat is the conversation chamber).
- Diff stays, restyled. Keep per-hunk accept and reject and the keyboard model (j and k move, a accept, r reject); only restyle.

Run the self-check in Part VI at the end of this phase. If any tell is true, it is not done.

### Phase 4. Give HIDE its own progress signature

Design HIDE's own progress indicator, not a stock spinner. See Part V (Aliveness). Build it from light entering the dark, make it recognizable and consistent across every surface, layer it on top of the real action feed rather than replacing it, and give it a reduced-motion fallback. This is yours to invent within the doctrine, so put your own spin on it.

### Operating rules

- Parts I to VI win every look-and-feel call. `BLACKHOLE.md` governs engineering. Read the vision (Part I) first.
- One coherent pass, not a patchwork. Protect the public contract, the existing behavior, the build, and the assets. Recast components in place.
- No em dashes, no en dashes, no middot separators in any copy.
- Copy voice is terse telemetry (Part V).

When done: the three chambers are dark grayscale concrete, the volumes float in generous void, light enters only where the agent works, there is no budget meter and no context cap, the progress signature is unmistakably HIDE, and every surface is wired to the real HIDE and Hawking engine. It no longer looks like anyone else shipped it. That is HIDE.

## Appendix. Consolidated tokens (copy-paste)

```css
:root {
  /* type */
  --font: "Geist Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, "Courier New", monospace;

  /* concrete (surface) */
  --void: #070707;
  --concrete-1: #0E0E0F;
  --concrete-2: #141416;
  --concrete-3: #1B1B1E;
  --concrete-4: #222226;
  --formwork: rgba(255,255,255,0.02);

  /* shadow-lines (replace all CSS borders) */
  --line: rgba(255,255,255,0.06);
  --line-strong: rgba(255,255,255,0.11);
  --hairline: 0 0 0 1px rgba(255,255,255,0.06);
  --hairline-strong: 0 0 0 1px rgba(255,255,255,0.11);
  --depth: 0 0 0 1px rgba(255,255,255,0.06), 0 12px 32px -16px rgba(0,0,0,0.7);

  /* light (the only accent) */
  --light: #F4F2EE;
  --light-soft: rgba(244,242,238,0.06);
  --light-bloom: 0 0 28px rgba(244,242,238,0.10);
  --inner-glow: inset 0 1px 0 rgba(255,255,255,0.06);
  --grade-wall: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0) 42%);

  /* text (chalk on concrete) */
  --text-1: #ECEAE6;
  --text-2: #9B9A95;
  --text-3: #6E6D68;
  --mute: #5C5B57;

  /* pigment (semantic punctuation only, glyph-paired) */
  --ok: #7E9E86;  --ok-bg: rgba(126,158,134,0.08);
  --bad: #C0807A; --bad-bg: rgba(192,128,122,0.08);

  /* void (ma scale) */
  --ma-1: 4px; --ma-2: 8px; --ma-3: 12px; --ma-4: 16px;
  --ma-6: 24px; --ma-8: 32px; --ma-10: 40px; --ma-14: 56px; --ma-18: 72px; --ma-24: 96px;

  /* form and motion */
  --radius: 8px; --radius-pill: 9999px;
  --ease: cubic-bezier(0.2, 0, 0, 1);
  --dur-fast: 120ms; --dur: 220ms; --dur-slow: 360ms; --dur-door: 480ms;
  --breathe: 2400ms;
}
```

One sentence to hold the whole thing: build a Tadao Ando building in grayscale concrete, leave generous void around everything, let no color in, make it come alive only where light enters the dark (which is where the agent works), and wire that to the real engine.
