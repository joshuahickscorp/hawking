# HIDE Design Doctrine
### Hawking IDE: the look, the feel, the interaction philosophy

> Part of the HIDE front-end bible. This is the **binding design system**: the look, feel, tokens, type, motion, voice, and interaction rules every surface and every harvested module conforms to. It answers the design brief and governs [01-surfaces.md](01-surfaces.md) (it turns "what panels exist" into "what they look and feel like"). When this doctrine and a harvested module disagree, the doctrine wins, and the module gets re-housed and re-skinned. The Self-check at the end is the ship gate.

> One line: HIDE is the IDE named after the man who proved the black box leaks. A black hole is the ultimate black box, nothing escapes, you cannot see in. Hawking proved that is false: black holes radiate. HIDE makes the agent's black box radiate everything it sees and does. The whole product is that one idea expressed in pixels.

> The dual face (resolves the ironic name): HIDE hides *you* from the cloud (offline, local, nothing leaves your machine) and hides *nothing* from you (the Context Stack). Privacy outward, transparency inward. Same object, two faces, exactly like a black hole.

---

## 0. The spine

Read this part first. Everything below is downstream of it.

**The feel:** an observatory, not a cockpit. You sit in the dark and watch enormous work happen with serene clarity. The danger with "show everything" is that it becomes a cockpit: alarms, gauges, white knuckles. We reject that. The agent does vast work; you observe it with the calm of an astronomer, not the stress of a fighter pilot. Calm, powerful, instrument-grade.

**The one adjective we optimize the whole design for: legible.** Not "clean," not "minimal," not "fast," not "powerful" (everyone in this space claims those). Legible. You can read what the agent is doing at a glance, the type is legible, the layout is legible, every state is legible. Radical transparency only works if it is legible, otherwise it is just noise, which is the failure mode of every "show the logs" tool. Legibility is also the craft value that separates material brutalism from vibecoded mush. Two supporting qualities under it: **calm** (so transparency never becomes a cockpit) and **material** (so it never becomes cheap flat).

**The through-line, the one thing that makes HIDE instantly HIDE:** a single luminous warm gold rim-light on near-black material surfaces. Everything alive or important wears a thin gold edge against deep near-black: the active agent, the approval control, the streaming edge, the mark itself, the live stratum of the Context Stack. It is the black box that radiates, made into a consistent visual device. It continues your CX anodized-capsule language (dark recessed surfaces, inset rim, glowing label) directly. When you see a thin gold glow on the edge of a dark recessed panel, that is HIDE and nothing else.

**The brand family insight (use this everywhere):** Hawking Condense and HIDE are the two phenomena of a black hole. Condense *compresses* (the model-maker drives matter toward singularity density, the ultimate compressor). HIDE *radiates* (it makes the agent's work escape and become visible). Compression and radiation. One family, two faces of the same physics. That is a tight, honest, ownable story and it is grounded in what the two products literally do.

---

## 1. North star / ethos

**Feel:** a calm, powerful instrument. Specifically an observatory. Not a friendly pair-programmer (too soft, too chatty, undersells the power and the transparency), not a fast dense cockpit (the thing we must actively design against, because "show everything" tends to drift there).

**Channel (look and feel to borrow from):**
- **Linear.** The gold standard for dev-tool craft: dark, fast, keyboard-first, command palette, immaculate micro-interactions, zero jank. We borrow the craft floor and the speed. We do not borrow the purple.
- **Things 3.** Calm, restraint, generous breathing room, opinionated curated views, delight that never gets loud. We borrow the calm and the opinionation.
- **Teenage Engineering (OP-1 / field gear).** The instrument: every control labeled, tactile, precise, functional-technical, material aluminum. This is the soul of "show everything and let me touch it." We borrow the labeled-instrument feel and the tactility.
- **032c.** Editorial typographic confidence: big assured display type, the big-number-as-statement moment. We borrow the confidence to set huge serif headings and editorial numbers in a tool that "should" be all small sans.
- **Aesop.** Muted premium restraint, warm neutrals, material, lots of negative space, considered. We borrow the restraint and the warmth.

One non-software reference worth keeping pinned: the Event Horizon Telescope image of M87\*. Black core, warm gold photon ring. That image *is* the brand.

**Avoid (deliberately):**
- **VS Code.** The clone trap. Generic chrome, the blue, infinitely-dockable patchwork, no point of view. This is the gravity well we are escaping.
- **Cursor.** A chat panel bolted onto VS Code. It is exactly the "chat box on a black box" pattern the product exists to reject. We must not look like it.
- The category to avoid as a whole: the vibecoded AI-startup look (purple-blue gradients, glassmorphism, rounded-everything, Inter plus a gradient logo, the ChatGPT-wrapper aesthetic). And its dark cousin, the fake-hacker neon-green terminal.

---

## 2. Brand and identity

**How "Hawking" reads visually:** cosmic and black-hole, yes, but as a *conceptual spine expressed minimally*, never as literal galaxy wallpaper or starfields. Material brutalism, not a screensaver. The cosmos lives in the mark, the single gold accent, and the radiation language. The chrome stays disciplined near-black.

**The mark:** the event horizon. A precise circle, a black disk with a thin luminous gold rim, abstracted from the EHT photon ring. The mark literally encodes the thesis: a black core (the unknown, the black box) that radiates light at its edge (the information escaping, the transparency). This is the CX rim-light treatment applied to a circle. It scales from a 16px favicon to a hero. Optional refinement: a faint asymmetry in the rim brightness (the accretion-disk lean) so it reads as observed light, not a UI ring.

**Wordmark:** Geist, logo only, per your standing rule. This is the single place Geist appears anywhere. "HIDE" as the product wordmark, "Hawking" as the maker line, the event-horizon ring as the family glyph that sits before the word. Geist nowhere else, ever.

**Shared identity with Condense:** yes, one family. Same ring glyph, same Geist maker wordmark, differentiated only by product name (and, if you want, a slightly different rim hue: Condense could rim cooler/whiter, HIDE rims gold). The family story from section 0 (compression and radiation) is the connective tissue. Lead with it whenever the two products appear together.

**Tagline:** I would carry two registers.
- Primary, the thesis in three words: **Open the box.** It plays directly against "black box" and states the whole product.
- Long form, the dual face: **Nothing leaves your machine. Nothing's hidden from you.**
- Bench/honest alternative if you want the local-lavish angle foregrounded: **Local. Legible. Lavish.** (on-device, transparent, free to spend compute without limit).

---

## 3. Color and theme

**Dark-first, dark primary, dark is the soul.** Not a preference, a function: an IDE you watch all night, organized around staring into the agent's work, is an observatory and observatories are dark. Light mode is a later accommodation, never the marketing hero, possibly a restrained "paper" mode in v2. Build dark, and build it properly.

**Material, not flat.** The base is not VS Code's flat #1e1e1e grey. It is your #060606 near-black with material depth: subtle two-layer gradients on surfaces, inset shadows for recessed channels, hairline rims, real elevation. Anodized, not painted. This is your CX capsule language applied to the whole shell.

**On the cliche:** "near-black with a single bright accent" is a known AI-default look. We are not it, and here is why: the dark base is mandated by the doctrine (not a lazy free choice), the accent is *derived from the subject* (the gold photon ring and Hawking radiation, not a generic acid-green or vermilion), and the surfaces are *material* (rim-lit, recessed, anodized) rather than flat. The serif-plus-mono type (section 4) seals it. If any screen starts to read as "flat dark with a neon dot," it has failed the doctrine.

**Tokens (near-black material ramp):**

| token | value | use |
|---|---|---|
| `--void` | `#060606` | base background (your established base) |
| `--surface-0` | `#0B0B0C` | raised panel |
| `--surface-1` | `#111113` | card |
| `--surface-2` | `#18181B` | elevated / hover |
| `--rim` | `rgba(255,255,255,0.06)` | hairline border, the recessed-channel edge |
| `--rim-strong` | `rgba(255,255,255,0.10)` | emphasized edge |
| `--text-hi` | `#F2F0EC` | primary text, warm off-white, AA on `--void` |
| `--text-mid` | `#A8A6A1` | secondary text |
| `--text-low` | `#7C7A75` | metadata (keep at AA, see constraints) |

**Signature (the radiation, the brand life-color):**

| token | value | use |
|---|---|---|
| `--radiation` | `#F0B95B` | agent active / streaming, the breathing glow |
| `--radiation-bright` | `#FFD888` | needs your approval, peak glow, lit controls |
| `--radiation-bloom` | `rgba(240,185,91,0.32)` | the rim-light bloom / glow shadow |

**Semantic palette:**

| meaning | token | value | notes |
|---|---|---|---|
| agent active / streaming | `--radiation` | `#F0B95B` | soft, breathing, never a spinner |
| needs your approval | `--radiation-bright` | `#FFD888` | steady, rim-lit control, plus icon and label |
| success / tests passed | `--success` | `#6FBF8B` | muted jade, never neon, never terminal-green |
| error / danger | `--danger` | `#E5635E` | refined signal red, a calm nod to 032c red |
| warning | `--warning` | `#E08A3C` | orange, deliberately separated from the gold |
| diff added | `--diff-add` | fg `#8FD0A6`, bg `rgba(111,191,139,0.10)` | plus a `+` marker, not color alone |
| diff removed | `--diff-del` | fg `#E58B86`, bg `rgba(229,99,94,0.10)` | plus a `-` marker, not color alone |

**Off-limits:**
- No blue and no purple, anywhere, at all. This is a hard rule and it is a feature: the absence of blue separates us from VS Code (blue) and the absence of purple separates us from Linear and Cursor and every AI wrapper. Our accent system is neutrals plus gold plus green/red/orange. Nothing cool.
- No true black `#000`. Use `#060606`. True black kills the material depth and is harsh under long sessions.
- No glassmorphism / frosted translucency as a primary device.
- No neon / acid saturation (the cheap-hacker tell).
- Color is never the only signal (see accessibility in section 14).

**Mood:** muted desaturated pro across roughly 95 percent of the surface (Aesop restraint), with the gold glow as the one place allowed to be luminous and alive (bounded maximalism: a calm near-monochrome field, one radiating accent). That contrast *is* the design: the dark observatory, and the glowing thing you are observing.

---

## 4. Typography

The typographic signature is a serif display paired with a mono everything-else. No IDE does this. That alone is half your distinctiveness.

**Display: Cormorant Garamond.** Big, confident, editorial (032c energy). An elegant high-contrast serif inside a coding IDE is a strong, deliberate statement: this is a considered instrument, not a generic tool. Used strictly for large display: surface titles, the overnight digest's hero number, empty-state hero lines, section heads. Strictly display sizes (28 to 72px); Cormorant gets spindly at body sizes, so it never drops below display.

**Everything functional: Geist Mono.** Not just code. Labels, buttons, status, metadata, the file tree, the Context Stack, the agent narration, the diffs. All of it in mono. This makes HIDE feel like a precision instrument where every control is labeled (the OP-1 / field-gear feel), and it is radically distinctive: no IDE runs mono-everything UI. Mono also *helps* legibility in dense lists (alignment, scanning), which is why terminals use it, which serves the Context Stack directly. Set it at comfortable sizes and line-heights for UI (11 to 14px, line-height 1.5 to 1.6), this is a legibility requirement, not an aesthetic afterthought.

**Logo: Geist.** Wordmark only. Never in UI text. (Standing rule.)

**Type scale:** high contrast on purpose. Huge confident Cormorant display against small precise Geist Mono chrome. The drama is the gap between elegant-large-serif and small-technical-mono. Not flat-utilitarian, the editorial confidence is part of the brand.

One allowance: if a genuinely prose-heavy agent explanation in Chat starts to feel tiring in mono, Cormorant (at a reading size, not display) is the only fallback. Default stays mono, which reads correctly as transcript/log and fits the terse voice.

---

## 5. Density and layout

**Differentiated density, not uniform.** The editor and the Context Stack are dense (pro IDE, information-rich, the density of a Bloomberg terminal with the craft of Linear) because that is where you work and observe. The Chat and the overview/"between" spaces breathe (Things 3 calm, generous spacing) because that is where you think and converse. You correctly intuited this split; we formalize it.

**What keeps it from feeling like two different apps:** a strict grid and one consistent spacing scale (4px base, 8px rhythm) everywhere, so even the dense surfaces read as composed, never cramped. Density through information, never through abandoning spacing discipline. The instrument model: a control panel is dense, but every control has alignment and breathing room. Dense but never cramped is the rule.

**Layout stance: opinionated and curated, not infinitely dockable.** VS Code's everything-is-draggable is precisely the identity-less, patchwork-enabling choice we are rejecting, and it is the thing that would let harvested modules stay a patchwork. Instead: a small number of designed, named, canonical layouts per surface (the Things 3 / Linear stance of fixed-but-perfect). Splits resize (drag to widen the Context Stack or the editor), but panels do not freely rearrange or dock anywhere. The opinionation *is* the design system that unifies the borrowed parts. This is the single most important structural decision for not shipping a patchwork: every harvested component gets re-housed into a fixed, designed slot, never dropped in as-is.

---

## 6. The three surfaces and how they relate

**One unified shell, three modes, with the Context Stack as the constant spine across all three.** You never "leave" one space for another. You change what is on the main stage while the Context Stack (the agent's live state) persists at the edge. That persistent rail is the thread that makes the three surfaces feel like one product: wherever you are, you are watching the agent.

**Center of gravity: observation-first.** Not editor-first (that is Cursor, the clone trap) and not purely chat-first. The product is organized around *watching the agent*, and the three surfaces are three lenses on that single activity:
- **AI IDE** = watch and touch the code the agent changes.
- **AI Chat** = watch and steer the agent's reasoning.
- **AI Workstation** = watch and manage many agents at once.

This is the honest expression of your actual differentiator (transparency plus lavish local parallelism), and "observation-first" is genuinely novel: nobody else is organized this way.

**The front door is the Workstation.** When you open HIDE you land on the overview: your agents, your runs, what happened overnight, project state at a glance. You dive *into* the IDE or Chat for focused work, the way Linear opens to your issues and you dive into one. Putting the most novel, most only-local surface at the front door is the right strategic framing. The IDE must still be a genuinely excellent editor (not an afterthought), it is just not the landing.

**Moving between surfaces:** a persistent mode switcher (three named modes as a left icon-rail or segmented control, the glanceable spatial anchor) plus a command palette (⌘K for everything, the keyboard-first power-user spine, Linear/Things style). Not a sea of tabs (VS Code tab-hell is out).

---

## 7. The Context Stack (the signature feature)

**Prominence: an always-visible side rail by default, expandable into a full inspector on demand.** This is how radical transparency stays empowering instead of overwhelming: the default is a calm, glanceable summary (legible at a glance, there is the adjective again), and you drill into depth only when curious. Progressive disclosure is the entire trick. Always present, never shouting.

**The metaphor stays literal: a stack.** A vertical column of strata, each a layer of what the agent is holding right now, top to bottom:
- retrieved files and symbols
- tools called
- memory in play
- tests and state
- current action (this stratum is a live feed, the now, streaming the agent's actual moves)

Scan the column to take in the whole state; expand any stratum for depth. Stable structure (the strata) with one live-feed stratum (the present). I deliberately did *not* choose a graph or a swarm or a galaxy visualization for this: a graph is the overwhelming choice you were worried about, and it would betray the "calm" rule. The cosmos lives in the brand and the glow, not in forcing the context into a busy diagram. Keep the literal metaphor literal.

**"Let me touch it" is the differentiator, and the design must make touch obvious.** Every stratum is editable: pin or unpin a file from context, evict a memory, mute a tool, inject a note. Every stratum carries its affordances visibly (a pin, an x, a drag handle). This is where the CX physical-control language pays off hardest: the strata should feel like tactile hardware modules you toggle, like an OP-1's labeled controls or a patch bay. Touch should feel material. That tactility is what turns "transparency" into "control," which is the whole promise.

Mission-control in spirit (you are monitoring), expressed as a clean stack, never as a busy dashboard.

---

## 8. Agent presence and aliveness

**Motion philosophy: restrained, but alive. The light is the heartbeat.** No anthropomorphic mascot, no bouncy character animation. Aliveness is carried by the gold radiation: when the agent thinks, a soft breathing pulse on the relevant element; when it streams, text arrives with a subtle gold leading edge (the radiation leaking out). Aliveness equals the box radiating. This ties the agent's life directly to the brand color and the black-hole concept, which is exactly the kind of coherence we want.

**Token streaming:** text streams with a refined leading cursor, a subtle gold block or glow at the streaming edge. No jank, no flicker, calm cadence. The edge is where the radiation is.

**Progress without anxiety, the most important rule in this section:** show the actual work, not abstract progress theater. Avoid spinners and percentage bars, they manufacture false precision and create the cockpit anxiety we are designing against. Convey "working" through (1) the breathing gold glow on the active stratum, (2) the live-feed stratum scrolling the agent's real actions (Reading auth.ts, Running 12 tests), and (3) slow ambient motion. Seeing the real action is calmer and more trustworthy than a mystery spinner. The anti-anxiety move *is* the product thesis: transparency reduces anxiety, so aliveness and transparency reinforce each other instead of fighting. Restrained motion, one living color, real-work-as-progress.

---

## 9. Parallel and overnight agents

**"Many agents working" is a board of cards, not a swarm.** A chaotic swarm visualization looks cool in a demo and reads as pure noise in use (the cockpit/overwhelming trap, again). Instead, a calm board/grid where each agent is a card showing: what it is working on, one line of its live feed (current action), and its status by glow (breathing gold = active, green = done, amber = waiting on you). A fleet you take in at a glance, like a Linear board or a Things project list where each row is a live agent. Calm, scannable, legible.

**Walking back after agents ran all night: a morning digest, not 50 notifications.** Fifty notifications is anxiety. One composed overnight view is the observatory at dawn: you sit down and the night's observations are composed into a legible report. What ran, what needs your review (ranked first, in gold/amber), what is ready to merge, what failed. Open it with a single ambient editorial number in big Cormorant Garamond ("7 agents ran. 312 files changed. 4 need you."), the 032c big-number-as-statement moment, which is also a genuine delight and entirely on-brand. Reading the morning's findings, calm and complete.

**Merge review of parallel outputs:** a calm queue, each agent's diff reviewed with the *same* hunk-by-hunk accept/reject as the IDE. Consistency of the diff-review interaction across every surface is essential given we are assembling harvested modules, the review gesture must be identical everywhere or the seams show.

---

## 10. Interaction model

**Keyboard-first, command-palette-centric, mouse-rich where things are physical.** ⌘K is the spine (jump anywhere, do anything, Linear-style), comprehensive shortcuts throughout, this fits the developer / LocalLLaMA audience. But not vim-modal-mandatory: vim is an *option* in the editor for those who want it, never forced, because forcing it narrows the audience. Mouse stays rich where the interaction is spatial or tactile: diff review, the Context Stack touch-interactions, the agent board. Linear's balance is the model.

**Steering and approving, three verbs: see, steer, gate.** This maps exactly to "show everything and let me touch it."
- **See:** the Context Stack (you always know what the agent is doing).
- **Steer:** a persistent steering input, always available, to redirect the agent mid-flight ("actually, use X"). The agent is interruptible and steerable, never fire-and-forget.
- **Gate:** an approval gate for consequential or irreversible actions (a destructive command, etc.). Clear and calm, unmistakable but not a trap. The CX glowing-button language is perfect here: the approval is a lit control waiting to be pressed, `--radiation-bright`, with an icon and a plain-language statement of what will happen.

**Diff review:** inline in the editor by default (changes shown in place, green/red blocks, accept or reject per hunk via keyboard: move with j/k, accept/reject with a/r, a calm flow), with side-by-side available on toggle for large or complex diffs. Line-by-line granularity available. Inline keeps you in context and stays calmer; side-by-side is there for careful comparison. Identical interaction in the Workstation merge review.

---

## 11. Motion and micro-interactions

**Polished but restrained: weighted calm.** Every motion is purposeful and refined (Linear-grade easing), never gratuitous, but there *are* signature moments. Bounded maximalism: a disciplined field with a few perfect motions. Things have mass; nothing snaps or bounces cheaply (cheap snap-bounce is the vibecoded tell).

Signature motions:
- **The radiation pulse.** The gold breathing glow on active elements, the agent's heartbeat, the through-line motion of the whole product.
- **How a diff lands.** Accept a hunk and the red/green settles cleanly into normal code with a brief, satisfying absorption (the change is taken in). Reject and it dissolves back out. The acceptance should feel like a clean physical action, a key resolving on the OP-1, your material language.
- **How a finished run resolves.** Breathing gold (active) transitions to a steady state (green done, or amber needs-you) with a brief calm settling: the radiation quiets. No fireworks (noise/anxiety), quiet completion. Calm closure. The overnight version of this is the morning digest.
- **How panels appear.** The Context Stack expanding to inspector slides and grows with weight and ease, never pops. Material, with mass.

Respect `prefers-reduced-motion` everywhere: the breathing glow and every transition need a static fallback. Given how central motion is, this is a real requirement, not a checkbox.

---

## 12. Voice and copy

**Telemetry, not chatter. Terse, technical, dry, precise.** The voice of a flight log or mission control. Not playful or cute (no "Oops!", no emoji, that is the cheap startup voice and it would undercut the instrument), not warmly chatty. It states what is happening, plainly, in the interface's own voice.

- **Empty states:** calm and matter-of-fact, with one elegant Cormorant hero line allowed as the editorial touch. An empty screen is an invitation to act, not a mood.
- **Errors:** direct, specific, blame-free, actionable. They never apologize and are never vague. "Couldn't reach the local engine. It may not be running." Not "Something went wrong."
- **Labels:** name things by what the user controls and recognizes, never by how the engine is built. A control says exactly what it does, and keeps the same word through the whole flow (the button that says Approve produces a state that says Approved).

**Agent personality:** competence and transparency, expressed by being specific and undramatic, like a good senior engineer thinking out loud in short phrases. Present-tense, specific narration: "Reading auth.ts", "Running 12 tests", "3 failed, fixing." Never "Sure! Let me take a look at that for you!" The agent earns trust by being precise and legible, not by being friendly. A flash of dry wit is allowed in non-critical copy (empty states); never cutesy, never in errors.

---

## 13. Distinctiveness: the one thing

If forced to one through-line, it is the look from section 0: **near-black material surfaces wearing a single luminous warm-gold rim-light, paired with confident Cormorant Garamond display type.** The rim-light is the device that repeats across every surface (the active agent, the approval control, the streaming edge, the mark, the live stratum). It is the black box that radiates, and it is unmistakably not VS Code or Cursor (flat, blue, sans, no material, no serif). When a thin gold glow sits on the edge of a dark recessed panel, that is HIDE.

The functional twin of that visual signature is the **Context Stack as a persistent spine** (observation-first, the agent always legible at the edge). The visual and the structural say the same thing: the box radiates, and you are always watching it.

If you want a one-sentence positioning to hang it all on: *most AI tools are a chat box on a black box; HIDE is the box, opened, lit at the rim, and yours alone.*

---

## 14. Hard constraints

**Standing rules (must-haves, non-negotiable):**
- No em dashes or en dashes anywhere in UI copy or content. Commas, colons, parentheses, restructured sentences.
- No middot as a separator. Use alternatives.
- Geist is logo only. Never in UI text.
- Geist Mono is the monospace and UI font.
- Dark brutalist aesthetic governs. Material brutalism, not flat vibecoded gradients or glassmorphism.
- Temperature/cooking references (if any ever appear in copy or examples): both Fahrenheit and Celsius.

**Absolute nevers (design):**
- Not a VS Code clone. No VS Code chrome, no VS Code blue, no infinitely-dockable patchwork. Every harvested module gets re-housed into a designed fixed slot.
- No purple or blue anywhere. The accent system is neutrals plus gold plus green/red/orange.
- No true black `#000`. Use `#060606`.
- No glassmorphism / frosted glass as a primary device.
- No neon / acid / fake-hacker terminal green.
- No cutesy or emoji voice.
- No gratuitous bounce or snap motion.

**Accessibility (treat as hard requirements):**
- The muted aesthetic must not cost text contrast. Body and label text meet WCAG AA against `--void` (`#060606`). Verify `--text-low` specifically, it is the one at risk. The gold accent must be legible where it carries text.
- Color is never the sole signal. Needs-approval, success, error, and especially diff-added/removed must also carry an icon, a shape, or a `+`/`-` marker, for red-green color blindness.
- Keyboard-navigable everything (it is keyboard-first anyway). Visible focus states.
- Respect `prefers-reduced-motion`: static fallbacks for the breathing glow and all transitions.
- Mono-everything must be set at comfortable size and line-height; do not let the instrument look cramp legibility.

**References to match in spirit:** Linear (craft, speed, keyboard, command palette, motion polish), Things 3 (calm, restraint, opinionated curated layouts), Teenage Engineering OP-1 (labeled tactile instrument, mono controls, material), 032c (editorial display type, the big-number moment), Aesop (muted premium restraint, warmth), the EHT M87\* image (the gold photon ring, the mark and the glow). Your own CX capsules (the material rim-light language) and the TailorAI base (`#060606`, Cormorant plus Geist Mono) for direct continuity.

**References to avoid literally:** VS Code, Cursor, generic vibecoded AI-wrapper aesthetics, neon-hacker terminals.

---

## Self-check: tells that we are failing the doctrine

Run this against any screen before it ships. If any are true, it has drifted.

- It could be mistaken for VS Code or Cursor.
- There is blue or purple on screen.
- The dark surfaces are flat (no rim, no recess, no material depth), so it reads as "flat dark with a neon dot."
- A spinner or a percentage bar is standing in for showing the agent's real work.
- The Context Stack is a busy graph or a swarm instead of a legible stack.
- The transparency feels like a cockpit (alarms, noise, anxiety) instead of an observatory (calm watching).
- The voice apologized, used an emoji, or got cute in an error.
- A harvested module was dropped in as-is instead of re-housed into a designed slot, and you can see the seam.
- Geist appears somewhere other than the logo.
- An em dash, en dash, or middot made it into copy.
- Motion bounced or snapped cheaply.
- Body or label text fails AA contrast on `#060606`.

If none are true: the box is radiating, at the rim, in gold, and it is legible. That is HIDE.
