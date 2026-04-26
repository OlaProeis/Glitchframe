# Suno Prompting Skill v6.1 — Bass Music Production Guide

> Target model: **Suno v5.5** (Custom Mode). Verified against the official Suno API limits, Suno blog/docs, the bitwize-music v5 best-practices reference, jackrighteous.com guides, r/SunoAI community consensus (2025–2026), and AceTagGen's prompt-scorer analysis.

---

## What Changed from v6.0 → v6.1

v6.0 correctly located arrangement intent in both fields but **didn't go hard enough on arrangement force-control**. The #1 user-facing problem with Suno 5.5 is **arrangement chaos**:

- Drops 15–30 seconds long that never come back
- Second drop completely different from the first
- 4 verses and zero drops, or 15 drops in a row
- `[Outro]` adds brand-new material instead of resolving

v6.1 adds a dedicated **Arrangement Force-Control Toolkit** with the two-layer "Structure Prompting" technique (r/SunoAI advanced guide), explicit repetition rules, bar-count targeting, mirror anchoring, and a pre-generation failure-mode checklist.

**New in v6.1:**

- Two-layer Structure Prompting — arrangement narrative in Style + bracket tags in Lyrics
- Section repetition rule — write `[Drop]` three times if you want three drops
- Explicit arrangement line in Style field (a required final descriptor)
- Mirror anchoring — echo core descriptor at end of Style field
- Bar-count suffixes: `[Intro 4]` `[Drop 16]` `[Break 4]`
- `[Silence: 2s]` pre-drop impact cut
- Frankenstein / Song-Editor Extend workflow for surgical drop-locking
- Pre-generation checklist keyed to the 4 failure modes

---

## What Changed from v5.0 → v6.0

v5.0 claimed a 120-char Style limit, treated bracket-prose as a serious tool, and let the Lyrics field carry all arrangement detail. Research against Suno 5.5 showed those assumptions wrong.

| Was (v5.0) | Is (v5.5 reality) |
|---|---|
| Style field "hard limit 120 chars" | **1000 chars max, 150–300 target** (120 was V4 only) |
| Arrangement lives in Lyrics bracket tags | **Arrangement arc lives in Style field + per-section parenthetical cues in Lyrics** |
| Long prose inside `[Drop — kick only, reese bass cycles...]` works | **Dense bracket prose gets diluted or ignored.** Short modifiers + parenthetical on the next line work |
| Tag ordering doesn't matter | **Position 1 carries 60–70% of output DNA**; weight decays per position |
| More descriptors = more control | **4–7 descriptors is the sweet spot**; 8+ causes prompt fatigue |
| "Vocal Description" is a separate field | Suno has 2 fields (Style + Lyrics). Vocal identity **top-anchors Style**; per-section delivery lives as **parenthetical cues under each tag in Lyrics** |

---

## Core Philosophy

Suno v5.5 has **two real input fields**: Style and Lyrics. Treat them as **two layers of the same plan** (Structure Prompting):

| Field | Job | Hard limit | v6.1 target |
|---|---|---|---|
| **Style** | Sonic world + **explicit arrangement line** naming sections, bar counts, drop count + mirror anchor | 1000 chars | **250–450** (arrangement line pushes past minimalist targets — worth it) |
| **Lyrics** | Section skeleton via bracket tags with **bar-count suffixes** + short parenthetical cues. Tags repeated N times for N plays. | 5000 chars | **400–2000** for light-vocal tracks |

Both fields describe the **same arrangement** from two angles. The Style field speaks it as narrative prose (`drop 16 reese + choir, break 4, build 4, drop 16 identical + gamelan`). The Lyrics field speaks it as bracket architecture (`[Drop 16]`, `[Break 4]`, `[Build 4]`, `[Drop 16]`). Cross-referenced. This is the mechanism that forces Suno to follow your plan.

We keep a **Vocal Identity Anchor** as a mental sub-field:

- The identity itself (gender / timbre / processing) → prepended to the **top of the Style field** (first 80 chars).
- Per-section delivery cues (how the voice moves in each part) → **parenthetical under each Lyrics tag**.

Never write actual sung lyric lines for bass-music drops. Use short tag vocabulary (ad-libs, phonemes, vowels) so Suno doesn't try to build a full verse where you want a drop.

---

## The Golden Rule of v5.5

> **Position 1 of the Style field dominates.** The first token carries roughly 60–70% of output DNA; each subsequent token's influence drops by `2/(1+k)`. Put your most important descriptor (genre collision + bass character) **first**.

Everything else is secondary to this.

---

## Arrangement Force-Control Toolkit

This is the most important section in the skill. Suno 5.5's single biggest weakness is arrangement chaos — short drops that never come back, wildly inconsistent drop 1 vs drop 2, bare outros that invent new material, or 15-drop spam. These are not accidents. They are Suno's default behavior. You must **mechanically force** the arrangement.

### The four failure modes

| Symptom | Root cause | Fix (minimum) |
|---|---|---|
| Drop is 15–30s and never returns | Only one `[Drop]` tag in Lyrics; no arrangement line in Style | Write `[Drop]` **three times** in Lyrics; add explicit arrangement line in Style |
| Drop 2 wildly different from Drop 1 | No recall signal; matched tags alone insufficient | Same tag + identical parenthetical cue + `same drop, one layer added` phrase |
| 15 drops in a single song | `[Build]` appearing on every section, or only `[Drop]` tags and no `[Break]` | Use exactly **one** `[Build]` per drop; always follow `[Drop]` with `[Break]` before the next `[Build]` |
| `[Outro]` adds new verse / material | Bare `[Outro]` tag; no decay instruction | `[Outro 4]` + `(element-by-element decay, no new phrases)` |

### The Two-Layer Structure Prompting technique

This is the mechanism. It is the r/SunoAI "ultimate powerhouse combo" technique from 2025.

**Layer 1 — Lyrics field (the architecture / "what"):**
Bracket tags on their own lines. Each tag = one section. If you want a section to play N times, **write the tag N times**. Suno does not repeat sections you didn't tag.

**Layer 2 — Style field (the arrangement narrative / "how"):**
A final descriptor in the Style field that names the sections as **plain words** (no brackets in Style — brackets in Style glitch or are ignored). This is the line that fixes "drop only plays once":

```
arrangement: intro 4, verse 8, pre 4, drop 16 reese + choir, break 4, build 4, drop 16 same as drop 1 with gamelan added, outro 4 fade, two drops only
```

Both layers cross-reference the same plan. Suno reads the plan from two angles and is much more likely to comply.

### Required arrangement line in the Style field

Starting in v6.1, every bass-music track includes this line as the **final descriptor** of the Style field:

```
arrangement: [section X bars, section X bars, ..., Z drops only, outro fade]
```

Rules:
- Use **plain words** for section names (`intro`, `verse`, `pre`, `drop`, `break`, `build`, `bridge`, `outro`). No brackets.
- Include bar counts: `intro 4`, `drop 16`, `break 4`, `outro 4` (approximate — Suno treats as targets).
- **State the total drop count explicitly**: `two drops only` or `three drops total`. This is the most important single anti-spam phrase.
- Describe the drop 2 relationship to drop 1: `drop 2 identical to drop 1 plus one gamelan layer` — force the recall.

### Section repetition — the `N tags = N plays` rule

This rule is under-used and it is the single most reliable mechanic in Suno:

> **If you write `[Drop]` once, it plays once. Write it three times, it plays three times. Same for `[Chorus]`, `[Hook]`, `[Verse]`.**

For a two-drop EDM structure:

```
[Drop 16]
(bass hit, reese locked, choir on chord tones)

[Break 4]
(strip to sub + snare tail)

[Build 4]
(riser, kick drops out)

[Drop 16]
(same bass pattern, one gamelan layer added)
```

The **two `[Drop]` tags are mandatory**. Matched text under each = better recall. Separating `[Break]` then `[Build]` between them = prevents merge.

### Bar-count suffixes (approximate, not guaranteed)

Numeric suffixes are a documented v5/v5.5 feature. Suno treats them as targets, not guarantees, but they meaningfully bias section length:

```
[Intro 4]   [Verse 1 8]   [Pre 4]   [Drop 16]
[Break 4]   [Build 4]     [Drop 16] [Outro 4]
```

Use them always for `[Intro]`, `[Drop]`, `[Break]`, `[Outro]` — these are where Suno makes the worst length decisions. Verses and choruses are less critical.

### Mirror anchoring

A reddit-proven technique for locking core character: **repeat your core genre descriptor at the start AND end of the Style field**. Example:

```
neuro bass gospel, 94 BPM, sharp morphing reese bass, broken halftime drums, gospel organ, Bb minor i-iv-VI-V, warm dark wide, arrangement: intro 4, verse 8, pre 4, drop 16 reese + choir, break 4, build 4, drop 16 identical + gamelan, outro 4 fade, two drops only — neuro bass gospel
```

The closing `— neuro bass gospel` is mirror anchoring. This stabilises genre across the generation, especially in 5.5 which otherwise pop-washes by minute 2.

### `[Silence: 2s]` — pre-drop impact cut

Placing `[Silence: 2s]` (or `[Silence: 1s]`) on its own line **immediately before** a `[Drop]` tag forces Suno to cut to silence before the hit. This does two things:

1. Makes the drop feel bigger (dynamic contrast)
2. Signals to Suno that what follows is **the peak**, making it less likely to undercook it

```
[Build 4]
(riser builds, kick drops out)

[Silence: 2s]

[Drop 16]
(bass hit, choir stack)
```

Use on the first drop only; not both drops, or it becomes a cliché rhythmic device Suno echoes throughout.

### Restrict `[Build]` to one-per-drop

If `[Build]` appears everywhere in your Lyrics field, Suno treats every section as a mini-climb and gives you 8 small drops instead of 2 big ones. **One `[Build]` per drop, maximum.** Verses do their own internal tension — they don't need the `[Build]` tag.

### Always use `[Break]` between drops

Without an explicit cooldown section between drops, Suno merges them into one long drop or loses the second drop entirely. The minimum skeleton for a two-drop track:

```
[Drop 16] ... [Break 4] ... [Build 4] ... [Drop 16]
```

Never:

```
[Drop 16] ... [Drop 16]  ← these merge
```

### Frankenstein workflow (Song Editor Extend) — the surgical option

When generation-time force fails, switch to the Song Editor. This is the r/SunoAI "Frankenstein Method":

1. Generate the full song with the two-layer arrangement force.
2. Listen. Identify the **best drop** in the generation.
3. Open the song in Song Editor. Find the bar position of the end of the best drop.
4. Use **Extend** from that point. Style field = identical. Lyrics field = trimmed to just the continuation (`[Break 4] [Build 4] [Drop 16] [Outro 4]`).
5. The Extend **inherits the first drop's timbre, groove, and vocal identity** because it's literally a continuation of that audio.
6. This is the most reliable way to get a matched second drop. Worth 3 credits over 30 regenerations.

### Pre-generation failure-mode checklist

Run this before every generation:

- [ ] Style field contains an explicit `arrangement: ...` line as its final descriptor
- [ ] Arrangement line states **drop count explicitly** (`two drops only` or `three drops total`)
- [ ] Arrangement line specifies **drop 2 relationship to drop 1** (`identical plus one layer`)
- [ ] Style field ends with mirror-anchor of core genre (`— neuro bass gospel`)
- [ ] Lyrics field has `[Drop]` tag **written N times** where N = intended drop count
- [ ] Every `[Drop]` followed by `[Break]` (never two `[Drop]` adjacent)
- [ ] Exactly one `[Build]` per `[Drop]`
- [ ] Drop parentheticals are **identical text** (or identical + "one layer added")
- [ ] Bar-count suffixes on `[Intro]`, `[Drop]`, `[Break]`, `[Outro]`
- [ ] `[Outro]` has decay cue (`(element-by-element decay, bass drops out first)`)
- [ ] Ends with `[End][End][End]`

If any box is unchecked, you are giving Suno permission to default. It will.

---

## Style Field — Authoring Rules

### Structure (v6.1 template)

```
[vocal anchor if light-vocal], [genre collision], [BPM],
[bass character], [drum character], [texture 1–2],
[key + progression], [atmosphere word],
arrangement: [explicit section list with bar counts and drop count],
— [mirror anchor: repeat core genre]
```

### Hard rules

- **No artist names.** Ever. They either hallucinate or get rejected.
- **No more than 7–8 comma-separated descriptors before the arrangement line.** 9+ causes "prompt fatigue" — Suno dilutes or drops tags.
- **No filler words** (`and`, `with`, `featuring`). Commas only.
- **No brackets in the Style field.** `[Drop]` inside Style glitches or is ignored. Use plain words: `drop`, `break`, `build`.
- **Put the genre collision at position 1.** If vocals are non-negotiable, put the vocal anchor at position 1 instead (e.g. `male warm tenor, neuro bass soul, 92 BPM, ...`).
- **Arrangement line is mandatory** (v6.1). Always include `arrangement: intro X, verse X, ..., N drops only, outro X fade` as a single comma-separated clause near the end.
- **Mirror anchor is mandatory** (v6.1). End the Style field with `— [core genre repeated]`. This stabilises character across the generation.
- **Target 250–450 chars.** The arrangement line pushes us past the 300-char "minimalist" target. That's fine — the clarity trade-off is worth it for arrangement control.

### v5.5-specific counters

v5.5 has a **"pop-wash" tendency** — it smooths niche genres into generic pop. Counter with:

- An **era anchor**: `90s UK garage`, `2012 brostep`, `2001 electroclash`
- A **raw-modifier**: `raw`, `unpolished`, `lo-fi tape warmth`, `dusty`
- A **specific sound-design anchor**: `reese bass`, `gamelan percussion`, `sidechained 808 sub` — Suno recognises these concretely

### Example Style fields (v6.1 format with arrangement line + mirror anchor)

Neuro Gospel, 94 BPM:
```
neuro bass gospel, 94 BPM, sharp morphing reese bass, halftime broken drums, gospel organ swells, Bb minor i-iv-VI-V, warm dark wide, arrangement: intro 4, verse 8, pre 4, drop 16 reese + choir stack, break 4, build 4, drop 16 identical plus one gamelan layer, outro 4 fade, two drops only — neuro bass gospel
```

Bass UK Garage Soul, 130 BPM:
```
male warm tenor ad-libs, bass UK garage soul, 130 BPM, warm reese sub, 2-step skippy shuffle, Rhodes keys, tape saturation, C minor i-VI-III-VII, dusty intimate warm wide, arrangement: intro 4, verse 8, pre 4, drop 16 bass hit + choir stack, break 4, build 4, drop 16 identical plus one gamelan layer, outro 4 fade, two drops only — bass UK garage soul
```

Both sit around 380–420 chars. Longer than v6.0's 150–300 target — and that's by design. The arrangement line is worth the extra length.

---

## Lyrics Field — Authoring Rules

The Lyrics field carries the **structure skeleton** via bracket tags. Arrangement and vocal detail for each section lives as a **short parenthetical on the line immediately under the tag**. This is the pattern the bitwize-music v5 reference confirms and that r/SunoAI pro users rely on for genre-switching tracks.

### Format

```
[Tag]
(short parenthetical cue — 6–12 words describing what enters, what the vocal does, what the bass does)
```

Not:

```
[Tag — long paragraph with multiple clauses of sonic description nobody will read...]
```

### Bracket tag vocabulary (v5.5 verified)

Structure:
`[Intro]` `[Verse 1]` `[Pre-Chorus]` `[Chorus]` `[Verse 2]` `[Bridge]` `[Final Chorus]` `[Outro]` `[End]`

Energy / electronic:
`[Build]` `[Build-Up]` `[Drop]` `[Break]` `[Breakdown]` `[Beat switch]` `[Drop-Out]`

Modifier / delivery (inline, 1–3 words only):
`[Instrumental]` `[Whisper]` `[Belted]` `[Choir: Gospel]` `[Harmony: High]` `[Vocal Ad-libs]` `[Callback: Chorus melody]`

Bar-count targeting (v5-only, approximate — Suno treats as targets, not guarantees):
`[INTRO 4]` `[VERSE 1 8]` `[PRE 4]` `[CHORUS 8]` `[DROP 16]` `[OUTRO 4]`

Ending (triple-tag is strongest stop signal observed in community testing):
`[End][End][End]`

### Tag rules

- **One tag per line.** Don't bury tags in prose.
- **N tags = N plays** (v6.1, critical). Want two drops? Write `[Drop]` twice. Want three choruses? Write `[Chorus]` three times. Suno does not repeat sections you didn't tag.
- **Always use bar-count suffixes** on `[Intro]`, `[Drop]`, `[Break]`, `[Outro]` (v6.1). These are the sections where Suno makes the worst length decisions.
- **Never place two `[Drop]` tags adjacent.** Always `[Drop] [Break] [Build] [Drop]`. Adjacent drops merge.
- **One `[Build]` per `[Drop]` maximum.** More than that and Suno gives you drop spam.
- **Parenthetical cues are 6–12 words.** If a cue starts reading like a sentence, trim it.
- **Drop 2 parenthetical matches Drop 1** + `"plus one [layer name] added"` as the only change. Identical text is the recall anchor.
- **Never skip `[Pre-Chorus]` / `[Pre 4]`** — it's the reliable tension mechanism before choruses and drops.
- **Never leave `[Outro]` bare**. Always `[Outro 4]` + `(element-by-element decay, bass drops out first)`.
- **End with `[End][End][End]`** — triple-tag is the strongest stop signal.
- **For genre-switch / beat-switch tracks**, put the target genre in a parenthetical on the next line: `[Beat switch]\n(double-time riddim, reese bass enters)`.

### Drop repeat — three verified methods

Suno has no native "repeat drop with variation" command. These three patterns work:

**Method A — Matched tag + modifier parenthetical** (fastest, single generation):

```
[Drop]
(bass hit, 2-step locked, choir stack on chord tones)

[Break]
(strip to sub + snare tail)

[Drop]
(same bass pattern, add gamelan layer on the second bar)
```

Tag names identical. Parenthetical on the second drop says explicitly "same bass pattern, add one layer". Community testing confirms matched tag names naturally produce close variations.

**Method B — Remix/Extend from the first drop** (most reliable, requires a second pass):

1. Generate the track.
2. Find the first drop in the Song Editor.
3. Use Extend from that point; feed the same Style + trimmed Lyrics (second drop + outro).
4. The extend inherits the first drop's timbre and groove.

**Method C — Upload first drop as audio reference** (when matching identity across sessions):

1. Export the first drop as a short clip.
2. Start a new generation with Audio Upload enabled.
3. Set **Audio Influence ~ 25%** — low enough to let the prompt lead, high enough to inherit bass timbre.
4. Style field identical to the original. Lyrics = second drop only.

### Full-song Lyrics template (v6.1 — two-drop force-control)

```
[Intro 4]
(filtered sub pad, riser starts, no drums)

[Verse 1 8]
(halftime drums enter, sparse vocal ad-libs, muted reese)

[Pre 4]
(white noise build, vocal chop stutter, kick drops out)

[Silence: 1s]

[Drop 16]
(bass hit, 2-step locked, choir-voiced harmonics on chord tones)

[Break 4]
(strip to sub + snare tail, one held vocal vowel)

[Build 4]
(riser builds, kick drops out)

[Drop 16]
(same bass pattern, gamelan layer added on second bar)

[Outro 4]
(element-by-element decay, bass drops out first, pad tail)

[End][End][End]
```

Note what changed from v6.0:

- Bar-count suffixes on every section
- `[Silence: 1s]` before the first drop for impact cut
- `[Build 4]` inserted between `[Break]` and the second `[Drop]` — never let drops be adjacent
- No `[Bridge]` or `[Final Chorus]` for light-vocal bass tracks — they dilute the arrangement. Add them only if the Style field specifically calls for them.
- Second `[Drop 16]` uses **identical parenthetical text** to the first, then `+ gamelan layer added`. This is the recall anchor.

---

## Vocal Handling — Light Vocals Preset

The user's preferred territory is **vocal-present but not vocal-led**: ad-libs, vowels, choir harmonics on drops, no full verse lines.

### Style field — vocal identity anchor (first 60–80 chars)

Pick one and front-load it:

```
male warm tenor ad-libs only, neuro bass gospel, 94 BPM, ...
female soul choir vowels only, bass UK garage, 130 BPM, ...
gender-neutral formant-shifted vocal chops, glitchhop bossa, 96 BPM, ...
```

### Lyrics field — per-section parentheticals

| Section | Cue pattern |
|---|---|
| Intro | `(no vocals, atmospheric pad only)` |
| Verse 1 | `(sparse ad-libs only — "yeah", "ooh", no full lines)` |
| Pre-Chorus | `(one held vowel rising in pitch, breath becomes tight)` |
| Chorus | `(choir-voiced harmonics on chord tones, each tone separate layer)` |
| Drop | `(pitched vocal chops cycling progression, no full phrases)` |
| Break | `(one exposed held vowel, no production)` |
| Bridge | `(single whispered ad-lib, unprocessed)` |
| Outro | `(vocal tail fades first, no new phrases)` |

### Vocal rules

- **Never write actual sung lyric lines** — if they appear in the Lyrics field, Suno will sing them. For light-vocal tracks, stick to parentheticals.
- Use `choir-voiced harmonics` — never `vocal chords` (Suno reads that as anatomy).
- Each chord tone **voiced by a separate layer** — not `stacked harmonies` (which tends to collapse into unison).
- For full-throat songs, use `[Harmony: High]`, `[Belted]`, `[Choir: Gospel]` as inline modifiers — 1–3 words only.

### Personas (Pro / Premier)

For vocal consistency across an album:

1. Generate one song with a voice you like.
2. Save the song as a Persona.
3. Apply the Persona to new generations.
4. **Simplify Style to 1–2 genres when a Persona is active** — the Persona overrides conflicting vocal descriptors.

---

## Style Influence DNA — Wordized Blocks

These are battle-tested descriptor blocks for the Style field. Never use artist names. Combine 2 blocks max; more dilutes.

### Sharp Heavy Bass
```
sharp edgy morphing reese bass, crushing sidechain,
deep sub underneath distorted mid bass, lush chord pads vs aggressive bass,
slow patient builds, warm cold contrast, cinematic weight
```

### Technical Melodic Chop
```
glass pitch-shifted synth, stutter micro-edits on specific beats,
melodic synth chops, technical offbeat groove, bright lead vs dark bass,
wide bright stereo, precise rhythmic editing
```

### Organic Psychedelic Warmth
```
organic psychedelic pad, warm breathing atmosphere,
bowed string textures dissolving into bass, patient slow builds,
oscillating pads, natural organic percussion
```

### Cinematic Emotional Melody
```
cinematic chord progressions, warm electronic atmosphere,
emotional melodic lead, sophisticated arrangement arc,
lush wide chord pads, tender emotional contrast, warm vintage electronics
```

### Combination template
```
[Block A first 2 phrases], [Block B first 2 phrases], [one atmosphere word]
```
Example:
```
sharp morphing reese bass, crushing sidechain, glass pitch-shifted synth, stutter micro-edits, dark warm wide
```

---

## Genre Collisions (Verified Strong)

Bass music is the foundation; the collision is the character.

- Bass + Soul / Gospel
- Bass + Bossa Nova / Jazz
- Bass + Flamenco / Phrygian
- Bass + 1930s Swing / 1960s Twist
- Bass + Orchestral / Cinematic
- Bass + Tribal / Gamelan
- Bass + Psytrance
- Bass + Complextro / Electro House
- Bass + Halftime DnB
- Bass + Glitchhop
- Bass + UK Garage / 2-Step
- Bass + Drill / Plugg

### Fusion phrasing (contribution, not mashup)

Instead of `neuro bass + gospel`, write what each genre contributes:

```
neuro bass drums and reese from dubstep,
choir harmonies and organ swells from gospel
```

This is the jackrighteous "contribution formula" — much stronger than listing two genre names.

---

## Chord Progressions

State key center in Style field, reference progression in parenthetical cues.

Short (4 chords, fits in 1-bar cycle):
```
i-VI-III-VII in A minor
```

Long (8 chords, cycles twice in drop):
```
i-VI-III-VII-v-iv-VI-VII in A minor
```

Jazz voicings (write chord symbols explicitly):
```
Fmaj7-Am7-Dm7-Bbmaj7-Gm7-C7 in F major
```

In the Lyrics field, reference the progression in the drop parenthetical:
```
[Drop]
(bass hit, choir stack voices i-VI-III-VII in A minor)
```

---

## Vocal Processing — Signature Effects

| Processing | Sonic Character | Best Used For |
|---|---|---|
| Underwater reverb + pitch-bend down on phrase endings | Sinking, dissolving | Loss, depth |
| Vinyl crackle + formant aging | Found recording, worn memory | Nostalgia |
| Half-BPM tremolo on sustained notes | Held breath, tension | Cinematic |
| Reverse reverb pre-delay (ghost arrives before voice) | Haunted, dissolving self | Psychedelic |
| Surprise octave harmony on last word only | Bittersweet, tender | Emotional |
| Telephone bandpass + spring reverb | Distant, circuit-worn | Genre collision |
| Formant shift up 2 octaves | Strange, almost childlike | Experimental |
| Stutter micro-edit on specific syllables | Technical, precise | Technical bass |
| Glass pitch shift on vowels only | Crystal, cold, precise | Technical melodic |
| Tape wobble + slight pitch instability | Imperfect, human, warm | Soul, organic |

Usage in Style field:
```
..., tape wobble vocal, warm dusty ...
```
Usage in Lyrics parenthetical:
```
[Bridge]
(single vowel, tape wobble, unprocessed otherwise)
```

---

## Token Biases — Lyrics-Lock

Suno 5.5 has observed default substitutions. If you don't prevent it, it will sneak these into your lyrics:

**Neon, Echo, Ghost, Silver, Shadow, Whisper, Crystal, Velvet, Midnight, Fire**

If you write actual lyric lines and want them sung verbatim, add this as the **first line of the Lyrics field** (above the intro tag):

```
Do not change any words. Sing exactly as written.
[Intro]
...
```

For our light-vocal instrumental-leaning tracks, this is usually unnecessary since we only use parenthetical cues.

---

## Creative Sliders (Suno UI)

| Slider | Low | Mid | High |
|---|---|---|---|
| **Weirdness** | predictable, hooky | balanced | experimental, unexpected |
| **Style Influence** | looser fusion, more Suno defaults | moderate | strict adherence to Style field |
| **Audio Influence** (uploads only) | loose reference | hybrid | strong clone of uploaded clip |

### Recommended defaults for bass music

- **Weirdness:** mid-high for genre collisions (forces unusual combinations to actually collide instead of averaging)
- **Style Influence:** mid-high if your Style field is dialed in; drop to mid if Suno is generating artifacts
- **Audio Influence:** ~25% when using Method C drop repeat

### Magic Wand warning

Turn Magic Wand **off** when pasting a precise Style field. In v5.5 it auto-rewrites your prompt and mixes in styles from previous songs, which silently breaks reproducibility.

---

## Bar-Count Targeting (optional)

For finer structural control, append bar numbers to structure tags in the Lyrics field:

```
[INTRO 4]
[VERSE 1 8]
[PRE 4]
[CHORUS 8]
[VERSE 2 8]
[DROP 16]
[BREAK 4]
[DROP 16]
[OUTRO 4]
[End][End][End]
```

Suno treats these as targets, not guarantees. Use them when you need a predictable intro/outro length for DJ mixing.

---

## Error Prevention

| Error | Cause | Fix |
|---|---|---|
| Song too short (1–2 min) | Not enough sections | Add Build, Break, Bridge — 9–12 tags minimum |
| Song too long (6–7 min) | Too many sections | Drop Bridge, trim to 8 tags, add explicit `[Outro]` |
| Drop cuts early | Over-stuffed bracket prose | Move detail to Style field, tag = `[Drop]`, cue = 1 parenthetical line |
| Second drop entirely different | No recall mechanism | Use matched tag name + `(same bass pattern, add one layer)` cue, or Extend from first drop |
| Outro adds new material | Bare `[Outro]` | Add cue: `(element-by-element decay, bass drops out first)` |
| Weak vocal harmonics | Wrong phrasing | Use `choir-voiced harmonics, each tone separate layer` |
| Style field rejected | Artist name or over 1000 chars | Remove artist names, trim to 150–300 |
| Vocals too repetitive | Same delivery every section | Vary per-section parentheticals; alternate "sparse ad-libs" / "one held vowel" / "pitched chops" |
| Drop has unwanted new verse | No vocal role in drop cue | Add `(pitched vocal chops only, no full phrases)` in drop parenthetical |
| Tags ignored entirely | Prose inside brackets | Strip bracket to tag name; move cue to parenthetical line below |
| Pop-wash on niche genre (5.5) | Model smoothing | Add "raw", "unpolished", era anchor to Style |
| Style field changed without consent | Magic Wand on | Turn off Magic Wand before generating |
| Token substitutions in lyrics | Suno bias defaults | Prepend `Do not change any words. Sing exactly as written.` |
| Same groove on every generation (5.5 bug) | Model default attractor | Raise Weirdness, change position-1 token, retry |

---

## Migration Rules (v4.5 → v5 / v5.5)

From Suno's own community consensus — your old prompts will fail:

1. **Cut the Style field by 30–50%.** v5 and v5.5 want less, not more.
2. **Move arrangement instructions out of bracket prose into parenthetical cues** under each tag.
3. **Add a numeric BPM** if it's missing — it anchors everything.
4. **Test with a 1:30 generation** before committing to a full 4-minute song.
5. **If pop-washed, re-add era + raw anchors** (v5.5 specific).

---

## Quick Reference Card (v6.1)

```
STYLE FIELD (250–450 chars, arrangement line mandatory)
────────────────────────────────────────────────────────
[vocal anchor], [genre collision], [BPM],
[bass character], [drum character], [texture],
[key + progression], [atmosphere],
arrangement: intro 4, verse 8, pre 4,
             drop 16 [character], break 4, build 4,
             drop 16 identical + [one added layer],
             outro 4 fade, two drops only
— [mirror anchor: repeat core genre]


LYRICS FIELD (bracket tags with bar counts + parentheticals)
─────────────────────────────────────────────────────────────
[Intro 4]
(cue)

[Verse 1 8]
(cue)

[Pre 4]
(cue)

[Silence: 1s]

[Drop 16]
(bass hit, locked, choir on chord tones)

[Break 4]
(strip to sub + snare tail)

[Build 4]
(riser builds, kick drops out)

[Drop 16]
(same bass pattern + one gamelan layer added)

[Outro 4]
(element-by-element decay, bass drops out first)

[End][End][End]


THE N TAGS = N PLAYS RULE
──────────────────────────
Want 2 drops? Write [Drop] twice. Never once.
Want 3 choruses? Write [Chorus] three times.
Suno does not repeat sections you didn't tag.


DROP REPEAT — PICK ONE
──────────────────────
A. Matched tag + identical parenthetical + "+ one layer"
B. Song Editor Extend from first drop (most reliable)
C. Audio Upload of first drop at ~25% Audio Influence


ANTI-PATTERNS (never do these)
───────────────────────────────
✗ Two [Drop] tags adjacent (merge into one)
✗ [Build] on every section (produces drop spam)
✗ Bare [Outro] (invents new material)
✗ Brackets in Style field (glitches or ignored)
✗ Prose paragraphs inside [Tag — long descriptions]
✗ No arrangement line in Style field


SLIDERS
───────
Weirdness: mid-high    Style Influence: mid-high
Audio Influence: ~25% (Method C only)
Magic Wand: OFF (silently rewrites your Style field)
```

---

*Suno Prompting Skill v6.1 — grounded in Suno 5.5 reality (2026), with dedicated arrangement force-control toolkit. Update when Suno releases v6 or the WMG-licensed models roll out.*
