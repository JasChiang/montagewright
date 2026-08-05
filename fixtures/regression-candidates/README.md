# Regression candidates

Captured failures from the `codex/autonomous-delivery-v1` tip-health run on the
Samsung Galaxy Z material, 2026-08-03. These are **candidates**, not golden
cases: per the work order's §11.1 governance rule, promotion is a human
decision, the set is capped at 20, and adding one retires another.

Each entry records what the semantic layer decided and what the execution
layer did with that decision, because the gap between those two is what the
rebuild exists to close.

## grounding-ambiguity

`grounding.json` — chapter `fold8_camera`, 16:9, source `C8351.MP4`.

Gemini was asked to ground "the vertically arranged rear camera module". The
frame holds two handsets, so it returned two boxes at equal confidence and
labelled each one:

- `Fold8 鏡頭模組 (左側灰色手機)` — 左側灰色手機背面的三鏡頭模組外框
- `Fold8 鏡頭模組 (右側深色手機)` — 右側深色手機背面的三鏡頭模組外框

`match_status: ambiguous`, and the whole aspect was abandoned:
`all 16:9 candidates failed quality, capacity, geometry or lineage preflight`.

**Why this is worth keeping.** Nothing was missing. The model produced both
options *and a disambiguation_reason for each*, which is exactly the input a
"which one?" question needs. The pipeline had no step that could ask, so
ambiguity fell through to abandonment.

This is the case §4.5's recognition chain is designed for — detect, overlay
numbered boxes, put the choice back to Gemini as a multiple-choice question,
then track. The assertion to write against it is behavioural, not literal
(§11.1 rule 2): *this segment produces a reframe*, not *it picks the left
handset*.

Two earlier attempts to dodge it by rewriting the brief's target description
both failed, and the second one is instructive: clarifying "treat the lens
cluster as one region" answered a question nobody was asking. The ambiguity
was never *what* the target is, it was *which handset*. A prompt cannot
resolve a choice the material genuinely leaves open; only asking can.
