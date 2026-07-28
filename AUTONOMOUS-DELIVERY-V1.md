# Autonomous Delivery V1 implementation and benchmark record

## Scope

This branch extends the existing feature-delivery chain; it does not add a free-form agent or a second render engine. V1 is limited to 30–120 second music-led product, feature, event montage, and visual UI demonstrations with non-narrative source audio.

The application-owned execution boundary is:

```text
Gemini semantic observations
  → immutable event/grounding evidence
  → local presentation and sequence compilers
  → segment-addressed FFmpeg render
  → deterministic delivery QA
  → observational final semantic QA
  → hash-bound AUTO_POLICY authority
```

## Implemented milestones

| Milestone | Implementation |
| --- | --- |
| M0 | Baseline test and historical artifact usage audit. |
| M1 | `AutonomousEditPolicy`, `DecisionAuthorityV2`, `BudgetLedger`, low/high media-resolution contracts, and text-only schema repair. |
| M2 | `EditorialBeatContract`, `ExactEventLockV2`, grouped frame-ID event selection, cue delta evidence, AUTO_POLICY MusicMap/CuePlan/TrimIntent. |
| M3 | Extracted existing vertical fit/crop/reversal logic into `presentation.py`; added one-call multi-target grounding, shared SAM seeds, two-panel, promoted solid fit, exact-event freeze, and source-motion suppression. |
| M4 | Evidence-first bounded beam search, duration reconciliation, continuity penalties, and content-addressed segment render cache. |
| M5 | Audible `autonomous_final_9x16`, typed QA issues, deterministic gate report, local repair mapping, one-replan/two-QA caps, and final `AUTO_POLICY` authority. |

Legacy `review_preview` and `production_review` continue to normalize final QA to human review and never receive autonomous delivery authority.

## Contract and cost boundaries

- Policy definition, every automatic decision, degradation manifest, and final authority are SHA-256 bound.
- Schema failure does not resend full media. ContentMap and direct-video plan repair use raw text plus schema only.
- General video uses low media resolution; exact event and bbox images use high.
- Direct-video-edit-plan-v2 does not trigger the legacy selected-clip framing rewatch.
- Two-panel orientation, rects, scale lock, gutter, and same-PTS checks are local and add zero paid calls.
- Hard evidence, identity, action completeness, required relation, scale, quality-safe interval, cue tolerance, reuse authority, and executable geometry are non-compensable constraints.
- Final semantic QA can observe an issue but cannot approve delivery. Only a fully passed deterministic report plus non-blocking requested-aspect QA can create `DecisionAuthorityV2`.

## Local verification

The final local suite completed with:

```text
626 passed, 1 warning
```

The warning is the existing Starlette/httpx deprecation warning and is unrelated to autonomous delivery.

Acceptance fixtures cover:

- tracker deadband and zero synthetic drift;
- minimal monotonic camera motion and no semantic reordering;
- source-pan suppression and reversal-to-hard-cut behavior;
- same-source/same-PTS two-panel and relative scale lock;
- no arbitrary three-panel layout;
- grouped exact event selection using existing frame IDs only;
- Samsung gesture/result/UI/underwater/laugh-freeze cue contracts;
- hard evidence cannot be offset by a high aesthetic score;
- optional omission and readability-based duration reconciliation;
- changed-segment-only render invalidation;
- audible final 9:16 QA with brief and all autonomous context;
- budget rejection before final-QA dispatch;
- deterministic cue delta failure;
- at most one semantic replan and two full final QA passes;
- strict delivery authority without a human approval artifact;
- best-effort degradation disclosure.

## Archived Samsung 9:16 evidence experiment

The first selected-window V1 experiment explicitly reused the historical
`direct-video-edit-plan-v2`, quality, geometry, and segment artifacts, then
created the missing exact-event evidence in the live picture run. It was useful
for validating grouped ExactEventLocks, but it is not a valid autonomous editing
benchmark and must not be presented as a new edit.

| Result | Value |
| --- | --- |
| Archived output | `artifacts/_archive/2026-07-28-stale-plan-reuse/samsung-galaxy-z-autonomous-v1-benchmark-01/picture/renders/feature-cut-9x16-clean.mp4` |
| Media | 82.624 s, 1080×1920, 30 fps, H.264 + 48 kHz stereo AAC |
| Paid interactions | 23 |
| Input / cached input | 124,058 / 0 tokens |
| Output / thought | 2,821 / 9,340 tokens |
| Estimated cost | US$0.27729450 |
| Picture runtime | 1,007.828 s |
| Technical QC | passed |
| Exact locks | 6 locks across 5 grouped selected windows |
| Presentation | 6 solid fits, 4 tracked crops, 2 review center crops, 0 panels |
| Audio | source audio only; music mux was never reached |
| Final semantic QA | not run; deterministic hard gate stopped first |
| Delivery state | blocked |

The evidence experiment remained below the 25-interaction warm-run target. It correctly
failed `autonomous_strict` cue sync: the AI result stable frame was 29 frames
from its principal downbeat, while the closing reaction and freeze were 38 and
41 frames from the phrase ending. Gesture, watch UI, and underwater apex were
within 0, 0, and 4 frames of their contract-specific cues. No tolerance was
relaxed and no second paid trial was made. Because the picture plan was reused
and no music delivery was assembled, these figures cannot be used to claim a
fresh-plan autonomous benchmark.

The run also exposed and fixed a local gate bug: an authorized
`solid_matte_fit_used` degradation was incorrectly classified as an editorial
omission. Presentation fallbacks are now independent from optional/preferred
beat omission authority, with regression tests.

The contaminated benchmark and its historical source run were moved to
`artifacts/_archive/2026-07-28-stale-plan-reuse/`. Autonomous profiles now
reject both editorial plan reuse flags before paid work. The Samsung strict
policy also forbids solid matte fit; an all-full-bleed request must switch
Top-K candidates or block.

Historical comparison baselines remain:

| Artifact | Paid interactions | Input | Cached input | Output | Thought | Estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `samsung-galaxy-z-simple-production-v2-01` | 127 | 598,036 | 21,608 | 104,306 | 14,640 | US$1.75997820 |
| `samsung-galaxy-z-delivery-v11` | 56 | 297,528 | 51,399 | 71,896 | 215 | US$0.91773585 |
| `samsung-galaxy-z-attention-camera-live-03` | 23 | 128,034 | 0 | 36,080 | 0 | US$0.46265100 |

## Selected-window integration

The historical feature-cut orchestration now emits the six autonomous inputs in the same picture run:

1. beat templates are mapped to feature IDs and rebound to the selected candidate's real `EvidenceQueryLockV2`;
2. the final immutable source in/out is decoded through the existing dense-frame path;
3. all events for one selected feature are resolved in one grouped exact-frame request;
4. local PTS mapping, cue delta, degradation, and deterministic evidence are persisted with hashes;
5. `feature-delivery` discovers the generated paths without a manually assembled context directory;
6. picture resume validates every context hash and the bundle index before reuse.

`autonomous-evidence-bundle.json` indexes all six artifacts. Hard beats without a selected query lock or exact event fail closed. Preferred/optional omissions require policy authority and are written to the degradation manifest.

The next valid benchmark must use a fresh output directory, generate a fresh
music-aware editorial plan, forbid solid matte fit, assemble the music-only
audition mux before it is shown to a user, and then apply cue-aware
trim/duration reconciliation before framing and render.
