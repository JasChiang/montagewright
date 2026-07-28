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
612 passed, 1 warning
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

## Paid Samsung benchmark readiness

No new paid Samsung request was made in this implementation run.

The paid benchmark preflight found:

- `.env` declares `GEMINI_API_KEY`;
- the historical Samsung outputs, Clip Card/plan artifacts, render manifests, source quality maps, final muxes, and SAM checkpoint are present;
- the historical runs do **not** contain V1 `editorial-beat-contracts.json`, `cue-plan.json`, `exact-event-locks.json`, policy-bound `reuse-degradation.json`, or `deterministic-delivery-evidence.json`.

Those artifacts are delivery evidence, not optional metadata. Synthesizing them from a playable historical MP4 would falsely upgrade review evidence into exact autonomous evidence. The pipeline therefore blocks before paid work instead of spending a final-QA call on an ineligible run.

Consequently there is no truthful V1 paid-call/token/cost/runtime/final-QA report yet. Historical comparison baselines remain:

| Artifact | Paid interactions | Input | Cached input | Output | Thought | Estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `samsung-galaxy-z-simple-production-v2-01` | 127 | 598,036 | 21,608 | 104,306 | 14,640 | US$1.75997820 |
| `samsung-galaxy-z-delivery-v11` | 56 | 297,528 | 51,399 | 71,896 | 215 | US$0.91773585 |
| `samsung-galaxy-z-attention-camera-live-03` | 23 | 128,034 | 0 | 36,080 | 0 | US$0.46265100 |

## Remaining blocking integration

The new event, presentation, optimizer, cache, QA, recovery, and authority contracts are implemented and tested, but the historical feature-cut orchestration does not yet emit the six required autonomous benchmark inputs as one live run. Until that wiring exists, a paid end-to-end Samsung V1 claim would be misleading.

The next implementation step is therefore narrow: make the selected-window stage persist the grouped ExactEventLocks, compile the beat/cue/degradation/deterministic evidence bundle from those exact artifacts, pass it directly to `feature-delivery`, and then perform one 9:16 paid benchmark under the policy ledger. It must not introduce another semantic planner or revive selected-clip framing refinement.
