# jascue_auto

Rushes in, a cut out. Gemini decides what the film is; local code decides what
frame that lands on.

```
jascue-auto render RUSHES/ --brief BRIEF.md --music TRACK.mp3 \
  --music-map MAP.lock.json --aspect 9:16 --output CUT/
```

## The division

| Question | Answered by |
| --- | --- |
| Is this take usable, and what is in it | Gemini, watching each clip once |
| What should this material become | Gemini, watching all of it with the music |
| Which shots, in what order, and why each | Gemini, watching the usable ones |
| How long each shot holds, and whether it cuts on the beat | Gemini, hearing the track |
| Which camera move, and where the subject sits in frame | Gemini |
| Where the beat actually falls | Local onset analysis |
| Where the subject actually is, frame by frame | SAM propagation from a Gemini seed |
| How far a crop may travel, and what it cost | Local arithmetic |

Semantic questions go to the model. Measurements stay local. The rule is not
about trust — it is that each side can only answer one kind of question, and
the failures come from asking the wrong side.

## Stages

```
clip cards   once per asset, cached by content hash, brief-free on purpose
direction    all the footage + the music + the brief
selection    the usable footage + the direction
rhythm       the track + each shot's neighbours
subject      the card's box, or a grounding call when the card cannot answer
reframe      a crop path inside the shot's energy budget
render       segments, concat, music, preview
review       the cut, the brief, and the direction it set for itself
```

The clip card deliberately does not see the brief. A card written against one
brief has to be rewritten for the next deliverable from the same shoot; a card
written against the material is good until the material changes. That is what
makes it worth caching, and caching it is what makes the subject boxes free on
every rerun.

## Things that were true and cost something to learn

**Prose asks; structure binds.** Three separate fixes here came down to the
same thing. The model answers subject boxes in its native 0..1000 space
however firmly a field description says otherwise, so the conversion happens
on receipt. A conditionally required field cannot be marked required, so it
travels as a required string with a sentinel. Asking for one short clause
inside a repeated item returned a 7453-character note and overran two output
ceilings, so the reasoning moved to the top level where it is written once.

**A constraint that lives only in a validator is a bill you pay later.** The
previous system had five: a classified motion role needing a reason, an atomic
region not also declaring a visible fraction, and three more. Each one cost a
full paid plan to discover, because the model had no way to know. Every
coupling in `schema.py` is written where the model reads it.

**Fields get added where they are used, not where they are decided.** Four
times: `pan_hint`, `composition`, `subjects`, `min_visible`. Each was defined,
consumed downstream, and unreachable from upstream. Nothing fails when this
happens — the default is plausible, the render succeeds, and the decision has
quietly moved from the planner to a constant.

**Numbers in the execution layer are usually taste in disguise.**
`MIN_DIRECTNESS` turned a planned follow into a hold without saying so.
`MAX_UPSCALE` answered "is this shot worth softening" on the edit's behalf.
Both read as measurements. The shape that works is: local code offers the
options and what each costs, the planner picks one, and any substitution is
recorded with the measurement that forced it.

**"Not too fast" and "legible" are different tests.** A sweep across three
handsets in 1.5 seconds is inside every speed limit and still arrives before
anyone has looked at the second one. Moves declare the seconds they need and
rhythm treats that as a floor.

**Watch the render.** Every composition bug here was found by pulling frames
and looking: a pan that overshot both subjects into empty background, a zoom
aimed at the middle of the frame instead of the subject, a wordmark cropped to
"y Unpacked". Every one of them reported travel, static and aligned-cut
figures that looked correct.

## Reading a run

`report.json` carries the direction, the selection with its reasons, which
cuts landed on music, every degradation with the measurement that forced it,
the enlargement applied per shot, and the spend per stage. Sharpness is
reported as a number rather than left for a reviewer to judge from a preview,
where a soft proxy and an over-enlarged shot look identical.
