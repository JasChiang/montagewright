"""Data contracts for the autonomous edit pipeline.

Two rules shape everything here, and both come from what the previous system
got wrong rather than from taste.

Coordinates are normalised floats in 0..1 everywhere inside the pipeline. The
old code carried pixel speeds and viewport fractions side by side, which made
a limit meaningful only next to the resolution it was written against. Pixels
appear once, at the ffmpeg boundary.

A constraint the local layer will reject has to be stated in the contract the
model reads. Five separate rules in the previous system lived only inside
validators -- a classified motion role needing a reason, an atomic region not
also declaring a visible fraction, and three more -- and each one cost a full
paid plan to discover. Every coupling below is written where the model sees
it, and every closed vocabulary carries an `other` escape so an unforeseen
answer degrades to a described value instead of a rejected response.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 0..1 across the frame, origin top-left. Never pixels.
Normalised = Annotated[float, Field(ge=0.0, le=1.0)]

CameraEnergy = Literal["calm", "active", "dynamic"]
CoverageType = Literal["literal", "implied", "tonal"]
EnergyIntent = Literal["low", "medium", "high"]


class ModelFacing(BaseModel):
    """Base for anything the model fills in.

    Unknown keys are kept rather than rejected: an extra field is the model
    volunteering something, and throwing away a paid response over it buys
    nothing. Callers log what they ignored.
    """

    model_config = ConfigDict(extra="allow")


class Local(BaseModel):
    """Base for anything only local code writes."""

    model_config = ConfigDict(extra="forbid")


class NamedFact(ModelFacing):
    """One product name, model number, year, or on-screen string, as seen.

    Transcribed before planning and quoted by everything downstream. The point
    is that a 2026 handset is newer than the model's training data, so its own
    memory of what a model number "should" be is the least reliable source in
    the room.
    """

    text: str = Field(
        description=(
            "The string exactly as it appears, including spacing and any "
            "character that looks like a typo. Never corrected, normalised, "
            "or completed."
        )
    )
    source_id: str = Field(description="Asset the string was read from.")
    at_seconds: float = Field(
        ge=0.0, description="Roughly when it is legible in that asset."
    )
    legibility: Literal["clear", "partial", "uncertain"] = Field(
        description=(
            "How readable it actually was. 'uncertain' is the honest answer "
            "for a guess and costs nothing; a confident wrong string is what "
            "downstream copy cannot recover from."
        )
    )


class MaterialNote(ModelFacing):
    """A judgement about one asset that only watching it can produce."""

    source_id: str
    usable: bool = Field(
        description=(
            "False only for footage that cannot serve any purpose -- a failed "
            "take, a duplicate attempt superseded by a better one. Camera "
            "shake, soft focus, and unusual framing are style choices "
            "available to the edit, not defects; say so in `note` and leave "
            "this true."
        )
    )
    note: str = Field(
        description="What is here and what it is good for, in one or two lines."
    )
    supersedes: list[str] = Field(
        default_factory=list,
        description=(
            "Other source_ids this take beats, when several attempts at the "
            "same action exist."
        ),
    )


class StyleDecision(ModelFacing):
    """Stage-one creative direction, formed after seeing everything.

    This replaces the fixed profile the previous system carried. A profile
    tuned on one shoot silently mis-fits the next; a decision made against the
    actual material can be read, argued with, and checked against the cut.
    """

    reasoning: str = Field(
        description=(
            "Why this material wants this treatment. Written before the "
            "decisions below, because a conclusion reached first and "
            "explained afterwards tends to be the conventional one."
        )
    )
    material_assessment: str = Field(
        description="What this footage is: kind, light, mood, what it affords."
    )
    material_notes: list[MaterialNote] = Field(default_factory=list)
    direction: str = Field(
        description="Overall tone, cutting rhythm, and grade direction."
    )
    target_seconds: float = Field(gt=0.0)
    aspect: Literal["16:9", "9:16", "1:1", "other"]
    aspect_other: str | None = Field(
        default=None,
        description="Required when aspect is 'other'; e.g. '4:5'.",
    )
    music_suggestion: str | None = Field(
        default=None,
        description=(
            "When no track was supplied: what music would suit this cut. The "
            "rhythm strategy in `direction` still has to stand on its own."
        ),
    )

    @model_validator(mode="after")
    def describe_other_aspect(self) -> "StyleDecision":
        if self.aspect == "other" and not (self.aspect_other or "").strip():
            raise ValueError("aspect 'other' needs aspect_other spelled out")
        return self


class Subject(ModelFacing):
    """Who or what the frame is about, in words a detector can be pointed at."""

    description: str = Field(
        description=(
            "The subject as a viewer would name it, distinguishing it from "
            "anything similar in the same frame. Two handsets on a table need "
            "'the left, grey one', not 'the handset' -- an ambiguous subject "
            "is the single most common reason a shot cannot be reframed."
        )
    )
    coarse_position: Literal[
        "top_left", "top_center", "top_right",
        "mid_left", "center", "mid_right",
        "bottom_left", "bottom_center", "bottom_right",
    ] = Field(description="Where it sits at the clip's start, roughly.")
    min_visible: Normalised = Field(
        default=0.85,
        description=(
            "Smallest share of the subject that still carries the meaning. "
            "This is the normal way to say a subject matters, because a "
            "fraction leaves a crop planable. Reserve 1.0 for content partial "
            "clipping destroys outright -- rendered text, a UI state, a "
            "readout -- since a subject demanding all of itself can only be "
            "honoured or abandoned."
        ),
    )


class Reframe(ModelFacing):
    """What the camera should attend to, never where to put the crop."""

    subject: Subject | None = Field(
        default=None,
        description="Null for a landscape or empty frame; set pan_hint instead.",
    )
    pan_hint: Literal["left", "right", "up", "down", "none"] = Field(
        default="none",
        description="Drift direction for an empty frame with no subject.",
    )
    intent: str = Field(
        default="",
        description="The move in words: hold, follow, ease in on the hands.",
    )
    then_subject: Subject | None = Field(
        default=None,
        description=(
            "A second subject this shot hands over to. The pipeline splits "
            "the shot in two rather than changing target mid-move, so both "
            "halves get a frame that commits to one thing."
        ),
    )
    camera_energy: CameraEnergy = Field(
        default="calm",
        description=(
            "How lively the move is. This picks a parameter set locally; it "
            "is not a speed. Say what the shot wants and let the executor "
            "size it against the measured motion."
        ),
    )

    @model_validator(mode="after")
    def empty_frames_need_a_direction(self) -> "Reframe":
        if self.subject is None and self.pan_hint == "none" and not self.intent:
            raise ValueError(
                "a subjectless reframe needs a pan_hint or a stated intent"
            )
        return self


class MusicSync(ModelFacing):
    """How this shot sits against the music.

    Timing resolves locally against a measured grid, so nothing here is a
    timestamp. What is here is the editorial part: whether this cut wants to
    be felt with the music or against it, and why.

    Cutting everything to a fixed beat count is the failure mode on the other
    side of ignoring the music entirely. A shot whose action needs another
    half-second should get it and land late; a hard accent should be hit even
    if the shot before it has to give way. Say which of those this is.
    """

    cut_on_beat: bool = Field(
        default=True,
        description=(
            "True to land the out-point on the nearest musical event. False "
            "when the content should govern -- an action mid-completion, a "
            "reaction still landing, a held beat of silence. Cutting through "
            "a finished gesture to hit an accent is worse than arriving a "
            "moment late."
        ),
    )
    sync_to: str | None = Field(
        default=None,
        description=(
            "A named point this shot should coincide with, when it matters: "
            "'chorus_1_start', 'drop', 'outro_start'. Leave null when no "
            "particular moment in the track is being targeted."
        ),
    )
    beats: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Length in beats, only when the shot genuinely wants a musical "
            "count -- a montage pulse, a repeated figure. Leave null and let "
            "the clip's own in and out times express the length whenever the "
            "content, not the metre, decides how long this is on screen."
        ),
    )
    rhythm_reason: str = Field(
        default="",
        description=(
            "Why this shot takes the length it does, in terms of what is "
            "happening on screen and in the track. One line. This is the "
            "record of an editorial decision, so 'eight beats' is not a "
            "reason -- 'the fold completes on the downbeat' is."
        ),
    )


class Clip(ModelFacing):
    """One shot. Times are approximate; grounding snaps them."""

    clip_id: str
    source_id: str
    approx_in_seconds: float = Field(ge=0.0)
    approx_out_seconds: float = Field(gt=0.0)
    in_looks_like: str = Field(
        default="",
        description=(
            "What is on screen at the in-point. Grounding uses this to snap "
            "to the right boundary when the timestamp lands near several."
        ),
    )
    out_looks_like: str = Field(default="")
    music_sync: MusicSync = Field(default_factory=MusicSync)
    energy_intent: EnergyIntent = Field(
        default="medium",
        description=(
            "How energetic this beat of the film is, so a high-energy stretch "
            "can be checked for actually holding motion."
        ),
    )
    reframe: Reframe | None = Field(
        default=None,
        description="Required when the output aspect differs from the source.",
    )
    named_facts: list[str] = Field(
        default_factory=list,
        description=(
            "Strings quoted verbatim from the transcription list. Anything "
            "here that was never transcribed is flagged, not silently trusted."
        ),
    )
    sync_group: str | None = Field(
        default=None,
        description=(
            "Reserved. Groups simultaneous angles of one moment once audio "
            "fingerprint alignment exists; unused today."
        ),
    )

    @model_validator(mode="after")
    def out_follows_in(self) -> "Clip":
        if self.approx_out_seconds <= self.approx_in_seconds:
            raise ValueError(
                f"clip {self.clip_id}: out must fall after in"
            )
        return self


class Goal(ModelFacing):
    """Something the audience should feel or understand.

    Satisfied by the film as a whole. The previous system read a brief as a
    shot list and went hunting for one picture per line, which is how a brief
    asking to convey craftsmanship turns into four redundant process shots
    instead of one good macro.
    """

    goal_id: str
    text: str
    covered_by: list[str] = Field(
        default_factory=list, description="clip_ids contributing. May overlap."
    )
    coverage_type: CoverageType = "implied"
    uncovered_reason: str | None = Field(
        default=None,
        description=(
            "Why the cut does not serve this goal. A stated, argued omission "
            "is a legitimate outcome, not a failure."
        ),
    )

    @model_validator(mode="after")
    def account_for_the_gap(self) -> "Goal":
        if not self.covered_by and not (self.uncovered_reason or "").strip():
            raise ValueError(
                f"goal {self.goal_id}: needs contributing clips or a reason"
            )
        return self


class HardConstraint(ModelFacing):
    """Something that must literally appear. Checked off one by one."""

    constraint_id: str
    text: str
    satisfied_by: list[str] = Field(default_factory=list)


class BriefCoverage(ModelFacing):
    hard_constraints: list[HardConstraint] = Field(default_factory=list)
    goals: list[Goal] = Field(default_factory=list)


class EDL(ModelFacing):
    """The edit decision list: the whole plan, from any source.

    The executor cannot tell whether this came from the planner or from a text
    editor, and that is the point -- a hand-written EDL is how an execution
    bug gets told apart from a planning one.
    """

    edl_version: Literal["jascue-auto-edl-v1"] = "jascue-auto-edl-v1"
    project_id: str
    style_decision: StyleDecision | None = Field(
        default=None,
        description=(
            "Carried so review can judge the cut against its own stated "
            "intent. Absent for a hand-written EDL."
        ),
    )
    clips: list[Clip] = Field(min_length=1)
    brief_coverage: BriefCoverage = Field(default_factory=BriefCoverage)
    named_facts: list[NamedFact] = Field(default_factory=list)

    @model_validator(mode="after")
    def clip_ids_are_unique(self) -> "EDL":
        ids = [clip.clip_id for clip in self.clips]
        duplicates = {value for value in ids if ids.count(value) > 1}
        if duplicates:
            raise ValueError(f"duplicate clip_ids: {sorted(duplicates)}")
        return self

    @model_validator(mode="after")
    def coverage_points_at_real_clips(self) -> "EDL":
        known = {clip.clip_id for clip in self.clips}
        for goal in self.brief_coverage.goals:
            unknown = set(goal.covered_by) - known
            if unknown:
                raise ValueError(
                    f"goal {goal.goal_id} cites unknown clips: {sorted(unknown)}"
                )
        for constraint in self.brief_coverage.hard_constraints:
            unknown = set(constraint.satisfied_by) - known
            if unknown:
                raise ValueError(
                    f"constraint {constraint.constraint_id} cites unknown "
                    f"clips: {sorted(unknown)}"
                )
        return self


class DegradationStep(Local):
    """One rung down from the plan, with the measurement that forced it.

    Degrading is not free. Each step needs its own evidence that the rung
    above was tried and failed, because a system allowed to skip straight to
    the safe option will take it every time -- which is the conservative bias
    this pipeline exists to remove.
    """

    clip_id: str
    ladder: Literal[
        "full_move", "reduced_zoom", "slower_follow", "static_on_subject",
        "center_crop", "other",
    ]
    ladder_other: str | None = None
    trigger: str = Field(description="What was attempted and how it failed.")
    measured: dict[str, float] = Field(
        default_factory=dict,
        description="The numbers behind the claim, so it can be argued with.",
    )
    adjudication: Literal["accept", "replan", "unadjudicated"] = "unadjudicated"
    adjudication_reason: str | None = None


class Issue(ModelFacing):
    """One actionable note from review."""

    issue_type: Literal[
        "pacing", "framing", "music_sync", "coverage", "named_fact",
        "continuity", "other",
    ]
    issue_type_other: str | None = None
    severity: Literal["minor", "major", "blocking"]
    clip_id: str | None = None
    at_seconds: float | None = Field(
        default=None,
        description="Roughly where, in the cut. Grounding snaps it; a rough "
        "position beats no position.",
    )
    description: str
    fix: str = Field(
        description=(
            "What to change. A note nobody can act on does not start another "
            "round, so 'this could feel better' belongs in `description`."
        )
    )


class ReviewVerdict(ModelFacing):
    """The reviewer's call. It may approve; that is how the loop converges."""

    verdict: Literal["approve", "revise"]
    overall: str
    issues: list[Issue] = Field(default_factory=list)

    @model_validator(mode="after")
    def revision_needs_something_to_do(self) -> "ReviewVerdict":
        if self.verdict == "revise" and not any(
            issue.severity in {"major", "blocking"} for issue in self.issues
        ):
            raise ValueError(
                "revise needs at least one major or blocking issue; minor "
                "notes alone do not justify another render"
            )
        return self
