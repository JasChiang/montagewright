from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .autonomous_policy import AutonomousEditPolicy
from .models import FrozenStrictModel


BeatPriority = Literal["hard", "preferred", "optional"]
SourceReuseMode = Literal[
    "none",
    "distinct_interval",
    "alternate_presentation",
    "editorial_reprise",
]
PresentationMode = Literal[
    "source_hold",
    "static_full_bleed_crop",
    "tracked_full_bleed_crop",
    "phase_virtual_camera",
    "hard_cut_between_views",
    "controlled_semantic_clip",
    "two_panel_layout",
    "solid_matte_fit",
    "intentional_freeze",
]


FrontierAttemptStage = Literal[
    "local_preflight",
    "exact_event",
    "grounding",
]
FrontierAttemptOutcome = Literal[
    "local_preflight_passed",
    "local_preflight_failed",
    "exact_event_passed",
    "exact_event_failed",
    "grounding_accepted",
    "grounding_failed",
]
FrontierBeatStatus = Literal[
    "pending",
    "accepted",
    "omitted",
    "exhausted",
]


class RoundRobinFrontierCandidate(FrozenStrictModel):
    """One candidate in a beat-local fallback order.

    ``candidate_order`` is explicit so the pure scheduler never derives
    execution order from caller container order or semantic scores.
    """

    beat_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    candidate_execution_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    candidate_order: int = Field(ge=0)
    requires_exact_event: bool = False


class RoundRobinFrontierBeat(FrozenStrictModel):
    """One story beat participating in the paid candidate frontier."""

    beat_id: str = Field(min_length=1)
    story_order: int = Field(ge=0)
    priority: BeatPriority = "preferred"
    candidates: tuple[RoundRobinFrontierCandidate, ...] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_candidates(self) -> "RoundRobinFrontierBeat":
        if any(
            candidate.beat_id != self.beat_id
            for candidate in self.candidates
        ):
            raise ValueError("frontier candidates must match their beat")
        candidate_keys = [
            (
                candidate.candidate_id,
                candidate.candidate_execution_sha256,
            )
            for candidate in self.candidates
        ]
        if len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError(
                "frontier candidate executions must be unique per beat"
            )
        candidate_orders = [
            candidate.candidate_order for candidate in self.candidates
        ]
        if len(candidate_orders) != len(set(candidate_orders)):
            raise ValueError(
                "frontier candidate order must be unique per beat"
            )
        return self


class RoundRobinFrontierAttempt(FrozenStrictModel):
    """The only next operation currently admitted by the frontier."""

    revision: int = Field(ge=0)
    beat_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    candidate_execution_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    story_order: int = Field(ge=0)
    priority: BeatPriority = "preferred"
    candidate_order: int = Field(ge=0)
    round_index: int = Field(ge=1)
    stage: FrontierAttemptStage
    paid: bool

    @model_validator(mode="after")
    def validate_paid_stage(self) -> "RoundRobinFrontierAttempt":
        expected_paid = self.stage != "local_preflight"
        if self.paid != expected_paid:
            raise ValueError(
                "only exact-event and grounding frontier stages are paid"
            )
        return self


class RoundRobinFrontierAttemptResult(FrozenStrictModel):
    attempt: RoundRobinFrontierAttempt
    outcome: FrontierAttemptOutcome
    accepted_candidate_id: str | None = None
    accepted_candidate_execution_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    beat_omitted: bool = False
    decision_codes: tuple[str, ...] = ()
    paid_calls_added: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_stage_outcome(self) -> "RoundRobinFrontierAttemptResult":
        allowed_by_stage = {
            "local_preflight": {
                "local_preflight_passed",
                "local_preflight_failed",
            },
            "exact_event": {
                "exact_event_passed",
                "exact_event_failed",
            },
            "grounding": {
                "grounding_accepted",
                "grounding_failed",
            },
        }
        if self.outcome not in allowed_by_stage[self.attempt.stage]:
            raise ValueError(
                "frontier attempt outcome does not match its stage"
            )
        if (
            self.accepted_candidate_id is not None
            and self.outcome
            not in {
                "local_preflight_failed",
                "exact_event_failed",
                "grounding_failed",
                "grounding_accepted",
            }
        ):
            raise ValueError(
                "candidate override requires a grounding acceptance or a "
                "terminal-stage failure that activates an earlier deferred "
                "fallback"
            )
        if (
            self.accepted_candidate_execution_sha256 is not None
            and self.accepted_candidate_id is None
        ):
            raise ValueError(
                "accepted candidate execution requires its candidate ID"
            )
        if self.beat_omitted and (
            not self.outcome.endswith("_failed")
            or self.accepted_candidate_id is not None
            or self.accepted_candidate_execution_sha256 is not None
        ):
            raise ValueError(
                "beat omission requires a terminal candidate failure and "
                "cannot also accept a candidate"
            )
        if self.attempt.stage == "local_preflight" and self.paid_calls_added not in {
            None,
            0,
        }:
            # A local-preflight failure may exhaust its route candidates and
            # synchronously invoke the one policy-authorized scoped semantic
            # replan.  The attempt itself remains a failed local preflight;
            # the explicit decision code makes the exceptional paid action
            # auditable instead of misclassifying it as silent local work.
            if not (
                self.outcome == "local_preflight_failed"
                and "earlier_deferred_fallback_accepted_before_next_priority_tier"
                in self.decision_codes
            ):
                raise ValueError("local preflight cannot add a paid call")
        return self


class RoundRobinFrontierBeatState(FrozenStrictModel):
    beat: RoundRobinFrontierBeat
    candidate_cursor: int = Field(default=0, ge=0)
    round_index: int = Field(default=1, ge=1)
    status: FrontierBeatStatus = "pending"
    active_candidate_id: str | None = None
    active_candidate_execution_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    next_stage: FrontierAttemptStage = "local_preflight"
    accepted_candidate_id: str | None = None
    accepted_candidate_execution_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_runtime_state(self) -> "RoundRobinFrontierBeatState":
        candidate_count = len(self.beat.candidates)
        if self.candidate_cursor > candidate_count:
            raise ValueError("frontier candidate cursor is out of range")
        if self.status == "pending":
            if self.candidate_cursor >= candidate_count:
                raise ValueError(
                    "pending frontier beat must have a remaining candidate"
                )
            current = self.beat.candidates[self.candidate_cursor]
            if (
                self.active_candidate_id is not None
                and self.active_candidate_id != current.candidate_id
            ):
                raise ValueError(
                    "active frontier candidate must match the cursor"
                )
            if (
                self.active_candidate_execution_sha256 is not None
                and self.active_candidate_execution_sha256
                != current.candidate_execution_sha256
            ):
                raise ValueError(
                    "active frontier execution must match the cursor"
                )
            if (
                self.next_stage != "local_preflight"
                and self.active_candidate_id is None
            ):
                raise ValueError(
                    "paid frontier stages require an active candidate"
                )
            if (
                self.next_stage != "local_preflight"
                and self.active_candidate_execution_sha256
                != current.candidate_execution_sha256
            ):
                raise ValueError(
                    "paid frontier stage must bind the active execution"
                )
        elif self.status == "accepted":
            if self.accepted_candidate_id is None:
                raise ValueError(
                    "accepted frontier beat must bind its candidate"
                )
            matching = [
                candidate
                for candidate in self.beat.candidates
                if candidate.candidate_id == self.accepted_candidate_id
                and (
                    self.accepted_candidate_execution_sha256 is None
                    or candidate.candidate_execution_sha256
                    == self.accepted_candidate_execution_sha256
                )
            ]
            if len(matching) != 1:
                raise ValueError(
                    "accepted frontier beat must bind one candidate execution"
                )
        elif (
            self.accepted_candidate_id is not None
            or self.accepted_candidate_execution_sha256 is not None
        ):
            raise ValueError(
                "only an accepted frontier beat may bind a candidate"
            )
        return self


class RoundRobinFrontierState(FrozenStrictModel):
    """Immutable state for fair, candidate-round paid execution."""

    contract_version: Literal["round-robin-paid-frontier-v2"] = (
        "round-robin-paid-frontier-v2"
    )
    revision: int = Field(default=0, ge=0)
    beats: tuple[RoundRobinFrontierBeatState, ...] = Field(min_length=1)
    attempt_history: tuple[RoundRobinFrontierAttemptResult, ...] = ()
    paid_calls_consumed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_beats(self) -> "RoundRobinFrontierState":
        beat_ids = [state.beat.beat_id for state in self.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("frontier beat IDs must be unique")
        story_orders = [state.beat.story_order for state in self.beats]
        if len(story_orders) != len(set(story_orders)):
            raise ValueError("frontier story order must be unique")
        if story_orders != sorted(story_orders):
            raise ValueError(
                "frontier state beats must remain in explicit story order"
            )
        return self


def initialize_round_robin_frontier(
    beats: Sequence[RoundRobinFrontierBeat],
) -> RoundRobinFrontierState:
    """Canonicalize caller input into deterministic story/candidate order."""

    if not beats:
        raise ValueError("round-robin frontier requires at least one beat")
    canonical_beats = []
    for beat in beats:
        canonical_beats.append(
            beat.model_copy(
                update={
                    "candidates": tuple(
                        sorted(
                            beat.candidates,
                            key=lambda candidate: (
                                candidate.candidate_order,
                                candidate.candidate_id,
                                candidate.candidate_execution_sha256 or "",
                            ),
                        )
                    )
                }
            )
        )
    canonical_beats.sort(
        key=lambda beat: (beat.story_order, beat.beat_id)
    )
    return RoundRobinFrontierState(
        beats=tuple(
            RoundRobinFrontierBeatState(beat=beat)
            for beat in canonical_beats
        )
    )


def next_round_robin_frontier_attempt(
    state: RoundRobinFrontierState,
) -> RoundRobinFrontierAttempt | None:
    """Return one deterministic operation without mutating frontier state.

    Hard beats are completely resolved before preferred beats, and preferred
    beats before optional beats.  Inside one priority class, beats in the
    lowest unfinished paid round always win.  Consequently a flexible earlier
    beat cannot consume source capacity needed by a later hard beat, while
    candidate retry fairness remains deterministic inside each class.
    """

    pending = [row for row in state.beats if row.status == "pending"]
    if not pending:
        return None
    priority_rank = {"hard": 0, "preferred": 1, "optional": 2}
    minimum_priority_rank = min(
        priority_rank[row.beat.priority] for row in pending
    )
    priority_pending = [
        row
        for row in pending
        if priority_rank[row.beat.priority] == minimum_priority_rank
    ]
    minimum_round = min(row.round_index for row in priority_pending)
    selected = min(
        (
            row
            for row in priority_pending
            if row.round_index == minimum_round
        ),
        key=lambda row: (row.beat.story_order, row.beat.beat_id),
    )
    candidate = selected.beat.candidates[selected.candidate_cursor]
    return RoundRobinFrontierAttempt(
        revision=state.revision,
        beat_id=selected.beat.beat_id,
        candidate_id=candidate.candidate_id,
        candidate_execution_sha256=(
            candidate.candidate_execution_sha256
        ),
        story_order=selected.beat.story_order,
        priority=selected.beat.priority,
        candidate_order=candidate.candidate_order,
        round_index=selected.round_index,
        stage=selected.next_stage,
        paid=selected.next_stage != "local_preflight",
    )


def _resolve_accepted_frontier_candidate(
    beat: RoundRobinFrontierBeat,
    current: RoundRobinFrontierCandidate,
    result: RoundRobinFrontierAttemptResult,
) -> RoundRobinFrontierCandidate:
    if result.accepted_candidate_id is None:
        return current
    matching = [
        candidate
        for candidate in beat.candidates
        if candidate.candidate_id == result.accepted_candidate_id
        and (
            result.accepted_candidate_execution_sha256 is None
            or candidate.candidate_execution_sha256
            == result.accepted_candidate_execution_sha256
        )
    ]
    if len(matching) != 1:
        raise ValueError(
            "accepted frontier override must identify one candidate execution"
        )
    return matching[0]


def record_round_robin_frontier_attempt(
    state: RoundRobinFrontierState,
    result: RoundRobinFrontierAttemptResult,
) -> RoundRobinFrontierState:
    """Apply exactly the currently admitted operation to immutable state."""

    expected = next_round_robin_frontier_attempt(state)
    if expected is None:
        raise ValueError("round-robin frontier is already complete")
    if result.attempt != expected:
        raise ValueError("stale or out-of-order frontier attempt result")

    updated_beats = list(state.beats)
    beat_index = next(
        index
        for index, row in enumerate(updated_beats)
        if row.beat.beat_id == expected.beat_id
    )
    beat_state = updated_beats[beat_index]
    candidate = beat_state.beat.candidates[
        beat_state.candidate_cursor
    ]

    if result.outcome == "local_preflight_passed":
        beat_state = beat_state.model_copy(
            update={
                "active_candidate_id": candidate.candidate_id,
                "active_candidate_execution_sha256": (
                    candidate.candidate_execution_sha256
                ),
                "next_stage": (
                    "exact_event"
                    if candidate.requires_exact_event
                    else "grounding"
                ),
            }
        )
    elif result.outcome == "exact_event_passed":
        beat_state = beat_state.model_copy(
            update={"next_stage": "grounding"}
        )
    elif result.outcome == "grounding_accepted":
        accepted_candidate = _resolve_accepted_frontier_candidate(
            beat_state.beat,
            candidate,
            result,
        )
        beat_state = beat_state.model_copy(
            update={
                "status": "accepted",
                "active_candidate_id": None,
                "active_candidate_execution_sha256": None,
                "accepted_candidate_id": accepted_candidate.candidate_id,
                "accepted_candidate_execution_sha256": (
                    accepted_candidate.candidate_execution_sha256
                ),
            }
        )
    elif result.beat_omitted:
        beat_state = beat_state.model_copy(
            update={
                "status": "omitted",
                "active_candidate_id": None,
                "active_candidate_execution_sha256": None,
                "accepted_candidate_id": None,
                "accepted_candidate_execution_sha256": None,
            }
        )
    elif result.accepted_candidate_id is not None:
        accepted_candidate = _resolve_accepted_frontier_candidate(
            beat_state.beat,
            candidate,
            result,
        )
        beat_state = beat_state.model_copy(
            update={
                "status": "accepted",
                "active_candidate_id": None,
                "active_candidate_execution_sha256": None,
                "accepted_candidate_id": accepted_candidate.candidate_id,
                "accepted_candidate_execution_sha256": (
                    accepted_candidate.candidate_execution_sha256
                ),
            }
        )
    else:
        next_cursor = beat_state.candidate_cursor + 1
        candidate_failure_was_paid = expected.paid
        beat_state = beat_state.model_copy(
            update={
                "candidate_cursor": next_cursor,
                "round_index": (
                    beat_state.round_index + 1
                    if candidate_failure_was_paid
                    else beat_state.round_index
                ),
                "status": (
                    "exhausted"
                    if next_cursor >= len(beat_state.beat.candidates)
                    else "pending"
                ),
                "active_candidate_id": None,
                "active_candidate_execution_sha256": None,
                "next_stage": "local_preflight",
            }
        )

    updated_beats[beat_index] = beat_state
    return state.model_copy(
        update={
            "revision": state.revision + 1,
            "beats": tuple(updated_beats),
            "attempt_history": (
                *state.attempt_history,
                result,
            ),
            "paid_calls_consumed": (
                state.paid_calls_consumed
                + (
                    int(expected.paid)
                    if result.paid_calls_added is None
                    else result.paid_calls_added
                )
            ),
        }
    )


class CandidateRouteOption(FrozenStrictModel):
    """One bounded pre-render sequence choice.

    Exact evidence, tracking, and pixel geometry are deliberately not claimed
    here.  The option binds every editorial fact that is already known before
    rendering (candidate, resolved trim duration, music exit, presentation
    family, and symbolic entry/exit composition).  Runtime execution must
    still pass the unresolved hard gates and may advance to the next
    globally-ranked fallback.
    """

    beat_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    source_asset_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    event_id: str = Field(min_length=1)
    planner_rank: int = Field(ge=1)
    semantic_confidence: float = Field(ge=0.0, le=1.0)
    presentation_intrusion_rank: int = Field(default=0, ge=0, le=10)
    trim_duration_ms: int = Field(default=1, gt=0)
    minimum_readable_ms: int = Field(default=1, gt=0)
    preferred_readable_ms: int = Field(default=1, gt=0)
    maximum_readable_ms: int = Field(default=1, gt=0)
    cue_id: str = Field(default="no-music", min_length=1)
    cue_aligned: bool = True
    presentation_mode: PresentationMode = "source_hold"
    entry_composition: str = Field(default="unresolved", min_length=1)
    exit_composition: str = Field(default="unresolved", min_length=1)
    technical_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    preflight_hard_failures: tuple[str, ...] = ()
    preflight_deferred_gates: tuple[str, ...] = ()
    source_clip_id: str | None = Field(default=None, min_length=1)
    safe_capacity_ms: int | None = Field(default=None, gt=0)
    safe_window_start_ms: int | None = Field(default=None, ge=0)
    safe_window_end_ms: int | None = Field(default=None, ge=0)
    source_anchor_ms: int | None = Field(default=None, ge=0)
    fixed_source_in_ms: int | None = Field(default=None, ge=0)
    fixed_source_out_ms: int | None = Field(default=None, ge=0)
    reuse_mode: SourceReuseMode = "none"
    reuse_justification: str | None = None
    candidate_timing_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_editorial_bounds(self) -> "CandidateRouteOption":
        if not (
            self.minimum_readable_ms
            <= self.preferred_readable_ms
            <= self.maximum_readable_ms
        ):
            raise ValueError("pre-render readability bounds must be ordered")
        if not (
            self.minimum_readable_ms
            <= self.trim_duration_ms
            <= self.maximum_readable_ms
        ):
            raise ValueError(
                "pre-render trim duration must remain inside readability bounds"
            )
        safe_fields = (
            self.safe_window_start_ms,
            self.safe_window_end_ms,
            self.source_anchor_ms,
        )
        if any(value is not None for value in safe_fields):
            if any(value is None for value in safe_fields):
                raise ValueError(
                    "candidate safe window requires start, end, and anchor"
                )
            assert self.safe_window_start_ms is not None
            assert self.safe_window_end_ms is not None
            assert self.source_anchor_ms is not None
            if not (
                self.safe_window_start_ms
                <= self.source_anchor_ms
                < self.safe_window_end_ms
            ):
                raise ValueError(
                    "candidate source anchor must lie inside its safe window"
                )
            measured_capacity = (
                self.safe_window_end_ms - self.safe_window_start_ms
            )
            if measured_capacity < 1:
                raise ValueError("candidate safe window must be positive")
            if (
                self.safe_capacity_ms is not None
                and self.safe_capacity_ms != measured_capacity
            ):
                raise ValueError(
                    "candidate safe capacity must match its safe window"
                )
        fixed_fields = (
            self.fixed_source_in_ms,
            self.fixed_source_out_ms,
        )
        if any(value is not None for value in fixed_fields):
            if any(value is None for value in fixed_fields):
                raise ValueError(
                    "fixed candidate trim requires both source boundaries"
                )
            assert self.fixed_source_in_ms is not None
            assert self.fixed_source_out_ms is not None
            if self.fixed_source_out_ms <= self.fixed_source_in_ms:
                raise ValueError("fixed candidate trim must be positive")
        if (
            self.reuse_mode != "none"
            and not (self.reuse_justification or "").strip()
        ):
            raise ValueError(
                "candidate source reuse authority requires a justification"
            )
        return self


class CandidateRouteBeat(FrozenStrictModel):
    beat_id: str = Field(min_length=1)
    options: tuple[CandidateRouteOption, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_options(self) -> "CandidateRouteBeat":
        if any(option.beat_id != self.beat_id for option in self.options):
            raise ValueError("candidate route options must match their beat")
        ids = [option.candidate_id for option in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate route option IDs must be unique")
        return self


class CandidateRouteSelection(FrozenStrictModel):
    beat_id: str
    candidate_id: str
    source_asset_id: str
    event_id: str
    trim_duration_ms: int = Field(gt=0)
    cue_id: str
    cue_aligned: bool
    presentation_mode: PresentationMode
    entry_composition: str
    exit_composition: str
    decision_codes: tuple[str, ...]
    source_clip_id: str | None = Field(default=None, min_length=1)
    source_in_ms: int | None = Field(default=None, ge=0)
    source_out_ms: int | None = Field(default=None, ge=0)
    candidate_execution_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    reuse_mode: SourceReuseMode = "none"
    reuse_justification: str | None = None

    @model_validator(mode="after")
    def validate_source_interval(self) -> "CandidateRouteSelection":
        boundaries = (self.source_in_ms, self.source_out_ms)
        if any(value is not None for value in boundaries):
            if any(value is None for value in boundaries):
                raise ValueError(
                    "candidate execution needs both source boundaries"
                )
            assert self.source_in_ms is not None
            assert self.source_out_ms is not None
            if self.source_out_ms - self.source_in_ms != self.trim_duration_ms:
                raise ValueError(
                    "candidate source interval must match trim duration"
                )
        return self


class CandidateCompleteRoute(FrozenStrictModel):
    """One complete, globally feasible candidate/duration/source route."""

    route_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    selections: tuple[CandidateRouteSelection, ...] = Field(min_length=1)
    objective_score: float
    total_duration_ms: int = Field(gt=0)
    panel_duration_ms: int = Field(ge=0)
    decision_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_complete_route(self) -> "CandidateCompleteRoute":
        beat_ids = [selection.beat_id for selection in self.selections]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("complete route may select each beat only once")
        if (
            sum(
                selection.trim_duration_ms
                for selection in self.selections
            )
            != self.total_duration_ms
        ):
            raise ValueError("complete route duration is inconsistent")
        measured_panel_duration_ms = sum(
            selection.trim_duration_ms
            for selection in self.selections
            if selection.presentation_mode == "two_panel_layout"
        )
        if measured_panel_duration_ms != self.panel_duration_ms:
            raise ValueError("complete route panel duration is inconsistent")
        return self


class SemanticReplanCandidateBinding(FrozenStrictModel):
    """A preflight-safe option reserved for Gemini, not local fallback.

    A candidate can be absent from every complete route solely because it
    requires an editorial reuse authorization in the eventual whole sequence.
    Keeping it here lets one bounded semantic replan consider the real
    candidate without letting the renderer execute it speculatively.
    """

    option: CandidateRouteOption
    replan_required_codes: tuple[str, ...] = ()


class SemanticReplanReuseAuthority(FrozenStrictModel):
    """Gemini's limited authority to reuse one immutable candidate."""

    beat_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    reuse_mode: SourceReuseMode = "none"
    reuse_justification: str | None = None
    reuse_of_beat_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_authority(self) -> "SemanticReplanReuseAuthority":
        if self.reuse_mode == "none":
            if self.reuse_justification is not None or self.reuse_of_beat_ids:
                raise ValueError(
                    "none reuse authority cannot include a reason or source beats"
                )
            return self
        if not (self.reuse_justification or "").strip():
            raise ValueError("semantic reuse authority requires a justification")
        if not self.reuse_of_beat_ids:
            raise ValueError("semantic reuse authority requires source beat IDs")
        if len(set(self.reuse_of_beat_ids)) != len(self.reuse_of_beat_ids):
            raise ValueError("semantic reuse authority source beat IDs must be unique")
        if self.beat_id in self.reuse_of_beat_ids:
            raise ValueError("semantic reuse authority cannot reference itself")
        return self


class CandidateRouteResult(FrozenStrictModel):
    contract_version: Literal["pre-render-sequence-frontier-v2"] = (
        "pre-render-sequence-frontier-v2"
    )
    selections: tuple[CandidateRouteSelection, ...]
    fallback_candidate_ids_by_beat: Mapping[str, tuple[str, ...]]
    option_bindings_by_beat: Mapping[
        str,
        Mapping[str, CandidateRouteOption],
    ]
    objective_score: float
    beam_width: int = Field(gt=0)
    total_duration_ms: int = Field(gt=0)
    ranked_routes: tuple[CandidateCompleteRoute, ...] = ()
    # These options are deliberately separate from ``option_bindings``:
    # being available for one Gemini replan does not authorize renderer
    # fallback or paid exact/grounding work.
    semantic_replan_candidate_bindings_by_beat: Mapping[
        str, tuple[SemanticReplanCandidateBinding, ...]
    ] = Field(default_factory=dict)
    unresolved_runtime_hard_gates: tuple[str, ...] = (
        "exact_event_evidence",
        "identity",
        "action_completeness",
        "required_relation_and_scale",
        "quality_safe_interval",
        "reuse_authority",
        "pixel_geometry",
    )


class RuntimeCueTimingBinding(FrozenStrictModel):
    """One exact event projected against an immutable output-music cue.

    ``event_offset_ms`` is measured inside the runtime-selected source window,
    not copied from the pre-render candidate.  The reconciler owns only
    project-timeline arithmetic; it never moves the source event or the cue.
    """

    event_id: str = Field(min_length=1)
    cue_id: str = Field(min_length=1)
    event_offset_ms: int = Field(ge=0)
    cue_time_ms: int = Field(ge=0)
    fps_numerator: int = Field(default=30, gt=0)
    fps_denominator: int = Field(default=1, gt=0)
    tolerance_frames: int = Field(default=0, ge=0, le=24)
    hard_sync: bool = True


class RuntimeSegmentTiming(FrozenStrictModel):
    """Measured timing for one runtime-selected candidate.

    The pre-render duration remains the editorial request.  Source capacity
    and actual duration are measured runtime facts.  An actual duration may
    be shorter, or differ by at most one output-frame of source-PTS boundary
    quantization.  It may never be padded, materially extended beyond the
    plan, exceed source capacity, or fall back to freeze/time-stretch inside
    this contract.
    """

    beat_id: str = Field(min_length=1)
    planned_candidate_id: str = Field(min_length=1)
    runtime_candidate_id: str = Field(min_length=1)
    source_asset_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    planned_duration_ms: int = Field(gt=0)
    actual_source_capacity_ms: int = Field(gt=0)
    actual_duration_ms: int = Field(gt=0)
    minimum_readable_ms: int = Field(gt=0)
    max_pts_quantization_delta_ms: int = Field(default=34, ge=0, le=100)
    cue_bindings: tuple[RuntimeCueTimingBinding, ...] = ()
    input_artifact_hashes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_runtime_duration(self) -> "RuntimeSegmentTiming":
        if self.actual_duration_ms > self.actual_source_capacity_ms:
            raise ValueError(
                "runtime duration cannot exceed measured source capacity"
            )
        if (
            self.actual_duration_ms
            > self.planned_duration_ms + self.max_pts_quantization_delta_ms
        ):
            raise ValueError(
                "runtime reconciliation cannot materially extend a pre-render "
                "duration beyond its PTS quantization tolerance"
            )
        if any(
            binding.event_offset_ms >= self.actual_duration_ms
            for binding in self.cue_bindings
        ):
            raise ValueError(
                "runtime cue event must lie inside the selected duration"
            )
        return self


class ReconciledRuntimeCueTiming(FrozenStrictModel):
    event_id: str
    cue_id: str
    planned_project_event_time_ms: int = Field(ge=0)
    resolved_project_event_time_ms: int = Field(ge=0)
    cue_time_ms: int = Field(ge=0)
    planned_delta_frames: int
    resolved_delta_frames: int
    delta_change_frames: int
    tolerance_frames: int = Field(ge=0, le=24)
    hard_sync: bool
    passed: bool

    @model_validator(mode="after")
    def validate_cue_result(self) -> "ReconciledRuntimeCueTiming":
        if (
            self.delta_change_frames
            != self.resolved_delta_frames - self.planned_delta_frames
        ):
            raise ValueError("runtime cue delta change is inconsistent")
        if self.passed != (
            abs(self.resolved_delta_frames) <= self.tolerance_frames
        ):
            raise ValueError("runtime cue pass flag is inconsistent")
        return self


class ReconciledRuntimeSegmentTiming(FrozenStrictModel):
    beat_id: str
    planned_candidate_id: str
    runtime_candidate_id: str
    source_asset_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    planned_project_start_ms: int = Field(ge=0)
    planned_project_end_ms: int = Field(gt=0)
    resolved_project_start_ms: int = Field(ge=0)
    resolved_project_end_ms: int = Field(gt=0)
    planned_duration_ms: int = Field(gt=0)
    actual_source_capacity_ms: int = Field(gt=0)
    resolved_duration_ms: int = Field(gt=0)
    duration_delta_ms: int
    project_shift_before_ms: int
    project_shift_after_ms: int
    runtime_substitute_selected: bool
    source_capacity_limited: bool
    synthetic_fill_ms: Literal[0] = 0
    time_stretch_ratio: Literal[1.0] = 1.0
    cue_timings: tuple[ReconciledRuntimeCueTiming, ...] = ()
    decision_codes: tuple[str, ...]

    @model_validator(mode="after")
    def validate_boundaries(self) -> "ReconciledRuntimeSegmentTiming":
        if (
            self.planned_project_end_ms - self.planned_project_start_ms
            != self.planned_duration_ms
        ):
            raise ValueError("planned project boundaries are inconsistent")
        if (
            self.resolved_project_end_ms - self.resolved_project_start_ms
            != self.resolved_duration_ms
        ):
            raise ValueError("resolved project boundaries are inconsistent")
        if (
            self.duration_delta_ms
            != self.resolved_duration_ms - self.planned_duration_ms
        ):
            raise ValueError("runtime duration delta is inconsistent")
        if (
            self.project_shift_before_ms
            != self.resolved_project_start_ms
            - self.planned_project_start_ms
        ):
            raise ValueError("runtime project start shift is inconsistent")
        if (
            self.project_shift_after_ms
            != self.resolved_project_end_ms - self.planned_project_end_ms
        ):
            raise ValueError("runtime project end shift is inconsistent")
        return self


class RuntimeSequenceTimingReconciliation(FrozenStrictModel):
    """Auditable replacement of planned timing with measured runtime timing."""

    contract_version: Literal["runtime-sequence-timing-reconciliation-v1"] = (
        "runtime-sequence-timing-reconciliation-v1"
    )
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    segments: tuple[ReconciledRuntimeSegmentTiming, ...]
    planned_total_duration_ms: int = Field(ge=0)
    resolved_total_duration_ms: int = Field(ge=0)
    total_duration_delta_ms: int
    outcome: Literal["unchanged", "reconciled", "blocked"]
    freeze_inserted: Literal[False] = False
    time_stretch_applied: Literal[False] = False
    decision_codes: tuple[str, ...]
    failure_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_totals(self) -> "RuntimeSequenceTimingReconciliation":
        if (
            self.total_duration_delta_ms
            != self.resolved_total_duration_ms
            - self.planned_total_duration_ms
        ):
            raise ValueError("runtime total duration delta is inconsistent")
        if self.outcome == "blocked" and not self.failure_codes:
            raise ValueError("blocked runtime reconciliation needs failures")
        if self.outcome != "blocked" and self.failure_codes:
            raise ValueError("successful runtime reconciliation has failures")
        return self


class ResolvedTimelineV1(FrozenStrictModel):
    """Per-aspect project-time authority after rendered durations are probed."""

    contract_version: Literal["resolved-timeline-v1"] = "resolved-timeline-v1"
    aspect: Literal["16:9", "9:16"]
    reconciliation_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_total_duration_ms: int = Field(ge=0)
    segments: tuple[ReconciledRuntimeSegmentTiming, ...]
    music_output_timeline_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_definition(self) -> "ResolvedTimelineV1":
        payload = self.model_dump(mode="json", exclude={"definition_sha256"})
        expected = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if self.definition_sha256 != expected:
            raise ValueError("resolved timeline definition hash is inconsistent")
        if (
            sum(segment.resolved_duration_ms for segment in self.segments)
            != self.resolved_total_duration_ms
        ):
            raise ValueError("resolved timeline total differs from its segments")
        return self


def build_resolved_timeline(
    *,
    aspect: Literal["16:9", "9:16"],
    reconciliation: RuntimeSequenceTimingReconciliation,
    music_output_timeline_sha256: str | None = None,
) -> ResolvedTimelineV1:
    if reconciliation.outcome == "blocked":
        raise ValueError("blocked reconciliation cannot become time authority")
    payload = {
        "contract_version": "resolved-timeline-v1",
        "aspect": aspect,
        "reconciliation_input_sha256": reconciliation.input_sha256,
        "resolved_total_duration_ms": (
            reconciliation.resolved_total_duration_ms
        ),
        "segments": [
            segment.model_dump(mode="json")
            for segment in reconciliation.segments
        ],
        "music_output_timeline_sha256": music_output_timeline_sha256,
    }
    definition_sha256 = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ResolvedTimelineV1(
        **payload,
        definition_sha256=definition_sha256,
    )


def reconcile_runtime_sequence_timing(
    segments: Sequence[RuntimeSegmentTiming],
    *,
    minimum_total_duration_ms: int | None = None,
    maximum_total_duration_ms: int | None = None,
    pts_duration_quantization_tolerance_ms: int = 0,
) -> RuntimeSequenceTimingReconciliation:
    """Rebase a pre-render sequence onto measured runtime source durations.

    This is intentionally deterministic and conservative.  It accepts a
    shorter, already-selected Top-K substitute only at its measured duration,
    shifts every downstream project boundary, and recomputes exact-event cue
    deltas.  It never fills the shortfall with a freeze, time-stretch, or an
    unplanned extension of another segment.
    """

    if (minimum_total_duration_ms is None) != (
        maximum_total_duration_ms is None
    ):
        raise ValueError("runtime total duration bounds must be supplied together")
    if (
        minimum_total_duration_ms is not None
        and maximum_total_duration_ms is not None
        and (
            minimum_total_duration_ms < 1
            or maximum_total_duration_ms < minimum_total_duration_ms
        )
    ):
        raise ValueError("runtime total duration bounds are invalid")
    if not 0 <= pts_duration_quantization_tolerance_ms <= 100:
        raise ValueError("runtime PTS duration quantization tolerance is invalid")
    beat_ids = [segment.beat_id for segment in segments]
    if len(beat_ids) != len(set(beat_ids)):
        raise ValueError("runtime sequence beat IDs must be unique")
    event_ids = [
        binding.event_id
        for segment in segments
        for binding in segment.cue_bindings
    ]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("runtime sequence cue event IDs must be unique")

    input_payload = {
        "segments": [
            segment.model_dump(mode="json") for segment in segments
        ],
        "minimum_total_duration_ms": minimum_total_duration_ms,
        "maximum_total_duration_ms": maximum_total_duration_ms,
        "pts_duration_quantization_tolerance_ms": (
            pts_duration_quantization_tolerance_ms
        ),
    }
    input_sha256 = hashlib.sha256(
        json.dumps(
            input_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    planned_cursor_ms = 0
    resolved_cursor_ms = 0
    reconciled: list[ReconciledRuntimeSegmentTiming] = []
    failures: list[str] = []
    for segment in segments:
        planned_start_ms = planned_cursor_ms
        resolved_start_ms = resolved_cursor_ms
        planned_end_ms = planned_start_ms + segment.planned_duration_ms
        resolved_end_ms = resolved_start_ms + segment.actual_duration_ms
        cue_timings: list[ReconciledRuntimeCueTiming] = []
        for binding in segment.cue_bindings:
            planned_event_ms = planned_start_ms + binding.event_offset_ms
            resolved_event_ms = resolved_start_ms + binding.event_offset_ms
            planned_delta = _project_cue_delta_frames(
                project_event_time_ms=planned_event_ms,
                cue_time_ms=binding.cue_time_ms,
                fps_numerator=binding.fps_numerator,
                fps_denominator=binding.fps_denominator,
            )
            resolved_delta = _project_cue_delta_frames(
                project_event_time_ms=resolved_event_ms,
                cue_time_ms=binding.cue_time_ms,
                fps_numerator=binding.fps_numerator,
                fps_denominator=binding.fps_denominator,
            )
            passed = abs(resolved_delta) <= binding.tolerance_frames
            cue_timings.append(
                ReconciledRuntimeCueTiming(
                    event_id=binding.event_id,
                    cue_id=binding.cue_id,
                    planned_project_event_time_ms=planned_event_ms,
                    resolved_project_event_time_ms=resolved_event_ms,
                    cue_time_ms=binding.cue_time_ms,
                    planned_delta_frames=planned_delta,
                    resolved_delta_frames=resolved_delta,
                    delta_change_frames=resolved_delta - planned_delta,
                    tolerance_frames=binding.tolerance_frames,
                    hard_sync=binding.hard_sync,
                    passed=passed,
                )
            )
            if binding.hard_sync and not passed:
                failures.append(
                    f"{binding.event_id}:runtime_cue_alignment_failed"
                )
        if segment.actual_duration_ms < segment.minimum_readable_ms:
            failures.append(
                f"{segment.beat_id}:runtime_duration_below_minimum_readable"
            )
        decision_codes = ["measured_runtime_duration_applied"]
        if (
            segment.runtime_candidate_id
            != segment.planned_candidate_id
        ):
            decision_codes.append("runtime_substitute_candidate_bound")
        if segment.actual_duration_ms < segment.planned_duration_ms:
            decision_codes.append("shortfall_preserved_without_synthetic_fill")
        else:
            decision_codes.append("pre_render_duration_preserved")
        if resolved_start_ms != planned_start_ms:
            decision_codes.append("downstream_project_boundary_rebased")
        if cue_timings:
            decision_codes.append("cue_deltas_recomputed_from_resolved_boundary")
        reconciled.append(
            ReconciledRuntimeSegmentTiming(
                beat_id=segment.beat_id,
                planned_candidate_id=segment.planned_candidate_id,
                runtime_candidate_id=segment.runtime_candidate_id,
                source_asset_id=segment.source_asset_id,
                planned_project_start_ms=planned_start_ms,
                planned_project_end_ms=planned_end_ms,
                resolved_project_start_ms=resolved_start_ms,
                resolved_project_end_ms=resolved_end_ms,
                planned_duration_ms=segment.planned_duration_ms,
                actual_source_capacity_ms=(
                    segment.actual_source_capacity_ms
                ),
                resolved_duration_ms=segment.actual_duration_ms,
                duration_delta_ms=(
                    segment.actual_duration_ms - segment.planned_duration_ms
                ),
                project_shift_before_ms=(
                    resolved_start_ms - planned_start_ms
                ),
                project_shift_after_ms=resolved_end_ms - planned_end_ms,
                runtime_substitute_selected=(
                    segment.runtime_candidate_id
                    != segment.planned_candidate_id
                ),
                source_capacity_limited=(
                    segment.actual_source_capacity_ms
                    < segment.planned_duration_ms
                ),
                cue_timings=tuple(cue_timings),
                decision_codes=tuple(decision_codes),
            )
        )
        planned_cursor_ms = planned_end_ms
        resolved_cursor_ms = resolved_end_ms

    if (
        minimum_total_duration_ms is not None
        and resolved_cursor_ms + pts_duration_quantization_tolerance_ms
        < minimum_total_duration_ms
    ):
        failures.append("resolved_total_duration_below_minimum")
    if (
        maximum_total_duration_ms is not None
        and resolved_cursor_ms - pts_duration_quantization_tolerance_ms
        > maximum_total_duration_ms
    ):
        failures.append("resolved_total_duration_above_maximum")
    changed = resolved_cursor_ms != planned_cursor_ms or any(
        segment.runtime_substitute_selected for segment in reconciled
    )
    outcome: Literal["unchanged", "reconciled", "blocked"]
    if failures:
        outcome = "blocked"
    elif changed:
        outcome = "reconciled"
    else:
        outcome = "unchanged"
    return RuntimeSequenceTimingReconciliation(
        input_sha256=input_sha256,
        segments=tuple(reconciled),
        planned_total_duration_ms=planned_cursor_ms,
        resolved_total_duration_ms=resolved_cursor_ms,
        total_duration_delta_ms=resolved_cursor_ms - planned_cursor_ms,
        outcome=outcome,
        decision_codes=(
            "runtime_source_capacity_audited",
            "all_project_boundaries_recomputed",
            "all_bound_cue_deltas_recomputed",
            "no_freeze_inserted",
            "no_time_stretch_applied",
        ),
        failure_codes=tuple(dict.fromkeys(failures)),
    )


def _project_cue_delta_frames(
    *,
    project_event_time_ms: int,
    cue_time_ms: int,
    fps_numerator: int,
    fps_denominator: int,
) -> int:
    event_frame = round(
        project_event_time_ms
        * fps_numerator
        / (1_000 * fps_denominator)
    )
    cue_frame = round(
        cue_time_ms * fps_numerator / (1_000 * fps_denominator)
    )
    return event_frame - cue_frame


class _CandidateRouteState(FrozenStrictModel):
    selections: tuple[CandidateRouteSelection, ...] = ()
    options: tuple[CandidateRouteOption, ...] = ()
    source_asset_ids: tuple[str, ...] = ()
    source_events: tuple[tuple[str, str], ...] = ()
    exit_composition: str | None = None
    duration_ms: int = Field(default=0, ge=0)
    panel_duration_ms: int = Field(default=0, ge=0)
    score: float = 0.0


def _candidate_duration_bounds(
    option: CandidateRouteOption,
) -> tuple[int, int, int] | None:
    maximum_ms = option.maximum_readable_ms
    if option.safe_capacity_ms is not None:
        maximum_ms = min(maximum_ms, option.safe_capacity_ms)
    if (
        option.fixed_source_in_ms is not None
        and option.fixed_source_out_ms is not None
    ):
        fixed_ms = option.fixed_source_out_ms - option.fixed_source_in_ms
        if not option.minimum_readable_ms <= fixed_ms <= maximum_ms:
            return None
        return fixed_ms, fixed_ms, fixed_ms
    minimum_ms = option.minimum_readable_ms
    if minimum_ms > maximum_ms:
        return None
    preferred_ms = min(
        maximum_ms,
        max(minimum_ms, option.preferred_readable_ms),
    )
    return minimum_ms, preferred_ms, maximum_ms


def _distribute_duration_headroom(
    durations_ms: list[int],
    ceilings_ms: Sequence[int],
    amount_ms: int,
) -> int:
    """Distribute integer duration deterministically without exceeding caps."""

    if amount_ms <= 0:
        return 0
    headroom = [
        max(0, ceiling - duration)
        for duration, ceiling in zip(durations_ms, ceilings_ms, strict=True)
    ]
    total_headroom = sum(headroom)
    if total_headroom <= 0:
        return amount_ms
    awarded = [
        min(
            room,
            amount_ms * room // total_headroom,
        )
        for room in headroom
    ]
    for index, value in enumerate(awarded):
        durations_ms[index] += value
    remaining = amount_ms - sum(awarded)
    for index, ceiling in enumerate(ceilings_ms):
        if remaining <= 0:
            break
        available = ceiling - durations_ms[index]
        if available <= 0:
            continue
        value = min(available, remaining)
        durations_ms[index] += value
        remaining -= value
    return remaining


def _allocate_candidate_route_durations(
    options: Sequence[CandidateRouteOption],
    *,
    target_duration_ms: int | None,
) -> tuple[int, ...] | None:
    bounds = [_candidate_duration_bounds(option) for option in options]
    if any(item is None for item in bounds):
        return None
    resolved_bounds = [
        item for item in bounds if item is not None
    ]
    if target_duration_ms is None:
        durations = tuple(option.trim_duration_ms for option in options)
        if any(
            not minimum <= duration <= maximum
            for duration, (minimum, _, maximum) in zip(
                durations,
                resolved_bounds,
                strict=True,
            )
        ):
            return None
        return durations
    minimum_total = sum(item[0] for item in resolved_bounds)
    maximum_total = sum(item[2] for item in resolved_bounds)
    if not minimum_total <= target_duration_ms <= maximum_total:
        return None
    durations = [item[0] for item in resolved_bounds]
    preferred = [item[1] for item in resolved_bounds]
    maximum = [item[2] for item in resolved_bounds]
    remaining = target_duration_ms - minimum_total
    remaining = _distribute_duration_headroom(
        durations,
        preferred,
        remaining,
    )
    remaining = _distribute_duration_headroom(
        durations,
        maximum,
        remaining,
    )
    if remaining:
        return None
    return tuple(durations)


def _redistribute_candidate_route_duration(
    options: Sequence[CandidateRouteOption],
    *,
    primary_durations_ms: Sequence[int],
    forced_index: int,
    forced_duration_ms: int,
) -> tuple[int, ...] | None:
    """Move one execution to a legal contract boundary and preserve runtime.

    This is deliberately a bounded local repair, not a second rhythm solver.
    The selected beat moves to one of its declared min/preferred/max
    durations, while the delta is absorbed by the fewest other beats with
    legal readability headroom.  Cue-aligned beats are changed last.
    """

    if len(options) != len(primary_durations_ms):
        raise ValueError("duration redistribution inputs must have equal size")
    if not 0 <= forced_index < len(options):
        raise ValueError("forced duration index is out of range")
    bounds = [_candidate_duration_bounds(option) for option in options]
    if any(item is None for item in bounds):
        return None
    resolved_bounds = [item for item in bounds if item is not None]
    minimum_ms, _, maximum_ms = resolved_bounds[forced_index]
    if not minimum_ms <= forced_duration_ms <= maximum_ms:
        return None

    durations = list(primary_durations_ms)
    delta_ms = durations[forced_index] - forced_duration_ms
    durations[forced_index] = forced_duration_ms
    if delta_ms == 0:
        return tuple(durations)

    if delta_ms > 0:
        # The forced beat became shorter. Give the released runtime to the
        # smallest possible set of non-cue-locked beats with largest headroom.
        recipient_rows = [
            (
                options[index].cue_aligned,
                -(
                    resolved_bounds[index][2]
                    - durations[index]
                ),
                index,
            )
            for index in range(len(options))
            if index != forced_index
            and durations[index] < resolved_bounds[index][2]
        ]
        for _, _, index in sorted(recipient_rows):
            awarded_ms = min(
                delta_ms,
                resolved_bounds[index][2] - durations[index],
            )
            durations[index] += awarded_ms
            delta_ms -= awarded_ms
            if delta_ms == 0:
                break
    else:
        # The forced beat became longer. Recover the required runtime from the
        # fewest other beats that remain above their declared minimum.
        remaining_ms = -delta_ms
        donor_rows = [
            (
                options[index].cue_aligned,
                -(
                    durations[index]
                    - resolved_bounds[index][0]
                ),
                index,
            )
            for index in range(len(options))
            if index != forced_index
            and durations[index] > resolved_bounds[index][0]
        ]
        for _, _, index in sorted(donor_rows):
            removed_ms = min(
                remaining_ms,
                durations[index] - resolved_bounds[index][0],
            )
            durations[index] -= removed_ms
            remaining_ms -= removed_ms
            if remaining_ms == 0:
                break
        delta_ms = -remaining_ms

    if delta_ms != 0:
        return None
    return tuple(durations)


def _candidate_route_duration_variants(
    options: Sequence[CandidateRouteOption],
    *,
    target_duration_ms: int | None,
    max_variants: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate a small execution frontier from editorial duration bounds.

    The first row is always the normal rhythm allocation. Additional rows
    move exactly one beat to a declared boundary and redistribute its delta.
    This avoids a Cartesian product while giving runtime geometry/quality
    failures a shorter or longer execution of the same semantic candidate.
    """

    if max_variants < 1:
        raise ValueError("candidate duration frontier must retain one variant")
    primary = _allocate_candidate_route_durations(
        options,
        target_duration_ms=target_duration_ms,
    )
    if primary is None:
        return ()
    if target_duration_ms is None or max_variants == 1:
        return (primary,)

    variants: list[tuple[int, ...]] = [primary]
    seen = {primary}
    bounds = [_candidate_duration_bounds(option) for option in options]
    if any(item is None for item in bounds):
        return (primary,)
    resolved_bounds = [item for item in bounds if item is not None]

    # Shorter executions are the common recovery for dirty heads/tails and
    # source-camera motion, so enumerate all minima before expansion variants.
    boundary_passes = (
        tuple(item[0] for item in resolved_bounds),
        tuple(item[1] for item in resolved_bounds),
        tuple(item[2] for item in resolved_bounds),
    )
    for boundary_values in boundary_passes:
        for forced_index, forced_duration_ms in enumerate(boundary_values):
            if forced_duration_ms == primary[forced_index]:
                continue
            variant = _redistribute_candidate_route_duration(
                options,
                primary_durations_ms=primary,
                forced_index=forced_index,
                forced_duration_ms=forced_duration_ms,
            )
            if variant is None or variant in seen:
                continue
            seen.add(variant)
            variants.append(variant)
            if len(variants) >= max_variants:
                return tuple(variants)
    return tuple(variants)


def _candidate_source_interval(
    option: CandidateRouteOption,
    *,
    duration_ms: int,
) -> tuple[int | None, int | None]:
    feasible = _candidate_source_start_range(
        option,
        duration_ms=duration_ms,
    )
    if feasible is None:
        return None, None
    _, _, centered_start_ms = feasible
    return centered_start_ms, centered_start_ms + duration_ms


def _candidate_source_start_range(
    option: CandidateRouteOption,
    *,
    duration_ms: int,
) -> tuple[int, int, int] | None:
    """Return inclusive start bounds and the anchor-centered preferred start."""

    if (
        option.fixed_source_in_ms is not None
        and option.fixed_source_out_ms is not None
    ):
        if option.fixed_source_out_ms - option.fixed_source_in_ms != duration_ms:
            return None
        return (
            option.fixed_source_in_ms,
            option.fixed_source_in_ms,
            option.fixed_source_in_ms,
        )
    if (
        option.safe_window_start_ms is None
        or option.safe_window_end_ms is None
        or option.source_anchor_ms is None
    ):
        return None
    if duration_ms > option.safe_window_end_ms - option.safe_window_start_ms:
        return None
    # Every movable interval must retain the semantic/event anchor.  The
    # interval may slide anywhere inside the safe window; centering is merely
    # the preferred position, never an immutable trim.
    minimum_start_ms = max(
        option.safe_window_start_ms,
        # Source intervals are half-open.  An anchor exactly at source_out_ms
        # is not part of the execution and cannot satisfy semantic evidence.
        option.source_anchor_ms - duration_ms + 1,
    )
    maximum_start_ms = min(
        option.source_anchor_ms,
        option.safe_window_end_ms - duration_ms,
    )
    if minimum_start_ms > maximum_start_ms:
        return None
    centered_start_ms = min(
        max(
            minimum_start_ms,
            option.source_anchor_ms - duration_ms // 2,
        ),
        maximum_start_ms,
    )
    return minimum_start_ms, maximum_start_ms, centered_start_ms


def _ordered_interval_placement(
    *,
    left_range: tuple[int, int, int],
    left_duration_ms: int,
    right_range: tuple[int, int, int],
    permitted_overlap_ms: float,
) -> tuple[int, int] | None:
    """Minimally move centers so left overlap with right stays under a cap."""

    left_min, left_max, left_center = left_range
    right_min, right_max, right_center = right_range
    required_start_delta_ms = math.ceil(
        left_duration_ms - permitted_overlap_ms
    )
    if left_min + required_start_delta_ms > right_max:
        return None
    if left_center + required_start_delta_ms <= right_center:
        return left_center, right_center
    movement_needed_ms = (
        left_center + required_start_delta_ms - right_center
    )
    left_capacity_ms = left_center - left_min
    right_capacity_ms = right_max - right_center
    left_move_ms = min(
        left_capacity_ms,
        movement_needed_ms // 2,
    )
    right_move_ms = min(
        right_capacity_ms,
        movement_needed_ms - left_move_ms,
    )
    remaining_ms = (
        movement_needed_ms - left_move_ms - right_move_ms
    )
    if remaining_ms:
        extra_left_ms = min(
            left_capacity_ms - left_move_ms,
            remaining_ms,
        )
        left_move_ms += extra_left_ms
        remaining_ms -= extra_left_ms
    if remaining_ms:
        extra_right_ms = min(
            right_capacity_ms - right_move_ms,
            remaining_ms,
        )
        right_move_ms += extra_right_ms
        remaining_ms -= extra_right_ms
    if remaining_ms:
        return None
    left_start_ms = left_center - left_move_ms
    right_start_ms = right_center + right_move_ms
    if left_start_ms + required_start_delta_ms > right_start_ms:
        return None
    return left_start_ms, right_start_ms


def _joint_pair_interval_placement(
    *,
    first_range: tuple[int, int, int],
    first_duration_ms: int,
    second_range: tuple[int, int, int],
    second_duration_ms: int,
    permitted_overlap_ms: float,
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    first_then_second = _ordered_interval_placement(
        left_range=first_range,
        left_duration_ms=first_duration_ms,
        right_range=second_range,
        permitted_overlap_ms=permitted_overlap_ms,
    )
    if first_then_second is not None:
        candidates.append(first_then_second)
    second_then_first = _ordered_interval_placement(
        left_range=second_range,
        left_duration_ms=second_duration_ms,
        right_range=first_range,
        permitted_overlap_ms=permitted_overlap_ms,
    )
    if second_then_first is not None:
        second_start_ms, first_start_ms = second_then_first
        candidates.append((first_start_ms, second_start_ms))
    if not candidates:
        return None
    first_center = first_range[2]
    second_center = second_range[2]
    return min(
        candidates,
        key=lambda starts: (
            abs(starts[0] - first_center)
            + abs(starts[1] - second_center),
            max(
                abs(starts[0] - first_center),
                abs(starts[1] - second_center),
            ),
            starts,
        ),
    )


def _candidate_route_intervals(
    options: Sequence[CandidateRouteOption],
    durations_ms: Sequence[int],
    *,
    max_editorial_reprise_overlap_fraction: float,
) -> tuple[tuple[int | None, int | None], ...] | None:
    """Jointly place every timed use of a source inside its safe window.

    Reusing a source more than twice is an editorial decision, not an
    optimizer limitation.  The later use owns the typed reuse edge to every
    earlier use: ``none`` rejects it, ``distinct_interval`` forbids overlap,
    ``alternate_presentation`` permits overlap, and ``editorial_reprise``
    uses the policy-bound overlap allowance.  The search only considers range
    endpoints, preferred centers and exact overlap boundaries, so it remains
    small and deterministic without adding a general solver dependency.
    """

    ranges = [
        _candidate_source_start_range(option, duration_ms=duration_ms)
        for option, duration_ms in zip(options, durations_ms, strict=True)
    ]
    intervals: list[tuple[int | None, int | None]] = [
        (
            (start_range[2], start_range[2] + duration_ms)
            if start_range is not None
            else (None, None)
        )
        for start_range, duration_ms in zip(
            ranges,
            durations_ms,
            strict=True,
        )
    ]
    source_indices: dict[str, list[int]] = {}
    for index, option in enumerate(options):
        source_indices.setdefault(_source_identity(option), []).append(index)
    for indices in source_indices.values():
        timed_indices = [
            index for index in indices if ranges[index] is not None
        ]
        # Missing legacy timing retains the old soft-repeat behavior.  Exact
        # interval authority starts only once both uses are measurable.
        if len(timed_indices) < 2:
            continue
        constraints: dict[tuple[int, int], int] = {}
        for later_position, later_index in enumerate(timed_indices[1:], start=1):
            later_option = options[later_index]
            if later_option.reuse_mode == "none":
                return None
            for earlier_index in timed_indices[:later_position]:
                if later_option.reuse_mode == "alternate_presentation":
                    continue
                permitted_overlap_ms = 0
                if later_option.reuse_mode == "editorial_reprise":
                    permitted_overlap_ms = math.floor(
                        durations_ms[later_index]
                        * max_editorial_reprise_overlap_fraction
                    )
                constraints[(earlier_index, later_index)] = permitted_overlap_ms

        candidate_starts: dict[int, set[int]] = {
            index: {
                ranges[index][0],  # type: ignore[index]
                ranges[index][1],  # type: ignore[index]
                ranges[index][2],  # type: ignore[index]
            }
            for index in timed_indices
        }
        # A feasible region can only start/end at a safe boundary, preferred
        # center, or an exact pairwise overlap boundary. Close the finite set
        # under those boundaries before the bounded backtracking pass.
        for _ in range(len(timed_indices) + 1):
            changed = False
            for (first_index, second_index), permitted_overlap_ms in constraints.items():
                first_range = ranges[first_index]
                second_range = ranges[second_index]
                assert first_range is not None
                assert second_range is not None
                for first_start in tuple(candidate_starts[first_index]):
                    for candidate_start in (
                        first_start + permitted_overlap_ms - durations_ms[second_index],
                        first_start + durations_ms[first_index] - permitted_overlap_ms,
                    ):
                        if second_range[0] <= candidate_start <= second_range[1] and candidate_start not in candidate_starts[second_index]:
                            candidate_starts[second_index].add(candidate_start)
                            changed = True
                for second_start in tuple(candidate_starts[second_index]):
                    for candidate_start in (
                        second_start + permitted_overlap_ms - durations_ms[first_index],
                        second_start + durations_ms[second_index] - permitted_overlap_ms,
                    ):
                        if first_range[0] <= candidate_start <= first_range[1] and candidate_start not in candidate_starts[first_index]:
                            candidate_starts[first_index].add(candidate_start)
                            changed = True
            if not changed:
                break

        ordered_indices = sorted(
            timed_indices,
            key=lambda index: (
                len(candidate_starts[index]),
                index,
            ),
        )
        best: tuple[int, tuple[int, ...], dict[int, int]] | None = None

        def overlap_ms(first_index: int, first_start: int, second_index: int, second_start: int) -> int:
            return max(
                0,
                min(
                    first_start + durations_ms[first_index],
                    second_start + durations_ms[second_index],
                )
                - max(first_start, second_start),
            )

        def search(position: int, chosen: dict[int, int], displacement: int) -> None:
            nonlocal best
            if best is not None and displacement > best[0]:
                return
            if position == len(ordered_indices):
                ordered_starts = tuple(chosen[index] for index in timed_indices)
                candidate = (displacement, ordered_starts, dict(chosen))
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
                return
            index = ordered_indices[position]
            current_range = ranges[index]
            assert current_range is not None
            for start_ms in sorted(
                candidate_starts[index],
                key=lambda value: (abs(value - current_range[2]), value),
            ):
                valid = True
                for other_index, other_start_ms in chosen.items():
                    key = (
                        (other_index, index)
                        if other_index < index
                        else (index, other_index)
                    )
                    permitted_overlap_ms = constraints.get(key)
                    if permitted_overlap_ms is not None and overlap_ms(
                        index,
                        start_ms,
                        other_index,
                        other_start_ms,
                    ) > permitted_overlap_ms:
                        valid = False
                        break
                if not valid:
                    continue
                chosen[index] = start_ms
                search(
                    position + 1,
                    chosen,
                    displacement + abs(start_ms - current_range[2]),
                )
                chosen.pop(index)

        search(0, {}, 0)
        if best is None:
            return None
        for index, start_ms in best[2].items():
            intervals[index] = (
                start_ms,
                start_ms + durations_ms[index],
            )
    return tuple(intervals)


def candidate_route_execution_sha256(
    option: CandidateRouteOption,
    *,
    duration_ms: int,
    source_in_ms: int | None,
    source_out_ms: int | None,
) -> str:
    """Hash the exact candidate execution, not just its semantic identity."""

    payload = {
        "contract_version": "candidate-route-execution-v1",
        "beat_id": option.beat_id,
        "candidate_id": option.candidate_id,
        "source_asset_id": option.source_asset_id,
        "source_clip_id": option.source_clip_id,
        "event_id": option.event_id,
        "candidate_timing_sha256": option.candidate_timing_sha256,
        "duration_ms": duration_ms,
        "source_in_ms": source_in_ms,
        "source_out_ms": source_out_ms,
        "cue_id": option.cue_id,
        "presentation_mode": option.presentation_mode,
        "reuse_mode": option.reuse_mode,
        "reuse_justification": option.reuse_justification,
        "entry_composition": option.entry_composition,
        "exit_composition": option.exit_composition,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _candidate_route_selection(
    option: CandidateRouteOption,
    *,
    duration_ms: int,
    decision_codes: tuple[str, ...],
    source_interval: tuple[int | None, int | None] | None = None,
) -> CandidateRouteSelection:
    if source_interval is None:
        source_in_ms, source_out_ms = _candidate_source_interval(
            option,
            duration_ms=duration_ms,
        )
    else:
        source_in_ms, source_out_ms = source_interval
    return CandidateRouteSelection(
        beat_id=option.beat_id,
        candidate_id=option.candidate_id,
        source_asset_id=option.source_asset_id,
        event_id=option.event_id,
        trim_duration_ms=duration_ms,
        cue_id=option.cue_id,
        cue_aligned=option.cue_aligned,
        presentation_mode=option.presentation_mode,
        entry_composition=option.entry_composition,
        exit_composition=option.exit_composition,
        decision_codes=decision_codes,
        source_clip_id=option.source_clip_id,
        source_in_ms=source_in_ms,
        source_out_ms=source_out_ms,
        candidate_execution_sha256=candidate_route_execution_sha256(
            option,
            duration_ms=duration_ms,
            source_in_ms=source_in_ms,
            source_out_ms=source_out_ms,
        ),
        reuse_mode=option.reuse_mode,
        reuse_justification=option.reuse_justification,
    )


def _source_identity(option: CandidateRouteOption) -> str:
    return option.source_clip_id or option.source_asset_id


def _candidate_reuse_failure(
    prior: Sequence[tuple[CandidateRouteOption, CandidateRouteSelection]],
    option: CandidateRouteOption,
    selection: CandidateRouteSelection,
    *,
    max_editorial_reprise_overlap_fraction: float,
) -> str | None:
    for prior_option, prior_selection in prior:
        if _source_identity(prior_option) != _source_identity(option):
            continue
        # Legacy options did not persist candidate-local timing.  Preserve
        # their old soft-repeat behavior instead of manufacturing intervals.
        if (
            prior_selection.source_in_ms is None
            or prior_selection.source_out_ms is None
            or selection.source_in_ms is None
            or selection.source_out_ms is None
        ):
            continue
        overlap_ms = max(
            0,
            min(prior_selection.source_out_ms, selection.source_out_ms)
            - max(prior_selection.source_in_ms, selection.source_in_ms),
        )
        if option.reuse_mode == "none":
            return "source_reuse_authority_missing"
        if option.reuse_mode == "distinct_interval" and overlap_ms:
            return "distinct_interval_reuse_overlaps"
        if option.reuse_mode == "editorial_reprise":
            overlap_fraction = overlap_ms / max(
                selection.trim_duration_ms,
                1,
            )
            if (
                overlap_fraction
                > max_editorial_reprise_overlap_fraction + 1e-9
            ):
                return "editorial_reprise_overlap_exceeded"
        # alternate_presentation is explicit authority for the same interval.
    return None


def _candidate_complete_route(
    state: _CandidateRouteState,
    *,
    target_duration_ms: int | None,
    max_panel_runtime_fraction: float | None,
    max_editorial_reprise_overlap_fraction: float,
    durations_ms: Sequence[int] | None = None,
    primary_durations_ms: Sequence[int] | None = None,
) -> CandidateCompleteRoute | None:
    durations = (
        tuple(durations_ms)
        if durations_ms is not None
        else _allocate_candidate_route_durations(
            state.options,
            target_duration_ms=target_duration_ms,
        )
    )
    if durations is None:
        return None
    source_intervals = _candidate_route_intervals(
        state.options,
        durations,
        max_editorial_reprise_overlap_fraction=(
            max_editorial_reprise_overlap_fraction
        ),
    )
    if source_intervals is None:
        return None
    selections: list[CandidateRouteSelection] = []
    prior: list[tuple[CandidateRouteOption, CandidateRouteSelection]] = []
    panel_duration_ms = 0
    repositioned = False
    for option, duration_ms, source_interval, provisional in zip(
        state.options,
        durations,
        source_intervals,
        state.selections,
        strict=True,
    ):
        centered_interval = _candidate_source_interval(
            option,
            duration_ms=duration_ms,
        )
        selection_codes = provisional.decision_codes
        if source_interval != centered_interval:
            repositioned = True
            selection_codes = (
                *selection_codes,
                "source_interval_jointly_repositioned",
            )
        selection = _candidate_route_selection(
            option,
            duration_ms=duration_ms,
            decision_codes=selection_codes,
            source_interval=source_interval,
        )
        if _candidate_reuse_failure(
            prior,
            option,
            selection,
            max_editorial_reprise_overlap_fraction=(
                max_editorial_reprise_overlap_fraction
            ),
        ):
            return None
        selections.append(selection)
        prior.append((option, selection))
        if option.presentation_mode == "two_panel_layout":
            panel_duration_ms += duration_ms
    total_duration_ms = sum(durations)
    if (
        max_panel_runtime_fraction is not None
        and panel_duration_ms / max(total_duration_ms, 1)
        > max_panel_runtime_fraction + 1e-9
    ):
        return None
    route_payload = {
        "contract_version": "candidate-complete-route-v1",
        "candidate_execution_sha256s": [
            selection.candidate_execution_sha256
            for selection in selections
        ],
    }
    route_id = hashlib.sha256(
        json.dumps(
            route_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    duration_deviation_ms = (
        sum(
            abs(duration_ms - primary_duration_ms)
            for duration_ms, primary_duration_ms in zip(
                durations,
                primary_durations_ms,
                strict=True,
            )
        )
        if primary_durations_ms is not None
        else 0
    )
    return CandidateCompleteRoute(
        route_id=route_id,
        selections=tuple(selections),
        objective_score=round(
            state.score
            - duration_deviation_ms
            / max(total_duration_ms, 1)
            * 0.05,
            6,
        ),
        total_duration_ms=total_duration_ms,
        panel_duration_ms=panel_duration_ms,
        decision_codes=(
            "complete_route_ranked_before_runtime",
            "candidate_local_timing_bound",
            "source_reuse_hard_checked",
            *(
                ("bounded_duration_recovery_variant",)
                if duration_deviation_ms
                else ()
            ),
            *(
                ("source_intervals_jointly_placed",)
                if repositioned
                else ()
            ),
        ),
    )


def _semantic_replan_bindings(
    beats: Sequence[CandidateRouteBeat],
    *,
    primary_route: CandidateCompleteRoute,
    max_editorial_reprise_overlap_fraction: float,
) -> dict[str, tuple[SemanticReplanCandidateBinding, ...]]:
    """Retain safe candidates that need Gemini's whole-route judgement.

    This is intentionally conservative: candidates with preflight hard
    failures never enter the semantic set.  A possible source-reuse conflict
    is reported as a requirement rather than locally solved; the later
    replan/rebuild must still prove the final intervals and all project gates.
    """

    option_by_beat_and_id = {
        (beat.beat_id, option.candidate_id): option
        for beat in beats
        for option in beat.options
    }
    bindings: dict[str, tuple[SemanticReplanCandidateBinding, ...]] = {}
    for beat in beats:
        rows: list[SemanticReplanCandidateBinding] = []
        for option in beat.options:
            if option.preflight_hard_failures:
                continue
            candidate_selection = _candidate_route_selection(
                option,
                duration_ms=option.trim_duration_ms,
                decision_codes=("semantic_replan_candidate_projection",),
            )
            prior: list[tuple[CandidateRouteOption, CandidateRouteSelection]] = []
            for selection in primary_route.selections:
                if selection.beat_id == beat.beat_id:
                    continue
                selected_option = option_by_beat_and_id.get(
                    (selection.beat_id, selection.candidate_id)
                )
                if selected_option is not None:
                    prior.append((selected_option, selection))
            reuse_failure = _candidate_reuse_failure(
                prior,
                option,
                candidate_selection,
                max_editorial_reprise_overlap_fraction=(
                    max_editorial_reprise_overlap_fraction
                ),
            )
            rows.append(
                SemanticReplanCandidateBinding(
                    option=option,
                    replan_required_codes=(
                        (reuse_failure,) if reuse_failure is not None else ()
                    ),
                )
            )
        bindings[beat.beat_id] = tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.option.planner_rank,
                    row.option.candidate_id,
                ),
            )
        )
    return bindings


def optimize_pre_render_candidate_route(
    beats: Sequence[CandidateRouteBeat],
    *,
    beam_width: int = 64,
    minimum_duration_ms: int | None = None,
    maximum_duration_ms: int | None = None,
    max_panel_runtime_fraction: float | None = None,
    target_duration_ms: int | None = None,
    max_editorial_reprise_overlap_fraction: float = 0.5,
    max_ranked_routes: int | None = None,
    max_duration_variants_per_route: int = 16,
) -> CandidateRouteResult:
    """Build an executable frontier without making editorial substitutions.

    Gemini ranks source/event/presentation candidates.  This routine may
    remove only candidates with a deterministic hard failure, bind legal
    source intervals and validate project limits.  When more than one complete
    route remains, its primary route is therefore the lexicographic Gemini
    rank vector, never a local blend of confidence, technical quality,
    apparent variety or presentation intrusion.  Those measurements stay in
    the artifacts for a bounded semantic replan; they are not local creative
    authority.
    """

    if beam_width < 1:
        raise ValueError("candidate route beam width must be positive")
    if (minimum_duration_ms is None) != (maximum_duration_ms is None):
        raise ValueError(
            "pre-render duration bounds must be supplied together"
        )
    if (
        minimum_duration_ms is not None
        and maximum_duration_ms is not None
        and (
            minimum_duration_ms < 1
            or maximum_duration_ms < minimum_duration_ms
        )
    ):
        raise ValueError("pre-render duration bounds are invalid")
    if max_panel_runtime_fraction is not None and not (
        0 <= max_panel_runtime_fraction <= 1
    ):
        raise ValueError(
            "pre-render panel runtime fraction must be between 0 and 1"
        )
    if target_duration_ms is not None and target_duration_ms < 1:
        raise ValueError("pre-render target duration must be positive")
    if (
        target_duration_ms is not None
        and minimum_duration_ms is not None
        and maximum_duration_ms is not None
        and not minimum_duration_ms
        <= target_duration_ms
        <= maximum_duration_ms
    ):
        raise ValueError("pre-render target duration is outside policy bounds")
    if not 0 <= max_editorial_reprise_overlap_fraction <= 1:
        raise ValueError(
            "editorial reprise overlap fraction must be between 0 and 1"
        )
    if max_ranked_routes is not None and max_ranked_routes < 1:
        raise ValueError("max ranked routes must be positive")
    if max_duration_variants_per_route < 1:
        raise ValueError(
            "max duration variants per route must be positive"
        )
    states = [_CandidateRouteState()]
    for beat in beats:
        feasible_options = [
            option
            for option in beat.options
            if not option.preflight_hard_failures
        ]
        if not feasible_options:
            failures = sorted(
                {
                    failure
                    for option in beat.options
                    for failure in option.preflight_hard_failures
                }
                or {"no_pre_render_candidate_options"}
            )
            raise ValueError(
                "pre-render sequence frontier has no hard-safe option for "
                f"{beat.beat_id}: {','.join(failures)}"
            )
        expanded: list[_CandidateRouteState] = []
        for state in states:
            for option in feasible_options:
                provisional_options = state.options + (option,)
                provisional_durations = tuple(
                    (
                        _candidate_duration_bounds(candidate)
                        or (0, 0, 0)
                    )[0]
                    for candidate in provisional_options
                )
                if any(duration < 1 for duration in provisional_durations):
                    continue
                # Centered trims are preferences.  The early hard gate solves
                # the full feasible start ranges, so a movable distinct
                # interval is not discarded merely because centers overlap.
                if _candidate_route_intervals(
                    provisional_options,
                    provisional_durations,
                    max_editorial_reprise_overlap_fraction=(
                        max_editorial_reprise_overlap_fraction
                    ),
                ) is None:
                    continue
                # This is deliberately *not* an aesthetic score.  Source
                # variety, confidence, technical quality and composition are
                # useful observations, but using them to promote an otherwise
                # feasible rank-2 candidate would turn the executor into a
                # second editor.  Reuse/identity/geometry constraints remain
                # hard gates elsewhere in this route construction.
                score = state.score - option.planner_rank
                codes = ["gemini_candidate_rank_preserved"]
                codes.extend(
                    (
                        "resolved_trim_bound_before_render",
                        "music_exit_bound_before_render",
                        "presentation_family_bound_before_render",
                        "entry_exit_composition_bound_before_render",
                        "runtime_hard_gates_remain_fail_closed",
                    )
                )
                expanded.append(
                    _CandidateRouteState(
                        selections=state.selections
                        + (
                            _candidate_route_selection(
                                option,
                                duration_ms=option.trim_duration_ms,
                                decision_codes=tuple(codes),
                            ),
                        ),
                        options=state.options + (option,),
                        source_asset_ids=state.source_asset_ids
                        + (option.source_asset_id,),
                        source_events=state.source_events
                        + ((option.source_asset_id, option.event_id),),
                        exit_composition=option.exit_composition,
                        duration_ms=state.duration_ms + option.trim_duration_ms,
                        panel_duration_ms=(
                            state.panel_duration_ms
                            + (
                                option.trim_duration_ms
                                if option.presentation_mode
                                == "two_panel_layout"
                                else 0
                            )
                        ),
                        score=score,
                    )
                )
        if not expanded:
            raise ValueError(
                f"candidate route has no options for beat {beat.beat_id}"
            )
        states = sorted(
            expanded,
            key=lambda state: (
                tuple(option.planner_rank for option in state.options),
                tuple(
                    selection.candidate_id
                    for selection in state.selections
                ),
            ),
        )[:beam_width]
    ranked_route_states: list[
        tuple[CandidateCompleteRoute, _CandidateRouteState]
    ] = []
    for state in states:
        duration_variants = _candidate_route_duration_variants(
            state.options,
            target_duration_ms=target_duration_ms,
            max_variants=max_duration_variants_per_route,
        )
        primary_durations_ms = (
            duration_variants[0] if duration_variants else None
        )
        for durations_ms in duration_variants:
            route = _candidate_complete_route(
                state,
                target_duration_ms=target_duration_ms,
                max_panel_runtime_fraction=max_panel_runtime_fraction,
                max_editorial_reprise_overlap_fraction=(
                    max_editorial_reprise_overlap_fraction
                ),
                durations_ms=durations_ms,
                primary_durations_ms=primary_durations_ms,
            )
            if route is None:
                continue
            if (
                minimum_duration_ms is not None
                and maximum_duration_ms is not None
                and not minimum_duration_ms
                <= route.total_duration_ms
                <= maximum_duration_ms
            ):
                continue
            ranked_route_states.append((route, state))
    ranked_route_states.sort(
        key=lambda item: (
            tuple(
                option.planner_rank for option in item[1].options
            ),
            tuple(
                selection.candidate_id
                for selection in item[0].selections
            ),
        ),
    )
    if max_ranked_routes is not None:
        ranked_route_states = ranked_route_states[:max_ranked_routes]
    if not ranked_route_states:
        raise ValueError(
            "pre-render sequence frontier has no route inside duration and "
            "panel-runtime policy bounds"
        )
    best_route, best_state = ranked_route_states[0]
    selected_by_beat = {
        selection.beat_id: selection.candidate_id
        for selection in best_route.selections
    }
    fallback_candidate_ids_by_beat: dict[str, tuple[str, ...]] = {}
    option_bindings_by_beat: dict[
        str,
        dict[str, CandidateRouteOption],
    ] = {}
    for beat in beats:
        selected_candidate_id = selected_by_beat[beat.beat_id]
        route_compatible_candidate_ids: list[str] = []
        for complete_route, _state in ranked_route_states:
            route_by_beat = {
                selection.beat_id: selection
                for selection in complete_route.selections
            }
            candidate_id = route_by_beat[beat.beat_id].candidate_id
            if candidate_id not in route_compatible_candidate_ids:
                route_compatible_candidate_ids.append(candidate_id)
        legal = [
            option
            for option in beat.options
            if not option.preflight_hard_failures
            and option.candidate_id in route_compatible_candidate_ids
        ]
        # Every candidate that occurs in a ranked complete route must retain
        # its own timing binding.  A global alternate is not a hypothetical
        # one-beat substitution: its source interval and duration can be
        # jointly feasible only with the other selections in *that* route.
        # Dropping it here previously produced an execution binding without
        # the candidate-local safe-window fact required by strict preflight.
        contextual_legal = list(legal)
        # The primary comes from Gemini's rank vector. Remaining options are
        # present only because at least one saved complete route proved them
        # jointly feasible; runtime never invents an unranked fallback or
        # locally promotes one for variety/quality reasons.
        alternatives = sorted(
            (
                option
                for option in contextual_legal
                if option.candidate_id != selected_candidate_id
            ),
            key=lambda option: (
                option.planner_rank,
                option.candidate_id,
            ),
        )
        fallback_candidate_ids_by_beat[beat.beat_id] = (
            selected_candidate_id,
            *(option.candidate_id for option in alternatives),
        )
        option_bindings_by_beat[beat.beat_id] = {
            option.candidate_id: option for option in contextual_legal
        }
    return CandidateRouteResult(
        selections=best_route.selections,
        fallback_candidate_ids_by_beat=fallback_candidate_ids_by_beat,
        option_bindings_by_beat=option_bindings_by_beat,
        objective_score=best_route.objective_score,
        beam_width=beam_width,
        total_duration_ms=best_route.total_duration_ms,
        ranked_routes=tuple(
            route for route, _ in ranked_route_states
        ),
        semantic_replan_candidate_bindings_by_beat=(
            _semantic_replan_bindings(
                beats,
                primary_route=best_route,
                max_editorial_reprise_overlap_fraction=(
                    max_editorial_reprise_overlap_fraction
                ),
            )
        ),
    )


def rebuild_route_with_semantic_authorities(
    frontier: CandidateRouteResult,
    *,
    selected_candidate_ids_by_beat: Mapping[str, str],
    reuse_authorities_by_beat: Mapping[str, SemanticReplanReuseAuthority],
    minimum_duration_ms: int | None = None,
    maximum_duration_ms: int | None = None,
    max_panel_runtime_fraction: float | None = None,
    target_duration_ms: int | None = None,
    max_editorial_reprise_overlap_fraction: float = 0.5,
) -> CandidateCompleteRoute:
    """Rebuild one full route after Gemini makes a bounded replan decision.

    No existing ``CandidateRouteSelection`` is patched.  Gemini may select
    only one retained candidate per affected beat and may add a typed reuse
    authority for that exact candidate.  The whole route is then re-solved so
    timing, source intervals, panel runtime and reuse constraints are proved
    together before any paid exact-event or grounding stage is admitted.
    """

    known_beats = tuple(frontier.semantic_replan_candidate_bindings_by_beat)
    if set(selected_candidate_ids_by_beat) - set(known_beats):
        raise ValueError("semantic replan selected an unknown beat")
    if set(reuse_authorities_by_beat) - set(known_beats):
        raise ValueError("semantic replan reuse authority references an unknown beat")

    primary_by_beat = {
        selection.beat_id: selection.candidate_id
        for selection in frontier.selections
    }
    rebuilt_beats: list[CandidateRouteBeat] = []
    for beat_id in known_beats:
        candidate_id = selected_candidate_ids_by_beat.get(
            beat_id,
            primary_by_beat.get(beat_id),
        )
        if candidate_id is None:
            raise ValueError("semantic replan has no primary candidate for " + beat_id)
        binding = next(
            (
                row
                for row in frontier.semantic_replan_candidate_bindings_by_beat[
                    beat_id
                ]
                if row.option.candidate_id == candidate_id
            ),
            None,
        )
        if binding is None:
            raise ValueError(
                "semantic replan selected a candidate outside its immutable "
                "candidate bindings: " + beat_id + ":" + candidate_id
            )
        if binding.replan_required_codes and any(
            code != "source_reuse_authority_missing"
            for code in binding.replan_required_codes
        ):
            raise ValueError(
                "semantic replan candidate retains a non-authorizable hard "
                "conflict: " + ",".join(binding.replan_required_codes)
            )
        authority = reuse_authorities_by_beat.get(beat_id)
        option = binding.option
        if authority is not None:
            if authority.candidate_id != candidate_id:
                raise ValueError(
                    "semantic reuse authority does not bind selected candidate"
                )
            if option.reuse_mode != "none":
                raise ValueError(
                    "semantic replan cannot replace existing candidate reuse authority"
                )
            option = option.model_copy(
                update={
                    "reuse_mode": authority.reuse_mode,
                    "reuse_justification": authority.reuse_justification,
                }
            )
        elif "source_reuse_authority_missing" in binding.replan_required_codes:
            raise ValueError(
                "semantic replan selected a candidate that requires explicit "
                "reuse authority"
            )
        rebuilt_beats.append(
            CandidateRouteBeat(beat_id=beat_id, options=(option,))
        )

    rebuilt = optimize_pre_render_candidate_route(
        tuple(rebuilt_beats),
        beam_width=1,
        minimum_duration_ms=minimum_duration_ms,
        maximum_duration_ms=maximum_duration_ms,
        max_panel_runtime_fraction=max_panel_runtime_fraction,
        target_duration_ms=target_duration_ms,
        max_editorial_reprise_overlap_fraction=(
            max_editorial_reprise_overlap_fraction
        ),
        max_ranked_routes=1,
    )
    route = rebuilt.ranked_routes[0]
    by_beat = {selection.beat_id: selection for selection in route.selections}
    for beat_id, authority in reuse_authorities_by_beat.items():
        current = by_beat[beat_id]
        source_beats = tuple(
            selection.beat_id
            for selection in route.selections
            if selection.beat_id != beat_id
            and selection.source_clip_id is not None
            and selection.source_clip_id == current.source_clip_id
        )
        if tuple(sorted(source_beats)) != tuple(sorted(authority.reuse_of_beat_ids)):
            raise ValueError(
                "semantic reuse authority does not name every reused source beat"
            )
    return route


def select_next_compatible_route(
    ranked_routes: Sequence[CandidateCompleteRoute],
    *,
    accepted_execution_sha256_by_beat: Mapping[str, str],
    failed_execution_sha256s: Sequence[str] = (),
    unavailable_execution_sha256s: Sequence[str] = (),
    after_route_id: str | None = None,
) -> CandidateCompleteRoute | None:
    """Advance only to a complete route compatible with accepted executions."""

    unavailable = {
        *failed_execution_sha256s,
        *unavailable_execution_sha256s,
    }
    after_seen = after_route_id is None
    for route in ranked_routes:
        if not after_seen:
            if route.route_id == after_route_id:
                after_seen = True
            continue
        by_beat = {
            selection.beat_id: selection.candidate_execution_sha256
            for selection in route.selections
        }
        if any(
            selection.candidate_execution_sha256 in unavailable
            for selection in route.selections
        ):
            continue
        if any(
            by_beat.get(beat_id) != execution_sha256
            for beat_id, execution_sha256
            in accepted_execution_sha256_by_beat.items()
        ):
            continue
        return route
    return None


class ConstraintResult(FrozenStrictModel):
    """Typed, evidence-carrying result shared by every editing capability."""

    constraint_id: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_.:-]*$",
    )
    level: Literal["hard", "preference"]
    status: Literal["pass", "fail", "unknown"]
    evidence_refs: tuple[str, ...] = ()
    measured_value: float | int | bool | str | None = None
    threshold: float | int | bool | str | None = None
    reason_code: str = Field(min_length=1)


class OptionMetrics(FrozenStrictModel):
    semantic_fit: float = Field(default=0.5, ge=0.0, le=1.0)
    readability: float = Field(default=0.5, ge=0.0, le=1.0)
    technical_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    music_flow: float = Field(default=0.5, ge=0.0, le=1.0)
    synthetic_motion_distance: float = Field(default=0.0, ge=0.0)
    intrusion_rank: int = Field(default=0, ge=0, le=10)
    local_cost_rank: int = Field(default=0, ge=0, le=10)


class ExecutableOptionV2(FrozenStrictModel):
    """Capability-neutral option. Payload stays in the owning executor."""

    option_id: str = Field(min_length=1)
    capability_id: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_hashes: tuple[str, ...] = ()
    constraints: tuple[ConstraintResult, ...] = Field(min_length=1)
    metrics: OptionMetrics = OptionMetrics()
    decision_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_constraints(self) -> "ExecutableOptionV2":
        ids = [item.constraint_id for item in self.constraints]
        if len(ids) != len(set(ids)):
            raise ValueError("option constraint IDs must be unique")
        return self

    @property
    def hard_failure_codes(self) -> tuple[str, ...]:
        # Unknown hard facts fail closed. A preference may remain unknown and
        # merely lower confidence; it can never authorize delivery.
        return tuple(
            result.reason_code
            for result in self.constraints
            if result.level == "hard" and result.status != "pass"
        )


class ExecutableOptionSelectionV2(FrozenStrictModel):
    contract_version: Literal["executable-option-selection-v2"] = (
        "executable-option-selection-v2"
    )
    selected_option_id: str | None
    generated_option_ids: tuple[str, ...]
    generated_capabilities: Mapping[str, str] = Field(default_factory=dict)
    rejected_options: Mapping[str, tuple[str, ...]]
    score: float | None
    runner_up_option_ids: tuple[str, ...] = ()
    score_gap: float | None = None
    semantic_negotiation_recommended: bool = False
    semantic_ambiguity_codes: tuple[str, ...] = ()
    decision_codes: tuple[str, ...]


def select_executable_option(
    options: Sequence[ExecutableOptionV2],
    *,
    preferred_capability_ids: Sequence[str] = (),
) -> ExecutableOptionSelectionV2:
    """Select one legal operation with hard-first, bounded minimality.

    The local compiler must not let a high aesthetic score compensate for a
    more intrusive construction.  Planner preferences order operations inside
    the same natural/constructed tier; they never promote panel or matte above
    a hard-safe single-canvas presentation.
    """

    generated = tuple(option.option_id for option in options)
    if len(generated) != len(set(generated)):
        raise ValueError("executable option IDs must be unique")
    rejected = {
        option.option_id: option.hard_failure_codes
        for option in options
        if option.hard_failure_codes
    }
    generated_capabilities = {
        option.option_id: option.capability_id for option in options
    }
    legal = [option for option in options if not option.hard_failure_codes]
    if not legal:
        return ExecutableOptionSelectionV2(
            selected_option_id=None,
            generated_option_ids=generated,
            generated_capabilities=generated_capabilities,
            rejected_options=rejected,
            score=None,
            decision_codes=("no_hard_constraint_safe_option",),
        )
    preference_rank = {
        capability_id: index
        for index, capability_id in enumerate(preferred_capability_ids)
    }

    def construction_tier(option: ExecutableOptionV2) -> int:
        # Ranks below four are single-canvas editorial operations.  Panel and
        # matte are constructed recovery/presentation families and may only
        # win after the lower tier has no hard-safe option.
        return 0 if option.metrics.intrusion_rank < 4 else 1

    def preference_tier(option: ExecutableOptionV2) -> int:
        return preference_rank.get(
            option.capability_id,
            len(preference_rank) + 1,
        )

    def rank(
        option: ExecutableOptionV2,
    ) -> tuple[int, int, int, float, float, float, float, int, str]:
        metrics = option.metrics
        return (
            construction_tier(option),
            preference_tier(option),
            metrics.intrusion_rank,
            -metrics.semantic_fit,
            -metrics.readability,
            -metrics.technical_quality,
            metrics.synthetic_motion_distance,
            metrics.local_cost_rank,
            option.option_id,
        )

    ranked = sorted(legal, key=rank)
    selected = ranked[0]
    selected_score = (
        selected.metrics.semantic_fit * 0.4
        + selected.metrics.readability * 0.35
        + selected.metrics.technical_quality * 0.25
    )
    runners = tuple(option.option_id for option in ranked[1:3])
    runner_score = (
        ranked[1].metrics.semantic_fit * 0.4
        + ranked[1].metrics.readability * 0.35
        + ranked[1].metrics.technical_quality * 0.25
        if len(ranked) >= 2
        else None
    )
    score_gap = (
        selected_score - runner_score
        if runner_score is not None
        else None
    )
    semantically_ambiguous = bool(
        len(ranked) >= 2
        and score_gap is not None
        and construction_tier(selected) == construction_tier(ranked[1])
        and preference_tier(selected) == preference_tier(ranked[1])
        and abs(score_gap) <= 0.04
        and selected.capability_id != ranked[1].capability_id
    )
    return ExecutableOptionSelectionV2(
        selected_option_id=selected.option_id,
        generated_option_ids=generated,
        generated_capabilities=generated_capabilities,
        rejected_options=rejected,
        score=round(selected_score, 6),
        runner_up_option_ids=runners,
        score_gap=round(score_gap, 6) if score_gap is not None else None,
        semantic_negotiation_recommended=semantically_ambiguous,
        semantic_ambiguity_codes=(
            ("different_capabilities_have_near_equal_local_scores",)
            if semantically_ambiguous
            else ()
        ),
        decision_codes=(
            "hard_constraints_passed",
            "single_canvas_minimality_applied",
            "lexicographic_hard_first_selection",
        ),
    )


class SequenceOption(FrozenStrictModel):
    """One executable candidate/trim/cue/presentation choice for a beat."""

    option_id: str = Field(min_length=1)
    beat_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    source_asset_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_in_pts: int = Field(ge=0)
    source_out_pts: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    minimum_readable_ms: int = Field(gt=0)
    preferred_readable_ms: int = Field(gt=0)
    maximum_readable_ms: int = Field(gt=0)
    cue_id: str = Field(min_length=1)
    cue_delta_frames: int
    cue_tolerance_frames: int = Field(ge=0)
    presentation_mode: PresentationMode
    presentation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tracking_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entry_x: float | None = Field(default=None, ge=0.0, le=1.0)
    exit_x: float | None = Field(default=None, ge=0.0, le=1.0)
    hard_evidence_satisfied: bool
    identity_satisfied: bool
    action_complete: bool
    required_relation_satisfied: bool
    relative_scale_satisfied: bool
    quality_safe: bool
    reuse_authorized: bool
    geometry_executable: bool
    legal_musical_exit: bool
    semantic_fit: float = Field(ge=0.0, le=1.0)
    readability: float = Field(ge=0.0, le=1.0)
    technical_quality: float = Field(ge=0.0, le=1.0)
    music_flow: float = Field(ge=0.0, le=1.0)
    synthetic_motion_distance: float = Field(default=0.0, ge=0.0)
    authorized_reprise: bool = False
    freeze_event_lock_id: str | None = None

    @model_validator(mode="after")
    def validate_option(self) -> "SequenceOption":
        if self.source_out_pts <= self.source_in_pts:
            raise ValueError("source interval must be non-empty")
        if not (
            self.minimum_readable_ms
            <= self.preferred_readable_ms
            <= self.maximum_readable_ms
        ):
            raise ValueError("readability bounds must be ordered")
        if not (
            self.minimum_readable_ms
            <= self.duration_ms
            <= self.maximum_readable_ms
        ):
            raise ValueError("duration must remain inside readability bounds")
        if (
            self.presentation_mode == "intentional_freeze"
            and self.freeze_event_lock_id is None
        ):
            raise ValueError("intentional freeze requires an exact event lock")
        return self

    def hard_constraint_failures(self, priority: BeatPriority) -> tuple[str, ...]:
        failures: list[str] = []
        checks = (
            ("identity_failed", self.identity_satisfied),
            ("action_incomplete", self.action_complete),
            ("required_relation_failed", self.required_relation_satisfied),
            ("relative_scale_failed", self.relative_scale_satisfied),
            ("quality_unsafe", self.quality_safe),
            (
                "music_hard_sync_failed",
                abs(self.cue_delta_frames) <= self.cue_tolerance_frames,
            ),
            ("reuse_unauthorized", self.reuse_authorized),
            ("geometry_infeasible", self.geometry_executable),
            ("musical_exit_illegal", self.legal_musical_exit),
        )
        if priority == "hard" and not self.hard_evidence_satisfied:
            failures.append("hard_evidence_missing")
        failures.extend(code for code, passed in checks if not passed)
        return tuple(failures)


class BeatOptionSet(FrozenStrictModel):
    beat_id: str = Field(min_length=1)
    priority: BeatPriority
    options: tuple[SequenceOption, ...]

    @model_validator(mode="after")
    def validate_options(self) -> "BeatOptionSet":
        if any(option.beat_id != self.beat_id for option in self.options):
            raise ValueError("all options must belong to the declared beat")
        return self


class SequenceSelection(FrozenStrictModel):
    beat_id: str
    priority: BeatPriority
    option: SequenceOption | None
    decision_codes: tuple[str, ...]


class SequenceOptimizationResult(FrozenStrictModel):
    contract_version: Literal["sequence-optimization-v1"] = (
        "sequence-optimization-v1"
    )
    selections: tuple[SequenceSelection, ...]
    total_duration_ms: int = Field(ge=0)
    target_duration_ms: int = Field(gt=0)
    duration_delta_ms: int
    score: float
    outcome: Literal["complete", "best_effort_shortened", "blocked"]
    omitted_beat_ids: tuple[str, ...] = ()
    hard_failure_codes: tuple[str, ...] = ()


class MusicBoundaryCue(FrozenStrictModel):
    """One immutable cue projected onto the assembled music timeline."""

    cue_id: str = Field(min_length=1)
    time_ms: int = Field(ge=0)
    kind: Literal["section_boundary", "downbeat", "accent", "ending_hit"]
    strength: float = Field(ge=0.0, le=1.0)


class MusicBoundarySpec(FrozenStrictModel):
    """Duration bounds and semantic preference for one picture chapter."""

    beat_id: str = Field(min_length=1)
    preferred_duration_ms: int = Field(gt=0)
    minimum_duration_ms: int = Field(gt=0)
    maximum_duration_ms: int = Field(gt=0)
    boundary_priority: Literal["low", "normal", "high"] = "normal"
    boundary_alignment: Literal[
        "free",
        "content_locked",
        "phrase_preferred",
        "accent_preferred",
    ] = "free"
    semantic_music_target: Literal[
        "phrase_start",
        "phrase_end",
        "downbeat",
        "accent",
        "section_change",
    ] | None = None

    @model_validator(mode="after")
    def validate_duration_bounds(self) -> "MusicBoundarySpec":
        if not (
            self.minimum_duration_ms
            <= self.preferred_duration_ms
            <= self.maximum_duration_ms
        ):
            raise ValueError("music boundary duration bounds must be ordered")
        return self


class MusicBoundarySelection(FrozenStrictModel):
    boundary_after_beat_id: str
    preferred_boundary_ms: int = Field(gt=0)
    resolved_boundary_ms: int = Field(gt=0)
    cue_id: str | None = None
    cue_kind: str | None = None
    snap_delta_ms: int
    snap_applied: bool


class MusicBoundarySolution(FrozenStrictModel):
    """Globally feasible picture boundaries for one fixed-duration timeline."""

    contract_version: Literal["music-boundary-solution-v1"] = (
        "music-boundary-solution-v1"
    )
    total_duration_ms: int = Field(gt=0)
    selections: tuple[MusicBoundarySelection, ...]
    chapter_durations_ms: tuple[int, ...]
    cue_aligned_boundary_count: int = Field(ge=0)


class SemanticRhythmSpec(FrozenStrictModel):
    """One beat's semantic dwell envelope before optional music snapping."""

    beat_id: str = Field(min_length=1)
    minimum_duration_ms: int = Field(gt=0)
    preferred_duration_ms: int = Field(gt=0)
    maximum_duration_ms: int = Field(gt=0)
    cut_pressure: float | None = Field(default=None, ge=0.0, le=1.0)
    energy_role: Literal[
        "low_hold",
        "rise",
        "peak",
        "release",
        "reset",
    ] | None = None
    boundary_alignment: Literal[
        "free",
        "content_locked",
        "phrase_preferred",
        "accent_preferred",
    ] = "free"

    @model_validator(mode="after")
    def validate_duration_bounds(self) -> "SemanticRhythmSpec":
        if not (
            self.minimum_duration_ms
            <= self.preferred_duration_ms
            <= self.maximum_duration_ms
        ):
            raise ValueError("semantic rhythm duration bounds must be ordered")
        return self


class SemanticRhythmSelection(FrozenStrictModel):
    beat_id: str
    duration_ms: int = Field(gt=0)
    semantic_target_ms: int = Field(gt=0)
    preferred_delta_ms: int
    decision_codes: tuple[str, ...]


class SemanticRhythmSolution(FrozenStrictModel):
    """Globally reconciled visual cadence, independent of soundtrack presence."""

    contract_version: Literal["semantic-rhythm-solution-v1"] = (
        "semantic-rhythm-solution-v1"
    )
    total_duration_ms: int = Field(gt=0)
    selections: tuple[SemanticRhythmSelection, ...]
    objective_cost: float = Field(ge=0.0)
    cadence_source: Literal[
        "semantic_attention_and_energy",
        "bounded_legacy_dwell",
    ]


class _RhythmBeamState(FrozenStrictModel):
    durations_ms: tuple[int, ...] = ()
    elapsed_ms: int = Field(default=0, ge=0)
    objective_cost: float = Field(default=0.0, ge=0.0)


def _semantic_rhythm_target_ms(spec: SemanticRhythmSpec) -> int:
    """Translate semantic pressure into a bounded target, never a hard rule."""

    if (
        spec.boundary_alignment == "content_locked"
        or spec.minimum_duration_ms == spec.maximum_duration_ms
    ):
        return spec.preferred_duration_ms
    span = spec.maximum_duration_ms - spec.minimum_duration_ms
    preferred_ratio = (
        spec.preferred_duration_ms - spec.minimum_duration_ms
    ) / max(span, 1)
    pressure_ratio = (
        1.0 - spec.cut_pressure
        if spec.cut_pressure is not None
        else preferred_ratio
    )
    # Gemini's preferred dwell remains the main authority. Cut pressure and
    # semantic energy only resolve flexibility already permitted by the
    # attention envelope.
    target_ratio = preferred_ratio * 0.72 + pressure_ratio * 0.28
    target_ratio += {
        "low_hold": 0.06,
        "rise": -0.04,
        "peak": -0.07,
        "release": 0.08,
        "reset": 0.04,
        None: 0.0,
    }[spec.energy_role]
    target_ratio = max(0.0, min(1.0, target_ratio))
    return round(spec.minimum_duration_ms + span * target_ratio)


def _semantic_duration_candidates(
    spec: SemanticRhythmSpec,
    *,
    step_ms: int,
) -> tuple[int, ...]:
    if spec.minimum_duration_ms == spec.maximum_duration_ms:
        return (spec.minimum_duration_ms,)
    first_grid = (
        (spec.minimum_duration_ms + step_ms - 1) // step_ms
    ) * step_ms
    grid = range(
        first_grid,
        spec.maximum_duration_ms + 1,
        step_ms,
    )
    return tuple(
        sorted(
            {
                spec.minimum_duration_ms,
                spec.preferred_duration_ms,
                spec.maximum_duration_ms,
                _semantic_rhythm_target_ms(spec),
                *grid,
            }
        )
    )


def solve_semantic_rhythm_durations(
    specs: Sequence[SemanticRhythmSpec],
    *,
    total_duration_ms: int,
    step_ms: int = 100,
    beam_width: int = 512,
) -> SemanticRhythmSolution:
    """Choose a globally coherent cadence before optional soundtrack snapping.

    The solver consumes Gemini's semantic attention and energy observations,
    but all durations stay inside application-measured readability/capacity
    bounds. With no music this is the production cadence authority; with music
    its output becomes the preferred timeline for the cue-boundary solver.
    """

    if not specs:
        raise ValueError("semantic rhythm solver requires at least one beat")
    if total_duration_ms <= 0:
        raise ValueError("semantic rhythm total duration must be positive")
    if step_ms <= 0:
        raise ValueError("semantic rhythm step must be positive")
    if beam_width <= 0:
        raise ValueError("semantic rhythm beam width must be positive")
    if (
        sum(spec.minimum_duration_ms for spec in specs) > total_duration_ms
        or sum(spec.maximum_duration_ms for spec in specs) < total_duration_ms
    ):
        raise ValueError("semantic rhythm bounds cannot satisfy total duration")
    beat_ids = [spec.beat_id for spec in specs]
    if len(beat_ids) != len(set(beat_ids)):
        raise ValueError("semantic rhythm beat IDs must be unique")

    minimum_suffix = [
        sum(spec.minimum_duration_ms for spec in specs[index:])
        for index in range(len(specs) + 1)
    ]
    maximum_suffix = [
        sum(spec.maximum_duration_ms for spec in specs[index:])
        for index in range(len(specs) + 1)
    ]
    states: list[_RhythmBeamState] = [_RhythmBeamState()]
    for index, spec in enumerate(specs):
        remaining_min = minimum_suffix[index + 1]
        remaining_max = maximum_suffix[index + 1]
        target = _semantic_rhythm_target_ms(spec)
        candidates = _semantic_duration_candidates(spec, step_ms=step_ms)
        expanded: list[_RhythmBeamState] = []
        for state in states:
            if index == len(specs) - 1:
                durations = (total_duration_ms - state.elapsed_ms,)
            else:
                durations = candidates
            for duration in durations:
                if not (
                    spec.minimum_duration_ms
                    <= duration
                    <= spec.maximum_duration_ms
                ):
                    continue
                elapsed = state.elapsed_ms + duration
                remaining = total_duration_ms - elapsed
                if not remaining_min <= remaining <= remaining_max:
                    continue
                span = max(
                    spec.maximum_duration_ms - spec.minimum_duration_ms,
                    step_ms,
                )
                semantic_cost = abs(duration - target) / span
                preferred_cost = (
                    abs(duration - spec.preferred_duration_ms) / span
                ) * 0.35
                cadence_cost = 0.0
                if state.durations_ms:
                    prior_duration = state.durations_ms[-1]
                    prior_spec = specs[index - 1]
                    transition_pressure = max(
                        float(spec.cut_pressure or 0.0),
                        float(prior_spec.cut_pressure or 0.0),
                    )
                    if (
                        transition_pressure >= 0.55
                        and abs(duration - prior_duration) < 150
                        and spec.boundary_alignment != "content_locked"
                        and prior_spec.boundary_alignment != "content_locked"
                    ):
                        cadence_cost = 0.08
                expanded.append(
                    _RhythmBeamState(
                        durations_ms=state.durations_ms + (duration,),
                        elapsed_ms=elapsed,
                        objective_cost=(
                            state.objective_cost
                            + semantic_cost
                            + preferred_cost
                            + cadence_cost
                        ),
                    )
                )
        if not expanded:
            raise ValueError(
                "no globally feasible semantic rhythm sequence satisfies "
                f"{spec.beat_id}"
            )
        # Retain distinct elapsed/last-duration frontiers. This keeps the
        # search bounded without collapsing all cadence shapes into one state.
        best_by_frontier: dict[tuple[int, int], _RhythmBeamState] = {}
        for state in sorted(
            expanded,
            key=lambda item: (
                item.objective_cost,
                item.elapsed_ms,
                item.durations_ms,
            ),
        ):
            frontier = (
                state.elapsed_ms,
                state.durations_ms[-1] // step_ms,
            )
            best_by_frontier.setdefault(frontier, state)
        states = list(best_by_frontier.values())[:beam_width]

    completed = [
        state for state in states if state.elapsed_ms == total_duration_ms
    ]
    if not completed:
        raise ValueError("semantic rhythm solver did not close the total duration")
    best = min(
        completed,
        key=lambda item: (item.objective_cost, item.durations_ms),
    )
    selections: list[SemanticRhythmSelection] = []
    for spec, duration in zip(specs, best.durations_ms, strict=True):
        target = _semantic_rhythm_target_ms(spec)
        codes = ["semantic_attention_envelope_respected"]
        if spec.energy_role is not None:
            codes.append(f"energy_role_{spec.energy_role}")
        if spec.boundary_alignment == "content_locked":
            codes.append("content_locked_dwell_preserved")
        selections.append(
            SemanticRhythmSelection(
                beat_id=spec.beat_id,
                duration_ms=duration,
                semantic_target_ms=target,
                preferred_delta_ms=duration - spec.preferred_duration_ms,
                decision_codes=tuple(codes),
            )
        )
    return SemanticRhythmSolution(
        total_duration_ms=total_duration_ms,
        selections=tuple(selections),
        objective_cost=round(best.objective_cost, 6),
        cadence_source=(
            "semantic_attention_and_energy"
            if any(spec.cut_pressure is not None for spec in specs)
            else "bounded_legacy_dwell"
        ),
    )


def solve_music_aligned_boundaries(
    specs: Sequence[MusicBoundarySpec],
    cues: Sequence[MusicBoundaryCue],
    *,
    total_duration_ms: int,
) -> MusicBoundarySolution:
    """Jointly choose legal picture boundaries instead of greedily snapping.

    Source/attention capacities are duration *bounds*, not reasons to disable
    music alignment.  The solver considers every internal boundary together,
    so moving one cut onto a cue can never make a later chapter unreadable or
    exceed its source-safe interval.  The unsnapped preferred boundary remains
    a legal deterministic fallback when no cue combination is globally safe.
    """

    if total_duration_ms <= 0:
        raise ValueError("music boundary total duration must be positive")
    if not specs:
        raise ValueError("music boundary solver requires at least one chapter")
    if (
        sum(spec.minimum_duration_ms for spec in specs) > total_duration_ms
        or sum(spec.maximum_duration_ms for spec in specs) < total_duration_ms
    ):
        raise ValueError("chapter bounds cannot satisfy the timeline duration")
    cue_ids = [cue.cue_id for cue in cues]
    if len(cue_ids) != len(set(cue_ids)):
        raise ValueError("music boundary cue IDs must be unique")
    ordered_cues = sorted(cues, key=lambda cue: (cue.time_ms, cue.cue_id))

    preferred_boundaries: list[int] = []
    running = 0
    for spec in specs[:-1]:
        running += spec.preferred_duration_ms
        preferred_boundaries.append(running)

    # State: accumulated objective cost, selected boundaries.  Candidate count
    # is bounded by the cue window and the public chapter limit (16).
    states: list[tuple[float, tuple[MusicBoundarySelection, ...]]] = [(0.0, ())]
    for index, spec in enumerate(specs[:-1]):
        preferred_boundary = preferred_boundaries[index]
        window_ms = {
            "low": 250,
            "normal": 450,
            "high": 650,
        }[spec.boundary_priority]
        if spec.boundary_alignment == "content_locked":
            window_ms = 0
            allowed_kinds: set[str] = set()
        elif spec.boundary_alignment == "phrase_preferred":
            window_ms = max(window_ms, 650)
            allowed_kinds = {"section_boundary", "downbeat", "ending_hit"}
        elif spec.boundary_alignment == "accent_preferred":
            window_ms = max(window_ms, 650)
            allowed_kinds = {
                "section_boundary",
                "downbeat",
                "accent",
                "ending_hit",
            }
        else:
            allowed_kinds = (
                {"section_boundary", "downbeat"}
                if spec.boundary_priority == "low"
                else {
                    "section_boundary",
                    "downbeat",
                    "accent",
                    "ending_hit",
                }
            )
        semantic_kinds = {
            "phrase_start": {"section_boundary", "downbeat"},
            "phrase_end": {"section_boundary", "ending_hit"},
            "downbeat": {"downbeat"},
            "accent": {"accent"},
            "section_change": {"section_boundary"},
        }.get(spec.semantic_music_target, set())
        candidates: list[tuple[int, MusicBoundaryCue | None]] = [
            (preferred_boundary, None)
        ]
        candidates.extend(
            (cue.time_ms, cue)
            for cue in ordered_cues
            if (
                cue.kind in allowed_kinds
                and abs(cue.time_ms - preferred_boundary) <= window_ms
            )
        )
        # Multiple cue labels may share a sample; retain the semantically
        # strongest candidate while keeping the preferred fallback.
        deduped: dict[tuple[int, str | None], MusicBoundaryCue | None] = {}
        for position, cue in candidates:
            deduped[(position, cue.cue_id if cue is not None else None)] = cue

        expanded: list[tuple[float, tuple[MusicBoundarySelection, ...]]] = []
        for cost, prior in states:
            previous_boundary = (
                prior[-1].resolved_boundary_ms if prior else 0
            )
            for (position, _), cue in deduped.items():
                chapter_duration = position - previous_boundary
                if not (
                    spec.minimum_duration_ms
                    <= chapter_duration
                    <= spec.maximum_duration_ms
                ):
                    continue
                remaining_specs = specs[index + 1 :]
                remaining_duration = total_duration_ms - position
                if not (
                    sum(item.minimum_duration_ms for item in remaining_specs)
                    <= remaining_duration
                    <= sum(item.maximum_duration_ms for item in remaining_specs)
                ):
                    continue
                if cue is None:
                    # Prefer real musical exits, especially at high-pressure
                    # boundaries, while keeping content-locked cuts untouched.
                    candidate_cost = (
                        0.0
                        if spec.boundary_alignment == "content_locked"
                        else {"low": 1.4, "normal": 2.0, "high": 2.6}[
                            spec.boundary_priority
                        ]
                    )
                else:
                    distance_cost = abs(position - preferred_boundary) / max(
                        window_ms,
                        1,
                    )
                    semantic_cost = (
                        0.0
                        if not semantic_kinds or cue.kind in semantic_kinds
                        else 0.8
                    )
                    structural_cost = (
                        0.0
                        if cue.kind == "section_boundary"
                        else 0.15 if cue.kind == "downbeat" else 0.3
                    )
                    candidate_cost = (
                        distance_cost
                        + semantic_cost
                        + structural_cost
                        - cue.strength * 0.25
                    )
                selection = MusicBoundarySelection(
                    boundary_after_beat_id=spec.beat_id,
                    preferred_boundary_ms=preferred_boundary,
                    resolved_boundary_ms=position,
                    cue_id=cue.cue_id if cue is not None else None,
                    cue_kind=cue.kind if cue is not None else None,
                    snap_delta_ms=position - preferred_boundary,
                    snap_applied=cue is not None,
                )
                expanded.append((cost + candidate_cost, prior + (selection,)))
        if not expanded:
            raise ValueError(
                "no globally feasible music boundary sequence satisfies "
                f"chapter {spec.beat_id}"
            )
        # Keep the solver bounded without discarding distinct timeline states.
        best_by_position: dict[
            int,
            tuple[float, tuple[MusicBoundarySelection, ...]],
        ] = {}
        for state in sorted(
            expanded,
            key=lambda item: (
                item[0],
                tuple(
                    selection.resolved_boundary_ms
                    for selection in item[1]
                ),
                tuple(selection.cue_id or "" for selection in item[1]),
            ),
        ):
            position = state[1][-1].resolved_boundary_ms
            best_by_position.setdefault(position, state)
        states = list(best_by_position.values())[:256]

    feasible: list[tuple[float, tuple[MusicBoundarySelection, ...]]] = []
    for state in states:
        prior = state[1]
        previous = prior[-1].resolved_boundary_ms if prior else 0
        final_duration = total_duration_ms - previous
        final_spec = specs[-1]
        if (
            final_spec.minimum_duration_ms
            <= final_duration
            <= final_spec.maximum_duration_ms
        ):
            feasible.append(state)
    if not feasible:
        raise ValueError(
            "no globally feasible music boundary sequence satisfies final chapter"
        )
    _, selections = min(
        feasible,
        key=lambda item: (
            item[0],
            tuple(selection.resolved_boundary_ms for selection in item[1]),
            tuple(selection.cue_id or "" for selection in item[1]),
        ),
    )
    boundaries = [
        0,
        *(selection.resolved_boundary_ms for selection in selections),
        total_duration_ms,
    ]
    durations = tuple(
        end - start
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    )
    return MusicBoundarySolution(
        total_duration_ms=total_duration_ms,
        selections=selections,
        chapter_durations_ms=durations,
        cue_aligned_boundary_count=sum(
            selection.snap_applied for selection in selections
        ),
    )


class _BeamState(FrozenStrictModel):
    selections: tuple[SequenceSelection, ...] = ()
    duration_ms: int = Field(default=0, ge=0)
    panel_duration_ms: int = Field(default=0, ge=0)
    score: float = 0.0
    source_intervals: tuple[tuple[str, int, int], ...] = ()


def optimize_sequence(
    beat_sets: Sequence[BeatOptionSet],
    *,
    policy: AutonomousEditPolicy,
    beam_width: int = 24,
) -> SequenceOptimizationResult:
    """Bounded evidence-first beam search; never lets quality buy a hard pass."""

    if beam_width < 1:
        raise ValueError("beam width must be positive")
    states: list[_BeamState] = [_BeamState()]
    hard_failures: list[str] = []
    total_beats = max(1, len(beat_sets))
    for beat_index, beat in enumerate(beat_sets):
        valid = [
            option
            for option in beat.options
            if not option.hard_constraint_failures(beat.priority)
        ]
        if not valid and beat.priority != "optional":
            failures = sorted(
                {
                    code
                    for option in beat.options
                    for code in option.hard_constraint_failures(beat.priority)
                }
                or {"no_candidate_options"}
            )
            hard_failures.extend(
                f"{beat.beat_id}:{failure}" for failure in failures
            )
            return _blocked_result(
                policy=policy,
                selections=states[0].selections,
                hard_failures=hard_failures,
            )

        choices: list[SequenceOption | None] = list(valid)
        if (
            beat.priority == "optional"
            and policy.editorial.allow_optional_beat_omission
        ):
            choices.append(None)
        if not choices:
            return _blocked_result(
                policy=policy,
                selections=states[0].selections,
                hard_failures=(f"{beat.beat_id}:optional_omission_forbidden",),
            )

        expanded: list[_BeamState] = []
        for state in states:
            for option in choices:
                expanded.append(_extend_state(state, beat, option))
        states = sorted(
            expanded,
            key=lambda state: _beam_rank(
                state,
                policy,
                completion_fraction=(beat_index + 1) / total_beats,
            ),
            reverse=True,
        )[:beam_width]

    panel_safe_states = [
        state
        for state in states
        if state.duration_ms == 0
        or (
            state.panel_duration_ms / state.duration_ms
            <= policy.presentation.max_panel_runtime_fraction + 1e-9
        )
    ]
    acceptable = [
        state
        for state in panel_safe_states
        if policy.duration.min_ms
        <= state.duration_ms
        <= policy.duration.max_ms
    ]
    outcome: Literal["complete", "best_effort_shortened", "blocked"]
    if acceptable:
        best = max(acceptable, key=lambda state: _final_rank(state, policy))
        outcome = "complete"
    elif (
        policy.execution_profile == "autonomous_best_effort"
        and panel_safe_states
        and max(
            state.duration_ms for state in panel_safe_states
        ) < policy.duration.min_ms
    ):
        best = max(
            panel_safe_states,
            key=lambda state: _final_rank(state, policy),
        )
        outcome = "best_effort_shortened"
    else:
        closest = max(states, key=lambda state: _final_rank(state, policy))
        failure = (
            "panel_runtime_fraction_exceeded"
            if states and not panel_safe_states
            else "duration_outside_policy_bounds"
        )
        return _blocked_result(
            policy=policy,
            selections=closest.selections,
            hard_failures=(failure,),
        )

    omitted = tuple(
        selection.beat_id
        for selection in best.selections
        if selection.option is None
    )
    return SequenceOptimizationResult(
        selections=best.selections,
        total_duration_ms=best.duration_ms,
        target_duration_ms=policy.duration.target_ms,
        duration_delta_ms=best.duration_ms - policy.duration.target_ms,
        score=round(_final_rank(best, policy), 6),
        outcome=outcome,
        omitted_beat_ids=omitted,
    )


def _extend_state(
    state: _BeamState,
    beat: BeatOptionSet,
    option: SequenceOption | None,
) -> _BeamState:
    if option is None:
        return _BeamState(
            selections=state.selections
            + (
                SequenceSelection(
                    beat_id=beat.beat_id,
                    priority=beat.priority,
                    option=None,
                    decision_codes=("policy_authorized_optional_omission",),
                ),
            ),
            duration_ms=state.duration_ms,
            panel_duration_ms=state.panel_duration_ms,
            score=state.score - 0.04,
            source_intervals=state.source_intervals,
        )

    interval = (
        option.source_asset_id,
        option.source_in_pts,
        option.source_out_pts,
    )
    reuse_penalty = 0.0
    if interval in state.source_intervals:
        reuse_penalty = 0.20 if option.authorized_reprise else 1.0
    continuity_penalty = _continuity_penalty(state, option)
    intrusion_penalty = {
        "two_panel_layout": 0.04,
        "solid_matte_fit": 0.08,
        "intentional_freeze": 0.05,
    }.get(option.presentation_mode, 0.0)
    preference = (
        option.semantic_fit * 0.32
        + option.readability * 0.26
        + option.technical_quality * 0.20
        + option.music_flow * 0.16
        - option.synthetic_motion_distance * 0.06
        - continuity_penalty
        - intrusion_penalty
        - reuse_penalty
    )
    decision_codes = ["hard_constraints_passed"]
    if option.duration_ms >= option.preferred_readable_ms:
        decision_codes.append("preferred_readability_reached")
    if option.authorized_reprise:
        decision_codes.append("editorial_reprise_authorized")
    return _BeamState(
        selections=state.selections
        + (
            SequenceSelection(
                beat_id=beat.beat_id,
                priority=beat.priority,
                option=option,
                decision_codes=tuple(decision_codes),
            ),
        ),
        duration_ms=state.duration_ms + option.duration_ms,
        panel_duration_ms=(
            state.panel_duration_ms
            + (
                option.duration_ms
                if option.presentation_mode == "two_panel_layout"
                else 0
            )
        ),
        score=state.score + preference,
        source_intervals=state.source_intervals + (interval,),
    )


def _continuity_penalty(
    state: _BeamState,
    option: SequenceOption,
) -> float:
    previous = next(
        (
            selection.option
            for selection in reversed(state.selections)
            if selection.option is not None
        ),
        None,
    )
    if (
        previous is None
        or previous.exit_x is None
        or option.entry_x is None
    ):
        return 0.0
    return abs(previous.exit_x - option.entry_x) * 0.08


def _beam_rank(
    state: _BeamState,
    policy: AutonomousEditPolicy,
    *,
    completion_fraction: float,
) -> float:
    # A bounded intermediate duration pressure prevents the beam from filling
    # with only long or only short variants before later beats are considered.
    expected = policy.duration.target_ms * completion_fraction
    duration_pressure = abs(state.duration_ms - expected) / max(
        policy.duration.target_ms,
        1,
    )
    return state.score - duration_pressure * 0.02


def _final_rank(
    state: _BeamState,
    policy: AutonomousEditPolicy,
) -> float:
    duration_error = abs(state.duration_ms - policy.duration.target_ms)
    duration_penalty = duration_error / max(policy.duration.target_ms, 1)
    return state.score - duration_penalty * 0.35


def _blocked_result(
    *,
    policy: AutonomousEditPolicy,
    selections: tuple[SequenceSelection, ...],
    hard_failures: Sequence[str],
) -> SequenceOptimizationResult:
    total = sum(
        selection.option.duration_ms
        for selection in selections
        if selection.option is not None
    )
    return SequenceOptimizationResult(
        selections=selections,
        total_duration_ms=total,
        target_duration_ms=policy.duration.target_ms,
        duration_delta_ms=total - policy.duration.target_ms,
        score=0.0,
        outcome="blocked",
        hard_failure_codes=tuple(hard_failures),
    )


class SegmentRenderCacheKey(FrozenStrictModel):
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_in_pts: int = Field(ge=0)
    source_out_pts: int = Field(gt=0)
    presentation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tracking_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    filter_graph_version: str = Field(min_length=1)
    aspect: Literal["16:9", "9:16"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps_numerator: int = Field(gt=0)
    fps_denominator: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_pts(self) -> "SegmentRenderCacheKey":
        if self.source_out_pts <= self.source_in_pts:
            raise ValueError("segment cache interval must be non-empty")
        return self

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class SegmentRenderRequest(FrozenStrictModel):
    segment_id: str = Field(min_length=1)
    cache_key: SegmentRenderCacheKey


class SegmentRenderRecord(FrozenStrictModel):
    segment_id: str
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_path: str
    cache_hit: bool


class IncrementalRenderResult(FrozenStrictModel):
    records: tuple[SegmentRenderRecord, ...]
    rendered_segment_ids: tuple[str, ...]
    reused_segment_ids: tuple[str, ...]


def render_segments_incrementally(
    requests: Sequence[SegmentRenderRequest],
    *,
    cache_dir: Path,
    renderer: Callable[[SegmentRenderRequest, Path], None],
    suffix: str = ".mp4",
) -> IncrementalRenderResult:
    """Render cache misses only; unchanged segments are never re-rendered."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[SegmentRenderRecord] = []
    rendered: list[str] = []
    reused: list[str] = []
    for request in requests:
        output = cache_dir / f"{request.cache_key.digest}{suffix}"
        hit = output.is_file() and output.stat().st_size > 0
        if hit:
            reused.append(request.segment_id)
        else:
            temporary = cache_dir / (
                f".{request.cache_key.digest}.{os.getpid()}.partial{suffix}"
            )
            renderer(request, temporary)
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise RuntimeError(
                    f"renderer did not create segment {request.segment_id}"
                )
            temporary.replace(output)
            rendered.append(request.segment_id)
        records.append(
            SegmentRenderRecord(
                segment_id=request.segment_id,
                cache_key=request.cache_key.digest,
                output_path=str(output),
                cache_hit=hit,
            )
        )
    return IncrementalRenderResult(
        records=tuple(records),
        rendered_segment_ids=tuple(rendered),
        reused_segment_ids=tuple(reused),
    )


def concat_manifest_lines(
    result: IncrementalRenderResult,
) -> tuple[str, ...]:
    """Return deterministic ffconcat entries in requested segment order."""

    return tuple(
        f"file {json.dumps(record.output_path)}" for record in result.records
    )
