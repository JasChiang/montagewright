from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .autonomous_policy import AutonomousEditPolicy
from .models import FrozenStrictModel


BeatPriority = Literal["hard", "preferred", "optional"]
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
